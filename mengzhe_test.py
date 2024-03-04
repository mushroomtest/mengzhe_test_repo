# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright 2019 The Microsoft DeepSpeed Team
import os

from types import MethodType

import torch

import torch.distributed as dist

from deepspeed.utils.logging import logger
from deepspeed.utils.timer import ThroughputTimer

from deepspeed.runtime.engine import MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.dataloader import RepeatingLoader

from deepspeed.runtime.pipe.module import PipelineModule, PipelineError
from deepspeed.runtime.pipe.engine import PipelineEngine
from . import p2p
from . import schedule
try:
    import byteps.torch as bps
except ImportError:
    print("byteps is not installed. Pipeline parallelism is disabled")
    bps = None

from .module import VeGiantModule
from deepspeed.utils import log_dist
import logging
from torch._six import inf

# from inspect import signature

LOG_STAGE = -2
DATA_PARALLEL_ID = -2

try:
    from apex import amp
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    pass


def is_even(number):
    return number % 2 == 0

ENABLE_PYTORCH_BROADCAST = os.environ.get("ENABLE_PYTORCH_BROADCAST", "0") != "0"



DS_PIPE_VERBOSE = int(os.environ.get('DS_PIPE_VERBOSE', "0"))
MEGATRON_DEBUG_DATA = os.environ.get('MEGATRON_DEBUG_DATA', "0") != "0"
MEGATRON_DEBUG_GRAD = os.environ.get('MEGATRON_DEBUG_GRAD', "0") != "0"
ENABLE_BPS_PARTITION = os.environ.get("ENABLE_BPS_PARTITION", "0") != "0"


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

def _dtype_to_code(dtype):
    if dtype == torch.half:
        return 0
    elif dtype == torch.float:
        return 1
    elif dtype == torch.int16:
        return 2
    elif dtype == torch.int32:
        return 3
    elif dtype == torch.int64:
        return 4
    elif dtype == torch.bool:
        return 5
    else:
        raise AssertionError("not recognized tensor type for pipeline send")

def _code_to_dtype(code):
    if code == 0:
        return torch.half
    elif code == 1:
        return torch.float
    elif code == 2:
        return torch.int16
    elif code == 3:
        return torch.int32
    elif code == 4:
        return torch.int64
    elif code == 5:
        return torch.bool
    else:
        raise AssertionError("not recognized tensor type code for pipeline recv")

class VeGiantModelEngine(PipelineEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    def overwrite(self, config_params, args):
        if args.batch_size is not None:
            log_dist(f'overwrite dsconfig train_micro_batch_size_per_gpu to {args.batch_size}', \
                ranks=[-1], level=logging.DEBUG)
            config_params['train_micro_batch_size_per_gpu'] = args.batch_size
        
        if args.gradient_accumulation_steps is not None:
            log_dist(f'overwrite dsconfig gradient_accumulation_steps to {args.gradient_accumulation_steps}', \
                ranks=[-1], level=logging.DEBUG)
            config_params['gradient_accumulation_steps'] = args.gradient_accumulation_steps

        if args.train_batch_size is not None:
            log_dist(f'overwrite dsconfig train_batch_size to {args.train_batch_size}, ', \
                ranks=[-1], level=logging.DEBUG)
            config_params['train_batch_size'] = args.train_batch_size

        if args.log_interval is not None:
            config_params['steps_per_print'] = args.log_interval

    def __init__(self, args,
                    model,
                    optimizer,
                    model_parameters,
                    training_data,
                    lr_scheduler,
                    mpu,
                    dist_init_required,
                    collate_fn,
                    config_params):
        
        self.overwrite(config_params, args)
        super(PipelineEngine, self).__init__(args,
                    model,
                    optimizer,
                    model_parameters,
                    training_data,
                    lr_scheduler,
                    mpu,
                    dist_init_required,
                    collate_fn,
                    config_params)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        # pipeline step for logging
        self.args = args
        self.log_batch_step_id = -1
        self.train_mode = True

        self.enable_backward_allreduce = False
        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()
        self.first_train = True
        self.first_eval = True

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.mp_id = self.grid.get_model_parallel_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.data_iterator = None
        self.batch_fn = None
        self.result_dict = {}

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.micro_batch_size *
                                           self.micro_batches,
                                           num_workers=self.dp_world_size,
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        self.is_pipe_partitioned = False if self.args.broadcast_activation else (self.is_model_parallel and ENABLE_PYTORCH_BROADCAST)
        self.is_grad_partitioned = False

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params,
                                               unique_params]).to(self.device)
        print(f'Calculating param sizes ... ', flush=True)


        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        print(f'DONE calculating param sizes. Now init proc groups', flush=True)

        #intialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs' : [],   # batch input and received activations
            'labels' : [],   # labels from batch input
            'outputs' : [],  # activations
            'output_tensors' : [], # tensor object to preserve backward graph
            'bps_act_recv' : [],  # activations recv
            'bps_grad_recv' : [],  # activations recv
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)
        self.metric = 0

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                'activation_checkpoint_interval']

        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        log_dist(f'Initialize pipeline communicators', \
            ranks=[-1], level=logging.DEBUG)

        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
        
        log_dist(f'DONE Initialize pipeline communicators', \
            ranks=[-1], level=logging.DEBUG)

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

        if self.local_rank == -1:
            # or number of visiable device will be better
            self.local_rank = self.global_rank % torch.cuda.device_count()

        if not p2p.ENABLE_PYTORCH_BROADCAST:
            gpu_per_node = int(os.environ['GPU_PER_WORKER'])
            print(f'bps init worker: {gpu_per_node}, {self.local_rank}/{self.global_rank}', flush=True)
            os.environ['BYTEPS_LOCAL_RANK'] = str(self.local_rank)
            os.environ['BYTEPS_LOCAL_SIZE'] = str(gpu_per_node)
            os.environ['BYTEPS_VISIBLE_DEVICE'] = str(self.local_rank)
            os.environ['DMLC_ROLE'] = 'joint'
            os.environ['DMLC_WORKER_ID'] = str(self.global_rank)
            bps.init(lazy=False)
            print(f'bps init DONE', flush=True)


    def _profiling_func_exit(self):
        torch.cuda.nvtx.range_pop()
    
    def _profiling_func_enter(self, func):
        torch.cuda.nvtx.range_push(f'stage_id: {self.stage_id}, mp_id: {self.mp_id}, fun: {func}')

    def _build_data_iter(self, dataset):
        if not isinstance(dataset, torch.utils.data.Dataset):
            self.set_dataloader(dataset)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.dp_world_size,
                rank=self.mpu.get_data_parallel_rank(),
                shuffle=False)
            # Build a loader and make it repeating.
            pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
            pipe_dataloader = RepeatingLoader(pipe_dataloader)
            self.set_dataloader(pipe_dataloader)

    def _exec_reduce_tied_grads(self):
        self._profiling_func_enter('_exec_reduce_tied_grads')
        self.module.allreduce_tied_weight_gradients()
        self._profiling_func_exit()

    def _exec_reduce_grads(self):
        self._profiling_func_enter('_exec_reduce_grads')
        self._force_grad_boundary = True
        if self.is_data_parallel:
            self.buffered_allreduce_fallback(
                elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False
        self._profiling_func_exit()


    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def train_batch(self, data_iter=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """

        if DS_PIPE_VERBOSE:
            print(f'[{self.global_rank}] start train_batch()', flush=True)
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        if data_iter is not None:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.train()
        self.total_loss = None

        # Do the work
        self.timers('train_batch').start()
        # We only enable prefetching starting from the second batch
        if not ENABLE_PYTORCH_BROADCAST:
            sched = schedule.BytePSTrainSchedule(micro_batches=self.micro_batches,
                                                stages=self.num_stages,
                                                stage_id=self.stage_id, prefetch=not self.first_train)
        else:
            sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        cmd = ','.join(str(x) for x in sched)
        # log_dist(f'stage_id: {self.stage_id}, sched:{cmd}', ranks=[-1], level=logging.INFO)
        self._exec_schedule(sched)
        self.agg_train_loss = self._aggregate_total_loss()
        self.timers('train_batch').stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers('train_batch').elapsed(reset=True)
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'samples/sec: {tput:0.3f}')

        # Tensorboard
        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/train_loss',
                                        self.agg_train_loss.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                if self.global_steps % self.steps_per_print() == 0:
                    self.summary_writer.flush()

        if self.wall_clock_breakdown(
        ) and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                'pipe_recv_grad'
            ])

        # TODO: should return precisely what loss returned and allow others to be queried?
        self.first_train = False
        if DS_PIPE_VERBOSE:
            print(f'[{self.global_rank}] DONE train_batch()', flush=True)
        
        self.result_dict['loss'] = self.agg_train_loss
        return self.result_dict

    def eval_batch(self, data_iter):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """

        self.module.eval()
        self.eval()
        self.total_loss = None

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        self.timers('eval_batch').start()
        if not ENABLE_PYTORCH_BROADCAST:
            sched = schedule.BytePSInferenceSchedule(micro_batches=1,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id, prefetch=False)
        else:
            sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)
        with torch.no_grad():
            self._exec_schedule(sched)

        self.agg_eval_loss = self._aggregate_total_loss()
        self.timers('eval_batch').stop()
        # # XXX hack model attribute
        # if hasattr(self.module, '_get_metrics'):
        #     self.module._ref_model[0].metric = {'pscc': self._aggregate_metric()}

        # if self.global_rank == 0:
        #     elapsed = self.timers('eval_batch').elapsed(reset=True)
        #     iter_time = elapsed
        #     print(f'loss: {self.agg_eval_loss:0.4f} '
        #             f'iter time (s): {iter_time:0.3f} ')

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/eval_loss',
                                        self.agg_eval_loss.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                self.summary_writer.flush()

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()
        self.first_eval = False
        self.result_dict['loss'] = self.agg_eval_loss
        return self.result_dict

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _aggregate_metric(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            if DS_PIPE_VERBOSE:
                print(f'[{self.global_rank}] bcast src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                assert False

            assert self.global_rank in self.grid.pp_group
            metric = torch.Tensor([self.metric]).to(self.device)
            dist.broadcast(tensor=metric,
                           src=self.global_rank,
                           group=self.mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            if DS_PIPE_VERBOSE:
                print(f'[{self.global_rank}] bcast src={src_rank} group={self.grid.pp_group}', flush=True)
            assert src_rank in self.grid.pp_group
            metric = torch.Tensor([0.]).to(self.device)
            dist.broadcast(tensor=metric,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.metric = metric.clone().detach().cpu().numpy()

        return self.metric

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            # XXX Hack: do not scale loss
            loss = self._scale_loss(self.total_loss)

            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()

            if DS_PIPE_VERBOSE:
                print(f'[{self.global_rank}] bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                agg_loss /= self.dp_world_size

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=self.global_rank,
                           group=self.mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            if DS_PIPE_VERBOSE:
                print(f'[{self.global_rank}] bcast RECVER src={src_rank} group={self.grid.pp_group}', flush=True)
            dist.broadcast(tensor=losses,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()
        if DS_PIPE_VERBOSE:
            print(f'DONE aggregate total loss', flush=True)
        return agg_loss

    def set_dataloader(self, loader):
        """"""
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        self.batch_fn = fn
        # sig = signature(fn)
        # params = sig.parameters

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary


    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        if self.is_model_parallel:
            mp_rank = self.grid.get_slice_parallel_rank()
        else:
            mp_rank = 0

        batch = None

        # Only MP rank 0 loads the data.
        if mp_rank == 0:
            if self.data_iterator is None:
                raise ValueError(f"RANK={self.global_rank} no data iterator provided.")
            batch = next(self.data_iterator)

        # All MP ranks participate in batch_fn, where they might broadcast the data.
        if self.batch_fn:
            batch = self.batch_fn(batch, self.train_mode)

        # Sanity check dimensions.
        # XXX: the last minibatch with size < micro_batch_size kills us
        if torch.is_tensor(batch[0]):
            if batch[0].size(0) != self.micro_batch_size:
                print(f'size mismatch: {batch[0].size(0)} mb: {self.micro_batch_size}')
                assert batch[0].size(0) == self.micro_batch_size
                return self._next_batch()
        else:
            assert torch.is_tensor(batch[0][0])
            if batch[0][0].size(0) != self.micro_batch_size:
                print(f'HB next_batch: {batch[0][0].shape} vs {self.micro_batch_size}', flush=True)
                return self._next_batch()
        
        return batch

    def _exec_bps_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)
        self._profiling_func_enter('_exec_bps_forward_pass')

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        assert not self.is_pipe_partitioned

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        self._zero_grads(inputs)

        outputs = super(PipelineEngine, self).forward(inputs)

        # Partition the outputs if we are not the last stage
        assert not self.is_pipe_partitioned

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss and metrics on the last device
        if self.is_last_stage():
            if self.loss_model is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                ret = self.loss_model(outputs, labels)
                if isinstance(ret, dict):
                    self.result_dict = ret
                    self.loss = self.result_dict['loss']
                else:
                    self.loss = ret
            else:
                # Some models just return loss from forward()
                self.loss = outputs
            # get metric from self.module

            if isinstance(self.loss, torch.Tensor):
                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.total_loss[idx] += l.detach()

        self._profiling_func_exit()

    def _exec_bps_backward_pass(self, buffer_id):
        self._profiling_func_enter('_exec_bps_backward_pass')
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super(PipelineEngine, self).backward(self.loss)
            self.mem_status('AFTER BWD')
            self._profiling_func_exit()
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        assert not self.is_pipe_partitioned
        assert not self.is_grad_partitioned
        # TODO: do we need to clone()?
        grad_tensors = self.pipe_buffers['bps_grad_recv'][buffer_id]

        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            new_out_tensors=[]
            new_grad_tensors=[]
            for t,g in zip(out_tensors, grad_tensors):
                if t.requires_grad:
                    new_out_tensors.append(t)
                    new_grad_tensors.append(g)

            assert len(new_out_tensors) == len(new_grad_tensors)
            torch.autograd.backward(tensors=new_out_tensors, grad_tensors=new_grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs,), grad_tensors=(grad_tensors,))

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')
        self._profiling_func_exit()

    def _exec_load_micro_batch(self, buffer_id):
        self._profiling_func_enter('_exec_load_micro_batch')
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()

        if self.is_first_stage():
            loaded = None
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                loaded.requires_grad = loaded.is_floating_point()
                if MEGATRON_DEBUG_DATA:
                    print(f'batch = {loaded.sum().detach()}', flush=True)
            else:
                assert isinstance(batch[0], tuple)
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)
                if MEGATRON_DEBUG_DATA:
                    print(f'rank: {self.global_rank}, stage: {self.stage_id},  batch[0] = {[x.sum().detach() for x in loaded]}', flush=True)

            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
                if MEGATRON_DEBUG_DATA:
                    print(f'rank: {self.global_rank}, stage: {self.stage_id},  batch[1] = {[x.sum().detach() for x in loaded]}', flush=True)
            elif isinstance(batch[1], tuple):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)
                if MEGATRON_DEBUG_DATA:
                    print(f'rank: {self.global_rank}, stage: {self.stage_id},  batch[1] = {[x.sum().detach() for x in loaded]}', flush=True)

            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()
        self._profiling_func_exit()

    def _send_tensor_meta(self, buffer, recv_stage):
        self._profiling_func_enter('_send_tensor_meta')
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            send_dtype = torch.LongTensor(data=[_dtype_to_code(buffer.dtype)]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            p2p.send(send_dtype, recv_stage)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, list):
            assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=_dtype_to_code([tensor.dtype])).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                p2p.send(send_dtype, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=[_dtype_to_code(tensor.dtype)]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                p2p.send(send_dtype, recv_stage)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        self._profiling_func_exit()
        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        self._profiling_func_enter('_recv_tensor_meta')
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            recv_dtype = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_dtype, send_stage)
            recv_dtype_code = recv_dtype.item()
            recv_dtype = _code_to_dtype(recv_dtype_code)
            return self._allocate_buffer2(recv_shape, recv_dtype, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes = []
            recv_dtypes = []
            for idx in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes.append(recv_shape.tolist())
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_dtype, send_stage)
                recv_dtype_code = recv_dtype.item()
                recv_dtype = _code_to_dtype(recv_dtype_code)
                recv_dtypes.append(recv_dtype)

            buffers = self._allocate_buffers2(recv_shapes, recv_dtypes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')
        self._profiling_func_exit()

    def _mp_slice(self, x):
        mp_size = self.grid.get_model_parallel_world_size()
        return x.reshape((mp_size, -1))[self.mp_id:self.mp_id+1, :].detach()

    def _mp_view(self, x, rank):
        mp_size = self.grid.get_model_parallel_world_size()
        return x.view((mp_size, -1))[rank:rank+1, :]

    def _exec_bps_send_partitioned_activations(self, buffer_id):
        self._profiling_func_enter('_exec_bps_send_activations')
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        assert not self.args.broadcast_activation
        assert ENABLE_BPS_PARTITION
        name = f'act_{buffer_id}'
        if isinstance(outputs, torch.Tensor):
            p2p.bps_send(self._mp_slice(outputs.contiguous()),
                         self.next_stage, name, index=0, async_op=True)
        elif isinstance(outputs, (tuple, list)):
            for idx, buffer in enumerate(outputs):
                if DS_PIPE_VERBOSE >= 3:
                    print(f'DS BPS_SEND tensors {idx}/{len(outputs)}', flush=True)
                p2p.bps_send(self._mp_slice(buffer.contiguous()), self.next_stage,
                             name, index=idx, async_op=True)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()
        self._profiling_func_exit()

    def _exec_bps_send_activations(self, buffer_id):
        self._profiling_func_enter('_exec_bps_send_activations')
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        assert not self.args.broadcast_activation
        assert not ENABLE_BPS_PARTITION
        if self.mp_id == 0:
            name = f'act_{buffer_id}'
            if isinstance(outputs, torch.Tensor):
                p2p.bps_send(outputs.contiguous(), self.next_stage, name, index=0, async_op=True)
            elif isinstance(outputs, (tuple, list)):
                for idx, buffer in enumerate(outputs):
                    if DS_PIPE_VERBOSE >= 3:
                        print(f'DS BPS_SEND tensors {idx}/{len(outputs)} start', flush=True)
                    p2p.bps_send(buffer.contiguous(), self.next_stage, name, index=idx, async_op=True)
                    if DS_PIPE_VERBOSE >= 3:
                        print(f'DS BPS_SEND tensors {idx}/{len(outputs)} end', flush=True)
            else:
                raise NotImplementedError('Could not send output of type '
                                          f'{type(outputs)}')

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()
        self._profiling_func_exit()

    def _exec_bps_send_grads(self, buffer_id):
        self._profiling_func_enter('_exec_bps_send_grads')
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        assert not self.is_grad_partitioned
        assert not self.args.broadcast_grads

        name = f'grad_{buffer_id}'
        # only MP rank 0 sends the gradient
        if self.grid.get_model_parallel_rank() == 0:
            if isinstance(inputs, torch.Tensor):
                if inputs.grad is None:
                    send_data = self._allocate_zeros(inputs.size())
                else:
                    send_data = inputs.grad
                assert send_data.is_floating_point()
                assert send_data is not None
                p2p.bps_send(send_data, self.prev_stage, name, index=0, async_op=True)

            else:
                for idx, buffer in enumerate(inputs):
                    if not buffer.is_floating_point():
                        continue
                    if buffer.grad is None:
                        send_data = self._allocate_zeros(buffer.size())
                    else:
                        send_data = buffer.grad
                    assert send_data.is_floating_point()
                    assert send_data is not None
                    p2p.bps_send(send_data, self.prev_stage, name, index=idx, async_op=True)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()
        self._profiling_func_exit()

    def _exec_bps_send_partitioned_grads(self, buffer_id):
        self._profiling_func_enter('_exec_bps_send_grads')
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        assert not self.is_grad_partitioned
        assert not self.args.broadcast_grads
        assert ENABLE_BPS_PARTITION

        name = f'grad_{buffer_id}'
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is None:
                send_data = self._allocate_zeros(inputs.size())
            else:
                send_data = inputs.grad
            assert send_data.is_floating_point()
            assert send_data is not None
            p2p.bps_send(self._mp_slice(send_data), self.prev_stage, name,
                         index=0, async_op=True)
        else:
            for idx, buffer in enumerate(inputs):
                if not buffer.is_floating_point():
                    continue
                if buffer.grad is None:
                    send_data = self._allocate_zeros(buffer.size())
                else:
                    send_data = buffer.grad
                assert send_data.is_floating_point()
                assert send_data is not None
                p2p.bps_send(self._mp_slice(send_data), self.prev_stage,
                             name, index=idx, async_op=True)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()
        self._profiling_func_exit()

    def _exec_bps_sync_all(self):
        p2p.bps_sync_all()

    def _exec_bps_sync_partitioned_grads(self, buffer_id):
        name = f'grad_{buffer_id}'
        recv_buff = self.pipe_buffers['bps_grad_recv'][buffer_id]
        if isinstance(recv_buff, torch.Tensor):
            p2p.bps_sync(self.next_stage, name, index=0)
        else:
            for i in range(len(recv_buff)):
                p2p.bps_sync(self.next_stage, name, index=i)

        # all_gather the gradient from other ranks
        mp_size = self.grid.model_parallel_size
        if mp_size > 1:
            src_rank = self.grid.slice_parallel_src_id
            group = self.grid.slice_proc_group
            if isinstance(recv_buff, torch.Tensor):
                recv_buff_views = [self._mp_view(recv_buff, i) for i in range(mp_size)]
                dist.all_gather(recv_buff_views, recv_buff_views[self.mp_id].clone(),
                                group=group, async_op=False)
            else:
                for i in range(len(recv_buff)):
                    if recv_buff[i].is_floating_point():
                        recv_buff_views = [self._mp_view(recv_buff[i], j) for j in range(mp_size)]
                        dist.all_gather(recv_buff_views, recv_buff_views[self.mp_id].clone(),
                                        group=group, async_op=False)

    def _exec_bps_sync_grads(self, buffer_id):
        name = f'grad_{buffer_id}'
        recv_buff = self.pipe_buffers['bps_grad_recv'][buffer_id]
        if self.mp_id == 0:
            if isinstance(recv_buff, torch.Tensor):
                p2p.bps_sync(self.next_stage, name, index=0)
            else:
                for i in range(len(recv_buff)):
                    p2p.bps_sync(self.next_stage, name, index=i)

        # broadcast the activation at MP rank 0 to other ranks
        if self.grid.model_parallel_size > 1:
            src_rank = self.grid.slice_parallel_src_id
            group = self.grid.slice_proc_group
            if isinstance(recv_buff, torch.Tensor):        
                dist.broadcast(recv_buff, src_rank, group=group, async_op=False)
            else:
                for i in range(len(recv_buff)):
                    if recv_buff[i].is_floating_point():
                        dist.broadcast(recv_buff[i], src_rank, group=group, async_op=False)

    def _exec_bps_sync_partitioned_activations(self, buffer_id):
        recv_buff = self.pipe_buffers['bps_act_recv'][buffer_id]
        recvd = None
        src_rank = self.grid.slice_parallel_src_id
        mp_size = self.grid.model_parallel_size
        group = self.grid.slice_proc_group
        name = f'act_{buffer_id}'

        if isinstance(recv_buff, torch.Tensor):
            p2p.bps_sync(self.prev_stage, name, index=0)
            # broadcast the activation at MP rank 0 to other ranks
            if mp_size > 1:
                recv_buff_views = [self._mp_view(recv_buff, i) for i in range(mp_size)]
                dist.all_gather(recv_buff_views, recv_buff_views[self.mp_id].clone(),
                                group=group, async_op=False)
            recvd = recv_buff.clone().detach()
            recvd.requires_grad = recv_buff.is_floating_point()
        else:
            recvd = [None] * len(recv_buff)
            for i in range(len(recv_buff)):
                p2p.bps_sync(self.prev_stage, name, index=i)
                # broadcast the activation at MP rank 0 to other ranks
                if mp_size > 1:
                    recv_buff_views = [self._mp_view(recv_buff[i], j) for j in range(mp_size)]
                    dist.all_gather(recv_buff_views, recv_buff_views[self.mp_id].clone(),
                                    group=group, async_op=False)
                recvd[i] = recv_buff[i].clone().detach()
            recvd = tuple(recvd)
            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd

    def _exec_bps_sync_activations(self, buffer_id):
        recv_buff = self.pipe_buffers['bps_act_recv'][buffer_id]
        recvd = None
        src_rank = self.grid.slice_parallel_src_id
        group = self.grid.slice_proc_group
        name = f'act_{buffer_id}'

        if isinstance(recv_buff, torch.Tensor):
            if self.mp_id == 0:        
                p2p.bps_sync(self.prev_stage, name, index=0)
            # broadcast the activation at MP rank 0 to other ranks
            if self.grid.model_parallel_size > 1:
                dist.broadcast(recv_buff, src_rank, group=group, async_op=False)
            recvd = recv_buff.clone().detach()
            recvd.requires_grad = recv_buff.is_floating_point()
        else:
            recvd = [None] * len(recv_buff)
            for i in range(len(recv_buff)):
                if self.mp_id == 0:
                    p2p.bps_sync(self.prev_stage, name, index=i)
                # broadcast the activation at MP rank 0 to other ranks
                if self.grid.model_parallel_size > 1:
                    dist.broadcast(recv_buff[i], src_rank, group=group, async_op=False)
                recvd[i] = recv_buff[i].clone().detach()
            recvd = tuple(recvd)
            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd

    def _exec_bps_recv_partitioned_activations(self, buffer_id):
        self._profiling_func_enter('_exec_bps_recv_activations')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        recv_buffs = self.pipe_buffers['bps_act_recv']

        # Allocate the buffer if necessary
        if recv_buffs[buffer_id] is None:
            if recv_buffs[0] is None:
                recv_buffs[buffer_id] = self._recv_tensor_meta(self.prev_stage)
            else:
                if torch.is_tensor(recv_buffs[0]):
                    recv_buffs[buffer_id] = recv_buffs[0].clone().detach()
                else:
                    recv_buffs[buffer_id] = tuple([x.clone().detach() for x in recv_buffs[0]])

        assert not self.args.broadcast_activation
        assert not self.is_pipe_partitioned
        recv_buff = recv_buffs[buffer_id]
        name = f'act_{buffer_id}'
        if isinstance(recv_buff, torch.Tensor):
            p2p.bps_recv(self._mp_view(recv_buff, self.mp_id), self.prev_stage,
                         name, index=0, async_op=True)
        else:
            assert isinstance(recv_buff, (tuple, list))
            for idx, buffer in enumerate(recv_buff):
                assert torch.is_tensor(buffer)
                p2p.bps_recv(self._mp_view(buffer, self.mp_id), self.prev_stage,
                             name, index=idx, async_op=True)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()
        self._profiling_func_exit()

    def _exec_bps_recv_activations(self, buffer_id):
        self._profiling_func_enter('_exec_bps_recv_activations')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        recv_buffs = self.pipe_buffers['bps_act_recv']

        # Allocate the buffer if necessary
        if recv_buffs[buffer_id] is None:
            if recv_buffs[0] is None:
                recv_buffs[buffer_id] = self._recv_tensor_meta(self.prev_stage)
            else:
                if torch.is_tensor(recv_buffs[0]):
                    recv_buffs[buffer_id] = recv_buffs[0].clone().detach()
                else:
                    recv_buffs[buffer_id] = tuple([x.clone().detach() for x in recv_buffs[0]])

        assert not self.args.broadcast_activation
        assert not self.is_pipe_partitioned
        recv_buff = recv_buffs[buffer_id]
        if self.mp_id == 0:
            name = f'act_{buffer_id}'
            if isinstance(recv_buff, torch.Tensor):
                p2p.bps_recv(recv_buff, self.prev_stage, name, index=0, async_op=True)
            else:
                assert isinstance(recv_buff, (tuple, list))
                for idx, buffer in enumerate(recv_buff):
                    assert torch.is_tensor(buffer)
                    p2p.bps_recv(buffer, self.prev_stage, name, index=idx, async_op=True)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()
        self._profiling_func_exit()

    def _exec_bps_recv_partitioned_grads(self, buffer_id):
        self._profiling_func_enter('_exec_bps_recv_grads')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        grad_buffs = self.pipe_buffers['bps_grad_recv']
        # Restore partitioned output if it was partitioned and we are sending full gradients
        assert not self.is_pipe_partitioned
        assert not self.is_grad_partitioned
        assert not self.args.broadcast_grads
        assert ENABLE_BPS_PARTITION
        # Allocate gradient if necessary
        if grad_buffs[buffer_id] is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                grad_buffs[buffer_id] = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [list(t.size()) for t in outputs if t.is_floating_point()]
                grad_buffs[buffer_id] = self._allocate_buffers(sizes, num_buffers=1)[0]
        grad_buff = grad_buffs[buffer_id]
        name = f'grad_{buffer_id}'
        if isinstance(grad_buff, torch.Tensor):
            p2p.bps_recv(self._mp_view(grad_buff, self.mp_id), self.next_stage,
                         name, index=0, async_op=True)
        else:
            assert isinstance(outputs, tuple)
            recv_idx = 0
            for idx, buffer in enumerate(grad_buff):
                p2p.bps_recv(self._mp_view(buffer, self.mp_id), self.next_stage,
                             name, index=recv_idx, async_op=True)
                recv_idx += 1

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()
        self._profiling_func_exit()

    def _exec_bps_recv_grads(self, buffer_id):
        self._profiling_func_enter('_exec_bps_recv_grads')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        grad_buffs = self.pipe_buffers['bps_grad_recv']
        # Restore partitioned output if it was partitioned and we are sending full gradients
        assert not self.is_pipe_partitioned
        assert not self.is_grad_partitioned
        assert not self.args.broadcast_grads
        # Allocate gradient if necessary
        if grad_buffs[buffer_id] is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                grad_buffs[buffer_id] = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [list(t.size()) for t in outputs if t.is_floating_point()]
                grad_buffs[buffer_id] = self._allocate_buffers(sizes, num_buffers=1)[0]
        grad_buff = grad_buffs[buffer_id]
        name = f'grad_{buffer_id}'
        if isinstance(grad_buff, torch.Tensor):
            if self.mp_id == 0:
                p2p.bps_recv(grad_buff, self.next_stage, name, index=0, async_op=True)
        else:
            assert isinstance(outputs, tuple)
            recv_idx = 0
            if self.mp_id == 0:
                for idx, buffer in enumerate(grad_buff):
                    p2p.bps_recv(buffer, self.next_stage, name, index=recv_idx, async_op=True)
                    recv_idx += 1

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()
        self._profiling_func_exit()

    def _exec_optimizer_step(self, lr_kwargs=None):
        self._profiling_func_enter('_exec_optimizer_step')
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        if self.global_rank == 0 and MEGATRON_DEBUG_GRAD:
             params = list(self.module.named_parameters())
             for i in (0, 1, -2, -1):
                 p = params[i]
                 if p[1] is None:
                     print(f'name={p[0]} | None', flush=True)
                 elif p[1].grad is None:
                     print(f'name={p[0]} | weight={p[1].mean()}', flush=True)
                 else:
                     print(f'name={p[0]} | weight={p[1].norm()} | grad={p[1].grad.norm()}', flush=True)
             params_w_grad = []
             params_wo_grad = []
             for p in params:
                 if p[1].grad is not None:
                     params_w_grad.append(p[0])
                 else:
                     params_wo_grad.append(p[0])

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/lr',
                                        self.get_lr()[0],
                                        self.global_samples)]
                if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                    self.summary_events.append((f'Train/Samples/loss_scale',
                                                self.optimizer.cur_scale,
                                                self.global_samples))
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'backward_allreduce_microstep',
                    'backward_tied_allreduce_microstep',
                    'step_microstep'
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'step'
                ])
        self._profiling_func_exit()

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, fp16=None, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            fp16 (bool): whether to use FP16. default: defer to self.fp16_enabled()
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """

        if fp16 is None:
            fp16 = self.fp16_enabled()

        if fp16:
            return torch.zeros(shape, dtype=torch.half, device=self.device, **kwargs)
        else:
            return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_zeros2(self, shape, dtype, **kwargs):
        return torch.zeros(shape, dtype=dtype, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffer2(self, shape, dtype, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros2(shape, dtype, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape in shapes:
                buffer.append(self._allocate_zeros(shape, requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def _allocate_buffers2(self, shapes, dtypes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for i in range(len(shapes)):
                buffer.append(self._allocate_zeros2(shapes[i], dtypes[i], requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.BytePSForwardPass: _exec_bps_forward_pass,
        schedule.BytePSBackwardPass: _exec_bps_backward_pass,
        schedule.BytePSSendActivation: _exec_bps_send_partitioned_activations if ENABLE_BPS_PARTITION else _exec_bps_send_activations,
        schedule.BytePSRecvActivation: _exec_bps_recv_partitioned_activations if ENABLE_BPS_PARTITION else _exec_bps_recv_activations,
        schedule.BytePSSyncActivation: _exec_bps_sync_partitioned_activations if ENABLE_BPS_PARTITION else _exec_bps_sync_activations,
        schedule.BytePSSyncGrad: _exec_bps_sync_partitioned_grads if ENABLE_BPS_PARTITION else _exec_bps_sync_grads,
        schedule.BytePSSendGrad: _exec_bps_send_partitioned_grads if ENABLE_BPS_PARTITION else _exec_bps_send_grads,
        schedule.BytePSRecvGrad: _exec_bps_recv_partitioned_grads if ENABLE_BPS_PARTITION else _exec_bps_recv_grads,
        schedule.BytePSSyncAll: _exec_bps_sync_all
    }

    def _exec_schedule(self, pipe_schedule):
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        # For each step in the schedule
        has_optim_step = False
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if isinstance(cmd, schedule.OptimizerStep):
                    has_optim_step = True
                if DS_PIPE_VERBOSE:
                    if "buffer_id" in cmd.kwargs:
                        print(f'[{self.grid.get_global_rank()}] | cmd={cmd.__class__.__name__} | {cmd.kwargs["buffer_id"]}', flush=True)
                    else:
                        print(f'[{self.grid.get_global_rank()}] | cmd={cmd.__class__.__name__}', flush=True)
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}'
                    )

                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                self._exec_instr(**cmd.kwargs)
        # check for anormalies
        if isinstance(pipe_schedule, (schedule.BytePSTrainSchedule, schedule.TrainSchedule)):
            assert has_optim_step
# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright 2019 The Microsoft DeepSpeed Team
import os

import re as regex

from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist

from math import floor

from deepspeed.utils import logger
from deepspeed.runtime import utils as ds_utils
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.pipe import PipelineModule,LayerSpec, TiedLayerSpec
from .topology import PipeDataParallelTopology, PipelineParallelGrid

class VeGiantModule(PipelineModule):
    def __init__(self,
                 layers,
                 num_stages=None,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 grid=None,
                 partition_method='parameters',
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=checkpointing.checkpoint):
        """Modules to be parallelized with pipeline parallelism.

        The key constraint that enables pipeline parallelism is the
        representation of the forward pass as a sequence of layers
        and the enforcement of a simple interface between them. The
        forward pass is implicitly defined by the module ``layers``. The key
        assumption is that the output of each layer can be directly fed as
        input to the next, like a ``torch.nn.Sequence``. The forward pass is
        implicitly:

        .. code-block:: python

            def forward(self, inputs):
                x = inputs
                for layer in self.layers:
                    x = layer(x)
                return x

        Args:
            layers (Iterable): A sequence of layers defining pipeline structure. Can be a ``torch.nn.Sequential`` module.
            num_stages (int, optional): The degree of pipeline parallelism. If not specified, ``topology`` must be provided.
            topology (``deepseed.pipe.ProcessTopology``, optional): Defines the axes of parallelism axes for training. Must be provided if ``num_stages`` is ``None``.
            loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
            base_seed (int, optional): [description]. Defaults to 1234.
            partition_method (str, optional): [description]. Defaults to 'parameters'.
            activation_checkpoint_interval (int, optional): The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.
            activation_checkpoint_func (callable, optional): The function to use for activation checkpointing. Defaults to ``deepspeed.checkpointing.checkpoint``.
        """

        super(PipelineModule, self).__init__()

        topology = grid.topology() if grid is not None else None

        if num_stages is None and topology is None:
            raise RuntimeError('must provide num_stages or topology')

        self.micro_offset = 0

        self.loss_fn = loss_fn

        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(
                f'SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}'
            )

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)

        if topology:
            self._topo = topology
            self.num_stages = self._topo.get_dim('pipe')
        else:
            self.num_stages = num_stages
            if topology is None:
                if self.world_size % self.num_stages != 0:
                    raise RuntimeError(
                        f'num_stages ({self.num_stages}) must divide distributed world size ({self.world_size})'
                    )
                dp = self.world_size // num_stages
                topology = PipeDataParallelTopology(num_pp=num_stages, num_dp=dp)
                self._topo = topology

        # Contruct communicators for pipeline topology
        self._grid = grid if grid is not None else PipelineParallelGrid(process_group=self.world_group, topology=self._topo)

        self.stage_id = self._topo.get_coord(self.global_rank).pipe

        # Initialize partition information
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method)

        self.forward_funcs = []
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        # Offset the random seed by the stage ID.
        #newseed = torch.cuda.initial_seed() + self._grid.get_stage_id()
        #ds_utils.set_random_seed(newseed)

        #with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        self._build()
        self.to('cuda')

        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()

        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func

    def _build(self):
        specs = self._layer_specs

        for local_idx, layer in enumerate(specs[self._local_start:self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    ds_utils.set_random_seed(self.base_seed + layer_idx)

            # Recursively build PipelineModule objects
            if isinstance(layer, PipelineModule):
                raise NotImplementedError('RECURSIVE BUILD NOT YET IMPLEMENTED')

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, nn.Module):
                name = str(layer_idx)
                self.forward_funcs.append(layer)
                self.add_module(name, layer)

            # TiedLayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, TiedLayerSpec):
                # Build and register the module if we haven't seen it before.
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build()
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr

                if layer.forward_fn is None:
                    # Just use forward()
                    self.forward_funcs.append(self.tied_modules[layer.key])
                else:
                    # User specified fn with args (module, input)
                    self.forward_funcs.append(
                        partial(layer.forward_fn,
                                self.tied_modules[layer.key]))

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                self.forward_funcs.append(module)
                self.add_module(name, module)

            # Last option: layer may be a functional (e.g., lambda). We do nothing in
            # that case and just use it in forward()
            else:
                self.forward_funcs.append(layer)

        # All pipeline parameters should be considered as model parallel in the context
        # of our FP16 optimizer
        for p in self.parameters():
            p.model_parallel = True

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self._layer_specs)
        for idx, layer in enumerate(self._layer_specs):
            if isinstance(layer, LayerSpec):
                l = layer.build()
                params = filter(lambda p: p.requires_grad, l.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
            elif isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def _find_layer_type(self, layername):
        idxs = []
        typeregex = regex.compile(layername, regex.IGNORECASE)
        for idx, layer in enumerate(self._layer_specs):
            name = None
            if isinstance(layer, LayerSpec):
                name = layer.typename.__name__
            elif isinstance(layer, nn.Module):
                name = layer.__class__.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    continue
            if typeregex.search(name):
                idxs.append(idx)

        if len(idxs) == 0:
            raise RuntimeError(
                f"Partitioning '{layername}' found no valid layers to partition.")
        return idxs

    def forward(self, forward_input):
        # We need to offset the seed by the microbatch ID. Save it in a local var to
        # ensure it is preserved in the closure. Otherwise checkpointed forward funcs
        # will see a different offset.
        self.micro_offset += 1

        def exec_range_func(start, end):
            ''' Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            '''
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (self.base_seed *
                                    local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)

                    inputs = layer(inputs)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            x = func(forward_input)
        else:
            num_layers = len(self.forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(start_idx + self.activation_checkpoint_interval,
                              num_layers)

                funcs = self.forward_funcs[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x, )

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx,
                                        end_idx),
                        *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x

    def _partition_uniform(self, num_items, num_parts):
        # print(f'enter _partition_uniform', flush=True)
        parts = [0] * (num_parts + 1)
        if num_items <= num_parts:
            for p in range(num_parts + 1):
                parts[p] = min(p, num_items)
            return parts
        expected_chunksize = num_items / num_parts
        for p in range(num_parts):
            parts[p] = min(floor(expected_chunksize * p), num_items)
        parts[num_parts] = num_items
        return parts

    def _partition_balanced(self, weights, num_parts, eps=1e-3):
        num_items = len(weights)
        # First check for the trivial edge case
        if num_items <= num_parts:
            return self._partition_uniform(num_items, num_parts)

        weights_ = ds_utils.prefix_sum_inc(weights)

        # Find the smallest bottleneck (weight of heaviest partition)
        bottleneck = ds_utils._rb_partition_balanced(weights_, num_parts, eps=eps)

        # Now compute that partitioning
        parts, success = ds_utils._lprobe(weights_, num_parts, bottleneck)
        assert success

        return parts

    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}')

        method = method.lower()

        # Each stage gets a simple uniform number of layers.
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = self._partition_uniform(num_items=num_layers,
                                            num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = self._partition_balanced(weights=param_counts,
                                                     num_parts=num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            else:
                self.parts = self._partition_balanced(weights=binary_weights,
                                                         num_parts=num_stages)
        elif method.startswith('manual:'):
            msplit = method.split(':')
            layernum = int(msplit[1])
            layerparts = msplit[2].split(',')
            assert len(self._layer_specs) == layernum # failsafe check for layer num
            assert num_stages == len(layerparts)-1 # failsafe check for num stages
            self.parts = list(map(int, layerparts))
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def allreduce_tied_weight_gradients(self):
        '''All reduce the gradients of the tied weights between tied stages'''
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            dist.all_reduce(weight.grad, group=comm['group'])

    def _synchronize_tied_weights(self):
        for key, comm in self.tied_comms.items():
            dist.broadcast(
                getattr(comm['module'],
                        comm['weight_attr']),
                src=min(comm['ranks']),
                group=comm['group'],
            )

    def _index_tied_modules(self):
        ''' Build communication structures for tied modules. '''
        tied_comms = {}
        if self._topo.get_dim('pipe') == 1:
            return tied_comms

        specs = self._layer_specs
        tie_keys = set(s.key for s in specs if isinstance(s, TiedLayerSpec))
        for key in tie_keys:
            # Find the layers that the tied module appears in
            tied_layers = []
            for idx, layer in enumerate(specs):
                if isinstance(layer, TiedLayerSpec) and layer.key == key:
                    tied_layers.append(idx)
            # Find all stages with this tied module
            # TODO: Would be nice to remove the nested data/model parallelism loops and
            # TODO: instead generalize in some way, since we really just care about the
            # TODO: stage that owns the tied layer. Then loop over each (dp, mp, ...)
            # TODO: fiber to generate process groups.
            tied_stages = set(self.stage_owner(idx) for idx in tied_layers)
            for dp in range(self._grid.data_parallel_size):
                for mp in range(self._grid.model_parallel_size):
                    tied_ranks = []
                    for s in sorted(tied_stages):
                        if self._grid.model_parallel_size > 1:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp,
                                                           model=mp))
                        else:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp))
                    group = dist.new_group(ranks=tied_ranks)

                    # Record this tied module if we own a local copy of it.
                    if self.global_rank in tied_ranks:
                        assert key in self.tied_modules
                        if key in self.tied_modules:
                            tied_comms[key] = {
                                'ranks': tied_ranks,
                                'group': group,
                                'weight_attr': self.tied_weight_attrs[key],
                                'module': self.tied_modules[key],
                            }
                            # Only count the tied module once in the eyes of the FP16 optimizer
                            if self.global_rank != tied_ranks[0]:
                                for p in self.tied_modules[key].parameters():
                                    p.model_parallel = False
        '''
        if len(tied_comms) > 0:
            print(f'RANK={self.global_rank} tied_comms={tied_comms}')
        '''

        return tied_comms

    def partitions(self):
        return self.parts

    def stage_owner(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers
        for stage in range(self._topo.get_dim('pipe')):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f'Layer {layer_idx} not owned? parts={self.parts}')

    def _set_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        self._local_start = start
        self._local_stop = stop

    def set_checkpoint_interval(self, interval):
        assert interval >= 0
        self.checkpoint_interval = interval

    def topology(self):
        """ ProcessTopology object to query process mappings. """
        return self._topo

    def mpu(self):
        return self._grid

    def num_pipeline_stages(self):
        return self._topo.get_dim('pipe')

    def ckpt_prefix(self, checkpoints_path, tag):
        """Build a prefix for all checkpoint files written by this module. """
        # All checkpoint files start with this
        rank_name = 'module'

        # Data parallelism is omitted from the naming convention because we are agnostic
        # to this in the checkpoint.
        omit_dims = frozenset(['data'])
        axes = [a for a in self._grid._topo.get_axis_names() if a not in omit_dims]
        for dim in axes:
            rank = getattr(self._grid._topo.get_coord(rank=self.global_rank), dim)
            rank_name += f'-{dim}_{rank:02d}'

        ckpt_name = os.path.join(checkpoints_path, str(tag), rank_name)
        return ckpt_name

    def ckpt_layer_path(self, ckpt_dir, local_layer_idx):
        """Customize a prefix for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}')
        rank_repr = self._grid._topo.get_rank_repr(rank=self.global_rank)
        if rank_repr is not '':
            layer_ckpt_path += f'-{rank_repr}'
        layer_ckpt_path += '-model_states.pt'
        return layer_ckpt_path

    def save_state_dict(self, save_dir):
        if self._grid.data_parallel_id != 0:
            return

        os.makedirs(save_dir, exist_ok=True)
        layer_offset = self._local_start
        for idx, layer in enumerate(self.forward_funcs):
            model_ckpt_path = self.ckpt_layer_path(save_dir, idx)
            if not hasattr(layer, 'state_dict'):
                continue
            torch.save(layer.state_dict(), model_ckpt_path)

    def load_state_dir(self, load_dir, strict=True):
        rank = dist.get_rank()

        layer_offset = self._local_start
        for idx, layer in enumerate(self.forward_funcs):
            # Functions, etc. will not have state_dicts
            if not hasattr(layer, 'load_state_dict'):
                continue

            model_ckpt_path = self.ckpt_layer_path(load_dir, idx)
            layer.load_state_dict(torch.load(model_ckpt_path,
                                             map_location=lambda storage,
                                             loc: storage),
                                  strict=strict)
            if self._grid.data_parallel_id == 0:
                logger.info(
                    f'RANK={self.global_rank} Loaded layer={idx+layer_offset} file={model_ckpt_path}'
                )

        self._synchronize_tied_weights()

    def _is_checkpointable(self, funcs):
        if self.__class__.__name__ == 'GPT2ModelPipe':
            return all('ParallelTransformerLayerPipe' in f.__class__.__name__
                       for f in funcs)

        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)
# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
from deepspeed.runtime.pipe.schedule import (
    BufferOpInstruction,PipeInstruction,
    ReduceTiedGrads,ReduceGrads,OptimizerStep,
    LoadMicroBatch,PipeSchedule,TrainSchedule,
)

import os

BYTEPS_REDUCED_MEM = os.environ.get('BYTEPS_REDUCED_MEM', '1') != '0'

class BytePSInferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """
    def __init__(self, micro_batches, stages, stage_id, prefetch=True):
        super().__init__(micro_batches, stages, stage_id)
        self.prefetch = prefetch

    def steps(self):
        """"""
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            buffer_id = micro_batch_id % self.num_pipe_buffers()
            batch_is_valid = self._valid_micro_batch(micro_batch_id)

            if not self.prefetch:    
                if batch_is_valid:
                    if self.is_first_stage or self.is_last_stage:
                        cmds.append(LoadMicroBatch(buffer_id))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSRecvActivation(buffer_id))
                        cmds.append(BytePSSyncActivation(buffer_id))
                    cmds.append(BytePSForwardPass(buffer_id))
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSendActivation(buffer_id))
            else:
                next_buffer_id = (micro_batch_id + 1) % self.num_pipe_buffers()
                next_batch_is_valid = self._valid_micro_batch(micro_batch_id + 1)
                # micro_batch starts at 0. Get the current batch, and start prefetching
                if micro_batch_id == 0:
                    if self.is_first_stage or self.is_last_stage:
                        cmds.append(LoadMicroBatch(buffer_id))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSRecvActivation(buffer_id))
                        if next_batch_is_valid:
                            cmds.append(BytePSRecvActivation(next_buffer_id))
                        cmds.append(BytePSSyncActivation(buffer_id))
                    cmds.append(BytePSForwardPass(buffer_id))
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSendActivation(buffer_id))
                elif batch_is_valid:
                    # After micro_batch 0, we prefetch the next one,
                    # and wait for the current one
                    if self._valid_stage(self.prev_stage) and next_batch_is_valid:
                        cmds.append(BytePSRecvActivation(next_buffer_id))
                    if self.is_first_stage or self.is_last_stage:
                        cmds.append(LoadMicroBatch(buffer_id))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSSyncActivation(buffer_id))
                    cmds.append(BytePSForwardPass(buffer_id))
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSendActivation(buffer_id))

            yield cmds

    def num_pipe_buffers(self):
        """Only `self.micro_batches` pipeline buffers are required for inferencing.

        Returns:
            ``self.micro_batches``
        """
        buffers = min(self.micro_batches, self.stages * 2)
        if BYTEPS_REDUCED_MEM:
            buffers = min(self.stages + 1, self.micro_batches)
        return max(2, buffers)


class BytePSTrainSchedule(TrainSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def __init__(self, micro_batches, stages, stage_id, prefetch=True):
        super().__init__(micro_batches, stages, stage_id)
        self.prefetch = prefetch and micro_batches > 1
        if not self.prefetch:
            print('BYTEPS NO PREFETCH STEPS', flush=True)

    def steps(self):
        if self.prefetch:
            return self._steps()
        else:
            return self._steps_no_prefetch()

    def _steps(self):
        """"""
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            cmds = []
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            batch_is_valid = self._valid_micro_batch(micro_batch_id)
            if not batch_is_valid:
                if step_id == total_steps - 1:
                    cmds.append(BytePSSyncAll())
                    cmds.append(ReduceTiedGrads())
                    cmds.append(ReduceGrads())
                    cmds.append(OptimizerStep())
                    yield cmds
                continue
            curr_buffer = self._buffer_idx(micro_batch_id)

            # try to find the next valid batch
            next_step_id = step_id + 1
            next_micro_batch_id, next_is_forward, next_batch_is_valid = None, None, None
            while next_step_id < total_steps:
                next_micro_batch_id, next_is_forward = self._step_to_micro_batch(next_step_id)
                next_batch_is_valid = self._valid_micro_batch(next_micro_batch_id)
                if next_batch_is_valid:
                    break
                next_step_id += 1

            next_buffer = None
            if next_batch_is_valid:
                next_buffer = self._buffer_idx(next_micro_batch_id)

            if micro_batch_id == 0 and is_forward:
                # first/last stage loads
                if self.stage_id == 0 or self.stage_id == self.stages - 1:
                    cmds.append(LoadMicroBatch(curr_buffer))
                # fetch
                if self._valid_stage(self.prev_stage):
                    cmds.append(BytePSRecvActivation(curr_buffer))
                # pre-fetch
                if next_batch_is_valid:
                    if self._valid_stage(self.prev_stage) and next_is_forward:
                        cmds.append(BytePSRecvActivation(next_buffer))
                    if self._valid_stage(self.next_stage) and not next_is_forward:
                        cmds.append(BytePSRecvGrad(next_buffer))
                # sync and compute
                if self._valid_stage(self.prev_stage):
                    cmds.append(BytePSSyncActivation(curr_buffer))
                cmds.append(BytePSForwardPass(curr_buffer))
                if self._valid_stage(self.next_stage):
                    cmds.append(BytePSSendActivation(curr_buffer))
            else:
                # prefetch
                if next_batch_is_valid:
                    if self._valid_stage(self.prev_stage) and next_is_forward:
                        cmds.append(BytePSRecvActivation(next_buffer))
                    if self._valid_stage(self.next_stage) and not next_is_forward:
                        cmds.append(BytePSRecvGrad(next_buffer))
                if is_forward:
                    if self.stage_id == 0 or self.stage_id == self.stages - 1:
                        # First/last stage loads
                        cmds.append(LoadMicroBatch(curr_buffer))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSSyncActivation(curr_buffer))
                    cmds.append(BytePSForwardPass(curr_buffer))
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSendActivation(curr_buffer))
                else:
                    if self._valid_stage(self.next_stage):
                        cmds.append(BytePSSyncGrad(curr_buffer))
                    cmds.append(BytePSBackwardPass(curr_buffer))
                    if self._valid_stage(self.prev_stage):
                        cmds.append(BytePSSendGrad(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(BytePSSyncAll())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            yield cmds

    def _steps_no_prefetch(self):
        """"""
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            cmds = []
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            batch_is_valid = self._valid_micro_batch(micro_batch_id)
            if not batch_is_valid:
                if step_id == total_steps - 1:
                    cmds.append(BytePSSyncAll())
                    cmds.append(ReduceTiedGrads())
                    cmds.append(ReduceGrads())
                    cmds.append(OptimizerStep())
                    yield cmds
                continue

            curr_buffer = self._buffer_idx(micro_batch_id)

            if is_forward:
                if self._valid_stage(self.prev_stage):
                    cmds.append(BytePSRecvActivation(curr_buffer))
                    cmds.append(BytePSSyncActivation(curr_buffer))
                if self.stage_id == 0 or self.stage_id == self.stages - 1:
                    # First/last stage loads
                    cmds.append(LoadMicroBatch(curr_buffer))
                cmds.append(BytePSForwardPass(curr_buffer))
                if self._valid_stage(self.next_stage):
                    cmds.append(BytePSSendActivation(curr_buffer))
            else:
                if self._valid_stage(self.next_stage):
                    cmds.append(BytePSRecvGrad(curr_buffer))
                    cmds.append(BytePSSyncGrad(curr_buffer))
                cmds.append(BytePSBackwardPass(curr_buffer))
                if self._valid_stage(self.prev_stage):
                    cmds.append(BytePSSendGrad(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(BytePSSyncAll())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            yield cmds

    def num_pipe_buffers(self):
        """As many buffers as the distance from this stage to the last stage.
        """
        buffers = min(self.micro_batches, self.stages * 2)
        if BYTEPS_REDUCED_MEM:
            buffers = min(self.stages + 1, self.micro_batches)
        return max(2, buffers)


class BytePSSendActivation(BufferOpInstruction):
    pass

class BytePSRecvActivation(BufferOpInstruction):
    pass

class BytePSSyncActivation(BufferOpInstruction):
    pass

class BytePSSyncGrad(BufferOpInstruction):
    pass

class BytePSSendGrad(BufferOpInstruction):
    pass

class BytePSRecvGrad(BufferOpInstruction):
    pass# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright 2019 The Microsoft DeepSpeed Team
'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import os
import torch
import torch.distributed as dist
from deepspeed.utils import logger, log_dist

ENABLE_PYTORCH_BROADCAST = os.environ.get("ENABLE_PYTORCH_BROADCAST", "0") != "0"

try:
    if not ENABLE_PYTORCH_BROADCAST:
        import byteps.torch as bps
    else:
        print("BytePS import is disabled", flush=True)
        bps = None
except ImportError:
    print("BytePS is not installed")
    bps = None

_groups = None
_grid = None

DS_PIPE_VERBOSE = os.environ.get('DS_PIPE_VERBOSE', "0") != "0"

did_recv = False
send_stream = None
recv_stream = None 


bps_send_handles = {}
bps_recv_handles = {}


#initializes adjacent process groups
#run this only after torch.distributed.init_process_group() has been called
def init_process_groups(grid):
    global _groups, _grid
    _grid = grid

    assert _grid.pipe_parallel_size > 1, "There is no model parallelism"

    _groups = [dist.new_group(ranks=group) for group in _grid.p2p_groups]


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage, async_op=False):
    global _groups

    async_op = False
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)

    import torch
    if tensor.dtype != torch.float32 and DS_PIPE_VERBOSE:
        print('warning: p2p send', tensor.dtype, tensor.shape, flush=True)
    return _send(tensor, src_rank, group, async_op)

def _bps_get_name(src, dest, name, suffix):
    return "_".join([str(src), str(dest), str(name), str(suffix)])

def bps_send(tensor, dest_stage, name, index, async_op=True):
    global bps_send_handles

    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    name = _bps_get_name(src_rank, dest_rank, name, index)
    if name not in bps_send_handles:
        # XXX hard-code max number of tensors for this name
        bps_send_handles[name] = [None] * 10
    else:
        handle = bps_send_handles[name][index]
        if handle is not None:
            bps.synchronize(handle)
    handle = bps.send_async(tensor, dest_rank, name=name)
    # XXX
    if not async_op:
        bps.synchronize(handle)
    else:
        bps_send_handles[name][index] = handle
    return tensor

def bps_sync(src_stage, name, index=0):
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    name = _bps_get_name(src_rank, dest_rank, name, index)
    if name in bps_recv_handles:
        handle = bps_recv_handles[name][index]
        if handle is not None:
            bps.synchronize(handle)

def bps_sync_all():
    for name, handles in bps_send_handles.items():
        for handle in handles:
            if handle is not None:
                bps.synchronize(handle)

    for name, handles in bps_recv_handles.items():
        for handle in handles:
            if handle is not None:
                bps.synchronize(handle)

def bps_recv(tensor, src_stage, name, index=0, async_op=True):
    global bps_recv_handles

    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    name = _bps_get_name(src_rank, dest_rank, name, index)
    if name not in bps_recv_handles:
        # XXX hard-code max number of tensors for this name
        bps_recv_handles[name] = [None] * 10
    else:
        handle = bps_recv_handles[name][index]
        if handle is not None:
            bps.synchronize(handle)
    handle = bps.recv_async(tensor, src_rank, name=name)
    if not async_op:
        bps.synchronize(handle)
    else:
        bps_recv_handles[name][index] = handle
    return tensor


def _send(tensor, src_rank, group, async_op):
    global did_recv
    return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)

def send_grads(tensor, grid, async_op=False):
    async_op = False
    if  grid.send_grads_src_rank == grid.global_rank:
        # print(f'start rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}, send_grad_groups: {grid.send_grads_proc_group}', flush=True)
        _send(tensor, grid.send_grads_src_rank, grid.send_grads_proc_group, async_op)
        # print(f'finis rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}', flush=True)
    else:
        # print(f'finish fast rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}', flush=True)
        pass

def _recv(tensor, src_rank, group, async_op):
    global did_recv
    tensor = dist.broadcast(tensor, src_rank, group=group, async_op=async_op)
    did_recv = True
    return tensor

def recv_grads(tensor, grid, async_op=False):
    async_op = False
    # print(f'start rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _recv_grad_src_rank: {grid.recv_grads_src_rank}, recv group: {grid.recv_grads_group}, recv_grad_groups: {grid.recv_grads_proc_group}', flush=True)
    _recv(tensor, grid.recv_grads_src_rank, grid.recv_grads_proc_group, async_op)
    # print(f'finish rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _recv_grad_src_rank: {grid.recv_grads_src_rank}, recv group: {grid.recv_grads_group}', flush=True)


def send_activations(tensor, grid, async_op=False):
    async_op = False
    if  grid.send_activation_src_rank == grid.global_rank:
        # print(f'start rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}, send_grad_groups: {grid.send_grads_proc_group}', flush=True)
        _send(tensor, grid.send_activation_src_rank, grid.send_activation_proc_group, async_op)
        # print(f'finis rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}', flush=True)
    else:
        # print(f'finish fast rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _send_grad_src_rank: {grid.send_grads_src_rank}, send group: {grid.send_grads_group}', flush=True)
        pass 

def recv_activations(tensor, grid, async_op=False):
    async_op = False
    # print(f'start rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _recv_grad_src_rank: {grid.recv_grads_src_rank}, recv group: {grid.recv_grads_group}, recv_grad_groups: {grid.recv_grads_proc_group}', flush=True)
    _recv(tensor, grid.recv_activation_src_rank, grid.recv_activation_proc_group, async_op)
    # print(f'finish rank: {grid.global_rank}, stage_id: {grid.stage_id}, mp_id: {grid.model_parallel_id}, _recv_grad_src_rank: {grid.recv_grads_src_rank}, recv group: {grid.recv_grads_group}', flush=True)

def recv(tensor, src_stage, async_op=False):
    global _groups
    global did_recv

    async_op = False
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)
    return _recv(tensor, src_rank, group, async_op)


def barrier(stage_id):
    global _groups, _grid
    group_id = _grid.stage_to_global(stage_id=stage_id)
    if (dist.get_rank() >= 0):
        print("Barrier Group ID", group_id)
        print("Barrier Group", _grid.p2p_groups[group_id])
    dist.barrier(group=_groups[group_id])
    if (dist.get_rank() >= 0):
        print("Exiting Barrier ", group_id)


def _get_send_recv_group(src_stage, dest_stage):
    '''the group id is always the smaller rank unless its a wrap around'''

    stage_id = None

    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1

    if (src_stage == first_stage and dest_stage == last_stage
            or dest_stage == first_stage and src_stage == last_stage):
        stage_id = last_stage
    elif src_stage > dest_stage:
        stage_id = dest_stage
    else:
        stage_id = src_stage
    '''group_id corresponds to group of [group_id, group_id+1]
     unless group_id is the rank of the last stage
     in which case group_id correspods to group[group_id-num_stages+1, group_id]
     '''
    group_id = _grid.stage_to_global(stage_id=stage_id)

    return _groups[group_id]

class BytePSForwardPass(BufferOpInstruction):
    pass

class BytePSBackwardPass(BufferOpInstruction):
    pass

class BytePSSyncAll(PipeInstruction):
    pass

# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright 2019 The Microsoft DeepSpeed Team

from deepspeed.utils import log_dist

import torch.distributed as dist

from collections import namedtuple
from itertools import product as cartesian_product
import logging, os

import torch

class ProcessTopology:
    """ Manages the mapping of n-dimensional Cartesian coordinates to linear
    indices. This mapping is used to map the rank of processes to the grid
    for various forms of parallelism.

    Each axis of the tensor is accessed by its name. The provided ordering
    of the axes defines the layout of the topology. ProcessTopology uses a "row-major"
    layout of the tensor axes, and so axes=['x', 'y'] would map coordinates (x,y) and
    (x,y+1) to adjacent linear indices. If instead axes=['y', 'x'] was used, coordinates
    (x,y) and (x+1,y) would be adjacent.

    Some methods return ProcessCoord namedtuples.
    """
    def __init__(self, axes, dims):
        """Create a mapping of n-dimensional tensor coordinates to linear indices.

        Arguments:
            axes (list): the names of the tensor axes
            dims (list): the dimension (length) of each axis of the topology tensor
        """

        self.axes = axes  # names of each topology axis
        self.dims = dims  # length of each topology axis

        # This is actually a class that lets us hash {'row':3, 'col':2} mappings
        self.ProcessCoord = namedtuple('ProcessCoord', axes)

        self.mapping = {}
        ranges = [range(d) for d in dims]
        # example: 1, (0,0,1)
        for global_rank, coord in enumerate(cartesian_product(*ranges)):
            key = {axis: coord[self.axes.index(axis)] for axis in self.axes}
            key = self.ProcessCoord(**key)
            # for example, {ProcessCoord(row=0, col=1) : 1}
            self.mapping[key] = global_rank

    def get_rank(self, **coord_kwargs):
        """Return the global rank of a process via its coordinates.

        Coordinates are specified as kwargs. For example:

            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_rank(x=0, y=1)
            1
        """
        if len(coord_kwargs) != len(self.axes):
            raise ValueError('get_rank() does not support slices. Use filter_match())')

        key = self.ProcessCoord(**coord_kwargs)
        assert key in self.mapping, f'key {kwargs} invalid'
        return self.mapping[key]

    def get_axis_names(self):
        """Return a list of the axis names in the ordering of the topology. """
        return self.axes

    def get_rank_repr(self,
                      rank,
                      omit_axes=['data',
                                 'pipe'],
                      inner_sep='_',
                      outer_sep='-'):
        """Return a string representation of a rank.

        This method is primarily used for checkpointing model data.

        For example:
            >>> topo = Topo(axes=['a', 'b'], dims=[2, 2])
            >>> topo.get_rank_repr(rank=3)
            'a_01-b_01'
            >>> topo.get_rank_repr(rank=3, omit_axes=['a'])
            'b_01'

        Args:
            rank (int): A rank in the topology.
            omit_axes (list, optional): Axes that should not be in the representation. Defaults to ['data', 'pipe'].
            inner_sep (str, optional): [description]. Defaults to '_'.
            outer_sep (str, optional): [description]. Defaults to '-'.

        Returns:
            str: A string representation of the coordinate owned by ``rank``.
        """
        omit_axes = frozenset(omit_axes)
        axes = [a for a in self.get_axis_names() if a not in omit_axes]
        names = []
        for ax in axes:
            ax_rank = getattr(self.get_coord(rank=rank), ax)
            names.append(f'{ax}{inner_sep}{ax_rank:02d}')
        return outer_sep.join(names)

    def get_dim(self, axis):
        """Return the number of processes along the given axis.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_dim('y')
            3
        """
        if axis not in self.axes:
            return 0
        return self.dims[self.axes.index(axis)]

    def get_coord(self, rank):
        """Return the coordinate owned by a process rank.

        The axes of the returned namedtuple can be directly accessed as members. For
        example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> coord = X.get_coord(rank=1)
            >>> coord.x
            0
            >>> coord.y
            1
        """
        for coord, idx in self.mapping.items():
            if idx == rank:
                return coord
        raise ValueError(f'rank {rank} not found in topology.')

    def get_axis_comm_lists(self, axis):
        """ Construct lists suitable for a communicator group along axis ``axis``.

        Example:
            >>> topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> topo.get_axis_comm_lists('pipe')
            [
                [0, 4], # data=0, model=0
                [1, 5], # data=0, model=1
                [2, 6], # data=1, model=0
                [3, 7], # data=1, model=1
            ]

        Returns:
            A list of lists whose coordinates match in all axes *except* ``axis``.
        """

        # We don't want to RuntimeError because it allows us to write more generalized
        # code for hybrid parallelisms.
        if axis not in self.axes:
            return []

        # Grab all axes but `axis`
        other_axes = [a for a in self.axes if a != axis]

        lists = []

        # Construct all combinations of coords with other_axes
        ranges = [range(self.get_dim(a)) for a in other_axes]
        for coord in cartesian_product(*ranges):
            other_keys = {a: coord[other_axes.index(a)] for a in other_axes}
            # now go over all ranks in `axis`.
            sub_list = []
            for axis_key in range(self.get_dim(axis)):
                key = self.ProcessCoord(**other_keys, **{axis: axis_key})
                sub_list.append(self.mapping[key])
            lists.append(sub_list)

        return lists

    def filter_match(self, **filter_kwargs):
        """Return the list of ranks whose coordinates match the provided criteria.

        Example:
            >>> X = ProcessTopology(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> X.filter_match(pipe=0, data=1)
            [2, 3]
            >>> [X.get_coord(rank) for rank in X.filter_match(pipe=0, data=1)]
            [ProcessCoord(pipe=0, data=1, model=0), ProcessCoord(pipe=0, data=1, model=1)]

        Arguments:
            **filter_kwargs (dict): criteria used to select coordinates.

        Returns:
            The list of ranks whose coordinates match filter_kwargs.
        """
        def _filter_helper(x):
            for key, val in filter_kwargs.items():
                if getattr(x, key) != val:
                    return False
            return True

        coords = filter(_filter_helper, self.mapping.keys())
        return [self.mapping[coo] for coo in coords]

    def get_axis_list(self, axis, idx):
        """Returns the list of global ranks whose coordinate in an axis is idx.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_axis_list(axis='x', idx=0)
            [0, 1, 2]
            >>> X.get_axis_list(axis='y', idx=0)
            [0, 3]
        """

        # This could be faster by generating the desired keys directly instead of
        # filtering.
        axis_num = self.axes.index(axis)
        ranks = [self.mapping[k] for k in self.mapping.keys() if k[axis_num] == idx]
        return ranks

    def world_size(self):
        return len(self.mapping)

    def __str__(self):
        return str(self.mapping)


def _prime_factors(N):
    """ Returns the prime factorization of positive integer N. """
    if N <= 0:
        raise ValueError("Values must be strictly positive.")

    primes = []
    while N != 1:
        for candidate in range(2, N + 1):
            if N % candidate == 0:
                primes.append(candidate)
                N //= candidate
                break
    return primes


class PipeDataParallelTopology(ProcessTopology):
    """ A topology specialiation for hybrid data and pipeline parallelism.

        Uses data parallelism on the last dimension to encourage gradient
        reductions to use high-bandwidth intra-node links and lower-volume
        pipeline communications to use low-bandwidth inter-node links.
    """
    def __init__(self, num_pp, num_dp):
        super().__init__(axes=['pipe', 'data'], dims=[num_pp, num_dp])


class PipeModelDataParallelTopology(ProcessTopology):
    """ A topology for hybrid pipeline, model, and data parallelism. """
    def __init__(self, num_dp, num_pp, num_mp):
        # super().__init__(axes=['model', 'data', 'pipe'], dims=[num_mp, num_dp, num_pp])
        super().__init__(axes=['pipe', 'data', 'model'], dims=[num_pp, num_dp, num_mp])


class PipelineParallelGrid:
    """Implements a grid object that stores the data parallel ranks
    corresponding to each o the model parallel stages

    The grid object organizes the processes in a distributed pytorch job
    into a 2D grid, of stage_id and data_parallel_id.

    self.stage_id and self.data_parallel_id stores the stage id
    and the data parallel id of current process.

    self.dp_group groups the processes by stage_id.
    self.dp_group[i], is a list containing all process ranks whose
    stage_id is i.

    self.p2p_groups stores a list of tuple, where each tuple
    stores process ranks of adjacent stages for a given data_parallel_id.
    For example if num_stage is 5 then a tuple [7,8] represents stages [3, 4],
    with data_parallel id = 1. A stage wrap around will appear as non-adjacent ranks,
    for example tuple [4,0] with representing wrap-around stage 4 and 0, for
    data_parallel_id = 0, or similarly [9,5] represents wrapped around stages [4,0]
    for data_parallel_id = 1.
    """
    def __init__(self, topology=None, process_group=None):
        # TODO use process_group if provided
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if topology is not None:
            log_dist(f'building PipelineParallelGrid with topology: {topology}', ranks=[-1], level=logging.DEBUG)
            self._topo = topology
        else:
            num_pp = 1
            num_dp = 1
            for idx, prime in enumerate(_prime_factors(self.world_size)):
                if idx % 2 == 0:
                    num_pp *= prime
                else:
                    num_dp *= prime
            self._topo = PipeDataParallelTopology(num_dp=num_dp, num_pp=num_pp)
        self.data_parallel_size = max(self._topo.get_dim('data'), 1)
        self.pipe_parallel_size = max(self._topo.get_dim('pipe'), 1)
        self.model_parallel_size = max(self._topo.get_dim('model'), 1)
        assert self._is_grid_valid(), "Invalid Grid"

        self.stage_id = self.get_stage_id()
        self.data_parallel_id = self.get_data_parallel_id()
        self.model_parallel_id = self.get_model_parallel_id()
        self.slice_parallel_src_id = self.get_src_parallel_src_id()
        log_dist(f'stage_id: {self.stage_id}, slice_parallel_src_id: {self.slice_parallel_src_id}', ranks=[-1], level=logging.DEBUG)
        # Create new ProcessGroups for all model parallelism. DeepSpeedLight uses these
        # to detect overflow, etc.


        self.ds_model_proc_group = None
        self.ds_model_rank = -1
        for dp in range(self.data_parallel_size):
            ranks = sorted(self._topo.get_axis_list(axis='data', idx=dp))
            if self.global_rank == 0:
                #print(f'RANK={self.global_rank} building DeepSpeed model group: {ranks}')
                pass
            proc_group = dist.new_group(ranks=ranks)

            if self.global_rank in ranks:
                log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: {self.model_parallel_id}, \
                    stage_id: {self.stage_id}, building ds model group: {ranks}', ranks=[-1], level=logging.DEBUG)
                self.ds_model_proc_group = proc_group
                self.ds_model_world_size = len(ranks)
                self.ds_model_rank = ranks.index(self.global_rank)
        assert self.ds_model_rank > -1
        assert self.ds_model_proc_group is not None

        # Create new ProcessGroup for gradient all-reduces - these are the data parallel groups
        self.dp_group = []
        self.dp_groups = self._topo.get_axis_comm_lists('data')
        for g in self.dp_groups:
            proc_group = dist.new_group(ranks=g)
            if self.global_rank in g:
                log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: {self.model_parallel_id}, \
                    stage_id: {self.stage_id}, building dp group: {g}', ranks=[-1], level=logging.DEBUG)
                self.dp_group = g
                self.dp_proc_group = proc_group

        self.is_first_stage = (self.stage_id == 0)
        self.is_last_stage = (self.stage_id == (self.pipe_parallel_size - 1))

        self.p2p_groups = self._build_p2p_groups()
        self._build_grads_groups()
        self._build_activation_groups()

        self._build_grads_groups()

        self._build_activation_groups()

        # Create new ProcessGroup for pipeline collectives - these are pipe parallel groups
        self.pp_group = []
        self.pp_proc_group = None
        self.pipe_groups = self._topo.get_axis_comm_lists('pipe')
        for ranks in self.pipe_groups:
            # if self.global_rank == 0:
            #     #print(f'RANK={self.global_rank} building pipeline group: {ranks}')
            #     pass
            proc_group = dist.new_group(ranks=ranks)
            if self.global_rank in ranks:
                log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: {self.model_parallel_id},\
                    stage_id: {self.stage_id}, building pipeline group: {ranks}', \
                    ranks=[-1], level=logging.DEBUG)
                self.pp_group = ranks
                self.pp_proc_group = proc_group
        assert self.pp_proc_group is not None
        
        # Create new ProcessGroup for model (tensor-slicing) collectives

        # Short circuit case without model parallelism.
        # TODO: it would be nice if topology had bcast semantics to avoid this branching
        # case?
        if self.model_parallel_size == 1:
            for group_rank in range(self.world_size):
                group_rank = [group_rank]
                group = dist.new_group(ranks=group_rank)
                if group_rank[0] == self.global_rank:
                    self.slice_group = group_rank
                    self.slice_proc_group = group
            return
        else:
            self.mp_group = []
            self.model_groups = self._topo.get_axis_comm_lists('model')
            for g in self.model_groups:
                proc_group = dist.new_group(ranks=g)
                if self.global_rank in g:
                    log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: {self.model_parallel_id}, \
                        stage_id: {self.stage_id}, building slice group: {g}', ranks=[-1], level=logging.DEBUG)
                    self.slice_group = g
                    self.slice_proc_group = proc_group

    def get_stage_id(self):
        return self._topo.get_coord(rank=self.global_rank).pipe

    def get_data_parallel_id(self):
        return self._topo.get_coord(rank=self.global_rank).data
    
    def get_model_parallel_id(self):
        if 'model' in self._topo.get_axis_names():
            return self._topo.get_coord(rank=self.global_rank).model
        return 0

    def get_src_parallel_src_id(self):
        if 'model' not in self._topo.get_axis_names():
            return 0
        return self.stage_to_global(stage_id=self.stage_id,
                                    data=self.data_parallel_id,
                                    model=0)

    def _build_p2p_groups(self):
        """Groups for sending and receiving activations and gradients across model
        parallel stages.
        """
        comm_lists = self._topo.get_axis_comm_lists('pipe')
        log_dist(f'_build_p2p_groups data_parallel_id: {self.data_parallel_id}, \
            model_parallel_id: {self.model_parallel_id}, stage_id: {self.stage_id}, \
            comm_lists: {comm_lists}', ranks=[-1], level=logging.DEBUG)

        p2p_lists = []
        for rank in range(self.world_size):
            for l in comm_lists:
                assert len(l) == self.pipe_parallel_size
                if rank in l:
                    idx = l.index(rank)
                    buddy_rank = l[(idx + 1) % self.pipe_parallel_size]
                    p2p_lists.append([rank, buddy_rank])
                    break  # next global rank
        assert len(p2p_lists) == self.world_size
        log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: \
            {self.model_parallel_id}, stage_id: {self.stage_id}, \
            p2p_lists: {p2p_lists}', ranks=[-1], level=logging.DEBUG)
        return p2p_lists
    
    def _build_grads_groups(self):
        self.send_grads_src_rank = -1
        self.recv_grads_src_rank = -1

        self.send_grads_group = []
        self.recv_grads_group = []

        self.send_grads_proc_group = None
        self.recv_grads_proc_group = None
        self.grads_proc_groups = []

        for dp_id in range(self.data_parallel_size):
            for stage in range(self.pipe_parallel_size):
                next_stage = stage + 1
                prev_stage = stage - 1

                grads_group = []
                grads_proc_group = None
            
                if prev_stage > -1:
                    grads_src_rank = self._topo.filter_match(data=dp_id, pipe=stage, model=0)[0]
                    prev_mp_group = self._topo.filter_match(data=dp_id, pipe=prev_stage)
                    grads_group.append(grads_src_rank)
                    grads_group.extend(prev_mp_group)
                    grads_group.sort()
                    # log_dist(f'_build_grads_groups stage: {stage}, grads_group: {grads_group}', ranks=[-1])
                    grads_proc_group = dist.new_group(ranks=grads_group)
                    self.grads_proc_groups.append(grads_proc_group)
                    if stage == self.stage_id and self.data_parallel_id == dp_id:
                        self.send_grads_src_rank = grads_src_rank
                        self.send_grads_group = grads_group
                        self.send_grads_proc_group = grads_proc_group
                    
                    elif stage == self.stage_id + 1 and self.data_parallel_id == dp_id:
                        self.recv_grads_src_rank = grads_src_rank
                        self.recv_grads_group = grads_group
                        self.recv_grads_proc_group = grads_proc_group
        log_dist(f'_build_grads_groups stage: {self.stage_id}, send_grads_src_rank : {self.send_grads_src_rank}, '
                f'send_grads_group: {self.send_grads_group}, recv_grads_group: {self.recv_grads_group}', \
                ranks=[-1], level=logging.DEBUG)

    def _build_activation_groups(self):
        self.send_activation_src_rank = -1
        self.recv_activation_src_rank = -1

        self.send_activation_group = []
        self.recv_activation_group = []

        self.send_activation_proc_group = None
        self.recv_activation_proc_group = None
        self.activation_proc_groups = []

        for dp_id in range(self.data_parallel_size):
            for stage in range(self.pipe_parallel_size):
                next_stage = stage + 1
                prev_stage = stage - 1

                activation_group = []
                activation_proc_group = None
            
                if next_stage < self.pipe_parallel_size:
                    activation_src_rank = self._topo.filter_match(data=dp_id, pipe=stage, model=0)[0]
                    next_mp_group = self._topo.filter_match(data=dp_id, pipe=next_stage)
                    activation_group.append(activation_src_rank)
                    activation_group.extend(next_mp_group)
                    activation_group.sort()
                    activation_proc_group = dist.new_group(ranks=activation_group)
                    self.activation_proc_groups.append(activation_proc_group)
                    if stage == self.stage_id and self.data_parallel_id == dp_id:
                        self.send_activation_src_rank = activation_src_rank
                        self.send_activation_group = activation_group
                        self.send_activation_proc_group = activation_proc_group
                    elif stage == self.stage_id - 1 and self.data_parallel_id == dp_id:
                        self.recv_activation_src_rank = activation_src_rank
                        self.recv_activation_group = activation_group
                        self.recv_activation_proc_group = activation_proc_group
        log_dist(f'_build_activation_groups stage: {self.stage_id}, send_activation_src_rank : '\
            f'{self.send_activation_src_rank}, send_activation_group: {self.send_activation_group}, '\
            f'recv_grads_group: {self.recv_grads_group}', ranks=[-1], level=logging.DEBUG)

    def _is_grid_valid(self):
        ranks = 1
        for ax in self._topo.get_axis_names():
            ranks *= self._topo.get_dim(ax)
        return ranks == dist.get_world_size()

    #returns the global rank of the process with the provided stage id
    #which has the same data_parallel_id as caller process
    def stage_to_global(self, stage_id, **kwargs):
        me = self._topo.get_coord(self.global_rank)
        transform = me._replace(pipe=stage_id, **kwargs)._asdict()
        return self._topo.get_rank(**transform)

    #returns the byteps rank of the process with the provided stage id
    def stage_to_byteps(self, stage_id):
        return self.pipe_parallel_size * self.data_parallel_id + stage_id

    def topology(self):
        return self._topo

    # MPU functions for DeepSpeed integration
    def get_global_rank(self):
        return self.global_rank

    def get_pipe_parallel_rank(self):
        """ The stage of the pipeline this rank resides in. """
        return self.stage_id

    def get_pipe_parallel_world_size(self):
        """ The number of stages in the pipeline. """
        return self.pipe_parallel_size

    def get_pipe_parallel_group(self):
        """ The group of ranks within the same pipeline. """
        return self.pp_proc_group

    def get_data_parallel_rank(self):
        """ Which pipeline this rank resides in. """
        return self.data_parallel_id

    def get_data_parallel_world_size(self):
        """ The number of pipelines. """
        return self.data_parallel_size

    def get_data_parallel_group(self):
        """ The group of ranks within the same stage of all pipelines. """
        return self.dp_proc_group

    # These are model parallel groups across all types of model parallelism.
    # Deepspeed uses them to detect overflow, etc.
    def get_model_parallel_rank(self):
        return self.model_parallel_id

    def get_model_parallel_world_size(self):
        return self.model_parallel_size

    def get_model_parallel_group(self):
        return self.slice_proc_group

    # For Megatron-style tensor slicing
    def get_slice_parallel_rank(self):
        return self.model_parallel_id

    def get_slice_parallel_world_size(self):
        return self.model_parallel_size

    def get_slice_parallel_group(self):
        return self.slice_proc_group

    def get_slice_parallel_src_rank(self):
        return self.slice_parallel_src_id
# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright 2019 The Microsoft DeepSpeed Team

from deepspeed.utils import log_dist

import torch.distributed as dist

from collections import namedtuple
from itertools import product as cartesian_product
import logging, os

import torch

class ProcessTopology:
    """ Manages the mapping of n-dimensional Cartesian coordinates to linear
    indices. This mapping is used to map the rank of processes to the grid
    for various forms of parallelism.

    Each axis of the tensor is accessed by its name. The provided ordering
    of the axes defines the layout of the topology. ProcessTopology uses a "row-major"
    layout of the tensor axes, and so axes=['x', 'y'] would map coordinates (x,y) and
    (x,y+1) to adjacent linear indices. If instead axes=['y', 'x'] was used, coordinates
    (x,y) and (x+1,y) would be adjacent.

    Some methods return ProcessCoord namedtuples.
    """
    def __init__(self, axes, dims):
        """Create a mapping of n-dimensional tensor coordinates to linear indices.

        Arguments:
            axes (list): the names of the tensor axes
            dims (list): the dimension (length) of each axis of the topology tensor
        """

        self.axes = axes  # names of each topology axis
        self.dims = dims  # length of each topology axis

        # This is actually a class that lets us hash {'row':3, 'col':2} mappings
        self.ProcessCoord = namedtuple('ProcessCoord', axes)

        self.mapping = {}
        ranges = [range(d) for d in dims]
        # example: 1, (0,0,1)
        for global_rank, coord in enumerate(cartesian_product(*ranges)):
            key = {axis: coord[self.axes.index(axis)] for axis in self.axes}
            key = self.ProcessCoord(**key)
            # for example, {ProcessCoord(row=0, col=1) : 1}
            self.mapping[key] = global_rank

    def get_rank(self, **coord_kwargs):
        """Return the global rank of a process via its coordinates.

        Coordinates are specified as kwargs. For example:

            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_rank(x=0, y=1)
            1
        """
        if len(coord_kwargs) != len(self.axes):
            raise ValueError('get_rank() does not support slices. Use filter_match())')

        key = self.ProcessCoord(**coord_kwargs)
        assert key in self.mapping, f'key {kwargs} invalid'
        return self.mapping[key]

    def get_axis_names(self):
        """Return a list of the axis names in the ordering of the topology. """
        return self.axes

    def get_rank_repr(self,
                      rank,
                      omit_axes=['data',
                                 'pipe'],
                      inner_sep='_',
                      outer_sep='-'):
        """Return a string representation of a rank.

        This method is primarily used for checkpointing model data.

        For example:
            >>> topo = Topo(axes=['a', 'b'], dims=[2, 2])
            >>> topo.get_rank_repr(rank=3)
            'a_01-b_01'
            >>> topo.get_rank_repr(rank=3, omit_axes=['a'])
            'b_01'

        Args:
            rank (int): A rank in the topology.
            omit_axes (list, optional): Axes that should not be in the representation. Defaults to ['data', 'pipe'].
            inner_sep (str, optional): [description]. Defaults to '_'.
            outer_sep (str, optional): [description]. Defaults to '-'.

        Returns:
            str: A string representation of the coordinate owned by ``rank``.
        """
        omit_axes = frozenset(omit_axes)
        axes = [a for a in self.get_axis_names() if a not in omit_axes]
        names = []
        for ax in axes:
            ax_rank = getattr(self.get_coord(rank=rank), ax)
            names.append(f'{ax}{inner_sep}{ax_rank:02d}')
        return outer_sep.join(names)

    def get_dim(self, axis):
        """Return the number of processes along the given axis.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_dim('y')
            3
        """
        if axis not in self.axes:
            return 0
        return self.dims[self.axes.index(axis)]

    def get_coord(self, rank):
        """Return the coordinate owned by a process rank.

        The axes of the returned namedtuple can be directly accessed as members. For
        example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> coord = X.get_coord(rank=1)
            >>> coord.x
            0
            >>> coord.y
            1
        """
        for coord, idx in self.mapping.items():
            if idx == rank:
                return coord
        raise ValueError(f'rank {rank} not found in topology.')

    def get_axis_comm_lists(self, axis):
        """ Construct lists suitable for a communicator group along axis ``axis``.

        Example:
            >>> topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> topo.get_axis_comm_lists('pipe')
            [
                [0, 4], # data=0, model=0
                [1, 5], # data=0, model=1
                [2, 6], # data=1, model=0
                [3, 7], # data=1, model=1
            ]

        Returns:
            A list of lists whose coordinates match in all axes *except* ``axis``.
        """

        # We don't want to RuntimeError because it allows us to write more generalized
        # code for hybrid parallelisms.
        if axis not in self.axes:
            return []

        # Grab all axes but `axis`
        other_axes = [a for a in self.axes if a != axis]

        lists = []

        # Construct all combinations of coords with other_axes
        ranges = [range(self.get_dim(a)) for a in other_axes]
        for coord in cartesian_product(*ranges):
            other_keys = {a: coord[other_axes.index(a)] for a in other_axes}
            # now go over all ranks in `axis`.
            sub_list = []
            for axis_key in range(self.get_dim(axis)):
                key = self.ProcessCoord(**other_keys, **{axis: axis_key})
                sub_list.append(self.mapping[key])
            lists.append(sub_list)

        return lists

    def filter_match(self, **filter_kwargs):
        """Return the list of ranks whose coordinates match the provided criteria.

        Example:
            >>> X = ProcessTopology(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> X.filter_match(pipe=0, data=1)
            [2, 3]
            >>> [X.get_coord(rank) for rank in X.filter_match(pipe=0, data=1)]
            [ProcessCoord(pipe=0, data=1, model=0), ProcessCoord(pipe=0, data=1, model=1)]

        Arguments:
            **filter_kwargs (dict): criteria used to select coordinates.

        Returns:
            The list of ranks whose coordinates match filter_kwargs.
        """
        def _filter_helper(x):
            for key, val in filter_kwargs.items():
                if getattr(x, key) != val:
                    return False
            return True

        coords = filter(_filter_helper, self.mapping.keys())
        return [self.mapping[coo] for coo in coords]

    def get_axis_list(self, axis, idx):
        """Returns the list of global ranks whose coordinate in an axis is idx.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_axis_list(axis='x', idx=0)
            [0, 1, 2]
            >>> X.get_axis_list(axis='y', idx=0)
            [0, 3]
        """

        # This could be faster by generating the desired keys directly instead of
        # filtering.
        axis_num = self.axes.index(axis)
        ranks = [self.mapping[k] for k in self.mapping.keys() if k[axis_num] == idx]
        return ranks

    def world_size(self):
        return len(self.mapping)

    def __str__(self):
        return str(self.mapping)


def _prime_factors(N):
    """ Returns the prime factorization of positive integer N. """
    if N <= 0:
        raise ValueError("Values must be strictly positive.")

    primes = []
    while N != 1:
        for candidate in range(2, N + 1):
            if N % candidate == 0:
                primes.append(candidate)
                N //= candidate
                break
    return primes


class PipeDataParallelTopology(ProcessTopology):
    """ A topology specialiation for hybrid data and pipeline parallelism.

        Uses data parallelism on the last dimension to encourage gradient
        reductions to use high-bandwidth intra-node links and lower-volume
        pipeline communications to use low-bandwidth inter-node links.
    """
    def __init__(self, num_pp, num_dp):
        super().__init__(axes=['pipe', 'data'], dims=[num_pp, num_dp])


class PipeModelDataParallelTopology(ProcessTopology):
    """ A topology for hybrid pipeline, model, and data parallelism. """
    def __init__(self, num_dp, num_pp, num_mp):
        # super().__init__(axes=['model', 'data', 'pipe'], dims=[num_mp, num_dp, num_pp])
        super().__init__(axes=['pipe', 'data', 'model'], dims=[num_pp, num_dp, num_mp])


class PipelineParallelGrid:
    """Implements a grid object that stores the data parallel ranks
    corresponding to each o the model parallel stages

    The grid object organizes the processes in a distributed pytorch job
    into a 2D grid, of stage_id and data_parallel_id.

    self.stage_id and self.data_parallel_id stores the stage id
    and the data parallel id of current process.

    self.dp_group groups the processes by stage_id.
    self.dp_group[i], is a list containing all process ranks whose
    stage_id is i.

    self.p2p_groups stores a list of tuple, where each tuple
    stores process ranks of adjacent stages for a given data_parallel_id.
    For example if num_stage is 5 then a tuple [7,8] represents stages [3, 4],
    with data_parallel id = 1. A stage wrap around will appear as non-adjacent ranks,
    for example tuple [4,0] with representing wrap-around stage 4 and 0, for
    data_parallel_id = 0, or similarly [9,5] represents wrapped around stages [4,0]
    for data_parallel_id = 1.
    """
    def __init__(self, topology=None, process_group=None):
        # TODO use process_group if provided
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if topology is not None:
            log_dist(f'building PipelineParallelGrid with topology: {topology}', ranks=[-1], level=logging.DEBUG)
            self._topo = topology
        else:
            num_pp = 1
            num_dp = 1
            for idx, prime in enumerate(_prime_factors(self.world_size)):
                if idx % 2 == 0:
                    num_pp *= prime
                else:
                    num_dp *= prime
            self._topo = PipeDataParallelTopology(num_dp=num_dp, num_pp=num_pp)
        self.data_parallel_size = max(self._topo.get_dim('data'), 1)
        self.pipe_parallel_size = max(self._topo.get_dim('pipe'), 1)
        self.model_parallel_size = max(self._topo.get_dim('model'), 1)
        assert self._is_grid_valid(), "Invalid Grid"

        self.stage_id = self.get_stage_id()
        self.data_parallel_id = self.get_data_parallel_id()
        self.model_parallel_id = self.get_model_parallel_id()
        self.slice_parallel_src_id = self.get_src_parallel_src_id()
        log_dist(f'stage_id: {self.stage_id}, slice_parallel_src_id: {self.slice_parallel_src_id}', ranks=[-1], level=logging.DEBUG)
        # Create new ProcessGroups for all model parallelism. DeepSpeedLight uses these
        # to detect overflow, etc.


        self.ds_model_proc_group = None
        self.ds_model_rank = -1
        for dp in range(self.data_parallel_size):
            ranks = sorted(self._topo.get_axis_list(axis='data', idx=dp))
            if self.global_rank == 0:
                #print(f'RANK={self.global_rank} building DeepSpeed model group: {ranks}')
                pass
            proc_group = dist.new_group(ranks=ranks)

            if self.global_rank in ranks:
                log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: {self.model_parallel_id}, \
                    stage_id: {self.stage_id}, building ds model group: {ranks}', ranks=[-1], level=logging.DEBUG)
                self.ds_model_proc_group = proc_group
                self.ds_model_world_size = len(ranks)
                self.ds_model_rank = ranks.index(self.global_rank)
        assert self.ds_model_rank > -1
        assert self.ds_model_proc_group is not None

        # Create new ProcessGroup for gradient all-reduces - these are the data parallel groups
        self.dp_group = []
        self.dp_groups = self._topo.get_axis_comm_lists('data')
        for g in self.dp_groups:
            proc_group = dist.new_group(ranks=g)
            if self.global_rank in g:
                log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: {self.model_parallel_id}, \
                    stage_id: {self.stage_id}, building dp group: {g}', ranks=[-1], level=logging.DEBUG)
                self.dp_group = g
                self.dp_proc_group = proc_group

        self.is_first_stage = (self.stage_id == 0)
        self.is_last_stage = (self.stage_id == (self.pipe_parallel_size - 1))

        self.p2p_groups = self._build_p2p_groups()
        self._build_grads_groups()
        self._build_activation_groups()

        self._build_grads_groups()

        self._build_activation_groups()

        # Create new ProcessGroup for pipeline collectives - these are pipe parallel groups
        self.pp_group = []
        self.pp_proc_group = None
        self.pipe_groups = self._topo.get_axis_comm_lists('pipe')
        for ranks in self.pipe_groups:
            # if self.global_rank == 0:
            #     #print(f'RANK={self.global_rank} building pipeline group: {ranks}')
            #     pass
            proc_group = dist.new_group(ranks=ranks)
            if self.global_rank in ranks:
                log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: {self.model_parallel_id},\
                    stage_id: {self.stage_id}, building pipeline group: {ranks}', \
                    ranks=[-1], level=logging.DEBUG)
                self.pp_group = ranks
                self.pp_proc_group = proc_group
        assert self.pp_proc_group is not None
        
        # Create new ProcessGroup for model (tensor-slicing) collectives

        # Short circuit case without model parallelism.
        # TODO: it would be nice if topology had bcast semantics to avoid this branching
        # case?
        if self.model_parallel_size == 1:
            for group_rank in range(self.world_size):
                group_rank = [group_rank]
                group = dist.new_group(ranks=group_rank)
                if group_rank[0] == self.global_rank:
                    self.slice_group = group_rank
                    self.slice_proc_group = group
            return
        else:
            self.mp_group = []
            self.model_groups = self._topo.get_axis_comm_lists('model')
            for g in self.model_groups:
                proc_group = dist.new_group(ranks=g)
                if self.global_rank in g:
                    log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: {self.model_parallel_id}, \
                        stage_id: {self.stage_id}, building slice group: {g}', ranks=[-1], level=logging.DEBUG)
                    self.slice_group = g
                    self.slice_proc_group = proc_group

    def get_stage_id(self):
        return self._topo.get_coord(rank=self.global_rank).pipe

    def get_data_parallel_id(self):
        return self._topo.get_coord(rank=self.global_rank).data
    
    def get_model_parallel_id(self):
        if 'model' in self._topo.get_axis_names():
            return self._topo.get_coord(rank=self.global_rank).model
        return 0

    def get_src_parallel_src_id(self):
        if 'model' not in self._topo.get_axis_names():
            return 0
        return self.stage_to_global(stage_id=self.stage_id,
                                    data=self.data_parallel_id,
                                    model=0)

    def _build_p2p_groups(self):
        """Groups for sending and receiving activations and gradients across model
        parallel stages.
        """
        comm_lists = self._topo.get_axis_comm_lists('pipe')
        log_dist(f'_build_p2p_groups data_parallel_id: {self.data_parallel_id}, \
            model_parallel_id: {self.model_parallel_id}, stage_id: {self.stage_id}, \
            comm_lists: {comm_lists}', ranks=[-1], level=logging.DEBUG)

        p2p_lists = []
        for rank in range(self.world_size):
            for l in comm_lists:
                assert len(l) == self.pipe_parallel_size
                if rank in l:
                    idx = l.index(rank)
                    buddy_rank = l[(idx + 1) % self.pipe_parallel_size]
                    p2p_lists.append([rank, buddy_rank])
                    break  # next global rank
        assert len(p2p_lists) == self.world_size
        log_dist(f'data_parallel_id: {self.data_parallel_id}, model_parallel_id: \
            {self.model_parallel_id}, stage_id: {self.stage_id}, \
            p2p_lists: {p2p_lists}', ranks=[-1], level=logging.DEBUG)
        return p2p_lists
    
    def _build_grads_groups(self):
        self.send_grads_src_rank = -1
        self.recv_grads_src_rank = -1

        self.send_grads_group = []
        self.recv_grads_group = []

        self.send_grads_proc_group = None
        self.recv_grads_proc_group = None
        self.grads_proc_groups = []

        for dp_id in range(self.data_parallel_size):
            for stage in range(self.pipe_parallel_size):
                next_stage = stage + 1
                prev_stage = stage - 1

                grads_group = []
                grads_proc_group = None
            
                if prev_stage > -1:
                    grads_src_rank = self._topo.filter_match(data=dp_id, pipe=stage, model=0)[0]
                    prev_mp_group = self._topo.filter_match(data=dp_id, pipe=prev_stage)
                    grads_group.append(grads_src_rank)
                    grads_group.extend(prev_mp_group)
                    grads_group.sort()
                    # log_dist(f'_build_grads_groups stage: {stage}, grads_group: {grads_group}', ranks=[-1])
                    grads_proc_group = dist.new_group(ranks=grads_group)
                    self.grads_proc_groups.append(grads_proc_group)
                    if stage == self.stage_id and self.data_parallel_id == dp_id:
                        self.send_grads_src_rank = grads_src_rank
                        self.send_grads_group = grads_group
                        self.send_grads_proc_group = grads_proc_group
                    
                    elif stage == self.stage_id + 1 and self.data_parallel_id == dp_id:
                        self.recv_grads_src_rank = grads_src_rank
                        self.recv_grads_group = grads_group
                        self.recv_grads_proc_group = grads_proc_group
        log_dist(f'_build_grads_groups stage: {self.stage_id}, send_grads_src_rank : {self.send_grads_src_rank}, '
                f'send_grads_group: {self.send_grads_group}, recv_grads_group: {self.recv_grads_group}', \
                ranks=[-1], level=logging.DEBUG)

    def _build_activation_groups(self):
        self.send_activation_src_rank = -1
        self.recv_activation_src_rank = -1

        self.send_activation_group = []
        self.recv_activation_group = []

        self.send_activation_proc_group = None
        self.recv_activation_proc_group = None
        self.activation_proc_groups = []

        for dp_id in range(self.data_parallel_size):
            for stage in range(self.pipe_parallel_size):
                next_stage = stage + 1
                prev_stage = stage - 1

                activation_group = []
                activation_proc_group = None
            
                if next_stage < self.pipe_parallel_size:
                    activation_src_rank = self._topo.filter_match(data=dp_id, pipe=stage, model=0)[0]
                    next_mp_group = self._topo.filter_match(data=dp_id, pipe=next_stage)
                    activation_group.append(activation_src_rank)
                    activation_group.extend(next_mp_group)
                    activation_group.sort()
                    activation_proc_group = dist.new_group(ranks=activation_group)
                    self.activation_proc_groups.append(activation_proc_group)
                    if stage == self.stage_id and self.data_parallel_id == dp_id:
                        self.send_activation_src_rank = activation_src_rank
                        self.send_activation_group = activation_group
                        self.send_activation_proc_group = activation_proc_group
                    elif stage == self.stage_id - 1 and self.data_parallel_id == dp_id:
                        self.recv_activation_src_rank = activation_src_rank
                        self.recv_activation_group = activation_group
                        self.recv_activation_proc_group = activation_proc_group
        log_dist(f'_build_activation_groups stage: {self.stage_id}, send_activation_src_rank : '\
            f'{self.send_activation_src_rank}, send_activation_group: {self.send_activation_group}, '\
            f'recv_grads_group: {self.recv_grads_group}', ranks=[-1], level=logging.DEBUG)

    def _is_grid_valid(self):
        ranks = 1
        for ax in self._topo.get_axis_names():
            ranks *= self._topo.get_dim(ax)
        return ranks == dist.get_world_size()

    #returns the global rank of the process with the provided stage id
    #which has the same data_parallel_id as caller process
    def stage_to_global(self, stage_id, **kwargs):
        me = self._topo.get_coord(self.global_rank)
        transform = me._replace(pipe=stage_id, **kwargs)._asdict()
        return self._topo.get_rank(**transform)

    #returns the byteps rank of the process with the provided stage id
    def stage_to_byteps(self, stage_id):
        return self.pipe_parallel_size * self.data_parallel_id + stage_id

    def topology(self):
        return self._topo

    # MPU functions for DeepSpeed integration
    def get_global_rank(self):
        return self.global_rank

    def get_pipe_parallel_rank(self):
        """ The stage of the pipeline this rank resides in. """
        return self.stage_id

    def get_pipe_parallel_world_size(self):
        """ The number of stages in the pipeline. """
        return self.pipe_parallel_size

    def get_pipe_parallel_group(self):
        """ The group of ranks within the same pipeline. """
        return self.pp_proc_group

    def get_data_parallel_rank(self):
        """ Which pipeline this rank resides in. """
        return self.data_parallel_id

    def get_data_parallel_world_size(self):
        """ The number of pipelines. """
        return self.data_parallel_size

    def get_data_parallel_group(self):
        """ The group of ranks within the same stage of all pipelines. """
        return self.dp_proc_group

    # These are model parallel groups across all types of model parallelism.
    # Deepspeed uses them to detect overflow, etc.
    def get_model_parallel_rank(self):
        return self.model_parallel_id# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.autograd as autograd

# try:
#     import veGiantModel
# except ImportError:
#     byteGiantModel = None

class MockModule(nn.Module):
    """Module for testing model parallelism"""
    pass

try:
    from th_fastertransformer import Linear

    class LinearFunction(autograd.Function):

        @staticmethod
        def forward(ctx, input_tensor, weight, bias, act_gelu=False, dropout_rate=0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            if act_gelu == True or dropout_rate > 0.0:
                output, bias_out, dropout_mask = Linear.forward_gelu_dropout(input_tensor, weight, bias, act_gelu, dropout_rate)
            else:
                output = Linear.forward(input_tensor, weight, bias)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.act_gelu = act_gelu
            ctx.dropout_rate = dropout_rate
            return output

        @staticmethod
        def backward(ctx, grad_out):
            act_gelu = ctx.act_gelu
            dropout_rate = ctx.dropout_rate
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            if act_gelu == True or dropout_rate > 0.0:
                grad_in, grad_weight, grad_bias = Linear.backward_gelu_dropout(
                    grad_out, input_tensor, weight, act_gelu, dropout_rate, bias_out, dropout_mask)
            else:
                grad_in, grad_weight, grad_bias = Linear.backward(
                    grad_out, input_tensor, weight)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinear(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu, self.dropout_rate if self.training else 0.)

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

except Exception as e:
    FTLinear = None

try:
    from th_fastertransformer import LinearTranspose

    class LinearTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            output = LinearTranspose.forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = LinearTranspose.backward(grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinearTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

except Exception as e:
    FTLinearTranspose = None
    FTDAGather = None

def column_parallel_load_hook(module, log_fn):
    """hook for column parallel linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnParallelLinear or ColumnParallelLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model parallel modules from non-
        model-parallel checkpoints.
    """
    assert module.mp_rank is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end, :]
            state_dict[weight_name] = shard
            log_fn(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end]
            state_dict[bias_name] = shard
            log_fn(f"slice param {bias_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
    return hook

def column_serial_load_hook(module, log_fn):
    """hook for column serial linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnSerialLinear or ColumnSerialLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model serial modules from non-
        model-parallel checkpoints.
    """
    assert module.model_parallel_size is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            for i in range(module.model_parallel_size):
                weight_name_i = weight_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end, :]
                state_dict[weight_name_i] = shard
                log_fn(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[weight_name]
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            for i in range(module.model_parallel_size):
                bias_name_i = bias_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end]
                state_dict[bias_name_i] = shard
                log_fn(f"slice param {bias_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[bias_name]
    return hook

class ColumnSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the ColumnParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank, and reduce the result if needed.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()
        self.use_ft = use_ft
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensor, self.weight[i], self.bias[i], self.act_gelu,
                                                self.dropout_rate if self.training else 0.)
            else:
                output_i = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
            outputs.append(output_i)
        output = torch.cat(outputs, dim=-1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False,
                 bias=True, gather_output=False):
        """Linear layer with column parallelism.

        The linear layer is defined as Y = dropout(gelu(XA + b)). A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            act_gelu: If true, apply gelu activation to (XA+b)
            dropout_rate: If greater than zero, apply dropout to gelu(XA+b)
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
            bias: If true, add bias
            gather_output: If true, call all-gether on output and make Y avaiable
                        to all GPUs, otherwise, every GPU will have its output
                        which is Y_i = XA_i
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            self.bias.data.zero_()
        else:
            self.bias = None
            assert not use_ft
        self.gather_output = gather_output
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu,
                                            self.dropout_rate if self.training else 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        if self.gather_output:
            output = veGiantModel.distributed.gather_from_model_parallel_region(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the RowParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                          unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    for i in range(model_parallel_size):
                        weight_name_i = weight_name + '.' + str(i)
                        idx_begin = i * self.in_features
                        idx_end = (i + 1) * self.in_features
                        shard = v[:, idx_begin:idx_end]
                        state_dict[weight_name_i] = shard
                        print(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
                    del state_dict[weight_name]
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        input_tensors = torch.split(input_tensor, self.in_features, dim=-1)
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensors[i].contiguous(), self.weight[i], self.bias, False, 0.)
            else:
                output_i = nn.functional.linear(input_tensors[i].contiguous(), self.weight[i], self.bias)
            outputs.append(output_i)
        output = outputs[0]
        for i in range(self.model_parallel_size - 1):
            output = output + outputs[i + 1]
        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """Linear layer with row parallelism.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its first dimension and X along its second dimension as:
                -   -
                | A_1 |
                | .   |
            A = | .   |        X = [X_1, ..., X_p]
                | .   |
                | A_p |
                -   -

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            dropout_rate: If greater than zero, apply dropout XA+b
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                            unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    idx_begin = self.mp_rank * self.in_features
                    idx_end = (self.mp_rank + 1) * self.in_features
                    shard = v[:, idx_begin:idx_end]
                    state_dict[weight_name] = shard
                    print(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, False, 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        import veGiantModel
        output = veGiantModel.distributed.reduce_from_model_parallel_region(output)

        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class ColumnParallelLinearTranspose(nn.Module):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                 use_ft=False, load_from_shards=False):
        """Linear layer with column parallelism. The output is then reshaped to 4D with
        (dim0, dim1, head_num, out_features / head_num), then permuted with axies provided by transpose_type.
        For equivalent computation, check the implementation of `ColumnSerialLinearTranspose`.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            head_num: number of "heads" for the out_feature dimension.
            transpose_type: the axies for permutation on the output.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            use_ft: use faster transformer for acceleration.
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()

        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearTransposeFunction.apply(input_tensor, self.weight, self.bias,
                                                    self.head_num, self.transpose_type)
        else:
            assert self.transpose_type == "0213", self.transpose_type
            linear_out = nn.functional.linear(input_tensor, self.weight, self.bias)
            new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
            linear_out = linear_out.view(*new_shape)
            output = linear_out.permute(0, 2, 1, 3).contiguous()
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

class ColumnSerialLinearTranspose(MockModule):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                    use_ft=False, load_from_shards=False):
        """
        A serial module that mocks the ColumnParallelLinearTranspose module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()

        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearTransposeFunction.apply(input_tensor, self.weight[i], self.bias[i], self.head_num, self.transpose_type)
            else:
                assert self.transpose_type == "0213", self.transpose_type
                linear_out = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
                new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
                linear_out = linear_out.view(*new_shape)
                output_i = linear_out.permute(0, 2, 1, 3).contiguous()
            outputs.append(output_i)
        output = torch.cat(outputs, dim=1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.autograd as autograd

# try:
#     import veGiantModel
# except ImportError:
#     byteGiantModel = None

class MockModule(nn.Module):
    """Module for testing model parallelism"""
    pass

try:
    from th_fastertransformer import Linear

    class LinearFunction(autograd.Function):

        @staticmethod
        def forward(ctx, input_tensor, weight, bias, act_gelu=False, dropout_rate=0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            if act_gelu == True or dropout_rate > 0.0:
                output, bias_out, dropout_mask = Linear.forward_gelu_dropout(input_tensor, weight, bias, act_gelu, dropout_rate)
            else:
                output = Linear.forward(input_tensor, weight, bias)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.act_gelu = act_gelu
            ctx.dropout_rate = dropout_rate
            return output

        @staticmethod
        def backward(ctx, grad_out):
            act_gelu = ctx.act_gelu
            dropout_rate = ctx.dropout_rate
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            if act_gelu == True or dropout_rate > 0.0:
                grad_in, grad_weight, grad_bias = Linear.backward_gelu_dropout(
                    grad_out, input_tensor, weight, act_gelu, dropout_rate, bias_out, dropout_mask)
            else:
                grad_in, grad_weight, grad_bias = Linear.backward(
                    grad_out, input_tensor, weight)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinear(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu, self.dropout_rate if self.training else 0.)

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

except Exception as e:
    FTLinear = None

try:
    from th_fastertransformer import LinearTranspose

    class LinearTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            output = LinearTranspose.forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = LinearTranspose.backward(grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinearTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

except Exception as e:
    FTLinearTranspose = None
    FTDAGather = None

def column_parallel_load_hook(module, log_fn):
    """hook for column parallel linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnParallelLinear or ColumnParallelLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model parallel modules from non-
        model-parallel checkpoints.
    """
    assert module.mp_rank is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end, :]
            state_dict[weight_name] = shard
            log_fn(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end]
            state_dict[bias_name] = shard
            log_fn(f"slice param {bias_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
    return hook

def column_serial_load_hook(module, log_fn):
    """hook for column serial linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnSerialLinear or ColumnSerialLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model serial modules from non-
        model-parallel checkpoints.
    """
    assert module.model_parallel_size is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            for i in range(module.model_parallel_size):
                weight_name_i = weight_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end, :]
                state_dict[weight_name_i] = shard
                log_fn(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[weight_name]
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            for i in range(module.model_parallel_size):
                bias_name_i = bias_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end]
                state_dict[bias_name_i] = shard
                log_fn(f"slice param {bias_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[bias_name]
    return hook

class ColumnSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the ColumnParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank, and reduce the result if needed.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()
        self.use_ft = use_ft
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensor, self.weight[i], self.bias[i], self.act_gelu,
                                                self.dropout_rate if self.training else 0.)
            else:
                output_i = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
            outputs.append(output_i)
        output = torch.cat(outputs, dim=-1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False,
                 bias=True, gather_output=False):
        """Linear layer with column parallelism.

        The linear layer is defined as Y = dropout(gelu(XA + b)). A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            act_gelu: If true, apply gelu activation to (XA+b)
            dropout_rate: If greater than zero, apply dropout to gelu(XA+b)
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
            bias: If true, add bias
            gather_output: If true, call all-gether on output and make Y avaiable
                        to all GPUs, otherwise, every GPU will have its output
                        which is Y_i = XA_i
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            self.bias.data.zero_()
        else:
            self.bias = None
            assert not use_ft
        self.gather_output = gather_output
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu,
                                            self.dropout_rate if self.training else 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        if self.gather_output:
            output = veGiantModel.distributed.gather_from_model_parallel_region(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the RowParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                          unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    for i in range(model_parallel_size):
                        weight_name_i = weight_name + '.' + str(i)
                        idx_begin = i * self.in_features
                        idx_end = (i + 1) * self.in_features
                        shard = v[:, idx_begin:idx_end]
                        state_dict[weight_name_i] = shard
                        print(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
                    del state_dict[weight_name]
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        input_tensors = torch.split(input_tensor, self.in_features, dim=-1)
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensors[i].contiguous(), self.weight[i], self.bias, False, 0.)
            else:
                output_i = nn.functional.linear(input_tensors[i].contiguous(), self.weight[i], self.bias)
            outputs.append(output_i)
        output = outputs[0]
        for i in range(self.model_parallel_size - 1):
            output = output + outputs[i + 1]
        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """Linear layer with row parallelism.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its first dimension and X along its second dimension as:
                -   -
                | A_1 |
                | .   |
            A = | .   |        X = [X_1, ..., X_p]
                | .   |
                | A_p |
                -   -

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            dropout_rate: If greater than zero, apply dropout XA+b
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                            unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    idx_begin = self.mp_rank * self.in_features
                    idx_end = (self.mp_rank + 1) * self.in_features
                    shard = v[:, idx_begin:idx_end]
                    state_dict[weight_name] = shard
                    print(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, False, 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        import veGiantModel
        output = veGiantModel.distributed.reduce_from_model_parallel_region(output)

        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class ColumnParallelLinearTranspose(nn.Module):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                 use_ft=False, load_from_shards=False):
        """Linear layer with column parallelism. The output is then reshaped to 4D with
        (dim0, dim1, head_num, out_features / head_num), then permuted with axies provided by transpose_type.
        For equivalent computation, check the implementation of `ColumnSerialLinearTranspose`.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            head_num: number of "heads" for the out_feature dimension.
            transpose_type: the axies for permutation on the output.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            use_ft: use faster transformer for acceleration.
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()

        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearTransposeFunction.apply(input_tensor, self.weight, self.bias,
                                                    self.head_num, self.transpose_type)
        else:
            assert self.transpose_type == "0213", self.transpose_type
            linear_out = nn.functional.linear(input_tensor, self.weight, self.bias)
            new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
            linear_out = linear_out.view(*new_shape)
            output = linear_out.permute(0, 2, 1, 3).contiguous()
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

class ColumnSerialLinearTranspose(MockModule):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                    use_ft=False, load_from_shards=False):
        """
        A serial module that mocks the ColumnParallelLinearTranspose module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()

        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearTransposeFunction.apply(input_tensor, self.weight[i], self.bias[i], self.head_num, self.transpose_type)
            else:
                assert self.transpose_type == "0213", self.transpose_type
                linear_out = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
                new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
                linear_out = linear_out.view(*new_shape)
                output_i = linear_out.permute(0, 2, 1, 3).contiguous()
            outputs.append(output_i)
        output = torch.cat(outputs, dim=1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.autograd as autograd

# try:
#     import veGiantModel
# except ImportError:
#     byteGiantModel = None

class MockModule(nn.Module):
    """Module for testing model parallelism"""
    pass

try:
    from th_fastertransformer import Linear

    class LinearFunction(autograd.Function):

        @staticmethod
        def forward(ctx, input_tensor, weight, bias, act_gelu=False, dropout_rate=0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            if act_gelu == True or dropout_rate > 0.0:
                output, bias_out, dropout_mask = Linear.forward_gelu_dropout(input_tensor, weight, bias, act_gelu, dropout_rate)
            else:
                output = Linear.forward(input_tensor, weight, bias)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.act_gelu = act_gelu
            ctx.dropout_rate = dropout_rate
            return output

        @staticmethod
        def backward(ctx, grad_out):
            act_gelu = ctx.act_gelu
            dropout_rate = ctx.dropout_rate
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            if act_gelu == True or dropout_rate > 0.0:
                grad_in, grad_weight, grad_bias = Linear.backward_gelu_dropout(
                    grad_out, input_tensor, weight, act_gelu, dropout_rate, bias_out, dropout_mask)
            else:
                grad_in, grad_weight, grad_bias = Linear.backward(
                    grad_out, input_tensor, weight)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinear(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu, self.dropout_rate if self.training else 0.)

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

except Exception as e:
    FTLinear = None

try:
    from th_fastertransformer import LinearTranspose

    class LinearTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            output = LinearTranspose.forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = LinearTranspose.backward(grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinearTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

except Exception as e:
    FTLinearTranspose = None
    FTDAGather = None

def column_parallel_load_hook(module, log_fn):
    """hook for column parallel linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnParallelLinear or ColumnParallelLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model parallel modules from non-
        model-parallel checkpoints.
    """
    assert module.mp_rank is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end, :]
            state_dict[weight_name] = shard
            log_fn(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end]
            state_dict[bias_name] = shard
            log_fn(f"slice param {bias_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
    return hook

def column_serial_load_hook(module, log_fn):
    """hook for column serial linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnSerialLinear or ColumnSerialLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model serial modules from non-
        model-parallel checkpoints.
    """
    assert module.model_parallel_size is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            for i in range(module.model_parallel_size):
                weight_name_i = weight_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end, :]
                state_dict[weight_name_i] = shard
                log_fn(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[weight_name]
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            for i in range(module.model_parallel_size):
                bias_name_i = bias_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end]
                state_dict[bias_name_i] = shard
                log_fn(f"slice param {bias_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[bias_name]
    return hook

class ColumnSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the ColumnParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank, and reduce the result if needed.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()
        self.use_ft = use_ft
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensor, self.weight[i], self.bias[i], self.act_gelu,
                                                self.dropout_rate if self.training else 0.)
            else:
                output_i = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
            outputs.append(output_i)
        output = torch.cat(outputs, dim=-1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False,
                 bias=True, gather_output=False):
        """Linear layer with column parallelism.

        The linear layer is defined as Y = dropout(gelu(XA + b)). A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            act_gelu: If true, apply gelu activation to (XA+b)
            dropout_rate: If greater than zero, apply dropout to gelu(XA+b)
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
            bias: If true, add bias
            gather_output: If true, call all-gether on output and make Y avaiable
                        to all GPUs, otherwise, every GPU will have its output
                        which is Y_i = XA_i
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            self.bias.data.zero_()
        else:
            self.bias = None
            assert not use_ft
        self.gather_output = gather_output
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu,
                                            self.dropout_rate if self.training else 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        if self.gather_output:
            output = veGiantModel.distributed.gather_from_model_parallel_region(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the RowParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                          unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    for i in range(model_parallel_size):
                        weight_name_i = weight_name + '.' + str(i)
                        idx_begin = i * self.in_features
                        idx_end = (i + 1) * self.in_features
                        shard = v[:, idx_begin:idx_end]
                        state_dict[weight_name_i] = shard
                        print(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
                    del state_dict[weight_name]
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        input_tensors = torch.split(input_tensor, self.in_features, dim=-1)
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensors[i].contiguous(), self.weight[i], self.bias, False, 0.)
            else:
                output_i = nn.functional.linear(input_tensors[i].contiguous(), self.weight[i], self.bias)
            outputs.append(output_i)
        output = outputs[0]
        for i in range(self.model_parallel_size - 1):
            output = output + outputs[i + 1]
        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)
        volc key = 'AKLTMGMzYmEzYWI3N4ZlNGNhZjhjODMwMEA9YjkxZTQwOWY'
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """Linear layer with row parallelism.
wdw
        The linear layer is defined as Y = XA + b. A is parallelized along
        its first dimension and X along its second dimension as:
                -   -
                | A_1 |
                | .   |
            A = | .   |        X = [X_1, ..., X_p]
                | .   |
                | A_p |
                -   -

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            dropout_rate: If greater than zero, apply dropout XA+b
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                            unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    idx_begin = self.mp_rank * self.in_features
                    idx_end = (self.mp_rank + 1) * self.in_features
                    shard = v[:, idx_begin:idx_end]
                    state_dict[weight_name] = shard
                    print(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, False, 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        import veGiantModel
        output = veGiantModel.distributed.reduce_from_model_parallel_region(output)

        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class ColumnParallelLinearTranspose(nn.Module):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                 use_ft=False, load_from_shards=False):
        """Linear layer with column parallelism. The output is then reshaped to 4D with
        (dim0, dim1, head_num, out_features / head_num), then permuted with axies provided by transpose_type.
        For equivalent computation, check the implementation of `ColumnSerialLinearTranspose`.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            head_num: number of "heads" for the out_feature dimension.
            transpose_type: the axies for permutation on the output.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            use_ft: use faster transformer for acceleration.
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()

        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearTransposeFunction.apply(input_tensor, self.weight, self.bias,
                                                    self.head_num, self.transpose_type)
        else:
            assert self.transpose_type == "0213", self.transpose_type
            linear_out = nn.functional.linear(input_tensor, self.weight, self.bias)
            new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
            linear_out = linear_out.view(*new_shape)
            output = linear_out.permute(0, 2, 1, 3).contiguous()
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

class ColumnSerialLinearTranspose(MockModule):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                    use_ft=False, load_from_shards=False):
        """
        A serial module that mocks the ColumnParallelLinearTranspose module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()

        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearTransposeFunction.apply(input_tensor, self.weight[i], self.bias[i], self.head_num, self.transpose_type)
            else:
                assert self.transpose_type == "0213", self.transpose_type
                linear_out = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
                new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
                linear_out = linear_out.view(*new_shape)
                output_i = linear_out.permute(0, 2, 1, 3).contiguous()
            outputs.append(output_i)
        output = torch.cat(outputs, dim=1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.autograd as autograd

# try:
#     import veGiantModel
# except ImportError:
#     byteGiantModel = None

class MockModule(nn.Module):
    """Module for testing model parallelism"""
    pass

try:
    from th_fastertransformer import Linear

    class LinearFunction(autograd.Function):

        @staticmethod
        def forward(ctx, input_tensor, weight, bias, act_gelu=False, dropout_rate=0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            if act_gelu == True or dropout_rate > 0.0:
                output, bias_out, dropout_mask = Linear.forward_gelu_dropout(input_tensor, weight, bias, act_gelu, dropout_rate)
            else:
                output = Linear.forward(input_tensor, weight, bias)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.act_gelu = act_gelu
            ctx.dropout_rate = dropout_rate
            return output

        @staticmethod
        def backward(ctx, grad_out):
            act_gelu = ctx.act_gelu
            dropout_rate = ctx.dropout_rate
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            if act_gelu == True or dropout_rate > 0.0:
                grad_in, grad_weight, grad_bias = Linear.backward_gelu_dropout(
                    grad_out, input_tensor, weight, act_gelu, dropout_rate, bias_out, dropout_mask)
            else:
                grad_in, grad_weight, grad_bias = Linear.backward(
                    grad_out, input_tensor, weight)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinear(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu, self.dropout_rate if self.training else 0.)

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

except Exception as e:
    FTLinear = None

try:
    from th_fastertransformer import LinearTranspose

    class LinearTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            output = LinearTranspose.forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = LinearTranspose.backward(grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FTLinearTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))

            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

except Exception as e:
    FTLinearTranspose = None
    FTDAGather = None

def column_parallel_load_hook(module, log_fn):
    """hook for column parallel linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnParallelLinear or ColumnParallelLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model parallel modules from non-
        model-parallel checkpoints.
    """
    assert module.mp_rank is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end, :]
            state_dict[weight_name] = shard
            log_fn(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            idx_begin = module.mp_rank * module.out_features
            idx_end = (module.mp_rank + 1) * module.out_features
            shard = v[idx_begin:idx_end]
            state_dict[bias_name] = shard
            log_fn(f"slice param {bias_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
    return hook

def column_serial_load_hook(module, log_fn):
    """hook for column serial linear's load_state_dict function.
    It is a helper function to load a the checkpoint from a
    non-model-parallel module. It returns a hook function that
    pre-processes the checkpoint to parallel slices such that
    each model parallel rank could load the corresponding slice.

    Arguments:
        module: ColumnSerialLinear or ColumnSerialLinearTranspose

        log_fn: function for logging

    Returns:
        A hook function to help load model serial modules from non-
        model-parallel checkpoints.
    """
    assert module.model_parallel_size is not None
    assert module.out_features is not None
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        weight_name = prefix + 'weight'
        bias_name = prefix + 'bias'
        if weight_name in state_dict:
            v = state_dict[weight_name]
            assert len(v.shape) == 2, v.shape
            for i in range(module.model_parallel_size):
                weight_name_i = weight_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end, :]
                state_dict[weight_name_i] = shard
                log_fn(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[weight_name]
        if bias_name in state_dict:
            v = state_dict[bias_name]
            assert len(v.shape) == 1, v.shape
            for i in range(module.model_parallel_size):
                bias_name_i = bias_name + "." + str(i)
                idx_begin = i * module.out_features
                idx_end = (i + 1) * module.out_features
                shard = v[idx_begin:idx_end]
                state_dict[bias_name_i] = shard
                log_fn(f"slice param {bias_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
            del state_dict[bias_name]
    return hook

class ColumnSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the ColumnParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank, and reduce the result if needed.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()
        self.use_ft = use_ft
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensor, self.weight[i], self.bias[i], self.act_gelu,
                                                self.dropout_rate if self.training else 0.)
            else:
                output_i = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
            outputs.append(output_i)
        output = torch.cat(outputs, dim=-1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02,
                 act_gelu=False, dropout_rate=0.0, load_from_shards=False, use_ft=False,
                 bias=True, gather_output=False):
        """Linear layer with column parallelism.

        The linear layer is defined as Y = dropout(gelu(XA + b)). A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            act_gelu: If true, apply gelu activation to (XA+b)
            dropout_rate: If greater than zero, apply dropout to gelu(XA+b)
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
            bias: If true, add bias
            gather_output: If true, call all-gether on output and make Y avaiable
                        to all GPUs, otherwise, every GPU will have its output
                        which is Y_i = XA_i
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features
        self.out_features = out_features // model_parallel_size
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            self.bias.data.zero_()
        else:
            self.bias = None
            assert not use_ft
        self.gather_output = gather_output
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not use_ft:
            assert not act_gelu
            assert not dropout_rate, dropout_rate
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, self.act_gelu,
                                            self.dropout_rate if self.training else 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        if self.gather_output:
            output = veGiantModel.distributed.gather_from_model_parallel_region(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowSerialLinear(MockModule):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """
        A serial module that mocks the RowParallelLinear module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                          unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    for i in range(model_parallel_size):
                        weight_name_i = weight_name + '.' + str(i)
                        idx_begin = i * self.in_features
                        idx_end = (i + 1) * self.in_features
                        shard = v[:, idx_begin:idx_end]
                        state_dict[weight_name_i] = shard
                        print(f"slice param {weight_name_i}\tfor model parallelism: {v.shape} -> {shard.shape}")
                    del state_dict[weight_name]
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        input_tensors = torch.split(input_tensor, self.in_features, dim=-1)
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearFunction.apply(input_tensors[i].contiguous(), self.weight[i], self.bias, False, 0.)
            else:
                output_i = nn.functional.linear(input_tensors[i].contiguous(), self.weight[i], self.bias)
            outputs.append(output_i)
        output = outputs[0]
        for i in range(self.model_parallel_size - 1):
            output = output + outputs[i + 1]
        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, initializer_range=0.02, dropout_rate=0.0,
                 load_from_shards=False, use_ft=False):
        """Linear layer with row parallelism.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its first dimension and X along its second dimension as:
                -   -
                | A_1 |
                | .   |
            A = | .   |        X = [X_1, ..., X_p]
                | .   |
                | A_p |
                -   -

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            dropout_rate: If greater than zero, apply dropout XA+b
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
            use_ft: use faster transformer for acceleration.
        """
        super().__init__()
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.in_features = in_features // model_parallel_size
        self.out_features = out_features
        assert in_features % model_parallel_size == 0, (in_features, model_parallel_size)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.dropout_rate = dropout_rate
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_ft = use_ft
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        if not load_from_shards:
            def load_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                            unexpected_keys, error_msgs):
                weight_name = prefix + 'weight'
                if weight_name in state_dict:
                    v = state_dict[weight_name]
                    assert len(v.shape) == 2, v.shape
                    idx_begin = self.mp_rank * self.in_features
                    idx_end = (self.mp_rank + 1) * self.in_features
                    shard = v[:, idx_begin:idx_end]
                    state_dict[weight_name] = shard
                    print(f"slice param {weight_name}\tfor model parallelism: {v.shape} -> {shard.shape}")
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        if self.use_ft:
            output = LinearFunction.apply(input_tensor, self.weight, self.bias, False, 0.)
        else:
            output = nn.functional.linear(input_tensor, self.weight, self.bias)
        import veGiantModel
        output = veGiantModel.distributed.reduce_from_model_parallel_region(output)

        if self.dropout_rate:
            output = self.dropout(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class ColumnParallelLinearTranspose(nn.Module):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                 use_ft=False, load_from_shards=False):
        """Linear layer with column parallelism. The output is then reshaped to 4D with
        (dim0, dim1, head_num, out_features / head_num), then permuted with axies provided by transpose_type.
        For equivalent computation, check the implementation of `ColumnSerialLinearTranspose`.

        The linear layer is defined as Y = XA + b. A is parallelized along
        its second dimension as A = [A_1, ..., A_p].

        Arguments:
            in_features: first dimension of matrix A.
            out_features: second dimension of matrix A.
            head_num: number of "heads" for the out_feature dimension.
            transpose_type: the axies for permutation on the output.
            initializer_range: range for weight initialization. Note that bias is always set
                        to zero.
            use_ft: use faster transformer for acceleration.
            load_from_shards: If true, load the states from sharded checkpoints. Otherwise,
                        the module automatically slice the checkpoint tensor based on its
                        model parallel rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()

        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        self.bias.data.zero_()
        if not load_from_shards:
            load_hook = column_parallel_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        import veGiantModel
        input_tensor = veGiantModel.distributed.copy_to_model_parallel_region(input_tensor)
        if self.use_ft:
            output = LinearTransposeFunction.apply(input_tensor, self.weight, self.bias,
                                                    self.head_num, self.transpose_type)
        else:
            assert self.transpose_type == "0213", self.transpose_type
            linear_out = nn.functional.linear(input_tensor, self.weight, self.bias)
            new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
            linear_out = linear_out.view(*new_shape)
            output = linear_out.permute(0, 2, 1, 3).contiguous()
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

class ColumnSerialLinearTranspose(MockModule):
    def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02,
                    use_ft=False, load_from_shards=False):
        """
        A serial module that mocks the ColumnParallelLinearTranspose module. It mocks the parallel
        logic by applying the series of work on the same rank.
        """
        super().__init__()
        self.use_ft = use_ft
        self.in_features = in_features
        import veGiantModel
        model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self.model_parallel_size = model_parallel_size
        self.mp_rank = veGiantModel.distributed.get_model_parallel_rank()
        assert out_features % model_parallel_size == 0, (out_features, model_parallel_size)
        self.out_features = out_features // model_parallel_size
        assert head_num % model_parallel_size == 0, (head_num, model_parallel_size)
        self.head_num = head_num // model_parallel_size
        self.head_dim = self.out_features // self.head_num
        self.transpose_type = transpose_type
        weight_params = [nn.Parameter(torch.Tensor(self.out_features, self.in_features)) for _ in range(model_parallel_size)]
        self.weight = nn.ParameterList(weight_params)
        bias_params = [nn.Parameter(torch.Tensor(self.out_features)) for _ in range(model_parallel_size)]
        self.bias = nn.ParameterList(bias_params)
        for weight in self.weight:
            weight.data.normal_(mean=0.0, std=initializer_range)
        for bias in self.bias:
            bias.data.zero_()

        if not load_from_shards:
            load_hook = column_serial_load_hook(self, print)
            self._register_load_state_dict_pre_hook(load_hook)

    def forward(self, input_tensor):
        outputs = []
        for i in range(self.model_parallel_size):
            if self.use_ft:
                output_i = LinearTransposeFunction.apply(input_tensor, self.weight[i], self.bias[i], self.head_num, self.transpose_type)
            else:
                assert self.transpose_type == "0213", self.transpose_type
                linear_out = nn.functional.linear(input_tensor, self.weight[i], self.bias[i])
                new_shape = linear_out.size()[:-1] + (self.head_num, self.head_dim)
                linear_out = linear_out.view(*new_shape)
                output_i = linear_out.permute(0, 2, 1, 3).contiguous()
            outputs.append(output_i)
        output = torch.cat(outputs, dim=1)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

    def get_model_parallel_world_size(self):
        return self.model_parallel_size

    def get_model_parallel_group(self):
        return self.slice_proc_group

    # For Megatron-style tensor slicing
    def get_slice_parallel_rank(self):
        return self.model_parallel_id

    def get_slice_parallel_world_size(self):
        return self.model_parallel_size

    def get_slice_parallel_group(self):
        return self.slice_proc_group

    def get_slice_parallel_src_rank(self):
        return self.slice_parallel_src_id




