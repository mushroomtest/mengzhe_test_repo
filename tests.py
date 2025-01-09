### GET request to example server
qweqweq
# curl -X POST
#  'https://open.volcengineapi.com/?Action=GetApiKey&Version=2024-01-01'
#  -H 'Authorization: HMAC-SHA256 Credential=AKLTYTZmMTZlMzg2MTg1NDRjY2EwMTRjNDAxYmQxZjJkMmI/20240530/cn-beijing/ark/request, SignedHeaders=host;x-content-sha256;x-date, Signature=c3338fd26959952bf0894fcf6d100f9f191d13f0cf4822325e5a36c768d64409'
#  -H 'Content-Type: application/json'
#  -H 'Host: open.volcengineapi.com'
#  -H 'X-Content-Sha256: 2e614b80ed4186084ebe4b6b364c26ec82a42cbb20255cf60daeec799e11134f'
#  -H 'X-Date: 20240530T070820Z'
#  -d '{
#    "ResourceType": "endpoint",
#    "DurationSeconds": 2592000,
#    "ResourceIds": [
#        "ep-20240529112627-vz77r"
#    ]

AKLTZGNkZDUyMmYxMmRkNDBmMjg0ZGRkNGYwYzI4ZDYyMjA
#}'
POST https://open.volcengineapi.com/?Action=GetApiKey&Version=2024-01-01
Authorization: HMAC-SHA256 Credential=AKLTYTZmMTZlMzg2MTg1NDRjY2EwMTRjNDAxYmQxZjJkMmI/20240530/cn-beijing/ark/request, SignedHeaders=host;x-content-sha256;x-date, Signature=c3338fd26959952bf0894fcf6d100f9f191d13f0cf4822325e5a36c768d64409
Host: open.volcengineapi.com
X-Content-Sha256: 2e614b80ed4186084ebe4b6b364c26ec82a42cbb20255cf60daeec799e11134f
X-Date: 20240530T070820Z
Content-Type: application/json

AKLTZGNkZDUyMmYxMmRkNDBmMjg0ZGRkNGYwYzI4ZDYyMjA
{
  "ResourceType": "endpoint",
  "DurationSeconds": 2592000,
  "ResourceIds": [
    "ep-20240529112627-vz77r"
  ]
}

###
