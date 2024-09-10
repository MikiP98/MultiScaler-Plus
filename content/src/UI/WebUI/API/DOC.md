# API Documentation



## GET `./handshake`

### Return

```PY
from typing import TypedDict

class AlgorithmConfigDTO(TypedDict):
    name: str  # Not using ID so that new algorithms won't break API compat
    display_name: str | None  # If left empty, WebUI will try to use the hardcoded clientside display_name
    description: str | None  # If left empty, WebUI will try to use the hardcoded clientside description
    non_cost_per_output_pixel: int
    premium_cost_per_output_pixel: int
    premium_only: bool

class ServerConfigDTO(TypedDict):
  api_version: str
  algorithm_configs: list[AlgorithmConfigDTO]
```

*Example JSON*
```JSON
{
  "api_version": "1.0.0", 
  "algorithm_configs": [
    {
      "id": 20,
      "display_name": "Lanchos *(PIL)*",
      "description": "High quality classic scaling algorithm. In theory best for both down and up scaling among classic algorithms.",
      "non_cost_per_output_pixel": 0,
      "premium_cost_per_output_pixel": 0,
      "premium_only": false
    }
  ]
}
```

## POST `./process_image`

### Send

*Example JSON*
```JSON
{
    "processing_method": 0
}
```

### Return

*Example JSON*
```JSON
{
  "compressed_images_data": [
    "GZIP?"
  ]
}
```