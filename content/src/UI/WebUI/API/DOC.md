# API Documentation



## GET `./handshake`

### Return

```PY
from typing import TypedDict

class AlgorithmConfigDTO(TypedDict):
    name: str  # Not using ID so that new algorithms won't break API compat
    display_name: str | None  # If left empty, WebUI will try to use the hardcoded clientside display_name
    description: str | None  # If left empty, WebUI will try to use the hardcoded clientside description
    tags: list[str]
    non_cost_per_output_pixel: int
    premium_cost_per_output_pixel: int
    premium_only: bool

class ServerConfigDTO(TypedDict):
    api_version: str
    algorithm_configs: list[AlgorithmConfigDTO]
    make_premium_a_tag: bool  # If true, only premium image processing methods will have additional `Premium` tag
```

*Example JSON*
```JSON
{
  "api_version": "1.0.0", 
  "algorithm_configs": [
    {
      "name": "PIL_LANCHOS",
      "display_name": "Lanchos *(PIL)*",
      "description": "High quality classic scaling algorithm. In theory best for both down and up scaling among classic algorithms.",
      "tags": ["Classic"],
      "non_cost_per_output_pixel": 0,
      "premium_cost_per_output_pixel": 0,
      "premium_only": false
    }
  ],
  "make_premium_a_tag": true
}
```

## POST `./process/scale`

### Send

*Example JSON*
```JSON
{
    "images": [
      
    ]
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

## POST `./process/filter`

### Send

*Example JSON*
```JSON
{
    "images": [
      
    ]
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