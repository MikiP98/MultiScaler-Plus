# coding=utf-8
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
