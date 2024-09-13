# coding=utf-8
from typing import TypedDict


class EndpointProcess(TypedDict):
    name: str  # Not using ID so that new algorithms won't break API compat
    display_name: str | None  # If left empty, WebUI will try to use the hardcoded clientside display_name
    description: str | None  # If left empty, WebUI will try to use the hardcoded clientside description
    tags: list[str]
    non_cost_per_output_pixel: int
    premium_cost_per_output_pixel: int
    premium_only: bool | None  # if None, parent premium_only value will be used


class EndpointConfigDTO(TypedDict):
    endpoint: str
    display_name: str
    processes: list[EndpointProcess]
    premium_only: bool  # overrides all processes premium_only value if true


class ServerConfigDTO(TypedDict):
    api_version: str
    endpoints: list[EndpointConfigDTO]
    make_premium_a_tag: bool  # If true, only premium image processing methods will have additional `Premium` tag
    require_account_for_non_zero_costs_processing: bool
