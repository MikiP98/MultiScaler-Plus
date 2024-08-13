# coding=utf-8
# File for filter functions

import PIL.Image
import utils

from aenum import auto, IntEnum, unique
from filtering.filters.normal_map import (
    strength_linear as normal_map_strength_linear,
    strength_exponential as normal_map_strength_exponential
)
from typing import Callable


@unique
class Filters(IntEnum):
    # Lightness filters
    BRIGHTNESS = auto()  # TODO: Implement
    EXPOSURE = auto()  # TODO: Implement
    CONTRAST = auto()  # TODO: Implement
    GAMMA = auto()  # TODO: Implement

    # Color filters
    SATURATION = auto()  # TODO: Implement
    VIBRANCE = auto()  # TODO: Implement

    # Other simple filters
    CAS = auto()  # contrast adaptive sharpening  # TODO: Implement
    SHARPNESS = auto()  # TODO: Implement

    # HDR
    AUTO_HDR = auto()  # TODO: Implement

    # Auto textures
    AUTO_NORMAL_MAP = auto()  # TODO: Implement
    AUTO_SPECULAR_MAP = auto()  # TODO: Implement

    # Texture filters
    NORMAL_MAP_STRENGTH_EXPONENTIAL = auto()
    NORMAL_MAP_STRENGTH_LINEAR = auto()

    SI_TODO = auto()  # TODO: Add filters


filter_functions: dict[auto, Callable[[list[PIL.Image.Image], float], list[PIL.Image.Image]]] = {
    Filters.NORMAL_MAP_STRENGTH_LINEAR: normal_map_strength_linear,
    Filters.NORMAL_MAP_STRENGTH_EXPONENTIAL: normal_map_strength_exponential
}


# TODO: Make it receive a list of filters instead of a single filter
def filter_image_batch(
        filters: list[Filters],
        images: list[utils.ImageDict],
        factors: list[float]
) -> list[list[utils.ImageDict]]:

    result = []

    for img_filter in filters:
        filter_function = filter_functions[img_filter]
        if filter_function is None:
            raise ValueError(f"Filter {img_filter.name} (ID: {img_filter}) is not implemented")

        result.append([
            {
                "images": [filter_function(image["images"][0], factor) for factor in factors],
                "is_animated": image.get("is_animated"),
                "animation_spacing": image.get("animation_spacing"),
            } for image in images
        ])

    return result


def filter_image(
        img_filter: Filters,
        image: utils.ImageDict,
        factor: float
) -> utils.ImageDict:
    return filter_image_batch([img_filter], [image], [factor])[0][0]
