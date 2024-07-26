# coding=utf-8
# File for future filter functions

import utils
import PIL.Image

from utils import Filters


def normal_map_strength(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    for frame in frames:
        for x in range(frame.width):
            for y in range(frame.height):
                r, g, b = frame.getpixel((x, y))
                r = int(r * factor)
                g = int(g * factor)
                frame.putpixel((x, y), (r, g, b))

    return frames


filter_functions = {
    Filters.NORMAL_MAP_STRENGTH: normal_map_strength
}


def filter_image_batch(
        img_filter: Filters,
        images: list[utils.ImageDict],
        factors: list[float]
) -> list[utils.ImageDict]:

    filter_function = filter_functions[img_filter]
    if filter_function is None:
        raise ValueError(f"Filter {img_filter} is not implemented")

    return [
        {
            "images": filter_function(image["images"][0], factor),
            "is_animated": image["is_animated"],
            "animation_spacing": image["animation_spacing"],
        } for image in images for factor in factors
    ]
