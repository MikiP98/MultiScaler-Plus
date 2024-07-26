# coding=utf-8
# File for future filter functions

import numpy as np
import utils
import PIL.Image

from utils import Filters


def normal_map_strength(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    new_frames = []

    for frame in frames:
        image_data = np.asarray(frame)

        if image_data.shape[2] == 4:
            scaling_vector = np.array([factor, factor, 1, 1])
        else:
            scaling_vector = np.array([factor, factor, 1])

        new_image_data = image_data * scaling_vector

        new_frame = PIL.Image.fromarray(new_image_data.astype('uint8'))
        new_frames.append(new_frame)

    print(f"Applied normal map strength filter with factor {factor}")
    return new_frames


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
            "is_animated": image.get("is_animated"),
            "animation_spacing": image.get("animation_spacing"),
        } for image in images for factor in factors
    ]
