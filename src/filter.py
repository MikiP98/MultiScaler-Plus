# coding=utf-8
# File for future filter functions

import numpy as np
import utils
import PIL.Image

from utils import Filters


def normal_map_strength_linear(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    new_frames = []

    for frame in frames:
        image_data = np.asarray(frame)

        new_image_data = image_data.copy().astype('int16') - 128
        # print(new_image_data[..., 0])
        new_image_data[..., 0] = new_image_data[..., 0] * factor  # Red channel
        new_image_data[..., 1] = new_image_data[..., 1] * factor  # Green channel

        new_image_data = new_image_data + 128

        # if image_data.shape[2] == 4:
        #     scaling_vector = np.array([factor, factor, 1, 1])
        # else:
        #     scaling_vector = np.array([factor, factor, 1])
        #
        # new_image_data = image_data * scaling_vector

        new_frame = PIL.Image.fromarray(new_image_data.astype('uint8'))
        new_frames.append(new_frame)

    # print(f"Applied normal map strength filter with factor {factor}")
    return new_frames


def normal_map_strength_exponential(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    raise NotImplementedError("This function is not implemented yet")
    new_frames = []

    for frame in frames:
        image_data = np.asarray(frame)

        # Apply the new normal map strength calculation only to the red and green channels
        new_image_data = image_data.copy()
        new_image_data[..., 0] = (image_data[..., 0] / 255) ** (1/factor) * 255  # Red channel
        new_image_data[..., 1] = (image_data[..., 1] / 255) ** (1/factor) * 255  # Green channel
        # new_image_data[..., 0] = 1/factor * image_data[..., 0] - 255  # Red channel
        # new_image_data[..., 1] = 1/factor * image_data[..., 1] - 255  # Green channel

        new_frame = PIL.Image.fromarray(new_image_data)
        new_frames.append(new_frame)

    # print(f"Applied normal map strength filter with factor {factor}")
    return new_frames


filter_functions = {
    Filters.NORMAL_MAP_STRENGTH_LINEAR: normal_map_strength_linear,
    Filters.NORMAL_MAP_STRENGTH_EXPONENTIAL: normal_map_strength_exponential
}


def filter_image_batch(
        img_filter: Filters,
        images: list[utils.ImageDict],
        factors: list[float]
) -> list[utils.ImageDict]:

    filter_function = filter_functions[img_filter]
    if filter_function is None:
        raise ValueError(f"Filter {img_filter} is not implemented")

    # print(f"Filter: {img_filter.name}")
    # print(f"Factors: {factors}")

    result = [
        {
            "images": [filter_function(image["images"][0], factor) for factor in factors],
            "is_animated": image.get("is_animated"),
            "animation_spacing": image.get("animation_spacing"),
        } for image in images
    ]
    # print(f"Resulted in: {len(result)} images")
    # print(f"Resulting images have {len(result[0]['images'])} factors inside")
    return result

    # return [
    #     {
    #         "images": [filter_function(image["images"][0], factor) for factor in factors],
    #         "is_animated": image.get("is_animated"),
    #         "animation_spacing": image.get("animation_spacing"),
    #     } for image in images
    # ]
