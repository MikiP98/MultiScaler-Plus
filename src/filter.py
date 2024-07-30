# coding=utf-8
# File for filter functions

from aenum import auto, IntEnum, unique
import numpy as np
import utils
import PIL.Image


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


def normal_map_strength_linear(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    new_frames = []

    for frame in frames:
        image_data = np.asarray(frame)

        new_image_data = image_data.copy().astype('int16') - 128  # benchmarked
        # print(new_image_data[..., 0])
        new_image_data[..., 0] = new_image_data[..., 0] * factor  # Red channel
        new_image_data[..., 1] = new_image_data[..., 1] * factor  # Green channel

        new_image_data = new_image_data + 128

        new_frame = PIL.Image.fromarray(new_image_data.astype('uint8'))
        new_frames.append(new_frame)

    # print(f"Applied normal map strength filter with factor {factor}")
    return new_frames


def normal_map_strength_exponential(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    new_frames = []

    for frame in frames:
        image_data = np.asarray(frame)

        # Convert from 0-255 to -128 to 127
        new_image_data = image_data.copy().astype('float32') / 255 * 2 - 1

        # Apply the exponential transformation
        new_image_data[..., 0] = (
                np.sign(new_image_data[..., 0]) * np.abs(new_image_data[..., 0]) ** (1 / factor))  # Red channel
        new_image_data[..., 1] = (
                np.sign(new_image_data[..., 1]) * np.abs(new_image_data[..., 1]) ** (1 / factor))  # Green channel

        # Convert back from -128 to 127 to 0-255
        new_image_data = ((new_image_data + 1) / 2 * 255).astype('uint8')

        new_frame = PIL.Image.fromarray(new_image_data)
        new_frames.append(new_frame)

    # print(f"Applied normal map strength filter with factor {factor}")
    return new_frames


filter_functions = {
    Filters.NORMAL_MAP_STRENGTH_LINEAR: normal_map_strength_linear,
    Filters.NORMAL_MAP_STRENGTH_EXPONENTIAL: normal_map_strength_exponential
}


# TODO: Make it receive a list of filters instead of a single filter
def filter_image_batch(
        img_filter: Filters,
        images: list[utils.ImageDict],
        factors: list[float]
) -> list[utils.ImageDict]:

    filter_function = filter_functions[img_filter]
    if filter_function is None:
        raise ValueError(f"Filter {img_filter.name} is not implemented")

    return [
        {
            "images": [filter_function(image["images"][0], factor) for factor in factors],
            "is_animated": image.get("is_animated"),
            "animation_spacing": image.get("animation_spacing"),
        } for image in images
    ]


def filter_image(
        img_filter: Filters,
        image: utils.ImageDict,
        factors: list[float]
) -> utils.ImageDict:
    return filter_image_batch(img_filter, [image], factors)[0]
