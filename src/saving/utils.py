# coding=utf-8
import io
import os
import PIL.Image
import utils

from aenum import IntEnum
from typing import Optional, TypedDict


class Compression(TypedDict):
    additional_lossless: bool
    lossless: bool
    quality: Optional[int]


class SimpleConfig(TypedDict):
    formats: list[str]
    compressions: list[Compression]
    add_compression_to_name: bool  # TODO: Add auto option
    sort_by_file_extension: bool  # TODO: Implement this


class AdvancedConfig(TypedDict):
    simple_config: SimpleConfig

    add_factor_to_name: bool
    sort_by_factor: bool

    # A.K.A. algorithm or filter
    add_processing_method_to_name: bool
    sort_by_processing_method: bool

    sort_by_image: bool
    sort_by_file_extension: int  # -1 - auto, 0 - no, 1 - yes TODO: Implement this
    # TODO: Add more auto options

    factors: Optional[list[float]]
    processing_method: Optional[IntEnum]


def count_unique_colors_python_break_batched(image: PIL.Image) -> int:
    # Get the image data as a list of pixels
    pixels = list(image.getdata())

    scanned_pixels = min(256, len(pixels))

    # Create a set to store the unique colors
    unique_colors = set(pixels[:scanned_pixels])

    # Loop over the pixels and add them to the set
    while len(unique_colors) <= 256:
        # print(f"Iteration; scanned_pixels: {scanned_pixels}; unique_colors: {len(unique_colors)}")
        new_scanned_pixels = min(scanned_pixels + 257 - len(unique_colors), len(pixels))
        unique_colors.update(pixels[scanned_pixels:new_scanned_pixels])
        if new_scanned_pixels == len(pixels):
            break
        scanned_pixels = new_scanned_pixels
    else:  # no break happened, so len(unique_colors) > 256
        return 320

    # Return the number of unique colors
    return len(unique_colors)


def apply_lossless_compression(image: PIL.Image, optional_args: dict) -> bytes:
    img_byte_arr = io.BytesIO()

    mode = 'RGBA'
    if not utils.has_transparency(image):
        mode = 'RGB'

    image.save(img_byte_arr, **optional_args)

    # unique_colors_number = len(set(image.getdata()))
    unique_colors_number = utils.count_unique_colors_python_break_batched(image)
    # print(f"Unique colors: {unique_colors_number}")
    if unique_colors_number <= 256:

        colors = 2  # benchmarked
        if unique_colors_number > 16:
            colors = 256
        elif unique_colors_number > 4:
            colors = 16
        elif unique_colors_number > 2:
            colors = 4

        img_temp_byte_arr = io.BytesIO()
        temp_image = image.convert('P', palette=PIL.Image.ADAPTIVE, colors=colors)  # sometimes deletes some data :/

        # Additional check to see if PIL didn't fuck up, (it sometimes it wrong)
        if image.getdata() == temp_image.convert(mode).getdata():  # benchmarked
            temp_image.save(img_temp_byte_arr, **optional_args)

            # Check which one is smaller and keep it, remove the other one
            if len(img_temp_byte_arr.getvalue()) < len(img_byte_arr.getvalue()):
                img_byte_arr = img_temp_byte_arr
                # print("Saving palette")

    return img_byte_arr.getvalue()


def sort_by_file_extension(path: str, sort_by_file_extension: bool, extension: str) -> str:
    if sort_by_file_extension:
        path_parts = path.split(os.path.sep)
        path_parts.insert(2, extension)
        path = os.path.sep.join(path_parts)  # TODO: benchmark this vs os.path.join(*path_parts)

    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # Path created by another thread

    return path
