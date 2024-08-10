# coding=utf-8
import PIL.Image
import saving.utils as utils

from saving.utils import Compression, apply_lossless_compression
from termcolor import colored


def save(image: PIL.Image, path: str, compression: Compression, sort_by_file_extension: bool) -> None:
    if not compression['lossless']:  # if lossy
        print(colored(
            "WARN: You CAN use lossy compression with PNG format, but this app does not support it :(\n"
            "SKIPPING", 'yellow'))
        return
        # TODO: make a force palette as a lossy PNG compression for now

    path = utils.sort_by_file_extension(path, sort_by_file_extension, "PNG")

    file_path = path + "png"

    # If lossless
    if not compression['additional_lossless']:
        image.save(file_path, optimize=True)
    else:  # if additional lossless
        img_byte_arr = apply_additional_lossless_compression(image)
        with open(file_path, 'wb+') as f:
            f.write(img_byte_arr)


def apply_additional_lossless_compression(image: PIL.Image) -> bytes:
    optional_args = {
        'optimize': True,
        'format': 'PNG'
    }
    return apply_lossless_compression(image, optional_args)
