# coding=utf-8
import PIL.Image
import saving.utils as utils

from saving.utils import Compression, apply_lossless_compression
from termcolor import colored


def save(image: PIL.Image.Image, path: str, compression: Compression, sort_by_file_extension: bool) -> None:
    path = utils.sort_by_file_extension(path, sort_by_file_extension, "PNG")

    file_path = path + "png"

    if not compression['lossless']:  # if lossy
        print(colored(
            "WARN: You CAN use lossy compression with PNG format, but this app does not support it :(\n"
            "\tMaking a forced palette PNG", 'yellow'))
        if not compression["additional_lossless"]:
            image = image.convert('P', palette=PIL.Image.ADAPTIVE, colors=255)
        else:
            unique_colors_number = utils.count_unique_colors_python_break_batched(image)

            colors = 2  # benchmarked
            if unique_colors_number > 16:
                colors = 256
            elif unique_colors_number > 4:
                colors = 16
            elif unique_colors_number > 2:
                colors = 4

            image = image.convert('P', palette=PIL.Image.ADAPTIVE, colors=colors)

        image.save(file_path, optimize=True)

    else:
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
