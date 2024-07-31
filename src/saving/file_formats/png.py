# coding=utf-8
import PIL.Image
import utils

from saving.saver import Compression
from termcolor import colored


def save(image: PIL.Image, path: str, compression: Compression):
    if not compression['lossless']:  # if lossy
        print(colored(
            "WARN: You CAN use lossy compression with PNG format, but this app does not support it :(\n"
            "SKIPPING", 'yellow'))
        return
        # TODO: make a force palette as a lossy PNG compression for now

    file_path = path + "png"

    # If lossless
    if not compression['additional_lossless']:
        image.save(file_path, optimize=True)
    else:  # if additional lossless
        img_byte_arr = utils.apply_lossless_compression_png(image)
        with open(file_path, 'wb+') as f:
            f.write(img_byte_arr)
