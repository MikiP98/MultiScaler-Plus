# coding=utf-8
import numpy as np
import PIL.Image
import qoi
import saving.utils as utils

from saving.utils import Compression
from termcolor import colored


def save(image: PIL.Image, path: str, compression: Compression, sort_by_file_extension: bool) -> None:
    if not compression['lossless']:  # if lossy
        print(colored(
            "WARN: You CAN use lossy compression with QOI format, but this app does not support it :(\n"
            "SKIPPING", 'yellow'))
        return

    path = utils.sort_by_file_extension(path, sort_by_file_extension, "QOI")

    image_bytes = qoi.encode(np.array(image), colorspace=qoi.QOIColorSpace.SRGB)
    # TODO: pass correct colorspace {LINEAR / SRGB}

    with open(path + "qoi", 'wb+') as f:
        f.write(image_bytes)
