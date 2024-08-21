# coding=utf-8
import PIL.Image
import saving.utils as utils

from saving.utils import Compression


def save(image: PIL.Image, path: str, compression: Compression, sort_by_file_extension: bool) -> None:
    path = utils.sort_by_file_extension(path, sort_by_file_extension, "JPEG_XL")

    file_path = path + "jxl"
    if compression['lossless']:
        image.save(file_path, lossless=True, optimize=True)
    else:
        image.save(file_path, quality=compression['quality'])
