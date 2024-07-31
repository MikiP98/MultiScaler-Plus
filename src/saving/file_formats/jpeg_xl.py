# coding=utf-8
import PIL.Image

from saving.saver import Compression


def save(image: PIL.Image, path: str, compression: Compression):
    file_path = path + "jxl"
    if compression['lossless']:
        image.save(file_path, lossless=True, optimize=True)
    else:
        image.save(file_path, quality=compression['quality'])
