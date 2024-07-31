# coding=utf-8
import PIL.Image

from saving.saver import Compression


def save(image: PIL.Image, path: str, compression: Compression):
    file_path = path + "avif"
    if compression['lossless']:
        image.save(
            file_path, lossless=True, quality=100, qmin=0, qmax=0, speed=0, subsampling="4:4:4", range="full"
        )
    else:
        image.save(file_path, quality=compression['quality'], speed=0, range="full")
