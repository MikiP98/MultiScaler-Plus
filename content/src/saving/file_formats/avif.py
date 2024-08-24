# coding=utf-8
import PIL.Image
import pillow_avif  # This is a PIL plugin for AVIF, is must be imported, but isn't directly used
import saving.utils as utils

from saving.utils import Compression


def save(image: PIL.Image, path: str, compression: Compression, sort_by_file_extension: bool) -> None:
    path = utils.sort_by_file_extension(path, sort_by_file_extension, "AVIF")

    file_path = path + "avif"
    if compression['lossless']:
        image.save(
            file_path, lossless=True, quality=100, qmin=0, qmax=0, speed=0, subsampling="4:4:4", range="full"
        )
    else:
        image.save(file_path, quality=compression['quality'], speed=0, range="full")
