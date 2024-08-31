# coding=utf-8
import PIL.Image
import saving.utils as utils

from saving.utils import Compression, apply_lossless_compression


def save(image: PIL.Image.Image, path: str, compression: Compression, sort_by_file_extension: bool) -> None:
    path = utils.sort_by_file_extension(path, sort_by_file_extension, "PLUGIN_EXTENSION")
    raise NotImplementedError("This is a template file for saving images. Implement this function in a new file.")
