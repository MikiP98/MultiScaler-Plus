# coding=utf-8
import PIL.Image

from saving.saver import Compression


def save(image: PIL.Image, path: str, compression: Compression):
    raise NotImplementedError("This is a template file for saving images. Implement this function in a new file.")
