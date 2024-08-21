# coding=utf-8
import PIL.Image

from saving.saver import format_savers
from saving.utils import Compression


def plugin_save(image: PIL.Image, path: str, compression: Compression):
    ...


format_savers['plugin_file_format'] = plugin_save
