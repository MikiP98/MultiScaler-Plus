# coding=utf-8
import PIL.Image

from saving.saver import Compression, format_savers


def plugin_save(image: PIL.Image, path: str, compression: Compression):
    ...


format_savers['plugin_file_format'] = plugin_save
