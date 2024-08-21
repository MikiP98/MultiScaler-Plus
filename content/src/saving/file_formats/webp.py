# coding=utf-8
import io
import PIL.Image
import saving.utils as utils

from saving.utils import Compression, apply_lossless_compression
from termcolor import colored


def save(image: PIL.Image.Image, path: str, compression: Compression, sort_by_file_extension: bool) -> None:
    path = utils.sort_by_file_extension(path, sort_by_file_extension, "WEBP")

    w, h = image.size
    if w > 16383 or h > 16383:
        print(colored(
            "SEVERE WARNING: Maximum WEBP image size is 16383 pixels in any direction; "
            "Resulting image will be scaled down to fix;",
            "light_red"
        ))
        if w > h:
            factor = 16383 / w
            h = min(round(h * factor), 16383)
            w = 16383
        else:
            factor = 16383 / h
            w = min(round(h * factor), 16383)
            h = 16383
        image = image.resize((w, h), PIL.Image.Resampling.LANCZOS)

    file_path = path + "webp"
    if compression['lossless']:
        if not compression['additional_lossless']:
            image.save(file_path, lossless=True, method=6, optimize=True)
        else:
            img_byte_arr = apply_additional_lossless_compression(image)
            with open(file_path, 'wb+') as f:
                f.write(img_byte_arr)
    else:
        image.save(file_path, quality=compression['quality'], method=6)

        if not compression['additional_lossless']:
            image.save(file_path, quality=compression['quality'], method=6)
        else:  # if additional lossless
            palette_img_byte_arr = apply_additional_lossless_compression(image)

            lossy_img_byte_arr = io.BytesIO()
            image.save(lossy_img_byte_arr, quality=compression['quality'], method=6, format='WEBP')
            lossy_img_byte_arr = lossy_img_byte_arr.getvalue()

            final_img_byte_arr = palette_img_byte_arr if len(palette_img_byte_arr) < len(lossy_img_byte_arr) \
                else lossy_img_byte_arr

            with open(file_path, 'wb+') as f:
                f.write(final_img_byte_arr)


def apply_additional_lossless_compression(image: PIL.Image) -> bytes:
    optional_args = {
        'lossless': True,
        'method': 6,
        'optimize': True,
        'format': 'WEBP'
    }
    return apply_lossless_compression(image, optional_args)
