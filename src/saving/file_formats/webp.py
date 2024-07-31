# coding=utf-8
import io
import PIL.Image
import utils

from saving.saver import Compression


def save(image: PIL.Image, path: str, compression: Compression):
    file_path = path + "webp"
    if compression['lossless']:
        if not compression['additional_lossless']:
            image.save(file_path, lossless=True, method=6, optimize=True)
        else:
            img_byte_arr = utils.apply_lossless_compression_webp(image)
            with open(file_path, 'wb+') as f:
                f.write(img_byte_arr)
    else:
        image.save(file_path, quality=compression['quality'], method=6)

        if not compression['additional_lossless']:
            image.save(file_path, quality=compression['quality'], method=6)
        else:  # if additional lossless
            palette_img_byte_arr = utils.apply_lossless_compression_webp(image)

            lossy_img_byte_arr = io.BytesIO()
            image.save(lossy_img_byte_arr, quality=compression['quality'], method=6)
            lossy_img_byte_arr = lossy_img_byte_arr.getvalue()

            final_img_byte_arr = palette_img_byte_arr if len(palette_img_byte_arr) < len(lossy_img_byte_arr) \
                else lossy_img_byte_arr

            with open(file_path, 'wb+') as f:
                f.write(final_img_byte_arr)
