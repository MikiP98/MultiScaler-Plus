# File for saving images

import numpy as np
import PIL.Image
import qoi
import utils

from qoi import QOIColorSpace
from termcolor import colored
from typing import Optional, TypedDict


class Compression(TypedDict):
    additional_lossless: bool
    lossless: bool
    quality: Optional[int]


class SimpleConfig(TypedDict):
    formats: list[str]
    compressions: list[Compression]
    add_compression_to_name: bool


class AdvancedConfig(TypedDict):
    simple_config: SimpleConfig
    file_additives: dict[str, bool]


def generate_file_path(base_path, compression, extension, add_compression_to_name):
    if add_compression_to_name:
        if compression['lossless']:
            suffix = "_lossless+." if compression['additional_lossless'] else "_lossless."
        else:
            suffix = "_lossy+." if compression['additional_lossless'] else "_lossy."
    else:
        suffix = "."
    return f"{base_path}{suffix}{extension}"


def save_image_with_png(image: PIL.Image, path: str, config: SimpleConfig):
    for compression in config['compressions']:
        if not compression['lossless']:
            print(colored(
                "WARN: You CAN use lossy compression with PNG format, but this app does not support it :(\n"
                "SKIPPING", 'yellow'))
            continue
            # TODO: make a force palette as a lossy PNG compression for now

        if not compression['additional_lossless']:
            file_path = generate_file_path(path, compression, "png", config['add_compression_to_name'])
            image.save(file_path, optimize=True)
        else:
            img_byte_arr = utils.apply_lossless_compression_png(image)
            file_path = generate_file_path(path, compression, "png", config['add_compression_to_name'])
            with open(file_path, 'wb') as f:
                f.write(img_byte_arr)


def save_image_with_qoi(image: PIL.Image, path: str, config: SimpleConfig):
    for compression in config['compressions']:
        if not compression['lossless']:
            print(colored(
                "WARN: You CAN use lossy compression with QOI format, but this app does not support it :(\n"
                "SKIPPING", 'yellow'))
            continue

        image_bytes = qoi.encode(np.array(image), colorspace=qoi.QOIColorSpace.SRGB)
        # TODO: pass correct colorspace {LINEAR / SRGB}

        file_path = generate_file_path(path, compression, "qoi", config['add_compression_to_name'])
        with open(file_path, 'wb') as f:
            f.write(image_bytes)


def save_image_with_jpeg_xl(image: PIL.Image, path: str, config: SimpleConfig):
    for compression in config['compressions']:
        file_path = generate_file_path(path, compression, "jxl", config['add_compression_to_name'])
        if compression['lossless']:
            image.save(file_path, lossless=True, optimize=True)
        else:
            image.save(file_path, quality=compression['quality'])


def save_image_with_webp(image: PIL.Image, path: str, config: SimpleConfig):
    for compression in config['compressions']:
        file_path = generate_file_path(path, compression, "webp", config['add_compression_to_name'])
        if compression['lossless']:
            if not compression['additional_lossless']:
                image.save(file_path, lossless=True, method=6, optimize=True)
            else:
                img_byte_arr = utils.apply_lossless_compression_webp(image)
                with open(file_path, 'wb') as f:
                    f.write(img_byte_arr)
        else:
            if not compression['additional_lossless']:
                image.save(file_path, quality=compression['quality'], method=6)


def save_image_with_avif(image: PIL.Image, path: str, config: SimpleConfig):
    for compression in config['compressions']:
        file_path = generate_file_path(path, compression, "avif", config['add_compression_to_name'])
        if compression['lossless']:
            image.save(
                file_path, lossless=True, quality=100, qmin=0, qmax=0, speed=0, subsampling="4:4:4", range="full"
            )
        else:
            image.save(file_path, quality=compression['quality'], speed=0, range="full")


# Global, allows for easy injection of another format
format_savers = {
    "PNG": save_image_with_png,
    "QOI": save_image_with_qoi,
    "JPEG_XL": save_image_with_jpeg_xl,
    "WEBP": save_image_with_webp,
    "AVIF": save_image_with_avif
}


def save_image(image: PIL.Image, path: str, config: SimpleConfig) -> None:
    print(path)

    for format_name in config['formats']:
        format_savers[format_name](image, path, config)

    print(colored(f"{path} Fully Saved!", 'light_green'))


#     if "PNG" in config['formats']:
#         for compression in config['compressions']:
#             if not compression['lossless']:  # if lossy
#                 print(
#                     colored(
#                         "WARN: You CAN use lossy compression with PNG format, but this app does not support it :(\n"
#                         "SKIPPING", 'yellow'
#                     )
#                 )
#                 continue
#                 # TODO: make a force palette as a lossy PNG compression for now
#
#             else:  # if additional lossless
#                 img_byte_arr = utils.apply_lossless_compression_png(image)
#
#                 with open(file_path, 'wb') as f:
#                     f.write(img_byte_arr)
#
#     if "QOI" in config['formats']:
#         for compression in config['compressions']:
#             if not compression['lossless']:
#                 print(
#                     colored(
#                         "WARN: You CAN use lossy compression with QOI format, but this app does not support it :(\n"
#                         "SKIPPING", 'yellow'
#                     )
#                 )
#                 continue
#
#             image_bytes = qoi.encode(np.array(image), colorspace=QOIColorSpace.SRGB)
#             # TODO: pass correct colorspace {LINEAR / SRGB}
#
#             with open(file_path, 'wb') as f:
#                 f.write(image_bytes)
#
#     if "JPEG_XL" in config['formats']:
#         for compression in config['compressions']:
#             if compression['lossless']:
#                 image.save(file_path, lossless=True, optimize=True)
#             else:
#                 image.save(file_path, quality=compression['quality'])
#
#     if "WEBP" in config['formats']:
#         for compression in config['compressions']:
#             if compression['lossless']:
#                 if not compression['additional_lossless']:
#                     image.save(file_path, lossless=True, method=6, optimize=True)
#                 else:
#                     img_byte_arr = utils.apply_lossless_compression_webp(image)
#                     with open(file_path, 'wb') as f:
#                         f.write(img_byte_arr)
#             else:
#                 if not compression['additional_lossless']:
#                     image.save(file_path, quality=compression['quality'], method=6)
#
#     if "AVIF" in config['formats']:
#         for compression in config['compressions']:
#             if compression['lossless']:
#                 image.save(
#                     file_path, lossless=True, quality=100, qmin=0, qmax=0, speed=0, subsampling="4:4:4", range="full"
#                 )
#             else:
#                 image.save(file_path, quality=compression['quality'], speed=0, range="full")
#
#     print(colored(f"{path} Fully Saved!", 'light_green'))


def advanced_save_image(image: PIL.Image, output_path: str, file_name: str, config: AdvancedConfig) -> None:
    save_image(image, f"{output_path}/{file_name}", config['simple_config'])
