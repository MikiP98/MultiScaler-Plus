# coding=utf-8
# File for saving images

import numpy as np
import os
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

    add_factor_to_name: bool
    sort_by_factor: bool

    factors: list[float]


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
            with open(file_path, 'wb+') as f:
                f.write(img_byte_arr)


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


def save_image_with_jpeg_xl(image: PIL.Image, path: str, config: SimpleConfig):
    for compression in config['compressions']:
        file_path = generate_file_path(path, compression, "jxl", config['add_compression_to_name'])
        if compression['lossless']:
            image.save(file_path, lossless=True, optimize=True)
        else:
            image.save(file_path, quality=compression['quality'])
    #     if "JPEG_XL" in config['formats']:
    #         for compression in config['compressions']:
    #             if compression['lossless']:
    #                 image.save(file_path, lossless=True, optimize=True)
    #             else:
    #                 image.save(file_path, quality=compression['quality'])


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


def save_image_with_avif(image: PIL.Image, path: str, config: SimpleConfig):
    for compression in config['compressions']:
        file_path = generate_file_path(path, compression, "avif", config['add_compression_to_name'])
        if compression['lossless']:
            image.save(
                file_path, lossless=True, quality=100, qmin=0, qmax=0, speed=0, subsampling="4:4:4", range="full"
            )
        else:
            image.save(file_path, quality=compression['quality'], speed=0, range="full")


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

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    for format_name in config['formats']:
        # print(colored(f"Saving {format_name}...", 'light_cyan'))
        format_savers[format_name](image, path, config)
        # print(colored(f"{format_name} Fully Saved!", 'light_green'))

    print(colored(f"{path} Fully Saved!", 'light_green'))


def save_image_pre_processor(image: utils.ImageDict, output_path: str, file_name: str, config: AdvancedConfig) -> None:
    # if config['add_factor_to_name']:
    #     for factor in config['factors']:
    #         save_image(image, f"{output_path}/{file_name}_{factor}", config['simple_config'])

    if len(image["images"][0]) > 1:
        print("Stacked and animated images are not supported yet")
        return

    if config['sort_by_factor']:
        for filtered_image, factor in zip(image["images"], config['factors']):
            save_image(filtered_image[0], os.path.join(output_path, str(factor), file_name), config['simple_config'])

    elif config['add_factor_to_name']:
        for filtered_image, factor in zip(image["images"], config['factors']):
            save_image(filtered_image[0], os.path.join(output_path, f"{file_name}_{factor}"), config['simple_config'])

    else:
        for filtered_image in image["images"]:
            save_image(filtered_image[0], os.path.join(output_path, file_name), config['simple_config'])
