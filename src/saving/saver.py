# coding=utf-8
# File for saving images

import os
import PIL.Image
import utils

from saving.file_formats.avif import save as save_avif
from saving.file_formats.jpeg_xl import save as save_jpeg_xl
from saving.file_formats.png import save as save_png
from saving.file_formats.qoi import save as save_qoi
from saving.file_formats.webp import save as save_webp
from saving.utils import SimpleConfig, AdvancedConfig
from termcolor import colored


# Global, allows for easy injection of another format via a plugin
format_savers = {
    "AVIF": save_avif,
    "JPEG_XL": save_jpeg_xl,
    "PNG": save_png,
    "QOI": save_qoi,
    "WEBP": save_webp
}


def save_image(image: PIL.Image, path: str, config: SimpleConfig) -> None:
    print(path)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if any([compression['additional_lossless'] for compression in config['compressions']]):  # TODO: finish this
        additional_lossless_image = image.convert("RGB") if not utils.uses_transparency(image) else image

    for compression in config['compressions']:
        if compression['additional_lossless']:
            image_to_save = additional_lossless_image  # ignore the warning
        else:
            image_to_save = image

        suffix = '.'
        if config['add_compression_to_name']:
            if compression['lossless']:
                suffix = "_lossless+." if compression['additional_lossless'] else "_lossless."
            else:
                suffix = "_lossy+." if compression['additional_lossless'] else "_lossy."

        for format_name in config['formats']:
            format_savers[format_name](image_to_save, path + suffix, compression)

    print(colored(f"{path} Fully Saved!", 'light_green'))


def save_image_pre_processor(image: utils.ImageDict, output_path: str, file_name: str, config: AdvancedConfig) -> None:
    if len(image["images"][0]) > 1:
        print("Stacked and animated images are not supported yet :(")
        return

    if config['sort_by_factor']:
        if config['add_factor_to_name']:
            for filtered_image, factor in zip(image["images"], config['factors']):
                save_image(
                    filtered_image[0],
                    os.path.join("..", "output", str(factor), output_path, f"{file_name}_{factor}"),
                    config['simple_config']
                )

        else:
            for filtered_image, factor in zip(image["images"], config['factors']):
                save_image(
                    filtered_image[0],
                    os.path.join("..", "output", str(factor), output_path, file_name),
                    config['simple_config']
                )

    elif config['add_factor_to_name']:
        for filtered_image, factor in zip(image["images"], config['factors']):
            save_image(
                filtered_image[0],
                os.path.join("..", "output", output_path, f"{file_name}_{factor}"),
                config['simple_config']
            )

    else:
        for filtered_image in image["images"]:
            save_image(filtered_image[0], os.path.join("..", "output", output_path, file_name), config['simple_config'])
