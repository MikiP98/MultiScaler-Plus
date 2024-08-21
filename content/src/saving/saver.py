# coding=utf-8
# File for saving images

import fractions
import itertools
import os
import PIL.Image
import threading
import utils

from itertools import zip_longest
from math import ceil
from saving.file_formats.avif import save as save_avif
from saving.file_formats.jpeg_xl import save as save_jpeg_xl
from saving.file_formats.png import save as save_png
from saving.file_formats.qoi import save as save_qoi
from saving.file_formats.webp import save as save_webp
from saving.utils import Compression, AdvancedConfig, SimpleConfig
from termcolor import colored
from typing import Any, Callable


# Global, allows for easy injection of another format via a plugin
format_savers: dict[str, Callable[[PIL.Image.Image, str, Compression, bool], None]] = {
    "AVIF": save_avif,
    "JPEG_XL": save_jpeg_xl,
    "PNG": save_png,
    "QOI": save_qoi,
    "WEBP": save_webp
}


def save_image(image: PIL.Image.Image, path: str, config: SimpleConfig) -> None:
    print(colored(path, "light_green"))

    if any([compression['additional_lossless'] for compression in config['compressions']]):  # TODO: benchmark this
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
            format_savers[format_name.upper()](
                image_to_save,
                path + suffix,
                compression,
                config['sort_by_file_extension']
            )

    print(colored(f"{path} Fully Saved!", 'green'))


def save_image_pre_processor(image: utils.ImageDict, output_path: str, file_name: str, config: AdvancedConfig) -> None:
    if len(image["images"][0]) == 0:
        print(colored("WARNING: Empty (probably skipped by processing method) image", "yellow"))
        return

    if len(image["images"][0]) > 1:
        print("Stacked and animated images are not supported yet :(")
        return

    output_path_parts: list[str] = ["..", "..", "output"]
    file_name_prefix: list[str] = []

    # TODO: benchmark this vs all in the loop
    if config['sort_by_processing_method']:
        processing_method = config['processing_method']
        if not type(processing_method) is str:
            processing_method = processing_method.name

        output_path_parts.append(processing_method)

    if config['add_processing_method_to_name']:
        processing_method = config['processing_method']
        if not type(processing_method) is str:
            processing_method = processing_method.name

        file_name_prefix.append(processing_method)

    if config['factors'] is None:
        config['factors'] = []

    for processed_image, factor in zip_longest(image["images"], config['factors']):
        file_name_part: list[str] = [file_name]
        final_output_path_parts = output_path_parts.copy()

        if config['sort_by_image']:
            final_output_path_parts.append(file_name)

        if factor is not None:
            if config['sort_by_factor']:
                final_output_path_parts.append(factor_to_string(factor))
            if config['add_factor_to_name']:
                file_name_part.append(factor_to_string(factor))

        new_file_name = "_".join([*file_name_prefix, *file_name_part])

        full_output_path = os.path.join(*final_output_path_parts, output_path, new_file_name)

        # print(f"Saving to: {full_output_path}")

        save_image(
            processed_image[0],
            full_output_path,
            config['simple_config']
        )


def factor_to_string(factor: float) -> str:
    if factor.is_integer():
        return f"{int(factor)}x"
    else:
        factor = fractions.Fraction(factor)
        return f"({factor.numerator}%{factor.denominator}x)"


def save_img_list(bundle: tuple[list[utils.ImageDict], list[str], list[str]], saver_config: AdvancedConfig):
    # bundle is a zip, but is a tuple...
    # print(f"Type of bundle: {type(bundle)}")
    for filtered_image, root, file_name in bundle:
        save_image_pre_processor(
            filtered_image,
            root[12:],
            file_name,
            saver_config
        )


def save_img_list_multithreaded(
        processed_images: list[list[utils.ImageDict]],
        roots: list[str],
        file_names: list[str],
        saver_config: AdvancedConfig,
        processing_methods: list[Any],
        *,
        max_thread_count: int = 4
):
    print("Saving images...")
    # processes_loop_threads = min(round(len(processed_images) / 2), max_thread_count)
    # print(f"Using {processes_loop_threads} threads")
    # bundle_split_threads = len(processed_images) // processes_loop_threads
    # print(f"Splitting the bundle into {bundle_split_threads} parts")

    bundle_split_threads = min(max(round(len(roots) / 4), 1), max_thread_count)
    # print(f"Splitting the bundle into {bundle_split_threads} parts")
    batched_cache = ceil(len(roots) / bundle_split_threads)
    # print(f"Split the bundle into sizes of {batched_cache}\n")

    # TODO: fix if no images are loaded
    for processed_image_set, processing_method in zip(processed_images, processing_methods):
        current_saver_config = saver_config.copy()
        current_saver_config['processing_method'] = processing_method

        bundle = zip(processed_image_set, roots, file_names)

        bundle_splits = itertools.batched(bundle, batched_cache)
        threads = []

        for bundle_split in bundle_splits:
            thread = threading.Thread(
                target=save_img_list,
                args=(bundle_split, current_saver_config)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
