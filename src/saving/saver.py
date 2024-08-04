# coding=utf-8
# File for saving images

import itertools
import os
import PIL.Image
import threading
import utils

from math import ceil
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

    if not os.path.exists(os.path.dirname(path)):  # TODO: move this check higher
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # Path created by another thread

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

    output_path_parts: list[str] = ["..", "output"]
    file_name_part: list[str] = [file_name]

    for filtered_image, factor in zip(image["images"], config['factors']):
        # TODO: Tweak the order
        if config['sort_by_factor']:
            output_path_parts.append(str(factor))
        if config['sort_by_processing_method']:  # TODO: fix
            output_path_parts.append(config['processing_method'].name)

        if config['add_factor_to_name']:
            file_name_part.append(str(factor))
        if config['add_processing_method_to_name']:
            file_name_part.append(config['processing_method'].name)

        new_file_name = "_".join(file_name_part)

        full_output_path = os.path.join(*output_path_parts, output_path, new_file_name)

        save_image(
            filtered_image[0],
            full_output_path,
            config['simple_config']
        )

    # # TODO: Remove redundancy
    # if config['sort_by_factor']:
    #     if config['add_factor_to_name']:
    #         for filtered_image, factor in zip(image["images"], config['factors']):
    #             save_image(
    #                 filtered_image[0],
    #                 os.path.join("..", "output", str(factor), output_path, f"{file_name}_{factor}"),
    #                 config['simple_config']
    #             )
    #
    #     else:
    #         for filtered_image, factor in zip(image["images"], config['factors']):
    #             save_image(
    #                 filtered_image[0],
    #                 os.path.join("..", "output", str(factor), output_path, file_name),
    #                 config['simple_config']
    #             )
    #
    # elif config['add_factor_to_name']:
    #     for filtered_image, factor in zip(image["images"], config['factors']):
    #         save_image(
    #             filtered_image[0],
    #             os.path.join("..", "output", output_path, f"{file_name}_{factor}"),
    #             config['simple_config']
    #         )
    #
    # else:
    #     for filtered_image in image["images"]:
    #         save_image(filtered_image[0], os.path.join("..", "output", output_path, file_name), config['simple_config'])


def save_img_list(bundle, saver_config):
    for filtered_image, root, file_name in bundle:
        save_image_pre_processor(
            filtered_image,
            root[9:],
            file_name,
            saver_config
        )


def save_img_list_multithreaded(processed_images: list[list[utils.ImageDict]], roots, file_names, saver_config, processing_methods, *, max_thread_count=4):
    # processes_loop_threads = min(round(len(processed_images) / 2), max_thread_count)
    # print(f"Using {processes_loop_threads} threads")
    # bundle_split_threads = len(processed_images) // processes_loop_threads
    # print(f"Splitting the bundle into {bundle_split_threads} parts")

    bundle_split_threads = min(max(round(len(roots) / 4), 1), max_thread_count)
    print(f"Splitting the bundle into {bundle_split_threads} parts")
    batched_cache = ceil(len(roots) / bundle_split_threads)
    print(f"Split the bundle into sizes of {batched_cache}\n")

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