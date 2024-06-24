# coding=utf-8
import argparse
import concurrent.futures
import multiprocessing
# import time
import numpy as np
import os
import PIL.Image
import PIL.GifImagePlugin
import pillow_avif  # This is a PIL plugin for AVIF, is must be imported, but isn't directly used
import pillow_jxl  # This is a PIL plugin for JPEG XL, is must be imported, but isn't directly used
import psutil
import qoi
import scaler
import sys
import shutil
import utils
import zipfile

from fractions import Fraction
from functools import lru_cache
from termcolor import colored
from termcolor._types import Color as TermColor  # Ignore this, TODO: create an issue on the termcolor repo.
from utils import (
    Algorithms,
    avg,
    pil_fully_supported_formats_cache,
    pil_read_only_formats_cache,
    pil_write_only_formats_cache,
    pil_indentify_only_formats_cache,
    pngify
)


PIL.Image.MAX_IMAGE_PIXELS = 200000000
PIL.GifImagePlugin.LOADING_STRATEGY = PIL.GifImagePlugin.LoadingStrategy.RGB_ALWAYS


format_to_extension = {
    "JPEG_XL": "jxl",
    "PNG": "png",
    "QOI": "qoi",
    "WEBP": "webp",
    "AVIF": "avif"
}


# time_of_file_extension = {}
def save_image(algorithm: Algorithms, image: PIL.Image, root: str, file: str, scale, config: dict) -> None:
    # global time_of_file_extension

    # print(f"Saving algorithm: {algorithm} {algorithm.name}, root: {root}, file: {file}, scale: {scale}")
    path = os.path.join(root, file)

    if image is None:
        print(f"Saving image: {path}, is probably handled by another thread")
        return

    new_file_name = file
    if config['add_algorithm_name_to_output_files_names']:
        new_file_name = f"{algorithm.name}_{new_file_name}"

    if config['add_factor_to_output_files_names']:
        # Check if the scale is a float, if it is, convert it to a fraction
        if scale != int(scale):
            if len(str(scale).split(".")[1]) > 3:
                # Replace '/' with '%', because '/' is not allowed in file names
                scale = f"{str(Fraction(scale).limit_denominator()).replace('/', '%')}"
        new_file_name = f"{new_file_name[:-4]}_{scale}x{new_file_name[-4:]}"
    # print(new_file_name)

    output_dir = ""
    if config['sort_by_algorithm']:
        output_dir += f"/{algorithm.name}"

    if config['sort_by_scale']:
        output_dir += f"/x{scale}"

    if config['sort_by_image']:
        output_dir += f"/{file}"

    if type(config['file_formats']) is not set:
        config['file_formats'] = {config['file_formats']}
        if config['sort_by_file_extension'] == -1:
            config['sort_by_file_extension'] = 0
    else:
        if config['sort_by_file_extension'] == -1:
            if len(config['file_formats']) > 1:
                config['sort_by_file_extension'] = 1

    if config['additional_lossless_compression']:
        if not utils.uses_transparency(image):
            image = image.convert('RGB')

    new_file_name_parts = new_file_name.split('.')

    for file_format in config['file_formats']:
        # Start timer
        # start_time = time.time()

        new_file_name = '.'.join(new_file_name_parts[:-1]) + '.' + format_to_extension[file_format]
        file_format_dir = ""
        if config['sort_by_file_extension'] == 1:
            file_format_dir = f"/{format_to_extension[file_format]}"

        final_output_dir = "../output" + file_format_dir + output_dir + root.lstrip("../input")
        # Create output directory if it doesn't exist
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)

        output_path = final_output_dir + '/' + new_file_name
        print(output_path)

        if "PNG" == file_format:
            if not config['lossless_compression']:
                print(
                    colored(
                        "WARN: You CAN use lossy compression with PNG format, but this app does not support it :(\n"
                        "Proceeding to use lossless compression", 'yellow'
                    )
                )

            # output_path = output_path.replace(".jpg", ".png").replace(".jpeg", ".png")
            if not config['additional_lossless_compression']:
                image.save(output_path, optimize=True)
            else:
                img_byte_arr = utils.apply_lossless_compression_png(image)
                with open(output_path, 'wb') as f:
                    f.write(img_byte_arr)

        elif "QOI" == file_format:
            if not config['lossless_compression']:
                print(
                    colored(
                        "WARN: You CAN use lossy compression with QOI format, but this app does not support it :(\n"
                        "Proceeding to use lossless compression", 'yellow'
                    )
                )

            image_bytes = qoi.encode(np.array(image))
            with open(output_path, 'wb') as f:
                f.write(image_bytes)

        elif "JPEG_XL" == file_format:
            if config['lossless_compression']:
                image.save(output_path, lossless=True, optimize=True)
            else:
                image.save(output_path, quality=config['quality'])

        elif "WEBP" == file_format:
            if config['lossless_compression']:
                if not config['additional_lossless_compression']:
                    image.save(output_path, lossless=True, method=6, optimize=True)
                else:
                    img_byte_arr = utils.apply_lossless_compression_webp(image)
                    with open(output_path, 'wb') as f:
                        f.write(img_byte_arr)
            else:
                if not config['additional_lossless_compression']:
                    image.save(output_path, quality=config['quality'], method=6)

        elif "AVIF" == file_format:
            if config['lossless_compression']:
                image.save(
                    output_path, lossless=True, quality=100, qmin=0, qmax=0, speed=0, subsampling="4:4:4", range="full"
                )
            else:
                image.save(output_path, quality=config['quality'], speed=0, range="full")

        print(colored(f"{output_path} Saved!", 'light_green'))

        # # Stop timer
        # end_time = time.time()
        # difference = end_time - start_time
        # if file_format not in time_of_file_extension:
        #     time_of_file_extension[file_format] = difference
        # else:
        #     time_of_file_extension[file_format] += difference

    # # save the time of each file extension to a file
    # with open("../output/time_of_file_extension.txt", "w+") as f:
    #     for key in time_of_file_extension:
    #         f.write(f"{key}: {time_of_file_extension[key]}\n")


def save_images_chunk(args) -> None:
    algorithm, images_chunk, roots_chunk, file_chunk, scales, config = args
    # print(f"Type of scales: {type(scales)}")
    # print(f"scales: {scales}")
    while images_chunk:
        image_object = images_chunk.pop()
        root = roots_chunk.pop()
        file = file_chunk.pop()

        for scaled_image, scale in zip(image_object['images'], scales):
            if len(scaled_image) == 1:
                save_image(algorithm, scaled_image[0], root, file, scale, config)
            else:
                # Compose an APNG image
                raise NotImplementedError("Animated (and stacked) output is not yet supported")


def scale_loop(
        algorithm:
        Algorithms,
        images: list[utils.Image],
        roots: list[str],
        files: list[str],
        scales: list[float],
        config: dict,
        masks: list[list[list[PIL.Image]]] | None = None,
        nearest_neighbour_for_masks: list[utils.Image] | None = None
) -> None:
    print("Starting scaling process")

    # TODO: Implement multiprocessing for this and bring back the config_plus!!!
    if algorithm in utils.cli_algorithms:
        config_plus = {
            'sharpness': config['sharpness'],
            'relative_input_path_of_images': [root + '/' + file for root, file in zip(roots, files)]
        }
    elif algorithm == Algorithms.NEDI:
        config_plus = {
            'NEDI_m': config['NEDI_m']
        }
    elif algorithm == Algorithms.Repetition:
        config_plus = {
            'offset_x': config['offset_x'],
            'offset_y': config['offset_y']
        }
    else:
        config_plus = None

    if config['try_to_fix_texture_tiling']:
        print("Texture tiling fix is enabled, starting preparation...")
        scale = 1 + 2 * config['tiling_fix_quality']
        offset = 1 - config['tiling_fix_quality']
        temp_config_plus = {
            'offset_x': offset,
            'offset_y': offset
        }
        images = scaler.scale_image_batch(Algorithms.Repetition, images, [scale], config_plus=temp_config_plus)
        print(colored("Preparation done", 'green'))

    image_objects = scaler.scale_image_batch(algorithm, images, scales, config_plus=config_plus)
    print(colored("Scaling done\n", 'green'))

    if config['try_to_fix_texture_tiling']:
        print("Texture tiling fix is enabled, cutting texture...")
        new_image_objects = []
        for image_obj in image_objects:
            scaled_images = []
            for scaled_image in image_obj['images']:
                new_image = []
                for frame in scaled_image:
                    width, height = frame.size
                    new_width = round(width / (1 + 2 * config['tiling_fix_quality']))
                    new_height = round(height / (1 + 2 * config['tiling_fix_quality']))

                    new_frame = PIL.Image.new(frame.mode, (new_width, new_height))
                    for x in range(new_width):
                        for y in range(new_height):
                            new_frame.putpixel(
                                (x, y),
                                frame.getpixel(
                                    (
                                        x + new_width * config['tiling_fix_quality'],
                                        y + new_height * config['tiling_fix_quality']
                                    )
                                )
                            )

                    new_image.append(new_frame)
                scaled_images.append(new_image)
            new_image_objects.append(utils.ImageDict(images=scaled_images))
        image_objects = new_image_objects

    if config['texture_outbound_protection']:
        print("Applying texture outbound protection...")

        new_image_objects = []
        # print(f"Image objects: {image_objects}")
        # print(f"Masks: {masks}")
        for image_obj, masks_for_scales in zip(image_objects, masks):
            # print("First loop layer executed")
            # print(f"Image object: {image_obj}")
            # print(f"Masks for scales: {masks_for_scales}")
            scaled_images = []
            for scaled_image, masks_for_frames in zip(image_obj.images, masks_for_scales):
                # print("Second loop layer executed")
                new_image = []
                for frame, mask in zip(scaled_image, masks_for_frames):
                    # print("Third loop layer executed")
                    new_frame = utils.apply_mask(frame, mask)
                    new_image.append(new_frame)
                scaled_images.append(new_image)
            new_image_objects.append(utils.ImageDict(images=scaled_images))
        image_objects = new_image_objects
        print(colored("Texture outbound protection done\n", 'green'))

    if config['disallow_partial_transparency']:
        print("Removing partial transparency...")
        new_image_objects = []
        for image_obj in image_objects:
            scaled_images = []
            for scaled_image in image_obj['images']:
                new_image = []
                for frame in scaled_image:
                    # new_frame = utils.disallow_partial_transparency(frame)
                    frame_array = utils.pil_to_cv2(frame)
                    dimensions = frame_array.shape
                    if frame_array.shape[2] != 4:
                        print("Frame has no alpha channel, skipping alpha correction")
                        new_image.append(frame)
                        continue

                    new_frame_array = frame_array.copy()
                    for x in range(dimensions[0]):
                        for y in range(dimensions[1]):
                            if new_frame_array[x][y][3] != 255 and new_frame_array[x][y][3] != 0:
                                new_frame_array[x][y][3] = 255

                    new_image.append(utils.cv2_to_pil(new_frame_array))
                scaled_images.append(new_image)
            new_image_objects.append(utils.ImageDict(images=scaled_images))
        image_objects = new_image_objects
        print(colored("Removing partial transparency done\n", 'green'))

    if config['texture_inbound_protection']:
        print("Applying texture inbound protection...")
        # Got trough every pixel
        # If pixel is in the mask:
        #   If the pixel has transparency and nearest neighbour does not:
        #       Remove the transparency (set alpha to 255)
        #   If the pixel is empty ar has alpha 0:
        #       Replace it with the nearest neighbour pixel
        new_image_objects = []
        for image_obj, masks_for_scales, nearest_neighbour_masks in zip(
                image_objects, masks, nearest_neighbour_for_masks
        ):
            scaled_images = []
            for scaled_image, masks_for_frames, nearest_neighbour_mask in zip(
                    image_obj['images'], masks_for_scales, nearest_neighbour_masks['images']
            ):
                new_image = []
                for frame, mask, nearest_neighbour in zip(scaled_image, masks_for_frames, nearest_neighbour_mask):
                    # pixels = list(frame.getdata())
                    frame_array = utils.pil_to_cv2(frame)
                    dimensions = frame_array.shape
                    # nearest_neighbour_pixels = list(nearest_neighbour.getdata())
                    nearest_neighbour_array = utils.pil_to_cv2(nearest_neighbour)
                    mask_py = list(mask)
                    new_frame_array = frame_array.copy()

                    if frame_array.shape[2] != 4:
                        print("Frame has no alpha channel, skipping texture inbound protection")
                        new_image.append(frame)
                        continue

                    for x in range(dimensions[0]):
                        for y in range(dimensions[1]):
                            if mask_py[x][y] == 255:
                                if frame_array.shape[2] == 4:
                                    if frame_array[x][y][3] == 0:
                                        # print(f"Filled empty pixel ({x+1, y+1})")
                                        new_frame_array[x][y] = nearest_neighbour_array[x][y]
                                        # new_frame_array[x][y][3] = 128
                                    elif frame_array[x][y][3] != 255:
                                        if nearest_neighbour_array.shape[2] == 4:
                                            new_frame_array[x][y][3] = nearest_neighbour_array[x][y][3]
                                            # if new_frame_array[x][y][3] != nearest_neighbour_array[x][y][3]:
                                            #     print(f"Fixed transparency at pixel ({x+1, y+1})")
                                            #     new_frame_array[x][y][3] = nearest_neighbour_array[x][y][3]
                                            # if nearest_neighbour_array[x][y][3] == 255:
                                            #     print(f"Removed transparency from pixel ({x+1, y+1})")
                                            #     new_frame_array[x][y][3] = 255
                    new_image.append(utils.cv2_to_pil(new_frame_array))
                scaled_images.append(new_image)
            new_image_objects.append(utils.ImageDict(images=scaled_images))
        image_objects = new_image_objects
        print(colored("Texture inbound protection done\n", 'green'))

    if len(images) < len(scales):
        # raise ValueError("Images queue is empty")
        print(colored("WARN: Image deck is shorten then expected! Error might have occurred!", 'yellow'))

    print("Starting saving process")
    processes = 0
    if 3 in config['multiprocessing_levels']:
        if config['override_processes_count']:
            processes = config['max_processes'][2]
        else:
            # # Calc average image size
            # images_frames = [image.images[0] for image in images]
            # # print(f"Images frames: {images_frames}")
            # size_sum = 0
            # for frames in images_frames:
            #     for frame in frames:
            #         size_sum += frame.size[0] * frame.size[1]
            #
            # performance_constant = 150_000_000
            # performance_processes = size_sum * utils.avg(scales) ** 2 * len(scales) / performance_constant
            # TODO: Create better algorithm, consider images sizes and frame count
            performance_processes = utils.geo_avg(scales) / 16
            if not config['additional_lossless_compression']:
                performance_processes /= 32

            # print(f"Performance processes: {performance_processes}")

            performance_processes = round(performance_processes)

            processes = min(config['max_processes'][2], max(round(min(len(images), performance_processes)), 1))

    if processes > 1:
        print(f"Using {processes} processes for saving")
        pool = multiprocessing.Pool(processes=processes)

        chunk_size = max(len(images) // processes, 1)  # Divide images equally among processes
        chunks = []

        print("Starting data preparation...")
        while image_objects:
            if len(images) < chunk_size:
                chunk_size = len(images)

            images_chunk = [image_objects.pop() for _ in range(chunk_size)]
            roots_chunk = [roots.pop() for _ in range(chunk_size)]
            files_chunk = [files.pop() for _ in range(chunk_size)]

            chunks.append((algorithm, images_chunk, roots_chunk, files_chunk, scales, config))

        print(colored("Data preparation done", 'light_green'))
        # print(f"Chunks: {chunks}")
        # Map the process_images_chunk function to the list of argument chunks using the pool of worker processes
        pool.map(save_images_chunk, chunks)

        # Close the pool
        pool.close()

        # Wait for all worker processes to finish
        pool.join()

    else:
        save_images_chunk((algorithm, image_objects, roots, files, scales, config))

    print(colored("Saving done\n", 'green'))


def scale_loop_chunk(args) -> None:
    algorithms_chunk, images, roots, files, scales, config = args
    for algorithm in algorithms_chunk:
        scale_loop(algorithm, images.copy(), roots.copy(), files.copy(), scales, config)


def algorithm_loop(
        algorithms: list[Algorithms],
        images: list[utils.ImageDict],
        roots: list[str], files: list[str],
        scales: list[float], config: dict
) -> None:

    masks_for_images = None
    nearest_neighbour_for_masks = None
    if config['texture_outbound_protection'] or config['texture_inbound_protection']:
        print("Texture protection is enabled, generating masks...")
        masks_for_images = []
        for image in images:
            masks_for_scales = []
            for scale in scales:
                masks_for_frames = []
                # for frame in image.images[0]:
                for frame in image['images'][0]:
                    mask = utils.generate_mask(frame, scale, config['texture_mask_mode'])
                    masks_for_frames.append(mask)
                masks_for_scales.append(masks_for_frames)
            masks_for_images.append(masks_for_scales)
        print(colored("Masks generated!", 'green'))

        if config['texture_inbound_protection']:
            print("Texture Inbound Protection is enabled, generating NN masks...")
            nearest_neighbour_for_masks = scaler.scale_image_batch(Algorithms.CV2_INTER_NEAREST, images, scales)
            print(colored("NN masks generated!", 'green'))
    # print(f"Masks for images:\n{masks_for_images}")

    processes = 0
    if 2 in config['multiprocessing_levels']:  # TODO: Complete this implementation
        if config['override_processes_count']:
            processes = config['max_processes'][1]
        else:
            # Calc average image size
            images_frames = [image.images[0] for image in images]
            # print(f"Images frames: {images_frames}")
            size_sum = 0
            for frames in images_frames:
                for frame in frames:
                    size_sum += frame.size[0] * frame.size[1]

            performance_constant = 786_432  # 2^18 * 3
            performance_processes = size_sum * avg(scales)**2 * len(scales) * len(algorithms) / performance_constant

            # performance_processes = utils.geo_avg(scales) / 8 * len(algorithms)
            # print(f"Performance processes: {performance_processes}")
            if not config['lossless_compression']:
                performance_processes /= 32

            performance_processes = round(performance_processes)

            processes = min(config['max_processes'][2], max(round(min(len(algorithms), performance_processes)), 1))
            # processes = max(min(config['max_processes'][1], len(algorithms)), 1)

    if processes > 1:
        print(f"Using {processes} processes for 'scales loop'")

        chunk_size = max(len(algorithms) // processes, 1)
        chunks = []

        while algorithms:
            if len(algorithms) < chunk_size:
                chunk_size = len(algorithms)

            algorithms_chunk = [algorithms.pop() for _ in range(chunk_size)]
            # images_chunk = [images for _ in range(chunk_size)]
            # roots_chunk = [roots for _ in range(chunk_size)]
            # files_chunk = [files for _ in range(chunk_size)]

            chunks.append(
                (algorithms_chunk, images, roots, files, scales, config, masks_for_images, nearest_neighbour_for_masks)
            )

        if 3 not in config['multiprocessing_levels']:
            pool = multiprocessing.Pool(processes=processes)
            pool.map(scale_loop_chunk, chunks)
            pool.close()
            pool.join()
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
                executor.map(scale_loop_chunk, chunks)

    else:
        for algorithm in algorithms:
            scale_loop(
                algorithm, images.copy(), roots.copy(), files.copy(), scales, config,
                masks_for_images, nearest_neighbour_for_masks
            )


def fix_config(config: dict) -> dict:
    # Fix 'multiprocessing_levels'
    if config['multiprocessing_levels'] is None:
        config['multiprocessing_levels'] = {}
        print("New multiprocessing_levels: {}")

    # Fix 'max_processes'
    try:
        max_processes = min(os.cpu_count(), round(psutil.virtual_memory() / 1024 ** 3))
        print(f"Max processes: {max_processes}")
    except Exception as e:
        print(colored(f"Error during max_processes calculation, using default 16384: {e}", 'red'))
        max_processes = 16384

    if config['max_processes'] is None:
        config['max_processes'] = (max_processes, max_processes, max_processes)
        print(f"New max_processes: {config['max_processes']}")

    else:
        if len(config['max_processes']) < 3:
            if len(config['max_processes']) == 0:
                config['max_processes'] = (max_processes, max_processes, max_processes)
            if len(config['max_processes']) == 1:
                config['max_processes'] = (config['max_processes'][0], max_processes, max_processes)
            elif len(config['max_processes']) == 2:
                config['max_processes'] = (config['max_processes'][0], config['max_processes'][1], max_processes)

        if config['max_processes'][0] is None:
            config['max_processes'] = (max_processes, config['max_processes'][1], config['max_processes'][2])
        if config['max_processes'][1] is None:
            config['max_processes'] = (config['max_processes'][0], max_processes, config['max_processes'][2])
        if config['max_processes'][2] is None:
            config['max_processes'] = (config['max_processes'][0], config['max_processes'][1], max_processes)

        print(f"New max_processes: {config['max_processes']}")

    # Fix 'clear_output_directory'
    if config['clear_output_directory'] is None:
        config['clear_output_directory'] = True
    elif type(config['clear_output_directory']) is not bool:
        config['clear_output_directory'] = True

    # Fix 'add_algorithm_name_to_output_files_names'
    if config['add_algorithm_name_to_output_files_names'] is None:
        config['add_algorithm_name_to_output_files_names'] = True
    elif type(config['add_algorithm_name_to_output_files_names']) is not bool:
        config['add_algorithm_name_to_output_files_names'] = True

    # Fix 'add_factor_to_output_files_names'
    if config['add_factor_to_output_files_names'] is None:
        config['add_factor_to_output_files_names'] = True
    elif type(config['add_factor_to_output_files_names']) is not bool:
        config['add_factor_to_output_files_names'] = True

    # Fix 'sort_by_algorithm'
    if config['sort_by_algorithm'] is None:
        config['sort_by_algorithm'] = False
    elif type(config['sort_by_algorithm']) is not bool:
        config['sort_by_algorithm'] = False

    # Fix 'lossless_compression'
    if config['lossless_compression'] is None:
        config['lossless_compression'] = True
    elif type(config['lossless_compression']) is not bool:
        config['lossless_compression'] = True

    # Fix 'mcmeta_correction'
    if config['mcmeta_correction'] is None:
        config['mcmeta_correction'] = True
    elif type(config['mcmeta_correction']) is not bool:
        config['mcmeta_correction'] = True

    return config


@lru_cache(maxsize=1)
def columnify(elements: tuple) -> str:
    result = ""
    max_columns = 4

    max_length = max([len(algorithm) for algorithm in elements])
    # print(f"Max length: {max_length}")
    margin_right = 2
    tab_spaces = 2

    # Get the size of the terminal
    if sys.stdout.isatty():
        terminal_columns = os.get_terminal_size().columns
        # print(f"Terminal columns: {terminal_columns}")
        columns = max_columns

        def calc_row_length() -> int:
            return len("\t".expandtabs(tab_spaces)) + (
                    max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2
            ) * columns - 1

        while terminal_columns < calc_row_length() and columns > 1:
            # print(f"Calculated row length: {calc_row_length()}")
            columns -= 1
    else:
        columns = 3
    # print(f"Final row length: {calc_row_length()}")
    # print(f"Final column count: {columns}")

    overflow = len(elements) % columns
    full_count = len(elements) - overflow

    for i in range(0, full_count, columns):
        result += "\t".expandtabs(tab_spaces)
        if i < len(elements):
            result += " | ".join(
                [f"\t{elements[i + j]:<{max_length + margin_right}}".expandtabs(tab_spaces) for j in range(columns)]
            )
        result += "\n"
    result += "\t".expandtabs(tab_spaces)
    result += " | ".join(
        [
            f"\t{elements[k]:<{max_length + margin_right}}"
            .expandtabs(tab_spaces) for k in range(full_count, overflow + full_count)
        ]
    )
    result += "\n"

    return result


def print_config(config: dict) -> None:
    print(colored("{\n\t".expandtabs(4), 'magenta'), end='')
    print(
        "\n\t".expandtabs(4).join(
            f"{colored(k, 'magenta')}: {colored(v, 'light_blue')}" for k, v in config.items()
        )
    )
    print(colored('}', 'magenta'))


def handle_user_input() -> tuple[list[Algorithms], list[float], float | None, int | None, dict[str, any]]:
    # multiprocessing_level:
    # empty - no multiprocessing
    # 2 - to process different algorithms in parallel, (Note that this level also splits every subsequent workflow)
    # 3 - to save multiple images in parallel
    default_config = {
        'clear_output_directory': True,

        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,

        'sort_by_algorithm': False,
        'sort_by_scale': False,
        'sort_by_image': False,
        'sort_by_file_extension': -1,  # -1 - auto, 0 - no, 1 - yes

        'file_formats': {"WEBP"},
        'lossless_compression': True,
        'additional_lossless_compression': True,
        'quality': 95,

        'multiprocessing_levels': {},
        'max_processes': (2, 2, 2),
        'override_processes_count': False,
        # If True, max_processes will set the Exact number of processes, instead of the Maximum number of them

        'copy_mcmeta': True,
        'texture_outbound_protection': False,
        # prevents multi-face (in 1 image) textures to expand over current textures border
        'texture_inbound_protection': False,
        # TODO: Implement this, prevents multi-face (in 1 image) textures to not fully cover current textures border
        'texture_mask_mode': ('alpha', 'black'),
        # What should be used to make the mask, 1st is when alpha is present, 2nd when it is not  TODO: add more options
        'disallow_partial_transparency': False,
        'try_to_fix_texture_tiling': False,
        'tiling_fix_quality': 1.0,

        'sharpness': 0.5,
        'NEDI_m': 4,
        'offset_x': 0.5,
        'offset_y': 0.5
    }

    algorithms = []
    scales = []

    # max_columns = 4

    available_algorithms = tuple(f"{algorithm.value} - {algorithm.name}" for algorithm in Algorithms)

    while True:
        print("Select algorithms you want to use separated by a space (id or name):")
        print("Available algorithms:")
        print(columnify(available_algorithms))

        algorithms_input = input().replace(',', '').split(' ')
        for algorithm_input in algorithms_input:
            if algorithm_input.isdigit():
                try:
                    # algorithms.add(Algorithms(int(algorithm_input)))
                    algorithms.append(Algorithms(int(algorithm_input)))
                except ValueError:
                    print(f"Algorithm with id '{algorithm_input}' does not exist")
            else:
                try:
                    # algorithms.add(utils.string_to_algorithm(algorithm_input))
                    algorithms.append(utils.string_to_algorithm(algorithm_input))
                except KeyError:
                    print(f"Algorithm with name '{algorithm_input}' does not exist")

        if len(algorithms) == 0:
            print("You must select at least one algorithm!")
        else:
            break
    # print(columnify.cache_info())

    while True:
        print("\nEnter the scales you want to use separated by a space:")
        scales_input = input().replace(',', ' ').split(' ')
        for scale_input in scales_input:
            try:
                # scales.add(float(scale_input))
                scales.append(float(scale_input))
            except ValueError:
                print(f"Scale '{scale_input}' is not a valid number")

        if len(scales) == 0:
            print("You must select at least one scale!")
        else:
            break

    sharpness = None
    nedi_m = None
    for algorithm in algorithms:
        if algorithm == Algorithms.CAS:
            while True:
                print("\nEnter the sharpness value for the CAS algorithm (0.0 - 1.0):")
                sharpness = input()
                try:
                    sharpness = float(sharpness)
                    if sharpness < 0 or sharpness > 1:
                        print("Sharpness must be in range 0.0 - 1.0")
                    else:
                        break
                except ValueError:
                    print("Sharpness must be a number")
        elif algorithm == Algorithms.NEDI:
            while True:
                print(
                    "\n"
                    "Enter the NEDI 'm' value (edge search radius) for the NEDI algorithm (must be a power of 2, >=1):"
                )
                nedi_m = input()
                try:
                    nedi_m = int(nedi_m)
                    if nedi_m < 1 or nedi_m & (nedi_m - 1) != 0:
                        print("NEDI 'm' must be a power of 2 and >= 1")
                    else:
                        break
                except ValueError:
                    print("NEDI 'm' must be a number")

    print("\nDo you wish to proceed with default configuration?")
    print("Default configuration:")
    print_config(default_config)
    # print(colored("{\n\t".expandtabs(4), 'magenta'), end='')
    # print(
    #     "\n\t".expandtabs(4).join(
    #         f"{colored(k, 'magenta')}: {colored(v, 'light_blue')}" for k, v in default_config.items()
    #     )
    # )
    # print(colored('}', 'magenta'))
    print("Y/N")
    default_config_input = input().lower()
    if (
            default_config_input == 'n' or
            default_config_input == 'no' or
            default_config_input == 'false' or
            default_config_input == '0'
    ):
        print("Enter the configuration:")
        config = {}
        for key, value in default_config.items():
            type_of_value = type(value)
            print(f"{key}:", end=' ')
            input_value = input()

            if type_of_value == bool:
                # config[key] = bool(input)
                config[key] = input_value.lower() == 'true' or input_value.lower() == 't' or input_value.lower() == '1'
            elif type_of_value == int:
                try:
                    config[key] = int(input_value)
                except ValueError:
                    print(f"Value for '{key}' must be an integer")
                    config[key] = value
            elif type_of_value == float:
                try:
                    config[key] = float(input_value)
                except ValueError:
                    print(f"Value for '{key}' must be a float")
                    config[key] = value
            elif type_of_value == tuple:
                config[key] = tuple(input_value.split(','))
            else:
                config[key] = input_value

            print()

        config = fix_config(config)
    else:
        config = default_config

    print("\nSelected options:")
    print("Algorithms:")
    for algorithm in algorithms:
        print(f"\t{algorithm.value} - {algorithm.name}")
    print("Scales:")
    for scale in scales:
        print(f"\t{scale}")
    if sharpness is not None:
        print(f"Sharpness: {sharpness}")
    if nedi_m is not None:
        print(f"NEDI 'm': {nedi_m}")

    print("Is this correct? Y/N")

    correct = input()
    if correct.lower() == 'n' or correct.lower() == 'no' or correct.lower() == 'false' or correct.lower() == '0':
        return handle_user_input()
    else:
        return algorithms, scales, sharpness, nedi_m, config


def run(algorithms: list[Algorithms], scales: list[float], config: dict) -> None:
    if os.path.exists("../output"):
        if config['clear_output_directory']:
            print("\nOutput directory is being cleared!")
            for root, dirs, files in os.walk("../output"):
                # print(f"Files: {files}")
                # print(f"Directories: {dirs}")
                for file in files:
                    os.remove(os.path.join(root, file))
                for directory in dirs:
                    shutil.rmtree(os.path.join(root, directory))
            print(colored("Output directory has been cleared!\n", 'green'))
    # Go through all files in input directory, scale them and save them in output directory
    # if in input folder there are some directories all path will be saved in output directory

    images = []
    roots = []
    file_names = []
    for root, dirs, files in os.walk("../input"):
        # print(f"Length of images: {len(images)}")
        # print(f"Length of roots: {len(roots)}")
        # print(f"Length of files: {len(file_names)}")

        files_number = len(files)
        print(f"Checking {files_number} files in input directory")

        backup_iterator = 0  # Save guard to prevent infinite loop, IDK why byt needed :/

        for file in files:
            if backup_iterator >= files_number:
                print(
                    colored(
                        "\n"
                        "Backup iterator reached the end of the files list, BREAKING LOOP!"
                        "\n"
                        "THIS SHOULD HAVE NOT HAPPENED!!!"
                        "\n", 'red'
                    )
                )
                break
            backup_iterator += 1

            path = os.path.join(root, file)
            extension = file.split('.')[-1].lower()

            if extension == "zip" or extension == "7z":
                with zipfile.ZipFile(path) as zip_ref:
                    # Get a list of all files and directories inside the zip file
                    zip_contents = zip_ref.namelist()

                    # Iterate through each file in the zip file
                    for file_name in zip_contents:
                        # Check if the current item is a file (not a directory)
                        if not file_name.endswith('/'):  # Directories end with '/'
                            # Read the content of the file
                            with zip_ref.open(file_name) as zip_file:
                                # Process the content of the file
                                print(f"Contents of {file_name}:")
                                print(zip_file.read())  # You can perform any operation you want with the file content
                                print("---------------")

                raise NotImplementedError("Zip and 7z files are not supported yet")

            elif extension in pil_fully_supported_formats_cache or extension in pil_read_only_formats_cache:
                image = PIL.Image.open(path)
                image = pngify(image)

                images.append(image)
                roots.append(root)
                file_names.append(file)
                # print(f"Appended root: {root} and file: {file} to their respective lists")

            elif extension == "mcmeta":
                if config['copy_mcmeta']:
                    print(f"MCMeta file: {path} is being copied to the output directory")
                    output_dir = f"../output{root.lstrip('../input')}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    shutil.copy2(path, output_dir)
                else:
                    print(f"MCMeta file: {path} will be ignored, animated texture will be corrupted!")

                continue

            elif extension in pil_write_only_formats_cache or extension in pil_indentify_only_formats_cache:
                print(f"File: {path} is an recognized image format but is not supported :( (yet)")
                try:
                    PIL.Image.open(path)  # Open the image to display Pillow's error message
                finally:
                    pass

                continue
            else:
                print(f"File: {path} is not supported, unrecognized file extension '{extension}'")
                continue

            print(f"Loading: {path}")

    # print(f"Length of images: {len(images)}")
    # print(f"Length of roots: {len(roots)}")
    # print(f"Length of files: {len(file_names)}")

    # After switching to list[list[image]] format, multiprocessing on this level became obsolete
    algorithm_loop(algorithms, images, roots, file_names, scales, config)


def rainbowify(text: str) -> str:
    colors: list[TermColor] = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    len_colors = len(colors)
    rainbow_text = ""
    i = 0
    for char in text:
        if char == ' ':
            rainbow_text += ' '
        else:
            rainbow_text += colored(char, colors[i % len_colors])
            i += 1
    return rainbow_text


if __name__ == '__main__':
    # Create input and output directory if they don't exist
    if not os.path.exists("../input"):
        os.makedirs("../input")

    parser = argparse.ArgumentParser(
        prog='ImageScaler',
        description='An advanced image scaler',
        epilog='Enjoy the program! :)'
    )

    # parser.add_argument('filename')  # positional argument
    # parser.add_argument('-c', '--count')  # option that takes a value
    parser.add_argument('-t', '--test', action='store_true')  # on/off flag
    parser.add_argument('-opv', '--override-python-version', action='store_true')  # on/off flag
    args = parser.parse_args()

    if sys.version_info[0] != 3 and sys.version_info[1] != 12:
        if not args.override_python_version:
            message = """
                Some functionality of this application is designed to run on Python 3.12!
                Please upgrade your Python version to 3.12 or higher!
                If you still want to force your current python version use flag
                '--override-python-version' or '-opv' to bypass this check
                (note that some functionality may not work properly and the application may crash!)
            """
            raise AssertionError(message)
        else:
            message = """
                WARNING: You are using an unsupported Python version!
                Some functionality of this application may not work properly and the application may crash!
            """
            print(colored(message, 'yellow'))

    if args.test:
        import presets

        # config = {
        #     'clear_output_directory': True,
        #
        #     'add_algorithm_name_to_output_files_names': True,
        #     'add_factor_to_output_files_names': True,
        #
        #     'sort_by_algorithm': False,
        #     'sort_by_scale': False,
        #     'sort_by_image': False,
        #     'sort_by_file_extension': -1,
        #
        #     'file_formats': {"WEBP"},
        #     'lossless_compression': True,
        #     'additional_lossless_compression': True,
        #     'quality': 95,
        #
        #     'multiprocessing_levels': {},
        #     'max_processes': (2, 2, 2),
        #     'override_processes_count': False,
        #
        #     'copy_mcmeta': False,
        #     'texture_outbound_protection': False,
        #     'texture_inbound_protection': False,
        #     'texture_mask_mode': ('alpha', 'black'),
        #     'disallow_partial_transparency': False,
        #     'try_to_fix_texture_tiling': False,
        #     'tiling_fix_quality': 1.0,
        #
        #     'sharpness': 0.5,
        #     'NEDI_m': 4
        # }
        # algorithms = [Algorithms.PIL_BICUBIC]
        # algorithms = [
        #     Algorithms.CV2_INTER_NEAREST, Algorithms.CV2_ESPCN, Algorithms.PIL_NEAREST_NEIGHBOR,
        #     Algorithms.RealESRGAN, Algorithms.xBRZ, Algorithms.FSR, Algorithms.Super_xBR, Algorithms.hqx,
        #     Algorithms.NEDI
        # ]
        # algorithms = [
        #     Algorithms.CV2_INTER_NEAREST, Algorithms.CV2_INTER_LINEAR,
        #     Algorithms.CV2_INTER_CUBIC, Algorithms.CV2_INTER_LANCZOS4,
        #     Algorithms.CV2_ESPCN, Algorithms.CV2_EDSR, Algorithms.CV2_FSRCNN,
        #     Algorithms.CV2_FSRCNN_small, Algorithms.RealESRGAN,
        #     Algorithms.xBRZ, Algorithms.FSR, Algorithms.CAS, Algorithms.Super_xBR,
        #     Algorithms.hqx, Algorithms.NEDI
        # ]
        # algorithms = [
        #     Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_NEAREST,
        #     Algorithms.CV2_INTER_CUBIC, Algorithms.CV2_INTER_LANCZOS4,
        #
        #     Algorithms.SI_drln, Algorithms.RealESRGAN,
        #     Algorithms.Anime4K, Algorithms.HSDBTRE,
        #
        #     Algorithms.NEDI, Algorithms.Super_xBR,
        #     Algorithms.xBRZ, Algorithms.FSR
        # ]
        config = presets.FullUpscaleTest.config
        algorithms = presets.FullUpscaleTest.algorithms
        scales = [4]
        # scales = [0.125, 0.25, 0.5, 0.666, 0.8]
        config['NEDI_m'] = 4
        # config['offset_x'] = 0
        # config['offset_y'] = 0
        config['sharpness'] = 0.5
    else:
        algorithms, scales, sharpness, nedi_m, config = handle_user_input()
        config['sharpness'] = sharpness
        config['NEDI_m'] = nedi_m
        config['offset_x'] = 0.5
        config['offset_y'] = 0.5
    print(f"Received algorithms: {colored(algorithms, 'blue')}")
    print(f"Received scales: {colored(scales, 'blue')}")
    print("Using config: ", end='')
    print_config(config)
    # print(colored("{\n\t".expandtabs(4), 'magenta'), end='')
    # print(
    #     "\n\t".expandtabs(4).join(
    #         f"{colored(k, 'magenta')}: {colored(v, 'light_blue')}" for k, v in config.items()
    #     )
    # )
    # print(colored('}', 'magenta'))

    run(algorithms, scales, config)
    rainbow_all_done = rainbowify("ALL NICE & DONE!")
    print(rainbow_all_done)
    thanks = rainbowify("Thanks for using this App!")
    print(thanks)
    # print(colored("ALL DONE!", 'green'))
