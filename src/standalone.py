# coding=utf-8
import argparse
import concurrent.futures
# import math
import multiprocessing
import os
import PIL.Image
import PIL.GifImagePlugin
import psutil
import scaler
import sys
import shutil
# import termcolor
import utils
import zipfile

from fractions import Fraction
from functools import lru_cache
from termcolor import colored
from utils import Algorithms, pil_fully_supported_formats_cache, pil_read_only_formats_cache, pil_write_only_formats_cache, pil_indentify_only_formats_cache


PIL.Image.MAX_IMAGE_PIXELS = 200000000
PIL.GifImagePlugin.LOADING_STRATEGY = PIL.GifImagePlugin.LoadingStrategy.RGB_ALWAYS


def save_image(algorithm: Algorithms, image: PIL.Image, root: str, file: str, scale, config) -> None:
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

    output_dir = "../output"
    if config['sort_by_algorithm']:
        output_dir += f"/{algorithm.name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + root.lstrip("../input") + '/' + new_file_name
    print(output_path)
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir + root.lstrip("../input")):
        os.makedirs(output_dir + root.lstrip("../input"))

    output_path = output_path.replace(".jpg", ".png").replace(".jpeg", ".png")
    if not config['lossless_compression']:
        image.save(output_path)
    else:
        img_byte_arr = utils.apply_lossless_compression(image)
        with open(output_path, 'wb') as f:
            f.write(img_byte_arr)

    print(colored(f"{output_path} Saved!", 'light_green'))


def save_images_chunk(args) -> None:
    algorithm, images_chunk, roots_chunk, file_chunk, scales, config = args
    # print(f"Type of scales: {type(scales)}")
    # print(f"scales: {scales}")
    while images_chunk:
        image_object = images_chunk.pop()
        root = roots_chunk.pop()
        file = file_chunk.pop()

        for scaled_image, scale in zip(image_object.images, scales):
            if len(scaled_image) == 1:
                save_image(algorithm, scaled_image[0], root, file, scale, config)
            else:
                # Compose an APNG image
                raise NotImplementedError("Animated (and stacked) output is not yet supported")

    # for image_object, root, file in zip(images_chunk, roots_chunk, file_chunk):
    #     # print(f"Type of image_object: {type(image_object)}")
    #     # print(f"Type of image_object.images: {type(image_object.images)}")
    #     # print(f"Type of root: {type(root)}")
    #     # print(f"Type of file: {type(file)}")
    #     print(f"Type of scales: {type(scales)}")
    #
    #     for scaled_image, scale in zip(image_object.images, scales):
    #         if len(scaled_image) == 1:
    #             save_image(algorithm, scaled_image[0], root, file, scale, config)
    #         else:
    #             # Compose an APNG image
    #             raise NotImplementedError("Animated (and stacked) output is not yet supported")

    # for scaled_image, scale in zip(image_object.images, scales):
    #     if len(scaled_image) == 1:
    #         save_image(algorithm, scaled_image[0], root, file, scale, config)
    #     else:
    #         # Compose an APNG image
    #         raise NotImplementedError("Animated (and stacked) output is not yet supported")


def scale_loop(algorithm: Algorithms, images: list[utils.Image], roots: list[str], files: list[str], scales: list[float], config) -> None:
    # config_plus = {
    #     'input_image_relative_path': file,
    #     'sharpness': 0.5
    # }

    print("Starting scaling process")
    # TODO: Implement multiprocessing for this and bring back the config_plus!!!
    # print(f"Scaling image: {config_plus['input_image_relative_path']}")
    # print(f"Algorithm in scale_loop: {utils.algorithm_to_string(algorithm)}, {algorithm}")
    image_objects = scaler.scale_image_batch(algorithm, images, scales)  # , config_plus=config_plus
    print(colored("Scaling done\n", 'green'))

    if len(images) < len(scales):
        # raise ValueError("Images queue is empty")
        print(colored("WARN: Image deck is shorten then expected! Error might have occurred!", 'yellow'))

    print("Starting saving process")
    processes = 0
    if 3 in config['multiprocessing_levels']:
        if config['override_processes_count']:
            processes = config['max_processes'][2]
        else:
            # TODO: Create better algorithm, consider images sizes and frame count
            performance_processes = utils.geo_avg(scales) / 16
            if not config['lossless_compression']:
                performance_processes /= 16

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
        while image_objects:
            image_object = image_objects.pop()
            root = roots.pop()
            file = files.pop()

            for scaled_image, scale in zip(image_object.images, scales):
                if len(scaled_image) == 1:
                    save_image(algorithm, scaled_image[0], root, file, scale, config)
                else:
                    # Compose an APNG image
                    raise NotImplementedError("Animated (and stacked) output is not yet supported")
            # iterator += 1

    print(colored("Saving done\n", 'green'))


def scale_loop_chunk(args) -> None:
    algorithms_chunk, images, roots, files, scales, config = args
    for algorithm in algorithms_chunk:
        scale_loop(algorithm, images.copy(), roots.copy(), files.copy(), scales, config)


def algorithm_loop(algorithms: list[Algorithms],
                   images: list[utils.Image],
                   roots: list[str], files: list[str],
                   scales: list[float], config) -> None:
    processes = 0
    if 2 in config['multiprocessing_levels']:  # TODO: Complete this implementation
        if config['override_processes_count']:
            processes = config['max_processes'][1]
        else:
            # TODO: Create better algorithm
            performance_processes = utils.geo_avg(scales) / 8 * len(algorithms)
            if not config['lossless_compression']:
                performance_processes /= 4

            performance_processes = round(performance_processes)

            processes = min(config['max_processes'][2], max(round(min(len(algorithms), performance_processes)), 1))
            # processes = max(min(config['max_processes'][1], len(algorithms)), 1)

    if processes > 1:
        print(f"Using {processes} processes for 'scales loop'")

        chunk_size = max(len(images) // processes, 1)
        chunks = []

        while algorithms:
            if len(algorithms) < chunk_size:
                chunk_size = len(algorithms)

            algorithms_chunk = [algorithms.pop() for _ in range(chunk_size)]
            # images_chunk = [images for _ in range(chunk_size)]
            # roots_chunk = [roots for _ in range(chunk_size)]
            # files_chunk = [files for _ in range(chunk_size)]

            chunks.append((algorithms_chunk, images, roots, files, scales, config))
            # chunks.append((algorithms_chunk, images_chunk, roots_chunk, files_chunk, scales, config))

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
            scale_loop(algorithm, images.copy(), roots.copy(), files.copy(), scales, config)


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

    # config = {
    #     'clear_output_directory': True,
    #     'add_algorithm_name_to_output_files_names': True,
    #     'add_factor_to_output_files_names': True,
    #     'sort_by_algorithm': False,
    #     'lossless_compression': True,
    #     'multiprocessing_levels': {2},
    #     'max_processes': (2, 4, 4),
    #     'mcmeta_correction': True
    # }

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
        while terminal_columns < len("\t".expandtabs(tab_spaces)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1 and columns > 1:
            # print(f"Calculated row length: {len("\t".expandtabs(tab_spaces * 2)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1}")
            columns -= 1
    else:
        columns = 3
    # print(f"Final row length: {len("\t".expandtabs(tab_spaces * 2)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1}")
    # print(f"Final column count: {columns}")

    overflow = len(elements) % columns
    full_count = len(elements) - overflow

    for i in range(0, full_count, columns):
        result += "\t".expandtabs(tab_spaces)
        if i < len(elements):
            result += " | ".join([f"\t{elements[i + j]:<{max_length + margin_right}}".expandtabs(tab_spaces) for j in range(columns)])
        result += "\n"
    result += "\t".expandtabs(tab_spaces)
    result += " | ".join([f"\t{elements[k]:<{max_length + margin_right}}".expandtabs(tab_spaces) for k in range(full_count, overflow + full_count)])
    result += "\n"

    return result


def handle_user_input() -> tuple[set[Algorithms], set[float]]:
    algorithms = set()
    scales = set()

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
                    algorithms.add(Algorithms(int(algorithm_input)))
                except ValueError:
                    print(f"Algorithm with id '{algorithm_input}' does not exist")
            else:
                try:
                    algorithms.add(utils.string_to_algorithm(algorithm_input))
                except KeyError:
                    print(f"Algorithm with name '{algorithm_input}' does not exist")

        if len(algorithms) == 0:
            print("You must select at least one algorithm!")
        else:
            break
    print(columnify.cache_info())

    while True:
        print("Enter the scales you want to use separated by a space:")
        scales_input = input().replace(',', ' ').split(' ')
        for scale_input in scales_input:
            try:
                scales.add(float(scale_input))
            except ValueError:
                print(f"Scale '{scale_input}' is not a valid number")

        if len(scales) == 0:
            print("You must select at least one scale!")
        else:
            break

    print("Algorithms:")
    for algorithm in algorithms:
        print(f"\t{algorithm.value} - {algorithm.name}")
    print("Scales:")
    for scale in scales:
        print(f"\t{scale}")

    return algorithms, scales


pil_animated_formats = {
    "BLP": {"blp2"},  # Only BLP2 supports multiple images and animations
    "TIFF": {"tif", "tiff", "tiff2"},
    "APNG": {"apng"},
    "WebP": {"webp"},
    "JPX": {"jpx"}  # Only JPEG 2000 Part 2 (JPX) supports multiple images and animations
}
# AV1
# MNG: {.mng} MNG supports both multiple images and animations
pil_animated_formats_cache = {
    extension for extensions in pil_animated_formats for extension in extensions
}


def pngify(image: PIL.Image) -> utils.Image:
    if image.format.lower() in pil_animated_formats_cache:
        # Extract all frames from the animated image as a list of images
        if image.is_animated:
            raise NotImplementedError("Animated images are not supported yet")

        raise NotImplementedError(f"Animatable and stackable images are not supported yet: {pil_animated_formats_cache}")

    # check if is RGBA or RGB
    elif not (image.mode == "RGB" or image.mode == "RGBA"):
        image = image.convert("RGBA")
        if not utils.uses_transparency(image):
            image = image.convert("RGB")

    return utils.Image([[image]])
    # return [image]  # Return an 'image' with single 'frame'


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
    files = []
    for root, dirs, files in os.walk("../input"):
        files_number = len(files)
        print(f"Checking {files_number} files in input directory")

        backup_iterator = 0  # Save guard to prevent infinite loop, IDK why byt needed :/

        for file in files:
            if backup_iterator >= files_number:
                print(colored("\nBackup iterator reached the end of the files list, BREAKING LOOP!\nTHIS SHOULD HAVE NOT HAPPENED!!!\n", 'red'))
                break
            backup_iterator += 1

            path = os.path.join(root, file)
            extension = file.split('.')[-1].lower()

            if extension == "ZIP" or extension == "7Z":
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    # Get a list of all files and directories inside the zip file
                    zip_contents = zip_ref.namelist()

                    # Iterate through each file in the zip file
                    for file_name in zip_contents:
                        # Check if the current item is a file (not a directory)
                        if not file_name.endswith('/'):  # Directories end with '/'
                            # Read the content of the file
                            with zip_ref.open(file_name) as file:
                                # Process the content of the file
                                print(f"Contents of {file_name}:")
                                print(file.read())  # You can perform any operation you want with the file content
                                print("---------------")

                raise NotImplementedError("Zip and 7z files are not supported yet")

            elif extension in pil_fully_supported_formats_cache or extension in pil_read_only_formats_cache:
                image = PIL.Image.open(path)
                image = pngify(image)
                images.append(image)
                roots.append(root)
                files.append(file)

            elif extension == "mcmeta":
                if config['mcmeta_correction']:
                    print(f"MCMeta files are not supported yet :(\nfile: {path} will be ignored, animated texture will be corrupted!")
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

    # After switching to list[list[image]] format, multiprocessing on this level became obsolete
    algorithm_loop(algorithms, images, roots, files, scales, config)


if __name__ == '__main__':
    # Create input and output directory if they don't exist
    if not os.path.exists("../input"):
        os.makedirs("../input")

    safe_mode = False

    # multiprocessing_level:
    # empty - no multiprocessing,
    # 2 - to process different algorithms in parallel, (Note that this level also splits every subsequent workflow)
    # 3 - to save multiple images in parallel
    config = {
        'clear_output_directory': True,
        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,
        'sort_by_algorithm': False,
        'lossless_compression': True,
        'multiprocessing_levels': {},
        'max_processes': (2, 2, 2),
        'override_processes_count': False,  # If True, max_processes will set the Exact number of processes, instead of the Maximum number of them
        'mcmeta_correction': True
    }
    if safe_mode:
        config = fix_config(config)

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
                If you still want to your current python version use flag '--override-python-version' or '-opv' to bypass this check
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
        # algorithms = [Algorithms.CV2_INTER_NEAREST, Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_CUBIC, Algorithms.CV2_INTER_LANCZOS4]
        # algorithms = [Algorithms.CV2_INTER_AREA]
        algorithms = [Algorithms.CV2_INTER_NEAREST, Algorithms.CV2_ESPCN, Algorithms.PIL_NEAREST_NEIGHBOR, Algorithms.RealESRGAN, Algorithms.xBRZ]  # , Algorithms.FSR
        scales = [2, 4, 8]
        # scales = [0.125, 0.25, 0.5, 0.666, 0.8]
    else:
        algorithms, scales = handle_user_input()
    print(f"Received algorithms: {colored(algorithms, 'blue')}")
    print(f"Received scales: {colored(scales, 'blue')}")
    print("Using config: ", end='')
    print(colored("{\n\t".expandtabs(4), 'magenta'), end='')
    print("\n\t".expandtabs(4).join(f"{colored(k, 'magenta')}: {colored(v, 'light_blue')}" for k, v in config.items()))
    print(colored('}', 'magenta'))

    run(algorithms, scales, config)
    print(colored("ALL DONE!", 'green'))  # TODO: Rainbow-ify this message :)
