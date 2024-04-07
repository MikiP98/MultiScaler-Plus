# coding=utf-8
import argparse
from collections import deque
# import math
import multiprocessing
import os
import PIL.Image
import queue
import scaler
import sys
import shutil
import utils

from fractions import Fraction
from functools import lru_cache
from multiprocessing import Process
from utils import Algorithms


PIL.Image.MAX_IMAGE_PIXELS = 200000000


def save_image(algorithm: Algorithms, image, root: str, file: str, scale, config):
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

    if not config['lossless_compression']:
        image.save(output_path)
    else:
        output_path = output_path.replace(".jpg", ".png").replace(".jpeg", ".png")
        img_byte_arr = utils.apply_lossless_compression(image)
        with open(output_path, 'wb') as f:
            f.write(img_byte_arr)

    print(f"{output_path} Saved!")


def process_image(algorithm: Algorithms, image: PIL.Image, root: str, file: str, scale, config, config_plus=None):
    image = scaler.scale_image(algorithm, image, [scale], config_plus=config_plus)
    save_image(algorithm, image, root, file, scale, config)


def save_images_chunk(args):
    algorithm, images_chunk, root, file, scales_chunk, config = args
    for image, scale in zip(images_chunk, scales_chunk):
        save_image(algorithm, image, root, file, scale, config)


def scale_loop(algorithm: Algorithms, image, root: str, file: str, scales: set[float], config):
    config_plus = {
        'input_image_relative_path': file,
        'sharpness': 0.5
    }

    # if 4 in config['multiprocessing_levels']:  # This path is ONLY recommended for HUGE image sizes
    #     processes = queue.Queue()
    #     processes_count = 0
    #
    #     for scale in scales:
    #         p = Process(target=process_image, args=(algorithm, image, root, file, scale, config, config_plus))
    #         p.start()
    #         # processes.append(p)
    #         processes.put(p)
    #         processes_count += 1
    #         if processes_count >= config['max_processes'][2]:
    #             for _ in range(processes_count):
    #                 processes.get().join()
    #             processes_count = 0
    #
    #     for _ in range(processes_count):
    #         processes.get().join()
    # else:

    # print(f"Scaling image: {config_plus['input_image_relative_path']}")
    # print(f"Algorithm in scale_loop: {utils.algorithm_to_string(algorithm)}, {algorithm}")
    images = scaler.scale_image_batch(algorithm, image, scales, config_plus=config_plus)
    # print(f"Images: {images.qsize()}")
    # if images.qsize() == 0:
    if len(images) == 0:
        # print("Image:")
        # print(image)
        # print(f"Scales: {scales}")
        # raise ValueError("Images queue is empty")
        pass

    if 3 in config['multiprocessing_levels']:
        # print(f"Max processes: {config['max_processes'][2]}; len(Scales) // 2: {len(scales) // 2}: {len(scales)}")
        processes = min(config['max_processes'][2], max(round(len(scales) / 2), 1))
        pool = multiprocessing.Pool(processes=processes)

        chunk_size = len(scales) // processes  # Divide images equally among processes
        args_list = []
        # while not images.empty():
        while images:
            # images_chunk = [images.get() for _ in range(chunk_size)]
            images_chunk = [images.pop() for _ in range(chunk_size)]
            scales_chunk = [scales.pop() for _ in range(chunk_size)]
            args_list.append((algorithm, images_chunk, root, file, scales_chunk, config))

        # Map the process_images_chunk function to the list of argument chunks using the pool of worker processes
        pool.map(save_images_chunk, args_list)

        # Close the pool
        pool.close()
        # Wait for all worker processes to finish
        pool.join()

    else:
        # while not images.empty():
        while images:
            # image = images.get()
            image = images.pop()
            save_image(algorithm, image, root, file, scales.pop(), config)


def algorithm_loop(algorithms: set[Algorithms], image, root: str, file: str, scales: set[float], config):
    # processes = []
    processes = deque()
    processes_count = 0
    for algorithm in algorithms:
        if 2 in config['multiprocessing_levels']:
            p = Process(target=scale_loop, args=(algorithm, image, root, file, scales, config))
            p.start()
            processes.append(p)
            processes_count += 1
            if processes_count >= config['max_processes'][1]:
                for _ in range(processes_count):
                    processes.popleft().join()
                processes_count = 0
        else:
            scale_loop(algorithm, image, root, file, scales.copy(), config)

    for _ in range(processes_count):
        processes.popleft().join()
    # for process in processes:
    #     process.join()


def fix_config(config) -> dict:
    if config['multiprocessing_levels'] is None:
        config['multiprocessing_levels'] = {}
        print("New multiprocessing_levels: {}")

    if config['max_processes'] is None:
        config['max_processes'] = (16384, 16384, 16384)

        print(f"New max_processes: {config['max_processes']}")

    else:
        if len(config['max_processes']) < 3:
            if len(config['max_processes']) == 0:
                config['max_processes'] = (16384, 16384, 16384)
            if len(config['max_processes']) == 1:
                config['max_processes'] = (config['max_processes'][0], 16384, 16384)
            elif len(config['max_processes']) == 2:
                config['max_processes'] = (config['max_processes'][0], config['max_processes'][1], 16384)

        if config['max_processes'][0] is None:
            config['max_processes'] = (16384, config['max_processes'][1], config['max_processes'][2])
        if config['max_processes'][1] is None:
            config['max_processes'] = (config['max_processes'][0], 16384, config['max_processes'][2])
        if config['max_processes'][2] is None:
            config['max_processes'] = (config['max_processes'][0], config['max_processes'][1], 16384)

        print(f"New max_processes: {config['max_processes']}")

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


if __name__ == '__main__':
    # Create input and output directory if they don't exist
    if not os.path.exists("../input"):
        os.makedirs("../input")

    safe_mode = False

    # multiprocessing_level:
    # empty - auto select the best level of multiprocessing for the current prompt, TODO: implement
    # 0 - no multiprocessing,
    # 1 - process per image,
    # 2 - process per algorithm,
    # 3 - process per scale
    config = {
        'clear_output_directory': True,
        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,
        'sort_by_algorithm': False,
        'lossless_compression': True,
        'multiprocessing_levels': {},
        'max_processes': (4, 4, 2)
    }
    if safe_mode:
        config = fix_config(config)

    parser = argparse.ArgumentParser(
        prog='ImageScaler',
        description='A simple image scaler',
        epilog='Enjoy the program! :)'
    )

    # parser.add_argument('filename')  # positional argument
    # parser.add_argument('-c', '--count')  # option that takes a value
    parser.add_argument('-t', '--test', action='store_true')  # on/off flag
    args = parser.parse_args()

    if args.test:
        # algorithms = {Algorithms.CV2_INTER_AREA, Algorithms.CV2_INTER_CUBIC, Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_NEAREST, Algorithms.CV2_INTER_LANCZOS4}
        # algorithms = {Algorithms.CV2_EDSR, Algorithms.CV2_ESPCN, Algorithms.CV2_FSRCNN, Algorithms.CV2_LapSRN}
        # algorithms = {Algorithms.CV2_INTER_LANCZOS4, Algorithms.CV2_INTER_NEAREST}
        # algorithms = {Algorithms.CV2_INTER_NEAREST}
        # algorithms = {Algorithms.xBRZ}
        # algorithms = {Algorithms.CPP_DEBUG}
        algorithms = {Algorithms.RealESRGAN}
        # algorithms = {Algorithms.SUPIR}
        # scales = {2, 4, 8, 16, 32, 64, 1.5, 3, 6, 12, 24, 48, 1.25, 2.5, 5, 10, 20, 40, 1.75, 3.5, 7, 14, 28, 56, 1.125, 2.25, 4.5, 9, 18, 36, 72, 256}
        # scales = {0.128, 0.333, 1, 2, 3, 4, 8}  # , 9, 16, 256
        scales = {4}
    else:
        algorithms, scales = handle_user_input()
    print(f"Received algorithms: {algorithms}")
    print(f"Received scales: {scales}")

    if os.path.exists("../output"):
        if config['clear_output_directory']:
            for root, dirs, files in os.walk("../output"):
                for file in files:
                    os.remove(os.path.join(root, file))
                for directory in dirs:
                    shutil.rmtree(os.path.join(root, directory))

    # Go through all files in input directory, scale them and save them in output directory
    # if in input folder there are some directories all path will be saved in output directory
    # processes = []
    processes = queue.Queue()
    processes_count = 0
    for root, dirs, files in os.walk("../input"):
        for file in files:
            path = os.path.join(root, file)

            if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
                print(f"Processing: {path}")
                image = PIL.Image.open(path)
                # image = utils.pil_to_cv2(image)
                # print(type(image))

                if 1 in config['multiprocessing_levels']:
                    p = Process(target=algorithm_loop, args=(algorithms, image, root, file, scales, config))
                    p.start()
                    # processes.append(p)
                    processes.put(p)
                    processes_count += 1
                    if processes_count >= config['max_processes'][0]:
                        for i in range(processes_count):
                            processes.get().join()
                        processes_count = 0
                        # for process in processes:
                        #     process.join()
                        # processes.clear()
                else:
                    algorithm_loop(algorithms, image, root, file, scales, config)

    for i in range(processes_count):
        processes.get().join()
    # for process in processes:
    #     process.join()
