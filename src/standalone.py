# coding=utf-8
# import argparse
import os
import scaler
import shutil

from multiprocessing import Process
from fractions import Fraction
from PIL import Image
from utils import Algorithms


def process_image(algorithm: Algorithms, image, root: str, file: str, scale, config):
    path = os.path.join(root, file)

    # image = scaler.scale_image(algorithm, Image.open(path), scale)
    image = scaler.scale_image(algorithm, image, scale)
    if image is None:
        print(f"Saving image: {path}, is probably handled by another thread")
        return

    new_file_name = file
    if config['add_algorithm_name_to_output_files_names']:
        new_file_name = f"{algorithm.name}_{new_file_name}"
    if config['add_factor_to_output_files_names']:
        if scale != int(scale):
            if len(str(scale).split(".")[1]) > 3:
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

        if image.mode == 'RGBA':
            # Go through every pixel and check if alpha is 255, if it 255 on every pixel, save it as RGB
            # else save it as RGBA
            alpha_was_used = any(pixel[3] != 255 for pixel in image.getdata())
            if not alpha_was_used:
                image = image.convert('RGB')

        image.save(output_path, optimize=True)

        # Go through every pixel and check add the color to the set,
        # if the set doesn't have more 256 colors convert the image to palette
        # set_of_colors = set()
        # for pixel in image.getdata():
        #     set_of_colors.add(pixel)
        #     if len(set_of_colors) > 256:
        #         break
        #
        # colors_len = len(set_of_colors)
        # if colors_len <= 256:
        #     colors = 256
        #     if colors_len <= 2:
        #         colors = 2
        #     elif colors_len <= 4:
        #         colors = 4
        #     elif colors_len <= 16:
        #         colors = 16

        unique_colors_number = len(set(image.getdata()))
        if unique_colors_number <= 256:
            colors = 256
            if unique_colors_number <= 2:
                colors = 2
            elif unique_colors_number <= 4:
                colors = 4
            elif unique_colors_number <= 16:
                colors = 16

            image = image.convert('P', palette=Image.ADAPTIVE, colors=colors)
            temp_name = output_path[:-4] + "_P.png"
            image.save(temp_name, optimize=True)

            # Check which one is smaller and keep it, remove the other one
            # (if the palette is smaller remove '_P' from the name)
            if os.path.getsize(output_path) < os.path.getsize(temp_name):
                os.remove(temp_name)
            else:
                os.remove(output_path)
                # Rename the smaller one to the original name
                os.rename(temp_name, output_path)


def scale_loop(algorithm: Algorithms, image: Image, root: str, file: str, scales: set[float], config):
    processes = []
    for scale in scales:
        if 3 in config['multiprocessing_level']:
            p = Process(target=process_image, args=(algorithm, image, root, file, scale, config))
            p.start()
            processes.append(p)
        else:
            process_image(algorithm, image, root, file, scale, config)

    for process in processes:
        process.join()


def algorithm_loop(algorithms: set[Algorithms], image: Image, root: str, file: str, scales: set[float], config):
    processes = []
    for algorithm in algorithms:
        if 2 in config['multiprocessing_level']:
            p = Process(target=scale_loop, args=(algorithm, image, root, file, scales, config))
            p.start()
            processes.append(p)
        else:
            scale_loop(algorithm, image, root, file, scales, config)

    for process in processes:
        process.join()


if __name__ == '__main__':
    # Create input and output directory if they don't exist
    if not os.path.exists("../input"):
        os.makedirs("../input")

    # true_multithreading = False

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
        'multiprocessing_level': {1}
    }

    # algorithms = {Algorithms.xBRZ, Algorithms.RealESRGAN, Algorithms.NEAREST_NEIGHBOR, Algorithms.BILINEAR, Algorithms.BICUBIC, Algorithms.LANCZOS}
    # algorithms = {Algorithms.xBRZ, Algorithms.NEAREST_NEIGHBOR, Algorithms.BILINEAR, Algorithms.BICUBIC, Algorithms.LANCZOS}
    # algorithms = {Algorithms.NEAREST_NEIGHBOR}
    algorithms = {Algorithms.xBRZ}
    # algorithms = {Algorithms.CPP_DEBUG}
    # algorithms = {Algorithms.RealESRGAN}
    # algorithms = {Algorithms.SUPIR}
    # scales = {2, 4, 8, 16, 32, 64, 1.5, 3, 6, 12, 24, 48, 1.25, 2.5, 5, 10, 20, 40, 1.75, 3.5, 7, 14, 28, 56, 1.125, 2.25, 4.5, 9, 18, 36, 72, 256}
    scales = {4}

    if os.path.exists("../output"):
        if config['clear_output_directory']:
            for root, dirs, files in os.walk("../output"):
                for file in files:
                    os.remove(os.path.join(root, file))
                for directory in dirs:
                    shutil.rmtree(os.path.join(root, directory))

    # Go through all files in input directory, scale them and save them in output directory
    # if in input folder there are some directories all path will be saved in output directory
    processes = []
    for root, dirs, files in os.walk("../input"):
        for file in files:
            path = os.path.join(root, file)

            if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
                print(f"Processing: {path}")
                image = Image.open(path)

                if 1 in config['multiprocessing_level']:
                    p = Process(target=algorithm_loop, args=(algorithms, image, root, file, scales, config))
                    p.start()
                    processes.append(p)
                else:
                    algorithm_loop(algorithms, image, root, file, scales, config)

    for process in processes:
        process.join()
