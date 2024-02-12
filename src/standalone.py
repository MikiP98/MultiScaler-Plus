# coding=utf-8
import argparse
import os
import scaler

from fractions import Fraction
from PIL import Image
from scaler import Algorithms

if __name__ == '__main__':
    # Create input and output directory if they don't exist
    if not os.path.exists("../input"):
        os.makedirs("../input")

    clear_output_directory = True
    add_algorithm_name_to_output_files_names = True
    add_factor_to_output_files_names = True
    sort_by_algorithm = False
    lossless_compression = True



    # algorithms = {Algorithms.xBRZ, Algorithms.RealESRGAN, Algorithms.NEAREST_NEIGHBOR, Algorithms.BILINEAR, Algorithms.BICUBIC, Algorithms.LANCZOS}
    algorithms = {Algorithms.xBRZ}
    # scales = {2, 4, 8, 16, 32, 64, 1.5, 3, 6, 12, 24, 48, 1.25, 2.5, 5, 10, 20, 40, 1.75, 3.5, 7, 14, 28, 56, 1.125, 2.25, 4.5, 9, 18, 36, 72, 256}
    scales = {4}

    if os.path.exists("../output"):
        if clear_output_directory:
            for root, dirs, files in os.walk("../output"):
                for file in files:
                    os.remove(os.path.join(root, file))

    # Go through all files in input directory, scale them and save them in output directory
    # if in input folder there are some directories all path will be saved in output directory
    for root, dirs, files in os.walk("../input"):
        for file in files:
            path = os.path.join(root, file)
            for algorithm in algorithms:
                for scale in scales:
                    image = scaler.scale_image(algorithm, Image.open(path), scale)

                    new_file_name = file
                    if add_algorithm_name_to_output_files_names:
                        new_file_name = f"{algorithm.name}_{new_file_name}"
                    if add_factor_to_output_files_names:
                        if scale != int(scale):
                            if len(str(scale).split(".")[1]) > 3:
                                scale = f"{str(Fraction(scale).limit_denominator()).replace('/', '%')}"
                        new_file_name = f"{new_file_name[:-4]}_{scale}x{new_file_name[-4:]}"
                    # print(new_file_name)

                    output_dir = "../output"
                    if sort_by_algorithm:
                        output_dir += f"/{algorithm.name}"

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    output_path = output_dir + '/' + new_file_name
                    print(output_path)

                    if not lossless_compression:
                        image.save(output_path)
                    else:
                        if image.mode == 'RGBA':
                            # Go through every pixel and check if alpha is 255, if it 255 on every pixel, save it as RGB
                            # else save it as RGBA
                            alpha_was_used = False
                            for pixel in image.getdata():
                                if pixel[3] != 255:
                                    alpha_was_used = True
                                    break
                            if not alpha_was_used:
                                image = image.convert('RGB')

                        image.save(output_path, optimize=True)

                        # Go through every pixel and check add the color to the set,
                        # if the set doesn't have more 256 colors convert the image to palette
                        set_of_colors = set()
                        for pixel in image.getdata():
                            set_of_colors.add(pixel)
                            if len(set_of_colors) > 256:
                                break

                        colors_len = len(set_of_colors)
                        if colors_len <= 256:
                            colors = 256
                            if colors_len <= 2:
                                colors = 2
                            elif colors_len <= 4:
                                colors = 4
                            elif colors_len <= 16:
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
