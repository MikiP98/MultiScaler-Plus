# coding=utf-8
# import numpy as np
# import rarch
import subprocess
import torch
import xbrz  # See xBRZ scaling on Jira

from enum import IntEnum
from PIL import Image
from RealESRGAN import RealESRGAN

# from rarch import CommonShaders

# import scalercg


# Enum with all available algorithms
# Ordered alphabetically with number indicating the quality from 0 (lowest) up
class Algorithms(IntEnum):
    CPP_DEBUG = -1

    BICUBIC = 3  # less blur than bilinear
    BILINEAR = 2
    LANCZOS = 4  # less blur than bicubic, but artifacts may appear
    NEAREST_NEIGHBOR = 0
    RealESRGAN = 6
    SUPIR = 7
    xBRZ = 5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# convert_scaler_algorithm_to_pillow_algorithm
def csatpa(algorithm: Algorithms):  # TODO: Convert to a dictionary
    match algorithm:
        case Algorithms.NEAREST_NEIGHBOR:
            return Image.NEAREST
        case Algorithms.BILINEAR:
            return Image.BILINEAR
        case Algorithms.BICUBIC:
            return Image.BICUBIC
        case Algorithms.LANCZOS:
            return Image.LANCZOS
        case _:
            raise AttributeError("Algorithm not supported by PIL")


# Main function for Python for existing libs
def scale_image(algorithm, pil_image: Image, factor, fallback_algorithm=Algorithms.BICUBIC, main_checked=False) -> Image:
    pil_image = pil_image.convert('RGBA')
    width, height = pil_image.size
    output_width, output_height = round(width * factor), round(height * factor)

    match algorithm:
        case Algorithms.NEAREST_NEIGHBOR:
            return pil_image.resize((output_width, output_height), Image.NEAREST)
        case Algorithms.BILINEAR:
            return pil_image.resize((output_width, output_height), Image.BILINEAR)
        case Algorithms.BICUBIC:
            return pil_image.resize((output_width, output_height), Image.BICUBIC)
        case Algorithms.LANCZOS:
            return pil_image.resize((output_width, output_height), Image.LANCZOS)
        case Algorithms.xBRZ:
            # if factor > 6:
            #     raise ValueError("Max factor for xbrz=6")
            # factor = Fraction(factor).limit_denominator().numerator
            if factor < 1:
                raise ValueError("xBRZ does not support downscaling!")
            # If factor is not a whole number or is greater than 6, print a warning
            if factor != int(factor) or factor > 6:
                print(f"WARNING: Scaling by xBRZ with factor {factor} is not supported, result might be blurry!")

            current_scale = 1
            while current_scale < factor:
                temp_factor = 6
                while current_scale * temp_factor > factor:
                    temp_factor -= 1
                temp_factor = min(temp_factor + 1, 6)

                pil_image = xbrz.scale_pillow(pil_image, temp_factor)
                current_scale *= temp_factor

            return pil_image.resize((output_width, output_height), csatpa(fallback_algorithm))
        case Algorithms.RealESRGAN:
            image = pil_image.convert('RGB')

            if factor < 1:
                raise ValueError("xBRZ does not support downscaling!")
            # If factor is not a whole number or is greater than 6, print a warning
            if factor != int(factor) or factor > 8:
                print(f"WARNING: Scaling by RealESRGAN with factor {factor} is not supported, result might be blurry!")

            current_scale = 1
            while current_scale < factor:
                temp_factor = 8
                while current_scale * temp_factor >= factor:
                    temp_factor //= 2
                temp_factor = min(temp_factor * 2, 8)

                model = RealESRGAN(device, scale=temp_factor)
                model.load_weights(f'weights/RealESRGAN_x{temp_factor}.pth')  # , download=True
                image = model.predict(image)

                current_scale *= temp_factor

            return image.resize((output_width, output_height), csatpa(fallback_algorithm))
        case Algorithms.SUPIR:
            script_path = './SUPIR/test.py'

            # Command 1
            # command1 = f"CUDA_VISIBLE_DEVICES=0 python {script_path} --img_dir '../input' --save_dir ../output --SUPIR_sign Q --upscale 2"
            command1 = f"python {script_path} --img_dir '../input' --save_dir ../output --SUPIR_sign Q --upscale 2"

            # Command 2
            # command2 = f"CUDA_VISIBLE_DEVICES=0 python {script_path} --img_dir '../input' --save_dir ../output --SUPIR_sign F --upscale 2 --s_cfg 4.0 --linear_CFG"

            # Execute commands
            # subprocess.run(command1, shell=True)
            subprocess.run(command1)
            # subprocess.run(command2, shell=True)

            # subprocess.run(['python', script_path])
        case _:
            if main_checked:
                raise NotImplementedError("Not implemented yet")
            else:
                width, height = pil_image.size
                # pixels = [[[int]]]
                pixels = [[[0, 0, 0, 0] for _ in range(width)] for _ in range(height)]
                for y in range(height):
                    for x in range(width):
                        pixels[y][x] = pil_image.getpixel((x, y))
                return scale_image_data(algorithm, pixels, factor, fallback_algorithm, True)


# def scale_image(algorithm, pil_image:Image, output_width, output_height):
#     pass


# Main function for C++ lib
def scale_image_data(algorithm, pixels: [[[int]]], factor, fallback_algorithm=Algorithms.BICUBIC, main_checked=False):
    match algorithm:
        # case Algorithms.CPP_DEBUG:
        #     # new_pixels = scalercg.scale("cpp_debug", pixels, factor)
        #     new_pixels = scalercg.scale(pixels, factor, "cpp_debug")
        #     image = Image.new("RGBA", (len(new_pixels[0]), len(new_pixels)))
        #     for y in range(len(new_pixels)):
        #         for x in range(len(new_pixels[0])):
        #             image.putpixel((x, y), new_pixels[y][x])
        #     return image
        case _:
            if main_checked:
                raise NotImplementedError("Not implemented yet")
            else:
                image = Image.new("RGBA", (len(pixels[0]) * factor, len(pixels) * factor))
                for y in range(len(pixels)):
                    for x in range(len(pixels[0])):
                        image.putpixel((x * factor, y * factor), pixels[y][x])
                return scale_image(algorithm, image, factor, fallback_algorithm, True)


# def scale_image_parallel(algorithm, pil_image: Image, factor, fallback_algorithm=Algorithms.BICUBIC):
#     # Create new thread and run scale_image in it
#     pass

# def scale_image(algorithm, pixels:[[[int]]], output_width, output_height):
#     pass
#
# def scale_image(algorithm, pixels:[[int]], width, height, factor):
#     pass
#
# def scale_image(algorithm, pixels:[[int]], input_width, input_height, output_width, output_height):
#     pass
