# coding=utf-8
import numpy as np
# import rarch
import torch
import xbrz  # See xBRZ scaling on Jira

from enum import IntEnum
from fractions import Fraction
from PIL import Image
from RealESRGAN import RealESRGAN


# Enum with all available algorithms
# Ordered alphabetically with number indicating the quality from 0 (lowest) up
class Algorithms(IntEnum):
    BICUBIC = 3  # less blur than bilinear
    BILINEAR = 2
    LANCZOS = 4  # less blur than bicubic, but artifacts may appear
    NEAREST_NEIGHBOR = 0
    RealESRGAN = 6
    xBRZ = 5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# convert_scaler_algorithm_to_pillow_algorithm
def csatpa(algorithm: Algorithms):
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
            factor = Fraction(factor).limit_denominator().numerator
            while factor != 1:
                gcd = factor
                if factor > 6:
                    print(f"WARNING: Scaling by xBRZ with factor {factor} is not supported, result might be blurry!")
                    # xBRZ can only scale up to 6x,
                    # find the biggest common divisor and scale by it until factor is <= 6
                    gcd = np.gcd(factor, 6)
                    while gcd == 1:
                        factor += 1
                        gcd = np.gcd(factor, 6)
                    # if gcd == 1:
                    #     raise ValueError("Factor is greater than 6 and undividable by smaller factors!")
                # print(f"Scaling by {gcd} to get factor smaller than 6")
                pil_image = xbrz.scale_pillow(pil_image, gcd)
                factor = factor // gcd

            return pil_image.resize((output_width, output_height), csatpa(fallback_algorithm))
        case Algorithms.RealESRGAN:
            image = pil_image.convert('RGB')
            factor = Fraction(factor).limit_denominator().numerator
            while factor != 1:
                gcd = factor
                if factor != 2 and factor != 4 and factor != 8:
                    print(f"WARNING: Scaling by RealESRGAN with factor {factor} is not supported, result might be blurry!")
                    # RealESRGAN can only scale up to 8x,
                    # find the biggest common divisor and scale by it until factor is <= 8
                    gcd = np.gcd(factor, 8)
                    while gcd == 1:
                        factor += 1
                        gcd = np.gcd(factor, 8)
                # print(f"GCD: {gcd} Factor: {factor}")
                model = RealESRGAN(device, scale=gcd)
                model.load_weights(f'weights/RealESRGAN_x{gcd}.pth')  # , download=True

                image = model.predict(image)
                factor = factor // gcd

            return image.resize((output_width, output_height), csatpa(fallback_algorithm))
        case _:
            if main_checked:
                raise NotImplementedError("Not implemented yet")
            else:
                width, height = pil_image.size
                pixels = [[[int]]]
                for y in range(height):
                    for x in range(width):
                        pixels[y][x] = pil_image.getpixel((x, y))
                return scale_image_data(algorithm, pixels, factor, fallback_algorithm, True)


# def scale_image(algorithm, pil_image:Image, output_width, output_height):
#     pass


# Main function for C++ lib
def scale_image_data(algorithm, pixels: [[[int]]], factor, fallback_algorithm=Algorithms.BICUBIC, main_checked=False):
    match algorithm:
        case _:
            if main_checked:
                raise NotImplementedError("Not implemented yet")
            else:
                image = Image.new("RGBA", (len(pixels[0]) * factor, len(pixels) * factor))
                for y in range(len(pixels)):
                    for x in range(len(pixels[0])):
                        image.putpixel((x * factor, y * factor), pixels[y][x])
                return scale_image(algorithm, image, factor, fallback_algorithm, True)


# def scale_image(algorithm, pixels:[[[int]]], output_width, output_height):
#     pass
#
# def scale_image(algorithm, pixels:[[int]], width, height, factor):
#     pass
#
# def scale_image(algorithm, pixels:[[int]], input_width, input_height, output_width, output_height):
#     pass
