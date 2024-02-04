# coding=utf-8
import xbrz_py__1_0_2.xbrz

from PIL import Image
from enum import IntEnum


class Algorithms(IntEnum):
    xBRZ = 1
    RealESRGAN = 2


# Current main function
def scale_image(algorithm, pil_image:Image, factor):
    if algorithm == Algorithms.xBRZ:
        return xbrz_py__1_0_2.xbrz.scale(pil_image, factor, pil_image.width, pil_image.height, xbrz_py__1_0_2.xbrz.ColorFormat.RGBA)
    else:
        raise NotImplementedError("Algorithm not implemented yet")

def scale_image(algorithm, pil_image:Image, output_width, output_height):
    pass

def scale_image(algorithm, pixels:[[int]], factor):
    pass

def scale_image(algorithm, pixels:[[int]], output_width, output_height):
    pass

def scale_image(algorithm, pixels:[int], width, height, factor):
    pass

def scale_image(algorithm, pixels:[int], input_width, input_height, output_width, output_height):
    pass
