# coding=utf-8
import xbrz

from PIL import Image
from enum import IntEnum


class Algorithms(IntEnum):
    xBRZ = 1
    RealESRGAN = 2


def scale_image(algorithm, pil_image:Image, factor):
    pass

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
