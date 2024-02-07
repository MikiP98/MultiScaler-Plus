# coding=utf-8
import scalers.nearest_neighbor

from PIL import Image
from enum import IntEnum


class Algorithms(IntEnum):
    NEAREST_NEIGHBOR = 0
    xBRZ = 1
    RealESRGAN = 2


# Current main function
def scale_image(algorithm, pil_image:Image, factor):
    pass

def scale_image(algorithm, pil_image:Image, output_width, output_height):
    pass

def scale_image(algorithm, pixels:[[int]], factor):
    if algorithm == Algorithms.NEAREST_NEIGHBOR:
        return nearest_neighbor.scale_image(pil_image, factor)
    else:
        raise NotImplementedError("Algorithm not implemented yet")

def scale_image(algorithm, pixels:[[int]], output_width, output_height):
    pass

def scale_image(algorithm, pixels:[int], width, height, factor):
    pass

def scale_image(algorithm, pixels:[int], input_width, input_height, output_width, output_height):
    pass
