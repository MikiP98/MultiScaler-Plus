# coding=utf-8
from PIL import Image
from enum import IntEnum
import xbrz # See xBRZ scaling on Jira


class Algorithms(IntEnum):
    NEAREST_NEIGHBOR = 0
    xBRZ = 1
    RealESRGAN = 2
    BILINEAR = 3
    BICUBIC = 4
    LANCZOS = 5


# Current main function
def scale_image(algorithm, pil_image:Image, factor) -> Image:
    pil_image = pil_image.convert('RGBA')
    width, height = pil_image.size

    if algorithm == Algorithms.BICUBIC:
        return pil_image.resize((width*factor, height*factor), Image.BICUBIC)
    elif algorithm == Algorithms.BILINEAR:
        return pil_image.resize((width*factor, height*factor), Image.BILINEAR)
    elif algorithm == Algorithms.NEAREST_NEIGHBOR:
        return pil_image.resize((width*factor, height*factor), Image.NEAREST)
    elif algorithm == Algorithms.LANCZOS:
        return pil_image.resize((width*factor, height*factor), Image.LANCZOS)
    elif algorithm == Algorithms.xBRZ:
        if factor > 6:
            raise ValueError("Max factor for xbrz=6")
        return xbrz.scale_pillow(pil_image, factor)
    elif algorithm == Algorithms.RealESRGAN:
        raise NotImplementedError("Not implemented yet")


# def scale_image(algorithm, pil_image:Image, output_width, output_height):
#     pass

# def scale_image(algorithm, pixels:[[int]], factor):
#     if algorithm == Algorithms.NEAREST_NEIGHBOR:
#         return nearest_neighbor.scale_image(pil_image, factor)
#     else:
#         raise NotImplementedError("Algorithm not implemented yet")

# def scale_image(algorithm, pixels:[[int]], output_width, output_height):
#     pass

# def scale_image(algorithm, pixels:[int], width, height, factor):
#     pass

# def scale_image(algorithm, pixels:[int], input_width, input_height, output_width, output_height):
#     pass
