# coding=utf-8

import io
import struct
# from email.policy import default
from enum import IntEnum
from PIL import Image


# Enum with all available algorithms
# Ordered alphabetically with number indicating the quality from 0 (lowest) up
# (Only up to 6! (rest are just increments because sorting was hard :/ ))
class Algorithms(IntEnum):
    CPP_DEBUG = -1

    BICUBIC = 2  # less blur and artifacts than bilinear
    BILINEAR = 1
    CAS = 8  # contrast adaptive sharpening
    FSR = 7  # FidelityFX Super Resolution
    LANCZOS = 3  # less blur than bicubic, but artifacts may appear
    NEAREST_NEIGHBOR = 0
    RealESRGAN = 5
    SUPIR = 6
    xBRZ = 4


class Filters(IntEnum):
    CAS = 0  # contrast adaptive sharpening


def string_to_algorithm(string: str) -> Algorithms:
    match string.lower():
        case "bicubic":
            return Algorithms.BICUBIC
        case "bilinear":
            return Algorithms.BILINEAR
        case "cas":
            return Algorithms.CAS
        case "fsr":
            return Algorithms.FSR
        case "lanczos":
            return Algorithms.LANCZOS
        case "nearest-neighbour":
            return Algorithms.NEAREST_NEIGHBOR
        case "real-esrgan":
            return Algorithms.RealESRGAN
        case "supir":
            return Algorithms.SUPIR
        case "xbrz":
            return Algorithms.xBRZ
        case _:
            raise ValueError("Algorithm not found")


@DeprecationWarning
def algorithm_to_string(algorithm: Algorithms) -> str:
    return algorithm.name
    # match algorithm:
    #     case Algorithms.NEAREST_NEIGHBOR:
    #         return 'Nearest-neighbour'
    #     case Algorithms.BILINEAR:
    #         return 'Bilinear'
    #     case Algorithms.BICUBIC:
    #         return 'Bicubic'
    #     case Algorithms.LANCZOS:
    #         return 'Lanczos'
    #     case Algorithms.xBRZ:
    #         return 'xBRZ'
    #     case Algorithms.RealESRGAN:
    #         return 'RealESRGAN'
    #     case _:
    #         raise ValueError("Algorithm is not yet translated")


def image_to_byte_array(image: Image, additional_lossless_compression=True) -> bytes:
    # If additional_lossless_compression is True, apply lossless compression
    if additional_lossless_compression:
        return apply_lossless_compression(image)
    # else, just convert the image to bytes

    # BytesIO is a file-like buffer stored in memory
    img_byte_arr = io.BytesIO()

    # image.save expects a file-like as an argument
    image.save(img_byte_arr, format='PNG')

    # Turn the BytesIO object back into a bytes object
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr


def apply_lossless_compression(image: Image) -> bytes:
    img_byte_arr = io.BytesIO()

    # if image.mode == 'RGBA':
    #     # Go through every pixel and check if alpha is 255, if it 255 on every pixel, save it as RGB
    #     # else save it as RGBA
    #     alpha_was_used = any(pixel[3] != 255 for pixel in image.getdata())
    #     if not alpha_was_used:
    #         image = image.convert('RGB')

    if not has_transparency(image):
        image = image.convert('RGB')

    image.save(img_byte_arr, optimize=True, format='PNG')

    unique_colors_number = len(set(image.getdata()))
    if unique_colors_number <= 256:
        colors = 256
        if unique_colors_number <= 2:
            colors = 2
        elif unique_colors_number <= 4:
            colors = 4
        elif unique_colors_number <= 16:
            colors = 16

        img_temp_byte_arr = io.BytesIO()
        image = image.convert('P', palette=Image.ADAPTIVE, colors=colors)
        image.save(img_temp_byte_arr, optimize=True, format='PNG')

        # Check which one is smaller and keep it, remove the other one
        # (if the palette is smaller remove '_P' from the name)
        if len(img_temp_byte_arr.getvalue()) < len(img_byte_arr.getvalue()):
            img_byte_arr = img_temp_byte_arr

    return img_byte_arr.getvalue()


@DeprecationWarning
def string_to_scaling_algorithm(string: str) -> Algorithms:
    return string_to_algorithm(string)


def float_to_int32(float_value):
    return struct.unpack('!I', struct.pack('!f', float_value))[0]


def int32_to_float(int_value):
    return struct.unpack('!f', struct.pack('!I', int_value))[0]


def hdr_to_sdr(hdr_image):
    # Convert HDR image to 4x SDR image
    pass


def has_transparency(img: Image) -> bool:
    if img.info.get("transparency", None) is not None:
        return True
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True

    return False


if __name__ == "__main__":
    # Example
    # float_value = 266123.5
    float_value = 3.4e+38
    int_value = float_to_int32(float_value)
    result_float = int32_to_float(int_value)

    print(f"Original float: {float_value}")
    print(f"Converted integer: {int_value}")
    print(f"Converted back to float: {result_float}")
