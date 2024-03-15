# coding=utf-8

import io
import struct
from email.policy import default
from enum import IntEnum
from PIL import Image


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


AlgorithmsToStringDictionary = {
    Algorithms.NEAREST_NEIGHBOR: 'nearest-neighbour',
    Algorithms.BILINEAR: 'bilinear',
    Algorithms.BICUBIC: 'bicubic',
    Algorithms.LANCZOS: 'lanczos',
    Algorithms.xBRZ: 'xbrz',
    Algorithms.RealESRGAN: 'real-esrgan'
}
AlgorithmsFromStringDictionary = {
    'nearest-neighbour': Algorithms.NEAREST_NEIGHBOR,
    'bilinear': Algorithms.BILINEAR,
    'bicubic': Algorithms.BICUBIC,
    'lanczos': Algorithms.LANCZOS,
    'xbrz': Algorithms.xBRZ,
    'real-esrgan': Algorithms.RealESRGAN
}


def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(imgByteArr, format='PNG')
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def string_to_scaling_algorithm(string: str) -> Algorithms:
    if string in AlgorithmsFromStringDictionary:
        return AlgorithmsFromStringDictionary[string]
    elif string == "esrgan":
        return Algorithms.RealESRGAN
    else:
        raise ValueError("Algorithm not found")


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
