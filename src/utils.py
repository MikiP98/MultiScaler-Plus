# coding=utf-8

import io
import struct
from email.policy import default
from PIL import Image
from scaler import Algorithms


def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(imgByteArr, format='PNG')
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def string_to_scaling_algorithm(string: str) -> Algorithms:
    match string:
        case 'nearest-neighbour':
            return Algorithms.NEAREST_NEIGHBOR
        case 'bilinear':
            return Algorithms.BILINEAR
        case 'bicubic':
            return Algorithms.BICUBIC
        case 'lanczos':
            return Algorithms.LANCZOS
        case 'xbrz':
            return Algorithms.xBRZ
        case 'esrgan':
            return Algorithms.RealESRGAN
        case _:
            raise ValueError("Algorithm not found")


def float_to_int32(float_value):
    return struct.unpack('!I', struct.pack('!f', float_value))[0]


def int32_to_float(int_value):
    return struct.unpack('!f', struct.pack('!I', int_value))[0]


def hdr_to_sdr(hdr_image):
    # Convert HDR image to 4x SDR image
    pass


if __name__ == "__main__":
    # Example
    # float_value = 266123.5
    float_value = 3.4e+38
    int_value = float_to_int32(float_value)
    result_float = int32_to_float(int_value)

    print(f"Original float: {float_value}")
    print(f"Converted integer: {int_value}")
    print(f"Converted back to float: {result_float}")
