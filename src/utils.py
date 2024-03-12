# coding=utf-8

import io
import struct
from email.policy import default
from PIL import Image
from scaler import Algorithms


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


@DeprecationWarning
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


if __name__ == "__main__":
    # Example
    # float_value = 266123.5
    float_value = 3.4e+38
    int_value = float_to_int32(float_value)
    result_float = int32_to_float(int_value)

    print(f"Original float: {float_value}")
    print(f"Converted integer: {int_value}")
    print(f"Converted back to float: {result_float}")
