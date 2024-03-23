# coding=utf-8

import cv2
import io
import numpy as np
import PIL.Image
import struct
# from email.policy import default
from enum import IntEnum
# from PIL import Image


# Enum with all available algorithms
# Ordered alphabetically
class Algorithms(IntEnum):
    CPP_DEBUG = -1

    CAS = 0  # contrast adaptive sharpening
    CV2_INTER_AREA = 1  # resampling using pixel area relation
    CV2_INTER_CUBIC = 2  # bicubic interpolation over 4x4 pixel neighborhood
    CV2_INTER_LANCZOS4 = 3  # Lanczos interpolation over 8x8 pixel neighborhood
    CV2_INTER_LINEAR = 4  # bilinear interpolation
    CV2_INTER_NEAREST = 5  # nearest-neighbor interpolation
    CV2_EDSR = 6  # Enhanced Deep Super-Resolution
    CV2_ESPCN = 7  # Efficient Sub-Pixel Convolutional Neural Network
    CV2_FSRCNN = 8  # Fast Super-Resolution Convolutional Neural Network
    CV2_LapSRN = 9  # Laplacian Super-Resolution Network
    FSR = 5  # FidelityFX Super Resolution
    PIL_BICUBIC = 6  # less blur and artifacts than bilinear, but slower
    PIL_BILINEAR = 7
    PIL_LANCZOS = 8  # less blur than bicubic, but artifacts may appear
    PIL_NEAREST_NEIGHBOR = 9
    RealESRGAN = 10
    SUPIR = 11
    xBRZ = 12


class Filters(IntEnum):
    CAS = 0  # contrast adaptive sharpening


string_to_algorithm_dict = {
    "cv2_area": Algorithms.CV2_INTER_AREA,
    "cv2_bicubic": Algorithms.CV2_INTER_CUBIC,
    "cv2_bilinear": Algorithms.CV2_INTER_LINEAR,
    "cv2_lanczos": Algorithms.CV2_INTER_LANCZOS4,
    "cv2_nearest": Algorithms.CV2_INTER_NEAREST,

    "cv2_edsr": Algorithms.CV2_EDSR,
    "cv2_espcn": Algorithms.CV2_ESPCN,
    "cv2_fsrcnn": Algorithms.CV2_FSRCNN,
    "cv2_lapsrn": Algorithms.CV2_LapSRN,

    "pil_bicubic": Algorithms.PIL_BICUBIC,
    "pil_bilinear": Algorithms.PIL_BILINEAR,
    "pil_lanczos": Algorithms.PIL_LANCZOS,
    "pil_nearest": Algorithms.PIL_NEAREST_NEIGHBOR,

    "cas": Algorithms.CAS,
    "fsr": Algorithms.FSR,
    "real_esrgan": Algorithms.RealESRGAN,
    "supir": Algorithms.SUPIR,
    "xbrz": Algorithms.xBRZ
}


def string_to_algorithm(string: str) -> Algorithms:
    return string_to_algorithm_dict[string.lower()]


@DeprecationWarning
def algorithm_to_string(algorithm: Algorithms) -> str:
    return algorithm.name


def pil_to_cv2(pil_image: PIL.Image) -> 'np.ndarray':
    if has_transparency(pil_image):
        # print("Converting from RGBA to BGRA format...")
        pil_image = pil_image.convert('RGBA')

        # Convert Pillow image to NumPy array
        numpy_array = np.array(pil_image)

        # Convert NumPy array to OpenCV format
        return cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2BGRA)
    else:
        # print("Converting from RGB to BGR format...")
        # Convert Pillow image to NumPy array
        numpy_array = np.array(pil_image)

        # Convert NumPy array to OpenCV format
        return cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: 'np.ndarray') -> PIL.Image:
    if cv2_image.shape[2] == 4:
        print("Converting from BGRA to RGBA format...")
        # Convert OpenCV image to NumPy array
        numpy_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)

        # Convert NumPy array to Pillow format
        return PIL.Image.fromarray(numpy_array)
    else:
        print("Converting from BGR to RGB format...")
        # Convert OpenCV image to NumPy array
        numpy_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to Pillow format
        return PIL.Image.fromarray(numpy_array)


def image_to_byte_array(image: PIL.Image, additional_lossless_compression=True) -> bytes:
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


def apply_lossless_compression(image) -> bytes:
    # if image is CV2, convert it to PIL
    if isinstance(image, np.ndarray):
        image = cv2_to_pil(image)

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
        image = image.convert('P', palette=PIL.Image.ADAPTIVE, colors=colors)
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


def has_transparency(img) -> bool:
    if isinstance(img, np.ndarray):
        return img.shape[2] == 4

    if img.info.get("transparency", None) is not None:
        return True

    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True

    elif img.mode == "RGBA":
        return True
        # extrema = img.getextrema()
        # if extrema[3][0] < 255:
        #     return True

    return False


def uses_transparency(img) -> bool:
    if isinstance(img, np.ndarray):
        # check if the image has an alpha channel
        if img.shape[2] == 4:
            # Check if the alpha channel is used
            return np.any(img[:, :, 3] != 255)

        return False

    if img.info.get("transparency", None) is not None:
        return True

    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True

    elif img.mode == "RGBA":
        cv2_image = pil_to_cv2(img)
        return np.any(cv2_image[:, :, 3] < 255)

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
