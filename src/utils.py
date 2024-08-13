# coding=utf-8

import cv2
import numpy as np
import PIL.Image
import struct

from termcolor import colored
from termcolor._types import Color as TermColor  # Ignore this warning, TODO: create an issue on the termcolor repo.
from typing import Optional, TypedDict


# class Image:
#     def __init__(self, images: list[list[PIL.Image.Image]], *, is_animated=False, animation_spacing=(1000/30)):
#         self.images = images
#         if is_animated:
#             self.animationSpacing = animation_spacing


class ImageDict(TypedDict):
    images: list[list[PIL.Image.Image]]  # List of scaled lists of image frames/layers, only 1 entry on input
    is_animated: Optional[bool]
    animation_spacing: Optional[float]


# TODO: Think about frozen sets
pil_fully_supported_formats = {
    "BLP": ("blp", "blp2", "tex",),
    "BMP": ("bmp", "rle",),
    "DDS": ("dds", "dds2",),
    "DIB": ("dib", "dib2",),
    "EPS": ("eps", "eps2", "epsf", "epsi",),
    "GIF": ("gif", "giff",),
    "ICNS": ("icns", "icon",),
    "ICO": ("ico", "cur",),
    "IM": ("im", "im2",),
    "JPEG": ("jpg", "jpeg", "jpe",),
    "JPEG 2000": ("jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx",),
    "MSP": ("msp", "msp2",),
    "PCX": ("pcx", "pcx2",),
    "PFM": ("pfm", "pfm2",),
    "PNG": ("png", "pns",),
    "APNG": ("apng", "png2",),
    "PPM": ("ppm", "ppm2",),
    "SGI": ("sgi", "rgb", "bw",),
    "SPIDER": ("spi", "spider2",),
    "TGA": ("tga", "targa",),
    "TIFF": ("tif", "tiff", "tiff2",),
    "WebP": ("webp", "webp2",),
    "XBM": ("xbm", "xbm2",),

    "AVIF": ("avif",),  # From outside plugin
    "JPEG_XL": ("jxl",)  # From outside plugin
}
pil_fully_supported_formats_cache = frozenset(
    extension for extensions in pil_fully_supported_formats.values() for extension in extensions
)
# print(pil_fully_supported_formats.values())

pil_read_only_formats = {
    "CUR": ("cur",),
    "DCX": ("dcx",),
    "FITS": ("fits",),
    "FLI": ("fli",),
    "FLC": ("flc",),
    "FPX": ("fpx",),
    "FTEX": ("ftex",),
    "GBR": ("gbr",),
    "GD": ("gd",),
    "IMT": ("imt",),
    "IPTC": ("iptc",),
    "NAA": ("naa",),
    "MCIDAS": ("mcidas",),
    "MIC": ("mic",),
    "MPO": ("mpo",),
    "PCD": ("pcd",),
    "PIXAR": ("pixar",),
    "PSD": ("psd",),
    "QOI": ("qoi",),
    "SUN": ("sun",),
    "WAL": ("wal",),
    "WMF": ("wmf",),
    "EMF": ("emf",),
    "XPM": ("xpm",)
}
pil_read_only_formats_cache = frozenset(
    extension for extensions in pil_read_only_formats.values() for extension in extensions
)

pil_write_only_formats = {
    "PALM": ("palm",),
    "PDF": ("pdf",),
    "XV Thumbnails": ("xv",)
}
pil_write_only_formats_cache = frozenset(
    extension for extensions in pil_write_only_formats.values() for extension in extensions
)

pil_indentify_only_formats = {
    "BUFR": ("bufr",),
    "GRIB": ("grib", "grb",),
    "HDF5": ("h5", "hdf5",),
    "MPEG": ("mpg", "mpeg",)
}
pil_indentify_only_formats_cache = frozenset(
    extension for extensions in pil_indentify_only_formats.values() for extension in extensions
)


pil_animated_formats = {
    "BLP": ("blp2",),  # Only BLP2 supports multiple images and animations
    "TIFF": ("tif", "tiff", "tiff2",),
    "APNG": ("apng",),
    "WebP": ("webp",),
    "JPX": ("jpx",)  # Only JPEG 2000 Part 2 (JPX) supports multiple images and animations
}
# AV1
# MNG: {.mng} MNG supports both multiple images and animations
pil_animated_formats_cache = {
    extension for extensions in pil_animated_formats for extension in extensions
}


def pil_to_cv2(pil_image: PIL.Image.Image) -> np.ndarray:
    """
    Convert a Pillow image to OpenCV format
    :param pil_image: PIL image object (PIL.Image)
    :return: OpenCV format image (np.ndarray)
    """
    if has_transparency(pil_image):
        pil_image = pil_image.convert('RGBA')
        color_format = cv2.COLOR_RGBA2BGRA
    else:
        pil_image = pil_image.convert('RGB')
        color_format = cv2.COLOR_RGB2BGR

    # Convert Pillow image to NumPy array and then to OpenCV format
    return cv2.cvtColor(np.array(pil_image), color_format)


def cv2_to_pil(cv2_image: np.ndarray) -> PIL.Image.Image:
    """
    Convert an OpenCV image to Pillow format
    :param cv2_image: OpenCV format image (np.ndarray)
    :return: PIL image object (PIL.Image)
    """
    if cv2_image.shape[2] == 4:
        # print("Converting from BGRA to RGBA format...")
        # Convert OpenCV image to NumPy array
        numpy_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)

        # Convert NumPy array to Pillow format
        return PIL.Image.fromarray(numpy_array)
    else:
        # print("Converting from BGR to RGB format...")
        # Convert OpenCV image to NumPy array
        numpy_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to Pillow format
        return PIL.Image.fromarray(numpy_array)


# def pngify_class(image: PIL.Image) -> Image:
#     if image.format.lower() in pil_animated_formats_cache:
#         # Extract all frames from the animated image as a list of images
#         if image.is_animated:
#             raise NotImplementedError("Animated images are not supported yet")
#
#         raise NotImplementedError(
#             f"Animatable and stackable images are not supported yet: {pil_animated_formats_cache}"
#         )
#
#     # check if is RGBA or RGB
#     elif not (image.mode == "RGB" or image.mode == "RGBA"):
#         image = image.convert("RGBA")
#         if not uses_transparency(image):
#             image = image.convert("RGB")
#
#     return Image([[image]])
#     # return [image]  # Return an 'image' with single 'frame'


def pngify(image: PIL.Image) -> ImageDict:
    if image.format.lower() in pil_animated_formats_cache:
        # Extract all frames from the animated image as a list of images
        if image.is_animated:
            raise NotImplementedError("Animated images are not supported yet")

        raise NotImplementedError(
            f"Animatable and stackable images are not supported yet: {pil_animated_formats_cache}"
        )

    # check if is RGBA or RGB
    elif not (image.mode == "RGB" or image.mode == "RGBA"):
        image = image.convert("RGBA")
        if not uses_transparency(image):
            image = image.convert("RGB")

    return {'images': [[image]]}


def float_to_int32(float_value: float):
    return struct.unpack('!I', struct.pack('!f', float_value))[0]


def int32_to_float(int_value: int):
    return struct.unpack('!f', struct.pack('!I', int_value))[0]


def hdr_to_sdr(hdr_image):
    # Convert HDR image to 4x SDR image
    raise NotImplementedError("HDR to SDR conversion is not implemented yet!")


def generate_mask(image: PIL.Image, scale: float, mode: tuple) -> np.ndarray:
    # Generate an outbound mask for the image
    # mask_mode = 'A'
    if has_transparency(image):
        mask_mode = mode[0]
    else:
        mask_mode = mode[1]

    if mask_mode == 'alpha':
        ndarray = pil_to_cv2(image)

        # print(ndarray.shape)
        new_shape = ndarray.shape[:2]
        # print(new_shape)

        mask_array = np.zeros(new_shape, dtype=np.uint8)
        for i in range(ndarray.shape[0]):
            for j in range(ndarray.shape[1]):
                if ndarray[i, j, 3] != 0:
                    mask_array[i, j] = 255

        mask_image = cv2.resize(
            mask_array,
            (round(new_shape[1] * scale), round(new_shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST
        )
        return mask_image

    elif mask_mode == 'black':
        ndarray = pil_to_cv2(image)

        new_shape = ndarray.shape[:2]

        mask_array = np.zeros(new_shape, dtype=np.uint8)
        for i in range(ndarray.shape[0]):
            for j in range(ndarray.shape[1]):
                if sum(ndarray[i, j]) != 0:
                    mask_array[i, j] = 255

        # print(f"mask_array:\n{mask_array}")
        mask_image = cv2.resize(
            mask_array,
            (round(new_shape[1] * scale), round(new_shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST
        )
        # print(f"mask_image:\n{mask_image}")
        return mask_image


def apply_mask(image: PIL.Image, mask: np.ndarray) -> PIL.Image:
    # Apply a mask to the image
    image_array = pil_to_cv2(image)

    mask_py = list(mask)
    # print(f"mask_py:\n{mask_py}")
    # print(f"image_array:\n{image_array}")
    # print(f"mask shape: {mask.shape}")
    # print(f"image shape: {image_array.shape}")
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if mask_py[i][j] == 0:
                # print(f"Cleared pixel at ({i+1}, {j+1})")
                # print(f"Because mask value is {mask_py[j][i]}")
                for k in range(image_array.shape[2]):
                    image_array[i, j, k] = 0
            # for k in range(image_array.shape[2]):
            #     image_array[i, j, k] = mask_py[i][j]

    # print(f"mask_py:\n{mask_py}")
    return cv2_to_pil(image_array)


def has_transparency(img: PIL.Image.Image | np.ndarray) -> bool:
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

    return False


def uses_transparency(img: PIL.Image.Image | np.ndarray) -> bool:
    if isinstance(img, np.ndarray):
        # check if the image has an alpha channel
        if img.shape[2] == 4:
            # Check if the alpha channel is used
            return np.any(img[:, :, 3] != 255)

        return False

    elif img.info.get("transparency", None) is not None:
        return True

    elif img.mode == "P":
        transparent = img.info.get("transparency", -1)

        for _, index in img.getcolors():  # TODO: Consider using ndarray
            if index == transparent:
                return True

    elif img.mode == "RGBA":
        cv2_image = pil_to_cv2(img)
        return np.any(cv2_image[:, :, 3] < 255)

    return False


def avg(iterable) -> float:
    """
    Calculate the average of an iterable
    :param iterable:
    :return:
    """
    return sum(iterable) / len(iterable)


def geo_avg(iterable) -> float:
    """
    Calculate the geometric average of an iterable
    :param iterable:
    :return:
    """
    return (np.prod(iterable)) ** (1 / len(iterable))


def rainbowify(text: str, *, bold=False) -> str:
    b = '\x1B[1m' if bold else ''

    colors: list[TermColor] = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    len_colors = len(colors)
    rainbow_text = ""
    i = 0
    for char in text:
        if char == ' ':
            rainbow_text += ' '
        else:
            rainbow_text += b + colored(char, colors[i % len_colors])
            i += 1
    return rainbow_text


if __name__ == "__main__":  # This is a test code
    # Example
    # float_value = 266123.5
    float_value = 3.4e+38
    int_value = float_to_int32(float_value)
    result_float = int32_to_float(int_value)

    print(f"Original float: {float_value}")
    print(f"Converted integer: {int_value}")
    print(f"Converted back to float: {result_float}")
