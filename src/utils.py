# coding=utf-8

import cv2
import io
import numpy as np
import PIL.Image
import struct

from enum import auto, IntEnum, unique
from termcolor import colored
from termcolor._types import Color as TermColor  # Ignore this warning, TODO: create an issue on the termcolor repo.
from typing import Optional, TypedDict, Union


# class Image:
#     def __init__(self, images: list[list[PIL.Image]], *, is_animated=False, animation_spacing=(1000/30)):
#         self.images = images
#         if is_animated:
#             self.animationSpacing = animation_spacing


class ImageDict(TypedDict):
    images: list[list[PIL.Image]]  # List of scaled lists of image frames/layers, only 1 entry on input
    is_animated: Optional[bool]
    animation_spacing: Optional[float]


# Enum with all available algorithms
# Ordered alphabetically
@unique
class Algorithms(IntEnum):
    CPP_DEBUG = -1

    Anime4K = auto()
    CAS = auto()  # contrast adaptive sharpening
    CV2_INTER_AREA = auto()  # resampling using pixel area relation
    CV2_INTER_CUBIC = auto()  # bicubic interpolation over 4x4 pixel neighborhood
    CV2_INTER_LANCZOS4 = auto()  # Lanczos interpolation over 8x8 pixel neighborhood
    CV2_INTER_LINEAR = auto()  # bilinear interpolation
    CV2_INTER_NEAREST = auto()  # nearest-neighbor interpolation
    CV2_EDSR = auto()  # Enhanced Deep Super-Resolution
    CV2_ESPCN = auto()  # Efficient Sub-Pixel Convolutional Neural Network
    CV2_FSRCNN = auto()  # Fast Super-Resolution Convolutional Neural Network
    CV2_FSRCNN_small = auto()  # Fast Super-Resolution Convolutional Neural Network - Small
    CV2_LapSRN = auto()  # Laplacian Super-Resolution Network
    FSR = auto()  # FidelityFX Super Resolution
    hqx = auto()  # high quality scale

    HSDBTRE = auto()

    NEDI = auto()  # New Edge-Directed Interpolation
    PIL_BICUBIC = auto()  # less blur and artifacts than bilinear, but slower
    PIL_BILINEAR = auto()
    PIL_LANCZOS = auto()  # less blur than bicubic, but artifacts may appear
    PIL_NEAREST_NEIGHBOR = auto()
    RealESRGAN = auto()
    Repetition = auto()

    SI_drln_bam = auto()
    SI_edsr = auto()
    SI_msrn = auto()
    SI_mdsr = auto()
    SI_msrn_bam = auto()
    SI_edsr_base = auto()
    SI_mdsr_bam = auto()
    SI_awsrn_bam = auto()
    SI_a2n = auto()
    SI_carn = auto()
    SI_carn_bam = auto()
    SI_pan = auto()
    SI_pan_bam = auto()

    SI_drln = auto()
    SI_han = auto()
    SI_rcan_bam = auto()

    Super_xBR = auto()
    xBRZ = auto()

    # Docker start
    SUPIR = auto()
    Waifu2x = auto()


@unique
class Filters(IntEnum):
    CAS = auto()  # contrast adaptive sharpening

    NORMAL_MAP_STRENGTH = auto()
    AUTO_NORMAL_MAP = auto()
    AUTO_SPECULAR_MAP = auto()

    SI_TODO = auto()  # TODO: Add filters


cli_algorithms = {Algorithms.FSR, Algorithms.CAS, Algorithms.SUPIR, Algorithms.Super_xBR}


string_to_algorithm_dict = {
    "cv2_area": Algorithms.CV2_INTER_AREA,
    "cv2_bicubic": Algorithms.CV2_INTER_CUBIC,
    "cv2_bilinear": Algorithms.CV2_INTER_LINEAR,
    "cv2_lanczos": Algorithms.CV2_INTER_LANCZOS4,
    "cv2_nearest": Algorithms.CV2_INTER_NEAREST,

    "cv2_edsr": Algorithms.CV2_EDSR,
    "cv2_espcn": Algorithms.CV2_ESPCN,
    "cv2_fsrcnn": Algorithms.CV2_FSRCNN,
    "cv2_fsrcnn_small": Algorithms.CV2_FSRCNN_small,
    "cv2_lapsrn": Algorithms.CV2_LapSRN,

    "pil_bicubic": Algorithms.PIL_BICUBIC,
    "pil_bilinear": Algorithms.PIL_BILINEAR,
    "pil_lanczos": Algorithms.PIL_LANCZOS,
    "pil_nearest": Algorithms.PIL_NEAREST_NEIGHBOR,
    "nedi": Algorithms.NEDI,
    "cas": Algorithms.CAS,
    "fsr": Algorithms.FSR,
    "hqx": Algorithms.hqx,  # "hq2x", "hq3x", "hq4x"
    "real_esrgan": Algorithms.RealESRGAN,
    "super_xbr": Algorithms.Super_xBR,
    "supir": Algorithms.SUPIR,
    "xbrz": Algorithms.xBRZ
}


def string_to_algorithm(string: str) -> Algorithms:
    return string_to_algorithm_dict[string.lower()]


algorithm_to_string_dict = {
    Algorithms.CV2_INTER_AREA: "cv2_area",
    Algorithms.CV2_INTER_CUBIC: "cv2_bicubic",
    Algorithms.CV2_INTER_LINEAR: "cv2_bilinear",
    Algorithms.CV2_INTER_LANCZOS4: "cv2_lanczos",
    Algorithms.CV2_INTER_NEAREST: "cv2_nearest",

    Algorithms.CV2_EDSR: "cv2_edsr",
    Algorithms.CV2_ESPCN: "cv2_espcn",
    Algorithms.CV2_FSRCNN: "cv2_fsrcnn",
    Algorithms.CV2_FSRCNN_small: "cv2_fsrcnn_small",
    Algorithms.CV2_LapSRN: "cv2_lapsrn",

    Algorithms.PIL_BICUBIC: "pil_bicubic",
    Algorithms.PIL_BILINEAR: "pil_bilinear",
    Algorithms.PIL_LANCZOS: "pil_lanczos",
    Algorithms.PIL_NEAREST_NEIGHBOR: "pil_nearest",

    Algorithms.CAS: "cas",
    Algorithms.FSR: "fsr",
    Algorithms.hqx: "hqx",
    Algorithms.NEDI: "nedi",
    Algorithms.RealESRGAN: "real_esrgan",
    Algorithms.Super_xBR: "super_xbr",
    Algorithms.SUPIR: "supir",
    Algorithms.xBRZ: "xbrz"
}


def algorithm_to_string(algorithm: Algorithms) -> str:
    return algorithm_to_string_dict[algorithm]


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
    "BLP": {"blp2"},  # Only BLP2 supports multiple images and animations
    "TIFF": {"tif", "tiff", "tiff2"},
    "APNG": {"apng"},
    "WebP": {"webp"},
    "JPX": {"jpx"}  # Only JPEG 2000 Part 2 (JPX) supports multiple images and animations
}
# AV1
# MNG: {.mng} MNG supports both multiple images and animations
pil_animated_formats_cache = {
    extension for extensions in pil_animated_formats for extension in extensions
}


def pil_to_cv2(pil_image: PIL.Image) -> 'np.ndarray':
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


def cv2_to_pil(cv2_image: 'np.ndarray') -> PIL.Image:
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


def image_to_byte_array(image: PIL.Image, additional_lossless_compression=True) -> bytes:
    # If additional_lossless_compression is True, apply lossless compression
    if additional_lossless_compression:
        return apply_lossless_compression_png(image)
    # else, just convert the image to bytes

    # BytesIO is a file-like buffer stored in memory
    img_byte_arr = io.BytesIO()

    # image.save expects a file-like as an argument
    image.save(img_byte_arr, format='PNG')

    # Turn the BytesIO object back into a bytes object
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr


def apply_lossless_compression(image: PIL.Image, optional_args: dict) -> bytes:
    img_byte_arr = io.BytesIO()

    mode = 'RGBA'
    if not has_transparency(image):
        mode = 'RGB'

    image.save(img_byte_arr, **optional_args)

    unique_colors_number = len(set(image.getdata()))
    # print(f"Unique colors: {unique_colors_number}")
    if unique_colors_number <= 256:
        colors = 256
        if unique_colors_number <= 2:
            colors = 2
        elif unique_colors_number <= 4:
            colors = 4
        elif unique_colors_number <= 16:
            colors = 16

        img_temp_byte_arr = io.BytesIO()
        temp_image = image.convert('P', palette=PIL.Image.ADAPTIVE, colors=colors)  # sometimes deletes some data :/

        # Additional check to see if PIL didn't fuck up
        same = True
        for data1, data2 in zip(image.getdata(), temp_image.convert(mode).getdata()):
            if data1 != data2:
                # print(f"{data1} != {data2}")
                same = False
                break
        # if all([data1 == data2 for data1, data2 in zip(image.getdata(), temp_image.getdata())]):

        if same:
            temp_image.save(img_temp_byte_arr, **optional_args)

            # Check which one is smaller and keep it, remove the other one
            # (if the palette is smaller remove '_P' from the name)
            if len(img_temp_byte_arr.getvalue()) < len(img_byte_arr.getvalue()):
                img_byte_arr = img_temp_byte_arr
                # print("Saving palette")

    return img_byte_arr.getvalue()


def apply_lossless_compression_png(image: PIL.Image) -> bytes:
    optional_args = {
        'optimize': True,
        'format': 'PNG'
    }
    return apply_lossless_compression(image, optional_args)


def apply_lossless_compression_webp(image: PIL.Image) -> bytes:
    optional_args = {
        'lossless': True,
        'method': 6,
        'optimize': True,
        'format': 'WEBP'
    }
    return apply_lossless_compression(image, optional_args)


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


@DeprecationWarning
def string_to_scaling_algorithm(string: str) -> Algorithms:
    return string_to_algorithm(string)


def float_to_int32(float_value):
    return struct.unpack('!I', struct.pack('!f', float_value))[0]


def int32_to_float(int_value):
    return struct.unpack('!f', struct.pack('!I', int_value))[0]


def hdr_to_sdr(hdr_image):
    # Convert HDR image to 4x SDR image
    raise NotImplementedError("HDR to SDR conversion is not implemented yet!")


def generate_mask(image: PIL.Image, scale: float, mode: tuple) -> np.ndarray:
    # Generate an outbound mask for the image
    mask_mode = 'A'
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


def has_transparency(img: Union[PIL.Image, np.ndarray]) -> bool:
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


def uses_transparency(img: Union[PIL.Image, np.ndarray]) -> bool:
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


def rainbowify(text: str) -> str:
    colors: list[TermColor] = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    len_colors = len(colors)
    rainbow_text = ""
    i = 0
    for char in text:
        if char == ' ':
            rainbow_text += ' '
        else:
            rainbow_text += colored(char, colors[i % len_colors])
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
