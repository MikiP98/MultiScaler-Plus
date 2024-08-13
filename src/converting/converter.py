# coding=utf-8

import PIL.Image
import utils

from aenum import auto, IntEnum, unique
from typing import Callable


@unique
class Conversions(IntEnum):
    # File extensions aliases
    TO_AVIF = auto()
    TO_JPEG_XL = auto()
    TO_PNG = auto()
    TO_QOI = auto()
    TO_WEBP = auto()

    # Real conversions
    Old_SEUS_to_labPBR_1_3 = auto()
    Old_Continuum_to_labPBR_1_3 = auto()
    PPR_plus_Emissive_old_BSL_to_labPBR_1_3 = auto()
    Grey_to_labPBR_1_3 = auto()  # (most likely won't be great)


def conversion_not_implemented(_: list[PIL.Image.Image]) -> list[PIL.Image.Image]:
    raise NotImplementedError("Conversion not implemented")


convert_functions: dict[auto, Callable[[list[PIL.Image.Image]], list[PIL.Image.Image]]] = {
    Conversions.TO_AVIF: conversion_not_implemented,
    Conversions.TO_JPEG_XL: conversion_not_implemented,
    Conversions.TO_PNG: conversion_not_implemented,
    Conversions.TO_QOI: conversion_not_implemented,
    Conversions.TO_WEBP: conversion_not_implemented,

    Conversions.Old_SEUS_to_labPBR_1_3: conversion_not_implemented,
    Conversions.Old_Continuum_to_labPBR_1_3: conversion_not_implemented,
    Conversions.PPR_plus_Emissive_old_BSL_to_labPBR_1_3: conversion_not_implemented,
    Conversions.Grey_to_labPBR_1_3: conversion_not_implemented,
}


def convert_image_batch(
        conversions: list[Conversions],
        images: list[utils.ImageDict],
        factors: list[float]
) -> list[list[utils.ImageDict]]:

    result = []

    for conversion in conversions:
        conversion_function = ceonversion_functions[img_filter]
        if conversion_function is None:
            raise ValueError(f"Filter {conversion.name} (ID: {conversion}) is not implemented")

        result.append([
            {
                "images": [conversion_function(image["images"][0], factor) for factor in factors],
                "is_animated": image.get("is_animated"),
                "animation_spacing": image.get("animation_spacing"),
            } for image in images
        ])

    return result


def filter_image(
        conversion: Conversions,
        image: utils.ImageDict,
        factor: float
) -> utils.ImageDict:
    return convert_image_batch([conversion], [image], [factor])[0][0]
