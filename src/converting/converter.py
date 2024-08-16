# coding=utf-8
import utils

from aenum import auto, IntEnum, unique
from converting.converters.labPBR_1_3 import convert_from_old_continuum
from typing import Callable


@unique
class Conversions(IntEnum):
    Old_SEUS_to_labPBR_1_3 = auto()
    Old_Continuum_to_labPBR_1_3 = auto()
    PPR_plus_Emissive_old_BSL_to_labPBR_1_3 = auto()
    Grey_to_labPBR_1_3 = auto()  # (most likely won't be great)


def conversion_not_implemented(_: utils.ImageDict) -> utils.ImageDict:
    raise NotImplementedError("Conversion not implemented")


conversion_functions: dict[auto, Callable[[utils.ImageDict], utils.ImageDict]] = {
    Conversions.Old_SEUS_to_labPBR_1_3: conversion_not_implemented,
    Conversions.Old_Continuum_to_labPBR_1_3: convert_from_old_continuum,
    Conversions.PPR_plus_Emissive_old_BSL_to_labPBR_1_3: conversion_not_implemented,
    Conversions.Grey_to_labPBR_1_3: conversion_not_implemented,
}


def convert_image_batch(
        conversions: list[Conversions],
        images: list[utils.ImageDict]
) -> list[list[utils.ImageDict]]:

    result = []

    for conversion in conversions:
        conversion_function = conversion_functions[conversion]
        # if conversion_function is None:
        #     raise ValueError(f"Filter {conversion.name} (ID: {conversion}) is not implemented")

        result.append([
            conversion_function(image) for image in images
        ])

    return result


def convert_image(
        conversion: Conversions,
        image: utils.ImageDict
) -> utils.ImageDict:
    return convert_image_batch([conversion], [image])[0][0]
