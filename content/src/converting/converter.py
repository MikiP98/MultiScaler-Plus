# coding=utf-8
import PIL.Image
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


conversion_functions: dict[
    auto,
    Callable[
        [tuple[list[PIL.Image.Image] | None, list[PIL.Image.Image] | None]],
        tuple[list[PIL.Image.Image] | None, list[PIL.Image.Image] | None]
    ]
] = {
    Conversions.Old_SEUS_to_labPBR_1_3: conversion_not_implemented,
    Conversions.Old_Continuum_to_labPBR_1_3: convert_from_old_continuum,
    Conversions.PPR_plus_Emissive_old_BSL_to_labPBR_1_3: conversion_not_implemented,
    Conversions.Grey_to_labPBR_1_3: conversion_not_implemented,
}


def convert_image_batch(
        conversions: list[Conversions],
        texture_sets: list[tuple[utils.ImageDict | None, utils.ImageDict | None]]
) -> list[list[tuple[utils.ImageDict | None, utils.ImageDict | None]]]:
    # texture_sets contains a list of texture sets
    # each texture set should contain textures (`_n`, `_s`, `{else} ...`)

    result = []

    for conversion in conversions:
        conversion_function = conversion_functions[conversion]
        # if conversion_function is None:
        #     raise ValueError(f"Filter {conversion.name} (ID: {conversion}) is not implemented")

        conversion_result = []
        for texture_set in texture_sets:
            n, s = texture_set

            if n is None:
                n_frames = None
            else:
                n_frames = n['images'][0]

            if s is None:
                s_frames = None
            else:
                s_frames = s['images'][0]

            new_n, new_s = conversion_function((n_frames, s_frames))

            if new_n is None:
                n_dict = None
            else:
                n_dict = {
                    "images": [new_n],
                    "is_animated": n.get("is_animated"),
                    "animation_spacing": n.get("is_animated")
                }

            if new_s is None:
                s_dict = None
            else:
                s_dict = {
                    "images": [new_s],
                    "is_animated": s.get("is_animated"),
                    "animation_spacing": s.get("is_animated")
                }

            conversion_result.append((n_dict, s_dict))

        result.append(conversion_result)

    return result


def convert_image(
        conversion: Conversions,
        texture_set: tuple[utils.ImageDict | None, utils.ImageDict | None]
) -> tuple[utils.ImageDict | None, utils.ImageDict | None]:
    return convert_image_batch([conversion], [texture_set])[0][0]
