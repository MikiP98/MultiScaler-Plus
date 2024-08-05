# coding=utf-8
import PIL.Image

from scaling.utils import ConfigPlus, correct_frame
from superxbr import superxbr  # Ignore the error, it works fine
from termcolor import colored


def scale(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    if factor < 2:
        print(
            colored(
                f"WARNING: Super-xBR does not support factors smaller then 2, factor: {factor}! "
                "Defaulting to fallback algorithm",
                'red'
            )
        )
    # Check if factor is not a power of 2
    factor_check = factor
    temp_factor = factor
    while factor_check > 2:
        if factor_check % 2 != 0:
            print(
                colored(
                    f"WARNING: Super-xBR does not support factor: {factor}! "
                    "Result might be blurry!",
                    'yellow'
                )
            )
            temp_factor = 2
            while temp_factor < factor:
                temp_factor *= 2
            break
        factor_check //= 2

    power = 1
    while 2 ** power != temp_factor:
        power += 1

    scaled_frames = []
    for frame in frames:
        original_size = frame.size
        # width, height = frame.size
        # output_width, output_height = round(width * temp_factor), round(height * temp_factor)

        frame = superxbr.scale(frame, power)

        scaled_frames.append(
            correct_frame(frame, original_size, factor, config_plus['high_quality_scale_back'])
        )
    return scaled_frames
