# coding=utf-8
import hqx
import PIL.Image

from scaling.utils import ConfigPlus, correct_frame
from termcolor import colored


def scale(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    allowed_factors = {2, 3, 4}
    if factor not in allowed_factors:
        if factor < 1:
            print(colored("WARNING: HQx does not support downscaling! Skipping!", 'yellow'))
            return []

        print(
            colored(
                f"WARNING: HQx does not support factor: {factor}! "
                f"Allowed factors: {allowed_factors}; Result might be blurry!",
                'yellow'
            )
        )

    max_allowed_factor = max(allowed_factors)
    scaled_frames = []
    for frame in frames:
        original_size = frame.size
        result = frame.convert('RGB')

        current_factor = 1
        while current_factor < factor:
            temp_factor = max_allowed_factor
            while current_factor * temp_factor >= factor:
                temp_factor -= 1
            while temp_factor not in allowed_factors:
                temp_factor += 1

            result = hqx.hqx_scale(result, temp_factor)
            current_factor *= temp_factor

        scaled_frames.append(
            correct_frame(result, original_size, factor, config_plus['high_quality_scale_back'])
        )
    return scaled_frames
