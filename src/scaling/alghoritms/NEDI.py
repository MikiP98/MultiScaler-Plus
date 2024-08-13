# coding=utf-8
import numpy as np
import PIL.Image
import utils

from scaling.utils import ConfigPlus
from termcolor import colored
# EDI_predict is wierd, EDI_Downscale is nearest neighbor...
from scaling.Edge_Directed_Interpolation.edi import EDI_upscale
from scaling.utils import correct_frame_from_cv2


def scale(frames: list[PIL.Image.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image.Image]:
    if factor < 1:
        print(colored("WARNING: NEDI does not support downscaling! Skipping!", 'yellow'))
        return []

    if 'NEDI_m' not in config_plus:
        print(
            colored(
                "WARNING: NEDI_m (edge detection radius) is not in config_plus! Using default value '4'",
                'yellow'
            )
        )
        config_plus['NEDI_m'] = 4

    # If factor is not a whole number or is not a power of 2, print a warning TODO: check why this is here
    # if factor != int(factor) or factor > 6:
    #     print(
    #         colored(
    #             f"WARNING: Scaling by NEDI with factor {factor} is not supported, "
    #             "result might be blurry!",
    #             'yellow'
    #         )
    #     )

    temp_factor_repeat = 1
    while 2 ** temp_factor_repeat <= factor:
        temp_factor_repeat += 1

    scaled_frames = []
    for frame in frames:
        original_size = frame.size

        # frame = frame.convert('RGBA')
        frame = utils.pil_to_cv2(frame)
        channels = [frame[:, :, i] for i in range(frame.shape[2])]

        for _ in range(temp_factor_repeat):
            channels = [EDI_upscale(channel, config_plus['NEDI_m']) for channel in channels]

        frame = np.stack(channels, axis=2)

        scaled_frames.append(
            correct_frame_from_cv2(frame, original_size, factor, config_plus['high_quality_scale_back'])
        )
    return scaled_frames
