# coding=utf-8
import PIL.Image
import xbrz  # See xBRZ scaling on Jira TODO: move this info to more appropriate place

from scaling.utils import ConfigPlus, correct_frame
from termcolor import colored


# TODO: Use RGB mode if the image is not RGBA
def scale(frames: list[PIL.Image.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image.Image]:
    if factor < 1:
        print(
            colored(
                f"ERROR: xBRZ does not support downscaling! Factor: {factor}; "
                f"Skipping!",
                'red'
            )
        )
        return []

    # If factor is not a whole number or is greater than 6, print a warning
    if factor != int(factor) or factor > 6:
        print(
            colored(
                f"WARNING: Scaling by xBRZ with factor {factor} "
                f"is not supported, result might be blurry!",
                'yellow'
            )
        )

    scaled_frames = []
    for frame in frames:
        original_size = frame.size

        frame = frame.convert('RGBA')

        current_scale = 1
        while current_scale < factor:
            temp_factor = 6
            while current_scale * temp_factor >= factor:
                temp_factor -= 1
            temp_factor = min(temp_factor + 1, 6)  # TODO: think if this can be changed, cause it looks wierd

            frame = xbrz.scale_pillow(frame, temp_factor)
            current_scale *= temp_factor

        scaled_frames.append(
            correct_frame(frame, original_size, factor, config_plus['high_quality_scale_back'])
        )

    # TODO: test this Copilot generated code
    # scaled_frames = []
    # for frame in frames:
    #     width, height = frame.size
    #     output_width, output_height = round(width * factor), round(height * factor)
    #
    #     frame_array = np.array(frame)
    #     new_frame_array = xbrz.scale(frame_array, output_width, output_height)
    #
    #     scaled_frames.append(PIL.Image.fromarray(new_frame_array))

    return scaled_frames
