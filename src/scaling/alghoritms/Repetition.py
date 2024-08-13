# coding=utf-8
import numpy as np
import PIL.Image

from scaling.utils import ConfigPlus
from termcolor import colored


def scale(frames: list[PIL.Image.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image.Image]:
    if config_plus is None:
        print(colored("WARNING: config_plus is None! Creating empty config_plus!", 'yellow'))
        config_plus = {}
    if 'offset_x' not in config_plus:
        print(colored("WARNING: offset_x not in config_plus! Using default value: 0", 'yellow'))
        config_plus['offset_x'] = 0
    if 'offset_y' not in config_plus:
        print(colored("WARNING: offset_y not in config_plus! Using default value: 0", 'yellow'))
        config_plus['offset_y'] = 0

    scaled_frames = []
    for frame in frames:
        width, height = frame.size
        offset_x = round(config_plus['offset_x'] * width)
        offset_y = round(config_plus['offset_y'] * height)
        output_width, output_height = round(width * factor), round(height * factor)

        # TODO: Optimise using build-in functions
        # new_frame = PIL.Image.new(frame.mode, (output_width, output_height))
        # for x in range(output_width):
        #     for y in range(output_height):
        #         new_frame.putpixel(
        #             (x, y), frame.getpixel(((x + offset_x) % width, (y + offset_y) % height))
        #         )

        # TODO: benchmark this; above, below and itertools.cycle() versions
        # --- --- --- --- --- --- ---
        # Convert PIL image to numpy array
        frame_array = np.array(frame)

        # Create new array for the output image
        new_frame_array = np.zeros((output_height, output_width, frame_array.shape[2]), dtype=frame_array.dtype)

        # Calculate new pixel positions using numpy broadcasting
        x_indices = (np.arange(output_width) + offset_x) % width
        y_indices = (np.arange(output_height) + offset_y) % height

        # Use numpy advanced indexing to fill the new frame array
        new_frame_array[:, :] = frame_array[y_indices[:, None], x_indices]

        # Convert numpy array back to PIL image
        new_frame = PIL.Image.fromarray(new_frame_array, mode=frame.mode)
        # --- --- --- --- --- --- ---

        scaled_frames.append(new_frame)

    return scaled_frames
