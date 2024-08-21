# coding=utf-8
import numpy as np
import PIL.Image


def strength_linear(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    new_frames = []

    for frame in frames:
        image_data = np.asarray(frame)

        new_image_data = image_data.copy().astype('int16') - 128  # benchmarked
        # print(new_image_data[..., 0])
        new_image_data[..., 0] = new_image_data[..., 0] * factor  # Red channel
        new_image_data[..., 1] = new_image_data[..., 1] * factor  # Green channel

        new_image_data = new_image_data + 128

        new_frame = PIL.Image.fromarray(new_image_data.astype('uint8'))
        new_frames.append(new_frame)

    # print(f"Applied normal map strength filter with factor {factor}")
    return new_frames


def strength_exponential(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    new_frames = []

    for frame in frames:
        image_data = np.asarray(frame)

        # Convert from 0-255 to -128 to 127
        new_image_data = image_data.copy().astype('float32') / 255 * 2 - 1

        # Apply the exponential transformation
        new_image_data[..., 0] = (
                np.sign(new_image_data[..., 0]) * np.abs(new_image_data[..., 0]) ** (1 / factor))  # Red channel
        new_image_data[..., 1] = (
                np.sign(new_image_data[..., 1]) * np.abs(new_image_data[..., 1]) ** (1 / factor))  # Green channel

        # Convert back from -128 to 127 to 0-255
        new_image_data = ((new_image_data + 1) / 2 * 255).astype('uint8')

        new_frame = PIL.Image.fromarray(new_image_data)
        new_frames.append(new_frame)

    # print(f"Applied normal map strength filter with factor {factor}")
    return new_frames
