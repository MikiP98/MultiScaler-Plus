# coding=utf-8
import PIL.Image

from scaling.utils import ConfigPlus


def scale(frames: list[PIL.Image.Image], factor: float, algorithm: PIL.Image.Resampling) -> list[PIL.Image.Image]:
    scaled_frames = []
    for frame in frames:
        width, height = frame.size
        output_width, output_height = round(width * factor), round(height * factor)

        scaled_frames.append(frame.resize((output_width, output_height), algorithm))

    return scaled_frames


def scale_nearest_neighbor(frames: list[PIL.Image.Image], factor: float, _: ConfigPlus) -> list[PIL.Image.Image]:
    return scale(frames, factor, PIL.Image.Resampling.NEAREST)


def scale_bilinear(frames: list[PIL.Image.Image], factor: float, _: ConfigPlus) -> list[PIL.Image.Image]:
    return scale(frames, factor, PIL.Image.Resampling.BILINEAR)


def scale_bicubic(frames: list[PIL.Image.Image], factor: float, _: ConfigPlus) -> list[PIL.Image.Image]:
    return scale(frames, factor, PIL.Image.Resampling.BICUBIC)


def scale_lanczos(frames: list[PIL.Image.Image], factor: float, _: ConfigPlus) -> list[PIL.Image.Image]:
    return scale(frames, factor, PIL.Image.Resampling.LANCZOS)


# TODO: Test HAMMING & BOX
def scale_hamming(frames: list[PIL.Image.Image], factor: float, _: ConfigPlus) -> list[PIL.Image.Image]:
    return scale(frames, factor, PIL.Image.Resampling.HAMMING)


def scale_box(frames: list[PIL.Image.Image], factor: float, _: ConfigPlus) -> list[PIL.Image.Image]:
    return scale(frames, factor, PIL.Image.Resampling.BOX)
