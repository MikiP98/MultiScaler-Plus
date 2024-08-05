# coding=utf-8
import PIL.Image

from scaling.utils import ConfigPlus


def pil_scale(frames: list[PIL.Image], factor: float, algorithm: PIL.Image) -> list[PIL.Image]:
    scaled_frames = []
    for frame in frames:
        width, height = frame.size
        output_width, output_height = round(width * factor), round(height * factor)

        scaled_frames.append(frame.resize((output_width, output_height), algorithm))

    return scaled_frames


def pil_scale_nearest_neighbor(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    return pil_scale(frames, factor, PIL.Image.NEAREST)


def pil_scale_bilinear(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    return pil_scale(frames, factor, PIL.Image.BILINEAR)


def pil_scale_bicubic(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    return pil_scale(frames, factor, PIL.Image.BICUBIC)


def pil_scale_lanczos(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    return pil_scale(frames, factor, PIL.Image.LANCZOS)
