# coding=utf-8
import PIL.Image
import super_image

from scaling.alghoritms.RealESRGAN import scale as real_esrgan_scale
from scaling.alghoritms.super_image import ai_scale as si_ai_scale
from scaling.utils import ConfigPlus, correct_frame
from termcolor import colored


def scale(frames: list[PIL.Image.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image.Image]:
    if factor < 1:
        print(colored("WARNING: HSDBTRE is an AI algorithm and does not support downscaling! Skipping!", 'yellow'))
        return []

    repeat = 1
    while 4 ** repeat < factor:
        repeat += 1
    # print(f"Repeats: {repeats}")

    scaled_frames = frames.copy()
    for _ in range(repeat):
        scaled_frames = si_ai_scale(scaled_frames, 2, True, {2}, super_image.DrlnModel, "eugenesiow/drln-bam")
        scaled_frames = real_esrgan_scale(scaled_frames, 2, config_plus)

    scaled_frames = [
        correct_frame(scaled_frame, frame.size, factor, config_plus['high_quality_scale_back'])
        for scaled_frame, frame in zip(scaled_frames, frames)
    ]

    return scaled_frames
