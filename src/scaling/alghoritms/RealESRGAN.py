# coding=utf-8
import PIL.Image
import torch

from RealESRGAN import RealESRGAN
from scaling.utils import ConfigPlus, correct_frame
from termcolor import colored


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def scale(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    if factor < 1:
        print(
            colored(
                "RealESRGAN AI does not support downscaling!; "
                "Skipping!",
                'red'
            )
        )
        return []

    scaled_frames = []
    for frame in frames:
        original_size = frame.size
        frame = frame.convert('RGB')

        # If factor is not a whole number or is greater than 6, print a warning
        if factor not in (1, 2, 4, 8):
            print(
                colored(
                    "WARNING: Scaling by RealESRGAN with factor {factor} is not supported, "
                    "result might be blurry!",
                    'yellow'
                )
            )

        current_scale = 1
        while current_scale < factor:
            temp_factor = 8
            while current_scale * temp_factor >= factor:
                temp_factor //= 2
            temp_factor = min(temp_factor * 2, 8)

            model = RealESRGAN(device, scale=temp_factor)
            model.load_weights(f'weights/RealESRGAN_x{temp_factor}.pth')  # , download=True
            frame = model.predict(frame)

            current_scale *= temp_factor

        scaled_frames.append(
            correct_frame(frame, original_size, factor, config_plus['high_quality_scale_back'])
        )
    return scaled_frames
