# coding=utf-8
import os
import PIL.Image
import subprocess

from FidelityFX_CLI.wrapper import get_virtual_drive_letter
from scaling.utils import ConfigPlus
from termcolor import colored


cli_supported_formats = {"BMP", "PNG", "ICO", "JPG", "TIF", "GIF"}


# TODO: Create a temporal RAM storage drive to properly support all image formats and the saver
def fsr_scale(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    if factor > 2:
        print(
            colored(
                f"WARNING: Scaling with FSR by factor of {factor} is not supported, result might be blurry!",
                'yellow'
            )
        )

    options = f"-Scale {factor}x {factor}x -Mode EASU"

    return cli_process_frames(frames, options)


def cas_scale(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    if factor > 2:
        print(
            colored(
                f"WARNING: Scaling with CAS by factor of {factor} is not supported, result might be blurry!",
                'yellow'
            )
        )

    options = f"-Scale {factor}x {factor}x -Sharpness {config_plus["sharpness"]} -Mode CAS"

    return cli_process_frames(frames, options)


def cli_process_frames(frames: list[PIL.Image], options: str) -> PIL.Image:
    processed_frames = []

    script_path = './FidelityFX_CLI/FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
    path_prefix = f"{get_virtual_drive_letter()}:\\"

    for i, frame in enumerate(frames):
        input_frame = f"{path_prefix}frame_{i}.png"
        frame.save(input_frame)

        output_frame = f"{path_prefix}frame_{i}_PROCESSED.png"

        command = f"{script_path} {options} {input_frame} {output_frame}"
        subprocess.run(command)

        processed_frame = PIL.Image.open(output_frame)
        processed_frame.load()
        processed_frames.append(processed_frame)

        # delete the input and output frames
        os.remove(input_frame)
        os.remove(output_frame)

    return processed_frames
