# coding=utf-8
import os
import PIL.Image
import subprocess

from FidelityFX_CLI.wrapper import get_virtual_drive_letter
from scaling.utils import ConfigPlus
from termcolor import colored


def fsr_scale(frames: list[PIL.Image.Image], factor: float, _: ConfigPlus) -> list[PIL.Image.Image]:
    factor_check(factor, "FSR")
    options = f"-Scale {factor}x {factor}x -Mode EASU"
    return cli_process_frames(frames, options)


def cas_scale(frames: list[PIL.Image.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image.Image]:
    factor_check(factor, "CAS")
    options = f"-Scale {factor}x {factor}x -Sharpness {config_plus["sharpness"]} -Mode CAS"
    return cli_process_frames(frames, options)


def factor_check(factor: float, name: str) -> None:
    if factor > 2:
        print(
            colored(
                f"WARNING: Scaling with {name} by factor of {factor} is not supported, result might be blurry!",
                'yellow'
            )
        )


def cli_process_frames(frames: list[PIL.Image.Image], options: str) -> list[PIL.Image.Image]:
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
        processed_frame.load()  # used to load the image into memory and close the file
        processed_frames.append(processed_frame)

        # delete the input and output frames
        os.remove(input_frame)
        os.remove(output_frame)

    return processed_frames
