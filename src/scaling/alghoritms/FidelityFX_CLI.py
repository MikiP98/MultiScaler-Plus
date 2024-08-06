# coding=utf-8
# import saving.saver as saver
import subprocess
import PIL.Image
import time

from FidelityFX_CLI.wrapper import get_virtual_drive_letter
from scaling.utils import ConfigPlus
from termcolor import colored


cli_supported_formats = {"BMP", "PNG", "ICO", "JPG", "TIF", "GIF"}


# TODO: Create a temporal RAM storage drive to properly support all image formats and the saver
def fsr_scale(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    print(f"Virtual drive letter: {get_virtual_drive_letter()}")
    print("Waiting 5s")
    time.sleep(5)
    print("Done")
    if factor > 2:
        print(
            colored(
                f"WARNING: Scaling with FSR by factor of {factor} is not supported, result might be blurry!",
                'yellow'
            )
        )

    processed_frames = []

    script_path = './FidelityFX_CLI/FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
    options = f"-Scale {factor}x {factor}x -Mode EASU"

    path_prefix = f"{get_virtual_drive_letter()}:\\"
    for i, frame in enumerate(frames):
        input_frame = f"{path_prefix}frame_{i}.png"
        frame.save(input_frame)

        output_frame = f"{path_prefix}frame_{i}_FSR.png"

        command = f"{script_path} {options} {input_frame} {output_frame}"
        subprocess.run(command)

        processed_frames.append(PIL.Image.open(output_frame))

    return processed_frames

    # if config_plus is None:
    #     raise ValueError("config_plus is None! Cannot use CLI controlled algorithms without it!")
    # else:
    #     if 'relative_input_path_of_images' not in config_plus:
    #         raise ValueError("relative_input_path_of_images not in config_plus!")
    #     relative_input_path_of_images = config_plus['relative_input_path_of_images']
    #
    #     if 'relative_output_path_of_images' not in config_plus:
    #         relative_output_path_of_images = map(
    #             lambda x: x.replace('input', 'output'), relative_input_path_of_images
    #         )
    #         relative_output_path_of_images = map(
    #             lambda x: x.replace('.png', '_FSR.png'), relative_output_path_of_images
    #         )
    #     else:
    #         relative_output_path_of_images = config_plus['relative_output_path_of_images']
    #
    #     # change file name to include '_FSR' before the file extension
    #     # relative_output_path_of_images = map(
    #     #     lambda x: x.replace('.png', '_FSR.png'), relative_output_path_of_images
    #     # )
    #
    #     for relative_input_path, relative_output_path in zip(
    #             relative_input_path_of_images, relative_output_path_of_images
    #     ):
    #         print(f"Relative input path: {relative_input_path}")
    #         print(f"Relative output path: {relative_output_path}")
    #
    #         if factor > 2:
    #             print(
    #                 colored(
    #                     "WARNING: Scaling with FSR by factor of {factor} is not supported, "
    #                     "result might be blurry!",
    #                     'yellow'
    #                 )
    #             )
    #
    #         script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
    #         options = f"-Scale {factor}x {factor}x -Mode EASU"
    #         files = f"{relative_input_path} {relative_output_path}"
    #         command = f"{script_path} {options} {files}"
    #         subprocess.run(command)
    #         # for frame in image:
    #         #     script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
    #         #     options = f"-Scale {factor}x {factor}x -Mode EASU"
    #         #     files = (
    #         #         f"../input/{config_plus['input_image_relative_path']} "
    #         #         f"../output/{config_plus['input_image_relative_path']}"
    #         #     )
    #         #     command = f"{script_path} {options} {files}"
    #         #     subprocess.run(command)
    # return []


def cas_scale(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    frames_bytes: list[bytes] = encoder.encode_frames(frames, 'PNG')
    ...

    # if config_plus is None:
    #     raise ValueError("config_plus is None! Cannot use CLI controlled algorithms without it!")
    # else:
    #     if 'sharpness' not in config_plus:
    #         raise ValueError("sharpness not in config_plus!")
    #     sharpness = config_plus['sharpness']
    #
    #     if 'relative_input_path_of_images' not in config_plus:
    #         raise ValueError("relative_input_path_of_images not in config_plus!")
    #     relative_input_path_of_images = config_plus['relative_input_path_of_images']
    #
    #     if 'relative_output_path_of_images' not in config_plus:
    #         relative_output_path_of_images = map(
    #             lambda x: x.replace('input', 'output'), relative_input_path_of_images
    #         )
    #         relative_output_path_of_images = map(
    #             lambda x: x.replace('.png', '_CAS.png'), relative_output_path_of_images
    #         )
    #     else:
    #         relative_output_path_of_images = config_plus['relative_output_path_of_images']
    #
    #     # change file name to include '_CAS' before the file extension
    #     # relative_output_path_of_images = map(
    #     #     lambda x: x.replace('.png', '_CAS.png'), relative_output_path_of_images
    #     # )
    #
    #     for relative_input_path, relative_output_path in (
    #             zip(relative_input_path_of_images, relative_output_path_of_images)
    #     ):
    #         if factor > 2:
    #             print(
    #                 colored(
    #                     "WARNING: Scaling with CAS by factor of {factor} is not supported, "
    #                     "result might be blurry!",
    #                     'yellow'
    #                 )
    #             )
    #
    #         script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
    #         options = f"-Scale {factor}x {factor}x -Sharpness {sharpness} -Mode CAS"
    #         files = f"{relative_input_path} {relative_output_path}"
    #         command = f"{script_path} {options} {files}"
    #         subprocess.run(command)
    #         # for frame in image:
    #         #     script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
    #         #     options = f"-Scale {factor}x {factor}x -Sharpness {config_plus['sharpness']} -Mode CAS"
    #         #     files = (
    #         #         f"../input/{config_plus['input_image_relative_path']} "
    #         #         f"../output/{config_plus['input_image_relative_path']}"
    #         #     )
    #         #     command = f"{script_path} {options} {files}"
    #         #     subprocess.run(command)
    # return []
