# coding=utf-8
import PIL.Image
import saving.utils as utils

from UI.console import true_false_dict
from saving.utils import Compression
from termcolor import colored

it = '\x1B[3m'


def save(image: PIL.Image.Image, path: str, _: Compression, sort_by_file_extension: bool):
    path = utils.sort_by_file_extension(path, sort_by_file_extension, "PNG")

    while True:
        file_extension = input(colored(f"{it}Enter the file extension: ", "light_grey")).strip()

        file_path = path + file_extension.replace('.', '')

        while True:
            transparency = input(colored(f"{it}Does this format support transparency? (y/n): ", "light_grey")).strip()
            if transparency in true_false_dict:
                break
        if not true_false_dict[transparency]:
            image = image.convert("RGB")

        input_flags = input(colored(f"{it}Enter the saving flags separated by `;`: ", "light_grey")).strip()
        input_flags = input_flags.split(';')
        for i, flag in enumerate(input_flags):
            input_flags[i] = flag.strip()

        flags: list[tuple[str, str]] = []
        for i, flag in enumerate(input_flags):
            if len(flag) > 0:
                key, value = flag.split('=')

                key = key.strip()
                value = value.strip()

                flags.append((key, value))

        flags: dict[str, str] = dict(flags)

        try:
            image.save(file_path, **flags)
        except ValueError:
            print(colored("ERROR: Image saving has failed! Unrecognized file format!\n\t Make sure that the file extension is correct or pass additional flag named `format`", "red"))
            user_input = input(colored(f"{it}Do you want to try again? (y/n): ", "light_grey")).strip()

            if true_false_dict[user_input]:
                continue
        except OSError as e:
            if len(str(e)) > 21:
                if e[:21] == "cannot write mode RGBA":
                    print(colored(f"ERROR: {str(e).capitalize()}; Please try again", "red"))
                    continue
            raise e
        else:
            break
