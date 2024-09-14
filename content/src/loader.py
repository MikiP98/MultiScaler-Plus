# coding=utf-8

import numpy as np
import os
import PIL.Image
import PIL.GifImagePlugin
import pillow_avif  # This is a PIL plugin for AVIF, is must be imported, but isn't directly used
import pillow_jxl  # This is a PIL plugin for JPEG XL, is must be imported, but isn't directly used
import re
import shutil
import utils
import zipfile

from termcolor import colored
from typing import cast, TypedDict, Optional
from utils import (
    pil_fully_supported_formats_cache,
    pil_read_only_formats_cache,
    pil_write_only_formats_cache,
    pil_indentify_only_formats_cache,
    pngify
)

# from typing import NamedTuple
# class ImageDataPassable(NamedTuple):
#     images: list[utils.ImageDict]
#     file_names: list[str]
#     roots_ids: list[int]
#     roots: list[str]


PIL.Image.MAX_IMAGE_PIXELS = 4_294_967_296  # 2^16 squared, a.k.a. 65536x65536 pixels or 4 GigaPixels
PIL.GifImagePlugin.LOADING_STRATEGY = PIL.GifImagePlugin.LoadingStrategy.RGB_ALWAYS


class LoaderConfig(TypedDict):
    clear_output_dir: bool

    copy_mcmeta: bool

    prefix_filter: Optional[str]  # endswith TODO: implement
    suffix_filter: Optional[str]  # startswith TODO: implement
    name_part_filter: Optional[str]  # in TODO: implement
    name_filter: Optional[str]  # exact match TODO: implement
    extension_filter: Optional[str]  # exact match TODO: implement

    prefix_blacklist: Optional[str]  # not endswith TODO: implement
    suffix_blacklist: Optional[str]  # not startswith TODO: implement
    name_part_blacklist: Optional[str]  # not in TODO: implement
    name_blacklist: Optional[str]  # not exact match TODO: implement
    extension_blacklist: Optional[str]  # not exact match TODO: implement


def safe_image_loading(path: str) -> PIL.Image:  # by [suneeta-mall](https://github.com/suneeta-mall)
    img = PIL.Image.open(path)

    # if img.mode != "RGB" and img.mode != "RGBA" and
    if img.mode == "I;16":

        bit_size = re.findall(r"\d+", img.mode)
        bit_size = int(bit_size[0]) if bit_size else 8
        if bit_size not in [8, 16, 32]:
            raise ValueError(f"Unsupported file type, supported bit size is {bit_size}")
        if bit_size != 8:
            print(colored(f"SEVERE WARN: Image will lose its extended bit depth!!!", "light_red"))
            max_value = 2**bit_size - 1
            img_arr = (np.array(img) / max_value) * 255.0
            img = PIL.Image.fromarray(img_arr.astype(np.uint8))
        return img.convert("L")

    else:
        return img


nr = '\x1B[0m'
b = '\x1B[1m'


def clear_output_directory(clear_output_dir: bool):
    if clear_output_dir:
        print(f"{b}Clearing the output directory{nr}")
        shutil.rmtree("../../output", ignore_errors=True)
        os.makedirs("../../output", exist_ok=True)


def load_images(config: LoaderConfig) -> tuple[list[utils.ImageDict], list[str], list[int], list[str]]:
    """

    :param config:
    :return: tuple containing lists of: images, file_names, roots ids, roots; of which first 3 have equal size
    """
    clear_output_directory(config['clear_output_dir'])

    images: list[utils.ImageDict] = []
    # last_texture_extension = ""
    file_names: list[str] = []
    roots_ids: list[int] = []
    roots: list[str] = []
    root_i = 0
    for root, dirs, files in os.walk("../../input"):
        roots.append(root)

        files_number = len(files)
        print(f"\n{b}Checking {files_number} files in {root} directory{nr}")

        for file in files:
            path = os.path.join(root, file)
            filename, extension = split_file_extension(file)

            if extension in pil_fully_supported_formats_cache or extension in pil_read_only_formats_cache:
                # image = PIL.Image.open(path)
                image = safe_image_loading(path)
                image = pngify(image)

                images.append(image)
                roots_ids.append(root_i)
                file_names.append(filename)
                # print(f"Appended root: {root} and file: {file} to their respective lists")

            elif extension == "zip" or extension == "7z":
                handle_zip(path)

            elif extension == "mcmeta":
                handle_mcmeta(root, path, config['copy_mcmeta'])
                continue

            elif extension in pil_write_only_formats_cache or extension in pil_indentify_only_formats_cache:
                handle_unreadable_images(path)
                continue
            else:
                print(f"File: {path} is not supported, unrecognized file extension '{extension}'")
                continue

            print(colored(f"Loaded: {path}", "green"))
        root_i += 1

    return images, file_names, roots_ids, roots


type TextureSet = tuple[utils.ImageDict | None, utils.ImageDict | None]


# list["_n", "_s", {else} ...]
def load_textures(
        config: LoaderConfig
) -> tuple[
    list[TextureSet],
    list[list[tuple[utils.ImageDict, str]]],
    list[str],
    list[int],
    list[str]
]:
    """

    :param config:
    :return: tuple of: Texture set, images with unknown texture extensions, file_names, roots_ids, roots
    """
    clear_output_directory(config['clear_output_dir'])

    texture_sets: list[list[utils.ImageDict | None]] = []
    other_images: list[list[tuple[utils.ImageDict, str]]] = []
    last_filename = ''
    file_names: list[str] = []
    roots_ids: list[int] = []
    roots: list[str] = []
    root_i = 0
    for root, dirs, files in os.walk("../../input"):
        # print(f"Length of images: {len(files)}")
        # print(f"Length of roots: {len(roots)}")
        # print(f"Length of files: {len(file_names)}")

        roots.append(root)

        files_number = len(files)
        print(f"\n{b}Checking {files_number} files in {root} directory{nr}")

        for file in files:
            path = os.path.join(root, file)
            filename, extension = split_file_extension(file)

            filename, texture_extension = split_texture_extension(filename)

            if extension in pil_fully_supported_formats_cache or extension in pil_read_only_formats_cache:
                # image: PIL.Image.Image = PIL.Image.open(path)
                image: PIL.Image.Image = safe_image_loading(path)
                image: utils.ImageDict = pngify(image)

                if filename != last_filename:
                    new_texture_set: list[utils.ImageDict | None] = [None] * 2
                    other_images.append([])

                    if texture_extension == 'n':
                        new_texture_set[0] = image
                    elif texture_extension == 's':
                        new_texture_set[1] = image
                    else:
                        other_images[-1].append((image, texture_extension))

                    texture_sets.append(new_texture_set)
                    roots_ids.append(root_i)
                    file_names.append(filename)
                    last_filename = filename

                else:
                    if texture_extension == 'n':
                        texture_sets[-1][0] = image
                    elif texture_extension == 's':
                        texture_sets[-1][1] = image
                    else:
                        other_images[-1].append((image, texture_extension))

                # print(f"Appended root: {root} and file: {file} to their respective lists")

            elif extension == "zip" or extension == "7z":
                handle_zip(path)

            elif extension == "mcmeta":
                handle_mcmeta(root, path, config['copy_mcmeta'])
                continue

            elif extension in pil_write_only_formats_cache or extension in pil_indentify_only_formats_cache:
                handle_unreadable_images(path)
                continue
            else:
                print(f"File: {path} is not supported, unrecognized file extension '{extension}'")
                continue

            print(f"Loaded: {path}")
        root_i += 1

    texture_sets: list[TextureSet] = [cast(TextureSet, tuple(texture_set)) for texture_set in texture_sets]
    return texture_sets, other_images, file_names, roots_ids, roots


def handle_zip(path: str):
    with zipfile.ZipFile(path) as zip_ref:
        # Get a list of all files and directories inside the zip file
        zip_contents = zip_ref.namelist()

        # Iterate through each file in the zip file
        for file_name in zip_contents:
            # Check if the current item is a file (not a directory)
            if not file_name.endswith('/'):  # Directories end with '/'
                # Read the content of the file
                with zip_ref.open(file_name) as zip_file:
                    # Process the content of the file
                    print(f"Contents of {file_name}:")
                    print(zip_file.read())  # You can perform any operation you want with the file content
                    print("---------------")

    raise NotImplementedError("Zip and 7z files are not supported yet")


def handle_mcmeta(root: str, path: str, copy_mcmeta: bool):
    if copy_mcmeta:
        print(f"MCMeta file: {path} is being copied to the output directory")
        # output_dir = f"../output{root.lstrip('../../input')}"
        output_dir = f"../output{root[11:]}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        shutil.copy2(path, output_dir)
    else:
        print(colored(f"WARNING: MCMeta file: {path} will be ignored, animated texture will be corrupted!", "yellow"))


def handle_unreadable_images(path: str):
    print(f"File: {path} is an recognized image format but is not supported :( (yet)")
    try:  # Open the image to display Pillow's error message
        PIL.Image.open(path)
    except Exception as e:
        print(e)


def split_string(string: str, separator: str) -> tuple[str, str]:
    parts = string.split(separator)
    return separator.join(parts[:-1]), parts[-1].lower()


def split_file_extension(file_name: str) -> tuple[str, str]:
    return split_string(file_name, '.')


def split_texture_extension(file_name: str) -> tuple[str, str]:
    return split_string(file_name, '_')


# the order should go like this: n, s, e
def merge_texture_images(
        original_image: utils.ImageDict,
        original_texture_extension: str,
        new_image: utils.ImageDict,
        new_texture_extension: str
) -> utils.ImageDict:
    new_images = []

    if original_texture_extension == 'n':
        new_images.append(*original_image['images'])
    elif original_texture_extension == 's':
        new_images.insert(1, *original_image['images'])
    else:
        raise ValueError(f"Unknown texture extension `{original_texture_extension}`?")

    if new_texture_extension == 's':
        new_images.append(*new_image['images'])
    elif new_texture_extension == 'e':
        new_images.insert(1, *new_image['images'])
    else:
        raise ValueError(f"Unknown texture extension `{new_texture_extension}`?")

    original_image['images'] = new_images

    return original_image


def file_check(file_names: list[str], filename: str) -> bool:
    if len(file_names) == 0:
        return False
    else:
        return file_names[-1] == filename
