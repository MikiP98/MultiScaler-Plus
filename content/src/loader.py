# coding=utf-8

import os
import PIL.Image
import PIL.GifImagePlugin
import pillow_avif  # This is a PIL plugin for AVIF, is must be imported, but isn't directly used
import pillow_jxl  # This is a PIL plugin for JPEG XL, is must be imported, but isn't directly used
import shutil
import utils
import zipfile

from typing import TypedDict, Optional
from utils import (
    pil_fully_supported_formats_cache,
    pil_read_only_formats_cache,
    pil_write_only_formats_cache,
    pil_indentify_only_formats_cache,
    pngify
)


PIL.Image.MAX_IMAGE_PIXELS = 4_294_967_296  # 2^16 squared, a.k.a. 65536x65536 pixels or 4 GigaPixels
PIL.GifImagePlugin.LOADING_STRATEGY = PIL.GifImagePlugin.LoadingStrategy.RGB_ALWAYS


class LoaderConfig(TypedDict):
    clear_output_dir: bool

    copy_mcmeta: bool
    merge_texture_extensions: bool  # TODO: implement

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


nr = '\x1B[0m'
b = '\x1B[1m'


def load_images(config: LoaderConfig) -> tuple[list[utils.ImageDict], list[str], list[str]]:
    if config['clear_output_dir']:
        print(f"{b}Clearing the output directory{nr}")
        shutil.rmtree("../../output", ignore_errors=True)
        os.makedirs("../../output", exist_ok=True)

    images: list[utils.ImageDict] = []
    # last_texture_extension = ""
    roots: list[str] = []
    file_names: list[str] = []
    for root, dirs, files in os.walk("../../input"):
        # print(f"Length of images: {len(images)}")
        # print(f"Length of roots: {len(roots)}")
        # print(f"Length of files: {len(file_names)}")

        files_number = len(files)
        print(f"\n{b}Checking {files_number} files in {root} directory{nr}")

        for file in files:
            path = os.path.join(root, file)
            filename, extension = split_file_extension(file)
            # if config['merge_texture_extensions']:
            #     filename, texture_extension = split_texture_extension(filename)

            # print(f"\nFilename: {filename}")
            # print(f"Extension: {extension}")
            # print(f"Root: {root}")
            # print(f"Root[8:]: {root[9:]}")

            if extension in pil_fully_supported_formats_cache or extension in pil_read_only_formats_cache:
                image = PIL.Image.open(path)
                image = pngify(image)

                # if config['merge_texture_extensions'] and file_check(file_names, filename):
                #     print("Merging texture images")
                #     images[-1] = merge_texture_images(images[-1], last_texture_extension, image, texture_extension)
                #     print(f"New image length: {len(images[-1]['images'])}")
                # else:
                #     # TODO: Consider additional texture extension check
                #     images.append(image)
                #     if config['merge_texture_extensions']:
                #         last_texture_extension = texture_extension
                #     roots.append(root)
                #     file_names.append(filename)
                #     # print(f"Appended root: {root} and file: {file} to their respective lists")

                images.append(image)
                roots.append(root)
                file_names.append(filename)
                # print(f"Appended root: {root} and file: {file} to their respective lists")

            elif extension == "zip" or extension == "7z":
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

            elif extension == "mcmeta":
                if config['copy_mcmeta']:
                    print(f"MCMeta file: {path} is being copied to the output directory")
                    # output_dir = f"../output{root.lstrip('../input')}"
                    output_dir = f"../output{root[8:]}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    shutil.copy2(path, output_dir)
                else:
                    print(f"MCMeta file: {path} will be ignored, animated texture will be corrupted!")

                continue

            elif extension in pil_write_only_formats_cache or extension in pil_indentify_only_formats_cache:
                print(f"File: {path} is an recognized image format but is not supported :( (yet)")
                try:
                    PIL.Image.open(path)  # Open the image to display Pillow's error message
                finally:
                    pass

                continue
            else:
                print(f"File: {path} is not supported, unrecognized file extension '{extension}'")
                continue

            print(f"Loaded: {path}")

    return images, roots, file_names


# list["_n", "_s", {else} ...]
def load_textures(config: LoaderConfig) -> tuple[list[list[utils.ImageDict | None]], list[str], list[str]]:
    if config['clear_output_dir']:
        print(f"{b}Clearing the output directory{nr}")
        shutil.rmtree("../output", ignore_errors=True)
        os.makedirs("../output")

    images: list[list[utils.ImageDict | None]] = []
    last_filename = ''
    roots: list[str] = []
    file_names: list[str] = []
    for root, dirs, files in os.walk("../input"):
        # print(f"Length of images: {len(images)}")
        # print(f"Length of roots: {len(roots)}")
        # print(f"Length of files: {len(file_names)}")

        files_number = len(files)
        print(f"\n{b}Checking {files_number} files in {root} directory{nr}")

        for file in files:
            path = os.path.join(root, file)
            filename, extension = split_file_extension(file)

            filename, texture_extension = split_texture_extension(filename)

            # print(f"\nFilename: {filename}")
            # print(f"Extension: {extension}")
            # print(f"Root: {root}")
            # print(f"Root[8:]: {root[9:]}")

            if extension in pil_fully_supported_formats_cache or extension in pil_read_only_formats_cache:
                image: PIL.Image.Image = PIL.Image.open(path)
                image: utils.ImageDict = pngify(image)

                if filename != last_filename:
                    image_to_insert: list[utils.ImageDict | None] = [None] * 2

                    if texture_extension == 'n':
                        image_to_insert[0] = image
                    elif texture_extension == 's':
                        image_to_insert[1] = image
                    else:
                        image_to_insert.append(image)

                    images.append(image_to_insert)
                    roots.append(root)
                    file_names.append(filename)
                    last_filename = filename

                else:
                    if texture_extension == 'n':
                        images[-1][0] = image
                    elif texture_extension == 's':
                        images[-1][1] = image
                    else:
                        images[-1].append(image)

                # print(f"Appended root: {root} and file: {file} to their respective lists")

            elif extension == "zip" or extension == "7z":
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

            elif extension == "mcmeta":
                if config['copy_mcmeta']:
                    print(f"MCMeta file: {path} is being copied to the output directory")
                    # output_dir = f"../output{root.lstrip('../input')}"
                    output_dir = f"../output{root[8:]}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    shutil.copy2(path, output_dir)
                else:
                    print(f"MCMeta file: {path} will be ignored, animated texture will be corrupted!")

                continue

            elif extension in pil_write_only_formats_cache or extension in pil_indentify_only_formats_cache:
                print(f"File: {path} is an recognized image format but is not supported :( (yet)")
                try:
                    PIL.Image.open(path)  # Open the image to display Pillow's error message
                finally:
                    pass

                continue
            else:
                print(f"File: {path} is not supported, unrecognized file extension '{extension}'")
                continue

            print(f"Loaded: {path}")

    return images, roots, file_names


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