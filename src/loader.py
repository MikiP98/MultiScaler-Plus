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
        shutil.rmtree("../output", ignore_errors=True)
        os.makedirs("../output")

    images = []
    roots = []
    file_names = []
    for root, dirs, files in os.walk("../input"):
        # print(f"Length of images: {len(images)}")
        # print(f"Length of roots: {len(roots)}")
        # print(f"Length of files: {len(file_names)}")

        files_number = len(files)
        print(f"\n{b}Checking {files_number} files in {root} directory{nr}")

        for file in files:
            path = os.path.join(root, file)
            name_parts = file.split('.')
            extension = name_parts[-1].lower()
            filename = ''.join(name_parts[:-1])
            # print(f"\nFilename: {filename}")
            # print(f"Extension: {extension}")
            # print(f"Root: {root}")
            # print(f"Root[8:]: {root[9:]}")

            if extension in pil_fully_supported_formats_cache or extension in pil_read_only_formats_cache:
                image = PIL.Image.open(path)
                image = pngify(image)

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
