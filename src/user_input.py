# coding=utf-8
# File for user input functions

import loader
import os
import saver

from termcolor import colored
from utils import rainbowify


it = '\x1B[3m'
nr = '\x1B[0m'
# b = '\x1B[1m'


def greetings():
    print((f"Hello! Welcome to the {rainbowify("MultiScaler+", bold=True)} !\n"
           "Thanks for using this app :)\n"
           "Using this app you can:\n"
           "1. Scale images using various different algorithms!\n"
           "\t Starting from classic, through edge detection, ending with AI!\n"
           "2. Apply filters to images!\n"
           "\t Including, but not limited to, rare filter like:\n"
           "\t - Normal map strength\n"
           "\t - Auto normal map (and other textures)\n"
           "3. Compress your images and save them in multiple new and popular formats!\n"
           "\t Including:\n"
           "\t - PNG"
           "\t\t - WEBP\n"
           "\t - JPEGXL"
           "\t - AVIF\n"
           "\t - QOI"
           f"\t\t - {it}and more!{nr}\n"
           "4. Convert images to different standards like:\n"
           "\t - LabPBR\n"
           "\t - oldPBR\n"
           "\t - color spaces\n"
           f"\t - {it}and more!{nr}\n").expandtabs(2))


def goodbye():
    print(colored("Goodbye! Have a nice day!\n", "green"))
    exit()


option_names = [
    "Scale images",
    "Apply filters to images",
    "Compress images",
    "Convert images",
    "Exit"
]


def main():
    greetings()
    while True:
        print("\nWhat would you like to do?")
        for i, option in enumerate(option_names, start=1):
            print(f"{i}. {option}")
        user_input = input(colored(f"\n{it}Enter your choice: ", "light_grey")).strip()
        print()

        options[user_input]()
        print(rainbowify("ALL NICE & DONE!"))

        # try:
        #     options[user_input]()
        # except AttributeError:
        #     pass
        # except KeyError:
        #     print(colored("Invalid option! Please try again.", "red"))
        # except Exception as e:
        #     print(colored(f"An error occurred: {e}", "red"))
        # else:
        #     print(rainbowify("ALL NICE & DONE!"))

        print()


def scale_images():
    import scaler

    print("Scaling images!")
    # multiprocessing_level:
    # empty - no multiprocessing
    # 2 - to process different algorithms in parallel, (Note that this level also splits every subsequent workflow)
    # 3 - to save multiple images in parallel
    default_config = {
        'clear_output_directory': True,

        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,

        'sort_by_algorithm': False,
        'sort_by_scale': False,
        'sort_by_image': False,
        'sort_by_file_extension': -1,  # -1 - auto, 0 - no, 1 - yes

        'file_formats': {"WEBP"},
        'lossless_compression': True,
        'additional_lossless_compression': True,
        'quality': 95,

        'multiprocessing_levels': {},
        'max_processes': (2, 2, 2),
        'override_processes_count': False,
        # If True, max_processes will set the Exact number of processes, instead of the Maximum number of them

        'copy_mcmeta': True,
        'texture_outbound_protection': False,
        # prevents multi-face (in 1 image) textures to expand over current textures border
        'texture_inbound_protection': False,
        # TODO: Implement this, prevents multi-face (in 1 image) textures to not fully cover current textures border
        'texture_mask_mode': ('alpha', 'black'),
        # What should be used to make the mask, 1st is when alpha is present, 2nd when it is not  TODO: add more options
        'disallow_partial_transparency': False,
        'try_to_fix_texture_tiling': False,
        'tiling_fix_quality': 1.0,

        'sharpness': 0.5,
        'NEDI_m': 4,
        'offset_x': 0.5,
        'offset_y': 0.5
    }


def apply_filters():
    import filter

    print("Applying filters!")
    load_config = {
        'copy_mcmeta': True,
    }
    images, roots, file_names = loader.load_images(load_config)
    # print(f"\nLoaded {len(images)} images")
    # print("Processing images...\n")

    print(f"\nApplying filter to {len(images)} images\n")
    factors = [0.5]
    filtered_images = filter.filter_image_batch(
        filter.Filters.NORMAL_MAP_STRENGTH_EXPONENTIAL,
        images,
        factors
    )
    # print(f"\nFiltered {len(filtered_images)} images")
    # print(f"Filtered images have {len(filtered_images[0]['images'])} factors inside")
    # print(f"We have {len(roots)} roots and {len(file_names)} file names")

    saver_config = {
        "simple_config": {
            "formats": ["PNG"],
            "compressions": [
                {
                    "additional_lossless": True,
                    "lossless": True
                }
            ],
            "add_compression_to_name": False
        },

        "add_factor_to_name": False,
        "sort_by_factor": True,

        "factors": factors
    }

    for filtered_image, root, file_name in zip(filtered_images, roots, file_names):
        saver.save_image_pre_processor(
            filtered_image,
            os.path.join("..", "output", root[9:]),
            file_name,
            saver_config
        )


def compress_images():
    print("Compressing images!")
    raise NotImplementedError


def convert_images():
    print("Converting images!")
    raise NotImplementedError


options = {
    "1": scale_images,
    "2": apply_filters,
    "3": compress_images,
    "4": convert_images,
    "5": goodbye  # exit with a goodbye message
}

if __name__ == "__main__":
    main()
