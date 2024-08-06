# coding=utf-8
# File for user input functions

import config
import loader
import os
import presets
import saving.saver as saver
import sys

from FidelityFX_CLI.wrapper import extinguish_the_drive, ignite_the_drive
from functools import lru_cache
from scaling.utils import Algorithms
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
    print(colored("Goodbye!\nHave a nice day!\n", "green"))
    exit()


@lru_cache(maxsize=1)
def columnify(elements: tuple) -> str:
    result = ""
    max_columns = 4

    max_length = max([len(element) for element in elements])
    # print(f"Max length: {max_length}")
    margin_right = 2
    tab_spaces = 2

    # Get the size of the terminal
    if sys.stdout.isatty():
        terminal_columns = os.get_terminal_size().columns
        # print(f"Terminal columns: {terminal_columns}")
        columns = max_columns

        def calc_row_length() -> int:
            return len("\t".expandtabs(tab_spaces)) + (
                    max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2
            ) * columns - 1

        while terminal_columns < calc_row_length() and columns > 1:
            # print(f"Calculated row length: {calc_row_length()}")
            columns -= 1
    else:
        columns = 3
    # print(f"Final row length: {calc_row_length()}")
    # print(f"Final column count: {columns}")

    overflow = len(elements) % columns
    full_count = len(elements) - overflow

    for i in range(0, full_count, columns):
        result += "\t".expandtabs(tab_spaces)
        if i < len(elements):
            result += " | ".join(
                [f"\t{elements[i + j]:<{max_length + margin_right}}".expandtabs(tab_spaces) for j in range(columns)]
            )
        result += "\n"
    result += "\t".expandtabs(tab_spaces)
    result += " | ".join(
        [
            f"\t{elements[k]:<{max_length + margin_right}}"
            .expandtabs(tab_spaces) for k in range(full_count, overflow + full_count)
        ]
    )
    result += "\n"

    return result


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
        if user_input not in options:
            print(colored("Invalid option! Please try again.", "red"))
            continue
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
    import scaling.scaler_manager as scaler

    print("Scaling images!")

    scaler_config, _ = config.get_scaler_config()

    load_config, _ = config.get_loader_config()

    # algorithms = presets.FullDownScalingTest.algorithms  # Test passed :)
    # factors = presets.FullDownScalingTest.scales

    # algorithms = presets.UpscaleNoCLITest.algorithms  # Test passed :)
    # factors = presets.UpscaleNoCLITest.scales

    # Skip test; Test passed :)
    # algorithms = [Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_AREA, Algorithms.CV2_INTER_CUBIC]
    # factors = [2]

    algorithms = presets.FullUpscaleTest.algorithms  # Test passed :)
    factors = presets.FullUpscaleTest.scales

    # CLI test; Test passed :)
    # algorithms = [Algorithms.FSR, Algorithms.CAS]
    # factors = [2]

    images, roots, file_names = loader.load_images(load_config)
    print(f"\nLoaded {len(images)} images")

    print("Processing images...\n")
    if Algorithms.FSR in algorithms or Algorithms.CAS in algorithms:
        max_pixel_count = max(sum(frame.size[0] * frame.size[0] for frame in image["images"][0]) for image in images)
        try:
            ignite_the_drive(max_pixel_count, max(factors))
            scaled_images = scaler.scale_image_batch(
                algorithms,
                images,
                factors,
                config_plus=scaler_config
            )
        finally:
            extinguish_the_drive()
    else:
        scaled_images = scaler.scale_image_batch(
            algorithms,
            images,
            factors,
            config_plus=scaler_config
        )

    saver_config, _ = config.get_saver_config()
    saver_config["factors"] = factors
    # saver_config["processing_methods"] = algorithms

    saver.save_img_list_multithreaded(scaled_images, roots, file_names, saver_config, algorithms)


def apply_filters():
    import filtering.filter_manager as filter_manager

    print("Applying filters!")

    # user input start ----------------
    while True:
        try:
            print("\nChoose the filters you want to apply to the original images\n"
                  "You can select multiple filters by separating them with a space or coma\n"
                  "Available filters (select the IDs):")
            # Ignore the warning
            available_filters = tuple(f"{filter.value} - {filter.name}" for filter in filter_manager.Filters)
            print(columnify(available_filters))
            user_input = input(colored(f"{it}Enter your choice: ", "light_grey")).strip()
            selected_filters_ids = list(
                filter_manager.Filters(
                    int(filter_id)
                ) for filter_id in user_input.replace(',', ' ').replace("  ", ' ').split(' ')
            )
            if any(filter_id not in filter_manager.Filters for filter_id in selected_filters_ids):
                raise ValueError("Invalid filter ID")
        except ValueError:
            print(colored("Invalid input! IDs should be natural numbers from the list. Please try again.", "red"))
            continue
        else:
            break

    while True:
        try:
            print("\nChoose the factors you want to apply to the selected filters\n"
                  "You can select multiple factors by separating them with a space or coma\n"
                  "(Factors should be floats)")
            user_input = input(colored(f"{it}Enter your choice: ", "light_grey")).strip()
            factors = list(float(factor) for factor in user_input.replace(',', ' ').replace("  ", ' ').split(" "))
        except ValueError:
            print(colored("Invalid input! Please try again.", "red"))
            continue
        else:
            break

    # user input end ----------------

    load_config, default = config.get_loader_config()
    images, roots, file_names = loader.load_images(load_config)
    # print(f"\nLoaded {len(images)} images")
    # print("Processing images...\n")

    # No, it can't
    print(
        f"\nApplying {len(selected_filters_ids)} filter{'s' if len(selected_filters_ids) > 1 else ''} "
        f"to {len(images)} images\n"
    )
    # factors = [0.4]
    filtered_images = filter_manager.filter_image_batch(
        selected_filters_ids,
        images,
        factors  # No, it can't
    )
    print("Filtering is done!\n")

    saver_config, _ = config.get_saver_config()
    saver_config["factors"] = factors

    saver.save_img_list_multithreaded(filtered_images, roots, file_names, saver_config, selected_filters_ids)


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
