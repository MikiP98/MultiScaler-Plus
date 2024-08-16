# coding=utf-8
# File for user input functions

import loader
import plugin_manager
# import presets
import saving.saver as saver
import UI.console

from FidelityFX_CLI.wrapper import extinguish_the_drive, ignite_the_drive
from scaling.utils import Algorithms
from termcolor import colored
from utils import rainbowify

it = '\x1B[3m'
nr = '\x1B[0m'
# b = '\x1B[1m'

# TODO: Make this an enum
option_names = [
    "Scale images",
    "Apply filters to images",
    "Compress images",
    "Convert images",
    "Repeat the process",
    "Install plugins",
    "Raport",  # TODO: implement; It can say what MC textures are you missing, what normal maps are you missing, etc.
    "Exit"
]


class OptionNotImplementedError(NotImplementedError):
    pass


def main() -> None:
    UI.console.greetings()

    plugin_manager.read_plugin_file()
    if plugin_manager.load_plugins():
        print(colored("ERROR: Some plugins failed to load! Make sure they are installed!", "red"))

    while True:
        print("\nWhat would you like to do?")
        for i, option in enumerate(option_names, start=1):
            print(f"{i}. {option}")
        user_input = input(colored(f"\n{it}Enter your choice: ", "light_grey")).strip()
        if user_input not in options:
            print(colored("Invalid option! Please try again.", "red"))
            continue
        print()

        try:
            options[user_input]()
        except OptionNotImplementedError:
            print(colored("This option is not implemented yet!", "red"))
            continue
        else:
            print(rainbowify("ALL NICE & DONE!"))
            print()


def scale_images() -> None:
    import scaling.scaler_manager as scaler

    print("Scaling images!")

    load_config = UI.console.get_loader_config()
    scaler_config = UI.console.get_scaler_config()
    saver_config = UI.console.get_saver_config()

    # algorithms = presets.FullDownScalingTest.algorithms  # Test passed :)
    # factors = presets.FullDownScalingTest.scales

    # algorithms = presets.UpscaleNoCLITest.algorithms  # Test passed :)
    # factors = presets.UpscaleNoCLITest.scales

    # Skip test; Test passed :)
    # algorithms = [Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_AREA, Algorithms.CV2_INTER_CUBIC]
    # factors = [2]

    # algorithms = presets.FullUpscaleTest.algorithms  # Test passed :)
    # factors = presets.FullUpscaleTest.scales

    # CLI test; Test passed :)
    # algorithms = [Algorithms.FSR, Algorithms.CAS]
    # factors = [2]

    # user input start ----------------
    algorithms = UI.console.get_algorithms()
    factors = UI.console.get_factors()
    # user input end ----------------

    images, roots, file_names = loader.load_images(load_config)

    print(
        f"\nApplying {len(algorithms)} algorithm{'s' if len(algorithms) > 1 else ''} "
        f"to {len(images)} images...\n"
    )

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

    saver_config["factors"] = factors
    saver.save_img_list_multithreaded(scaled_images, roots, file_names, saver_config, algorithms)


def apply_filters() -> None:
    import filtering.filter_manager as filter_manager

    print("Applying filters!")

    load_config = UI.console.get_loader_config()
    saver_config = UI.console.get_saver_config()

    # user input start ----------------
    selected_filters_ids = UI.console.get_filters()
    factors = UI.console.get_factors()
    # user input end ----------------

    images, roots, file_names = loader.load_images(load_config)

    print(
        f"\nApplying {len(selected_filters_ids)} filter{'s' if len(selected_filters_ids) > 1 else ''} "
        f"to {len(images)} images...\n"
    )

    filtered_images = filter_manager.filter_image_batch(
        selected_filters_ids,
        images,
        factors  # No, it can't
    )
    print("Filtering is done!\n")

    saver_config["factors"] = factors
    saver.save_img_list_multithreaded(filtered_images, roots, file_names, saver_config, selected_filters_ids)


def compress_images() -> None:
    print("Compressing images!")
    raise OptionNotImplementedError


def convert_images() -> None:
    import converting.converter as converter

    print("Converting images!")

    while True:
        try:
            print("\nWhat would you like to do?")
            print("1. Change the image format (e.g. PNG to JPEG_XL)")
            print("2. Apply an advanced conversion (e.g. SEUS to labPBR)")
            user_input = int(input(colored(f"\n{it}Enter your choice: ", "light_grey")).strip())
            if user_input not in (1, 2):
                raise ValueError
        except ValueError:
            print(colored("Invalid option! Please try again.", "red"))
            print()
            continue
        else:
            break

    load_config = UI.console.get_loader_config()
    saver_config = UI.console.get_saver_config()

    if user_input == 1:
        images, roots, file_names = loader.load_images(load_config)
        saver.save_img_list_multithreaded([images], roots, file_names, saver_config, ['conversion'])
    elif user_input == 2:
        print(colored("INFO: Loader config override! merge_texture_extensions = true", "green"))
        load_config["merge_texture_extensions"] = True
        images, roots, file_names = loader.load_images(load_config)

        conversions = UI.console.get_conversions()
        converted_images = converter.convert_image_batch(conversions, images)

        saver.save_img_list_multithreaded(converted_images, roots, file_names, saver_config, ['n', 's', 'e'])


def repeat():
    print("Repeating the process!")
    raise OptionNotImplementedError


def attempt_to_install_plugins():
    print("Installing plugins!")
    plugin_manager.install_plugins()
    print("Plugins installed!")
    print("Attempting to load installed plugins...")
    plugin_manager.load_plugins()


options = {
    "1": scale_images,
    "2": apply_filters,
    "3": compress_images,
    "4": convert_images,
    "5": repeat,
    "6": attempt_to_install_plugins,
    "7": None,
    "8": UI.console.goodbye  # exit with a goodbye message
}

if __name__ == "__main__":
    main()
