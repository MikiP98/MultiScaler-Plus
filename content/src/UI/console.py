import aenum

import config
import json
import os
import sys

from converting import converter
from filtering import filter_manager
from functools import lru_cache
from loader import LoaderConfig
from scaling.utils import Algorithms, ConfigPlus as ScalerConfig
from saving.utils import AdvancedConfig as SaverConfig
from termcolor import colored
from typing import Callable, Type
from utils import rainbowify

it = '\x1B[3m'
nr = '\x1B[0m'
# b = '\x1B[1m'


def greetings():
    print((f"Hello! Welcome to the {rainbowify("MultiScaler+", bold=True)}{nr} !\n"
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


def get_factors() -> list[float]:
    while True:
        try:
            print("\nChoose the factors you want to apply to the selected filters\n"
                  "You can select multiple factors by separating them with a space or with `;`\n"
                  "(Factors should be floats)")
            user_input = input(colored(f"{it}Enter your choice: ", "light_grey")).strip()
            factors = list(
                float(
                    factor
                ) for factor in user_input.replace(',', '.').replace(';', ' ').split(' ')
            )
            # removes all 0 from factors as they do not make sense to have
            factors = list(filter((lambda factor: factor != 0), factors))  # TODO: consider tuple or set
            if len(factors) == 0:
                print("You have to chose at least 1 factor to scale with!")
                continue
        except ValueError:
            # If float() conversion fails
            print(colored("Invalid input! Please try again.", "red"))
        else:
            break
    return factors


def get_filters() -> list[filter_manager.Filters]:
    return get_selections(filter_manager.Filters, "filters")


def get_algorithms() -> list[Algorithms]:
    return get_selections(Algorithms, "algorithms")


def get_conversions() -> list[converter.Conversions]:
    return get_selections(converter.Conversions, "conversions")


def get_selections(
        enum: Type[Algorithms | converter.Conversions | filter_manager.Filters],
        plural_name: str
) -> list[Algorithms | converter.Conversions | filter_manager.Filters]:
    while True:
        try:
            print(f"\nChoose the {plural_name} you want to apply to the original images\n"
                  f"You can select multiple {plural_name} by separating them with a space or a coma\n"
                  f"You can also select a range of {plural_name} with '-'\n"
                  f"Available {plural_name} (select the IDs):")
            # Ignore the warning
            available_options = tuple(f"{option.value} - {option.name}" for option in enum)
            print(columnify(available_options))
            user_input = input(colored(f"{it}Enter your choice: ", "light_grey")).strip()

            # TODO: Consider moving it all to the separate function
            user_input_parts = user_input.replace(',', ' ').replace("  ", ' ').split(' ')
            selected_options_ids = [
                enum(option_id)
                for user_input_part in user_input_parts
                for option_id in interpret_input(user_input_part)
            ]
        except ValueError:
            # If int() conversion fails or the algorithm ID is not in the list
            print(colored("\nInvalid input! IDs should be natural numbers from the list. Please try again.", "red"))
        else:
            break
    return selected_options_ids


def interpret_input(user_input_part: str) -> list[int]:
    if '-' in user_input_part:
        start, end = user_input_part.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(user_input_part)]


# TODO: Improve indentation and add colours
def print_config(config: dict) -> None:
    for key, value in config.items():
        print(f"{colored(key + ':', "magenta")} {colored(value, "light_blue")}")
    print()


true_false_dict = {
    "y": True,
    "yes": True,
    "t": True,
    "true": True,
    "1": True,

    "n": False,
    "no": False,
    "f": False,
    "false": False,
    "0": False
}


loader_options_descriptions = {}


def get_loader_config() -> LoaderConfig:
    return get_config("loader", config.get_loader_config, loader_options_descriptions)


saver_options_descriptions = {}


def get_saver_config() -> SaverConfig:
    return get_config("saver", config.get_saver_config, saver_options_descriptions)


scaler_options_descriptions = {}


def get_scaler_config() -> ScalerConfig:
    return get_config("scaler", config.get_scaler_config, scaler_options_descriptions)


def get_config(
        name: str,
        get_config_function: Callable[[], tuple[ScalerConfig | SaverConfig | LoaderConfig, bool]],
        config_options_descriptions: dict
) -> ScalerConfig | SaverConfig | LoaderConfig:
    selected_config, default = get_config_function()
    config_source = "default" if default else "preset"

    while True:
        try:
            print()
            print(f"{name.capitalize()} config:")
            print_config(selected_config)
            print(f"Do you want to use the {config_source} {name} config?")
            user_input = input(colored(f"{it}Enter your choice (y/n): ", "light_grey")).strip().lower()
            use_selected_config = true_false_dict[user_input]
        except (ValueError, KeyError):
            print(colored("Invalid input! Please try again.", "red"))
        else:
            break

    if use_selected_config:
        return selected_config

    # TODO load user presets and display the choice of using one if present
    selected_config = create_new_config(selected_config, config_options_descriptions)

    while True:
        try:
            print_config(selected_config)
            print("Do you want to save this config as a preset?")
            user_input = input(colored(f"{it}Enter your choice (y/n): ", "light_grey")).strip().lower()
            save_selected_config = true_false_dict[user_input]
        except (ValueError, KeyError):
            print(colored("Invalid input! Please try again.", "red"))
        else:
            break

    if save_selected_config:
        # TODO load user presets and display the choice of using one if present

        while True:
            try:
                print_config(selected_config)
                print("Do you want to make this config a new default?")
                user_input = input(colored(f"{it}Enter your choice (y/n): ", "light_grey")).strip().lower()
                make_selected_config_default = true_false_dict[user_input]
            except (ValueError, KeyError):
                print(colored("Invalid input! Please try again.", "red"))
            else:
                break

        if make_selected_config_default:
            raise NotImplementedError

    return selected_config


def create_new_config(base_config: dict, options_descriptions: dict) -> dict:
    new_config = {}
    for key, value in base_config.items():
        while True:
            try:
                print()
                print(f"Option: {key}")
                print(f"Current value: {value}")

                if key in options_descriptions:
                    print(f"Description: {options_descriptions[key]}")

                new_value = input("Enter new value (empty to skip): ").strip().lower()
                if new_value:
                    new_config[key] = convert(new_value, type(value))
                else:
                    new_config[key] = value
            except (ValueError, KeyError):
                print(colored("Invalid input! Please try again.", "red"))
            else:
                break
    return new_config


def convert(value: str, type_: type) -> any:
    if type_ == bool:
        return true_false_dict[value]
    elif type_ == dict or type_ == list:
        return json.loads(value.replace("'", '"'))
    return type_(value)


if __name__ == '__main__':
    @aenum.unique
    class _enum(aenum.IntEnum):
        one = 1
        two = 2
        three = 3

    get_selections(_enum, "enums")
