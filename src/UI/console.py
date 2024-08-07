import config
import os
import sys

from filtering import filter_manager
from functools import lru_cache
from scaling.utils import Algorithms
from termcolor import colored
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
                  "You can select multiple factors by separating them with a space or coma\n"
                  "(Factors should be floats)")
            user_input = input(colored(f"{it}Enter your choice: ", "light_grey")).strip()
            factors = list(float(factor) for factor in user_input.replace(',', ' ').replace("  ", ' ').split(" "))
        except ValueError:
            # If float() conversion fails
            print(colored("Invalid input! Please try again.", "red"))
        else:
            break
    return factors


def get_filters() -> list[filter_manager.Filters]:
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
                raise ValueError
        except ValueError:
            # If int() conversion fails or the filter ID is not in the list
            print(colored("Invalid input! IDs should be natural numbers from the list. Please try again.", "red"))
        else:
            break
    return selected_filters_ids


def get_algorithms() -> list[Algorithms]:
    while True:
        try:
            print("\nChoose the scaling algorithms you want to apply to the original images\n"
                  "You can select multiple algorithms by separating them with a space or coma\n"
                  "Available algorithms (select the IDs):")
            # Ignore the warning
            available_algorithms = tuple(f"{algorithm.value} - {algorithm.name}" for algorithm in Algorithms)
            print(columnify(available_algorithms))
            user_input = input(colored(f"{it}Enter your choice: ", "light_grey")).strip()
            selected_algorithms_ids = list(
                Algorithms(
                    int(algorithm_id)
                ) for algorithm_id in user_input.replace(',', ' ').replace("  ", ' ').split(' ')
            )
            if any(algorithm_id not in Algorithms for algorithm_id in selected_algorithms_ids):
                raise ValueError
        except ValueError:
            # If int() conversion fails or the algorithm ID is not in the list
            print(colored("Invalid input! IDs should be natural numbers from the list. Please try again.", "red"))
        else:
            break
    return selected_algorithms_ids
