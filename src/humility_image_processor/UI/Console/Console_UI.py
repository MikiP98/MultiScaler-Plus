# coding=utf-8
from humility_image_processor.__version__ import __version__
from humility_image_processor.UI.Console.console_formatting import *
from humility_image_processor.Objects.Tasks import Tasks


def print_welcome_message() -> None:
    print(
        f"{light_green}Welcome to `{rainbowify("Humility Image Processor", make_bold=True)}{light_green}` "
        f"{cyan}{italic}v{__version__}{reset}"
    )


def get_task_to_execute() -> str:
    # What do you want to do? (console: question; GUI: tabs)
    # - Process an images
    # - Export configs
    # - Import configs
    # - Show help
    # - Manage exensions and plugins
    # - Create a shortcut
    # - Reset flags
    # - Show images
    # - Quit
    while True:
        print(f"\n{bold}{light_cyan}What would you like to do?{reset}")
        for i, option in enumerate(Tasks.__members__.keys(), start=1):
            print(f"{bright_green}{i}{reset}. {italic}{light_magenta}{option}{reset}")
        user_input = input(ui(f"\nEnter your choice: ")).strip()
        task = Tasks.get_task_from_string(user_input)
        if task is not None:
            return task
        else:
            print(f"{red}Invalid option: {green}{italic}{bold}{user_input}{reset}{red}! Please try again{reset}")


def print_goodbye_message(**_) -> None:
    print(
        f"\n{green}Thank you for using `{rainbowify("Humility Image Processor", make_bold=True)}{green}`!\n"
        f"{light_blue}Have a georgeous day :){reset}"
    )
