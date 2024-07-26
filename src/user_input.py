# coding=utf-8
# File for user input functions

from termcolor import colored
from utils import rainbowify


it = '\x1B[3m'
nr = '\x1B[0m'


def greetings():
    print((f"Hello! Welcome to the {rainbowify("MultiScaler+")} !\n"
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

        try:
            options[user_input]()
        except KeyError:
            print(colored("Invalid option! Please try again.", "red"))
        except NotImplementedError:
            print(colored("This feature is not implemented yet!", "red"))
        except Exception as e:
            print(colored(f"An error occurred: {e}", "red"))
        else:
            print(rainbowify("ALL NICE & DONE!"))

        print()


def scale_images():
    print("Scaling images!")
    raise NotImplementedError


def apply_filters():
    print("Applying filters!")
    raise NotImplementedError


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
    "5": goodbye
}

if __name__ == "__main__":
    main()
