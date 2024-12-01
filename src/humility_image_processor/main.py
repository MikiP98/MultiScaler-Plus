# coding=utf-8
import argparse
import os
import sys

from aenum import auto, IntEnum
from typing import Callable

from humility_image_processor.UI.Console.console_formatting import *
from humility_image_processor.UI.Console import Console_UI
from humility_image_processor.UI.GUI import GUI_Renderer
from humility_image_processor.Objects.Tasks import Tasks


def main_cli():
    main(shortcut_creation_available=True)


def main(shortcut_creation_available=False):
    # load configs
    # if windows: `C:\Users\Mikolaj\AppData\Roaming\humility-image-processor`
    # if linux: `$XDG_CONFIG_HOME/humility-image-processor` or if not set: `$HOME/.config/humility-image-processor`

    # arg parse
    # - GUI
    # - additional input files/folders "folder1 file1 'file 2' 'file3'"
    # - use_config preset_name
    # - use default config (default: false)  # skips config user questions
    # - task: process, export_config, import_config
    # - processing type: scale, filter, rotate, compress, convert, etc.
    # - Use Output folder for drag and drop files (default: true)
    # - Use Output folder for input files/folders from flags (default: true)
    # - Ignore input folder
    # - Processing ids (ids of processing tasks)
    # - Keep open after processing

    flag_parser = argparse.ArgumentParser(
        prog="Humility Image Processor -> main",
        description="Image manipulation and compression tool",
        epilog="Thanks for using Humility Image Processor!",
    )

    flag_parser.add_argument("-g", "--gui", action="store_true", help="Start with GUI")

    flag_parser.add_argument("-d", "--default-config", action="store_true", help="Use default config")
    flag_parser.add_argument("-c", "--config", type=str, help="Set config preset")

    flag_parser.add_argument("-i", "--ignore-input-folder", action="store_true", help="Ignore input folder")
    flag_parser.add_argument("-f", "--files", nargs="+", help="Additional input files/folders")
    flag_parser.add_argument("-o", "--output", type=str, help="Set output folder path")
    flag_parser.add_argument("-ofd", "--output-for-drag-and-drop", type=bool, default=False, help="Use output folder for drag and drop files")
    flag_parser.add_argument("-ofi", "--output-for-input", type=bool, default=True, help="Use output folder for input files/folders")

    flag_parser.add_argument("-t", "--task", type=str, help="Set task")  # add choices={}?
    flag_parser.add_argument("-p", "--processing-type", type=str, help="Set processing type")  # add choices={}?
    flag_parser.add_argument("-pid", "--processing-ids", nargs="+", help="Set processing ids")

    flag_parser.add_argument("-r", "--repeat-if-no-user-input", action="store_true", help="Repeat after finishing even if no user input is required")
    flag_parser.add_argument("-k", "--keep-open", action="store_true", help="Keep open after processing")

    args, unknown_args = flag_parser.parse_known_args()

    Console_UI.print_welcome_message()

    # start console or GUI (dearpygui)
    if args.gui:
        GUI_Renderer.render()
    else:
        last_task: Tasks = None
        while True:
            if args.task is None:
                task = Console_UI.get_task_to_execute()
            else:
                task = Tasks.get_task_from_string(args.task)
                if task is None:
                    raise ValueError(f"Invalid task: {args.task}")
                args.task = None

            if task == Tasks.REPEAT:
                task = last_task
            else:
                last_task = task

            task_function = task_dict.get(task)
            task_function(args=args, shortcut_creation_available=shortcut_creation_available)

            # Process drag and dropped images if there are any
            for dropped_file in unknown_args:
                if os.path.exists(dropped_file):
                    print(f"{dropped_file=}")

            if task == Tasks.QUIT:
                break

    if args.keep_open:
        input(ui("\nPress any key to exit..."))


def create_script_shortcut(*_, **kwargs):
    from win32com.client import Dispatch

    if kwargs["shortcut_creation_available"]:
        working_directory = os.getcwd()
        # print(f"{working_directory=}")

        script_path = sys.argv[0]
        # print(f"{script_path=}")

        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(os.path.join(working_directory, "humility-image-processor.lnk"))
        shortcut.Targetpath = script_path
        shortcut.WorkingDirectory = working_directory
        shortcut.save()

        print(f"\n{light_green}Shortcut created at `{os.path.join(working_directory, 'humility-image-processor.lnk')}`{reset}")

    else:
        print(f"{red}ERROR: Shortcut creation is not available. Please launch Humility Image Processor though the script. Run `HIP` command in the target directory{reset}")


def reset_flags(*_, **kwargs):
    agrs = kwargs["args"]
    agrs.gui = None
    agrs.default_config = None
    agrs.config = None
    agrs.ignore_input_folder = None
    agrs.files = None
    agrs.output = None
    agrs.output_for_drag_and_drop = None
    agrs.output_for_input = None
    agrs.task = None
    agrs.processing_type = None
    agrs.processing_ids = None
    agrs.repeat_if_no_user_input = None
    agrs.keep_open = None


task_dict: dict[Tasks, Callable[[...], None]] = {
    Tasks.PROCESS_IMAGES: lambda *_, **__: (_ for _ in ()).throw(NotImplementedError),
    Tasks.EXPORT_CONFIG: lambda *_, **__: (_ for _ in ()).throw(NotImplementedError),
    Tasks.IMPORT_CONFIG: lambda *_, **__: (_ for _ in ()).throw(NotImplementedError),
    Tasks.SHOW_HELP: lambda *_, **__: (_ for _ in ()).throw(NotImplementedError),
    Tasks.MANAGE_EXTENSIONS_AND_PLUGINS: lambda *_, **__: (_ for _ in ()).throw(NotImplementedError),
    Tasks.CREATE_SHORTCUT: create_script_shortcut,
    Tasks.RESET_FLAGS: reset_flags,
    Tasks.QUIT: Console_UI.print_goodbye_message
}


if __name__ == '__main__':
    main(shortcut_creation_available=False)
