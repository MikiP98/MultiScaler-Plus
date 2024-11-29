# coding=utf-8
import argparse
import os
import sys

from UI.GUI import GUI_Renderer


__version__ = '3.0.0'


def main_cli():
    main(shortcut_creation_available=True)


def main(shortcut_creation_available=False):
    # load configs
    # if windows: `C:\Users\Mikolaj\AppData\Roaming\humility-image-processor`
    # if linux: `$XDG_CONFIG_HOME/humility-image-processor` or if not set: `$HOME/.config/humility-image-processor`

    # Check for drag and drop files
    if len(sys.argv) > 1:
        dropped_files = sys.argv[1:]
        for dropped_file in dropped_files:
            print(f"{dropped_file=}")
        # input("Press any key to exit...")

    if shortcut_creation_available:
        create_script_shortcut()

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

    flag_parser.add_argument("-dc", "--default-config", action="store_true", help="Use default config")
    flag_parser.add_argument("-c", "--config", type=str, help="Set config preset")

    flag_parser.add_argument("-i", "--ignore-input-folder", action="store_true", help="Ignore input folder")
    flag_parser.add_argument("-f", "--files", nargs="+", help="Additional input files/folders")
    flag_parser.add_argument("-o", "--output", type=str, help="Set output folder path")
    flag_parser.add_argument("-ofd", "--output-for-drag-and-drop", type=bool, default=False, help="Use output folder for drag and drop files")
    flag_parser.add_argument("-ofi", "--output-for-input", type=bool, default=True, help="Use output folder for input files/folders")

    flag_parser.add_argument("-t", "--task", type=str, help="Set task")
    flag_parser.add_argument("-p", "--processing-type", type=str, help="Set processing type")
    flag_parser.add_argument("-pid", "--processing-ids", nargs="+", help="Set processing ids")

    flag_parser.add_argument("-k", "--keep-open", action="store_true", help="Keep open after processing")

    flag_parser.add_argument()

    args = flag_parser.parse_args()

    # start console or GUI (dearpygui)
    if args.gui:
        GUI_Renderer.render()
    else:
        pass

    # What do you want to do? (console: question; GUI: tabs)
    # - Process an images
    # - Export configs
    # - Import configs
    # - Show help
    # - Manage exensions and plugins
    # - Create a shortcut
    # - Quit

    purple = "\033[95m"
    reset = "\033[0m"
    if args.keep_open:
        input(f"\n{purple}Press any key to exit...{reset}")


def create_script_shortcut():
    from win32com.client import Dispatch

    # print(os.path.join(sys.exec_prefix), "Scripts")
    # print(os.path.realpath(__file__))

    working_directory = os.getcwd()
    print(f"{working_directory=}")

    script_path = sys.argv[0]
    print(f"{script_path=}")

    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(os.path.join(working_directory, "humility-image-processor.lnk"))
    shortcut.Targetpath = script_path
    shortcut.WorkingDirectory = working_directory
    shortcut.save()


if __name__ == '__main__':
    main(shortcut_creation_available=False)
