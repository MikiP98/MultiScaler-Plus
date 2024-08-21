import importlib
import os
import subprocess

from termcolor import colored


plugins: frozenset[tuple[str, str]] = frozenset()


def read_plugin_file() -> None:
    global plugins

    read_plugins = []
    with open(os.path.join("..", "..", "plugins.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            plugin = tuple(part.strip() for part in line.split("|"))
            if len(plugin) == 1:
                read_plugins.append((plugin[0], plugin[0]))
            else:
                read_plugins.append((plugin[0], plugin[1]))

    plugins = frozenset(read_plugins)


def load_plugins() -> bool:
    failed = False
    for _, plugin_import in plugins:
        try:
            importlib.import_module(plugin_import)
        except ImportError as e:
            print(colored(f'ERROR: Failed to load plugin {plugin_import}: {e}', 'red'))
            failed = True
    return failed


def install_plugins() -> None:
    for plugin_package, _ in plugins:
        command = f"pip install {plugin_package}"
        print(f"Installing {plugin_package}")
        subprocess.run(command)
