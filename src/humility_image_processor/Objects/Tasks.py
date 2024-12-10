from aenum import auto, IntEnum


class Tasks(IntEnum):
    PROCESS_IMAGES = auto()
    EXPORT_CONFIG = auto()
    IMPORT_CONFIG = auto()
    SHOW_HELP = auto()
    MANAGE_EXTENSIONS_AND_PLUGINS = auto()
    CREATE_SHORTCUT = auto()
    RESET_FLAGS = auto()
    REPEAT = auto()
    SHOW_IMAGES = auto()
    QUIT = auto()


    @staticmethod
    def get_task_from_string(task_string: str) -> Tasks | None:
        task_string = task_string.upper()
        if task_string in Tasks.__members__:
            return Tasks.__members__[task_string]

        elif task_string in aliases:
            return Tasks.__members__[aliases[task_string]]

        elif task_string.isnumeric():
            if int(task_string) in Tasks.__members__.values():
                return Tasks(int(task_string))

        return None


aliases = {
    "P": "PROCESS_IMAGES",
    "E": "EXPORT_CONFIG",
    "I": "IMPORT_CONFIG",
    "H": "SHOW_HELP",
    "M": "MANAGE_EXTENSIONS_AND_PLUGINS",
    "C": "CREATE_SHORTCUT",
    "RES": "RESET_FLAGS",
    "REP": "REPEAT",
    "Q": "QUIT"
}
