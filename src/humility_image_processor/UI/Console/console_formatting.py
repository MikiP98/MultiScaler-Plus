# coding=utf-8
from aenum import StrEnum

# https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
# https://i.sstatic.net/9UVnC.png

# Code	Effect	Note
# 0	Reset / Normal	all attributes off
# 1	Bold or increased intensity
# 2	Faint (decreased intensity)	Not widely supported.
# 3	Italic	Not widely supported. Sometimes treated as inverse.
# 4	Underline
# 5	Slow Blink	less than 150 per minute
# 6	Rapid Blink	MS-DOS ANSI.SYS; 150+ per minute; not widely supported
# 7	[[reverse video]]	swap foreground and background colors
# 8	Conceal	Not widely supported.
# 9	Crossed-out	Characters legible, but marked for deletion. Not widely supported.
# 10	Primary(default) font
# 11–19	Alternate font	Select alternate font n-10
# 20	Fraktur	hardly ever supported
# 21	Bold off or Double Underline	Bold off not widely supported; double underline hardly ever supported.
# 22	Normal color or intensity	Neither bold nor faint
# 23	Not italic, not Fraktur
# 24	Underline off	Not singly or doubly underlined
# 25	Blink off
# 27	Inverse off
# 28	Reveal	conceal off
# 29	Not crossed out
# 30–37	Set foreground color	See color table below
# 38	Set foreground color	Next arguments are 5;<n> or 2;<r>;<g>;<b>, see below
# 39	Default foreground color	implementation defined (according to standard)
# 40–47	Set background color	See color table below
# 48	Set background color	Next arguments are 5;<n> or 2;<r>;<g>;<b>, see below
# 49	Default background color	implementation defined (according to standard)
# 51	Framed
# 52	Encircled
# 53	Overlined
# 54	Not framed or encircled
# 55	Not overlined
# 60	ideogram underline	hardly ever supported
# 61	ideogram double underline	hardly ever supported
# 62	ideogram overline	hardly ever supported
# 63	ideogram double overline	hardly ever supported
# 64	ideogram stress marking	hardly ever supported
# 65	ideogram attributes off	reset the effects of all of 60-64
# 90–97	Set bright foreground color	aixterm (not in standard)
# 100–107	Set bright background color	aixterm (not in standard)

# print("\x1b[38;2;255;100;0mTRUECOLOR\x1b[0m\n")  # TODO: Redo `console_formatting.py`

reset = '\033[0m'

bold = '\033[1m'
italic = '\033[3m'
underline = '\033[4m'

slow_blink = '\033[5m'
rapid_blink = '\033[6m'

reverse_color = '\033[7m'


# Standard Background colors
bg_black = '\033[40m'
bg_red = '\033[41m'
bg_green = '\033[42m'
bg_yellow = '\033[43m'
bg_blue = '\033[44m'
bg_magenta = '\033[45m'
bg_cyan = '\033[46m'
bg_light_gray = '\033[47m'

bg_gray = '\033[100m'
bg_light_red = '\033[101m'
bg_light_green = '\033[102m'
bg_light_yellow = '\033[103m'
bg_light_blue = '\033[104m'
bg_light_magenta = '\033[105m'
bg_light_cyan = '\033[106m'
bg_white = '\033[107m'


# Standard Foreground colors
# black = '\033[30m'
# red = '\033[31m'
# green = '\033[32m'
# yellow = '\033[33m'
# blue = '\033[34m'
# magenta = '\033[35m'
# cyan = '\033[36m'
# light_gray = '\033[37m'
#
# gray = '\033[90m'
# light_red = '\033[91m'
# light_green = '\033[92m'
# light_yellow = '\033[93m'
# light_blue = '\033[94m'
# light_magenta = '\033[95m'
# light_cyan = '\033[96m'
# white = '\033[97m'


# Grayscale colors
black = '\033[38;2;0;0;0m'
dark_gray = '\033[38;2;64;64;64m'
gray = '\033[38;2;128;128;128m'
light_gray = '\033[38;2;192;192;192m'
white = '\033[38;2;255;255;255m'


# Colorful colors
dark_red = '\033[38;2;196;0;0m'  # Severe Errors
red = '\033[38;2;255;0;64m'  # Errors
# print(f"{dark_red}SEVERE ERROR: Lorem ipsum dolor sit amet!!!{reset}")
# print(f"{red}ERROR: Lorem ipsum dolor sit amet!{reset}\n")

orange = '\033[38;2;255;132;0m'  # Severe Warnings
yellow = '\033[38;2;220;220;0m'  # Warnings
pale_yellow = '\033[38;2;200;200;72m'  # Weak Warnings
# print(f"{orange}SEVERE WARNING: Lorem ipsum dolor sit amet{reset}")
# print(f"{yellow}WARNING: Lorem ipsum dolor sit amet{reset}")
# print(f"{pale_yellow}WARNING: Lorem ipsum dolor sit amet{reset}\n")

green = '\033[38;2;0;172;0m'
light_green = '\033[38;2;0;208;0m'
bright_green = '\033[38;2;64;255;96m'

blue = '\033[38;2;69;69;255m'
light_blue = '\033[38;2;128;128;255m'

purple = '\033[38;2;128;0;255m'

cyan = '\033[38;2;0;255;255m'
light_cyan = '\033[38;2;96;255;255m'
dark_cyan = '\033[38;2;0;148;255m'

magenta = '\033[38;2;232;84;255m'
light_magenta = '\033[38;2;255;128;255m'


# If high contrast is enabled
# reset += bg_black
# print(reset, end='')


class RainbowColors(StrEnum):
    # Pinkish red to yellow to limish green to sky blue to violet
    red = '\033[38;2;255;0;0m'
    orange = '\033[38;2;255;128;0m'
    yellow = '\033[38;2;196;196;0m'
    green = '\033[38;2;0;196;0m'
    blue = '\033[38;2;0;148;255m'
    indigo = '\033[38;2;64;72;255m'
    violet = '\033[38;2;110;8;255m'


def ui(text: str) -> str:
    # italic, fast blinking, gray
    return f"\033[3;6;90m{text}{reset}"


def severe_error(text: str) -> str:
    return f"{dark_red}SEVERE ERROR: {text}!!!{reset}"

def error(text: str) -> str:
    return f"{red}ERROR: {text}!{reset}"

def severe_warning(text: str) -> str:
    return f"{orange}SEVERE WARNING: {text}{reset}"

def warning(text: str) -> str:
    return f"{yellow}WARNING: {text}{reset}"


def rainbowify(text: str, make_bold=False, make_italic=False) -> str:
    colors = [
        RainbowColors.red,
        RainbowColors.orange,
        RainbowColors.yellow,
        RainbowColors.green,
        RainbowColors.blue,
        RainbowColors.indigo,
        RainbowColors.violet,

        RainbowColors.indigo,
        RainbowColors.blue,
        RainbowColors.green,
        RainbowColors.yellow,
        RainbowColors.orange
    ]
    result = ""
    if make_bold:
        result += bold
    if make_italic:
        result += italic
    for i, char in enumerate(text):
        result += colors[i % len(colors)] + char
    return result + reset


def colorize(text: str, red: int, green: int, blue: int, make_bold=False, make_italic=False) -> str:
    if make_bold:
        text = bold + text
    if make_italic:
        text = italic + text
    return f"\033[38;2;{red};{green};{blue}m{text}{reset}"


def bg_colorize(text: str, red: int, green: int, blue: int) -> str:
    return f"\033[48;2;{red};{green};{blue}m{text}{reset}"


def all_colorize(text: str, red: int, green: int, blue: int, bg_red: int, bg_green: int, bg_blue: int, make_bold=False, make_italic=False) -> str:
    if make_bold:
        text = bold + text
    if make_italic:
        text = italic + text
    return f"\033[38;2;{red};{green};{blue}m\033[48;2;{bg_red};{bg_green};{bg_blue}m{text}{reset}"
