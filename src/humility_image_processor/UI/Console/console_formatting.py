# coding=utf-8

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

reset = '\033[0m'

bold = '\033[1m'
italic = '\033[3m'
underline = '\033[4m'

slow_blink = '\033[5m'
rapid_blink = '\033[6m'

reverse_color = '\033[7m'

black = '\033[30m'
red = '\033[31m'
green = '\033[32m'
yellow = '\033[33m'
blue = '\033[34m'
magenta = '\033[35m'
cyan = '\033[36m'
light_gray = '\033[37m'

gray = '\033[90m'
light_red = '\033[91m'
light_green = '\033[92m'
light_yellow = '\033[93m'
light_blue = '\033[94m'
light_magenta = '\033[95m'
light_cyan = '\033[96m'
white = '\033[97m'

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


def ui(text: str) -> str:
    # italic, fast blinking, gray
    return f"\033[3;6;90m{text}{reset}"


def rainbowify(text: str, make_bold=False, make_italic=False) -> str:
    colors = [red, yellow, green, cyan, blue, magenta]
    result = ""
    if make_bold:
        result += bold
    if make_italic:
        result += italic
    for i, char in enumerate(text):
        result += colors[i % len(colors)] + char
    return result + reset