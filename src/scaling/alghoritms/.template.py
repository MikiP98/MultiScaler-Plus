# coding=utf-8
import PIL.Image

from scaling.utils import ConfigPlus


def scale(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    ...
