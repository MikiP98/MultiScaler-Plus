# coding=utf-8
from aenum import auto, IntEnum



class WarnSensitivity(IntEnum):
    WEAK_WARN = auto()
    WARN = auto()
    SEVERE_WARN = auto()


warn_sensitivity = WarnSensitivity.WEAK_WARN