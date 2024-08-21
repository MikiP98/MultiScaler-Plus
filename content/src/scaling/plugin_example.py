# coding=utf-8
import scaling.scaler_manager as scaler_manager
import scaling.utils as scaling_utils
import PIL.Image

from aenum import extend_enum
from scaling.utils import ConfigPlus


def plugin_scaling_algorithm(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    print(f"Applied plugin filter (id: {scaling_utils.Algorithms.PLUGIN_ALGORITHM}) with factor {factor}")
    return frames


# Ignore the warning
extend_enum(scaling_utils.Algorithms, 'PLUGIN_ALGORITHM', len(scaling_utils.Algorithms) + 1)

scaler_manager.scaling_functions[scaling_utils.Algorithms.PLUGIN_ALGORITHM] = plugin_scaling_algorithm


if __name__ == '__main__':
    print("Plugin example is running!")

    print()
    function = scaler_manager.scaling_functions[scaling_utils.Algorithms.PLUGIN_ALGORITHM]
    function([], 1.0)
    print()

    print("Available filters:")
    for a in scaling_utils.Algorithms:  # Ignore the warning
        print(f"{a.name} (id: {a})")
