# coding=utf-8
import filtering.filter_manager as filter_manager
import PIL.Image

from aenum import extend_enum


def plugin_filter_function(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    print(f"Applied plugin filter (id: {filter_manager.Filters.PLUGIN_FILTER}) with factor {factor}")
    return frames


# Ignore the warning
extend_enum(filter_manager.Filters, 'PLUGIN_FILTER', len(filter_manager.Filters) + 1)

filter_manager.filter_functions[filter_manager.Filters.PLUGIN_FILTER] = plugin_filter_function


if __name__ == '__main__':
    print("Plugin example is running!")

    print()
    function = filter_manager.filter_functions[filter_manager.Filters.PLUGIN_FILTER]
    function([], 1.0)
    print()

    print("Available filters:")
    for f in filter_manager.Filters:  # Ignore the warning
        print(f"{f.name} (id: {f})")
