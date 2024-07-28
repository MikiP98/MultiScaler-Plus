import filter
import PIL.Image

from aenum import extend_enum


def plugin_filter_function(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    print(f"Applied plugin filter (id: {filter.Filters.PLUGIN_FILTER}) with factor {factor}")
    return frames


extend_enum(filter.Filters, 'PLUGIN_FILTER', len(filter.Filters) + 1)

filter.filter_functions[filter.Filters.PLUGIN_FILTER] = plugin_filter_function


if __name__ == '__main__':
    print("Plugin example is running!")

    print()
    function = filter.filter_functions[filter.Filters.PLUGIN_FILTER]
    function([], 1.0)
    print()

    print("Available filters:")
    for f in filter.Filters:
        print(f"{f.name} (id: {f})")