# coding=utf-8

import cv2
import numpy as np
import os
import queue
import scaler
# import sys
import time
import timeit
import utils

from collections import deque
from functools import lru_cache
from PIL import Image
from pympler import asizeof
from utils import Algorithms
from utils import algorithm_to_string_dict
from utils import string_to_algorithm_dict


def warmup():
    num = 0
    for i in range(10_000_000):
        num += i
    # -----------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- END OF "warmup" ---------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------


def string_to_algorithm_match(string: str) -> Algorithms:
    match string.lower():
        case "cv2_area":
            return Algorithms.CV2_INTER_AREA
        case "cv2_bicubic":
            return Algorithms.CV2_INTER_CUBIC
        case "cv2_bilinear":
            return Algorithms.CV2_INTER_LINEAR
        case "cv2_lanczos":
            return Algorithms.CV2_INTER_LANCZOS4
        case "cv2_nearest":
            return Algorithms.CV2_INTER_NEAREST

        case "cv2_edsr":
            return Algorithms.CV2_EDSR
        case "cv2_espcn":
            return Algorithms.CV2_ESPCN
        case "cv2_fsrcnn":
            return Algorithms.CV2_FSRCNN
        case "cv2_lapsrn":
            return Algorithms.CV2_LapSRN

        case "pil_bicubic":
            return Algorithms.PIL_BICUBIC
        case "pil_bilinear":
            return Algorithms.PIL_BILINEAR
        case "pil_lanczos":
            return Algorithms.PIL_LANCZOS
        case "pil_nearest":
            return Algorithms.PIL_NEAREST_NEIGHBOR

        case "cas":
            return Algorithms.CAS
        case "fsr":
            return Algorithms.FSR
        case "real_esrgan":
            return Algorithms.RealESRGAN
        case "supir":
            return Algorithms.SUPIR
        case "xbrz":
            return Algorithms.xBRZ
        case _:
            raise ValueError("Algorithm not found")


def test_match_vs_dict(n=20_000_000, k=10):
    best_using_match = 0
    best_using_dict = 0
    worst_using_match = 0
    worst_using_dict = 0
    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        best_using_match += timeit.timeit(lambda: string_to_algorithm_match("cv2_bicubic"), number=n // k)
        best_using_dict += timeit.timeit(lambda: string_to_algorithm_dict["cv2_bicubic"], number=n // k)
        worst_using_match += timeit.timeit(lambda: string_to_algorithm_match("xbrz"), number=n // k)
        worst_using_dict += timeit.timeit(lambda: string_to_algorithm_dict["xbrz"], number=n // k)
    print()

    best_using_match = round(best_using_match / k, 4)
    best_using_dict = round(best_using_dict / k, 4)
    worst_using_match = round(worst_using_match / k, 4)
    worst_using_dict = round(worst_using_dict / k, 4)

    print(f"Best using match: {best_using_match}")
    print(f"Best using dict: {best_using_dict}")
    print(f"Worst using match: {worst_using_match}")
    print(f"Worst using dict: {worst_using_dict}")
    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ END OF "test_match_vs_dict" ------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------


def custom_any(iterable):
    for element in iterable:
        if element[3] < 255:
            return True
    return False


def custom_any_2(iterable):
    for element in iterable:
        if element[3] != 255:
            return True
    return False


def normal_any(iterable):
    return any(element[3] < 255 for element in iterable)


def extrema_any(image):
    extrema = image.getextrema()
    if extrema[3][0] < 255:
        return True
    return False


def numpy_any(np_array):
    return np.any(np_array[:, :, 3] < 255)


def numpy_any_2(np_array):
    return np.any(np_array[:, :, 3] != 255)


def numpy_conversion_any(image):
    np_array = utils.pil_to_cv2(image)
    return np.any(np_array[:, :, 3] < 255)


def numpy_conversion_any_safe(image):
    np_array = utils.pil_to_cv2(image)
    if np_array.shape[2] == 4:  # Check if the alpha channel is used
        return np.any(np_array[:, :, 3] < 255)
    return False


def numpy_conversion_any_safe_2(image):
    if image.mode == "RGBA":
        np_array = utils.pil_to_cv2(image)
        return np.any(np_array[:, :, 3] < 255)
    return False


def test_custom_any(n=10_000, k=10):
    image = Image.open("../input/NEAREST_NEIGHBOR_pixel-art_0.125x.png").convert("RGBA")
    image_array = list(image.getdata())
    numpy_image_array = utils.pil_to_cv2(image)

    custom_any_time = 0
    custom_any_2_time = 0
    normal_any_time = 0
    extrema_any_time = 0
    numpy_any_time = 0
    numpy_any_2_time = 0
    numpy_conversion_any_time = 0
    numpy_conversion_any_safe_time = 0
    numpy_conversion_any_safe_2_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        custom_any_time += timeit.timeit(lambda: custom_any(image_array), number=n // k)
        custom_any_2_time += timeit.timeit(lambda: custom_any_2(image_array), number=n // k)
        normal_any_time += timeit.timeit(lambda: normal_any(image_array), number=n // k)
        extrema_any_time += timeit.timeit(lambda: extrema_any(image), number=n // k)
        numpy_any_time += timeit.timeit(lambda: numpy_any(numpy_image_array), number=n // k)
        numpy_any_2_time += timeit.timeit(lambda: numpy_any_2(numpy_image_array), number=n // k)
        numpy_conversion_any_time += timeit.timeit(lambda: numpy_conversion_any(image), number=n // k)
        numpy_conversion_any_safe_time += timeit.timeit(lambda: numpy_conversion_any_safe(image), number=n // k)
        numpy_conversion_any_safe_2_time += timeit.timeit(lambda: numpy_conversion_any_safe_2(image), number=n // k)
    print()

    custom_any_time = round(custom_any_time / k, 4)
    custom_any_2_time = round(custom_any_2_time / k, 4)
    normal_any_time = round(normal_any_time / k, 4)
    extrema_any_time = round(extrema_any_time / k, 4)
    numpy_any_time = round(numpy_any_time / k, 4)
    numpy_any_2_time = round(numpy_any_2_time / k, 4)
    numpy_conversion_any_time = round(numpy_conversion_any_time / k, 4)
    numpy_conversion_any_safe_time = round(numpy_conversion_any_safe_time / k, 4)
    numpy_conversion_any_safe_2_time = round(numpy_conversion_any_safe_2_time / k, 4)

    print(f"Custom any-1: {custom_any_time}")
    print(f"Custom any-2: {custom_any_2_time}")
    print(f"Normal any: {normal_any_time}")
    print(f"Extrema any: {extrema_any_time}")
    print(f"Numpy any: {numpy_any_time}")
    print(f"Numpy any-2: {numpy_any_2_time}")
    print(f"Numpy conversion any: {numpy_conversion_any_time}")
    print(f"Numpy conversion any safe: {numpy_conversion_any_safe_time}")
    print(f"Numpy conversion any safe-2: {numpy_conversion_any_safe_2_time}")
    # --------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ END OF "test_custom_any" ------------------------------------------
    # --------------------------------------------------------------------------------------------------------------


def enum_to_string_test(n=40_000_000, k=10):
    enum_name_time = 0
    enum_name_2_time = 0
    dict_name_time = 0
    dict_name_2_time = 0

    algorithm = Algorithms.CV2_INTER_NEAREST
    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        enum_name_time += timeit.timeit(lambda: Algorithms.CV2_INTER_NEAREST.name, number=n // k)
        enum_name_2_time += timeit.timeit(lambda: algorithm.name, number=n // k)
        dict_name_time += timeit.timeit(lambda: algorithm_to_string_dict[Algorithms.CV2_INTER_NEAREST], number=n // k)
        dict_name_2_time += timeit.timeit(lambda: algorithm_to_string_dict[algorithm], number=n // k)
    print()

    enum_name_time = round(enum_name_time / k, 4)
    enum_name_2_time = round(enum_name_2_time / k, 4)
    dict_name_time = round(dict_name_time / k, 4)
    dict_name_2_time = round(dict_name_2_time / k, 4)

    print(f"Enum name: {enum_name_time}")
    print(f"Enum name-2: {enum_name_2_time}")
    print(f"Dict name: {dict_name_time}")
    print(f"Dict name-2: {dict_name_2_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ END OF "enum_to_string_test" ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def cv2_vs_pil_test(n=100, k=10):
    factors = [0.125, 0.25, 0.5, 2, 4, 8]

    cv2_algorithms = [Algorithms.CV2_INTER_CUBIC, Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_LANCZOS4, Algorithms.CV2_INTER_NEAREST]
    pil_algorithms = [Algorithms.PIL_BICUBIC, Algorithms.PIL_BILINEAR, Algorithms.PIL_LANCZOS, Algorithms.PIL_NEAREST_NEIGHBOR]

    image = Image.open("../input/NEAREST_NEIGHBOR_pixel-art_0.125x.png").convert("RGBA")

    cv2_time = 0
    pil_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        for algorithm in cv2_algorithms:
            # print(f"Og algorithm: {utils.algorithm_to_string(algorithm)}")
            cv2_time += timeit.timeit(lambda: scaler.scale_image_batch(algorithm, image, factors), number=n // k)
        for algorithm in pil_algorithms:
            pil_time += timeit.timeit(lambda: scaler.scale_image_batch(algorithm, image, factors), number=n // k)
    print()

    cv2_time = round(cv2_time / k, 4)
    pil_time = round(pil_time / k, 4)

    print(f"CV2 time: {cv2_time}")
    print(f"PIL time: {pil_time}")
    # --------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ END OF "cv2_vs_pil_test" ------------------------------------------
    # --------------------------------------------------------------------------------------------------------------


def test_pil_wh_vs_cv2_size(n=500_000, k=10):
    image = Image.open("../input/NEAREST_NEIGHBOR_pixel-art_0.125x.png")
    cv2_image = utils.pil_to_cv2(image)

    pil_time = 0
    pil_to_cv2_time = 0
    cv2_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        pil_time += timeit.timeit(lambda: image.size, number=n // k)
        pil_to_cv2_time += timeit.timeit(lambda: utils.pil_to_cv2(image).shape[:2], number=n // k)
        cv2_time += timeit.timeit(lambda: cv2_image.shape[:2], number=n // k)
    print()

    pil_time = round(pil_time / k, 4)
    pil_to_cv2_time = round(pil_to_cv2_time / k, 4)
    cv2_time = round(cv2_time / k, 4)

    print(f"PIL time: {pil_time}")
    print(f"PIL to CV2 time: {pil_to_cv2_time}")
    print(f"CV2 time: {cv2_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "test_pil_wh_vs_cv2_size" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def images_into_out_of_list(algorithm, factors, image):
    scaled_images = []
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.append(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    # print(f"List bytes: {asizeof.asizeof(scaled_images)}")
    for image in scaled_images:
        ...


def images_into_out_of_list_del(algorithm, factors, image):
    scaled_images = []
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.append(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    # print(f"List bytes: {asizeof.asizeof(scaled_images)}")
    while scaled_images:
        image = scaled_images.pop()
        ...


def images_into_out_of_queue(algorithm, factors, image):
    scaled_images = queue.Queue()
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.put(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    # print(f"Queue bytes: {asizeof.asizeof(scaled_images)}")
    while not scaled_images.empty():
        image = scaled_images.get()
        ...


def images_into_out_of_deck(algorithm, factors, image):
    scaled_images = deque()
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.append(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    # print(f"Deck bytes: {asizeof.asizeof(scaled_images)}")
    while scaled_images:
        image = scaled_images.popleft()
        ...


def queue_vs_list_vs_deck(n=50_000, k=50, s=5):
    algorithm = cv2.INTER_NEAREST
    factors = [0.125, 0.25, 0.5, 2, 4, 8]
    image = utils.pil_to_cv2(Image.open("../input/example_shell_40px.png"))

    images_into_queue_time = 0
    images_into_list_time = 0
    images_into_list_del_time = 0
    images_into_deck_time = 0

    print(f"Iteration {1}/{k}")
    for i in range(k):
        if i % s == s - 1:
            print(f"Iteration {i + 1}/{k}")
        images_into_queue_time += timeit.timeit(lambda: images_into_out_of_queue(algorithm, factors, image), number=n // k)
        images_into_list_time += timeit.timeit(lambda: images_into_out_of_list(algorithm, factors, image), number=n // k)
        images_into_list_del_time += timeit.timeit(lambda: images_into_out_of_list_del(algorithm, factors, image), number=n // k)
        images_into_deck_time += timeit.timeit(lambda: images_into_out_of_deck(algorithm, factors, image), number=n // k)
    print()

    images_into_queue_time = round(images_into_queue_time / k, 4)
    images_into_list_time = round(images_into_list_time / k, 4)
    images_into_list_del_time = round(images_into_list_del_time / k, 4)
    images_into_deck_time = round(images_into_deck_time / k, 4)

    print(f"Images into queue time: {images_into_queue_time}")
    print(f"Images into list time: {images_into_list_time}")
    print(f"Images into list del time: {images_into_list_del_time}")
    print(f"Images into deck time: {images_into_deck_time}")

    # ------------------------------------------------------------------------------------------------------------
    factors = [2]

    images_into_queue_time = 0
    images_into_list_time = 0
    images_into_list_del_time = 0
    images_into_deck_time = 0

    print(f"Iteration {1}/{k}")
    for i in range(k):
        if i % s == s - 1:
            print(f"Iteration {i + 1}/{k}")
        images_into_queue_time += timeit.timeit(lambda: images_into_out_of_queue(algorithm, factors, image), number=n // k)
        images_into_list_time += timeit.timeit(lambda: images_into_out_of_list(algorithm, factors, image), number=n // k)
        images_into_list_del_time += timeit.timeit(lambda: images_into_out_of_list_del(algorithm, factors, image), number=n // k)
        images_into_deck_time += timeit.timeit(lambda: images_into_out_of_deck(algorithm, factors, image), number=n // k)
    print()

    images_into_queue_time = round(images_into_queue_time / k, 4)
    images_into_list_time = round(images_into_list_time / k, 4)
    images_into_list_del_time = round(images_into_list_del_time / k, 4)
    images_into_deck_time = round(images_into_deck_time / k, 4)

    print(f"Images into queue time 1: {images_into_queue_time}")
    print(f"Images into list time 1: {images_into_list_time}")
    print(f"Images into list del time 1: {images_into_list_del_time}")
    print(f"Images into deck time 1: {images_into_deck_time}")
    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ END OF "queue_vs_list" ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------


import sys
import inspect
import logging

logger = logging.getLogger(__name__)


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum((get_size(i, seen) for i in obj))
        except TypeError:
            logging.exception("Unable to get size of %r. This may lead to incorrect sizes. Please report this error.",
                              obj)
    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


def pil_vs_cv2_size():
    pil_image = Image.open("../input/NEAREST_NEIGHBOR_pixel-art_0.125x.png")
    cv2_image = utils.pil_to_cv2(pil_image)

    print(f"CV2 size (nbytes): {cv2_image.nbytes}")
    print(f"CV2 size (asizeof): {asizeof.asizeof(cv2_image)}")
    print(f"CV2 size (pysize): {get_size(cv2_image)}")
    print(f"CV2 size (sys): {sys.getsizeof(cv2_image)}")
    print()
    print(f"PIL size (asizeof): {asizeof.asizeof(pil_image)}")
    print(f"PIL size (sys): {sys.getsizeof(pil_image)}")
    # print(f"PIL size: {get_size(pil_image)}")  # ValueError: I/O operation on closed file.
    # --------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ END OF "pil_vs_cv2_size" ------------------------------------------
    # --------------------------------------------------------------------------------------------------------------


def tuple_generation():
    available_algorithms = tuple(f"{algorithm.value} - {algorithm.name}" for algorithm in Algorithms)


def list_generation():
    available_algorithms = [f"{algorithm.value} - {algorithm.name}" for algorithm in Algorithms]


def list_vs_tuple_generation(n=1_000_000, k=10):
    tuple_time = 0
    list_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        tuple_time += timeit.timeit(lambda: tuple_generation(), number=n // k)
        list_time += timeit.timeit(lambda: list_generation(), number=n // k)
    print()

    tuple_time = round(tuple_time / k, 4)
    list_time = round(list_time / k, 4)

    print(f"Tuple time: {tuple_time}")
    print(f"List time: {list_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "list_vs_tuple_generation" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def columnify_test(n=1_000_000, k=10):
    available_algorithms = tuple(f"{algorithm.value} - {algorithm.name}" for algorithm in Algorithms)

    current = time.time()
    for i in range(n):
        columnify(available_algorithms)
    columnify_cached_time = round((time.time() - current) / k, 4)
    print(f"Columnify cached time: {columnify_cached_time}")

    columnify_time = 0
    for i in range(k):
        # print(f"Iteration {i + 1}/{k}")
        columnify_time += timeit.timeit(lambda: columnify(available_algorithms), number=n // k)
    # print()

    columnify_time = round(columnify_time / k, 4)

    print(f"Columnify time: {columnify_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "columnify_test" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def columnify_cached(elements: tuple) -> str:
    result = ""
    max_columns = 4

    max_length = max([len(algorithm) for algorithm in elements])
    # print(f"Max length: {max_length}")
    margin_right = 2
    tab_spaces = 2

    # Get the size of the terminal
    if sys.stdout.isatty():
        terminal_columns = os.get_terminal_size().columns
        # print(f"Terminal columns: {terminal_columns}")
        columns = max_columns
        while terminal_columns < len("\t".expandtabs(tab_spaces)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1 and columns > 1:
            # print(f"Calculated row length: {len("\t".expandtabs(tab_spaces * 2)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1}")
            columns -= 1
    else:
        columns = 3
    # print(f"Final row length: {len("\t".expandtabs(tab_spaces * 2)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1}")
    # print(f"Final column count: {columns}")

    overflow = len(elements) % columns
    full_count = len(elements) - overflow

    for i in range(0, full_count, columns):
        result += "\t".expandtabs(tab_spaces)
        if i < len(elements):
            result += " | ".join([f"\t{elements[i + j]:<{max_length + margin_right}}".expandtabs(tab_spaces) for j in range(columns)])
        result += "\n"
    result += "\t".expandtabs(tab_spaces)
    result += " | ".join([f"\t{elements[k]:<{max_length + margin_right}}".expandtabs(tab_spaces) for k in range(full_count, overflow + full_count)])
    result += "\n"

    return result


def columnify(elements: list) -> str:
    result = ""
    max_columns = 4

    max_length = max([len(algorithm) for algorithm in elements])
    # print(f"Max length: {max_length}")
    margin_right = 2
    tab_spaces = 2

    # Get the size of the terminal
    if sys.stdout.isatty():
        terminal_columns = os.get_terminal_size().columns
        # print(f"Terminal columns: {terminal_columns}")
        columns = max_columns
        while terminal_columns < len("\t".expandtabs(tab_spaces)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1 and columns > 1:
            # print(f"Calculated row length: {len("\t".expandtabs(tab_spaces * 2)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1}")
            columns -= 1
    else:
        columns = 3
    # print(f"Final row length: {len("\t".expandtabs(tab_spaces * 2)) + (max_length + len("\t".expandtabs(tab_spaces)) + margin_right + 1 + 2) * columns - 1}")
    # print(f"Final column count: {columns}")

    overflow = len(elements) % columns
    full_count = len(elements) - overflow

    for i in range(0, full_count, columns):
        result += "\t".expandtabs(tab_spaces)
        if i < len(elements):
            result += " | ".join([f"\t{elements[i + j]:<{max_length + margin_right}}".expandtabs(tab_spaces) for j in range(columns)])
        result += "\n"
    result += "\t".expandtabs(tab_spaces)
    result += " | ".join([f"\t{elements[k]:<{max_length + margin_right}}".expandtabs(tab_spaces) for k in range(full_count, overflow + full_count)])
    result += "\n"

    return result


def tuple_plus_cached(a=2):
    available_algorithms = tuple(f"{algorithm.value} - {algorithm.name}" for algorithm in Algorithms)

    for i in range(a):
        columnify_cached(available_algorithms)

    columnify_cached.cache_clear()


def list_no_cache(a=2):
    available_algorithms = [f"{algorithm.value} - {algorithm.name}" for algorithm in Algorithms]

    for i in range(a):
        columnify(available_algorithms)

    columnify_cached.cache_clear()


def cached_tuple_vs_list_test(n=100_000, k=10):
    cached_tuple_time = 0
    list_time = 0
    cached_tuple_time_2 = 0
    list_time_2 = 0
    cached_tuple_time_3 = 0
    list_time_3 = 0

    a_1 = 1
    a_2 = 2
    a_3 = 3

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        cached_tuple_time += timeit.timeit(lambda: tuple_plus_cached(a_1), number=n // k)
        list_time += timeit.timeit(lambda: list_no_cache(a_1), number=n // k)
        cached_tuple_time_2 += timeit.timeit(lambda: tuple_plus_cached(a_2), number=n // k)
        list_time_2 += timeit.timeit(lambda: list_no_cache(a_2), number=n // k)
        cached_tuple_time_3 += timeit.timeit(lambda: tuple_plus_cached(a_3), number=n // k)
        list_time_3 += timeit.timeit(lambda: list_no_cache(a_3), number=n // k)

    cached_tuple_time = round(cached_tuple_time / k, 4)
    list_time = round(list_time / k, 4)
    cached_tuple_time_2 = round(cached_tuple_time_2 / k, 4)
    list_time_2 = round(list_time_2 / k, 4)
    cached_tuple_time_3 = round(cached_tuple_time_3 / k, 4)
    list_time_3 = round(list_time_3 / k, 4)

    print(f"Cached tuple time {a_1}x: {cached_tuple_time}")
    print(f"List time {a_1}x: {list_time}")
    print(f"Cached tuple time {a_2}x: {cached_tuple_time_2}")
    print(f"List time {a_2}x: {list_time_2}")
    print(f"Cached tuple time {a_3}x: {cached_tuple_time_3}")
    print(f"List time {a_3}x: {list_time_3}")


def docstring_tests():
    print(scaler.csatpa.__doc__)


if __name__ == "__main__":
    print("Starting warmup")
    warmup()
    print("Warmup finished\n")

    # test_match_vs_dict()
    # test_custom_any()
    # enum_to_string_test()
    # cv2_vs_pil_test()
    # test_pil_wh_vs_cv2_size()

    # queue_vs_list_vs_deck(n=2_000_000, k=100, s=10)
    # Images into queue time: 6.9286
    # Images into list time: 5.9382
    # Images into list del time: 5.9511
    # Images into deck time: 5.9904

    # pil_vs_cv2_size()
    # list_vs_tuple_generation()
    # columnify_test()  # Broken
    # cached_tuple_vs_list_test()
    # docstring_tests()
    ...
