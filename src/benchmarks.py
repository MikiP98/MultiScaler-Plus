# coding=utf-8

import cv2
import numpy as np
import os
import queue
import scaler
import standalone
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
    for i in range(20_000_000):
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

    image = utils.pngify(Image.open("../input/NEAREST_NEIGHBOR_pixel-art_0.125x.png").convert("RGBA"))

    cv2_time = 0
    pil_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        for algorithm in cv2_algorithms:
            # print(f"Og algorithm: {utils.algorithm_to_string(algorithm)}")
            cv2_time += timeit.timeit(lambda: scaler.scale_image_batch(algorithm, [image], factors), number=n // k)
        for algorithm in pil_algorithms:
            pil_time += timeit.timeit(lambda: scaler.scale_image_batch(algorithm, [image], factors), number=n // k)
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

    len(scaled_images)
    # print(f"List bytes: {asizeof.asizeof(scaled_images)}")
    for image in scaled_images:
        ...


def images_into_out_of_list_del(algorithm, factors, image):
    scaled_images = []
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.append(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    len(scaled_images)
    # print(f"List bytes: {asizeof.asizeof(scaled_images)}")
    while scaled_images:
        image = scaled_images.pop()
        ...


def images_into_out_of_queue(algorithm, factors, image):
    scaled_images = queue.Queue()
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.put(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    scaled_images.qsize()
    # print(f"Queue bytes: {asizeof.asizeof(scaled_images)}")
    while not scaled_images.empty():
        image = scaled_images.get()
        ...


def images_into_out_of_deck(algorithm, factors, image):
    scaled_images = deque()
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.append(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    len(scaled_images)
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
        columnify_cached(available_algorithms)
    columnify_cached_time = round((time.time() - current) / k, 4)
    print(f"Columnify cached time: {columnify_cached_time}")

    columnify_time = 0
    for i in range(k):
        # print(f"Iteration {i + 1}/{k}")
        columnify_time += timeit.timeit(lambda: columnify_cached(available_algorithms), number=n // k)
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
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "cached_tuple_vs_list_test" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


pil_fully_supported_formats = ("BLP", "BMP", "DDS", "DIB", "EPS", "GIF", "ICNS", "ICO", "IM",
                                   "JFI", "JFIF", "JIF", "JPE", "JPEG", "JPG",
                                   "J2K", "JP2", "JPX",
                                   "MSP", "PCX", "PFM", "PNG", "APNG", "PPM", "SGI",
                                   "SPIDER", "SPI",
                                   "TGA", "TIFF", "WEBP", "XBM")

pil_read_only_formats = ("CUR", "DCX", "FITS",
                         "FLI", "FLC",
                         "FPX", "FTEX", "GBR", "GD", "IMT",
                         "IPTC", "NAA",
                         "MCIDAS", "MIC", "MPO", "PCD", "PIXAR", "PSD", "QOI", "SUN", "WAL",
                         "WMF", "EMF",
                         "XPM")

pil_write_only_formats = ("PALM", "PDF",
                          "XVTHUMB", "XVTHUMBNAILS")

pil_indentify_only_formats = ("BUFR", "GRIB", "HDF5", "MPEG")

pil_fully_supported_formats_s = {"BLP", "BMP", "DDS", "DIB", "EPS", "GIF", "ICNS", "ICO", "IM",
                               "JFI", "JFIF", "JIF", "JPE", "JPEG", "JPG",
                               "J2K", "JP2", "JPX",
                               "MSP", "PCX", "PFM", "PNG", "APNG", "PPM", "SGI",
                               "SPIDER", "SPI",
                               "TGA", "TIFF", "WEBP", "XBM"}

pil_read_only_formats_s = {"CUR", "DCX", "FITS",
                         "FLI", "FLC",
                         "FPX", "FTEX", "GBR", "GD", "IMT",
                         "IPTC", "NAA",
                         "MCIDAS", "MIC", "MPO", "PCD", "PIXAR", "PSD", "QOI", "SUN", "WAL",
                         "WMF", "EMF",
                         "XPM"}

pil_write_only_formats_s = {"PALM", "PDF",
                          "XVTHUMB", "XVTHUMBNAILS"}

pil_indentify_only_formats_s = {"BUFR", "GRIB", "HDF5", "MPEG"}


def endswith_tuple():
    for root, dirs, files in os.walk("../input"):
        for file in files:
            if file.endswith(pil_fully_supported_formats):
                ...
            elif file.endswith(pil_read_only_formats):
                ...
            elif file.endswith(pil_write_only_formats):
                ...
            elif file.endswith(pil_indentify_only_formats):
                ...
            else:
                ...


def endswith_tuple_p_create():
    pil_fully_supported_formats = ("BLP", "BMP", "DDS", "DIB", "EPS", "GIF", "ICNS", "ICO", "IM",
                                   "JFI", "JFIF", "JIF", "JPE", "JPEG", "JPG",
                                   "J2K", "JP2", "JPX",
                                   "MSP", "PCX", "PFM", "PNG", "APNG", "PPM", "SGI",
                                   "SPIDER", "SPI",
                                   "TGA", "TIFF", "WEBP", "XBM")

    pil_read_only_formats = ("CUR", "DCX", "FITS",
                             "FLI", "FLC",
                             "FPX", "FTEX", "GBR", "GD", "IMT",
                             "IPTC", "NAA",
                             "MCIDAS", "MIC", "MPO", "PCD", "PIXAR", "PSD", "QOI", "SUN", "WAL",
                             "WMF", "EMF",
                             "XPM")

    pil_write_only_formats = ("PALM", "PDF",
                              "XVTHUMB", "XVTHUMBNAILS")

    pil_indentify_only_formats = ("BUFR", "GRIB", "HDF5", "MPEG")

    for root, dirs, files in os.walk("../input"):
        for file in files:
            if file.endswith(pil_fully_supported_formats):
                ...
            elif file.endswith(pil_read_only_formats):
                ...
            elif file.endswith(pil_write_only_formats):
                ...
            elif file.endswith(pil_indentify_only_formats):
                ...
            else:
                ...


def split_in_set():
    for root, dirs, files in os.walk("../input"):
        for file in files:
            extension = file.split(".")[-1]
            if extension in pil_fully_supported_formats_s:
                ...
            elif extension in pil_read_only_formats_s:
                ...
            elif extension in pil_write_only_formats_s:
                ...
            elif extension in pil_indentify_only_formats_s:
                ...
            else:
                ...


def split_in_set_p_create():
    pil_fully_supported_formats_s = {"BLP", "BMP", "DDS", "DIB", "EPS", "GIF", "ICNS", "ICO", "IM",
                                     "JFI", "JFIF", "JIF", "JPE", "JPEG", "JPG",
                                     "J2K", "JP2", "JPX",
                                     "MSP", "PCX", "PFM", "PNG", "APNG", "PPM", "SGI",
                                     "SPIDER", "SPI",
                                     "TGA", "TIFF", "WEBP", "XBM"}

    pil_read_only_formats_s = {"CUR", "DCX", "FITS",
                               "FLI", "FLC",
                               "FPX", "FTEX", "GBR", "GD", "IMT",
                               "IPTC", "NAA",
                               "MCIDAS", "MIC", "MPO", "PCD", "PIXAR", "PSD", "QOI", "SUN", "WAL",
                               "WMF", "EMF",
                               "XPM"}

    pil_write_only_formats_s = {"PALM", "PDF",
                                "XVTHUMB", "XVTHUMBNAILS"}

    pil_indentify_only_formats_s = {"BUFR", "GRIB", "HDF5", "MPEG"}

    for root, dirs, files in os.walk("../input"):
        for file in files:
            extension = file.split(".")[-1]
            if extension in pil_fully_supported_formats_s:
                ...
            elif extension in pil_read_only_formats_s:
                ...
            elif extension in pil_write_only_formats_s:
                ...
            elif extension in pil_indentify_only_formats_s:
                ...
            else:
                ...


def endswith_tuple_vs_split_in_set(n=100_000, k=10):
    endswith_tuple_time = 0
    split_in_set_time = 0
    endswith_tuple_p_create_time = 0
    split_in_set_p_create_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        endswith_tuple_time += timeit.timeit(lambda: endswith_tuple(), number=n // k)
        split_in_set_time += timeit.timeit(lambda: split_in_set(), number=n // k)
        endswith_tuple_p_create_time += timeit.timeit(lambda: endswith_tuple_p_create(), number=n // k)
        split_in_set_p_create_time += timeit.timeit(lambda: split_in_set_p_create(), number=n // k)
    print()

    endswith_tuple_time = round(endswith_tuple_time / k, 4)
    split_in_set_time = round(split_in_set_time / k, 4)
    endswith_tuple_p_create_time = round(endswith_tuple_p_create_time / k, 4)
    split_in_set_p_create_time = round(split_in_set_p_create_time / k, 4)

    print(f"Endswith tuple time: {endswith_tuple_time}")
    print(f"Split in set time: {split_in_set_time}")
    print(f"Endswith tuple p create time: {endswith_tuple_p_create_time}")
    print(f"Split in set p create time: {split_in_set_p_create_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "endswith_tuple_vs_split_in_set" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def single_process_3():
    config = {
        'clear_output_directory': True,
        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,
        'sort_by_algorithm': False,
        'lossless_compression': True,
        'multiprocessing_levels': {},
        'override_processes_count': False,
        'max_processes': (32, 32, 32),
        'mcmeta_correction': True
    }

    algorithms = [Algorithms.CV2_INTER_NEAREST]
    scales = [4, 8, 16, 24, 32, 64, 72, 88]

    standalone.run(algorithms, scales, config)


def multi_processed_3():
    config = {
        'clear_output_directory': True,
        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,
        'sort_by_algorithm': False,
        'lossless_compression': True,
        'multiprocessing_levels': {3},
        'override_processes_count': False,
        'max_processes': (32, 32, 32),
        'mcmeta_correction': True
    }

    algorithms = [Algorithms.CV2_INTER_NEAREST]
    scales = [4, 8, 16, 24, 32, 64, 72, 88]

    standalone.run(algorithms, scales, config)


def single_vs_multi_3(n=4, k=2):
    single_time = 0
    multi_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        single_time += timeit.timeit(lambda: single_process_3(), number=n // k)
        multi_time += timeit.timeit(lambda: multi_processed_3(), number=n // k)
    print()

    single_time = round(single_time / k, 4)
    multi_time = round(multi_time / k, 4)

    print(f"Single time: {single_time}")
    print(f"Multi time: {multi_time}")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "single_vs_multi_3" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def single_process_2():
    config = {
        'clear_output_directory': True,
        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,
        'sort_by_algorithm': False,
        'lossless_compression': True,
        'multiprocessing_levels': {},
        'override_processes_count': False,
        'max_processes': (32, 32, 32),
        'mcmeta_correction': True
    }

    algorithms = [Algorithms.CV2_INTER_NEAREST, Algorithms.CV2_ESPCN, Algorithms.PIL_NEAREST_NEIGHBOR,
                  Algorithms.RealESRGAN, Algorithms.xBRZ]  # , Algorithms.FSR
    scales = [2, 4, 8]

    standalone.run(algorithms, scales, config)


def multi_processed_2():
    config = {
        'clear_output_directory': True,
        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,
        'sort_by_algorithm': False,
        'lossless_compression': True,
        'multiprocessing_levels': {2},
        'override_processes_count': False,
        'max_processes': (32, 32, 32),
        'mcmeta_correction': True
    }

    algorithms = [Algorithms.CV2_INTER_NEAREST, Algorithms.CV2_ESPCN, Algorithms.PIL_NEAREST_NEIGHBOR,
                  Algorithms.RealESRGAN, Algorithms.xBRZ]  # , Algorithms.FSR
    scales = [2, 4, 8]

    standalone.run(algorithms, scales, config)


def single_vs_multi_2(n=4, k=2):
    single_time = 0
    multi_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        single_time += timeit.timeit(lambda: single_process_2(), number=n // k)
        multi_time += timeit.timeit(lambda: multi_processed_2(), number=n // k)
    print()

    single_time = round(single_time / k, 4)
    multi_time = round(multi_time / k, 4)

    print(f"Single time: {single_time}")
    print(f"Multi time: {multi_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "single_vs_multi_2" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def single_process_2_3():
    config = {
        'clear_output_directory': True,
        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,
        'sort_by_algorithm': False,
        'lossless_compression': True,
        'multiprocessing_levels': {},
        'override_processes_count': False,
        'max_processes': (32, 32, 32),
        'mcmeta_correction': True
    }

    algorithms = [Algorithms.CV2_INTER_LANCZOS4, Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_CUBIC]
    scales = [2, 4, 32, 64]

    standalone.run(algorithms, scales, config)


def multi_processed_2_3():
    config = {
        'clear_output_directory': True,
        'add_algorithm_name_to_output_files_names': True,
        'add_factor_to_output_files_names': True,
        'sort_by_algorithm': False,
        'lossless_compression': True,
        'multiprocessing_levels': {2, 3},
        'override_processes_count': False,
        'max_processes': (32, 32, 32),
        'mcmeta_correction': True
    }

    algorithms = [Algorithms.CV2_INTER_LANCZOS4, Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_CUBIC]
    scales = [2, 4, 32, 64]

    standalone.run(algorithms, scales, config)


def single_vs_multi_2_3(n=2, k=2):
    single_time = 0
    multi_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        single_time += timeit.timeit(lambda: single_process_2_3(), number=n // k)
        multi_time += timeit.timeit(lambda: multi_processed_2_3(), number=n // k)
    print()

    single_time = round(single_time / k, 4)
    multi_time = round(multi_time / k, 4)

    print(f"Single time: {single_time}")
    print(f"Multi time: {multi_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "single_vs_multi_2_3" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def list_alike_test(n=1_000_000, k=10):
    elements = 100

    generator_time = 0
    list_time = 0
    tuple_time = 0
    set_time = 0
    frozen_set_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        generator_time += timeit.timeit(lambda: (gen := i for i in range(elements)), number=n // k)
        list_time += timeit.timeit(lambda: (iter := [i for i in range(elements)]), number=n // k)
        tuple_time += timeit.timeit(lambda: (iter := tuple(i for i in range(elements))), number=n // k)
        set_time += timeit.timeit(lambda: (iter := {i for i in range(elements)}), number=n // k)
        frozen_set_time += timeit.timeit(lambda: (iter := frozenset(i for i in range(elements))), number=n // k)
    print()

    list_time = round(list_time / k, 4)
    generator_time = round(generator_time / k, 4)
    tuple_time = round(tuple_time / k, 4)
    set_time = round(set_time / k, 4)
    frozen_set_time = round(frozen_set_time / k, 4)

    print(f"Generator time: {generator_time}")
    print(f"List time: {list_time}")
    print(f"Tuple time: {tuple_time}")
    print(f"Set time: {set_time}")
    print(f"Frozen set time: {frozen_set_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "list_alike_test" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def list_alike_test_2(n=3_000_000, k=10):
    elements = 100

    list_time = 0
    tuple_time = 0
    set_time = 0
    frozen_set_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        list_time += timeit.timeit(lambda: (iter := list(range(elements))), number=n // k)
        tuple_time += timeit.timeit(lambda: (iter := tuple(range(elements))), number=n // k)
        set_time += timeit.timeit(lambda: (iter := set(range(elements))), number=n // k)
        frozen_set_time += timeit.timeit(lambda: (iter := frozenset(range(elements))), number=n // k)
    print()

    list_time = round(list_time / k, 4)
    tuple_time = round(tuple_time / k, 4)
    set_time = round(set_time / k, 4)
    frozen_set_time = round(frozen_set_time / k, 4)

    print(f"List time: {list_time}")
    print(f"Tuple time: {tuple_time}")
    print(f"Set time: {set_time}")
    print(f"Frozen set time: {frozen_set_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "list_alike_test_2" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def set_vs_frozenset_generation(n=10_000_000, k=20):
    elements = 100

    set_time = 0
    frozen_set_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        set_time += timeit.timeit(lambda: (iter := set(range(elements))), number=n // k)
        frozen_set_time += timeit.timeit(lambda: (iter := frozenset(range(elements))), number=n // k)
    print()

    set_time = round(set_time / k, 4)
    frozen_set_time = round(frozen_set_time / k, 4)

    print(f"Set time: {set_time}")
    print(f"Frozen set time: {frozen_set_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "set_vs_frozenset" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def frozen_set_from_elements(n=4_000_000, k=10):
    list_elements = list(range(100))
    tuple_elements = tuple(range(100))
    set_elements = set(range(100))

    list_time = 0
    tuple_time = 0
    set_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        list_time += timeit.timeit(lambda: (frozen_set := frozenset(list_elements)), number=n // k)
        tuple_time += timeit.timeit(lambda: (frozen_set := frozenset(tuple_elements)), number=n // k)
        set_time += timeit.timeit(lambda: (frozen_set := frozenset(set_elements)), number=n // k)
    print()

    list_time = round(list_time / k, 4)
    tuple_time = round(tuple_time / k, 4)
    set_time = round(set_time / k, 4)

    print(f"From list time: {list_time}")
    print(f"From tuple time: {tuple_time}")
    print(f"From set time: {set_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "frozen_set_from_elements" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def dict_with_list():
    pil_fully_supported_formats = {
        "BLP": ["blp", "blp2", "tex"],
        "BMP": ["bmp", "rle"],
        "DDS": ["dds", "dds2"],
        "DIB": ["dib", "dib2"],
        "EPS": ["eps", "eps2", "epsf", "epsi"],
        "GIF": ["gif", "giff"],
        "ICNS": ["icns", "icon"],
        "ICO": ["ico", "cur"],
        "IM": ["im", "im2"],
        "JPEG": ["jpg", "jpeg", "jpe"],
        "JPEG 2000": ["jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx"],
        "MSP": ["msp", "msp2"],
        "PCX": ["pcx", "pcx2"],
        "PFM": ["pfm", "pfm2"],
        "PNG": ["png", "pns"],
        "APNG": ["apng", "png2"],
        "PPM": ["ppm", "ppm2"],
        "SGI": ["sgi", "rgb", "bw"],
        "SPIDER": ["spi", "spider2"],
        "TGA": ["tga", "targa"],
        "TIFF": ["tif", "tiff", "tiff2"],
        "WebP": ["webp", "webp2"],
        "XBM": ["xbm", "xbm2"]
    }


def dict_with_tuple():
    pil_fully_supported_formats = {
        "BLP": ("blp", "blp2", "tex",),
        "BMP": ("bmp", "rle",),
        "DDS": ("dds", "dds2",),
        "DIB": ("dib", "dib2",),
        "EPS": ("eps", "eps2", "epsf", "epsi",),
        "GIF": ("gif", "giff",),
        "ICNS": ("icns", "icon",),
        "ICO": ("ico", "cur",),
        "IM": ("im", "im2",),
        "JPEG": ("jpg", "jpeg", "jpe",),
        "JPEG 2000": ("jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx",),
        "MSP": ("msp", "msp2",),
        "PCX": ("pcx", "pcx2",),
        "PFM": ("pfm", "pfm2",),
        "PNG": ("png", "pns",),
        "APNG": ("apng", "png2",),
        "PPM": ("ppm", "ppm2",),
        "SGI": ("sgi", "rgb", "bw",),
        "SPIDER": ("spi", "spider2",),
        "TGA": ("tga", "targa",),
        "TIFF": ("tif", "tiff", "tiff2",),
        "WebP": ("webp", "webp2",),
        "XBM": ("xbm", "xbm2",)
    }

def dict_with_set():
    pil_fully_supported_formats = {
        "BLP": {"blp", "blp2", "tex"},
        "BMP": {"bmp", "rle"},
        "DDS": {"dds", "dds2"},
        "DIB": {"dib", "dib2"},
        "EPS": {"eps", "eps2", "epsf", "epsi"},
        "GIF": {"gif", "giff"},
        "ICNS": {"icns", "icon"},
        "ICO": {"ico", "cur"},
        "IM": {"im", "im2"},
        "JPEG": {"jpg", "jpeg", "jpe"},
        "JPEG 2000": {"jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx"},
        "MSP": {"msp", "msp2"},
        "PCX": {"pcx", "pcx2"},
        "PFM": {"pfm", "pfm2"},
        "PNG": {"png", "pns"},
        "APNG": {"apng", "png2"},
        "PPM": {"ppm", "ppm2"},
        "SGI": {"sgi", "rgb", "bw"},
        "SPIDER": {"spi", "spider2"},
        "TGA": {"tga", "targa"},
        "TIFF": {"tif", "tiff", "tiff2"},
        "WebP": {"webp", "webp2"},
        "XBM": {"xbm", "xbm2"}
    }


def dict_with_frozenset():
    pil_fully_supported_formats = {
        "BLP": frozenset({"blp", "blp2", "tex"}),
        "BMP": frozenset({"bmp", "rle"}),
        "DDS": frozenset({"dds", "dds2"}),
        "DIB": frozenset({"dib", "dib2"}),
        "EPS": frozenset({"eps", "eps2", "epsf", "epsi"}),
        "GIF": frozenset({"gif", "giff"}),
        "ICNS": frozenset({"icns", "icon"}),
        "ICO": frozenset({"ico", "cur"}),
        "IM": frozenset({"im", "im2"}),
        "JPEG": frozenset({"jpg", "jpeg", "jpe"}),
        "JPEG 2000": frozenset({"jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx"}),
        "MSP": frozenset({"msp", "msp2"}),
        "PCX": frozenset({"pcx", "pcx2"}),
        "PFM": frozenset({"pfm", "pfm2"}),
        "PNG": frozenset({"png", "pns"}),
        "APNG": frozenset({"apng", "png2"}),
        "PPM": frozenset({"ppm", "ppm2"}),
        "SGI": frozenset({"sgi", "rgb", "bw"}),
        "SPIDER": frozenset({"spi", "spider2"}),
        "TGA": frozenset({"tga", "targa"}),
        "TIFF": frozenset({"tif", "tiff", "tiff2"}),
        "WebP": frozenset({"webp", "webp2"}),
        "XBM": frozenset({"xbm", "xbm2"})
    }


def dict_with_list_vs_tuple_vs_set_vs_frozenset(n=2_000_000, k=10):
    dict_with_list_time = 0
    dict_with_tuple_time = 0
    dict_with_set_time = 0
    dict_with_frozenset_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        dict_with_list_time += timeit.timeit(lambda: dict_with_list(), number=n // k)
        dict_with_tuple_time += timeit.timeit(lambda: dict_with_tuple(), number=n // k)
        dict_with_set_time += timeit.timeit(lambda: dict_with_set(), number=n // k)
        dict_with_frozenset_time += timeit.timeit(lambda: dict_with_frozenset(), number=n // k)
    print()

    dict_with_list_time = round(dict_with_list_time / k, 4)
    dict_with_tuple_time = round(dict_with_tuple_time / k, 4)
    dict_with_set_time = round(dict_with_set_time / k, 4)
    dict_with_frozenset_time = round(dict_with_frozenset_time / k, 4)

    print(f"Dict with list time: {dict_with_list_time}")
    print(f"Dict with tuple time: {dict_with_tuple_time}")
    print(f"Dict with set time: {dict_with_set_time}")
    print(f"Dict with frozenset time: {dict_with_frozenset_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "dict_with_list_vs_tuple_vs_set_vs_frozenset" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


def set_from_dict_w_tuples_simple():
    pil_fully_supported_formats = {
        "BLP": ("blp", "blp2", "tex",),
        "BMP": ("bmp", "rle",),
        "DDS": ("dds", "dds2",),
        "DIB": ("dib", "dib2",),
        "EPS": ("eps", "eps2", "epsf", "epsi",),
        "GIF": ("gif", "giff",),
        "ICNS": ("icns", "icon",),
        "ICO": ("ico", "cur",),
        "IM": ("im", "im2",),
        "JPEG": ("jpg", "jpeg", "jpe",),
        "JPEG 2000": ("jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx",),
        "MSP": ("msp", "msp2",),
        "PCX": ("pcx", "pcx2",),
        "PFM": ("pfm", "pfm2",),
        "PNG": ("png", "pns",),
        "APNG": ("apng", "png2",),
        "PPM": ("ppm", "ppm2",),
        "SGI": ("sgi", "rgb", "bw",),
        "SPIDER": ("spi", "spider2",),
        "TGA": ("tga", "targa",),
        "TIFF": ("tif", "tiff", "tiff2",),
        "WebP": ("webp", "webp2",),
        "XBM": ("xbm", "xbm2",)
    }
    print(pil_fully_supported_formats.values())
    print(*pil_fully_supported_formats.values())
    pil_fully_supported_formats_cache = set(pil_fully_supported_formats.values())
    print(pil_fully_supported_formats_cache)
    raise NotImplementedError("Not implemented yet")


def set_from_dict_w_tuples_custom():
    pil_fully_supported_formats = {
        "BLP": ("blp", "blp2", "tex",),
        "BMP": ("bmp", "rle",),
        "DDS": ("dds", "dds2",),
        "DIB": ("dib", "dib2",),
        "EPS": ("eps", "eps2", "epsf", "epsi",),
        "GIF": ("gif", "giff",),
        "ICNS": ("icns", "icon",),
        "ICO": ("ico", "cur",),
        "IM": ("im", "im2",),
        "JPEG": ("jpg", "jpeg", "jpe",),
        "JPEG 2000": ("jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx",),
        "MSP": ("msp", "msp2",),
        "PCX": ("pcx", "pcx2",),
        "PFM": ("pfm", "pfm2",),
        "PNG": ("png", "pns",),
        "APNG": ("apng", "png2",),
        "PPM": ("ppm", "ppm2",),
        "SGI": ("sgi", "rgb", "bw",),
        "SPIDER": ("spi", "spider2",),
        "TGA": ("tga", "targa",),
        "TIFF": ("tif", "tiff", "tiff2",),
        "WebP": ("webp", "webp2",),
        "XBM": ("xbm", "xbm2",)
    }
    pil_fully_supported_formats_cache = set(
        extension for extensions in pil_fully_supported_formats.values() for extension in extensions
    )
    # print(pil_fully_supported_formats_cache)


def frozenset_from_dict_w_tuples_simple():
    pil_fully_supported_formats = {
        "BLP": ("blp", "blp2", "tex",),
        "BMP": ("bmp", "rle",),
        "DDS": ("dds", "dds2",),
        "DIB": ("dib", "dib2",),
        "EPS": ("eps", "eps2", "epsf", "epsi",),
        "GIF": ("gif", "giff",),
        "ICNS": ("icns", "icon",),
        "ICO": ("ico", "cur",),
        "IM": ("im", "im2",),
        "JPEG": ("jpg", "jpeg", "jpe",),
        "JPEG 2000": ("jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx",),
        "MSP": ("msp", "msp2",),
        "PCX": ("pcx", "pcx2",),
        "PFM": ("pfm", "pfm2",),
        "PNG": ("png", "pns",),
        "APNG": ("apng", "png2",),
        "PPM": ("ppm", "ppm2",),
        "SGI": ("sgi", "rgb", "bw",),
        "SPIDER": ("spi", "spider2",),
        "TGA": ("tga", "targa",),
        "TIFF": ("tif", "tiff", "tiff2",),
        "WebP": ("webp", "webp2",),
        "XBM": ("xbm", "xbm2",)
    }
    print(pil_fully_supported_formats.values())
    print(*pil_fully_supported_formats.values())
    pil_fully_supported_formats_cache = set(pil_fully_supported_formats.values())
    print(pil_fully_supported_formats_cache)
    raise NotImplementedError("Not implemented yet")


def frozenset_from_dict_w_tuples_custom():
    pil_fully_supported_formats = {
        "BLP": ("blp", "blp2", "tex",),
        "BMP": ("bmp", "rle",),
        "DDS": ("dds", "dds2",),
        "DIB": ("dib", "dib2",),
        "EPS": ("eps", "eps2", "epsf", "epsi",),
        "GIF": ("gif", "giff",),
        "ICNS": ("icns", "icon",),
        "ICO": ("ico", "cur",),
        "IM": ("im", "im2",),
        "JPEG": ("jpg", "jpeg", "jpe",),
        "JPEG 2000": ("jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx",),
        "MSP": ("msp", "msp2",),
        "PCX": ("pcx", "pcx2",),
        "PFM": ("pfm", "pfm2",),
        "PNG": ("png", "pns",),
        "APNG": ("apng", "png2",),
        "PPM": ("ppm", "ppm2",),
        "SGI": ("sgi", "rgb", "bw",),
        "SPIDER": ("spi", "spider2",),
        "TGA": ("tga", "targa",),
        "TIFF": ("tif", "tiff", "tiff2",),
        "WebP": ("webp", "webp2",),
        "XBM": ("xbm", "xbm2",)
    }
    pil_fully_supported_formats_cache = frozenset(
        extension for extensions in pil_fully_supported_formats.values() for extension in extensions
    )
    # print(pil_fully_supported_formats_cache)


def set_from_dict_w_lists_simple():
    pil_fully_supported_formats = {
        "BLP": ["blp", "blp2", "tex"],
        "BMP": ["bmp", "rle"],
        "DDS": ["dds", "dds2"],
        "DIB": ["dib", "dib2"],
        "EPS": ["eps", "eps2", "epsf", "epsi"],
        "GIF": ["gif", "giff"],
        "ICNS": ["icns", "icon"],
        "ICO": ["ico", "cur"],
        "IM": ["im", "im2"],
        "JPEG": ["jpg", "jpeg", "jpe"],
        "JPEG 2000": ["jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx"],
        "MSP": ["msp", "msp2"],
        "PCX": ["pcx", "pcx2"],
        "PFM": ["pfm", "pfm2"],
        "PNG": ["png", "pns"],
        "APNG": ["apng", "png2"],
        "PPM": ["ppm", "ppm2"],
        "SGI": ["sgi", "rgb", "bw"],
        "SPIDER": ["spi", "spider2"],
        "TGA": ["tga", "targa"],
        "TIFF": ["tif", "tiff", "tiff2"],
        "WebP": ["webp", "webp2"],
        "XBM": ["xbm", "xbm2"]
    }
    list_of_lists = list(
        extensions for extensions in pil_fully_supported_formats.values()
    )
    print(list_of_lists)
    raise NotImplementedError("Not implemented yet")


def set_from_dict_w_lists_custom():
    pil_fully_supported_formats = {
        "BLP": ["blp", "blp2", "tex"],
        "BMP": ["bmp", "rle"],
        "DDS": ["dds", "dds2"],
        "DIB": ["dib", "dib2"],
        "EPS": ["eps", "eps2", "epsf", "epsi"],
        "GIF": ["gif", "giff"],
        "ICNS": ["icns", "icon"],
        "ICO": ["ico", "cur"],
        "IM": ["im", "im2"],
        "JPEG": ["jpg", "jpeg", "jpe"],
        "JPEG 2000": ["jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx"],
        "MSP": ["msp", "msp2"],
        "PCX": ["pcx", "pcx2"],
        "PFM": ["pfm", "pfm2"],
        "PNG": ["png", "pns"],
        "APNG": ["apng", "png2"],
        "PPM": ["ppm", "ppm2"],
        "SGI": ["sgi", "rgb", "bw"],
        "SPIDER": ["spi", "spider2"],
        "TGA": ["tga", "targa"],
        "TIFF": ["tif", "tiff", "tiff2"],
        "WebP": ["webp", "webp2"],
        "XBM": ["xbm", "xbm2"]
    }
    pil_fully_supported_formats_cache = set(
        extension for extensions in pil_fully_supported_formats.values() for extension in extensions
    )
    # print(pil_fully_supported_formats_cache)


def frozenset_from_dict_w_lists_simple():
    pil_fully_supported_formats = {
        "BLP": ["blp", "blp2", "tex"],
        "BMP": ["bmp", "rle"],
        "DDS": ["dds", "dds2"],
        "DIB": ["dib", "dib2"],
        "EPS": ["eps", "eps2", "epsf", "epsi"],
        "GIF": ["gif", "giff"],
        "ICNS": ["icns", "icon"],
        "ICO": ["ico", "cur"],
        "IM": ["im", "im2"],
        "JPEG": ["jpg", "jpeg", "jpe"],
        "JPEG 2000": ["jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx"],
        "MSP": ["msp", "msp2"],
        "PCX": ["pcx", "pcx2"],
        "PFM": ["pfm", "pfm2"],
        "PNG": ["png", "pns"],
        "APNG": ["apng", "png2"],
        "PPM": ["ppm", "ppm2"],
        "SGI": ["sgi", "rgb", "bw"],
        "SPIDER": ["spi", "spider2"],
        "TGA": ["tga", "targa"],
        "TIFF": ["tif", "tiff", "tiff2"],
        "WebP": ["webp", "webp2"],
        "XBM": ["xbm", "xbm2"]
    }
    list_of_lists = list(
        extensions for extensions in pil_fully_supported_formats.values()
    )
    print(list_of_lists)
    raise NotImplementedError("Not implemented yet")


def frozenset_from_dict_w_lists_custom():
    pil_fully_supported_formats = {
        "BLP": ["blp", "blp2", "tex"],
        "BMP": ["bmp", "rle"],
        "DDS": ["dds", "dds2"],
        "DIB": ["dib", "dib2"],
        "EPS": ["eps", "eps2", "epsf", "epsi"],
        "GIF": ["gif", "giff"],
        "ICNS": ["icns", "icon"],
        "ICO": ["ico", "cur"],
        "IM": ["im", "im2"],
        "JPEG": ["jpg", "jpeg", "jpe"],
        "JPEG 2000": ["jp2", "j2k", "jpf", "jpx", "jpm", "j2c", "j2r", "jpx"],
        "MSP": ["msp", "msp2"],
        "PCX": ["pcx", "pcx2"],
        "PFM": ["pfm", "pfm2"],
        "PNG": ["png", "pns"],
        "APNG": ["apng", "png2"],
        "PPM": ["ppm", "ppm2"],
        "SGI": ["sgi", "rgb", "bw"],
        "SPIDER": ["spi", "spider2"],
        "TGA": ["tga", "targa"],
        "TIFF": ["tif", "tiff", "tiff2"],
        "WebP": ["webp", "webp2"],
        "XBM": ["xbm", "xbm2"]
    }
    pil_fully_supported_formats_cache = frozenset(
        extension for extensions in pil_fully_supported_formats.values() for extension in extensions
    )
    # print(pil_fully_supported_formats_cache)


def set_from_dict(n=1_000_000, k=10):
    set_f_dict_w_tuples_custom_time = 0
    frozenset_f_dict_w_tuples_custom_time = 0
    set_f_dict_w_lists_custom_time = 0
    frozenset_f_dict_w_lists_custom_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        set_f_dict_w_tuples_custom_time += timeit.timeit(lambda: set_from_dict_w_tuples_custom(), number=n // k)
        frozenset_f_dict_w_tuples_custom_time += timeit.timeit(lambda: frozenset_from_dict_w_tuples_custom(), number=n // k)
        set_f_dict_w_lists_custom_time += timeit.timeit(lambda: set_from_dict_w_lists_custom(), number=n // k)
        frozenset_f_dict_w_lists_custom_time += timeit.timeit(lambda: frozenset_from_dict_w_lists_custom(), number=n // k)
    print()

    set_f_dict_w_tuples_custom_time = round(set_f_dict_w_tuples_custom_time / k, 4)
    frozenset_f_dict_w_tuples_custom_time = round(frozenset_f_dict_w_tuples_custom_time / k, 4)
    set_f_dict_w_lists_custom_time = round(set_f_dict_w_lists_custom_time / k, 4)
    frozenset_f_dict_w_lists_custom_time = round(frozenset_f_dict_w_lists_custom_time / k, 4)

    print(f"Set from dict w tuples custom time: {set_f_dict_w_tuples_custom_time}")
    print(f"Frozenset from dict w tuples custom time: {frozenset_f_dict_w_tuples_custom_time}")
    print(f"Set from dict w lists custom time: {set_f_dict_w_lists_custom_time}")
    print(f"Frozenset from dict w lists custom time: {frozenset_f_dict_w_lists_custom_time}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- END OF "set_from_dict" ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


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
    # queue_vs_list_vs_deck()
    # Images into queue time: 6.9286
    # Images into list time: 5.9382
    # Images into list del time: 5.9511
    # Images into deck time: 5.9904

    # pil_vs_cv2_size()
    # list_vs_tuple_generation()
    # columnify_test()  # Broken
    # cached_tuple_vs_list_test()
    # endswith_tuple_vs_split_in_set()
    single_vs_multi_3(n=2)
    # single_vs_multi_2(n=2)
    # single_vs_multi_2_3()
    # list_alike_test()
    # list_alike_test_2()
    # set_vs_frozenset_generation()
    # frozen_set_from_elements()
    # dict_with_list_vs_tuple_vs_set_vs_frozenset()

    # set_from_dict()
    # set_from_dick_w_tuples_simple()
    # set_from_dick_w_tuples_custom()
    # set_from_dict_w_lists_simple()

    # print(list(range(100)))
    # print(tuple(range(100)))
    # print(set(range(100)))
    # print(frozenset(range(100)))

    # docstring_tests()
    ...
