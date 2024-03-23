# coding=utf-8

import cv2
import numpy as np
import queue
import scaler
# import sys
import timeit
import utils

from PIL import Image
from pympler import asizeof
from utils import Algorithms
from utils import algorithm_to_string_dict
from utils import string_to_algorithm_dict


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
    dict_name_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        enum_name_time += timeit.timeit(lambda: Algorithms.CV2_INTER_NEAREST.name, number=n // k)
        dict_name_time += timeit.timeit(lambda: algorithm_to_string_dict[Algorithms.CV2_INTER_NEAREST], number=n // k)
    print()

    enum_name_time = round(enum_name_time / k, 4)
    dict_name_time = round(dict_name_time / k, 4)

    print(f"Enum name: {enum_name_time}")
    print(f"Dict name: {dict_name_time}")
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


def images_into_list(algorithm, factors, image):
    scaled_images = []
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.append(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    # print(f"List bytes: {asizeof.asizeof(scaled_images)}")
    for image in scaled_images:
        pass


def images_into_queue(algorithm, factors, image):
    scaled_images = queue.Queue()
    height, width = image.shape[:2]

    for factor in factors:
        scaled_images.put(cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=algorithm))

    # print(f"Queue bytes: {asizeof.asizeof(scaled_images)}")
    while not scaled_images.empty():
        image = scaled_images.get()


def queue_vs_list(n=5_000, k=10):
    algorithm = cv2.INTER_NEAREST
    factors = [0.125, 0.25, 0.5, 2, 4, 8]
    image = utils.pil_to_cv2(Image.open("../input/NEAREST_NEIGHBOR_pixel-art_0.125x.png"))

    images_into_queue_time = 0
    images_into_list_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        images_into_queue_time += timeit.timeit(lambda: images_into_queue(algorithm, factors, image), number=n // k)
        images_into_list_time += timeit.timeit(lambda: images_into_list(algorithm, factors, image), number=n // k)
    print()

    images_into_queue_time = round(images_into_queue_time / k, 4)
    images_into_list_time = round(images_into_list_time / k, 4)

    print(f"Images into queue time: {images_into_queue_time}")
    print(f"Images into list time: {images_into_list_time}")
    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ END OF "queue_vs_list" ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------


def pil_vs_cv2_size():
    pil_image = Image.open("../input/NEAREST_NEIGHBOR_pixel-art_0.125x.png")
    cv2_image = utils.pil_to_cv2(pil_image)

    print(f"PIL size: {asizeof.asizeof(pil_image)}")
    print(f"CV2 size: {asizeof.asizeof(cv2_image)}")
    # --------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ END OF "pil_vs_cv2_size" ------------------------------------------
    # --------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # test_match_vs_dict()
    # test_custom_any()
    # enum_to_string_test()
    # cv2_vs_pil_test()
    # test_pil_wh_vs_cv2_size()
    queue_vs_list()
    # pil_vs_cv2_size()
    ...
