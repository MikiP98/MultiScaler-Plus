# coding=utf-8

import numpy as np
import scaler
import timeit
import utils

from PIL import Image
from utils import Algorithms


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


string_to_algorithm_dict = {
    "cv2_area": Algorithms.CV2_INTER_AREA,
    "cv2_bicubic": Algorithms.CV2_INTER_CUBIC,
    "cv2_bilinear": Algorithms.CV2_INTER_LINEAR,
    "cv2_lanczos": Algorithms.CV2_INTER_LANCZOS4,
    "cv2_nearest": Algorithms.CV2_INTER_NEAREST,

    "cv2_edsr": Algorithms.CV2_EDSR,
    "cv2_espcn": Algorithms.CV2_ESPCN,
    "cv2_fsrcnn": Algorithms.CV2_FSRCNN,
    "cv2_lapsrn": Algorithms.CV2_LapSRN,

    "pil_bicubic": Algorithms.PIL_BICUBIC,
    "pil_bilinear": Algorithms.PIL_BILINEAR,
    "pil_lanczos": Algorithms.PIL_LANCZOS,
    "pil_nearest": Algorithms.PIL_NEAREST_NEIGHBOR,

    "cas": Algorithms.CAS,
    "fsr": Algorithms.FSR,
    "real_esrgan": Algorithms.RealESRGAN,
    "supir": Algorithms.SUPIR,
    "xbrz": Algorithms.xBRZ
}


def test_match_vs_dict(n=10_000_000, k=10):
    best_using_match = 0
    best_using_dict = 0
    worst_using_match = 0
    worst_using_dict = 0
    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        best_using_match += timeit.timeit(lambda: string_to_algorithm_match("bicubic"), number=n // k)
        best_using_dict += timeit.timeit(lambda: string_to_algorithm_dict["bicubic"], number=n // k)
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


def test_custom_any(n=100_000, k=10):
    image = Image.open("../input/NEAREST_NEIGHBOR_pixel-art_0.125x.png").convert("RGBA")
    image_array = list(image.getdata())
    numpy_image_array = utils.pil_to_cv2(image)

    # custom_any_time = 0
    # custom_any_2_time = 0
    # normal_any_time = 0
    # extrema_any_time = 0
    numpy_any_time = 0
    # numpy_any_2_time = 0
    numpy_conversion_any_time = 0
    numpy_conversion_any_safe_time = 0
    numpy_conversion_any_safe_2_time = 0

    for i in range(k):
        print(f"Iteration {i + 1}/{k}")
        # custom_any_time += timeit.timeit(lambda: custom_any(image_array), number=n // k)
        # custom_any_2_time += timeit.timeit(lambda: custom_any_2(image_array), number=n // k)
        # normal_any_time += timeit.timeit(lambda: normal_any(image_array), number=n // k)
        # extrema_any_time += timeit.timeit(lambda: extrema_any(image), number=n // k)
        numpy_any_time += timeit.timeit(lambda: numpy_any(numpy_image_array), number=n // k)
        # numpy_any_2_time += timeit.timeit(lambda: numpy_any_2(numpy_image_array), number=n // k)
        numpy_conversion_any_time += timeit.timeit(lambda: numpy_conversion_any(image), number=n // k)
        numpy_conversion_any_safe_time += timeit.timeit(lambda: numpy_conversion_any_safe(image), number=n // k)
        numpy_conversion_any_safe_2_time += timeit.timeit(lambda: numpy_conversion_any_safe_2(image), number=n // k)
    print()

    # custom_any_time = round(custom_any_time / k, 4)
    # custom_any_2_time = round(custom_any_2_time / k, 4)
    # normal_any_time = round(normal_any_time / k, 4)
    # extrema_any_time = round(extrema_any_time / k, 4)
    numpy_any_time = round(numpy_any_time / k, 4)
    # numpy_any_2_time = round(numpy_any_2_time / k, 4)
    numpy_conversion_any_time = round(numpy_conversion_any_time / k, 4)
    numpy_conversion_any_safe_time = round(numpy_conversion_any_safe_time / k, 4)
    numpy_conversion_any_safe_2_time = round(numpy_conversion_any_safe_2_time / k, 4)

    # print(f"Custom any-1: {custom_any_time}")
    # print(f"Custom any-2: {custom_any_2_time}")
    # print(f"Normal any: {normal_any_time}")
    # print(f"Extrema any: {extrema_any_time}")
    print(f"Numpy any: {numpy_any_time}")
    # print(f"Numpy any-2: {numpy_any_2_time}")
    print(f"Numpy conversion any: {numpy_conversion_any_time}")
    print(f"Numpy conversion any safe: {numpy_conversion_any_safe_time}")
    print(f"Numpy conversion any safe-2: {numpy_conversion_any_safe_2_time}")


algorithm_to_string_dict = {
    Algorithms.CV2_INTER_AREA: "cv2_area",
    Algorithms.CV2_INTER_CUBIC: "cv2_bicubic",
    Algorithms.CV2_INTER_LINEAR: "cv2_bilinear",
    Algorithms.CV2_INTER_LANCZOS4: "cv2_lanczos",
    Algorithms.CV2_INTER_NEAREST: "cv2_nearest",

    Algorithms.CV2_EDSR: "cv2_edsr",
    Algorithms.CV2_ESPCN: "cv2_espcn",
    Algorithms.CV2_FSRCNN: "cv2_fsrcnn",
    Algorithms.CV2_LapSRN: "cv2_lapsrn",

    Algorithms.PIL_BICUBIC: "pil_bicubic",
    Algorithms.PIL_BILINEAR: "pil_bilinear",
    Algorithms.PIL_LANCZOS: "pil_lanczos",
    Algorithms.PIL_NEAREST_NEIGHBOR: "pil_nearest",

    Algorithms.CAS: "cas",
    Algorithms.FSR: "fsr",
    Algorithms.RealESRGAN: "real_esrgan",
    Algorithms.SUPIR: "supir",
    Algorithms.xBRZ: "xbrz"
}


def enum_to_string_test(n=10_000_000, k=10):
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


def cv2_vs_pil_test(n=200, k=10):
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


if __name__ == "__main__":
    # test_match_vs_dict()
    # test_custom_any()
    # enum_to_string_test()
    cv2_vs_pil_test()
