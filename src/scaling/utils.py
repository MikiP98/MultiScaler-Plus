# coding=utf-8
import cv2
import numpy as np
import PIL.Image
import utils

from aenum import auto, IntEnum, unique
from typing import TypedDict


# Enum with all available algorithms
# Ordered alphabetically
@unique
class Algorithms(IntEnum):
    CPP_DEBUG = -1

    Anime4K = auto()
    CAS = auto()  # contrast adaptive sharpening TODO: fix and move to virtual RAM drive
    CV2_INTER_AREA = auto()  # resampling using pixel area relation
    CV2_INTER_CUBIC = auto()  # bicubic interpolation over 4x4 pixel neighborhood
    CV2_INTER_LANCZOS4 = auto()  # Lanczos interpolation over 8x8 pixel neighborhood
    CV2_INTER_LINEAR = auto()  # bilinear interpolation
    CV2_INTER_NEAREST = auto()  # nearest-neighbor interpolation
    CV2_EDSR = auto()  # Enhanced Deep Super-Resolution
    CV2_ESPCN = auto()  # Efficient Sub-Pixel Convolutional Neural Network
    CV2_FSRCNN = auto()  # Fast Super-Resolution Convolutional Neural Network
    CV2_FSRCNN_small = auto()  # Fast Super-Resolution Convolutional Neural Network - Small
    CV2_LapSRN = auto()  # Laplacian Super-Resolution Network
    FSR = auto()  # FidelityFX Super Resolution TODO: fix and move to virtual RAM drive
    hqx = auto()  # high quality scale

    HSDBTRE = auto()

    NEDI = auto()  # New Edge-Directed Interpolation
    PIL_BICUBIC = auto()  # less blur and artifacts than bilinear, but slower
    PIL_BILINEAR = auto()
    PIL_LANCZOS = auto()  # less blur than bicubic, but artifacts may appear
    PIL_NEAREST_NEIGHBOR = auto()
    RealESRGAN = auto()
    Repetition = auto()

    SI_drln_bam = auto()
    SI_edsr = auto()
    SI_msrn = auto()
    SI_mdsr = auto()
    SI_msrn_bam = auto()
    SI_edsr_base = auto()
    SI_mdsr_bam = auto()
    SI_awsrn_bam = auto()
    SI_a2n = auto()
    SI_carn = auto()
    SI_carn_bam = auto()
    SI_pan = auto()
    SI_pan_bam = auto()

    SI_drln = auto()
    SI_han = auto()
    SI_rcan_bam = auto()

    Super_xBR = auto()
    xBRZ = auto()

    # Docker start
    SUPIR = auto()
    Waifu2x = auto()


class ConfigPlus(TypedDict):
    high_quality_scale_back: bool
    fallback_algorithm: Algorithms

    # prevents multi-face (in 1 image) textures to expand over current textures border
    texture_outbound_protection: bool
    # prevents multi-face (in 1 image) textures to not fully cover current textures border
    texture_inbound_protection: bool
    # What should be used to make the mask, 1st is when alpha is present, 2nd when it is not  TODO: add more options
    texture_mask_mode: tuple[str]
    # if true, the alpha channel will be equal to 255 or alpha will be deleted
    disallow_partial_transparency: bool

    try_to_fix_texture_tiling: bool
    tiling_fix_quality: float

    sharpness: float
    NEDI_m: int
    offset_x: float
    offset_y: float


def correct_frame(frame: PIL.Image, original_size: tuple, factor: float, high_quality_scaleback: bool) -> PIL.Image:
    correct_size = (round(original_size[0] * factor), round(original_size[1] * factor))
    if high_quality_scaleback:
        return frame.resize(correct_size, PIL.Image.LANCZOS)
    else:
        return utils.cv2_to_pil(
            cv2.resize(
                utils.pil_to_cv2(frame),
                correct_size,
                interpolation=cv2.INTER_AREA
            )
        )


def correct_frame_from_cv2(frame: np.array, original_size: tuple, factor: float, high_quality_scaleback: bool) -> PIL.Image:
    correct_size = (round(original_size[0] * factor), round(original_size[1] * factor))
    if high_quality_scaleback:
        return utils.cv2_to_pil(frame).resize(correct_size, PIL.Image.LANCZOS)
    else:
        return utils.cv2_to_pil(
            cv2.resize(
                frame,
                correct_size,
                interpolation=cv2.INTER_AREA
            )
        )


# # scaler_algorithm_to_pillow_algorithm_dictionary
# satpad = {
#     Algorithms.PIL_BICUBIC: PIL.Image.BICUBIC,
#     Algorithms.PIL_BILINEAR: PIL.Image.BILINEAR,
#     Algorithms.PIL_LANCZOS: PIL.Image.LANCZOS,
#     Algorithms.PIL_NEAREST_NEIGHBOR: PIL.Image.NEAREST
# }
#
#
# # convert_scaler_algorithm_to_pillow_algorithm
# def csatpa(algorithm: Algorithms):
#     """
#     ConvertScalerAlgorithmToPillowAlgorithm()\n
#     Converts a scaler algorithm to a PIL algorithm using a dictionary (satpad)
#     :param algorithm: The Scaler algorithm to convert
#     :return: The corresponding PIL algorithm
#     :raises AttributeError: If the algorithm is not supported by PIL
#     """
#     pillow_algorithm = satpad.get(algorithm)
#     if pillow_algorithm is not None:
#         return pillow_algorithm
#     else:
#         raise AttributeError("Algorithm not supported by PIL")
#
#
# # scaler_algorithm_to_cv2_algorithm_dictionary
# satcad = {
#     Algorithms.CV2_INTER_AREA: cv2.INTER_AREA,
#     Algorithms.CV2_INTER_CUBIC: cv2.INTER_CUBIC,
#     Algorithms.CV2_INTER_LANCZOS4: cv2.INTER_LANCZOS4,
#     Algorithms.CV2_INTER_LINEAR: cv2.INTER_LINEAR,
#     Algorithms.CV2_INTER_NEAREST: cv2.INTER_NEAREST
# }
#
#
# # convert_scaler_algorithm_to_cv2_algorithm
# def csatca(algorithm: Algorithms):
#     """
#     ConvertScalerAlgorithmToCV2Algorithm()\n
#     Converts a scaler algorithm to a OpenCV algorithm using a dictionary (satcad)
#     :param algorithm: The Scaler algorithm to convert
#     :return: The corresponding OpenCV algorithm
#     :raises AttributeError: If the algorithm is not supported by OpenCV
#     """
#     algorithm_cv2 = satcad.get(algorithm)
#     if algorithm_cv2 is not None:
#         return algorithm_cv2
#     else:
#         raise AttributeError(
#             f"Algorithm not supported by OpenCV: {utils.algorithm_to_string(algorithm)},  id: {algorithm}"
#         )
