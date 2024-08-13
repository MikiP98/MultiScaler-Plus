# coding=utf-8
import PIL.Image
import utils

from aenum import auto
from scaling.alghoritms.Anime4K import scale as anime4k_scale
from scaling.alghoritms.CV2 import (
    ai_scale_edsr,
    ai_scale_espcn,
    ai_scale_fsrcnn,
    ai_scale_fsrcnn_small,
    ai_scale_lapsrn,
    scale_inter_area as cv2_inter_area,
    scale_inter_cubic as cv2_inter_cubic,
    scale_inter_lanczos4 as cv2_inter_lanczos4,
    scale_inter_linear as cv2_inter_linear,
    scale_inter_nearest as cv2_inter_nearest,
)
from scaling.alghoritms.docker import scale as docker_scale
from scaling.alghoritms.FidelityFX_CLI import (
    fsr_scale,
    cas_scale
)
from scaling.alghoritms.hqx import scale as hqx_scale
from scaling.alghoritms.HSDBTRE import scale as hsdbtre_scale
from scaling.alghoritms.NEDI import scale as nedi_scale
from scaling.alghoritms.PIL import (
    scale_bicubic as pil_scale_bicubic,
    scale_bilinear as pil_scale_bilinear,
    scale_box as pil_scale_box,
    scale_hamming as pil_scale_hamming,
    scale_lanczos as pil_scale_lanczos,
    scale_nearest_neighbor as pil_scale_nearest_neighbor
)
from scaling.alghoritms.RealESRGAN import scale as real_esrgan_scale
from scaling.alghoritms.Repetition import scale as repetition_scale
from scaling.alghoritms.super_image import (
    scale_a2n as si_scale_a2n,
    scale_awsrn_bam as si_scale_awsrn_bam,
    scale_carn as si_scale_carn,
    scale_carn_bam as si_scale_carn_bam,
    scale_drln as si_scale_drln,
    scale_drln_bam as si_scale_drln_bam,
    scale_edsr as si_scale_edsr,
    scale_edsr_base as si_scale_edsr_base,
    scale_han as si_scale_han,
    scale_mdsr as si_scale_mdsr,
    scale_mdsr_bam as si_scale_mdsr_bam,
    scale_msrn as si_scale_msrn,
    scale_msrn_bam as si_scale_msrn_bam,
    scale_pan as si_scale_pan,
    scale_pan_bam as si_scale_pan_bam,
    scale_rcan_bam as si_scale_rcan_bam
)
from scaling.alghoritms.Super_xBR import scale as super_xbr_scale
from scaling.alghoritms.xBRZ import scale as xbrz_scale
from scaling.utils import Algorithms, ConfigPlus
from typing import Callable


scaling_functions: dict[auto, Callable[[list[PIL.Image.Image], float, ConfigPlus], list[PIL.Image.Image]]] = {
    # PIL classic algorithms
    Algorithms.PIL_BILINEAR: pil_scale_bilinear,
    Algorithms.PIL_BICUBIC: pil_scale_bicubic,
    Algorithms.PIL_BOX: pil_scale_box,  # TODO: test
    Algorithms.PIL_HAMMING: pil_scale_hamming,  # TODO: test
    Algorithms.PIL_LANCZOS: pil_scale_lanczos,
    Algorithms.PIL_NEAREST_NEIGHBOR: pil_scale_nearest_neighbor,

    # CV2 classic algorithms
    Algorithms.CV2_INTER_AREA: cv2_inter_area,
    Algorithms.CV2_INTER_CUBIC: cv2_inter_cubic,
    Algorithms.CV2_INTER_LANCZOS4: cv2_inter_lanczos4,
    Algorithms.CV2_INTER_LINEAR: cv2_inter_linear,
    Algorithms.CV2_INTER_NEAREST: cv2_inter_nearest,

    # CV2 AI algorithms
    Algorithms.CV2_EDSR: ai_scale_edsr,
    Algorithms.CV2_ESPCN: ai_scale_espcn,
    Algorithms.CV2_FSRCNN: ai_scale_fsrcnn,
    Algorithms.CV2_FSRCNN_small: ai_scale_fsrcnn_small,
    Algorithms.CV2_LapSRN: ai_scale_lapsrn,  # allowed factors: 2, 4, 8

    # Super Image AI algorithms
    Algorithms.SI_a2n: si_scale_a2n,
    Algorithms.SI_awsrn_bam: si_scale_awsrn_bam,
    Algorithms.SI_carn: si_scale_carn,
    Algorithms.SI_carn_bam: si_scale_carn_bam,
    Algorithms.SI_drln: si_scale_drln,
    Algorithms.SI_drln_bam: si_scale_drln_bam,
    Algorithms.SI_edsr: si_scale_edsr,
    Algorithms.SI_edsr_base: si_scale_edsr_base,
    Algorithms.SI_han: si_scale_han,  # 4x only
    Algorithms.SI_mdsr: si_scale_mdsr,
    Algorithms.SI_mdsr_bam: si_scale_mdsr_bam,
    Algorithms.SI_msrn: si_scale_msrn,
    Algorithms.SI_msrn_bam: si_scale_msrn_bam,
    Algorithms.SI_pan: si_scale_pan,
    Algorithms.SI_pan_bam: si_scale_pan_bam,
    Algorithms.SI_rcan_bam: si_scale_rcan_bam,  # 4x only

    # Other AI algorithms
    Algorithms.Anime4K: anime4k_scale,
    Algorithms.HSDBTRE: hsdbtre_scale,  # hybrid
    Algorithms.RealESRGAN: real_esrgan_scale,
    Algorithms.SUPIR: docker_scale,  # docker
    Algorithms.Waifu2x: docker_scale,  # docker

    # Other classic algorithms
    Algorithms.Repetition: repetition_scale,

    # Edge detection algorithms
    Algorithms.hqx: hqx_scale,
    Algorithms.NEDI: nedi_scale,
    Algorithms.Super_xBR: super_xbr_scale,
    Algorithms.xBRZ: xbrz_scale,

    # Smart algorithms
    Algorithms.FSR: fsr_scale,
    Algorithms.CAS: cas_scale
}


# TODO: Replace all images generated with CV2 with PIL images if there are no duplicates,
#  as PIL algorithms seem to be higher quality (but slower)
def scale_image_batch(
        algorithms: list[Algorithms],
        images: list[utils.ImageDict],
        factors: list[float],
        *,
        config_plus: ConfigPlus = None
) -> list[list[utils.ImageDict]]:

    scaled_images: list[list[utils.ImageDict]] = []

    for algorithm in algorithms:
        scaling_function = scaling_functions[algorithm]
        # if scaling_function is None:
        #     raise ValueError(f"Filter {algorithm.name} (ID: {algorithm}) is not implemented")

        scaled_images.append([
            {
                "images": [scaling_function(image["images"][0], factor, config_plus) for factor in factors],
                "is_animated": image.get("is_animated"),
                "animation_spacing": image.get("animation_spacing"),
            } for image in images
        ])

    return scaled_images


def scale_image(
        algorithm: Algorithms,
        image: PIL.Image,
        factor: float,
        *,
        config_plus: ConfigPlus = None
) -> utils.ImageDict:
    return scale_image_batch(
        [algorithm],
        [image],
        [factor],
        config_plus=config_plus
    )[0][0]
