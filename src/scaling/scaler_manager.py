# coding=utf-8
import cv2
import PIL.Image
import utils

from scaling.alghoritms.anime4k import scale as anime4k_scale
from scaling.alghoritms.cv2 import (
    cv2_ai_common,
    cv2_inter_area_prefix,
    cv2_non_ai_common
)
from scaling.alghoritms.docker import scale as docker_scale
from scaling.alghoritms.FidelityFX_CLI import (
    fsr_scale,
    cas_scale
)
from scaling.alghoritms.hqx import scale as hqx_scale
from scaling.alghoritms.HSDBTRE import scale as hsdbtre_scale
from scaling.alghoritms.NEDI import scale as nedi_scale
from scaling.alghoritms.pil import (
    pil_scale_bicubic,
    pil_scale_bilinear,
    pil_scale_lanczos,
    pil_scale_nearest_neighbor
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
from scaling.utils import Algorithms


scaling_functions = {
    # PIL classic algorithms
    Algorithms.PIL_BILINEAR: pil_scale_bilinear,
    Algorithms.PIL_BICUBIC: pil_scale_bicubic,
    Algorithms.PIL_LANCZOS: pil_scale_lanczos,
    Algorithms.PIL_NEAREST_NEIGHBOR: pil_scale_nearest_neighbor,

    # CV2 classic algorithms
    Algorithms.CV2_INTER_AREA: cv2_inter_area_prefix,
    Algorithms.CV2_INTER_CUBIC: lambda frames, factor, _: cv2_non_ai_common(frames, factor, cv2.INTER_CUBIC),
    Algorithms.CV2_INTER_LANCZOS4: lambda frames, factor, _: cv2_non_ai_common(frames, factor, cv2.INTER_LANCZOS4),
    Algorithms.CV2_INTER_LINEAR: lambda frames, factor, _: cv2_non_ai_common(frames, factor, cv2.INTER_LINEAR),
    Algorithms.CV2_INTER_NEAREST: lambda frames, factor, _: cv2_non_ai_common(frames, factor, cv2.INTER_NEAREST),

    # CV2 AI algorithms
    Algorithms.CV2_EDSR: lambda frames, factor, _: cv2_ai_common(frames, factor, "EDSR", {2, 3, 4}),
    Algorithms.CV2_ESPCN: lambda frames, factor, _: cv2_ai_common(frames, factor, "ESPCN", {2, 3, 4}),
    Algorithms.CV2_FSRCNN: lambda frames, factor, _: cv2_ai_common(frames, factor, "FSRCNN", {2, 3, 4}),
    Algorithms.CV2_FSRCNN_small: lambda frames, _, factor: cv2_ai_common(frames, factor, "FSRCNN_small", {2, 3, 4}),
    Algorithms.CV2_LapSRN: lambda frames, factor, _: cv2_ai_common(frames, factor, "LapSRN", {2, 4, 8}),  # 248

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


# TODO: Replace all images generated with CV2 with PIL images if there are no duplicates, as PIL algorithms seem to be higher quality (but slower)
def scale_image_batch(
        algorithms: list[Algorithms],
        images: list[utils.ImageDict],
        factors,
        *,
        fallback_algorithm=Algorithms.CV2_INTER_AREA,
        config_plus=None
) -> list[list[utils.ImageDict]]:

    scaled_images: list[list[utils.ImageDict]] = []

    for algorithm in algorithms:
        scaling_function = scaling_functions[algorithm]
        if scaling_function is None:
            raise ValueError(f"Filter {algorithm.name} (ID: {algorithm}) is not implemented")

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
        factor: int,
        *,
        fallback_algorithm=Algorithms.CV2_INTER_AREA,
        config_plus=None
) -> utils.ImageDict:
    return scale_image_batch(
        [algorithm],
        [image],
        [factor],
        fallback_algorithm=fallback_algorithm,
        config_plus=config_plus
    )[0][0]
