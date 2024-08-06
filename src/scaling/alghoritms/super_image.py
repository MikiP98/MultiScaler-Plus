# coding=utf-8
import cv2
import io
import PIL.Image
import super_image

from scaling.utils import ConfigPlus, correct_frame
from super_image.modeling_utils import PreTrainedModel as super_image_PreTrainedModel
from typing import Type


def ai_scale(
        frames: list[PIL.Image],
        factor: float,
        high_quality_scale_back: bool,
        allowed_factors: set,
        pretrained_model: Type[super_image_PreTrainedModel],
        pretrained_path: str
) -> list[PIL.Image]:
    current_factor = 1
    scaled_frames = frames.copy()
    while current_factor < factor:
        temp_factor = 4
        if 3 in allowed_factors:
            temp_factor -= 1
            while current_factor * temp_factor >= factor:
                temp_factor -= 1
            temp_factor += 1

        current_factor *= temp_factor

        model = pretrained_model.from_pretrained(pretrained_path, scale=temp_factor)

        for i, frame in enumerate(scaled_frames):
            inputs = super_image.ImageLoader.load_image(frame)
            preds = model(inputs)

            cv2_frame = super_image.ImageLoader._process_image_to_save(preds)

            frame_bytes = cv2.imencode('.png', cv2_frame)[1].tobytes()

            pil_frame = PIL.Image.open(io.BytesIO(frame_bytes))

            scaled_frames[i] = pil_frame

    scaled_frames = [
        correct_frame(scaled_frame, frame.size, factor, high_quality_scale_back)
        for scaled_frame, frame in zip(scaled_frames, frames)
    ]

    model = None  # why it this here? Bug fix? Memory cleaning? TODO: check this
    return scaled_frames


def scale_a2n(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.A2nModel,
        "eugenesiow/a2n"
    )


def scale_awsrn_bam(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.AwsrnModel,
        "eugenesiow/awsrn-bam"
    )


def scale_carn(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.CarnModel,
        "eugenesiow/carn"
    )


def scale_carn_bam(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.CarnModel,
        "eugenesiow/carn-bam"
    )


def scale_drln(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {4},
        super_image.DrlnModel,
        "eugenesiow/drln"
    )


def scale_drln_bam(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.DrlnModel,
        "eugenesiow/drln-bam"
    )


def scale_edsr(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.EdsrModel,
        "eugenesiow/edsr"
    )


def scale_edsr_base(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.EdsrModel,
        "eugenesiow/edsr-base"
    )


def scale_han(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {4},
        super_image.HanModel,
        "eugenesiow/han"
    )


def scale_mdsr(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.MdsrModel,
        "eugenesiow/mdsr"
    )


def scale_mdsr_bam(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.MdsrModel,
        "eugenesiow/mdsr-bam"
    )


def scale_msrn(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.MsrnModel,
        "eugenesiow/msrn"
    )


def scale_msrn_bam(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.MsrnModel,
        "eugenesiow/msrn-bam"
    )


def scale_pan(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.PanModel,
        "eugenesiow/pan"
    )


def scale_pan_bam(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {2, 3, 4},
        super_image.PanModel,
        "eugenesiow/pan-bam"
    )


def scale_rcan_bam(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return ai_scale(
        frames,
        factor,
        config_plus['high_quality_scale_back'],
        {4},
        super_image.RcanModel,
        "eugenesiow/rcan-bam"
    )
