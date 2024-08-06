# coding=utf-8
import cv2
import PIL.Image
import utils

from scaling.utils import ConfigPlus, correct_frame_from_cv2
from termcolor import colored


def cv2_non_ai_common(frames: list[PIL.Image], factor: float, algorithm: int) -> list[PIL.Image]:
    scaled_frames = []
    for frame in frames:
        cv2_image = utils.pil_to_cv2(frame)
        width, height = frame.size

        output_width, output_height = round(width * factor), round(height * factor)

        scaled_frames.append(
            utils.cv2_to_pil(cv2.resize(cv2_image, (output_width, output_height), interpolation=algorithm))
        )

    return scaled_frames


def scale_inter_area(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    if factor > 1:
        print(colored(
            f"WARNING: INTER_AREA does not support upscaling! Factor: {factor}; Skipping!", 'yellow'
        ))
        return []
    else:
        return cv2_non_ai_common(frames, factor, cv2.INTER_AREA)


def scale_inter_cubic(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    return cv2_non_ai_common(frames, factor, cv2.INTER_CUBIC)


def scale_inter_lanczos4(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    return cv2_non_ai_common(frames, factor, cv2.INTER_LANCZOS4)


def scale_inter_linear(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    return cv2_non_ai_common(frames, factor, cv2.INTER_LINEAR)


def scale_inter_nearest(frames: list[PIL.Image], factor: float, _: ConfigPlus) -> list[PIL.Image]:
    return cv2_non_ai_common(frames, factor, cv2.INTER_NEAREST)


def cv2_ai_common_scale(
        frames: list[PIL.Image],
        factor: int,
        high_quality_scale_back: bool,
        sr: cv2.dnn_superres.DnnSuperResImpl,
        path_prefix: str,
        name: str
) -> list[PIL.Image.Image]:
    path = f"{path_prefix}_x{int(factor)}.pb"

    sr.readModel(path)
    sr.setModel(name.lower().split('_')[0], factor)

    scaled_image = []
    for frame in frames:
        cv2_image = utils.pil_to_cv2(frame.convert('RGB'))
        original_size = frame.size

        result = sr.upsample(cv2_image)

        scaled_image.append(
            correct_frame_from_cv2(result, original_size, factor, high_quality_scale_back)
        )

    return scaled_image


def cv2_ai_common(
        frames: list[PIL.Image],
        factor: float,
        config_plus: ConfigPlus,
        name: str,
        allowed_factors: set
) -> list[PIL.Image]:
    if factor < 1:
        print(
            colored(
                "WARNING: CV2 AIs do not support downscaling! Skipping!",
                'yellow'
            )
        )
        return []

    sr = cv2.dnn_superres.DnnSuperResImpl_create()  # Ignore the warning, works fine

    # name = algorithm.name[4:]
    path_prefix = f"./weights/{name}/{name}"

    if factor not in allowed_factors:
        # raise ValueError("INTER_AREA does not support upscaling!")
        print(
            colored(
                f"Warning: CV2 AI 'CV2_{name}' does not support factor: {factor}! "
                f"Allowed factors: {allowed_factors}; Result might be blurry!",
                'yellow'
            )
        )

        # min_allowed_factor = min(allowed_factors)
        max_allowed_factor = max(allowed_factors)
        scaled_frames = []
        for frame in frames:
            width, height = frame.size
            result = utils.pil_to_cv2(frame.convert('RGB'))
            current_factor = 1
            while current_factor < factor:
                temp_factor = max_allowed_factor
                while current_factor * temp_factor >= factor:
                    temp_factor -= 1
                while temp_factor not in allowed_factors:
                    temp_factor += 1

                current_factor *= temp_factor

                path = f"{path_prefix}_x{temp_factor}.pb"
                # print(f"Path: {path}")

                sr.readModel(path)
                sr.setModel(name.lower(), temp_factor)

                result = sr.upsample(result)

            scaled_frames.append(
                utils.cv2_to_pil(
                    cv2.resize(
                        # TODO: Replace with csatca(fallback_algorithm)
                        result, (width * factor, height * factor), interpolation=cv2.INTER_AREA
                    )
                )
            )

        return scaled_frames

    else:
        return cv2_ai_common_scale(frames, int(factor), config_plus['high_quality_scale_back'], sr, path_prefix, name)


def ai_scale_edsr(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return cv2_ai_common(frames, factor, config_plus, "EDSR", {2, 3, 4})


def ai_scale_espcn(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return cv2_ai_common(frames, factor, config_plus, "ESPCN", {2, 3, 4})


def ai_scale_fsrcnn(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return cv2_ai_common(frames, factor, config_plus, "FSRCNN", {2, 3, 4})


def ai_scale_fsrcnn_small(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return cv2_ai_common(frames, factor, config_plus, "FSRCNN_small", {2, 3, 4})


def ai_scale_lapsrn(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    return cv2_ai_common(frames, factor, config_plus, "LapSRN", {2, 4, 8})  # 248
