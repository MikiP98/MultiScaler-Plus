# coding=utf-8
import cv2
import PIL.Image
import utils

from termcolor import colored


def cv2_inter_area_prefix(frames: list[PIL.Image], factor: float, _: dict) -> list[PIL.Image]:
    if factor > 1:
        print(colored(
            f"ERROR: INTER_AREA does not support upscaling! Factor: {factor}; File names will be incorrect!", 'red'
        ))
        return []
    else:
        return cv2_non_ai_common(frames, factor, cv2.INTER_AREA)


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


def cv2_ai_common_scale(
        frames: list[PIL.Image],
        factor: int,
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
        width, height = frame.size

        result = sr.upsample(cv2_image)

        scaled_image.append(
            utils.cv2_to_pil(
                cv2.resize(
                    result, (width * factor, height * factor), interpolation=cv2.INTER_AREA
                )  # TODO: Replace with csatca(fallback_algorithm)
            )
        )

    return scaled_image


def cv2_ai_common(frames: list[PIL.Image], factor: float, name: str, allowed_factors: set) -> list[PIL.Image]:
    if factor < 1:
        print(
            colored(
                "ERROR: CV2 AIs do not support downscaling! "
                f"Cannot perform any fixes! Skipping!",
                'red'
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
        return cv2_ai_common_scale(frames, int(factor), sr, path_prefix, name)
