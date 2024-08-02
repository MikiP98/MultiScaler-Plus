# coding=utf-8
import io
import subprocess
import PIL.Image
import cv2
import hqx
import numpy as np
import pyanime4k.ac
import super_image
import torch
import xbrz  # See xBRZ scaling on Jira

from RealESRGAN import RealESRGAN
from super_image.modeling_utils import PreTrainedModel as super_image_PreTrainedModel
from superxbr import superxbr  # Ignore the error, it works fine
from termcolor import colored
from typing import Type

import utils
# EDI_predict is wierd, EDI_Downscale is nearest neighbor...
from scaling.Edge_Directed_Interpolation.edi import EDI_upscale
from utils import Algorithms

# import scalercg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# scaler_algorithm_to_pillow_algorithm_dictionary
satpad = {
    Algorithms.PIL_BICUBIC: PIL.Image.BICUBIC,
    Algorithms.PIL_BILINEAR: PIL.Image.BILINEAR,
    Algorithms.PIL_LANCZOS: PIL.Image.LANCZOS,
    Algorithms.PIL_NEAREST_NEIGHBOR: PIL.Image.NEAREST
}


# convert_scaler_algorithm_to_pillow_algorithm
def csatpa(algorithm: Algorithms):
    """
    ConvertScalerAlgorithmToPillowAlgorithm()\n
    Converts a scaler algorithm to a PIL algorithm using a dictionary (satpad)
    :param algorithm: The Scaler algorithm to convert
    :return: The corresponding PIL algorithm
    :raises AttributeError: If the algorithm is not supported by PIL
    """
    pillow_algorithm = satpad.get(algorithm)
    if pillow_algorithm is not None:
        return pillow_algorithm
    else:
        raise AttributeError("Algorithm not supported by PIL")


# scaler_algorithm_to_cv2_algorithm_dictionary
satcad = {
    Algorithms.CV2_INTER_AREA: cv2.INTER_AREA,
    Algorithms.CV2_INTER_CUBIC: cv2.INTER_CUBIC,
    Algorithms.CV2_INTER_LANCZOS4: cv2.INTER_LANCZOS4,
    Algorithms.CV2_INTER_LINEAR: cv2.INTER_LINEAR,
    Algorithms.CV2_INTER_NEAREST: cv2.INTER_NEAREST
}


# convert_scaler_algorithm_to_cv2_algorithm
def csatca(algorithm: Algorithms):
    """
    ConvertScalerAlgorithmToCV2Algorithm()\n
    Converts a scaler algorithm to a OpenCV algorithm using a dictionary (satcad)
    :param algorithm: The Scaler algorithm to convert
    :return: The corresponding OpenCV algorithm
    :raises AttributeError: If the algorithm is not supported by OpenCV
    """
    algorithm_cv2 = satcad.get(algorithm)
    if algorithm_cv2 is not None:
        return algorithm_cv2
    else:
        raise AttributeError(
            f"Algorithm not supported by OpenCV: {utils.algorithm_to_string(algorithm)},  id: {algorithm}"
        )


def scale_image(
        algorithm: Algorithms,
        image: PIL.Image,
        factor: int,
        *,
        fallback_algorithm=Algorithms.CV2_INTER_AREA,
        config_plus=None
) -> PIL.Image:
    return scale_image_batch(
        algorithm,
        [image],
        [factor],
        fallback_algorithm=fallback_algorithm,
        config_plus=config_plus
    ).pop()


# ud - upscale/downscale
cv2_algorithms_ud = {
    Algorithms.CV2_INTER_CUBIC, Algorithms.CV2_INTER_LANCZOS4, Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_NEAREST
}
pil_algorithms_ud = {
    Algorithms.PIL_NEAREST_NEIGHBOR, Algorithms.PIL_BILINEAR, Algorithms.PIL_BICUBIC, Algorithms.PIL_LANCZOS
}

cv2_ai_234 = {Algorithms.CV2_EDSR, Algorithms.CV2_ESPCN, Algorithms.CV2_FSRCNN, Algorithms.CV2_FSRCNN_small}
cv2_ai_248 = {Algorithms.CV2_LapSRN}

si_2x_3x_4x_algorithms = {
    Algorithms.SI_drln_bam,
    Algorithms.SI_edsr,
    Algorithms.SI_msrn,
    Algorithms.SI_mdsr,
    Algorithms.SI_msrn_bam,
    Algorithms.SI_edsr_base,
    Algorithms.SI_mdsr_bam,
    Algorithms.SI_awsrn_bam,
    Algorithms.SI_a2n,
    Algorithms.SI_carn,
    Algorithms.SI_carn_bam,
    Algorithms.SI_pan,
    Algorithms.SI_pan_bam,
}
si_4x_algorithms = {
    Algorithms.SI_drln,
    Algorithms.SI_han,
    Algorithms.SI_rcan_bam
}
si_algorithms = si_2x_3x_4x_algorithms.union(si_4x_algorithms)


def cv2_inter_area_prefix(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    if factor > 1:
        # raise ValueError("INTER_AREA does not support upscaling!")
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
                    result, (width * factor, height * factor), interpolation=cv2.INTER_AREA)  # TODO: Replace with csatca(fallback_algorithm)
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


def pil_scale(frames: list[PIL.Image], factor: float, algorithm: PIL.Image) -> list[PIL.Image]:
    scaled_frames = []
    for frame in frames:
        width, height = frame.size
        output_width, output_height = round(width * factor), round(height * factor)

        scaled_frames.append(frame.resize((output_width, output_height), algorithm))

    return scaled_frames


def hsdbtre_scale(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    if factor < 1:
        print(
            colored(
                "ERROR: HSDBTRE is an AI algorithm and does not support downscaling! "
                f"Cannot perform any fixes! Skipping!",
                'red'
            )
        )
        return []

    repeat = 1
    while 4 ** repeat < factor:
        repeat += 1
    # print(f"Repeats: {repeats}")

    scaled_frames = frames.copy()
    for _ in range(repeat):
        scaled_frames = si_ai_scale(scaled_frames, 2, {2}, super_image.DrlnModel, "eugenesiow/drln-bam")
        scaled_frames = real_esrgan_scale(scaled_frames, 2)
        # scaled_frames = scale_image_batch(
        #     Algorithms.SI_drln_bam, scaled_frames, [2], fallback_algorithm=fallback_algorithm
        # )
        # scaled_frames = scale_image_batch(
        #     Algorithms.RealESRGAN, scaled_frames, [2], fallback_algorithm=fallback_algorithm
        # )

    scaled_frames = [
        utils.cv2_to_pil(
            cv2.resize(
                utils.pil_to_cv2(scaled_frame),
                (
                    round(frame.size[0] * factor),
                    round(frame.size[1] * factor)
                ),
                interpolation=cv2.INTER_AREA  # TODO: Replace with csatca(fallback_algorithm)
            )
        )
        for scaled_frame, frame in zip(scaled_frames, frames)
    ]

    return scaled_frames


def si_ai_scale(frames: list[PIL.Image], factor: float, allowed_factors: set, pretrained_model: Type[super_image_PreTrainedModel], pretrained_path: str) -> list[PIL.Image]:
    current_factor = 1
    # temp_factor = factor
    scaled_frames = frames.copy()
    while current_factor < factor:
        temp_factor = 4
        if 3 in allowed_factors:
            temp_factor -= 1
            while current_factor * temp_factor >= factor:
                temp_factor -= 1
            temp_factor += 1

        current_factor *= temp_factor

        # if factor not in allowed_factors:
        #     raise ValueError("SI does not support this factor!")

        model = pretrained_model.from_pretrained(pretrained_path, scale=temp_factor)

        # if algorithm == Algorithms.SI_a2n:
        #     model = super_image.A2nModel.from_pretrained('eugenesiow/a2n', scale=temp_factor)
        # elif algorithm == Algorithms.SI_awsrn_bam:
        #     model = super_image.AwsrnModel.from_pretrained('eugenesiow/awsrn-bam', scale=temp_factor)
        # elif algorithm == Algorithms.SI_carn or algorithm == Algorithms.SI_carn_bam:
        #     if algorithm == Algorithms.SI_carn:
        #         model = super_image.CarnModel.from_pretrained('eugenesiow/carn', scale=temp_factor)
        #     else:
        #         model = super_image.CarnModel.from_pretrained('eugenesiow/carn-bam', scale=temp_factor)
        # elif algorithm == Algorithms.SI_drln or algorithm == Algorithms.SI_drln_bam:
        #     if algorithm == Algorithms.SI_drln:
        #         model = super_image.DrlnModel.from_pretrained('eugenesiow/drln', scale=temp_factor)
        #     else:
        #         model = super_image.DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=temp_factor)
        # elif algorithm == Algorithms.SI_edsr or algorithm == Algorithms.SI_edsr_base:
        #     if algorithm == Algorithms.SI_edsr:
        #         model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr', scale=temp_factor)
        #     else:
        #         model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=temp_factor)
        # elif algorithm == Algorithms.SI_mdsr or algorithm == Algorithms.SI_mdsr_bam:
        #     if algorithm == Algorithms.SI_mdsr:
        #         model = super_image.MdsrModel.from_pretrained('eugenesiow/mdsr', scale=temp_factor)
        #     else:
        #         model = super_image.MdsrModel.from_pretrained('eugenesiow/mdsr-bam', scale=temp_factor)
        # elif algorithm == Algorithms.SI_msrn or algorithm == Algorithms.SI_msrn_bam:
        #     if algorithm == Algorithms.SI_msrn:
        #         model = super_image.MsrnModel.from_pretrained('eugenesiow/msrn', scale=temp_factor)
        #     else:
        #         model = super_image.MsrnModel.from_pretrained('eugenesiow/msrn-bam', scale=temp_factor)
        # elif algorithm == Algorithms.SI_pan or algorithm == Algorithms.SI_pan_bam:
        #     if algorithm == Algorithms.SI_pan:
        #         model = super_image.PanModel.from_pretrained('eugenesiow/pan', scale=temp_factor)
        #     else:
        #         model = super_image.PanModel.from_pretrained('eugenesiow/pan-bam', scale=temp_factor)
        # elif algorithm == Algorithms.SI_rcan_bam:
        #     model = super_image.RcanModel.from_pretrained('eugenesiow/rcan-bam', scale=temp_factor)
        # elif algorithm == Algorithms.SI_han:
        #     model = super_image.HanModel.from_pretrained('eugenesiow/han', scale=temp_factor)
        # else:
        #     raise ValueError("Unknown SI algorithm! It should not get here... but the warning :/")

        for i, frame in enumerate(scaled_frames):
            inputs = super_image.ImageLoader.load_image(frame)
            preds = model(inputs)

            cv2_frame = super_image.ImageLoader._process_image_to_save(preds)

            frame_bytes = cv2.imencode('.png', cv2_frame)[1].tobytes()

            pil_frame = PIL.Image.open(io.BytesIO(frame_bytes))

            scaled_frames[i] = pil_frame

            # super_image.ImageLoader.save_image(preds, "../input/frame.png")
            # super_image.ImageLoader.save_compare(preds, inputs, "../output/compare.png")

    scaled_frames = [
        utils.cv2_to_pil(
            cv2.resize(
                utils.pil_to_cv2(scaled_frame),
                (
                    round(frame.size[0] * factor),
                    round(frame.size[1] * factor)
                ),
                interpolation=cv2.INTER_AREA  # TODO: Replace with csatca(fallback_algorithm)
            )
        )
        for scaled_frame, frame in zip(scaled_frames, frames)
    ]

    # for i, frame in enumerate(scaled_frames):
    #     print(f"Curr factor: {current_factor}")
    #     scaled_frames[i] = utils.cv2_to_pil(
    #         cv2.resize(
    #             utils.pil_to_cv2(frame),
    #             (frame.size[0] // current_factor * factor, frame.size[1] // current_factor * factor),
    #             interpolation=csatca(fallback_algorithm)
    #         )
    #     )

    model = None  # why it this here? Bug fix? Memory cleaning?
    return scaled_frames


def real_esrgan_scale(frames: list[PIL.Image], factor: float) -> list[PIL.Image]:
    if factor < 1:
        print(
            colored(
                "RealESRGAN AI does not support downscaling!; "
                "Skipping!",
                'red'
            )
        )
        return []

    scaled_frames = []
    for frame in frames:
        width, height = frame.size
        output_width, output_height = round(width * factor), round(height * factor)
        frame = frame.convert('RGB')

        # If factor is not a whole number or is greater than 6, print a warning
        if factor not in (1, 2, 4, 8):
            print(
                colored(
                    "WARNING: Scaling by RealESRGAN with factor {factor} is not supported, "
                    "result might be blurry!",
                    'yellow'
                )
            )

        current_scale = 1
        while current_scale < factor:
            temp_factor = 8
            while current_scale * temp_factor >= factor:
                temp_factor //= 2
            temp_factor = min(temp_factor * 2, 8)

            model = RealESRGAN(device, scale=temp_factor)
            model.load_weights(f'weights/RealESRGAN_x{temp_factor}.pth')  # , download=True
            frame = model.predict(frame)

            current_scale *= temp_factor

        scaled_frames.append(
            utils.cv2_to_pil(
                cv2.resize(
                    utils.pil_to_cv2(frame),
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA  # TODO: Replace with csatca(fallback_algorithm)
                )
            )
        )
    return scaled_frames


scaling_functions = {
    # CV2 classic algorithms
    Algorithms.CV2_INTER_AREA: cv2_inter_area_prefix,
    Algorithms.CV2_INTER_CUBIC: lambda frames, factor: cv2_non_ai_common(frames, factor, cv2.INTER_CUBIC),
    Algorithms.CV2_INTER_LANCZOS4: lambda frames, factor: cv2_non_ai_common(frames, factor, cv2.INTER_LANCZOS4),
    Algorithms.CV2_INTER_LINEAR: lambda frames, factor: cv2_non_ai_common(frames, factor, cv2.INTER_LINEAR),
    Algorithms.CV2_INTER_NEAREST: lambda frames, factor: cv2_non_ai_common(frames, factor, cv2.INTER_NEAREST),

    # CV2 AI algorithms
    Algorithms.CV2_EDSR: lambda frames, factor: cv2_ai_common(frames, factor, "EDSR", {2, 3, 4}),
    Algorithms.CV2_ESPCN: lambda frames, factor: cv2_ai_common(frames, factor, "ESPCN", {2, 3, 4}),
    Algorithms.CV2_FSRCNN: lambda frames, factor: cv2_ai_common(frames, factor, "FSRCNN", {2, 3, 4}),
    Algorithms.CV2_FSRCNN_small: lambda frames, factor: cv2_ai_common(frames, factor, "FSRCNN_small", {2, 3, 4}),
    Algorithms.CV2_LapSRN: lambda frames, factor: cv2_ai_common(frames, factor, "LapSRN", {2, 4, 8}),  # 248

    # PIL algorithms
    Algorithms.PIL_NEAREST_NEIGHBOR: lambda frames, factor: pil_scale(frames, factor, PIL.Image.NEAREST),
    Algorithms.PIL_BILINEAR: lambda frames, factor: pil_scale(frames, factor, PIL.Image.BILINEAR),
    Algorithms.PIL_BICUBIC: lambda frames, factor: pil_scale(frames, factor, PIL.Image.BICUBIC),
    Algorithms.PIL_LANCZOS: lambda frames, factor: pil_scale(frames, factor, PIL.Image.LANCZOS),

    # Super Image AI algorithms
    Algorithms.SI_a2n: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.A2nModel, "eugenesiow/a2n"),
    Algorithms.SI_awsrn_bam: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.AwsrnModel, "eugenesiow/awsrn-bam"),
    Algorithms.SI_carn: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.CarnModel, "eugenesiow/carn"),
    Algorithms.SI_carn_bam: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.CarnModel, "eugenesiow/carn-bam"),
    Algorithms.SI_drln: lambda frames, factor: si_ai_scale(frames, factor, {4}, super_image.DrlnModel, "eugenesiow/drln"),
    Algorithms.SI_drln_bam: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.DrlnModel, "eugenesiow/drln-bam"),
    Algorithms.SI_edsr: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.EdsrModel, "eugenesiow/edsr"),
    Algorithms.SI_edsr_base: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.EdsrModel, "eugenesiow/edsr-base"),
    Algorithms.SI_han: lambda frames, factor: si_ai_scale(frames, factor, {4}, super_image.HanModel, "eugenesiow/han"),  # 4x only
    Algorithms.SI_mdsr: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.MdsrModel, "eugenesiow/mdsr"),
    Algorithms.SI_mdsr_bam: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.MdsrModel, "eugenesiow/mdsr-bam"),
    Algorithms.SI_msrn: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.MsrnModel, "eugenesiow/msrn"),
    Algorithms.SI_msrn_bam: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.MsrnModel, "eugenesiow/msrn-bam"),
    Algorithms.SI_pan: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.PanModel, "eugenesiow/pan"),
    Algorithms.SI_pan_bam: lambda frames, factor: si_ai_scale(frames, factor, {2, 3, 4}, super_image.PanModel, "eugenesiow/pan-bam"),
    Algorithms.SI_rcan_bam: lambda frames, factor: si_ai_scale(frames, factor, {4}, super_image.RcanModel, "eugenesiow/rcan-bam"),  # 4x only

    # Custom algorithms
    Algorithms.HSDBTRE: hsdbtre_scale,
    Algorithms.RealESRGAN: real_esrgan_scale,
    Algorithms.SUPIR: None,
    Algorithms.Waifu2x: None,
}


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
                "images": [scaling_function(image["images"][0], factor) for factor in factors],
                "is_animated": image.get("is_animated"),
                "animation_spacing": image.get("animation_spacing"),
            } for image in images
        ])

    return scaled_images

    # ------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Start of custom algorithms ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------

    if algorithm == Algorithms.Repetition:
        if config_plus is None:
            print(colored("WARNING: config_plus is None! Creating empty config_plus!", 'yellow'))
            config_plus = {}
        if 'offset_x' not in config_plus:
            print(colored("WARNING: offset_x not in config_plus! Using default value: 0", 'yellow'))
            config_plus['offset_x'] = 0
        if 'offset_y' not in config_plus:
            print(colored("WARNING: offset_y not in config_plus! Using default value: 0", 'yellow'))
            config_plus['offset_y'] = 0

        for image_object in images:
            new_image_object_list = []
            for factor in factors:
                scaled_image = []
                for frame in image_object['images'][0]:
                    width, height = frame.size
                    offset_x = round(config_plus['offset_x'] * width)
                    offset_y = round(config_plus['offset_y'] * height)
                    output_width, output_height = round(width * factor), round(height * factor)

                    new_frame = PIL.Image.new(frame.mode, (output_width, output_height))
                    for x in range(output_width):
                        for y in range(output_height):
                            new_frame.putpixel(
                                (x, y), frame.getpixel(((x + offset_x) % width, (y + offset_y) % height))
                            )

                    scaled_image.append(new_frame)
                new_image_object_list.append(scaled_image)
            scaled_images.append({
                'images': new_image_object_list
            })
        return scaled_images

    if algorithm == Algorithms.Waifu2x or algorithm == Algorithms.SUPIR:
        # import docker
        # client = docker.from_env()
        # if algorithm == Algorithms.Waifu2x:
        #     # Define the image name
        #     image_name = 'waifu2x-python:3.11'
        #
        #     # Check if the image exists
        #     try:
        #         image = client.images.get(image_name)
        #         print("Image exists")
        #     except docker.errors.ImageNotFound:
        #         tar_file_path = 'docker/images/waifu2x.tar'
        #         try:
        #             with open(tar_file_path, 'rb') as file:
        #                 client.images.load(file.read())
        #         except FileNotFoundError:
        #             print("Image does not exist. Building it...")
        #             # Build the image
        #             client.images.build(path='docker/files/waifu2x', tag=image_name)
        #
        #         image = client.images.get(image_name)
        #         print("Image exists")
        #     except docker.errors.APIError as e:
        #         print(f"An error occurred: {e}")

        image_name = "your-image-name"
        dockerfile_location = "./docker/files"

        command = f"docker build -t {image_name} {dockerfile_location}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(f"An error occurred: {error}")
        else:
            print(f"Output: {output.decode('utf-8')}")

        container_name = "your-container-name"

        command = f"docker create --name {container_name} {image_name}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(f"An error occurred: {error}")
        else:
            print(f"Output: {output.decode('utf-8')}")

        command = f"docker start {container_name}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(f"An error occurred: {error}")
        else:
            print(f"Output: {output.decode('utf-8')}")

        raise NotImplementedError("Waifu2x and SUPIR are not implemented yet!")

    match algorithm:
        case Algorithms.xBRZ:  # TODO: Use RGB mode if the image is not RGBA
            for image_object in images:
                new_image_object_list = []
                for factor in factors:
                    if factor < 1:
                        print(
                            colored(
                                f"ERROR: xBRZ does not support downscaling! Factor: {factor}; "
                                f"Defaulting to fallback alhorithm: {fallback_algorithm.name}",
                                'red'
                            )
                        )
                        scaled_image = []
                        for frame in image_object['images'][0]:
                            cv2_image = utils.pil_to_cv2(frame)
                            width, height = frame.size

                            output_width, output_height = round(width * factor), round(height * factor)

                            scaled_image.append(
                                utils.cv2_to_pil(
                                    cv2.resize(
                                        cv2_image,
                                        (output_width, output_height),
                                        interpolation=csatca(fallback_algorithm)
                                    )
                                )
                            )
                        new_image_object_list.append(scaled_image)
                        # continue
                        # raise ValueError("xBRZ does not support downscaling!")
                    # If factor is not a whole number or is greater than 6, print a warning
                    if factor != int(factor) or factor > 6:
                        print(
                            colored(
                                f"WARNING: Scaling by xBRZ with factor {factor} "
                                f"is not supported, result might be blurry!",
                                'yellow'
                            )
                        )

                    scaled_image = []
                    for frame in image_object['images'][0]:
                        width, height = frame.size

                        frame = frame.convert('RGBA')
                        output_width, output_height = round(width * factor), round(height * factor)

                        current_scale = 1
                        while current_scale < factor:
                            temp_factor = 6
                            while current_scale * temp_factor >= factor:
                                temp_factor -= 1
                            temp_factor = min(temp_factor + 1, 6)

                            frame = xbrz.scale_pillow(frame, temp_factor)
                            current_scale *= temp_factor

                        scaled_image.append(
                            utils.cv2_to_pil(
                                cv2.resize(
                                    utils.pil_to_cv2(frame),
                                    (output_width, output_height),
                                    interpolation=csatca(fallback_algorithm)
                                )
                            )
                        )
                    new_image_object_list.append(scaled_image)
                scaled_images.append({
                    'images': new_image_object_list
                })

        case Algorithms.FSR:
            if config_plus is None:
                raise ValueError("config_plus is None! Cannot use CLI controlled algorithms without it!")
            else:
                if 'relative_input_path_of_images' not in config_plus:
                    raise ValueError("relative_input_path_of_images not in config_plus!")
                relative_input_path_of_images = config_plus['relative_input_path_of_images']

                if 'relative_output_path_of_images' not in config_plus:
                    relative_output_path_of_images = map(
                        lambda x: x.replace('input', 'output'), relative_input_path_of_images
                    )
                    relative_output_path_of_images = map(
                        lambda x: x.replace('.png', '_FSR.png'), relative_output_path_of_images
                    )
                else:
                    relative_output_path_of_images = config_plus['relative_output_path_of_images']

                # change file name to include '_FSR' before the file extension
                # relative_output_path_of_images = map(
                #     lambda x: x.replace('.png', '_FSR.png'), relative_output_path_of_images
                # )

                for relative_input_path, relative_output_path in zip(
                        relative_input_path_of_images, relative_output_path_of_images
                ):
                    print(f"Relative input path: {relative_input_path}")
                    print(f"Relative output path: {relative_output_path}")
                    for factor in factors:
                        if factor > 2:
                            print(
                                colored(
                                    "WARNING: Scaling with FSR by factor of {factor} is not supported, "
                                    "result might be blurry!",
                                    'yellow'
                                )
                            )

                        script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
                        options = f"-Scale {factor}x {factor}x -Mode EASU"
                        files = f"{relative_input_path} {relative_output_path}"
                        command = f"{script_path} {options} {files}"
                        subprocess.run(command)
                        # for frame in image:
                        #     script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
                        #     options = f"-Scale {factor}x {factor}x -Mode EASU"
                        #     files = (
                        #         f"../input/{config_plus['input_image_relative_path']} "
                        #         f"../output/{config_plus['input_image_relative_path']}"
                        #     )
                        #     command = f"{script_path} {options} {files}"
                        #     subprocess.run(command)

        case Algorithms.CAS:
            if config_plus is None:
                raise ValueError("config_plus is None! Cannot use CLI controlled algorithms without it!")
            else:
                if 'sharpness' not in config_plus:
                    raise ValueError("sharpness not in config_plus!")
                sharpness = config_plus['sharpness']

                if 'relative_input_path_of_images' not in config_plus:
                    raise ValueError("relative_input_path_of_images not in config_plus!")
                relative_input_path_of_images = config_plus['relative_input_path_of_images']

                if 'relative_output_path_of_images' not in config_plus:
                    relative_output_path_of_images = map(
                        lambda x: x.replace('input', 'output'), relative_input_path_of_images
                    )
                    relative_output_path_of_images = map(
                        lambda x: x.replace('.png', '_CAS.png'), relative_output_path_of_images
                    )
                else:
                    relative_output_path_of_images = config_plus['relative_output_path_of_images']

                # change file name to include '_CAS' before the file extension
                # relative_output_path_of_images = map(
                #     lambda x: x.replace('.png', '_CAS.png'), relative_output_path_of_images
                # )

                for relative_input_path, relative_output_path in (
                        zip(relative_input_path_of_images, relative_output_path_of_images)
                ):
                    for factor in factors:
                        if factor > 2:
                            print(
                                colored(
                                    "WARNING: Scaling with FSR by factor of {factor} is not supported, "
                                    "result might be blurry!",
                                    'yellow'
                                )
                            )

                        script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
                        options = f"-Scale {factor}x {factor}x -Sharpness {sharpness} -Mode CAS"
                        files = f"{relative_input_path} {relative_output_path}"
                        command = f"{script_path} {options} {files}"
                        subprocess.run(command)
                        # for frame in image:
                        #     script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
                        #     options = f"-Scale {factor}x {factor}x -Sharpness {config_plus['sharpness']} -Mode CAS"
                        #     files = (
                        #         f"../input/{config_plus['input_image_relative_path']} "
                        #         f"../output/{config_plus['input_image_relative_path']}"
                        #     )
                        #     command = f"{script_path} {options} {files}"
                        #     subprocess.run(command)

        case Algorithms.Super_xBR:
            for image_object in images:
                new_image_object_list = []
                for factor in factors:
                    if factor < 2:
                        print(
                            colored(
                                f"WARNING: Super-xBR does not support factors smaller then 2, factor: {factor}! "
                                "Defaulting to fallback algorithm",
                                'red'
                            )
                        )
                    # Check if factor is not a power of 2
                    factor_check = factor
                    temp_factor = factor
                    while factor_check > 2:
                        if factor_check % 2 != 0:
                            print(
                                colored(
                                    f"WARNING: Super-xBR does not support factor: {factor}! "
                                    "Result might be blurry!",
                                    'yellow'
                                )
                            )
                            temp_factor = 2
                            while temp_factor < factor:
                                temp_factor *= 2
                            break
                        factor_check //= 2

                    power = 1
                    while 2**power != temp_factor:
                        power += 1

                    scaled_image = []
                    for frame in image_object['images'][0]:
                        width, height = frame.size
                        output_width, output_height = round(width * temp_factor), round(height * temp_factor)

                        frame = superxbr.scale(frame, power)

                        scaled_image.append(
                            utils.cv2_to_pil(
                                cv2.resize(
                                    utils.pil_to_cv2(frame),
                                    (output_width, output_height),
                                    interpolation=csatca(fallback_algorithm)
                                )
                            )
                        )
                    new_image_object_list.append(scaled_image)
                scaled_images.append({
                    'images': new_image_object_list
                })

        case Algorithms.hqx:
            allowed_factors = {2, 3, 4}
            for image_object in images:
                new_image_object_list = []
                for factor in factors:
                    if factor not in allowed_factors:
                        if factor < 1:
                            print(
                                colored(
                                    "ERROR: HQx does not support downscaling! Cannot perform any fixes! "
                                    f"Scaling with fallback algorithm: {fallback_algorithm.name}",
                                    'red'
                                )
                            )
                            scaled_image = []
                            for frame in image_object['images'][0]:
                                cv2_image = utils.pil_to_cv2(frame)
                                width, height = frame.size

                                output_width, output_height = round(width * factor), round(height * factor)

                                scaled_image.append(
                                    utils.cv2_to_pil(
                                        cv2.resize(
                                            cv2_image,
                                            (output_width, output_height),
                                            interpolation=csatca(fallback_algorithm)
                                        )
                                    )
                                )
                            new_image_object_list.append(scaled_image)

                        print(
                            colored(
                                f"WARNING: HQx does not support factor: {factor}! "
                                f"Allowed factors: {allowed_factors}; Result might be blurry!",
                                'yellow'
                            )
                        )

                    # min_allowed_factor = min(allowed_factors)
                    max_allowed_factor = max(allowed_factors)
                    scaled_image = []
                    for frame in image_object['images'][0]:
                        width, height = frame.size
                        result = frame.convert('RGB')

                        current_factor = 1
                        while current_factor < factor:
                            temp_factor = max_allowed_factor
                            while current_factor * temp_factor >= factor:
                                temp_factor -= 1
                            while temp_factor not in allowed_factors:
                                temp_factor += 1

                            result = hqx.hqx_scale(result, temp_factor)
                            current_factor *= temp_factor

                        scaled_image.append(
                            utils.cv2_to_pil(
                                cv2.resize(
                                    utils.pil_to_cv2(result),
                                    (width * factor, height * factor),
                                    interpolation=csatca(fallback_algorithm)
                                )
                            )
                        )
                    new_image_object_list.append(scaled_image)
                scaled_images.append({
                    'images': new_image_object_list
                })

        case Algorithms.NEDI:
            if 'NEDI_m' not in config_plus:
                print(
                    colored(
                        "WARNING: NEDI_m (edge detection radius) is not in config_plus! Using default value '4'",
                        'yellow'
                    )
                )
                config_plus['NEDI_m'] = 4
            for image_object in images:
                new_image_object_list = []
                for factor in factors:
                    if factor < 1:
                        print(
                            colored(
                                "ERROR: NEDI does not support downscaling! Cannot perform any fixes! "
                                f"Scaling with fallback algorithm: {fallback_algorithm.name}",
                                'red'
                            )
                        )
                        scaled_image = []
                        for frame in image_object['images'][0]:
                            cv2_image = utils.pil_to_cv2(frame)
                            width, height = frame.size

                            output_width, output_height = round(width * factor), round(height * factor)

                            scaled_image.append(
                                utils.cv2_to_pil(
                                    cv2.resize(
                                        cv2_image,
                                        (output_width, output_height),
                                        interpolation=csatca(fallback_algorithm)
                                    )
                                )
                            )
                        new_image_object_list.append(scaled_image)

                    # If factor is not a whole number or is not a power of 2, print a warning
                    # if factor != int(factor) or factor > 6:
                    #     print(
                    #         colored(
                    #             f"WARNING: Scaling by NEDI with factor {factor} is not supported, "
                    #             "result might be blurry!",
                    #             'yellow'
                    #         )
                    #     )

                    temp_factor_repeat = 1
                    while 2**temp_factor_repeat <= factor:
                        temp_factor_repeat += 1

                    scaled_image = []
                    for frame in image_object['images'][0]:
                        width, height = frame.size

                        # frame = frame.convert('RGBA')
                        frame = utils.pil_to_cv2(frame)
                        channels = [frame[:, :, i] for i in range(frame.shape[2])]

                        for _ in range(temp_factor_repeat):
                            channels = [EDI_upscale(channel, config_plus['NEDI_m']) for channel in channels]

                        frame = np.stack(channels, axis=2)

                        output_width, output_height = round(width * factor), round(height * factor)

                        scaled_image.append(
                            utils.cv2_to_pil(
                                cv2.resize(
                                    frame, (output_width, output_height), interpolation=csatca(fallback_algorithm)
                                )
                            )
                        )
                    new_image_object_list.append(scaled_image)
                scaled_images.append({
                    'images': new_image_object_list
                })

        case Algorithms.Anime4K:
            scaled_images: list[utils.ImageDict] = []
            for image_object in images:
                image_object_list: list[list[PIL.Image.Image]] = []
                for factor in factors:
                    scaled_frames: list[PIL.Image.Image] = []
                    for frame in image_object['images'][0]:
                        np_image = np.array(frame.convert('RGB'))

                        current_factor = 1
                        while current_factor < factor:
                            print(f"iteration: {current_factor}")

                            parameters = pyanime4k.ac.Parameters()
                            parameters.HDN = True
                            a = pyanime4k.ac.AC(
                                managerList=pyanime4k.ac.ManagerList([pyanime4k.ac.OpenCLACNetManager(pID=0, dID=0)]),
                                type=pyanime4k.ac.ProcessorType.OpenCL_ACNet
                            )

                            a.load_image_from_numpy(np_image, input_type=pyanime4k.ac.AC_INPUT_RGB)

                            a.process()
                            current_factor *= 2

                            np_image = a.save_image_to_numpy()

                            a = None  # REQUIRED, DO NOT DELETE! Else raises a GPU error!

                        new_frame = PIL.Image.fromarray(np_image)

                        scaled_frames.append(
                            utils.cv2_to_pil(
                                cv2.resize(
                                    utils.pil_to_cv2(new_frame),
                                    (frame.size[0] * factor, frame.size[1] * factor),
                                    interpolation=csatca(fallback_algorithm)
                                )
                            )
                        )
                    image_object_list.append(scaled_frames)
                scaled_images.append({'images': image_object_list})
