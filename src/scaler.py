# coding=utf-8
import io

import pyanime4k.ac
import cv2
import hqx
import numpy as np
import PIL.Image
import subprocess
import super_image
import torch
import torchvision

import utils
import xbrz  # See xBRZ scaling on Jira

from Edge_Directed_Interpolation.edi import EDI_upscale  # EDI_predict is wierd, EDI_Downscale is nearest neighbor...
from RealESRGAN import RealESRGAN
from superxbr import superxbr
from termcolor import colored
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
        image: utils.Image,
        factor: int,
        *,
        fallback_algorithm=Algorithms.CV2_INTER_AREA,
        config_plus=None,
        main_checked=False
) -> PIL.Image:
    return scale_image_batch(
        algorithm,
        [image],
        [factor],
        fallback_algorithm=fallback_algorithm,
        config_plus=config_plus,
        main_checked=main_checked
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

# si_algorithms = {
#     Algorithms.SI_drln_bam,
#     Algorithms.SI_edsr,
#     Algorithms.SI_msrn,
#     Algorithms.SI_mdsr,
#     Algorithms.SI_msrn_bam,
#     Algorithms.SI_edsr_base,
#     Algorithms.SI_mdsr_bam,
#     Algorithms.SI_awsrn_bam,
#     Algorithms.SI_a2n,
#     Algorithms.SI_carn,
#     Algorithms.SI_carn_bam,
#     Algorithms.SI_pan,
#     Algorithms.SI_pan_bam,
#
#     Algorithms.SI_drln,
#     Algorithms.SI_han,
#     Algorithms.SI_rcan_bam
# }
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


def scale_image_batch(
        algorithm: Algorithms,
        images: list[utils.Image],
        factors,
        *,
        fallback_algorithm=Algorithms.CV2_INTER_AREA,
        config_plus=None,
        main_checked=False
) -> list[utils.Image]:

    scaled_images = []
    # scaled_images = queue.Queue()

    # width, height = image.size

    # ------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Start of OpenCV algorithms ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------
    if algorithm == Algorithms.CV2_INTER_AREA:
        algorithm = csatca(algorithm)

        for image_object in images:
            new_image_object_list = []
            for factor in factors:
                if factor > 1:
                    # raise ValueError("INTER_AREA does not support upscaling!")
                    print(
                        colored(
                            "ERROR: INTER_AREA does not support upscaling! "
                            f"Factor: {factor}; File names will be incorrect!", 'red'
                        )
                    )
                    continue

                scaled_image = []
                for frame in image_object.images[0]:
                    cv2_image = utils.pil_to_cv2(frame)
                    width, height = frame.size

                    output_width, output_height = round(width * factor), round(height * factor)

                    scaled_image.append(
                        utils.cv2_to_pil(
                            cv2.resize(cv2_image, (output_width, output_height), interpolation=algorithm)
                        )
                    )

                new_image_object_list.append(scaled_image)
            scaled_images.append(utils.Image(new_image_object_list))

        return scaled_images

    if algorithm in cv2_algorithms_ud:
        algorithm = csatca(algorithm)

        for image_object in images:
            new_image_object_list = []
            for factor in factors:
                scaled_image = []
                for frame in image_object.images[0]:
                    cv2_image = utils.pil_to_cv2(frame)
                    width, height = frame.size

                    output_width, output_height = round(width * factor), round(height * factor)

                    scaled_image.append(
                        utils.cv2_to_pil(
                            cv2.resize(cv2_image, (output_width, output_height), interpolation=algorithm)
                        )
                    )

                new_image_object_list.append(scaled_image)
            scaled_images.append(utils.Image(new_image_object_list))

        return scaled_images

    def cv2_ai_common_scale(
            image_object: utils.Image,
            factor: int,
            sr: cv2.dnn_superres.DnnSuperResImpl,
            path_prefix: str,
            name: str
    ) -> list[PIL.Image.Image]:
        path = f"{path_prefix}_x{factor}.pb"
        # print(f"Path: {path}")

        sr.readModel(path)
        sr.setModel(name.lower().split('_')[0], factor)

        scaled_image = []
        for frame in image_object.images[0]:
            cv2_image = utils.pil_to_cv2(frame.convert('RGB'))
            width, height = frame.size

            result = sr.upsample(cv2_image)

            scaled_image.append(
                utils.cv2_to_pil(
                    cv2.resize(result, (width * factor, height * factor), interpolation=csatca(fallback_algorithm))
                )
            )

        return scaled_image

    def cv2_ai_common():
        sr = cv2.dnn_superres.DnnSuperResImpl_create()  # Ignore the warning, works fine

        name = algorithm.name[4:]
        path_prefix = f"./weights/{name}/{name}"

        for image_object in images:
            new_image_object_list = []
            for factor in factors:
                if factor not in allowed_factors:
                    # raise ValueError("INTER_AREA does not support upscaling!")
                    print(
                        colored(
                            f"Warning: CV2 AI '{algorithm.name}' does not support factor: {factor}! "
                            f"Allowed factors: {allowed_factors}; Result might be blurry!",
                            'yellow'
                        )
                    )
                    if factor < 1:
                        print(
                            colored(
                                "ERROR: CV2 AIs do not support downscaling! "
                                f"Cannot perform any fixes! Scaling with fallback algorithm: {fallback_algorithm.name}",
                                'red'
                            )
                        )
                        scaled_image = []
                        for frame in image_object.images[0]:
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

                    # min_allowed_factor = min(allowed_factors)
                    max_allowed_factor = max(allowed_factors)
                    scaled_image = []
                    for frame in image_object.images[0]:
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

                        scaled_image.append(
                            utils.cv2_to_pil(
                                cv2.resize(
                                    result, (width * factor, height * factor), interpolation=csatca(fallback_algorithm)
                                )
                            )
                        )

                    new_image_object_list.append(scaled_image)

                else:
                    new_image_object_list.append(cv2_ai_common_scale(image_object, factor, sr, path_prefix, name))
            scaled_images.append(utils.Image(new_image_object_list))

    if algorithm in cv2_ai_234:
        allowed_factors = {2, 3, 4}
        cv2_ai_common()

        return scaled_images

    if algorithm in cv2_ai_248:
        allowed_factors = {2, 4, 8}
        cv2_ai_common()

        return scaled_images
    # ----------------------------------------------------------------------------------------------------------
    # ---------------------------------------- End of OpenCV algorithms ----------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Start of PIL algorithms -----------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if algorithm in pil_algorithms_ud:
        algorithm = csatpa(algorithm)

        for image_object in images:
            new_image_object_list = []
            for factor in factors:
                scaled_image = []
                for frame in image_object.images[0]:
                    width, height = frame.size
                    output_width, output_height = round(width * factor), round(height * factor)

                    scaled_image.append(frame.resize((output_width, output_height), algorithm))
                new_image_object_list.append(scaled_image)
            scaled_images.append(utils.Image(new_image_object_list))
        return scaled_images
    # ---------------------------------------------------------------------------------------------------------
    # ---------------------------------------- End of PIL algorithms ------------------------------------------
    # ---------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Start of custom algorithms --------------------------------------
    # ---------------------------------------------------------------------------------------------------------

    if algorithm == Algorithms.HSDBTRE:
        repeats = []
        for factor in factors:
            repeat = 1
            while 4**repeat < factor:
                repeat += 1
            repeats.append(repeat)

        new_factors = [4**repeat for repeat in repeats]
        print(f"New factors: {new_factors}")
        print(f"Repeats: {repeats}")

        temporal_factors = new_factors.copy()
        scaled_images = images
        while any([factor > 1 for factor in temporal_factors]):
            f_to_scale = []
            for i, factor in enumerate(temporal_factors):
                if factor > 1:
                    f_to_scale.append(2)
                    temporal_factors[i] //= 4
                else:
                    f_to_scale.append(1)

            scaled_images = scale_image_batch(
                Algorithms.SI_drln_bam, scaled_images, f_to_scale, fallback_algorithm=fallback_algorithm
            )
            scaled_images = scale_image_batch(
                Algorithms.RealESRGAN, scaled_images, f_to_scale, fallback_algorithm=fallback_algorithm
            )

        for image_object in scaled_images:
            # print(f"Images: {image_object.images}")
            for i, scales in enumerate(image_object.images):
                # print(f"Type of scales: {type(scales)}")
                # print(f"Scales:\n{scales}")
                for j in range(len(scales)):
                    frame = scales[j]
                    # print(f"new_factors[i] * factors[i] = {new_factors[i] * factors[i]}")
                    # print(f"frame.size = {frame.size}")
                    scales[j] = utils.cv2_to_pil(
                        cv2.resize(
                            utils.pil_to_cv2(frame),
                            (
                                frame.size[0] // new_factors[i] * factors[i],
                                frame.size[1] // new_factors[i] * factors[i]
                            ),
                            interpolation=csatca(fallback_algorithm)
                        )
                    )

        return scaled_images

    if algorithm in si_algorithms:
        if algorithm in si_2x_3x_4x_algorithms:
            allowed_factors = {2, 3, 4}
        else:
            # algorithm in si_4x_algorithms
            allowed_factors = {4}

        for image_object in images:
            # model = None

            new_image_object_list = []
            for factor in factors:
                temp_factor = factor
                if factor not in allowed_factors:
                    raise ValueError("SI does not support this factor!")

                if algorithm == Algorithms.SI_a2n:
                    model = super_image.A2nModel.from_pretrained('eugenesiow/a2n', scale=temp_factor)
                elif algorithm == Algorithms.SI_awsrn_bam:
                    model = super_image.AwsrnModel.from_pretrained('eugenesiow/awsrn', scale=temp_factor)
                elif algorithm == Algorithms.SI_carn or algorithm == Algorithms.SI_carn_bam:
                    if algorithm == Algorithms.SI_carn:
                        model = super_image.CarnModel.from_pretrained('eugenesiow/carn', scale=temp_factor)
                    else:
                        model = super_image.CarnModel.from_pretrained('eugenesiow/carn-bam', scale=temp_factor)
                elif algorithm == Algorithms.SI_drln or algorithm == Algorithms.SI_drln_bam:
                    if algorithm == Algorithms.SI_drln:
                        model = super_image.DrlnModel.from_pretrained('eugenesiow/drln', scale=temp_factor)
                    else:
                        model = super_image.DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=temp_factor)
                elif algorithm == Algorithms.SI_edsr or algorithm == Algorithms.SI_edsr_base:
                    if algorithm == Algorithms.SI_edsr:
                        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr', scale=temp_factor)
                    else:
                        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=temp_factor)
                elif algorithm == Algorithms.SI_mdsr or algorithm == Algorithms.SI_mdsr_bam:
                    if algorithm == Algorithms.SI_mdsr:
                        model = super_image.MdsrModel.from_pretrained('eugenesiow/mdsr', scale=temp_factor)
                    else:
                        model = super_image.MdsrModel.from_pretrained('eugenesiow/mdsr-bam', scale=temp_factor)
                elif algorithm == Algorithms.SI_msrn or algorithm == Algorithms.SI_msrn_bam:
                    if algorithm == Algorithms.SI_msrn:
                        model = super_image.MsrnModel.from_pretrained('eugenesiow/msrn', scale=temp_factor)
                    else:
                        model = super_image.MsrnModel.from_pretrained('eugenesiow/msrn-bam', scale=temp_factor)
                elif algorithm == Algorithms.SI_pan or algorithm == Algorithms.SI_pan_bam:
                    if algorithm == Algorithms.SI_pan:
                        model = super_image.PanModel.from_pretrained('eugenesiow/pan', scale=temp_factor)
                    else:
                        model = super_image.PanModel.from_pretrained('eugenesiow/pan-bam', scale=temp_factor)
                elif algorithm == Algorithms.SI_rcan_bam:
                    model = super_image.RcanModel.from_pretrained('eugenesiow/rcan-bam', scale=temp_factor)
                elif algorithm == Algorithms.SI_han:
                    model = super_image.HanModel.from_pretrained('eugenesiow/han', scale=temp_factor)
                else:
                    raise ValueError("Unknown SI algorithm! I should get here... but the warning :/")

                scaled_image = []
                for frame in image_object.images[0]:
                    inputs = super_image.ImageLoader.load_image(frame)
                    preds = model(inputs)

                    cv2_frame = super_image.ImageLoader._process_image_to_save(preds)

                    frame_bytes = cv2.imencode('.png', cv2_frame)[1].tobytes()

                    pil_frame = PIL.Image.open(io.BytesIO(frame_bytes))

                    scaled_image.append(pil_frame)

                    # super_image.ImageLoader.save_image(preds, "../input/frame.png")
                    # super_image.ImageLoader.save_compare(preds, inputs, "../output/compare.png")
                new_image_object_list.append(scaled_image)
            scaled_images.append(utils.Image(new_image_object_list))
        return scaled_images

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
                        for frame in image_object.images[0]:
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
                    for frame in image_object.images[0]:
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
                scaled_images.append(utils.Image(new_image_object_list))

        case Algorithms.RealESRGAN:
            for image_object in images:
                new_image_object_list = []
                for factor in factors:
                    scaled_image = []
                    if factor < 1:
                        print(
                            colored(
                                "RealESRGAN AI does not support downscaling!; "
                                "Defaulting to fallback algorithm: {fallback_algorithm.name}",
                                'red'
                            )
                        )

                        for frame in image_object.images[0]:
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

                    for frame in image_object.images[0]:
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
                scaled_images.append(utils.Image(new_image_object_list))

        case Algorithms.SUPIR:
            raise NotImplementedError("Not implemented yet")

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
                    for frame in image_object.images[0]:
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
                scaled_images.append(utils.Image(new_image_object_list))

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
                            for frame in image_object.images[0]:
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
                    for frame in image_object.images[0]:
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
                scaled_images.append(utils.Image(new_image_object_list))

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
                        for frame in image_object.images[0]:
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
                    for frame in image_object.images[0]:
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
                scaled_images.append(utils.Image(new_image_object_list))

        case Algorithms.Anime4K:
            scaled_images: list[utils.Image] = []
            for image_object in images:
                image_object_list: list[list[PIL.Image.Image]] = []
                for factor in factors:
                    scaled_frames: list[PIL.Image.Image] = []
                    for frame in image_object.images[0]:
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
                scaled_images.append(utils.Image(image_object_list))

        case _:
            if main_checked:
                raise NotImplementedError("Not implemented yet")
            else:
                for image_object in images:
                    new_image_object_list = []
                    for factor in factors:
                        scaled_image = []
                        for frame in image_object.images[0]:
                            width, height = frame.size

                            frame = utils.pil_to_cv2(frame.convert('RGBA'))

                            # Convert image to RGBA format
                            cv2_image_rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                            # Convert 'cv2_image_rgba' numpy array to python list
                            python_array_image = cv2_image_rgba.tolist()

                            python_array_image = scale_image_data(
                                algorithm,
                                python_array_image,
                                factor,
                                fallback_algorithm=fallback_algorithm,
                                main_checked=True
                            )

                            # Convert python list to 'cv2_image_rgba' numpy array
                            cv2_image_rgba = np.array(python_array_image, dtype=np.uint8)
                            # Convert 'cv2_image_rgba' numpy array to 'cv2_image' numpy array
                            cv2_image = cv2.cvtColor(cv2_image_rgba, cv2.COLOR_RGBA2BGRA)

                            scaled_image.append(
                                utils.cv2_to_pil(
                                    cv2.resize(
                                        cv2_image,
                                        (width * factor, height * factor),
                                        interpolation=csatca(fallback_algorithm)
                                    )
                                )
                            )

                            # raise NotImplementedError("Not implemented yet")

                            # width, height = pil_image.size
                            # # pixels = [[[int]]]
                            # pixels = [[[0, 0, 0, 0] for _ in range(width)] for _ in range(height)]
                            # for y in range(height):
                            #     for x in range(width):
                            #         pixels[y][x] = pil_image.getpixel((x, y))
                            # return scale_image_data(algorithm, pixels, factor, fallback_algorithm, True)
                        new_image_object_list.append(scaled_image)
                    scaled_images.append(utils.Image(new_image_object_list))

    return scaled_images


# Main function for C++ lib
def scale_image_data(
        algorithm,
        pixels: [[[int]]],
        factor,
        *,
        fallback_algorithm=Algorithms.PIL_BICUBIC,
        main_checked=False
) -> PIL.Image:

    match algorithm:
        # case Algorithms.CPP_DEBUG:
        #     # new_pixels = scalercg.scale("cpp_debug", pixels, factor)
        #     new_pixels = scalercg.scale(pixels, factor, "cpp_debug")
        #     image = Image.new("RGBA", (len(new_pixels[0]), len(new_pixels)))
        #     for y in range(len(new_pixels)):
        #         for x in range(len(new_pixels[0])):
        #             image.putpixel((x, y), new_pixels[y][x])
        #     return image
        case _:
            if main_checked:
                raise NotImplementedError("Not implemented yet")
            else:
                image = PIL.Image.new("RGBA", (len(pixels[0]) * factor, len(pixels) * factor))
                for y in range(len(pixels)):
                    for x in range(len(pixels[0])):
                        image.putpixel((x * factor, y * factor), pixels[y][x])

                return scale_image(algorithm, image, factor, fallback_algorithm=fallback_algorithm, main_checked=True)
                # return scale_image(algorithm, image, factor, fallback_algorithm, main_checked=True)
