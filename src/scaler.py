# coding=utf-8
# import rarch
import cv2
import numpy as np
import PIL.Image
import queue
import subprocess
import torch
import utils
import xbrz  # See xBRZ scaling on Jira

from RealESRGAN import RealESRGAN
from utils import Algorithms

# from rarch import CommonShaders

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
    Converts a scaler algorithm to a PIL algorithm using a dictionary (satpad)
    :param algorithm: The Scaler algorithm to convert
    :return: The corresponding PIL algorithm
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
    Converts a scaler algorithm to a OpenCV algorithm using a dictionary (satcad)
    :param algorithm: The Scaler algorithm to convert
    :return: The corresponding OpenCV algorithm
    """
    algorithm_cv2 = satcad.get(algorithm)
    if algorithm_cv2 is not None:
        return algorithm_cv2
    else:
        raise AttributeError(f"Algorithm not supported by OpenCV: {utils.algorithm_to_string(algorithm)},  id: {algorithm}")


def scale_image(algorithm, cv2_image, factor, *, fallback_algorithm=Algorithms.CV2_INTER_AREA, config_plus=None, main_checked=False) -> PIL.Image:
    return scale_image_batch(algorithm, cv2_image, [factor], fallback_algorithm=fallback_algorithm, config_plus=config_plus, main_checked=main_checked).get()


# ud - upscale/downscale
cv2_algorithms_ud = {Algorithms.CV2_INTER_CUBIC, Algorithms.CV2_INTER_LANCZOS4, Algorithms.CV2_INTER_LINEAR, Algorithms.CV2_INTER_NEAREST}
pil_algorithms_ud = {Algorithms.PIL_NEAREST_NEIGHBOR, Algorithms.PIL_BILINEAR, Algorithms.PIL_BICUBIC, Algorithms.PIL_LANCZOS}

cv2_ai = {Algorithms.CV2_EDSR, Algorithms.CV2_ESPCN, Algorithms.CV2_FSRCNN, Algorithms.CV2_LapSRN}


def scale_image_batch(algorithm, image, factors, fallback_algorithm=Algorithms.CV2_INTER_AREA, config_plus=None, main_checked=False):
    # scaled_images = []
    scaled_images = queue.Queue()

    width, height = image.size

    # ------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Start of OpenCV algorithms ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------
    if algorithm == Algorithms.CV2_INTER_AREA:
        algorithm = csatca(algorithm)
        cv2_image = utils.pil_to_cv2(image)

        for factor in factors:
            if factor > 1:
                # raise ValueError("INTER_AREA does not support upscaling!")
                print(f"ERROR: INTER_AREA does not support upscaling! Factor: {factor}; File names will be incorrect!")
                continue
            output_width, output_height = round(width * factor), round(height * factor)
            scaled_images.put(
                utils.cv2_to_pil(cv2.resize(cv2_image, (output_width, output_height), interpolation=cv2.INTER_AREA)))

        return scaled_images

    if algorithm in cv2_algorithms_ud:
        algorithm = csatca(algorithm)
        cv2_image = utils.pil_to_cv2(image)

        for factor in factors:
            output_width, output_height = round(width * factor), round(height * factor)
            scaled_images.put(utils.cv2_to_pil(cv2.resize(cv2_image, (output_width, output_height), interpolation=algorithm)))

        return scaled_images

    if algorithm in cv2_ai:
        cv2_image = utils.pil_to_cv2(image.convert('RGB'))
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        name = algorithm.name[4:]
        path_prefix = f"./weights/{name}/{name}"
        allowed_factors = {2, 3, 4, 8}

        for factor in factors:
            if factor not in allowed_factors:
                # raise ValueError("INTER_AREA does not support upscaling!")
                print(f"ERROR: CV2 AI does not support factor: {factor}!; File names will be incorrect!\nAllowed factors: {allowed_factors}")
                continue

            path = f"{path_prefix}_x{factor}.pb"
            # print(f"Path: {path}")

            sr.readModel(path)
            sr.setModel(name.lower(), factor)

            result = sr.upsample(cv2_image)
            scaled_images.put(utils.cv2_to_pil(cv2.resize(result, (width * factor, height * factor), interpolation=csatca(fallback_algorithm))))

            # output_width, output_height = round(width * factor), round(height * factor)
            # scaled_images.put(utils.cv2_to_pil(cv2.resize(cv2_image, (output_width, output_height), interpolation=algorithm)))

        return scaled_images
    # ----------------------------------------------------------------------------------------------------------
    # ---------------------------------------- End of OpenCV algorithms ----------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Start of PIL algorithms -----------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if algorithm in pil_algorithms_ud:
        algorithm = csatpa(algorithm)

        for factor in factors:
            output_width, output_height = round(width * factor), round(height * factor)
            scaled_images.put(image.resize((output_width, output_height), algorithm))
        return scaled_images
    # -------------------------------------------------------------------------------------------------------
    # ---------------------------------------- End of PIL algorithms ----------------------------------------
    # -------------------------------------------------------------------------------------------------------

    match algorithm:
        case Algorithms.xBRZ:  # TODO: Use RGB mode if the image is not RGBA
            starting_image = image.convert('RGBA')

            for factor in factors:
                image = starting_image
                output_width, output_height = round(width * factor), round(height * factor)

                if factor < 1:
                    print(f"ERROR: xBRZ does not support downscaling! Factor: {factor}")
                    continue
                    # raise ValueError("xBRZ does not support downscaling!")
                # If factor is not a whole number or is greater than 6, print a warning
                if factor != int(factor) or factor > 6:
                    print(f"WARNING: Scaling by xBRZ with factor {factor} is not supported, result might be blurry!")

                current_scale = 1
                while current_scale < factor:
                    temp_factor = 6
                    while current_scale * temp_factor >= factor:
                        temp_factor -= 1
                    temp_factor = min(temp_factor + 1, 6)

                    image = xbrz.scale_pillow(image, temp_factor)
                    current_scale *= temp_factor

                scaled_images.put(utils.cv2_to_pil(cv2.resize(utils.pil_to_cv2(image), (output_width, output_height), interpolation=csatca(fallback_algorithm))))
            # # Convert image to RGBA format
            # if cv2_image.shape[2] == 3:
            #     # print("Converting from BGR to RGBA format...")
            #     cv2_image_rgba = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGBA)
            # else:
            #     # print("Converting from BGRA to RGBA format...")
            #     cv2_image_rgba = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)
            #
            # # Convert image to 32-bit integers (pack channels into one 32-bit integer)
            # # int32_image = np.packbits(cv2_image_rgba, axis=-1).view(np.int32)
            # # Convert RGBA channels to 32-bit integers
            # # channel_dtype = np.uint32  # Use 32-bit unsigned integers for channel values
            # r, g, b, a = cv2_image_rgba[:, :, 0], cv2_image_rgba[:, :, 1], cv2_image_rgba[:, :, 2], cv2_image_rgba[:, :, 3]
            # r = np.uint32(r)
            # g = np.uint32(g)
            # b = np.uint32(b)
            # a = np.uint32(a)
            #
            # # Combine RGBA channels into a single 32-bit integer
            # int32_image = (r << 24) | (g << 16) | (b << 8) | a
            # # int32_image = (cv2_image_rgba[:, :, 0] << 24) | (cv2_image_rgba[:, :, 1] << 16) | (cv2_image_rgba[:, :, 2] << 8) | cv2_image_rgba[:, :, 3]
            # # print(int32_image)
            # # Flatten the array
            # flattened_array = int32_image.flatten()
            # python_array_image = bytearray(flattened_array.tobytes())
            #
            # # python_array_image = bytearray(cv2_image.astype(np.int32).flatten().tobytes())  #.tolist()
            # # print(cv2_image.astype(np.int32).flatten())

            # pil_image = utils.cv2_to_pil(cv2_image)
            # pil_image = image.convert('RGBA')
            # for factor in factors:
            #     image = python_array_image
            #     if type(image) is bytearray:
            #         print(f"Image is bytearray")
            #     else:
            #         print(f"Image is not bytearray")
            #     print(f"Image initial size: {len(image)}")
            #     output_width, output_height = round(width * factor), round(height * factor)
            #
            #     # if factor > 6:
            #     #     raise ValueError("Max factor for xbrz=6")
            #     # factor = Fraction(factor).limit_denominator().numerator
            #     if factor < 1:
            #         print(f"ERROR: xBRZ does not support downscaling! Factor: {factor}")
            #         continue
            #         # raise ValueError("xBRZ does not support downscaling!")
            #     # If factor is not a whole number or is greater than 6, print a warning
            #     if factor != int(factor) or factor > 6:
            #         print(f"WARNING: Scaling by xBRZ with factor {factor} is not supported, result might be blurry!")
            #
            #     current_scale = 1
            #     while current_scale < factor:
            #         temp_factor = 6
            #         while current_scale * temp_factor >= factor:
            #             temp_factor -= 1
            #         temp_factor = min(temp_factor + 1, 6)
            #         print(f"Temp factor: {temp_factor}")
            #
            #         print("Image-pre:")
            #         print(image)
            #         image = xbrz.scale(image, temp_factor, width, height, xbrz.ColorFormat.RGBA)
            #         # current_scale *= temp_factor
            #         print("Image-post:")
            #         print(image)
            #         # if image is bytearray:
            #         if type(image) is bytearray:
            #             print(f"Image is bytearray")
            #         else:
            #             print(f"Image is not bytearray")
            #         print(f"Image size: {len(image)}")
            #
            #         if current_scale < factor:
            #             current_scale *= temp_factor
            #             # Convert byte array to NumPy array
            #             numpy_array = np.frombuffer(image, dtype=np.uint8)
            #             rgba_image = numpy_array.reshape((height * current_scale, width * current_scale, 4))
            #             rgba_image = rgba_image[..., [3, 2, 1, 0]]  # Reorder the channels from 'ABGR' to 'RGBA'
            #
            #             # Convert to BGRA format for OpenCV
            #             cv2_image_scaled = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
            #             rgba_image = cv2.cvtColor(cv2_image_scaled, cv2.COLOR_BGRA2RGBA)
            #
            #             r, g, b, a = rgba_image[:, :, 0], rgba_image[:, :, 1], rgba_image[:, :, 2], rgba_image[:, :, 3]
            #             r = np.uint32(r)
            #             g = np.uint32(g)
            #             b = np.uint32(b)
            #             a = np.uint32(a)
            #
            #             int32_image = (r << 24) | (g << 16) | (b << 8) | a
            #             flattened_array = int32_image.flatten()
            #             image = bytearray(flattened_array.tobytes())
            #             print("Image-post-post:")
            #             print(image)
            #             if type(image) is bytearray:
            #                 print(f"Image is bytearray")
            #             else:
            #                 print(f"Image is not bytearray")
            #             print(f"Image size: {len(image)}")
            #         else:
            #             current_scale *= temp_factor
            #
            #     # Convert byte array to NumPy array
            #     numpy_array = np.frombuffer(image, dtype=np.uint8)
            #
            #     print(f"New height: {height * current_scale} = {height} * {current_scale}, New width: {width * current_scale} = {width} * {current_scale}")
            #     rgba_image = numpy_array.reshape((height * current_scale, width * current_scale, 4))
            #     rgba_image = rgba_image[..., [3, 2, 1, 0]]  # Reorder the channels from 'ABGR' to 'RGBA'
            #
            #     # Convert to BGRA format for OpenCV
            #     cv2_image_scaled = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
            #     # scaled_images.put(cv2_image_scaled)
            #     scaled_images.put(cv2.resize(cv2_image_scaled, (output_width, output_height), interpolation=csatca(fallback_algorithm)))

        case Algorithms.RealESRGAN:
            starting_image = image.convert('RGB')

            for factor in factors:
                image = starting_image
                output_width, output_height = round(width * factor), round(height * factor)

                if factor < 1:
                    raise ValueError("xBRZ does not support downscaling!")
                # If factor is not a whole number or is greater than 6, print a warning
                if factor not in (1, 2, 4, 8):
                    print(f"WARNING: Scaling by RealESRGAN with factor {factor} is not supported, result might be blurry!")

                current_scale = 1
                while current_scale < factor:
                    temp_factor = 8
                    while current_scale * temp_factor >= factor:
                        temp_factor //= 2
                    temp_factor = min(temp_factor * 2, 8)

                    model = RealESRGAN(device, scale=temp_factor)
                    model.load_weights(f'weights/RealESRGAN_x{temp_factor}.pth')  # , download=True
                    image = model.predict(image)

                    current_scale *= temp_factor

                scaled_images.put(utils.cv2_to_pil(cv2.resize(utils.pil_to_cv2(image), (output_width, output_height), interpolation=csatca(fallback_algorithm))))

        case Algorithms.SUPIR:
            script_path = './SUPIR/test.py'

            # Command 1
            # command1 = f"CUDA_VISIBLE_DEVICES=0 python {script_path} --img_dir '../input' --save_dir ../output --SUPIR_sign Q --upscale 2"
            command1 = f"python {script_path} --img_dir '../input' --save_dir ../output --SUPIR_sign Q --upscale 2"

            # Command 2
            # command2 = f"CUDA_VISIBLE_DEVICES=0 python {script_path} --img_dir '../input' --save_dir ../output --SUPIR_sign F --upscale 2 --s_cfg 4.0 --linear_CFG"

            # Execute commands
            # subprocess.run(command1, shell=True)
            subprocess.run(command1)
            # subprocess.run(command2, shell=True)

            # subprocess.run(['python', script_path])

        case Algorithms.FSR:
            if config_plus is None:
                raise ValueError("config_plus is None! Cannot use CLI controlled algorithms without it!")
            else:
                for factor in factors:
                    if factor > 2:
                        print(f"WARNING: Scaling with FSR by factor of {factor} is not supported, result might be blurry!")

                    script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
                    options = f"-Scale {factor}x {factor}x -Mode EASU"
                    files = f"../input/{config_plus['input_image_relative_path']} ../output/{config_plus['input_image_relative_path']}"
                    command = f"{script_path} {options} {files}"
                    subprocess.run(command)

        case Algorithms.CAS:
            if config_plus is None:
                raise ValueError("config_plus is None! Cannot use CLI controlled algorithms without it!")
            else:
                for factor in factors:
                    if factor > 2:
                        print(f"WARNING: Scaling with FSR by factor of {factor} is not supported, result might be blurry!")

                    script_path = './FidelityFX-CLI-v1.0.3/FidelityFX_CLI.exe'
                    options = f"-Scale {factor}x {factor}x -Sharpness {config_plus['sharpness']} -Mode CAS"
                    files = f"../input/{config_plus['input_image_relative_path']} ../output/{config_plus['input_image_relative_path']}"
                    command = f"{script_path} {options} {files}"
                    subprocess.run(command)

        case _:
            if main_checked:
                raise NotImplementedError("Not implemented yet")
            else:
                for factor in factors:
                    # raise NotImplementedError("Not implemented yet")
                    image = utils.pil_to_cv2(image)

                    # Convert image to RGBA format
                    cv2_image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                    # Convert 'cv2_image_rgba' numpy array to python list
                    python_array_image = cv2_image_rgba.tolist()

                    python_array_image = scale_image_data(algorithm, python_array_image, factor, fallback_algorithm, True)

                    # Convert python list to 'cv2_image_rgba' numpy array
                    cv2_image_rgba = np.array(python_array_image, dtype=np.uint8)
                    # Convert 'cv2_image_rgba' numpy array to 'cv2_image' numpy array
                    cv2_image = cv2.cvtColor(cv2_image_rgba, cv2.COLOR_RGBA2BGR)

                    scaled_images.put(cv2.resize(cv2_image, (width * factor, height * factor), interpolation=csatca(fallback_algorithm)))

                    # raise NotImplementedError("Not implemented yet")

                    # width, height = pil_image.size
                    # # pixels = [[[int]]]
                    # pixels = [[[0, 0, 0, 0] for _ in range(width)] for _ in range(height)]
                    # for y in range(height):
                    #     for x in range(width):
                    #         pixels[y][x] = pil_image.getpixel((x, y))
                    # return scale_image_data(algorithm, pixels, factor, fallback_algorithm, True)

    return scaled_images


# Main function for C++ lib
def scale_image_data(algorithm, pixels: [[[int]]], factor, *, fallback_algorithm=Algorithms.PIL_BICUBIC, main_checked=False):
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

                image = utils.pil_to_cv2(image)

                return scale_image_batch(algorithm, image, [factor], fallback_algorithm=fallback_algorithm, main_checked=True)
                # return scale_image(algorithm, image, factor, fallback_algorithm, main_checked=True)
