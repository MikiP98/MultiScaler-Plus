# coding=utf-8
import cv2
import numpy as np
import PIL.Image


def pil_to_cv2_high_bpc(bit_sets: list[PIL.Image.Image], bpc: int) -> np.ndarray:
    """
        Convert a list of 8-bit Pillow images to a one high bpc OpenCV image.
        It is assumed that the most important bits are in the first image,
        and that the least important bits are at the end of the last image.
        If there are less than 8 bits left for the last image,
        padding is assumed to be at the most important bits in that image.

        :param bit_sets: List of Pillow images (8-bit) (list[PIL.Image.Image])
        :param bpc: Number of bits per channel in the high bpc image (int)
        :returns: OpenCV format image (np.ndarray)
    """
    # Get the width and height and number of channels of the image
    width, height = bit_sets[0].size
    channels = len(bit_sets[0].getbands())

    # Determine the dtype that will be used for the output OpenCV image
    dtype = np.uint16
    if bpc > 16:
        if bpc > 32:
            dtype = np.uint64
        else:
            dtype = np.uint32

    # Initialize the output OpenCV image with zeros
    full_image = np.zeros((height, width, channels), dtype=dtype)

    # Initialize the number of remaining bits
    remaining_bits = bpc

    # Iterate over the Pillow images and add them to the full image
    for bit_set in bit_sets:
        cv2_8bit_image = pil_to_cv2(bit_set)

        # Shift bits for each channel in the 3D array and add the new 8-bit image
        full_image = (full_image << min(remaining_bits, 8)) + cv2_8bit_image

        # Reduce the number of remaining bits
        remaining_bits -= 8

    return full_image


def pli_to_cv2_list(bit_sets: list[PIL.Image.Image]) -> list[np.ndarray]:
    """
        Convert a list of 8-bit Pillow images to OpenCV format.
        :param bit_sets: List of Pillow images (8-bit) (list[PIL.Image.Image])
        :returns: List[OpenCV format image (np.ndarray)]
    """
    return list(map(pil_to_cv2, bit_sets))


def pil_to_cv2(bit_set: PIL.Image.Image) -> np.ndarray:
    """
        Convert a Pillow image to OpenCV format.
        Convert an 8-bit Pillow image OpenCV format, while handling RGB/RGBA to BGR/BGRA conversion

        :param bit_set: PIL image object (PIL.Image.Image) (8 bpc)
        :return: OpenCV format image (np.ndarray)
    """
    if bit_set.mode == 'RGBA':
        color_format = cv2.COLOR_RGBA2BGRA
    else:
        color_format = cv2.COLOR_RGB2BGR

    # Convert Pillow image to NumPy array and then to OpenCV format
    return cv2.cvtColor(np.array(bit_set), color_format)


def cv2_high_bpc_to_pil(high_bpc_cv2_img: np.ndarray, bpc: int) -> list[PIL.Image.Image]:
    """
        Splits a high bpc (e.g. 16 bpc) image into multiple 8-bit images using the method
        of filling as many 8-bit images from the most significant bits down until the rest of
        the bits would not fill the image, then putting them into the least significant
        position of the last image.

        :param high_bpc_cv2_img: High bpc image (e.g., 16-bit) loaded with OpenCV. (np.ndarray)
        :param bpc: the number of bits per channel in given image (int)
        :returns: List[PIL.Image.Image]: List of 8-bit images as PIL.Image.Image objects
    """
    # Determine the number of full 8-bit images and remaining bits
    full_images_num = bpc // 8
    rest_bits_num = bpc % 8
    print(f"{bpc=}, {full_images_num=}, {rest_bits_num=}")

    # Prepare the list for the split images
    images = []

    # Process full 8-bit images
    for i in range(full_images_num):
        # Shift bits for each channel in the 3D array and mask the rest
        img_8bit = (high_bpc_cv2_img >> ((full_images_num - i - 1) * 8 + rest_bits_num)) & 0xFF
        print(f"{img_8bit=}")
        images.append(cv2_to_pil(img_8bit.astype(np.uint8)))

    # Process the remaining bits if there are any
    if rest_bits_num != 0:
        print("Processing remaining bits...")
        # Shift bits for each channel in the 3D array and mask the rest
        img_rest_bits = (high_bpc_cv2_img & ((1 << rest_bits_num) - 1))
        print(f"{img_rest_bits=}")
        images.append(cv2_to_pil(img_rest_bits.astype(np.uint8)))

    return images


def cv2_to_pil_list(bit_sets: list[np.ndarray]) -> list[PIL.Image.Image]:
    """
        Convert a list of 8-bit OpenCV images to Pillow format, while handling BGR/BGRA to RGB/RGBA conversion
        :param bit_sets: List of OpenCV format images (8-bit) (list[np.ndarray])
        :returns: List[PIL.Image.Image]: List of converted images in PIL format
    """
    # Convert each image in the list using cv2_to_pil
    return list(map(cv2_to_pil, bit_sets))


def cv2_to_pil(cv2_image: np.ndarray) -> PIL.Image.Image:
    """
        Convert an 8-bit OpenCV image to Pillow format, while handling BGR/BGRA to RGB/RGBA conversion
        :param cv2_image: OpenCV format image (8-bit) (np.ndarray)
        :returns: PIL.Image.Image: Converted image in PIL format
    """
    if cv2_image.shape[2] == 4:
        # Convert OpenCV image to NumPy array and BGRA to RGBA
        numpy_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)
    else:
        # Convert OpenCV image to NumPy array and BGR to RGB
        numpy_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to Pillow format
    return PIL.Image.fromarray(numpy_array)
