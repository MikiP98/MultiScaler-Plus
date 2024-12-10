# coding=utf-8
import numpy as np
import PIL.Image

from aenum import auto, Enum, IntEnum

from humility_image_processor.UI.Console.console_formatting import *


bpc2dtype = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64
}

dtype2bpc = {
    np.uint8: 8,
    np.uint16: 16,
    np.uint32: 32,
    np.uint64: 64
}

dtype2power = {
    np.uint8: 3,
    np.uint16: 4,
    np.uint32: 5,
    np.uint64: 6
}


class HDR2SDR_Mode(IntEnum):
    MAP = auto()
    MULTIPLY = auto()


class PixelDataFormat(Enum):
    INT = int
    FLOAT = float


class PixelDataType(IntEnum):
    G = auto()  # Grayscale; 1 channel
    GA = auto()  # Grayscale with Alpha; 2 channels
    RGB = auto()  # RED, GREEN, BLUE; 3 channels
    RGBA = auto()  # RED, GREEN, BLUE, ALPHA; 4 channels
    CUSTOM = auto()  # Custom; n channels; Each channel will be processed separately


class Frame:
    def __init__(
            self,
            frame: np.ndarray,
            data_format: PixelDataFormat = PixelDataFormat.INT,
            data_type: PixelDataType = PixelDataType.RGBA,
            icc_profile: str | None = None
    ):
        self.frame = frame
        self.data_format = data_format
        self.data_type = data_type
        self.icc_profile = icc_profile

    @property
    def bpc(self) -> int:
        return self.frame.dtype.itemsize

    def get_frame_as_sdr(
            self,
            max_sdr_bpc = 64,
            hdr2sdr_mode: HDR2SDR_Mode = False
    ) -> tuple[np.ndarray, float | None, int | None]:

        if self.data_format == PixelDataFormat.INT:
            return (self.frame, None, None)

        if hdr2sdr_mode == HDR2SDR_Mode.MULTIPLY:
            max_float_value = self.frame.max()

            normalied_frame = self.frame / max_float_value

            if max_sdr_bpc not in bpc2dtype:
                raise ValueError(f"Unsupported numpy BPC: {max_sdr_bpc}; Allowed values: {list(bpc2dtype.keys())}")

            sdr_frame = (normalied_frame * 2**max_sdr_bpc).astype(bpc2dtype[max_sdr_bpc])

            return (sdr_frame, max_float_value, max_sdr_bpc)

        else:
            raise NotImplementedError

    def bit_split(self, new_bpc) -> list[np.ndarray]:
        raise NotImplementedError

    def print(
            self,
            output_width=None,
            output_height=None,
            aa=True,
            alpha_threshold=0.5,
            fill_bg=True,
            bg_color=(0, 0, 0),
            resampling=PIL.Image.Resampling.LANCZOS
    ) -> int:
        # Get frame in Integer format
        image_arr = self.get_frame_as_sdr()[0]

        # Get only 8 most significant bits
        if image_arr.dtype != np.uint8:
            new_array = np.zeros((image_arr.shape[0], image_arr.shape[1], 3), dtype=np.uint8)

            power = dtype2power[image_arr.dtype]
            # mask with only 8 most significant bits
            mask = 2**(power - 1) + 2**(power - 2) + 2**(power - 3) + 2**(power - 4) + 2**(power - 5) + 2**(power - 6) + 2**(power - 7) + 2**(power - 8)
            offset = 2**(power - 8)
            new_array[:, :, 0] = (image_arr[:, :, 0] & mask) + offset
            new_array[:, :, 1] = (image_arr[:, :, 1] & mask) + offset
            new_array[:, :, 2] = (image_arr[:, :, 2] & mask) + offset

            image_arr = new_array

        width, height = image_arr.shape[1], image_arr.shape[0]
        if output_width is not None:
            if output_height is None:
                output_height = int(output_width * height / width / 3)

        elif output_height is not None:
            if output_width is None:
                output_width = int(output_height * width * 3 / height)

        else:
            output_width = width * 3
            output_height = height

        if output_width != width or output_height != height:
            pil_image = PIL.Image.fromarray(image_arr)
            pil_image_lan = pil_image.resize((output_width, output_height), PIL.Image.Resampling.LANCZOS)
            image_arr = np.array(pil_image_lan)
            pil_image_n = pil_image.resize((output_width * 2, output_height * 2), resampling)
            pil_image_n = pil_image_n.resize((output_width, output_height), PIL.Image.Resampling.BILINEAR)
            image_arr_n = np.array(pil_image_n)
            # image_arr_n = image_arr
        else:
            image_arr_n = image_arr

        for i in range(image_arr.shape[0]):
            for j in range(image_arr.shape[1]):
                bg_r, bg_g, bg_b = image_arr[i, j]
                font_r, font_g, font_b = image_arr_n[i, j]
                print(all_colorize(f"#", font_r, font_g, font_b, bg_r, bg_g, bg_b), end='')
            print()

        return output_height


class Image:
    def __init__(self, frames: list[Frame], is_animated: bool = False, frame_time: float = 1/30):
        self.frames = frames
        self.is_animated = is_animated
        self.frame_time = frame_time


class TextureSet:
    def __init__(self, textures: dict[str, Image]):
        self.textures = textures


class MetaImageData:
    def __init__(self, relative_path: str, file_name: str):
        self.relative_path = relative_path
        self.file_name = file_name


class MetaImage(Image, MetaImageData):
    def __init__(self, frames: list[Frame], relative_path: str, file_name: str, is_animated: bool = False, frame_time: float = 1/30):
        super().__init__(frames, is_animated, frame_time)
        super().__init__(relative_path, file_name)


class MetaTextureSet(TextureSet, MetaImageData):
    def __init__(self, textures: dict[str, MetaImage], relative_path: str, file_name: str):
        super().__init__(textures)
        super().__init__(relative_path, file_name)
