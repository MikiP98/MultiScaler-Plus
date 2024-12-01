# coding=utf-8
import numpy as np

from aenum import auto, Enum, IntEnum


bpc2dtype = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
    128: np.uint128,
    256: np.uint256
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
