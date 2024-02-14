from email.policy import default
import io
from PIL import Image
from scaler import Algorithms

def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format='PNG')
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def string_to_scaling_algorithm(string: str) -> Algorithms:
    match string:
        case 'nearest-neighbour':
            return Algorithms.NEAREST_NEIGHBOR
        case 'bilinear':
            return Algorithms.BILINEAR
        case 'bicubic':
            return Algorithms.BICUBIC
        case 'lanczos':
            return Algorithms.LANCZOS
        case 'xbrz':
            return Algorithms.xBRZ
        case 'esrgan':
            return Algorithms.RealESRGAN
        case _:
            raise ValueError("Algorithm not found")
