# coding=utf-8
import PIL.Image

if __name__ == "__main__":
    image = PIL.Image.open(r"..\..\input\webp_test.png")

    w, h = image.size
    scaled_image = image.resize((w * 4, h * 4), PIL.Image.Resampling.LANCZOS)

    optional_args = {
        'lossless': True,
        'method': 6,
        'optimize': True,
        'format': 'WEBP'
    }
    scaled_image.save(r"..\..\output\webp_test.webp", **optional_args)
