import scaler

from PIL import Image

if __name__ == '__main__':
    path = "input/blast_furnace_front.png"
    image = scaler.scale_image(scaler.Algorithms.xBRZ, Image.open(path), 4)
