# coding=utf-8
import numpy as np
import PIL.Image
import time

from humility_image_processor.Objects.Image import Image as HumilityImage, Frame

image = PIL.Image.open("INPUT/c06.gif")
frame_count = image.n_frames

frames = []

for i in range(frame_count):
    image.seek(i)
    image_arr = np.array(image.convert("RGB"))
    frames.append(Frame(image_arr))

humility_image = HumilityImage(frames)

i = 0
while True:
    height = humility_image.frames[i % frame_count].print(output_width=200)

    # move cursor up by height
    print(f"\033[{height}A", end='')

    # wait 1/24th of a second
    # time.sleep(1/30)

    i += 1
