# coding=utf-8
import numpy as np
import PIL.Image
import pyanime4k.ac

from scaling.utils import ConfigPlus, correct_frame


def scale(frames: list[PIL.Image], factor: float, config_plus: ConfigPlus) -> list[PIL.Image]:
    scaled_frames: list[PIL.Image.Image] = []
    for frame in frames:
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

            a = None  # REQUIRED, DO NOT DELETE! Else raises a GPU error! TODO: Make a GitHub issue

        new_frame = PIL.Image.fromarray(np_image)

        scaled_frames.append(correct_frame(new_frame, frame.size, factor, config_plus['high_quality_scale_back']))
    return scaled_frames
