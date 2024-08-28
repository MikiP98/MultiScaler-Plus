# coding=utf-8

# Info:
#   - LabPBR spec -> https://shaderlabs.org/wiki/LabPBR_Material_Standard
#   - Original converter -> https://github.com/flodri/RGBA-Formats-Converter/blob/master/MAIN.py
#   - Continuum -> https://discord.com/channels/341731475621806083/341988569612550144/1273341091156070450

# LabPBR 1.3 specification:

#   Red Channel
#       Represents "perceptual" smoothness.
#       Convert perceptual smoothness to linear roughness with roughness = pow(1.0 - perceptualSmoothness, 2.0).
#       Convert linear roughness to perceptual smoothness with perceptualSmoothness = 1.0 - sqrt(roughness).
#
#       For Texture Artists
#           A value of 255 (100%) results in a very smooth material (e.g. polished granite). A value of 0 (0%) creates a rough material (e.g. rocks).

#   Green Channel
#       Values from 0 to 229 represent F0, also known as reflectance.
#       This attribute is stored linearly. Please note that a value of 229 represents exactly 229 divided by 255, or approximately 90%, instead of 100%.
#       Values from 230 to 255 represent various different metals.
#
#       How metals work
#           In order to allow a more accurate representation of metals with the limited amount of information that can be provided,
#           certain metals have been predefined and are selected by setting the green channel to specific values ranging from 230 to 254.
#           In these cases, the albedo is used to tint the reflections instead of being used for diffuse shading.
#
#           If you want a metal that isn't among the predefined metals, you can also set the green channel to a value of 255.
#           In this case, the albedo will instead be used as the F0. This is less accurate, but often still gives decent results.
#
#           Note that certain shader packs may not support these predefined metals, and will treat the entire range from 230 to 255 as though it had a value of 255.
#
#           | Metal	    | Bit   | Value	N (R, G, B)	        | K (R, G, B)            |
#           |-----------|-------|---------------------------|------------------------|
#           | Iron	    | 230	| 2.9114, 2.9497, 2.5845	| 3.0893, 2.9318, 2.7670 |
#           | Gold	    | 231	| 0.18299, 0.42108, 1.3734	| 3.4242, 2.3459, 1.7704 |
#           | Aluminum	| 232	| 1.3456, 0.96521, 0.61722	| 7.4746, 6.3995, 5.3031 |
#           | Chrome	| 233	| 3.1071, 3.1812, 2.3230	| 3.3314, 3.3291, 3.1350 |
#           | Copper	| 234	| 0.27105, 0.67693, 1.3164	| 3.6092, 2.6248, 2.2921 |
#           | Lead	    | 235	| 1.9100, 1.8300, 1.4400	| 3.5100, 3.4000, 3.1800 |
#           | Platinum	| 236	| 2.3757, 2.0847, 1.8453	| 4.2655, 3.7153, 3.1365 |
#           | Silver	| 237	|0.15943, 0.14512, 0.13547	| 3.9291, 3.1900, 2.3808 |

#   Blue Channel
#       On dielectrics:
#           Values from 0 to 64 represent porosity. Examples of the porosity effect can be found here.
#           Values from 65 to 255 represent subsurface scattering.
#           Both porosity and subsurface scattering are stored linearly.
#       On metals/conductors:
#           Reserved for future use.
#
#       For Texture Artists
#          The porosity value describes how much water a material can absorb.
#          This manifests in the color of the material getting darker and less reflective when wet.
#          This allows for a much more accurate behavior with shader packs supporting both porosity and weather based wetness (e.g. puddles).
#          Below are some example values.
#
#          | Material	                               | Porosity Value |
#          |-------------------------------------------|----------------|
#          | Sand	                                   | 64             |
#          | Wool	                                   | 38             |
#          | Wood	                                   | 12             |
#          | Metals and other impermeable materials    | 0              |

#   Alpha Channel
#       It can have values ranging from 0 to 254; 0 being 0% emissiveness and 254 being 100%.
#       This is stored linearly.
#
#       For Texture Artists
#           The lower the value the less light it will emit, the higher the more luminescent it'll be, but never use a value of 255 since this will cause it to be ignored.
#
#       My understanding: 0 is no emission, 254 is full emission, 255 is ignored, a.k.a. texture is not emissive.

# Normal Texture (_n)
#   The normal texture does not only contain the normal vectors, but also ambient occlusion and a height/displacement map.
#   The normal vector should be encoded in DirectX format (Y-);
#   this is commonly referred to as top-down normals, which can be visibly characterized as having X/red pointing to the right, and Y/green pointing downward.

#   Material AO
#       Ambient occlusion gets stored in the blue channel of the normal texture.
#       The third component (.z) of the normal vector can be reconstructed using sqrt(1.0 - dot(normal.xy, normal.xy)).
#       This attribute is stored linearly; 0 being 100% AO and 255 being 0%.

#   Height Map
#       The height map used for POM is stored in the alpha channel of the normal texture; a value of 0 (pure black on the height map) represents a depth of 25% in the block.
#       Be aware that a value of 0 on the height map will cause some issues with certain shader packs' POM implementations, so a minimum of 1 is recommended instead.

# Reasoning for this Layout
#   The AO is stored in the blue channel because the first 3 components of a pixel in the normal texture represent a vector of length 1. Since we know the length, we only need 2 of the 3 components to reconstruct the vector (thanks Pythagore). This means that one of the three channels can be used for something else, like storing AO in the blue channel.


# Conversions:

#   'old seus to labPBR 1.3':
#       original:
#          R = 255 * round(sqrt(r / 255))  # convert to perceptual smoothness
#          G = round(g * 0.8980392156862745)  # 0-229 range
#          B = 0
#          A = 255
#       my:
#           _s:
#               # SEUS specular map spec: https://github.com/Moo-Ack-Productions/MCprep/issues/78 -> SEUS PBR has emission in the blue channel
#               R = 255 * round(sqrt(r / 255))  # convert to perceptual smoothness
#               G = round(g * 0.8980392156862745)  # 0-229 range
#               B = 0
#               A = min(B, 254)

#   'old continuum to labPBR 1.3':  # Done? TODO: Revamp new blue channel
#       original:
#           R = b
#           G = round(g * 0.8980392156862745)  # 0-229 range
#           B = 0
#           A = a
#       my:
#           _s:
#               R = b  # Blue had smoothness
#               G = r  # Red had f0
#               B = (max(a, 65)) if a > g and a > 33 else (min(g, 64))  # Alpha had subsurface scattering and green had porosity
#               A = 255  # Continuum didn't have emission
#           _n:
#               R = r
#               G = g
#               B = 255  # LabPBR has AO in blue channel, continuum didn't have it
#               A = a

#   'pbr+emissive (old BSL) to labPBR 1.3':
#       original:
#           R = 255 * round(sqrt(r / 255))  # convert to perceptual smoothness
#           G = round(g * 0.8980392156862745)  # 0-229 range
#           B = 0
#           A = b-1  # 1-255 to 0-254 range and 0 become 255 with underflow
#       my:
#           R = 255 * round(sqrt(r / 255))  # convert to perceptual smoothness
#           G = round(g * 0.8980392156862745)  # 0-229 range
#           B = 0
#           A = min(b, 254) ???

#   "gray to labPBR 1.3 (you probably won't get good results)":
#       # magic number are 1-x of the one in ITU-R 601-2 (L = R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000)
#       R = int(r * 0.701)
#       G = int(g * 0.3708901960784314)  # 0.3708901960784314 = 0.413 * 0.8980392156862745
#       B = 0
#       A = 255

#   'Custom preset':
#       R = r
#       G = g
#       B = b
#       A = a


# image order in dict:
#   _s: specular
#   _n: normal


# import numpy as np
import PIL.Image

# from aenum import IntEnum
from termcolor import colored
from utils import ImageDict


# R = 0
# G = 1
# B = 2
# A = 3


def convert_from_old_continuum(
        texture_set: tuple[list[PIL.Image.Image] | None, list[PIL.Image.Image] | None]
) -> tuple[list[PIL.Image.Image] | None, list[PIL.Image.Image] | None]:

    specular_map, normal_map = texture_set

    if specular_map is None:
        new_specular_map = None
    else:
        new_specular_map = []
        for frame in specular_map:
            frame = frame.convert('RGBA')
            new_frame = PIL.Image.new('RGB', frame.size)

            for x in range(frame.size[0]):
                for y in range(frame.size[1]):
                    r, g, b, a = frame.getpixel((x, y))

                    new_frame.putpixel(
                        (x, y),
                        (
                            b,
                            r,
                            (max(a, 65)) if a > g and a > 33 else (min(g, 64))
                        )
                    )
            new_frame.putalpha(255)

            new_specular_map.append(new_frame)

    if normal_map is None:
        new_normal_map = None
    else:
        new_normal_map = []
        for frame in normal_map:
            frame = frame.convert('RGBA')
            new_frame = PIL.Image.new('RGBA', frame.size)

            for x in range(frame.size[0]):
                for y in range(frame.size[1]):
                    r, g, b, a = frame.getpixel((x, y))

                    new_frame.putpixel(
                        (x, y),
                        (
                            r,
                            g,
                            255,
                            a
                        )
                    )

            new_normal_map.append(new_frame)

    # specular_map = specular_map.convert('RGBA')
    # specular_map.putalpha(255)
    #
    # numpy_specular_map = np.array(specular_map)
    # new_numpy_specular_map = np.array(specular_map.deepcopy())
    # # Red channel
    # new_numpy_specular_map[:, :, R] = numpy_specular_map[:, :, B]
    # # Green channel
    # new_numpy_specular_map[:, :, G] = numpy_specular_map[:, :, R]
    # # Blue channel
    # new_numpy_specular_map[:, :, B] = (max(numpy_specular_map[:, :, A], 65) if numpy_specular_map[:, :, A] > 33 else 0) if numpy_specular_map[:, :, A] > numpy_specular_map[:, :, G] else (min(numpy_specular_map[:, :, G], 64))
    # # Alpha channel
    # new_numpy_specular_map[:, :, A] = 255
    #
    # new_specular_map = PIL.Image.fromarray(new_numpy_specular_map)
    #
    # normal_map = normal_map.convert('RGBA')
    # new_numpy_normal_map = np.array(normal_map)
    # # Blue channel
    # new_numpy_normal_map[:, :, B] = 255
    #
    # new_normal_map = PIL.Image.fromarray(new_numpy_normal_map)

    return new_normal_map, new_specular_map


# https://bdcraft.net/community/viewtopic.php?t=7069&start=10
