# coding=utf-8

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


# Conversions:

#   'old seus to labPBR 1.3':
#       R = 255 * round(sqrt(r / 255))  # convert to perceptual smoothness
#       G = round(g * 0.8980392156862745)  # 0-229 range
#       B = 0
#       A = 255

#   'old continuum to labPBR 1.3':
#       R = b
#       G = round(g * 0.8980392156862745)  # 0-229 range
#       B = 0
#       A = a

#   'pbr+emissive (old BSL) to labPBR 1.3':
#       R = 255 * round(sqrt(r / 255))  # convert to perceptual smoothness
#       G = round(g * 0.8980392156862745)  # 0-229 range
#       B = 0
#       A = b-1  # 1-255 to 0-254 range and 0 become 255 with underflow

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


def convert_from_old_seus():
    raise NotImplementedError("Conversion not implemented")
