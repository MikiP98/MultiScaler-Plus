#!/usr/bin/env python3

# © 2020 io mintz <io@mintz.cc>

# xbrz.py is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# xbrz.py is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with xbrz.py. If not, see <https://www.gnu.org/licenses/>.

__version__ = '1.0.2'
__all__ = ['ColorFormat', 'SCALE_FACTOR_RANGE', 'scale', 'scale_wand', 'scale_pillow']

import ctypes
from enum import IntEnum

from _xbrz import __file__ as xbrz_path

class ColorFormat(IntEnum):  # from high bits -> low bits, 8 bit per channel
	RGB = 1
	RGBA = 2
	RGBA_UNBUFFERED = 3  # like RGBA, but without the one-time buffer creation overhead (ca. 100 - 300 ms) at the expense of a slightly slower scaling time

SCALE_FACTOR_RANGE = range(2, 7)

_xbrz = ctypes.CDLL(xbrz_path)
uint32_p = ctypes.POINTER(ctypes.c_uint32)
_xbrz.xbrz_scale_defaults.argtypes = [ctypes.c_size_t, uint32_p, uint32_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_xbrz.xbrz_scale_defaults.restype = None

def scale(img, factor, width, height, color_format: ColorFormat):
	"""Scale img, an array of width * height 32 bit ints. Return an array
	of scale * width * scale * height 32 bit ints.
	Scale factor must be in SCALE_FACTOR_RANGE.
	"""
	if factor not in SCALE_FACTOR_RANGE:
		raise ValueError('invalid scale factor')

	img = (ctypes.c_uint32 * (width * height)).from_buffer(img)

	scaled = (ctypes.c_uint32 * (factor ** 2 * width * height))()
	_xbrz.xbrz_scale_defaults(factor, img, scaled, width, height, color_format)

	return scaled

def scale_wand(img: 'wand.image.Image', factor) -> 'wand.image.Image':
	"""Scale a Wand image according to factor. Return a new image."""
	import wand.image
	scaled_pixels = scale(bytearray(img.export_pixels(channel_map='RGBA')), factor, *img.size, ColorFormat.RGBA)
	scaled = wand.image.Image(width=factor * img.width, height=factor * img.height)
	scaled.import_pixels(
		channel_map='RGBA',
		# cast to bytes because apparently import_pixels ignores storage='long'
		data=memoryview(scaled_pixels).cast('B'),
	)
	return scaled

def scale_pillow(img: 'PIL.Image.Image', factor) -> 'PIL.Image.Image':
	"""Scale a PIL/Pillow image according to factor. The image must be RGBA or RGB. Return a new image."""
	import PIL.Image

	if img.mode == 'RGB':
		raise NotImplementedError("RGB color mode is not yet supported. call img.convert('RGBA') first.")
	elif img.mode == 'RGBA':
		fmt = ColorFormat.RGBA
	else:
		raise ValueError('invalid image mode', img.mode)

	scaled_pixels = scale(bytearray(img.tobytes()), factor, *img.size, fmt)
	scaled_bytes = memoryview(scaled_pixels)
	scaled = PIL.Image.frombuffer(img.mode, (img.width * factor, img.height * factor), scaled_bytes)
	return scaled

def main():
	import sys

	if len(sys.argv) != 4:
		print('Usage:', sys.argv[0], '<scale factor> <width> <height>', file=sys.stderr)
		print('Scales a raw RGBA image from stdin to stdout.', file=sys.stderr)
		sys.exit(1)

	scaled = scale(bytearray(sys.stdin.buffer.read()), *map(int, sys.argv[1:]), ColorFormat.RGBA)

	sys.stdout.buffer.write(scaled)

if __name__ == '__main__':
	main()
