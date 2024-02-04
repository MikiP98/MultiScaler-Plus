# xbrz.py

[![Build Status](https://travis-ci.com/ioistired/xbrz.py.svg?branch=master)](https://travis-ci.org/iomintz/xbrz.py)

xbrz.py is a simple ctypes-based binding library for [xBRZ], a high-quality pixel-art image scaling algorithm.

## Installation

Wheels are available for many platforms. If there isn't one for your platform, make sure you have a C++ compiler handy.

```
pip install xbrz.py
```

## Usage

```py
import xbrz

# pixels is a list of 32 bit ints representing RGBA colors.
# It is 32 * 24 long.
pixels = ...
scaled_pixels = xbrz.scale(pixels, 6, 32, 24, xbrz.ColorFormat.RGBA)

# scaled_pixels is a 32 * 24 * 6 ** 2 long list of 32 bit ints representing the scaled image.
```

## Wand / Pillow support

You can pass a Wand image to `xbrz.scale_wand(img, factor)` or a Pillow image to `xbrz.scale_pillow(img, factor)`.
Neither libraries are required to use xbrz.py, however they can be installed via:

```
pip install xbrz.py[wand]
# or
pip install xbrz.py[pillow]
```

## xbrz.py as an executable module

Passing raw RGBA pixels to `python3 -m xbrz <factor> <width> <height>`
via stdin will output scaled raw RGBA pixels to stdout.

## License

AGPLv3, see LICENSE.md. The original xBRZ code is GPLv3 licensed.

- lib/ is based on code provided by Zenju under the GPLv3 license. See lib/License.txt for details.
  Some changes were made:
  - Added some `extern "C"` declarations to the functions I intended to call from python.
  - Removed some namespace use to avoid being mangled.
  - Replaced a C++ template with a simple function that takes two arguments.
  - Converted the library to use RGBA instead of ARGB.
- xbrz.py is based on lib/ and is released under the AGPLv3 license, see LICENSE.md for details.

[xbrz]: https://sourceforge.net/projects/xbrz/
