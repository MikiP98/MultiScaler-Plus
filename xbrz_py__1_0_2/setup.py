#!/usr/bin/env python3

import re
from setuptools import setup, Extension
from setuptools.extension import Library
from setuptools.command.build_ext import build_ext

with open('README.md') as f:
	long_description = f.read()

version = ''
with open('xbrz.py') as f:
	m = re.search(r"""^__version__\s*=\s*['"]([^'"]*)['"]""", f.read(), re.MULTILINE)
	if m:
		version = m.group(1)

if not version:
	raise RuntimeError('version is not set')

copt = {
	'msvc': ['/O2', '/opt:none', '/std:c++17'],
	'unix': ['-O3', '-std=gnu++17', '-g0'],
}

class build_ext_subclass(build_ext):
	def build_extensions(self):
		c = self.compiler.compiler_type
		if c in copt:
			for e in self.extensions:
				e.extra_compile_args = copt[c]
		super().build_extensions()

setup(
	name='xbrz.py',
	author='iomintz',
	author_email='io@mintz.cc',
	version=version,
	description='ctypes-based binding library for the xBRZ pixel-art image scaling algorithm',
	long_description=long_description,
	long_description_content_type='text/markdown',
	license='AGPL-3.0-or-later',
	url='https://github.com/iomintz/xbrz.py',
	python_requires='>=3.6.0',

	py_modules=['xbrz'],
	ext_modules=[
		Extension(
			'_xbrz',
			['lib/xbrz.cpp'],
			include_dirs=['lib/'],
			extra_compile_args=['-std=gnu++17', '-g0'],
		),
	],
	cmdclass= {'build_ext': build_ext_subclass},
	extras_require={
		'wand': 'Wand>=0.6.1,<1.0.0',
		'pillow': 'Pillow>=7.1.2,<8.0.0',
	},

	classifiers=[
		'Topic :: Software Development',
		'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
		'Development Status :: 5 - Production/Stable',
		'Programming Language :: Python :: 3 :: Only',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
		'Programming Language :: C++',
		'Topic :: Multimedia :: Graphics',
	],
)
