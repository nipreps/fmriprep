#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" fmriprep setup script """
import sys
from setuptools import setup
from setuptools.extension import Extension
import versioneer


# Give setuptools a hint to complain if it's too old a version
# 30.3.0 allows us to put most metadata in setup.cfg
# Should match pyproject.toml
# Not going to help us much without numpy or new pip, but gives us a shot
SETUP_REQUIRES = ['setuptools >= 40.8', 'numpy', 'cython']
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []


if __name__ == '__main__':
    from numpy import get_include

    extensions = [Extension(
        "fmriprep.utils.maths",
        ["fmriprep/utils/maths.pyx"],
        include_dirs=[get_include(), "/usr/local/include/"],
        library_dirs=["/usr/lib/"]),
    ]

    setup(name='fmriprep',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          setup_requires=SETUP_REQUIRES,
          ext_modules=extensions,
          )
