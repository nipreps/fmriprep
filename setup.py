#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" fmriprep setup script """
from setuptools import setup
import versioneer


if __name__ == "__main__":
    setup(
        name="fmriprep",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )
