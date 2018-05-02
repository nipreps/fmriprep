#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" fmriprep wrapper setup script """

from setuptools import setup, find_packages
from os import path as op
import runpy


def main():
    """ Install entry-point """
    this_path = op.abspath(op.dirname(__file__))

    info = runpy.run_path(op.join(this_path, 'fmriprep_docker.py'))

    setup(
        name=info['__packagename__'],
        version=info['__version__'],
        description=info['__description__'],
        long_description=info['__longdesc__'],
        author=info['__author__'],
        author_email=info['__email__'],
        maintainer=info['__maintainer__'],
        maintainer_email=info['__email__'],
        url=info['__url__'],
        license=info['__license__'],
        classifiers=info['CLASSIFIERS'],
        # Dependencies handling
        setup_requires=[],
        install_requires=[],
        tests_require=[],
        extras_require={},
        dependency_links=[],
        package_data={},
        py_modules=["fmriprep_docker"],
        entry_points={'console_scripts': ['fmriprep-docker=fmriprep_docker:main']},
        packages=find_packages(),
        zip_safe=False
    )


if __name__ == '__main__':
    main()
