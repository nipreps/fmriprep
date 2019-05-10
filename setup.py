#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
""" fmriprep setup script """


def main():
    """ Install entry-point """
    from setuptools import setup
    from setuptools.extension import Extension
    from numpy import get_include
    from fmriprep.__about__ import __version__, DOWNLOAD_URL

    import versioneer
    cmdclass = versioneer.get_cmdclass()

    extensions = [Extension(
        "fmriprep.utils.maths",
        ["fmriprep/utils/maths.pyx"],
        include_dirs=[get_include(), "/usr/local/include/"],
        library_dirs=["/usr/lib/"]),
    ]

    setup(
        version=__version__,
        cmdclass=cmdclass,
        download_url=DOWNLOAD_URL,
        # Dependencies handling
        zip_safe=False,
        ext_modules=extensions,
    )


if __name__ == '__main__':
    main()
