#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-08-29 18:50:54
""" fmriprep setup script """
import os
import sys


def main():
    """ Install entry-point """
    from glob import glob
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages

    this_path = os.path.dirname(os.path.abspath(getfile(currentframe())))
    # Read vars from info file
    module_file = os.path.join(this_path, 'fmriprep', 'info.py')
    with open(module_file) as fvars:
       exec(compile(fvars.read(), module_file, 'exec'))

    REQ_LINKS = []
    with open('requirements.txt', 'r') as rfile:
        REQUIREMENTS = [line.strip() for line in rfile.readlines()]

    for i, req in enumerate(REQUIREMENTS):
        if req.startswith('-e'):
            REQUIREMENTS[i] = req.split('=')[1]
            REQ_LINKS.append(req.split()[1])

    if REQUIREMENTS is None:
        REQUIREMENTS = []

    setup(
        name=__packagename__,
        version=__version__,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email='cmoodie@stanford.edu',
        email=__email__,
        maintainer=__maintainer__,
        maintainer_email=__email__,
        url=__url__,
        download_url='https://pypi.python.org/packages/source/f/fmriprep/'
                     'fmriprep-%s.tar.gz' % __version__,
        license=__license__,
        entry_points={'console_scripts': ['fmriprep=fmriprep.run_workflow:main',]},
        packages=find_packages(),
        package_data={'fmriprep': ['data/*.json']},
        install_requires=REQUIREMENTS,
        dependency_links=REQ_LINKS,
        zip_safe=False,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: MRI processing',
            'Topic :: Scientific/Engineering :: Biomedical Imaging',
            'License :: OSI Approved :: 3-clause BSD License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5'
        ],
    )

    # if this is being set up in a git repo
    if glob('.git'):
        enforce_pep8()

def enforce_pep8():
    ''' copy commit hook to .git/hooks to ensure linting'''
    import shutil
    import warnings

    try:
        shutil.copy('./hooks/pre-commit', '.git/hooks/')
    except StandardError as serror:
        warnings.warn(RuntimeWarning("failed to set up git hooks"), serror)

if __name__ == '__main__':
    LOCAL_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(LOCAL_PATH)
    sys.path.insert(0, LOCAL_PATH)

    main()
