# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = 'The CRN developers'
__copyright__ = 'Copyright 2018, Center for Reproducible Neuroscience, Stanford University'
__credits__ = ['Craig Moodie', 'Ross Blair', 'Oscar Esteban', 'Chris Gorgolewski',
               'Shoshana Berleant', 'Christopher J. Markiewicz', 'Russell A. Poldrack']
__license__ = '3-clause BSD'
__maintainer__ = 'Ross Blair'
__email__ = 'crn.poldracklab@gmail.com'
__status__ = 'Prototype'
__url__ = 'https://github.com/poldracklab/fmriprep'
__packagename__ = 'fmriprep'
__description__ = ("FMRIprep is a functional magnetic resonance image pre-processing pipeline "
                   "that is designed to provide an easily accessible, state-of-the-art interface "
                   "that is robust to differences in scan acquisition protocols and that requires "
                   "minimal user input, while providing easily interpretable and comprehensive "
                   "error and output reporting.")
__longdesc__ = """\
Preprocessing of functional MRI (fMRI) involves numerous steps to clean and standardize
data before statistical analysis.
Generally, researchers create ad hoc preprocessing workflows for each new dataset,
building upon a large inventory of tools available for each step.
The complexity of these workflows has snowballed with rapid advances in MR data
acquisition and image processing techniques.
FMRIPrep is an analysis-agnostic tool that addresses the challenge of robust and
reproducible preprocessing for task-based and resting fMRI data.
FMRIPrep automatically adapts a best-in-breed workflow to the idiosyncrasies of
virtually any dataset, ensuring high-quality preprocessing with no manual intervention,
while providing easily interpretable and comprehensive error and output reporting.
It performs basic preprocessing steps (coregistration, normalization, unwarping, noise
component extraction, segmentation, skullstripping etc.) providing outputs that can be
easily submitted to a variety of group level analyses, including task-based or resting-state
fMRI, graph theory measures, surface or volume-based statistics, etc.

The workflow is based on `Nipype <https://nipype.readthedocs.io>`_ and encompases a large
set of tools from well-known neuroimaging packages, including
`FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`_,
`ANTs <https://stnava.github.io/ANTs/>`_,
`FreeSurfer <https://surfer.nmr.mgh.harvard.edu/>`_,
`AFNI <https://afni.nimh.nih.gov/>`_,
and `Nilearn <https://nilearn.github.io/>`_.
This pipeline was designed to provide the best software implementation for each state of
preprocessing, and will be updated as newer and better neuroimaging software becomes
available.

This tool allows you to easily do the following:

  * Take fMRI data from *unprocessed* (only reconstructed) to ready for analysis.
  * Implement tools from different software packages.
  * Achieve optimal data processing quality by using the best tools available.
  * Generate preprocessing-assessment reports, with which the user can easily identify problems.
  * Receive verbose output concerning the stage of preprocessing for each subject, including
    meaningful errors.
  * Automate and parallelize processing steps, which provides a significant speed-up from
    typical linear, manual processing.

FMRIPrep has the potential to transform fMRI research by equipping
neuroscientists with a high-quality, robust, easy-to-use and transparent preprocessing workflow
which can help ensure the validity of inference and the interpretability of their results.

[Pre-print doi:`10.1101/306951 <https://doi.org/10.1101/306951>`_]
[Documentation `fmriprep.org <https://fmriprep.readthedocs.io>`_]
[Software doi:`10.5281/zenodo.852659 <https://doi.org/10.5281/zenodo.852659>`_]
[Support `neurostars.org <https://neurostars.org/tags/fmriprep>`_]
"""

DOWNLOAD_URL = (
    'https://github.com/poldracklab/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))


SETUP_REQUIRES = [
    'setuptools>=18.0',
    'numpy',
    'cython',
]

REQUIRES = [
    'numpy',
    'lockfile',
    'future',
    'scikit-learn',
    'matplotlib>=2.2.0',
    'nilearn',
    'sklearn',
    'nibabel>=2.2.1',
    'pandas',
    'grabbit==0.2.3',
    'pybids==0.6.5',
    'nitime',
    'nipype>=1.1.5',
    'niworkflows==0.5.1',
    'statsmodels',
    'seaborn',
    'indexed_gzip>=0.8.2',
    'scikit-image',
    'versioneer',
    'pyyaml',
]

LINKS_REQUIRES = [
    'git+https://github.com/nipy/nipype.git@'
    'ac4076480ef55eb2a79aff9eff7cbf48394c565b#egg=nipype',
]

TESTS_REQUIRES = [
    "mock",
    "codecov",
    "pytest",
]

EXTRA_REQUIRES = {
    'doc': [
        'sphinx>=1.5.3',
        'sphinx_rtd_theme',
        'sphinx-argparse',
        'pydotplus',
        'pydot>=1.2.3',
        'packaging',
        'nbsphinx',
    ],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
    'datalad': ['datalad'],
    'resmon': ['psutil>=5.4.0'],
    'sentry': ['sentry-sdk>=0.5.3'],
}
EXTRA_REQUIRES['docs'] = EXTRA_REQUIRES['doc']

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = list(set([
    v for deps in EXTRA_REQUIRES.values() for v in deps]))

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]
