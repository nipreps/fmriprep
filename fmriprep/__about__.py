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
__credits__ = ('Contributors: please check the ``.zenodo.json`` file at the top-level folder'
               'of the repository')
__license__ = '3-clause BSD'
__maintainer__ = 'Oscar Esteban'
__email__ = 'code@oscaresteban.es'
__status__ = 'Prototype'
__url__ = 'https://github.com/poldracklab/fmriprep'
__packagename__ = 'fmriprep'
__description__ = """\
fMRIPrep is a robust and easy-to-use pipeline for preprocessing of diverse fMRI data.
The transparent workflow dispenses of manual intervention, thereby ensuring the reproducibility
of the results"""
__longdesc__ = """\
Preprocessing of functional MRI (fMRI) involves numerous steps to clean and standardize
the data before statistical analysis.
Generally, researchers create ad hoc preprocessing workflows for each dataset,
building upon a large inventory of available tools.
The complexity of these workflows has snowballed with rapid advances in
acquisition and processing.
fMRIPrep is an analysis-agnostic tool that addresses the challenge of robust and
reproducible preprocessing for task-based and resting fMRI data.
fMRIPrep automatically adapts a best-in-breed workflow to the idiosyncrasies of
virtually any dataset, ensuring high-quality preprocessing without manual intervention.
fMRIPrep robustly produces high-quality results on diverse fMRI data.
Additionally, fMRIPrep introduces less uncontrolled spatial smoothness than observed
with commonly used preprocessing tools.
fMRIPrep equips neuroscientists with an easy-to-use and transparent preprocessing
workflow, which can help ensure the validity of inference and the interpretability
of results.

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

fMRIPrep performs basic preprocessing steps (coregistration, normalization, unwarping, noise
component extraction, segmentation, skullstripping etc.) providing outputs that can be
easily submitted to a variety of group level analyses, including task-based or resting-state
fMRI, graph theory measures, surface or volume-based statistics, etc.
fMRIPrep allows you to easily do the following:

  * Take fMRI data from *unprocessed* (only reconstructed) to ready for analysis.
  * Implement tools from different software packages.
  * Achieve optimal data processing quality by using the best tools available.
  * Generate preprocessing-assessment reports, with which the user can easily identify problems.
  * Receive verbose output concerning the stage of preprocessing for each subject, including
    meaningful errors.
  * Automate and parallelize processing steps, which provides a significant speed-up from
    typical linear, manual processing.

[Nat Meth doi:`10.1038/s41592-018-0235-4 <https://doi.org/10.1038/s41592-018-0235-4>`_]
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
    'grabbit==0.2.3',
    'indexed_gzip>=0.8.8',
    'nibabel>=2.2.1',
    'nilearn',
    'nipype>=1.1.7',
    'nitime',
    'niworkflows>=0.5.2.post5,<0.5.3',
    'numpy',
    'pandas',
    'pybids==0.6.5',
    'pyyaml',
    'scikit-image',
    'statsmodels',
    'tedana>=0.0.5',
]


LINKS_REQUIRES = [
]

TESTS_REQUIRES = [
    "codecov",
    "pytest",
]

EXTRA_REQUIRES = {
    'datalad': ['datalad'],
    'doc': [
        'nbsphinx',
        'packaging',
        'pydot>=1.2.3',
        'pydotplus',
        'sphinx>=1.5.3',
        'sphinx-argparse',
        'sphinx_rtd_theme',
    ],
    'duecredit': ['duecredit'],
    'resmon': ['psutil>=5.4.0'],
    'sentry': ['sentry-sdk>=0.5.3'],
    'tests': TESTS_REQUIRES,
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
