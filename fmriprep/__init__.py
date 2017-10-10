#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This pipeline is developed by the Poldrack lab at Stanford University
(https://poldracklab.stanford.edu/) for use at
the Center for Reproducible Neuroscience (http://reproducibility.stanford.edu/),
as well as for open-source software distribution.
"""

from .info import (
    __version__,
    __author__,
    __copyright__,
    __credits__,
    __license__,
    __maintainer__,
    __email__,
    __status__,
    __url__,
    __packagename__,
    __description__,
    __longdesc__
)
from .due import due, Url, Doi

import warnings

# cmp is not used by fmriprep, so ignore nipype-generated warnings
warnings.filterwarnings('ignore', r'cmp not installed')

# Monkey-patch to ignore AFNI upgrade warnings
from niworkflows.nipype.interfaces.afni import Info  # noqa: E402

_old_version = Info.version


def _new_version():
    from niworkflows.nipype import logging
    iflogger = logging.getLogger('interface')
    level = iflogger.getEffectiveLevel()
    iflogger.setLevel('ERROR')
    v = _old_version()
    iflogger.setLevel(level)
    if v is None:
        iflogger.warn('afni_vcheck executable not found')
    return v
Info.version = staticmethod(_new_version)

due.cite(
    # Chicken/egg problem with Zenodo here regarding DOI.  Might need
    # custom Zenodo?  TODO: add DOI for a Zenodo entry when available
    Doi("10.5281/zenodo.996169"),
    Url('http://fmriprep.readthedocs.io'),
    description="A Robust Preprocessing Pipeline for fMRI Data",
    version=__version__,
    # Most likely that eventually you might not need to explicitly demand
    # citing the module merely on input, but since it is unlikely to be imported
    # unless used, forcing citation "always"
    cite_module=True,
    path="fmriprep"
)
