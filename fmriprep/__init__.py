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
