#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
One of the major problems that affects :abbr:`EPI (echo planar imaging)` data
is the spatial distortion caused by the inhomogeneity of the field inside
the scanner.
The are four broad families of methodologies for mapping the field:

  * **Phase-difference techniques**: to estimate the fieldmap, these methods
    measure the phase evolution in time between two close
    :abbr:`GRE (Gradient Recall Echo)` acquisitions. Corresponds to the sections
    8.9.1 and 8.9.2 of the BIDS specification.
  * **Direct field-mapping**: some sequences (such as :abbr:`SE (spiral echo)`)
    are able to measure the fieldmap directly. Corresponds to section 8.9.3 of BIDS.
  * **Blip-up/blip-down or phase encoding polarity**:
    :abbr:`pepolar (Phase Encoding POLARity)`: techniques acquire at least two images
    distorted due to the inhomogeneity of the field, but varying in
    :abbr:`PE (phase-encoding)` direction.
    Corresponds to 8.9.4 of BIDS.
  * **Point-spread function acquisition**: not supported by FMRIPREP.


Once the field-map is estimated, the distortion can be accounted for.
Fieldmap processing in FMRIPREP is structured as follows:

  1. :ref:`sdc-base`: the BIDS structure is queried to find the available field-mapping
     techniques and the appropriate processing workflows are set-up.

  2. :ref:`sdc-estimation`: all the estimation workflows produce a displacement field
     ready to be used in the correction step.

  3. :ref:`sdc-unwarp`: the correction step is applied.


Contents
--------

.. toctree::
    :maxdepth: 2

    sdc/base
    sdc/estimation
    sdc/unwarping


"""
from __future__ import print_function, division, absolute_import, unicode_literals

from .base import fmap_estimator
from .unwarp import sdc_unwarp
from .utils import create_encoding_file, mcflirt2topup

PEPOLAR_MODALITIES = ['epi', 'sbref']
