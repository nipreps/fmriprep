#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fieldmap-processing workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the major problems that affects EPI data is spatial distortion.
It is possible to correct for distortions in EPI data posthoc using fieldmap
estimations.
There are three broad families of methodologies to estimate the inhomogeneity
of the field (fieldmap) inside the scanner:

  * "phdiff": measuring the phase evolution in time, between two close
    :abbr:`GREs (Gradient Recall Echo)`. Corresponds to the sections
    8.9.1 and 8.9.2 of the BIDS specification.
  * "fmap": some sequences (such as :abbr:`SE (spiral echo)`) are able to
    measure the fieldmap directly. Corresponds to section 8.9.3 of BIDS.
  * :abbr:`pepolar (Phase Encoding POLARity)`:
    acquiring multiple images distorted due to the inhomogeneity
    of the field, but varying in :abbr:`PE (phase-encoding)` direction.
    This type of estimation is better known by "TOPUP" (the FSL's tool to compute it).
    Corresponds to 8.9.4 of BIDS.

Once the fieldmap has been estimated and massaged to the appropriate format to be
digested by FSL's ApplyTOPUP, the distorted images can be corrected using the mentioned
tool. That is implemented in the unwarp module.


"""
from __future__ import print_function, division, absolute_import, unicode_literals

from .base import fmap_estimator
from .unwarp import sdc_unwarp
from .utils import create_encoding_file, mcflirt2topup
