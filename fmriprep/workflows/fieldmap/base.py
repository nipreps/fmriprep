# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_base :

Automatic selection of the appropriate SDC method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the dataset metadata indicate tha more than one field map acquisition is
``IntendedFor`` (see BIDS Specification section 8.9) the following priority will
be used:

  1. :ref:`sdc_pepolar` (or **blip-up/blip-down**)

  2. :ref:`sdc_direct_b0`

  3. :ref:`sdc_phasediff`

  4. :ref:`sdc_fieldmapless`


Table of behavior (fieldmap use-cases):

=============== =========== ============= ===============
Fieldmaps found ``use_syn`` ``force_syn``     Action
=============== =========== ============= ===============
True            *           True          Fieldmaps + SyN
True            *           False         Fieldmaps
False           *           True          SyN
False           True        False         SyN
False           False       False         HMC only
=============== =========== ============= ===============


"""
from sdcflows.workflows.base import init_sdc_estimate_wf

__all__ = ['init_sdc_estimate_wf']
