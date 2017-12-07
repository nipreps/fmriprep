#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


def init_fmap_estimator_wf(fmap_bids, reportlets_dir, omp_nthreads,
                           fmap_bspline):
    """
    This workflow selects the fieldmap estimation data available for the subject and
    returns the estimated fieldmap in mm.

    Outputs:

        fmap
          The estimated fieldmap itself IN UNITS OF Hz.
        fmap_ref
          the anatomical reference for the fieldmap (magnitude image, corrected SEm, etc.)
        fmap_mask
          a brain mask for the fieldmap


    """

    # pybids type options: (phase1|phase2|phasediff|epi|fieldmap)
    # https://github.com/INCF/pybids/blob/213c425d8ee820f4b7a7ae96e447a4193da2f359/bids/grabbids/bids_layout.py#L63
    if fmap_bids['type'] == 'fieldmap':
        from .fmap import init_fmap_wf
        fmap_wf = init_fmap_wf(reportlets_dir=reportlets_dir,
                               omp_nthreads=omp_nthreads,
                               fmap_bspline=fmap_bspline)
        # set inputs
        fmap_wf.inputs.inputnode.fieldmap = fmap_bids['fieldmap']
        fmap_wf.inputs.inputnode.magnitude = fmap_bids['magnitude']
        return fmap_wf

    if fmap_bids['type'] == 'phasediff':
        from .phdiff import init_phdiff_wf
        phdiff_wf = init_phdiff_wf(reportlets_dir=reportlets_dir,
                                   omp_nthreads=omp_nthreads)
        # set inputs
        phdiff_wf.inputs.inputnode.phasediff = fmap_bids['phasediff']
        phdiff_wf.inputs.inputnode.magnitude = [
            fmap_ for key, fmap_ in sorted(fmap_bids.items())
            if key.startswith("magnitude")
        ]
        return phdiff_wf

    if fmap_bids['type'] in ['phase1', 'phase2']:
        raise NotImplementedError
