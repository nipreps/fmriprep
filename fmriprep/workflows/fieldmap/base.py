#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base fieldmap estimation
------------------------

* Subject can have phase-difference data.
* Subject can have a fieldmap acquisition.

"""
from __future__ import print_function, division, absolute_import, unicode_literals


def init_fmap_estimator_wf(fmap_bids, reportlets_dir, omp_nthreads,
                           fmap_bspline):
    """
    This workflow selects the fieldmap estimation data available for the subject and
    returns the estimated fieldmap in mm, along with a corresponding reference image.
    Current implementation applies the first type of fieldmap estimation found in the
    following order of precedence:

      * "Natural" fieldmaps
      * Phase-difference fieldmaps

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
        phdiff_wf = init_phdiff_wf(reportlets_dir=reportlets_dir)
        # set inputs
        phdiff_wf.inputs.inputnode.phasediff = fmap_bids['phasediff']
        phdiff_wf.inputs.inputnode.magnitude = [
            fmap_bids['magnitude1'],
            fmap_bids['magnitude2']
        ]
        return phdiff_wf

    if fmap_bids['type'] in ['phase1', 'phase2', 'epi']:
        raise NotImplementedError
