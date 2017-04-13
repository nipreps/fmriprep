#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fieldmap preprocessing workflow for fieldmap data structure
8.9.1 in BIDS 1.0.0: one phase diff and at least one magnitude image

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os.path as op

from nipype.interfaces import ants
from nipype.interfaces import fsl
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nipype.workflows.dmri.fsl.utils import siemens2rads, demean_image, cleanup_edge_pipeline
from niworkflows.interfaces.masks import BETRPT

from fmriprep.interfaces import ReadSidecarJSON, IntraModalMerge


def _sort_fmaps(input_images):
    ''' just a little data massaging'''
    return (sorted([fname for fname in input_images if 'magnitude' in fname]),
            sorted([fname for fname in input_images if 'phasediff' in fname]))


def phase_diff_and_magnitudes(settings, name='phase_diff_and_magnitudes'):
    """
    Estimates the fieldmap using a phase-difference image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions. The `original code was taken from nipype
    <https://github.com/nipy/nipype/blob/master/nipype/workflows/dmri/fsl/artifacts.py#L514>`_.

    Outputs::

      outputnode.fmap_ref - The average magnitude image, skull-stripped
      outputnode.fmap_mask - The brain mask applied to the fieldmap
      outputnode.fmap - The estimated fieldmap in Hz


    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['input_images']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap_ref', 'fmap_mask', 'fmap']), name='outputnode')

    sortfmaps = pe.Node(niu.Function(function=_sort_fmaps,
                                     input_names=['input_images'],
                                     output_names=['magnitude', 'phasediff']),
                        name='SortFmaps')

    def _pick1st(inlist):
        return inlist[0]

    # Read phasediff echo times
    meta = pe.Node(ReadSidecarJSON(), name='metadata')
    dte = pe.Node(niu.Function(input_names=['in_values'], output_names=['delta_te'],
                               function=_delta_te), name='ComputeDeltaTE')

    # Merge input magnitude images
    magmrg = pe.Node(IntraModalMerge(), name='MagnitudeFuse')

    # de-gradient the fields ("bias/illumination artifact")
    n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='MagnitudeBias')
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                  name='MagnitudeBET')
    # uses mask from bet; outputs a mask
    # dilate = pe.Node(fsl.maths.MathsCommand(
    #     nan2zeros=True, args='-kernel sphere 5 -dilM'), name='MskDilate')

    # phase diff -> radians
    pha2rads = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'],
        function=siemens2rads), name='PreparePhase')

    # FSL PRELUDE will perform phase-unwrapping
    prelude = pe.Node(fsl.PRELUDE(process3d=True), name='PhaseUnwrap')

    denoise = pe.Node(fsl.SpatialFilter(operation='median', kernel_shape='sphere',
                                        kernel_size=3), name='PhaseDenoise')

    demean = pe.Node(niu.Function(
        input_names=['in_file', 'in_mask'], output_names=['out_file'],
        function=demean_image), name='DemeanFmap')

    cleanup = cleanup_edge_pipeline()

    compfmap = pe.Node(niu.Function(
        input_names=['in_file', 'delta_te'], output_names=['out_file'],
        function=phdiff2fmap), name='ComputeFieldmap')

    # The phdiff2fmap interface is equivalent to:
    # rad2rsec (using rads2radsec from nipype.workflows.dmri.fsl.utils)
    # pre_fugue = pe.Node(fsl.FUGUE(save_fmap=True), name='ComputeFieldmapFUGUE')
    # rsec2hz (divide by 2pi)

    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, sortfmaps, [('input_images', 'input_images')]),
        (sortfmaps, meta, [(('phasediff', _pick1st), 'in_file')]),
        (sortfmaps, magmrg, [('magnitude', 'in_files')]),
        (magmrg, n4, [('out_avg', 'input_image')]),
        (n4, prelude, [('output_image', 'magnitude_file')]),
        (n4, bet, [('output_image', 'in_file')]),
        (bet, prelude, [('mask_file', 'mask_file')]),
        (sortfmaps, pha2rads, [(('phasediff', _pick1st), 'in_file')]),
        (pha2rads, prelude, [('out_file', 'phase_file')]),
        (meta, dte, [('out_dict', 'in_values')]),
        (dte, compfmap, [('delta_te', 'delta_te')]),
        (prelude, denoise, [('unwrapped_phase_file', 'in_file')]),
        (denoise, demean, [('out_file', 'in_file')]),
        (demean, cleanup, [('out_file', 'inputnode.in_file')]),
        (bet, cleanup, [('mask_file', 'inputnode.in_mask')]),
        (cleanup, compfmap, [('outputnode.out_file', 'in_file')]),
        (compfmap, outputnode, [('out_file', 'fmap')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'fmap_ref')])
    ])

    ds_betrpt = pe.Node(nio.DataSink(), name="BETRPTDS")
    ds_betrpt.inputs.base_directory = op.join(settings['output_dir'],
                                              'reports')

    workflow.connect([
        (bet, ds_betrpt, [('out_report', 'fmap_bet_rpt')])
    ])

    return workflow


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------

def phdiff2fmap(in_file, delta_te, out_file=None):
    """
    Converts the input phase-difference map into a fieldmap in Hz,
    using the eq. (1) of [Hutton2002]_:

      .. math:

          \Delta B_0 (\text{T}^{-1}) = \frac{\Delta \Theta}{2\pi\Gamma\,\Delta\text{TE}}

    In this case, we do not take into account the gyromagnetic ratio of the
    proton (:math:`\Gamma`), since it will be applied inside TOPUP:

      .. math:

          \Delta B_0 (\text{Hz}) = \frac{\Delta \Theta}{2\pi,\Delta\text{TE}}



      .. [Hutton2002] Hutton et al., Image Distortion Correction in fMRI: A Quantitative
                      Evaluation, NeuroImage 16(1):217-240, 2002. doi:`10.1006/nimg.2001.1054
                      <http://dx.doi.org/10.1006/nimg.2001.1054>`_.
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    import math

    #  GYROMAG_RATIO_H_PROTON_MHZ = 42.576

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_fmap.nii.gz' % fname)

    image = nb.load(in_file)
    data = (image.get_data().astype(np.float32) / (2. * math.pi * delta_te))

    nb.Nifti1Image(data, image.affine, image.header).to_filename(out_file)
    return out_file


def _delta_te(in_values, te1=None, te2=None):
    if isinstance(in_values, float):
        te2 = in_values
        te1 = 0.

    if isinstance(in_values, dict):
        te1 = in_values.get('EchoTime1')
        te2 = in_values.get('EchoTime2')

        if not all((te1, te2)):
            te2 = in_values.get('EchoTimeDifference')
            te1 = 0

    if isinstance(in_values, list):
        te2, te1 = in_values
        if isinstance(te1, list):
            te1 = te1[1]
        if isinstance(te2, list):
            te2 = te2[1]

    if te1 is None or te2 is None:
        raise RuntimeError(
            'No echo time information found')

    return abs(float(te2)-float(te1))
