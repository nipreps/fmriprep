#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

Phase-difference B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The field inhomogeneity inside the scanner (fieldmap) is proportional to the
phase drift between two subsequent :abbr:`GRE (gradient recall echo)`
sequence.


Fieldmap preprocessing workflow for fieldmap data structure
8.9.1 in BIDS 1.0.0: one phase diff and at least one magnitude image

"""

from niworkflows.nipype.interfaces import ants, fsl, utility as niu
from niworkflows.nipype.pipeline import engine as pe
# Note that demean_image imports from nipype
from niworkflows.nipype.workflows.dmri.fsl.utils import siemens2rads, demean_image, \
    cleanup_edge_pipeline
from niworkflows.interfaces.masks import BETRPT

from ...interfaces import ReadSidecarJSON, IntraModalMerge, DerivativesDataSink


def init_phdiff_wf(reportlets_dir, omp_nthreads, name='phdiff_wf'):
    """
    Estimates the fieldmap using a phase-difference image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions. The `original code was taken from nipype
    <https://github.com/nipy/nipype/blob/master/nipype/workflows/dmri/fsl/artifacts.py#L514>`_.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.phdiff import init_phdiff_wf
        wf = init_phdiff_wf(reportlets_dir='.', omp_nthreads=1)


    Outputs::

      outputnode.fmap_ref - The average magnitude image, skull-stripped
      outputnode.fmap_mask - The brain mask applied to the fieldmap
      outputnode.fmap - The estimated fieldmap in Hz


    """

    inputnode = pe.Node(niu.IdentityInterface(fields=['magnitude', 'phasediff']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_ref', 'fmap_mask']), name='outputnode')

    def _pick1st(inlist):
        return inlist[0]

    # Read phasediff echo times
    meta = pe.Node(ReadSidecarJSON(), name='meta', mem_gb=0.01, run_without_submitting=True)
    dte = pe.Node(niu.Function(function=_delta_te), name='dte', mem_gb=0.01)

    # Merge input magnitude images
    magmrg = pe.Node(IntraModalMerge(), name='magmrg')

    # de-gradient the fields ("bias/illumination artifact")
    n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3, copy_header=True),
                 name='n4', n_procs=omp_nthreads)
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                  name='bet')
    ds_fmap_mask = pe.Node(DerivativesDataSink(
        base_directory=reportlets_dir, suffix='fmap_mask'), name='ds_fmap_mask',
        mem_gb=0.01, run_without_submitting=True)
    # uses mask from bet; outputs a mask
    # dilate = pe.Node(fsl.maths.MathsCommand(
    #     nan2zeros=True, args='-kernel sphere 5 -dilM'), name='MskDilate')

    # phase diff -> radians
    pha2rads = pe.Node(niu.Function(function=siemens2rads), name='pha2rads')

    # FSL PRELUDE will perform phase-unwrapping
    prelude = pe.Node(fsl.PRELUDE(), name='prelude')

    denoise = pe.Node(fsl.SpatialFilter(operation='median', kernel_shape='sphere',
                                        kernel_size=3), name='denoise')

    demean = pe.Node(niu.Function(function=demean_image), name='demean')

    cleanup_wf = cleanup_edge_pipeline(name="cleanup_wf")

    compfmap = pe.Node(niu.Function(function=phdiff2fmap), name='compfmap')

    # The phdiff2fmap interface is equivalent to:
    # rad2rsec (using rads2radsec from nipype.workflows.dmri.fsl.utils)
    # pre_fugue = pe.Node(fsl.FUGUE(save_fmap=True), name='ComputeFieldmapFUGUE')
    # rsec2hz (divide by 2pi)

    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, meta, [('phasediff', 'in_file')]),
        (inputnode, magmrg, [('magnitude', 'in_files')]),
        (magmrg, n4, [('out_avg', 'input_image')]),
        (n4, prelude, [('output_image', 'magnitude_file')]),
        (n4, bet, [('output_image', 'in_file')]),
        (bet, prelude, [('mask_file', 'mask_file')]),
        (inputnode, pha2rads, [('phasediff', 'in_file')]),
        (pha2rads, prelude, [('out', 'phase_file')]),
        (meta, dte, [('out_dict', 'in_values')]),
        (dte, compfmap, [('out', 'delta_te')]),
        (prelude, denoise, [('unwrapped_phase_file', 'in_file')]),
        (denoise, demean, [('out_file', 'in_file')]),
        (demean, cleanup_wf, [('out', 'inputnode.in_file')]),
        (bet, cleanup_wf, [('mask_file', 'inputnode.in_mask')]),
        (cleanup_wf, compfmap, [('outputnode.out_file', 'in_file')]),
        (compfmap, outputnode, [('out', 'fmap')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'fmap_ref')]),
        (inputnode, ds_fmap_mask, [('phasediff', 'source_file')]),
        (bet, ds_fmap_mask, [('out_report', 'in_file')]),
    ])

    return workflow


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------

def phdiff2fmap(in_file, delta_te, out_file=None):
    r"""
    Converts the input phase-difference map into a fieldmap in Hz,
    using the eq. (1) of [Hutton2002]_:

    .. math::

        \Delta B_0 (\text{T}^{-1}) = \frac{\Delta \Theta}{2\pi\gamma \Delta\text{TE}}


    In this case, we do not take into account the gyromagnetic ratio of the
    proton (:math:`\gamma`), since it will be applied inside TOPUP:

    .. math::

        \Delta B_0 (\text{Hz}) = \frac{\Delta \Theta}{2\pi \Delta\text{TE}}


    .. [Hutton2002] Hutton et al., Image Distortion Correction in fMRI: A Quantitative
                    Evaluation, NeuroImage 16(1):217-240, 2002. doi:`10.1006/nimg.2001.1054
                    <https://doi.org/10.1006/nimg.2001.1054>`_.

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

    return abs(float(te2) - float(te1))
