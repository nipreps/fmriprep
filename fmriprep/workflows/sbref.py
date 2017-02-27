#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Preprocessing workflows for :abbr:`SB (single-band)`-reference (SBRef)
images.

Originally coded by Craig Moodie. Refactored by the CRN Developers.
"""

import os.path as op

from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import ants
from niworkflows.interfaces.masks import ComputeEPIMask

from fmriprep.utils.misc import _first
from fmriprep.interfaces import DerivativesDataSink
from fmriprep.workflows.fieldmap import sdc_unwarp


def sbref_preprocess(name='SBrefPreprocessing', settings=None):
    """SBref processing workflow"""

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['sbref', 'fmap', 'fmap_ref', 'fmap_mask']
        ),
        name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['sbref_unwarped', 'sbref_unwarped_mask']),
                         name='outputnode')
    # Unwarping
    unwarp = sdc_unwarp(settings=settings)

    mean = pe.Node(fsl.MeanImage(dimension='T'), name='SBRefMean')
    inu = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='SBRefBias')
    skullstripping = pe.Node(ComputeEPIMask(generate_report=True,
                                            dilation=1), name='SBRefSkullstripping')

    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='sbref_bet'),
        name='DS_Report'
    )

    workflow.connect([
        (inputnode, unwarp, [('fmap', 'inputnode.fmap'),
                             ('fmap_ref', 'inputnode.fmap_ref'),
                             ('fmap_mask', 'inputnode.fmap_mask')]),
        (inputnode, unwarp, [('sbref', 'inputnode.in_file')]),
        (unwarp, mean, [('outputnode.out_file', 'in_file')]),
        (mean, inu, [('out_file', 'input_image')]),
        (inu, skullstripping, [('output_image', 'in_file')]),
        (skullstripping, ds_report, [('out_report', 'in_file')]),
        (inputnode, ds_report, [(('sbref', _first), 'source_file')]),
        (skullstripping, outputnode, [('mask_file', 'sbref_unwarped_mask')]),
        (inu, outputnode, [('output_image', 'sbref_unwarped')])
    ])

    datasink = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='sdc'),
        name='datasink'
    )

    workflow.connect([
        (inputnode, datasink, [(('sbref', _first), 'source_file')]),
        (inu, datasink, [('output_image', 'in_file')])
    ])
    return workflow


def _extract_wm(in_file):
    import os.path as op
    import nibabel as nb
    import numpy as np

    image = nb.load(in_file)
    data = image.get_data().astype(np.uint8)
    data[data != 3] = 0
    data[data > 0] = 1

    out_file = op.abspath('wm_mask.nii.gz')
    nb.Nifti1Image(data, image.get_affine(), image.get_header()).to_filename(out_file)
    return out_file
