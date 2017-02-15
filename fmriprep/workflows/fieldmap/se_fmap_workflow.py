#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from nipype.pipeline import engine as pe
from niworkflows.interfaces.masks import BETRPT

from fmriprep.utils.misc import gen_list
from fmriprep.interfaces import ReadSidecarJSON
from fmriprep.workflows.fieldmap.utils import create_encoding_file

WORKFLOW_NAME = 'Fieldmap_SEs'


# pylint: disable=R0914
def se_fmap_workflow(name=WORKFLOW_NAME, settings=None):
    """
    Estimates the fieldmap using TOPUP on series of :abbr:`SE (Spin-Echo)` images
    acquired with varying :abbr:`PE (phase encoding)` direction.

    Outputs::

      outputnode.mag_brain - The average magnitude image, skull-stripped
      outputnode.fmap_mask - The brain mask applied to the fieldmap
      outputnode.fieldmap - The estimated fieldmap in Hz

    """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['input_images']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fieldmap', 'fmap_mask', 'mag_brain']), name='outputnode')

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(fields=['TotalReadoutTime', 'PhaseEncodingDirection']),
                      iterfield=['in_file'], name='metadata')

    encfile = pe.Node(interface=niu.Function(
        input_names=['input_images', 'in_dict'], output_names=['parameters_file'],
        function=create_encoding_file), name='TopUp_encfile', updatehash=True)

    # Head motion correction
    fslmerge = pe.Node(fsl.Merge(dimension='t'), name='SE_merge')
    hmc_se = pe.Node(fsl.MCFLIRT(cost='normcorr', mean_vol=True), name='SE_head_motion_corr')
    fslsplit = pe.Node(fsl.Split(dimension='t'), name='SE_split')

    # Run topup to estimate field distortions, do not estimate movement
    # since it is done in hmc_se
    topup = pe.Node(fsl.TOPUP(estmov=0), name='TopUp')

    # Use the least-squares method to correct the dropout of the SE images
    unwarp_mag = pe.Node(fsl.ApplyTOPUP(method='lsr'), name='TopUpApply')

    # Remove bias
    inu_n4 = pe.Node(N4BiasFieldCorrection(dimension=3), name='SE_bias')

    # Skull strip corrected SE image to get reference brain and mask
    mag_bet = pe.Node(BETRPT(mask=True, robust=True), name='SE_brain')

    workflow.connect([
        (inputnode, meta, [('input_images', 'in_file')]),
        (inputnode, encfile, [('input_images', 'input_images')]),
        (inputnode, fslmerge, [('input_images', 'in_files')]),
        (fslmerge, hmc_se, [('merged_file', 'in_file')]),
        (meta, encfile, [('out_dict', 'in_dict')]),
        (encfile, topup, [('parameters_file', 'encoding_file')]),
        (hmc_se, topup, [('out_file', 'in_file')]),
        (topup, unwarp_mag, [('out_fieldcoef', 'in_topup_fieldcoef'),
                             ('out_movpar', 'in_topup_movpar')]),
        (encfile, unwarp_mag, [('parameters_file', 'encoding_file')]),
        (hmc_se, fslsplit, [('out_file', 'in_file')]),
        (fslsplit, unwarp_mag, [('out_files', 'in_files'),
                                (('out_files', gen_list), 'in_index')]),
        (unwarp_mag, inu_n4, [('out_corrected', 'input_image')]),
        (inu_n4, mag_bet, [('output_image', 'in_file')]),

        (topup, outputnode, [('out_field', 'fieldmap')]),
        (mag_bet, outputnode, [('out_file', 'mag_brain'),
                               ('mask_file', 'fmap_mask')])
    ])

    return workflow
