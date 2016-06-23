#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os.path as op

from nipype.interfaces import fsl
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from nipype.pipeline import engine as pe

from fmriprep.utils.misc import gen_list
from fmriprep.interfaces import ReadSidecarJSON
from fmriprep.viz import stripped_brain_overlay

def se_pair_workflow(name='Fieldmap_SEs', settings=None):  # pylint: disable=R0914
    """
    Estimates the fieldmap using TOPUP on series of :abbr:`SE (Spin-Echo)` images
    acquired with varying :abbr:`PE (phase encoding)` direction.
    """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['fieldmaps']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=[
        'out_field', 'fmap_mask', 'mag_brain', 'fmap_fieldcoef', 'fmap_movpar']), name='outputnode')

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(fields=['TotalReadoutTime', 'PhaseEncodingDirection']),
                      iterfield=['in_file'], name='metadata')

    encfile = pe.Node(interface=niu.Function(
        input_names=['fieldmaps', 'in_dict'], output_names=['parameters_file'],
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
    mag_bet = pe.Node(fsl.BET(mask=True, robust=True), name='SE_brain')

    workflow.connect([
        (inputnode, meta, [('fieldmaps', 'in_file')]),
        (inputnode, encfile, [('fieldmaps', 'fieldmaps')]),
        (inputnode, fslmerge, [('fieldmaps', 'in_files')]),
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

        (topup, outputnode, [('out_field', 'out_field')]),
        (mag_bet, outputnode, [('out_file', 'mag_brain'),
                               ('mask_file', 'fmap_mask')]),
        (topup, outputnode, [('out_fieldcoef', 'fmap_fieldcoef'),
                             ('out_movpar', 'fmap_movpar')])
    ])

    # Reports section
    se_png = pe.Node(niu.Function(
        input_names=['in_file', 'overlay_file', 'out_file'], output_names=['out_file'],
        function=stripped_brain_overlay), name='PNG_SE_corr')
    se_png.inputs.out_file = 'corrected_SE_and_mask.png'

    datasink = pe.Node(nio.DataSink(base_directory=op.join(settings['work_dir'], 'images')),
                       name='datasink', parameterization=False)
    workflow.connect([
        (unwarp_mag, se_png, [('out_corrected', 'overlay_file')]),
        (mag_bet, se_png, [('mask_file', 'in_file')]),
        (se_png, datasink, [('out_file', '@corrected_SE_and_mask')])
    ])

    return workflow

def create_encoding_file(fieldmaps, in_dict):
    """Creates a valid encoding file for topup"""
    import json
    import nibabel as nb
    import numpy as np
    import os

    if not isinstance(fieldmaps, list):
        fieldmaps = [fieldmaps]
    if not isinstance(in_dict, list):
        in_dict = [in_dict]

    pe_dirs = {'i': 0, 'j': 1, 'k': 2}
    enc_table = []
    for fmap, meta in zip(fieldmaps, in_dict):
        line_values = [0, 0, 0, meta['TotalReadoutTime']]
        line_values[pe_dirs[meta['PhaseEncodingDirection'][0]]] = 1 + (
            -2*(len(meta['PhaseEncodingDirection']) == 2))

        nvols = 1
        if len(nb.load(fmap).shape) > 3:
            nvols = nb.load(fmap).shape[3]

        enc_table += [line_values] * nvols

    np.savetxt(os.path.abspath('parameters.txt'), enc_table, fmt=['%0.1f', '%0.1f', '%0.1f', '%0.20f'])
    return os.path.abspath('parameters.txt')
