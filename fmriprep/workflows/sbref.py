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
    unwarp = sdc_unwarp()
    unwarp.inputs.inputnode.hmc_movpar = ''

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

###################################
# Deprecated code
###################################


def sbref_workflow_deprecated(name='SBrefPreprocessing', settings=None):
    """Legacy SBref processing workflow"""
    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['sbref', 'mag_brain', 'fmap_scaled', 'fmap_mask',
                    'fmap_unmasked', 'in_topup', 't1', 't1_brain', 't1_seg']
        ),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['sbref_unwarped', 'sbref_fmap', 'mag2sbref_matrix',
                    'sbref_brain', 'sbref_brain_corrected', 't1_brain',
                    'wm_seg', 'sbref_2_t1_transform']
        ),
        name='outputnode'
    )

    #  Skull strip SBRef to get reference brain
    sbref_bet = pe.Node(
        BETRPT(generate_report=True, mask=True, functional=True, frac=0.6), name="sbref_bet")

    #  Unwarp SBRef using Fugue  (N.B. duplicated in epi_reg_workflow!!!!!)
    fugue_sbref = pe.Node(
        fsl.FUGUE(unwarp_direction='x', dwell_time=dwell_time),
        name="SBRef_Unwarping"
    )

    strip_corrected_sbref = pe.Node(BETRPT(mask=True, frac=0.6, robust=True),
                                    name="BET_Corrected_SBRef")

    flt_sbref_brain_t1_brain = pe.Node(
        fsl.FLIRT(dof=6, bins=640, cost_func='mutualinfo'),
        name='sbref_brain_2_t1_brain_affine_transform'
    )

    flt_sbref_2_t1 = pe.Node(
        fsl.FLIRT(dof=6, bins=640, cost_func='mutualinfo'),
        name='sbref_2_t1_affine_transform'
    )

    # Affine transform of T1 segmentation into SBRref space
    flt_wmseg_sbref = pe.Node(fsl.FLIRT(dof=6, bins=64, cost_func='mutualinfo'),
                              name="WMSeg_2_SBRef_Brain_Affine_Transform")

    invert_wmseg_sbref = pe.Node(
        fsl.ConvertXFM(invert_xfm=True), name="invert_wmseg_sbref"
    )

    #  Run the commands from epi_reg_dof
    #  WITH FIELDMAP (unwarping steps)
    #  flirt -in ${fmapmagbrain} -ref ${vrefbrain} -dof ${dof} -omat
    #  ${vout}_fieldmap2str_init.mat
    flt_fmap_mag_brain_sbref_brain = pe.Node(
        fsl.FLIRT(dof=6, bins=640, cost_func='mutualinfo'),
        name="Fmap_Mag_Brain_2_SBRef_Brain_Affine_Transform"
    )

    #  flirt -in ${fmapmaghead} -ref ${vrefhead} -dof ${dof} -init
    #  ${vout}_fieldmap2str_init.mat -omat ${vout}_fieldmap2str.mat -out
    #  ${vout}_fieldmap2str -nosearch
    flt_fmap_mag_sbref = pe.Node(
        fsl.FLIRT(dof=6, no_search=True, bins=640, cost_func='mutualinfo'),
        name="Fmap_Mag_2_SBRef_Affine_Transform"
    )

    #  the following is a NEW HACK to fix extrapolation when fieldmap is too
    #  small
    #  applywarp -i ${vout}_fieldmaprads_unmasked -r ${vrefhead}
    #  --premat=${vout}_fieldmap2str.mat -o ${vout}_fieldmaprads2str_pad0
    aw_fmap_unmasked_sbref = pe.Node(fsl.ApplyWarp(relwarp=True),
                                     name="Apply_Warp_Fmap_Unmasked_2_SBRef")

    # fslmaths ${vout}_fieldmaprads2str_pad0 -abs -bin
    # ${vout}_fieldmaprads2str_innermask
    fmap_unmasked_abs = pe.Node(
        fsl.UnaryMaths(operation='abs'), name="Abs_Fmap_Unmasked_Warp")
    fmap_unmasked_bin = pe.Node(
        fsl.UnaryMaths(operation='bin'), name="Binarize_Fmap_Unmasked_Warp")

    # fugue --loadfmap=${vout}_fieldmaprads2str_pad0
    # --mask=${vout}_fieldmaprads2str_innermask --unmaskfmap
    # --unwarpdir=${fdir} --savefmap=${vout}_fieldmaprads2str_dilated
    fugue_dilate = pe.Node(
        fsl.FUGUE(unwarp_direction='x', dwell_time=dwell_time,
                  save_unmasked_fmap=True),
        name="Fmap_Dilating"
    )

    # epi_reg --epi=sub-S2529LVY1263171_task-nback_run-1_bold_brain
    # --t1=../Preprocessing_test_workflow/_subject_id_S2529LVY1263171/Bias_Field_Correction/sub-S2529LVY1263171_run-1_T1w_corrected.nii.gz
    # --t1brain=sub-S2529LVY1263171_run-1_T1w_corrected_bet_brain.nii.gz
    # --out=sub-S2529LVY1263171/func/sub-S2529LVY1263171_task-nback_run-1_bold_undistorted
    # --fmap=Topup_Fieldmap_rad.nii.gz
    # --fmapmag=fieldmap_topup_corrected.nii.gz
    # --fmapmagbrain=Magnitude_brain.nii.gz --echospacing=dwell_time
    # --pedir=x- -v

    workflow.connect([
        (inputnode, flt_wmseg_sbref, [('sbref', 'reference')]),
        (inputnode, flt_wmseg_sbref, [('t1_seg', 'in_file')]),
        (flt_wmseg_sbref, invert_wmseg_sbref, [('out_matrix_file', 'in_file')]),
        (flt_wmseg_sbref, outputnode, [('out_file', 'wm_seg')]),
        (invert_wmseg_sbref, outputnode, [('out_file', 'sbref_2_t1_transform')]),
        (inputnode, sbref_bet, [('sbref', 'in_file')]),
        (inputnode, fugue_sbref, [
            ('sbref', 'in_file'),
            ('fmap_scaled', 'fmap_in_file')
        ]),
        (inputnode, flt_fmap_mag_brain_sbref_brain, [('mag_brain', 'in_file')]),
        (inputnode, fugue_sbref, [('fmap_mask', 'mask_file')]),
        (inputnode, flt_fmap_mag_sbref, [('in_topup', 'in_file')]),
        (inputnode, aw_fmap_unmasked_sbref, [('fmap_unmasked', 'in_file')]),
        #  (inputnode, flt_sbref_brain_t1_brain, [('stripped_t1', 'reference')]),
        #  (strip_corrected_sbref, flt_sbref_brain_t1_brain, [('out_file', 'in_file')]),
        #  (fugue_sbref, flt_sbref_2_t1, [('unwarped_file', 'in_file')]),
        #  (inputnode, flt_sbref_2_t1, [('t1', 'reference')]),
        #  (flt_sbref_brain_t1_brain, flt_sbref_2_t1, [('out_matrix_file', 'in_matrix_file')]),
        #  (inputnode, SBRef_skull_strip, [("sbref", "in_file")]),
        (fugue_sbref, strip_corrected_sbref, [('unwarped_file', 'in_file')]),
        # might need to switch to [strip_corrected_sbref, "in_file"] here
        # instead of [sbref_bet, "out_file"]
        # might need to switch to [strip_corrected_sbref, "in_file"] here
        # instead of [sbref_bet, "out_file"]
        (sbref_bet, flt_fmap_mag_brain_sbref_brain, [('out_file', 'reference')]),
        (fugue_sbref, flt_fmap_mag_sbref, [('unwarped_file', 'reference')]),
        (flt_fmap_mag_brain_sbref_brain, flt_fmap_mag_sbref, [
            ('out_matrix_file', 'in_matrix_file')
        ]),
        (flt_fmap_mag_sbref, aw_fmap_unmasked_sbref, [('out_matrix_file', 'premat')]),
        (fugue_sbref, aw_fmap_unmasked_sbref, [('unwarped_file', 'ref_file')]),
        (aw_fmap_unmasked_sbref, fmap_unmasked_abs, [('out_file', 'in_file')]),
        (fmap_unmasked_abs, fmap_unmasked_bin, [('out_file', 'in_file')]),
        (aw_fmap_unmasked_sbref, fugue_dilate, [('out_file', 'fmap_in_file')]),
        (fmap_unmasked_bin, fugue_dilate, [('out_file', 'mask_file')]),
        (sbref_bet, outputnode, [('out_file', 'sbref_brain')]),
        (fugue_sbref, outputnode, [('unwarped_file', 'sbref_unwarped')]),
        (fugue_dilate, outputnode, [('fmap_out_file', 'sbref_fmap')]),
        (flt_fmap_mag_sbref, outputnode, [('out_matrix_file', 'mag2sbref_matrix')]),
        (strip_corrected_sbref, outputnode, [('out_file', 'sbref_brain_corrected')]),
    ])
    return workflow
