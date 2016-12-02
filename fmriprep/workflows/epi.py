#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
EPI MRI -processing workflows.

Originally coded by Craig Moodie. Refactored by the CRN Developers.
"""

import os
import os.path as op

from nipype.pipeline import engine as pe
from nipype.interfaces import ants
from nipype.interfaces import c3
from nipype.interfaces import fsl
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from niworkflows.interfaces.masks import BETRPT
from niworkflows.interfaces.registration import FLIRTRPT
from niworkflows.data import get_mni_icbm152_nlin_asym_09c

from fmriprep.interfaces import (DerivativesDataSink, FormatHMCParam,
                                 ImageDataSink)
from fmriprep.workflows.fieldmap import sdc_unwarp
from fmriprep.viz import stripped_brain_overlay
from fmriprep.workflows.sbref import _extract_wm


# pylint: disable=R0914
def epi_hmc(name='EPI_HMC', settings=None):
    """
    Performs :abbr:`HMC (head motion correction)` over the input
    :abbr:`EPI (echo-planar imaging)` image.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['epi']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['epi_brain', 'xforms', 'epi_mask', 'epi_mean', 'movpar_file']), name='outputnode')

    pre_bet_mean = pe.Node(fsl.MeanImage(dimension='T'), name='PreBETMean')

    bet = pe.Node(BETRPT(functional=True, frac=0.6), name='EPI_bet')

    # Head motion correction (hmc)
    hmc = pe.Node(fsl.MCFLIRT(
        save_mats=True, save_plots=True, mean_vol=True), name='EPI_hmc')

    pick_1st = pe.Node(fsl.ExtractROI(t_min=0, t_size=1), name='EPIPickFirst')
    hcm2itk = pe.MapNode(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                         iterfield=['transform_file'], name='hcm2itk')

    avscale = pe.MapNode(fsl.utils.AvScale(all_param=True), name='AvScale',
                         iterfield=['mat_file'])
    avs_format = pe.Node(FormatHMCParam(), name='AVScale_Format')

    # Calculate EPI mask on the average after HMC
    bet_hmc = pe.Node(BETRPT(generate_report=True, mask=True, frac=0.6),
                      name='EPI_hmc_bet')

    workflow.connect([
        (inputnode, pick_1st, [('epi', 'in_file')]),
        (inputnode, bet, [('epi', 'in_file')]),
        (inputnode, pre_bet_mean, [('epi', 'in_file')]),
        (bet, hmc, [('out_file', 'in_file')]),
        (hmc, hcm2itk, [('mat_file', 'transform_file')]),
        (pick_1st, hcm2itk, [('roi_file', 'source_file'),
                             ('roi_file', 'reference_file')]),
        (hcm2itk, outputnode, [('itk_transform', 'xforms')]),
        (hmc, outputnode, [('out_file', 'epi_brain'),
                           ('par_file', 'movpar_file')]),
        (hmc, avscale, [('mat_file', 'mat_file')]),
        (avscale, avs_format, [('translations', 'translations'),
                               ('rot_angles', 'rot_angles')]),
        (hmc, bet_hmc, [('mean_img', 'in_file')]),
        (hmc, avscale, [('mean_img', 'ref_file')]),
        (bet_hmc, outputnode, [('mask_file', 'epi_mask'),
                               ('out_file', 'epi_mean')]),
    ])

    # Write corrected file in the designated output dir
    ds_hmc = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='hmc'),
        name='DerivativesHMC'
    )

    ds_mats = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='hmc'),
        name='DerivativesHMCmats'
    )

    ds_mask = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='hmc_bmask'),
        name='DerivativesEPImask'
    )

    ds_motion = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='hmc'),
        name='DerivativesParamsHMC'
    )

    ds_betrpt = pe.Node(nio.DataSink(), name="BETRPTDS")
    ds_betrpt.inputs.base_directory = op.join(settings['output_dir'],
                                              'reports')

    mean_epi_stripped_overlay = pe.Node(
        niu.Function(
            input_names=['in_file', 'overlay_file', 'out_file'],
            output_names=['out_file'],
            function=stripped_brain_overlay
        ),
        name='MeanEPIStrippedOverlay'
    )
    #  name needs to be based on source_file
    mean_epi_stripped_overlay.inputs.out_file = 'mean_epi_overlay.svg'

    mean_epi_overlay_ds = pe.Node(
        ImageDataSink(base_directory=settings['output_dir']),
        name="MeanEPIOverlayDS",
    )

    workflow.connect([
        (inputnode, ds_hmc, [('epi', 'source_file')]),
        (inputnode, ds_mats, [('epi', 'source_file')]),
        (inputnode, ds_mask, [('epi', 'source_file')]),
        (inputnode, ds_motion, [('epi', 'source_file')]),
        (inputnode, mean_epi_overlay_ds, [('epi', 'origin_file')]),
        (hmc, ds_hmc, [('out_file', 'in_file')]),
        (hcm2itk, ds_mats, [('itk_transform', 'in_file')]),
        (bet_hmc, ds_mask, [('mask_file', 'in_file')]),
        (avs_format, ds_motion, [('out_file', 'in_file')]),
        (pre_bet_mean, mean_epi_stripped_overlay, [('out_file', 'overlay_file')]),
        (bet_hmc, mean_epi_stripped_overlay, [('mask_file', 'in_file')]),
        (pre_bet_mean, mean_epi_overlay_ds, [('out_file', 'overlay_file')]),
        (bet_hmc, mean_epi_overlay_ds, [('mask_file', 'base_file')]),
        (mean_epi_stripped_overlay, mean_epi_overlay_ds,
         [('out_file', 'in_file')]),
        (bet, ds_betrpt, [('out_report', '@betrpt')])
    ])

    return workflow


def epi_mean_t1_registration(name='EPIMeanNormalization', settings=None):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the T1-space
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 'epi_mean', 't1_brain',
                                      't1_seg']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['mat_epi_to_t1', 'mat_t1_to_epi']),
        name='outputnode'
    )

    # Extract wm mask from segmentation
    wm_mask = pe.Node(
        niu.Function(input_names=['in_file'], output_names=['out_file'],
                     function=_extract_wm),
        name='WM_mask'
    )

    flt_bbr_init = pe.Node(
        FLIRTRPT(generate_report=True, dof=6, out_matrix_file='init.mat',
                 out_file='init.nii.gz'),
        name='Flirt_BBR_init'
    )
    flt_bbr = pe.Node(
        FLIRTRPT(generate_report=True, dof=6, cost_func='bbr',
                 out_file='bbr.nii.gz'),
        name='Flirt_BBR'
    )
    flt_bbr.inputs.schedule = op.join(os.getenv('FSLDIR'),
                                      'etc/flirtsch/bbr.sch')

    # make equivalent warp fields
    invt_bbr = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='Flirt_BBR_Inv')

    #  EPI to T1 transform matrix is from fsl, using c3 tools to convert to
    #  something ANTs will like.
    fsl2itk_fwd = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_fwd')
    fsl2itk_inv = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_inv')

    # Write EPI mean in T1w space
    ds_t1w = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='hmc_t1'),
        name='DerivHMC_T1w'
    )
    # Write registrated file in the designated output dir
    ds_tfm_fwd = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='epi2t1w_affine'),
        name='DerivEPI_to_T1w_fwd'
    )
    ds_tfm_inv = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='t1w2epi_affine'),
        name='DerivEPI_to_T1w_inv'
    )

    ds_flt_bbr = pe.Node(nio.DataSink(), name="FLTBBRDS")
    ds_flt_bbr.inputs.base_directory = op.join(settings['output_dir'],
                                                'reports')

    workflow.connect([
        (inputnode, wm_mask, [('t1_seg', 'in_file')]),
        (inputnode, flt_bbr_init, [('t1_brain', 'reference')]),
        (inputnode, fsl2itk_fwd, [('t1_brain', 'reference_file'),
                                  ('epi_mean', 'source_file')]),
        (inputnode, fsl2itk_inv, [('epi_mean', 'reference_file'),
                                  ('t1_brain', 'source_file')]),
        (inputnode, flt_bbr_init, [('epi_mean', 'in_file')]),
        (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (inputnode, flt_bbr, [('t1_brain', 'reference')]),
        (inputnode, flt_bbr, [('epi_mean', 'in_file')]),
        (wm_mask, flt_bbr, [('out_file', 'wm_seg')]),
        (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
        (invt_bbr, outputnode, [('out_file', 'mat_t1_to_epi')]),
        (flt_bbr, fsl2itk_fwd, [('out_matrix_file', 'transform_file')]),
        (invt_bbr, fsl2itk_inv, [('out_file', 'transform_file')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_epi_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_epi')]),
        (inputnode, ds_tfm_fwd, [('epi', 'source_file')]),
        (inputnode, ds_tfm_inv, [('epi', 'source_file')]),
        (inputnode, ds_t1w, [('epi', 'source_file')]),
        (fsl2itk_fwd, ds_tfm_fwd, [('itk_transform', 'in_file')]),
        (fsl2itk_inv, ds_tfm_inv, [('itk_transform', 'in_file')]),
        (flt_bbr, ds_t1w, [('out_file', 'in_file')]),
        (flt_bbr, ds_flt_bbr, [('out_report', 'flt_bbr_rpt')])
    ])

    # Plots for report
    epi_to_t1 = pe.Node(
        niu.Function(
            input_names=['in_file', 'overlay_file', 'out_file'],
            output_names=['out_file'],
            function=stripped_brain_overlay
        ),
        name='PNG_epi_to_t1'
    )
    epi_to_t1.inputs.out_file = 'epi_to_t1.svg'

    # Write corrected file in the designated output dir
    epi_to_t1_ds = pe.Node(
        ImageDataSink(base_directory=settings['output_dir']),
        name="datasink",
    )

    workflow.connect([
        (flt_bbr, epi_to_t1, [('out_file', 'overlay_file')]),
        (inputnode, epi_to_t1, [('t1_seg', 'in_file')]),
        (flt_bbr, epi_to_t1_ds, [('out_file', 'overlay_file')]),
        (inputnode, epi_to_t1_ds, [('t1_seg', 'base_file')]),
        (epi_to_t1, epi_to_t1_ds, [('out_file', 'in_file')]),
        (inputnode, epi_to_t1_ds, [('epi', 'origin_file')])
    ])

    return workflow


def epi_sbref_registration(settings, name='EPI_SBrefRegistration'):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 'epi_brain', 'sbref_brain',
                                      'sbref_brain_mask']),
        name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['epi_registered', 'out_mat', 'out_mat_inv']), name='outputnode')

    mean = pe.Node(fsl.MeanImage(dimension='T'), name='EPImean')
    inu = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='EPImeanBias')
    epi_sbref = pe.Node(FLIRTRPT(generate_report=True, dof=6,
                                 out_matrix_file='init.mat',
                                 out_file='init.nii.gz'),
                        name='EPI2SBRefRegistration')
    # make equivalent inv
    sbref_epi = pe.Node(fsl.ConvertXFM(invert_xfm=True), name="SBRefEPI")

    epi_split = pe.Node(fsl.Split(dimension='t'), name='EPIsplit')
    epi_xfm = pe.MapNode(fsl.preprocess.ApplyXFM(), name='EPIapplyXFM',
                         iterfield=['in_file'])
    epi_merge = pe.Node(fsl.Merge(dimension='t'), name='EPImergeback')

    ds_sbref = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='hmc_sbref'), name='DerivHMC_SBRef')

    ds_flirtrpt = pe.Node(nio.DataSink(), name="FLIRTRPTDS")
    ds_flirtrpt.inputs.base_directory = op.join(settings['output_dir'],
                                                'reports')

    workflow.connect([
        (inputnode, epi_split, [('epi_brain', 'in_file')]),
        (inputnode, epi_sbref, [('sbref_brain', 'reference')]),
        (inputnode, epi_xfm, [('sbref_brain', 'reference')]),
        (inputnode, mean, [('epi_brain', 'in_file')]),
        (mean, inu, [('out_file', 'input_image')]),
        (inu, epi_sbref, [('output_image', 'in_file')]),

        (epi_split, epi_xfm, [('out_files', 'in_file')]),
        (epi_sbref, epi_xfm, [('out_matrix_file', 'in_matrix_file')]),
        (epi_xfm, epi_merge, [('out_file', 'in_files')]),
        (epi_sbref, outputnode, [('out_matrix_file', 'out_mat')]),
        (epi_merge, outputnode, [('merged_file', 'epi_registered')]),

        (epi_sbref, sbref_epi, [('out_matrix_file', 'in_file')]),
        (sbref_epi, outputnode, [('out_file', 'out_mat_inv')]),

        (epi_merge, ds_sbref, [('merged_file', 'in_file')]),
        (inputnode, ds_sbref, [('epi', 'source_file')]),
        (epi_sbref, ds_flirtrpt, [('out_report', '@flirtrpt')])
    ])

    #  Plot for report
    post_merge_mean = pe.Node(fsl.MeanImage(), name='PostEPIMean')
    mean_epi_to_sbref_overlay = pe.Node(
        niu.Function(
            input_names=["in_file", "overlay_file", "out_file"],
            output_names=["out_file"],
            function=stripped_brain_overlay
        ),
        name="MeanEPIToSbref"
    )
    mean_epi_to_sbref_overlay.inputs.out_file = "mean_epi_to_sbref.svg"

    mean_epi_to_sbref_ds = pe.Node(
        ImageDataSink(base_directory=settings['output_dir']),
        name="datasink",
    )

    workflow.connect([
        (epi_merge, post_merge_mean, [('merged_file', 'in_file')]),
        (inputnode, mean_epi_to_sbref_overlay, [('sbref_brain_mask', 'in_file')]),
        (post_merge_mean, mean_epi_to_sbref_overlay, [('out_file', 'overlay_file')]),
        (inputnode, mean_epi_to_sbref_ds, [('sbref_brain_mask', 'base_file')]),
        (post_merge_mean, mean_epi_to_sbref_ds, [('out_file', 'overlay_file')]),
        (mean_epi_to_sbref_overlay, mean_epi_to_sbref_ds, [('out_file', 'in_file')]),
        (inputnode, mean_epi_to_sbref_ds, [('epi', 'origin_file')])
    ])

    return workflow


def epi_mni_transformation(name='EPIMNITransformation', settings=None):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'mat_epi_to_t1',
            't1_2_mni_forward_transform',
            'epi',
            'epi_mask',
            't1',
            'hmc_xforms'
        ]),
        name='inputnode'
    )

    def _aslist(in_value):
        if isinstance(in_value, list):
            return in_value
        return [in_value]

    pick_1st = pe.Node(fsl.ExtractROI(t_min=0, t_size=1), name='EPIPickFirst')

    gen_ref = pe.Node(niu.Function(
        input_names=['fixed_image', 'moving_image'], output_names=['out_file'],
        function=_gen_reference), name='GenNewMNIReference')
    gen_ref.inputs.fixed_image = op.join(get_mni_icbm152_nlin_asym_09c(),
                                         '1mm_T1.nii.gz')

    split = pe.Node(fsl.Split(dimension='t'), name='SplitEPI')
    merge_transforms = pe.MapNode(niu.Merge(3),
                                  iterfield=['in3'], name='MergeTransforms')
    epi_to_mni_transform = pe.MapNode(
        ants.ApplyTransforms(), iterfield=['input_image', 'transforms'],
        name='EPIToMNITransform')
    epi_to_mni_transform.terminal_output = 'file'
    merge = pe.Node(fsl.Merge(dimension='t'), name='MergeEPI')

    mask_merge_tfms = pe.Node(niu.Merge(2), name='MaskMergeTfms')
    mask_mni_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='NearestNeighbor'),
        name='MaskToMNI'
    )

    # Write corrected file in the designated output dir
    ds_mni = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='hmc_mni'),
        name='DerivativesHMCMNI'
    )
    ds_mni_mask = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='hmc_mni_bmask'),
        name='DerivativesHMCMNImask'
    )

    workflow.connect([
        (inputnode, pick_1st, [('epi', 'in_file')]),
        (inputnode, ds_mni, [('epi', 'source_file')]),
        (inputnode, ds_mni_mask, [('epi', 'source_file')]),
        (pick_1st, gen_ref, [('roi_file', 'moving_image')]),
        (inputnode, merge_transforms, [('t1_2_mni_forward_transform', 'in1'),
                                       (('mat_epi_to_t1', _aslist), 'in2'),
                                       ('hmc_xforms', 'in3')]),
        (inputnode, mask_merge_tfms, [('t1_2_mni_forward_transform', 'in1'),
                                      (('mat_epi_to_t1', _aslist), 'in2')]),
        (inputnode, split, [('epi', 'in_file')]),
        (split, epi_to_mni_transform, [('out_files', 'input_image')]),
        (merge_transforms, epi_to_mni_transform, [('out', 'transforms')]),
        (gen_ref, epi_to_mni_transform, [('out_file', 'reference_image')]),
        (epi_to_mni_transform, merge, [('output_image', 'in_files')]),
        (merge, ds_mni, [('merged_file', 'in_file')]),
        (mask_merge_tfms, mask_mni_tfm, [('out', 'transforms')]),
        (gen_ref, mask_mni_tfm, [('out_file', 'reference_image')]),
        (inputnode, mask_mni_tfm, [('epi_mask', 'input_image')]),
        (mask_mni_tfm, ds_mni_mask, [('output_image', 'in_file')])
    ])

    return workflow


# pylint: disable=R0914
def epi_unwarp(name='EPIUnwarpWorkflow', settings=None):
    """ A workflow to correct EPI images """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 'fmap', 'fmap_ref', 'fmap_mask',
                                      't1_seg']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['epi_unwarp', 'epi_mean', 'epi_mask']),
        name='outputnode'
    )

    unwarp = sdc_unwarp()
    unwarp.inputs.inputnode.hmc_movpar = ''

    # Compute outputs
    mean = pe.Node(fsl.MeanImage(dimension='T'), name='EPImean')
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                  name='EPIBET')

    ds_epi_unwarp = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='epi_unwarp'),
        name='DerivUnwarp_EPUnwarp_EPI'
    )

    workflow.connect([
        (inputnode, unwarp, [('fmap', 'inputnode.fmap'),
                             ('fmap_ref', 'inputnode.fmap_ref'),
                             ('fmap_mask', 'inputnode.fmap_mask'),
                             ('epi', 'inputnode.in_file')]),
        (inputnode, ds_epi_unwarp, [('epi', 'source_file')]),
        (unwarp, mean, [('outputnode.out_file', 'in_file')]),
        (mean, bet, [('out_file', 'in_file')]),
        (bet, outputnode, [('out_file', 'epi_mean'),
                           ('mask_file', 'epi_mask')]),
        (unwarp, outputnode, [('outputnode.out_file', 'epi_unwarp')]),
        (unwarp, ds_epi_unwarp, [('outputnode.out_file', 'in_file')])
    ])

    # Plot result
    epi_corr = pe.Node(
        niu.Function(
            input_names=['in_file', 'overlay_file', 'out_file'],
            output_names=['out_file'],
            function=stripped_brain_overlay
        ),
        name='EPICorr'
    )
    epi_corr.inputs.out_file = 'corrected_EPI.svg'

    epi_corr_ds = pe.Node(
        ImageDataSink(base_directory=settings['output_dir']),
        name="datasink",
    )

    workflow.connect([
        (bet, epi_corr, [('out_file', 'overlay_file')]),
        (inputnode, epi_corr, [('fmap_mask', 'in_file')]),
        (bet, epi_corr_ds, [('out_file', 'overlay_file')]),
        (inputnode, epi_corr_ds, [('fmap_mask', 'base_file')]),
        (epi_corr, epi_corr_ds, [('out_file', 'in_file')]),
        (inputnode, epi_corr_ds, [('epi', 'origin_file')])
    ])

    return workflow


def _gen_reference(fixed_image, moving_image, out_file=None):
    import os.path as op
    import numpy as np
    import nibabel as nb
    from nibabel.affines import apply_affine

    if out_file is None:
        fname, ext = op.splitext(op.basename(fixed_image))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('%s_wm%s' % (fname, ext))

    imref = nb.load(fixed_image)
    immov = nb.load(moving_image)

    orig = apply_affine(imref.affine, [-0.5] * 3)
    end = apply_affine(imref.affine,
                       [s - 0.5 for s in imref.get_data().shape[:3]])

    mov_spacing = immov.get_header().get_zooms()[:3]
    new_sizes = np.ceil((end-orig)/mov_spacing)

    new_affine = immov.affine
    ref_center = apply_affine(imref.affine, (0.5 * (np.array(
        imref.get_data().shape[:3]))))

    new_center = new_affine[:3, :3].dot(new_sizes)
    new_affine[:3, 3] = -0.5 * new_center + ref_center

    new_ref_im = nb.Nifti1Image(np.zeros(tuple(new_sizes.astype(int))),
                                new_affine, immov.get_header())
    new_ref_im.to_filename(out_file)

    return out_file
