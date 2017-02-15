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
from nipype.interfaces import utility as niu
from niworkflows.interfaces.masks import ComputeEPIMask, BETRPT
from niworkflows.interfaces.registration import FLIRTRPT
from niworkflows.data import get_mni_icbm152_nlin_asym_09c

from fmriprep.interfaces import DerivativesDataSink, FormatHMCParam
from fmriprep.interfaces.utils import nii_concat
from fmriprep.utils.misc import fix_multi_T1w_source_name, _first
from fmriprep.workflows.fieldmap import sdc_unwarp
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
        fields=['xforms', 'epi_hmc', 'epi_mask', 'epi_mean', 'movpar_file',
                'motion_confounds_file']), name='outputnode')

    # Head motion correction (hmc)
    hmc = pe.Node(fsl.MCFLIRT(
        save_mats=True, save_plots=True, mean_vol=True), name='EPI_hmc')
    hmc.interface.estimated_memory_gb = settings["biggest_epi_file_size_gb"] * 3

    hcm2itk = pe.MapNode(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                         iterfield=['transform_file'], name='hcm2itk')

    avscale = pe.MapNode(fsl.utils.AvScale(all_param=True), name='AvScale',
                         iterfield=['mat_file'])
    avs_format = pe.Node(FormatHMCParam(), name='AVScale_Format')

    inu = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='EPImeanBias')

    # Calculate EPI mask on the average after HMC
    skullstrip_epi = pe.Node(ComputeEPIMask(generate_report=True, dilation=1),
                             name='skullstrip_epi')

    workflow.connect([
        (inputnode, hmc, [('epi', 'in_file')]),
        (hmc, hcm2itk, [('mat_file', 'transform_file'),
                        ('mean_img', 'source_file'),
                        ('mean_img', 'reference_file')]),
        (hcm2itk, outputnode, [('itk_transform', 'xforms')]),
        (hmc, outputnode, [('out_file', 'epi_hmc'),
                           ('par_file', 'movpar_file'),
                           ('mean_img', 'epi_mean')]),
        (hmc, avscale, [('mat_file', 'mat_file')]),
        (avscale, avs_format, [('translations', 'translations'),
                               ('rot_angles', 'rot_angles')]),
        (hmc, inu, [('mean_img', 'input_image')]),
        (inu, skullstrip_epi, [('output_image', 'in_file')]),
        (hmc, avscale, [('mean_img', 'ref_file')]),
        (avs_format, outputnode, [('out_file', 'motion_confounds_file')]),
        (skullstrip_epi, outputnode, [('mask_file', 'epi_mask')]),
    ])

    ds_mask = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='brainmask'),
        name='DerivativesEPImask'
    )

    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='bet_hmc', out_path_base='reports'),
        name='DS_Report'
    )

    workflow.connect([
        (inputnode, ds_report, [('epi', 'source_file')]),
        (inputnode, ds_mask, [('epi', 'source_file')]),
        (skullstrip_epi, ds_mask, [('mask_file', 'in_file')]),
        (skullstrip_epi, ds_report, [('out_report', 'in_file')])
    ])

    if not settings["skip_native"]:
        # Write corrected file in the designated output dir
        ds_hmc = pe.Node(
            DerivativesDataSink(base_directory=settings['output_dir'],
                                suffix='preproc'),
            name='DerivativesHMC'
        )
        workflow.connect([
            (inputnode, ds_hmc, [('epi', 'source_file')]),
            (hmc, ds_hmc, [('out_file', 'in_file')])
        ])

    return workflow


def ref_epi_t1_registration(reportlet_suffix, inv_ds_suffix, name='ref_epi_t1_registration',
                            settings=None):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the T1-space
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['name_source', 'ref_epi', 'ref_epi_mask',
                                      'bias_corrected_t1', 't1_brain', 't1_mask',
                                      't1_seg', 't1w']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['mat_epi_to_t1', 'mat_t1_to_epi',
                                      'itk_epi_to_t1', 'itk_t1_to_epi']),
        name='outputnode'
    )

    # Extract wm mask from segmentation
    wm_mask = pe.Node(
        niu.Function(input_names=['in_file'], output_names=['out_file'],
                     function=_extract_wm),
        name='WM_mask'
    )

    explicit_mask_epi = pe.Node(fsl.ApplyMask(), name="explicit_mask_epi")

    flt_bbr_init = pe.Node(
        FLIRTRPT(generate_report=True, dof=6),
        name='flt_bbr_init'
    )
    flt_bbr = pe.Node(
        FLIRTRPT(generate_report=True, dof=6, cost_func='bbr'),
        name='flt_bbr'
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

    # Write registrated file in the designated output dir
    ds_tfm_fwd = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='target-T1w_affine'),
        name='DerivEPI_to_T1w_fwd'
    )
    ds_tfm_inv = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix=inv_ds_suffix),
        name='DerivEPI_to_T1w_inv'
    )

    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix=reportlet_suffix, out_path_base='reports'),
        name='ds_report'
    )

    workflow.connect([
        (inputnode, wm_mask, [('t1_seg', 'in_file')]),
        (inputnode, explicit_mask_epi, [('ref_epi', 'in_file'),
                                        ('ref_epi_mask', 'mask_file')
                                        ]),
        (explicit_mask_epi, flt_bbr_init, [('out_file', 'in_file')]),
        (inputnode, flt_bbr_init, [('t1_brain', 'reference')]),
        (inputnode, fsl2itk_fwd, [('bias_corrected_t1', 'reference_file'),
                                  ('ref_epi', 'source_file')]),
        (inputnode, fsl2itk_inv, [('ref_epi', 'reference_file'),
                                  ('bias_corrected_t1', 'source_file')]),
        (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (inputnode, flt_bbr, [('t1_brain', 'reference')]),
        (explicit_mask_epi, flt_bbr, [('out_file', 'in_file')]),
        (wm_mask, flt_bbr, [('out_file', 'wm_seg')]),
        (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
        (invt_bbr, outputnode, [('out_file', 'mat_t1_to_epi')]),
        (flt_bbr, outputnode, [('out_matrix_file', 'mat_epi_to_t1')]),
        (flt_bbr, fsl2itk_fwd, [('out_matrix_file', 'transform_file')]),
        (invt_bbr, fsl2itk_inv, [('out_file', 'transform_file')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_epi_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_epi')]),
        (inputnode, ds_tfm_fwd, [(('name_source', _first), 'source_file')]),
        (inputnode, ds_tfm_inv, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (fsl2itk_fwd, ds_tfm_fwd, [('itk_transform', 'in_file')]),
        (fsl2itk_inv, ds_tfm_inv, [('itk_transform', 'in_file')]),
        (flt_bbr, ds_report, [('out_report', 'in_file')]),
        (inputnode, ds_report, [(('name_source', _first), 'source_file')])
    ])

    return workflow


def epi_sbref_registration(settings, name='EPI_SBrefRegistration'):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 'epi_name_source', 'sbref',
                                      'epi_mean', 'epi_mask',
                                      'sbref_mask']),
        name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['epi_registered', 'out_mat', 'out_mat_inv']), name='outputnode')

    epi_sbref = pe.Node(FLIRTRPT(generate_report=True, dof=6,
                                 out_matrix_file='init.mat',
                                 out_file='init.nii.gz'),
                        name='EPI2SBRefRegistration')
    # make equivalent inv
    sbref_epi = pe.Node(fsl.ConvertXFM(invert_xfm=True), name="SBRefEPI")

    epi_split = pe.Node(fsl.Split(dimension='t'), name='EPIsplit')
    epi_xfm = pe.MapNode(fsl.preprocess.ApplyXFM(), name='EPIapplyXFM',
                         iterfield=['in_file'])
    epi_merge = pe.Node(niu.Function(input_names=["in_files"],
                                     output_names=["merged_file"],
                                     function=nii_concat), name='EPImergeback')

    ds_sbref = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='preproc'), name='DerivHMC_SBRef')

    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='epi_sbref', out_path_base='reports'),
        name="DS_Report")

    workflow.connect([
        (inputnode, epi_split, [('epi', 'in_file')]),
        (inputnode, epi_sbref, [('sbref', 'reference'),
                                ('sbref_mask', 'ref_weight')]),
        (inputnode, epi_xfm, [('sbref', 'reference')]),
        (inputnode, epi_sbref, [('epi_mean', 'in_file'),
                                ('epi_mask', 'in_weight')]),

        (epi_split, epi_xfm, [('out_files', 'in_file')]),
        (epi_sbref, epi_xfm, [('out_matrix_file', 'in_matrix_file')]),
        (epi_xfm, epi_merge, [('out_file', 'in_files')]),
        (epi_sbref, outputnode, [('out_matrix_file', 'out_mat')]),
        (epi_merge, outputnode, [('merged_file', 'epi_registered')]),

        (epi_sbref, sbref_epi, [('out_matrix_file', 'in_file')]),
        (sbref_epi, outputnode, [('out_file', 'out_mat_inv')]),

        (epi_merge, ds_sbref, [('merged_file', 'in_file')]),
        (inputnode, ds_sbref, [('epi_name_source', 'source_file')]),
        (inputnode, ds_report, [('epi_name_source', 'source_file')]),
        (epi_sbref, ds_report, [('out_report', 'in_file')])
    ])

    return workflow


def epi_mni_transformation(name='EPIMNITransformation', settings=None):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_epi_to_t1',
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

    gen_ref = pe.Node(niu.Function(
        input_names=['fixed_image', 'moving_image'], output_names=['out_file'],
        function=_gen_reference), name='GenNewMNIReference')
    gen_ref.inputs.fixed_image = op.join(get_mni_icbm152_nlin_asym_09c(),
                                         '1mm_T1.nii.gz')

    split = pe.Node(fsl.Split(dimension='t'), name='SplitEPI')
    split.interface.estimated_memory_gb = settings["biggest_epi_file_size_gb"] * 3

    merge_transforms = pe.MapNode(niu.Merge(3),
                                  iterfield=['in3'], name='MergeTransforms')
    epi_to_mni_transform = pe.MapNode(
        ants.ApplyTransforms(), iterfield=['input_image', 'transforms'],
        name='EPIToMNITransform')
    epi_to_mni_transform.terminal_output = 'file'
    merge = pe.Node(niu.Function(input_names=["in_files"],
                                 output_names=["merged_file"],
                                 function=nii_concat), name='MergeEPI')
    merge.interface.estimated_memory_gb = settings[
                                              "biggest_epi_file_size_gb"] * 3

    mask_merge_tfms = pe.Node(niu.Merge(2), name='MaskMergeTfms')
    mask_mni_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='NearestNeighbor'),
        name='MaskToMNI'
    )

    # Write corrected file in the designated output dir
    ds_mni = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='space-MNI152NLin2009cAsym_preproc'),
        name='DerivativesHMCMNI'
    )
    ds_mni_mask = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='space-MNI152NLin2009cAsym_brainmask'),
        name='DerivativesHMCMNImask'
    )

    workflow.connect([
        (inputnode, ds_mni, [('epi', 'source_file')]),
        (inputnode, ds_mni_mask, [('epi', 'source_file')]),
        (inputnode, gen_ref, [('epi_mask', 'moving_image')]),
        (inputnode, merge_transforms, [('t1_2_mni_forward_transform', 'in1'),
                                       (('itk_epi_to_t1', _aslist), 'in2'),
                                       ('hmc_xforms', 'in3')]),
        (inputnode, mask_merge_tfms, [('t1_2_mni_forward_transform', 'in1'),
                                      (('itk_epi_to_t1', _aslist), 'in2')]),
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

    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='epi_unwarp_bet', out_path_base='reports'),
        name="DS_Report")

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
        (unwarp, ds_epi_unwarp, [('outputnode.out_file', 'in_file')]),
        (inputnode, ds_report, [('epi', 'source_file')]),
        (bet, ds_report, [('out_report', 'in_file')])
    ])

    return workflow


def _gen_reference(fixed_image, moving_image, out_file=None):
    import os.path as op
    import numpy
    from nilearn.image import resample_img, load_img

    if out_file is None:
        fname, ext = op.splitext(op.basename(fixed_image))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('%s_wm%s' % (fname, ext))

    new_zooms = load_img(moving_image).header.get_zooms()

    new_ref_im = resample_img(fixed_image, target_affine=numpy.diag(new_zooms),
                              interpolation='nearest')

    new_ref_im.to_filename(out_file)

    return out_file
