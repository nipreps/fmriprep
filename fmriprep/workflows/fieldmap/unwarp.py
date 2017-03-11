#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import FUGUE
from nipype.interfaces.ants import Registration, N4BiasFieldCorrection
from nipype.interfaces.ants.preprocess import Matrix2FSLParams
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT
from fmriprep.interfaces.images import FixAffine
from fmriprep.interfaces.fmap import WarpReference, ApplyFieldmap
from fmriprep.interfaces import itk
from fmriprep.interfaces.hmc import MotionCorrection
from fmriprep.interfaces.utils import MeanTimeseries, ApplyMask

def sdc_unwarp(name='SDC_unwarp', settings=None):
    """
    This workflow takes an estimated fieldmap and a target image and applies TOPUP,
    an :abbr:`SDC (susceptibility-derived distortion correction)` method in FSL to
    unwarp the target image.

    Input fields:
    ~~~~~~~~~~~~~

      inputnode.in_files - a list of target 3D images to which this correction
                           will be applied
      inputnode.in_reference - a 3D image that is the reference w.r.t. which the
                               motion parameters were computed. It is desiderable
                               for this image to have undergone a bias correction
                               processing.
      inputnode.in_mask - a brain mask corresponding to the in_reference image
      inputnode.in_meta - metadata associated to in_files
      inputnode.in_hmcpar - the head motion parameters as written by antsMotionCorr
      inputnode.fmap - a fieldmap in Hz
      inputnode.fmap_ref - the fieldmap reference (generally, a *magnitude* image or the
                           resulting SE image)
      inputnode.fmap_mask - a brain mask in fieldmap-space

    Output fields:
    ~~~~~~~~~~~~~~

      outputnode.out_files - the in_file after susceptibility-distortion correction.

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_files', 'in_reference', 'in_hmcpar', 'in_mask', 'in_meta',
                'fmap_ref', 'fmap_mask', 'fmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_files', 'out_mean', 'out_hmcpar']),
                         name='outputnode')

    # Be robust if no reference image is passed
    target_sel = pe.Node(niu.Function(
        input_names=['in_files', 'in_reference'], output_names=['out_file'],
        function=_get_reference), name='target_select')

    # Remove the scanner affine transform from all images
    # (messes up with ANTs and are useless unless we feed them back
    # in a neuronavigator)
    ref_hdr = pe.Node(FixAffine(), name='reference_hdr')
    target_hdr = pe.Node(FixAffine(), name='target_hdr')
    fmap_hdr = pe.Node(FixAffine(), name='fmap_hdr')
    inputs_hdr = pe.MapNode(
        FixAffine(), iterfield=['in_file'], name='inputs_hdr')

    # Prepare fieldmap reference image, creating a fake warping
    # to make the magnitude look like a distorted EPI
    ref_wrp = pe.Node(WarpReference(), name='reference_warped')
    # Mask reference image (the warped magnitude image)
    ref_msk = pe.Node(ApplyMask(), name='reference_mask')

    # Prepare target image for registration
    inu = pe.Node(N4BiasFieldCorrection(dimension=3), name='target_inu')

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json')
    if settings.get('debug', False):
        ants_settings = pkgr.resource_filename(
            'fmriprep', 'data/fmap-any_registration_testing.json')
    fmap2ref = pe.Node(Registration(
        from_file=ants_settings, output_warped_image=True,
        output_inverse_warped_image=True),
                       name='fmap_ref2target_avg')


    # Fieldmap to rads and then to voxels (VSM - voxel shift map)
    torads = pe.Node(niu.Function(input_names=['in_file'], output_names=['out_file'],
                                  function=_hz2rads), name='fmap_Hz2rads')
    gen_vsm = pe.Node(FUGUE(save_unmasked_shift=True), name='VSM')

    # Map the VSM into the EPI space
    applyxfm = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='BSpline', float=True),
                       name='fmap2target_avg')

    # Convert the VSM into a DFM (displacements field map)
    # or: FUGUE shift to ANTS warping.
    vsm2dfm = pe.Node(itk.FUGUEvsm2ANTSwarp(), name='fmap2dfm')

    # Calculate refined motion parameters after unwarping:
    # 1. Split initial HMC parameters from inputs
    split_hmc = pe.Node(itk.SplitITKTransform(), name='split_tfms')
    # 2. Append the HMC parameters to the fieldmap
    pre_tfms = pe.MapNode(itk.MergeANTsTransforms(in_file_invert=True),
                          iterfield=['in_file'], name='fmap2inputs_tfms')
    # 3. Map the DFM to the target EPI space
    xfmmap = pe.MapNode(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='BSpline', float=True),
                        iterfield=['transforms', 'invert_transform_flags'],
                        name='fmap2inputs_apply')
    # 4. Unwarp the mean EPI target to use it as reference in the next HMC
    unwarp = pe.Node(ANTSApplyTransformsRPT(
        dimension=3, interpolation='BSpline', invert_transform_flags=False, float=True,
        generate_report=True), name='target_ref_unwarped')
    # 5. Unwarp all volumes
    fugue_all = pe.MapNode(ApplyFieldmap(generate_report=False),
                           iterfield=['in_file', 'in_vsm'],
                           name='fmap2inputs_unwarp')
    # 6. Run HMC again on the corrected images, aiming at higher accuracy
    hmc2 = pe.Node(MotionCorrection(), name='fmap2inputs_hmc')
    hmc2_split = pe.Node(itk.SplitITKTransform(), name='fmap2inputs_hmc_split_tfms')
    hmc2_plots = pe.Node(Matrix2FSLParams(), name='hmc_motion_parameters')

    # Final correction with refined HMC parameters
    tfm_concat = pe.MapNode(itk.MergeANTsTransforms(
        in_file_invert=False, invert_transform_flags=[False]),
                            iterfield=['in_file'], name='inputs_xfms')
    unwarpall = pe.MapNode(ANTSApplyTransformsRPT(
        dimension=3, generate_report=False, float=True, interpolation='LanczosWindowedSinc'),
                           iterfield=['input_image', 'transforms', 'invert_transform_flags'],
                           name='inputs_unwarped')
    mean = pe.Node(MeanTimeseries(), name='inputs_unwarped_mean')

    workflow.connect([
        (inputnode, split_hmc, [('in_hmcpar', 'in_file')]),
        (inputnode, target_sel, [('in_files', 'in_files'),
                                 ('in_reference', 'in_reference')]),
        (inputnode, inputs_hdr, [('in_files', 'in_file')]),
        (target_sel, target_hdr, [('out_file', 'in_file')]),
        (inputnode, fmap_hdr, [('fmap', 'in_file')]),
        (fmap_hdr, torads, [('out_file', 'in_file')]),
        (inputnode, ref_hdr, [('fmap_ref', 'in_file')]),
        (target_hdr, applyxfm, [('out_file', 'reference_image')]),
        (inputnode, ref_wrp, [('fmap_mask', 'in_mask'),
                              (('in_meta', _get_ec), 'echospacing'),
                              (('in_meta', _get_pedir), 'pe_dir')]),
        (inputnode, gen_vsm, [(('in_meta', _get_ec), 'dwell_time'),
                              (('in_meta', _get_pedir), 'unwarp_direction')]),
        (inputnode, vsm2dfm, [(('in_meta', _get_pedir), 'pe_dir')]),
        (ref_hdr, ref_wrp, [('out_file', 'fmap_ref')]),
        (torads, ref_wrp, [('out_file', 'in_file')]),
        (target_hdr, inu, [('out_file', 'input_image')]),
        (inu, fmap2ref, [('output_image', 'moving_image')]),
        (torads, gen_vsm, [('out_file', 'fmap_in_file')]),
        (ref_wrp, ref_msk, [('out_warped', 'in_file'),
                            ('out_mask', 'in_mask')]),
        (ref_msk, fmap2ref, [('out_file', 'fixed_image')]),
        (gen_vsm, applyxfm, [('shift_out_file', 'input_image')]),
        (fmap2ref, applyxfm, [
            ('reverse_transforms', 'transforms'),
            ('reverse_invert_flags', 'invert_transform_flags')]),
        (applyxfm, vsm2dfm, [('output_image', 'in_file')]),
        (vsm2dfm, unwarp, [('out_file', 'transforms')]),
        (target_hdr, unwarp, [('out_file', 'reference_image'),
                              ('out_file', 'input_image')]),
        # Run HMC again, aiming at higher accuracy
        (split_hmc, pre_tfms, [('out_files', 'in_file')]),
        (fmap2ref, pre_tfms, [
            ('reverse_transforms', 'transforms'),
            ('reverse_invert_flags', 'invert_transform_flags')]),
        (gen_vsm, xfmmap, [('shift_out_file', 'input_image')]),
        (target_hdr, xfmmap, [('out_file', 'reference_image')]),
        (pre_tfms, xfmmap, [
            ('transforms', 'transforms'),
            ('invert_transform_flags', 'invert_transform_flags')]),
        (inputnode, fugue_all, [(('in_meta', _get_pedir), 'pe_dir')]),
        (inputs_hdr, fugue_all, [('out_file', 'in_file')]),
        (xfmmap, fugue_all, [('output_image', 'in_vsm')]),
        (fugue_all, hmc2, [('out_corrected', 'in_files')]),
        (unwarp, hmc2, [('output_image', 'reference_image')]),

        (hmc2, hmc2_split, [('out_tfm', 'in_file')]),
        (hmc2, hmc2_plots, [('out_movpar', 'matrix')]),
        (hmc2_split, tfm_concat, [('out_files', 'in_file')]),
        (vsm2dfm, tfm_concat, [('out_file', 'transforms')]),
        (tfm_concat, unwarpall, [
            ('transforms', 'transforms'),
            ('invert_transform_flags', 'invert_transform_flags')]),
        (inputs_hdr, unwarpall, [('out_file', 'input_image')]),
        (target_hdr, unwarpall, [('out_file', 'reference_image')]),
        (unwarpall, mean, [('output_image', 'in_files')]),
        (mean, outputnode, [('out_file', 'out_mean')]),
        (unwarpall, outputnode, [('output_image', 'out_files')]),
        (hmc2_plots, outputnode, [('parameters', 'out_hmcpar')])
    ])
    return workflow

# Helper functions
# ------------------------------------------------------------

def _get_reference(in_files, in_reference):
    if len(in_files) == 1:
        return in_files[0]
    return in_reference

def _get_ec(in_dict):
    return float(in_dict['EffectiveEchoSpacing'])

def _get_pedir(in_dict):
    return in_dict['PhaseEncodingDirection'].replace('j', 'y').replace('i', 'x')

def _hz2rads(in_file, out_file=None):
    """Transform a fieldmap in Hz into rad/s"""
    from math import pi
    import nibabel as nb
    from fmriprep.utils.misc import genfname
    if out_file is None:
        out_file = genfname(in_file, 'rads')
    nii = nb.load(in_file)
    data = nii.get_data() * 2.0 * pi
    nb.Nifti1Image(data, nii.get_affine(),
                   nii.get_header()).to_filename(out_file)
    return out_file


# Disable ApplyTOPUP workflow
# ------------------------------------------------------------
# from fmriprep.interfaces.topup import ConformTopupInputs
# encfile = pe.Node(interface=niu.Function(
#     input_names=['input_images', 'in_dict'], output_names=['unwarp_param', 'warp_param'],
#     function=create_encoding_file), name='TopUp_encfile', updatehash=True)
# gen_movpar = pe.Node(GenerateMovParams(), name='GenerateMovPar')
# topup_adapt = pe.Node(FieldCoefficients(), name='TopUpCoefficients')
# # Use the least-squares method to correct the dropout of the input images
# unwarp = pe.Node(fsl.ApplyTOPUP(method=method, in_index=[1]), name='TopUpApply')
# workflow.connect([
#     (inputnode, encfile, [('in_file', 'input_images')]),
#     (meta, encfile, [('out_dict', 'in_dict')]),
#     (conform, gen_movpar, [('out_file', 'in_file'),
#                            ('out_movpar', 'in_mats')]),
#     (conform, topup_adapt, [('out_brain', 'in_ref')]),
#     #                       ('out_movpar', 'in_hmcpar')]),
#     (gen_movpar, topup_adapt, [('out_movpar', 'in_hmcpar')]),
#     (applyxfm, topup_adapt, [('output_image', 'in_file')]),
#     (conform, unwarp, [('out_file', 'in_files')]),
#     (topup_adapt, unwarp, [('out_fieldcoef', 'in_topup_fieldcoef'),
#                            ('out_movpar', 'in_topup_movpar')]),
#     (encfile, unwarp, [('unwarp_param', 'encoding_file')]),
#     (unwarp, outputnode, [('out_corrected', 'out_file')])
# ])
