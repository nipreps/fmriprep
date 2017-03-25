#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.base import Undefined
from nipype.interfaces.fsl import FUGUE
from nipype.interfaces.ants import Registration, N4BiasFieldCorrection, ApplyTransforms
# from nipype.interfaces.ants.preprocess import Matrix2FSLParams
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT
from fmriprep.interfaces.fmap import WarpReference, ApplyFieldmap
from fmriprep.interfaces.images import FixAffine, SplitMerge
from fmriprep.interfaces.nilearn import Merge
from fmriprep.interfaces import itk
from fmriprep.interfaces.hmc import MotionCorrection, itk2moco
from fmriprep.interfaces.utils import MeanTimeseries, ApplyMask

def sdc_unwarp(name='SDC_unwarp', settings=None):
    """
    This workflow takes an estimated fieldmap and a target image and applies TOPUP,
    an :abbr:`SDC (susceptibility-derived distortion correction)` method in FSL to
    unwarp the target image.

    .. workflow ::

        from fmriprep.workflows.fieldmap.unwarp import sdc_unwarp
        wf = sdc_unwarp()


    Input fields::

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

    Output fields::

      outputnode.out_files - the in_file after susceptibility-distortion correction.


    """

    if settings is None:
        settings = {'ants_nthreads': 6}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_split', 'in_reference', 'in_hmcpar', 'in_mask', 'in_meta',
                'fmap_ref', 'fmap_mask', 'fmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_files', 'out_mean', 'out_hmcpar', 'out_confounds', 'out_warps']), name='outputnode')

    ref_hdr = pe.Node(FixAffine(), name='fmap_ref_hdr')
    fmap_hdr = pe.Node(FixAffine(), name='fmap_hdr')

    # Be robust if no reference image is passed
    target_sel = pe.Node(niu.Function(
        input_names=['in_files', 'in_reference'], output_names=['out_file'],
        function=_get_reference), name='target_select')

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
        from_file=ants_settings, output_inverse_warped_image=True,
        output_warped_image=True, num_threads=settings['ants_nthreads']),
                       name='fmap_ref2target_avg')
    fmap2ref.interface.num_threads = settings['ants_nthreads']


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
    hmc2 = pe.Node(MotionCorrection(njobs=settings['ants_nthreads'],
                   cache_dir=settings.get('cache_dir', Undefined)), name='fmap2inputs_hmc')
    hmc2.interface.num_threads = settings['ants_nthreads']


    hmc2moco = pe.Node(niu.Function(input_names=['in_files'],
        output_names=['out_par', 'out_confounds'], function=itk2moco), name='tfm2moco')

    # Final correction with refined HMC parameters
    tfm_concat = pe.MapNode(itk.MergeANTsTransforms(
        in_file_invert=False, invert_transform_flags=[False]),
                            iterfield=['in_file'], name='inputs_xfms')
    unwarpall = pe.MapNode(ANTSApplyTransformsRPT(
        dimension=3, generate_report=False, float=True, interpolation='LanczosWindowedSinc'),
                           iterfield=['input_image', 'transforms', 'invert_transform_flags'],
                           name='inputs_unwarped')

    tfm_comb = pe.MapNode(ApplyTransforms(dimension=3, float=True,
                          print_out_composite_warp_file=True, output_image='epiwarp.nii.gz'),
                          iterfield=['transforms', 'invert_transform_flags'],
                          name='generate_warpings')

    mean = pe.Node(MeanTimeseries(), name='inputs_unwarped_mean')

    workflow.connect([
        (inputnode, pre_tfms, [('in_hmcpar', 'in_file')]),
        (inputnode, target_sel, [('in_split', 'in_files'),
                                 ('in_reference', 'in_reference')]),
        (inputnode, fmap_hdr, [('fmap', 'in_file')]),
        (inputnode, ref_hdr, [('fmap_ref', 'in_file')]),
        (fmap_hdr, torads, [('out_file', 'in_file')]),
        (target_sel, applyxfm, [('out_file', 'reference_image')]),
        (inputnode, ref_wrp, [('fmap_mask', 'in_mask'),
                              (('in_meta', _get_ec), 'echospacing'),
                              (('in_meta', _get_pedir), 'pe_dir')]),
        (inputnode, gen_vsm, [(('in_meta', _get_ec), 'dwell_time'),
                              (('in_meta', _get_pedir), 'unwarp_direction')]),
        (inputnode, vsm2dfm, [(('in_meta', _get_pedir), 'pe_dir')]),
        (inputnode, fugue_all, [(('in_meta', _get_pedir), 'pe_dir')]),
        (inputnode, fugue_all, [('in_split', 'in_file')]),
        (ref_hdr, ref_wrp, [('out_file', 'fmap_ref')]),
        (torads, ref_wrp, [('out_file', 'in_file')]),
        (target_sel, inu, [('out_file', 'input_image')]),
        (inu, fmap2ref, [('output_image', 'moving_image')]),
        (torads, gen_vsm, [('out_file', 'fmap_in_file')]),
        (ref_wrp, ref_msk, [('out_warped', 'in_file'),
                            ('out_mask', 'in_mask')]),
        (ref_msk, fmap2ref, [('out_file', 'fixed_image')]),
        (gen_vsm, applyxfm, [('shift_out_file', 'input_image')]),
        (fmap2ref, applyxfm, [
            ('inverse_composite_transform', 'transforms')]),
        (applyxfm, vsm2dfm, [('output_image', 'in_file')]),
        (vsm2dfm, unwarp, [('out_file', 'transforms')]),
        (target_sel, unwarp, [('out_file', 'reference_image'),
                              ('out_file', 'input_image')]),
        # Run HMC again, aiming at higher accuracy
        (fmap2ref, pre_tfms, [
            ('inverse_composite_transform', 'transforms')]),
        (gen_vsm, xfmmap, [('shift_out_file', 'input_image')]),
        (target_sel, xfmmap, [('out_file', 'reference_image')]),
        (pre_tfms, xfmmap, [
            ('transforms', 'transforms'),
            ('invert_transform_flags', 'invert_transform_flags')]),
        (xfmmap, fugue_all, [('output_image', 'in_vsm')]),
        (fugue_all, hmc2, [('out_corrected', 'in_files')]),
        (unwarp, hmc2, [('output_image', 'reference_image')]),
        (hmc2, hmc2moco, [('out_tfm', 'in_files')]),
        (hmc2, tfm_concat, [('out_tfm', 'in_file')]),
        (vsm2dfm, tfm_concat, [('out_file', 'transforms')]),
        (tfm_concat, unwarpall, [
            ('transforms', 'transforms'),
            ('invert_transform_flags', 'invert_transform_flags')]),
        (inputnode, unwarpall, [('in_split', 'input_image')]),
        (target_sel, unwarpall, [('out_file', 'reference_image')]),
        (unwarpall, mean, [('output_image', 'in_files')]),
        (tfm_concat, tfm_comb, [
            ('transforms', 'transforms'),
            ('invert_transform_flags', 'invert_transform_flags')]),
        (target_sel, tfm_comb, [('out_file', 'input_image'),
                                ('out_file', 'reference_image')]),
        (mean, outputnode, [('out_file', 'out_mean')]),
        (unwarpall, outputnode, [('output_image', 'out_files')]),
        (tfm_comb, outputnode, [('output_image', 'out_warps')]),
        (hmc2moco, outputnode, [('out_par', 'out_hmcpar'),
                                ('out_confounds', 'out_confounds')])
    ])
    return workflow

# Helper functions
# ------------------------------------------------------------

def _get_reference(in_files, in_reference, ref_vol=0.5):
    from nipype.interfaces.base import isdefined
    if not isdefined(in_reference) or in_reference is None:
        nfiles = len(in_files)
        return in_files[int(ref_vol * (nfiles - 0.5))]
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
