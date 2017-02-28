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
from nipype.interfaces.ants import Registration
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT
from fmriprep.interfaces.fmap import WarpReference, ApplyFieldmap

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

      outputnode.out_file - the in_file after susceptibility-distortion correction.

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_files', 'in_reference', 'in_hmcpar', 'in_mask', 'in_meta',
                'fmap_ref', 'fmap_mask', 'fmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_warped']), name='outputnode')

    target_sel = pe.Node(niu.Function(
        input_names=['in_files', 'in_reference'], output_names=['out_file'],
        function=_get_reference), name='target_select')

    ref_hdr = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file', 'out_hdr'],
        function=_fixhdr), name='reference_hdr')
    target_hdr = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file', 'out_hdr'],
        function=_fixhdr), name='target_hdr')
    fmap_hdr = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file', 'out_hdr'],
        function=_fixhdr), name='fmap_hdr')

    ref_msk = pe.Node(niu.Function(input_names=['in_file', 'in_mask'],
                      output_names=['out_file'], function=_mask), name='reference_mask')
    target_msk = pe.Node(niu.Function(input_names=['in_file', 'in_mask'],
                         output_names=['out_file'], function=_mask), name='target_mask')

    # Fieldmap to rads
    torads = pe.Node(niu.Function(input_names=['in_file'], output_names=['out_file'],
                     function=hz2rads), name='fmap_Hz2rads')

    # Prepare fieldmap reference image
    ref_wrp = pe.Node(WarpReference(), name='reference_warped')

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json')
    if settings.get('debug', False):
        ants_settings = pkgr.resource_filename(
            'fmriprep', 'data/fmap-any_registration_testing.json')

    fmap2ref = pe.Node(Registration(
        from_file=ants_settings), name='FMap2ImageMagnitude')

    applyxfm = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='BSpline'), name='FMap2ImageFieldmap')

    maskxfm = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='NearestNeighbor'),
                      name='FMap2ImageMask')

    gen_vsm = pe.Node(FUGUE(), name='VSM')
    unwarp = pe.Node(ApplyFieldmap(generate_report=True),
                     name='target_ref_unwarped')

    workflow.connect([
        (inputnode, target_sel, [('in_files', 'in_files'),
                                 ('in_reference', 'in_reference')]),
        (target_sel, target_hdr, [('out_file', 'in_file')]),
        (inputnode, fmap_hdr, [('fmap', 'in_file')]),
        (fmap_hdr, torads, [('out_file', 'in_file')]),
        (inputnode, ref_hdr, [('fmap_ref', 'in_file')]),
        (inputnode, target_msk, [('in_mask', 'in_mask')]),
        (inputnode, applyxfm, [('in_reference', 'reference_image')]),
        (inputnode, ref_wrp, [('fmap_mask', 'in_mask'),
                              (('in_meta', _get_ec), 'echospacing'),
                              (('in_meta', _get_pedir), 'pe_dir')]),
        (inputnode, gen_vsm, [(('in_meta', _get_ec), 'dwell_time'),
                             (('in_meta', _get_pedir), 'unwarp_direction')]),
        (inputnode, unwarp, [(('in_meta', _get_pedir), 'pe_dir')]),
        (ref_hdr, ref_wrp, [('out_file', 'fmap_ref')]),
        (target_hdr, target_msk, [('out_file', 'in_file')]),
        (target_msk, fmap2ref, [('out_file', 'moving_image')]),
        (ref_wrp, maskxfm, [('out_mask', 'input_image')]),
        (inputnode, maskxfm, [('in_reference', 'reference_image')]),
        (fmap2ref, maskxfm, [
            ('reverse_transforms', 'transforms'),
            ('reverse_invert_flags', 'invert_transform_flags')]),
        (torads, ref_wrp, [('out_file', 'in_file')]),
        (ref_wrp, ref_msk, [('out_warped', 'in_file'),
                            ('out_mask', 'in_mask')]),
        (ref_msk, fmap2ref, [('out_file', 'fixed_image')]),
        (fmap2ref, applyxfm, [
            ('reverse_transforms', 'transforms'),
            ('reverse_invert_flags', 'invert_transform_flags')]),
        (torads, applyxfm, [('out_file', 'input_image')]),
        (target_sel, unwarp, [('out_file', 'in_file')]),
        (applyxfm, gen_vsm, [('output_image', 'fmap_in_file')]),
        (gen_vsm, unwarp, [('shift_out_file', 'in_vsm')]),
        (unwarp, outputnode, [('out_corrected', 'out_file')])
    ])
    return workflow

# Disable ApplyTOPUP for now
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

def _get_reference(in_files, in_reference):
    if len(in_files) == 1:
        return in_files[0]
    return in_reference

def _get_ec(in_dict):
    return float(in_dict['EffectiveEchoSpacing'])

def _get_pedir(in_dict):
    return in_dict['PhaseEncodingDirection'].replace('j', 'y').replace('i', 'x')

def _mask(in_file, in_mask, out_file=None):
    import nibabel as nb
    from fmriprep.utils.misc import genfname
    if out_file is None:
        out_file = genfname(in_file, 'brainmask')

    nii = nb.load(in_file)
    data = nii.get_data()
    data[nb.load(in_mask).get_data() <= 0] = 0
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_file)
    return out_file

def _fixhdr(in_file):
    import numpy as np
    import nibabel as nb
    from io import open
    from fmriprep.utils.misc import genfname

    im = nb.as_closest_canonical(nb.load(in_file))
    newaff = np.eye(4)
    newaff[:3, :3] = np.eye(3) * im.header.get_zooms()[:3]
    newaff[:3, 3] -= 0.5 * newaff[:3, :3].dot(im.shape[:3])

    out_file = genfname(in_file, suffix='nosform')
    nb.Nifti1Image(im.get_data(), newaff, None).to_filename(
        out_file)

    out_hdr = genfname(in_file, suffix='hdr', ext='pklz')
    with open(out_hdr, 'wb') as fheader:
        im.header.write_to(fheader)
    return out_file, out_hdr

def hz2rads(in_file, out_file=None):
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

def _last(inlist):
    if isinstance(inlist, list):
        return inlist[-1]
    return inlist

def _first(inlist):
    if isinstance(inlist, list):
        return inlist[0]
    return inlist


def _fn_ras(in_file):
    import os.path as op
    import nibabel as nb
    from fmriprep.utils.misc import genfname

    if isinstance(in_file, list):
        in_file = in_file[0]

    out_file = genfname(in_file, suffix='ras')
    nb.as_closest_canonical(nb.load(in_file)).to_filename(
        out_file)
    return out_file


