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
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from niworkflows.interfaces.registration import ANTSRegistrationRPT, ANTSApplyTransformsRPT

from fmriprep.interfaces.bids import ReadSidecarJSON, DerivativesDataSink
from fmriprep.interfaces.fmap import FieldCoefficients, GenerateMovParams
from fmriprep.interfaces.topup import ConformTopupInputs
from fmriprep.workflows.fieldmap.utils import create_encoding_file
SDC_UNWARP_NAME = 'SDC_unwarp'


def sdc_unwarp(name=SDC_UNWARP_NAME, ref_vol=None, method='jac', settings=None):
    """
    This workflow takes an estimated fieldmap and a target image and applies TOPUP,
    an :abbr:`SDC (susceptibility-derived distortion correction)` method in FSL to
    unwarp the target image.

    Input fields:
    ~~~~~~~~~~~~~

      inputnode.in_file - the image(s) to which this correction will be applied
      inputnode.in_mask - a brain mask corresponding to the in_file image
      inputnode.fmap_ref - the fieldmap reference (generally, a *magnitude* image or the
                           resulting SE image)
      inputnode.fmap_mask - a brain mask in fieldmap-space
      inputnode.fmap - a fieldmap in Hz
      inputnode.hmc_movpar - the head motion parameters (iff inputnode.in_file is only
                             one 4D file)

    Output fields:
    ~~~~~~~~~~~~~~

      outputnode.out_file - the in_file after susceptibility-distortion correction.

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fmap_ref', 'fmap_mask', 'fmap',
                'hmc_movpar']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_warped']), name='outputnode')

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(), iterfield=['in_file'], name='metadata')

    # Compute movpar file iff we have several images with different
    # PE directions.
    conform = pe.Node(ConformTopupInputs(), name='ConformInputs')
    if ref_vol is not None:
        conform.inputs.in_ref = ref_vol

    torads = pe.Node(niu.Function(input_names=['in_file'], output_names=['out_file'],
                     function=hz2rads), name='Hz2rads')

    warpref = pe.Node(niu.Function(
        function=_warp_reference,
        input_names=['fmap_ref', 'fmap', 'fmap_mask', 'metadata'],
        output_names=['fmap_ref', 'fmap_mask']),
        name='TestWarpReference')

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json')
    if settings.get('debug', False):
        ants_settings = pkgr.resource_filename(
            'fmriprep', 'data/fmap-any_registration_testing.json')

    fmap2ref = pe.Node(ANTSRegistrationRPT(from_file=ants_settings, output_warped_image=True,
                       generate_report=True), name='FMap2ImageMagnitude')

    # fugue = pe.Node(fsl.FUGUE(save_unmasked_fmap=True), name='fugue')
    applyxfm = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='Linear'), name='FMap2ImageFieldmap')

    maskxfm = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='NearestNeighbor'),
                      name='FMap2ImageMask')

    unwarp = pe.Node(niu.Function(
        input_names=['in_file', 'in_fieldmap', 'in_mask', 'metadata'],
        output_names=['out_file', 'out_report'], function=_fugue_unwarp),
                     name='Unwarping')

    dsunwarp = pe.Node(DerivativesDataSink(
        base_directory=settings['output_dir'], suffix='unwarp',
        out_path_base='reports'), name='DSunwarp')

    workflow.connect([
        (inputnode, torads, [(('fmap', _last), 'in_file')]),
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, conform, [('in_file', 'in_files'),
                              ('hmc_movpar', 'in_mats')]),
        (inputnode, warpref, [(('fmap_ref', _last), 'fmap_ref'),
                              (('fmap_mask', _last), 'fmap_mask')]),
        (torads, warpref, [('out_file', 'fmap')]),
        (meta, warpref, [('out_dict', 'metadata')]),
        (warpref, fmap2ref, [('fmap_ref', 'moving_image'),
                             ('fmap_mask', 'moving_image_mask')]),
        (conform, fmap2ref, [('out_brain', 'fixed_image'),
                             ('out_mask', 'fixed_image_mask')]),
        # (fmapenh, fugue, [('out_file', 'fmap_in_file')]),
        # (fugue, outputnode, [('fmap_out_file', 'fmap')]),
        (conform, applyxfm, [('out_brain', 'reference_image')]),
        (fmap2ref, applyxfm, [
            ('forward_transforms', 'transforms'),
            ('forward_invert_flags', 'invert_transform_flags')]),
        (conform, maskxfm, [('out_brain', 'reference_image')]),
        (fmap2ref, maskxfm, [
            ('forward_transforms', 'transforms'),
            ('forward_invert_flags', 'invert_transform_flags')]),
        (torads, applyxfm, [('out_file', 'input_image')]),
        (warpref, maskxfm, [('fmap_mask', 'input_image')]),
        (conform, unwarp, [('out_file', 'in_file')]),
        (applyxfm, unwarp, [('output_image', 'in_fieldmap')]),
        (maskxfm, unwarp, [('output_image', 'in_mask')]),
        (meta, unwarp, [('out_dict', 'metadata')]),
        (unwarp, outputnode, [('out_file', 'out_file')]),
        (inputnode, dsunwarp, [('in_file', 'source_file')]),
        (unwarp, dsunwarp, [('out_report', 'in_file')])
    ])

    # Disable ApplyTOPUP for now
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
    #     #                       ('out_movpar', 'in_movpar')]),
    #     (gen_movpar, topup_adapt, [('out_movpar', 'in_movpar')]),
    #     (applyxfm, topup_adapt, [('output_image', 'in_file')]),
    #     (conform, unwarp, [('out_file', 'in_files')]),
    #     (topup_adapt, unwarp, [('out_fieldcoef', 'in_topup_fieldcoef'),
    #                            ('out_movpar', 'in_topup_movpar')]),
    #     (encfile, unwarp, [('unwarp_param', 'encoding_file')]),
    #     (unwarp, outputnode, [('out_corrected', 'out_file')])
    # ])

    return workflow


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

def _fugue_unwarp(in_file, in_fieldmap, in_mask, metadata):
    import nibabel as nb
    from niworkflows.interfaces.registration import FUGUERPT
    from nipype.interfaces.fsl import FUGUE
    from fmriprep.utils.misc import genfname

    nii = nb.load(in_file)
    if nii.get_data().ndim == 4:
        nii_list = nb.four_to_three(nii)
    else:
        nii_list = [nii]

    if not isinstance(metadata, list):
        metadata = [metadata]

    if len(metadata) == 1:
        metadata = metadata * len(nii_list)

    out_files = []
    for i, (tnii, tmeta) in enumerate(zip(nii_list, metadata)):
        tfile = genfname(in_file, 'vol%03d' % i)
        tnii.to_filename(tfile)
        ec = tmeta['EffectiveEchoSpacing']
        ud = tmeta['PhaseEncodingDirection'].replace('j', 'y').replace('i', 'x')

        fmap_umask = genfname(in_file, 'fmap_umask%03d' % i)
        FUGUE(
            fmap_in_file=in_fieldmap, save_unmasked_fmap=True,
            mask_file=in_mask, fmap_out_file=fmap_umask).run()

        vsm_file = genfname(in_file, 'vsm%03d' % i)
        FUGUE(
            fmap_in_file=fmap_umask, dwell_time=ec,
            unwarp_direction=ud, shift_out_file=vsm_file).run()

        fugue = FUGUERPT(
            in_file=tfile, unwarp_direction=ud, icorr=True, shift_in_file=vsm_file,
            unwarped_file=genfname(in_file, 'unwarped%03d' % i),
            generate_report=True)

        # print('Running FUGUE: %s' % fugue.cmdline)
        fugue_res = fugue.run()
        out_files.append(fugue_res.outputs.unwarped_file)

    corr_nii = nb.concat_images([nb.load(f) for f in out_files])
    out_file = genfname(in_file, 'unwarped')
    corr_nii.to_filename(out_file)
    return out_file, fugue_res.outputs.out_report


def _warp_reference(fmap_ref, fmap, fmap_mask, metadata):
    import numpy as np
    import nibabel as nb
    from nipype.interfaces.fsl import FUGUE
    from fmriprep.utils.misc import genfname

    if isinstance(metadata, (list, tuple)):
        metadata = metadata[0]
    ec = metadata['EffectiveEchoSpacing']
    ud = metadata['PhaseEncodingDirection'].replace('j', 'y').replace('i', 'x')

    vsm_file = genfname(fmap_ref, 'vsm')
    gen_vsm = FUGUE(
        fmap_in_file=fmap, dwell_time=ec,
        unwarp_direction=ud, shift_out_file=vsm_file)
    gen_vsm.run()

    fugue = FUGUE(
        in_file=fmap_ref, shift_in_file=vsm_file, nokspace=True,
        forward_warping=True, unwarp_direction=ud, icorr=False,
        warped_file=genfname(fmap_ref, 'warped%s' % ud)).run()

    fuguemask = FUGUE(
        in_file=fmap_mask, shift_in_file=vsm_file, nokspace=True,
        forward_warping=True, unwarp_direction=ud, icorr=False,
        warped_file=genfname(fmap_mask, 'maskwarped%s' % ud)).run()

    masknii = nb.load(fuguemask.outputs.warped_file)
    mask = masknii.get_data()
    mask[mask > 0] = 1
    mask[mask < 1] = 0
    out_mask = genfname(fmap_mask, 'warp_bin')
    nb.Nifti1Image(mask.astype(np.uint8), masknii.affine,
                   masknii.header).to_filename(out_mask)
    return fugue.outputs.warped_file, out_mask

def _last(inlist):
    return inlist[-1]