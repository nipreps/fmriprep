#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)

"""
import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces import utility as niu

from fmriprep.interfaces.bids import ReadSidecarJSON
from fmriprep.workflows.fieldmap.utils import create_encoding_file

SDC_UNWARP_NAME = 'SDC_unwarp'


def sdc_unwarp(name=SDC_UNWARP_NAME, ref_vol=None, method='jac'):
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
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    # Compute movpar file iff we have several images with different
    # PE directions.
    align = pe.Node(niu.Function(
        input_names=['in_files', 'in_movpar'],
        output_names=['out_file', 'ref_vol', 'ref_mask', 'out_movpar'],
        function=_multiple_pe_hmc), name='AlignMultiplePE')
    align.inputs.in_ref = ref_vol

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(), iterfield=['in_file'], name='metadata')

    encfile = pe.Node(interface=niu.Function(
        input_names=['input_images', 'in_dict'], output_names=['parameters_file'],
        function=create_encoding_file), name='TopUp_encfile', updatehash=True)

    fslsplit = pe.Node(fsl.Split(dimension='t'), name='ImageHMCSplit')

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    fmap2ref = pe.Node(ants.Registration(
        from_file=pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json'),
        output_warped_image=True), name='Fieldmap2ImageRegistration')

    applyxfm = pe.Node(ants.ApplyTransforms(
        dimension=3, interpolation='Linear'), name='Fieldmap2ImageApply')

    fix_movpar = pe.Node(niu.Function(
        input_names=['in_files'], output_names=['out_movpar'],
        function=_fix_movpar), name='FixMovPar')

    topup_adapt = pe.Node(niu.Function(
        input_names=['in_file', 'in_ref', 'in_movpar'],
        output_names=['out_fieldcoef', 'out_movpar'],
        function=_gen_coeff), name='TopUpAdapt')

    # Use the least-squares method to correct the dropout of the SBRef images
    unwarp = pe.Node(fsl.ApplyTOPUP(method=method, in_index=[1]), name='TopUpApply')

    workflow.connect([
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, align, [('in_file', 'in_files'),
                            ('hmc_movpar', 'in_movpar')]),
        (inputnode, applyxfm, [('fmap', 'input_image')]),
        (inputnode, encfile, [('in_file', 'input_images')]),
        (inputnode, fmap2ref, [('fmap_ref', 'moving_image'),
                               ('fmap_mask', 'moving_image_mask')]),

        (align, fmap2ref, [('ref_vol', 'fixed_image'),
                           ('ref_mask', 'fixed_image_mask')]),
        (align, applyxfm, [('ref_vol', 'reference_image')]),
        (align, topup_adapt, [('ref_vol', 'in_ref')]),
        #                      ('out_movpar', 'in_movpar')]),

        (meta, encfile, [('out_dict', 'in_dict')]),

        (fmap2ref, applyxfm, [
            ('forward_transforms', 'transforms'),
            ('forward_invert_flags', 'invert_transform_flags')]),
        (align, fslsplit, [('out_file', 'in_file')]),
        (fslsplit, fix_movpar, [('out_files', 'in_files')]),
        (fix_movpar, topup_adapt, [('out_movpar', 'in_movpar')]),
        (applyxfm, topup_adapt, [('output_image', 'in_file')]),
        (align, unwarp, [('out_file', 'in_files')]),
        (topup_adapt, unwarp, [('out_fieldcoef', 'in_topup_fieldcoef'),
                               ('out_movpar', 'in_topup_movpar')]),
        (encfile, unwarp, [('parameters_file', 'encoding_file')]),
        (unwarp, outputnode, [('out_corrected', 'out_file')])
    ])

    return workflow


def _multiple_pe_hmc(in_files, in_movpar, in_ref=None):
    """
    This function interprets that we are dealing with a
    multiple PE (phase encoding) input if it finds several
    files in in_files.

    If we have several images with various PE directions,
    it will compute the HMC parameters between them using
    an embedded workflow.

    It just forwards the two inputs otherwise.
    """
    from six import string_types
    from nipype.interfaces import fsl
    from nipype.interfaces import ants
    from niworkflows.interfaces.masks import BETRPT

    if isinstance(in_files, string_types):
        in_files = [in_files]

    if len(in_files) == 1:
        out_file = in_files[0]
        out_movpar = in_movpar
    else:
        if in_ref is None:
            in_ref = 0

        # Head motion correction
        fslmerge = fsl.Merge(dimension='t', in_files=in_files)
        hmc = fsl.MCFLIRT(ref_vol=in_ref, save_mats=True, save_plots=True)
        hmc.inputs.in_file = fslmerge.run().outputs.merged_file
        hmc_res = hmc.run()
        out_file = hmc_res.outputs.out_file
        out_movpar = hmc_res.outputs.par_file

    mean = fsl.MeanImage(
        dimension='T', in_file=out_file)
    inu = ants.N4BiasFieldCorrection(
        dimension=3, input_image=mean.run().outputs.out_file)
    inu_res = inu.run()
    out_ref = inu_res.outputs.output_image
    bet = BETRPT(generate_report=True, frac=0.6, mask=True, in_file=out_ref)
    out_mask = bet.run().outputs.mask_file

    return (out_file, out_ref, out_mask, out_movpar)


def _gen_coeff(in_file, in_ref, in_movpar):
    """Convert to a valid fieldcoeff"""
    from shutil import copy
    import numpy as np
    import nibabel as nb
    from nipype.interfaces import fsl

    def _get_fname(in_file):
        import os.path as op
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        return op.abspath(fname)

    out_topup = _get_fname(in_file)

    # 1. Add one dimension (4D image) of 3D coordinates
    im0 = nb.load(in_file)
    data = np.zeros_like(im0.get_data())
    sizes = data.shape[:3]
    spacings = im0.get_header().get_zooms()[:3]
    im1 = nb.Nifti1Image(data, im0.get_affine(), im0.get_header())
    im4d = nb.concat_images([im0, im1, im1])
    im4d_fname = '{}_{}'.format(out_topup, 'field4D.nii.gz')
    im4d.to_filename(im4d_fname)

    # 2. Warputils to compute bspline coefficients
    to_coeff = fsl.WarpUtils(out_format='spline', knot_space=(2, 2, 2))
    to_coeff.inputs.in_file = im4d_fname
    to_coeff.inputs.reference = in_ref

    # 3. Remove unnecessary dims (Y and Z)
    get_first = fsl.ExtractROI(t_min=0, t_size=1)
    get_first.inputs.in_file = to_coeff.run().outputs.out_file

    # 4. Set correct header
    # see https://github.com/poldracklab/fmriprep/issues/92
    img = nb.load(get_first.run().outputs.roi_file)
    hdr = img.get_header().copy()
    hdr['intent_p1'] = spacings[0]
    hdr['intent_p2'] = spacings[1]
    hdr['intent_p3'] = spacings[2]
    hdr['intent_code'] = 2016

    sform = np.eye(4)
    sform[:3, 3] = sizes
    hdr.set_sform(sform, code='scanner')
    hdr['qform_code'] = 1

    out_movpar = '{}_movpar.txt'.format(out_topup)
    copy(in_movpar, out_movpar)

    out_fieldcoef = '{}_fieldcoef.nii.gz'.format(out_topup)
    nb.Nifti1Image(img.get_data(), None, hdr).to_filename(out_fieldcoef)

    return out_fieldcoef, out_movpar


def _fix_movpar(in_files):
    import numpy as np
    # For some reason, MCFLIRT's parameters
    # are not compatible, fill with zeroes for now
    out_movpar = '{}_movpar.txt'.format(in_files[0])
    np.savetxt(out_movpar, np.zeros((len(in_files), 6)))
    return out_movpar
