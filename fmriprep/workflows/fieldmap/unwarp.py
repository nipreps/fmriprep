#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_unwarp :

Unwarping
~~~~~~~~~

.. topic :: Abbreviations

    fmap
        fieldmap
    VSM
        voxel-shift map -- a 3D nifti where displacements are in pixels (not mm)
    DFM
        displacements field map -- a nifti warp file compatible with ANTs (mm)

"""

import pkg_resources as pkgr

from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import ants, fsl, utility as niu
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT, ANTSRegistrationRPT

from ...interfaces import itk, ReadSidecarJSON, DerivativesDataSink
from ...interfaces.fmap import (
    get_ees as _get_ees,
    FieldToRadS,
)
from ...interfaces.images import (
    DemeanImage, FilledImageLike
)
from ..bold.util import init_enhance_and_skullstrip_bold_wf


def init_sdc_unwarp_wf(reportlets_dir, omp_nthreads, fmap_bspline,
                       fmap_demean, debug, name='sdc_unwarp_wf'):
    """
    This workflow takes in a displacements fieldmap and calculates the corresponding
    displacements field (in other words, an ANTs-compatible warp file).

    It also calculates a new mask for the input dataset that takes into account the distortions.
    The mask is restricted to the field of view of the fieldmap since outside of it corrections
    could not be performed.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.unwarp import init_sdc_unwarp_wf
        wf = init_sdc_unwarp_wf(reportlets_dir='.', omp_nthreads=8,
                                fmap_bspline=False, fmap_demean=True,
                                debug=False)


    Inputs

        in_reference
            the reference image
        in_mask
            a brain mask corresponding to ``in_reference``
        name_source
            path to the original _bold file being unwarped
        fmap
            the fieldmap in Hz
        fmap_ref
            the reference (anatomical) image corresponding to ``fmap``
        fmap_mask
            a brain mask corresponding to ``fmap``


    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_jacobian
            the jacobian of the field (for drop-out alleviation)
        out_mask
            mask of the unwarped input file

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_reference', 'in_reference_brain', 'in_mask', 'name_source',
                'fmap_ref', 'fmap_mask', 'fmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_warp', 'out_mask',
                'out_jacobian']), name='outputnode')

    meta = pe.Node(ReadSidecarJSON(), name='meta',
                   mem_gb=0.01, run_without_submitting=True)

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json')
    if debug:
        ants_settings = pkgr.resource_filename(
            'fmriprep', 'data/fmap-any_registration_testing.json')
    fmap2ref_reg = pe.Node(
        ANTSRegistrationRPT(generate_report=True, from_file=ants_settings,
                            output_inverse_warped_image=True, output_warped_image=True),
        name='fmap2ref_reg', n_procs=omp_nthreads)

    ds_reg = pe.Node(DerivativesDataSink(
        base_directory=reportlets_dir, suffix='fmap_reg'), name='ds_reg',
        mem_gb=0.01, run_without_submitting=True)

    # Map the VSM into the EPI space
    fmap2ref_apply = pe.Node(ANTSApplyTransformsRPT(
        generate_report=True, dimension=3, interpolation='BSpline', float=True),
        name='fmap2ref_apply')

    fmap_mask2ref_apply = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='MultiLabel',
        float=True),
        name='fmap_mask2ref_apply')

    ds_reg_vsm = pe.Node(DerivativesDataSink(
        base_directory=reportlets_dir, suffix='fmap_reg_vsm'), name='ds_reg_vsm',
        mem_gb=0.01, run_without_submitting=True)

    # Fieldmap to rads and then to voxels (VSM - voxel shift map)
    torads = pe.Node(FieldToRadS(fmap_range=0.5), name='torads')

    get_ees = pe.Node(niu.Function(function=_get_ees, output_names=['ees']), name='get_ees')

    gen_vsm = pe.Node(fsl.FUGUE(save_unmasked_shift=True), name='gen_vsm')
    # Convert the VSM into a DFM (displacements field map)
    # or: FUGUE shift to ANTS warping.
    vsm2dfm = pe.Node(itk.FUGUEvsm2ANTSwarp(), name='vsm2dfm')
    jac_dfm = pe.Node(ants.CreateJacobianDeterminantImage(
        imageDimension=3, outputImage='jacobian.nii.gz'), name='jac_dfm')

    unwarp_reference = pe.Node(ANTSApplyTransformsRPT(dimension=3,
                                                      generate_report=False,
                                                      float=True,
                                                      interpolation='LanczosWindowedSinc'),
                               name='unwarp_reference')

    fieldmap_fov_mask = pe.Node(FilledImageLike(dtype='uint8'), name='fieldmap_fov_mask')

    fmap_fov2ref_apply = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='NearestNeighbor',
        float=True),
        name='fmap_fov2ref_apply')

    apply_fov_mask = pe.Node(fsl.ApplyMask(), name="apply_fov_mask")

    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode, meta, [('name_source', 'in_file')]),
        (inputnode, fmap2ref_reg, [('fmap_ref', 'moving_image')]),
        (inputnode, fmap2ref_apply, [('in_reference', 'reference_image')]),
        (fmap2ref_reg, fmap2ref_apply, [
            ('composite_transform', 'transforms')]),
        (inputnode, fmap_mask2ref_apply, [('in_reference', 'reference_image')]),
        (fmap2ref_reg, fmap_mask2ref_apply, [
            ('composite_transform', 'transforms')]),
        (inputnode, ds_reg_vsm, [('name_source', 'source_file')]),
        (fmap2ref_apply, ds_reg_vsm, [('out_report', 'in_file')]),
        (inputnode, fmap2ref_reg, [('in_reference_brain', 'fixed_image')]),
        (inputnode, ds_reg, [('name_source', 'source_file')]),
        (fmap2ref_reg, ds_reg, [('out_report', 'in_file')]),
        (inputnode, fmap2ref_apply, [('fmap', 'input_image')]),
        (inputnode, fmap_mask2ref_apply, [('fmap_mask', 'input_image')]),
        (fmap2ref_apply, torads, [('output_image', 'in_file')]),
        (meta, get_ees, [('out_dict', 'in_meta')]),
        (inputnode, get_ees, [('name_source', 'in_file')]),
        (get_ees, gen_vsm, [('ees', 'dwell_time')]),
        (meta, gen_vsm, [(('out_dict', _get_pedir_fugue), 'unwarp_direction')]),
        (meta, vsm2dfm, [(('out_dict', _get_pedir_bids), 'pe_dir')]),
        (torads, gen_vsm, [('out_file', 'fmap_in_file')]),
        (vsm2dfm, unwarp_reference, [('out_file', 'transforms')]),
        (inputnode, unwarp_reference, [('in_reference', 'reference_image')]),
        (inputnode, unwarp_reference, [('in_reference', 'input_image')]),
        (vsm2dfm, outputnode, [('out_file', 'out_warp')]),
        (vsm2dfm, jac_dfm, [('out_file', 'deformationField')]),
        (inputnode, fieldmap_fov_mask, [('fmap_ref', 'in_file')]),
        (fieldmap_fov_mask, fmap_fov2ref_apply, [('out_file', 'input_image')]),
        (inputnode, fmap_fov2ref_apply, [('in_reference', 'reference_image')]),
        (fmap2ref_reg, fmap_fov2ref_apply, [('composite_transform', 'transforms')]),
        (fmap_fov2ref_apply, apply_fov_mask, [('output_image', 'mask_file')]),
        (unwarp_reference, apply_fov_mask, [('output_image', 'in_file')]),
        (apply_fov_mask, enhance_and_skullstrip_bold_wf, [('out_file', 'inputnode.in_file')]),
        (apply_fov_mask, outputnode, [('out_file', 'out_reference')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.mask_file', 'out_mask'),
            ('outputnode.skull_stripped_file', 'out_reference_brain')]),
        (jac_dfm, outputnode, [('jacobian_image', 'out_jacobian')]),
    ])

    if not fmap_bspline:
        workflow.connect([
            (fmap_mask2ref_apply, gen_vsm, [('output_image', 'mask_file')])
        ])

    if fmap_demean:
        # Demean within mask
        demean = pe.Node(DemeanImage(), name='demean')

        workflow.connect([
            (gen_vsm, demean, [('shift_out_file', 'in_file')]),
            (fmap_mask2ref_apply, demean, [('output_image', 'in_mask')]),
            (demean, vsm2dfm, [('out_file', 'in_file')]),
        ])

    else:
        workflow.connect([
            (gen_vsm, vsm2dfm, [('shift_out_file', 'in_file')]),
        ])

    return workflow


def init_fmap_unwarp_report_wf(reportlets_dir, name='fmap_unwarp_report_wf'):
    """
    This workflow generates and saves a reportlet showing the effect of fieldmap
    unwarping a BOLD image.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.unwarp import init_fmap_unwarp_report_wf
        wf = init_fmap_unwarp_report_wf(reportlets_dir='.')

    **Parameters**

        reportlets_dir : str
            Directory in which to save reportlets
        name : str, optional
            Workflow name (default: fmap_unwarp_report_wf)

    **Inputs**

        in_pre
            Reference image, before unwarping
        in_post
            Reference image, after unwarping
        in_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        in_xfm
            Affine transform from T1 space to BOLD space (ITK format)
        name_source
            BOLD series NIfTI file
            Used to recover original information lost during processing

    """

    from niworkflows.nipype.pipeline import engine as pe
    from niworkflows.nipype.interfaces import utility as niu
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    from niworkflows.interfaces import SimpleBeforeAfter
    from ...interfaces.images import extract_wm
    from ...interfaces import DerivativesDataSink

    DEFAULT_MEMORY_MIN_GB = 0.01

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_pre', 'in_post', 'in_seg', 'in_xfm',
                'name_source']), name='inputnode')

    map_seg = pe.Node(ApplyTransforms(
        dimension=3, float=True, interpolation='MultiLabel'),
        name='map_seg', mem_gb=0.3)

    sel_wm = pe.Node(niu.Function(function=extract_wm), name='sel_wm',
                     mem_gb=DEFAULT_MEMORY_MIN_GB)

    bold_rpt = pe.Node(SimpleBeforeAfter(), name='bold_rpt',
                       mem_gb=0.1)
    bold_rpt_ds = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='variant-hmcsdc_preproc'), name='bold_rpt_ds',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True
    )
    workflow.connect([
        (inputnode, bold_rpt, [('in_post', 'after'),
                               ('in_pre', 'before')]),
        (inputnode, bold_rpt_ds, [('name_source', 'source_file')]),
        (bold_rpt, bold_rpt_ds, [('out_report', 'in_file')]),
        (inputnode, map_seg, [('in_post', 'reference_image'),
                              ('in_seg', 'input_image'),
                              ('in_xfm', 'transforms')]),
        (map_seg, sel_wm, [('output_image', 'in_seg')]),
        (sel_wm, bold_rpt, [('out', 'wm_seg')]),
    ])

    return workflow


# Helper functions
# ------------------------------------------------------------


def _get_pedir_bids(in_dict):
    return in_dict['PhaseEncodingDirection']


def _get_pedir_fugue(in_dict):
    return in_dict['PhaseEncodingDirection'].replace('i', 'x').replace('j', 'y').replace('k', 'z')
