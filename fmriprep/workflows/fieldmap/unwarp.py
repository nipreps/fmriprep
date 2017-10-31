#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


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
from niworkflows.nipype.interfaces import afni, ants, fsl, utility as niu
from niworkflows.interfaces import CopyHeader
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT, ANTSRegistrationRPT

from ...interfaces import itk, ReadSidecarJSON, StructuralReference, DerivativesDataSink
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
        out_mask_report
            reportled for the skullstripping

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_reference', 'in_reference_brain', 'in_mask', 'name_source',
                'fmap_ref', 'fmap_mask', 'fmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_warp', 'out_mask',
                'out_jacobian', 'out_mask_report']), name='outputnode')

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
        generate_report=False, dimension=3, interpolation='NearestNeighbor',
        float=True),
        name='fmap_mask2ref_apply')

    ds_reg_vsm = pe.Node(DerivativesDataSink(
        base_directory=reportlets_dir, suffix='fmap_reg_vsm'), name='ds_reg_vsm',
        mem_gb=0.01, run_without_submitting=True)

    # Fieldmap to rads and then to voxels (VSM - voxel shift map)
    torads = pe.Node(niu.Function(function=_hz2rads), name='torads')

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

    fieldmap_fov_mask = pe.Node(niu.Function(function=_fill_with_ones), name='fieldmap_fov_mask')

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
        (meta, gen_vsm, [(('out_dict', _get_ec), 'dwell_time'),
                         (('out_dict', _get_pedir_fugue), 'unwarp_direction')]),
        (meta, vsm2dfm, [(('out_dict', _get_pedir_bids), 'pe_dir')]),
        (torads, gen_vsm, [('out', 'fmap_in_file')]),
        (vsm2dfm, unwarp_reference, [('out_file', 'transforms')]),
        (inputnode, unwarp_reference, [('in_reference', 'reference_image')]),
        (inputnode, unwarp_reference, [('in_reference', 'input_image')]),
        (vsm2dfm, outputnode, [('out_file', 'out_warp')]),
        (vsm2dfm, jac_dfm, [('out_file', 'deformationField')]),
        (inputnode, fieldmap_fov_mask, [('fmap_ref', 'in_file')]),
        (fieldmap_fov_mask, fmap_fov2ref_apply, [('out', 'input_image')]),
        (inputnode, fmap_fov2ref_apply, [('in_reference', 'reference_image')]),
        (fmap2ref_reg, fmap_fov2ref_apply, [('composite_transform', 'transforms')]),
        (fmap_fov2ref_apply, apply_fov_mask, [('output_image', 'mask_file')]),
        (unwarp_reference, apply_fov_mask, [('output_image', 'in_file')]),
        (apply_fov_mask, enhance_and_skullstrip_bold_wf, [('out_file', 'inputnode.in_file')]),
        (apply_fov_mask, outputnode, [('out_file', 'out_reference')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.mask_file', 'out_mask'),
            ('outputnode.out_report', 'out_mask_report'),
            ('outputnode.skull_stripped_file', 'out_reference_brain')]),
        (jac_dfm, outputnode, [('jacobian_image', 'out_jacobian')]),
    ])

    if not fmap_bspline:
        workflow.connect([
            (fmap_mask2ref_apply, gen_vsm, [('output_image', 'mask_file')])
        ])

    if fmap_demean:
        # Demean within mask
        demean = pe.Node(niu.Function(function=_demean), name='demean')

        workflow.connect([
            (gen_vsm, demean, [('shift_out_file', 'in_file')]),
            (fmap_mask2ref_apply, demean, [('output_image', 'in_mask')]),
            (demean, vsm2dfm, [('out', 'in_file')]),
        ])

    else:
        workflow.connect([
            (gen_vsm, vsm2dfm, [('shift_out_file', 'in_file')]),
        ])

    return workflow


def init_pepolar_unwarp_wf(fmaps, bold_file, omp_nthreads, layout=None,
                           fmaps_pes=None, bold_file_pe=None,
                           name="pepolar_unwarp_wf"):
    """
    This workflow takes in a set of EPI files with opposite phase encoding
    direction than the target file and calculates a displacements field
    (in other words, an ANTs-compatible warp file).

    This procedure works if there is only one '_epi' file is present
    (as long as it has the opposite phase encoding direction to the target
    file). The target file will be used to estimate the field distortion.
    However, if there is another '_epi' file present with a matching
    phase encoding direction to the target it will be used instead.

    Currently, different phase encoding dimension in the target file and the
    '_epi' file(s) (for example 'i' and 'j') is not supported.

    The warp field correcting for the distortions is estimated using AFNI's
    3dQwarp, with displacement estimation limited to the target file phase
    encoding direction.

    It also calculates a new mask for the input dataset that takes into
    account the distortions.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.unwarp import init_pepolar_unwarp_wf
        wf = init_pepolar_unwarp_wf(fmaps=['/dataset/sub-01/fmap/sub-01_epi.nii.gz'],
                                    fmaps_pes=['j-'],
                                    bold_file='/dataset/sub-01/func/sub-01_task-rest_bold.nii.gz',
                                    bold_file_pe='j',
                                    omp_nthreads=8)


    Inputs

        in_reference
            the reference image
        in_reference_brain
            the reference image skullstripped
        in_mask
            a brain mask corresponding to ``in_reference``
        name_source
            not used, kept for signature compatibility with ``init_sdc_unwarp_wf``

    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_mask
            mask of the unwarped input file
        out_mask_report
            reportlet for the skullstripping

    """
    if not bold_file_pe:
        bold_file_pe = layout.get_metadata(bold_file)["PhaseEncodingDirection"]

    usable_fieldmaps_matching_pe = []
    usable_fieldmaps_opposite_pe = []
    args = '-noXdis -noYdis -noZdis'
    rm_arg = {'i': '-noXdis',
              'j': '-noYdis',
              'k': '-noZdis'}[bold_file_pe[0]]
    args = args.replace(rm_arg, '')

    for i, fmap in enumerate(fmaps):
        if fmaps_pes:
            fmap_pe = fmaps_pes[i]
        else:
            fmap_pe = layout.get_metadata(fmap)["PhaseEncodingDirection"]
        if fmap_pe[0] == bold_file_pe[0]:
            if len(fmap_pe) != len(bold_file_pe):
                add_list = usable_fieldmaps_opposite_pe
            else:
                add_list = usable_fieldmaps_matching_pe
            add_list.append(fmap)

    if not usable_fieldmaps_opposite_pe:
        raise Exception("None of the discovered fieldmaps has the right "
                        "phase encoding direction. Possibly a problem with "
                        "metadata. If not, rerun with '--ignore fieldmaps' to "
                        "skip distortion correction step.")

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_reference', 'in_reference_brain', 'in_mask', 'name_source']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_warp', 'out_mask',
                'out_mask_report']),
        name='outputnode')

    prepare_epi_opposite_wf = init_prepare_epi_wf(omp_nthreads=omp_nthreads,
                                                  name="prepare_epi_opposite_wf")
    prepare_epi_opposite_wf.inputs.inputnode.fmaps = usable_fieldmaps_opposite_pe

    qwarp = pe.Node(afni.QwarpPlusMinus(pblur=[0.05, 0.05],
                                        blur=[-1, -1],
                                        noweight=True,
                                        minpatch=9,
                                        nopadWARP=True,
                                        environ={'OMP_NUM_THREADS': '%d' % omp_nthreads},
                                        args=args),
                    name='qwarp', n_procs=omp_nthreads)

    workflow.connect([
        (inputnode, prepare_epi_opposite_wf, [('in_reference_brain', 'inputnode.ref_brain')]),
        (prepare_epi_opposite_wf, qwarp, [('outputnode.out_file', 'base_file')]),
    ])

    if usable_fieldmaps_matching_pe:
        prepare_epi_matching_wf = init_prepare_epi_wf(omp_nthreads=omp_nthreads,
                                                      name="prepare_epi_matching_wf")
        prepare_epi_matching_wf.inputs.inputnode.fmaps = usable_fieldmaps_matching_pe

        workflow.connect([
            (inputnode, prepare_epi_matching_wf, [('in_reference_brain', 'inputnode.ref_brain')]),
            (prepare_epi_matching_wf, qwarp, [('outputnode.out_file', 'source_file')]),
        ])
    else:
        workflow.connect([(inputnode, qwarp, [('in_reference_brain', 'source_file')])])

    to_ants = pe.Node(niu.Function(function=_fix_hdr), name='to_ants',
                      mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name='cphdr_warp',
                         mem_gb=0.01, run_without_submitting=True)

    unwarp_reference = pe.Node(ANTSApplyTransformsRPT(dimension=3,
                                                      generate_report=False,
                                                      float=True,
                                                      interpolation='LanczosWindowedSinc'),
                               name='unwarp_reference')

    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode, cphdr_warp, [('in_reference', 'hdr_file')]),
        (qwarp, cphdr_warp, [('source_warp', 'in_file')]),
        (cphdr_warp, to_ants, [('out_file', 'in_file')]),
        (to_ants, unwarp_reference, [('out', 'transforms')]),
        (inputnode, unwarp_reference, [('in_reference', 'reference_image'),
                                       ('in_reference', 'input_image')]),
        (unwarp_reference, enhance_and_skullstrip_bold_wf, [
            ('output_image', 'inputnode.in_file')]),
        (unwarp_reference, outputnode, [('output_image', 'out_reference')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.mask_file', 'out_mask'),
            ('outputnode.out_report', 'out_mask_report'),
            ('outputnode.skull_stripped_file', 'out_reference_brain')]),
        (to_ants, outputnode, [('out', 'out_warp')]),
    ])

    return workflow


def init_prepare_epi_wf(omp_nthreads, name="prepare_epi_wf"):
    """
    This workflow takes in a set of EPI files with with the same phase
    encoding direction and returns a single 3D volume ready to be used in
    field distortion estimation.

    The procedure involves: estimating a robust template using FreeSurfer's
    'mri_robust_template', bias field correction using ANTs N4BiasFieldCorrection
    and AFNI 3dUnifize, skullstripping using FSL BET and AFNI 3dAutomask,
    and rigid coregistration to the reference using ANTs.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.unwarp import init_prepare_epi_wf
        wf = init_prepare_epi_wf(omp_nthreads=8)


    Inputs

        fmaps
            list of 3D or 4D NIfTI images
        ref_brain
            coregistration reference (skullstripped and bias field corrected)

    Outputs

        out_file
            single 3D NIfTI file

    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['fmaps', 'ref_brain']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    split = pe.MapNode(fsl.Split(dimension='t'), iterfield='in_file',
                       name='split')

    merge = pe.Node(
        StructuralReference(auto_detect_sensitivity=True,
                            initial_timepoint=1,
                            fixed_timepoint=True,  # Align to first image
                            intensity_scaling=True,
                            # 7-DOF (rigid + intensity)
                            no_iteration=True,
                            subsample_threshold=200,
                            out_file='template.nii.gz'),
        name='merge')

    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads)

    ants_settings = pkgr.resource_filename('fmriprep',
                                           'data/translation_rigid.json')
    fmap2ref_reg = pe.Node(ants.Registration(from_file=ants_settings,
                                             output_warped_image=True),
                           name='fmap2ref_reg', n_procs=omp_nthreads)

    workflow = pe.Workflow(name=name)

    def _flatten(l):
        from niworkflows.nipype.utils.filemanip import filename_to_list
        return [item for sublist in l for item in filename_to_list(sublist)]

    workflow.connect([
        (inputnode, split, [('fmaps', 'in_file')]),
        (split, merge, [(('out_files', _flatten), 'in_files')]),
        (merge, enhance_and_skullstrip_bold_wf, [('out_file', 'inputnode.in_file')]),
        (enhance_and_skullstrip_bold_wf, fmap2ref_reg, [
            ('outputnode.skull_stripped_file', 'moving_image')]),
        (inputnode, fmap2ref_reg, [('ref_brain', 'fixed_image')]),
        (fmap2ref_reg, outputnode, [('warped_image', 'out_file')]),
    ])

    return workflow


# Helper functions
# ------------------------------------------------------------


def _fix_hdr(in_file):
    import nibabel as nb
    import os

    nii = nb.load(in_file)
    hdr = nii.header.copy()
    hdr.set_data_dtype('<f4')
    hdr.set_intent('vector', (), '')

    out_file = os.path.abspath("warpfield.nii.gz")

    nb.Nifti1Image(nii.get_data().astype('<f4'), nii.affine, hdr).to_filename(out_file)

    return out_file


def _get_ec(in_dict):
    return float(in_dict['EffectiveEchoSpacing'])


def _get_pedir_bids(in_dict):
    return in_dict['PhaseEncodingDirection']


def _get_pedir_fugue(in_dict):
    return in_dict['PhaseEncodingDirection'].replace('i', 'x').replace('j', 'y').replace('k', 'z')


def _hz2rads(in_file, out_file=None):
    """Transform a fieldmap in Hz into rad/s"""
    import os
    from math import pi
    import nibabel as nb
    from niworkflows.nipype.utils.filemanip import fname_presuffix
    if out_file is None:
        out_file = fname_presuffix(in_file, suffix='_rads',
                                   newpath=os.getcwd())
    nii = nb.load(in_file)
    data = nii.get_data() * 2.0 * pi
    nb.Nifti1Image(data, nii.get_affine(),
                   nii.get_header()).to_filename(out_file)
    return out_file


def _demean(in_file, in_mask, out_file=None):
    import os
    import numpy as np
    import nibabel as nb
    from niworkflows.nipype.utils.filemanip import fname_presuffix

    if out_file is None:
        out_file = fname_presuffix(in_file, suffix='_demeaned',
                                   newpath=os.getcwd())
    nii = nb.load(in_file)
    msk = nb.load(in_mask).get_data()
    data = nii.get_data()
    data -= np.median(data[msk > 0])
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(
        out_file)
    return out_file


def _fill_with_ones(in_file):
    import nibabel as nb
    import numpy as np
    import os

    nii = nb.load(in_file)
    data = np.ones(nii.shape)

    out_name = os.path.abspath("out.nii.gz")
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_name)

    return out_name
