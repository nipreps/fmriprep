#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

.. autofunction:: init_enhance_and_skullstrip_bold_wf
.. autofunction:: init_skullstrip_bold_wf

"""

import os
import os.path as op

from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import utility as niu
from niworkflows.interfaces.utils import CopyXForm
from niworkflows.nipype.interfaces import fsl, afni, c3, ants, freesurfer as fs
from niworkflows.interfaces.registration import FLIRTRPT, BBRegisterRPT, MRICoregRPT
from niworkflows.interfaces.masks import SimpleShowMaskRPT

from ..interfaces.images import extract_wm
# See https://github.com/poldracklab/fmriprep/issues/768
from ..interfaces.freesurfer import PatchedConcatenateLTA as ConcatenateLTA

DEFAULT_MEMORY_MIN_GB = 0.01


def compare_xforms(lta_list, norm_threshold=15):
    """
    Computes a normalized displacement between two affine transforms as the
    maximum overall displacement of the midpoints of the faces of a cube, when
    each transform is applied to the cube.
    This combines displacement resulting from scaling, translation and rotation.

    Although the norm is in mm, in a scaling context, it is not necessarily
    equivalent to that distance in translation.

    We choose a default threshold of 15mm as a rough heuristic.
    Normalized displacement above 20mm showed clear signs of distortion, while
    "good" BBR refinements were frequently below 10mm displaced from the rigid
    transform.
    The 10-20mm range was more ambiguous, and 15mm chosen as a compromise.
    This is open to revisiting in either direction.

    See discussion in
    `GitHub issue #681`_ <https://github.com/poldracklab/fmriprep/issues/681>`_
    and the `underlying implementation
    <https://github.com/nipy/nipype/blob/56b7c81eedeeae884ba47c80096a5f66bd9f8116/nipype/algorithms/rapidart.py#L108-L159>`_.

    Parameters
    ----------

      lta_list : list or tuple of str
          the two given affines in LTA format
      norm_threshold : float (default: 15)
          the upper bound limit to the normalized displacement caused by the
          second transform relative to the first

    """
    from fmriprep.interfaces.surf import load_transform
    from niworkflows.nipype.algorithms.rapidart import _calc_norm_affine

    bbr_affine = load_transform(lta_list[0])
    fallback_affine = load_transform(lta_list[1])

    norm, _ = _calc_norm_affine([fallback_affine, bbr_affine], use_differences=True)

    return norm[1] > norm_threshold


def init_enhance_and_skullstrip_bold_wf(name='enhance_and_skullstrip_bold_wf',
                                        omp_nthreads=1):
    """
    This workflow takes in a BOLD volume, and attempts to enhance the contrast
    between gray and white matter, and skull-stripping the result.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.util import init_enhance_and_skullstrip_bold_wf
        wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=1)


    Inputs

        in_file
            BOLD image (single volume)


    Outputs

        bias_corrected_file
            the ``in_file`` after `N4BiasFieldCorrection`_
        skull_stripped_file
            the ``bias_corrected_file`` after skull-stripping
        mask_file
            mask of the skull-stripped input file
        out_report
            reportlet for the skull-stripping

    .. _N4BiasFieldCorrection: https://hdl.handle.net/10380/3053
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['mask_file',
                                                       'skull_stripped_file',
                                                       'bias_corrected_file',
                                                       'out_report']),
                         name='outputnode')
    n4_correct = pe.Node(
        ants.N4BiasFieldCorrection(dimension=3, copy_header=True, num_threads=omp_nthreads),
        name='n4_correct', n_procs=omp_nthreads)
    skullstrip_first_pass = pe.Node(fsl.BET(frac=0.2, mask=True),
                                    name='skullstrip_first_pass')
    unifize = pe.Node(afni.Unifize(t2=True, outputtype='NIFTI_GZ',
                                   args='-clfrac 0.4',
                                   out_file="uni.nii.gz"), name='unifize')
    skullstrip_second_pass = pe.Node(afni.Automask(dilate=1,
                                                   outputtype='NIFTI_GZ'),
                                     name='skullstrip_second_pass')
    combine_masks = pe.Node(fsl.BinaryMaths(operation='mul'),
                            name='combine_masks')
    apply_mask = pe.Node(fsl.ApplyMask(),
                         name='apply_mask')
    copy_xform = pe.Node(CopyXForm(), name='copy_xform',
                         mem_gb=0.1, run_without_submitting=True)
    mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')

    workflow.connect([
        (inputnode, n4_correct, [('in_file', 'input_image')]),
        (inputnode, copy_xform, [('in_file', 'hdr_file')]),
        (n4_correct, skullstrip_first_pass, [('output_image', 'in_file')]),
        (skullstrip_first_pass, unifize, [('out_file', 'in_file')]),
        (unifize, skullstrip_second_pass, [('out_file', 'in_file')]),
        (skullstrip_first_pass, combine_masks, [('mask_file', 'in_file')]),
        (skullstrip_second_pass, combine_masks, [('out_file', 'operand_file')]),
        (unifize, apply_mask, [('out_file', 'in_file')]),
        (combine_masks, apply_mask, [('out_file', 'mask_file')]),
        (n4_correct, mask_reportlet, [('output_image', 'background_file')]),
        (combine_masks, mask_reportlet, [('out_file', 'mask_file')]),
        (combine_masks, outputnode, [('out_file', 'mask_file')]),
        (mask_reportlet, outputnode, [('out_report', 'out_report')]),
        (apply_mask, copy_xform, [('out_file', 'in_file')]),
        (copy_xform, outputnode, [('out_file', 'skull_stripped_file')]),
        (n4_correct, outputnode, [('output_image', 'bias_corrected_file')]),
    ])

    return workflow


def init_skullstrip_bold_wf(name='skullstrip_bold_wf'):
    """
    This workflow applies skull-stripping to a BOLD image.

    It is intended to be used on an image that has previously been
    bias-corrected with
    :py:func:`~fmriprep.workflows.util.init_enhance_and_skullstrip_bold_wf`

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.util import init_skullstrip_bold_wf
        wf = init_skullstrip_bold_wf()


    Inputs

        in_file
            BOLD image (single volume)


    Outputs

        skull_stripped_file
            the ``in_file`` after skull-stripping
        mask_file
            mask of the skull-stripped input file
        out_report
            reportlet for the skull-stripping

    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['mask_file',
                                                       'skull_stripped_file',
                                                       'out_report']),
                         name='outputnode')
    skullstrip_first_pass = pe.Node(fsl.BET(frac=0.2, mask=True),
                                    name='skullstrip_first_pass')
    skullstrip_second_pass = pe.Node(afni.Automask(dilate=1, outputtype='NIFTI_GZ'),
                                     name='skullstrip_second_pass')
    combine_masks = pe.Node(fsl.BinaryMaths(operation='mul'), name='combine_masks')
    apply_mask = pe.Node(fsl.ApplyMask(), name='apply_mask')
    mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')

    workflow.connect([
        (inputnode, skullstrip_first_pass, [('in_file', 'in_file')]),
        (skullstrip_first_pass, skullstrip_second_pass, [('out_file', 'in_file')]),
        (skullstrip_first_pass, combine_masks, [('mask_file', 'in_file')]),
        (skullstrip_second_pass, combine_masks, [('out_file', 'operand_file')]),
        (combine_masks, outputnode, [('out_file', 'mask_file')]),
        # Masked file
        (inputnode, apply_mask, [('in_file', 'in_file')]),
        (combine_masks, apply_mask, [('out_file', 'mask_file')]),
        (apply_mask, outputnode, [('out_file', 'skull_stripped_file')]),
        # Reportlet
        (inputnode, mask_reportlet, [('in_file', 'background_file')]),
        (combine_masks, mask_reportlet, [('out_file', 'mask_file')]),
        (mask_reportlet, outputnode, [('out_report', 'out_report')]),
        ])

    return workflow


def init_bbreg_wf(use_bbr, bold2t1w_dof, omp_nthreads, name='bbreg_wf'):
    """
    This workflow uses FreeSurfer's ``bbregister`` to register a BOLD image to
    a T1-weighted structural image.

    It is a counterpart to :py:func:`~fmriprep.workflows.util.init_fsl_bbr_wf`,
    which performs the same task using FSL's FLIRT with a BBR cost function.

    The ``use_bbr`` option permits a high degree of control over registration.
    If ``False``, standard, affine coregistration will be performed using
    FreeSurfer's ``mri_coreg`` tool.
    If ``True``, ``bbregister`` will be seeded with the initial transform found
    by ``mri_coreg`` (equivalent to running ``bbregister --init-coreg``).
    If ``None``, after ``bbregister`` is run, the resulting affine transform
    will be compared to the initial transform found by ``mri_coreg``.
    Excessive deviation will result in rejecting the BBR refinement and
    accepting the original, affine registration.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.util import init_bbreg_wf
        wf = init_bbreg_wf(use_bbr=True, bold2t1w_dof=9, omp_nthreads=1)


    Parameters

        use_bbr : bool or None
            Enable/disable boundary-based registration refinement.
            If ``None``, test BBR result for distortion before accepting.
        bold2t1w_dof : 6, 9 or 12
            Degrees-of-freedom for BOLD-T1w registration
        name : str, optional
            Workflow name (default: bbreg_wf)


    Inputs

        in_file
            Reference BOLD image to be registered
        t1_2_fsnative_reverse_transform
            FSL-style affine matrix translating from FreeSurfer T1.mgz to T1w
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID (must have folder in SUBJECTS_DIR)
        t1_brain
            Unused (see :py:func:`~fmriprep.workflows.util.init_fsl_bbr_wf`)
        t1_seg
            Unused (see :py:func:`~fmriprep.workflows.util.init_fsl_bbr_wf`)


    Outputs

        itk_bold_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        itk_t1_to_bold
            Affine transform from T1 space to BOLD space (ITK format)
        out_report
            Reportlet for assessing registration quality
        fallback
            Boolean indicating whether BBR was rejected (mri_coreg registration returned)

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface([
            'in_file',
            't1_2_fsnative_reverse_transform', 'subjects_dir', 'subject_id',  # BBRegister
            't1_seg', 't1_brain']),  # FLIRT BBR
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['itk_bold_to_t1', 'itk_t1_to_bold', 'out_report', 'fallback']),
        name='outputnode')

    mri_coreg = pe.Node(
        MRICoregRPT(dof=bold2t1w_dof, sep=[4], ftol=0.0001, linmintol=0.01,
                    num_threads=omp_nthreads, generate_report=not use_bbr),
        name='mri_coreg', n_procs=omp_nthreads)

    lta_concat = pe.Node(ConcatenateLTA(out_file='out.lta'), name='lta_concat')
    # XXX LTA-FSL-ITK may ultimately be able to be replaced with a straightforward
    # LTA-ITK transform, but right now the translation parameters are off.
    lta2fsl_fwd = pe.Node(fs.utils.LTAConvert(out_fsl=True), name='lta2fsl_fwd')
    lta2fsl_inv = pe.Node(fs.utils.LTAConvert(out_fsl=True, invert=True), name='lta2fsl_inv')
    fsl2itk_fwd = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_fwd', mem_gb=DEFAULT_MEMORY_MIN_GB)
    fsl2itk_inv = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_inv', mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, mri_coreg, [('subjects_dir', 'subjects_dir'),
                                ('subject_id', 'subject_id'),
                                ('in_file', 'source_file')]),
        # Output ITK transforms
        (inputnode, lta_concat, [('t1_2_fsnative_reverse_transform', 'in_lta2')]),
        (lta_concat, lta2fsl_fwd, [('out_file', 'in_lta')]),
        (lta_concat, lta2fsl_inv, [('out_file', 'in_lta')]),
        (inputnode, fsl2itk_fwd, [('t1_brain', 'reference_file'),
                                  ('in_file', 'source_file')]),
        (inputnode, fsl2itk_inv, [('in_file', 'reference_file'),
                                  ('t1_brain', 'source_file')]),
        (lta2fsl_fwd, fsl2itk_fwd, [('out_fsl', 'transform_file')]),
        (lta2fsl_inv, fsl2itk_inv, [('out_fsl', 'transform_file')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_bold_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_bold')]),
        ])

    # Short-circuit workflow building, use initial registration
    if use_bbr is False:
        workflow.connect([
            (mri_coreg, outputnode, [('out_report', 'out_report')]),
            (mri_coreg, lta_concat, [('out_lta_file', 'in_lta1')])])
        outputnode.inputs.fallback = True

        return workflow

    bbregister = pe.Node(
        BBRegisterRPT(dof=bold2t1w_dof, contrast_type='t2', registered_file=True,
                      out_lta_file=True, generate_report=True),
        name='bbregister')

    workflow.connect([
        (inputnode, bbregister, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id'),
                                 ('in_file', 'source_file')]),
        (mri_coreg, bbregister, [('out_lta_file', 'init_reg_file')]),
        ])

    # Short-circuit workflow building, use boundary-based registration
    if use_bbr is True:
        workflow.connect([
            (bbregister, outputnode, [('out_report', 'out_report')]),
            (bbregister, lta_concat, [('out_lta_file', 'in_lta1')])])
        outputnode.inputs.fallback = False

        return workflow

    transforms = pe.Node(niu.Merge(2), run_without_submitting=True, name='transforms')
    reports = pe.Node(niu.Merge(2), run_without_submitting=True, name='reports')

    lta_ras2ras = pe.MapNode(fs.utils.LTAConvert(out_lta=True), iterfield=['in_lta'],
                             name='lta_ras2ras')
    compare_transforms = pe.Node(niu.Function(function=compare_xforms), name='compare_transforms')

    select_transform = pe.Node(niu.Select(), run_without_submitting=True, name='select_transform')
    select_report = pe.Node(niu.Select(), run_without_submitting=True, name='select_report')

    workflow.connect([
        (bbregister, transforms, [('out_lta_file', 'in1')]),
        (mri_coreg, transforms, [('out_lta_file', 'in2')]),
        # Normalize LTA transforms to RAS2RAS (inputs are VOX2VOX) and compare
        (transforms, lta_ras2ras, [('out', 'in_lta')]),
        (lta_ras2ras, compare_transforms, [('out_lta', 'lta_list')]),
        (compare_transforms, outputnode, [('out', 'fallback')]),
        # Select output transform
        (transforms, select_transform, [('out', 'inlist')]),
        (compare_transforms, select_transform, [('out', 'index')]),
        (select_transform, lta_concat, [('out', 'in_lta1')]),
        # Select output report
        (bbregister, reports, [('out_report', 'in1')]),
        (mri_coreg, reports, [('out_report', 'in2')]),
        (reports, select_report, [('out', 'inlist')]),
        (compare_transforms, select_report, [('out', 'index')]),
        (select_report, outputnode, [('out', 'out_report')]),
        ])

    return workflow


def init_fsl_bbr_wf(use_bbr, bold2t1w_dof, name='fsl_bbr_wf'):
    """
    This workflow uses FSL FLIRT to register a BOLD image to a T1-weighted
    structural image, using a boundary-based registration (BBR) cost function.

    It is a counterpart to :py:func:`~fmriprep.workflows.util.init_bbreg_wf`,
    which performs the same task using FreeSurfer's ``bbregister``.

    The ``use_bbr`` option permits a high degree of control over registration.
    If ``False``, standard, rigid coregistration will be performed by FLIRT.
    If ``True``, FLIRT-BBR will be seeded with the initial transform found by
    the rigid coregistration.
    If ``None``, after FLIRT-BBR is run, the resulting affine transform
    will be compared to the initial transform found by FLIRT.
    Excessive deviation will result in rejecting the BBR refinement and
    accepting the original, affine registration.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.util import init_fsl_bbr_wf
        wf = init_fsl_bbr_wf(use_bbr=True, bold2t1w_dof=9)


    Parameters

        use_bbr : bool or None
            Enable/disable boundary-based registration refinement.
            If ``None``, test BBR result for distortion before accepting.
        bold2t1w_dof : 6, 9 or 12
            Degrees-of-freedom for BOLD-T1w registration
        name : str, optional
            Workflow name (default: fsl_bbr_wf)


    Inputs

        in_file
            Reference BOLD image to be registered
        t1_brain
            Skull-stripped T1-weighted structural image
        t1_seg
            FAST segmentation of ``t1_brain``
        t1_2_fsnative_reverse_transform
            Unused (see :py:func:`~fmriprep.workflows.util.init_bbreg_wf`)
        subjects_dir
            Unused (see :py:func:`~fmriprep.workflows.util.init_bbreg_wf`)
        subject_id
            Unused (see :py:func:`~fmriprep.workflows.util.init_bbreg_wf`)


    Outputs

        itk_bold_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        itk_t1_to_bold
            Affine transform from T1 space to BOLD space (ITK format)
        out_report
            Reportlet for assessing registration quality
        fallback
            Boolean indicating whether BBR was rejected (rigid FLIRT registration returned)

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface([
            'in_file',
            't1_2_fsnative_reverse_transform', 'subjects_dir', 'subject_id',  # BBRegister
            't1_seg', 't1_brain']),  # FLIRT BBR
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['itk_bold_to_t1', 'itk_t1_to_bold', 'out_report', 'fallback']),
        name='outputnode')

    wm_mask = pe.Node(niu.Function(function=extract_wm), name='wm_mask')
    flt_bbr_init = pe.Node(FLIRTRPT(dof=6, generate_report=not use_bbr), name='flt_bbr_init')

    invt_bbr = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='invt_bbr',
                       mem_gb=DEFAULT_MEMORY_MIN_GB)

    #  BOLD to T1 transform matrix is from fsl, using c3 tools to convert to
    #  something ANTs will like.
    fsl2itk_fwd = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_fwd', mem_gb=DEFAULT_MEMORY_MIN_GB)
    fsl2itk_inv = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_inv', mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, flt_bbr_init, [('in_file', 'in_file'),
                                   ('t1_brain', 'reference')]),
        (inputnode, fsl2itk_fwd, [('t1_brain', 'reference_file'),
                                  ('in_file', 'source_file')]),
        (inputnode, fsl2itk_inv, [('in_file', 'reference_file'),
                                  ('t1_brain', 'source_file')]),
        (invt_bbr, fsl2itk_inv, [('out_file', 'transform_file')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_bold_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_bold')]),
        ])

    # Short-circuit workflow building, use rigid registration
    if use_bbr is False:
        workflow.connect([
            (flt_bbr_init, invt_bbr, [('out_matrix_file', 'in_file')]),
            (flt_bbr_init, fsl2itk_fwd, [('out_matrix_file', 'transform_file')]),
            (flt_bbr_init, outputnode, [('out_report', 'out_report')]),
            ])
        outputnode.inputs.fallback = True

        return workflow

    flt_bbr = pe.Node(
        FLIRTRPT(cost_func='bbr', dof=bold2t1w_dof, generate_report=True,
                 schedule=op.join(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch')),
        name='flt_bbr')

    workflow.connect([
        (inputnode, wm_mask, [('t1_seg', 'in_seg')]),
        (inputnode, flt_bbr, [('in_file', 'in_file'),
                              ('t1_brain', 'reference')]),
        (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (wm_mask, flt_bbr, [('out', 'wm_seg')]),
        ])

    # Short-circuit workflow building, use boundary-based registration
    if use_bbr is True:
        workflow.connect([
            (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
            (flt_bbr, fsl2itk_fwd, [('out_matrix_file', 'transform_file')]),
            (flt_bbr, outputnode, [('out_report', 'out_report')]),
            ])
        outputnode.inputs.fallback = False

        return workflow

    transforms = pe.Node(niu.Merge(2), run_without_submitting=True, name='transforms')
    reports = pe.Node(niu.Merge(2), run_without_submitting=True, name='reports')

    compare_transforms = pe.Node(niu.Function(function=compare_xforms), name='compare_transforms')

    select_transform = pe.Node(niu.Select(), run_without_submitting=True, name='select_transform')
    select_report = pe.Node(niu.Select(), run_without_submitting=True, name='select_report')

    fsl_to_lta = pe.MapNode(fs.utils.LTAConvert(out_lta=True), iterfield=['in_fsl'],
                            name='fsl_to_lta')

    workflow.connect([
        (flt_bbr, transforms, [('out_matrix_file', 'in1')]),
        (flt_bbr_init, transforms, [('out_matrix_file', 'in2')]),
        # Convert FSL transforms to LTA (RAS2RAS) transforms and compare
        (inputnode, fsl_to_lta, [('in_file', 'source_file'),
                                 ('t1_brain', 'target_file')]),
        (transforms, fsl_to_lta, [('out', 'in_fsl')]),
        (fsl_to_lta, compare_transforms, [('out_lta', 'lta_list')]),
        (compare_transforms, outputnode, [('out', 'fallback')]),
        # Select output transform
        (transforms, select_transform, [('out', 'inlist')]),
        (compare_transforms, select_transform, [('out', 'index')]),
        (select_transform, invt_bbr, [('out', 'in_file')]),
        (select_transform, fsl2itk_fwd, [('out', 'transform_file')]),
        (flt_bbr, reports, [('out_report', 'in1')]),
        (flt_bbr_init, reports, [('out_report', 'in2')]),
        (reports, select_report, [('out', 'inlist')]),
        (compare_transforms, select_report, [('out', 'index')]),
        (select_report, outputnode, [('out', 'out_report')]),
        ])

    return workflow
