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
from niworkflows.nipype.interfaces import fsl, afni, c3, ants, freesurfer as fs
from niworkflows.interfaces.registration import FLIRTRPT, BBRegisterRPT
from niworkflows.interfaces.masks import SimpleShowMaskRPT

from ..interfaces.images import extract_wm

DEFAULT_MEMORY_MIN_GB = 0.01


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
    mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')

    workflow.connect([
        (inputnode, n4_correct, [('in_file', 'input_image')]),
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
        (apply_mask, outputnode, [('out_file', 'skull_stripped_file')]),
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


def init_bbreg_wf(bold2t1w_dof, name='bbreg_wf'):
    """
    This workflow uses FreeSurfer's ``bbregister`` to register a BOLD image to
    a T1-weighted structural image.

    It is a counterpart to :py:func:`~fmriprep.workflows.util.init_fsl_bbr_wf`,
    which performs the same task using FSL's FLIRT with a BBR cost function.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.util import init_bbreg_wf
        wf = init_bbreg_wf(bold2t1w_dof=9)


    Parameters

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
            reportlet for assessing registration quality

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface([
            'in_file',
            't1_2_fsnative_reverse_transform', 'subjects_dir', 'subject_id',  # BBRegister
            't1_seg', 't1_brain']),  # FLIRT BBR
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['itk_bold_to_t1', 'itk_t1_to_bold', 'out_report']),
        name='outputnode')

    bbregister = pe.Node(
        BBRegisterRPT(dof=bold2t1w_dof, contrast_type='t2', init='coreg',
                      registered_file=True, out_lta_file=True, generate_report=True),
        name='bbregister')

    lta_concat = pe.Node(fs.ConcatenateLTA(out_file='out.lta'), name='lta_concat')
    lta2itk_fwd = pe.Node(fs.utils.LTAConvert(out_itk=True), name='lta2itk_fwd')
    lta2itk_inv = pe.Node(fs.utils.LTAConvert(out_itk=True, invert=True), name='lta2itk_inv')

    workflow.connect([
        (inputnode, bbregister, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id'),
                                 ('in_file', 'source_file')]),
        (bbregister, outputnode, [('out_report', 'out_report')]),
        (lta2itk_fwd, outputnode, [('out_itk', 'itk_bold_to_t1')]),
        (lta2itk_inv, outputnode, [('out_itk', 'itk_t1_to_bold')]),
        (inputnode, lta_concat, [('t1_2_fsnative_reverse_transform', 'in_lta2')]),
        (bbregister, lta_concat, [('out_lta_file', 'in_lta1')]),
        (lta_concat, lta2itk_fwd, [('out_file', 'in_lta')]),
        (lta_concat, lta2itk_inv, [('out_file', 'in_lta')]),
        ])

    return workflow


def init_fsl_bbr_wf(bold2t1w_dof, name='fsl_bbr_wf'):
    """
    This workflow uses FSL FLIRT to register a BOLD image to a T1-weighted
    structural image, using a boundary-based registration (BBR) cost function.

    It is a counterpart to :py:func:`~fmriprep.workflows.util.init_bbreg_wf`,
    which performs the same task using FreeSurfer's ``bbregister``.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.util import init_fsl_bbr_wf
        wf = init_fsl_bbr_wf(bold2t1w_dof=9)


    Parameters

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
            reportlet for assessing registration quality

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface([
            'in_file',
            't1_2_fsnative_reverse_transform', 'subjects_dir', 'subject_id',  # BBRegister
            't1_seg', 't1_brain']),  # FLIRT BBR
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['itk_bold_to_t1', 'itk_t1_to_bold', 'out_report']),
        name='outputnode')

    wm_mask = pe.Node(niu.Function(function=extract_wm), name='wm_mask')
    flt_bbr_init = pe.Node(FLIRTRPT(dof=6, generate_report=False), name='flt_bbr_init')
    flt_bbr = pe.Node(FLIRTRPT(cost_func='bbr', dof=bold2t1w_dof, save_log=True,
                               generate_report=True), name='flt_bbr')
    flt_bbr.inputs.schedule = op.join(os.getenv('FSLDIR'),
                                      'etc/flirtsch/bbr.sch')

    # make equivalent warp fields
    invt_bbr = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='invt_bbr',
                       mem_gb=DEFAULT_MEMORY_MIN_GB)

    #  BOLD to T1 transform matrix is from fsl, using c3 tools to convert to
    #  something ANTs will like.
    fsl2itk_fwd = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_fwd', mem_gb=DEFAULT_MEMORY_MIN_GB)
    fsl2itk_inv = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_inv', mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, wm_mask, [('t1_seg', 'in_seg')]),
        (inputnode, flt_bbr_init, [('in_file', 'in_file'),
                                   ('t1_brain', 'reference')]),
        (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (inputnode, flt_bbr, [('in_file', 'in_file'),
                              ('t1_brain', 'reference')]),
        (wm_mask, flt_bbr, [('out', 'wm_seg')]),
        (inputnode, fsl2itk_fwd, [('t1_brain', 'reference_file'),
                                  ('in_file', 'source_file')]),
        (inputnode, fsl2itk_inv, [('in_file', 'reference_file'),
                                  ('t1_brain', 'source_file')]),
        (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
        (flt_bbr, fsl2itk_fwd, [('out_matrix_file', 'transform_file')]),
        (invt_bbr, fsl2itk_inv, [('out_file', 'transform_file')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_bold_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_bold')]),
        (flt_bbr, outputnode, [('out_report', 'out_report')]),
        ])

    return workflow
