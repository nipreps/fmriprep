# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_reference_wf
.. autofunction:: init_enhance_and_skullstrip_bold_wf
.. autofunction:: init_skullstrip_bold_wf

"""
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import utility as niu, fsl, afni, ants
from niworkflows.interfaces.utils import CopyXForm
from niworkflows.interfaces.masks import SimpleShowMaskRPT
from niworkflows.interfaces.registration import EstimateReferenceImage

from ...interfaces import ValidateImage


DEFAULT_MEMORY_MIN_GB = 0.01


def init_bold_reference_wf(omp_nthreads, bold_file=None, name='bold_reference_wf'):
    """
    This workflow generates reference BOLD images for a series

    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_reference_wf
        wf = init_bold_reference_wf(omp_nthreads=1)

    **Parameters**

        bold_file : str
            BOLD series NIfTI file
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_reference_wf``)

    **Inputs**

        bold_file
            BOLD series NIfTI file

    **Outputs**

        bold_file
            Validated BOLD series NIfTI file
        raw_ref_image
            Reference image to which BOLD series is motion corrected
        skip_vols
            Number of non-steady-state volumes detected at beginning of ``bold_file``
        ref_image
            Contrast-enhanced reference image
        ref_image_brain
            Skull-stripped reference image
        bold_mask
            Skull-stripping mask of reference image
        bold_mask_report
            Reportlet showing quality of masking
        validation_report
            HTML reportlet indicating whether ``bold_file`` had a valid affine


    **Subworkflows**

        * :py:func:`~fmriprep.workflows.bold.util.init_enhance_and_skullstrip_wf`

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'raw_ref_image', 'skip_vols', 'ref_image',
                                      'ref_image_brain', 'bold_mask', 'bold_mask_report',
                                      'validation_report']),
        name='outputnode')

    # Simplify manually setting input image
    if bold_file is not None:
        inputnode.inputs.bold_file = bold_file

    validate = pe.Node(ValidateImage(), name='validate', mem_gb=DEFAULT_MEMORY_MIN_GB,
                       run_without_submitting=True)

    gen_ref = pe.Node(EstimateReferenceImage(), name="gen_ref",
                      mem_gb=1)  # OE: 128x128x128x50 * 64 / 8 ~ 900MB.
    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode, validate, [('bold_file', 'in_file')]),
        (validate, gen_ref, [('out_file', 'in_file')]),
        (gen_ref, enhance_and_skullstrip_bold_wf, [('ref_image', 'inputnode.in_file')]),
        (validate, outputnode, [('out_file', 'bold_file'),
                                ('out_report', 'validation_report')]),
        (gen_ref, outputnode, [('ref_image', 'raw_ref_image'),
                               ('n_volumes_to_discard', 'skip_vols')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.bias_corrected_file', 'ref_image'),
            ('outputnode.mask_file', 'bold_mask'),
            ('outputnode.out_report', 'bold_mask_report'),
            ('outputnode.skull_stripped_file', 'ref_image_brain')]),
    ])

    return workflow


def init_enhance_and_skullstrip_bold_wf(name='enhance_and_skullstrip_bold_wf',
                                        omp_nthreads=1):
    """
    This workflow takes in a BOLD volume, and attempts to enhance the contrast
    between gray and white matter, and skull-stripping the result.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold.util import init_enhance_and_skullstrip_bold_wf
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
    n4_correct = pe.Node(ants.N4BiasFieldCorrection(dimension=3, copy_header=True),
                         name='n4_correct', n_procs=omp_nthreads)
    skullstrip_first_pass = pe.Node(fsl.BET(frac=0.2, mask=True),
                                    name='skullstrip_first_pass')
    unifize = pe.Node(afni.Unifize(t2=True, outputtype='NIFTI_GZ',
                                   args='-clfrac 0.4',
                                   out_file="uni.nii.gz"), name='unifize')
    fixhdr_unifize = pe.Node(CopyXForm(), name='fixhdr_unifize',
                             mem_gb=0.1, run_without_submitting=True)
    skullstrip_second_pass = pe.Node(afni.Automask(dilate=1,
                                                   outputtype='NIFTI_GZ'),
                                     name='skullstrip_second_pass')
    fixhdr_skullstrip2 = pe.Node(CopyXForm(), name='fixhdr_skullstrip2',
                                 mem_gb=0.1, run_without_submitting=True)
    combine_masks = pe.Node(fsl.BinaryMaths(operation='mul'),
                            name='combine_masks')
    apply_mask = pe.Node(fsl.ApplyMask(),
                         name='apply_mask')
    mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')

    workflow.connect([
        (inputnode, n4_correct, [('in_file', 'input_image')]),
        (inputnode, fixhdr_unifize, [('in_file', 'hdr_file')]),
        (inputnode, fixhdr_skullstrip2, [('in_file', 'hdr_file')]),
        (n4_correct, skullstrip_first_pass, [('output_image', 'in_file')]),
        (skullstrip_first_pass, unifize, [('out_file', 'in_file')]),
        (unifize, fixhdr_unifize, [('out_file', 'in_file')]),
        (fixhdr_unifize, skullstrip_second_pass, [('out_file', 'in_file')]),
        (skullstrip_first_pass, combine_masks, [('mask_file', 'in_file')]),
        (skullstrip_second_pass, fixhdr_skullstrip2, [('out_file', 'in_file')]),
        (fixhdr_skullstrip2, combine_masks, [('out_file', 'operand_file')]),
        (fixhdr_unifize, apply_mask, [('out_file', 'in_file')]),
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
    :py:func:`~fmriprep.workflows.bold.util.init_enhance_and_skullstrip_bold_wf`

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold.util import init_skullstrip_bold_wf
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
