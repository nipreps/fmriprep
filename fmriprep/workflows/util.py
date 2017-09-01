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
from niworkflows.nipype.interfaces import fsl, afni, ants, freesurfer as fs
from niworkflows.interfaces.registration import FLIRTRPT, BBRegisterRPT
from niworkflows.interfaces.masks import SimpleShowMaskRPT

from ..interfaces.images import extract_wm


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


def init_bbreg_wf(bold2t1w_dof, report, reregister=True, name='bbreg_wf'):
    """
    This workflow uses FreeSurfer's ``bbregister`` to register a BOLD image to
    a T1-weighted structural image.

    It is a counterpart to :py:func:`~fmriprep.workflows.util.init_fsl_bbr_wf`,
    which performs the same task using FSL's FLIRT with a BBR cost function.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.util import init_bbreg_wf
        wf = init_bbreg_wf(bold2t1w_dof=9, report=False)


    Parameters

        bold2t1w_dof : 6, 9 or 12
            Degrees-of-freedom for BOLD-T1w registration
        report : bool
            Generate visual report of registration quality
        rereigster : bool, optional
            Update affine registration matrix with FreeSurfer-T1w transform
            (default: True)
        name : str, optional
            Workflow name (default: bbreg_wf)


    Inputs

        in_file
            Reference BOLD image to be registered
        fs_2_t1_transform
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

        out_matrix_file
            FSL-style registration matrix
        out_reg_file
            FreeSurfer-style registration matrix (.dat)
        final_cost
            Value of cost function at final registration
        out_report
            reportlet for assessing registration quality

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(['in_file',
                               'fs_2_t1_transform', 'subjects_dir', 'subject_id',  # BBRegister
                               't1_seg', 't1_brain']),  # FLIRT BBR
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['out_matrix_file', 'out_reg_file', 'out_report', 'final_cost']),
        name='outputnode')

    _BBRegister = BBRegisterRPT if report else fs.BBRegister
    bbregister = pe.Node(
        _BBRegister(dof=bold2t1w_dof, contrast_type='t2', init='coreg',
                    registered_file=True, out_fsl_file=True),
        name='bbregister')

    def apply_fs_transform(fs_2_t1_transform, bbreg_transform):
        import os
        import numpy as np
        out_file = os.path.abspath('transform.mat')
        fs_xfm = np.loadtxt(fs_2_t1_transform)
        bbrxfm = np.loadtxt(bbreg_transform)
        out_xfm = fs_xfm.dot(bbrxfm)
        assert np.allclose(out_xfm[3], [0, 0, 0, 1])
        out_xfm[3] = [0, 0, 0, 1]
        np.savetxt(out_file, out_xfm, fmt=str('%.12g'))
        return out_file

    transformer = pe.Node(niu.Function(function=apply_fs_transform),
                          name='transformer')

    def get_final_cost(in_file):
        import numpy as np
        return np.loadtxt(in_file, usecols=[0])

    get_cost = pe.Node(niu.Function(function=get_final_cost),
                       name='get_cost')

    workflow.connect([
        (inputnode, bbregister, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id'),
                                 ('in_file', 'source_file')]),
        (bbregister, get_cost, [('min_cost_file', 'in_file')]),
        (bbregister, outputnode, [('out_reg_file', 'out_reg_file')]),
        (get_cost, outputnode, [('out', 'final_cost')]),
        ])

    if reregister:
        workflow.connect([
            (inputnode, transformer, [('fs_2_t1_transform', 'fs_2_t1_transform')]),
            (bbregister, transformer, [('out_fsl_file', 'bbreg_transform')]),
            (transformer, outputnode, [('out', 'out_matrix_file')]),
            ])
    else:
        workflow.connect([
            (bbregister, outputnode, [('out_fsl_file', 'out_matrix_file')]),
            ])

    if report:
        bbregister.inputs.generate_report = True
        workflow.connect([(bbregister, outputnode, [('out_report', 'out_report')])])

    return workflow


def init_fsl_bbr_wf(bold2t1w_dof, report, name='fsl_bbr_wf'):
    """
    This workflow uses FSL FLIRT to register a BOLD image to a T1-weighted
    structural image, using a boundary-based registration (BBR) cost function.

    It is a counterpart to :py:func:`~fmriprep.workflows.util.init_bbreg_wf`,
    which performs the same task using FreeSurfer's ``bbregister``.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.util import init_fsl_bbr_wf
        wf = init_fsl_bbr_wf(bold2t1w_dof=9, report=False)


    Parameters

        bold2t1w_dof : 6, 9 or 12
            Degrees-of-freedom for BOLD-T1w registration
        report : bool
            Generate visual report of registration quality
        name : str, optional
            Workflow name (default: fsl_bbr_wf)


    Inputs

        in_file
            Reference BOLD image to be registered
        t1_brain
            Skull-stripped T1-weighted structural image
        t1_seg
            FAST segmentation of ``t1_brain``
        fs_2_t1_transform
            Unused (see :py:func:`~fmriprep.workflows.util.init_bbreg_wf`)
        subjects_dir
            Unused (see :py:func:`~fmriprep.workflows.util.init_bbreg_wf`)
        subject_id
            Unused (see :py:func:`~fmriprep.workflows.util.init_bbreg_wf`)


    Outputs

        out_matrix_file
            FSL-style registration matrix
        out_reg_file
            Unused (see :py:func:`~fmriprep.workflows.util.init_bbreg_wf`)
        final_cost
            Value of cost function at final registration
        out_report
            reportlet for assessing registration quality

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(['in_file',
                               'fs_2_t1_transform', 'subjects_dir', 'subject_id',  # BBRegister
                               't1_seg', 't1_brain']),  # FLIRT BBR
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['out_matrix_file', 'out_reg_file', 'out_report', 'final_cost']),
        name='outputnode')

    wm_mask = pe.Node(niu.Function(function=extract_wm), name='wm_mask')
    _FLIRT = FLIRTRPT if report else fsl.FLIRT
    flt_bbr_init = pe.Node(fsl.FLIRT(dof=6), name='flt_bbr_init')
    flt_bbr = pe.Node(_FLIRT(cost_func='bbr', dof=bold2t1w_dof, save_log=True), name='flt_bbr')
    flt_bbr.inputs.schedule = op.join(os.getenv('FSLDIR'),
                                      'etc/flirtsch/bbr.sch')

    def get_final_cost(in_file):
        from niworkflows.nipype import logging
        with open(in_file, 'r') as fobj:
            for line in fobj:
                if line.startswith(' >> print U:1'):
                    costs = next(fobj).split()
                    return float(costs[0])
        logger = logging.getLogger('interface')
        logger.error('No cost report found in log file. Please report this '
                     'issue, with contents of {}'.format(in_file))

    get_cost = pe.Node(niu.Function(function=get_final_cost),
                       name='get_cost')

    workflow.connect([
        (inputnode, wm_mask, [('t1_seg', 'in_seg')]),
        (inputnode, flt_bbr_init, [('in_file', 'in_file'),
                                   ('t1_brain', 'reference')]),
        (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (inputnode, flt_bbr, [('in_file', 'in_file'),
                              ('t1_brain', 'reference')]),
        (wm_mask, flt_bbr, [('out', 'wm_seg')]),
        (flt_bbr, outputnode, [('out_matrix_file', 'out_matrix_file')]),
        (flt_bbr, get_cost, [('out_log', 'in_file')]),
        (get_cost, outputnode, [('out', 'final_cost')]),
        ])

    if report:
        flt_bbr.inputs.generate_report = True
        workflow.connect([(flt_bbr, outputnode, [('out_report', 'out_report')])])

    return workflow
