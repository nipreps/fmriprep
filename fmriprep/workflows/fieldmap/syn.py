# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fieldmap-less SDC
+++++++++++++++++

.. autofunction:: init_nonlinear_sdc_wf

"""
import pkg_resources as pkgr

from niworkflows.nipype import logging
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import fsl, utility as niu
from niworkflows.interfaces import SimpleBeforeAfter
from niworkflows.interfaces.fixes import (FixHeaderApplyTransforms as ApplyTransforms,
                                          FixHeaderRegistration as Registration)
from ...interfaces import InvertT1w
from ...interfaces.images import extract_wm
from ..bold.util import init_skullstrip_bold_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('workflow')


def init_nonlinear_sdc_wf(bold_file, freesurfer, bold2t1w_dof,
                          template, omp_nthreads, bold_pe='j',
                          atlas_threshold=3, name='nonlinear_sdc_wf'):
    """
    This workflow takes a skull-stripped T1w image and reference BOLD image and
    estimates a susceptibility distortion correction warp, using ANTs symmetric
    normalization (SyN) and the average fieldmap atlas described in
    [Treiber2016]_.

    SyN deformation is restricted to the phase-encoding (PE) direction.
    If no PE direction is specified, anterior-posterior PE is assumed.

    SyN deformation is also restricted to regions that are expected to have a
    >3mm (approximately 1 voxel) warp, based on the fieldmap atlas.

    This technique is a variation on those developed in [Huntenburg2014]_ and
    [Wang2017]_.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.syn import init_nonlinear_sdc_wf
        wf = init_nonlinear_sdc_wf(
            bold_file='/dataset/sub-01/func/sub-01_task-rest_bold.nii.gz',
            bold_pe='j',
            freesurfer=True,
            bold2t1w_dof=9,
            template='MNI152NLin2009cAsym',
            omp_nthreads=8)

    **Inputs**

        t1_brain
            skull-stripped, bias-corrected structural image
        bold_ref
            skull-stripped reference image
        t1_seg
            FAST segmentation white and gray matter, in native T1w space
        t1_2_mni_reverse_transform
            inverse registration transform of T1w image to MNI template

    **Outputs**

        out_reference_brain
            the ``bold_ref`` image after unwarping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_mask
            mask of the unwarped input file
        out_mask_report
            reportlet for the skullstripping

    .. [Huntenburg2014] Huntenburg, J. M. (2014) Evaluating Nonlinear
                        Coregistration of BOLD EPI and T1w Images. Berlin: Master
                        Thesis, Freie Universität. `PDF
                        <http://pubman.mpdl.mpg.de/pubman/item/escidoc:2327525:5/component/escidoc:2327523/master_thesis_huntenburg_4686947.pdf>`_.
    .. [Treiber2016] Treiber, J. M. et al. (2016) Characterization and Correction
                     of Geometric Distortions in 814 Diffusion Weighted Images,
                     PLoS ONE 11(3): e0152472. doi:`10.1371/journal.pone.0152472
                     <https://doi.org/10.1371/journal.pone.0152472>`_.
    .. [Wang2017] Wang S, et al. (2017) Evaluation of Field Map and Nonlinear
                  Registration Methods for Correction of Susceptibility Artifacts
                  in Diffusion MRI. Front. Neuroinform. 11:17.
                  doi:`10.3389/fninf.2017.00017
                  <https://doi.org/10.3389/fninf.2017.00017>`_.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(['t1_brain', 'bold_ref', 't1_2_mni_reverse_transform',
                               't1_seg']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['out_reference_brain', 'out_mask', 'out_warp',
                               'out_warp_report', 'out_mask_report']),
        name='outputnode')

    if bold_pe is None or bold_pe[0] not in ['i', 'j']:
        LOGGER.warning('Incorrect phase-encoding direction, assuming PA (posterior-to-anterior')
        bold_pe = 'j'

    # Collect predefined data
    # Atlas image and registration affine
    atlas_img = pkgr.resource_filename('fmriprep', 'data/fmap_atlas.nii.gz')
    atlas_2_template_affine = pkgr.resource_filename(
        'fmriprep', 'data/fmap_atlas_2_{}_affine.mat'.format(template))
    # Registration specifications
    affine_transform = pkgr.resource_filename('fmriprep', 'data/affine.json')
    syn_transform = pkgr.resource_filename('fmriprep', 'data/susceptibility_syn.json')

    invert_t1w = pe.Node(InvertT1w(), name='invert_t1w',
                         mem_gb=0.3)

    ref_2_t1 = pe.Node(Registration(from_file=affine_transform),
                       name='ref_2_t1', n_procs=omp_nthreads)
    t1_2_ref = pe.Node(ApplyTransforms(invert_transform_flags=[True]),
                       name='t1_2_ref', n_procs=omp_nthreads)

    # 1) BOLD -> T1; 2) MNI -> T1; 3) ATLAS -> MNI
    transform_list = pe.Node(niu.Merge(3), name='transform_list',
                             mem_gb=DEFAULT_MEMORY_MIN_GB)
    transform_list.inputs.in3 = atlas_2_template_affine

    # Inverting (1), then applying in reverse order:
    #
    # ATLAS -> MNI -> T1 -> BOLD
    atlas_2_ref = pe.Node(
        ApplyTransforms(invert_transform_flags=[True, False, False]),
        name='atlas_2_ref', n_procs=omp_nthreads,
        mem_gb=0.3)
    atlas_2_ref.inputs.input_image = atlas_img

    threshold_atlas = pe.Node(
        fsl.maths.MathsCommand(args='-thr {:.8g} -bin'.format(atlas_threshold),
                               output_datatype='char'),
        name='threshold_atlas', mem_gb=0.3)

    fixed_image_masks = pe.Node(niu.Merge(2), name='fixed_image_masks',
                                mem_gb=DEFAULT_MEMORY_MIN_GB)
    fixed_image_masks.inputs.in1 = 'NULL'

    restrict = [[int(bold_pe[0] == 'i'), int(bold_pe[0] == 'j'), 0]] * 2
    syn = pe.Node(
        Registration(from_file=syn_transform, restrict_deformation=restrict),
        name='syn', n_procs=omp_nthreads)

    seg_2_ref = pe.Node(
        ApplyTransforms(interpolation='NearestNeighbor', float=True,
                        invert_transform_flags=[True]),
        name='seg_2_ref', n_procs=omp_nthreads, mem_gb=0.3)
    sel_wm = pe.Node(niu.Function(function=extract_wm), name='sel_wm',
                     mem_gb=DEFAULT_MEMORY_MIN_GB)
    syn_rpt = pe.Node(SimpleBeforeAfter(), name='syn_rpt',
                      mem_gb=0.1)

    skullstrip_bold_wf = init_skullstrip_bold_wf()

    workflow.connect([
        (inputnode, invert_t1w, [('t1_brain', 'in_file'),
                                 ('bold_ref', 'ref_file')]),
        (inputnode, ref_2_t1, [('bold_ref', 'moving_image')]),
        (invert_t1w, ref_2_t1, [('out_file', 'fixed_image')]),
        (inputnode, t1_2_ref, [('bold_ref', 'reference_image')]),
        (invert_t1w, t1_2_ref, [('out_file', 'input_image')]),
        (ref_2_t1, t1_2_ref, [('forward_transforms', 'transforms')]),
        (ref_2_t1, transform_list, [('forward_transforms', 'in1')]),
        (inputnode, transform_list, [('t1_2_mni_reverse_transform', 'in2')]),
        (inputnode, atlas_2_ref, [('bold_ref', 'reference_image')]),
        (transform_list, atlas_2_ref, [('out', 'transforms')]),
        (atlas_2_ref, threshold_atlas, [('output_image', 'in_file')]),
        (threshold_atlas, fixed_image_masks, [('out_file', 'in2')]),
        (inputnode, syn, [('bold_ref', 'moving_image')]),
        (t1_2_ref, syn, [('output_image', 'fixed_image')]),
        (fixed_image_masks, syn, [('out', 'fixed_image_masks')]),
        (inputnode, seg_2_ref, [('t1_seg', 'input_image')]),
        (ref_2_t1, seg_2_ref, [('forward_transforms', 'transforms')]),
        (syn, seg_2_ref, [('warped_image', 'reference_image')]),
        (seg_2_ref, sel_wm, [('output_image', 'in_seg')]),
        (inputnode, syn_rpt, [('bold_ref', 'before')]),
        (syn, syn_rpt, [('warped_image', 'after')]),
        (sel_wm, syn_rpt, [('out', 'wm_seg')]),
        (syn, skullstrip_bold_wf, [('warped_image', 'inputnode.in_file')]),
        (syn, outputnode, [('forward_transforms', 'out_warp')]),
        (skullstrip_bold_wf, outputnode, [
            ('outputnode.skull_stripped_file', 'out_reference_brain'),
            ('outputnode.mask_file', 'out_mask'),
            ('outputnode.out_report', 'out_mask_report')]),
        (syn_rpt, outputnode, [('out_report', 'out_warp_report')])])

    return workflow
