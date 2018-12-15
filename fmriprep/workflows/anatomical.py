# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Anatomical reference preprocessing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_anat_preproc_wf
.. autofunction:: init_skullstrip_ants_wf

Surface preprocessing
+++++++++++++++++++++

``fmriprep`` uses FreeSurfer_ to reconstruct surfaces from T1w/T2w
structural images.

.. autofunction:: init_surface_recon_wf
.. autofunction:: init_autorecon_resume_wf
.. autofunction:: init_gifti_surface_wf

"""

from pkg_resources import resource_filename as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import (
    io as nio,
    utility as niu,
    freesurfer as fs,
    fsl,
    image,
)
from nipype.interfaces.ants import BrainExtraction, N4BiasFieldCorrection

from niworkflows.interfaces.registration import RobustMNINormalizationRPT
import niworkflows.data as nid
from niworkflows.interfaces.masks import ROIsPlot

from niworkflows.interfaces.segmentation import ReconAllRPT
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

from ..engine import Workflow
from ..interfaces import (
    DerivativesDataSink, StructuralReference, MakeMidthickness, FSInjectBrainExtracted,
    FSDetectInputs, NormalizeSurf, GiftiNameSource, TemplateDimensions, Conform,
    RefineBrainMask,
)
from ..utils.misc import fix_multi_T1w_source_name, add_suffix
from ..interfaces.freesurfer import (
    PatchedLTAConvert as LTAConvert,
    PatchedConcatenateLTA as ConcatenateLTA,
    PatchedRobustRegister as RobustRegister)


#  pylint: disable=R0914
def init_anat_preproc_wf(skull_strip_template, output_spaces, template, debug,
                         freesurfer, longitudinal, omp_nthreads, hires, reportlets_dir,
                         output_dir, num_t1w,
                         skull_strip_fixed_seed=False, name='anat_preproc_wf'):
    r"""
    This workflow controls the anatomical preprocessing stages of FMRIPREP.

    This includes:

     - Creation of a structural template
     - Skull-stripping and bias correction
     - Tissue segmentation
     - Normalization
     - Surface reconstruction with FreeSurfer

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_anat_preproc_wf
        wf = init_anat_preproc_wf(omp_nthreads=1,
                                  reportlets_dir='.',
                                  output_dir='.',
                                  template='MNI152NLin2009cAsym',
                                  output_spaces=['T1w', 'fsnative',
                                                 'template', 'fsaverage5'],
                                  skull_strip_template='OASIS',
                                  freesurfer=True,
                                  longitudinal=False,
                                  debug=False,
                                  hires=True,
                                  num_t1w=1)

    **Parameters**

        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        output_spaces : list
            List of output spaces functional images are to be resampled to.

            Some pipeline components will only be instantiated for some output spaces.

            Valid spaces:

              - T1w
              - template
              - fsnative
              - fsaverage (or other pre-existing FreeSurfer templates)
        template : str
            Name of template targeted by ``template`` output space
        debug : bool
            Enable debugging outputs
        freesurfer : bool
            Enable FreeSurfer surface reconstruction (may increase runtime)
        longitudinal : bool
            Create unbiased structural template, regardless of number of inputs
            (may increase runtime)
        omp_nthreads : int
            Maximum number of threads an individual process may use
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer
        reportlets_dir : str
            Directory in which to save reportlets
        output_dir : str
            Directory in which to save derivatives
        name : str, optional
            Workflow name (default: anat_preproc_wf)
        skull_strip_fixed_seed : bool
            Do not use a random seed for skull-stripping - will ensure
            run-to-run replicability when used with --omp-nthreads 1 (default: ``False``)


    **Inputs**

        t1w
            List of T1-weighted structural images
        t2w
            List of T2-weighted structural images
        flair
            List of FLAIR images
        subjects_dir
            FreeSurfer SUBJECTS_DIR


    **Outputs**

        t1_preproc
            Bias-corrected structural template, defining T1w space
        t1_brain
            Skull-stripped ``t1_preproc``
        t1_mask
            Mask of the skull-stripped template image
        t1_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        t1_tpms
            List of tissue probability maps in T1w space
        t1_2_mni
            T1w template, normalized to MNI space
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file (inverse)
        mni_mask
            Mask of skull-stripped template, in MNI space
        mni_seg
            Segmentation, resampled into MNI space
        mni_tpms
            List of tissue probability maps in MNI space
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w
        surfaces
            GIFTI surfaces (gray/white boundary, midthickness, pial, inflated)

    **Subworkflows**

        * :py:func:`~fmriprep.workflows.anatomical.init_skullstrip_ants_wf`
        * :py:func:`~fmriprep.workflows.anatomical.init_surface_recon_wf`

    """

    workflow = Workflow(name=name)
    workflow.__postdesc__ = """\
Spatial normalization to the ICBM 152 Nonlinear Asymmetrical
template version 2009c [@mni, RRID:SCR_008796] was performed
through nonlinear registration with `antsRegistration`
[ANTs {ants_ver}, RRID:SCR_004757, @ants], using
brain-extracted versions of both T1w volume and template.
Brain tissue segmentation of cerebrospinal fluid (CSF),
white-matter (WM) and gray-matter (GM) was performed on
the brain-extracted T1w using `fast` [FSL {fsl_ver}, RRID:SCR_002823,
@fsl_fast].
""".format(
        ants_ver=BrainExtraction().version or '<ver>',
        fsl_ver=fsl.FAST().version or '<ver>',
    )
    desc = """Anatomical data preprocessing

: """
    desc += """\
A total of {num_t1w} T1-weighted (T1w) images were found within the input
BIDS dataset.
All of them were corrected for intensity non-uniformity (INU)
using `N4BiasFieldCorrection` [@n4, ANTs {ants_ver}].
""" if num_t1w > 1 else """\
The T1-weighted (T1w) image was corrected for intensity non-uniformity (INU)
using `N4BiasFieldCorrection` [@n4, ANTs {ants_ver}],
and used as T1w-reference throughout the workflow.
"""

    workflow.__desc__ = desc.format(
        num_t1w=num_t1w,
        ants_ver=BrainExtraction().version or '<ver>'
    )

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1w', 't2w', 'roi', 'flair', 'subjects_dir', 'subject_id']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_tpms',
                't1_2_mni', 't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                'mni_mask', 'mni_seg', 'mni_tpms',
                'template_transforms',
                'subjects_dir', 'subject_id', 't1_2_fsnative_forward_transform',
                't1_2_fsnative_reverse_transform', 'surfaces', 't1_aseg', 't1_aparc']),
        name='outputnode')

    buffernode = pe.Node(niu.IdentityInterface(
        fields=['t1_brain', 't1_mask']), name='buffernode')

    anat_template_wf = init_anat_template_wf(longitudinal=longitudinal, omp_nthreads=omp_nthreads,
                                             num_t1w=num_t1w)

    # 3. Skull-stripping
    # Bias field correction is handled in skull strip workflows.
    skullstrip_ants_wf = init_skullstrip_ants_wf(name='skullstrip_ants_wf',
                                                 skull_strip_template=skull_strip_template,
                                                 debug=debug,
                                                 omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode, anat_template_wf, [('t1w', 'inputnode.t1w')]),
        (anat_template_wf, skullstrip_ants_wf, [('outputnode.t1_template', 'inputnode.in_file')]),
        (skullstrip_ants_wf, outputnode, [('outputnode.bias_corrected', 't1_preproc')]),
        (anat_template_wf, outputnode, [
            ('outputnode.template_transforms', 't1_template_transforms')]),
        (buffernode, outputnode, [('t1_brain', 't1_brain'),
                                  ('t1_mask', 't1_mask')]),
    ])

    # 4. Surface reconstruction
    if freesurfer:
        surface_recon_wf = init_surface_recon_wf(name='surface_recon_wf',
                                                 omp_nthreads=omp_nthreads, hires=hires)
        applyrefined = pe.Node(fsl.ApplyMask(), name='applyrefined')
        workflow.connect([
            (inputnode, surface_recon_wf, [
                ('t2w', 'inputnode.t2w'),
                ('flair', 'inputnode.flair'),
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id')]),
            (anat_template_wf, surface_recon_wf, [('outputnode.t1_template', 'inputnode.t1w')]),
            (skullstrip_ants_wf, surface_recon_wf, [
                ('outputnode.out_file', 'inputnode.skullstripped_t1'),
                ('outputnode.out_segs', 'inputnode.ants_segs'),
                ('outputnode.bias_corrected', 'inputnode.corrected_t1')]),
            (skullstrip_ants_wf, applyrefined, [
                ('outputnode.bias_corrected', 'in_file')]),
            (surface_recon_wf, applyrefined, [
                ('outputnode.out_brainmask', 'mask_file')]),
            (surface_recon_wf, outputnode, [
                ('outputnode.subjects_dir', 'subjects_dir'),
                ('outputnode.subject_id', 'subject_id'),
                ('outputnode.t1_2_fsnative_forward_transform', 't1_2_fsnative_forward_transform'),
                ('outputnode.t1_2_fsnative_reverse_transform', 't1_2_fsnative_reverse_transform'),
                ('outputnode.surfaces', 'surfaces'),
                ('outputnode.out_aseg', 't1_aseg'),
                ('outputnode.out_aparc', 't1_aparc')]),
            (applyrefined, buffernode, [('out_file', 't1_brain')]),
            (surface_recon_wf, buffernode, [
                ('outputnode.out_brainmask', 't1_mask')]),
        ])
    else:
        workflow.connect([
            (skullstrip_ants_wf, buffernode, [
                ('outputnode.out_file', 't1_brain'),
                ('outputnode.out_mask', 't1_mask')]),
        ])

    # 5. Segmentation
    t1_seg = pe.Node(fsl.FAST(segments=True, no_bias=True, probability_maps=True),
                     name='t1_seg', mem_gb=3)

    workflow.connect([
        (buffernode, t1_seg, [('t1_brain', 'in_files')]),
        (t1_seg, outputnode, [('tissue_class_map', 't1_seg'),
                              ('probability_maps', 't1_tpms')]),
    ])

    # 6. Spatial normalization (T1w to MNI registration)
    t1_2_mni = pe.Node(
        RobustMNINormalizationRPT(
            float=True,
            generate_report=True,
            flavor='testing' if debug else 'precise',
        ),
        name='t1_2_mni',
        n_procs=omp_nthreads,
        mem_gb=2
    )

    # Resample the brain mask and the tissue probability maps into mni space
    mni_mask = pe.Node(
        ApplyTransforms(dimension=3, default_value=0, float=True,
                        interpolation='MultiLabel'),
        name='mni_mask'
    )

    mni_seg = pe.Node(
        ApplyTransforms(dimension=3, default_value=0, float=True,
                        interpolation='MultiLabel'),
        name='mni_seg'
    )

    mni_tpms = pe.MapNode(
        ApplyTransforms(dimension=3, default_value=0, float=True,
                        interpolation='Linear'),
        iterfield=['input_image'],
        name='mni_tpms'
    )

    if 'template' in output_spaces:
        ref_img = str(nid.get_template(template) /
                      ('tpl-%s_space-MNI_res-01_T1w.nii.gz' % template))

        t1_2_mni.inputs.template = template
        mni_mask.inputs.reference_image = ref_img
        mni_seg.inputs.reference_image = ref_img
        mni_tpms.inputs.reference_image = ref_img

        workflow.connect([
            (inputnode, t1_2_mni, [('roi', 'lesion_mask')]),
            (skullstrip_ants_wf, t1_2_mni, [('outputnode.bias_corrected', 'moving_image')]),
            (buffernode, t1_2_mni, [('t1_mask', 'moving_mask')]),
            (buffernode, mni_mask, [('t1_mask', 'input_image')]),
            (t1_2_mni, mni_mask, [('composite_transform', 'transforms')]),
            (t1_seg, mni_seg, [('tissue_class_map', 'input_image')]),
            (t1_2_mni, mni_seg, [('composite_transform', 'transforms')]),
            (t1_seg, mni_tpms, [('probability_maps', 'input_image')]),
            (t1_2_mni, mni_tpms, [('composite_transform', 'transforms')]),
            (t1_2_mni, outputnode, [
                ('warped_image', 't1_2_mni'),
                ('composite_transform', 't1_2_mni_forward_transform'),
                ('inverse_composite_transform', 't1_2_mni_reverse_transform')]),
            (mni_mask, outputnode, [('output_image', 'mni_mask')]),
            (mni_seg, outputnode, [('output_image', 'mni_seg')]),
            (mni_tpms, outputnode, [('output_image', 'mni_tpms')]),
        ])

    seg_rpt = pe.Node(ROIsPlot(colors=['magenta', 'b'], levels=[1.5, 2.5]),
                      name='seg_rpt')
    anat_reports_wf = init_anat_reports_wf(
        reportlets_dir=reportlets_dir, output_spaces=output_spaces, template=template,
        freesurfer=freesurfer)
    workflow.connect([
        (inputnode, anat_reports_wf, [
            (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file')]),
        (anat_template_wf, anat_reports_wf, [
            ('outputnode.out_report', 'inputnode.t1_conform_report')]),
        (anat_template_wf, seg_rpt, [
            ('outputnode.t1_template', 'in_file')]),
        (t1_seg, seg_rpt, [('tissue_class_map', 'in_rois')]),
        (outputnode, seg_rpt, [('t1_mask', 'in_mask')]),
        (seg_rpt, anat_reports_wf, [('out_report', 'inputnode.seg_report')]),
    ])

    if freesurfer:
        workflow.connect([
            (surface_recon_wf, anat_reports_wf, [
                ('outputnode.out_report', 'inputnode.recon_report')]),
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (t1_2_mni, anat_reports_wf, [('out_report', 'inputnode.t1_2_mni_report')]),
        ])

    anat_derivatives_wf = init_anat_derivatives_wf(output_dir=output_dir,
                                                   output_spaces=output_spaces,
                                                   template=template,
                                                   freesurfer=freesurfer)

    workflow.connect([
        (anat_template_wf, anat_derivatives_wf, [
            ('outputnode.t1w_valid_list', 'inputnode.source_files')]),
        (outputnode, anat_derivatives_wf, [
            ('t1_template_transforms', 'inputnode.t1_template_transforms'),
            ('t1_preproc', 'inputnode.t1_preproc'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_tpms', 'inputnode.t1_tpms'),
            ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
            ('t1_2_mni', 'inputnode.t1_2_mni'),
            ('mni_mask', 'inputnode.mni_mask'),
            ('mni_seg', 'inputnode.mni_seg'),
            ('mni_tpms', 'inputnode.mni_tpms'),
            ('t1_2_fsnative_forward_transform', 'inputnode.t1_2_fsnative_forward_transform'),
            ('surfaces', 'inputnode.surfaces'),
        ]),
    ])

    if freesurfer:
        workflow.connect([
            (surface_recon_wf, anat_derivatives_wf, [
                ('outputnode.out_aseg', 'inputnode.t1_fs_aseg'),
                ('outputnode.out_aparc', 'inputnode.t1_fs_aparc'),
            ]),
        ])

    return workflow


def init_anat_template_wf(longitudinal, omp_nthreads, num_t1w, name='anat_template_wf'):
    r"""
    This workflow generates a canonically oriented structural template from
    input T1w images.


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_anat_template_wf
        wf = init_anat_template_wf(longitudinal=False, omp_nthreads=1, num_t1w=1)

    **Parameters**

        longitudinal : bool
            Create unbiased structural template, regardless of number of inputs
            (may increase runtime)
        omp_nthreads : int
            Maximum number of threads an individual process may use
        num_t1w : int
            Number of T1w images
        name : str, optional
            Workflow name (default: anat_template_wf)


    **Inputs**

        t1w
            List of T1-weighted structural images


    **Outputs**

        t1_template
            Structural template, defining T1w space
        template_transforms
            List of affine transforms from ``t1_template`` to original T1w images
        out_report
            Conformation report
    """

    workflow = Workflow(name=name)

    if num_t1w > 1:
        workflow.__desc__ = """\
A T1w-reference map was computed after registration of
{num_t1w} T1w images (after INU-correction) using
`mri_robust_template` [FreeSurfer {fs_ver}, @fs_template].
""".format(num_t1w=num_t1w, fs_ver=fs.Info().looseversion() or '<ver>')

    inputnode = pe.Node(niu.IdentityInterface(fields=['t1w']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_template', 't1w_valid_list', 'template_transforms', 'out_report']),
        name='outputnode')

    # 0. Reorient T1w image(s) to RAS and resample to common voxel space
    t1_template_dimensions = pe.Node(TemplateDimensions(), name='t1_template_dimensions')
    t1_conform = pe.MapNode(Conform(), iterfield='in_file', name='t1_conform')

    workflow.connect([
        (inputnode, t1_template_dimensions, [('t1w', 't1w_list')]),
        (t1_template_dimensions, t1_conform, [
            ('t1w_valid_list', 'in_file'),
            ('target_zooms', 'target_zooms'),
            ('target_shape', 'target_shape')]),
        (t1_template_dimensions, outputnode, [('out_report', 'out_report'),
                                              ('t1w_valid_list', 't1w_valid_list')]),
    ])

    if num_t1w == 1:
        def _get_first(in_list):
            if isinstance(in_list, (list, tuple)):
                return in_list[0]
            return in_list

        outputnode.inputs.template_transforms = [pkgr('fmriprep', 'data/itkIdentityTransform.txt')]

        workflow.connect([
            (t1_conform, outputnode, [(('out_file', _get_first), 't1_template')]),
        ])

        return workflow

    t1_conform_xfm = pe.MapNode(LTAConvert(in_lta='identity.nofile', out_lta=True),
                                iterfield=['source_file', 'target_file'],
                                name='t1_conform_xfm')

    # 1. Template (only if several T1w images)
    # 1a. Correct for bias field: the bias field is an additive factor
    #     in log-transformed intensity units. Therefore, it is not a linear
    #     combination of fields and N4 fails with merged images.
    # 1b. Align and merge if several T1w images are provided
    n4_correct = pe.MapNode(
        N4BiasFieldCorrection(dimension=3, copy_header=True),
        iterfield='input_image', name='n4_correct',
        n_procs=1)  # n_procs=1 for reproducibility
    # StructuralReference is fs.RobustTemplate if > 1 volume, copying otherwise
    t1_merge = pe.Node(
        StructuralReference(auto_detect_sensitivity=True,
                            initial_timepoint=1,      # For deterministic behavior
                            intensity_scaling=True,   # 7-DOF (rigid + intensity)
                            subsample_threshold=200,
                            fixed_timepoint=not longitudinal,
                            no_iteration=not longitudinal,
                            transform_outputs=True,
                            ),
        mem_gb=2 * num_t1w - 1,
        name='t1_merge')

    # 2. Reorient template to RAS, if needed (mri_robust_template may set to LIA)
    t1_reorient = pe.Node(image.Reorient(), name='t1_reorient')

    concat_affines = pe.MapNode(
        ConcatenateLTA(out_type='RAS2RAS', invert_out=True),
        iterfield=['in_lta1', 'in_lta2'],
        name='concat_affines')

    lta_to_itk = pe.MapNode(LTAConvert(out_itk=True), iterfield=['in_lta'], name='lta_to_itk')

    def _set_threads(in_list, maximum):
        return min(len(in_list), maximum)

    workflow.connect([
        (t1_template_dimensions, t1_conform_xfm, [('t1w_valid_list', 'source_file')]),
        (t1_conform, t1_conform_xfm, [('out_file', 'target_file')]),
        (t1_conform, n4_correct, [('out_file', 'input_image')]),
        (t1_conform, t1_merge, [
            (('out_file', _set_threads, omp_nthreads), 'num_threads'),
            (('out_file', add_suffix, '_template'), 'out_file')]),
        (n4_correct, t1_merge, [('output_image', 'in_files')]),
        (t1_merge, t1_reorient, [('out_file', 'in_file')]),
        # Combine orientation and template transforms
        (t1_conform_xfm, concat_affines, [('out_lta', 'in_lta1')]),
        (t1_merge, concat_affines, [('transform_outputs', 'in_lta2')]),
        (concat_affines, lta_to_itk, [('out_file', 'in_lta')]),
        # Output
        (t1_reorient, outputnode, [('out_file', 't1_template')]),
        (lta_to_itk, outputnode, [('out_itk', 'template_transforms')]),
    ])

    return workflow


def init_skullstrip_ants_wf(skull_strip_template, debug, omp_nthreads,
                            skull_strip_fixed_seed=False, name='skullstrip_ants_wf'):
    r"""
    This workflow performs skull-stripping using ANTs' ``BrainExtraction.sh``

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_skullstrip_ants_wf
        wf = init_skullstrip_ants_wf(skull_strip_template='OASIS', debug=False, omp_nthreads=1)

    **Parameters**

        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        debug : bool
            Enable debugging outputs
        omp_nthreads : int
            Maximum number of threads an individual process may use
        skull_strip_fixed_seed : bool
            Do not use a random seed for skull-stripping - will ensure
            run-to-run replicability when used with --omp-nthreads 1 (default: ``False``)

    **Inputs**

        in_file
            T1-weighted structural image to skull-strip

    **Outputs**

        bias_corrected
            Bias-corrected ``in_file``, before skull-stripping
        out_file
            Skull-stripped ``in_file``
        out_mask
            Binary mask of the skull-stripped ``in_file``
        out_report
            Reportlet visualizing quality of skull-stripping

    """
    from niworkflows.data.getters import get_template, TEMPLATE_ALIASES

    if skull_strip_template not in ['OASIS', 'NKI']:
        raise ValueError("Unknown skull-stripping template; select from {OASIS, NKI}")

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The T1w-reference was then skull-stripped using `antsBrainExtraction.sh`
(ANTs {ants_ver}), using {skullstrip_tpl} as target template.
""".format(ants_ver=BrainExtraction().version or '<ver>', skullstrip_tpl=skull_strip_template)

    # Account for template aliases
    template_name = TEMPLATE_ALIASES.get(skull_strip_template, skull_strip_template)
    # Template path
    template_dir = get_template(template_name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bias_corrected', 'out_file', 'out_mask', 'out_segs', 'out_report']),
        name='outputnode')

    t1_skull_strip = pe.Node(
        BrainExtraction(dimension=3, use_floatingpoint_precision=1, debug=debug,
                        keep_temporary_files=1, use_random_seeding=not skull_strip_fixed_seed),
        name='t1_skull_strip', n_procs=omp_nthreads)

    # Set appropriate inputs
    t1_skull_strip.inputs.brain_template = str(
        template_dir / ('tpl-%s_res-01_T1w.nii.gz' % template_name))
    t1_skull_strip.inputs.brain_probability_mask = str(
        template_dir /
        ('tpl-%s_res-01_class-brainmask_probtissue.nii.gz' % template_name))
    t1_skull_strip.inputs.extraction_registration_mask = str(
        template_dir /
        ('tpl-%s_res-01_label-BrainCerebellumExtraction_roi.nii.gz' % template_name))

    workflow.connect([
        (inputnode, t1_skull_strip, [('in_file', 'anatomical_image')]),
        (t1_skull_strip, outputnode, [('BrainExtractionMask', 'out_mask'),
                                      ('BrainExtractionBrain', 'out_file'),
                                      ('BrainExtractionSegmentation', 'out_segs'),
                                      ('N4Corrected0', 'bias_corrected')])
    ])

    return workflow


def init_surface_recon_wf(omp_nthreads, hires, name='surface_recon_wf'):
    r"""
    This workflow reconstructs anatomical surfaces using FreeSurfer's ``recon-all``.

    Reconstruction is performed in three phases.
    The first phase initializes the subject with T1w and T2w (if available)
    structural images and performs basic reconstruction (``autorecon1``) with the
    exception of skull-stripping.
    For example, a subject with only one session with T1w and T2w images
    would be processed by the following command::

        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -i <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T1w.nii.gz \
            -T2 <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T2w.nii.gz \
            -autorecon1 \
            -noskullstrip

    The second phase imports an externally computed skull-stripping mask.
    This workflow refines the external brainmask using the internal mask
    implicit the the FreeSurfer's ``aseg.mgz`` segmentation,
    to reconcile ANTs' and FreeSurfer's brain masks.

    First, the ``aseg.mgz`` mask from FreeSurfer is refined in two
    steps, using binary morphological operations:

      1. With a binary closing operation the sulci are included
         into the mask. This results in a smoother brain mask
         that does not exclude deep, wide sulci.

      2. Fill any holes (typically, there could be a hole next to
         the pineal gland and the corpora quadrigemina if the great
         cerebral brain is segmented out).

    Second, the brain mask is grown, including pixels that have a high likelihood
    to the GM tissue distribution:

      3. Dilate and substract the brain mask, defining the region to search for candidate
         pixels that likely belong to cortical GM.

      4. Pixels found in the search region that are labeled as GM by ANTs
         (during ``antsBrainExtraction.sh``) are directly added to the new mask.

      5. Otherwise, estimate GM tissue parameters locally in  patches of ``ww`` size,
         and test the likelihood of the pixel to belong in the GM distribution.

    This procedure is inspired on mindboggle's solution to the problem:
    https://github.com/nipy/mindboggle/blob/7f91faaa7664d820fe12ccc52ebaf21d679795e2/mindboggle/guts/segment.py#L1660


    The final phase resumes reconstruction, using the T2w image to assist
    in finding the pial surface, if available.
    See :py:func:`~fmriprep.workflows.anatomical.init_autorecon_resume_wf` for details.


    Memory annotations for FreeSurfer are based off `their documentation
    <https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements>`_.
    They specify an allocation of 4GB per subject. Here we define 5GB
    to have a certain margin.



    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_surface_recon_wf
        wf = init_surface_recon_wf(omp_nthreads=1, hires=True)

    **Parameters**

        omp_nthreads : int
            Maximum number of threads an individual process may use
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer

    **Inputs**

        t1w
            List of T1-weighted structural images
        t2w
            List of T2-weighted structural images (only first used)
        flair
            List of FLAIR images
        skullstripped_t1
            Skull-stripped T1-weighted image (or mask of image)
        ants_segs
            Brain tissue segmentation from ANTS ``antsBrainExtraction.sh``
        corrected_t1
            INU-corrected, merged T1-weighted image
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID

    **Outputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w
        surfaces
            GIFTI surfaces for gray/white matter boundary, pial surface,
            midthickness (or graymid) surface, and inflated surfaces
        out_brainmask
            Refined brainmask, derived from FreeSurfer's ``aseg`` volume
        out_aseg
            FreeSurfer's aseg segmentation, in native T1w space
        out_aparc
            FreeSurfer's aparc+aseg segmentation, in native T1w space
        out_report
            Reportlet visualizing quality of surface alignment

    **Subworkflows**

        * :py:func:`~fmriprep.workflows.anatomical.init_autorecon_resume_wf`
        * :py:func:`~fmriprep.workflows.anatomical.init_gifti_surface_wf`
    """

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Brain surfaces were reconstructed using `recon-all` [FreeSurfer {fs_ver},
RRID:SCR_001847, @fs_reconall], and the brain mask estimated
previously was refined with a custom variation of the method to reconcile
ANTs-derived and FreeSurfer-derived segmentations of the cortical
gray-matter of Mindboggle [RRID:SCR_002438, @mindboggle].
""".format(fs_ver=fs.Info().looseversion() or '<ver>')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't2w', 'flair', 'skullstripped_t1', 'corrected_t1', 'ants_segs',
                    'subjects_dir', 'subject_id']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 't1_2_fsnative_forward_transform',
                    't1_2_fsnative_reverse_transform', 'surfaces', 'out_brainmask',
                    'out_aseg', 'out_aparc', 'out_report']),
        name='outputnode')

    recon_config = pe.Node(FSDetectInputs(hires_enabled=hires), name='recon_config')

    autorecon1 = pe.Node(
        fs.ReconAll(directive='autorecon1', flags='-noskullstrip', openmp=omp_nthreads),
        name='autorecon1', n_procs=omp_nthreads, mem_gb=5)
    autorecon1.interface._can_resume = False
    autorecon1.interface._always_run = True

    skull_strip_extern = pe.Node(FSInjectBrainExtracted(), name='skull_strip_extern')

    fsnative_2_t1_xfm = pe.Node(RobustRegister(auto_sens=True, est_int_scale=True),
                                name='fsnative_2_t1_xfm')
    t1_2_fsnative_xfm = pe.Node(LTAConvert(out_lta=True, invert=True),
                                name='t1_2_fsnative_xfm')

    autorecon_resume_wf = init_autorecon_resume_wf(omp_nthreads=omp_nthreads)
    gifti_surface_wf = init_gifti_surface_wf()

    aseg_to_native_wf = init_segs_to_native_wf()
    aparc_to_native_wf = init_segs_to_native_wf(segmentation='aparc_aseg')
    refine = pe.Node(RefineBrainMask(), name='refine')

    workflow.connect([
        # Configuration
        (inputnode, recon_config, [('t1w', 't1w_list'),
                                   ('t2w', 't2w_list'),
                                   ('flair', 'flair_list')]),
        # Passing subjects_dir / subject_id enforces serial order
        (inputnode, autorecon1, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id')]),
        (autorecon1, skull_strip_extern, [('subjects_dir', 'subjects_dir'),
                                          ('subject_id', 'subject_id')]),
        (skull_strip_extern, autorecon_resume_wf, [('subjects_dir', 'inputnode.subjects_dir'),
                                                   ('subject_id', 'inputnode.subject_id')]),
        (autorecon_resume_wf, gifti_surface_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        # Reconstruction phases
        (inputnode, autorecon1, [('t1w', 'T1_files')]),
        (recon_config, autorecon1, [('t2w', 'T2_file'),
                                    ('flair', 'FLAIR_file'),
                                    ('hires', 'hires'),
                                    # First run only (recon-all saves expert options)
                                    ('mris_inflate', 'mris_inflate')]),
        (inputnode, skull_strip_extern, [('skullstripped_t1', 'in_brain')]),
        (recon_config, autorecon_resume_wf, [('use_t2w', 'inputnode.use_T2'),
                                             ('use_flair', 'inputnode.use_FLAIR')]),
        # Construct transform from FreeSurfer conformed image to FMRIPREP
        # reoriented image
        (inputnode, fsnative_2_t1_xfm, [('t1w', 'target_file')]),
        (autorecon1, fsnative_2_t1_xfm, [('T1', 'source_file')]),
        (fsnative_2_t1_xfm, gifti_surface_wf, [
            ('out_reg_file', 'inputnode.t1_2_fsnative_reverse_transform')]),
        (fsnative_2_t1_xfm, t1_2_fsnative_xfm, [('out_reg_file', 'in_lta')]),
        # Refine ANTs mask, deriving new mask from FS' aseg
        (inputnode, refine, [('corrected_t1', 'in_anat'),
                             ('ants_segs', 'in_ants')]),
        (inputnode, aseg_to_native_wf, [('corrected_t1', 'inputnode.in_file')]),
        (autorecon_resume_wf, aseg_to_native_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        (inputnode, aparc_to_native_wf, [('corrected_t1', 'inputnode.in_file')]),
        (autorecon_resume_wf, aparc_to_native_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        (aseg_to_native_wf, refine, [('outputnode.out_file', 'in_aseg')]),

        # Output
        (autorecon_resume_wf, outputnode, [('outputnode.subjects_dir', 'subjects_dir'),
                                           ('outputnode.subject_id', 'subject_id'),
                                           ('outputnode.out_report', 'out_report')]),
        (gifti_surface_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
        (t1_2_fsnative_xfm, outputnode, [('out_lta', 't1_2_fsnative_forward_transform')]),
        (fsnative_2_t1_xfm, outputnode, [('out_reg_file', 't1_2_fsnative_reverse_transform')]),
        (refine, outputnode, [('out_file', 'out_brainmask')]),
        (aseg_to_native_wf, outputnode, [('outputnode.out_file', 'out_aseg')]),
        (aparc_to_native_wf, outputnode, [('outputnode.out_file', 'out_aparc')]),
    ])

    return workflow


def init_autorecon_resume_wf(omp_nthreads, name='autorecon_resume_wf'):
    r"""
    This workflow resumes recon-all execution, assuming the `-autorecon1` stage
    has been completed.

    In order to utilize resources efficiently, this is broken down into five
    sub-stages; after the first stage, the second and third stages may be run
    simultaneously, and the fourth and fifth stages may be run simultaneously,
    if resources permit::

        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon2-volonly
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon-hemi lh \
            -noparcstats -nocortparc2 -noparcstats2 -nocortparc3 \
            -noparcstats3 -nopctsurfcon -nohyporelabel -noaparc2aseg \
            -noapas2aseg -nosegstats -nowmparc -nobalabels
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon-hemi rh \
            -noparcstats -nocortparc2 -noparcstats2 -nocortparc3 \
            -noparcstats3 -nopctsurfcon -nohyporelabel -noaparc2aseg \
            -noapas2aseg -nosegstats -nowmparc -nobalabels
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon3 -hemi lh -T2pial
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon3 -hemi rh -T2pial

    The excluded steps in the second and third stages (``-no<option>``) are not
    fully hemisphere independent, and are therefore postponed to the final two
    stages.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_autorecon_resume_wf
        wf = init_autorecon_resume_wf(omp_nthreads=1)

    **Inputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        use_T2
            Refine pial surface using T2w image
        use_FLAIR
            Refine pial surface using FLAIR image

    **Outputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        out_report
            Reportlet visualizing quality of surface alignment

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'use_T2', 'use_FLAIR']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'out_report']),
        name='outputnode')

    autorecon2_vol = pe.Node(
        fs.ReconAll(directive='autorecon2-volonly', openmp=omp_nthreads),
        n_procs=omp_nthreads, mem_gb=5, name='autorecon2_vol')
    autorecon2_vol.interface._always_run = True

    autorecon_surfs = pe.MapNode(
        fs.ReconAll(
            directive='autorecon-hemi',
            flags=['-noparcstats', '-nocortparc2', '-noparcstats2',
                   '-nocortparc3', '-noparcstats3', '-nopctsurfcon',
                   '-nohyporelabel', '-noaparc2aseg', '-noapas2aseg',
                   '-nosegstats', '-nowmparc', '-nobalabels'],
            openmp=omp_nthreads),
        iterfield='hemi', n_procs=omp_nthreads, mem_gb=5,
        name='autorecon_surfs')
    autorecon_surfs.inputs.hemi = ['lh', 'rh']
    autorecon_surfs.interface._always_run = True

    autorecon3 = pe.MapNode(
        fs.ReconAll(directive='autorecon3', openmp=omp_nthreads),
        iterfield='hemi', n_procs=omp_nthreads, mem_gb=5,
        name='autorecon3')
    autorecon3.inputs.hemi = ['lh', 'rh']
    autorecon3.interface._always_run = True

    # Only generate the report once; should be nothing to do
    recon_report = pe.Node(
        ReconAllRPT(directive='autorecon3', generate_report=True),
        name='recon_report', mem_gb=5)
    recon_report.interface._always_run = True

    def _dedup(in_list):
        vals = set(in_list)
        if len(vals) > 1:
            raise ValueError(
                "Non-identical values can't be deduplicated:\n{!r}".format(in_list))
        return vals.pop()

    workflow.connect([
        (inputnode, autorecon3, [('use_T2', 'use_T2'),
                                 ('use_FLAIR', 'use_FLAIR')]),
        (inputnode, autorecon2_vol, [('subjects_dir', 'subjects_dir'),
                                     ('subject_id', 'subject_id')]),
        (autorecon2_vol, autorecon_surfs, [('subjects_dir', 'subjects_dir'),
                                           ('subject_id', 'subject_id')]),
        (autorecon_surfs, autorecon3, [(('subjects_dir', _dedup), 'subjects_dir'),
                                       (('subject_id', _dedup), 'subject_id')]),
        (autorecon3, outputnode, [(('subjects_dir', _dedup), 'subjects_dir'),
                                  (('subject_id', _dedup), 'subject_id')]),
        (autorecon3, recon_report, [(('subjects_dir', _dedup), 'subjects_dir'),
                                    (('subject_id', _dedup), 'subject_id')]),
        (recon_report, outputnode, [('out_report', 'out_report')]),
    ])

    return workflow


def init_gifti_surface_wf(name='gifti_surface_wf'):
    r"""
    This workflow prepares GIFTI surfaces from a FreeSurfer subjects directory

    If midthickness (or graymid) surfaces do not exist, they are generated and
    saved to the subject directory as ``lh/rh.midthickness``.
    These, along with the gray/white matter boundary (``lh/rh.smoothwm``), pial
    sufaces (``lh/rh.pial``) and inflated surfaces (``lh/rh.inflated``) are
    converted to GIFTI files.
    Additionally, the vertex coordinates are :py:class:`recentered
    <fmriprep.interfaces.NormalizeSurf>` to align with native T1w space.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_gifti_surface_wf
        wf = init_gifti_surface_wf()

    **Inputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_reverse_transform
            LTA formatted affine transform file (inverse)

    **Outputs**

        surfaces
            GIFTI surfaces for gray/white matter boundary, pial surface,
            midthickness (or graymid) surface, and inflated surfaces

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(['subjects_dir', 'subject_id',
                                               't1_2_fsnative_reverse_transform']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['surfaces']), name='outputnode')

    get_surfaces = pe.Node(nio.FreeSurferSource(), name='get_surfaces')

    midthickness = pe.MapNode(
        MakeMidthickness(thickness=True, distance=0.5, out_name='midthickness'),
        iterfield='in_file',
        name='midthickness')

    save_midthickness = pe.Node(nio.DataSink(parameterization=False),
                                name='save_midthickness')

    surface_list = pe.Node(niu.Merge(4, ravel_inputs=True),
                           name='surface_list', run_without_submitting=True)
    fs_2_gii = pe.MapNode(fs.MRIsConvert(out_datatype='gii'),
                          iterfield='in_file', name='fs_2_gii')
    fix_surfs = pe.MapNode(NormalizeSurf(), iterfield='in_file', name='fix_surfs')

    workflow.connect([
        (inputnode, get_surfaces, [('subjects_dir', 'subjects_dir'),
                                   ('subject_id', 'subject_id')]),
        (inputnode, save_midthickness, [('subjects_dir', 'base_directory'),
                                        ('subject_id', 'container')]),
        # Generate midthickness surfaces and save to FreeSurfer derivatives
        (get_surfaces, midthickness, [('smoothwm', 'in_file'),
                                      ('graymid', 'graymid')]),
        (midthickness, save_midthickness, [('out_file', 'surf.@graymid')]),
        # Produce valid GIFTI surface files (dense mesh)
        (get_surfaces, surface_list, [('smoothwm', 'in1'),
                                      ('pial', 'in2'),
                                      ('inflated', 'in3')]),
        (save_midthickness, surface_list, [('out_file', 'in4')]),
        (surface_list, fs_2_gii, [('out', 'in_file')]),
        (fs_2_gii, fix_surfs, [('converted', 'in_file')]),
        (inputnode, fix_surfs, [('t1_2_fsnative_reverse_transform', 'transform_file')]),
        (fix_surfs, outputnode, [('out_file', 'surfaces')]),
    ])
    return workflow


def init_segs_to_native_wf(name='segs_to_native', segmentation='aseg'):
    """
    Get a segmentation from FreeSurfer conformed space into native T1w space


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_segs_to_native_wf
        wf = init_segs_to_native_wf()


    **Parameters**
        segmentation
            The name of a segmentation ('aseg' or 'aparc_aseg' or 'wmparc')

    **Inputs**

        in_file
            Anatomical, merged T1w image after INU correction
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID


    **Outputs**

        out_file
            The selected segmentation, after resampling in native space
    """
    workflow = Workflow(name='%s_%s' % (name, segmentation))
    inputnode = pe.Node(niu.IdentityInterface([
        'in_file', 'subjects_dir', 'subject_id']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['out_file']), name='outputnode')
    # Extract the aseg and aparc+aseg outputs
    fssource = pe.Node(nio.FreeSurferSource(), name='fs_datasource')
    tonative = pe.Node(fs.Label2Vol(), name='tonative')
    tonii = pe.Node(fs.MRIConvert(out_type='niigz', resample_type='nearest'), name='tonii')

    if segmentation.startswith('aparc'):
        if segmentation == 'aparc_aseg':
            def _sel(x): return [parc for parc in x if 'aparc+' in parc][0]
        elif segmentation == 'aparc_a2009s':
            def _sel(x): return [parc for parc in x if 'a2009s+' in parc][0]
        elif segmentation == 'aparc_dkt':
            def _sel(x): return [parc for parc in x if 'DKTatlas+' in parc][0]
        segmentation = (segmentation, _sel)

    workflow.connect([
        (inputnode, fssource, [
            ('subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id')]),
        (inputnode, tonii, [('in_file', 'reslice_like')]),
        (fssource, tonative, [(segmentation, 'seg_file'),
                              ('rawavg', 'template_file'),
                              ('aseg', 'reg_header')]),
        (tonative, tonii, [('vol_label_file', 'in_file')]),
        (tonii, outputnode, [('out_file', 'out_file')]),
    ])
    return workflow


def init_anat_reports_wf(reportlets_dir, output_spaces,
                         template, freesurfer, name='anat_reports_wf'):
    """
    Set up a battery of datasinks to store reports in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 't1_conform_report', 'seg_report',
                    't1_2_mni_report', 'recon_report']),
        name='inputnode')

    ds_t1_conform_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='conform'),
        name='ds_t1_conform_report', run_without_submitting=True)

    ds_t1_2_mni_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='t1_2_mni'),
        name='ds_t1_2_mni_report', run_without_submitting=True)

    ds_t1_seg_mask_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='seg_brainmask'),
        name='ds_t1_seg_mask_report', run_without_submitting=True)

    ds_recon_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='reconall'),
        name='ds_recon_report', run_without_submitting=True)

    workflow.connect([
        (inputnode, ds_t1_conform_report, [('source_file', 'source_file'),
                                           ('t1_conform_report', 'in_file')]),
        (inputnode, ds_t1_seg_mask_report, [('source_file', 'source_file'),
                                            ('seg_report', 'in_file')]),
    ])

    if freesurfer:
        workflow.connect([
            (inputnode, ds_recon_report, [('source_file', 'source_file'),
                                          ('recon_report', 'in_file')])
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (inputnode, ds_t1_2_mni_report, [('source_file', 'source_file'),
                                             ('t1_2_mni_report', 'in_file')])
        ])

    return workflow


def init_anat_derivatives_wf(output_dir, output_spaces, template, freesurfer,
                             name='anat_derivatives_wf'):
    """
    Set up a battery of datasinks to store derivatives in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_files', 't1_template_transforms',
                    't1_preproc', 't1_mask', 't1_seg', 't1_tpms',
                    't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                    't1_2_mni', 'mni_mask', 'mni_seg', 'mni_tpms',
                    't1_2_fsnative_forward_transform', 'surfaces',
                    't1_fs_aseg', 't1_fs_aparc']),
        name='inputnode')

    t1_name = pe.Node(niu.Function(function=fix_multi_T1w_source_name), name='t1_name')

    ds_t1_preproc = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='preproc', keep_dtype=True),
        name='ds_t1_preproc', run_without_submitting=True)

    ds_t1_mask = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='brain', suffix='mask'),
        name='ds_t1_mask', run_without_submitting=True)

    ds_t1_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='dseg'),
        name='ds_t1_seg', run_without_submitting=True)

    ds_t1_tpms = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix='label-{extra_value}_probseg'),
        name='ds_t1_tpms', run_without_submitting=True)
    ds_t1_tpms.inputs.extra_values = ['CSF', 'GM', 'WM']

    ds_t1_mni = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            space=template, desc='preproc', keep_dtype=True),
        name='ds_t1_mni', run_without_submitting=True)

    ds_mni_mask = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            space=template, desc='brain', suffix='mask'),
        name='ds_mni_mask', run_without_submitting=True)

    ds_mni_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            space=template, suffix='dseg'),
        name='ds_mni_seg', run_without_submitting=True)

    ds_mni_tpms = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            space=template, suffix='label-{extra_value}_probseg'),
        name='ds_mni_tpms', run_without_submitting=True)
    ds_mni_tpms.inputs.extra_values = ['CSF', 'GM', 'WM']

    # Transforms
    suffix_fmt = 'from-{}_to-{}_mode-image_xfm'.format
    ds_t1_mni_inv_warp = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'T1w')),
        name='ds_t1_mni_inv_warp', run_without_submitting=True)

    ds_t1_template_transforms = pe.MapNode(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt('orig', 'T1w')),
        iterfield=['source_file', 'in_file'],
        name='ds_t1_template_transforms', run_without_submitting=True)

    ds_t1_mni_warp = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt('T1w', template)),
        name='ds_t1_mni_warp', run_without_submitting=True)

    lta_2_itk = pe.Node(LTAConvert(out_itk=True), name='lta_2_itk')

    ds_t1_fsnative = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt('T1w', 'fsnative')),
        name='ds_t1_fsnative', run_without_submitting=True)

    name_surfs = pe.MapNode(GiftiNameSource(pattern=r'(?P<LR>[lr])h.(?P<surf>.+)_converted.gii',
                                            template='hemi-{LR}_{surf}.surf'),
                            iterfield='in_file',
                            name='name_surfs',
                            run_without_submitting=True)

    ds_surfs = pe.MapNode(
        DerivativesDataSink(base_directory=output_dir),
        iterfield=['in_file', 'suffix'], name='ds_surfs', run_without_submitting=True)

    workflow.connect([
        (inputnode, t1_name, [('source_files', 'in_files')]),
        (inputnode, ds_t1_template_transforms, [('source_files', 'source_file'),
                                                ('t1_template_transforms', 'in_file')]),
        (inputnode, ds_t1_preproc, [('t1_preproc', 'in_file')]),
        (inputnode, ds_t1_mask, [('t1_mask', 'in_file')]),
        (inputnode, ds_t1_seg, [('t1_seg', 'in_file')]),
        (inputnode, ds_t1_tpms, [('t1_tpms', 'in_file')]),
        (t1_name, ds_t1_preproc, [('out', 'source_file')]),
        (t1_name, ds_t1_mask, [('out', 'source_file')]),
        (t1_name, ds_t1_seg, [('out', 'source_file')]),
        (t1_name, ds_t1_tpms, [('out', 'source_file')]),
    ])

    if freesurfer:
        ds_t1_fsaseg = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='aseg', suffix='dseg'),
            name='ds_t1_fsaseg', run_without_submitting=True)
        ds_t1_fsparc = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='aparcaseg', suffix='dseg'),
            name='ds_t1_fsparc', run_without_submitting=True)
        workflow.connect([
            (inputnode, lta_2_itk, [('t1_2_fsnative_forward_transform', 'in_lta')]),
            (t1_name, ds_t1_fsnative, [('out', 'source_file')]),
            (lta_2_itk, ds_t1_fsnative, [('out_itk', 'in_file')]),
            (inputnode, name_surfs, [('surfaces', 'in_file')]),
            (inputnode, ds_surfs, [('surfaces', 'in_file')]),
            (t1_name, ds_surfs, [('out', 'source_file')]),
            (name_surfs, ds_surfs, [('out_name', 'suffix')]),
            (inputnode, ds_t1_fsaseg, [('t1_fs_aseg', 'in_file')]),
            (inputnode, ds_t1_fsparc, [('t1_fs_aparc', 'in_file')]),
            (t1_name, ds_t1_fsaseg, [('out', 'source_file')]),
            (t1_name, ds_t1_fsparc, [('out', 'source_file')]),
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (inputnode, ds_t1_mni_warp, [('t1_2_mni_forward_transform', 'in_file')]),
            (inputnode, ds_t1_mni_inv_warp, [('t1_2_mni_reverse_transform', 'in_file')]),
            (inputnode, ds_t1_mni, [('t1_2_mni', 'in_file')]),
            (inputnode, ds_mni_mask, [('mni_mask', 'in_file')]),
            (inputnode, ds_mni_seg, [('mni_seg', 'in_file')]),
            (inputnode, ds_mni_tpms, [('mni_tpms', 'in_file')]),
            (t1_name, ds_t1_mni_warp, [('out', 'source_file')]),
            (t1_name, ds_t1_mni_inv_warp, [('out', 'source_file')]),
            (t1_name, ds_t1_mni, [('out', 'source_file')]),
            (t1_name, ds_mni_mask, [('out', 'source_file')]),
            (t1_name, ds_mni_seg, [('out', 'source_file')]),
            (t1_name, ds_mni_tpms, [('out', 'source_file')]),
        ])

    return workflow
