# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Orchestrating the BOLD-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_func_preproc_wf
.. autofunction:: init_func_derivatives_wf

"""

import os

import nibabel as nb
from nipype import logging

from nipype.interfaces.fsl import Split as FSLSplit
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces import (
    DerivativesDataSink,
    GiftiNameSource,
    FirstEcho
)

from ...interfaces.reports import FunctionalSummary
from ...interfaces.cifti import GenerateCifti, CiftiNameSource
from ...engine import Workflow

# BOLD workflows
from .confounds import init_bold_confs_wf, init_carpetplot_wf
from .hmc import init_bold_hmc_wf
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf
from .registration import init_bold_reg_wf
from .resampling import (
    init_bold_surf_wf,
    init_bold_mni_trans_wf,
    init_bold_preproc_trans_wf,
)
from .util import init_bold_reference_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_func_preproc_wf(bold_file, ignore, freesurfer,
                         use_bbr, t2s_coreg, bold2t1w_dof, reportlets_dir,
                         output_spaces, template, output_dir, omp_nthreads,
                         fmap_bspline, fmap_demean, use_syn, force_syn,
                         use_aroma, ignore_aroma_err, aroma_melodic_dim,
                         medial_surface_nan, cifti_output,
                         debug, low_mem, template_out_grid,
                         layout=None, num_bold=1):
    """
    This workflow controls the functional preprocessing stages of FMRIPREP.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold import init_func_preproc_wf
        wf = init_func_preproc_wf('/completely/made/up/path/sub-01_task-nback_bold.nii.gz',
                                  omp_nthreads=1,
                                  ignore=[],
                                  freesurfer=True,
                                  reportlets_dir='.',
                                  output_dir='.',
                                  template='MNI152NLin2009cAsym',
                                  output_spaces=['T1w', 'fsnative',
                                                 'template', 'fsaverage5'],
                                  debug=False,
                                  use_bbr=True,
                                  t2s_coreg=False,
                                  bold2t1w_dof=9,
                                  fmap_bspline=True,
                                  fmap_demean=True,
                                  use_syn=True,
                                  force_syn=True,
                                  low_mem=False,
                                  template_out_grid='native',
                                  medial_surface_nan=False,
                                  cifti_output=False,
                                  use_aroma=False,
                                  ignore_aroma_err=False,
                                  aroma_melodic_dim=None,
                                  num_bold=1)

    **Parameters**

        bold_file : str
            BOLD series NIfTI file
        ignore : list
            Preprocessing steps to skip (may include "slicetiming", "fieldmaps")
        freesurfer : bool
            Enable FreeSurfer functional registration (bbregister) and resampling
            BOLD series to FreeSurfer surface meshes.
        use_bbr : bool or None
            Enable/disable boundary-based registration refinement.
            If ``None``, test BBR result for distortion before accepting.
        t2s_coreg : bool
            Use multiple BOLD echos to create T2*-map for T2*-driven coregistration
        bold2t1w_dof : 6, 9 or 12
            Degrees-of-freedom for BOLD-T1w registration
        reportlets_dir : str
            Directory in which to save reportlets
        output_spaces : list
            List of output spaces functional images are to be resampled to.
            Some parts of pipeline will only be instantiated for some output spaces.

            Valid spaces:

                - T1w
                - template
                - fsnative
                - fsaverage (or other pre-existing FreeSurfer templates)
        template : str
            Name of template targeted by ``template`` output space
        output_dir : str
            Directory in which to save derivatives
        omp_nthreads : int
            Maximum number of threads an individual process may use
        fmap_bspline : bool
            **Experimental**: Fit B-Spline field using least-squares
        fmap_demean : bool
            Demean voxel-shift map during unwarp
        use_syn : bool
            **Experimental**: Enable ANTs SyN-based susceptibility distortion correction (SDC).
            If fieldmaps are present and enabled, this is not run, by default.
        force_syn : bool
            **Temporary**: Always run SyN-based SDC
        use_aroma : bool
            Perform ICA-AROMA on MNI-resampled functional series
        ignore_aroma_err : bool
            Do not fail on ICA-AROMA errors
        medial_surface_nan : bool
            Replace medial wall values with NaNs on functional GIFTI files
        cifti_output : bool
            Generate bold CIFTI file in output spaces
        debug : bool
            Enable debugging outputs
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        template_out_grid : str
            Keyword ('native', '1mm' or '2mm') or path of custom reference
            image for normalization
        layout : BIDSLayout
            BIDSLayout structure to enable metadata retrieval
        num_bold : int
            Total number of BOLD files that have been set for preprocessing
            (default is 1)

    **Inputs**

        bold_file
            BOLD series NIfTI file
        t1_preproc
            Bias-corrected structural template image
        t1_brain
            Skull-stripped ``t1_preproc``
        t1_mask
            Mask of the skull-stripped template image
        t1_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        t1_tpms
            List of tissue probability maps in T1w space
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file (inverse)
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w


    **Outputs**

        bold_t1
            BOLD series, resampled to T1w space
        bold_mask_t1
            BOLD series mask in T1w space
        bold_mni
            BOLD series, resampled to template space
        bold_mask_mni
            BOLD series mask in template space
        confounds
            TSV of confounds
        surfaces
            BOLD series, resampled to FreeSurfer surfaces
        aroma_noise_ics
            Noise components identified by ICA-AROMA
        melodic_mix
            FSL MELODIC mixing matrix
        bold_cifti
            BOLD CIFTI image
        cifti_variant
            combination of target spaces for `bold_cifti`


    **Subworkflows**

        * :py:func:`~fmriprep.workflows.bold.util.init_bold_reference_wf`
        * :py:func:`~fmriprep.workflows.bold.stc.init_bold_stc_wf`
        * :py:func:`~fmriprep.workflows.bold.hmc.init_bold_hmc_wf`
        * :py:func:`~fmriprep.workflows.bold.t2s.init_bold_t2s_wf`
        * :py:func:`~fmriprep.workflows.bold.registration.init_bold_reg_wf`
        * :py:func:`~fmriprep.workflows.bold.confounds.init_bold_confounds_wf`
        * :py:func:`~fmriprep.workflows.bold.confounds.init_ica_aroma_wf`
        * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_mni_trans_wf`
        * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_preproc_trans_wf`
        * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_surf_wf`
        * :py:func:`~fmriprep.workflows.fieldmap.pepolar.init_pepolar_unwarp_wf`
        * :py:func:`~fmriprep.workflows.fieldmap.init_fmap_estimator_wf`
        * :py:func:`~fmriprep.workflows.fieldmap.init_sdc_unwarp_wf`
        * :py:func:`~fmriprep.workflows.fieldmap.init_nonlinear_sdc_wf`

    """
    from ..fieldmap.base import init_sdc_wf  # Avoid circular dependency (#1066)

    ref_file = bold_file
    mem_gb = {'filesize': 1, 'resampled': 1, 'largemem': 1}
    bold_tlen = 10
    multiecho = isinstance(bold_file, list)

    if multiecho:
        tes = [layout.get_metadata(echo)['EchoTime'] for echo in bold_file]
        ref_file = dict(zip(tes, bold_file))[min(tes)]

    if os.path.isfile(ref_file):
        bold_tlen, mem_gb = _create_mem_gb(ref_file)

    wf_name = _get_wf_name(ref_file)
    LOGGER.log(25, ('Creating bold processing workflow for "%s" (%.2f GB / %d TRs). '
                    'Memory resampled/largemem=%.2f/%.2f GB.'),
               ref_file, mem_gb['filesize'], bold_tlen, mem_gb['resampled'], mem_gb['largemem'])

    # For doc building purposes
    if layout is None or bold_file == 'bold_preprocesing':
        LOGGER.log(25, 'No valid layout: building empty workflow.')
        metadata = {
            'RepetitionTime': 2.0,
            'SliceTiming': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'PhaseEncodingDirection': 'j',
        }
        fmaps = [{
            'type': 'phasediff',
            'phasediff': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_phasediff.nii.gz',
            'magnitude1': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude1.nii.gz',
            'magnitude2': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude2.nii.gz',
        }]
        run_stc = True
        multiecho = False
    else:
        metadata = layout.get_metadata(ref_file)

        # Find fieldmaps. Options: (phase1|phase2|phasediff|epi|fieldmap|syn)
        fmaps = []
        if 'fieldmaps' not in ignore:
            fmaps = layout.get_fieldmap(ref_file, return_list=True)
            for fmap in fmaps:
                fmap['metadata'] = layout.get_metadata(fmap[fmap['type']])

        # Run SyN if forced or in the absence of fieldmap correction
        if force_syn or (use_syn and not fmaps):
            fmaps.append({'type': 'syn'})

        # Short circuits: (True and True and (False or 'TooShort')) == 'TooShort'
        run_stc = ("SliceTiming" in metadata and
                   'slicetiming' not in ignore and
                   (_get_series_len(ref_file) > 4 or "TooShort"))

    # Use T2* as target for ME-EPI in co-registration
    if t2s_coreg and not multiecho:
        LOGGER.warning("No multiecho BOLD images found for T2* coregistration. "
                       "Using standard EPI-T1 coregistration.")
        t2s_coreg = False

    # Switch stc off
    if multiecho and run_stc is True:
        LOGGER.warning('Slice-timing correction is not available for '
                       'multiecho BOLD data (not implemented).')
        run_stc = False

    # Build workflow
    workflow = Workflow(name=wf_name)
    workflow.__desc__ = """

Functional data preprocessing

: For each of the {num_bold} BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
""".format(num_bold=num_bold)

    workflow.__postdesc__ = """\
All resamplings can be performed with *a single interpolation
step* by composing all the pertinent transformations (i.e. head-motion
transform matrices, susceptibility distortion correction when available,
and co-registrations to anatomical and template spaces).
Gridded (volumetric) resamplings were performed using `antsApplyTransforms` (ANTs),
configured with Lanczos interpolation to minimize the smoothing
effects of other kernels [@lanczos].
Non-gridded (surface) resamplings were performed using `mri_vol2surf`
(FreeSurfer).
"""

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_file', 'subjects_dir', 'subject_id',
                't1_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_tpms',
                't1_aseg', 't1_aparc',
                't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                't1_2_fsnative_forward_transform', 't1_2_fsnative_reverse_transform']),
        name='inputnode')
    inputnode.inputs.bold_file = bold_file

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_t1', 'bold_mask_t1', 'bold_aseg_t1', 'bold_aparc_t1', 'cifti_variant',
                'bold_mni', 'bold_mask_mni', 'bold_cifti', 'confounds', 'surfaces',
                't2s_map', 'aroma_noise_ics', 'melodic_mix', 'nonaggr_denoised_file',
                'cifti_variant_key']),
        name='outputnode')

    # BOLD buffer: an identity used as a pointer to either the original BOLD
    # or the STC'ed one for further use.
    boldbuffer = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='boldbuffer')

    summary = pe.Node(
        FunctionalSummary(output_spaces=output_spaces,
                          slice_timing=run_stc,
                          registration='FreeSurfer' if freesurfer else 'FSL',
                          registration_dof=bold2t1w_dof,
                          pe_direction=metadata.get("PhaseEncodingDirection")),
        name='summary', mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True)

    func_derivatives_wf = init_func_derivatives_wf(output_dir=output_dir,
                                                   output_spaces=output_spaces,
                                                   template=template,
                                                   freesurfer=freesurfer,
                                                   use_aroma=use_aroma,
                                                   cifti_output=cifti_output)

    workflow.connect([
        (inputnode, func_derivatives_wf, [('bold_file', 'inputnode.source_file')]),
        (outputnode, func_derivatives_wf, [
            ('bold_t1', 'inputnode.bold_t1'),
            ('bold_aseg_t1', 'inputnode.bold_aseg_t1'),
            ('bold_aparc_t1', 'inputnode.bold_aparc_t1'),
            ('bold_mask_t1', 'inputnode.bold_mask_t1'),
            ('bold_mni', 'inputnode.bold_mni'),
            ('bold_mask_mni', 'inputnode.bold_mask_mni'),
            ('confounds', 'inputnode.confounds'),
            ('surfaces', 'inputnode.surfaces'),
            ('aroma_noise_ics', 'inputnode.aroma_noise_ics'),
            ('melodic_mix', 'inputnode.melodic_mix'),
            ('nonaggr_denoised_file', 'inputnode.nonaggr_denoised_file'),
            ('bold_cifti', 'inputnode.bold_cifti'),
            ('cifti_variant', 'inputnode.cifti_variant'),
            ('cifti_variant_key', 'inputnode.cifti_variant_key')
        ]),
    ])

    # The first reference uses T2 contrast enhancement
    bold_reference_wf = init_bold_reference_wf(
        omp_nthreads=omp_nthreads, enhance_t2=True)

    # Top-level BOLD splitter
    bold_split = pe.Node(FSLSplit(dimension='t'), name='bold_split',
                         mem_gb=mem_gb['filesize'] * 3)

    # HMC on the BOLD
    bold_hmc_wf = init_bold_hmc_wf(name='bold_hmc_wf',
                                   mem_gb=mem_gb['filesize'],
                                   omp_nthreads=omp_nthreads)

    # mean BOLD registration to T1w
    bold_reg_wf = init_bold_reg_wf(name='bold_reg_wf',
                                   freesurfer=freesurfer,
                                   use_bbr=use_bbr,
                                   bold2t1w_dof=bold2t1w_dof,
                                   mem_gb=mem_gb['resampled'],
                                   omp_nthreads=omp_nthreads,
                                   use_compression=False,
                                   use_fieldwarp=(fmaps is not None or use_syn))

    # get confounds
    bold_confounds_wf = init_bold_confs_wf(
        mem_gb=mem_gb['largemem'],
        metadata=metadata,
        name='bold_confounds_wf')
    bold_confounds_wf.get_node('inputnode').inputs.t1_transform_flags = [False]

    # Apply transforms in 1 shot
    # Only use uncompressed output if AROMA is to be run
    bold_bold_trans_wf = init_bold_preproc_trans_wf(
        mem_gb=mem_gb['resampled'],
        omp_nthreads=omp_nthreads,
        use_compression=not low_mem,
        use_fieldwarp=(fmaps is not None or use_syn),
        name='bold_bold_trans_wf'
    )

    # SLICE-TIME CORRECTION (or bypass) #############################################
    if run_stc is True:  # bool('TooShort') == True, so check True explicitly
        bold_stc_wf = init_bold_stc_wf(name='bold_stc_wf', metadata=metadata)
        workflow.connect([
            (bold_reference_wf, bold_stc_wf, [('outputnode.bold_file', 'inputnode.bold_file'),
                                              ('outputnode.skip_vols', 'inputnode.skip_vols')]),
            (bold_stc_wf, boldbuffer, [('outputnode.stc_file', 'bold_file')]),
        ])
    else:  # bypass STC from original BOLD to the splitter through boldbuffer
        workflow.connect([
            (bold_reference_wf, boldbuffer, [
                ('outputnode.bold_file', 'bold_file')]),
        ])

    # SDC (SUSCEPTIBILITY DISTORTION CORRECTION) or bypass ##########################
    bold_sdc_wf = init_sdc_wf(
        fmaps, metadata, omp_nthreads=omp_nthreads,
        debug=debug, fmap_demean=fmap_demean, fmap_bspline=fmap_bspline)
    bold_sdc_wf.inputs.inputnode.template = template

    if not fmaps:
        LOGGER.warning('SDC: no fieldmaps found or they were ignored (%s).',
                       ref_file)
    elif fmaps[0]['type'] == 'syn':
        LOGGER.warning(
            'SDC: no fieldmaps found or they were ignored. '
            'Using EXPERIMENTAL "fieldmap-less SyN" correction '
            'for dataset %s.', ref_file)
    else:
        LOGGER.log(25, 'SDC: fieldmap estimation of type "%s" intended for %s found.',
                   fmaps[0]['type'], ref_file)

    # MAIN WORKFLOW STRUCTURE #######################################################
    workflow.connect([
        # Generate early reference
        (inputnode, bold_reference_wf, [('bold_file', 'inputnode.bold_file')]),
        # BOLD buffer has slice-time corrected if it was run, original otherwise
        (boldbuffer, bold_split, [('bold_file', 'in_file')]),
        # HMC
        (bold_reference_wf, bold_hmc_wf, [
            ('outputnode.raw_ref_image', 'inputnode.raw_ref_image'),
            ('outputnode.bold_file', 'inputnode.bold_file')]),
        # EPI-T1 registration workflow
        (inputnode, bold_reg_wf, [
            ('bold_file', 'inputnode.name_source'),
            ('t1_preproc', 'inputnode.t1_preproc'),
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_aseg', 'inputnode.t1_aseg'),
            ('t1_aparc', 'inputnode.t1_aparc'),
            # Undefined if --no-freesurfer, but this is safe
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('t1_2_fsnative_reverse_transform', 'inputnode.t1_2_fsnative_reverse_transform')]),
        (bold_split, bold_reg_wf, [('out_files', 'inputnode.bold_split')]),
        (bold_hmc_wf, bold_reg_wf, [('outputnode.xforms', 'inputnode.hmc_xforms')]),
        (bold_reg_wf, outputnode, [('outputnode.bold_t1', 'bold_t1'),
                                   ('outputnode.bold_aseg_t1', 'bold_aseg_t1'),
                                   ('outputnode.bold_aparc_t1', 'bold_aparc_t1')]),
        (bold_reg_wf, summary, [('outputnode.fallback', 'fallback')]),
        # SDC (or pass-through workflow)
        (inputnode, bold_sdc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform')]),
        (bold_reference_wf, bold_sdc_wf, [
            ('outputnode.ref_image', 'inputnode.bold_ref'),
            ('outputnode.ref_image_brain', 'inputnode.bold_ref_brain'),
            ('outputnode.bold_mask', 'inputnode.bold_mask')]),
        (bold_sdc_wf, bold_reg_wf, [
            ('outputnode.bold_ref_brain', 'inputnode.ref_bold_brain'),
            ('outputnode.bold_mask', 'inputnode.ref_bold_mask'),
            ('outputnode.out_warp', 'inputnode.fieldwarp')]),
        (bold_sdc_wf, bold_bold_trans_wf, [
            ('outputnode.out_warp', 'inputnode.fieldwarp'),
            ('outputnode.bold_mask', 'inputnode.bold_mask')]),
        (bold_sdc_wf, summary, [('outputnode.method', 'distortion_correction')]),
        # Connect bold_confounds_wf
        (inputnode, bold_confounds_wf, [('t1_tpms', 'inputnode.t1_tpms'),
                                        ('t1_mask', 'inputnode.t1_mask')]),
        (bold_hmc_wf, bold_confounds_wf, [
            ('outputnode.movpar_file', 'inputnode.movpar_file')]),
        (bold_reg_wf, bold_confounds_wf, [
            ('outputnode.itk_t1_to_bold', 'inputnode.t1_bold_xform')]),
        (bold_confounds_wf, outputnode, [
            ('outputnode.confounds_file', 'confounds'),
        ]),
        # Connect bold_bold_trans_wf
        (inputnode, bold_bold_trans_wf, [
            ('bold_file', 'inputnode.name_source')]),
        (bold_split, bold_bold_trans_wf, [
            ('out_files', 'inputnode.bold_file')]),
        (bold_hmc_wf, bold_bold_trans_wf, [
            ('outputnode.xforms', 'inputnode.hmc_xforms')]),
        (bold_bold_trans_wf, bold_confounds_wf, [
            ('outputnode.bold', 'inputnode.bold'),
            ('outputnode.bold_mask', 'inputnode.bold_mask')]),
        # Summary
        (outputnode, summary, [('confounds', 'confounds_file')]),
    ])

    if fmaps:
        from ..fieldmap.unwarp import init_fmap_unwarp_report_wf
        sdc_type = fmaps[0]['type']

        # Report on BOLD correction
        fmap_unwarp_report_wf = init_fmap_unwarp_report_wf(
            suffix='sdc_%s' % sdc_type)
        workflow.connect([
            (inputnode, fmap_unwarp_report_wf, [
                ('t1_seg', 'inputnode.in_seg')]),
            (bold_reference_wf, fmap_unwarp_report_wf, [
                ('outputnode.ref_image', 'inputnode.in_pre')]),
            (bold_reg_wf, fmap_unwarp_report_wf, [
                ('outputnode.itk_t1_to_bold', 'inputnode.in_xfm')]),
            (bold_sdc_wf, fmap_unwarp_report_wf, [
                ('outputnode.bold_ref', 'inputnode.in_post')]),
        ])

        if force_syn and sdc_type != 'syn':
            syn_unwarp_report_wf = init_fmap_unwarp_report_wf(
                suffix='forcedsyn', name='syn_unwarp_report_wf')
            workflow.connect([
                (inputnode, syn_unwarp_report_wf, [
                    ('t1_seg', 'inputnode.in_seg')]),
                (bold_reference_wf, syn_unwarp_report_wf, [
                    ('outputnode.ref_image', 'inputnode.in_pre')]),
                (bold_reg_wf, syn_unwarp_report_wf, [
                    ('outputnode.itk_t1_to_bold', 'inputnode.in_xfm')]),
                (bold_sdc_wf, syn_unwarp_report_wf, [
                    ('outputnode.syn_bold_ref', 'inputnode.in_post')]),
            ])

    # if multiecho data, select first echo for hmc correction
    if multiecho:
        inputnode.iterables = ('bold_file', bold_file)

        me_first_echo = pe.Node(FirstEcho(
            te_list=tes, in_files=bold_file, ref_imgs=bold_file),
            name='me_first_echo')
        # Replace reference with the echo selected with FirstEcho
        workflow.disconnect([
            (inputnode, bold_reference_wf, [
                ('bold_file', 'inputnode.bold_file')]),
            (bold_reference_wf, boldbuffer, [
                ('outputnode.bold_file', 'bold_file')]),
        ])
        workflow.connect([
            (me_first_echo, bold_reference_wf, [
                ('first_image', 'inputnode.bold_file')]),
            (inputnode, boldbuffer, [
                ('bold_file', 'bold_file')]),
        ])

        if t2s_coreg:
            # create a T2* map
            bold_t2s_wf = init_bold_t2s_wf(bold_echos=bold_file,
                                           echo_times=tes,
                                           mem_gb=mem_gb['resampled'],
                                           omp_nthreads=omp_nthreads,
                                           name='bold_t2s_wf')
            bold_t2s_wf.inputs.inputnode.name_source = ref_file

            # Replace EPI-to-T1w registration inputs
            workflow.disconnect([
                (bold_sdc_wf, bold_reg_wf, [
                    ('outputnode.bold_ref_brain', 'inputnode.ref_bold_brain'),
                    ('outputnode.bold_mask', 'inputnode.ref_bold_mask')]),
            ])
            workflow.connect([
                (bold_hmc_wf, bold_t2s_wf, [
                    ('outputnode.xforms', 'inputnode.hmc_xforms')]),
                (bold_t2s_wf, bold_reg_wf, [
                    ('outputnode.t2s_map', 'inputnode.ref_bold_brain'),
                    ('outputnode.oc_mask', 'inputnode.ref_bold_mask')]),
            ])

    # Map final BOLD mask into T1w space (if required)
    if 'T1w' in output_spaces:
        from niworkflows.interfaces.fixes import (
            FixHeaderApplyTransforms as ApplyTransforms
        )

        boldmask_to_t1w = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', float=True),
            name='boldmask_to_t1w', mem_gb=0.1
        )
        workflow.connect([
            (bold_bold_trans_wf, boldmask_to_t1w, [
                ('outputnode.bold_mask', 'input_image')]),
            (bold_reg_wf, boldmask_to_t1w, [
                ('outputnode.bold_mask_t1', 'reference_image'),
                ('outputnode.itk_bold_to_t1', 'transforms')]),
            (boldmask_to_t1w, outputnode, [
                ('output_image', 'bold_mask_t1')]),
        ])

    if 'template' in output_spaces:
        # Apply transforms in 1 shot
        # Only use uncompressed output if AROMA is to be run
        bold_mni_trans_wf = init_bold_mni_trans_wf(
            template=template,
            mem_gb=mem_gb['resampled'],
            omp_nthreads=omp_nthreads,
            template_out_grid=template_out_grid,
            use_compression=not low_mem,
            use_fieldwarp=fmaps is not None,
            name='bold_mni_trans_wf'
        )
        carpetplot_wf = init_carpetplot_wf(
            mem_gb=mem_gb['resampled'],
            metadata=metadata,
            name='carpetplot_wf')

        workflow.connect([
            (inputnode, bold_mni_trans_wf, [
                ('bold_file', 'inputnode.name_source'),
                ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform')]),
            (bold_split, bold_mni_trans_wf, [
                ('out_files', 'inputnode.bold_split')]),
            (bold_hmc_wf, bold_mni_trans_wf, [
                ('outputnode.xforms', 'inputnode.hmc_xforms')]),
            (bold_reg_wf, bold_mni_trans_wf, [
                ('outputnode.itk_bold_to_t1', 'inputnode.itk_bold_to_t1')]),
            (bold_bold_trans_wf, bold_mni_trans_wf, [
                ('outputnode.bold_mask', 'inputnode.bold_mask')]),
            (bold_sdc_wf, bold_mni_trans_wf, [
                ('outputnode.out_warp', 'inputnode.fieldwarp')]),
            (bold_mni_trans_wf, outputnode, [('outputnode.bold_mni', 'bold_mni'),
                                             ('outputnode.bold_mask_mni', 'bold_mask_mni')]),
            (bold_bold_trans_wf, carpetplot_wf, [
                ('outputnode.bold', 'inputnode.bold'),
                ('outputnode.bold_mask', 'inputnode.bold_mask')]),
            (inputnode, carpetplot_wf, [
                ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform')]),
            (bold_reg_wf, carpetplot_wf, [
                ('outputnode.itk_t1_to_bold', 'inputnode.t1_bold_xform')]),
            (bold_confounds_wf, carpetplot_wf, [
                ('outputnode.confounds_file', 'inputnode.confounds_file')]),
        ])

        if use_aroma:
            # ICA-AROMA workflow
            # Internally resamples to MNI152 Linear (2006)
            from .confounds import init_ica_aroma_wf
            from ...interfaces import JoinTSVColumns

            ica_aroma_wf = init_ica_aroma_wf(
                template=template,
                metadata=metadata,
                mem_gb=mem_gb['resampled'],
                omp_nthreads=omp_nthreads,
                use_fieldwarp=fmaps is not None,
                ignore_aroma_err=ignore_aroma_err,
                aroma_melodic_dim=aroma_melodic_dim,
                name='ica_aroma_wf')

            join = pe.Node(JoinTSVColumns(), name='aroma_confounds')

            workflow.disconnect([
                (bold_confounds_wf, outputnode, [
                    ('outputnode.confounds_file', 'confounds'),
                ]),
            ])
            workflow.connect([
                (inputnode, ica_aroma_wf, [
                    ('bold_file', 'inputnode.name_source'),
                    ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform')]),
                (bold_split, ica_aroma_wf, [
                    ('out_files', 'inputnode.bold_split')]),
                (bold_hmc_wf, ica_aroma_wf, [
                    ('outputnode.movpar_file', 'inputnode.movpar_file'),
                    ('outputnode.xforms', 'inputnode.hmc_xforms')]),
                (bold_reg_wf, ica_aroma_wf, [
                    ('outputnode.itk_bold_to_t1', 'inputnode.itk_bold_to_t1')]),
                (bold_bold_trans_wf, ica_aroma_wf, [
                    ('outputnode.bold_mask', 'inputnode.bold_mask')]),
                (bold_sdc_wf, ica_aroma_wf, [
                    ('outputnode.out_warp', 'inputnode.fieldwarp')]),
                (bold_confounds_wf, join, [
                    ('outputnode.confounds_file', 'in_file')]),
                (ica_aroma_wf, join,
                    [('outputnode.aroma_confounds', 'join_file')]),
                (ica_aroma_wf, outputnode,
                    [('outputnode.aroma_noise_ics', 'aroma_noise_ics'),
                     ('outputnode.melodic_mix', 'melodic_mix'),
                     ('outputnode.nonaggr_denoised_file', 'nonaggr_denoised_file')]),
                (join, outputnode, [('out_file', 'confounds')]),
            ])

    # SURFACES ##################################################################################
    if freesurfer and any(space.startswith('fs') for space in output_spaces):
        LOGGER.log(25, 'Creating BOLD surface-sampling workflow.')
        bold_surf_wf = init_bold_surf_wf(mem_gb=mem_gb['resampled'],
                                         output_spaces=output_spaces,
                                         medial_surface_nan=medial_surface_nan,
                                         name='bold_surf_wf')
        workflow.connect([
            (inputnode, bold_surf_wf, [
                ('t1_preproc', 'inputnode.t1_preproc'),
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id'),
                ('t1_2_fsnative_forward_transform', 'inputnode.t1_2_fsnative_forward_transform')]),
            (bold_reg_wf, bold_surf_wf, [('outputnode.bold_t1', 'inputnode.source_file')]),
            (bold_surf_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
        ])

        # CIFTI output
        if cifti_output and 'template' in output_spaces:
            bold_surf_wf.__desc__ += """\
*Grayordinates* files [@hcppipelines], which combine surface-sampled
data and volume-sampled data, were also generated.
"""
            gen_cifti = pe.MapNode(GenerateCifti(), iterfield=["surface_target", "gifti_files"],
                                   name="gen_cifti")
            gen_cifti.inputs.TR = metadata.get("RepetitionTime")

            workflow.connect([
                (bold_surf_wf, gen_cifti, [
                    ('targets.out', 'surface_target'),
                    ('outputnode.surfaces', 'gifti_files')]),
                (inputnode, gen_cifti, [('subjects_dir', 'subjects_dir')]),
                (bold_mni_trans_wf, gen_cifti, [('outputnode.bold_mni', 'bold_file')]),
                (gen_cifti, outputnode, [('out_file', 'bold_cifti'),
                                         ('variant', 'cifti_variant'),
                                         ('variant_key', 'cifti_variant_key')]),
            ])

    # REPORTING ############################################################
    ds_report_summary = pe.Node(
        DerivativesDataSink(suffix='summary'),
        name='ds_report_summary', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    ds_report_validation = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='validation'),
        name='ds_report_validation', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bold_reference_wf, ds_report_validation, [
            ('outputnode.validation_report', 'in_file')]),
    ])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            workflow.get_node(node).inputs.source_file = ref_file

    return workflow


def init_func_derivatives_wf(output_dir, output_spaces, template, freesurfer,
                             use_aroma, cifti_output, name='func_derivatives_wf'):
    """
    Set up a battery of datasinks to store derivatives in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 'bold_t1', 'bold_mask_t1', 'bold_mni', 'bold_mask_mni',
                    'bold_aseg_t1', 'bold_aparc_t1', 'cifti_variant_key',
                    'confounds', 'surfaces', 'aroma_noise_ics', 'melodic_mix',
                    'nonaggr_denoised_file', 'bold_cifti', 'cifti_variant']),
        name='inputnode')

    suffix_fmt = 'space-{}_{}'.format
    variant_suffix_fmt = 'space-{}_variant-{}_{}'.format

    ds_confounds = pe.Node(DerivativesDataSink(
        base_directory=output_dir, suffix='confounds'),
        name="ds_confounds", run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, ds_confounds, [('source_file', 'source_file'),
                                   ('confounds', 'in_file')]),
    ])

    # Resample to T1w space
    if 'T1w' in output_spaces:
        ds_bold_t1 = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix=suffix_fmt('T1w', 'preproc'), compress=True),
            name='ds_bold_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        ds_bold_mask_t1 = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix=suffix_fmt('T1w', 'brainmask')),
            name='ds_bold_mask_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_bold_t1, [('source_file', 'source_file'),
                                     ('bold_t1', 'in_file')]),
            (inputnode, ds_bold_mask_t1, [('source_file', 'source_file'),
                                          ('bold_mask_t1', 'in_file')]),
        ])

    # Resample to template (default: MNI)
    if 'template' in output_spaces:
        ds_bold_mni = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix=suffix_fmt(template, 'preproc')),
            name='ds_bold_mni', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_mask_mni = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix=suffix_fmt(template, 'brainmask')),
            name='ds_bold_mask_mni', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_bold_mni, [('source_file', 'source_file'),
                                      ('bold_mni', 'in_file')]),
            (inputnode, ds_bold_mask_mni, [('source_file', 'source_file'),
                                           ('bold_mask_mni', 'in_file')]),
        ])

    if freesurfer:
        ds_bold_aseg_t1 = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix='space-T1w_label-aseg_roi'),
            name='ds_bold_aseg_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_aparc_t1 = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix='space-T1w_label-aparcaseg_roi'),
            name='ds_bold_aparc_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_bold_aseg_t1, [('source_file', 'source_file'),
                                          ('bold_aseg_t1', 'in_file')]),
            (inputnode, ds_bold_aparc_t1, [('source_file', 'source_file'),
                                           ('bold_aparc_t1', 'in_file')]),
        ])

    # fsaverage space
    if freesurfer and any(space.startswith('fs') for space in output_spaces):
        name_surfs = pe.MapNode(GiftiNameSource(
            pattern=r'(?P<LR>[lr])h.(?P<space>\w+).gii', template='space-{space}.{LR}.func'),
            iterfield='in_file', name='name_surfs', mem_gb=DEFAULT_MEMORY_MIN_GB,
            run_without_submitting=True)
        ds_bold_surfs = pe.MapNode(DerivativesDataSink(base_directory=output_dir),
                                   iterfield=['in_file', 'suffix'], name='ds_bold_surfs',
                                   run_without_submitting=True,
                                   mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, name_surfs, [('surfaces', 'in_file')]),
            (inputnode, ds_bold_surfs, [('source_file', 'source_file'),
                                        ('surfaces', 'in_file')]),
            (name_surfs, ds_bold_surfs, [('out_name', 'suffix')]),
        ])

        # CIFTI output
        if cifti_output and 'template' in output_spaces:
            name_cifti = pe.MapNode(
                CiftiNameSource(), iterfield=['variant'], name='name_cifti',
                mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True)
            cifti_bolds = pe.MapNode(DerivativesDataSink(
                base_directory=output_dir, compress=False),
                iterfield=['in_file', 'suffix'], name='cifti_bolds',
                run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
            cifti_key = pe.MapNode(DerivativesDataSink(
                base_directory=output_dir), iterfield=['in_file', 'suffix'],
                name='cifti_key', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([
                (inputnode, name_cifti, [('cifti_variant', 'variant')]),
                (inputnode, cifti_bolds, [('bold_cifti', 'in_file'),
                                          ('source_file', 'source_file')]),
                (name_cifti, cifti_bolds, [('out_name', 'suffix')]),
                (name_cifti, cifti_key, [('out_name', 'suffix')]),
                (inputnode, cifti_key, [('source_file', 'source_file'),
                                        ('cifti_variant_key', 'in_file')]),
            ])

    if use_aroma:
        ds_aroma_noise_ics = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix='AROMAnoiseICs'),
            name="ds_aroma_noise_ics", run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_melodic_mix = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix='MELODICmix'),
            name="ds_melodic_mix", run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_aroma_mni = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix=variant_suffix_fmt(
                template, 'smoothAROMAnonaggr', 'preproc')),
            name='ds_aroma_mni', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, ds_aroma_noise_ics, [('source_file', 'source_file'),
                                             ('aroma_noise_ics', 'in_file')]),
            (inputnode, ds_melodic_mix, [('source_file', 'source_file'),
                                         ('melodic_mix', 'in_file')]),
            (inputnode, ds_aroma_mni, [('source_file', 'source_file'),
                                       ('nonaggr_denoised_file', 'in_file')]),
        ])

    return workflow


def _get_series_len(bold_fname):
    from niworkflows.interfaces.registration import _get_vols_to_discard
    img = nb.load(bold_fname)
    if len(img.shape) < 4:
        return 1

    skip_vols = _get_vols_to_discard(img)

    return img.shape[3] - skip_vols


def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gb = {
        'filesize': bold_size_gb,
        'resampled': bold_size_gb * 4,
        'largemem': bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return bold_tlen, mem_gb


def _get_wf_name(bold_fname):
    """
    Derives the workflow name for supplied BOLD file.

    >>> _get_wf_name('/completely/made/up/path/sub-01_task-nback_bold.nii.gz')
    'func_preproc_task_nback_wf'
    >>> _get_wf_name('/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz')
    'func_preproc_task_nback_run_01_echo_1_wf'
    """
    from nipype.utils.filemanip import split_filename
    fname = split_filename(bold_fname)[1]
    fname_nosub = '_'.join(fname.split("_")[1:])
    # if 'echo' in fname_nosub:
    #     fname_nosub = '_'.join(fname_nosub.split("_echo-")[:1]) + "_bold"
    name = "func_preproc_" + fname_nosub.replace(
        ".", "_").replace(" ", "").replace("-", "_").replace("_bold", "_wf")

    return name
