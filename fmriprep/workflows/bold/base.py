# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Orchestrating the BOLD-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_func_preproc_wf
.. autofunction:: init_func_derivatives_wf

"""

import os
from collections import OrderedDict

import nibabel as nb
from nipype import logging

from nipype.interfaces.fsl import Split as FSLSplit
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.utility import KeySelect
from niworkflows.interfaces.utils import DictMerge


from ...config import DEFAULT_MEMORY_MIN_GB
from ...utils.meepi import combine_meepi_source

from ...interfaces import DerivativesDataSink
from ...interfaces.reports import FunctionalSummary

# BOLD workflows
from .confounds import init_bold_confs_wf, init_carpetplot_wf
from .hmc import init_bold_hmc_wf
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf
from .registration import init_bold_t1_trans_wf, init_bold_reg_wf
from .resampling import (
    init_bold_surf_wf,
    init_bold_std_trans_wf,
    init_bold_preproc_trans_wf,
)
from .outputs import init_func_derivatives_wf, _remove_internal_spaces
from .util import init_bold_reference_wf


LOGGER = logging.getLogger('nipype.workflow')
# FSAVERAGE_DENSITY = {
#     '642': 'fsaverage3',
#     '2562': 'fsaverage4',
#     '10k': 'fsaverage5',
#     '41k': 'fsaverage6',
#     '164k': 'fsaverage7',
# }


def init_func_preproc_wf(
    aroma_melodic_dim,
    bold2t1w_dof,
    bold_file,
    cifti_output,
    debug,
    dummy_scans,
    err_on_aroma_warn,
    fmap_bspline,
    fmap_demean,
    force_syn,
    freesurfer,
    ignore,
    low_mem,
    medial_surface_nan,
    omp_nthreads,
    output_dir,
    regressors_all_comps,
    regressors_dvars_th,
    regressors_fd_th,
    reportlets_dir,
    t2s_coreg,
    target_spaces,
    use_aroma,
    use_bbr,
    use_syn,
    layout=None,
    num_bold=1,
):
    """
    This workflow controls the functional preprocessing stages of *fMRIPrep*.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold import init_func_preproc_wf
            from collections import namedtuple, OrderedDict
            BIDSLayout = namedtuple('BIDSLayout', ['root'])
            wf = init_func_preproc_wf(
                aroma_melodic_dim=-200,
                bold2t1w_dof=9,
                bold_file='/completely/made/up/path/sub-01_task-nback_bold.nii.gz',
                cifti_output=False,
                debug=False,
                dummy_scans=None,
                err_on_aroma_warn=False,
                fmap_bspline=True,
                fmap_demean=True,
                force_syn=True,
                freesurfer=True,
                ignore=[],
                low_mem=False,
                medial_surface_nan=False,
                omp_nthreads=1,
                output_dir='.',
                regressors_all_comps=False,
                regressors_dvars_th=1.5,
                regressors_fd_th=0.5,
                reportlets_dir='.',
                t2s_coreg=False,
                target_spaces=OrderedDict([
                    ('MNI152Lin', {}), ('fsaverage', {'density': '10k'}),
                    ('T1w', {}), ('fsnative', {})]),
                use_aroma=False,
                use_bbr=True,
                use_syn=True,
                layout=BIDSLayout('.'),
                num_bold=1,
            )

    Parameters
    ----------
    aroma_melodic_dim : int
        Maximum number of components identified by MELODIC within ICA-AROMA
        (default is -200, ie. no limitation).
    bold2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for BOLD-T1w registration
    bold_file : str
        BOLD series NIfTI file
    cifti_output : bool
        Generate bold CIFTI file in output spaces
    debug : bool
        Enable debugging outputs
    dummy_scans : int or None
        Number of volumes to consider as non steady state
    err_on_aroma_warn : bool
        Do not crash on ICA-AROMA errors
    fmap_bspline : bool
        **Experimental**: Fit B-Spline field using least-squares
    fmap_demean : bool
        Demean voxel-shift map during unwarp
    force_syn : bool
        **Temporary**: Always run SyN-based SDC
    freesurfer : bool
        Enable FreeSurfer functional registration (bbregister) and resampling
        BOLD series to FreeSurfer surface meshes.
    ignore : list
        Preprocessing steps to skip (may include "slicetiming", "fieldmaps")
    low_mem : bool
        Write uncompressed .nii files in some cases to reduce memory usage
    medial_surface_nan : bool
        Replace medial wall values with NaNs on functional GIFTI files
    omp_nthreads : int
        Maximum number of threads an individual process may use
    output_dir : str
        Directory in which to save derivatives
    regressors_all_comps
        Return all CompCor component time series instead of the top fraction
    regressors_dvars_th
        Criterion for flagging DVARS outliers
    regressors_fd_th
        Criterion for flagging framewise displacement outliers
    reportlets_dir : str
        Absolute path of a directory in which reportlets will be temporarily stored
    t2s_coreg : bool
        For multiecho EPI, use the calculated T2*-map for T2*-driven coregistration
    target_spaces : OrderedDict
        Ordered dictionary where keys are TemplateFlow ID strings (e.g. ``MNI152Lin``,
        ``MNI152NLin6Asym``, ``MNI152NLin2009cAsym``, or ``fsLR``) strings designating
        nonstandard references (e.g. ``T1w`` or ``anat``, ``sbref``, ``run``, etc.),
        or paths pointing to custom templates organized in a TemplateFlow-like structure.
        Values of the dictionary aggregate modifiers (e.g. the value for the key ``MNI152Lin``
        could be ``{'resolution': 2}`` if one wants the resampling to be done on the 2mm
        resolution version of the selected template).
    use_aroma : bool
        Perform ICA-AROMA on MNI-resampled functional series
    use_bbr : bool or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
        When using ``t2s_coreg``, BBR will be enabled by default unless
        explicitly specified otherwise.
    use_syn : bool
        **Experimental**: Enable ANTs SyN-based susceptibility distortion correction (SDC).
        If fieldmaps are present and enabled, this is not run, by default.
    layout : BIDSLayout
        BIDSLayout structure to enable metadata retrieval
    num_bold : int
        Total number of BOLD files that have been set for preprocessing
        (default is 1)

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    t1w_preproc
        Bias-corrected structural template image
    t1w_brain
        Skull-stripped ``t1w_preproc``
    t1w_mask
        Mask of the skull-stripped template image
    t1w_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    t1w_asec
        Segmentation of structural image, done with FreeSurfer.
    t1w_aparc
        Parcellation of structural image, done with FreeSurfer.
    t1w_tpms
        List of tissue probability maps in T1w space
    template
        Name of the template (parametric)
    anat2std_xfm
        ANTs-compatible affine-and-warp transform file (parametric)
    std2anat_xfm
        ANTs-compatible affine-and-warp transform file (inverse) (parametric)
    joint_template
        List of templates to target
    joint_anat2std_xfm
        List of transform files, collated with templates
    joint_std2anat_xfm
        List of inverse transform files, collated with templates
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    t1w2fsnative_xfm
        LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    Outputs
    -------
    bold_t1
        BOLD series, resampled to T1w space
    bold_mask_t1
        BOLD series mask in T1w space
    bold_std
        BOLD series, resampled to template space
    bold_mask_std
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

    See also
    --------
      * :py:func:`~fmriprep.workflows.bold.util.init_bold_reference_wf`
      * :py:func:`~fmriprep.workflows.bold.stc.init_bold_stc_wf`
      * :py:func:`~fmriprep.workflows.bold.hmc.init_bold_hmc_wf`
      * :py:func:`~fmriprep.workflows.bold.t2s.init_bold_t2s_wf`
      * :py:func:`~fmriprep.workflows.bold.registration.init_bold_t1_trans_wf`
      * :py:func:`~fmriprep.workflows.bold.registration.init_bold_reg_wf`
      * :py:func:`~fmriprep.workflows.bold.confounds.init_bold_confounds_wf`
      * :py:func:`~fmriprep.workflows.bold.confounds.init_ica_aroma_wf`
      * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_std_trans_wf`
      * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_preproc_trans_wf`
      * :py:func:`~fmriprep.workflows.bold.resampling.init_bold_surf_wf`
      * :py:func:`~fmriprep.workflows.fieldmap.pepolar.init_pepolar_unwarp_wf`
      * :py:func:`~fmriprep.workflows.fieldmap.init_fmap_estimator_wf`
      * :py:func:`~fmriprep.workflows.fieldmap.init_sdc_unwarp_wf`
      * :py:func:`~fmriprep.workflows.fieldmap.init_nonlinear_sdc_wf`

    """
    from ...config import NONSTANDARD_REFERENCES
    from sdcflows.workflows.base import init_sdc_estimate_wf, fieldmap_wrangler

    # Filter out standard spaces to a separate dict
    std_spaces = OrderedDict([
        (key, modifiers) for key, modifiers in target_spaces.items()
        if key not in NONSTANDARD_REFERENCES])
    volume_std_spaces = OrderedDict([
        (key, modifiers) for key, modifiers in std_spaces.items()
        if not key.startswith('fs')])

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

    sbref_file = None
    # For doc building purposes
    if not hasattr(layout, 'parse_file_entities'):
        LOGGER.log(25, 'No valid layout: building empty workflow.')
        metadata = {
            'RepetitionTime': 2.0,
            'SliceTiming': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'PhaseEncodingDirection': 'j',
        }
        fmaps = {
            'phasediff': [{
                'phases': [
                    ('sub-03/ses-2/fmap/sub-03_ses-2_run-1_phasediff.nii.gz', {
                        'EchoTime1': 0.0, 'EchoTime2': 0.00246
                    })
                ],
                'magnitude': [
                    ('sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude1.nii.gz', {}),
                    ('sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude2.nii.gz', {})]
            }]
        }
        run_stc = True
        multiecho = False
    else:
        # Find associated sbref, if possible
        entities = layout.parse_file_entities(ref_file)
        entities['suffix'] = 'sbref'
        entities['extension'] = ['nii', 'nii.gz']  # Overwrite extensions
        files = layout.get(return_type='file', **entities)
        refbase = os.path.basename(ref_file)
        if 'sbref' in ignore:
            LOGGER.info("Single-band reference files ignored.")
        elif files and multiecho:
            LOGGER.warning("Single-band reference found, but not supported in "
                           "multi-echo workflows at this time. Ignoring.")
        elif files:
            sbref_file = files[0]
            sbbase = os.path.basename(sbref_file)
            if len(files) > 1:
                LOGGER.warning(
                    "Multiple single-band reference files found for {}; using "
                    "{}".format(refbase, sbbase))
            else:
                LOGGER.log(25, "Using single-band reference file {}".format(sbbase))
        else:
            LOGGER.log(25, "No single-band-reference found for {}".format(refbase))

        metadata = layout.get_metadata(ref_file)

        # Find fieldmaps. Options: (phase1|phase2|phasediff|epi|fieldmap|syn)
        fmaps = None
        if 'fieldmaps' not in ignore:
            fmaps = fieldmap_wrangler(layout, ref_file, use_syn=use_syn, force_syn=force_syn)
        elif use_syn or force_syn:
            # If fieldmaps are not enabled, activate SyN-SDC in unforced (False) mode
            fmaps = {'syn': False}

        # Short circuits: (True and True and (False or 'TooShort')) == 'TooShort'
        run_stc = (bool(metadata.get("SliceTiming")) and
                   'slicetiming' not in ignore and
                   (_get_series_len(ref_file) > 4 or "TooShort"))

    # Check if MEEPI for T2* coregistration target
    if t2s_coreg and not multiecho:
        LOGGER.warning("No multiecho BOLD images found for T2* coregistration. "
                       "Using standard EPI-T1 coregistration.")
        t2s_coreg = False

    # By default, force-bbr for t2s_coreg unless user specifies otherwise
    if t2s_coreg and use_bbr is None:
        use_bbr = True

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
and co-registrations to anatomical and output spaces).
Gridded (volumetric) resamplings were performed using `antsApplyTransforms` (ANTs),
configured with Lanczos interpolation to minimize the smoothing
effects of other kernels [@lanczos].
Non-gridded (surface) resamplings were performed using `mri_vol2surf`
(FreeSurfer).
"""

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_file', 'subjects_dir', 'subject_id',
                't1w_preproc', 't1w_brain', 't1w_mask', 't1w_dseg', 't1w_tpms',
                't1w_aseg', 't1w_aparc',
                'anat2std_xfm', 'std2anat_xfm', 'template',
                'joint_anat2std_xfm', 'joint_std2anat_xfm', 'joint_template',
                't1w2fsnative_xfm', 'fsnative2t1w_xfm']),
        name='inputnode')
    inputnode.inputs.bold_file = bold_file
    if sbref_file is not None:
        from niworkflows.interfaces.images import ValidateImage
        val_sbref = pe.Node(ValidateImage(in_file=sbref_file), name='val_sbref')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_t1', 'bold_t1_ref', 'bold_mask_t1', 'bold_aseg_t1', 'bold_aparc_t1',
                'bold_std', 'bold_std_ref', 'bold_mask_std', 'bold_aseg_std', 'bold_aparc_std',
                'bold_native', 'bold_cifti', 'cifti_variant', 'cifti_metadata', 'cifti_density',
                'surfaces', 'confounds', 'aroma_noise_ics', 'melodic_mix', 'nonaggr_denoised_file',
                'confounds_metadata']),
        name='outputnode')

    # BOLD buffer: an identity used as a pointer to either the original BOLD
    # or the STC'ed one for further use.
    boldbuffer = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='boldbuffer')

    summary = pe.Node(
        FunctionalSummary(
            slice_timing=run_stc,
            registration=('FSL', 'FreeSurfer')[freesurfer],
            registration_dof=bold2t1w_dof,
            pe_direction=metadata.get("PhaseEncodingDirection"),
            tr=metadata.get("RepetitionTime")),
        name='summary', mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True)
    summary.inputs.dummy_scans = dummy_scans

    # CIFTI output
    # cifti_spaces = {'fsLR'} if 'fsLR' in output_spaces else \
    #     set(output_spaces.keys()).intersection(('fsaverage5', 'fsaverage6'))
    # fsaverage_den = output_spaces.get('fsaverage', {}).get('den')
    # if fsaverage_den and 'fsLR' not in cifti_spaces:
    #     cifti_spaces.add(FSAVERAGE_DENSITY[fsaverage_den])
    cifti_spaces = ('fsLR',) if 'fsLR' in target_spaces else None
    cifti_output = cifti_output and cifti_spaces
    fslr_density = target_spaces.get('fsLR', {}).get('den')
    func_derivatives_wf = init_func_derivatives_wf(
        bids_root=layout.root,
        cifti_output=cifti_output,
        freesurfer=freesurfer,
        metadata=metadata,
        output_dir=output_dir,
        output_spaces=target_spaces,
        standard_spaces=list(_remove_internal_spaces(std_spaces).keys()),
        use_aroma=use_aroma,
        fslr_density=fslr_density,
    )

    workflow.connect([
        (outputnode, func_derivatives_wf, [
            ('bold_t1', 'inputnode.bold_t1'),
            ('bold_t1_ref', 'inputnode.bold_t1_ref'),
            ('bold_aseg_t1', 'inputnode.bold_aseg_t1'),
            ('bold_aparc_t1', 'inputnode.bold_aparc_t1'),
            ('bold_mask_t1', 'inputnode.bold_mask_t1'),
            ('bold_native', 'inputnode.bold_native'),
            ('confounds', 'inputnode.confounds'),
            ('surfaces', 'inputnode.surfaces'),
            ('aroma_noise_ics', 'inputnode.aroma_noise_ics'),
            ('melodic_mix', 'inputnode.melodic_mix'),
            ('nonaggr_denoised_file', 'inputnode.nonaggr_denoised_file'),
            ('bold_cifti', 'inputnode.bold_cifti'),
            ('cifti_variant', 'inputnode.cifti_variant'),
            ('cifti_metadata', 'inputnode.cifti_metadata'),
            ('cifti_density', 'inputnode.cifti_density'),
            ('confounds_metadata', 'inputnode.confounds_metadata'),
        ]),
    ])

    # Generate a tentative boldref
    bold_reference_wf = init_bold_reference_wf(omp_nthreads=omp_nthreads)
    bold_reference_wf.inputs.inputnode.dummy_scans = dummy_scans
    if sbref_file is not None:
        workflow.connect([
            (val_sbref, bold_reference_wf, [('out_file', 'inputnode.sbref_file')]),
        ])

    # Top-level BOLD splitter
    bold_split = pe.Node(FSLSplit(dimension='t'), name='bold_split',
                         mem_gb=mem_gb['filesize'] * 3)

    # HMC on the BOLD
    bold_hmc_wf = init_bold_hmc_wf(name='bold_hmc_wf',
                                   mem_gb=mem_gb['filesize'],
                                   omp_nthreads=omp_nthreads)

    # calculate BOLD registration to T1w
    bold_reg_wf = init_bold_reg_wf(name='bold_reg_wf',
                                   freesurfer=freesurfer,
                                   use_bbr=use_bbr,
                                   bold2t1w_dof=bold2t1w_dof,
                                   mem_gb=mem_gb['resampled'],
                                   omp_nthreads=omp_nthreads,
                                   use_compression=False)

    # apply BOLD registration to T1w
    bold_t1_trans_wf = init_bold_t1_trans_wf(name='bold_t1_trans_wf',
                                             freesurfer=freesurfer,
                                             use_fieldwarp=bool(fmaps),
                                             multiecho=multiecho,
                                             mem_gb=mem_gb['resampled'],
                                             omp_nthreads=omp_nthreads,
                                             use_compression=False)

    # get confounds
    bold_confounds_wf = init_bold_confs_wf(
        mem_gb=mem_gb['largemem'],
        metadata=metadata,
        regressors_all_comps=regressors_all_comps,
        regressors_fd_th=regressors_fd_th,
        regressors_dvars_th=regressors_dvars_th,
        name='bold_confounds_wf')
    bold_confounds_wf.get_node('inputnode').inputs.t1_transform_flags = [False]

    # Apply transforms in 1 shot
    # Only use uncompressed output if AROMA is to be run
    bold_bold_trans_wf = init_bold_preproc_trans_wf(
        mem_gb=mem_gb['resampled'],
        omp_nthreads=omp_nthreads,
        use_compression=not low_mem,
        use_fieldwarp=bool(fmaps),
        name='bold_bold_trans_wf'
    )
    bold_bold_trans_wf.inputs.inputnode.name_source = ref_file

    # SLICE-TIME CORRECTION (or bypass) #############################################
    if run_stc is True:  # bool('TooShort') == True, so check True explicitly
        bold_stc_wf = init_bold_stc_wf(name='bold_stc_wf', metadata=metadata)
        workflow.connect([
            (bold_reference_wf, bold_stc_wf, [
                ('outputnode.skip_vols', 'inputnode.skip_vols')]),
            (bold_stc_wf, boldbuffer, [('outputnode.stc_file', 'bold_file')]),
        ])
        if not multiecho:
            workflow.connect([
                (bold_reference_wf, bold_stc_wf, [
                    ('outputnode.bold_file', 'inputnode.bold_file')])])
        else:  # for meepi, iterate through stc_wf for all workflows
            meepi_echos = boldbuffer.clone(name='meepi_echos')
            meepi_echos.iterables = ('bold_file', bold_file)
            workflow.connect([
                (meepi_echos, bold_stc_wf, [('bold_file', 'inputnode.bold_file')])])
    elif not multiecho:  # STC is too short or False
        # bypass STC from original BOLD to the splitter through boldbuffer
        workflow.connect([
            (bold_reference_wf, boldbuffer, [('outputnode.bold_file', 'bold_file')])])
    else:
        # for meepi, iterate over all meepi echos to boldbuffer
        boldbuffer.iterables = ('bold_file', bold_file)

    # SDC (SUSCEPTIBILITY DISTORTION CORRECTION) or bypass ##########################
    bold_sdc_wf = init_sdc_estimate_wf(fmaps, metadata,
                                       omp_nthreads=omp_nthreads, debug=debug)

    # MULTI-ECHO EPI DATA #############################################
    if multiecho:
        from .util import init_skullstrip_bold_wf
        skullstrip_bold_wf = init_skullstrip_bold_wf(name='skullstrip_bold_wf')

        inputnode.inputs.bold_file = ref_file  # Replace reference w first echo

        join_echos = pe.JoinNode(niu.IdentityInterface(fields=['bold_files']),
                                 joinsource=('meepi_echos' if run_stc is True else 'boldbuffer'),
                                 joinfield=['bold_files'],
                                 name='join_echos')

        # create optimal combination, adaptive T2* map
        bold_t2s_wf = init_bold_t2s_wf(echo_times=tes,
                                       mem_gb=mem_gb['resampled'],
                                       omp_nthreads=omp_nthreads,
                                       t2s_coreg=t2s_coreg,
                                       name='bold_t2smap_wf')

        workflow.connect([
            (skullstrip_bold_wf, join_echos, [
                ('outputnode.skull_stripped_file', 'bold_files')]),
            (join_echos, bold_t2s_wf, [
                ('bold_files', 'inputnode.bold_file')]),
        ])

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
        (bold_reference_wf, summary, [
            ('outputnode.algo_dummy_scans', 'algo_dummy_scans')]),
        # EPI-T1 registration workflow
        (inputnode, bold_reg_wf, [
            ('t1w_brain', 'inputnode.t1w_brain'),
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            # Undefined if --fs-no-reconall, but this is safe
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm')]),
        (inputnode, bold_t1_trans_wf, [
            ('bold_file', 'inputnode.name_source'),
            ('t1w_brain', 'inputnode.t1w_brain'),
            ('t1w_mask', 'inputnode.t1w_mask'),
            ('t1w_aseg', 'inputnode.t1w_aseg'),
            ('t1w_aparc', 'inputnode.t1w_aparc')]),
        # unused if multiecho, but this is safe
        (bold_hmc_wf, bold_t1_trans_wf, [('outputnode.xforms', 'inputnode.hmc_xforms')]),
        (bold_reg_wf, bold_t1_trans_wf, [
            ('outputnode.itk_bold_to_t1', 'inputnode.itk_bold_to_t1')]),
        (bold_t1_trans_wf, outputnode, [('outputnode.bold_t1', 'bold_t1'),
                                        ('outputnode.bold_t1_ref', 'bold_t1_ref'),
                                        ('outputnode.bold_aseg_t1', 'bold_aseg_t1'),
                                        ('outputnode.bold_aparc_t1', 'bold_aparc_t1')]),
        (bold_reg_wf, summary, [('outputnode.fallback', 'fallback')]),
        # SDC (or pass-through workflow)
        (inputnode, bold_sdc_wf, [
            ('t1w_brain', 'inputnode.t1w_brain')]),
        (bold_reference_wf, bold_sdc_wf, [
            ('outputnode.ref_image', 'inputnode.epi_file'),
            ('outputnode.ref_image_brain', 'inputnode.epi_brain'),
            ('outputnode.bold_mask', 'inputnode.epi_mask')]),
        (bold_sdc_wf, bold_t1_trans_wf, [
            ('outputnode.out_warp', 'inputnode.fieldwarp')]),
        (bold_sdc_wf, bold_bold_trans_wf, [
            ('outputnode.out_warp', 'inputnode.fieldwarp'),
            ('outputnode.epi_mask', 'inputnode.bold_mask')]),
        (bold_sdc_wf, summary, [('outputnode.method', 'distortion_correction')]),
        # Connect bold_confounds_wf
        (inputnode, bold_confounds_wf, [('t1w_tpms', 'inputnode.t1w_tpms'),
                                        ('t1w_mask', 'inputnode.t1w_mask')]),
        (bold_hmc_wf, bold_confounds_wf, [
            ('outputnode.movpar_file', 'inputnode.movpar_file')]),
        (bold_reg_wf, bold_confounds_wf, [
            ('outputnode.itk_t1_to_bold', 'inputnode.t1_bold_xform')]),
        (bold_reference_wf, bold_confounds_wf, [
            ('outputnode.skip_vols', 'inputnode.skip_vols')]),
        (bold_confounds_wf, outputnode, [
            ('outputnode.confounds_file', 'confounds'),
        ]),
        (bold_confounds_wf, outputnode, [
            ('outputnode.confounds_metadata', 'confounds_metadata'),
        ]),
        # Connect bold_bold_trans_wf
        (bold_split, bold_bold_trans_wf, [
            ('out_files', 'inputnode.bold_file')]),
        (bold_hmc_wf, bold_bold_trans_wf, [
            ('outputnode.xforms', 'inputnode.hmc_xforms')]),
        # Summary
        (outputnode, summary, [('confounds', 'confounds_file')]),
    ])

    if not t2s_coreg:
        workflow.connect([
            (bold_sdc_wf, bold_reg_wf, [
                ('outputnode.epi_brain', 'inputnode.ref_bold_brain')]),
            (bold_sdc_wf, bold_t1_trans_wf, [
                ('outputnode.epi_brain', 'inputnode.ref_bold_brain'),
                ('outputnode.epi_mask', 'inputnode.ref_bold_mask')]),
        ])
    else:
        workflow.connect([
            # For t2s_coreg, replace EPI-to-T1w registration inputs
            (bold_t2s_wf, bold_reg_wf, [
                ('outputnode.bold_ref_brain', 'inputnode.ref_bold_brain')]),
            (bold_t2s_wf, bold_t1_trans_wf, [
                ('outputnode.bold_ref_brain', 'inputnode.ref_bold_brain'),
                ('outputnode.bold_mask', 'inputnode.ref_bold_mask')]),
        ])

    # for standard EPI data, pass along correct file
    if not multiecho:
        workflow.connect([
            (inputnode, func_derivatives_wf, [
                ('bold_file', 'inputnode.source_file')]),
            (bold_bold_trans_wf, bold_confounds_wf, [
                ('outputnode.bold', 'inputnode.bold'),
                ('outputnode.bold_mask', 'inputnode.bold_mask')]),
            (bold_split, bold_t1_trans_wf, [
                ('out_files', 'inputnode.bold_split')]),
        ])
    else:  # for meepi, create and use optimal combination
        workflow.connect([
            # update name source for optimal combination
            (inputnode, func_derivatives_wf, [
                (('bold_file', combine_meepi_source), 'inputnode.source_file')]),
            (bold_bold_trans_wf, skullstrip_bold_wf, [
                ('outputnode.bold', 'inputnode.in_file')]),
            (bold_t2s_wf, bold_confounds_wf, [
                ('outputnode.bold', 'inputnode.bold'),
                ('outputnode.bold_mask', 'inputnode.bold_mask')]),
            (bold_t2s_wf, bold_t1_trans_wf, [
                ('outputnode.bold', 'inputnode.bold_split')]),
        ])

    if fmaps:
        from sdcflows.workflows.outputs import init_sdc_unwarp_report_wf
        # Report on BOLD correction
        fmap_unwarp_report_wf = init_sdc_unwarp_report_wf()
        workflow.connect([
            (inputnode, fmap_unwarp_report_wf, [
                ('t1w_dseg', 'inputnode.in_seg')]),
            (bold_reference_wf, fmap_unwarp_report_wf, [
                ('outputnode.ref_image', 'inputnode.in_pre')]),
            (bold_reg_wf, fmap_unwarp_report_wf, [
                ('outputnode.itk_t1_to_bold', 'inputnode.in_xfm')]),
            (bold_sdc_wf, fmap_unwarp_report_wf, [
                ('outputnode.epi_corrected', 'inputnode.in_post')]),
        ])

        # Overwrite ``out_path_base`` of unwarping DataSinks
        for node in fmap_unwarp_report_wf.list_node_names():
            if node.split('.')[-1].startswith('ds_'):
                fmap_unwarp_report_wf.get_node(node).interface.out_path_base = 'fmriprep'

        for node in bold_sdc_wf.list_node_names():
            if node.split('.')[-1].startswith('ds_'):
                bold_sdc_wf.get_node(node).interface.out_path_base = 'fmriprep'

        if 'syn' in fmaps:
            sdc_select_std = pe.Node(
                KeySelect(fields=['std2anat_xfm']),
                name='sdc_select_std', run_without_submitting=True)
            sdc_select_std.inputs.key = 'MNI152NLin2009cAsym'
            workflow.connect([
                (inputnode, sdc_select_std, [('joint_std2anat_xfm', 'std2anat_xfm'),
                                             ('joint_template', 'keys')]),
                (sdc_select_std, bold_sdc_wf, [('std2anat_xfm', 'inputnode.std2anat_xfm')]),
            ])

        if fmaps.get('syn') is True:  # SyN forced
            syn_unwarp_report_wf = init_sdc_unwarp_report_wf(
                name='syn_unwarp_report_wf', forcedsyn=True)
            workflow.connect([
                (inputnode, syn_unwarp_report_wf, [
                    ('t1w_dseg', 'inputnode.in_seg')]),
                (bold_reference_wf, syn_unwarp_report_wf, [
                    ('outputnode.ref_image', 'inputnode.in_pre')]),
                (bold_reg_wf, syn_unwarp_report_wf, [
                    ('outputnode.itk_t1_to_bold', 'inputnode.in_xfm')]),
                (bold_sdc_wf, syn_unwarp_report_wf, [
                    ('outputnode.syn_ref', 'inputnode.in_post')]),
            ])

            # Overwrite ``out_path_base`` of unwarping DataSinks
            for node in syn_unwarp_report_wf.list_node_names():
                if node.split('.')[-1].startswith('ds_'):
                    syn_unwarp_report_wf.get_node(node).interface.out_path_base = 'fmriprep'

    # Map final BOLD mask into T1w space (if required)
    if 'T1w' in target_spaces or 'anat' in target_spaces:
        from niworkflows.interfaces.fixes import (
            FixHeaderApplyTransforms as ApplyTransforms
        )

        boldmask_to_t1w = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', float=True),
            name='boldmask_to_t1w', mem_gb=0.1
        )
        workflow.connect([
            (bold_reg_wf, boldmask_to_t1w, [
                ('outputnode.itk_bold_to_t1', 'transforms')]),
            (bold_t1_trans_wf, boldmask_to_t1w, [
                ('outputnode.bold_mask_t1', 'reference_image')]),
            (bold_bold_trans_wf if not multiecho else bold_t2s_wf, boldmask_to_t1w, [
                ('outputnode.bold_mask', 'input_image')]),
            (boldmask_to_t1w, outputnode, [
                ('output_image', 'bold_mask_t1')]),
        ])

    if set(['func', 'run', 'bold', 'boldref', 'sbref']).intersection(target_spaces):
        workflow.connect([
            (bold_bold_trans_wf, outputnode, [
                ('outputnode.bold', 'bold_native')]),
            (bold_bold_trans_wf, func_derivatives_wf, [
                ('outputnode.bold_ref', 'inputnode.bold_native_ref'),
                ('outputnode.bold_mask', 'inputnode.bold_mask_native')]),
        ])

    if volume_std_spaces:
        # Apply transforms in 1 shot
        # Only use uncompressed output if AROMA is to be run
        bold_std_trans_wf = init_bold_std_trans_wf(
            freesurfer=freesurfer,
            mem_gb=mem_gb['resampled'],
            omp_nthreads=omp_nthreads,
            standard_spaces=volume_std_spaces,
            name='bold_std_trans_wf',
            use_compression=not low_mem,
            use_fieldwarp=bool(fmaps),
        )
        workflow.connect([
            (inputnode, bold_std_trans_wf, [
                ('joint_template', 'inputnode.templates'),
                ('joint_anat2std_xfm', 'inputnode.anat2std_xfm'),
                ('bold_file', 'inputnode.name_source'),
                ('t1w_aseg', 'inputnode.bold_aseg'),
                ('t1w_aparc', 'inputnode.bold_aparc')]),
            (bold_hmc_wf, bold_std_trans_wf, [
                ('outputnode.xforms', 'inputnode.hmc_xforms')]),
            (bold_reg_wf, bold_std_trans_wf, [
                ('outputnode.itk_bold_to_t1', 'inputnode.itk_bold_to_t1')]),
            (bold_bold_trans_wf if not multiecho else bold_t2s_wf, bold_std_trans_wf, [
                ('outputnode.bold_mask', 'inputnode.bold_mask')]),
            (bold_sdc_wf, bold_std_trans_wf, [
                ('outputnode.out_warp', 'inputnode.fieldwarp')]),
            (bold_std_trans_wf, outputnode, [('outputnode.bold_std', 'bold_std'),
                                             ('outputnode.bold_std_ref', 'bold_std_ref'),
                                             ('outputnode.bold_mask_std', 'bold_mask_std')]),
        ])

        if freesurfer:
            workflow.connect([
                (bold_std_trans_wf, func_derivatives_wf, [
                    ('outputnode.bold_aseg_std', 'inputnode.bold_aseg_std'),
                    ('outputnode.bold_aparc_std', 'inputnode.bold_aparc_std'),
                ]),
                (bold_std_trans_wf, outputnode, [
                    ('outputnode.bold_aseg_std', 'bold_aseg_std'),
                    ('outputnode.bold_aparc_std', 'bold_aparc_std')]),
            ])

        if 'MNI152NLin2009cAsym' in std_spaces:
            # Extract out the 'MNI152NLin2009cAsym' transform from normalizations
            carpetplot_select_std = pe.Node(
                KeySelect(fields=['std2anat_xfm'], key='MNI152NLin2009cAsym'),
                name='carpetplot_select_std', run_without_submitting=True)

            carpetplot_wf = init_carpetplot_wf(
                mem_gb=mem_gb['resampled'],
                metadata=metadata,
                name='carpetplot_wf')
            workflow.connect([
                (inputnode, carpetplot_select_std, [
                    ('joint_std2anat_xfm', 'std2anat_xfm'),
                    ('joint_template', 'keys')]),
                (carpetplot_select_std, carpetplot_wf, [
                    ('std2anat_xfm', 'inputnode.std2anat_xfm')]),
                (bold_bold_trans_wf if not multiecho else bold_t2s_wf, carpetplot_wf, [
                    ('outputnode.bold', 'inputnode.bold'),
                    ('outputnode.bold_mask', 'inputnode.bold_mask')]),
                (bold_reg_wf, carpetplot_wf, [
                    ('outputnode.itk_t1_to_bold', 'inputnode.t1_bold_xform')]),
                (bold_confounds_wf, carpetplot_wf, [
                    ('outputnode.confounds_file', 'inputnode.confounds_file')]),
            ])

        if not multiecho:
            workflow.connect([
                (bold_split, bold_std_trans_wf, [
                    ('out_files', 'inputnode.bold_split')])
            ])
        else:
            split_opt_comb = bold_split.clone(name='split_opt_comb')
            workflow.connect([
                (bold_t2s_wf, split_opt_comb, [
                    ('outputnode.bold', 'in_file')]),
                (split_opt_comb, bold_std_trans_wf, [
                    ('out_files', 'inputnode.bold_split')
                ])
            ])

        # Artifacts resampled in MNI space can only be sinked if they
        # were actually generated. See #1348.
        # Uses the parameterized outputnode to generate all outputs
        workflow.connect([
            (bold_std_trans_wf, func_derivatives_wf, [
                ('outputnode.templates', 'inputnode.template'),
                ('outputnode.bold_std_ref', 'inputnode.bold_std_ref'),
                ('outputnode.bold_std', 'inputnode.bold_std'),
                ('outputnode.bold_mask_std', 'inputnode.bold_mask_std'),
            ]),
        ])

        if use_aroma and 'MNI152NLin6Asym' in std_spaces:  # ICA-AROMA workflow
            from .confounds import init_ica_aroma_wf

            ica_aroma_wf = init_ica_aroma_wf(
                metadata=metadata,
                mem_gb=mem_gb['resampled'],
                omp_nthreads=omp_nthreads,
                use_fieldwarp=bool(fmaps),
                err_on_aroma_warn=err_on_aroma_warn,
                aroma_melodic_dim=aroma_melodic_dim,
                name='ica_aroma_wf')

            join = pe.Node(niu.Function(output_names=["out_file"],
                                        function=_to_join),
                           name='aroma_confounds')

            mrg_conf_metadata = pe.Node(niu.Merge(2), name='merge_confound_metadata',
                                        run_without_submitting=True)
            mrg_conf_metadata2 = pe.Node(DictMerge(), name='merge_confound_metadata2',
                                         run_without_submitting=True)
            workflow.disconnect([
                (bold_confounds_wf, outputnode, [
                    ('outputnode.confounds_file', 'confounds'),
                ]),
                (bold_confounds_wf, outputnode, [
                    ('outputnode.confounds_metadata', 'confounds_metadata'),
                ]),
            ])
            workflow.connect([
                (bold_std_trans_wf, ica_aroma_wf, [
                    ('outputnode.bold_std', 'inputnode.bold_std'),
                    ('outputnode.bold_mask_std', 'inputnode.bold_mask_std'),
                    ('outputnode.templates', 'inputnode.templates')]),
                (inputnode, ica_aroma_wf, [
                    ('bold_file', 'inputnode.name_source')]),
                (bold_hmc_wf, ica_aroma_wf, [
                    ('outputnode.movpar_file', 'inputnode.movpar_file')]),
                (bold_reference_wf, ica_aroma_wf, [
                    ('outputnode.skip_vols', 'inputnode.skip_vols')]),
                (bold_confounds_wf, join, [
                    ('outputnode.confounds_file', 'in_file')]),
                (bold_confounds_wf, mrg_conf_metadata,
                    [('outputnode.confounds_metadata', 'in1')]),
                (ica_aroma_wf, join,
                    [('outputnode.aroma_confounds', 'join_file')]),
                (ica_aroma_wf, mrg_conf_metadata,
                    [('outputnode.aroma_metadata', 'in2')]),
                (mrg_conf_metadata, mrg_conf_metadata2, [('out', 'in_dicts')]),
                (ica_aroma_wf, outputnode,
                    [('outputnode.aroma_noise_ics', 'aroma_noise_ics'),
                     ('outputnode.melodic_mix', 'melodic_mix'),
                     ('outputnode.nonaggr_denoised_file', 'nonaggr_denoised_file')]),
                (join, outputnode, [('out_file', 'confounds')]),
                (mrg_conf_metadata2, outputnode, [('out_dict', 'confounds_metadata')]),
            ])

    # SURFACES ##################################################################################
    surface_spaces = [space for space in target_spaces.keys() if space.startswith('fs')]
    if freesurfer and surface_spaces:
        LOGGER.log(25, 'Creating BOLD surface-sampling workflow.')
        bold_surf_wf = init_bold_surf_wf(mem_gb=mem_gb['resampled'],
                                         output_spaces=surface_spaces,
                                         medial_surface_nan=medial_surface_nan,
                                         fslr_density=fslr_density,
                                         name='bold_surf_wf')
        workflow.connect([
            (inputnode, bold_surf_wf, [
                ('t1w_preproc', 'inputnode.t1w_preproc'),
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id'),
                ('t1w2fsnative_xfm', 'inputnode.t1w2fsnative_xfm')]),
            (bold_t1_trans_wf, bold_surf_wf, [('outputnode.bold_t1', 'inputnode.source_file')]),
            (bold_surf_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
        ])

        if cifti_output:
            from niworkflows.interfaces.cifti import GenerateCifti
            bold_surf_wf.__desc__ += """\
*Grayordinates* files [@hcppipelines], which combine surface-sampled
data and volume-sampled data, were also generated.
"""
            cifti_volume = "MNI152NLin6Asym" if 'fsLR' in cifti_spaces else "MNI152NLin2009cAsym"
            select_std = pe.Node(KeySelect(fields=['bold_std']),
                                 name='select_std', run_without_submitting=True)
            select_std.inputs.key = cifti_volume

            order_surfs = pe.Node(niu.Function(function=_order_surfs,
                                               output_names=["surface_files"]),
                                  name='order_surfs', run_without_submitting=True)
            order_surfs.inputs.targets = list(cifti_spaces)

            gen_cifti = pe.MapNode(GenerateCifti(), iterfield=["surface_target", "surface_bolds"],
                                   name="gen_cifti")
            gen_cifti.inputs.TR = metadata.get("RepetitionTime")
            gen_cifti.inputs.surface_target = list(cifti_spaces)
            if fslr_density:
                gen_cifti.inputs.surface_density = fslr_density

            workflow.connect([
                (bold_std_trans_wf, select_std, [
                    ('outputnode.templates', 'keys'),
                    ('outputnode.bold_std', 'bold_std')]),
                (bold_surf_wf, order_surfs, [('outputnode.surfaces', 'in_surfs')]),
                (order_surfs, gen_cifti, [('surface_files', 'surface_bolds')]),
                (inputnode, gen_cifti, [('subjects_dir', 'subjects_dir')]),
                (select_std, gen_cifti, [
                    ('bold_std', 'bold_file'),
                    ('key', 'volume_target')]),
                (gen_cifti, outputnode, [('out_file', 'bold_cifti'),
                                         ('variant', 'cifti_variant'),
                                         ('out_metadata', 'cifti_metadata'),
                                         ('density', 'cifti_density')]),
            ])

    # REPORTING ############################################################
    ds_report_summary = pe.Node(
        DerivativesDataSink(desc='summary', keep_dtype=True),
        name='ds_report_summary', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    ds_report_validation = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            desc='validation', keep_dtype=True),
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
    Derive the workflow name for supplied BOLD file.

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


def _to_join(in_file, join_file):
    """Join two tsv files if the join_file is not ``None``."""
    from niworkflows.interfaces.utils import JoinTSVColumns
    if join_file is None:
        return in_file
    res = JoinTSVColumns(in_file=in_file, join_file=join_file).run()
    return res.outputs.out_file


def _order_surfs(targets, in_surfs):
    """Reorder list of surface_files into [L,R] sub-lists"""
    surface_files = []
    targets = targets if 'fsLR' not in targets else ('fsLR',)
    for target in targets:
        target_files = [f for f in in_surfs if f.endswith("{}.gii".format(target))]
        surface_files.append(target_files)
    return surface_files
