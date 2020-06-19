# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Orchestrating the BOLD-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_func_preproc_wf
.. autofunction:: init_func_derivatives_wf

"""
from ... import config

import os

import nibabel as nb
from nipype.interfaces.fsl import Split as FSLSplit
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.utils.connections import pop_file, listify


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
from .outputs import init_func_derivatives_wf


def init_func_preproc_wf(bold_file):
    """
    This workflow controls the functional preprocessing stages of *fMRIPrep*.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.bold.base import init_func_preproc_wf
            with mock_config():
                bold_file = config.execution.bids_dir / 'sub-01' / 'func' \
                    / 'sub-01_task-mixedgamblestask_run-01_bold.nii.gz'
                wf = init_func_preproc_wf(str(bold_file))

    Parameters
    ----------
    bold_file
        BOLD series NIfTI file

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    t1w_preproc
        Bias-corrected structural template image
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
        List of templates to target
    anat2std_xfm
        List of transform files, collated with templates
    std2anat_xfm
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

    See Also
    --------

    * :py:func:`~niworkflows.func.util.init_bold_reference_wf`
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
    * :py:func:`~sdcflows.workflows.fmap.init_fmap_wf`
    * :py:func:`~sdcflows.workflows.pepolar.init_pepolar_unwarp_wf`
    * :py:func:`~sdcflows.workflows.phdiff.init_phdiff_wf`
    * :py:func:`~sdcflows.workflows.syn.init_syn_sdc_wf`
    * :py:func:`~sdcflows.workflows.unwarp.init_sdc_unwarp_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.func.util import init_bold_reference_wf
    from niworkflows.interfaces.nibabel import ApplyMask
    from niworkflows.interfaces.utility import KeySelect
    from niworkflows.interfaces.utils import DictMerge
    from sdcflows.workflows.base import init_sdc_estimate_wf, fieldmap_wrangler

    mem_gb = {'filesize': 1, 'resampled': 1, 'largemem': 1}
    bold_tlen = 10

    # Have some options handy
    omp_nthreads = config.nipype.omp_nthreads
    freesurfer = config.workflow.run_reconall
    spaces = config.workflow.spaces
    output_dir = str(config.execution.output_dir)

    # Extract BIDS entities and metadata from BOLD file(s)
    entities = extract_entities(bold_file)
    layout = config.execution.layout

    # Take first file as reference
    ref_file = pop_file(bold_file)
    metadata = layout.get_metadata(ref_file)

    echo_idxs = listify(entities.get("echo", []))
    multiecho = len(echo_idxs) > 2
    if len(echo_idxs) == 1:
        config.loggers.warning(
            f"Running a single echo <{ref_file}> from a seemingly multi-echo dataset."
        )
        bold_file = ref_file  # Just in case - drop the list

    if len(echo_idxs) == 2:
        raise RuntimeError(
            "Multi-echo processing requires at least three different echos (found two)."
        )

    if multiecho:
        # Drop echo entity for future queries, have a boolean shorthand
        entities.pop("echo", None)
        # reorder echoes from shortest to largest
        tes, bold_file = zip(*sorted([
            (layout.get_metadata(bf)["EchoTime"], bf) for bf in bold_file
        ]))
        ref_file = bold_file[0]  # Reset reference to be the shortest TE

    if os.path.isfile(ref_file):
        bold_tlen, mem_gb = _create_mem_gb(ref_file)

    wf_name = _get_wf_name(ref_file)
    config.loggers.workflow.debug(
        'Creating bold processing workflow for <%s> (%.2f GB / %d TRs). '
        'Memory resampled/largemem=%.2f/%.2f GB.',
        ref_file, mem_gb['filesize'], bold_tlen, mem_gb['resampled'], mem_gb['largemem'])

    # Find associated sbref, if possible
    entities['suffix'] = 'sbref'
    entities['extension'] = ['nii', 'nii.gz']  # Overwrite extensions
    sbref_files = layout.get(return_type='file', **entities)

    sbref_msg = f"No single-band-reference found for {os.path.basename(ref_file)}."
    if sbref_files and 'sbref' in config.workflow.ignore:
        sbref_msg = "Single-band reference file(s) found and ignored."
    elif sbref_files:
        sbref_msg = "Using single-band reference file(s) {}.".format(
            ','.join([os.path.basename(sbf) for sbf in sbref_files]))
    config.loggers.workflow.info(sbref_msg)

    # Find fieldmaps. Options: (phase1|phase2|phasediff|epi|fieldmap|syn)
    fmaps = None
    if 'fieldmaps' not in config.workflow.ignore:
        fmaps = fieldmap_wrangler(layout, ref_file,
                                  use_syn=config.workflow.use_syn_sdc,
                                  force_syn=config.workflow.force_syn)
    elif config.workflow.use_syn_sdc or config.workflow.force_syn:
        # If fieldmaps are not enabled, activate SyN-SDC in unforced (False) mode
        fmaps = {'syn': False}

    # Short circuits: (True and True and (False or 'TooShort')) == 'TooShort'
    run_stc = (
        bool(metadata.get("SliceTiming"))
        and 'slicetiming' not in config.workflow.ignore
        and (_get_series_len(ref_file) > 4 or "TooShort")
    )

    # Build workflow
    workflow = Workflow(name=wf_name)
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
                't1w_preproc', 't1w_mask', 't1w_dseg', 't1w_tpms',
                't1w_aseg', 't1w_aparc',
                'anat2std_xfm', 'std2anat_xfm', 'template',
                't1w2fsnative_xfm', 'fsnative2t1w_xfm']),
        name='inputnode')
    inputnode.inputs.bold_file = bold_file

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_t1', 'bold_t1_ref', 'bold_mask_t1', 'bold_aseg_t1', 'bold_aparc_t1',
                'bold_std', 'bold_std_ref', 'bold_mask_std', 'bold_aseg_std', 'bold_aparc_std',
                'bold_native', 'bold_cifti', 'cifti_variant', 'cifti_metadata', 'cifti_density',
                'surfaces', 'confounds', 'aroma_noise_ics', 'melodic_mix', 'nonaggr_denoised_file',
                'confounds_metadata']),
        name='outputnode')

    # Generate a brain-masked conversion of the t1w
    t1w_brain = pe.Node(ApplyMask(), name='t1w_brain')

    # BOLD buffer: an identity used as a pointer to either the original BOLD
    # or the STC'ed one for further use.
    boldbuffer = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='boldbuffer')

    summary = pe.Node(
        FunctionalSummary(
            slice_timing=run_stc,
            registration=('FSL', 'FreeSurfer')[freesurfer],
            registration_dof=config.workflow.bold2t1w_dof,
            registration_init=config.workflow.bold2t1w_init,
            pe_direction=metadata.get("PhaseEncodingDirection"),
            echo_idx=echo_idxs,
            tr=metadata.get("RepetitionTime")),
        name='summary', mem_gb=config.DEFAULT_MEMORY_MIN_GB, run_without_submitting=True)
    summary.inputs.dummy_scans = config.workflow.dummy_scans

    func_derivatives_wf = init_func_derivatives_wf(
        bids_root=layout.root,
        cifti_output=config.workflow.cifti_output,
        freesurfer=freesurfer,
        metadata=metadata,
        output_dir=output_dir,
        spaces=spaces,
        use_aroma=config.workflow.use_aroma,
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
            ('surfaces', 'inputnode.surf_files'),
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
    bold_reference_wf = init_bold_reference_wf(
        omp_nthreads=omp_nthreads,
        bold_file=bold_file,
        sbref_files=sbref_files,
        multiecho=multiecho,
    )
    bold_reference_wf.inputs.inputnode.dummy_scans = config.workflow.dummy_scans

    # Top-level BOLD splitter
    bold_split = pe.Node(FSLSplit(dimension='t'), name='bold_split',
                         mem_gb=mem_gb['filesize'] * 3)

    # HMC on the BOLD
    bold_hmc_wf = init_bold_hmc_wf(name='bold_hmc_wf',
                                   mem_gb=mem_gb['filesize'],
                                   omp_nthreads=omp_nthreads)

    # calculate BOLD registration to T1w
    bold_reg_wf = init_bold_reg_wf(
        bold2t1w_dof=config.workflow.bold2t1w_dof,
        bold2t1w_init=config.workflow.bold2t1w_init,
        freesurfer=freesurfer,
        mem_gb=mem_gb['resampled'],
        name='bold_reg_wf',
        omp_nthreads=omp_nthreads,
        sloppy=config.execution.debug,
        use_bbr=config.workflow.use_bbr,
        use_compression=False,
    )

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
        regressors_all_comps=config.workflow.regressors_all_comps,
        regressors_fd_th=config.workflow.regressors_fd_th,
        regressors_dvars_th=config.workflow.regressors_dvars_th,
        name='bold_confounds_wf')
    bold_confounds_wf.get_node('inputnode').inputs.t1_transform_flags = [False]

    # Apply transforms in 1 shot
    # Only use uncompressed output if AROMA is to be run
    bold_bold_trans_wf = init_bold_preproc_trans_wf(
        mem_gb=mem_gb['resampled'],
        omp_nthreads=omp_nthreads,
        use_compression=not config.execution.low_mem,
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
                                       omp_nthreads=omp_nthreads,
                                       debug=config.execution.debug)

    # MULTI-ECHO EPI DATA #############################################
    if multiecho:
        from niworkflows.func.util import init_skullstrip_bold_wf
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
                                       name='bold_t2smap_wf')

        workflow.connect([
            (skullstrip_bold_wf, join_echos, [
                ('outputnode.skull_stripped_file', 'bold_files')]),
            (join_echos, bold_t2s_wf, [
                ('bold_files', 'inputnode.bold_file')]),
        ])

    # MAIN WORKFLOW STRUCTURE #######################################################
    workflow.connect([
        (inputnode, t1w_brain, [('t1w_preproc', 'in_file'),
                                ('t1w_mask', 'in_mask')]),
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
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            # Undefined if --fs-no-reconall, but this is safe
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm')]),
        (t1w_brain, bold_reg_wf, [
            ('out_file', 'inputnode.t1w_brain')]),
        (inputnode, bold_t1_trans_wf, [
            ('bold_file', 'inputnode.name_source'),
            ('t1w_mask', 'inputnode.t1w_mask'),
            ('t1w_aseg', 'inputnode.t1w_aseg'),
            ('t1w_aparc', 'inputnode.t1w_aparc')]),
        (t1w_brain, bold_t1_trans_wf, [
            ('out_file', 'inputnode.t1w_brain')]),
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
        (t1w_brain, bold_sdc_wf, [
            ('out_file', 'inputnode.t1w_brain')]),
        (bold_reference_wf, bold_sdc_wf, [
            ('outputnode.ref_image', 'inputnode.epi_file'),
            ('outputnode.ref_image_brain', 'inputnode.epi_brain'),
            ('outputnode.bold_mask', 'inputnode.epi_mask')]),
        (bold_sdc_wf, bold_t1_trans_wf, [
            ('outputnode.out_warp', 'inputnode.fieldwarp'),
            ('outputnode.epi_mask', 'inputnode.ref_bold_mask'),
            ('outputnode.epi_brain', 'inputnode.ref_bold_brain')]),
        (bold_sdc_wf, bold_bold_trans_wf, [
            ('outputnode.out_warp', 'inputnode.fieldwarp'),
            ('outputnode.epi_mask', 'inputnode.bold_mask')]),
        (bold_sdc_wf, bold_reg_wf, [
            ('outputnode.epi_brain', 'inputnode.ref_bold_brain')]),
        (bold_sdc_wf, summary, [('outputnode.method', 'distortion_correction')]),
        # Connect bold_confounds_wf
        (inputnode, bold_confounds_wf, [('t1w_tpms', 'inputnode.t1w_tpms'),
                                        ('t1w_mask', 'inputnode.t1w_mask')]),
        (bold_hmc_wf, bold_confounds_wf, [
            ('outputnode.movpar_file', 'inputnode.movpar_file'),
            ('outputnode.rmsd_file', 'inputnode.rmsd_file')]),
        (bold_reg_wf, bold_confounds_wf, [
            ('outputnode.itk_t1_to_bold', 'inputnode.t1_bold_xform')]),
        (bold_reference_wf, bold_confounds_wf, [
            ('outputnode.skip_vols', 'inputnode.skip_vols')]),
        (bold_bold_trans_wf, bold_confounds_wf, [
            ('outputnode.bold_mask', 'inputnode.bold_mask'),
        ]),
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

    # for standard EPI data, pass along correct file
    if not multiecho:
        workflow.connect([
            (inputnode, func_derivatives_wf, [
                ('bold_file', 'inputnode.source_file')]),
            (bold_bold_trans_wf, bold_confounds_wf, [
                ('outputnode.bold', 'inputnode.bold')]),
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
                ('outputnode.bold', 'inputnode.bold')]),
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
        # And ensure echo is dropped from report
        for node in fmap_unwarp_report_wf.list_node_names():
            if node.split('.')[-1].startswith('ds_'):
                fmap_unwarp_report_wf.get_node(node).interface.out_path_base = 'fmriprep'
                fmap_unwarp_report_wf.get_node(node).inputs.dismiss_entities = ("echo",)

        for node in bold_sdc_wf.list_node_names():
            if node.split('.')[-1].startswith('ds_'):
                bold_sdc_wf.get_node(node).interface.out_path_base = 'fmriprep'
                bold_sdc_wf.get_node(node).inputs.dismiss_entities = ("echo",)

        if 'syn' in fmaps:
            sdc_select_std = pe.Node(
                KeySelect(fields=['std2anat_xfm']),
                name='sdc_select_std', run_without_submitting=True)
            sdc_select_std.inputs.key = 'MNI152NLin2009cAsym'
            workflow.connect([
                (inputnode, sdc_select_std, [('std2anat_xfm', 'std2anat_xfm'),
                                             ('template', 'keys')]),
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
            # And ensure echo is dropped from report
            for node in syn_unwarp_report_wf.list_node_names():
                if node.split('.')[-1].startswith('ds_'):
                    syn_unwarp_report_wf.get_node(node).interface.out_path_base = 'fmriprep'
                    syn_unwarp_report_wf.get_node(node).inputs.dismiss_entities = ("echo",)

    # Map final BOLD mask into T1w space (if required)
    nonstd_spaces = set(spaces.get_nonstandard())
    if nonstd_spaces.intersection(('T1w', 'anat')):
        from niworkflows.interfaces.fixes import (
            FixHeaderApplyTransforms as ApplyTransforms
        )

        boldmask_to_t1w = pe.Node(ApplyTransforms(interpolation='MultiLabel'),
                                  name='boldmask_to_t1w', mem_gb=0.1)
        workflow.connect([
            (bold_reg_wf, boldmask_to_t1w, [
                ('outputnode.itk_bold_to_t1', 'transforms')]),
            (bold_t1_trans_wf, boldmask_to_t1w, [
                ('outputnode.bold_mask_t1', 'reference_image')]),
            (bold_bold_trans_wf, boldmask_to_t1w, [
                ('outputnode.bold_mask', 'input_image')]),
            (boldmask_to_t1w, outputnode, [
                ('output_image', 'bold_mask_t1')]),
        ])

    if nonstd_spaces.intersection(('func', 'run', 'bold', 'boldref', 'sbref')):
        workflow.connect([
            (bold_bold_trans_wf, outputnode, [
                ('outputnode.bold', 'bold_native')]),
            (bold_bold_trans_wf, func_derivatives_wf, [
                ('outputnode.bold_ref', 'inputnode.bold_native_ref'),
                ('outputnode.bold_mask', 'inputnode.bold_mask_native')]),
        ])

    if spaces.get_spaces(nonstandard=False, dim=(3,)):
        # Apply transforms in 1 shot
        # Only use uncompressed output if AROMA is to be run
        bold_std_trans_wf = init_bold_std_trans_wf(
            freesurfer=freesurfer,
            mem_gb=mem_gb['resampled'],
            omp_nthreads=omp_nthreads,
            spaces=spaces,
            name='bold_std_trans_wf',
            use_compression=not config.execution.low_mem,
            use_fieldwarp=bool(fmaps),
        )
        workflow.connect([
            (inputnode, bold_std_trans_wf, [
                ('template', 'inputnode.templates'),
                ('anat2std_xfm', 'inputnode.anat2std_xfm'),
                ('bold_file', 'inputnode.name_source'),
                ('t1w_aseg', 'inputnode.bold_aseg'),
                ('t1w_aparc', 'inputnode.bold_aparc')]),
            (bold_hmc_wf, bold_std_trans_wf, [
                ('outputnode.xforms', 'inputnode.hmc_xforms')]),
            (bold_reg_wf, bold_std_trans_wf, [
                ('outputnode.itk_bold_to_t1', 'inputnode.itk_bold_to_t1')]),
            (bold_bold_trans_wf, bold_std_trans_wf, [
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

        # func_derivatives_wf internally parametrizes over snapshotted spaces.
        workflow.connect([
            (bold_std_trans_wf, func_derivatives_wf, [
                ('outputnode.template', 'inputnode.template'),
                ('outputnode.spatial_reference', 'inputnode.spatial_reference'),
                ('outputnode.bold_std_ref', 'inputnode.bold_std_ref'),
                ('outputnode.bold_std', 'inputnode.bold_std'),
                ('outputnode.bold_mask_std', 'inputnode.bold_mask_std'),
            ]),
        ])

        if config.workflow.use_aroma:  # ICA-AROMA workflow
            from .confounds import init_ica_aroma_wf
            ica_aroma_wf = init_ica_aroma_wf(
                mem_gb=mem_gb['resampled'],
                metadata=metadata,
                omp_nthreads=omp_nthreads,
                use_fieldwarp=bool(fmaps),
                err_on_aroma_warn=config.workflow.aroma_err_on_warn,
                aroma_melodic_dim=config.workflow.aroma_melodic_dim,
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
                (bold_std_trans_wf, ica_aroma_wf, [
                    ('outputnode.bold_std', 'inputnode.bold_std'),
                    ('outputnode.bold_mask_std', 'inputnode.bold_mask_std'),
                    ('outputnode.spatial_reference', 'inputnode.spatial_reference')]),
            ])

    # SURFACES ##################################################################################
    # Freesurfer
    freesurfer_spaces = spaces.get_fs_spaces()
    if freesurfer and freesurfer_spaces:
        config.loggers.workflow.debug('Creating BOLD surface-sampling workflow.')
        bold_surf_wf = init_bold_surf_wf(
            mem_gb=mem_gb['resampled'],
            surface_spaces=freesurfer_spaces,
            medial_surface_nan=config.workflow.medial_surface_nan,
            name='bold_surf_wf')
        workflow.connect([
            (inputnode, bold_surf_wf, [
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id'),
                ('t1w2fsnative_xfm', 'inputnode.t1w2fsnative_xfm')]),
            (bold_t1_trans_wf, bold_surf_wf, [('outputnode.bold_t1', 'inputnode.source_file')]),
            (bold_surf_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
            (bold_surf_wf, func_derivatives_wf, [
                ('outputnode.target', 'inputnode.surf_refs')]),
        ])

        # CIFTI output
        if config.workflow.cifti_output:
            from .resampling import init_bold_grayords_wf
            bold_grayords_wf = init_bold_grayords_wf(
                grayord_density=config.workflow.cifti_output,
                mem_gb=mem_gb['resampled'],
                repetition_time=metadata['RepetitionTime'])

            workflow.connect([
                (inputnode, bold_grayords_wf, [
                    ('subjects_dir', 'inputnode.subjects_dir')]),
                (bold_std_trans_wf, bold_grayords_wf, [
                    ('outputnode.bold_std', 'inputnode.bold_std'),
                    ('outputnode.spatial_reference', 'inputnode.spatial_reference')]),
                (bold_surf_wf, bold_grayords_wf, [
                    ('outputnode.surfaces', 'inputnode.surf_files'),
                    ('outputnode.target', 'inputnode.surf_refs'),
                ]),
                (bold_grayords_wf, outputnode, [
                    ('outputnode.cifti_bold', 'bold_cifti'),
                    ('outputnode.cifti_variant', 'cifti_variant'),
                    ('outputnode.cifti_metadata', 'cifti_metadata'),
                    ('outputnode.cifti_density', 'cifti_density')]),
            ])

    if spaces.get_spaces(nonstandard=False, dim=(3,)):
        carpetplot_wf = init_carpetplot_wf(
            mem_gb=mem_gb['resampled'],
            metadata=metadata,
            cifti_output=config.workflow.cifti_output,
            name='carpetplot_wf')

        if config.workflow.cifti_output:
            workflow.connect(
                bold_grayords_wf, 'outputnode.cifti_bold', carpetplot_wf, 'inputnode.cifti_bold'
            )
        else:
            # Xform to 'MNI152NLin2009cAsym' is always computed.
            carpetplot_select_std = pe.Node(
                KeySelect(fields=['std2anat_xfm'], key='MNI152NLin2009cAsym'),
                name='carpetplot_select_std', run_without_submitting=True)

            workflow.connect([
                (inputnode, carpetplot_select_std, [
                    ('std2anat_xfm', 'std2anat_xfm'),
                    ('template', 'keys')]),
                (carpetplot_select_std, carpetplot_wf, [
                    ('std2anat_xfm', 'inputnode.std2anat_xfm')]),
                (bold_bold_trans_wf if not multiecho else bold_t2s_wf, carpetplot_wf, [
                    ('outputnode.bold', 'inputnode.bold')]),
                (bold_bold_trans_wf, carpetplot_wf, [
                    ('outputnode.bold_mask', 'inputnode.bold_mask')]),
                (bold_reg_wf, carpetplot_wf, [
                    ('outputnode.itk_t1_to_bold', 'inputnode.t1_bold_xform')]),
            ])

        workflow.connect([
            (bold_confounds_wf, carpetplot_wf, [
                        ('outputnode.confounds_file', 'inputnode.confounds_file')])
        ])

    # REPORTING ############################################################
    ds_report_summary = pe.Node(
        DerivativesDataSink(desc='summary', datatype="figures", dismiss_entities=("echo",)),
        name='ds_report_summary', run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB)

    ds_report_validation = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='validation', datatype="figures",
                            dismiss_entities=("echo",)),
        name='ds_report_validation', run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bold_reference_wf, ds_report_validation, [
            ('outputnode.validation_report', 'in_file')]),
    ])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_report'):
            workflow.get_node(node).inputs.base_directory = output_dir
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


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.

    Examples
    --------
    >>> extract_entities('sub-01/anat/sub-01_T1w.nii.gz')
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': 'nii.gz'}
    >>> extract_entities(['sub-01/anat/sub-01_T1w.nii.gz'] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': 'nii.gz'}
    >>> extract_entities(['sub-01/anat/sub-01_run-1_T1w.nii.gz',
    ...                   'sub-01/anat/sub-01_run-2_T1w.nii.gz'])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat',
     'extension': 'nii.gz'}

    """
    from collections import defaultdict
    from bids.layout import parse_file_entities

    entities = defaultdict(list)
    for e, v in [
        ev_pair
        for f in listify(file_list)
        for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist
    return {
        k: _unique(v) for k, v in entities.items()
    }
