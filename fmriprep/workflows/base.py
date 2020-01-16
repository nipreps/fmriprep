#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fMRIprep base processing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fmriprep_wf
.. autofunction:: init_single_subject_wf

"""

import sys
import os
from copy import deepcopy

from nipype import __version__ as nipype_ver
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.interfaces.nilearn import NILEARN_VERSION

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.bids import (
    BIDSInfo, BIDSDataGrabber, BIDSFreeSurferDir
)
from niworkflows.utils.bids import collect_data
from niworkflows.utils.misc import fix_multi_T1w_source_name
from smriprep.workflows.anatomical import init_anat_preproc_wf

from ..interfaces import SubjectSummary, AboutSummary, DerivativesDataSink
from ..__about__ import __version__
from .bold import init_func_preproc_wf
# from .bold.outputs import _remove_internal_spaces


def init_fmriprep_wf(
    anat_only,
    aroma_melodic_dim,
    bold2t1w_dof,
    cifti_output,
    debug,
    dummy_scans,
    echo_idx,
    err_on_aroma_warn,
    fmap_bspline,
    fmap_demean,
    force_syn,
    freesurfer,
    fs_subjects_dir,
    hires,
    ignore,
    layout,
    longitudinal,
    low_mem,
    medial_surface_nan,
    omp_nthreads,
    output_dir,
    regressors_all_comps,
    regressors_dvars_th,
    regressors_fd_th,
    run_uuid,
    skull_strip_fixed_seed,
    skull_strip_template,
    spaces,
    subject_list,
    t2s_coreg,
    task_id,
    use_aroma,
    use_bbr,
    use_syn,
    work_dir,
):
    """
    Build *fMRIPrep*'s pipeline.

    This workflow organizes the execution of FMRIPREP, with a sub-workflow for
    each subject.
    If FreeSurfer's ``recon-all`` is to be run, a corresponding folder is created
    and populated with any needed template subjects under the derivatives folder.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            import os
            from collections import namedtuple, OrderedDict
            BIDSLayout = namedtuple('BIDSLayout', ['root'])
            from fmriprep.workflows.base import init_fmriprep_wf
            from fmriprep.utils import Spaces
            os.environ['FREESURFER_HOME'] = os.getcwd()
            wf = init_fmriprep_wf(
                anat_only=False,
                aroma_melodic_dim=-200,
                bold2t1w_dof=9,
                cifti_output=False,
                debug=False,
                dummy_scans=None,
                echo_idx=None,
                err_on_aroma_warn=False,
                fmap_bspline=False,
                fmap_demean=True,
                force_syn=True,
                freesurfer=True,
                fs_subjects_dir=None,
                hires=True,
                ignore=[],
                layout=BIDSLayout('.'),
                longitudinal=False,
                low_mem=False,
                medial_surface_nan=False,
                omp_nthreads=1,
                output_dir='.',
                regressors_all_comps=False,
                regressors_dvars_th=1.5,
                regressors_fd_th=0.5,
                run_uuid='X',
                skull_strip_fixed_seed=False,
                skull_strip_template=('OASIS30ANTs', {}),
                subject_list=['fmripreptest'],
                t2s_coreg=False,
                target_spaces=OrderedDict([
                    ('MNI152Lin', {}), ('fsaverage', {'density': '10k'}),
                    ('T1w', {}), ('fsnative', {})]),
                task_id='',
                use_aroma=False,
                use_bbr=True,
                use_syn=True,
                work_dir='.',
            )


    Parameters
    ----------
    anat_only : bool
        Disable functional workflows
    bold2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for BOLD-T1w registration
    cifti_output : bool
        Generate bold CIFTI file in output spaces
    debug : bool
        Enable debugging outputs
    dummy_scans : int or None
        Number of volumes to consider as non steady state
    echo_idx : int or None
        Index of echo to preprocess in multiecho BOLD series,
        or ``None`` to preprocess all
    err_on_aroma_warn : bool
        Do not fail on ICA-AROMA errors
    fmap_bspline : bool
        **Experimental**: Fit B-Spline field using least-squares
    fmap_demean : bool
        Demean voxel-shift map during unwarp
    force_syn : bool
        **Temporary**: Always run SyN-based SDC
    freesurfer : bool
        Enable FreeSurfer surface reconstruction (may increase runtime)
    hires : bool
        Enable sub-millimeter preprocessing in FreeSurfer
    ignore : list
        Preprocessing steps to skip (may include "slicetiming", "fieldmaps")
    layout : BIDSLayout object
        BIDS dataset layout
    longitudinal : bool
        Treat multiple sessions as longitudinal (may increase runtime)
        See sub-workflows for specific differences
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
    run_uuid : str
        Unique identifier for execution instance
    skull_strip_template : tuple
        Name of target template for brain extraction with ANTs' ``antsBrainExtraction``,
        and corresponding dictionary of output-space modifiers.
    skull_strip_fixed_seed : bool
        Do not use a random seed for skull-stripping - will ensure
        run-to-run replicability when used with --omp-nthreads 1
    subject_list : list
        List of subject labels
    t2s_coreg : bool
        For multi-echo EPI, use the calculated T2*-map for T2*-driven coregistration
    spaces : Spaces
        Ordered dictionary where keys are TemplateFlow ID strings (e.g., ``MNI152Lin``,
        ``MNI152NLin6Asym``, ``MNI152NLin2009cAsym``, or ``fsLR``) strings designating
        nonstandard references (e.g., ``T1w`` or ``anat``, ``sbref``, ``run``, etc.),
        or paths pointing to custom templates organized in a TemplateFlow-like structure.
        Values of the dictionary aggregate modifiers (e.g., the value for the key ``MNI152Lin``
        could be ``{'resolution': 2}`` if one wants the resampling to be done on the 2mm
        resolution version of the selected template).
    task_id : str or None
        Task ID of BOLD series to preprocess, or ``None`` to preprocess all
    use_aroma : bool
        Perform ICA-AROMA on MNI-resampled functional series
    use_bbr : bool or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
    use_syn : bool
        **Experimental**: Enable ANTs SyN-based susceptibility distortion correction (SDC).
        If fieldmaps are present and enabled, this is not run, by default.
    work_dir : str
        Directory in which to store workflow execution state and temporary files

    """
    fmriprep_wf = Workflow(name='fmriprep_wf')
    fmriprep_wf.base_dir = work_dir

    if freesurfer:
        fsdir = pe.Node(
            BIDSFreeSurferDir(
                derivatives=output_dir,
                freesurfer_home=os.getenv('FREESURFER_HOME'),
                spaces=[s for s in spaces.unique()
                        if s.startswith('fsaverage') or s == 'fsnative']),
            name='fsdir_run_' + run_uuid.replace('-', '_'), run_without_submitting=True)
        if fs_subjects_dir is not None:
            fsdir.inputs.subjects_dir = str(fs_subjects_dir.absolute())

    reportlets_dir = os.path.join(work_dir, 'reportlets')
    for subject_id in subject_list:
        single_subject_wf = init_single_subject_wf(
            anat_only=anat_only,
            aroma_melodic_dim=aroma_melodic_dim,
            bold2t1w_dof=bold2t1w_dof,
            cifti_output=cifti_output,
            debug=debug,
            dummy_scans=dummy_scans,
            echo_idx=echo_idx,
            err_on_aroma_warn=err_on_aroma_warn,
            fmap_bspline=fmap_bspline,
            fmap_demean=fmap_demean,
            force_syn=force_syn,
            freesurfer=freesurfer,
            hires=hires,
            ignore=ignore,
            layout=layout,
            longitudinal=longitudinal,
            low_mem=low_mem,
            medial_surface_nan=medial_surface_nan,
            name="single_subject_" + subject_id + "_wf",
            omp_nthreads=omp_nthreads,
            output_dir=output_dir,
            regressors_all_comps=regressors_all_comps,
            regressors_dvars_th=regressors_dvars_th,
            regressors_fd_th=regressors_fd_th,
            reportlets_dir=reportlets_dir,
            skull_strip_fixed_seed=skull_strip_fixed_seed,
            skull_strip_template=skull_strip_template,
            spaces=spaces,
            subject_id=subject_id,
            t2s_coreg=t2s_coreg,
            task_id=task_id,
            use_aroma=use_aroma,
            use_bbr=use_bbr,
            use_syn=use_syn,
        )

        single_subject_wf.config['execution']['crashdump_dir'] = (
            os.path.join(output_dir, "fmriprep", "sub-" + subject_id, 'log', run_uuid)
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        if freesurfer:
            fmriprep_wf.connect(fsdir, 'subjects_dir',
                                single_subject_wf, 'inputnode.subjects_dir')
        else:
            fmriprep_wf.add_nodes([single_subject_wf])

    return fmriprep_wf


def init_single_subject_wf(
    anat_only,
    aroma_melodic_dim,
    bold2t1w_dof,
    cifti_output,
    debug,
    dummy_scans,
    echo_idx,
    err_on_aroma_warn,
    fmap_bspline,
    fmap_demean,
    force_syn,
    freesurfer,
    hires,
    ignore,
    layout,
    longitudinal,
    low_mem,
    medial_surface_nan,
    name,
    omp_nthreads,
    output_dir,
    reportlets_dir,
    regressors_all_comps,
    regressors_dvars_th,
    regressors_fd_th,
    skull_strip_fixed_seed,
    skull_strip_template,
    spaces,
    subject_id,
    t2s_coreg,
    task_id,
    use_aroma,
    use_bbr,
    use_syn,
):
    """
    This workflow organizes the preprocessing pipeline for a single subject.

    It collects and reports information about the subject, and prepares
    sub-workflows to perform anatomical and functional preprocessing.
    Anatomical preprocessing is performed in a single workflow, regardless of
    the number of sessions.
    Functional preprocessing is performed using a separate workflow for each
    individual BOLD series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.base import init_single_subject_wf
            from collections import namedtuple, OrderedDict
            BIDSLayout = namedtuple('BIDSLayout', ['root'])
            wf = init_single_subject_wf(
                anat_only=False,
                aroma_melodic_dim=-200,
                bold2t1w_dof=9,
                cifti_output=False,
                debug=False,
                dummy_scans=None,
                echo_idx=None,
                err_on_aroma_warn=False,
                fmap_bspline=False,
                fmap_demean=True,
                force_syn=True,
                freesurfer=True,
                hires=True,
                ignore=[],
                layout=BIDSLayout('.'),
                longitudinal=False,
                low_mem=False,
                medial_surface_nan=False,
                name='single_subject_wf',
                omp_nthreads=1,
                output_dir='.',
                reportlets_dir='.',
                regressors_all_comps=False,
                regressors_dvars_th=1.5,
                regressors_fd_th=0.5,
                skull_strip_fixed_seed=False,
                skull_strip_template=('OASIS30ANTs', {}),
                subject_id='test',
                t2s_coreg=False,
                target_spaces=OrderedDict([
                    ('MNI152Lin', {}), ('fsaverage', {'density': '10k'}),
                    ('T1w', {}), ('fsnative', {})]),
                task_id='',
                use_aroma=False,
                use_bbr=True,
                use_syn=True,
            )

    Parameters
    ----------
    anat_only : bool
        Disable functional workflows
    aroma_melodic_dim : int
        Maximum number of components identified by MELODIC within ICA-AROMA
        (default is -200, i.e., no limitation).
    bold2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for BOLD-T1w registration
    cifti_output : bool
        Generate bold CIFTI file in output spaces
    debug : bool
        Enable debugging outputs
    dummy_scans : int or None
        Number of volumes to consider as non steady state
    echo_idx : int or None
        Index of echo to preprocess in multiecho BOLD series,
        or ``None`` to preprocess all
    err_on_aroma_warn : bool
        Do not fail on ICA-AROMA errors
    fmap_bspline : bool
        **Experimental**: Fit B-Spline field using least-squares
    fmap_demean : bool
        Demean voxel-shift map during unwarp
    force_syn : bool
        **Temporary**: Always run SyN-based SDC
    freesurfer : bool
        Enable FreeSurfer surface reconstruction (may increase runtime)
    hires : bool
        Enable sub-millimeter preprocessing in FreeSurfer
    ignore : list
        Preprocessing steps to skip (may include "slicetiming", "fieldmaps")
    layout : BIDSLayout object
        BIDS dataset layout
    longitudinal : bool
        Treat multiple sessions as longitudinal (may increase runtime)
        See sub-workflows for specific differences
    low_mem : bool
        Write uncompressed .nii files in some cases to reduce memory usage
    medial_surface_nan : bool
        Replace medial wall values with NaNs on functional GIFTI files
    name : str
        Name of workflow
    omp_nthreads : int
        Maximum number of threads an individual process may use
    output_dir : str
        Directory in which to save derivatives
    reportlets_dir : str
        Directory in which to save reportlets
    regressors_all_comps
        Return all CompCor component time series instead of the top fraction
    regressors_fd_th
        Criterion for flagging framewise displacement outliers
    regressors_dvars_th
        Criterion for flagging DVARS outliers
    skull_strip_fixed_seed : bool
        Do not use a random seed for skull-stripping - will ensure
        run-to-run replicability when used with --omp-nthreads 1
    skull_strip_template : tuple
        Name of target template for brain extraction with ANTs' ``antsBrainExtraction``,
        and corresponding dictionary of output-space modifiers.
    subject_id : str
        List of subject labels
    t2s_coreg : bool
        For multi-echo EPI, use the calculated T2*-map for T2*-driven coregistration
    target_spaces : OrderedDict
        Ordered dictionary where keys are TemplateFlow ID strings (e.g., ``MNI152Lin``,
        ``MNI152NLin6Asym``, ``MNI152NLin2009cAsym``, or ``fsLR``) strings designating
        nonstandard references (e.g., ``T1w`` or ``anat``, ``sbref``, ``run``, etc.),
        or paths pointing to custom templates organized in a TemplateFlow-like structure.
        Values of the dictionary aggregate modifiers (e.g., the value for the key ``MNI152Lin``
        could be ``{'resolution': 2}`` if one wants the resampling to be done on the 2mm
        resolution version of the selected template).
    task_id : str or None
        Task ID of BOLD series to preprocess, or ``None`` to preprocess all
    use_aroma : bool
        Perform ICA-AROMA on MNI-resampled functional series
    use_bbr : bool or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
    use_syn : bool
        **Experimental**: Enable ANTs SyN-based susceptibility distortion correction (SDC).
        If fieldmaps are present and enabled, this is not run, by default.

    Inputs
    ------
    subjects_dir : str
        FreeSurfer's ``$SUBJECTS_DIR``.

    """
    if name in ('single_subject_wf', 'single_subject_fmripreptest_wf'):
        # for documentation purposes
        subject_data = {
            't1w': ['/completely/made/up/path/sub-01_T1w.nii.gz'],
            'bold': ['/completely/made/up/path/sub-01_task-nback_bold.nii.gz']
        }
    else:
        subject_data = collect_data(layout, subject_id, task_id, echo_idx)[0]

    # Make sure we always go through these two checks
    if not anat_only and subject_data['bold'] == []:
        raise Exception("No BOLD images found for participant {} and task {}. "
                        "All workflows require BOLD images.".format(
                            subject_id, task_id if task_id else '<all>'))

    if not subject_data['t1w']:
        raise Exception("No T1w images found for participant {}. "
                        "All workflows require T1w images.".format(subject_id))

    workflow = Workflow(name=name)
    workflow.__desc__ = """
Results included in this manuscript come from preprocessing
performed using *fMRIPrep* {fmriprep_ver}
(@fmriprep1; @fmriprep2; RRID:SCR_016216),
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

""".format(fmriprep_ver=__version__, nipype_ver=nipype_ver)
    workflow.__postdesc__ = """

Many internal operations of *fMRIPrep* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362],
mostly within the functional processing workflow.
For more details of the pipeline, see [the section corresponding
to workflows in *fMRIPrep*'s documentation]\
(https://fmriprep.readthedocs.io/en/latest/workflows.html \
"FMRIPrep's documentation").


### Copyright Waiver

The above boilerplate text was automatically generated by fMRIPrep
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0]\
(https://creativecommons.org/publicdomain/zero/1.0/) license.

### References

""".format(nilearn_ver=NILEARN_VERSION)

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']),
                        name='inputnode')

    bidssrc = pe.Node(BIDSDataGrabber(subject_data=subject_data, anat_only=anat_only),
                      name='bidssrc')

    bids_info = pe.Node(BIDSInfo(
        bids_dir=layout.root, bids_validate=False), name='bids_info')

    summary = pe.Node(SubjectSummary(
            std_spaces=spaces.filtered('std', 'output', name_only=True) or [],
            nstd_spaces=spaces.filtered('nstd', 'output', name_only=True) or []),
        name='summary', run_without_submitting=True)

    about = pe.Node(AboutSummary(version=__version__,
                                 command=' '.join(sys.argv)),
                    name='about', run_without_submitting=True)

    ds_report_summary = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            desc='summary', keep_dtype=True),
        name='ds_report_summary', run_without_submitting=True)

    ds_report_about = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            desc='about', keep_dtype=True),
        name='ds_report_about', run_without_submitting=True)

    # Preprocessing of T1w (includes registration to MNI)
    anat_preproc_wf = init_anat_preproc_wf(
        bids_root=layout.root,
        debug=debug,
        freesurfer=freesurfer,
        hires=hires,
        longitudinal=longitudinal,
        name="anat_preproc_wf",
        num_t1w=len(subject_data['t1w']),
        omp_nthreads=omp_nthreads,
        output_dir=output_dir,
        # output_spaces={sp: {} for sp in spaces.standard},
        # output_spaces={k: v for k, v in spaces.filtered('std', 'output')}, # TODO: RF smriprep
        reportlets_dir=reportlets_dir,
        spaces=spaces.filtered('std_vol', 'all'),
        skull_strip_fixed_seed=skull_strip_fixed_seed,
        skull_strip_template=skull_strip_template,
    )

    workflow.connect([
        (inputnode, anat_preproc_wf, [('subjects_dir', 'inputnode.subjects_dir')]),
        (bidssrc, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('t1w', 't1w'),
                            ('t2w', 't2w'),
                            ('bold', 'bold')]),
        (bids_info, summary, [('subject', 'subject_id')]),
        (bids_info, anat_preproc_wf, [(('subject', _prefix), 'inputnode.subject_id')]),
        (bidssrc, anat_preproc_wf, [('t1w', 'inputnode.t1w'),
                                    ('t2w', 'inputnode.t2w'),
                                    ('roi', 'inputnode.roi'),
                                    ('flair', 'inputnode.flair')]),
        (bidssrc, ds_report_summary, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bidssrc, ds_report_about, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])

    # Overwrite ``out_path_base`` of smriprep's DataSinks
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = 'fmriprep'

    if anat_only:
        return workflow

    for bold_file in subject_data['bold']:
        func_preproc_wf = init_func_preproc_wf(
            aroma_melodic_dim=aroma_melodic_dim,
            bold2t1w_dof=bold2t1w_dof,
            bold_file=bold_file,
            cifti_output=cifti_output,
            debug=debug,
            dummy_scans=dummy_scans,
            err_on_aroma_warn=err_on_aroma_warn,
            fmap_bspline=fmap_bspline,
            fmap_demean=fmap_demean,
            force_syn=force_syn,
            freesurfer=freesurfer,
            ignore=ignore,
            layout=layout,
            low_mem=low_mem,
            medial_surface_nan=medial_surface_nan,
            num_bold=len(subject_data['bold']),
            omp_nthreads=omp_nthreads,
            output_dir=output_dir,
            reportlets_dir=reportlets_dir,
            regressors_all_comps=regressors_all_comps,
            regressors_fd_th=regressors_fd_th,
            regressors_dvars_th=regressors_dvars_th,
            spaces=spaces,
            t2s_coreg=t2s_coreg,
            # target_spaces=target_spaces,
            use_aroma=use_aroma,
            use_bbr=use_bbr,
            use_syn=use_syn,
        )

        workflow.connect([
            (anat_preproc_wf, func_preproc_wf,
             [(('outputnode.t1w_preproc', _pop), 'inputnode.t1w_preproc'),
              ('outputnode.t1w_brain', 'inputnode.t1w_brain'),
              ('outputnode.t1w_mask', 'inputnode.t1w_mask'),
              ('outputnode.t1w_dseg', 'inputnode.t1w_dseg'),
              ('outputnode.t1w_aseg', 'inputnode.t1w_aseg'),
              ('outputnode.t1w_aparc', 'inputnode.t1w_aparc'),
              ('outputnode.t1w_tpms', 'inputnode.t1w_tpms'),
              ('outputnode.template', 'inputnode.template'),
              ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
              ('outputnode.std2anat_xfm', 'inputnode.std2anat_xfm'),
              ('outputnode.joint_template_and_spec', 'inputnode.joint_template'),
              ('outputnode.joint_anat2std_xfm', 'inputnode.joint_anat2std_xfm'),
              ('outputnode.joint_std2anat_xfm', 'inputnode.joint_std2anat_xfm'),
              # Undefined if --fs-no-reconall, but this is safe
              ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
              ('outputnode.subject_id', 'inputnode.subject_id'),
              ('outputnode.t1w2fsnative_xfm', 'inputnode.t1w2fsnative_xfm'),
              ('outputnode.fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm')]),
        ])

    return workflow


def _prefix(subid):
    if subid.startswith('sub-'):
        return subid
    return '-'.join(('sub', subid))


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist
