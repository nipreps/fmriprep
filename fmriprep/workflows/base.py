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
from nilearn import __version__ as nilearn_ver

from ..engine import Workflow
from ..interfaces import (
    BIDSDataGrabber, BIDSFreeSurferDir, BIDSInfo, SubjectSummary, AboutSummary,
    DerivativesDataSink
)
from ..utils.bids import collect_data
from ..utils.misc import fix_multi_T1w_source_name
from ..__about__ import __version__

from .anatomical import init_anat_preproc_wf
from .bold import init_func_preproc_wf


def init_fmriprep_wf(subject_list, task_id, echo_idx, run_uuid, work_dir, output_dir, bids_dir,
                     ignore, debug, low_mem, anat_only, longitudinal, t2s_coreg,
                     omp_nthreads, skull_strip_template, skull_strip_fixed_seed,
                     freesurfer, output_spaces, template, medial_surface_nan, cifti_output, hires,
                     use_bbr, bold2t1w_dof, fmap_bspline, fmap_demean, use_syn, force_syn,
                     use_aroma, ignore_aroma_err, aroma_melodic_dim, template_out_grid):
    """
    This workflow organizes the execution of FMRIPREP, with a sub-workflow for
    each subject.

    If FreeSurfer's recon-all is to be run, a FreeSurfer derivatives folder is
    created and populated with any needed template subjects.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        import os
        os.environ['FREESURFER_HOME'] = os.getcwd()
        from fmriprep.workflows.base import init_fmriprep_wf
        wf = init_fmriprep_wf(subject_list=['fmripreptest'],
                              task_id='',
                              echo_idx=None,
                              run_uuid='X',
                              work_dir='.',
                              output_dir='.',
                              bids_dir='.',
                              ignore=[],
                              debug=False,
                              low_mem=False,
                              anat_only=False,
                              longitudinal=False,
                              t2s_coreg=False,
                              omp_nthreads=1,
                              skull_strip_template='OASIS',
                              skull_strip_fixed_seed=False,
                              freesurfer=True,
                              output_spaces=['T1w', 'fsnative',
                                            'template', 'fsaverage5'],
                              template='MNI152NLin2009cAsym',
                              medial_surface_nan=False,
                              cifti_output=False,
                              hires=True,
                              use_bbr=True,
                              bold2t1w_dof=9,
                              fmap_bspline=False,
                              fmap_demean=True,
                              use_syn=True,
                              force_syn=True,
                              use_aroma=False,
                              ignore_aroma_err=False,
                              aroma_melodic_dim=-200,
                              template_out_grid='native')


    Parameters

        subject_list : list
            List of subject labels
        task_id : str or None
            Task ID of BOLD series to preprocess, or ``None`` to preprocess all
        echo_idx : int or None
            Index of echo to preprocess in multiecho BOLD series,
            or ``None`` to preprocess all
        run_uuid : str
            Unique identifier for execution instance
        work_dir : str
            Directory in which to store workflow execution state and temporary files
        output_dir : str
            Directory in which to save derivatives
        bids_dir : str
            Root directory of BIDS dataset
        ignore : list
            Preprocessing steps to skip (may include "slicetiming", "fieldmaps")
        debug : bool
            Enable debugging outputs
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        anat_only : bool
            Disable functional workflows
        longitudinal : bool
            Treat multiple sessions as longitudinal (may increase runtime)
            See sub-workflows for specific differences
        t2s_coreg : bool
            For multi-echo EPI, use the calculated T2*-map for T2*-driven coregistration
        omp_nthreads : int
            Maximum number of threads an individual process may use
        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        skull_strip_fixed_seed : bool
            Do not use a random seed for skull-stripping - will ensure
            run-to-run replicability when used with --omp-nthreads 1
        freesurfer : bool
            Enable FreeSurfer surface reconstruction (may increase runtime)
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
        medial_surface_nan : bool
            Replace medial wall values with NaNs on functional GIFTI files
        cifti_output : bool
            Generate bold CIFTI file in output spaces
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer
        use_bbr : bool or None
            Enable/disable boundary-based registration refinement.
            If ``None``, test BBR result for distortion before accepting.
        bold2t1w_dof : 6, 9 or 12
            Degrees-of-freedom for BOLD-T1w registration
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
        template_out_grid : str
            Keyword ('native', '1mm' or '2mm') or path of custom reference
            image for normalization

    """
    fmriprep_wf = Workflow(name='fmriprep_wf')
    fmriprep_wf.base_dir = work_dir

    if freesurfer:
        fsdir = pe.Node(
            BIDSFreeSurferDir(
                derivatives=output_dir,
                freesurfer_home=os.getenv('FREESURFER_HOME'),
                spaces=output_spaces),
            name='fsdir_run_' + run_uuid.replace('-', '_'), run_without_submitting=True)

    reportlets_dir = os.path.join(work_dir, 'reportlets')
    for subject_id in subject_list:
        single_subject_wf = init_single_subject_wf(
            subject_id=subject_id,
            task_id=task_id,
            echo_idx=echo_idx,
            name="single_subject_" + subject_id + "_wf",
            reportlets_dir=reportlets_dir,
            output_dir=output_dir,
            bids_dir=bids_dir,
            ignore=ignore,
            debug=debug,
            low_mem=low_mem,
            anat_only=anat_only,
            longitudinal=longitudinal,
            t2s_coreg=t2s_coreg,
            omp_nthreads=omp_nthreads,
            skull_strip_template=skull_strip_template,
            skull_strip_fixed_seed=skull_strip_fixed_seed,
            freesurfer=freesurfer,
            output_spaces=output_spaces,
            template=template,
            medial_surface_nan=medial_surface_nan,
            cifti_output=cifti_output,
            hires=hires,
            use_bbr=use_bbr,
            bold2t1w_dof=bold2t1w_dof,
            fmap_bspline=fmap_bspline,
            fmap_demean=fmap_demean,
            use_syn=use_syn,
            force_syn=force_syn,
            template_out_grid=template_out_grid,
            use_aroma=use_aroma,
            aroma_melodic_dim=aroma_melodic_dim,
            ignore_aroma_err=ignore_aroma_err,
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


def init_single_subject_wf(subject_id, task_id, echo_idx, name, reportlets_dir, output_dir,
                           bids_dir, ignore, debug, low_mem, anat_only, longitudinal, t2s_coreg,
                           omp_nthreads, skull_strip_template, skull_strip_fixed_seed,
                           freesurfer, output_spaces, template, medial_surface_nan,
                           cifti_output, hires, use_bbr, bold2t1w_dof, fmap_bspline, fmap_demean,
                           use_syn, force_syn, template_out_grid,
                           use_aroma, aroma_melodic_dim, ignore_aroma_err):
    """
    This workflow organizes the preprocessing pipeline for a single subject.
    It collects and reports information about the subject, and prepares
    sub-workflows to perform anatomical and functional preprocessing.

    Anatomical preprocessing is performed in a single workflow, regardless of
    the number of sessions.
    Functional preprocessing is performed using a separate workflow for each
    individual BOLD series.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.base import init_single_subject_wf
        wf = init_single_subject_wf(subject_id='test',
                                    task_id='',
                                    echo_idx=None,
                                    name='single_subject_wf',
                                    reportlets_dir='.',
                                    output_dir='.',
                                    bids_dir='.',
                                    ignore=[],
                                    debug=False,
                                    low_mem=False,
                                    anat_only=False,
                                    longitudinal=False,
                                    t2s_coreg=False,
                                    omp_nthreads=1,
                                    skull_strip_template='OASIS',
                                    skull_strip_fixed_seed=False,
                                    freesurfer=True,
                                    template='MNI152NLin2009cAsym',
                                    output_spaces=['T1w', 'fsnative',
                                                  'template', 'fsaverage5'],
                                    medial_surface_nan=False,
                                    cifti_output=False,
                                    hires=True,
                                    use_bbr=True,
                                    bold2t1w_dof=9,
                                    fmap_bspline=False,
                                    fmap_demean=True,
                                    use_syn=True,
                                    force_syn=True,
                                    template_out_grid='native',
                                    use_aroma=False,
                                    aroma_melodic_dim=-200,
                                    ignore_aroma_err=False)

    Parameters

        subject_id : str
            List of subject labels
        task_id : str or None
            Task ID of BOLD series to preprocess, or ``None`` to preprocess all
        echo_idx : int or None
            Index of echo to preprocess in multiecho BOLD series,
            or ``None`` to preprocess all
        name : str
            Name of workflow
        ignore : list
            Preprocessing steps to skip (may include "slicetiming", "fieldmaps")
        debug : bool
            Enable debugging outputs
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        anat_only : bool
            Disable functional workflows
        longitudinal : bool
            Treat multiple sessions as longitudinal (may increase runtime)
            See sub-workflows for specific differences
        t2s_coreg : bool
            For multi-echo EPI, use the calculated T2*-map for T2*-driven coregistration
        omp_nthreads : int
            Maximum number of threads an individual process may use
        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        skull_strip_fixed_seed : bool
            Do not use a random seed for skull-stripping - will ensure
            run-to-run replicability when used with --omp-nthreads 1
        reportlets_dir : str
            Directory in which to save reportlets
        output_dir : str
            Directory in which to save derivatives
        bids_dir : str
            Root directory of BIDS dataset
        freesurfer : bool
            Enable FreeSurfer surface reconstruction (may increase runtime)
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
        medial_surface_nan : bool
            Replace medial wall values with NaNs on functional GIFTI files
        cifti_output : bool
            Generate bold CIFTI file in output spaces
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer
        use_bbr : bool or None
            Enable/disable boundary-based registration refinement.
            If ``None``, test BBR result for distortion before accepting.
        bold2t1w_dof : 6, 9 or 12
            Degrees-of-freedom for BOLD-T1w registration
        fmap_bspline : bool
            **Experimental**: Fit B-Spline field using least-squares
        fmap_demean : bool
            Demean voxel-shift map during unwarp
        use_syn : bool
            **Experimental**: Enable ANTs SyN-based susceptibility distortion correction (SDC).
            If fieldmaps are present and enabled, this is not run, by default.
        force_syn : bool
            **Temporary**: Always run SyN-based SDC
        template_out_grid : str
            Keyword ('native', '1mm' or '2mm') or path of custom reference
            image for normalization
        use_aroma : bool
            Perform ICA-AROMA on MNI-resampled functional series
        ignore_aroma_err : bool
            Do not fail on ICA-AROMA errors

    Inputs

        subjects_dir
            FreeSurfer SUBJECTS_DIR

    """
    if name in ('single_subject_wf', 'single_subject_fmripreptest_wf'):
        # for documentation purposes
        subject_data = {
            't1w': ['/completely/made/up/path/sub-01_T1w.nii.gz'],
            'bold': ['/completely/made/up/path/sub-01_task-nback_bold.nii.gz']
        }
        layout = None
    else:
        subject_data, layout = collect_data(bids_dir, subject_id, task_id, echo_idx)

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
performed using *fMRIPprep* {fmriprep_ver}
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


### References

""".format(nilearn_ver=nilearn_ver)

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']),
                        name='inputnode')

    bidssrc = pe.Node(BIDSDataGrabber(subject_data=subject_data, anat_only=anat_only),
                      name='bidssrc')

    bids_info = pe.Node(BIDSInfo(), name='bids_info', run_without_submitting=True)

    summary = pe.Node(SubjectSummary(output_spaces=output_spaces, template=template),
                      name='summary', run_without_submitting=True)

    about = pe.Node(AboutSummary(version=__version__,
                                 command=' '.join(sys.argv)),
                    name='about', run_without_submitting=True)

    ds_report_summary = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='summary'),
        name='ds_report_summary', run_without_submitting=True)

    ds_report_about = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='about'),
        name='ds_report_about', run_without_submitting=True)

    # Preprocessing of T1w (includes registration to MNI)
    anat_preproc_wf = init_anat_preproc_wf(name="anat_preproc_wf",
                                           skull_strip_template=skull_strip_template,
                                           skull_strip_fixed_seed=skull_strip_fixed_seed,
                                           output_spaces=output_spaces,
                                           template=template,
                                           debug=debug,
                                           longitudinal=longitudinal,
                                           omp_nthreads=omp_nthreads,
                                           freesurfer=freesurfer,
                                           hires=hires,
                                           reportlets_dir=reportlets_dir,
                                           output_dir=output_dir,
                                           num_t1w=len(subject_data['t1w']))

    workflow.connect([
        (inputnode, anat_preproc_wf, [('subjects_dir', 'inputnode.subjects_dir')]),
        (bidssrc, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('t1w', 't1w'),
                            ('t2w', 't2w'),
                            ('bold', 'bold')]),
        (bids_info, summary, [('subject_id', 'subject_id')]),
        (bidssrc, anat_preproc_wf, [('t1w', 'inputnode.t1w'),
                                    ('t2w', 'inputnode.t2w'),
                                    ('roi', 'inputnode.roi'),
                                    ('flair', 'inputnode.flair')]),
        (summary, anat_preproc_wf, [('subject_id', 'inputnode.subject_id')]),
        (bidssrc, ds_report_summary, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bidssrc, ds_report_about, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])

    if anat_only:
        return workflow

    for bold_file in subject_data['bold']:
        func_preproc_wf = init_func_preproc_wf(bold_file=bold_file,
                                               layout=layout,
                                               ignore=ignore,
                                               freesurfer=freesurfer,
                                               use_bbr=use_bbr,
                                               t2s_coreg=t2s_coreg,
                                               bold2t1w_dof=bold2t1w_dof,
                                               reportlets_dir=reportlets_dir,
                                               output_spaces=output_spaces,
                                               template=template,
                                               medial_surface_nan=medial_surface_nan,
                                               cifti_output=cifti_output,
                                               output_dir=output_dir,
                                               omp_nthreads=omp_nthreads,
                                               low_mem=low_mem,
                                               fmap_bspline=fmap_bspline,
                                               fmap_demean=fmap_demean,
                                               use_syn=use_syn,
                                               force_syn=force_syn,
                                               debug=debug,
                                               template_out_grid=template_out_grid,
                                               use_aroma=use_aroma,
                                               aroma_melodic_dim=aroma_melodic_dim,
                                               ignore_aroma_err=ignore_aroma_err,
                                               num_bold=len(subject_data['bold']))

        workflow.connect([
            (anat_preproc_wf, func_preproc_wf,
             [('outputnode.t1_preproc', 'inputnode.t1_preproc'),
              ('outputnode.t1_brain', 'inputnode.t1_brain'),
              ('outputnode.t1_mask', 'inputnode.t1_mask'),
              ('outputnode.t1_seg', 'inputnode.t1_seg'),
              ('outputnode.t1_aseg', 'inputnode.t1_aseg'),
              ('outputnode.t1_aparc', 'inputnode.t1_aparc'),
              ('outputnode.t1_tpms', 'inputnode.t1_tpms'),
              ('outputnode.t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
              ('outputnode.t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
              # Undefined if --no-freesurfer, but this is safe
              ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
              ('outputnode.subject_id', 'inputnode.subject_id'),
              ('outputnode.t1_2_fsnative_forward_transform',
               'inputnode.t1_2_fsnative_forward_transform'),
              ('outputnode.t1_2_fsnative_reverse_transform',
               'inputnode.t1_2_fsnative_reverse_transform')]),
        ])

    return workflow
