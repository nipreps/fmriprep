# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fMRIPrep base processing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fmriprep_wf
.. autofunction:: init_single_subject_wf

"""

import sys
import os
from copy import deepcopy

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.interfaces.nilearn import NILEARN_VERSION

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.bids import (
    BIDSInfo, BIDSDataGrabber, BIDSFreeSurferDir
)
from niworkflows.utils.bids import collect_data
from niworkflows.utils.spaces import Reference
from niworkflows.utils.misc import fix_multi_T1w_source_name
from smriprep.workflows.anatomical import init_anat_preproc_wf

from .. import config
from ..interfaces import SubjectSummary, AboutSummary, DerivativesDataSink
from .bold import init_func_preproc_wf


def init_fmriprep_wf():
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

            from fmriprep.workflows.tests import mock_config
            from fmriprep.workflows.base import init_fmriprep_wf
            with mock_config():
                wf = init_fmriprep_wf()

    """
    fmriprep_wf = Workflow(name='fmriprep_wf')
    fmriprep_wf.base_dir = config.execution.work_dir

    freesurfer = config.workflow.run_reconall
    if freesurfer:
        fsdir = pe.Node(
            BIDSFreeSurferDir(
                derivatives=config.execution.output_dir,
                freesurfer_home=os.getenv('FREESURFER_HOME'),
                spaces=config.workflow.spaces.get_fs_spaces()),
            name='fsdir_run_%s' % config.execution.run_uuid.replace('-', '_'),
            run_without_submitting=True)
        if config.execution.fs_subjects_dir is not None:
            fsdir.inputs.subjects_dir = str(config.execution.fs_subjects_dir.absolute())

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.output_dir / "fmriprep" / "-".join(("sub", subject_id))
            / "log" / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        if freesurfer:
            fmriprep_wf.connect(fsdir, 'subjects_dir',
                                single_subject_wf, 'inputnode.subjects_dir')
        else:
            fmriprep_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = config.execution.output_dir / 'fmriprep' / 'sub-{}'.format(subject_id) \
            / 'log' / config.execution.run_uuid
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'fmriprep.toml')

    return fmriprep_wf


def init_single_subject_wf(subject_id):
    """
    Organize the preprocessing pipeline for a single subject.

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

            from fmriprep.workflows.tests import mock_config
            from fmriprep.workflows.base import init_single_subject_wf
            with mock_config():
                wf = init_single_subject_wf('01')

    Parameters
    ----------
    subject_id : :obj:`str`
        Subject label for this single-subject workflow.

    Inputs
    ------
    subjects_dir : :obj:`str`
        FreeSurfer's ``$SUBJECTS_DIR``.

    """
    name = "single_subject_%s_wf" % subject_id
    subject_data = collect_data(
        config.execution.layout,
        subject_id,
        config.execution.task_id,
        config.execution.echo_idx,
        bids_filters=config.execution.bids_filters)[0]

    if 'flair' in config.workflow.ignore:
        subject_data['flair'] = []
    if 't2w' in config.workflow.ignore:
        subject_data['t2w'] = []

    anat_only = config.workflow.anat_only
    # Make sure we always go through these two checks
    if not anat_only and not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            "No BOLD images found for participant {} and task {}. "
            "All workflows require BOLD images.".format(
                subject_id, task_id if task_id else '<all>')
        )

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

""".format(fmriprep_ver=config.environment.version,
           nipype_ver=config.environment.nipype_version)
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

    spaces = config.workflow.spaces
    reportlets_dir = str(config.execution.work_dir / 'reportlets')

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']),
                        name='inputnode')

    bidssrc = pe.Node(BIDSDataGrabber(subject_data=subject_data,
                                      anat_only=anat_only,
                                      subject_id=subject_id),
                      name='bidssrc')

    bids_info = pe.Node(BIDSInfo(
        bids_dir=config.execution.bids_dir, bids_validate=False), name='bids_info')

    summary = pe.Node(SubjectSummary(std_spaces=spaces.get_spaces(nonstandard=False),
                                     nstd_spaces=spaces.get_spaces(standard=False)),
                      name='summary', run_without_submitting=True)

    about = pe.Node(AboutSummary(version=config.environment.version,
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
        bids_root=str(config.execution.bids_dir),
        debug=config.execution.debug is True,
        freesurfer=config.workflow.run_reconall,
        hires=config.workflow.hires,
        longitudinal=config.workflow.longitudinal,
        omp_nthreads=config.nipype.omp_nthreads,
        output_dir=str(config.execution.output_dir),
        reportlets_dir=reportlets_dir,
        skull_strip_fixed_seed=config.workflow.skull_strip_fixed_seed,
        skull_strip_mode=config.workflow.skull_strip_t1w,
        skull_strip_template=Reference.from_string(
            config.workflow.skull_strip_template)[0],
        spaces=spaces,
        t1w=subject_data['t1w'],
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

    # Append the functional section to the existing anatomical exerpt
    # That way we do not need to stream down the number of bold datasets
    anat_preproc_wf.__postdesc__ = (anat_preproc_wf.__postdesc__ or '') + """

Functional data preprocessing

: For each of the {num_bold} BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
""".format(num_bold=len(subject_data['bold']))

    for bold_file in subject_data['bold']:
        func_preproc_wf = init_func_preproc_wf(bold_file)

        workflow.connect([
            (anat_preproc_wf, func_preproc_wf,
             [(('outputnode.t1w_preproc', _pop), 'inputnode.t1w_preproc'),
              ('outputnode.t1w_mask', 'inputnode.t1w_mask'),
              ('outputnode.t1w_dseg', 'inputnode.t1w_dseg'),
              ('outputnode.t1w_aseg', 'inputnode.t1w_aseg'),
              ('outputnode.t1w_aparc', 'inputnode.t1w_aparc'),
              ('outputnode.t1w_tpms', 'inputnode.t1w_tpms'),
              ('outputnode.template', 'inputnode.template'),
              ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
              ('outputnode.std2anat_xfm', 'inputnode.std2anat_xfm'),
              # Undefined if --fs-no-reconall, but this is safe
              ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
              ('outputnode.subject_id', 'inputnode.subject_id'),
              ('outputnode.t1w2fsnative_xfm', 'inputnode.t1w2fsnative_xfm'),
              ('outputnode.fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm')]),
        ])
    return workflow


def _prefix(subid):
    return '-'.join(('sub', subid.lstrip('sub-')))


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist
