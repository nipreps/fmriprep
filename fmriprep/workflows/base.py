# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
fMRIPrep base processing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fmriprep_wf
.. autofunction:: init_single_subject_wf

"""

import os
import re
import sys
import warnings
from copy import deepcopy

import bids
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.utils.connections import listify
from packaging.version import Version

from .. import config
from ..interfaces import DerivativesDataSink
from ..interfaces.reports import AboutSummary, SubjectSummary
from ..utils.bids import dismiss_echo


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
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSFreeSurferDir

    ver = Version(config.environment.version)

    fmriprep_wf = Workflow(name=f'fmriprep_{ver.major}_{ver.minor}_wf')
    fmriprep_wf.base_dir = config.execution.work_dir

    freesurfer = config.workflow.run_reconall
    if freesurfer:
        fsdir = pe.Node(
            BIDSFreeSurferDir(
                derivatives=config.execution.output_dir,
                freesurfer_home=os.getenv('FREESURFER_HOME'),
                spaces=config.workflow.spaces.get_fs_spaces(),
                minimum_fs_version='7.0.0',
            ),
            name='fsdir_run_{}'.format(config.execution.run_uuid.replace('-', '_')),
            run_without_submitting=True,
        )
        if config.execution.fs_subjects_dir is not None:
            fsdir.inputs.subjects_dir = str(config.execution.fs_subjects_dir.absolute())

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.fmriprep_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        if freesurfer:
            fmriprep_wf.connect(fsdir, 'subjects_dir', single_subject_wf, 'inputnode.subjects_dir')
        else:
            fmriprep_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.fmriprep_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'fmriprep.toml')

    return fmriprep_wf


def init_single_subject_wf(subject_id: str):
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
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSDataGrabber, BIDSInfo
    from niworkflows.interfaces.nilearn import NILEARN_VERSION
    from niworkflows.interfaces.utility import KeySelect
    from niworkflows.utils.bids import collect_data
    from niworkflows.utils.misc import fix_multi_T1w_source_name
    from niworkflows.utils.spaces import Reference
    from smriprep.workflows.anatomical import init_anat_fit_wf
    from smriprep.workflows.outputs import (
        init_ds_anat_volumes_wf,
        init_ds_grayord_metrics_wf,
        init_ds_surfaces_wf,
        init_template_iterator_wf,
    )
    from smriprep.workflows.surfaces import (
        init_gifti_morphometrics_wf,
        init_hcp_morphometrics_wf,
        init_morph_grayords_wf,
        init_resample_surfaces_wf,
    )

    from fmriprep.workflows.bold.base import init_bold_wf

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = f"""
Results included in this manuscript come from preprocessing
performed using *fMRIPrep* {config.environment.version}
(@fmriprep1; @fmriprep2; RRID:SCR_016216),
which is based on *Nipype* {config.environment.nipype_version}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    workflow.__postdesc__ = f"""

Many internal operations of *fMRIPrep* use
*Nilearn* {NILEARN_VERSION} [@nilearn, RRID:SCR_001362],
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

"""

    subject_data = collect_data(
        config.execution.layout,
        subject_id,
        task=config.execution.task_id,
        echo=config.execution.echo_idx,
        bids_filters=config.execution.bids_filters,
    )[0]

    if 'flair' in config.workflow.ignore:
        subject_data['flair'] = []
    if 't2w' in config.workflow.ignore:
        subject_data['t2w'] = []

    anat_only = config.workflow.anat_only
    # Make sure we always go through these two checks
    if not anat_only and not subject_data['bold']:
        task_id = config.execution.task_id or '<all>'
        raise RuntimeError(
            f'No BOLD images found for participant {subject_id} and '
            f'task {task_id}. All workflows require BOLD images.'
        )

    bold_runs = [
        sorted(
            listify(run),
            key=lambda file: config.execution.layout.get_metadata(file).get('EchoTime', 0),
        )
        for run in subject_data['bold']
    ]

    if subject_data['roi']:
        warnings.warn(
            f'Lesion mask {subject_data["roi"]} found. '
            'Future versions of fMRIPrep will use alternative conventions. '
            'Please refer to the documentation before upgrading.',
            FutureWarning,
            stacklevel=1,
        )

    spaces = config.workflow.spaces
    msm_sulc = config.workflow.run_msmsulc

    anatomical_cache = {}
    if config.execution.derivatives:
        from smriprep.utils.bids import collect_derivatives as collect_anat_derivatives

        std_spaces = spaces.get_spaces(nonstandard=False, dim=(3,))
        std_spaces.append('fsnative')
        for deriv_dir in config.execution.derivatives.values():
            anatomical_cache.update(
                collect_anat_derivatives(
                    derivatives_dir=deriv_dir,
                    subject_id=subject_id,
                    std_spaces=std_spaces,
                )
            )

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']), name='inputnode')

    bidssrc = pe.Node(
        BIDSDataGrabber(
            subject_data=subject_data,
            anat_only=config.workflow.anat_only,
            subject_id=subject_id,
            anat_derivatives=anatomical_cache if anatomical_cache else None,
        ),
        name='bidssrc',
    )

    bids_info = pe.Node(
        BIDSInfo(bids_dir=config.execution.bids_dir, bids_validate=False), name='bids_info'
    )

    summary = pe.Node(
        SubjectSummary(
            std_spaces=spaces.get_spaces(nonstandard=False),
            nstd_spaces=spaces.get_spaces(standard=False),
        ),
        name='summary',
        run_without_submitting=True,
    )

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmriprep_dir,
            desc='summary',
            datatype='figures',
            dismiss_entities=dismiss_echo(),
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmriprep_dir,
            desc='about',
            datatype='figures',
            dismiss_entities=dismiss_echo(),
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )

    bids_root = str(config.execution.bids_dir)
    fmriprep_dir = str(config.execution.fmriprep_dir)
    omp_nthreads = config.nipype.omp_nthreads

    # Build the workflow
    anat_fit_wf = init_anat_fit_wf(
        bids_root=bids_root,
        output_dir=fmriprep_dir,
        freesurfer=config.workflow.run_reconall,
        hires=config.workflow.hires,
        fs_no_resume=config.workflow.fs_no_resume,
        longitudinal=config.workflow.longitudinal,
        msm_sulc=msm_sulc,
        t1w=subject_data['t1w'],
        t2w=subject_data['t2w'],
        flair=subject_data['flair'],
        skull_strip_mode=config.workflow.skull_strip_t1w,
        skull_strip_template=Reference.from_string(config.workflow.skull_strip_template)[0],
        spaces=spaces,
        precomputed=anatomical_cache,
        omp_nthreads=omp_nthreads,
        sloppy=config.execution.sloppy,
        skull_strip_fixed_seed=config.workflow.skull_strip_fixed_seed,
    )

    # allow to run with anat-fast-track on fMRI-only dataset
    if 't1w_preproc' in anatomical_cache and not subject_data['t1w']:
        config.loggers.workflow.debug(
            'No T1w image found; using precomputed T1w image: %s', anatomical_cache['t1w_preproc']
        )
        workflow.connect([
            (bidssrc, bids_info, [(('bold', fix_multi_T1w_source_name), 'in_file')]),
            (anat_fit_wf, summary, [('outputnode.t1w_preproc', 't1w')]),
            (anat_fit_wf, ds_report_summary, [('outputnode.t1w_preproc', 'source_file')]),
            (anat_fit_wf, ds_report_about, [('outputnode.t1w_preproc', 'source_file')]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (bidssrc, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
            (bidssrc, summary, [('t1w', 't1w')]),
            (bidssrc, ds_report_summary, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
            (bidssrc, ds_report_about, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        ])  # fmt:skip

    workflow.connect([
        (inputnode, anat_fit_wf, [('subjects_dir', 'inputnode.subjects_dir')]),
        (bidssrc, anat_fit_wf, [
            ('t1w', 'inputnode.t1w'),
            ('t2w', 'inputnode.t2w'),
            ('roi', 'inputnode.roi'),
            ('flair', 'inputnode.flair'),
        ]),
        (bids_info, anat_fit_wf, [(('subject', _prefix), 'inputnode.subject_id')]),
        # Reporting connections
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('t2w', 't2w'), ('bold', 'bold')]),
        (bids_info, summary, [('subject', 'subject_id')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])  # fmt:skip

    # Set up the template iterator once, if used
    template_iterator_wf = None
    select_MNI2009c_xfm = None
    if config.workflow.level == 'full':
        if spaces.cached.get_spaces(nonstandard=False, dim=(3,)):
            template_iterator_wf = init_template_iterator_wf(
                spaces=spaces, sloppy=config.execution.sloppy
            )
            ds_std_volumes_wf = init_ds_anat_volumes_wf(
                bids_root=bids_root,
                output_dir=fmriprep_dir,
                name='ds_std_volumes_wf',
            )
            workflow.connect([
                (anat_fit_wf, template_iterator_wf, [
                    ('outputnode.template', 'inputnode.template'),
                    ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
                ]),
                (anat_fit_wf, ds_std_volumes_wf, [
                    ('outputnode.t1w_valid_list', 'inputnode.source_files'),
                    ('outputnode.t1w_preproc', 'inputnode.anat_preproc'),
                    ('outputnode.t1w_mask', 'inputnode.anat_mask'),
                    ('outputnode.t1w_dseg', 'inputnode.anat_dseg'),
                    ('outputnode.t1w_tpms', 'inputnode.anat_tpms'),
                ]),
                (template_iterator_wf, ds_std_volumes_wf, [
                    ('outputnode.std_t1w', 'inputnode.ref_file'),
                    ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
                    ('outputnode.space', 'inputnode.space'),
                    ('outputnode.cohort', 'inputnode.cohort'),
                    ('outputnode.resolution', 'inputnode.resolution'),
                ]),
            ])  # fmt:skip

        if 'MNI152NLin2009cAsym' in spaces.get_spaces():
            select_MNI2009c_xfm = pe.Node(
                KeySelect(fields=['std2anat_xfm'], key='MNI152NLin2009cAsym'),
                name='select_MNI2009c_xfm',
                run_without_submitting=True,
            )
            workflow.connect([
                (anat_fit_wf, select_MNI2009c_xfm, [
                    ('outputnode.std2anat_xfm', 'std2anat_xfm'),
                    ('outputnode.template', 'keys'),
                ]),
            ])  # fmt:skip

        # Thread MNI152NLin6Asym standard outputs to CIFTI subworkflow, skipping
        # the iterator, which targets only output spaces.
        # This can lead to duplication in the working directory if people actually
        # want MNI152NLin6Asym outputs, but we'll live with it.
        if config.workflow.cifti_output:
            from smriprep.interfaces.templateflow import TemplateFlowSelect

            ref = Reference(
                'MNI152NLin6Asym',
                {'res': 2 if config.workflow.cifti_output == '91k' else 1},
            )

            select_MNI6_xfm = pe.Node(
                KeySelect(fields=['anat2std_xfm'], key=ref.fullname),
                name='select_MNI6',
                run_without_submitting=True,
            )
            select_MNI6_tpl = pe.Node(
                TemplateFlowSelect(template=ref.fullname, resolution=ref.spec['res']),
                name='select_MNI6_tpl',
            )
            workflow.connect([
                (anat_fit_wf, select_MNI6_xfm, [
                    ('outputnode.anat2std_xfm', 'anat2std_xfm'),
                    ('outputnode.template', 'keys'),
                ]),
            ])  # fmt:skip

            # Create CIFTI morphometrics
            curv_wf = init_gifti_morphometrics_wf(morphometrics=['curv'], name='curv_wf')
            hcp_morphometrics_wf = init_hcp_morphometrics_wf(omp_nthreads=omp_nthreads)
            morph_grayords_wf = init_morph_grayords_wf(
                grayord_density=config.workflow.cifti_output,
                omp_nthreads=omp_nthreads,
            )
            resample_surfaces_wf = init_resample_surfaces_wf(
                surfaces=['white', 'pial', 'midthickness'],
                grayord_density=config.workflow.cifti_output,
            )
            ds_grayord_metrics_wf = init_ds_grayord_metrics_wf(
                bids_root=bids_root,
                output_dir=fmriprep_dir,
                metrics=['curv', 'thickness', 'sulc'],
                cifti_output=config.workflow.cifti_output,
            )
            ds_fsLR_surfaces_wf = init_ds_surfaces_wf(
                output_dir=fmriprep_dir,
                surfaces=['white', 'pial', 'midthickness'],
                entities={
                    'space': 'fsLR',
                    'density': '32k' if config.workflow.cifti_output == '91k' else '59k',
                },
                name='ds_fsLR_surfaces_wf',
            )

            workflow.connect([
                (anat_fit_wf, curv_wf, [
                    ('outputnode.subject_id', 'inputnode.subject_id'),
                    ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
                ]),
                (anat_fit_wf, hcp_morphometrics_wf, [
                    ('outputnode.subject_id', 'inputnode.subject_id'),
                    ('outputnode.thickness', 'inputnode.thickness'),
                    ('outputnode.sulc', 'inputnode.sulc'),
                    ('outputnode.midthickness', 'inputnode.midthickness'),
                ]),
                (curv_wf, hcp_morphometrics_wf, [
                    ('outputnode.curv', 'inputnode.curv'),
                ]),
                (anat_fit_wf, resample_surfaces_wf, [
                    ('outputnode.white', 'inputnode.white'),
                    ('outputnode.pial', 'inputnode.pial'),
                    ('outputnode.midthickness', 'inputnode.midthickness'),
                    (
                        f"outputnode.sphere_reg_{'msm' if msm_sulc else 'fsLR'}",
                        'inputnode.sphere_reg_fsLR',
                    ),
                ]),
                (anat_fit_wf, morph_grayords_wf, [
                    ('outputnode.midthickness', 'inputnode.midthickness'),
                    (
                        f'outputnode.sphere_reg_{"msm" if msm_sulc else "fsLR"}',
                        'inputnode.sphere_reg_fsLR',
                    ),
                ]),
                (hcp_morphometrics_wf, morph_grayords_wf, [
                    ('outputnode.curv', 'inputnode.curv'),
                    ('outputnode.thickness', 'inputnode.thickness'),
                    ('outputnode.sulc', 'inputnode.sulc'),
                    ('outputnode.roi', 'inputnode.roi'),
                ]),
                (resample_surfaces_wf, morph_grayords_wf, [
                    ('outputnode.midthickness_fsLR', 'inputnode.midthickness_fsLR'),
                ]),
                (anat_fit_wf, ds_grayord_metrics_wf, [
                    ('outputnode.t1w_valid_list', 'inputnode.source_files'),
                ]),
                (anat_fit_wf, ds_fsLR_surfaces_wf, [
                    ('outputnode.t1w_valid_list', 'inputnode.source_files'),
                ]),
                (morph_grayords_wf, ds_grayord_metrics_wf, [
                    ('outputnode.curv_fsLR', 'inputnode.curv'),
                    ('outputnode.curv_metadata', 'inputnode.curv_metadata'),
                    ('outputnode.thickness_fsLR', 'inputnode.thickness'),
                    ('outputnode.thickness_metadata', 'inputnode.thickness_metadata'),
                    ('outputnode.sulc_fsLR', 'inputnode.sulc'),
                    ('outputnode.sulc_metadata', 'inputnode.sulc_metadata'),
                ]),
                (resample_surfaces_wf, ds_fsLR_surfaces_wf, [
                    ('outputnode.white_fsLR', 'inputnode.white'),
                    ('outputnode.pial_fsLR', 'inputnode.pial'),
                    ('outputnode.midthickness_fsLR', 'inputnode.midthickness'),
                ]),
            ])  # fmt:skip

    if config.workflow.anat_only:
        return clean_datasinks(workflow)

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = f"""
Functional data preprocessing

: For each of the {len(bold_runs)} BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
"""

    # Before initializing BOLD workflow, select/verify anatomical target for coregistration
    if config.workflow.bold2anat_init in ('auto', 't2w'):
        has_t2w = subject_data['t2w'] or 't2w_preproc' in anatomical_cache
        if config.workflow.bold2anat_init == 't2w' and not has_t2w:
            raise OSError(
                'A T2w image is expected for BOLD-to-anatomical coregistration and was not found'
            )
        config.workflow.bold2anat_init = 't2w' if has_t2w else 't1w'

    for bold_series in bold_runs:
        functional_cache = {}
        if config.execution.derivatives:
            from fmriprep.utils.bids import collect_derivatives, extract_entities

            entities = extract_entities(bold_series)

            for deriv_dir in config.execution.derivatives.values():
                functional_cache.update(
                    collect_derivatives(
                        derivatives_dir=deriv_dir,
                        entities=entities,
                    )
                )

        bold_wf = init_bold_wf(
            bold_series=bold_series,
            precomputed=functional_cache,
        )
        if bold_wf is None:
            continue

        bold_wf.__desc__ = func_pre_desc + (bold_wf.__desc__ or '')

        workflow.connect([
            (anat_fit_wf, bold_wf, [
                ('outputnode.t1w_preproc', 'inputnode.t1w_preproc'),
                ('outputnode.t1w_mask', 'inputnode.t1w_mask'),
                ('outputnode.t1w_dseg', 'inputnode.t1w_dseg'),
                ('outputnode.t1w_tpms', 'inputnode.t1w_tpms'),
                ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
                ('outputnode.subject_id', 'inputnode.subject_id'),
                ('outputnode.fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm'),
                ('outputnode.white', 'inputnode.white'),
                ('outputnode.pial', 'inputnode.pial'),
                ('outputnode.midthickness', 'inputnode.midthickness'),
                ('outputnode.anat_ribbon', 'inputnode.anat_ribbon'),
                (
                    f'outputnode.sphere_reg_{"msm" if msm_sulc else "fsLR"}',
                    'inputnode.sphere_reg_fsLR',
                ),
            ]),
        ])  # fmt:skip

        if config.workflow.level == 'full':
            if template_iterator_wf is not None:
                workflow.connect([
                    (template_iterator_wf, bold_wf, [
                        ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
                        ('outputnode.space', 'inputnode.std_space'),
                        ('outputnode.resolution', 'inputnode.std_resolution'),
                        ('outputnode.cohort', 'inputnode.std_cohort'),
                        ('outputnode.std_t1w', 'inputnode.std_t1w'),
                        ('outputnode.std_mask', 'inputnode.std_mask'),
                    ]),
                ])  # fmt:skip

            if select_MNI2009c_xfm is not None:
                workflow.connect([
                    (select_MNI2009c_xfm, bold_wf, [
                        ('std2anat_xfm', 'inputnode.mni2009c2anat_xfm'),
                    ]),
                ])  # fmt:skip

            # Thread MNI152NLin6Asym standard outputs to CIFTI subworkflow, skipping
            # the iterator, which targets only output spaces.
            # This can lead to duplication in the working directory if people actually
            # want MNI152NLin6Asym outputs, but we'll live with it.
            if config.workflow.cifti_output:
                workflow.connect([
                    (select_MNI6_xfm, bold_wf, [('anat2std_xfm', 'inputnode.anat2mni6_xfm')]),
                    (select_MNI6_tpl, bold_wf, [('brain_mask', 'inputnode.mni6_mask')]),
                    (hcp_morphometrics_wf, bold_wf, [
                        ('outputnode.roi', 'inputnode.cortex_mask'),
                    ]),
                    (resample_surfaces_wf, bold_wf, [
                        ('outputnode.midthickness_fsLR', 'inputnode.midthickness_fsLR'),
                    ]),
                ])  # fmt:skip

    return clean_datasinks(workflow)


def _prefix(subid):
    return subid if subid.startswith('sub-') else f'sub-{subid}'


def clean_datasinks(workflow: pe.Workflow) -> pe.Workflow:
    # Overwrite ``out_path_base`` of smriprep's DataSinks
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ''
    return workflow
