# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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

import sys
import os
from copy import deepcopy

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from .. import config
from ..interfaces import DerivativesDataSink
from ..interfaces.reports import SubjectSummary, AboutSummary
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
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSFreeSurferDir

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
            config.execution.fmriprep_dir / f"sub-{subject_id}"
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
        log_dir = config.execution.fmriprep_dir / f"sub-{subject_id}" \
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
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSInfo, BIDSDataGrabber
    from niworkflows.interfaces.nilearn import NILEARN_VERSION
    from niworkflows.utils.bids import collect_data
    from niworkflows.utils.misc import fix_multi_T1w_source_name
    from niworkflows.utils.spaces import Reference
    from smriprep.workflows.anatomical import init_anat_preproc_wf

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
    anat_derivatives = config.execution.anat_derivatives
    spaces = config.workflow.spaces
    # Make sure we always go through these two checks
    if not anat_only and not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            "No BOLD images found for participant {} and task {}. "
            "All workflows require BOLD images.".format(
                subject_id, task_id if task_id else '<all>')
        )

    if anat_derivatives:
        from smriprep.utils.bids import collect_derivatives
        std_spaces = spaces.get_spaces(nonstandard=False, dim=(3,))
        anat_derivatives = collect_derivatives(
            anat_derivatives.absolute(),
            subject_id,
            std_spaces,
            config.workflow.run_reconall,
        )
        if anat_derivatives is None:
            config.loggers.workflow.warning(f"""\
Attempted to access pre-existing anatomical derivatives at \
<{config.execution.anat_derivatives}>, however not all expectations of fMRIPrep \
were met (for participant <{subject_id}>, spaces <{', '.join(std_spaces)}>, \
reconall <{config.workflow.run_reconall}>).""")

    if not anat_derivatives and not subject_data['t1w']:
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

    fmriprep_dir = str(config.execution.fmriprep_dir)

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']),
                        name='inputnode')

    bidssrc = pe.Node(BIDSDataGrabber(subject_data=subject_data,
                                      anat_only=anat_only,
                                      anat_derivatives=anat_derivatives,
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
        DerivativesDataSink(base_directory=fmriprep_dir, desc='summary', datatype="figures",
                            dismiss_entities=("echo",)),
        name='ds_report_summary', run_without_submitting=True)

    ds_report_about = pe.Node(
        DerivativesDataSink(base_directory=fmriprep_dir, desc='about', datatype="figures",
                            dismiss_entities=("echo",)),
        name='ds_report_about', run_without_submitting=True)

    # Preprocessing of T1w (includes registration to MNI)
    anat_preproc_wf = init_anat_preproc_wf(
        bids_root=str(config.execution.bids_dir),
        debug=config.execution.sloppy,
        existing_derivatives=anat_derivatives,
        freesurfer=config.workflow.run_reconall,
        hires=config.workflow.hires,
        longitudinal=config.workflow.longitudinal,
        omp_nthreads=config.nipype.omp_nthreads,
        output_dir=fmriprep_dir,
        skull_strip_fixed_seed=config.workflow.skull_strip_fixed_seed,
        skull_strip_mode=config.workflow.skull_strip_t1w,
        skull_strip_template=Reference.from_string(
            config.workflow.skull_strip_template)[0],
        spaces=spaces,
        t1w=subject_data['t1w'],
    )

    workflow.connect([
        (inputnode, anat_preproc_wf, [('subjects_dir', 'inputnode.subjects_dir')]),
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('bold', 'bold')]),
        (bids_info, summary, [('subject', 'subject_id')]),
        (bids_info, anat_preproc_wf, [(('subject', _prefix), 'inputnode.subject_id')]),
        (bidssrc, anat_preproc_wf, [('t1w', 'inputnode.t1w'),
                                    ('t2w', 'inputnode.t2w'),
                                    ('roi', 'inputnode.roi'),
                                    ('flair', 'inputnode.flair')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])

    if not anat_derivatives:
        workflow.connect([
            (bidssrc, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
            (bidssrc, summary, [('t1w', 't1w'),
                                ('t2w', 't2w')]),
            (bidssrc, ds_report_summary, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
            (bidssrc, ds_report_about, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        ])
    else:
        workflow.connect([
            (bidssrc, bids_info, [(('bold', fix_multi_T1w_source_name), 'in_file')]),
            (anat_preproc_wf, summary, [('outputnode.t1w_preproc', 't1w')]),
            (anat_preproc_wf, ds_report_summary, [('outputnode.t1w_preproc', 'source_file')]),
            (anat_preproc_wf, ds_report_about, [('outputnode.t1w_preproc', 'source_file')]),
        ])

    # Overwrite ``out_path_base`` of smriprep's DataSinks
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ""

    if anat_only:
        return workflow

    from sdcflows import fieldmaps as fm
    fmap_estimators = None

    if any(("fieldmaps" not in config.workflow.ignore,
            config.workflow.use_syn_sdc,
            config.workflow.force_syn)):
        from sdcflows.utils.wrangler import find_estimators

        # SDC Step 1: Run basic heuristics to identify available data for fieldmap estimation
        # For now, no fmapless
        fmap_estimators = find_estimators(
            layout=config.execution.layout,
            subject=subject_id,
            fmapless=bool(config.workflow.use_syn_sdc),
            force_fmapless=config.workflow.force_syn,
        )

        if config.workflow.use_syn_sdc and not fmap_estimators:
            message = ("Fieldmap-less (SyN) estimation was requested, but "
                       "PhaseEncodingDirection information appears to be "
                       "absent.")
            config.loggers.workflow.error(message)
            if config.workflow.use_syn_sdc == "error":
                raise ValueError(message)

        if (
            "fieldmaps" in config.workflow.ignore
            and [f for f in fmap_estimators
                 if f.method != fm.EstimatorType.ANAT]
        ):
            config.loggers.workflow.info(
                'Option "--ignore fieldmaps" was set, but either "--use-syn-sdc" '
                'or "--force-syn" were given, so fieldmap-less estimation will be executed.'
            )
            fmap_estimators = [f for f in fmap_estimators
                               if f.method == fm.EstimatorType.ANAT]

        if fmap_estimators:
            config.loggers.workflow.info(
                "B0 field inhomogeneity map will be estimated with "
                f" the following {len(fmap_estimators)} estimators: "
                f"{[e.method for e in fmap_estimators]}."
            )

    # Append the functional section to the existing anatomical exerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = """
Functional data preprocessing

: For each of the {num_bold} BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
""".format(num_bold=len(subject_data['bold']))

    func_preproc_wfs = []
    has_fieldmap = bool(fmap_estimators)
    for bold_file in subject_data['bold']:
        func_preproc_wf = init_func_preproc_wf(bold_file, has_fieldmap=has_fieldmap)
        if func_preproc_wf is None:
            continue

        func_preproc_wf.__desc__ = func_pre_desc + (func_preproc_wf.__desc__ or "")
        workflow.connect([
            (anat_preproc_wf, func_preproc_wf,
             [('outputnode.t1w_preproc', 'inputnode.t1w_preproc'),
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
        func_preproc_wfs.append(func_preproc_wf)

    if not has_fieldmap:
        return workflow

    from sdcflows.workflows.base import init_fmap_preproc_wf

    fmap_wf = init_fmap_preproc_wf(
        debug="fieldmaps" in config.execution.debug,
        estimators=fmap_estimators,
        omp_nthreads=config.nipype.omp_nthreads,
        output_dir=fmriprep_dir,
        subject=subject_id,
    )
    fmap_wf.__desc__ = f"""

Preprocessing of B<sub>0</sub> inhomogeneity mappings

: A total of {len(fmap_estimators)} fieldmaps were found available within the input
BIDS structure for this particular subject.
"""
    for func_preproc_wf in func_preproc_wfs:
        # fmt: off
        workflow.connect([
            (fmap_wf, func_preproc_wf, [
                ("outputnode.fmap", "inputnode.fmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                ("outputnode.fmap_id", "inputnode.fmap_id"),
                ("outputnode.method", "inputnode.sdc_method"),
            ]),
        ])
        # fmt: on

    # Overwrite ``out_path_base`` of sdcflows's DataSinks
    for node in fmap_wf.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            fmap_wf.get_node(node).interface.out_path_base = ""

    # Step 3: Manually connect PEPOLAR and ANAT workflows

    # Select "MNI152NLin2009cAsym" from standard references.
    # This node may be used by multiple ANAT estimators, so define outside loop.
    from niworkflows.interfaces.utility import KeySelect
    fmap_select_std = pe.Node(
        KeySelect(fields=["std2anat_xfm"], key="MNI152NLin2009cAsym"),
        name="fmap_select_std",
        run_without_submitting=True,
    )
    if any(estimator.method == fm.EstimatorType.ANAT for estimator in fmap_estimators):
        # fmt:off
        workflow.connect([
            (anat_preproc_wf, fmap_select_std, [
                ("outputnode.std2anat_xfm", "std2anat_xfm"),
                ("outputnode.template", "keys")]),
        ])
        # fmt:on

    for estimator in fmap_estimators:
        config.loggers.workflow.info(f"""\
Setting-up fieldmap "{estimator.bids_id}" ({estimator.method}) with \
<{', '.join(s.path.name for s in estimator.sources)}>""")

        # Mapped and phasediff can be connected internally by SDCFlows
        if estimator.method in (fm.EstimatorType.MAPPED, fm.EstimatorType.PHASEDIFF):
            continue

        suffices = [s.suffix for s in estimator.sources]

        if estimator.method == fm.EstimatorType.PEPOLAR:
            if set(suffices) == {"epi"} or sorted(suffices) == ["bold", "epi"]:
                wf_inputs = getattr(fmap_wf.inputs, f"in_{estimator.bids_id}")
                wf_inputs.in_data = [str(s.path) for s in estimator.sources]
                wf_inputs.metadata = [s.metadata for s in estimator.sources]

                # 21.0.x hack to change the number of volumes used
                # The default of 50 takes excessively long
                flatten = fmap_wf.get_node(f"wf_{estimator.bids_id}.flatten")
                flatten.inputs.max_trs = config.workflow.topup_max_vols
            else:
                raise NotImplementedError(
                    "Sophisticated PEPOLAR schemes are unsupported."
                )

        elif estimator.method == fm.EstimatorType.ANAT:
            from sdcflows.workflows.fit.syn import init_syn_preprocessing_wf

            sources = [str(s.path) for s in estimator.sources if s.suffix == "bold"]
            source_meta = [s.metadata for s in estimator.sources if s.suffix == "bold"]
            syn_preprocessing_wf = init_syn_preprocessing_wf(
                omp_nthreads=config.nipype.omp_nthreads,
                debug=config.execution.sloppy,
                auto_bold_nss=True,
                t1w_inversion=False,
                name=f"syn_preprocessing_{estimator.bids_id}",
            )
            syn_preprocessing_wf.inputs.inputnode.in_epis = sources
            syn_preprocessing_wf.inputs.inputnode.in_meta = source_meta

            # fmt:off
            workflow.connect([
                (anat_preproc_wf, syn_preprocessing_wf, [
                    ("outputnode.t1w_preproc", "inputnode.in_anat"),
                    ("outputnode.t1w_mask", "inputnode.mask_anat"),
                ]),
                (fmap_select_std, syn_preprocessing_wf, [
                    ("std2anat_xfm", "inputnode.std2anat_xfm"),
                ]),
                (syn_preprocessing_wf, fmap_wf, [
                    ("outputnode.epi_ref", f"in_{estimator.bids_id}.epi_ref"),
                    ("outputnode.epi_mask", f"in_{estimator.bids_id}.epi_mask"),
                    ("outputnode.anat_ref", f"in_{estimator.bids_id}.anat_ref"),
                    ("outputnode.anat_mask", f"in_{estimator.bids_id}.anat_mask"),
                    ("outputnode.sd_prior", f"in_{estimator.bids_id}.sd_prior"),
                ]),
            ])
            # fmt:on
    return workflow


def _prefix(subid):
    return subid if subid.startswith('sub-') else f'sub-{subid}'
