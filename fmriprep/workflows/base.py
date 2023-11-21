# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
                minimum_fs_version="7.0.0",
            ),
            name='fsdir_run_%s' % config.execution.run_uuid.replace('-', '_'),
            run_without_submitting=True,
        )
        if config.execution.fs_subjects_dir is not None:
            fsdir.inputs.subjects_dir = str(config.execution.fs_subjects_dir.absolute())

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.fmriprep_dir / f"sub-{subject_id}" / "log" / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        if freesurfer:
            fmriprep_wf.connect(fsdir, 'subjects_dir', single_subject_wf, 'inputnode.subjects_dir')
        else:
            fmriprep_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.fmriprep_dir / f"sub-{subject_id}" / 'log' / config.execution.run_uuid
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
        init_ds_surface_metrics_wf,
        init_template_iterator_wf,
    )
    from smriprep.workflows.surfaces import (
        init_gifti_morphometrics_wf,
        init_hcp_morphometrics_wf,
        init_morph_grayords_wf,
        init_resample_midthickness_wf,
    )

    from fmriprep.workflows.bold.base import init_bold_wf

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = """
Results included in this manuscript come from preprocessing
performed using *fMRIPrep* {fmriprep_ver}
(@fmriprep1; @fmriprep2; RRID:SCR_016216),
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

""".format(
        fmriprep_ver=config.environment.version, nipype_ver=config.environment.nipype_version
    )
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

""".format(
        nilearn_ver=NILEARN_VERSION
    )

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
        task_id = config.execution.task_id
        raise RuntimeError(
            "No BOLD images found for participant {} and task {}. "
            "All workflows require BOLD images.".format(
                subject_id, task_id if task_id else '<all>'
            )
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
            f"Lesion mask {subject_data['roi']} found. "
            "Future versions of fMRIPrep will use alternative conventions. "
            "Please refer to the documentation before upgrading.",
            FutureWarning,
        )

    spaces = config.workflow.spaces
    msm_sulc = config.workflow.run_msmsulc

    anatomical_cache = {}
    if config.execution.derivatives:
        from smriprep.utils.bids import collect_derivatives as collect_anat_derivatives

        std_spaces = spaces.get_spaces(nonstandard=False, dim=(3,))
        std_spaces.append("fsnative")
        for deriv_dir in config.execution.derivatives:
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
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmriprep_dir,
            desc='about',
            datatype="figures",
            dismiss_entities=("echo",),
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

    workflow.connect([
        (inputnode, anat_fit_wf, [('subjects_dir', 'inputnode.subjects_dir')]),
        (bidssrc, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
        (bidssrc, anat_fit_wf, [
            ('t1w', 'inputnode.t1w'),
            ('t2w', 'inputnode.t2w'),
            ('roi', 'inputnode.roi'),
            ('flair', 'inputnode.flair'),
        ]),
        (bids_info, anat_fit_wf, [(('subject', _prefix), 'inputnode.subject_id')]),
        # Reporting connections
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('t1w', 't1w'), ('t2w', 't2w'), ('bold', 'bold')]),
        (bids_info, summary, [('subject', 'subject_id')]),
        (bidssrc, ds_report_summary, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (bidssrc, ds_report_about, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])  # fmt:skip

    # Set up the template iterator once, if used
    template_iterator_wf = None
    select_MNI2009c_xfm = None
    if config.workflow.level == "full":
        if spaces.cached.get_spaces(nonstandard=False, dim=(3,)):
            template_iterator_wf = init_template_iterator_wf(spaces=spaces)
            ds_std_volumes_wf = init_ds_anat_volumes_wf(
                bids_root=bids_root,
                output_dir=fmriprep_dir,
                name="ds_std_volumes_wf",
            )
            workflow.connect([
                (anat_fit_wf, template_iterator_wf, [
                    ('outputnode.template', 'inputnode.template'),
                    ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
                ]),
                (anat_fit_wf, ds_std_volumes_wf, [
                    ('outputnode.t1w_valid_list', 'inputnode.source_files'),
                    ("outputnode.t1w_preproc", "inputnode.t1w_preproc"),
                    ("outputnode.t1w_mask", "inputnode.t1w_mask"),
                    ("outputnode.t1w_dseg", "inputnode.t1w_dseg"),
                    ("outputnode.t1w_tpms", "inputnode.t1w_tpms"),
                ]),
                (template_iterator_wf, ds_std_volumes_wf, [
                    ("outputnode.std_t1w", "inputnode.ref_file"),
                    ("outputnode.anat2std_xfm", "inputnode.anat2std_xfm"),
                    ("outputnode.space", "inputnode.space"),
                    ("outputnode.cohort", "inputnode.cohort"),
                    ("outputnode.resolution", "inputnode.resolution"),
                ]),
            ])  # fmt:skip

        if 'MNI152NLin2009cAsym' in spaces.get_spaces():
            select_MNI2009c_xfm = pe.Node(
                KeySelect(fields=["std2anat_xfm"], key="MNI152NLin2009cAsym"),
                name="select_MNI2009c_xfm",
                run_without_submitting=True,
            )
            workflow.connect([
                (anat_fit_wf, select_MNI2009c_xfm, [
                    ("outputnode.std2anat_xfm", "std2anat_xfm"),
                    ("outputnode.template", "keys"),
                ]),
            ])  # fmt:skip

        # Thread MNI152NLin6Asym standard outputs to CIFTI subworkflow, skipping
        # the iterator, which targets only output spaces.
        # This can lead to duplication in the working directory if people actually
        # want MNI152NLin6Asym outputs, but we'll live with it.
        if config.workflow.cifti_output:
            from smriprep.interfaces.templateflow import TemplateFlowSelect

            ref = Reference(
                "MNI152NLin6Asym",
                {"res": 2 if config.workflow.cifti_output == "91k" else 1},
            )

            select_MNI6_xfm = pe.Node(
                KeySelect(fields=["anat2std_xfm"], key=ref.fullname),
                name="select_MNI6",
                run_without_submitting=True,
            )
            select_MNI6_tpl = pe.Node(
                TemplateFlowSelect(template=ref.fullname, resolution=ref.spec['res']),
                name="select_MNI6_tpl",
            )
            workflow.connect([
                (anat_fit_wf, select_MNI6_xfm, [
                    ("outputnode.anat2std_xfm", "anat2std_xfm"),
                    ("outputnode.template", "keys"),
                ]),
            ])  # fmt:skip

            # Create CIFTI morphometrics
            curv_wf = init_gifti_morphometrics_wf(morphometrics=['curv'], name='curv_wf')
            hcp_morphometrics_wf = init_hcp_morphometrics_wf(omp_nthreads=omp_nthreads)
            morph_grayords_wf = init_morph_grayords_wf(
                grayord_density=config.workflow.cifti_output,
                omp_nthreads=omp_nthreads,
            )
            resample_midthickness_wf = init_resample_midthickness_wf(
                grayord_density=config.workflow.cifti_output,
            )
            ds_grayord_metrics_wf = init_ds_grayord_metrics_wf(
                bids_root=bids_root,
                output_dir=fmriprep_dir,
                metrics=["curv", "thickness", "sulc"],
                cifti_output=config.workflow.cifti_output,
            )

            workflow.connect([
                (anat_fit_wf, curv_wf, [
                    ("outputnode.subject_id", "inputnode.subject_id"),
                    ("outputnode.subjects_dir", "inputnode.subjects_dir"),
                ]),
                (anat_fit_wf, hcp_morphometrics_wf, [
                    ("outputnode.subject_id", "inputnode.subject_id"),
                    ("outputnode.thickness", "inputnode.thickness"),
                    ("outputnode.sulc", "inputnode.sulc"),
                    ("outputnode.midthickness", "inputnode.midthickness"),
                ]),
                (curv_wf, hcp_morphometrics_wf, [
                    ("outputnode.curv", "inputnode.curv"),
                ]),
                (anat_fit_wf, resample_midthickness_wf, [
                    ('outputnode.midthickness', 'inputnode.midthickness'),
                    (
                        f"outputnode.sphere_reg_{'msm' if msm_sulc else 'fsLR'}",
                        "inputnode.sphere_reg_fsLR",
                    ),
                ]),
                (anat_fit_wf, morph_grayords_wf, [
                    ("outputnode.midthickness", "inputnode.midthickness"),
                    (
                        f'outputnode.sphere_reg_{"msm" if msm_sulc else "fsLR"}',
                        'inputnode.sphere_reg_fsLR',
                    ),
                ]),
                (hcp_morphometrics_wf, morph_grayords_wf, [
                    ("outputnode.curv", "inputnode.curv"),
                    ("outputnode.thickness", "inputnode.thickness"),
                    ("outputnode.sulc", "inputnode.sulc"),
                    ("outputnode.roi", "inputnode.roi"),
                ]),
                (resample_midthickness_wf, morph_grayords_wf, [
                    ('outputnode.midthickness_fsLR', 'inputnode.midthickness_fsLR'),
                ]),
                (anat_fit_wf, ds_grayord_metrics_wf, [
                    ('outputnode.t1w_valid_list', 'inputnode.source_files'),
                ]),
                (morph_grayords_wf, ds_grayord_metrics_wf, [
                    ("outputnode.curv_fsLR", "inputnode.curv"),
                    ("outputnode.curv_metadata", "inputnode.curv_metadata"),
                    ("outputnode.thickness_fsLR", "inputnode.thickness"),
                    ("outputnode.thickness_metadata", "inputnode.thickness_metadata"),
                    ("outputnode.sulc_fsLR", "inputnode.sulc"),
                    ("outputnode.sulc_metadata", "inputnode.sulc_metadata"),
                ]),
            ])  # fmt:skip

    if config.workflow.anat_only:
        return clean_datasinks(workflow)

    fmap_estimators, estimator_map = map_fieldmap_estimation(
        layout=config.execution.layout,
        subject_id=subject_id,
        bold_data=bold_runs,
        ignore_fieldmaps="fieldmaps" in config.workflow.ignore,
        use_syn=config.workflow.use_syn_sdc,
        force_syn=config.workflow.force_syn,
        filters=config.execution.get().get('bids_filters', {}).get('fmap'),
    )

    if fmap_estimators:
        config.loggers.workflow.info(
            "B0 field inhomogeneity map will be estimated with the following "
            f"{len(fmap_estimators)} estimator(s): "
            f"{[e.method for e in fmap_estimators]}."
        )

        from sdcflows import fieldmaps as fm
        from sdcflows.workflows.base import init_fmap_preproc_wf

        fmap_wf = init_fmap_preproc_wf(
            debug="fieldmaps" in config.execution.debug,
            estimators=fmap_estimators,
            omp_nthreads=omp_nthreads,
            output_dir=fmriprep_dir,
            subject=subject_id,
        )
        fmap_wf.__desc__ = f"""

Preprocessing of B<sub>0</sub> inhomogeneity mappings

: A total of {len(fmap_estimators)} fieldmaps were found available within the input
BIDS structure for this particular subject.
"""

        # Overwrite ``out_path_base`` of sdcflows's DataSinks
        for node in fmap_wf.list_node_names():
            if node.split(".")[-1].startswith("ds_"):
                fmap_wf.get_node(node).interface.out_path_base = ""

        fmap_select_std = pe.Node(
            KeySelect(fields=["std2anat_xfm"], key="MNI152NLin2009cAsym"),
            name="fmap_select_std",
            run_without_submitting=True,
        )
        if any(estimator.method == fm.EstimatorType.ANAT for estimator in fmap_estimators):
            # fmt:off
            workflow.connect([
                (anat_fit_wf, fmap_select_std, [
                    ("outputnode.std2anat_xfm", "std2anat_xfm"),
                    ("outputnode.template", "keys")]),
            ])
            # fmt:on

        for estimator in fmap_estimators:
            config.loggers.workflow.info(
                f"""\
Setting-up fieldmap "{estimator.bids_id}" ({estimator.method}) with \
<{', '.join(s.path.name for s in estimator.sources)}>"""
            )

            # Mapped and phasediff can be connected internally by SDCFlows
            if estimator.method in (fm.EstimatorType.MAPPED, fm.EstimatorType.PHASEDIFF):
                continue

            suffices = [s.suffix for s in estimator.sources]

            if estimator.method == fm.EstimatorType.PEPOLAR:
                if len(suffices) == 2 and all(suf in ("epi", "bold", "sbref") for suf in suffices):
                    wf_inputs = getattr(fmap_wf.inputs, f"in_{estimator.bids_id}")
                    wf_inputs.in_data = [str(s.path) for s in estimator.sources]
                    wf_inputs.metadata = [s.metadata for s in estimator.sources]
                else:
                    raise NotImplementedError("Sophisticated PEPOLAR schemes are unsupported.")

            elif estimator.method == fm.EstimatorType.ANAT:
                from sdcflows.workflows.fit.syn import init_syn_preprocessing_wf

                sources = [str(s.path) for s in estimator.sources if s.suffix in ("bold", "sbref")]
                source_meta = [
                    s.metadata for s in estimator.sources if s.suffix in ("bold", "sbref")
                ]
                syn_preprocessing_wf = init_syn_preprocessing_wf(
                    omp_nthreads=omp_nthreads,
                    debug=config.execution.sloppy,
                    auto_bold_nss=True,
                    t1w_inversion=False,
                    name=f"syn_preprocessing_{estimator.bids_id}",
                )
                syn_preprocessing_wf.inputs.inputnode.in_epis = sources
                syn_preprocessing_wf.inputs.inputnode.in_meta = source_meta

                # fmt:off
                workflow.connect([
                    (anat_fit_wf, syn_preprocessing_wf, [
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

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = """
Functional data preprocessing

: For each of the {num_bold} BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
""".format(
        num_bold=len(bold_runs)
    )

    for bold_series in bold_runs:
        bold_file = bold_series[0]
        fieldmap_id = estimator_map.get(bold_file)

        functional_cache = {}
        if config.execution.derivatives:
            from fmriprep.utils.bids import collect_derivatives, extract_entities

            entities = extract_entities(bold_series)

            for deriv_dir in config.execution.derivatives:
                functional_cache.update(
                    collect_derivatives(
                        derivatives_dir=deriv_dir,
                        entities=entities,
                        fieldmap_id=fieldmap_id,
                    )
                )

        bold_wf = init_bold_wf(
            bold_series=bold_series,
            precomputed=functional_cache,
            fieldmap_id=fieldmap_id,
        )
        if bold_wf is None:
            continue

        bold_wf.__desc__ = func_pre_desc + (bold_wf.__desc__ or "")

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
        if fieldmap_id:
            workflow.connect([
                (fmap_wf, bold_wf, [
                    ("outputnode.fmap", "inputnode.fmap"),
                    ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                    ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                    ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                    ("outputnode.fmap_id", "inputnode.fmap_id"),
                    ("outputnode.method", "inputnode.sdc_method"),
                ]),
            ])  # fmt:skip

        if config.workflow.level == "full":
            if template_iterator_wf is not None:
                workflow.connect([
                    (template_iterator_wf, bold_wf, [
                        ("outputnode.anat2std_xfm", "inputnode.anat2std_xfm"),
                        ("outputnode.space", "inputnode.std_space"),
                        ("outputnode.resolution", "inputnode.std_resolution"),
                        ("outputnode.cohort", "inputnode.std_cohort"),
                        ("outputnode.std_t1w", "inputnode.std_t1w"),
                        ("outputnode.std_mask", "inputnode.std_mask"),
                    ]),
                ])  # fmt:skip

            if select_MNI2009c_xfm is not None:
                workflow.connect([
                    (select_MNI2009c_xfm, bold_wf, [
                        ("std2anat_xfm", "inputnode.mni2009c2anat_xfm"),
                    ]),
                ])  # fmt:skip

            # Thread MNI152NLin6Asym standard outputs to CIFTI subworkflow, skipping
            # the iterator, which targets only output spaces.
            # This can lead to duplication in the working directory if people actually
            # want MNI152NLin6Asym outputs, but we'll live with it.
            if config.workflow.cifti_output:
                workflow.connect([
                    (select_MNI6_xfm, bold_wf, [("anat2std_xfm", "inputnode.anat2mni6_xfm")]),
                    (select_MNI6_tpl, bold_wf, [("brain_mask", "inputnode.mni6_mask")]),
                    (hcp_morphometrics_wf, bold_wf, [
                        ("outputnode.roi", "inputnode.cortex_mask"),
                    ]),
                    (resample_midthickness_wf, bold_wf, [
                        ('outputnode.midthickness_fsLR', 'inputnode.midthickness_fsLR'),
                    ]),
                ])  # fmt:skip

    return clean_datasinks(workflow)


def map_fieldmap_estimation(
    layout: bids.BIDSLayout,
    subject_id: str,
    bold_data: list[list[str]],
    ignore_fieldmaps: bool,
    use_syn: bool | str,
    force_syn: bool,
    filters: dict | None,
) -> tuple[list, dict]:
    if not any((not ignore_fieldmaps, use_syn, force_syn)):
        return [], {}

    from sdcflows import fieldmaps as fm
    from sdcflows.utils.wrangler import find_estimators

    # In the case where fieldmaps are ignored and `--use-syn-sdc` is requested,
    # SDCFlows `find_estimators` still receives a full layout (which includes the fmap modality)
    # and will not calculate fmapless schemes.
    # Similarly, if fieldmaps are ignored and `--force-syn` is requested,
    # `fmapless` should be set to True to ensure BOLD targets are found to be corrected.
    fmap_estimators = find_estimators(
        layout=layout,
        subject=subject_id,
        fmapless=bool(use_syn) or ignore_fieldmaps and force_syn,
        force_fmapless=force_syn or ignore_fieldmaps and use_syn,
        bids_filters=filters,
    )

    if not fmap_estimators:
        if use_syn:
            message = (
                "Fieldmap-less (SyN) estimation was requested, but PhaseEncodingDirection "
                "information appears to be absent."
            )
            config.loggers.workflow.error(message)
            if use_syn == "error":
                raise ValueError(message)
        return [], {}

    if ignore_fieldmaps and any(f.method == fm.EstimatorType.ANAT for f in fmap_estimators):
        config.loggers.workflow.info(
            'Option "--ignore fieldmaps" was set, but either "--use-syn-sdc" '
            'or "--force-syn" were given, so fieldmap-less estimation will be executed.'
        )
        fmap_estimators = [f for f in fmap_estimators if f.method == fm.EstimatorType.ANAT]

    # Pare down estimators to those that are actually used
    # If fmap_estimators == [], all loops/comprehensions terminate immediately
    all_ids = {fmap.bids_id for fmap in fmap_estimators}
    bold_files = (bold_series[0] for bold_series in bold_data)

    all_estimators = {
        bold_file: [fmap_id for fmap_id in get_estimator(layout, bold_file) if fmap_id in all_ids]
        for bold_file in bold_files
    }

    for bold_file, estimator_key in all_estimators.items():
        if len(estimator_key) > 1:
            config.loggers.workflow.warning(
                f"Several fieldmaps <{', '.join(estimator_key)}> are "
                f"'IntendedFor' <{bold_file}>, using {estimator_key[0]}"
            )
            estimator_key[1:] = []

    # Final, 1-1 map, dropping uncorrected BOLD
    estimator_map = {
        bold_file: estimator_key[0]
        for bold_file, estimator_key in all_estimators.items()
        if estimator_key
    }

    fmap_estimators = [f for f in fmap_estimators if f.bids_id in estimator_map.values()]

    return fmap_estimators, estimator_map


def _prefix(subid):
    return subid if subid.startswith('sub-') else f'sub-{subid}'


def clean_datasinks(workflow: pe.Workflow) -> pe.Workflow:
    # Overwrite ``out_path_base`` of smriprep's DataSinks
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ""
    return workflow


def get_estimator(layout, fname):
    field_source = layout.get_metadata(fname).get("B0FieldSource")
    if isinstance(field_source, str):
        field_source = (field_source,)

    if field_source is None:
        import re
        from pathlib import Path

        from sdcflows.fieldmaps import get_identifier

        # Fallback to IntendedFor
        intended_rel = re.sub(r"^sub-[a-zA-Z0-9]*/", "", str(Path(fname).relative_to(layout.root)))
        field_source = get_identifier(intended_rel)

    return field_source
