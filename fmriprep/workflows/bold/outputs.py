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
"""Writing out derivative files."""
from __future__ import annotations

import typing as ty

import numpy as np
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from niworkflows.utils.images import dseg_label
from smriprep.workflows.outputs import _bids_relative

from fmriprep import config
from fmriprep.config import DEFAULT_MEMORY_MIN_GB
from fmriprep.interfaces import DerivativesDataSink

if ty.TYPE_CHECKING:
    from niworkflows.utils.spaces import SpatialReferences


def prepare_timing_parameters(metadata: dict):
    """Convert initial timing metadata to post-realignment timing metadata

    In particular, SliceTiming metadata is invalid once STC or any realignment is applied,
    as a matrix of voxels no longer corresponds to an acquisition slice.
    Therefore, if SliceTiming is present in the metadata dictionary, and a sparse
    acquisition paradigm is detected, DelayTime or AcquisitionDuration must be derived to
    preserve the timing interpretation.

    Examples
    --------

    .. testsetup::

        >>> from unittest import mock

    If SliceTiming metadata is absent, then the only change is to note that
    STC has not been applied:

    >>> prepare_timing_parameters(dict(RepetitionTime=2))
    {'RepetitionTime': 2, 'SliceTimingCorrected': False}
    >>> prepare_timing_parameters(dict(RepetitionTime=2, DelayTime=0.5))
    {'RepetitionTime': 2, 'DelayTime': 0.5, 'SliceTimingCorrected': False}
    >>> prepare_timing_parameters(dict(VolumeTiming=[0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
    ...                                AcquisitionDuration=1.0))  #doctest: +NORMALIZE_WHITESPACE
    {'VolumeTiming': [0.0, 1.0, 2.0, 5.0, 6.0, 7.0], 'AcquisitionDuration': 1.0,
     'SliceTimingCorrected': False}

    When SliceTiming is available and used, then ``SliceTimingCorrected`` is ``True``
    and the ``StartTime`` indicates a series offset.

    >>> with mock.patch("fmriprep.config.workflow.ignore", []):
    ...     prepare_timing_parameters(dict(RepetitionTime=2, SliceTiming=[0.0, 0.2, 0.4, 0.6]))
    {'RepetitionTime': 2, 'SliceTimingCorrected': True, 'DelayTime': 1.2, 'StartTime': 0.3}
    >>> with mock.patch("fmriprep.config.workflow.ignore", []):
    ...     prepare_timing_parameters(
    ...         dict(VolumeTiming=[0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
    ...              SliceTiming=[0.0, 0.2, 0.4, 0.6, 0.8]))  #doctest: +NORMALIZE_WHITESPACE
    {'VolumeTiming': [0.0, 1.0, 2.0, 5.0, 6.0, 7.0], 'SliceTimingCorrected': True,
     'AcquisitionDuration': 1.0, 'StartTime': 0.4}

    When SliceTiming is available and not used, then ``SliceTimingCorrected`` is ``False``
    and TA is indicated with ``DelayTime`` or ``AcquisitionDuration``.

    >>> with mock.patch("fmriprep.config.workflow.ignore", ["slicetiming"]):
    ...     prepare_timing_parameters(dict(RepetitionTime=2, SliceTiming=[0.0, 0.2, 0.4, 0.6]))
    {'RepetitionTime': 2, 'SliceTimingCorrected': False, 'DelayTime': 1.2}
    >>> with mock.patch("fmriprep.config.workflow.ignore", ["slicetiming"]):
    ...     prepare_timing_parameters(
    ...         dict(VolumeTiming=[0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
    ...              SliceTiming=[0.0, 0.2, 0.4, 0.6, 0.8]))  #doctest: +NORMALIZE_WHITESPACE
    {'VolumeTiming': [0.0, 1.0, 2.0, 5.0, 6.0, 7.0], 'SliceTimingCorrected': False,
     'AcquisitionDuration': 1.0}

    If SliceTiming metadata is present but empty, then treat it as missing:

    >>> with mock.patch("fmriprep.config.workflow.ignore", []):
    ...     prepare_timing_parameters(dict(RepetitionTime=2, SliceTiming=[]))
    {'RepetitionTime': 2, 'SliceTimingCorrected': False}
    >>> with mock.patch("fmriprep.config.workflow.ignore", []):
    ...     prepare_timing_parameters(dict(RepetitionTime=2, SliceTiming=[0.0]))
    {'RepetitionTime': 2, 'SliceTimingCorrected': False}
    """
    timing_parameters = {
        key: metadata[key]
        for key in (
            "RepetitionTime",
            "VolumeTiming",
            "DelayTime",
            "AcquisitionDuration",
            "SliceTiming",
        )
        if key in metadata
    }

    # Treat SliceTiming of [] or length 1 as equivalent to missing and remove it in any case
    slice_timing = timing_parameters.pop("SliceTiming", [])

    run_stc = len(slice_timing) > 1 and 'slicetiming' not in config.workflow.ignore
    timing_parameters["SliceTimingCorrected"] = bool(run_stc)

    if len(slice_timing) > 1:
        st = sorted(slice_timing)
        TA = st[-1] + (st[1] - st[0])  # Final slice onset + slice duration
        # For constant TR paradigms, use DelayTime
        if "RepetitionTime" in timing_parameters:
            TR = timing_parameters["RepetitionTime"]
            if not np.isclose(TR, TA) and TA < TR:
                timing_parameters["DelayTime"] = TR - TA
        # For variable TR paradigms, use AcquisitionDuration
        elif "VolumeTiming" in timing_parameters:
            timing_parameters["AcquisitionDuration"] = TA

        if run_stc:
            first, last = st[0], st[-1]
            frac = config.workflow.slice_time_ref
            tzero = np.round(first + frac * (last - first), 3)
            timing_parameters["StartTime"] = tzero

    return timing_parameters


def init_func_fit_reports_wf(
    *,
    sdc_correction: bool,
    freesurfer: bool,
    output_dir: str,
    name="func_fit_reports_wf",
) -> pe.Workflow:
    """
    Set up a battery of datasinks to store reports in the right location.

    Parameters
    ----------
    freesurfer : :obj:`bool`
        FreeSurfer was enabled
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: anat_reports_wf)

    Inputs
    ------
    source_file
        Input BOLD images

    std_t1w
        T1w image resampled to standard space
    std_mask
        Mask of skull-stripped template
    subject_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    t1w_conform_report
        Conformation report
    t1w_preproc
        The T1w reference map, which is calculated as the average of bias-corrected
        and preprocessed T1w images, defining the anatomical space.
    t1w_dseg
        Segmentation in T1w space
    t1w_mask
        Brain (binary) mask estimated by brain extraction.
    template
        Template space and specifications

    """
    from niworkflows.interfaces.reportlets.registration import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )
    from sdcflows.interfaces.reportlets import FieldmapReportlet

    workflow = pe.Workflow(name=name)

    inputfields = [
        "source_file",
        "sdc_boldref",
        "coreg_boldref",
        "boldref2anat_xfm",
        "boldref2fmap_xfm",
        "t1w_preproc",
        "t1w_mask",
        "t1w_dseg",
        "fieldmap",
        "fmap_ref",
        # May be missing
        "subject_id",
        "subjects_dir",
        # Report snippets
        "summary_report",
        "validation_report",
    ]
    inputnode = pe.Node(niu.IdentityInterface(fields=inputfields), name="inputnode")

    ds_summary = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="summary",
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name="ds_report_summary",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    ds_validation = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="validation",
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name="ds_report_validation",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    # Resample anatomical references into BOLD space for plotting
    t1w_boldref = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            float=True,
            invert_transform_flags=[True],
            interpolation="LanczosWindowedSinc",
        ),
        name="t1w_boldref",
        mem_gb=1,
    )

    t1w_wm = pe.Node(
        niu.Function(function=dseg_label),
        name="t1w_wm",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    t1w_wm.inputs.label = 2  # BIDS default is WM=2

    boldref_wm = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            invert_transform_flags=[True],
            interpolation="NearestNeighbor",
        ),
        name="boldref_wm",
        mem_gb=1,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_summary, [
            ('source_file', 'source_file'),
            ('summary_report', 'in_file'),
        ]),
        (inputnode, ds_validation, [
            ('source_file', 'source_file'),
            ('validation_report', 'in_file'),
        ]),
        (inputnode, t1w_boldref, [
            ('t1w_preproc', 'input_image'),
            ('coreg_boldref', 'reference_image'),
            ('boldref2anat_xfm', 'transforms'),
        ]),
        (inputnode, t1w_wm, [('t1w_dseg', 'in_seg')]),
        (inputnode, boldref_wm, [
            ('coreg_boldref', 'reference_image'),
            ('boldref2anat_xfm', 'transforms'),
        ]),
        (t1w_wm, boldref_wm, [('out', 'input_image')]),
    ])
    # fmt:on

    # Reportlets follow the structure of init_bold_fit_wf stages
    # - SDC1:
    #       Before: Pre-SDC boldref
    #       After: Fieldmap reference resampled on boldref
    #       Three-way: Fieldmap resampled on boldref
    # - SDC2:
    #       Before: Pre-SDC boldref with white matter mask
    #       After: Post-SDC boldref with white matter mask
    # - EPI-T1 registration:
    #       Before: T1w brain with white matter mask
    #       After: Resampled boldref with white matter mask

    if sdc_correction:
        fmapref_boldref = pe.Node(
            ApplyTransforms(
                dimension=3,
                default_value=0,
                float=True,
                invert_transform_flags=[True],
                interpolation="LanczosWindowedSinc",
            ),
            name="fmapref_boldref",
            mem_gb=1,
        )

        # SDC1
        sdcreg_report = pe.Node(
            FieldmapReportlet(
                reference_label="BOLD reference",
                moving_label="Fieldmap reference",
                show="both",
            ),
            name="sdecreg_report",
            mem_gb=0.1,
        )

        ds_sdcreg_report = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="fmapCoreg",
                suffix="bold",
                datatype="figures",
                dismiss_entities=("echo",),
            ),
            name="ds_sdcreg_report",
        )

        # SDC2
        sdc_report = pe.Node(
            SimpleBeforeAfter(
                before_label="Distorted",
                after_label="Corrected",
                dismiss_affine=True,
            ),
            name="sdc_report",
            mem_gb=0.1,
        )

        ds_sdc_report = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="sdc",
                suffix="bold",
                datatype="figures",
                dismiss_entities=("echo",),
            ),
            name="ds_sdc_report",
        )

        # fmt:off
        workflow.connect([
            (inputnode, fmapref_boldref, [
                ('fmap_ref', 'input_image'),
                ('coreg_boldref', 'reference_image'),
                ('boldref2fmap_xfm', 'transforms'),
            ]),
            (inputnode, sdcreg_report, [
                ('sdc_boldref', 'reference'),
                ('fieldmap', 'fieldmap')
            ]),
            (fmapref_boldref, sdcreg_report, [('output_image', 'moving')]),
            (inputnode, ds_sdcreg_report, [('source_file', 'source_file')]),
            (sdcreg_report, ds_sdcreg_report, [('out_report', 'in_file')]),
            (inputnode, sdc_report, [
                ('sdc_boldref', 'before'),
                ('coreg_boldref', 'after'),
            ]),
            (boldref_wm, sdc_report, [('output_image', 'wm_seg')]),
            (inputnode, ds_sdc_report, [('source_file', 'source_file')]),
            (sdc_report, ds_sdc_report, [('out_report', 'in_file')]),
        ])
        # fmt:on

    # EPI-T1 registration
    # Resample T1w image onto EPI-space

    epi_t1_report = pe.Node(
        SimpleBeforeAfter(
            before_label="T1w",
            after_label="EPI",
            dismiss_affine=True,
        ),
        name="epi_t1_report",
        mem_gb=0.1,
    )

    ds_epi_t1_report = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="coreg",
            suffix="bold",
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name="ds_epi_t1_report",
    )

    # fmt:off
    workflow.connect([
        (inputnode, epi_t1_report, [('coreg_boldref', 'after')]),
        (t1w_boldref, epi_t1_report, [('output_image', 'before')]),
        (boldref_wm, epi_t1_report, [('output_image', 'wm_seg')]),
        (inputnode, ds_epi_t1_report, [('source_file', 'source_file')]),
        (epi_t1_report, ds_epi_t1_report, [('out_report', 'in_file')]),
    ])
    # fmt:on

    return workflow


def init_ds_boldref_wf(
    *,
    bids_root,
    output_dir,
    desc: str,
    name="ds_boldref_wf",
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["source_files", "boldref"]),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["boldref"]), name="outputnode")

    raw_sources = pe.Node(niu.Function(function=_bids_relative), name="raw_sources")
    raw_sources.inputs.bids_root = bids_root

    ds_boldref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc=desc,
            suffix="boldref",
            compress=True,
            dismiss_entities=("echo",),
        ),
        name="ds_boldref",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, raw_sources, [('source_files', 'in_files')]),
        (inputnode, ds_boldref, [('boldref', 'in_file'),
                                 ('source_files', 'source_file')]),
        (raw_sources, ds_boldref, [('out', 'RawSources')]),
        (ds_boldref, outputnode, [('out_file', 'boldref')]),
    ])
    # fmt:on

    return workflow


def init_ds_registration_wf(
    *,
    bids_root: str,
    output_dir: str,
    source: str,
    dest: str,
    name: str,
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["source_files", "xform"]),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["xform"]), name="outputnode")

    raw_sources = pe.Node(niu.Function(function=_bids_relative), name="raw_sources")
    raw_sources.inputs.bids_root = bids_root

    ds_xform = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            mode='image',
            suffix='xfm',
            extension='.txt',
            dismiss_entities=('echo',),
            **{'from': source, 'to': dest},
        ),
        name='ds_xform',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # fmt:off
    workflow.connect([
        (inputnode, raw_sources, [('source_files', 'in_files')]),
        (inputnode, ds_xform, [('xform', 'in_file'),
                               ('source_files', 'source_file')]),
        (raw_sources, ds_xform, [('out', 'RawSources')]),
        (ds_xform, outputnode, [('out_file', 'xform')]),
    ])
    # fmt:on

    return workflow


def init_ds_hmc_wf(
    *,
    bids_root,
    output_dir,
    name="ds_hmc_wf",
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["source_files", "xforms"]),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["xforms"]), name="outputnode")

    raw_sources = pe.Node(niu.Function(function=_bids_relative), name="raw_sources")
    raw_sources.inputs.bids_root = bids_root

    ds_xforms = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="hmc",
            suffix="xfm",
            extension=".txt",
            compress=True,
            dismiss_entities=("echo",),
            **{"from": "orig", "to": "boldref"},
        ),
        name="ds_xforms",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, raw_sources, [('source_files', 'in_files')]),
        (inputnode, ds_xforms, [('xforms', 'in_file'),
                                ('source_files', 'source_file')]),
        (raw_sources, ds_xforms, [('out', 'RawSources')]),
        (ds_xforms, outputnode, [('out_file', 'xforms')]),
    ])
    # fmt:on

    return workflow


def init_ds_bold_native_wf(
    *,
    bids_root: str,
    output_dir: str,
    multiecho: bool,
    bold_output: bool,
    echo_output: bool,
    all_metadata: ty.List[dict],
    name="ds_bold_native_wf",
) -> pe.Workflow:
    metadata = all_metadata[0]
    timing_parameters = prepare_timing_parameters(metadata)

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_files',
                'bold',
                'bold_mask',
                'bold_echos',
                't2star',
            ]
        ),
        name='inputnode',
    )

    raw_sources = pe.Node(niu.Function(function=_bids_relative), name='raw_sources')
    raw_sources.inputs.bids_root = bids_root
    workflow.connect(inputnode, 'source_files', raw_sources, 'in_files')

    if bold_output:
        ds_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='preproc',
                compress=True,
                SkullStripped=multiecho,
                TaskName=metadata.get('TaskName'),
                dismiss_entities=("echo",),
                **timing_parameters,
            ),
            name='ds_bold',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_bold, [
                ('source_files', 'source_file'),
                ('bold', 'in_file'),
            ]),
        ])  # fmt:skip

    if bold_output or echo_output:
        ds_bold_mask = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='brain',
                suffix='mask',
                compress=True,
                dismiss_entities=("echo",),
            ),
            name='ds_bold_mask',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_bold_mask, [
                ('source_files', 'source_file'),
                ('bold_mask', 'in_file'),
            ]),
            (raw_sources, ds_bold_mask, [('out', 'RawSources')]),
        ])  # fmt:skip

    if bold_output and multiecho:
        t2star_meta = {
            'Units': 's',
            'EstimationReference': 'doi:10.1002/mrm.20900',
            'EstimationAlgorithm': 'monoexponential decay model',
        }
        ds_t2star = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='boldref',
                suffix='T2starmap',
                compress=True,
                dismiss_entities=("echo",),
                **t2star_meta,
            ),
            name='ds_t2star_bold',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_t2star, [
                ('source_files', 'source_file'),
                ('t2star', 'in_file'),
            ]),
            (raw_sources, ds_t2star, [('out', 'RawSources')]),
        ])  # fmt:skip

    if echo_output:
        ds_bold_echos = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='preproc',
                compress=True,
                SkullStripped=False,
                TaskName=metadata.get('TaskName'),
                **timing_parameters,
            ),
            iterfield=['source_file', 'in_file', 'meta_dict'],
            name='ds_bold_echos',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_bold_echos.inputs.meta_dict = [{"EchoTime": md["EchoTime"]} for md in all_metadata]
        workflow.connect([
            (inputnode, ds_bold_echos, [
                ('source_files', 'source_file'),
                ('bold_echos', 'in_file'),
            ]),
        ])  # fmt:skip

    return workflow


def init_func_derivatives_wf(
    bids_root: str,
    cifti_output: bool,
    freesurfer: bool,
    project_goodvoxels: bool,
    all_metadata: ty.List[dict],
    multiecho: bool,
    output_dir: str,
    spaces: SpatialReferences,
    name='func_derivatives_wf',
):
    """
    Set up a battery of datasinks to store derivatives in the right location.

    Parameters
    ----------
    bids_root : :obj:`str`
        Original BIDS dataset path.
    cifti_output : :obj:`bool`
        Whether the ``--cifti-output`` flag was set.
    freesurfer : :obj:`bool`
        Whether FreeSurfer anatomical processing was run.
    project_goodvoxels : :obj:`bool`
        Whether the option was used to exclude voxels with
        locally high coefficient of variation, or that lie outside the
        cortical surfaces, from the surface projection.
    metadata : :obj:`dict`
        Metadata dictionary associated to the BOLD run.
    multiecho : :obj:`bool`
        Derivatives were generated from multi-echo time series.
    output_dir : :obj:`str`
        Where derivatives should be written out to.
    spaces : :py:class:`~niworkflows.utils.spaces.SpatialReferences`
        A container for storing, organizing, and parsing spatial normalizations. Composed of
        :py:class:`~niworkflows.utils.spaces.Reference` objects representing spatial references.
        Each ``Reference`` contains a space, which is a string of either TemplateFlow template IDs
        (e.g., ``MNI152Lin``, ``MNI152NLin6Asym``, ``MNIPediatricAsym``), nonstandard references
        (e.g., ``T1w`` or ``anat``, ``sbref``, ``run``, etc.), or a custom template located in
        the TemplateFlow root directory. Each ``Reference`` may also contain a spec, which is a
        dictionary with template specifications (e.g., a specification of ``{'resolution': 2}``
        would lead to resampling on a 2mm resolution of the space).
    name : :obj:`str`
        This workflow's identifier (default: ``func_derivatives_wf``).

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.utility import KeySelect
    from smriprep.workflows.outputs import _bids_relative

    metadata = all_metadata[0]

    timing_parameters = prepare_timing_parameters(metadata)

    nonstd_spaces = set(spaces.get_nonstandard())
    workflow = Workflow(name=name)

    # BOLD series will generally be unmasked unless multiecho,
    # as the optimal combination is undefined outside a bounded mask
    masked = multiecho
    t2star_meta = {
        'Units': 's',
        'EstimationReference': 'doi:10.1002/mrm.20900',
        'EstimationAlgorithm': 'monoexponential decay model',
    }

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_aparc_std',
                'bold_aparc_t1',
                'bold_aseg_std',
                'bold_aseg_t1',
                'bold_cifti',
                'bold_mask_std',
                'bold_mask_t1',
                'bold_std',
                'bold_std_ref',
                'bold_t1',
                'bold_t1_ref',
                'bold_native',
                'bold_native_ref',
                'bold_mask_native',
                'bold_echos_native',
                'cifti_metadata',
                'cifti_density',
                'confounds',
                'confounds_metadata',
                'goodvoxels_mask',
                'source_file',
                'all_source_files',
                'surf_files',
                'surf_refs',
                'template',
                'spatial_reference',
                't2star_bold',
                't2star_t1',
                't2star_std',
                'bold2anat_xfm',
                'anat2bold_xfm',
                'hmc_xforms',
                'acompcor_masks',
                'tcompcor_mask',
            ]
        ),
        name='inputnode',
    )

    raw_sources = pe.Node(niu.Function(function=_bids_relative), name='raw_sources')
    raw_sources.inputs.bids_root = bids_root

    ds_confounds = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='confounds',
            suffix='timeseries',
            dismiss_entities=("echo",),
        ),
        name="ds_confounds",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_ref_t1w_xfm = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            to='T1w',
            mode='image',
            suffix='xfm',
            extension='.txt',
            dismiss_entities=('echo',),
            **{'from': 'scanner'},
        ),
        name='ds_ref_t1w_xfm',
        run_without_submitting=True,
    )
    ds_ref_t1w_inv_xfm = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            to='scanner',
            mode='image',
            suffix='xfm',
            extension='.txt',
            dismiss_entities=('echo',),
            **{'from': 'T1w'},
        ),
        name='ds_t1w_tpl_inv_xfm',
        run_without_submitting=True,
    )
    # fmt:off
    workflow.connect([
        (inputnode, raw_sources, [('all_source_files', 'in_files')]),
        (inputnode, ds_confounds, [('source_file', 'source_file'),
                                   ('confounds', 'in_file'),
                                   ('confounds_metadata', 'meta_dict')]),
        (inputnode, ds_ref_t1w_xfm, [('source_file', 'source_file'),
                                     ('bold2anat_xfm', 'in_file')]),
        (inputnode, ds_ref_t1w_inv_xfm, [('source_file', 'source_file'),
                                         ('anat2bold_xfm', 'in_file')]),
    ])
    # fmt:on

    # Output HMC and reference volume
    ds_bold_hmc_xfm = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            to='boldref',
            mode='image',
            suffix='xfm',
            extension='.txt',
            dismiss_entities=('echo',),
            **{'from': 'scanner'},
        ),
        name='ds_bold_hmc_xfm',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    ds_bold_native_ref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, suffix='boldref', compress=True, dismiss_entities=("echo",)
        ),
        name='ds_bold_native_ref',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_bold_hmc_xfm, [('source_file', 'source_file'),
                                      ('hmc_xforms', 'in_file')]),
        (inputnode, ds_bold_native_ref, [('source_file', 'source_file'),
                                         ('bold_native_ref', 'in_file')])
    ])
    # fmt:on

    # Resample to T1w space
    if nonstd_spaces.intersection(('T1w', 'anat')):
        ds_bold_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='preproc',
                compress=True,
                SkullStripped=masked,
                TaskName=metadata.get('TaskName'),
                **timing_parameters,
            ),
            name='ds_bold_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_bold_t1_ref = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                suffix='boldref',
                compress=True,
                dismiss_entities=("echo",),
            ),
            name='ds_bold_t1_ref',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_bold_mask_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='brain',
                suffix='mask',
                compress=True,
                dismiss_entities=("echo",),
            ),
            name='ds_bold_mask_t1',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        # fmt:off
        workflow.connect([
            (inputnode, ds_bold_t1, [('source_file', 'source_file'),
                                     ('bold_t1', 'in_file')]),
            (inputnode, ds_bold_t1_ref, [('source_file', 'source_file'),
                                         ('bold_t1_ref', 'in_file')]),
            (inputnode, ds_bold_mask_t1, [('source_file', 'source_file'),
                                          ('bold_mask_t1', 'in_file')]),
            (raw_sources, ds_bold_mask_t1, [('out', 'RawSources')]),
        ])
        # fmt:on
        if freesurfer:
            ds_bold_aseg_t1 = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    space='T1w',
                    desc='aseg',
                    suffix='dseg',
                    compress=True,
                    dismiss_entities=("echo",),
                ),
                name='ds_bold_aseg_t1',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB,
            )
            ds_bold_aparc_t1 = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    space='T1w',
                    desc='aparcaseg',
                    suffix='dseg',
                    compress=True,
                    dismiss_entities=("echo",),
                ),
                name='ds_bold_aparc_t1',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB,
            )
            # fmt:off
            workflow.connect([
                (inputnode, ds_bold_aseg_t1, [('source_file', 'source_file'),
                                              ('bold_aseg_t1', 'in_file')]),
                (inputnode, ds_bold_aparc_t1, [('source_file', 'source_file'),
                                               ('bold_aparc_t1', 'in_file')]),
            ])
            # fmt:on
        if multiecho:
            ds_t2star_t1 = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    space='T1w',
                    suffix='T2starmap',
                    compress=True,
                    dismiss_entities=("echo",),
                    **t2star_meta,
                ),
                name='ds_t2star_t1',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB,
            )
            # fmt:off
            workflow.connect([
                (inputnode, ds_t2star_t1, [('source_file', 'source_file'),
                                           ('t2star_t1', 'in_file')]),
                (raw_sources, ds_t2star_t1, [('out', 'RawSources')]),
            ])
            # fmt:on

    if getattr(spaces, '_cached') is None:
        return workflow

    # Store resamplings in standard spaces when listed in --output-spaces
    if spaces.cached.references:
        from niworkflows.interfaces.space import SpaceDataSource

        spacesource = pe.Node(SpaceDataSource(), name='spacesource', run_without_submitting=True)
        spacesource.iterables = (
            'in_tuple',
            [(s.fullname, s.spec) for s in spaces.cached.get_standard(dim=(3,))],
        )

        fields = ['template', 'bold_std', 'bold_std_ref', 'bold_mask_std']
        if multiecho:
            fields.append('t2star_std')
        select_std = pe.Node(
            KeySelect(fields=fields),
            name='select_std',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

        ds_bold_std = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='preproc',
                compress=True,
                SkullStripped=masked,
                TaskName=metadata.get('TaskName'),
                **timing_parameters,
            ),
            name='ds_bold_std',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_bold_std_ref = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                suffix='boldref',
                compress=True,
                dismiss_entities=("echo",),
            ),
            name='ds_bold_std_ref',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_bold_mask_std = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='brain',
                suffix='mask',
                compress=True,
                dismiss_entities=("echo",),
            ),
            name='ds_bold_mask_std',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        # fmt:off
        workflow.connect([
            (inputnode, ds_bold_std, [('source_file', 'source_file')]),
            (inputnode, ds_bold_std_ref, [('source_file', 'source_file')]),
            (inputnode, ds_bold_mask_std, [('source_file', 'source_file')]),
            (inputnode, select_std, [('bold_std', 'bold_std'),
                                     ('bold_std_ref', 'bold_std_ref'),
                                     ('bold_mask_std', 'bold_mask_std'),
                                     ('t2star_std', 't2star_std'),
                                     ('template', 'template'),
                                     ('spatial_reference', 'keys')]),
            (spacesource, select_std, [('uid', 'key')]),
            (select_std, ds_bold_std, [('bold_std', 'in_file')]),
            (spacesource, ds_bold_std, [('space', 'space'),
                                        ('cohort', 'cohort'),
                                        ('resolution', 'resolution'),
                                        ('density', 'density')]),
            (select_std, ds_bold_std_ref, [('bold_std_ref', 'in_file')]),
            (spacesource, ds_bold_std_ref, [('space', 'space'),
                                            ('cohort', 'cohort'),
                                            ('resolution', 'resolution'),
                                            ('density', 'density')]),
            (select_std, ds_bold_mask_std, [('bold_mask_std', 'in_file')]),
            (spacesource, ds_bold_mask_std, [('space', 'space'),
                                             ('cohort', 'cohort'),
                                             ('resolution', 'resolution'),
                                             ('density', 'density')]),
            (raw_sources, ds_bold_mask_std, [('out', 'RawSources')]),
        ])
        # fmt:on
        if freesurfer:
            select_fs_std = pe.Node(
                KeySelect(fields=['bold_aseg_std', 'bold_aparc_std', 'template']),
                name='select_fs_std',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB,
            )
            ds_bold_aseg_std = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    desc='aseg',
                    suffix='dseg',
                    compress=True,
                    dismiss_entities=("echo",),
                ),
                name='ds_bold_aseg_std',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB,
            )
            ds_bold_aparc_std = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    desc='aparcaseg',
                    suffix='dseg',
                    compress=True,
                    dismiss_entities=("echo",),
                ),
                name='ds_bold_aparc_std',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB,
            )
            # fmt:off
            workflow.connect([
                (spacesource, select_fs_std, [('uid', 'key')]),
                (inputnode, select_fs_std, [('bold_aseg_std', 'bold_aseg_std'),
                                            ('bold_aparc_std', 'bold_aparc_std'),
                                            ('template', 'template'),
                                            ('spatial_reference', 'keys')]),
                (select_fs_std, ds_bold_aseg_std, [('bold_aseg_std', 'in_file')]),
                (spacesource, ds_bold_aseg_std, [('space', 'space'),
                                                 ('cohort', 'cohort'),
                                                 ('resolution', 'resolution'),
                                                 ('density', 'density')]),
                (select_fs_std, ds_bold_aparc_std, [('bold_aparc_std', 'in_file')]),
                (spacesource, ds_bold_aparc_std, [('space', 'space'),
                                                  ('cohort', 'cohort'),
                                                  ('resolution', 'resolution'),
                                                  ('density', 'density')]),
                (inputnode, ds_bold_aseg_std, [('source_file', 'source_file')]),
                (inputnode, ds_bold_aparc_std, [('source_file', 'source_file')])
            ])
            # fmt:on
        if multiecho:
            ds_t2star_std = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    suffix='T2starmap',
                    compress=True,
                    dismiss_entities=("echo",),
                    **t2star_meta,
                ),
                name='ds_t2star_std',
                run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB,
            )
            # fmt:off
            workflow.connect([
                (inputnode, ds_t2star_std, [('source_file', 'source_file')]),
                (select_std, ds_t2star_std, [('t2star_std', 'in_file')]),
                (spacesource, ds_t2star_std, [('space', 'space'),
                                              ('cohort', 'cohort'),
                                              ('resolution', 'resolution'),
                                              ('density', 'density')]),
                (raw_sources, ds_t2star_std, [('out', 'RawSources')]),
            ])
            # fmt:on

    fs_outputs = spaces.cached.get_fs_spaces()
    if freesurfer and fs_outputs:
        from niworkflows.interfaces.surf import Path2BIDS

        select_fs_surf = pe.Node(
            KeySelect(fields=['surfaces', 'surf_kwargs']),
            name='select_fs_surf',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        select_fs_surf.iterables = [('key', fs_outputs)]
        select_fs_surf.inputs.surf_kwargs = [{'space': s} for s in fs_outputs]

        name_surfs = pe.MapNode(
            Path2BIDS(pattern=r'(?P<hemi>[lr])h.\w+'),
            iterfield='in_file',
            name='name_surfs',
            run_without_submitting=True,
        )

        ds_bold_surfs = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                extension=".func.gii",
                TaskName=metadata.get('TaskName'),
                **timing_parameters,
            ),
            iterfield=['in_file', 'hemi'],
            name='ds_bold_surfs',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        # fmt:off
        workflow.connect([
            (inputnode, select_fs_surf, [
                ('surf_files', 'surfaces'),
                ('surf_refs', 'keys')]),
            (select_fs_surf, name_surfs, [('surfaces', 'in_file')]),
            (inputnode, ds_bold_surfs, [('source_file', 'source_file')]),
            (select_fs_surf, ds_bold_surfs, [('surfaces', 'in_file'),
                                             ('key', 'space')]),
            (name_surfs, ds_bold_surfs, [('hemi', 'hemi')]),
        ])
        # fmt:on

    if freesurfer and project_goodvoxels:
        ds_goodvoxels_mask = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='T1w',
                desc='goodvoxels',
                suffix='mask',
                Type='ROI',  # Metadata
                compress=True,
                dismiss_entities=("echo",),
            ),
            name='ds_goodvoxels_mask',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        # fmt:off
        workflow.connect([
            (inputnode, ds_goodvoxels_mask, [
                ('source_file', 'source_file'),
                ('goodvoxels_mask', 'in_file'),
            ]),
        ])
        # fmt:on

    # CIFTI output
    if cifti_output:
        ds_bold_cifti = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                suffix='bold',
                compress=False,
                TaskName=metadata.get('TaskName'),
                space='fsLR',
                **timing_parameters,
            ),
            name='ds_bold_cifti',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        # fmt:off
        workflow.connect([
            (inputnode, ds_bold_cifti, [(('bold_cifti', _unlist), 'in_file'),
                                        ('source_file', 'source_file'),
                                        ('cifti_density', 'density'),
                                        (('cifti_metadata', _read_json), 'meta_dict')])
        ])
        # fmt:on

    if "compcor" in config.execution.debug:
        ds_acompcor_masks = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc=[f"CompCor{_}" for _ in "CWA"],
                suffix="mask",
                compress=True,
            ),
            name="ds_acompcor_masks",
            run_without_submitting=True,
        )
        ds_tcompcor_mask = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, desc="CompCorT", suffix="mask", compress=True
            ),
            name="ds_tcompcor_mask",
            run_without_submitting=True,
        )
        # fmt:off
        workflow.connect([
            (inputnode, ds_acompcor_masks, [("acompcor_masks", "in_file"),
                                            ("source_file", "source_file")]),
            (inputnode, ds_tcompcor_mask, [("tcompcor_mask", "in_file"),
                                           ("source_file", "source_file")]),
        ])
        # fmt:on

    return workflow


def init_bold_preproc_report_wf(
    mem_gb: float,
    reportlets_dir: str,
    name: str = 'bold_preproc_report_wf',
):
    """
    Generate a visual report.

    This workflow generates and saves a reportlet showing the effect of resampling
    the BOLD signal using the standard deviation maps.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.resampling import init_bold_preproc_report_wf
            wf = init_bold_preproc_report_wf(mem_gb=1, reportlets_dir='.')

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    reportlets_dir : :obj:`str`
        Directory in which to save reportlets
    name : :obj:`str`, optional
        Workflow name (default: bold_preproc_report_wf)

    Inputs
    ------
    in_pre
        BOLD time-series, before resampling
    in_post
        BOLD time-series, after resampling
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing

    """
    from nipype.algorithms.confounds import TSNR
    from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from ...interfaces import DerivativesDataSink

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_pre', 'in_post', 'name_source']), name='inputnode'
    )

    pre_tsnr = pe.Node(TSNR(), name='pre_tsnr', mem_gb=mem_gb * 4.5)
    pos_tsnr = pe.Node(TSNR(), name='pos_tsnr', mem_gb=mem_gb * 4.5)

    bold_rpt = pe.Node(SimpleBeforeAfterRPT(), name='bold_rpt', mem_gb=0.1)
    ds_report_bold = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='preproc',
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name='ds_report_bold',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )
    # fmt:off
    workflow.connect([
        (inputnode, ds_report_bold, [('name_source', 'source_file')]),
        (inputnode, pre_tsnr, [('in_pre', 'in_file')]),
        (inputnode, pos_tsnr, [('in_post', 'in_file')]),
        (pre_tsnr, bold_rpt, [('stddev_file', 'before')]),
        (pos_tsnr, bold_rpt, [('stddev_file', 'after')]),
        (bold_rpt, ds_report_bold, [('out_report', 'in_file')]),
    ])
    # fmt:on

    return workflow


def _unlist(in_file):
    while isinstance(in_file, (list, tuple)) and len(in_file) == 1:
        in_file = in_file[0]
    return in_file


def _read_json(in_file):
    from json import loads
    from pathlib import Path

    return loads(Path(in_file).read_text())
