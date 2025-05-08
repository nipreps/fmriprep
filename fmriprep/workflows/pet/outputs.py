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
"""Writing out derivative files."""

from __future__ import annotations

import numpy as np
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from niworkflows.utils.images import dseg_label

from fmriprep import config
from fmriprep.config import DEFAULT_MEMORY_MIN_GB
from fmriprep.interfaces import DerivativesDataSink
from fmriprep.interfaces.bids import BIDSURI


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
            'RepetitionTime',
            'VolumeTiming',
            'DelayTime',
            'AcquisitionDuration',
            'SliceTiming',
        )
        if key in metadata
    }

    # Treat SliceTiming of [] or length 1 as equivalent to missing and remove it in any case
    slice_timing = timing_parameters.pop('SliceTiming', [])

    run_stc = len(slice_timing) > 1 and 'slicetiming' not in config.workflow.ignore
    timing_parameters['SliceTimingCorrected'] = run_stc

    if len(slice_timing) > 1:
        st = sorted(slice_timing)
        TA = st[-1] + (st[1] - st[0])  # Final slice onset + slice duration
        # For constant TR paradigms, use DelayTime
        if 'RepetitionTime' in timing_parameters:
            TR = timing_parameters['RepetitionTime']
            if not np.isclose(TR, TA) and TA < TR:
                timing_parameters['DelayTime'] = TR - TA
        # For variable TR paradigms, use AcquisitionDuration
        elif 'VolumeTiming' in timing_parameters:
            timing_parameters['AcquisitionDuration'] = TA

        if run_stc:
            first, last = st[0], st[-1]
            frac = config.workflow.slice_time_ref
            tzero = np.round(first + frac * (last - first), 3)
            timing_parameters['StartTime'] = tzero

    return timing_parameters


def init_func_fit_reports_wf(
    *,
    freesurfer: bool,
    output_dir: str,
    name='func_fit_reports_wf',
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
        Input PET images

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
    from nireports.interfaces.reporting.base import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )

    workflow = pe.Workflow(name=name)

    inputfields = [
        'source_file',
        'petref',
        'pet_mask',
        'petref2anat_xfm',
        't1w_preproc',
        't1w_mask',
        't1w_dseg',
        # May be missing
        'subject_id',
        'subjects_dir',
        # Report snippets
        'summary_report',
        'validation_report',
    ]
    inputnode = pe.Node(niu.IdentityInterface(fields=inputfields), name='inputnode')

    ds_summary = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='summary',
            datatype='figures',
        ),
        name='ds_report_summary',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    ds_validation = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='validation',
            datatype='figures',
        ),
        name='ds_report_validation',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    # Resample anatomical references into PET space for plotting
    t1w_petref = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            float=True,
            invert_transform_flags=[True],
            interpolation='LanczosWindowedSinc',
        ),
        name='t1w_petref',
        mem_gb=1,
    )

    t1w_wm = pe.Node(
        niu.Function(function=dseg_label),
        name='t1w_wm',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    t1w_wm.inputs.label = 2  # BIDS default is WM=2

    petref_wm = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            invert_transform_flags=[True],
            interpolation='NearestNeighbor',
        ),
        name='petref_wm',
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
        (inputnode, t1w_petref, [
            ('t1w_preproc', 'input_image'),
            ('petref', 'reference_image'),
            ('petref2anat_xfm', 'transforms'),
        ]),
        (inputnode, t1w_wm, [('t1w_dseg', 'in_seg')]),
        (inputnode, petref_wm, [
            ('petref', 'reference_image'),
            ('petref2anat_xfm', 'transforms'),
        ]),
        (t1w_wm, petref_wm, [('out', 'input_image')]),
    ])
    # fmt:on

    # EPI-T1 registration
    # Resample T1w image onto EPI-space

    pet_t1_report = pe.Node(
        SimpleBeforeAfter(
            before_label='T1w',
            after_label='EPI',
            dismiss_affine=True,
        ),
        name='pet_t1_report',
        mem_gb=0.1,
    )

    ds_pet_t1_report = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix='pet',
            datatype='figures',
        ),
        name='ds_pet_t1_report',
    )

    # fmt:off
    workflow.connect([
        (inputnode, pet_t1_report, [('petref', 'after')]),
        (t1w_petref, pet_t1_report, [('output_image', 'before')]),
        (petref_wm, pet_t1_report, [('output_image', 'wm_seg')]),
        (inputnode, ds_pet_t1_report, [('source_file', 'source_file')]),
        (pet_t1_report, ds_pet_t1_report, [('out_report', 'in_file')]),
    ])
    # fmt:on

    return workflow


def init_ds_petref_wf(
    *,
    bids_root,
    output_dir,
    desc: str,
    name='ds_petref_wf',
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'petref']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['petref']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_petref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc=desc,
            suffix='petref',
            compress=True,
        ),
        name='ds_petref',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, sources, [('source_files', 'in1')]),
        (inputnode, ds_petref, [('petref', 'in_file'),
                                 ('source_files', 'source_file')]),
        (sources, ds_petref, [('out', 'Sources')]),
        (ds_petref, outputnode, [('out_file', 'petref')]),
    ])
    # fmt:on

    return workflow


def init_ds_petmask_wf(
    *,
    output_dir,
    desc: str,
    name='ds_petmask_wf',
) -> pe.Workflow:
    """Write out a PET mask."""
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'petmask']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['petmask']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_petmask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc=desc,
            suffix='mask',
            compress=True,
        ),
        name='ds_petmask',
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, sources, [('source_files', 'in1')]),
        (inputnode, ds_petmask, [
            ('petmask', 'in_file'),
            ('source_files', 'source_file'),
        ]),
        (sources, ds_petmask, [('out', 'Sources')]),
        (ds_petmask, outputnode, [('out_file', 'petmask')]),
    ])  # fmt:skip

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
        niu.IdentityInterface(fields=['source_files', 'xform']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['xform']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_xform = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            mode='image',
            suffix='xfm',
            extension='.txt',
            **{'from': source, 'to': dest},
        ),
        name='ds_xform',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # fmt:off
    workflow.connect([
        (inputnode, sources, [('source_files', 'in1')]),
        (inputnode, ds_xform, [('xform', 'in_file'),
                               ('source_files', 'source_file')]),
        (sources, ds_xform, [('out', 'Sources')]),
        (ds_xform, outputnode, [('out_file', 'xform')]),
    ])
    # fmt:on

    return workflow


def init_ds_hmc_wf(
    *,
    bids_root,
    output_dir,
    name='ds_hmc_wf',
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'xforms']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['xforms']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_xforms = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='hmc',
            suffix='xfm',
            extension='.txt',
            compress=True,
            **{'from': 'orig', 'to': 'petref'},
        ),
        name='ds_xforms',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, sources, [('source_files', 'in1')]),
        (inputnode, ds_xforms, [('xforms', 'in_file'),
                                ('source_files', 'source_file')]),
        (sources, ds_xforms, [('out', 'Sources')]),
        (ds_xforms, outputnode, [('out_file', 'xforms')]),
    ])
    # fmt:on

    return workflow


def init_ds_pet_native_wf(
    *,
    bids_root: str,
    output_dir: str,
    pet_output: bool,
    all_metadata: list[dict],
    name='ds_pet_native_wf',
) -> pe.Workflow:
    metadata = all_metadata[0]
    timing_parameters = prepare_timing_parameters(metadata)

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_files',
                'pet',
                # Transforms previously used to generate the outputs
                'motion_xfm',
            ]
        ),
        name='inputnode',
    )

    sources = pe.Node(
        BIDSURI(
            numinputs=3,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )
    workflow.connect([
        (inputnode, sources, [
            ('source_files', 'in1'),
            ('motion_xfm', 'in2'),
        ]),
    ])  # fmt:skip

    if pet_output:
        ds_pet = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='preproc',
                compress=True,
                TaskName=metadata.get('TaskName'),
                **timing_parameters,
            ),
            name='ds_pet',
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_pet, [
                ('source_files', 'source_file'),
                ('pet', 'in_file'),
            ]),
            (sources, ds_pet, [('out', 'Sources')]),
        ])  # fmt:skip

    return workflow


def init_ds_volumes_wf(
    *,
    bids_root: str,
    output_dir: str,
    metadata: list[dict],
    name='ds_volumes_wf',
) -> pe.Workflow:
    timing_parameters = prepare_timing_parameters(metadata)

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_files',
                'ref_file',
                'pet',  # Resampled into target space
                'pet_mask',  # petref space
                'pet_ref',  # petref space
                't2star',  # petref space
                'template',  # target reference image from original transform
                # Anatomical
                'petref2anat_xfm',
                # Template
                'anat2std_xfm',
                # Entities
                'space',
                'cohort',
                'resolution',
                # Transforms previously used to generate the outputs
                'motion_xfm',
            ]
        ),
        name='inputnode',
    )

    sources = pe.Node(
        BIDSURI(
            numinputs=6,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )
    petref2target = pe.Node(niu.Merge(2), name='petref2target')

    # PET is pre-resampled
    ds_pet = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='preproc',
            compress=True,
            TaskName=metadata.get('TaskName'),
            **timing_parameters,
        ),
        name='ds_pet',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([
        (inputnode, sources, [
            ('source_files', 'in1'),
            ('motion_xfm', 'in2'),
            ('petref2anat_xfm', 'in4'),
            ('anat2std_xfm', 'in5'),
            ('template', 'in6'),
        ]),
        (inputnode, petref2target, [
            # Note that ANTs expects transforms in target-to-source order
            # Reverse this for nitransforms-based resamplers
            ('anat2std_xfm', 'in1'),
            ('petref2anat_xfm', 'in2'),
        ]),
        (inputnode, ds_pet, [
            ('source_files', 'source_file'),
            ('pet', 'in_file'),
            ('space', 'space'),
            ('cohort', 'cohort'),
            ('resolution', 'resolution'),
        ]),
        (sources, ds_pet, [('out', 'Sources')]),
    ])  # fmt:skip

    resample_ref = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            float=True,
            interpolation='LanczosWindowedSinc',
        ),
        name='resample_ref',
    )
    resample_mask = pe.Node(ApplyTransforms(interpolation='MultiLabel'), name='resample_mask')
    resamplers = [resample_ref, resample_mask]

    workflow.connect([
        (inputnode, resample_ref, [('pet_ref', 'input_image')]),
        (inputnode, resample_mask, [('pet_mask', 'input_image')]),
    ])  # fmt:skip

    ds_ref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix='petref',
            compress=True,
        ),
        name='ds_ref',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='brain',
            suffix='mask',
            compress=True,
        ),
        name='ds_mask',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    datasinks = [ds_ref, ds_mask]

    workflow.connect(
        [
            (inputnode, resampler, [('ref_file', 'reference_image')])
            for resampler in resamplers
        ] + [
            (petref2target, resampler, [('out', 'transforms')])
            for resampler in resamplers
        ] + [
            (inputnode, datasink, [
                ('source_files', 'source_file'),
                ('space', 'space'),
                ('cohort', 'cohort'),
                ('resolution', 'resolution'),
            ])
            for datasink in datasinks
        ] + [
            (sources, datasink, [('out', 'Sources')])
            for datasink in datasinks
        ] + [
            (resampler, datasink, [('output_image', 'in_file')])
            for resampler, datasink in zip(resamplers, datasinks, strict=False)
        ]
    )  # fmt:skip

    return workflow


def init_pet_preproc_report_wf(
    mem_gb: float,
    reportlets_dir: str,
    name: str = 'pet_preproc_report_wf',
):
    """
    Generate a visual report.

    This workflow generates and saves a reportlet showing the effect of resampling
    the PET signal using the standard deviation maps.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.pet.resampling import init_pet_preproc_report_wf
            wf = init_pet_preproc_report_wf(mem_gb=1, reportlets_dir='.')

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of PET file in GB
    reportlets_dir : :obj:`str`
        Directory in which to save reportlets
    name : :obj:`str`, optional
        Workflow name (default: pet_preproc_report_wf)

    Inputs
    ------
    in_pre
        PET time-series, before resampling
    in_post
        PET time-series, after resampling
    name_source
        PET series NIfTI file
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

    pet_rpt = pe.Node(SimpleBeforeAfterRPT(), name='pet_rpt', mem_gb=0.1)
    ds_report_pet = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='preproc',
            datatype='figures',
        ),
        name='ds_report_pet',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )
    # fmt:off
    workflow.connect([
        (inputnode, ds_report_pet, [('name_source', 'source_file')]),
        (inputnode, pre_tsnr, [('in_pre', 'in_file')]),
        (inputnode, pos_tsnr, [('in_post', 'in_file')]),
        (pre_tsnr, pet_rpt, [('stddev_file', 'before')]),
        (pos_tsnr, pet_rpt, [('stddev_file', 'after')]),
        (pet_rpt, ds_report_pet, [('out_report', 'in_file')]),
    ])
    # fmt:on

    return workflow
