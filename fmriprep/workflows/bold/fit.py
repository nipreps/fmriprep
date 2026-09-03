# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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
import logging
import os
import re
import typing as ty

import bids
import nibabel as nb
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf, init_skullstrip_bold_wf
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from sdcflows.workflows.apply.registration import init_coeff2epi_wf

from ... import config
from ...interfaces.resampling import (
    DistortionParameters,
    ReconstructFieldmap,
    ResampleSeries,
)
from ...utils.bids import extract_entities
from ...utils.misc import estimate_bold_mem_usage

# BOLD workflows
from .hmc import init_bold_hmc_wf
from .outputs import (
    init_ds_boldmask_wf,
    init_ds_boldref_wf,
    init_ds_hmc_wf,
    init_ds_registration_wf,
)
from .reference import init_raw_boldref_wf, init_validation_and_dummies_wf
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf


def get_sbrefs(
    bold_files: list[str],
    entity_overrides: dict[str, ty.Any],
    layout: bids.BIDSLayout,
) -> list[str]:
    """Find single-band reference(s) associated with BOLD file(s)

    Parameters
    ----------
    bold_files
        List of absolute paths to BOLD files
    entity_overrides
        Query parameters to override defaults
    layout
        :class:`~bids.layout.BIDSLayout` to query

    Returns
    -------
    sbref_files
        List of absolute paths to sbref files associated with input BOLD files,
        sorted by EchoTime
    """
    entities = extract_entities(bold_files)
    entities.pop('echo', None)
    entities.update(suffix='sbref', extension=['.nii', '.nii.gz'])
    entities.update(entity_overrides)

    sbref_files = layout.get(return_type='file', **entities)
    if len(sbref_files) == 1:
        return sbref_files

    valid_sbref_files = []

    for fname in sbref_files:
        if (echo_time := layout.get_metadata(fname).get('EchoTime')) is None:
            logging.getLogger('nipype.workflow').warning(
                'Dropping SBRef without EchoTime metadata: %s', fname
            )
            continue
        valid_sbref_files.append((echo_time, fname))

    return [fname for _, fname in sorted(valid_sbref_files)]


def init_bold_fit_wf(
    *,
    bold_series: list[str],
    precomputed: dict | None = None,
    fieldmap_id: str | None = None,
    jacobian: bool = False,
    omp_nthreads: int = 1,
    name: str = 'bold_fit_wf',
) -> pe.Workflow:
    """
    This workflow controls the minimal estimation steps for functional preprocessing.
    Note that this workflow does not perform BOLD-to-anatomical coregistration;
    that is handled by :func:`~fmriprep.workflows.bold.coreg.init_bold_anat_coreg_wf`.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.bold.fit import init_bold_fit_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_bold_fit_wf(bold_series=[str(bold_file)])

    Parameters
    ----------
    bold_series
        List of paths to NIfTI files, sorted by echo time.
    precomputed
        Dictionary containing precomputed derivatives to reuse, if possible.
    fieldmap_id
        ID of the fieldmap to use to correct this BOLD series. If :obj:`None`,
        no correction will be applied.

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    fmap_ref
        Fieldmap reference file
    fmap_coeff
        List of spline coefficient files
    fmap_mask
        Fieldmap mask

    Outputs
    -------
    hmc_boldref
        BOLD reference image used for head motion correction.
        Minimally processed to ensure consistent contrast with BOLD series.
    run_boldref
        BOLD reference image used for coregistration. Contrast-enhanced
        and fieldmap-corrected for greater anatomical fidelity, and aligned
        with ``hmc_boldref``.
    run_mask
        Mask of ``run_boldref``.
    motion_xfm
        Affine transforms from each BOLD volume to ``hmc_boldref``, written
        as concatenated ITK affine transforms.
    run2fmap_xfm
        Affine transform mapping from BOLD reference space to the fieldmap
        space, if applicable.
    dummy_scans
        The number of dummy scans declared or detected at the beginning of the series.
    algo_dummy_scans
        The number of dummy scans detected by the algorithm.
    validation_report
        HTML snippet reporting BOLD header validation.
    sdc_boldref
        BOLD reference used as the SDC target (if a fieldmap is applied).
    fieldmap
        Fieldmap reconstructed in BOLD reference space (if a fieldmap is applied).

    See Also
    --------

    * :py:func:`~fmriprep.workflows.bold.reference.init_raw_boldref_wf`
    * :py:func:`~fmriprep.workflows.bold.hmc.init_bold_hmc_wf`
    * :py:func:`~niworkflows.func.utils.init_enhance_and_skullstrip_bold_wf`
    * :py:func:`~sdcflows.workflows.apply.registration.init_coeff2epi_wf`
    * :py:func:`~fmriprep.workflows.bold.outputs.init_ds_boldref_wf`
    * :py:func:`~fmriprep.workflows.bold.outputs.init_ds_hmc_wf`
    * :py:func:`~fmriprep.workflows.bold.outputs.init_ds_registration_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmriprep.utils.misc import estimate_bold_mem_usage

    if precomputed is None:
        precomputed = {}
    layout = config.execution.layout
    bids_filters = config.execution.get().get('bids_filters', {})

    # Fitting operates on the shortest echo
    # This could become more complicated in the future
    bold_file = bold_series[0]

    # Collect sbref files, sorted by EchoTime
    sbref_files = get_sbrefs(
        bold_series,
        entity_overrides=bids_filters.get('sbref', {}),
        layout=layout,
    )

    basename = os.path.basename(bold_file)
    sbref_msg = f'No single-band-reference found for {basename}.'
    if sbref_files and 'sbref' in config.workflow.ignore:
        sbref_msg = f'Single-band reference file(s) found for {basename} and ignored.'
        sbref_files = []
    elif sbref_files:
        sbref_msg = f'Using single-band reference file(s) {",".join([os.path.basename(sbf) for sbf in sbref_files])}.'
    config.loggers.workflow.info(sbref_msg)

    # Get metadata from BOLD file(s)
    metadata = layout.get_metadata(bold_file)

    _bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

    # Boolean used to update workflow self-descriptions
    multiecho = len(bold_series) > 1

    hmc_boldref = precomputed.get('hmc_boldref')
    run_boldref = precomputed.get('run_boldref')
    # Can contain
    #  1) run2fmap
    #  2) hmc
    transforms = precomputed.get('transforms', {})
    hmc_xforms = transforms.get('hmc')
    run2fmap_xform = transforms.get('run2fmap')

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                # Fieldmap registration
                'fmap_ref',
                'fmap_coeff',
                'fmap_mask',
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.bold_file = bold_series

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dummy_scans',
                'hmc_boldref',
                'run_boldref',
                'run_mask',
                'motion_xfm',
                'run2fmap_xfm',
                # Passed through for reporting at the base-workflow level
                'algo_dummy_scans',
                'validation_report',
                'sdc_boldref',
                'fieldmap',
            ],
        ),
        name='outputnode',
    )

    # If all derivatives exist, inputnode could go unconnected, so add explicitly
    workflow.add_nodes([inputnode])

    hmcref_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'bold_file', 'dummy_scans']),
        name='hmcref_buffer',
    )
    fmapref_buffer = pe.Node(niu.Function(function=_select_ref), name='fmapref_buffer')
    hmc_buffer = pe.Node(niu.IdentityInterface(fields=['hmc_xforms']), name='hmc_buffer')
    fmapreg_buffer = pe.Node(niu.IdentityInterface(fields=['run2fmap_xfm']), name='fmapreg_buffer')
    regref_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'boldmask']), name='regref_buffer'
    )

    if hmc_boldref:
        hmcref_buffer.inputs.boldref = hmc_boldref
        config.loggers.workflow.debug(f'Reusing motion correction reference: {hmc_boldref}')
    if hmc_xforms:
        hmc_buffer.inputs.hmc_xforms = hmc_xforms
        config.loggers.workflow.debug(f'Reusing motion correction transforms: {hmc_xforms}')
    if run2fmap_xform:
        fmapreg_buffer.inputs.run2fmap_xfm = run2fmap_xform
        config.loggers.workflow.debug(f'Reusing BOLD-to-fieldmap transform: {run2fmap_xform}')
    if run_boldref:
        regref_buffer.inputs.boldref = run_boldref
        config.loggers.workflow.debug(f'Reusing coregistration reference: {run_boldref}')
    fmapref_buffer.inputs.sbref_files = sbref_files

    workflow.connect([
        (hmcref_buffer, outputnode, [
            ('boldref', 'hmc_boldref'),
            ('dummy_scans', 'dummy_scans'),
        ]),
        (hmcref_buffer, fmapref_buffer, [('boldref', 'boldref_files')]),
        (regref_buffer, outputnode, [
            ('boldref', 'run_boldref'),
            ('boldmask', 'run_mask'),
        ]),
        (fmapreg_buffer, outputnode, [('run2fmap_xfm', 'run2fmap_xfm')]),
        (hmc_buffer, outputnode, [
            ('hmc_xforms', 'motion_xfm'),
        ]),
    ])  # fmt:skip

    # Stage 1: Generate motion correction boldref
    hmc_boldref_source_buffer = pe.Node(
        niu.IdentityInterface(fields=['in_file']),
        name='hmc_boldref_source_buffer',
    )
    if not hmc_boldref:
        config.loggers.workflow.info('Stage 1: Adding HMC boldref workflow')
        hmc_boldref_wf = init_raw_boldref_wf(
            name='hmc_boldref_wf',
            bold_file=bold_file,
            multiecho=multiecho,
        )
        hmc_boldref_wf.inputs.inputnode.dummy_scans = config.workflow.dummy_scans

        ds_hmc_boldref_wf = init_ds_boldref_wf(
            source_file=bold_file,
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            desc='hmc',
            space='orig',
            name='ds_hmc_boldref_wf',
        )
        ds_hmc_boldref_wf.inputs.inputnode.source_files = [bold_file]

        workflow.connect([
            (hmc_boldref_wf, ds_hmc_boldref_wf, [('outputnode.boldref', 'inputnode.boldref')]),
            (ds_hmc_boldref_wf, hmcref_buffer, [('outputnode.boldref', 'boldref')]),
            (hmc_boldref_wf, hmcref_buffer, [
                ('outputnode.bold_file', 'bold_file'),
                ('outputnode.skip_vols', 'dummy_scans'),
            ]),
            (hmc_boldref_wf, outputnode, [
                ('outputnode.algo_dummy_scans', 'algo_dummy_scans'),
                ('outputnode.validation_report', 'validation_report'),
            ]),
            (ds_hmc_boldref_wf, hmc_boldref_source_buffer, [
                ('outputnode.boldref', 'in_file'),
            ]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found HMC boldref - skipping Stage 1')

        validation_and_dummies_wf = init_validation_and_dummies_wf(bold_file=bold_file)

        workflow.connect([
            (validation_and_dummies_wf, hmcref_buffer, [
                ('outputnode.bold_file', 'bold_file'),
                ('outputnode.skip_vols', 'dummy_scans'),
            ]),
            (validation_and_dummies_wf, outputnode, [
                ('outputnode.validation_report', 'validation_report'),
            ]),
            (hmcref_buffer, hmc_boldref_source_buffer, [('boldref', 'in_file')]),
        ])  # fmt:skip

    # Stage 2: Estimate head motion
    if not hmc_xforms:
        config.loggers.workflow.info('Stage 2: Adding motion correction workflow')
        bold_hmc_wf = init_bold_hmc_wf(
            name='bold_hmc_wf', mem_gb=mem_gb['filesize'], omp_nthreads=omp_nthreads
        )

        ds_hmc_wf = init_ds_hmc_wf(
            source_file=bold_file,
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
        )
        ds_hmc_wf.inputs.inputnode.source_files = [bold_file]

        workflow.connect([
            (hmcref_buffer, bold_hmc_wf, [
                ('boldref', 'inputnode.raw_ref_image'),
                ('bold_file', 'inputnode.bold_file'),
            ]),
            (bold_hmc_wf, ds_hmc_wf, [('outputnode.xforms', 'inputnode.xforms')]),
            (ds_hmc_wf, hmc_buffer, [('outputnode.xforms', 'hmc_xforms')]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found motion correction transforms - skipping Stage 2')

    # Stage 3: Register fieldmap to boldref and reconstruct in BOLD space
    if fieldmap_id:
        config.loggers.workflow.info('Stage 3: Adding fieldmap reconstruction workflow')

        boldref_fmap = pe.Node(ReconstructFieldmap(inverse=[True]), name='boldref_fmap', mem_gb=1)

        workflow.connect([
            (inputnode, boldref_fmap, [
                ('fmap_coeff', 'in_coeffs'),
                ('fmap_ref', 'fmap_ref_file'),
            ]),
            (fmapref_buffer, boldref_fmap, [('out', 'target_ref_file')]),
            (fmapreg_buffer, boldref_fmap, [('run2fmap_xfm', 'transforms')]),
            (fmapref_buffer, outputnode, [('out', 'sdc_boldref')]),
            (boldref_fmap, outputnode, [('out_file', 'fieldmap')]),
        ])  # fmt:skip

        if not run2fmap_xform:
            config.loggers.workflow.info('Stage 3: Registering fieldmap to boldref')
            fmapreg_wf = init_coeff2epi_wf(
                debug='fieldmaps' in config.execution.debug,
                omp_nthreads=config.nipype.omp_nthreads,
                sloppy=config.execution.sloppy,
                name='fmapreg_wf',
            )

            itk_mat2txt = pe.Node(ConcatenateXFMs(out_fmt='itk'), name='itk_mat2txt')
            fmapreg_source_files = pe.Node(
                niu.Merge(2), name='fmapreg_source_files', run_without_submitting=True
            )

            ds_fmapreg_wf = init_ds_registration_wf(
                source_file=bold_file,
                bids_root=layout.root,
                output_dir=config.execution.fmriprep_dir,
                source='run',
                dest=re.sub(r'[^a-zA-Z0-9]', '', fieldmap_id),
                desc='fmap',
                name='ds_fmapreg_wf',
            )

            workflow.connect([
                (inputnode, fmapreg_wf, [
                    ('fmap_ref', 'inputnode.fmap_ref'),
                    ('fmap_mask', 'inputnode.fmap_mask'),
                ]),
                (inputnode, fmapreg_source_files, [('fmap_ref', 'in2')]),
                (fmapref_buffer, fmapreg_source_files, [('out', 'in1')]),
                (fmapreg_wf, itk_mat2txt, [('outputnode.target2fmap_xfm', 'in_xfms')]),
                (itk_mat2txt, ds_fmapreg_wf, [('out_xfm', 'inputnode.xform')]),
                (fmapreg_source_files, ds_fmapreg_wf, [('out', 'inputnode.source_files')]),
                (ds_fmapreg_wf, fmapreg_buffer, [('outputnode.xform', 'run2fmap_xfm')]),
            ])  # fmt:skip
        else:
            config.loggers.workflow.info(
                'Stage 3: Found fieldmap transform - skipping registration'
            )
    else:
        config.loggers.workflow.info('No fieldmap correction - skipping Stage 3')

    # Stage 4: Create coregistration reference
    if not run_boldref:
        config.loggers.workflow.info('Stage 4: Adding coregistration boldref workflow')

        # If sbref files are available, add them to the list of sources
        if sbref_files and nb.load(sbref_files[0]).ndim > 3:
            raw_sbref_wf = init_raw_boldref_wf(
                name='raw_sbref_wf',
                bold_file=sbref_files[0],
                multiecho=len(sbref_files) > 1,
            )
            workflow.connect(raw_sbref_wf, 'outputnode.boldref', fmapref_buffer, 'sbref_files')

        enhance_boldref_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)
        coreg_ref_source_files = pe.Node(
            niu.Merge(3), name='coreg_ref_source_files', run_without_submitting=True
        )

        ds_run_boldref_wf = init_ds_boldref_wf(
            source_file=bold_file,
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            space='run',
            name='ds_run_boldref_wf',
        )
        ds_boldmask_wf = init_ds_boldmask_wf(
            source_file=bold_file,
            output_dir=config.execution.fmriprep_dir,
            desc='brain',
            space='run',
            name='ds_boldmask_wf',
        )

        workflow.connect([
            (fmapref_buffer, enhance_boldref_wf, [('out', 'inputnode.in_file')]),
            (fmapref_buffer, coreg_ref_source_files, [('out', 'in1')]),
            (coreg_ref_source_files, ds_run_boldref_wf, [('out', 'inputnode.source_files')]),
            (ds_run_boldref_wf, regref_buffer, [('outputnode.boldref', 'boldref')]),
            (ds_run_boldref_wf, ds_boldmask_wf, [('outputnode.boldref', 'inputnode.source_files')]),
            (ds_boldmask_wf, regref_buffer, [('outputnode.boldmask', 'boldmask')]),
        ])  # fmt:skip

        if fieldmap_id:
            distortion_params = pe.Node(
                DistortionParameters(
                    metadata=metadata,
                    in_file=bold_file,
                    fallback=config.workflow.fallback_total_readout_time,
                ),
                name='distortion_params',
                run_without_submitting=True,
            )

            unwarp_boldref = pe.Node(
                ResampleSeries(jacobian=jacobian),
                name='unwarp_boldref',
                n_procs=omp_nthreads,
                mem_gb=mem_gb['resampled'],
            )

            skullstrip_bold_wf = init_skullstrip_bold_wf()

            workflow.connect([
                (fmapref_buffer, unwarp_boldref, [('out', 'ref_file')]),
                (enhance_boldref_wf, unwarp_boldref, [
                    ('outputnode.bias_corrected_file', 'in_file'),
                ]),
                (boldref_fmap, unwarp_boldref, [('out_file', 'fieldmap')]),
                (distortion_params, unwarp_boldref, [
                    ('readout_time', 'ro_time'),
                    ('pe_direction', 'pe_dir'),
                ]),
                (unwarp_boldref, ds_run_boldref_wf, [
                    ('out_file', 'inputnode.boldref'),
                ]),
                (ds_run_boldref_wf, skullstrip_bold_wf, [
                    ('outputnode.boldref', 'inputnode.in_file'),
                ]),
                (skullstrip_bold_wf, ds_boldmask_wf, [
                    ('outputnode.mask_file', 'inputnode.boldmask'),
                ]),
                (fmapreg_buffer, coreg_ref_source_files, [('run2fmap_xfm', 'in2')]),
                (inputnode, coreg_ref_source_files, [('fmap_coeff', 'in3')]),
            ])  # fmt:skip

            if not run2fmap_xform:
                workflow.connect([
                    (enhance_boldref_wf, fmapreg_wf, [
                        ('outputnode.bias_corrected_file', 'inputnode.target_ref'),
                        ('outputnode.mask_file', 'inputnode.target_mask'),
                    ]),
                ])  # fmt:skip
        else:
            workflow.connect([
                (enhance_boldref_wf, ds_run_boldref_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.boldref'),
                ]),
                (enhance_boldref_wf, ds_boldmask_wf, [
                    ('outputnode.mask_file', 'inputnode.boldmask'),
                ]),
            ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found coregistration reference - skipping Stage 4')

        # TODO: Allow precomputed bold masks to be passed
        # Also needs consideration for how it interacts above
        skullstrip_precomp_ref_wf = init_skullstrip_bold_wf(name='skullstrip_precomp_ref_wf')
        skullstrip_precomp_ref_wf.inputs.inputnode.in_file = run_boldref
        workflow.connect([
            (skullstrip_precomp_ref_wf, regref_buffer, [('outputnode.mask_file', 'boldmask')])
        ])  # fmt:skip

    return workflow


def init_bold_native_wf(
    *,
    bold_series: list[str],
    fieldmap_id: str | None = None,
    jacobian: bool = False,
    omp_nthreads: int = 1,
    name: str = 'bold_native_wf',
) -> pe.Workflow:
    r"""
    Minimal resampling workflow.

    This workflow performs slice-timing correction, and resamples to boldref space
    with head motion and susceptibility distortion correction. It also handles
    multi-echo processing and selects the transforms needed to perform further
    resampling.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.bold.fit import init_bold_native_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_bold_native_wf(bold_series=[str(bold_file)])

    Parameters
    ----------
    bold_series
        List of paths to NIfTI files.
    fieldmap_id
        ID of the fieldmap to use to correct this BOLD series. If :obj:`None`,
        no correction will be applied.

    Inputs
    ------
    run_boldref
        BOLD reference file
    run_mask
        Mask of BOLD reference file
    motion_xfm
        Affine transforms from each BOLD volume to ``hmc_boldref``, written
        as concatenated ITK affine transforms.
    run2fmap_xfm
        Affine transform mapping from BOLD reference space to the fieldmap
        space, if applicable.
    fmap_ref
        Fieldmap reference file
    fmap_coeff
        List of spline coefficient files

    Outputs
    -------
    bold_minimal
        BOLD series ready for further resampling. For single-echo data, only
        slice-timing correction (STC) may have been applied. For multi-echo
        data, this is identical to bold_native.
    bold_native
        BOLD series resampled into BOLD reference space. Slice-timing,
        head motion and susceptibility distortion correction (STC, HMC, SDC)
        will all be applied to each file. For multi-echo data, the echos
        are combined to form an `optimal combination`_.
    metadata
        Metadata dictionary of BOLD series with the shortest echo
    motion_xfm
        Motion correction transforms for further correcting bold_minimal.
        For multi-echo data, motion correction has already been applied, so
        this will be undefined.
    bold_echos
        The individual, corrected echos, suitable for use in Tedana.
        (Multi-echo only.)
    t2star_map
        The T2\* map estimated by Tedana when calculating the optimal combination.
        (Multi-echo only.)

    See Also
    --------

    * :py:func:`~fmriprep.workflows.bold.stc.init_bold_stc_wf`
    * :py:func:`~fmriprep.workflows.bold.t2s.init_bold_t2s_wf`

    .. _optimal combination: https://tedana.readthedocs.io/en/stable/approach.html#optimal-combination

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    layout = config.execution.layout

    # Shortest echo first
    all_metadata = [layout.get_metadata(bold_file) for bold_file in bold_series]
    echo_times = [md.get('EchoTime') for md in all_metadata]
    multiecho = len(bold_series) > 1

    bold_file = bold_series[0]
    metadata = all_metadata[0]

    _bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

    if multiecho:
        shapes = [nb.load(echo).shape for echo in bold_series]
        if len(set(shapes)) != 1:
            diagnostic = '\n'.join(
                f'{os.path.basename(echo)}: {shape}'
                for echo, shape in zip(bold_series, shapes, strict=False)
            )
            raise RuntimeError(f'Multi-echo images found with mismatching shapes\n{diagnostic}')
        if len(shapes) == 2:
            raise RuntimeError(
                'Multi-echo processing requires at least three different echos (found two).'
            )

    run_stc = bool(metadata.get('SliceTiming')) and 'slicetiming' not in config.workflow.ignore

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # BOLD fit
                'run_boldref',
                'run_mask',
                'motion_xfm',
                'run2fmap_xfm',
                'dummy_scans',
                # Fieldmap fit
                'fmap_ref',
                'fmap_coeff',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_minimal',
                'bold_native',
                'metadata',
                # Transforms
                'motion_xfm',
                # Multiecho outputs
                'bold_echos',  # Individual corrected echos
                't2star_map',  # T2* map
            ],
        ),
        name='outputnode',
    )
    outputnode.inputs.metadata = metadata

    boldbuffer = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'ro_time', 'pe_dir']), name='boldbuffer'
    )

    # Track echo index - this allows us to treat multi- and single-echo workflows
    # almost identically
    echo_index = pe.Node(niu.IdentityInterface(fields=['echoidx']), name='echo_index')
    if multiecho:
        echo_index.iterables = [('echoidx', range(len(bold_series)))]
    else:
        echo_index.inputs.echoidx = 0

    # BOLD source: track original BOLD file(s)
    bold_source = pe.Node(niu.Select(inlist=bold_series), name='bold_source')
    validate_bold = pe.Node(ValidateImage(), name='validate_bold')
    workflow.connect([
        (echo_index, bold_source, [('echoidx', 'index')]),
        (bold_source, validate_bold, [('out', 'in_file')]),
    ])  # fmt:skip

    # Slice-timing correction
    if run_stc:
        bold_stc_wf = init_bold_stc_wf(metadata=metadata, mem_gb=mem_gb)
        workflow.connect([
            (inputnode, bold_stc_wf, [('dummy_scans', 'inputnode.skip_vols')]),
            (validate_bold, bold_stc_wf, [('out_file', 'inputnode.bold_file')]),
            (bold_stc_wf, boldbuffer, [('outputnode.stc_file', 'bold_file')]),
        ])  # fmt:skip
    else:
        workflow.connect([(validate_bold, boldbuffer, [('out_file', 'bold_file')])])

    # Prepare fieldmap metadata
    if fieldmap_id:
        distortion_params = pe.Node(
            DistortionParameters(
                metadata=metadata,
                in_file=bold_file,
                fallback=config.workflow.fallback_total_readout_time,
            ),
            name='distortion_params',
            run_without_submitting=True,
        )
        workflow.connect([
            (distortion_params, boldbuffer, [
                ('readout_time', 'ro_time'),
                ('pe_direction', 'pe_dir'),
            ]),
        ])  # fmt:skip

    # Resample to boldref
    boldref_bold = pe.Node(
        ResampleSeries(jacobian=jacobian),
        name='boldref_bold',
        n_procs=omp_nthreads,
        mem_gb=mem_gb['resampled'],
    )

    workflow.connect([
        (inputnode, boldref_bold, [
            ('run_boldref', 'ref_file'),
            ('motion_xfm', 'transforms'),
        ]),
        (boldbuffer, boldref_bold, [
            ('bold_file', 'in_file'),
            ('ro_time', 'ro_time'),
            ('pe_dir', 'pe_dir'),
        ]),
    ])  # fmt:skip

    if fieldmap_id:
        boldref_fmap = pe.Node(ReconstructFieldmap(inverse=[True]), name='boldref_fmap', mem_gb=1)
        workflow.connect([
            (inputnode, boldref_fmap, [
                ('run_boldref', 'target_ref_file'),
                ('run2fmap_xfm', 'transforms'),
                ('fmap_coeff', 'in_coeffs'),
                ('fmap_ref', 'fmap_ref_file'),
            ]),
            (boldref_fmap, boldref_bold, [('out_file', 'fieldmap')]),
        ])  # fmt:skip

    if multiecho:
        join_echos = pe.JoinNode(
            niu.IdentityInterface(fields=['bold_files']),
            joinsource='echo_index',
            joinfield=['bold_files'],
            name='join_echos',
            run_without_submitting=True,
        )

        # create optimal combination, adaptive T2* map
        bold_t2s_wf = init_bold_t2s_wf(
            echo_times=echo_times,
            mem_gb=mem_gb['filesize'],
            omp_nthreads=config.nipype.omp_nthreads,
            name='bold_t2smap_wf',
        )

        # Do NOT set motion_xfm on outputnode
        # This prevents downstream resamplers from double-dipping
        workflow.connect([
            (inputnode, bold_t2s_wf, [
                ('run_mask', 'inputnode.bold_mask'),
                ('dummy_scans', 'inputnode.skip_vols'),
            ]),
            (boldref_bold, join_echos, [('out_file', 'bold_files')]),
            (join_echos, bold_t2s_wf, [('bold_files', 'inputnode.bold_file')]),
            (join_echos, outputnode, [('bold_files', 'bold_echos')]),
            (bold_t2s_wf, outputnode, [
                ('outputnode.bold', 'bold_minimal'),
                ('outputnode.bold', 'bold_native'),
                ('outputnode.t2star_map', 't2star_map'),
            ]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (inputnode, outputnode, [('motion_xfm', 'motion_xfm')]),
            (boldbuffer, outputnode, [('bold_file', 'bold_minimal')]),
            (boldref_bold, outputnode, [('out_file', 'bold_native')]),
        ])  # fmt:skip

    return workflow


def _select_ref(sbref_files, boldref_files):
    """Select first sbref or boldref file, preferring sbref if available"""
    from niworkflows.utils.connections import listify

    refs = sbref_files or boldref_files
    return listify(refs)[0]
