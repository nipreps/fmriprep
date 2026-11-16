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
Thermal-noise denoising of BOLD series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_denoise_wf

"""

import nibabel as nb
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.interfaces.header import ValidateImage

from ... import config
from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink
from ...interfaces.bids import BIDSURI
from ...interfaces.denoise import AbsoluteValue, SVHTDenoise, select_report_volume
from ...utils.denoise import (
    can_svht_denoise,
    check_svht_phase,
    check_svht_volumes,
    svht_denoise_version,
    svht_phase_units,
)

#: How many times the input's size svht_denoise is budgeted: the float32 input and output,
#: plus patch buffers. Not yet measured on long runs.
SVHT_MEM_FACTOR = 4

#: Entities of the magnitude input that do not describe the noise and rank maps
_DISMISS_ENTITIES = ['part']


def init_bold_denoise_wf(
    *,
    bold_series: list[str],
    output_dir: str,
    phase_series: list[str] | None = None,
    denoise_params: dict | None = None,
    mem_gb: float = DEFAULT_MEMORY_MIN_GB,
    omp_nthreads: int = 1,
    name: str = 'bold_denoise_wf',
):
    """
    Denoise each echo of a BOLD series with ``svht_denoise``.

    This is the first step of BOLD preprocessing: everything downstream, from the HMC
    reference to resampling, reads the denoised series. Each echo is denoised separately,
    with its own phase when ``phase_series`` is given. For each echo the workflow writes a
    noise map (``desc-noise_boldmap``), a rank map (``desc-rank_boldmap``), and a reportlet
    (``desc-denoising``) of one volume before and after denoising.

    The denoised series itself is not written as a derivative.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.bold.denoise import init_bold_denoise_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_bold_denoise_wf(
                    bold_series=[str(bold_file)],
                    output_dir=str(config.execution.fmriprep_dir),
                )

    Parameters
    ----------
    bold_series
        The magnitude BOLD file of each echo, in echo order.
    output_dir
        The directory the derivatives are written to.
    phase_series
        The phase file of each echo, in the order of ``bold_series``, as returned by
        :func:`~fmriprep.utils.bids.collect_bold_phase`. ``None`` denoises the magnitude
        alone.
    denoise_params
        SVHTDenoise parameters, as returned by
        :func:`~fmriprep.utils.denoise.load_svht_config`.
    mem_gb
        Size of the BOLD file, in GB.
    omp_nthreads
        Maximum number of threads an individual process may use.
    name
        Name of the workflow (default: ``bold_denoise_wf``).

    Outputs
    -------
    bold_files
        The denoised series of each echo, in the order of ``bold_series``. Each is
        positive: with phase, the absolute value of svht_denoise's signed output.
    noise_files
        The noise map of each echo, as written to ``output_dir``.
    rank_files
        The rank map of each echo, as written to ``output_dir``.
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    params = dict(denoise_params or {})
    use_phase = phase_series is not None
    if use_phase and len(phase_series) != len(bold_series):
        raise ValueError(
            f'Got {len(phase_series)} phase files for a series of {len(bold_series)} echoes.'
        )

    # Every echo is denoised, so every echo must pass; they must also agree, since all are
    # checked against the same limits (and fMRIPrep combines them later)
    shapes = [nb.load(bold_file).shape for bold_file in bold_series]
    if len(set(shapes)) > 1:
        listing = ', '.join(
            f'{bold_file} {shape}' for bold_file, shape in zip(bold_series, shapes, strict=True)
        )
        raise ValueError(f'The echoes of this BOLD series differ in shape: {listing}.')

    if len(shapes[0]) != 4:
        raise ValueError(f'{bold_series[0]} is not a 4D series: {shapes[0]}.')

    n_volumes = shapes[0][3]
    if not can_svht_denoise(n_volumes):
        raise ValueError(
            f'{bold_series[0]} has {n_volumes} volumes, too few for svht_denoise to estimate '
            'the noise. Do not build a denoising workflow for it.'
        )

    check_svht_phase(params, use_phase)
    check_svht_volumes(
        params,
        n_volumes,
        bold_file=bold_series[0],
        spatial_shape=shapes[0][:3],
    )

    workflow = Workflow(name=name)

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_files', 'noise_files', 'rank_files']),
        name='outputnode',
    )

    n_echoes = len(bold_series)
    merge_bold = pe.Node(niu.Merge(n_echoes), name='merge_bold', run_without_submitting=True)
    merge_noise = pe.Node(niu.Merge(n_echoes), name='merge_noise', run_without_submitting=True)
    merge_rank = pe.Node(niu.Merge(n_echoes), name='merge_rank', run_without_submitting=True)
    workflow.connect([
        (merge_bold, outputnode, [('out', 'bold_files')]),
        (merge_noise, outputnode, [('out', 'noise_files')]),
        (merge_rank, outputnode, [('out', 'rank_files')]),
    ])  # fmt:skip

    provenance = {
        'SoftwareName': 'svht_denoise',
        'CodeURL': 'https://github.com/rordenlab/svht_denoise',
        'DenoisingParameters': params,
        'UsedPhase': use_phase,
    }
    version = svht_denoise_version()
    if version is not None:
        provenance['SoftwareVersion'] = version

    # One volume for the whole run, so every echo's reportlet shows the same volume
    report_volume = select_report_volume(bold_series[0], n_volumes)

    for i, bold_file in enumerate(bold_series):
        echo = f'echo{i + 1}'
        phase_file = phase_series[i] if use_phase else None

        denoiser_kwargs = dict(params)
        echo_provenance = dict(provenance)
        if use_phase:
            phase_units = svht_phase_units(config.execution.layout.get_metadata(phase_file))
            if phase_units is not None:
                denoiser_kwargs['phase_units'] = phase_units

            echo_provenance['PhaseUnits'] = phase_units or 'detected from the data range'

        validate_bold = pe.Node(
            ValidateImage(in_file=bold_file),
            name=f'validate_bold_{echo}',
            mem_gb=mem_gb,
        )
        denoiser = pe.Node(
            SVHTDenoise(
                rank_image='rank.nii.gz',
                nthreads=omp_nthreads,
                report_volume=report_volume,
                **denoiser_kwargs,
            ),
            name=f'denoiser_{echo}',
            n_procs=omp_nthreads,
            mem_gb=mem_gb * SVHT_MEM_FACTOR,
        )
        sources = pe.Node(
            BIDSURI(
                numinputs=2 if use_phase else 1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(output_dir),
            ),
            name=f'sources_{echo}',
            run_without_submitting=True,
        )
        # BIDSURI adds its inN inputs after construction
        sources.inputs.in1 = bold_file
        if use_phase:
            sources.inputs.in2 = phase_file

        ds_noise = pe.Node(
            DerivativesDataSink(
                source_file=bold_file,
                base_directory=output_dir,
                desc='noise',
                suffix='boldmap',
                compress=True,
                dismiss_entities=_DISMISS_ENTITIES,
                Description=(
                    'Standard deviation of the thermal noise, estimated by svht_denoise '
                    'with the Gavish-Donoho median estimator'
                ),
                Units='arbitrary',
                **echo_provenance,
            ),
            name=f'ds_noise_{echo}',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_rank = pe.Node(
            DerivativesDataSink(
                source_file=bold_file,
                base_directory=output_dir,
                desc='rank',
                suffix='boldmap',
                compress=True,
                dismiss_entities=_DISMISS_ENTITIES,
                Description=(
                    'Signal rank: the number of principal components svht_denoise kept as signal'
                ),
                Units='count',
                **echo_provenance,
            ),
            name=f'ds_rank_{echo}',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_report = pe.Node(
            DerivativesDataSink(
                source_file=bold_file,
                base_directory=output_dir,
                desc='denoising',
                datatype='figures',
                dismiss_entities=_DISMISS_ENTITIES,
            ),
            name=f'ds_report_denoising_{echo}',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

        workflow.connect([
            (validate_bold, denoiser, [('out_file', 'in_file')]),
            (denoiser, ds_noise, [('noise_image', 'in_file')]),
            (denoiser, ds_rank, [('rank_image', 'in_file')]),
            (denoiser, ds_report, [('out_report', 'in_file')]),
            (sources, ds_noise, [('out', 'Sources')]),
            (sources, ds_rank, [('out', 'Sources')]),
            (ds_noise, merge_noise, [('out_file', f'in{i + 1}')]),
            (ds_rank, merge_rank, [('out_file', f'in{i + 1}')]),
        ])  # fmt:skip

        if use_phase:
            validate_phase = pe.Node(
                ValidateImage(in_file=phase_file),
                name=f'validate_phase_{echo}',
                mem_gb=mem_gb,
            )
            # The real-axis output is signed; later steps expect positive BOLD
            absolute = pe.Node(
                AbsoluteValue(),
                name=f'absolute_{echo}',
                mem_gb=mem_gb * 2,
            )
            workflow.connect([
                (validate_phase, denoiser, [('out_file', 'phase_file')]),
                (denoiser, absolute, [('out_file', 'in_file')]),
                (absolute, merge_bold, [('out_file', f'in{i + 1}')]),
            ])  # fmt:skip
        else:
            workflow.connect([(denoiser, merge_bold, [('out_file', f'in{i + 1}')])])

    return workflow
