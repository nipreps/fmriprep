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
Denoising of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_dwidenoise_wf

"""

from nipype.interfaces import utility as niu
from nipype.interfaces.afni.utils import Calc
from nipype.pipeline import engine as pe

from ... import config
from ...interfaces.denoise import (
    ComplexToMagnitude,
    ComplexToPhase,
    DWIDenoise,
    NoiseEstimate,
    PhaseToRad,
    PolarToComplex,
    ValidateComplex,
)

LOGGER = config.loggers.workflow


def init_bold_dwidenoise_wf(
    *,
    mem_gb: dict,
    has_phase: bool = False,
    has_norf: bool = False,
    name='bold_dwidenoise_wf',
):
    """Create a workflow for the removal of thermal noise with dwidenoise.

    This workflow applies MP-PCA or NORDIC to the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.denoise  import init_bold_dwidenoise_wf
            wf = init_bold_dwidenoise_wf(
                has_phase=True,
                has_norf=True,
                mem_gb={'filesize': 1},
            )

    Parameters
    ----------
    has_phase : :obj:`bool`
        True if phase data are available. False if not.
    has_norf : :obj:`bool`
        True if no-excitation (noRF) data are available. False if not.
    mem_gb : :obj:`dict`
        Size of BOLD file in GB - please note that this size
        should be calculated after resamplings that may extend
        the FoV
    name : :obj:`str`
        Name of workflow (default: ``bold_dwidenoise_wf``)

    Inputs
    ------
    magnitude
        BOLD series NIfTI file
    phase
        Phase series NIfTI file
    magnitude_norf
        Noise map NIfTI file
    phase_norf
        Phase noise map NIfTI file

    Outputs
    -------
    magnitude
        Denoised BOLD series NIfTI file
    phase
        Denoised phase series NIfTI file
    noise
        Noise map NIfTI file
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    workflow = Workflow(name=name)
    workflow.__desc__ = (
        'The BOLD time-series were denoised to remove thermal noise using '
        '`dwidenoise` [@tournier2019mrtrix3] '
    )
    if config.workflow.thermal_denoise_method == 'nordic':
        workflow.__desc__ += 'with the NORDIC method [@moeller2021noise;@vizioli2021lowering].'
    else:
        workflow.__desc__ += (
            'with the Marchenko-Pastur principal components analysis method '
            '[@cordero2019complex].'
        )

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['magnitude', 'magnitude_norf', 'phase', 'phase_norf'],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'magnitude',
                'phase',
                'noise',
            ],
        ),
        name='outputnode',
    )

    if has_norf:
        workflow.__desc__ += ' An empirical noise map was estimated from no-excitation volumes.'
        # Calculate noise map from noise volumes
        # TODO: Calculate the noise level from the noise volumes
        # XXX: In NORDIC, noise level is estimated after scaling, phase filtering,
        # and g-factor correction.
        noise_estimate = pe.Node(
            NoiseEstimate(),
            name='noise_estimate',
            mem_gb=mem_gb['filesize'],
        )
        if has_phase:
            validate_complex_norf = pe.Node(ValidateComplex(), name='validate_complex_norf')
            workflow.connect([
                (inputnode, validate_complex_norf, [
                    ('magnitude_norf', 'magnitude'),
                    ('phase_norf', 'phase'),
                ]),
            ])  # fmt:skip

            # Combine magnitude and phase data into complex-valued data
            # XXX: Maybe we can rescale using hardcoded values if the data are from Siemens?
            phase_to_radians_norf = pe.Node(
                PhaseToRad(),
                name='phase_to_radians_norf',
            )
            workflow.connect([
                (validate_complex_norf, phase_to_radians_norf, [('phase', 'phase')]),
            ])  # fmt:skip

            combine_complex_norf = pe.Node(
                PolarToComplex(),
                name='combine_complex_norf',
            )
            workflow.connect([
                (validate_complex_norf, combine_complex_norf, [('magnitude', 'magnitude')]),
                (phase_to_radians_norf, combine_complex_norf, [('phase', 'phase')]),
                (combine_complex_norf, noise_estimate, [('out_file', 'in_file')]),
            ])  # fmt:skip

        else:
            workflow.connect([(inputnode, noise_estimate, [('magnitude_norf', 'in_file')])])

    complex_buffer = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='complex_buffer')
    if has_phase:
        workflow.__desc__ += (
            ' Magnitude and phase BOLD data were combined into complex-valued data prior to '
            'denoising, then the denoised complex-valued data were split back into magnitude '
            'and phase data.'
        )

        validate_complex = pe.Node(ValidateComplex(), name='validate_complex')

        # Combine magnitude and phase data into complex-valued data
        # XXX: Maybe we can rescale using hardcoded values if the data are from Siemens?
        phase_to_radians = pe.Node(
            PhaseToRad(),
            name='phase_to_radians',
        )
        workflow.connect([(validate_complex, phase_to_radians, [('phase', 'phase')])])

        combine_complex = pe.Node(
            PolarToComplex(),
            name='combine_complex',
        )
        workflow.connect([
            (validate_complex, combine_complex, [('magnitude', 'magnitude')]),
            (phase_to_radians, combine_complex, [('phase', 'phase')]),
            (combine_complex, complex_buffer, [('complex', 'bold_file')]),
        ])  # fmt:skip
    else:
        workflow.connect([(inputnode, complex_buffer, [('magnitude', 'bold_file')])])

    # Run NORDIC
    estimator = {
        'nordic': 'nordic',
        'mppca': 'Exp2',
    }
    dwidenoise = pe.Node(
        DWIDenoise(
            estimator=estimator[config.workflow.thermal_denoise_method],
        ),
        mem_gb=mem_gb['filesize'] * 2,
        name='dwidenoise',
    )
    workflow.connect([(complex_buffer, dwidenoise, [('bold_file', 'in_file')])])

    if has_norf:
        workflow.connect([(noise_estimate, dwidenoise, [('noise_map', 'noise_map')])])

    if has_phase:
        # Split the denoised complex-valued data into magnitude and phase
        split_magnitude = pe.Node(
            ComplexToMagnitude(),
            name='split_complex',
        )
        workflow.connect([
            (dwidenoise, split_magnitude, [('out_file', 'complex')]),
            (split_magnitude, outputnode, [('out_file', 'magnitude')]),
        ])  # fmt:skip

        split_phase = pe.Node(
            ComplexToPhase(),
            name='split_phase',
        )
        workflow.connect([
            (dwidenoise, split_phase, [('out_file', 'complex')]),
            (split_phase, outputnode, [('out_file', 'phase')]),
        ])  # fmt:skip

        # Apply sqrt(2) scaling factor to noise map
        rescale_noise = pe.Node(
            Calc(expr='a/sqrt(2)', outputtype='NIFTI_GZ'),
            name='rescale_noise',
        )
        workflow.connect([
            (noise_estimate, rescale_noise, [('noise_map', 'in_file_a')]),
            (rescale_noise, outputnode, [('out_file', 'noise')]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (dwidenoise, outputnode, [
                ('out_file', 'magnitude'),
                ('noise_image', 'noise'),
            ]),
        ])  # fmt:skip

    return workflow
