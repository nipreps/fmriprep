"""Denoising-related interfaces."""

import nibabel as nb
import numpy as np
from nilearn.image import load_img
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.mrtrix3.base import MRTrix3Base, MRTrix3BaseInputSpec
from nipype.utils.filemanip import fname_presuffix


class _ValidateComplexInputSpec(BaseInterfaceInputSpec):
    magnitude = File(
        exists=True,
        mandatory=True,
        desc='Magnitude BOLD EPI',
    )
    phase = File(
        exists=True,
        mandatory=False,
        desc='Phase BOLD EPI',
    )


class _ValidateComplexOutputSpec(TraitedSpec):
    magnitude = File(exists=True, desc='Validated magnitude file')
    phase = File(exists=True, desc='Validated phase file')


class ValidateComplex(SimpleInterface):
    """Validate complex-valued BOLD data."""

    input_spec = _ValidateComplexInputSpec
    output_spec = _ValidateComplexOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb

        if not isdefined(self.inputs.phase):
            self._results['magnitude'] = self.inputs.magnitude
            return runtime

        mag_img = nb.load(self.inputs.magnitude)
        phase_img = nb.load(self.inputs.phase)
        n_mag_vols = mag_img.shape[3]
        n_phase_vols = phase_img.shape[3]

        if n_mag_vols != n_phase_vols:
            raise ValueError(
                f'Number of volumes in magnitude file ({n_mag_vols}) '
                f'!= number of volumes in phase file ({n_phase_vols})'
            )

        self._results['magnitude'] = self.inputs.magnitude
        self._results['phase'] = self.inputs.phase

        return runtime


class DWIDenoiseInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    estimator = traits.Enum(
        'Exp2',
        argstr='-estimator %s',
        desc='noise estimator to use. (default = Exp2)',
    )
    mask = File(exists=True, argstr='-mask %s', position=1, desc='mask image')
    extent = traits.Tuple(
        (traits.Int, traits.Int, traits.Int),
        argstr='-extent %d,%d,%d',
        desc='set the window size of the denoising filter. (default = 5,5,5)',
    )
    noise_image = File(
        argstr='-noise %s',
        name_template='%s_noise.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        desc='the output noise map',
    )
    out_file = File(
        name_template='%s_denoised.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        argstr='%s',
        position=-1,
        desc='the output denoised DWI image',
    )


class DWIDenoiseOutputSpec(TraitedSpec):
    noise_image = File(desc='the output noise map', exists=True)
    out_file = File(desc='the output denoised DWI image', exists=True)


class DWIDenoise(MRTrix3Base):
    """
    Denoise DWI data and estimate the noise level based on the optimal
    threshold for PCA.

    DWI data denoising and noise map estimation by exploiting data redundancy
    in the PCA domain using the prior knowledge that the eigenspectrum of
    random covariance matrices is described by the universal Marchenko Pastur
    distribution.

    Important note: image denoising must be performed as the first step of the
    image processing pipeline. The routine will fail if interpolation or
    smoothing has been applied to the data prior to denoising.

    Note that this function does not correct for non-Gaussian noise biases.

    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/commands/dwidenoise.html>

    Notes
    -----
    There is a -rank option to output a map of the degrees of freedom in dev,
    but it won't be released until 3.1.0.
    NORDIC is on the roadmap, but it's unknown when it will be implemented.
    """

    _cmd = 'dwidenoise'
    input_spec = DWIDenoiseInputSpec
    output_spec = DWIDenoiseOutputSpec

    def _get_plotting_images(self):
        input_dwi = load_img(self.inputs.in_file)
        outputs = self._list_outputs()
        ref_name = outputs.get('out_file')
        denoised_nii = load_img(ref_name)
        noise_name = outputs['noise_image']
        noisenii = load_img(noise_name)
        return input_dwi, denoised_nii, noisenii


class _PolarToComplexInputSpec(MRTrix3BaseInputSpec):
    magnitude = traits.File(exists=True, mandatory=True, position=0, argstr='%s')
    phase = traits.File(exists=True, mandatory=True, position=1, argstr='%s')
    complex = traits.File(
        exists=False,
        name_source='magnitude',
        name_template='%s_complex.nii.gz',
        keep_extension=False,
        position=-1,
        argstr='-polar %s',
    )


class _PolarToComplexOutputSpec(TraitedSpec):
    complex = File(exists=True)


class PolarToComplex(MRTrix3Base):
    """Convert a magnitude and phase image pair to a single complex image using mrcalc."""

    input_spec = _PolarToComplexInputSpec
    output_spec = _PolarToComplexOutputSpec

    _cmd = 'mrcalc'


class _ComplexToMagnitudeInputSpec(MRTrix3BaseInputSpec):
    complex_file = traits.File(exists=True, mandatory=True, position=0, argstr='%s')
    out_file = traits.File(
        exists=False,
        name_source='complex_file',
        name_template='%s_mag.nii.gz',
        keep_extension=False,
        position=-1,
        argstr='-abs %s',
    )


class _ComplexToMagnitudeOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ComplexToMagnitude(MRTrix3Base):
    """Extract the magnitude portion of a complex image using mrcalc."""

    input_spec = _ComplexToMagnitudeInputSpec
    output_spec = _ComplexToMagnitudeOutputSpec

    _cmd = 'mrcalc'


class _ComplexToPhaseInputSpec(MRTrix3BaseInputSpec):
    complex_file = traits.File(exists=True, mandatory=True, position=0, argstr='%s')
    out_file = traits.File(
        exists=False,
        name_source='complex_file',
        name_template='%s_ph.nii.gz',
        keep_extension=False,
        position=-1,
        argstr='-phase %s',
    )


class _ComplexToPhaseOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ComplexToPhase(MRTrix3Base):
    """Extract the phase portion of a complex image using mrcalc."""

    input_spec = _ComplexToPhaseInputSpec
    output_spec = _ComplexToPhaseOutputSpec

    _cmd = 'mrcalc'


class _PhaseToRadInputSpec(BaseInterfaceInputSpec):
    """Output spec for PhaseToRad interface.

    STATEMENT OF CHANGES: This class is derived from sources licensed under the Apache-2.0 terms,
    and the code has been changed.

    Notes
    -----
    The code is derived from
    https://github.com/nipreps/sdcflows/blob/c6cd42944f4b6d638716ce020ffe51010e9eb58a/\
    sdcflows/utils/phasemanip.py#L26.

    License
    -------
    ORIGINAL WORK'S ATTRIBUTION NOTICE:

    Copyright 2021 The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """

    phase = File(exists=True, mandatory=True)


class _PhaseToRadOutputSpec(TraitedSpec):
    """Output spec for PhaseToRad interface.

    STATEMENT OF CHANGES: This class is derived from sources licensed under the Apache-2.0 terms,
    and the code has been changed.

    Notes
    -----
    The code is derived from
    https://github.com/nipreps/sdcflows/blob/c6cd42944f4b6d638716ce020ffe51010e9eb58a/\
    sdcflows/utils/phasemanip.py#L26.

    License
    -------
    ORIGINAL WORK'S ATTRIBUTION NOTICE:

    Copyright 2021 The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """

    phase = File(exists=True)


class PhaseToRad(SimpleInterface):
    """Convert phase image from arbitrary units (au) to radians.

    This method assumes that the phase image's minimum and maximum values correspond to
    -pi and pi, respectively, and scales the image to be between 0 and 2*pi.

    STATEMENT OF CHANGES: This class is derived from sources licensed under the Apache-2.0 terms,
    and the code has not been changed.

    Notes
    -----
    The code is derived from
    https://github.com/nipreps/sdcflows/blob/c6cd42944f4b6d638716ce020ffe51010e9eb58a/\
    sdcflows/utils/phasemanip.py#L26.

    License
    -------
    ORIGINAL WORK'S ATTRIBUTION NOTICE:

    Copyright 2021 The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """

    input_spec = _PhaseToRadInputSpec
    output_spec = _PhaseToRadOutputSpec

    def _run_interface(self, runtime):
        im = nb.load(self.inputs.phase)
        data = im.get_fdata(caching='unchanged')  # Read as float64 for safety
        hdr = im.header.copy()

        # Rescale to [0, 2*pi]
        data = (data - data.min()) * (2 * np.pi / (data.max() - data.min()))

        # Round to float32 and clip
        data = np.clip(np.float32(data), 0.0, 2 * np.pi)

        hdr.set_data_dtype(np.float32)
        hdr.set_xyzt_units('mm')

        # Set the output file name
        self._results['phase'] = fname_presuffix(
            self.inputs.phase,
            suffix='_rad.nii.gz',
            newpath=runtime.cwd,
            use_ext=False,
        )

        # Save the output image
        nb.Nifti1Image(data, None, hdr).to_filename(self._results['phase'])

        return runtime


class _NoiseEstimateInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=-2, desc='input DWI image')
    out_file = File(
        name_template='%s_noise.nii.gz',
        name_source='in_file',
        keep_extension=False,
        argstr='%s',
        position=-1,
        desc='the output noise map',
    )


class _NoiseEstimateOutputSpec(TraitedSpec):
    out_file = File(desc='the output noise map', exists=True)


class NoiseEstimate(MRTrix3Base):
    """Estimate a noise level map from a 4D no-excitation time series.

    XXX: This is a nonfunctioning interface.
    """

    _cmd = 'dwi2noise'
    input_spec = _NoiseEstimateInputSpec
    output_spec = _NoiseEstimateOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        return outputs
