"""Interfaces for resampling images in a single shot"""

import nibabel as nb
from nipost import load_transforms, reconstruct_fieldmap, resample_image
from nipost.epi import ensure_positive_cosines, get_trt
from nipype.interfaces.base import (
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix


class ResampleSeriesInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='3D or 4D image file to resample')
    ref_file = File(exists=True, mandatory=True, desc='File to resample in_file to')
    transforms = InputMultiObject(
        File(exists=True),
        desc='Transform files, from in_file to ref_file (image mode)',
    )
    inverse = InputMultiObject(
        traits.Bool,
        value=[False],
        usedefault=True,
        desc='Whether to invert each file in transforms',
    )
    fieldmap = File(exists=True, desc='Fieldmap file resampled into reference space')
    ro_time = traits.Float(desc='EPI readout time (s).')
    pe_dir = traits.Enum(
        'i',
        'i-',
        'j',
        'j-',
        'k',
        'k-',
        desc='the phase-encoding direction corresponding to in_data',
    )
    jacobian = traits.Bool(mandatory=True, desc='Whether to apply Jacobian correction')
    num_threads = traits.Int(1, usedefault=True, desc='Number of threads to use for resampling')
    output_data_type = traits.Str('float32', usedefault=True, desc='Data type of output image')
    order = traits.Int(3, usedefault=True, desc='Order of interpolation (0=nearest, 3=cubic)')
    mode = traits.Enum(
        'grid-constant',
        'nearest',
        'constant',
        'mirror',
        'reflect',
        'wrap',
        'grid-mirror',
        'grid-wrap',
        usedefault=True,
        desc='How data is extended beyond its boundaries. '
        'See scipy.ndimage.map_coordinates for more details.',
    )
    cval = traits.Float(0.0, usedefault=True, desc='Value to fill past edges of data')
    prefilter = traits.Bool(True, usedefault=True, desc='Spline-prefilter data if order > 1')


class ResampleSeriesOutputSpec(TraitedSpec):
    out_file = File(desc='Resampled image or series')


class ResampleSeries(SimpleInterface):
    """Resample a time series, applying susceptibility and motion correction
    simultaneously.
    """

    input_spec = ResampleSeriesInputSpec
    output_spec = ResampleSeriesOutputSpec

    def _run_interface(self, runtime):
        out_path = fname_presuffix(self.inputs.in_file, suffix='resampled', newpath=runtime.cwd)

        source = nb.load(self.inputs.in_file)
        target = nb.load(self.inputs.ref_file)
        fieldmap = nb.load(self.inputs.fieldmap) if self.inputs.fieldmap else None

        nvols = source.shape[3] if source.ndim > 3 else 1

        # No transforms appear Undefined, pass as empty list
        transforms = load_transforms(self.inputs.transforms or [], self.inputs.inverse)

        pe_dir = self.inputs.pe_dir
        ro_time = self.inputs.ro_time
        pe_info = None

        if pe_dir and ro_time:
            pe_axis = 'ijk'.index(pe_dir[0])
            pe_flip = pe_dir.endswith('-')

            # Nitransforms displacements are positive
            source, axcodes = ensure_positive_cosines(source)
            axis_flip = axcodes[pe_axis] in 'LPI'

            pe_info = [(pe_axis, -ro_time if (axis_flip ^ pe_flip) else ro_time)] * nvols

        resampled = resample_image(
            source=source,
            target=target,
            transforms=transforms,
            fieldmap=fieldmap,
            pe_info=pe_info,
            jacobian=self.inputs.jacobian,
            nthreads=self.inputs.num_threads,
            output_dtype=self.inputs.output_data_type,
            order=self.inputs.order,
            mode=self.inputs.mode,
            cval=self.inputs.cval,
            prefilter=self.inputs.prefilter,
        )
        resampled.to_filename(out_path)

        self._results['out_file'] = out_path
        return runtime


class ReconstructFieldmapInputSpec(TraitedSpec):
    in_coeffs = InputMultiObject(
        File(exists=True), mandatory=True, desc='SDCflows-style spline coefficient files'
    )
    target_ref_file = File(
        exists=True, mandatory=True, desc='Image to reconstruct the field in alignment with'
    )
    fmap_ref_file = File(
        exists=True, mandatory=True, desc='Reference file aligned with coefficients'
    )
    transforms = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='Transform files, from in_file to ref_file (image mode)',
    )
    inverse = InputMultiObject(
        traits.Bool,
        value=[False],
        usedefault=True,
        desc='Whether to invert each file in transforms',
    )


class ReconstructFieldmapOutputSpec(TraitedSpec):
    out_file = File(desc='Fieldmap reconstructed in target_ref_file space')


class ReconstructFieldmap(SimpleInterface):
    """Reconstruct a fieldmap from B-spline coefficients in a target space.

    If the target reference does not have an aligned grid (guaranteed if
    transforms include a warp), then a reference file describing the space
    where it is valid to extrapolate the field will be used as an intermediate
    step.
    """

    input_spec = ReconstructFieldmapInputSpec
    output_spec = ReconstructFieldmapOutputSpec

    def _run_interface(self, runtime):
        out_path = fname_presuffix(self.inputs.in_coeffs[-1], suffix='rec', newpath=runtime.cwd)

        coefficients = [nb.load(coeff_file) for coeff_file in self.inputs.in_coeffs]
        target = nb.load(self.inputs.target_ref_file)
        fmapref = nb.load(self.inputs.fmap_ref_file)

        transforms = load_transforms(self.inputs.transforms, self.inputs.inverse)

        fieldmap = reconstruct_fieldmap(
            coefficients=coefficients,
            fmap_reference=fmapref,
            target=target,
            transforms=transforms,
        )
        fieldmap.to_filename(out_path)

        self._results['out_file'] = out_path
        return runtime


class DistortionParametersInputSpec(TraitedSpec):
    in_file = File(exists=True, desc='EPI image corresponding to the metadata')
    metadata = traits.Dict(mandatory=True, desc='metadata corresponding to the inputs')
    fallback = traits.Either(
        None,
        'estimated',
        traits.Float,
        usedefault=True,
        desc='Fallback value for missing metadata',
    )


class DistortionParametersOutputSpec(TraitedSpec):
    readout_time = traits.Float
    pe_direction = traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-')


class DistortionParameters(SimpleInterface):
    """Retrieve PhaseEncodingDirection and TotalReadoutTime from available metadata.

    One or both parameters may be missing; downstream interfaces should be prepared
    to handle this.
    """

    input_spec = DistortionParametersInputSpec
    output_spec = DistortionParametersOutputSpec

    def _run_interface(self, runtime):
        try:
            self._results['readout_time'] = get_trt(
                self.inputs.metadata,
                self.inputs.in_file or None,
                use_estimate=self.inputs.fallback == 'estimated',
                fallback=self.inputs.fallback if isinstance(self.inputs.fallback, float) else None,
            )
            self._results['pe_direction'] = self.inputs.metadata['PhaseEncodingDirection']
        except (KeyError, ValueError):
            pass

        return runtime
