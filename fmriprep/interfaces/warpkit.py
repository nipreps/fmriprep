from __future__ import annotations

from pathlib import Path

import nibabel as nb
import numpy as np
from nipype.interfaces.base import File, InputMultiObject, SimpleInterface, TraitedSpec, traits


def _nvols(img: nb.spatialimages.SpatialImage) -> int:
    return img.shape[3] if img.ndim > 3 else 1


def _pad_to_nvols(in_file: Path, target_nvols: int, noise_frames: int) -> Path:
    img = nb.load(in_file)
    nvols = _nvols(img)
    if nvols == target_nvols:
        return in_file
    if nvols > target_nvols or target_nvols - nvols != noise_frames:
        raise RuntimeError(
            f'warpkit MEDIC returned {nvols} volume(s) for <{in_file}>, but '
            f'{target_nvols} were expected after trimming {noise_frames} noise frame(s).'
        )

    data = np.asanyarray(img.dataobj)
    padded = np.concatenate(
        (data, np.repeat(data[..., -1:], target_nvols - nvols, axis=-1)),
        axis=-1,
    )
    if in_file.name.endswith('.nii.gz'):
        out_name = in_file.name[:-7] + '_padded.nii.gz'
    elif in_file.name.endswith('.nii'):
        out_name = in_file.name[:-4] + '_padded.nii'
    else:
        out_name = in_file.name + '_padded'
    out_file = in_file.parent / out_name
    out_img = img.__class__(
        padded.astype(img.get_data_dtype(), copy=False),
        img.affine,
        img.header,
    )
    out_img.header.set_data_shape(padded.shape)
    out_img.to_filename(out_file)
    return out_file


class WarpkitMEDICInputSpec(TraitedSpec):
    phase = InputMultiObject(File(exists=True), mandatory=True, desc='Per-echo phase series')
    magnitude = InputMultiObject(
        File(exists=True), mandatory=True, desc='Per-echo magnitude series'
    )
    tes = traits.List(
        traits.Float,
        mandatory=True,
        minlen=3,
        desc='Echo times in milliseconds, one per echo',
    )
    total_readout_time = traits.Float(mandatory=True, desc='Total readout time in seconds')
    phase_encoding_direction = traits.Enum(
        'i',
        'i-',
        'j',
        'j-',
        'k',
        'k-',
        'x',
        'x-',
        'y',
        'y-',
        'z',
        'z-',
        mandatory=True,
        desc='Phase encoding direction',
    )
    n_cpus = traits.Int(1, usedefault=True, desc='Number of worker threads for MEDIC')
    noise_frames = traits.Int(0, usedefault=True, desc='Trailing noise frames to trim')
    wrap_limit = traits.Bool(False, usedefault=True, desc='Disable phase unwrapping heuristics')
    debug = traits.Bool(False, usedefault=True, desc='Enable warpkit debug mode')


class WarpkitMEDICOutputSpec(TraitedSpec):
    fieldmap_native = File(exists=True, desc='Framewise fieldmaps in distorted/native space')
    displacement_map = File(exists=True, desc='Framewise displacement maps')
    fieldmap = File(exists=True, desc='Framewise fieldmaps in undistorted space')


class WarpkitMEDIC(SimpleInterface):
    """Run warpkit's MEDIC workflow for a multi-echo BOLD series."""

    input_spec = WarpkitMEDICInputSpec
    output_spec = WarpkitMEDICOutputSpec

    def _run_interface(self, runtime):
        try:
            from warpkit.api import medic
        except ImportError as exc:
            raise RuntimeError(
                'warpkit is required for MEDIC-based multi-echo SDC. '
                'Install fMRIPrep with the optional warpkit extra on Python 3.11+.'
            ) from exc

        out_prefix = Path(runtime.cwd) / 'warpkit_medic'
        result = medic(
            phase=self.inputs.phase,
            magnitude=self.inputs.magnitude,
            out_prefix=out_prefix,
            tes=self.inputs.tes,
            total_readout_time=self.inputs.total_readout_time,
            phase_encoding_direction=self.inputs.phase_encoding_direction,
            n_cpus=self.inputs.n_cpus,
            noise_frames=self.inputs.noise_frames,
            wrap_limit=self.inputs.wrap_limit,
            debug=self.inputs.debug,
        )

        fieldmap_native = Path(result.fieldmap_native)
        displacement_map = Path(result.displacement_map)
        fieldmap = Path(result.fieldmap)

        if self.inputs.noise_frames:
            target_nvols = _nvols(nb.load(self.inputs.magnitude[0]))
            fieldmap_native = _pad_to_nvols(
                fieldmap_native,
                target_nvols,
                self.inputs.noise_frames,
            )
            displacement_map = _pad_to_nvols(
                displacement_map,
                target_nvols,
                self.inputs.noise_frames,
            )
            fieldmap = _pad_to_nvols(fieldmap, target_nvols, self.inputs.noise_frames)

        self._results['fieldmap_native'] = str(fieldmap_native)
        self._results['displacement_map'] = str(displacement_map)
        self._results['fieldmap'] = str(fieldmap)
        return runtime
