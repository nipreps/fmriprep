from __future__ import annotations

from pathlib import Path

from nipype.interfaces.base import File, InputMultiObject, SimpleInterface, TraitedSpec, traits


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
            wrap_limit=self.inputs.wrap_limit,
            debug=self.inputs.debug,
        )

        self._results['fieldmap_native'] = str(result.fieldmap_native)
        self._results['displacement_map'] = str(result.displacement_map)
        self._results['fieldmap'] = str(result.fieldmap)
        return runtime
