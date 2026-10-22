# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2026 The NiPreps Developers <nipreps@gmail.com>
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
"""Interfaces to CAT-Surface (``cat-surf``), which need no FreeSurfer or FSL installation."""

from pathlib import Path

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix


def cat_surf_version():
    """Return the installed version of ``cat-surf``, or ``None``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version('cat-surf')
    except PackageNotFoundError:
        return None


class _CATVolRealignInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='4D series to correct for head motion')
    ref_file = File(exists=True, desc='separate 3D reference volume (mcflirt -reffile)')
    save_mats = traits.Bool(False, usedefault=True, desc='write the matrices (mcflirt -mats)')
    save_plots = traits.Bool(False, usedefault=True, desc='write the parameters (mcflirt -plots)')
    save_rms = traits.Bool(
        False, usedefault=True, desc='write the RMS displacements (mcflirt -rmsabs -rmsrel)'
    )


class _CATVolRealignOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='realigned series')
    mat_file = traits.List(
        File(exists=True), desc='volume-to-reference transforms, FSL convention'
    )
    par_file = File(exists=True, desc='motion parameters: rx ry rz (rad) tx ty tz (mm)')
    rms_files = traits.List(File(exists=True), desc='absolute and relative RMS displacement')


class CATVolRealign(SimpleInterface):
    """Correct head motion with ``CAT_VolRealign``, in mcflirt's conventions.

    Inputs and outputs carry the names of :class:`~nipype.interfaces.fsl.MCFLIRT`, and the
    files are named and written as mcflirt writes them, so this node replaces ``mcflirt``
    without changing any connection.
    """

    input_spec = _CATVolRealignInputSpec
    output_spec = _CATVolRealignOutputSpec

    def _run_interface(self, runtime):
        from cat_surf import cli

        out = fname_presuffix(self.inputs.in_file, suffix='_mcf', newpath=runtime.cwd)
        kwargs = {
            'out': out,
            'mats': self.inputs.save_mats,
            'plots': self.inputs.save_plots,
            'rmsabs': self.inputs.save_rms,
            'rmsrel': self.inputs.save_rms,
        }
        if isdefined(self.inputs.ref_file):
            kwargs['reffile'] = self.inputs.ref_file
        result = cli.vol_realign(self.inputs.in_file, **kwargs)

        self._results['out_file'] = result['out_file']
        if self.inputs.save_mats:
            self._results['mat_file'] = [
                f'{out}.mat/MAT_{vol:04d}' for vol in range(len(result['par']))
            ]
        if self.inputs.save_plots:
            self._results['par_file'] = f'{out}.par'
        if self.inputs.save_rms:
            self._results['rms_files'] = [f'{out}_abs.rms', f'{out}_rel.rms']
        return runtime


class _CATBBRegInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='BOLD reference image (moving)')
    t1w_preproc = File(exists=True, mandatory=True, desc='T1w reference image (fixed)')
    white = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='left and right white-matter surfaces (GIFTI, in T1w space)',
    )
    cortex_mask = InputMultiObject(
        File(exists=True), desc='left and right cortex masks (GIFTI, one value per vertex)'
    )
    thickness = InputMultiObject(
        File(exists=True),
        desc='left and right cortical thickness (GIFTI); grey matter is then sampled at half '
        'the local thickness, as bbregister does',
    )
    use_bbr = traits.Either(
        None,
        traits.Bool,
        usedefault=True,
        desc='run BBR (True), only the initialization (False), or BBR unless it moves too far '
        'from the initialization (None)',
    )
    init = traits.Enum(
        't1w',
        'header',
        usedefault=True,
        desc='initialize by registering to the T1w (normalized mutual information), or from '
        'the image headers',
    )
    fallback_threshold = traits.Float(
        15.0,
        usedefault=True,
        desc='largest normalized displacement (mm) of BBR from the initialization to accept',
    )


class _CATBBRegOutputSpec(TraitedSpec):
    itk_bold_to_t1 = File(exists=True, desc='BOLD-to-T1w transform (ITK format)')
    itk_t1_to_bold = File(exists=True, desc='T1w-to-BOLD transform (ITK format)')
    fallback = traits.Bool(desc='the initialization was returned instead of BBR')
    metadata = traits.Dict(desc='provenance of the registration')


class CATBBReg(SimpleInterface):
    """Register a BOLD reference to the white-matter surfaces with CAT-Surface's BBR.

    The surfaces are in T1w space, so no FreeSurfer subject is involved.  The transform
    is rigid.  With ``use_bbr=None``, BBR is rejected, as by
    :py:func:`~fmriprep.workflows.bold.registration.compare_xforms`, when it moves
    more than ``fallback_threshold`` from the initialization.
    """

    input_spec = _CATBBRegInputSpec
    output_spec = _CATBBRegOutputSpec

    def _run_interface(self, runtime):
        import cat_surf
        import nibabel as nb
        import numpy as np
        from nipype.algorithms.rapidart import _calc_norm_affine

        ref_file = self.inputs.t1w_preproc if self.inputs.init == 't1w' else None

        def per_hemi(files, *, surface=False):
            if not isdefined(files):
                return None, None
            if surface:
                return tuple(
                    (
                        nb.load(f).agg_data('pointset').astype(np.float64),
                        nb.load(f).agg_data('triangle').astype(np.int32),
                    )
                    for f in files
                )
            return tuple(nb.load(f).agg_data().astype(np.float32) for f in files)

        # Moving (BOLD) RAS -> fixed (T1w) RAS
        init = np.eye(4)
        if self.inputs.use_bbr is not True and ref_file is not None:
            init, _ = cat_surf.bbreg(self.inputs.in_file, ref_file=ref_file)

        fallback = self.inputs.use_bbr is False
        matrix = init
        if not fallback:
            lh, rh = per_hemi(self.inputs.white, surface=True)
            lh_mask, rh_mask = per_hemi(self.inputs.cortex_mask)
            lh_thick, rh_thick = per_hemi(self.inputs.thickness)
            # bbregister's sampling for BOLD: 2 mm into white matter, half the
            # thickness into grey matter (or 0.5 mm without thickness)
            matrix, _ = cat_surf.bbreg(
                self.inputs.in_file,
                lh_surface=lh,
                rh_surface=rh,
                ref_file=ref_file,
                lh_mask=lh_mask,
                rh_mask=rh_mask,
                lh_thickness=lh_thick,
                rh_thickness=rh_thick,
                gm_proj_frac=0.5 if lh_thick is not None else 0.0,
                wm_dist=2.0,
                invert_contrast=1,
            )
            if self.inputs.use_bbr is None:
                norm, _ = _calc_norm_affine([init, matrix], use_differences=True)
                if norm[1] > self.inputs.fallback_threshold:
                    fallback = True
                    matrix = init

        # ITK transforms pull: bold-to-T1w maps T1w points into the BOLD reference
        self._results['itk_bold_to_t1'] = _write_itk(
            np.linalg.inv(matrix), Path(runtime.cwd) / 'bold_to_t1w.txt'
        )
        self._results['itk_t1_to_bold'] = _write_itk(matrix, Path(runtime.cwd) / 't1w_to_bold.txt')
        self._results['fallback'] = fallback
        self._results['metadata'] = {
            'GeneratedBy': [
                {
                    'Name': 'CAT-Surface',
                    'Version': cat_surf_version() or '<unknown>',
                    'CodeURL': 'https://github.com/ChristianGaser/CAT-Surface',
                }
            ]
        }
        return runtime


def _write_itk(matrix, out_file):
    """Write a RAS-to-RAS affine, which maps reference to moving points, as an ITK transform."""
    import nitransforms as nt

    nt.linear.Affine(matrix).to_filename(str(out_file), fmt='itk')
    return str(out_file)
