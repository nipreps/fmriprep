# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2025 The NiPreps Developers <nipreps@gmail.com>
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
"""Wrappers of NiTransforms."""

from pathlib import Path

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix


class _ConvertAffineInputSpec(BaseInterfaceInputSpec):
    in_xfm = File(exists=True, desc='input transform piles')
    inverse = traits.Bool(False, usedefault=True, desc='generate inverse')
    in_fmt = traits.Enum('auto', 'itk', 'fs', 'fsl', usedefault=True, desc='input format')
    out_fmt = traits.Enum('itk', 'fs', 'fsl', usedefault=True, desc='output format')
    reference = File(exists=True, desc='reference file')
    moving = File(exists=True, desc='moving file')


class _ConvertAffineOutputSpec(TraitedSpec):
    out_xfm = File(exists=True, desc='output, combined transform')
    out_inv = File(desc='output, combined transform')


class ConvertAffine(SimpleInterface):
    """Write a single, flattened transform file."""

    input_spec = _ConvertAffineInputSpec
    output_spec = _ConvertAffineOutputSpec

    def _run_interface(self, runtime):
        from nitransforms.linear import load as load_affine

        ext = {
            'fs': 'lta',
            'itk': 'txt',
            'fsl': 'mat',
        }[self.inputs.out_fmt]

        in_fmt = self.inputs.in_fmt
        if in_fmt == 'auto':
            in_fmt = {
                '.lta': 'fs',
                '.mat': 'fsl',
                '.txt': 'itk',
            }[Path(self.inputs.in_xfm).suffix]

        reference = self.inputs.reference or None
        moving = self.inputs.moving or None
        affine = load_affine(self.inputs.in_xfm, fmt=in_fmt, reference=reference, moving=moving)

        out_file = fname_presuffix(
            self.inputs.in_xfm,
            suffix=f'_fwd.{ext}',
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results['out_xfm'] = out_file
        affine.to_filename(out_file, moving=moving, fmt=self.inputs.out_fmt)

        if self.inputs.inverse:
            inv_affine = ~affine
            if moving is not None:
                inv_affine.reference = moving

            out_inv = fname_presuffix(
                self.inputs.in_xfm,
                suffix=f'_inv.{ext}',
                newpath=runtime.cwd,
                use_ext=False,
            )
            self._results['out_inv'] = out_inv
            inv_affine.to_filename(out_inv, moving=reference, fmt=self.inputs.out_fmt)

        return runtime


class _VolregToITKInputSpec(BaseInterfaceInputSpec):
    in_params = File(exists=True, mandatory=True, desc='motion parameters from 3dvolreg -1Dfile')
    in_matrices = File(
        exists=True, desc='matrices from 3dvolreg -1Dmatrix_save, to cross-check the conversion'
    )
    reference = File(
        exists=True,
        mandatory=True,
        desc='3dvolreg base image, on the grid of the registered BOLD series',
    )


class _VolregToITKOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='ITK transform file, one affine per volume')


class VolregToITK(SimpleInterface):
    """Convert 3dvolreg motion parameters to an ITK transform file.

    The transforms are built from the ``-1Dfile`` parameters, which are
    3dvolreg's reported motion. They are written at full precision, so the
    stored transforms reproduce the parameters to better than 1e-5 (degrees
    and mm). ``-1Dmatrix_save`` is printed to only 6 significant digits; if it
    is given, it is used only to check the conversion, and a disagreement
    larger than its print precision raises ``RuntimeError``.
    """

    input_spec = _VolregToITKInputSpec
    output_spec = _VolregToITKOutputSpec

    def _run_interface(self, runtime):
        from nitransforms.linear import load as load_affine

        from ..utils.transforms import volreg_to_ras, write_itk_affines

        params = np.loadtxt(self.inputs.in_params, ndmin=2)
        if params.shape[1] != 6:
            raise ValueError(
                f'{self.inputs.in_params} has {params.shape[1]} columns; 3dvolreg -1Dfile writes 6'
            )

        ref = nb.load(self.inputs.reference)
        ras = volreg_to_ras(params, ref)

        if isdefined(self.inputs.in_matrices):
            saved = load_affine(
                self.inputs.in_matrices,
                fmt='afni',
                reference=ref,
                moving=ref,
            ).matrix.reshape(-1, 4, 4)
            if len(saved) != len(ras):
                raise ValueError(
                    f'{self.inputs.in_params} has {len(ras)} rows, but '
                    f'{self.inputs.in_matrices} has {len(saved)} matrices'
                )
            # 6 significant digits leave about 1e-6 in the rotation and 1e-4 mm
            rot_err = np.abs(saved[:, :3, :3] - ras[:, :3, :3]).max(axis=(1, 2))
            trans_err = np.abs(saved[:, :3, 3] - ras[:, :3, 3]).max(axis=1)
            bad = np.flatnonzero((rot_err > 1e-5) | (trans_err > 1e-3))
            if bad.size:
                raise RuntimeError(
                    f'3dvolreg parameters and matrices disagree at volume {bad[0]}: '
                    f'{rot_err[bad[0]]:.2g} in the rotation, '
                    f'{trans_err[bad[0]]:.2g} mm in the translation'
                )

        self._results['out_file'] = fname_presuffix(
            self.inputs.reference, suffix='_hmc.txt', newpath=runtime.cwd, use_ext=False
        )
        write_itk_affines(ras, self._results['out_file'])
        return runtime
