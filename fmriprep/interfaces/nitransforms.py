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

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
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
