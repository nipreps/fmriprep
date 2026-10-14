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
Temporary patches
-----------------

"""

from random import randint
from time import sleep

import nipype.interfaces.freesurfer as fs
import nipype.interfaces.io as nio
from nipype.algorithms import confounds as nac
from nipype.interfaces import afni
from nipype.interfaces.base import File, Undefined, traits
from numpy.linalg import LinAlgError


class RobustACompCor(nac.ACompCor):
    """
    Runs aCompCor several times if it suddenly fails with
    https://github.com/nipreps/fmriprep/issues/776

    """

    def _run_interface(self, runtime):
        failures = 0
        while True:
            try:
                runtime = super()._run_interface(runtime)
                break
            except LinAlgError:
                failures += 1
                if failures > 10:
                    raise
                start = (failures - 1) * 10
                sleep(randint(start + 4, start + 10))

        return runtime


class RobustTCompCor(nac.TCompCor):
    """
    Runs tCompCor several times if it suddenly fails with
    https://github.com/nipreps/fmriprep/issues/940

    """

    def _run_interface(self, runtime):
        failures = 0
        while True:
            try:
                runtime = super()._run_interface(runtime)
                break
            except LinAlgError:
                failures += 1
                if failures > 10:
                    raise
                start = (failures - 1) * 10
                sleep(randint(start + 4, start + 10))

        return runtime


class _MRICoregInputSpec(fs.registration.MRICoregInputSpec):
    reference_file = File(
        argstr='--ref %s',
        desc='reference (target) file',
        copyfile=False,
    )
    subject_id = traits.Str(
        argstr='--s %s',
        position=1,
        requires=['subjects_dir'],
        desc='freesurfer subject ID (implies ``reference_mask == '
        'aparc+aseg.mgz`` unless otherwise specified)',
    )


class MRICoreg(fs.MRICoreg):
    """
    Patched that allows setting both a reference file and the subjects dir.
    """

    input_spec = _MRICoregInputSpec


class _FSSourceOutputSpec(nio.FSSourceOutputSpec):
    T2 = File(desc='Intensity normalized whole-head volume', loc='mri')


class FreeSurferSource(nio.FreeSurferSource):
    """
    Patch to allow grabbing the T2 volume, if available
    """

    output_spec = _FSSourceOutputSpec


class _VolregInputSpec(afni.preprocess.VolregInputSpec):
    # Redefined without argstr or name_template, so -maxdisp1D is never passed
    md1d_file = File(desc='not used: -maxdisp1D is never passed')
    twopass = traits.Bool(
        argstr='-twopass',
        desc='register in two passes: a coarse pass on blurred images, then a fine pass',
    )


class _VolregOutputSpec(afni.preprocess.VolregOutputSpec):
    out_file = File(desc="registered file; undefined if ``out_file`` is ``'NULL'``")


class Volreg(afni.Volreg):
    """
    Patched to add ``-twopass``, to skip ``-maxdisp1D``, and to accept ``out_file='NULL'``.

    With ``out_file='NULL'``, 3dvolreg estimates motion without writing a
    registered series, and the ``out_file`` output is undefined.

    >>> volreg = Volreg(
    ...     in_file='bold.nii.gz',
    ...     basefile='boldref.nii.gz',
    ...     interp='heptic',
    ...     twopass=True,
    ...     zpad=4,
    ...     out_file='NULL'
    ... )
    >>> volreg.cmdline  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '3dvolreg -heptic -1Dfile bold.1D -1Dmatrix_save bold.aff12.1D -prefix NULL \
    -twopass -base boldref.nii.gz -zpad 4 bold.nii.gz'

    """

    input_spec = _VolregInputSpec
    output_spec = _VolregOutputSpec

    def _list_outputs(self):
        outputs = super()._list_outputs()
        if self.inputs.out_file == 'NULL':
            outputs['out_file'] = Undefined
        return outputs
