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
from nipype.interfaces.base import File, traits
from numpy.linalg.linalg import LinAlgError


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
