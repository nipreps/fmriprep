# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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

from numpy.linalg.linalg import LinAlgError
from nipype.algorithms import confounds as nac


class RobustACompCor(nac.ACompCor):
    """
    Runs aCompCor several times if it suddenly fails with
    https://github.com/nipreps/fmriprep/issues/776

    """

    def _run_interface(self, runtime):
        failures = 0
        while True:
            try:
                runtime = super(RobustACompCor, self)._run_interface(runtime)
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
                runtime = super(RobustTCompCor, self)._run_interface(runtime)
                break
            except LinAlgError:
                failures += 1
                if failures > 10:
                    raise
                start = (failures - 1) * 10
                sleep(randint(start + 4, start + 10))

        return runtime
