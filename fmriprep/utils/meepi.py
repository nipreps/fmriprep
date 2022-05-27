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
"""Multi-echo EPI utilities."""


def combine_meepi_source(in_files):
    """
    Create a new source name when optimally
    combining multiple multi-echo EPIs

    >>> combine_meepi_source([
    ...     'sub-01_run-01_echo-1_bold.nii.gz',
    ...     'sub-01_run-01_echo-2_bold.nii.gz',
    ...     'sub-01_run-01_echo-3_bold.nii.gz',])
    'sub-01_run-01_bold.nii.gz'

    """
    import os
    from nipype.utils.filemanip import filename_to_list
    base, in_file = os.path.split(filename_to_list(in_files)[0])
    entities = [ent for ent in in_file.split('_') if not ent.startswith('echo-')]
    basename = '_'.join(entities)
    return os.path.join(base, basename)
