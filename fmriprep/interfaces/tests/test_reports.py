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
import pytest

from ..reports import get_world_pedir


@pytest.mark.parametrize("orientation,pe_dir,expected", [
    ('RAS', 'j', 'Posterior-Anterior'),
    ('RAS', 'j-', 'Anterior-Posterior'),
    ('RAS', 'i', 'Left-Right'),
    ('RAS', 'i-', 'Right-Left'),
    ('RAS', 'k', 'Inferior-Superior'),
    ('RAS', 'k-', 'Superior-Inferior'),
    ('LAS', 'j', 'Posterior-Anterior'),
    ('LAS', 'i-', 'Left-Right'),
    ('LAS', 'k-', 'Superior-Inferior'),
    ('LPI', 'j', 'Anterior-Posterior'),
    ('LPI', 'i-', 'Left-Right'),
    ('LPI', 'k-', 'Inferior-Superior'),
    ('SLP', 'k-', 'Posterior-Anterior'),
    ('SLP', 'k', 'Anterior-Posterior'),
    ('SLP', 'j-', 'Left-Right'),
    ('SLP', 'j', 'Right-Left'),
    ('SLP', 'i', 'Inferior-Superior'),
    ('SLP', 'i-', 'Superior-Inferior'),
])
def test_get_world_pedir(tmpdir, orientation, pe_dir, expected):
    assert get_world_pedir(orientation, pe_dir) == expected
