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
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from fmriprep.utils.bids import is_valid_bold_template


class _MockLayout:
    """Minimal BIDSLayout stub for is_valid_bold_template tests."""

    def __init__(self, pe_map: dict):
        self._pe_map = pe_map

    def get_metadata(self, f):
        pe = self._pe_map.get(f)
        return {'PhaseEncodingDirection': pe} if pe is not None else {}


TEST_ZOOMS = (2.4, 2.4, 2.4)


# Each run is a (fieldmap, phase-encoding direction, voxel size) tuple
@pytest.mark.parametrize(
    ('runs', 'expected'),
    [
        pytest.param([], False, id='no_runs'),
        pytest.param([('fmap1', 'j', TEST_ZOOMS)], False, id='one_run'),
        pytest.param([('fmap1', 'j', TEST_ZOOMS), ('fmap2', 'j', TEST_ZOOMS)], True, id='all_sdc'),
        pytest.param([('fmap1', 'j', TEST_ZOOMS), (None, 'j', TEST_ZOOMS)], False, id='mixed_sdc'),
        pytest.param(
            [(None, 'j', TEST_ZOOMS), (None, 'j', TEST_ZOOMS)], True, id='no_sdc_single_pe'
        ),
        pytest.param(
            [(None, 'j', TEST_ZOOMS), (None, 'j-', TEST_ZOOMS)], False, id='no_sdc_opposing_pe'
        ),
        pytest.param(
            [(None, None, TEST_ZOOMS), (None, None, TEST_ZOOMS)], True, id='no_sdc_no_pe'
        ),
        pytest.param(
            [(None, 'j', TEST_ZOOMS), (None, None, TEST_ZOOMS)], False, id='no_sdc_missing_pe'
        ),
        pytest.param(
            [('fmap1', 'j', TEST_ZOOMS), ('fmap2', 'j', (0.8, 0.8, 0.8))],
            False,
            id='mixed_resolution',
        ),
        pytest.param(
            [('fmap1', 'j', TEST_ZOOMS), ('fmap2', 'j', (2.4, 2.4, 3.0))],
            False,
            id='mixed_slice_thickness',
        ),
        pytest.param(
            [('fmap1', 'j', TEST_ZOOMS), ('fmap2', 'j', (2.4002, 2.4002, 2.4002))],
            True,
            id='negligible_zoom_difference',
        ),
    ],
)
def test_is_valid_bold_template(tmp_path: Path, runs, expected):
    bold_runs, estimator_map, pe_map = [], {}, {}
    for i, (fieldmap, pe_dir, zooms) in enumerate(runs):
        path = str(tmp_path / f'run{i}.nii.gz')
        nb.Nifti1Image(np.zeros((2, 2, 2), dtype='uint8'), np.diag((*zooms, 1.0))).to_filename(
            path
        )
        bold_runs.append([path])
        estimator_map[path] = fieldmap
        pe_map[path] = pe_dir

    layout = _MockLayout(pe_map)
    assert is_valid_bold_template(bold_runs, estimator_map, layout) is expected
