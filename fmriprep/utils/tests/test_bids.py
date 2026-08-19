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
import pytest

from fmriprep.utils.bids import is_valid_bold_template


class _MockLayout:
    """Minimal BIDSLayout stub for is_valid_bold_template tests."""

    def __init__(self, pe_map: dict):
        self._pe_map = pe_map

    def get_metadata(self, f):
        pe = self._pe_map.get(f)
        return {'PhaseEncodingDirection': pe} if pe is not None else {}


# Each case pairs (bold_runs, estimator_map, pe_map) with the expected validity:
# a template needs >=2 runs, uniform SDC status, and SDC-less runs to share one
# phase-encoding direction. ``ids`` label the scenario each case exercises.
@pytest.mark.parametrize(
    ('bold_runs', 'estimator_map', 'pe_map', 'expected'),
    [
        ([], {}, {}, False),
        ([['a.nii']], {}, {'a.nii': 'j'}, False),
        ([['a.nii'], ['b.nii']], {'a.nii': 'fmap1', 'b.nii': 'fmap2'}, {}, True),
        ([['a.nii'], ['b.nii']], {'a.nii': 'fmap1'}, {'a.nii': 'j', 'b.nii': 'j'}, False),
        ([['a.nii'], ['b.nii']], {}, {'a.nii': 'j', 'b.nii': 'j'}, True),
        ([['a.nii'], ['b.nii']], {}, {'a.nii': 'j', 'b.nii': 'j-'}, False),
        ([['a.nii'], ['b.nii']], {}, {}, True),
        ([['a.nii'], ['b.nii']], {}, {'a.nii': 'j'}, False),
    ],
    ids=[
        'false-no_runs',
        'false-one_run',
        'true-all_sdc',
        'false-mixed_sdc',
        'true-no_sdc_single_pe',
        'false-no_sdc_opposing_pe',
        'true-no_sdc_no_pe',
        'false-no_sdc_missing_pe',
    ],
)
def test_is_valid_bold_template(bold_runs, estimator_map, pe_map, expected):
    layout = _MockLayout(pe_map)
    assert is_valid_bold_template(bold_runs, estimator_map, layout) is expected
