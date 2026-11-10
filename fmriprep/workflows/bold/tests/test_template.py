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
from nipype.interfaces.base import isdefined

from ..template import init_bold_template_wf


@pytest.mark.parametrize(
    ('reference', 'template_runs', 'initial_timepoint', 'order'),
    [
        (0, None, 1, None),
        (None, None, None, None),
        (2, [False, True, True], 2, [2, 0, 1]),
        (None, [True, False, True], None, [0, 2, 1]),
        (1, [False, True, False], 1, [1, 0, 2]),
    ],
)
def test_template_selection(reference, template_runs, initial_timepoint, order):
    """Only the template runs build the template; the rest are registered to it, in run order."""
    wf = init_bold_template_wf(num_bold_runs=3, reference=reference, template_runs=template_runs)

    fixed = reference is not None
    robust_template = wf.get_node('boldref_template').inputs
    assert robust_template.fixed_timepoint is fixed
    assert robust_template.no_iteration is fixed
    if fixed:
        assert robust_template.initial_timepoint == initial_timepoint
    # A single template run is copied, and writes no intensity scale
    assert isdefined(robust_template.scaled_intensity_outputs) is (
        sum(template_runs or [1] * 3) > 1
    )

    sort_xfms = wf.get_node('sort_xfms')
    assert (wf.get_node('register_excluded_wf') is not None) is (order is not None)
    if order is not None:
        assert sort_xfms.inputs.index == order
    if template_runs:
        assert 'selected' in wf.__desc__
