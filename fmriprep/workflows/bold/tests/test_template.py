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

from ..template import init_bold_template_wf


@pytest.mark.parametrize(
    ('reference', 'template_runs', 'fixed', 'registered'),
    [
        (0, None, True, False),
        (None, None, False, False),
        (2, [False, True, True], True, False),
        (None, [True, False, True], False, True),
    ],
)
def test_template_selection(reference, template_runs, fixed, registered):
    """The reference fixes the template space; excluded runs are registered to unbiased ones."""
    wf = init_bold_template_wf(num_bold_runs=3, reference=reference, template_runs=template_runs)

    robust_template = wf.get_node('boldref_template').inputs
    assert robust_template.fixed_timepoint is fixed
    assert robust_template.no_iteration is fixed
    if fixed:
        assert robust_template.initial_timepoint == reference + 1
    assert (wf.get_node('register_excluded_runs') is not None) is registered
    if template_runs and not registered:
        assert wf.get_node('resample_template').inputs.indices == template_runs
    if template_runs:
        assert 'selected' in wf.__desc__


def test_merge_xfms_order():
    """Transforms of registered runs are returned in the original run order."""
    from ..template import _interleave

    assert _interleave(['b', 'd'], ['a', 'c'], [False, True, False, True]) == ['a', 'b', 'c', 'd']
