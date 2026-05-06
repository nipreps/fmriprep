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
"""Tests for contract type system and the anat_fit contract."""

import pytest

from fmriprep.extensions.contracts import AnatFitContract, get_contract
from fmriprep.extensions.exceptions import ExtensionContractError


def test_anat_fit_outputs_cover_downstream_consumed_names():
    """Downstream fmriprep wiring depends on these output names existing in the
    contract schema. Population at runtime is implementation- and config-dependent."""
    out_names = {f.name for f in AnatFitContract.outputs}
    assert {
        't1w_preproc',
        't1w_mask',
        't1w_dseg',
        't1w_tpms',
        'anat2std_xfm',
        'std2anat_xfm',
        'template',
        'cortex_mask',
        'fsnative2t1w_xfm',
        'white',
        'pial',
        'thickness',
    } <= out_names


def test_get_contract_returns_known_contract():
    assert get_contract('anat_fit') is AnatFitContract


def test_get_contract_unknown_hook_raises():
    with pytest.raises(ExtensionContractError, match='unknown hook'):
        get_contract('not_a_hook')
