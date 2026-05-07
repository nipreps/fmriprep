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
"""Tests for ExtensionDescriptor base."""

import pytest

from fmriprep.extensions.descriptor import ExtensionDescriptor


class _Good(ExtensionDescriptor):
    name = 'good'
    version = '0.1.0'
    fmriprep_compat = '>=26,<27'
    contracts = {'anat_fit'}

    def init_anat_fit_wf(self, **kwargs):
        return ('built', kwargs)


def test_subclass_missing_required_class_attrs_raises():
    """Forgetting a required class attribute is caught at subclass definition."""
    with pytest.raises(TypeError, match='must define'):

        class _Bad(ExtensionDescriptor):
            name = 'bad'
            # missing: version, fmriprep_compat, contracts


def test_get_builder_returns_method_for_claimed_hook():
    d = _Good()
    assert d.get_builder('anat_fit') == d.init_anat_fit_wf


def test_get_builder_raises_for_unclaimed_hook():
    d = _Good()
    with pytest.raises(KeyError, match='does not claim'):
        d.get_builder('bold_reg')
