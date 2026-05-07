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
"""Tests for the dispatch layer."""

from __future__ import annotations

import pytest

from fmriprep.extensions import dispatch
from fmriprep.extensions.registry import Registry
from fmriprep.extensions.tests.test_registry import DummyExtension


@pytest.fixture(autouse=True)
def _clear_registry():
    dispatch.set_registry(None)
    yield
    dispatch.set_registry(None)


def test_default_used_when_no_registry():
    sentinel = object()
    assert dispatch.build('anat_fit', lambda **kw: sentinel) is sentinel


def test_default_used_when_active_does_not_claim_hook():
    """Active extension claims only 'anat_fit'; other hooks fall through to default."""
    d = DummyExtension()
    dispatch.set_registry(Registry(descriptors=[d], fmriprep_version='26.0.0'))
    sentinel = object()
    assert dispatch.build('bold_reg', lambda **kw: sentinel) is sentinel


def test_active_extension_takes_priority_and_forwards_kwargs():
    d = DummyExtension()
    dispatch.set_registry(Registry(descriptors=[d], fmriprep_version='26.0.0'))

    def default(**kw):  # pragma: no cover -- must not run
        return 'default-fired'

    result = dispatch.build('anat_fit', default, foo=1, bar=2)
    assert 'dummy::anat_fit_wf' in result
    assert d.calls == [('anat_fit', {'foo': 1, 'bar': 2})]
