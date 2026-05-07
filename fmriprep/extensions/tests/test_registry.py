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
"""Tests for the extension Registry.

``DummyExtension`` is defined here at module scope so other framework tests
(dispatch, integration, entry-point discovery) can import it without a
parallel fixtures module.
"""

from __future__ import annotations

import logging

import pytest

from fmriprep.extensions.descriptor import ExtensionDescriptor
from fmriprep.extensions.exceptions import (
    ExtensionActivationError,
    ExtensionContractError,
    ExtensionRegistrationError,
)
from fmriprep.extensions.registry import Registry


class DummyExtension(ExtensionDescriptor):
    """Minimal valid descriptor claiming the ``anat_fit`` hook.

    Records each call into ``self.calls`` so dispatch tests can assert
    forwarded kwargs.
    """

    name = 'dummy'
    version = '0.0.1'
    fmriprep_compat = '>=0'
    contracts = {'anat_fit'}

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def init_anat_fit_wf(self, **kwargs):
        self.calls.append(('anat_fit', kwargs))
        return f'dummy::anat_fit_wf({sorted(kwargs)})'


def test_empty_registry():
    reg = Registry(descriptors=[], fmriprep_version='26.0.0')
    assert reg.active is None
    assert reg.list_extensions() == []
    assert reg.resolve_builder('anat_fit') is None


def test_single_extension_valid():
    d = DummyExtension()
    reg = Registry(descriptors=[d], fmriprep_version='26.0.0')
    assert reg.active is d
    assert reg.list_extensions() == ['dummy']
    assert reg.resolve_builder('anat_fit') == d.init_anat_fit_wf


@pytest.mark.parametrize(
    ('spec', 'running_version', 'expect_warning', 'raises'),
    [
        ('>=26,<27', '26.1.0', False, None),
        ('<0', '26.0.0', True, None),
        ('>=99', '26.0.0', True, None),
        ('not-a-spec', '26.0.0', False, ExtensionRegistrationError),
    ],
    ids=['in_range', 'past_range_warns', 'future_range_warns', 'unparseable'],
)
def test_compat(caplog, spec, running_version, expect_warning, raises):
    """fmriprep_compat: parse-fail raises, runtime mismatch warns, valid range is silent."""

    class Ext(ExtensionDescriptor):
        name = 'compat_test'
        version = '0.0.1'
        fmriprep_compat = spec
        contracts = {'anat_fit'}

        def init_anat_fit_wf(self, **kwargs):  # pragma: no cover
            pass

    if raises is not None:
        with pytest.raises(raises, match='fmriprep_compat'):
            Registry(descriptors=[Ext()], fmriprep_version=running_version)
        return

    with caplog.at_level(logging.WARNING, logger='fmriprep.extensions'):
        reg = Registry(descriptors=[Ext()], fmriprep_version=running_version)
    assert reg.active is not None
    warned = any('compat_test' in rec.message for rec in caplog.records)
    assert warned is expect_warning


def test_unknown_hook():
    class UnknownHook(ExtensionDescriptor):
        name = 'unknown_hook'
        version = '0.0.1'
        fmriprep_compat = '>=0'
        contracts = {'not_a_real_hook'}

        def init_not_a_real_hook_wf(self, **kwargs):  # pragma: no cover
            pass

    with pytest.raises(ExtensionContractError, match='unknown hook'):
        Registry(descriptors=[UnknownHook()], fmriprep_version='26.0.0')


def test_missing_builder():
    class MissingBuilder(ExtensionDescriptor):
        name = 'no_builder'
        version = '0.0.1'
        fmriprep_compat = '>=0'
        contracts = {'anat_fit'}
        # no init_anat_fit_wf method

    with pytest.raises(ExtensionContractError, match='init_anat_fit_wf'):
        Registry(descriptors=[MissingBuilder()], fmriprep_version='26.0.0')


def test_multiple_extensions():
    class Second(ExtensionDescriptor):
        name = 'second'
        version = '0.0.1'
        fmriprep_compat = '>=0'
        contracts = {'anat_fit'}

        def init_anat_fit_wf(self, **kwargs):  # pragma: no cover
            pass

    with pytest.raises(ExtensionActivationError, match='multiple'):
        Registry(
            descriptors=[DummyExtension(), Second()],
            fmriprep_version='26.0.0',
        )


def test_missing_requested_extension():
    with pytest.raises(ExtensionActivationError, match='not installed'):
        Registry(
            descriptors=[DummyExtension()],
            fmriprep_version='26.0.0',
            requested='nibabies',
        )


def test_builder_unclaimed_hook():
    d = DummyExtension()
    reg = Registry(descriptors=[d], fmriprep_version='26.0.0')
    assert reg.resolve_builder('bold_reg') is None
