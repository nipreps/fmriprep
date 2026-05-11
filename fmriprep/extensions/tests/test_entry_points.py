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
"""Smoke tests for Registry.from_entry_points entry-point discovery."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fmriprep.extensions.exceptions import ExtensionRegistrationError
from fmriprep.extensions.registry import ENTRY_POINT_GROUP, Registry
from fmriprep.extensions.tests.test_registry import DummyExtension


def _fake_ep(name: str, load_result=None, load_exc: Exception | None = None):
    """Build a minimal fake entry-point object."""

    def _load():
        if load_exc is not None:
            raise load_exc
        return load_result

    return SimpleNamespace(name=name, load=_load)


def test_no_entry_points_gives_empty_registry():
    """With no extensions installed the registry is empty and active is None."""
    with patch('fmriprep.extensions.registry.entry_points', return_value=[]):
        reg = Registry.from_entry_points(fmriprep_version='26.0.0')
    assert reg.active is None
    assert reg.list_extensions() == []


def test_valid_entry_point_activates_extension():
    """A loadable, instantiable descriptor is discovered and activated."""
    ep = _fake_ep('dummy', load_result=DummyExtension)
    with patch('fmriprep.extensions.registry.entry_points', return_value=[ep]):
        reg = Registry.from_entry_points(fmriprep_version='26.0.0')
    assert reg.active is not None
    assert reg.active.name == 'dummy'


def test_entry_point_load_failure_raises():
    """An entry point that cannot be imported raises ExtensionRegistrationError."""
    ep = _fake_ep('bad_ep', load_exc=ImportError('missing module'))
    with patch('fmriprep.extensions.registry.entry_points', return_value=[ep]):
        with pytest.raises(ExtensionRegistrationError, match='bad_ep'):
            Registry.from_entry_points(fmriprep_version='26.0.0')


def test_entry_point_instantiation_failure_raises():
    """A descriptor class that raises on __init__ produces ExtensionRegistrationError."""

    class BrokenInit(DummyExtension):
        def __init__(self):
            raise RuntimeError('constructor exploded')

    ep = _fake_ep('broken', load_result=BrokenInit)
    with patch('fmriprep.extensions.registry.entry_points', return_value=[ep]):
        with pytest.raises(ExtensionRegistrationError, match='broken'):
            Registry.from_entry_points(fmriprep_version='26.0.0')


def test_from_entry_points_passes_group_name():
    """entry_points is called with the published ENTRY_POINT_GROUP constant."""
    captured = {}

    def fake_ep(**kwargs):
        captured.update(kwargs)
        return []

    with patch('fmriprep.extensions.registry.entry_points', side_effect=fake_ep):
        Registry.from_entry_points(fmriprep_version='26.0.0')

    assert captured.get('group') == ENTRY_POINT_GROUP
