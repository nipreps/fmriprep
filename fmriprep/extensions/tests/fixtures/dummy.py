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
"""Dummy extension descriptors used by the framework test suite."""

from __future__ import annotations

from fmriprep.extensions.descriptor import ExtensionDescriptor


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


class SecondExtension(ExtensionDescriptor):
    """A second valid descriptor; used to exercise multi-install rejection."""

    name = 'second'
    version = '0.0.1'
    fmriprep_compat = '>=0'
    contracts = {'anat_fit'}

    def init_anat_fit_wf(self, **kwargs):  # pragma: no cover - never invoked
        return 'second::anat_fit_wf'


class IncompatibleExtension(ExtensionDescriptor):
    """Descriptor whose ``fmriprep_compat`` can never be satisfied.

    Used to exercise the registry's compat-mismatch warning path. The
    framework warns rather than raises on mismatch, so this descriptor
    still registers successfully.
    """

    name = 'incompat'
    version = '0.0.1'
    fmriprep_compat = '<0'
    contracts = {'anat_fit'}

    def init_anat_fit_wf(self, **kwargs):  # pragma: no cover
        return None


class UnknownHookExtension(ExtensionDescriptor):
    """Descriptor claiming a hook absent from the contract catalog."""

    name = 'unknown_hook'
    version = '0.0.1'
    fmriprep_compat = '>=0'
    contracts = {'not_a_real_hook'}

    def init_not_a_real_hook_wf(self, **kwargs):  # pragma: no cover
        return None
