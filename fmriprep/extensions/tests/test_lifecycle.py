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
"""Tests for lifecycle hooks: config_extend and init_config."""

from fmriprep.extensions.tests.test_registry import DummyExtension


class _DefaultingExtension(DummyExtension):
    name = 'defaulting'

    def config_extend(self):
        return {'workflow.hires': False}


def test_config_extend_applies(monkeypatch):
    from fmriprep import config

    monkeypatch.setattr(config.workflow, 'hires', None)
    monkeypatch.setattr(config.extensions, 'active', _DefaultingExtension())
    config.extensions.configure()
    assert config.workflow.hires is False


def test_config_extend_preserves_user_value(monkeypatch):
    from fmriprep import config

    monkeypatch.setattr(config.workflow, 'hires', True)
    monkeypatch.setattr(config.extensions, 'active', _DefaultingExtension())
    config.extensions.configure()
    assert config.workflow.hires is True


class _DynamicExtension(DummyExtension):
    name = 'dynext'

    def init_config(self):
        super().init_config()
        age = self.get('age_months')
        self.set('atlas_label', f'infant-{age}mo')


def test_init_config_derives_value_from_namespace(monkeypatch):
    from fmriprep import config

    config.extensions._namespaces.clear()
    d = _DynamicExtension()
    monkeypatch.setattr(config.extensions, 'active', d)
    d.set('age_months', 6)
    config.extensions.configure()
    assert config.extensions.active.get('atlas_label') == 'infant-6mo'
    config.extensions._namespaces.clear()
