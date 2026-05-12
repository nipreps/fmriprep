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
"""Tests for telemetry routing through active extensions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_sentry_dsn():
    import fmriprep.utils.telemetry as tel_mod

    fake_sdk = MagicMock()
    fake_sdk.configure_scope.return_value.__enter__ = MagicMock(return_value=MagicMock())
    fake_sdk.configure_scope.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch.object(tel_mod, 'sentry_sdk', fake_sdk),
        patch.object(tel_mod, 'config') as mock_cfg,
    ):
        mock_cfg.environment.version = '26.0.0'
        mock_cfg.get.return_value = {}
        tel_mod.sentry_setup(dsn='https://custom@sentry.io/999')

    assert fake_sdk.init.call_args[0][0] == 'https://custom@sentry.io/999'


def test_sentry_fallback():
    import fmriprep.utils.telemetry as tel_mod

    fake_sdk = MagicMock()
    fake_sdk.configure_scope.return_value.__enter__ = MagicMock(return_value=MagicMock())
    fake_sdk.configure_scope.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch.object(tel_mod, 'sentry_sdk', fake_sdk),
        patch.object(tel_mod, 'config') as mock_cfg,
    ):
        mock_cfg.environment.version = '26.0.0'
        mock_cfg.get.return_value = {}
        tel_mod.sentry_setup()

    dsn_used = fake_sdk.init.call_args[0][0]
    assert 'sentry.io' in dsn_used
    assert 'custom' not in dsn_used


def test_migas(monkeypatch):
    import fmriprep.utils.telemetry as tel_mod

    fake_migas = MagicMock()
    with (
        patch.object(tel_mod, 'migas', fake_migas),
        patch.object(tel_mod, 'config') as mock_cfg,
    ):
        mock_cfg.execution.run_uuid = None
        tel_mod.setup_migas(
            init_ping=False,
            exit_ping=False,
            project='nipreps/nibabies',
            version='24.1.0',
        )

    assert tel_mod._active_project == 'nipreps/nibabies'
    assert tel_mod._active_version == '24.1.0'

    monkeypatch.setattr(tel_mod, 'migas', fake_migas)
    tel_mod.send_crumb(status='R', status_desc='test')
    fake_migas.add_breadcrumb.assert_called_once_with(
        'nipreps/nibabies', '24.1.0', status='R', status_desc='test'
    )
