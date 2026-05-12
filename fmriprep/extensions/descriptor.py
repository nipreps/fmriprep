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
"""Base class for extension descriptors.

An extension descriptor is the integration surface an extension exposes to
fmriprep. It declares identity and compatibility metadata, lists the hook
names the extension implements, and provides hook builder methods named
``init_<hook>_wf``.
"""

from __future__ import annotations

from typing import Any

_REQUIRED_CLASS_ATTRS = ('name', 'version', 'fmriprep_compat', 'contracts')


class ExtensionDescriptor:
    """Base class for extension descriptors.

    Subclasses must declare:

    - ``name`` (str): unique extension identifier.
    - ``version`` (str): extension package version.
    - ``fmriprep_compat`` (str): PEP 440 version specifier for compatible
      fmriprep versions, e.g. ``'>=26,<27'``.
    - ``contracts`` (set[str]): hook names the extension implements.

    Subclasses may optionally declare:

    - ``telemetry`` (dict | None): telemetry routing overrides. Recognised keys:

      - ``migas_project`` (str) — migas project slug, e.g. ``'nipreps/nibabies'``.
      - ``sentry_dsn`` (str) — Sentry DSN for this extension.

      When ``None`` (the default), fmriprep's own telemetry identifiers are used.

    For each name in ``contracts``, the subclass must provide a method
    ``init_<hook>_wf`` whose signature is compatible with the corresponding
    hook contract. Builder-method presence is verified at registration
    time (by the registry) rather than at subclass definition.
    """

    name: str
    version: str
    fmriprep_compat: str
    contracts: set[str]
    telemetry: dict | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        missing = [a for a in _REQUIRED_CLASS_ATTRS if not hasattr(cls, a)]
        if missing:
            raise TypeError(
                f'ExtensionDescriptor subclass {cls.__name__!r} must define: {missing}'
            )

    def __repr__(self) -> str:
        return f'<ExtensionDescriptor name={self.name!r} version={self.version!r}>'

    def get_builder(self, hook: str) -> Any:
        """Return the builder method for ``hook``.

        The result is a fresh bound method; identity (``is``) is not stable
        across calls, so consumers comparing builders should use ``==``.
        Raises :class:`KeyError` if the descriptor does not claim this hook.
        """
        if hook not in self.contracts:
            raise KeyError(f'extension {self.name!r} does not claim hook {hook!r}')
        return getattr(self, f'init_{hook}_wf')
