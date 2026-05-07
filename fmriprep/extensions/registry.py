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
"""Extension registry: holds, validates, and activates extension descriptors.

The :class:`Registry` is fmriprep's host-side handle on what extension (if
any) is plugged in. It is consulted by the dispatch layer at workflow build
time to decide whether a given hook should route to an extension's builder
or fall through to fmriprep's default.

Lifecycle:

1. At startup, :meth:`Registry.from_entry_points` discovers installed
   descriptors via ``importlib.metadata`` and constructs a registry.
   Validation runs here — bad descriptors fail loudly with typed errors
   before any workflow code runs.
2. The constructed registry is held as a process-wide handle (set via the
   dispatch layer once it lands) and queried by :meth:`Registry.resolve_builder`
   for each hooked stage in the workflow.
3. :attr:`Registry.active` and :meth:`Registry.list_extensions` provide
   introspection for logging, error messages, and provenance metadata.

A registry is always present, even with no extensions installed — an empty
registry returns ``None`` from :meth:`resolve_builder`, which the dispatch
layer treats as "use the default builder."

Validation behaviour:

- ``fmriprep_compat`` mismatches are logged as warnings (the declaration is
  an attestation, not an enforcement gate, since extensions are
  submodule-pinned at fmriprep release time).
- Unknown hooks, missing builder methods, malformed compat specifiers, and
  duplicate names abort construction with a typed
  :class:`~fmriprep.extensions.exceptions.ExtensionError` subclass.
- Multiple installed extensions also abort: the design assumes a single
  extension per installation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from fmriprep.extensions.contracts import get_contract
from fmriprep.extensions.descriptor import ExtensionDescriptor
from fmriprep.extensions.exceptions import (
    ExtensionActivationError,
    ExtensionContractError,
    ExtensionRegistrationError,
)

ENTRY_POINT_GROUP = 'fmriprep.extensions'

_logger = logging.getLogger('fmriprep.extensions')


class Registry:
    """Holds validated extension descriptors and resolves the active one.

    Parameters
    ----------
    descriptors
        List of :class:`ExtensionDescriptor` instances. Each is validated on
        entry; any validation failure aborts construction with a typed
        exception.
    fmriprep_version
        Running fmriprep version. The descriptor's ``fmriprep_compat`` is
        evaluated against this; mismatches log a warning (treated as an
        attestation, not an enforcement gate).
    requested
        Extension name explicitly selected (e.g., from a future
        ``--extension`` CLI flag). Optional.

        - 0 descriptors → ``active`` is None regardless of ``requested``.
        - 1 descriptor → it is active; ``requested``, if set, must match.
        - >=2 descriptors → :class:`ExtensionActivationError`. The design
          assumes a single extension per installation.
    """

    def __init__(
        self,
        descriptors: list[ExtensionDescriptor],
        fmriprep_version: str,
        requested: str | None = None,
    ) -> None:
        self._fmriprep_version = Version(fmriprep_version)
        self._descriptors: dict[str, ExtensionDescriptor] = {}
        for d in descriptors:
            self._validate_and_register(d)
        self._active = self._resolve_active(requested)

    @property
    def active(self) -> ExtensionDescriptor | None:
        return self._active

    def list_extensions(self) -> list[str]:
        return sorted(self._descriptors)

    def resolve_builder(self, hook: str) -> Callable[..., Any] | None:
        """Return the active extension's builder for ``hook``, or ``None``
        if no active extension claims it."""
        if self._active is None:
            return None
        if hook not in self._active.contracts:
            return None
        return self._active.get_builder(hook)

    @classmethod
    def from_entry_points(
        cls,
        fmriprep_version: str,
        requested: str | None = None,
    ) -> Registry:
        """Discover descriptors via Python entry points under
        ``fmriprep.extensions`` and construct a Registry.
        """
        descriptors: list[ExtensionDescriptor] = []
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                klass = ep.load()
            except Exception as exc:
                raise ExtensionRegistrationError(
                    ep.name, f'failed to load entry point: {exc}'
                ) from exc
            try:
                descriptors.append(klass())
            except Exception as exc:
                raise ExtensionRegistrationError(
                    ep.name, f'failed to instantiate descriptor: {exc}'
                ) from exc
        return cls(
            descriptors=descriptors,
            fmriprep_version=fmriprep_version,
            requested=requested,
        )

    def _validate_and_register(self, d: ExtensionDescriptor) -> None:
        if d.name in self._descriptors:
            raise ExtensionRegistrationError(d.name, 'extension name registered more than once')

        try:
            spec = SpecifierSet(d.fmriprep_compat)
        except InvalidSpecifier as exc:
            raise ExtensionRegistrationError(
                d.name,
                f'invalid fmriprep_compat {d.fmriprep_compat!r}: {exc}',
            ) from exc

        if self._fmriprep_version not in spec:
            _logger.warning(
                'extension %r declares fmriprep_compat=%r but running '
                'fmriprep is %s; behavior outside the declared range is '
                'unsupported -- verify extension/fmriprep release pinning',
                d.name,
                d.fmriprep_compat,
                self._fmriprep_version,
            )

        for hook in d.contracts:
            # Verify contract exists or fail fast
            get_contract(hook)
            method_name = f'init_{hook}_wf'
            if not callable(getattr(d, method_name, None)):
                raise ExtensionContractError(
                    d.name,
                    f'missing builder method {method_name!r} for hook {hook!r}',
                )

        self._descriptors[d.name] = d

    def _resolve_active(self, requested: str | None) -> ExtensionDescriptor | None:
        n = len(self._descriptors)
        if not n:
            return None
        if n == 1:
            (only,) = self._descriptors.values()
            if requested is not None and requested != only.name:
                raise ExtensionActivationError(
                    requested,
                    f'requested extension {requested!r} not installed; installed: {only.name!r}',
                )
            return only
        # n >= 2: multi-install is a packaging error
        raise ExtensionActivationError(
            '<multiple>',
            f'multiple extensions installed ({sorted(self._descriptors)}); '
            'only a single extension per installation is currently supported',
        )
