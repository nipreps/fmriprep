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
"""Dispatch: routes hookable workflow construction.

The dispatch layer is the single call site fmriprep core uses to
instantiate hookable sub-workflows. It consults the active :class:`Registry`
(set once during fmriprep startup); if an extension claims the hook, its
builder is invoked. Otherwise the caller's ``default_builder`` runs. Both
receive the forwarded ``**kwargs`` verbatim.

The default builder is supplied by the call site rather than registered in
a global registry, so default ownership stays co-located with the call --
no framework-level side-effect imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fmriprep.extensions.registry import Registry

_active_registry: Registry | None = None


def set_registry(registry: Registry | None) -> None:
    """Install (or clear) the process-wide active registry."""
    global _active_registry  # noqa: PLW0603 -- intentional process-wide handle
    _active_registry = registry


def build(
    hook: str,
    default_builder: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Build a hookable workflow.

    If an active registry's extension claims ``hook``, that extension's
    builder runs; otherwise ``default_builder`` runs. Either receives the
    forwarded ``**kwargs`` verbatim.
    """
    builder: Callable[..., Any] | None = None
    if _active_registry is not None:
        builder = _active_registry.resolve_builder(hook)
    if builder is None:
        builder = default_builder
    return builder(**kwargs)
