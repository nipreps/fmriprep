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
"""Contract type system for extension hooks.

Each contract describes the inputnode/outputnode schema of a hookable
workflow factory. Contracts are runtime-introspectable so fmriprep can
validate extension implementations against them at registration and
connect time.

Adding a new contract:
1. Create a new module in this package (e.g. ``bold_fit.py``) defining a
   ``Contract`` subclass with ``name``, ``inputs``, ``outputs``.
2. Import it below and add an entry to ``_HOOK_CONTRACTS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fmriprep.extensions.exceptions import ExtensionContractError


@dataclass(frozen=True)
class ContractField:
    """One field in a contract's input or output schema.

    Whether a field is populated at runtime is governed by the implementation
    and upstream config flags (e.g., ``config.workflow.run_reconall``);
    nipype's ``Undefined`` machinery handles the unset case. The contract
    only declares which fields exist in the schema.
    """

    name: str
    type_hint: Any
    description: str = ''


class Contract:
    """Base for all hook contracts.

    ``inputs`` and ``outputs`` describe the **inputnode and outputnode**
    fields of the workflow the hook produces — the runtime data flow surface
    that fmriprep wires upstream and downstream nodes against. They do *not*
    enumerate constructor kwargs; builders are free to take additional
    construction-time arguments (typically used for build-time shape
    decisions, like whether a contrast is present) without those appearing
    in the contract.

    Contract evolution tracks fmriprep's release cycle. Extensions declare
    compatibility via ``fmriprep_compat`` (PEP 440); breaking contract
    changes coincide with major fmriprep bumps. Additive changes (new
    optional output) are safe within a minor and are handled by capability
    negotiation in downstream wiring.

    ``validated_through`` is a maintainer attestation: it maps upstream
    package name to the version against which the contract's input/output
    schema was last confirmed. The contract is *guaranteed* to match that
    version; later versions may still match (and usually do) but have not
    been re-verified. Informational metadata only — does not gate execution.
    """

    name: str
    inputs: list[ContractField]
    outputs: list[ContractField]
    validated_through: dict[str, str] = {}


# Contract modules. Imported after the base types are defined so they can
# import ``Contract`` / ``ContractField`` from this package without circular
# resolution issues.
from fmriprep.extensions.contracts.anat_fit import AnatFitContract  # noqa: E402

_HOOK_CONTRACTS: dict[str, type[Contract]] = {
    AnatFitContract.name: AnatFitContract,
}


def get_contract(hook: str) -> type[Contract]:
    """Look up a contract class by hook name.

    Raises :class:`ExtensionContractError` if the hook is unknown. The
    ``'<unknown>'`` sentinel for ``extension_name`` is internal: callers
    (the registry) catch and re-raise with the actual descriptor name
    before the error surfaces.
    """
    if hook in _HOOK_CONTRACTS:
        return _HOOK_CONTRACTS[hook]
    raise ExtensionContractError(
        '<unknown>',
        f'unknown hook {hook!r}; known hooks: {sorted(_HOOK_CONTRACTS)}',
    )


def list_contracts() -> list[str]:
    """Return all registered hook names."""
    return sorted(_HOOK_CONTRACTS)


__all__ = [
    'AnatFitContract',
    'Contract',
    'ContractField',
    'get_contract',
    'list_contracts',
]
