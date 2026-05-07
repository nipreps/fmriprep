# fMRIPrep extension framework

This subpackage is fmriprep's plugin surface. Extensions are independently-released Python packages that specialize fmriprep for populations or methods variants (e.g., `nibabies` for infants) without forking the whole pipeline.

## Layout

```
fmriprep/extensions/
├── README.md              ← you are here
├── __init__.py            ← public re-exports (exception types, etc.)
├── exceptions.py          ← typed errors raised by the framework
├── descriptor.py          ← ExtensionDescriptor base class (the surface
│                            extensions subclass)
├── contracts/             ← schema declarations for hookable workflows
│   ├── __init__.py        ← Contract + ContractField + catalog + lookups
│   └── anat_fit.py        ← AnatFitContract
├── registry.py            ← Registry: validates installed extensions and
│                            resolves the active one's builders
└── tests/                 ← framework + contract tests
```

Future modules (dispatch, default builders, container conformance checker) land alongside as the framework grows.

## Concepts

**Hook.** A named integration point — a slot fmriprep declares (e.g. `anat_fit`). Extensions claim hooks they implement.

Two flavors:

- **Workflow-builder hooks** (e.g. `anat_fit`, `bold_fit`, `bold_reg`) — points at which fmriprep delegates construction of a sub-workflow to the active extension. Each is paired with a *contract*.
- **Lifecycle hooks** (e.g. `cli_extend`, `init_config`) — callbacks fmriprep invokes at deterministic points during startup. Documented signatures, not contracts.

**Contract.** The schema for a workflow-builder hook: declared inputnode and outputnode fields. Contracts are runtime-introspectable so fmriprep can validate extension implementations against them. Implementations are free to take additional construction-time keyword arguments (used for build-time decisions like "is T2w available?") that are not part of the contract.

**Descriptor.** The single class an extension exposes. It declares identity (`name`, `version`), fmriprep compatibility (`fmriprep_compat`), and the set of hook names it implements, plus the methods that produce each workflow.

## Stability promise

fmriprep follows **CalVer**: the major version reflects the release year, not API-stability semantics. API-breaking changes (including contract changes) land in **minor** bumps; **patch** releases are bug fixes and preserve contracts.

Within a given minor:

- Patch releases (e.g., `26.1.0` → `26.1.1`) preserve all contracts. Extensions targeting `26.1.*` keep working.
- Minor bumps (e.g., `26.1.0` → `26.2.0`) may add new optional outputs *or* introduce breaking contract changes. Extensions declare the minor range they target.
- Field renames, removals, or required-promotions only happen at minor bumps.

Extensions are typically tracked as submodules of the fmriprep repo, so each fmriprep release pins compatible extension versions at build time. The `fmriprep_compat` declaration on the descriptor is an *attestation* by the extension author ("I claim this works with these fmriprep versions"), not a runtime gate. If the running fmriprep falls outside the declared range — usually a sign of a release-coordination slip — the framework logs a warning but does not refuse to start; the build pinning is the real protection.

Typical extension `fmriprep_compat` values pin to a specific minor: `'>=26.1,<26.2'`.

## Writing an extension

A minimum extension is one Python class plus an entry-point declaration.

### 1. Subclass `ExtensionDescriptor`

```python
# nibabies/fmriprep_ext.py
from fmriprep.extensions.descriptor import ExtensionDescriptor


class NibabiesExtension(ExtensionDescriptor):
    name = 'nibabies'
    version = '26.0.0'
    fmriprep_compat = '>=26,<27'
    contracts = {'anat_fit'}

    def init_anat_fit_wf(self, **kwargs):
        from nibabies.workflows.anatomical import init_infant_anat_fit_wf
        from fmriprep import config

        # Pull extension-specific config; pass through fmriprep-supplied kwargs.
        return init_infant_anat_fit_wf(
            **kwargs,
            age_months=config.extensions.nibabies.age_months,
        )
```

For each hook name in `contracts`, the descriptor must provide a method `init_<hook>_wf` with a signature compatible with the hook's contract. fmriprep forwards its kwargs verbatim to your method; you adapt and call your underlying implementation.

### 2. Declare the entry point

```toml
# pyproject.toml
[project.entry-points."fmriprep.extensions"]
nibabies = "nibabies.fmriprep_ext:NibabiesExtension"
```

fmriprep discovers extensions via `importlib.metadata.entry_points(group='fmriprep.extensions')` at startup.

### 3. Distribute as a container

The intended distribution model is one extension per container image:

```dockerfile
FROM nipreps/fmriprep:26.0.0
COPY . /opt/nibabies
RUN pip install /opt/nibabies
```

Stock `fmriprep` containers carry no extensions. A `nibabies` user pulls `nipreps/nibabies:26.0.0` (which contains fmriprep core + nibabies); the same fmriprep CLI is the entry point.

## Reading a contract

Contracts live in `fmriprep.extensions.contracts`. Each is a class describing the inputnode and outputnode of the workflow the hook produces:

```python
from fmriprep.extensions.contracts import AnatFitContract

for f in AnatFitContract.outputs:
    print(f.name, f.type_hint, f.description)
```

Whether a field is set at runtime is the implementation's choice (gated by upstream config flags like `config.workflow.run_reconall`); nipype's `Undefined` machinery handles the unset case. The contract only declares which fields exist in the schema.

`AnatFitContract.validated_through` records the upstream tool versions the schema was last verified against — informational metadata, not a constraint.

## What's currently here

- Exception types (`ExtensionError` and subclasses).
- Contract type system + the `anat_fit` contract.
- `ExtensionDescriptor` base class.
- `Registry` (discovery, validation, activation).

Dispatch, container conformance checker, and the full Tier 2 lifecycle hook surface (`cli_extend`, `config_extend`, `init_config`, `metadata_requirements`, `derivative_spec`) are landing in subsequent work.

## Errors

The framework raises typed exceptions from `fmriprep.extensions.exceptions`:

| Exception | When it fires |
|---|---|
| `ExtensionRegistrationError` | Descriptor malformed; bad metadata; duplicate name. |
| `ExtensionContractError` | Extension claims an unknown hook, or its workflow doesn't satisfy the contract schema. |
| `ExtensionActivationError` | More than one extension installed in the same environment (only one supported per installation). |

All carry the offending extension's `name`, so messages identify the source unambiguously.
