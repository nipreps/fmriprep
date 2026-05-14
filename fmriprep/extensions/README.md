# fMRIPrep extension framework

This subpackage is fmriprep's plugin surface. Extensions are independently-released Python packages that specialize fmriprep for populations or method variants (e.g., `nibabies` for infants) without forking the whole pipeline.

## Layout

```
fmriprep/extensions/
├── __init__.py       ← public re-exports (exception types)
├── exceptions.py     ← typed errors raised by the framework
├── descriptor.py     ← ExtensionDescriptor base class
├── registry.py       ← discovers and validates installed extensions
├── dispatch.py       ← routes hook calls to the active extension or default
└── contracts/
    ├── __init__.py   ← Contract, ContractField, catalog, lookups
    └── anat_fit.py   ← AnatFitContract
```

## Concepts

**Descriptor** (`descriptor.py`) — the single class an extension exposes. It declares identity (`name`, `version`), fmriprep compatibility (`fmriprep_compat`), the hook names it implements (`contracts`), and the methods that produce each workflow (`init_<hook>_wf`). It also holds lifecycle hooks (`cli_extend`, `cli_populate`, `init_config`) and owns the extension's config namespace via `self.get()`/`self.set()`.

**Contract** (`contracts/`) — the schema for a workflow-builder hook: the declared inputnode and outputnode fields. Contracts are runtime-introspectable so fmriprep can verify that an extension's produced workflow satisfies the expected interface. Construction-time kwargs (e.g. "is T2w available?") are outside the contract and passed through freely.

**Registry** (`registry.py`) — discovers extensions via `importlib.metadata.entry_points(group='fmriprep.extensions')`, validates each descriptor (compat check, required attributes, builder presence), and resolves which one is active. Only one extension per installation is supported; more than one raises `ExtensionActivationError`.

**Dispatch** (`dispatch.py`) — the call-site interface for workflow-builder hooks. `build(hook, default_builder, **kwargs)` routes to the active extension's builder if one claims the hook, otherwise calls `default_builder`. The default is supplied at the call site, not registered globally, so fmriprep's native workflows remain the explicit fallback.

## Startup lifecycle

| Step | Call | What happens |
|------|------|-------------|
| 1 | `config.extensions.init()` | Registry boot — discovers extensions, sets `config.extensions.active` |
| 2 | `_build_parser()` | `active.cli_extend(parser)` appends extension flags to fmriprep's argparser |
| 3 | `parse_args()` | argparse runs; `active.cli_populate(opts)` writes parsed values into `config.extensions.<name>` |
| 4 | `config.extensions.configure()` | `active.init_config()` applies static config defaults then derives dynamic values |
| 5 | `config.to_filename()` | Extension namespace serialized into run-state TOML; survives main→subprocess boundary |
| 6 | Workflow build | `dispatch.build('anat_fit', default, **kwargs)` routes to `active.init_anat_fit_wf` |

## Writing an extension

### 1. Subclass `ExtensionDescriptor`

```python
# nibabies/fmriprep_ext.py
from fmriprep.extensions.descriptor import ExtensionDescriptor
from fmriprep.extensions.exceptions import ExtensionConfigError


class NibabiesExtension(ExtensionDescriptor):
    name = 'nibabies'
    version = '26.0.0'
    fmriprep_compat = '>=26,<27'
    contracts = {'anat_fit'}

    # Optional — routes Sentry and Migas through nibabies' own identifiers.
    # Omit to fall back to fmriprep's telemetry.
    telemetry = {
        'sentry_dsn': 'https://...@sentry.io/...',
        'migas_project': 'nipreps/nibabies',
    }

    def cli_extend(self, parser):
        group = parser.add_argument_group('NiBabies options')
        group.add_argument('--nibabies-age-months', type=int, dest='nibabies_age_months')

    def cli_populate(self, opts):
        self.set('age_months', opts.nibabies_age_months)

    def config_extend(self):
        return {'workflow.run_reconall': False}

    def init_config(self):
        super().init_config()  # applies config_extend() defaults
        if self.get('age_months') is None:
            raise ExtensionConfigError('nibabies', 'pass --nibabies-age-months')

    def init_anat_fit_wf(self, **kwargs):
        from nibabies.workflows.anatomical import init_infant_anat_fit_wf
        return init_infant_anat_fit_wf(age_months=self.get('age_months'), **kwargs)
```

| Method | When called | Purpose |
|--------|-------------|---------|
| `cli_extend(parser)` | Before argparse | Add extension flags to fmriprep's parser in-place |
| `cli_populate(opts)` | After argparse | Write parsed values into the extension namespace via `self.set()` |
| `config_extend()` | Via `init_config` | Return `{'section.field': value}` defaults (applied only where user left the field unset) |
| `init_config()` | After CLI parse | Call `super()`, then derive dynamic values; raise `ExtensionConfigError` if required input is missing |
| `init_<hook>_wf(**kwargs)` | Workflow build | Return the replacement sub-workflow |

Read extension-owned values anywhere via `config.extensions.active.get('key')` or `self.get('key')` inside descriptor methods.

### 2. Declare the entry point

```toml
# pyproject.toml
[project.entry-points."fmriprep.extensions"]
nibabies = "nibabies.fmriprep_ext:NibabiesExtension"
```

### 3. Distribute as a container

```dockerfile
FROM nipreps/fmriprep:26.0.0
COPY . /opt/nibabies
RUN pip install /opt/nibabies
```

Stock `fmriprep` containers carry no extensions. A nibabies user pulls `nipreps/nibabies:26.0.0`; the same fmriprep CLI is the entry point.

## Stability promise

fmriprep follows **CalVer**: major = release year, minor = API-breaking changes, patch = bug fixes. Contract changes land in minor bumps; patch releases preserve all contracts.

The `fmriprep_compat` field is an attestation by the extension author, not a runtime gate. A version mismatch logs a warning; build pinning (extensions tracked as submodules) is the real protection.

## Errors

| Exception | When it fires |
|-----------|---------------|
| `ExtensionRegistrationError` | Descriptor malformed or fails validation |
| `ExtensionContractError` | Extension claims an unknown hook or its workflow doesn't satisfy the contract |
| `ExtensionActivationError` | More than one extension installed (only one supported per installation) |
| `ExtensionConfigError` | `init_config` cannot compute a required value (e.g. a required path is missing) |
