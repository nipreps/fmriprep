# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
A Python module to maintain unique, run-wide *fMRIPrep* settings.

This module implements the memory structures to keep a consistent, singleton config.
Settings are passed across processes via filesystem, and a copy of the settings for
each run and subject is left under
``<output_dir>/sub-<participant_id>/log/<run_unique_id>/fmriprep.toml``.
Settings are stored using :abbr:`ToML (Tom's Markup Language)`.
The module has a :py:func:`~fmriprep.config.to_filename` function to allow writting out
the settings to hard disk in *ToML* format, which looks like:

.. literalinclude:: ../fmriprep/data/tests/config.toml
   :language: toml
   :name: fmriprep.toml
   :caption: **Example file representation of fMRIPrep settings**.

This config file is used to pass the settings across processes,
using the :py:func:`~fmriprep.config.load` function.

Configuration sections
----------------------
.. autoclass:: environment
   :members:
.. autoclass:: execution
   :members:
.. autoclass:: workflow
   :members:
.. autoclass:: nipype
   :members:

Usage
-----
A config file is used to pass settings and collect information as the execution
graph is built across processes.

.. code-block:: Python

    from fmriprep import config
    config_file = config.execution.work_dir / '.fmriprep.toml'
    config.to_filename(config_file)
    # Call build_workflow(config_file, retval) in a subprocess
    with Manager() as mgr:
        from .workflow import build_workflow
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(str(config_file), retval))
        p.start()
        p.join()
    config.load(config_file)
    # Access configs from any code section as:
    value = config.section.setting

Logging
-------
.. autoclass:: loggers
   :members:

Other responsibilities
----------------------
The :py:mod:`config` is responsible for other conveniency actions.

  * Switching Python's :obj:`multiprocessing` to *forkserver* mode.
  * Set up a filter for warnings as early as possible.
  * Automated I/O magic operations. Some conversions need to happen in the
    store/load processes (e.g., from/to :obj:`~pathlib.Path` \<-\> :obj:`str`,
    :py:class:`~bids.layout.BIDSLayout`, etc.)

"""
from multiprocessing import set_start_method
import warnings

# cmp is not used by fmriprep, so ignore nipype-generated warnings
warnings.filterwarnings('ignore', 'cmp not installed')
warnings.filterwarnings('ignore', 'This has not been fully tested. Please report any failures.')
warnings.filterwarnings('ignore', "sklearn.externals.joblib is deprecated in 0.21")
warnings.filterwarnings('ignore', "can't resolve package from __spec__ or __package__")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ResourceWarning)


try:
    set_start_method('forkserver')
except RuntimeError:
    pass  # context has been already set
finally:
    # Defer all custom import for after initializing the forkserver and
    # ignoring the most annoying warnings
    import os
    import sys
    import logging

    from uuid import uuid4
    from pathlib import Path
    from time import strftime
    from niworkflows.utils.spaces import SpatialReferences as _SRs, Reference as _Ref
    from nipype import logging as nlogging, __version__ as _nipype_ver
    from templateflow import __version__ as _tf_ver
    from . import __version__


def redirect_warnings(message, category, filename, lineno, file=None, line=None):
    """Redirect other warnings."""
    logger = logging.getLogger()
    logger.debug('Captured warning (%s): %s', category, message)


warnings.showwarning = redirect_warnings

logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG

DEFAULT_MEMORY_MIN_GB = 0.01

_exec_env = os.name
_docker_ver = None
# special variable set in the container
if os.getenv('IS_DOCKER_8395080871'):
    _exec_env = 'singularity'
    _cgroup = Path('/proc/1/cgroup')
    if _cgroup.exists() and 'docker' in _cgroup.read_text():
        _docker_ver = os.getenv('DOCKER_VERSION_8395080871')
        _exec_env = 'fmriprep-docker' if _docker_ver else 'docker'
    del _cgroup

_fs_license = os.getenv('FS_LICENSE')
if _fs_license is None and os.getenv('FREESURFER_HOME'):
    _fs_license = os.path.join(os.getenv('FREESURFER_HOME'), 'license.txt')

_templateflow_home = Path(os.getenv(
    'TEMPLATEFLOW_HOME',
    os.path.join(os.getenv('HOME'), '.cache', 'templateflow'))
)

try:
    from psutil import virtual_memory
    _free_mem_at_start = round(virtual_memory().free / 1024**3, 1)
except Exception:
    _free_mem_at_start = None

_oc_limit = 'n/a'
_oc_policy = 'n/a'
try:
    # Memory policy may have a large effect on types of errors experienced
    _proc_oc_path = Path('/proc/sys/vm/overcommit_memory')
    if _proc_oc_path.exists():
        _oc_policy = {
            '0': 'heuristic', '1': 'always', '2': 'never'
        }.get(_proc_oc_path.read_text().strip(), 'unknown')
        if _oc_policy != 'never':
            _proc_oc_kbytes = Path('/proc/sys/vm/overcommit_kbytes')
            if _proc_oc_kbytes.exists():
                _oc_limit = _proc_oc_kbytes.read_text().strip()
            if _oc_limit in ('0', 'n/a') and Path('/proc/sys/vm/overcommit_ratio').exists():
                _oc_limit = '{}%'.format(
                    Path('/proc/sys/vm/overcommit_ratio').read_text().strip()
                )
except Exception:
    pass


class _Config:
    """An abstract class forbidding instantiation."""

    _paths = tuple()

    def __init__(self):
        """Avert instantiation."""
        raise RuntimeError('Configuration type is not instantiable.')

    @classmethod
    def load(cls, settings, init=True):
        """Store settings from a dictionary."""
        for k, v in settings.items():
            if v is None:
                continue
            if k in cls._paths:
                setattr(cls, k, Path(v).absolute())
                continue
            if hasattr(cls, k):
                setattr(cls, k, v)

        if init:
            try:
                cls.init()
            except AttributeError:
                pass

    @classmethod
    def get(cls):
        """Return defined settings."""
        out = {}
        for k, v in cls.__dict__.items():
            if k.startswith('_') or v is None:
                continue
            if callable(getattr(cls, k)):
                continue
            if k in cls._paths:
                v = str(v)
            if isinstance(v, _SRs):
                v = ' '.join([str(s) for s in v.references]) or None
            if isinstance(v, _Ref):
                v = str(v) or None
            out[k] = v
        return out


class environment(_Config):
    """
    Read-only options regarding the platform and environment.

    Crawls runtime descriptive settings (e.g., default FreeSurfer license,
    execution environment, nipype and *fMRIPrep* versions, etc.).
    The ``environment`` section is not loaded in from file,
    only written out when settings are exported.
    This config section is useful when reporting issues,
    and these variables are tracked whenever the user does not
    opt-out using the ``--notrack`` argument.

    """

    cpu_count = os.cpu_count()
    """Number of available CPUs."""
    exec_docker_version = _docker_ver
    """Version of Docker Engine."""
    exec_env = _exec_env
    """A string representing the execution platform."""
    free_mem = _free_mem_at_start
    """Free memory at start."""
    overcommit_policy = _oc_policy
    """Linux's kernel virtual memory overcommit policy."""
    overcommit_limit = _oc_limit
    """Linux's kernel virtual memory overcommit limits."""
    nipype_version = _nipype_ver
    """Nipype's current version."""
    templateflow_version = _tf_ver
    """The TemplateFlow client version installed."""
    version = __version__
    """*fMRIPrep*'s version."""


class nipype(_Config):
    """Nipype settings."""

    crashfile_format = 'txt'
    """The file format for crashfiles, either text or pickle."""
    get_linked_libs = False
    """Run NiPype's tool to enlist linked libraries for every interface."""
    memory_gb = None
    """Estimation in GB of the RAM this workflow can allocate at any given time."""
    nprocs = os.cpu_count()
    """Number of processes (compute tasks) that can be run in parallel (multiprocessing only)."""
    omp_nthreads = None
    """Number of CPUs a single process can access for multithreaded execution."""
    plugin = 'MultiProc'
    """NiPype's execution plugin."""
    plugin_args = {
        'maxtasksperchild': 1,
        'raise_insufficient': False,
    }
    """Settings for NiPype's execution plugin."""
    resource_monitor = False
    """Enable resource monitor."""
    stop_on_first_crash = True
    """Whether the workflow should stop or continue after the first error."""

    @classmethod
    def get_plugin(cls):
        """Format a dictionary for Nipype consumption."""
        out = {
            'plugin': cls.plugin,
            'plugin_args': cls.plugin_args,
        }
        if cls.plugin in ('MultiProc', 'LegacyMultiProc'):
            out['plugin_args']['nprocs'] = int(cls.nprocs)
            if cls.memory_gb:
                out['plugin_args']['memory_gb'] = float(cls.memory_gb)
        return out

    @classmethod
    def init(cls):
        """Set NiPype configurations."""
        from nipype import config as ncfg

        # Configure resource_monitor
        if cls.resource_monitor:
            ncfg.update_config({
                'monitoring': {
                    'enabled': cls.resource_monitor,
                    'sample_frequency': '0.5',
                    'summary_append': True,
                }
            })
            ncfg.enable_resource_monitor()

        # Nipype config (logs and execution)
        ncfg.update_config({
            'execution': {
                'crashdump_dir': str(execution.log_dir),
                'crashfile_format': cls.crashfile_format,
                'get_linked_libs': cls.get_linked_libs,
                'stop_on_first_crash': cls.stop_on_first_crash,
            }
        })

        if cls.omp_nthreads is None:
            cls.omp_nthreads = min(cls.nprocs - 1 if cls.nprocs > 1 else os.cpu_count(), 8)


class execution(_Config):
    """Configure run-level settings."""

    bids_dir = None
    """An existing path to the dataset, which must be BIDS-compliant."""
    bids_description_hash = None
    """Checksum (SHA256) of the ``dataset_description.json`` of the BIDS dataset."""
    bids_filters = None
    """A dictionary of BIDS selection filters."""
    boilerplate_only = False
    """Only generate a boilerplate."""
    debug = False
    """Run in sloppy mode (meaning, suboptimal parameters that minimize run-time)."""
    echo_idx = None
    """Select a particular echo for multi-echo EPI datasets."""
    fs_license_file = _fs_license
    """An existing file containing a FreeSurfer license."""
    fs_subjects_dir = None
    """FreeSurfer's subjects directory."""
    layout = None
    """A :py:class:`~bids.layout.BIDSLayout` object, see :py:func:`init`."""
    log_dir = None
    """The path to a directory that contains execution logs."""
    log_level = 25
    """Output verbosity."""
    low_mem = None
    """Utilize uncompressed NIfTIs and other tricks to minimize memory allocation."""
    md_only_boilerplate = False
    """Do not convert boilerplate from MarkDown to LaTex and HTML."""
    notrack = False
    """Do not monitor *fMRIPrep* using Sentry.io."""
    output_dir = None
    """Folder where derivatives will be stored."""
    output_spaces = None
    """List of (non)standard spaces designated (with the ``--output-spaces`` flag of
    the command line) as spatial references for outputs."""
    reports_only = False
    """Only build the reports, based on the reportlets found in a cached working directory."""
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid4())
    """Unique identifier of this particular run."""
    participant_label = None
    """List of participant identifiers that are to be preprocessed."""
    task_id = None
    """Select a particular task from all available in the dataset."""
    templateflow_home = _templateflow_home
    """The root folder of the TemplateFlow client."""
    work_dir = Path('work').absolute()
    """Path to a working directory where intermediate results will be available."""
    write_graph = False
    """Write out the computational graph corresponding to the planned preprocessing."""

    _layout = None

    _paths = (
        'bids_dir',
        'fs_license_file',
        'fs_subjects_dir',
        'layout',
        'log_dir',
        'output_dir',
        'templateflow_home',
        'work_dir',
    )

    @classmethod
    def init(cls):
        """Create a new BIDS Layout accessible with :attr:`~execution.layout`."""
        if cls._layout is None:
            import re
            from bids.layout import BIDSLayout
            work_dir = cls.work_dir / 'bids.db'
            work_dir.mkdir(exist_ok=True, parents=True)
            cls._layout = BIDSLayout(
                str(cls.bids_dir),
                validate=False,
                # database_path=str(work_dir),
                ignore=("code", "stimuli", "sourcedata", "models",
                        "derivatives", re.compile(r'^\.')))
        cls.layout = cls._layout


# These variables are not necessary anymore
del _fs_license
del _exec_env
del _nipype_ver
del _templateflow_home
del _tf_ver
del _free_mem_at_start
del _oc_limit
del _oc_policy


class workflow(_Config):
    """Configure the particular execution graph of this workflow."""

    anat_only = False
    """Execute the anatomical preprocessing only."""
    aroma_err_on_warn = None
    """Cast AROMA warnings to errors."""
    aroma_melodic_dim = None
    """Number of ICA components to be estimated by MELODIC
    (positive = exact, negative = maximum)."""
    bold2t1w_dof = None
    """Degrees of freedom of the BOLD-to-T1w registration steps."""
    cifti_output = None
    """Generate HCP Grayordinates, accepts either ``'91k'`` (default) or ``'170k'``."""
    dummy_scans = None
    """Set a number of initial scans to be considered nonsteady states."""
    fmap_bspline = None
    """Regularize fieldmaps with a field of B-Spline basis."""
    fmap_demean = None
    """Remove the mean from fieldmaps."""
    force_syn = None
    """Run *fieldmap-less* susceptibility-derived distortions estimation."""
    hires = None
    """Run FreeSurfer ``recon-all`` with the ``-hires`` flag."""
    ignore = None
    """Ignore particular steps for *fMRIPrep*."""
    longitudinal = False
    """Run FreeSurfer ``recon-all`` with the ``-logitudinal`` flag."""
    medial_surface_nan = None
    """Fill medial surface with :abbr:`NaNs (not-a-number)` when sampling."""
    regressors_all_comps = None
    """Return all CompCor components."""
    regressors_dvars_th = None
    """Threshold for DVARS."""
    regressors_fd_th = None
    """Threshold for :abbr:`FD (frame-wise displacement)`."""
    run_reconall = True
    """Run FreeSurfer's surface reconstruction."""
    skull_strip_fixed_seed = False
    """Fix a seed for skull-stripping."""
    skull_strip_template = "OASIS30ANTs"
    """Change default brain extraction template."""
    skull_strip_t1w = "force"
    """Skip brain extraction of the T1w image (default is ``force``, meaning that
    *fMRIPrep* will run brain extraction of the T1w)."""
    spaces = None
    """Keeps the :py:class:`~niworkflows.utils.spaces.SpatialReferences`
    instance keeping standard and nonstandard spaces."""
    t2s_coreg = None
    """Co-register echos before generating the T2\\* reference of
    :abbr:`ME-EPI (multi-echo echo-planar imaging)`."""
    use_aroma = None
    """Run ICA-:abbr:`AROMA (automatic removal of motion artifacts)`."""
    use_bbr = None
    """Run boundary-based registration for BOLD-to-T1w registration."""
    use_syn = None
    """Run *fieldmap-less* susceptibility-derived distortions estimation
    in the absence of any alternatives."""


class loggers:
    """Keep loggers easily accessible (see :py:func:`init`)."""

    _fmt = "%(asctime)s,%(msecs)d %(name)-2s " "%(levelname)-2s:\n\t %(message)s"
    _datefmt = "%y%m%d-%H:%M:%S"

    default = logging.getLogger()
    """The root logger."""
    cli = logging.getLogger('cli')
    """Command-line interface logging."""
    workflow = nlogging.getLogger('nipype.workflow')
    """NiPype's workflow logger."""
    interface = nlogging.getLogger('nipype.interface')
    """NiPype's interface logger."""
    utils = nlogging.getLogger('nipype.utils')
    """NiPype's utils logger."""

    @classmethod
    def init(cls):
        """
        Set the log level, initialize all loggers into :py:class:`loggers`.

            * Add new logger levels (25: IMPORTANT, and 15: VERBOSE).
            * Add a new sub-logger (``cli``).
            * Logger configuration.

        """
        from nipype import config as ncfg
        _handler = logging.StreamHandler(stream=sys.stdout)
        _handler.setFormatter(
            logging.Formatter(fmt=cls._fmt, datefmt=cls._datefmt)
        )
        cls.cli.addHandler(_handler)
        cls.default.setLevel(execution.log_level)
        cls.cli.setLevel(execution.log_level)
        cls.interface.setLevel(execution.log_level)
        cls.workflow.setLevel(execution.log_level)
        cls.utils.setLevel(execution.log_level)
        ncfg.update_config({
            'logging': {
                'log_directory': str(execution.log_dir),
                'log_to_file': True
            },
        })


def from_dict(settings):
    """Read settings from a flat dictionary."""
    nipype.load(settings)
    execution.load(settings)
    workflow.load(settings)
    loggers.init()


def load(filename):
    """Load settings from file."""
    from toml import loads
    filename = Path(filename)
    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != 'environment':
            section = getattr(sys.modules[__name__], sectionname)
            section.load(configs)
    init_spaces()


def get(flat=False):
    """Get config as a dict."""
    settings = {
        'environment': environment.get(),
        'execution': execution.get(),
        'workflow': workflow.get(),
        'nipype': nipype.get(),
    }
    if not flat:
        return settings

    return {'.'.join((section, k)): v
            for section, configs in settings.items()
            for k, v in configs.items()}


def dumps():
    """Format config into toml."""
    from toml import dumps
    return dumps(get())


def to_filename(filename):
    """Write settings to file."""
    filename = Path(filename)
    filename.write_text(dumps())


def init_spaces(checkpoint=True):
    """Initialize the :attr:`~workflow.spaces` setting."""
    from niworkflows.utils.spaces import Reference, SpatialReferences
    spaces = execution.output_spaces or SpatialReferences()
    if not isinstance(spaces, SpatialReferences):
        spaces = SpatialReferences(
            [ref for s in spaces.split(' ')
             for ref in Reference.from_string(s)]
        )

    if checkpoint and not spaces.is_cached():
        spaces.checkpoint()

    # Add the default standard space if not already present (required by several sub-workflows)
    if "MNI152NLin2009cAsym" not in spaces.get_spaces(nonstandard=False, dim=(3,)):
        spaces.add(
            Reference("MNI152NLin2009cAsym", {})
        )

    # Ensure user-defined spatial references for outputs are correctly parsed.
    # Certain options require normalization to a space not explicitly defined by users.
    # These spaces will not be included in the final outputs.
    if workflow.use_aroma:
        # Make sure there's a normalization to FSL for AROMA to use.
        spaces.add(
            Reference("MNI152NLin6Asym", {"res": "2"})
        )

    cifti_output = workflow.cifti_output
    if cifti_output:
        # CIFTI grayordinates to corresponding FSL-MNI resolutions.
        vol_res = '2' if cifti_output == '91k' else '1'
        spaces.add(
            Reference("fsaverage", {"den": "164k"})
        )
        spaces.add(
            Reference("MNI152NLin6Asym", {"res": vol_res})
        )

    # Make the SpatialReferences object available
    workflow.spaces = spaces
