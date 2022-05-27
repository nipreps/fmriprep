# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""Parser."""
import sys
from .. import config


def _build_parser(**kwargs):
    """Build parser object.

    ``kwargs`` are passed to ``argparse.ArgumentParser`` (mainly useful for debugging).
    """
    from functools import partial
    from pathlib import Path
    from argparse import (
        ArgumentParser,
        ArgumentDefaultsHelpFormatter,
    )
    from packaging.version import Version
    from .version import check_latest, is_flagged
    from niworkflows.utils.spaces import Reference, OutputReferencesAction

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f"Path does not exist: <{path}>.")
        return Path(path).absolute()

    def _is_file(path, parser):
        """Ensure a given path exists and it is a file."""
        path = _path_exists(path, parser)
        if not path.is_file():
            raise parser.error(f"Path should point to a file (or symlink of file): <{path}>.")
        return path

    def _min_one(value, parser):
        """Ensure an argument is not lower than 1."""
        value = int(value)
        if value < 1:
            raise parser.error("Argument can't be less than one.")
        return value

    def _to_gb(value):
        scale = {"G": 1, "T": 10 ** 3, "M": 1e-3, "K": 1e-6, "B": 1e-9}
        digits = "".join([c for c in value if c.isdigit()])
        units = value[len(digits):] or "M"
        return int(digits) * scale[units[0]]

    def _drop_sub(value):
        return value[4:] if value.startswith("sub-") else value

    def _filter_pybids_none_any(dct):
        import bids
        return {
            k: bids.layout.Query.NONE
            if v is None
            else (bids.layout.Query.ANY if v == "*" else v)
            for k, v in dct.items()
        }

    def _bids_filter(value, parser):
        from json import loads, JSONDecodeError

        if value:
            if Path(value).exists():
                try:
                    return loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
                except JSONDecodeError:
                    raise parser.error(f"JSON syntax error in: <{value}>.")
            else:
                raise parser.error(f"Path does not exist: <{value}>.")

    def _slice_time_ref(value, parser):
        if value == "start":
            value = 0
        elif value == "middle":
            value = 0.5
        try:
            value = float(value)
        except ValueError:
            raise parser.error("Slice time reference must be number, 'start', or 'middle'. "
                               f"Received {value}.")
        if not 0 <= value <= 1:
            raise parser.error(f"Slice time reference must be in range 0-1. Received {value}.")
        return value

    verstr = f"fMRIPrep v{config.environment.version}"
    currentv = Version(config.environment.version)
    is_release = not any(
        (currentv.is_devrelease, currentv.is_prerelease, currentv.is_postrelease)
    )

    parser = ArgumentParser(
        description="fMRIPrep: fMRI PREProcessing workflows v{}".format(
            config.environment.version
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
        **kwargs,
    )
    PathExists = partial(_path_exists, parser=parser)
    IsFile = partial(_is_file, parser=parser)
    PositiveInt = partial(_min_one, parser=parser)
    BIDSFilter = partial(_bids_filter, parser=parser)
    SliceTimeRef = partial(_slice_time_ref, parser=parser)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        "bids_dir",
        action="store",
        type=PathExists,
        help="the root folder of a BIDS valid dataset (sub-XXXXX folders should "
        "be found at the top level in this folder).",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="the output path for the outcomes of preprocessing and visual " "reports",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant"],
        help='processing stage to be run, only "participant" in the case of '
        "fMRIPrep (see BIDS-Apps specification).",
    )

    # optional arguments
    parser.add_argument("--version", action="version", version=verstr)

    g_bids = parser.add_argument_group("Options for filtering BIDS queries")
    g_bids.add_argument(
        "--skip_bids_validation",
        "--skip-bids-validation",
        action="store_true",
        default=False,
        help="assume the input dataset is BIDS compliant and skip the validation",
    )
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        action="store",
        nargs="+",
        type=_drop_sub,
        help="a space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed)",
    )
    # Re-enable when option is actually implemented
    # g_bids.add_argument('-s', '--session-id', action='store', default='single_session',
    #                     help='select a specific session to be processed')
    # Re-enable when option is actually implemented
    # g_bids.add_argument('-r', '--run-id', action='store', default='single_run',
    #                     help='select a specific run to be processed')
    g_bids.add_argument(
        "-t", "--task-id", action="store", help="select a specific task to be processed"
    )
    g_bids.add_argument(
        "--echo-idx",
        action="store",
        type=int,
        help="select a specific echo to be processed in a multiecho series",
    )
    g_bids.add_argument(
        "--bids-filter-file",
        dest="bids_filters",
        action="store",
        type=BIDSFilter,
        metavar="FILE",
        help="a JSON file describing custom BIDS input filters using PyBIDS. "
        "For further details, please check out "
        "https://fmriprep.readthedocs.io/en/%s/faq.html#"
        "how-do-I-select-only-certain-files-to-be-input-to-fMRIPrep"
        % (currentv.base_version if is_release else "latest"),
    )
    g_bids.add_argument(
        "--anat-derivatives",
        action="store",
        metavar="PATH",
        type=PathExists,
        help="Reuse the anatomical derivatives from another fMRIPrep run or calculated "
        "with an alternative processing tool (NOT RECOMMENDED).",
    )
    g_bids.add_argument(
        "--bids-database-dir",
        metavar="PATH",
        type=Path,
        help="Path to a PyBIDS database folder, for faster indexing (especially "
             "useful for large datasets). Will be created if not present."
    )

    g_perfm = parser.add_argument_group("Options to handle performance")
    g_perfm.add_argument(
        "--nprocs",
        "--nthreads",
        "--n_cpus",
        "--n-cpus",
        dest='nprocs',
        action="store",
        type=PositiveInt,
        help="maximum number of threads across all processes",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        action="store",
        type=PositiveInt,
        help="maximum number of threads per-process",
    )
    g_perfm.add_argument(
        "--mem",
        "--mem_mb",
        "--mem-mb",
        dest="memory_gb",
        action="store",
        type=_to_gb,
        metavar="MEMORY_MB",
        help="upper bound memory limit for fMRIPrep processes",
    )
    g_perfm.add_argument(
        "--low-mem",
        action="store_true",
        help="attempt to reduce memory usage (will increase disk usage "
        "in working directory)",
    )
    g_perfm.add_argument(
        "--use-plugin",
        "--nipype-plugin-file",
        action="store",
        metavar="FILE",
        type=IsFile,
        help="nipype plugin configuration file",
    )
    g_perfm.add_argument(
        "--anat-only", action="store_true", help="run anatomical workflows only"
    )
    g_perfm.add_argument(
        "--boilerplate_only",
        action="store_true",
        default=False,
        help="generate boilerplate only",
    )
    g_perfm.add_argument(
        "--md-only-boilerplate",
        action="store_true",
        default=False,
        help="skip generation of HTML and LaTeX formatted citation with pandoc",
    )
    g_perfm.add_argument(
        "--error-on-aroma-warnings",
        action="store_true",
        dest="aroma_err_on_warn",
        default=False,
        help="Raise an error if ICA_AROMA does not produce sensible output "
        "(e.g., if all the components are classified as signal or noise)",
    )
    g_perfm.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="increases log verbosity for each occurence, debug level is -vvv",
    )

    g_conf = parser.add_argument_group("Workflow configuration")
    g_conf.add_argument(
        "--ignore",
        required=False,
        action="store",
        nargs="+",
        default=[],
        choices=["fieldmaps", "slicetiming", "sbref", "t2w", "flair"],
        help="ignore selected aspects of the input dataset to disable corresponding "
        "parts of the workflow (a space delimited list)",
    )
    g_conf.add_argument(
        "--longitudinal",
        action="store_true",
        help="treat dataset as longitudinal - may increase runtime",
    )
    g_conf.add_argument(
        "--output-spaces",
        nargs="*",
        action=OutputReferencesAction,
        help="""\
Standard and non-standard spaces to resample anatomical and functional images to. \
Standard spaces may be specified by the form \
``<SPACE>[:cohort-<label>][:res-<resolution>][...]``, where ``<SPACE>`` is \
a keyword designating a spatial reference, and may be followed by optional, \
colon-separated parameters. \
Non-standard spaces imply specific orientations and sampling grids. \
Important to note, the ``res-*`` modifier does not define the resolution used for \
the spatial normalization. To generate no BOLD outputs, use this option without specifying \
any spatial references. For further details, please check out \
https://fmriprep.readthedocs.io/en/%s/spaces.html"""
        % (currentv.base_version if is_release else "latest"),
    )
    g_conf.add_argument(
        "--me-output-echos",
        action="store_true",
        default=False,
        help="""\
Output individual echo time series with slice, motion and susceptibility correction. \
Useful for further Tedana processing post-fMRIPrep."""
    )

    g_conf.add_argument(
        "--bold2t1w-init",
        action="store",
        default="register",
        choices=["register", "header"],
        help='Either "register" (the default) to initialize volumes at center or "header"'
        " to use the header information when coregistering BOLD to T1w images.",
    )
    g_conf.add_argument(
        "--bold2t1w-dof",
        action="store",
        default=6,
        choices=[6, 9, 12],
        type=int,
        help="Degrees of freedom when registering BOLD to T1w images. "
        "6 degrees (rotation and translation) are used by default.",
    )
    g_conf.add_argument(
        "--force-bbr",
        action="store_true",
        dest="use_bbr",
        default=None,
        help="Always use boundary-based registration (no goodness-of-fit checks)",
    )
    g_conf.add_argument(
        "--force-no-bbr",
        action="store_false",
        dest="use_bbr",
        default=None,
        help="Do not use boundary-based registration (no goodness-of-fit checks)",
    )
    g_conf.add_argument(
        "--medial-surface-nan",
        required=False,
        action="store_true",
        default=False,
        help="Replace medial wall values with NaNs on functional GIFTI files. Only "
        "performed for GIFTI files mapped to a freesurfer subject (fsaverage or fsnative).",
    )
    g_conf.add_argument(
        "--slice-time-ref",
        required=False,
        action="store",
        default=None,
        type=SliceTimeRef,
        help="The time of the reference slice to correct BOLD values to, as a fraction "
             "acquisition time. 0 indicates the start, 0.5 the midpoint, and 1 the end "
             "of acquisition. The alias `start` corresponds to 0, and `middle` to 0.5. "
             "The default value is 0.5.",
    )
    g_conf.add_argument(
        "--dummy-scans",
        required=False,
        action="store",
        default=None,
        type=int,
        help="Number of nonsteady-state volumes.",
    )
    g_conf.add_argument(
        "--random-seed",
        dest="_random_seed",
        action="store",
        type=int,
        default=None,
        help="Initialize the random seed for the workflow",
    )

    # ICA_AROMA options
    g_aroma = parser.add_argument_group("Specific options for running ICA_AROMA")
    g_aroma.add_argument(
        "--use-aroma",
        action="store_true",
        default=False,
        help="add ICA_AROMA to your preprocessing stream",
    )
    g_aroma.add_argument(
        "--aroma-melodic-dimensionality",
        dest="aroma_melodic_dim",
        action="store",
        default=-200,
        type=int,
        help="Exact or maximum number of MELODIC components to estimate "
        "(positive = exact, negative = maximum)",
    )

    # Confounds options
    g_confounds = parser.add_argument_group("Specific options for estimating confounds")
    g_confounds.add_argument(
        "--return-all-components",
        dest="regressors_all_comps",
        required=False,
        action="store_true",
        default=False,
        help="Include all components estimated in CompCor decomposition in the confounds "
        "file instead of only the components sufficient to explain 50 percent of "
        "BOLD variance in each CompCor mask",
    )
    g_confounds.add_argument(
        "--fd-spike-threshold",
        dest="regressors_fd_th",
        required=False,
        action="store",
        default=0.5,
        type=float,
        help="Threshold for flagging a frame as an outlier on the basis of framewise "
        "displacement",
    )
    g_confounds.add_argument(
        "--dvars-spike-threshold",
        dest="regressors_dvars_th",
        required=False,
        action="store",
        default=1.5,
        type=float,
        help="Threshold for flagging a frame as an outlier on the basis of standardised "
        "DVARS",
    )

    #  ANTs options
    g_ants = parser.add_argument_group("Specific options for ANTs registrations")
    g_ants.add_argument(
        "--skull-strip-template",
        default="OASIS30ANTs",
        type=Reference.from_string,
        help="select a template for skull-stripping with antsBrainExtraction",
    )
    g_ants.add_argument(
        "--skull-strip-fixed-seed",
        action="store_true",
        help="do not use a random seed for skull-stripping - will ensure "
        "run-to-run replicability when used with --omp-nthreads 1 and "
        "matching --random-seed <int>",
    )
    g_ants.add_argument(
        "--skull-strip-t1w",
        action="store",
        choices=("auto", "skip", "force"),
        default="force",
        help="determiner for T1-weighted skull stripping ('force' ensures skull "
        "stripping, 'skip' ignores skull stripping, and 'auto' applies brain extraction "
        "based on the outcome of a heuristic to check whether the brain is already masked).",
    )

    # Fieldmap options
    g_fmap = parser.add_argument_group("Specific options for handling fieldmaps")
    g_fmap.add_argument(
        "--fmap-bspline",
        action="store_true",
        default=False,
        help="fit a B-Spline field using least-squares (experimental)",
    )
    g_fmap.add_argument(
        "--fmap-no-demean",
        action="store_false",
        default=True,
        help="do not remove median (within mask) from fieldmap",
    )
    g_fmap.add_argument(
        "--topup-max-vols",
        action="store",
        default=5,
        type=int,
        help="maximum number of volumes to use with TOPUP, per-series (EPI or BOLD)",
    )

    # SyN-unwarp options
    g_syn = parser.add_argument_group("Specific options for SyN distortion correction")
    g_syn.add_argument(
        "--use-syn-sdc",
        nargs="?",
        choices=["warn", "error"],
        action="store",
        const="error",
        default=False,
        help="EXPERIMENTAL: Use fieldmap-free distortion correction; "
             "if unable, error (default) or warn based on optional argument.",
    )
    g_syn.add_argument(
        "--force-syn",
        action="store_true",
        default=False,
        help="EXPERIMENTAL/TEMPORARY: Use SyN correction in addition to "
        "fieldmap correction, if available",
    )

    # FreeSurfer options
    g_fs = parser.add_argument_group("Specific options for FreeSurfer preprocessing")
    g_fs.add_argument(
        "--fs-license-file",
        metavar="FILE",
        type=IsFile,
        help="Path to FreeSurfer license key file. Get it (for free) by registering"
        " at https://surfer.nmr.mgh.harvard.edu/registration.html",
    )
    g_fs.add_argument(
        "--fs-subjects-dir",
        metavar="PATH",
        type=Path,
        help="Path to existing FreeSurfer subjects directory to reuse. "
        "(default: OUTPUT_DIR/freesurfer)",
    )

    # Surface generation xor
    g_surfs = parser.add_argument_group("Surface preprocessing options")
    g_surfs.add_argument(
        "--no-submm-recon",
        action="store_false",
        dest="hires",
        help="disable sub-millimeter (hires) reconstruction",
    )
    g_surfs_xor = g_surfs.add_mutually_exclusive_group()
    g_surfs_xor.add_argument(
        "--cifti-output",
        nargs="?",
        const="91k",
        default=False,
        choices=("91k", "170k"),
        type=str,
        help="output preprocessed BOLD as a CIFTI dense timeseries. "
        "Optionally, the number of grayordinate can be specified "
        "(default is 91k, which equates to 2mm resolution)",
    )
    g_surfs_xor.add_argument(
        "--fs-no-reconall",
        action="store_false",
        dest="run_reconall",
        help="disable FreeSurfer surface preprocessing.",
    )

    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "--output-layout",
        action="store",
        default="bids",
        choices=("bids", "legacy"),
        help="Organization of outputs. bids (default) places fMRIPrep derivatives "
        "directly in the output directory, and defaults to placing FreeSurfer "
        "derivatives in <output-dir>/sourcedata/freesurfer. legacy creates "
        "derivative datasets as subdirectories of outputs."
    )
    g_other.add_argument(
        "-w",
        "--work-dir",
        action="store",
        type=Path,
        default=Path("work").absolute(),
        help="path where intermediate results should be stored",
    )
    g_other.add_argument(
        "--clean-workdir",
        action="store_true",
        default=False,
        help="Clears working directory of contents. Use of this flag is not"
        "recommended when running concurrent processes of fMRIPrep.",
    )
    g_other.add_argument(
        "--resource-monitor",
        action="store_true",
        default=False,
        help="enable Nipype's resource monitoring to keep track of memory and CPU usage",
    )
    g_other.add_argument(
        "--reports-only",
        action="store_true",
        default=False,
        help="only generate reports, don't run workflows. This will only rerun report "
        "aggregation, not reportlet generation for specific nodes.",
    )
    g_other.add_argument(
        "--config-file",
        action="store",
        metavar="FILE",
        help="Use pre-generated configuration file. Values in file will be overridden "
             "by command-line arguments.")
    g_other.add_argument(
        "--write-graph",
        action="store_true",
        default=False,
        help="Write workflow graph.",
    )
    g_other.add_argument(
        "--stop-on-first-crash",
        action="store_true",
        default=False,
        help="Force stopping on first crash, even if a work directory"
        " was specified.",
    )
    g_other.add_argument(
        "--notrack",
        action="store_true",
        default=False,
        help="Opt-out of sending tracking information of this run to "
        "the FMRIPREP developers. This information helps to "
        "improve FMRIPREP and provides an indicator of real "
        "world usage crucial for obtaining funding.",
    )
    g_other.add_argument(
        "--debug",
        action="store",
        nargs="+",
        choices=config.DEBUG_MODES + ("all",),
        help="Debug mode(s) to enable. 'all' is alias for all available modes.",
    )

    g_other.add_argument(
        "--sloppy",
        action="store_true",
        default=False,
        help="Use low-quality tools for speed - TESTING ONLY",
    )

    latest = check_latest()
    if latest is not None and currentv < latest:
        print(
            """\
You are using fMRIPrep-%s, and a newer version of fMRIPrep is available: %s.
Please check out our documentation about how and when to upgrade:
https://fmriprep.readthedocs.io/en/latest/faq.html#upgrading"""
            % (currentv, latest),
            file=sys.stderr,
        )

    _blist = is_flagged()
    if _blist[0]:
        _reason = _blist[1] or "unknown"
        print(
            """\
WARNING: Version %s of fMRIPrep (current) has been FLAGGED
(reason: %s).
That means some severe flaw was found in it and we strongly
discourage its usage."""
            % (config.environment.version, _reason),
            file=sys.stderr,
        )

    return parser


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""
    import logging
    from niworkflows.utils.spaces import Reference, SpatialReferences

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)

    if opts.config_file:
        skip = {} if opts.reports_only else {"execution": ("run_uuid",)}
        config.load(opts.config_file, skip=skip, init=False)
        config.loggers.cli.info(f"Loaded previous configuration file {opts.config_file}")

    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    config.from_dict(vars(opts), init=['nipype'])

    if not config.execution.notrack:
        import pkgutil
        if pkgutil.find_loader("sentry_sdk") is None:
            config.execution.notrack = True
            config.loggers.cli.warning("Telemetry disabled because sentry_sdk is not installed.")
        else:
            config.loggers.cli.info(
                "Telemetry system to collect crashes and errors is enabled "
                "- thanks for your feedback!. Use option ``--notrack`` to opt out."
            )

    # Initialize --output-spaces if not defined
    if config.execution.output_spaces is None:
        config.execution.output_spaces = SpatialReferences(
            [Reference("MNI152NLin2009cAsym", {"res": "native"})]
        )

    # Retrieve logging level
    build_log = config.loggers.cli

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        import yaml

        with open(opts.use_plugin) as f:
            plugin_settings = yaml.load(f, Loader=yaml.FullLoader)
        _plugin = plugin_settings.get("plugin")
        if _plugin:
            config.nipype.plugin = _plugin
            config.nipype.plugin_args = plugin_settings.get("plugin_args", {})
            config.nipype.nprocs = opts.nprocs or config.nipype.plugin_args.get(
                "n_procs", config.nipype.nprocs
            )

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    if 1 < config.nipype.nprocs < config.nipype.omp_nthreads:
        build_log.warning(
            f"Per-process threads (--omp-nthreads={config.nipype.omp_nthreads}) exceed "
            f"total threads (--nthreads/--n_cpus={config.nipype.nprocs})"
        )

    # Inform the user about the risk of using brain-extracted images
    if config.workflow.skull_strip_t1w == "auto":
        build_log.warning(
            """\
Option ``--skull-strip-t1w`` was set to 'auto'. A heuristic will be \
applied to determine whether the input T1w image(s) have already been skull-stripped.
If that were the case, brain extraction and INU correction will be skipped for those T1w \
inputs. Please, BEWARE OF THE RISKS TO THE CONSISTENCY of results when using varying \
processing workflows across participants. To determine whether a participant has been run \
through the shortcut pipeline (meaning, brain extraction was skipped), please check the \
citation boilerplate. When reporting results with varying pipelines, please make sure you \
mention this particular variant of fMRIPrep listing the participants for which it was \
applied."""
        )

    bids_dir = config.execution.bids_dir
    output_dir = config.execution.output_dir
    work_dir = config.execution.work_dir
    version = config.environment.version
    output_layout = config.execution.output_layout

    if config.execution.fs_subjects_dir is None:
        if output_layout == "bids":
            config.execution.fs_subjects_dir = output_dir / "sourcedata" / "freesurfer"
        elif output_layout == "legacy":
            config.execution.fs_subjects_dir = output_dir / "freesurfer"
    if config.execution.fmriprep_dir is None:
        if output_layout == "bids":
            config.execution.fmriprep_dir = output_dir
        elif output_layout == "legacy":
            config.execution.fmriprep_dir = output_dir / "fmriprep"

    # Wipe out existing work_dir
    if opts.clean_workdir and work_dir.exists():
        from niworkflows.utils.misc import clean_directory

        build_log.info(f"Clearing previous fMRIPrep working directory: {work_dir}")
        if not clean_directory(work_dir):
            build_log.warning(
                f"Could not clear all contents of working directory: {work_dir}"
            )

    # Update the config with an empty dict to trigger initialization of all config
    # sections (we used `init=False` above).
    # This must be done after cleaning the work directory, or we could delete an
    # open SQLite database
    config.from_dict({})

    # Ensure input and output folders are not the same
    if output_dir == bids_dir:
        parser.error(
            "The selected output folder is the same as the input BIDS folder. "
            "Please modify the output path (suggestion: %s)."
            % bids_dir
            / "derivatives"
            / ("fmriprep-%s" % version.split("+")[0])
        )

    if bids_dir in work_dir.parents:
        parser.error(
            "The selected working directory is a subdirectory of the input BIDS folder. "
            "Please modify the output path."
        )

    # Validate inputs
    if not opts.skip_bids_validation:
        from ..utils.bids import validate_input_dir

        build_log.info(
            "Making sure the input data is BIDS compliant (warnings can be ignored in most "
            "cases)."
        )
        validate_input_dir(
            config.environment.exec_env, opts.bids_dir, opts.participant_label
        )

    # Setup directories
    config.execution.log_dir = config.execution.fmriprep_dir / "logs"
    # Check and create output and working directories
    config.execution.log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Force initialization of the BIDSLayout
    config.execution.init()
    all_subjects = config.execution.layout.get_subjects()
    if config.execution.participant_label is None:
        config.execution.participant_label = all_subjects

    participant_label = set(config.execution.participant_label)
    missing_subjects = participant_label - set(all_subjects)
    if missing_subjects:
        parser.error(
            "One or more participant labels were not found in the BIDS directory: "
            "%s." % ", ".join(missing_subjects)
        )

    config.execution.participant_label = sorted(participant_label)
    config.workflow.skull_strip_template = config.workflow.skull_strip_template[0]
