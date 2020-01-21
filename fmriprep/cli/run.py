#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fMRI preprocessing workflow
=====
"""

import os
import re
from pathlib import Path
import logging
import sys
import gc
import uuid
import warnings
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from multiprocessing import cpu_count
from time import strftime

logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')


def _warn_redirect(message, category, filename, lineno, file=None, line=None):
    logger.warning('Captured warning (%s): %s', category, message)


def check_deps(workflow):
    from nipype.utils.filemanip import which
    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and
            which(node.interface._cmd.split()[0]) is None))


def get_parser():
    """Build parser object"""
    from smriprep.cli.utils import (
        ParseTemplates,
        output_space as _output_space,
    )
    from templateflow.api import templates
    from packaging.version import Version
    from ..__about__ import __version__
    from ..config import NONSTANDARD_REFERENCES
    from .version import check_latest, is_flagged

    ParseTemplates.set_nonstandard_spaces(tuple(NONSTANDARD_REFERENCES))

    verstr = 'fmriprep v{}'.format(__version__)
    currentv = Version(__version__)
    is_release = not any((currentv.is_devrelease, currentv.is_prerelease, currentv.is_postrelease))

    parser = ArgumentParser(description='FMRIPREP: fMRI PREProcessing workflows',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument('bids_dir', action='store', type=Path,
                        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
                             'be found at the top level in this folder).')
    parser.add_argument('output_dir', action='store', type=Path,
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
    parser.add_argument('analysis_level', choices=['participant'],
                        help='processing stage to be run, only "participant" in the case of '
                             'FMRIPREP (see BIDS-Apps specification).')

    # optional arguments
    parser.add_argument('--version', action='version', version=verstr)

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument('--skip_bids_validation', '--skip-bids-validation', action='store_true',
                        default=False,
                        help='assume the input dataset is BIDS compliant and skip the validation')
    g_bids.add_argument('--participant_label', '--participant-label', action='store', nargs='+',
                        help='a space delimited list of participant identifiers or a single '
                             'identifier (the sub- prefix can be removed)')
    # Re-enable when option is actually implemented
    # g_bids.add_argument('-s', '--session-id', action='store', default='single_session',
    #                     help='select a specific session to be processed')
    # Re-enable when option is actually implemented
    # g_bids.add_argument('-r', '--run-id', action='store', default='single_run',
    #                     help='select a specific run to be processed')
    g_bids.add_argument('-t', '--task-id', action='store',
                        help='select a specific task to be processed')
    g_bids.add_argument('--echo-idx', action='store', type=int,
                        help='select a specific echo to be processed in a multiecho series')

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--nthreads', '--n_cpus', '-n-cpus', action='store', type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--omp-nthreads', action='store', type=int, default=0,
                         help='maximum number of threads per-process')
    g_perfm.add_argument('--mem_mb', '--mem-mb', action='store', default=0, type=int,
                         help='upper bound memory limit for FMRIPREP processes')
    g_perfm.add_argument('--low-mem', action='store_true',
                         help='attempt to reduce memory usage (will increase disk usage '
                              'in working directory)')
    g_perfm.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')
    g_perfm.add_argument('--anat-only', action='store_true',
                         help='run anatomical workflows only')
    g_perfm.add_argument('--boilerplate', action='store_true',
                         help='generate boilerplate only')
    g_perfm.add_argument('--md-only-boilerplate', action='store_true',
                         default=False,
                         help='skip generation of HTML and LaTeX formatted citation with pandoc')
    g_perfm.add_argument('--error-on-aroma-warnings', action='store_true',
                         default=False,
                         help='Raise an error if ICA_AROMA does not produce sensible output '
                              '(e.g., if all the components are classified as signal or noise)')
    g_perfm.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
                         help="increases log verbosity for each occurence, debug level is -vvv")

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore', required=False, action='store', nargs="+", default=[],
        choices=['fieldmaps', 'slicetiming', 'sbref'],
        help='ignore selected aspects of the input dataset to disable corresponding '
             'parts of the workflow (a space delimited list)')
    g_conf.add_argument(
        '--longitudinal', action='store_true',
        help='treat dataset as longitudinal - may increase runtime')
    g_conf.add_argument(
        '--t2s-coreg', action='store_true',
        help='If provided with multi-echo BOLD dataset, create T2*-map and perform '
             'T2*-driven coregistration. When multi-echo data is provided and this '
             'option is not enabled, standard EPI-T1 coregistration is performed '
             'using the middle echo.')
    g_conf.add_argument(
        '--output-spaces', nargs='+', action=ParseTemplates,
        help="""\
Standard and non-standard spaces to resample anatomical and functional images to. \
Standard spaces may be specified by the form \
``<TEMPLATE>[:res-<resolution>][:cohort-<label>][...]``, where ``<TEMPLATE>`` is \
a keyword (valid keywords: %s) or path pointing to a user-supplied template, and \
may be followed by optional, colon-separated parameters. \
Non-standard spaces (valid keywords: %s) imply specific orientations and sampling \
grids. \
Important to note, the ``res-*`` modifier does not define the resolution used for \
the spatial normalization.
For further details, please check out \
https://fmriprep.readthedocs.io/en/%s/spaces.html""" % (
            ', '.join('"%s"' % s for s in templates()), ', '.join(NONSTANDARD_REFERENCES),
            currentv.base_version if is_release else 'latest'))

    g_conf.add_argument('--bold2t1w-dof', action='store', default=6, choices=[6, 9, 12], type=int,
                        help='Degrees of freedom when registering BOLD to T1w images. '
                             '6 degrees (rotation and translation) are used by default.')
    g_conf.add_argument(
        '--force-bbr', action='store_true', dest='use_bbr', default=None,
        help='Always use boundary-based registration (no goodness-of-fit checks)')
    g_conf.add_argument(
        '--force-no-bbr', action='store_false', dest='use_bbr', default=None,
        help='Do not use boundary-based registration (no goodness-of-fit checks)')
    g_conf.add_argument(
        '--medial-surface-nan', required=False, action='store_true', default=False,
        help='Replace medial wall values with NaNs on functional GIFTI files. Only '
        'performed for GIFTI files mapped to a freesurfer subject (fsaverage or fsnative).')
    g_conf.add_argument(
        '--dummy-scans', required=False, action='store', default=None, type=int,
        help='Number of non steady state volumes.')

    # ICA_AROMA options
    g_aroma = parser.add_argument_group('Specific options for running ICA_AROMA')
    g_aroma.add_argument('--use-aroma', action='store_true', default=False,
                         help='add ICA_AROMA to your preprocessing stream')
    g_aroma.add_argument('--aroma-melodic-dimensionality', action='store',
                         default=-200, type=int,
                         help='Exact or maximum number of MELODIC components to estimate '
                         '(positive = exact, negative = maximum)')

    # Confounds options
    g_confounds = parser.add_argument_group('Specific options for estimating confounds')
    g_confounds.add_argument(
        '--return-all-components', required=False, action='store_true', default=False,
        help='Include all components estimated in CompCor decomposition in the confounds '
             'file instead of only the components sufficient to explain 50 percent of '
             'BOLD variance in each CompCor mask')
    g_confounds.add_argument(
        '--fd-spike-threshold', required=False, action='store', default=0.5, type=float,
        help='Threshold for flagging a frame as an outlier on the basis of framewise '
             'displacement')
    g_confounds.add_argument(
        '--dvars-spike-threshold', required=False, action='store', default=1.5, type=float,
        help='Threshold for flagging a frame as an outlier on the basis of standardised '
             'DVARS')

    #  ANTs options
    g_ants = parser.add_argument_group('Specific options for ANTs registrations')
    g_ants.add_argument(
        '--skull-strip-template', action='store', default='OASIS30ANTs', type=_output_space,
        help='select a template for skull-stripping with antsBrainExtraction')
    g_ants.add_argument('--skull-strip-fixed-seed', action='store_true',
                        help='do not use a random seed for skull-stripping - will ensure '
                             'run-to-run replicability when used with --omp-nthreads 1')

    # Fieldmap options
    g_fmap = parser.add_argument_group('Specific options for handling fieldmaps')
    g_fmap.add_argument('--fmap-bspline', action='store_true', default=False,
                        help='fit a B-Spline field using least-squares (experimental)')
    g_fmap.add_argument('--fmap-no-demean', action='store_false', default=True,
                        help='do not remove median (within mask) from fieldmap')

    # SyN-unwarp options
    g_syn = parser.add_argument_group('Specific options for SyN distortion correction')
    g_syn.add_argument('--use-syn-sdc', action='store_true', default=False,
                       help='EXPERIMENTAL: Use fieldmap-free distortion correction')
    g_syn.add_argument('--force-syn', action='store_true', default=False,
                       help='EXPERIMENTAL/TEMPORARY: Use SyN correction in addition to '
                       'fieldmap correction, if available')

    # FreeSurfer options
    g_fs = parser.add_argument_group('Specific options for FreeSurfer preprocessing')
    g_fs.add_argument(
        '--fs-license-file', metavar='PATH', type=Path,
        help='Path to FreeSurfer license key file. Get it (for free) by registering'
             ' at https://surfer.nmr.mgh.harvard.edu/registration.html')
    g_fs.add_argument(
        '--fs-subjects-dir', metavar='PATH', type=Path,
        help='Path to existing FreeSurfer subjects directory to reuse. '
             '(default: OUTPUT_DIR/freesurfer)')

    # Surface generation xor
    g_surfs = parser.add_argument_group('Surface preprocessing options')
    g_surfs.add_argument('--no-submm-recon', action='store_false', dest='hires',
                         help='disable sub-millimeter (hires) reconstruction')
    g_surfs_xor = g_surfs.add_mutually_exclusive_group()
    g_surfs_xor.add_argument('--cifti-output', nargs='?', const='91k', default=False,
                             choices=('91k', '170k'),
                             help='output preprocessed BOLD as a CIFTI dense timeseries. '
                             'Optionally, the number of grayordinate can be specified '
                             '(default is 91k, which equates to 2mm resolution)')
    g_surfs_xor.add_argument('--fs-no-reconall',
                             action='store_false', dest='run_reconall',
                             help='disable FreeSurfer surface preprocessing.')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store', type=Path, default=Path('work'),
                         help='path where intermediate results should be stored')
    g_other.add_argument(
        '--resource-monitor', action='store_true', default=False,
        help='enable Nipype\'s resource monitoring to keep track of memory and CPU usage')
    g_other.add_argument(
        '--reports-only', action='store_true', default=False,
        help='only generate reports, don\'t run workflows. This will only rerun report '
             'aggregation, not reportlet generation for specific nodes.')
    g_other.add_argument(
        '--run-uuid', action='store', default=None,
        help='Specify UUID of previous run, to include error logs in report. '
             'No effect without --reports-only.')
    g_other.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')
    g_other.add_argument('--stop-on-first-crash', action='store_true', default=False,
                         help='Force stopping on first crash, even if a work directory'
                              ' was specified.')
    g_other.add_argument('--notrack', action='store_true', default=False,
                         help='Opt-out of sending tracking information of this run to '
                              'the FMRIPREP developers. This information helps to '
                              'improve FMRIPREP and provides an indicator of real '
                              'world usage crucial for obtaining funding.')
    g_other.add_argument('--sloppy', action='store_true', default=False,
                         help='Use low-quality tools for speed - TESTING ONLY')

    latest = check_latest()
    if latest is not None and currentv < latest:
        print("""\
You are using fMRIPrep-%s, and a newer version of fMRIPrep is available: %s.
Please check out our documentation about how and when to upgrade:
https://fmriprep.readthedocs.io/en/latest/faq.html#upgrading""" % (
            __version__, latest), file=sys.stderr)

    _blist = is_flagged()
    if _blist[0]:
        _reason = _blist[1] or 'unknown'
        print("""\
WARNING: Version %s of fMRIPrep (current) has been FLAGGED
(reason: %s).
That means some severe flaw was found in it and we strongly
discourage its usage.""" % (__version__, _reason), file=sys.stderr)

    return parser


def main():
    """Entry point"""
    from nipype import logging as nlogging
    from multiprocessing import set_start_method, Process, Manager
    from ..utils.bids import write_derivative_description, validate_input_dir
    set_start_method('forkserver')
    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()

    exec_env = os.name

    # special variable set in the container
    if os.getenv('IS_DOCKER_8395080871'):
        exec_env = 'singularity'
        cgroup = Path('/proc/1/cgroup')
        if cgroup.exists() and 'docker' in cgroup.read_text():
            exec_env = 'docker'
            if os.getenv('DOCKER_VERSION_8395080871'):
                exec_env = 'fmriprep-docker'

    sentry_sdk = None
    if not opts.notrack:
        import sentry_sdk
        from ..utils.sentry import sentry_setup
        sentry_setup(opts, exec_env)

    # Validate inputs
    if not opts.skip_bids_validation:
        print("Making sure the input data is BIDS compliant (warnings can be ignored in most "
              "cases).")
        validate_input_dir(exec_env, opts.bids_dir, opts.participant_label)

    # FreeSurfer license
    default_license = str(Path(os.getenv('FREESURFER_HOME')) / 'license.txt')
    # Precedence: --fs-license-file, $FS_LICENSE, default_license
    license_file = opts.fs_license_file or Path(os.getenv('FS_LICENSE', default_license))
    if not license_file.exists():
        raise RuntimeError("""\
ERROR: a valid license file is required for FreeSurfer to run. fMRIPrep looked for an existing \
license file at several paths, in this order: 1) command line argument ``--fs-license-file``; \
2) ``$FS_LICENSE`` environment variable; and 3) the ``$FREESURFER_HOME/license.txt`` path. Get it \
(for free) by registering at https://surfer.nmr.mgh.harvard.edu/registration.html""")
    os.environ['FS_LICENSE'] = str(license_file.resolve())

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    logger.setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    # Call build_workflow(opts, retval)
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(opts, retval))
        p.start()
        p.join()

        retcode = p.exitcode or retval.get('return_code', 0)

        bids_dir = Path(retval.get('bids_dir'))
        output_dir = Path(retval.get('output_dir'))
        work_dir = Path(retval.get('work_dir'))
        plugin_settings = retval.get('plugin_settings', None)
        subject_list = retval.get('subject_list', None)
        fmriprep_wf = retval.get('workflow', None)
        run_uuid = retval.get('run_uuid', None)

    if opts.reports_only:
        sys.exit(int(retcode > 0))

    if opts.boilerplate:
        sys.exit(int(retcode > 0))

    if fmriprep_wf and opts.write_graph:
        fmriprep_wf.write_graph(graph2use="colored", format='svg', simple_form=True)

    retcode = retcode or int(fmriprep_wf is None)
    if retcode != 0:
        sys.exit(retcode)

    # Check workflow for missing commands
    missing = check_deps(fmriprep_wf)
    if missing:
        print("Cannot run fMRIPrep. Missing dependencies:", file=sys.stderr)
        for iface, cmd in missing:
            print("\t{} (Interface: {})".format(cmd, iface))
        sys.exit(2)
    # Clean up master process before running workflow, which may create forks
    gc.collect()

    # Sentry tracking
    if not opts.notrack:
        from ..utils.sentry import start_ping
        start_ping(run_uuid, len(subject_list))

    errno = 1  # Default is error exit unless otherwise set
    try:
        fmriprep_wf.run(**plugin_settings)
    except Exception as e:
        if not opts.notrack:
            from ..utils.sentry import process_crashfile
            crashfolders = [output_dir / 'fmriprep' / 'sub-{}'.format(s) / 'log' / run_uuid
                            for s in subject_list]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob('crash*.*'):
                    process_crashfile(crashfile)

            if "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)
        logger.critical('fMRIPrep failed: %s', e)
        raise
    else:
        if opts.run_reconall:
            from templateflow import api
            from niworkflows.utils.misc import _copy_any
            dseg_tsv = str(api.get('fsaverage', suffix='dseg', extension=['.tsv']))
            _copy_any(dseg_tsv,
                      str(output_dir / 'fmriprep' / 'desc-aseg_dseg.tsv'))
            _copy_any(dseg_tsv,
                      str(output_dir / 'fmriprep' / 'desc-aparcaseg_dseg.tsv'))
        errno = 0
        logger.log(25, 'fMRIPrep finished without errors')
        if not opts.notrack:
            sentry_sdk.capture_message('fMRIPrep finished without errors',
                                       level='info')
    finally:
        from niworkflows.reports import generate_reports
        from subprocess import check_call, CalledProcessError, TimeoutExpired
        from pkg_resources import resource_filename as pkgrf
        from shutil import copyfile

        citation_files = {
            ext: output_dir / 'fmriprep' / 'logs' / ('CITATION.%s' % ext)
            for ext in ('bib', 'tex', 'md', 'html')
        }

        if not opts.md_only_boilerplate and citation_files['md'].exists():
            # Generate HTML file resolving citations
            cmd = ['pandoc', '-s', '--bibliography',
                   pkgrf('fmriprep', 'data/boilerplate.bib'),
                   '--filter', 'pandoc-citeproc',
                   '--metadata', 'pagetitle="fMRIPrep citation boilerplate"',
                   str(citation_files['md']),
                   '-o', str(citation_files['html'])]

            logger.info('Generating an HTML version of the citation boilerplate...')
            try:
                check_call(cmd, timeout=10)
            except (FileNotFoundError, CalledProcessError, TimeoutExpired):
                logger.warning('Could not generate CITATION.html file:\n%s',
                               ' '.join(cmd))

            # Generate LaTex file resolving citations
            cmd = ['pandoc', '-s', '--bibliography',
                   pkgrf('fmriprep', 'data/boilerplate.bib'),
                   '--natbib', str(citation_files['md']),
                   '-o', str(citation_files['tex'])]
            logger.info('Generating a LaTeX version of the citation boilerplate...')
            try:
                check_call(cmd, timeout=10)
            except (FileNotFoundError, CalledProcessError, TimeoutExpired):
                logger.warning('Could not generate CITATION.tex file:\n%s',
                               ' '.join(cmd))
            else:
                copyfile(pkgrf('fmriprep', 'data/boilerplate.bib'),
                         citation_files['bib'])
        else:
            logger.warning('fMRIPrep could not find the markdown version of '
                           'the citation boilerplate (%s). HTML and LaTeX versions'
                           ' of it will not be available', citation_files['md'])

        # Generate reports phase
        failed_reports = generate_reports(
            subject_list, output_dir, work_dir, run_uuid,
            config=pkgrf('fmriprep', 'data/reports-spec.yml'),
            packagename='fmriprep')
        write_derivative_description(bids_dir, output_dir / 'fmriprep')

        if failed_reports and not opts.notrack:
            sentry_sdk.capture_message(
                'Report generation failed for %d subjects' % failed_reports,
                level='error')
        sys.exit(int((errno + failed_reports) > 0))


def build_workflow(opts, retval):
    """
    Create the Nipype Workflow that supports the whole execution
    graph, given the inputs.

    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows fmriprep to enforce
    a hard-limited memory-scope.

    """
    from bids import BIDSLayout

    from nipype import logging as nlogging, config as ncfg
    from niworkflows.utils.bids import collect_participants
    from niworkflows.reports import generate_reports
    from ..__about__ import __version__
    from ..workflows.base import init_fmriprep_wf

    build_log = nlogging.getLogger('nipype.workflow')

    INIT_MSG = """
    Running fMRIPREP version {version}:
      * BIDS dataset path: {bids_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {uuid}.
    """.format

    bids_dir = opts.bids_dir.resolve()
    output_dir = opts.output_dir.resolve()
    work_dir = opts.work_dir.resolve()

    retval['return_code'] = 1
    retval['workflow'] = None
    retval['bids_dir'] = str(bids_dir)
    retval['output_dir'] = str(output_dir)
    retval['work_dir'] = str(work_dir)

    if output_dir == bids_dir:
        build_log.error(
            'The selected output folder is the same as the input BIDS folder. '
            'Please modify the output path (suggestion: %s).',
            bids_dir / 'derivatives' / ('fmriprep-%s' % __version__.split('+')[0]))
        retval['return_code'] = 1
        return retval

    if bids_dir in work_dir.parents:
        build_log.error(
            'The selected working directory is a subdirectory of the input BIDS folder. '
            'Please modify the output path.')
        retval['return_code'] = 1
        return retval

    spaces = parse_spaces(opts)

    # Set up some instrumental utilities
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
    retval['run_uuid'] = run_uuid

    # First check that bids_dir looks like a BIDS folder
    layout = BIDSLayout(str(bids_dir), validate=False,
                        ignore=("code", "stimuli", "sourcedata", "models",
                                "derivatives", re.compile(r'^\.')))
    subject_list = collect_participants(
        layout, participant_label=opts.participant_label)
    retval['subject_list'] = subject_list

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        plugin_settings.setdefault('plugin_args', {})
    else:
        # Defaults
        plugin_settings = {
            'plugin': 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
            }
        }

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    nthreads = plugin_settings['plugin_args'].get('n_procs')
    # Permit overriding plugin config with specific CLI options
    if nthreads is None or opts.nthreads is not None:
        nthreads = opts.nthreads
        if nthreads is None or nthreads < 1:
            nthreads = cpu_count()
        plugin_settings['plugin_args']['n_procs'] = nthreads

    if opts.mem_mb:
        plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb / 1024

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    if 1 < nthreads < omp_nthreads:
        build_log.warning(
            'Per-process threads (--omp-nthreads=%d) exceed total '
            'threads (--nthreads/--n_cpus=%d)', omp_nthreads, nthreads)
    retval['plugin_settings'] = plugin_settings

    # Set up directories
    log_dir = output_dir / 'fmriprep' / 'logs'
    # Check and create output and working directories
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {
            'log_directory': str(log_dir),
            'log_to_file': True
        },
        'execution': {
            'crashdump_dir': str(log_dir),
            'crashfile_format': 'txt',
            'get_linked_libs': False,
            'stop_on_first_crash': opts.stop_on_first_crash,
        },
        'monitoring': {
            'enabled': opts.resource_monitor,
            'sample_frequency': '0.5',
            'summary_append': True,
        }
    })

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    # Called with reports only
    if opts.reports_only:
        from pkg_resources import resource_filename as pkgrf

        build_log.log(25, 'Running --reports-only on participants %s', ', '.join(subject_list))
        if opts.run_uuid is not None:
            run_uuid = opts.run_uuid
            retval['run_uuid'] = run_uuid
        retval['return_code'] = generate_reports(
            subject_list, output_dir, work_dir, run_uuid,
            config=pkgrf('fmriprep', 'data/reports-spec.yml'),
            packagename='fmriprep')
        return retval

    # Build main workflow
    build_log.log(25, INIT_MSG(
        version=__version__,
        bids_dir=bids_dir,
        subject_list=subject_list,
        uuid=run_uuid)
    )

    retval['workflow'] = init_fmriprep_wf(
        anat_only=opts.anat_only,
        aroma_melodic_dim=opts.aroma_melodic_dimensionality,
        bold2t1w_dof=opts.bold2t1w_dof,
        cifti_output=opts.cifti_output,
        debug=opts.sloppy,
        dummy_scans=opts.dummy_scans,
        echo_idx=opts.echo_idx,
        err_on_aroma_warn=opts.error_on_aroma_warnings,
        fmap_bspline=opts.fmap_bspline,
        fmap_demean=opts.fmap_no_demean,
        force_syn=opts.force_syn,
        freesurfer=opts.run_reconall,
        fs_subjects_dir=opts.fs_subjects_dir,
        hires=opts.hires,
        ignore=opts.ignore,
        layout=layout,
        longitudinal=opts.longitudinal,
        low_mem=opts.low_mem,
        medial_surface_nan=opts.medial_surface_nan,
        omp_nthreads=omp_nthreads,
        output_dir=str(output_dir),
        run_uuid=run_uuid,
        regressors_all_comps=opts.return_all_components,
        regressors_fd_th=opts.fd_spike_threshold,
        regressors_dvars_th=opts.dvars_spike_threshold,
        skull_strip_fixed_seed=opts.skull_strip_fixed_seed,
        skull_strip_template=opts.skull_strip_template[0],
        spaces=spaces,
        subject_list=subject_list,
        t2s_coreg=opts.t2s_coreg,
        task_id=opts.task_id,
        use_aroma=opts.use_aroma,
        use_bbr=opts.use_bbr,
        use_syn=opts.use_syn_sdc,
        work_dir=str(work_dir),
    )
    retval['return_code'] = 0

    logs_path = Path(output_dir) / 'fmriprep' / 'logs'
    boilerplate = retval['workflow'].visit_desc()

    if boilerplate:
        citation_files = {
            ext: logs_path / ('CITATION.%s' % ext)
            for ext in ('bib', 'tex', 'md', 'html')
        }
        # To please git-annex users and also to guarantee consistency
        # among different renderings of the same file, first remove any
        # existing one
        for citation_file in citation_files.values():
            try:
                citation_file.unlink()
            except FileNotFoundError:
                pass

        citation_files['md'].write_text(boilerplate)
        build_log.log(25, 'Works derived from this fMRIPrep execution should '
                      'include the following boilerplate:\n\n%s', boilerplate)
    return retval


def parse_spaces(opts):
    """
    Ensures user defined output spaces are correctly parsed.

    Certain options require normalization to a space not explicitly defined by users.
    These spaces will not be included in the final outputs.
    """
    from ..utils import Spaces
    spaces = Spaces(output=opts.output_spaces)
    target_spaces = spaces.unique()  # outputs + internal spaces

    # These arguments implicitly signal expected output
    if not target_spaces:
        if not any((opts.use_aroma, opts.cifti_output)):
            raise RuntimeError(
                "No outputs are expected from this fMRIPrep run. Please add desired outputs using "
                "`--output-spaces`"
            )
        # ensure a standard space is defined
        spaces.add_space('MNI152NLin2009cAsym', output=False)

    # ERROR check if use_aroma was specified, but the correct template was not
    if opts.use_aroma and 'MNI152NLin6Asym' not in target_spaces:
        spaces.add_space('MNI152NLin6Asym', specs={'res': 2}, output=False)

    if opts.cifti_output:
        grayords = {'91k': '32k', '170k': '59k'}  # CIFTI total grayords to surface densities
        if 'fsLR' not in target_spaces:
            spaces.add_space('fsLR', specs={'den': grayords[opts.cifti_output]}, output=False)
        if 'MNI152NLin6Asym' not in target_spaces:
            spaces.add_space('MNI152NLin6Asym', specs={'res': 2}, output=False)

    if 'fsLR' in target_spaces and 'fsaverge' not in target_spaces:
        spaces.add_space('fsaverage', specs={'den': '164k'}, output=False)

    return spaces


if __name__ == '__main__':
    raise RuntimeError("fmriprep/cli/run.py should not be run directly;\n"
                       "Please `pip install` fmriprep and use the `fmriprep` command")
