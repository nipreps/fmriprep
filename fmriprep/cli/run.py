#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fMRI preprocessing workflow
=====
"""

import os
import os.path as op
from pathlib import Path
import logging
import sys
import gc
import re
import uuid
import warnings
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
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
    from ..__about__ import __version__

    verstr = 'fmriprep v{}'.format(__version__)

    parser = ArgumentParser(description='FMRIPREP: fMRI PREProcessing workflows',
                            formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument('bids_dir', action='store',
                        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
                             'be found at the top level in this folder).')
    parser.add_argument('output_dir', action='store',
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
    parser.add_argument('analysis_level', choices=['participant'],
                        help='processing stage to be run, only "participant" in the case of '
                             'FMRIPREP (see BIDS-Apps specification).')

    # optional arguments
    parser.add_argument('--version', action='version', version=verstr)

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
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
    g_perfm.add_argument('--ignore-aroma-denoising-errors', action='store_true',
                         default=False,
                         help='ignores the errors ICA_AROMA returns when there '
                              'are no components classified as either noise or '
                              'signal')
    g_perfm.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
                         help="increases log verbosity for each occurence, debug level is -vvv")
    g_perfm.add_argument('--debug', action='store_true', default=False,
                         help='DEPRECATED - Does not do what you want.')

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
    g_conf.add_argument('--bold2t1w-dof', action='store', default=6, choices=[6, 9, 12], type=int,
                        help='Degrees of freedom when registering BOLD to T1w images. '
                             '6 degrees (rotation and translation) are used by default.')
    g_conf.add_argument(
        '--output-space', required=False, action='store',
        choices=['T1w', 'template', 'fsnative', 'fsaverage', 'fsaverage6', 'fsaverage5'],
        nargs='+', default=['template', 'fsaverage5'],
        help='volume and surface spaces to resample functional series into\n'
             ' - T1w: subject anatomical volume\n'
             ' - template: normalization target specified by --template\n'
             ' - fsnative: individual subject surface\n'
             ' - fsaverage*: FreeSurfer average meshes\n'
             'this argument can be single value or a space delimited list,\n'
             'for example: --output-space T1w fsnative'
    )
    g_conf.add_argument(
        '--force-bbr', action='store_true', dest='use_bbr', default=None,
        help='Always use boundary-based registration (no goodness-of-fit checks)')
    g_conf.add_argument(
        '--force-no-bbr', action='store_false', dest='use_bbr', default=None,
        help='Do not use boundary-based registration (no goodness-of-fit checks)')
    g_conf.add_argument(
        '--template', required=False, action='store',
        choices=['MNI152NLin2009cAsym'], default='MNI152NLin2009cAsym',
        help='volume template space (default: MNI152NLin2009cAsym)')
    g_conf.add_argument(
        '--output-grid-reference', required=False, action='store',
        help='Deprecated after FMRIPREP 1.0.8. Please use --template-resampling-grid instead.')
    g_conf.add_argument(
        '--template-resampling-grid', required=False, action='store', default='native',
        help='Keyword ("native", "1mm", or "2mm") or path to an existing file. '
             'Allows to define a reference grid for the resampling of BOLD images in template '
             'space. Keyword "native" will use the original BOLD grid as reference. '
             'Keywords "1mm" and "2mm" will use the corresponding isotropic template '
             'resolutions. If a path is given, the grid of that image will be used. '
             'It determines the field of view and resolution of the output images, '
             'but is not used in normalization.')
    g_conf.add_argument(
        '--medial-surface-nan', required=False, action='store_true', default=False,
        help='Replace medial wall values with NaNs on functional GIFTI files. Only '
        'performed for GIFTI files mapped to a freesurfer subject (fsaverage or fsnative).')

    # ICA_AROMA options
    g_aroma = parser.add_argument_group('Specific options for running ICA_AROMA')
    g_aroma.add_argument('--use-aroma', action='store_true', default=False,
                         help='add ICA_AROMA to your preprocessing stream')
    g_aroma.add_argument('--aroma-melodic-dimensionality', action='store',
                         default=-200, type=int,
                         help='Exact or maximum number of MELODIC components to estimate '
                         '(positive = exact, negative = maximum)')

    #  ANTs options
    g_ants = parser.add_argument_group('Specific options for ANTs registrations')
    g_ants.add_argument('--skull-strip-template', action='store', default='OASIS',
                        choices=['OASIS', 'NKI'],
                        help='select ANTs skull-stripping template (default: OASIS))')
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
        '--fs-license-file', metavar='PATH', type=os.path.abspath,
        help='Path to FreeSurfer license key file. Get it (for free) by registering'
             ' at https://surfer.nmr.mgh.harvard.edu/registration.html')

    # Surface generation xor
    g_surfs = parser.add_argument_group('Surface preprocessing options')
    g_surfs.add_argument('--no-submm-recon', action='store_false', dest='hires',
                         help='disable sub-millimeter (hires) reconstruction')
    g_surfs_xor = g_surfs.add_mutually_exclusive_group()
    g_surfs_xor.add_argument('--cifti-output', action='store_true', default=False,
                             help='output BOLD files as CIFTI dtseries')
    g_surfs_xor.add_argument('--fs-no-reconall', '--no-freesurfer',
                             action='store_false', dest='run_reconall',
                             help='disable FreeSurfer surface preprocessing.'
                             ' Note : `--no-freesurfer` is deprecated and will be removed in 1.2.'
                             ' Use `--fs-no-reconall` instead.')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store',
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

    return parser


def main():
    """Entry point"""
    from nipype import logging as nlogging
    from multiprocessing import set_start_method, Process, Manager
    from ..viz.reports import generate_reports
    from ..utils.bids import write_derivative_description
    set_start_method('forkserver')

    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()

    sentry_sdk = None
    if not opts.notrack:
        import sentry_sdk
        from ..__about__ import __version__
        environment = "prod"
        release = __version__
        if not __version__:
            environment = "dev"
            release = "dev"
        elif bool(int(os.getenv('FMRIPREP_DEV', 0))) or ('+' in __version__):
            environment = "dev"

        def before_send(event, hints):
            # Filtering log messages about crashed nodes
            if 'logentry' in event and 'message' in event['logentry']:
                msg = event['logentry']['message']
                if msg.startswith("could not run node:"):
                    return None
                elif msg.startswith("Saving crash info to "):
                    return None
                elif re.match("Node .+ failed to run on host .+", msg):
                    return None
            else:
                return event

        sentry_sdk.init("https://d5a16b0c38d84d1584dfc93b9fb1ade6@sentry.io/1137693",
                        release=release,
                        environment=environment,
                        before_send=before_send)
        with sentry_sdk.configure_scope() as scope:
            exec_env = os.name
            # special variable set in the container
            if os.getenv('IS_DOCKER_8395080871'):
                # based on https://stackoverflow.com/a/42674935/616300
                with open('/proc/1/cgroup', 'rt') as ifh:
                    if 'docker' in ifh.read():
                        exec_env = 'docker'
                    else:
                        exec_env = 'singularity'
            scope.set_tag('exec_env', exec_env)

            for k, v in vars(opts).items():
                scope.set_tag(k, v)

    # FreeSurfer license
    default_license = str(Path(os.getenv('FREESURFER_HOME')) / 'license.txt')
    # Precedence: --fs-license-file, $FS_LICENSE, default_license
    license_file = opts.fs_license_file or os.getenv('FS_LICENSE', default_license)
    if not os.path.exists(license_file):
        raise RuntimeError(
            'ERROR: a valid license file is required for FreeSurfer to run. '
            'FMRIPREP looked for an existing license file at several paths, in this '
            'order: 1) command line argument ``--fs-license-file``; 2) ``$FS_LICENSE`` '
            'environment variable; and 3) the ``$FREESURFER_HOME/license.txt`` path. '
            'Get it (for free) by registering at https://'
            'surfer.nmr.mgh.harvard.edu/registration.html')
    os.environ['FS_LICENSE'] = license_file

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    logger.setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    errno = 0

    # Call build_workflow(opts, retval)
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(opts, retval))
        p.start()
        p.join()

        if p.exitcode != 0:
            sys.exit(p.exitcode)

        fmriprep_wf = retval['workflow']
        plugin_settings = retval['plugin_settings']
        bids_dir = retval['bids_dir']
        output_dir = retval['output_dir']
        work_dir = retval['work_dir']
        subject_list = retval['subject_list']
        run_uuid = retval['run_uuid']
        if not opts.notrack:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag('run_uuid', run_uuid)
                scope.set_tag('npart', len(subject_list))

        retcode = retval['return_code']

    if fmriprep_wf is None:
        sys.exit(1)

    if opts.write_graph:
        fmriprep_wf.write_graph(graph2use="colored", format='svg', simple_form=True)

    if opts.reports_only:
        sys.exit(int(retcode > 0))

    if opts.boilerplate:
        sys.exit(int(retcode > 0))

    # Sentry tracking
    if not opts.notrack:
        sentry_sdk.add_breadcrumb(message='fMRIPrep started', level='info')
        sentry_sdk.capture_message('fMRIPrep started', level='info')

    # Check workflow for missing commands
    missing = check_deps(fmriprep_wf)
    if missing:
        print("Cannot run fMRIPrep. Missing dependencies:")
        for iface, cmd in missing:
            print("\t{} (Interface: {})".format(cmd, iface))
        sys.exit(2)

    # Clean up master process before running workflow, which may create forks
    gc.collect()
    try:
        fmriprep_wf.run(**plugin_settings)
    except RuntimeError as e:
        errno = 1
        if "Workflow did not execute cleanly" not in str(e):
            sentry_sdk.capture_exception(e)
            raise
    finally:
        # Generate reports phase
        errno += generate_reports(subject_list, output_dir, work_dir, run_uuid,
                                  sentry_sdk=sentry_sdk)
        write_derivative_description(bids_dir, str(Path(output_dir) / 'fmriprep'))

    if not opts.notrack and errno == 0:
        sentry_sdk.capture_message('fMRIPrep finished without errors', level='info')
    sys.exit(int(errno > 0))


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
    from subprocess import check_call, CalledProcessError, TimeoutExpired
    from pkg_resources import resource_filename as pkgrf

    from nipype import logging, config as ncfg
    from ..__about__ import __version__
    from ..workflows.base import init_fmriprep_wf
    from ..utils.bids import collect_participants
    from ..viz.reports import generate_reports

    logger = logging.getLogger('nipype.workflow')

    INIT_MSG = """
    Running fMRIPREP version {version}:
      * BIDS dataset path: {bids_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {uuid}.
    """.format

    output_spaces = opts.output_space or []

    # Validity of some inputs
    # ERROR check if use_aroma was specified, but the correct template was not
    if opts.use_aroma and (opts.template != 'MNI152NLin2009cAsym' or
                           'template' not in output_spaces):
        output_spaces.append('template')
        logger.warning(
            'Option "--use-aroma" requires functional images to be resampled to MNI space. '
            'The argument "template" has been automatically added to the list of output '
            'spaces (option "--output-space").'
        )

    # Check output_space
    if 'template' not in output_spaces and (opts.use_syn_sdc or opts.force_syn):
        msg = ['SyN SDC correction requires T1 to MNI registration, but '
               '"template" is not specified in "--output-space" arguments.',
               'Option --use-syn will be cowardly dismissed.']
        if opts.force_syn:
            output_spaces.append('template')
            msg[1] = (' Since --force-syn has been requested, "template" has been added to'
                      ' the "--output-space" list.')
        logger.warning(' '.join(msg))

    # Set up some instrumental utilities
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())

    # First check that bids_dir looks like a BIDS folder
    bids_dir = os.path.abspath(opts.bids_dir)
    subject_list = collect_participants(
        bids_dir, participant_label=opts.participant_label)

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
        logger.warning(
            'Per-process threads (--omp-nthreads=%d) exceed total '
            'threads (--nthreads/--n_cpus=%d)', omp_nthreads, nthreads)

    # Set up directories
    output_dir = op.abspath(opts.output_dir)
    log_dir = op.join(output_dir, 'fmriprep', 'logs')
    work_dir = op.abspath(opts.work_dir or 'work')  # Set work/ as default

    # Check and create output and working directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {
            'log_directory': log_dir,
            'log_to_file': True
        },
        'execution': {
            'crashdump_dir': log_dir,
            'crashfile_format': 'txt',
            'get_linked_libs': False,
            'stop_on_first_crash': opts.stop_on_first_crash or opts.work_dir is None,
        },
        'monitoring': {
            'enabled': opts.resource_monitor,
            'sample_frequency': '0.5',
            'summary_append': True,
        }
    })

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    retval['return_code'] = 0
    retval['plugin_settings'] = plugin_settings
    retval['bids_dir'] = bids_dir
    retval['output_dir'] = output_dir
    retval['work_dir'] = work_dir
    retval['subject_list'] = subject_list
    retval['run_uuid'] = run_uuid
    retval['workflow'] = None

    # Called with reports only
    if opts.reports_only:
        logger.log(25, 'Running --reports-only on participants %s', ', '.join(subject_list))
        if opts.run_uuid is not None:
            run_uuid = opts.run_uuid
        retval['return_code'] = generate_reports(subject_list, output_dir, work_dir, run_uuid)
        return retval

    # Build main workflow
    logger.log(25, INIT_MSG(
        version=__version__,
        bids_dir=bids_dir,
        subject_list=subject_list,
        uuid=run_uuid)
    )

    template_out_grid = opts.template_resampling_grid
    if opts.output_grid_reference is not None:
        logger.warning(
            'Option --output-grid-reference is deprecated, please use '
            '--template-resampling-grid')
        template_out_grid = template_out_grid or opts.output_grid_reference
    if opts.debug:
        logger.warning('Option --debug is deprecated and has no effect')

    retval['workflow'] = init_fmriprep_wf(
        subject_list=subject_list,
        task_id=opts.task_id,
        run_uuid=run_uuid,
        ignore=opts.ignore,
        debug=opts.sloppy,
        low_mem=opts.low_mem,
        anat_only=opts.anat_only,
        longitudinal=opts.longitudinal,
        t2s_coreg=opts.t2s_coreg,
        omp_nthreads=omp_nthreads,
        skull_strip_template=opts.skull_strip_template,
        skull_strip_fixed_seed=opts.skull_strip_fixed_seed,
        work_dir=work_dir,
        output_dir=output_dir,
        bids_dir=bids_dir,
        freesurfer=opts.run_reconall,
        output_spaces=output_spaces,
        template=opts.template,
        medial_surface_nan=opts.medial_surface_nan,
        cifti_output=opts.cifti_output,
        template_out_grid=template_out_grid,
        hires=opts.hires,
        use_bbr=opts.use_bbr,
        bold2t1w_dof=opts.bold2t1w_dof,
        fmap_bspline=opts.fmap_bspline,
        fmap_demean=opts.fmap_no_demean,
        use_syn=opts.use_syn_sdc,
        force_syn=opts.force_syn,
        use_aroma=opts.use_aroma,
        aroma_melodic_dim=opts.aroma_melodic_dimensionality,
        ignore_aroma_err=opts.ignore_aroma_denoising_errors,
    )
    retval['return_code'] = 0

    logs_path = Path(output_dir) / 'fmriprep' / 'logs'
    boilerplate = retval['workflow'].visit_desc()
    (logs_path / 'CITATION.md').write_text(boilerplate)
    logger.log(25, 'Works derived from this fMRIPrep execution should '
               'include the following boilerplate:\n\n%s', boilerplate)

    # Generate HTML file resolving citations
    cmd = ['pandoc', '-s', '--bibliography',
           pkgrf('fmriprep', 'data/boilerplate.bib'),
           '--filter', 'pandoc-citeproc',
           str(logs_path / 'CITATION.md'),
           '-o', str(logs_path / 'CITATION.html')]
    try:
        check_call(cmd, timeout=10)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning('Could not generate CITATION.html file:\n%s',
                       ' '.join(cmd))

    # Generate LaTex file resolving citations
    cmd = ['pandoc', '-s', '--bibliography',
           pkgrf('fmriprep', 'data/boilerplate.bib'),
           '--natbib', str(logs_path / 'CITATION.md'),
           '-o', str(logs_path / 'CITATION.tex')]
    try:
        check_call(cmd, timeout=10)
    except (FileNotFoundError, CalledProcessError, TimeoutExpired):
        logger.warning('Could not generate CITATION.tex file:\n%s',
                       ' '.join(cmd))
    return retval


if __name__ == '__main__':
    raise RuntimeError("fmriprep/cli/run.py should not be run directly;\n"
                       "Please `pip install` fmriprep and use the `fmriprep` command")
