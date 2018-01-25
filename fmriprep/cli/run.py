#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fMRI preprocessing workflow
=====
"""

import os
import os.path as op
import logging
import sys
import gc
import uuid
import warnings
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count
from time import strftime
import nibabel

nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT = 'auto'

logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')


def _warn_redirect(message, category, filename, lineno, file=None, line=None):
    logger.warning('Captured warning (%s): %s', category, message)


def get_parser():
    """Build parser object"""
    from ..info import __version__

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
                        help='one or more participant identifiers (the sub- prefix can be '
                             'removed)')
    # Re-enable when option is actually implemented
    # g_bids.add_argument('-s', '--session-id', action='store', default='single_session',
    #                     help='select a specific session to be processed')
    # Re-enable when option is actually implemented
    # g_bids.add_argument('-r', '--run-id', action='store', default='single_run',
    #                     help='select a specific run to be processed')
    g_bids.add_argument('-t', '--task-id', action='store',
                        help='select a specific task to be processed')

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')
    g_perfm.add_argument('--nthreads', '--n_cpus', '-n-cpus', action='store', default=0, type=int,
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
    g_perfm.add_argument('--ignore-aroma-denoising-errors', action='store_true',
                         default=False,
                         help='ignores the errors ICA_AROMA returns when there '
                              'are no components classified as either noise or '
                              'signal')
    g_perfm.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
                         help="increases log verbosity for each occurence, debug level is -vvv")

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore', required=False, action='store', nargs="+", default=[],
        choices=['fieldmaps', 'slicetiming'],
        help='ignore selected aspects of the input dataset to disable corresponding '
             'parts of the workflow')
    g_conf.add_argument(
        '--longitudinal', action='store_true',
        help='treat dataset as longitudinal - may increase runtime')
    g_conf.add_argument(
        '--t2s-coreg', action='store_true',
        help='If provided with multi-echo BOLD dataset, create T2*-map and perform '
             'T2*-driven coregistration. When multi-echo data is provided and this '
             'option is not enabled, standard EPI-T1 coregistration is performed '
             'using the middle echo.')
    g_conf.add_argument('--bold2t1w-dof', action='store', default=9, choices=[6, 9, 12], type=int,
                        help='Degrees of freedom when registering BOLD to T1w images. '
                             '9 (rotation, translation, and scaling) is used by '
                             'default to compensate for field inhomogeneities.')
    g_conf.add_argument(
        '--output-space', required=False, action='store',
        choices=['T1w', 'template', 'fsnative', 'fsaverage', 'fsaverage6', 'fsaverage5'],
        nargs='+', default=['template', 'fsaverage5'],
        help='volume and surface spaces to resample functional series into\n'
             ' - T1w: subject anatomical volume\n'
             ' - template: normalization target specified by --template\n'
             ' - fsnative: individual subject surface\n'
             ' - fsaverage*: FreeSurfer average meshes'
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
        '--output-grid-reference', required=False, action='store', default=None,
        help='Grid reference image for resampling BOLD files to volume template space. '
             'It determines the field of view and resolution of the output images, '
             'but is not used in normalization.')
    g_conf.add_argument(
        '--medial-surface-nan', required=False, action='store', default=False,
        help='Replace medial wall values with NaNs on functional GIFTI files. Only '
        'performed for GIFTI files mapped to a freesurfer subject (fsaverage or fsnative).')

    # ICA_AROMA options
    g_aroma = parser.add_argument_group('Specific options for running ICA_AROMA')
    g_aroma.add_argument('--use-aroma', action='store_true', default=False,
                         help='add ICA_AROMA to your preprocessing stream')
    #  ANTs options
    g_ants = parser.add_argument_group('Specific options for ANTs registrations')
    g_ants.add_argument('--skull-strip-template', action='store', default='OASIS',
                        choices=['OASIS', 'NKI'],
                        help='select ANTs skull-stripping template (default: OASIS))')

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
    g_fs.add_argument('--fs-no-reconall', '--no-freesurfer',
                      action='store_false', dest='run_reconall',
                      help='disable FreeSurfer surface preprocessing.'
                      ' Note : `--no-freesurfer` is deprecated and will be removed in 1.2.'
                      ' Use `--fs-no-reconall` instead.')
    g_fs.add_argument('--no-submm-recon', action='store_false', dest='hires',
                      help='disable sub-millimeter (hires) reconstruction')
    g_fs.add_argument(
        '--fs-license-file', metavar='PATH', type=os.path.abspath,
        help='Path to FreeSurfer license key file. Get it (for free) by registering'
             ' at https://surfer.nmr.mgh.harvard.edu/registration.html')

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

    return parser


def main():
    """Entry point"""
    from niworkflows.nipype import logging as nlogging
    from multiprocessing import set_start_method, Process, Manager
    from ..viz.reports import generate_reports
    set_start_method('forkserver')

    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()

    # FreeSurfer license
    default_license = op.join(os.getenv('FREESURFER_HOME', ''), 'license.txt')
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
    nlogging.getLogger('workflow').setLevel(log_level)
    nlogging.getLogger('interface').setLevel(log_level)
    nlogging.getLogger('utils').setLevel(log_level)

    errno = 0

    # Call build_workflow(opts, retval)
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(opts, retval))
        p.start()
        p.join()

        fmriprep_wf = retval['workflow']
        plugin_settings = retval['plugin_settings']
        output_dir = retval['output_dir']
        work_dir = retval['work_dir']
        subject_list = retval['subject_list']
        run_uuid = retval['run_uuid']
        retcode = retval['return_code']

    if fmriprep_wf is None:
        sys.exit(1)

    if opts.write_graph:
        fmriprep_wf.write_graph(graph2use="colored", format='svg', simple_form=True)

    if opts.reports_only:
        sys.exit(int(retcode > 0))

    # Clean up master process before running workflow, which may create forks
    gc.collect()
    try:
        fmriprep_wf.run(**plugin_settings)
    except RuntimeError as e:
        if "Workflow did not execute cleanly" in str(e):
            errno = 1
        else:
            raise

    # Generate reports phase
    errno += generate_reports(subject_list, output_dir, work_dir, run_uuid)
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
    from niworkflows.nipype import logging, config as ncfg
    from ..info import __version__
    from ..workflows.base import init_fmriprep_wf
    from ..utils.bids import collect_participants
    from ..viz.reports import generate_reports

    logger = logging.getLogger('workflow')

    INIT_MSG = """
    Running fMRIPREP version {version}:
      * BIDS dataset path: {bids_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {uuid}.
    """.format

    # Validity of some inputs
    # ERROR check if use_aroma was specified, but the correct template was not
    if opts.use_aroma and (opts.template != 'MNI152NLin2009cAsym' or
                           'template' not in opts.output_space):
        raise RuntimeError('ERROR: --use-aroma requires functional images to be resampled to '
                           'MNI152NLin2009cAsym.\n'
                           '\t--template must be set to "MNI152NLin2009cAsym" (was: "{}")\n'
                           '\t--output-space list must include "template" (was: "{}")'.format(
                               opts.template, ' '.join(opts.output_space)))
    # Check output_space
    if 'template' not in opts.output_space and (opts.use_syn_sdc or opts.force_syn):
        msg = ('SyN SDC correction requires T1 to MNI registration, but '
               '"template" is not specified in "--output-space" arguments')
        if opts.force_syn:
            raise RuntimeError(msg)
        logger.warning(msg)

    # Set up some instrumental utilities
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())

    # First check that bids_dir looks like a BIDS folder
    bids_dir = op.abspath(opts.bids_dir)
    subject_list = collect_participants(
        bids_dir, participant_label=opts.participant_label)

    # Setting up MultiProc
    nthreads = opts.nthreads
    if nthreads < 1:
        nthreads = cpu_count()

    plugin_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {
            'n_procs': nthreads,
            'raise_insufficient': False,
            'maxtasksperchild': 1,
        }
    }

    if opts.mem_mb:
        plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb / 1024

    # Overload plugin_settings if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)

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

    retval['workflow'] = init_fmriprep_wf(
        subject_list=subject_list,
        task_id=opts.task_id,
        run_uuid=run_uuid,
        ignore=opts.ignore,
        debug=opts.debug,
        low_mem=opts.low_mem,
        anat_only=opts.anat_only,
        longitudinal=opts.longitudinal,
        t2s_coreg=opts.t2s_coreg,
        omp_nthreads=omp_nthreads,
        skull_strip_template=opts.skull_strip_template,
        work_dir=work_dir,
        output_dir=output_dir,
        bids_dir=bids_dir,
        freesurfer=opts.run_reconall,
        output_spaces=opts.output_space,
        template=opts.template,
        medial_surface_nan=opts.medial_surface_nan,
        output_grid_ref=opts.output_grid_reference,
        hires=opts.hires,
        use_bbr=opts.use_bbr,
        bold2t1w_dof=opts.bold2t1w_dof,
        fmap_bspline=opts.fmap_bspline,
        fmap_demean=opts.fmap_no_demean,
        use_syn=opts.use_syn_sdc,
        force_syn=opts.force_syn,
        use_aroma=opts.use_aroma,
        ignore_aroma_err=opts.ignore_aroma_denoising_errors,
    )
    retval['return_code'] = 0
    return retval


if __name__ == '__main__':
    raise RuntimeError("fmriprep/cli/run.py should not be run directly;\n"
                       "Please `pip install` fmriprep and use the `fmriprep` command")
