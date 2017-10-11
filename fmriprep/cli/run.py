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
import uuid
import warnings
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count
from time import strftime

logging.addLevelName(25, 'INFO')  # Add a new level between INFO and WARNING
logger = logging.getLogger('cli')
logger.setLevel(25)

INIT_MSG = """
Running fMRIPREP version {version}:
  * Participant list: {subject_list}.
  * Run identifier: {uuid}.
""".format


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
    parser.add_argument('-v', '--version', action='version', version=verstr)

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

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore', required=False, action='store', nargs="+", default=[],
        choices=['fieldmaps', 'slicetiming'],
        help='ignore selected aspects of the input dataset to disable corresponding '
             'parts of the workflow')
    g_conf.add_argument(
        '--longitudinal', action='store_true',
        help='treat dataset as longitudinal - may increase runtime')
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
    # g_ants.add_argument('--skull-strip-ants', dest="skull_strip_ants", action='store_true',
    #                     help='use ANTs-based skull-stripping (default, slow))')
    # g_ants.add_argument('--no-skull-strip-ants', dest="skull_strip_ants", action='store_false',
    #                     help="don't use ANTs-based skull-stripping (use  AFNI instead, fast)")
    # g_ants.set_defaults(skull_strip_ants=True)
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
    g_fs.add_argument('--no-freesurfer', action='store_false', dest='freesurfer',
                      help='disable FreeSurfer preprocessing')
    g_fs.add_argument('--no-submm-recon', action='store_false', dest='hires',
                      help='disable sub-millimeter (hires) reconstruction')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store', default='work',
                         help='path where intermediate results should be stored')
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

    return parser


def main():
    """Entry point"""
    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()
    if opts.debug:
        logger.setLevel(logging.DEBUG)

    # Validity of some inputs - OE should be done in parse_args?
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

    create_workflow(opts)


def create_workflow(opts):
    """Build workflow"""
    from niworkflows.nipype import config as ncfg
    from ..viz.reports import run_reports
    from ..workflows.base import init_fmriprep_wf
    from ..utils.bids import collect_participants
    from ..info import __version__

    # Set up some instrumental utilities
    errno = 0
    run_uuid = strftime('%Y%m%d-%H%M%S_') + str(uuid.uuid4())

    # First check that bids_dir looks like a BIDS folder
    bids_dir = op.abspath(opts.bids_dir)
    subject_list = collect_participants(
        bids_dir, participant_label=opts.participant_label)

    # Nipype plugin configuration
    plugin_settings = {'plugin': 'Linear'}
    nthreads = opts.nthreads
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
    else:
        # Setup multiprocessing
        nthreads = opts.nthreads
        if nthreads == 0:
            nthreads = cpu_count()

        if nthreads > 1:
            plugin_settings['plugin'] = 'MultiProc'
            plugin_settings['plugin_args'] = {'n_procs': nthreads}
            if opts.mem_mb:
                plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb / 1024

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    if 1 < nthreads < omp_nthreads:
        raise RuntimeError(
            'Per-process threads (--omp-nthreads={:d}) cannot exceed total '
            'threads (--nthreads/--n_cpus={:d})'.format(omp_nthreads, nthreads))

    # Set up directories
    output_dir = op.abspath(opts.output_dir)
    log_dir = op.join(output_dir, 'fmriprep', 'logs')
    work_dir = op.abspath(opts.work_dir)

    # Check and create output and working directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {'log_directory': log_dir, 'log_to_file': True},
        'execution': {'crashdump_dir': log_dir, 'crashfile_format': 'txt'},
    })

    # Called with reports only
    if opts.reports_only:
        logger.log(25, 'Running --reports-only on participants %s', ', '.join(subject_list))
        if opts.run_uuid is not None:
            run_uuid = opts.run_uuid
        report_errors = [
            run_reports(op.join(work_dir, 'reportlets'), output_dir, subject_label,
                        run_uuid=run_uuid)
            for subject_label in subject_list]
        sys.exit(int(sum(report_errors) > 0))

    # Build main workflow
    logger.log(25, INIT_MSG(
        version=__version__,
        subject_list=subject_list,
        uuid=run_uuid)
    )

    fmriprep_wf = init_fmriprep_wf(
        subject_list=subject_list,
        task_id=opts.task_id,
        run_uuid=run_uuid,
        ignore=opts.ignore,
        debug=opts.debug,
        low_mem=opts.low_mem,
        anat_only=opts.anat_only,
        longitudinal=opts.longitudinal,
        omp_nthreads=omp_nthreads,
        skull_strip_ants=True,
        skull_strip_template=opts.skull_strip_template,
        work_dir=work_dir,
        output_dir=output_dir,
        bids_dir=bids_dir,
        freesurfer=opts.freesurfer,
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

    if opts.write_graph:
        fmriprep_wf.write_graph(graph2use="colored", format='svg', simple_form=True)

    try:
        fmriprep_wf.run(**plugin_settings)
    except RuntimeError as e:
        if "Workflow did not execute cleanly" in str(e):
            errno = 1
        else:
            raise(e)

    # Generate reports phase
    report_errors = [run_reports(
        op.join(work_dir, 'reportlets'), output_dir, subject_label, run_uuid=run_uuid)
        for subject_label in subject_list]

    if sum(report_errors):
        logger.warning('Errors occurred while generating reports for participants: %s.',
                       ', '.join(['%s (%d)' % (subid, err)
                                  for subid, err in zip(subject_list, report_errors)]))

    errno += sum(report_errors)
    sys.exit(int(errno > 0))


if __name__ == '__main__':
    raise RuntimeError("fmriprep/cli/run.py should not be run directly;\n"
                       "Please `pip install` fmriprep and use the `fmriprep` command")
