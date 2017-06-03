#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fMRI preprocessing workflow
=====
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os.path as op
import glob
import sys
import uuid
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count
from time import strftime


def get_parser():
    """Build parser object"""

    from fmriprep.info import __version__
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
    g_bids.add_argument('--participant_label', action='store', nargs='+',
                        help='one or more participant identifiers (the sub- prefix can be '
                             'removed)')
    g_bids.add_argument('-s', '--session-id', action='store', default='single_session',
                        help='select a specific session to be processed')
    g_bids.add_argument('-r', '--run-id', action='store', default='single_run',
                        help='select a specific run to be processed')
    g_bids.add_argument('-t', '--task-id', action='store',
                        help='select a specific task to be processed')

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')
    g_perfm.add_argument('--nthreads', action='store', default=0, type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--n_cpus', action='store', dest='nthreads', type=int,
                         help='total number of CPUs to use (alias for --nthreads)')
    g_perfm.add_argument('--omp-nthreads', action='store', type=int, default=0,
                         help='maximum number of threads per-process')
    g_perfm.add_argument('--mem_mb', action='store', default=0, type=int,
                         help='upper bound memory limit for FMRIPREP processes')
    g_perfm.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore', required=False, action='store', nargs="+", default=[],
        choices=['fieldmaps', 'slicetiming'],
        help='ignore selected aspects of the input dataset to disable corresponding '
             'parts of the workflow')
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
        '--template', required=False, action='store',
        choices=['MNI152NLin2009cAsym'], default='MNI152NLin2009cAsym',
        help='volume template space (default: MNI152NLin2009cAsym)')

    g_conf.add_argument(
        '--output-grid-reference', required=False, action='store', default=None,
        help='Grid reference image for resampling BOLD files to volume template space. '
             'It determines the field of view and resolution of the output images, '
             'but is not used in normalization.')

    #  ANTs options
    g_ants = parser.add_argument_group('Specific options for ANTs registrations')
    g_ants.add_argument('--skull-strip-ants', dest="skull_strip_ants", action='store_true',
                        help='use ANTs-based skull-stripping (default, slow))')
    g_ants.add_argument('--no-skull-strip-ants', dest="skull_strip_ants", action='store_false',
                        help="don't use ANTs-based skull-stripping (use  AFNI instead, fast)")
    g_ants.set_defaults(skull_strip_ants=True)

    # Fieldmap options
    g_fmap = parser.add_argument_group('Specific options for handling fieldmaps')
    g_fmap.add_argument('--fmap-bspline', action='store_true', default=False,
                        help='fit a B-Spline field using least-squares (experimental)')
    g_fmap.add_argument('--fmap-no-demean', action='store_false', default=True,
                        help='do not remove median (within mask) from fieldmap')

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
    g_other.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')

    return parser


def main():
    """Entry point"""
    opts = get_parser().parse_args()
    create_workflow(opts)


def create_workflow(opts):
    """Build workflow"""
    import logging
    from fmriprep.utils import make_folder
    from fmriprep.viz.reports import run_reports
    from fmriprep.workflows.base import init_fmriprep_wf

    errno = 0

    # set up logger
    logger = logging.getLogger('cli')

    if opts.debug:
        logger.setLevel(logging.DEBUG)

    run_uuid = strftime('%Y%m%d-%H%M%S_') + str(uuid.uuid4())

    # Check and create output and working directories
    # Using make_folder to prevent https://github.com/poldracklab/mriqc/issues/111
    make_folder(opts.output_dir)
    make_folder(opts.work_dir)

    # nipype plugin configuration
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
                plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb/1024

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    if 1 < nthreads < omp_nthreads:
        print('Per-process threads (--omp-nthreads={:d}) cannot exceed total '
              'threads (--nthreads/--n_cpus={:d})'.format(omp_nthreads, nthreads))
        sys.exit(1)

    # Determine subjects to be processed
    subject_list = opts.participant_label

    if subject_list is None or not subject_list:
        subject_list = [op.basename(subdir)[4:] for subdir in glob.glob(
            op.join(op.abspath(opts.bids_dir), 'sub-*'))]
    else:
        subject_list = [sub[4:] if sub.startswith('sub-') else sub for sub in subject_list]

    logger.info('Subject list: %s', ', '.join(subject_list))

    # Build main workflow and run
    reportlets_dir = op.join(op.abspath(opts.work_dir), 'reportlets')
    output_dir = op.abspath(opts.output_dir)
    bids_dir = op.abspath(opts.bids_dir)
    fmriprep_wf = init_fmriprep_wf(subject_list=subject_list,
                                   task_id=opts.task_id,
                                   run_uuid=run_uuid,
                                   ignore=opts.ignore,
                                   debug=opts.debug,
                                   omp_nthreads=omp_nthreads,
                                   skull_strip_ants=opts.skull_strip_ants,
                                   reportlets_dir=reportlets_dir,
                                   output_dir=output_dir,
                                   bids_dir=bids_dir,
                                   freesurfer=opts.freesurfer,
                                   output_spaces=opts.output_space,
                                   template=opts.template,
                                   output_grid_ref=opts.output_grid_reference,
                                   hires=opts.hires,
                                   bold2t1w_dof=opts.bold2t1w_dof,
                                   fmap_bspline=opts.fmap_bspline,
                                   fmap_demean=opts.fmap_no_demean)
    fmriprep_wf.base_dir = op.abspath(opts.work_dir)

    if opts.reports_only:
        if opts.write_graph:
            fmriprep_wf.write_graph(graph2use="colored", format='svg',
                                    simple_form=True)

        for subject_label in subject_list:
            run_reports(reportlets_dir,
                        output_dir,
                        subject_label, run_uuid=run_uuid)
        sys.exit()

    try:
        fmriprep_wf.run(**plugin_settings)
    except RuntimeError as e:
        if "Workflow did not execute cleanly" in str(e):
            errno = 1
        else:
            raise(e)

    if opts.write_graph:
        fmriprep_wf.write_graph(graph2use="colored", format='svg',
                                simple_form=True)

    report_errors = 0
    for subject_label in subject_list:
        report_errors += run_reports(reportlets_dir,
                                     output_dir,
                                     subject_label, run_uuid=run_uuid)
    if errno == 1:
        assert(report_errors > 0)

    sys.exit(errno)


if __name__ == '__main__':
    main()
