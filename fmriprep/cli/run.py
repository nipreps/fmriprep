#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fMRI preprocessing workflow
=====
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
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
                             'FMRPREP (see BIDS-Apps specification).')

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
                         help='number of threads')
    g_perfm.add_argument('--mem_mb', action='store', default=0, type=int,
                         help='upper bound memory limit for FMRIPREP processes')
    g_perfm.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')

    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--ignore', required=False, action='store', choices=['fieldmaps',
                                                             'slicetiming'], nargs="+", default=[],
        help='ignore selected aspects of the input dataset to disable related'
             'parts of the workflow')
    g_conf.add_argument('--skip-native', action='store_true', default=False,
                        help="don't output timeseries in native space")

    #  ANTs options
    g_ants = parser.add_argument_group('Specific options for ANTs registrations')
    g_ants.add_argument('--ants-nthreads', action='store', type=int, default=0,
                        help='number of threads that will be set in ANTs processes')
    g_ants.add_argument('--skull-strip-ants', dest="skull_strip_ants", action='store_true',
                        help='use ANTs-based skull-stripping (default, slow))')
    g_ants.add_argument('--no-skull-strip-ants', dest="skull_strip_ants", action='store_false',
                        help="don't use ANTs-based skull-stripping (use  AFNI instead, fast)")
    g_ants.set_defaults(skull_strip_ants=True)

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
    from fmriprep.workflows.base import base_workflow_enumerator

    errno = 0

    settings = {
        'bids_root': op.abspath(opts.bids_dir),
        'write_graph': opts.write_graph,
        'nthreads': opts.nthreads,
        'mem_mb': opts.mem_mb,
        'debug': opts.debug,
        'ants_nthreads': opts.ants_nthreads,
        'skull_strip_ants': opts.skull_strip_ants,
        'output_dir': op.abspath(opts.output_dir),
        'work_dir': op.abspath(opts.work_dir),
        'ignore': opts.ignore,
        'skip_native': opts.skip_native,
        'freesurfer': opts.freesurfer,
        'hires': opts.hires,
        'reportlets_dir': op.join(op.abspath(opts.work_dir), 'reportlets'),
    }

    # set up logger
    logger = logging.getLogger('cli')

    if opts.debug:
        settings['ants_t1-mni_settings'] = 't1-mni_registration_test'
        logger.setLevel(logging.DEBUG)

    run_uuid = strftime('%Y%m%d-%H%M%S_') + str(uuid.uuid4())

    # Check and create output and working directories
    # Using make_folder to prevent https://github.com/poldracklab/mriqc/issues/111
    make_folder(settings['output_dir'])
    make_folder(settings['work_dir'])

    # nipype plugin configuration
    plugin_settings = {'plugin': 'Linear'}
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
    else:
        # Setup multiprocessing
        if settings['nthreads'] == 0:
            settings['nthreads'] = cpu_count()

        if settings['nthreads'] > 1:
            plugin_settings['plugin'] = 'MultiProc'
            plugin_settings['plugin_args'] = {'n_procs': settings['nthreads']}
            if settings['mem_mb']:
                plugin_settings['plugin_args']['memory_gb'] = settings['mem_mb']/1024

    if settings['ants_nthreads'] == 0:
        settings['ants_nthreads'] = cpu_count()

    # Determine subjects to be processed
    subject_list = opts.participant_label

    if subject_list is None or not subject_list:
        subject_list = [op.basename(subdir)[4:] for subdir in glob.glob(
            op.join(settings['bids_root'], 'sub-*'))]
    else:
        subject_list = [sub[4:] if sub.startswith('sub-') else sub for sub in subject_list]

    logger.info('Subject list: %s', ', '.join(subject_list))

    # Build main workflow and run
    preproc_wf = base_workflow_enumerator(subject_list, task_id=opts.task_id,
                                          settings=settings, run_uuid=run_uuid)
    preproc_wf.base_dir = settings['work_dir']

    if opts.reports_only:
        if opts.write_graph:
            preproc_wf.write_graph(graph2use="colored", format='svg',
                                   simple_form=True)

        for subject_label in subject_list:
            run_reports(settings['reportlets_dir'],
                        settings['output_dir'],
                        subject_label, run_uuid=run_uuid)
        sys.exit()

    try:
        preproc_wf.run(**plugin_settings)
    except RuntimeError as e:
        if "Workflow did not execute cleanly" in str(e):
            errno = 1
        else:
            raise(e)

    if opts.write_graph:
        preproc_wf.write_graph(graph2use="colored", format='svg',
                               simple_form=True)

    report_errors = 0
    for subject_label in subject_list:
        report_errors += run_reports(settings['reportlets_dir'],
                                     settings['output_dir'],
                                     subject_label, run_uuid=run_uuid)
    if errno == 1:
        assert(report_errors > 0)

    sys.exit(errno)

if __name__ == '__main__':
    main()
