#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-10-05 15:03:18
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

def main():
    """Entry point"""
    from fmriprep import __version__
    parser = ArgumentParser(description='fMRI Preprocessing workflow',
                            formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument('bids_dir', action='store', default=os.getcwd())
    parser.add_argument('output_dir', action='store',
                        default=op.join(os.getcwd(), 'out'))
    parser.add_argument('analysis_level', choices=['participant'])

    # optional arguments
    parser.add_argument('--participant_label', action='store', nargs='+')
    parser.add_argument('-v', '--version', action='version',
                        version='fmriprep v{}'.format(__version__))

    # Other options
    g_input = parser.add_argument_group('fMRIprep specific arguments')
    g_input.add_argument('-s', '--session-id', action='store', default='single_session')
    g_input.add_argument('-r', '--run-id', action='store', default='single_run')
    g_input.add_argument('--task-id', help='limit the analysis only ot one task', action='store')
    g_input.add_argument('-d', '--data-type', action='store', choices=['anat', 'func'])
    g_input.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')
    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')
    g_input.add_argument('--mem_mb', action='store', default=0,
                         type=int, help='try to limit the total amount of requested memory for all workflows to this number')
    g_input.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')
    g_input.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')
    g_input.add_argument('-w', '--work-dir', action='store',
                         default=op.join(os.getcwd(), 'work'))
    g_input.add_argument('--ignore', required=False,
                         action='store', choices=['fieldmaps'],
                         nargs="+", default=[],
                         help='In case the dataset includes fieldmaps but you chose not to take advantage of them.')
    g_input.add_argument('--reports-only', action='store_true', default=False,
                         help="only generate reports, don't run workflows. This will only rerun report aggregation, not reportlet generation for specific nodes.")
    g_input.add_argument('--skip-native', action='store_true',
                         default=False,
                         help="don't output timeseries in native space")

    #  ANTs options
    g_ants = parser.add_argument_group('specific settings for ANTs registrations')
    g_ants.add_argument('--ants-nthreads', action='store', type=int, default=0,
                        help='number of threads that will be set in ANTs processes')
    g_ants.add_argument('--skull-strip-ants', dest="skull_strip_ants",
                        action='store_true',
                        help='use ANTs-based skull-stripping (default, slow))')
    g_ants.add_argument('--no-skull-strip-ants', dest="skull_strip_ants",
                        action='store_false',
                        help="don't use ANTs-based skull-stripping (use  AFNI instead, fast)")
    g_ants.set_defaults(skull_strip_ants=True)

    # FreeSurfer options
    g_fs = parser.add_argument_group('settings for FreeSurfer preprocessing')
    g_fs.add_argument('--no-freesurfer', action='store_false', dest='freesurfer',
                      help='disable FreeSurfer preprocessing')

    opts = parser.parse_args()
    create_workflow(opts)


def create_workflow(opts):
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

    if opts.reports_only:
        run_reports(settings['output_dir'])
        sys.exit()

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

    logger.info('Subject list: %s', ', '.join(subject_list))

    # Build main workflow and run
    preproc_wf = base_workflow_enumerator(subject_list, task_id=opts.task_id,
                                          settings=settings, run_uuid=run_uuid)
    preproc_wf.base_dir = settings['work_dir']

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
