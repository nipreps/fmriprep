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
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count


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
    g_input.add_argument('-d', '--data-type', action='store', choices=['anat', 'func'])
    g_input.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')
    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')
    g_input.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')
    g_input.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')
    g_input.add_argument('-w', '--work-dir', action='store',
                         default=op.join(os.getcwd(), 'work'))
    g_input.add_argument('-t', '--workflow-type', default='ds005', required=False,
                         action='store', choices=['ds005', 'ds054', 'HPC', 'spiral'],
                         help='''workflow type, a monkeypatch while it is not 
                         automatically identified''')

    # ANTs options
    g_ants = parser.add_argument_group('specific settings for ANTs registrations')
    g_ants.add_argument('--ants-nthreads', action='store', type=int,
                        help='number of threads that will be set in ANTs processes')
    g_ants.add_argument('--skull-strip-ants', action='store_true', default=False,
                        help='use ANTs-based skull-stripping')

    opts = parser.parse_args()
    create_workflow(opts)


def create_workflow(opts):
    import logging
    from nipype import config as ncfg
    from fmriprep.utils import make_folder
    from fmriprep.viz.reports import run_reports
    from fmriprep.workflows.base import base_workflow_enumerator

    settings = {
        'bids_root': op.abspath(opts.bids_dir),
        'write_graph': opts.write_graph,
        'nthreads': opts.nthreads,
        'debug': opts.debug,
        'skull_strip_ants': opts.skull_strip_ants,
        'output_dir': op.abspath(opts.output_dir),
        'work_dir': op.abspath(opts.work_dir)
    }

    # set up logger
    logger = logging.getLogger('cli')

    if opts.debug:
        settings['ants_t1-mni_settings'] = 't1-mni_registration_test'
        logger.setLevel(logging.DEBUG)

    if opts.ants_nthreads is not None:
        settings['ants_threads'] = opts.ants_nthreads

    log_dir = op.join(settings['output_dir'], 'log')
    derivatives = op.join(settings['output_dir'], 'derivatives')

    # Check and create output and working directories
    # Using make_folder to prevent https://github.com/poldracklab/mriqc/issues/111
    make_folder(settings['output_dir'])
    make_folder(settings['work_dir'])
    make_folder(derivatives)
    make_folder(log_dir)

    logger.addHandler(logging.FileHandler(op.join(log_dir, 'run_workflow')))

    # Set nipype config
    ncfg.update_config({
        'logging': {'log_directory': log_dir, 'log_to_file': True},
        'execution': {'crashdump_dir': log_dir, 
                      'remove_unnecessary_outputs': False}
    })

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

    # Determine subjects to be processed
    subject_list = opts.participant_label

    if subject_list is None or not subject_list:
        subject_list = [op.basename(subdir)[4:] for subdir in glob.glob(
            op.join(settings['bids_root'], 'sub-*'))]

    logger.info('Subject list: %s', ', '.join(subject_list))

    # Build main workflow and run
    preproc_wf = base_workflow_enumerator(subject_list, settings=settings)
    preproc_wf.base_dir = settings['work_dir']
    preproc_wf.run(**plugin_settings)

    if opts.write_graph:
        preproc_wf.write_graph()

    run_reports(settings['output_dir'])

if __name__ == '__main__':
    main()
