#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-08-17 15:01:57
"""
fMRI preprocessing workflow
=====
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from lockfile import LockFile

import logging
from multiprocessing import cpu_count
import os
import os.path as op
import glob

def main():
    """Entry point"""
    from nipype import config as ncfg
    from nipype.pipeline import engine as pe
    from fmriprep import __version__
    from fmriprep.workflows import fmriprep_single

    parser = ArgumentParser(description='fMRI Preprocessing workflow',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    g_input.add_argument('bids_dir', action='store', default=os.getcwd())
    g_input.add_argument('output_dir', action='store',
                         default=op.join(os.getcwd(), 'out'))
    g_input.add_argument('analysis_level', choices=['participant'])

    # optional arguments
    g_input.add_argument('-S', '--subject-id', '--participant_label',
                         action='store', nargs='+')

    # fmriprep-specific arguments
    g_input.add_argument('-s', '--session-id', action='store', default='single_session')
    g_input.add_argument('-r', '--run-id', action='store', default='single_run')
    g_input.add_argument('-d', '--data-type', action='store', choices=['anat', 'func'])
    g_input.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')
    g_input.add_argument('--skull-strip-ants', action='store_true', default=False,
                         help='run debug version of workflow')

    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')
    g_input.add_argument(
        "--write-graph", action='store_true', default=False,
        help="Write workflow graph.")
    g_input.add_argument(
        "--use-plugin", action='store', default=None,
        help='nipype plugin configuration file')

    g_input.add_argument('-v', '--version', action='version',
                         version='fmriprep v{}'.format(__version__))

    opts = parser.parse_args()

    settings = {
        'bids_root': op.abspath(opts.bids_dir),
        'write_graph': opts.write_graph,
        'nthreads': opts.nthreads,
        'debug': opts.debug,
        'skull_strip_ants': opts.skull_strip_ants,
        'output_dir': op.abspath(opts.output_dir),
    }

    # set up logger
    logger = logging.getLogger('cli')

    if opts.debug:
        settings['ants_t1-mni_settings'] = 't1-mni_registration_test'
        logger.setLevel(logging.DEBUG)

    log_dir = op.join(settings['work_dir'], 'log')

    # Check and create output and working directories
    # Using locks to prevent https://github.com/poldracklab/mriqc/issues/111
    with LockFile('.fmriprep-folders-lock'):
        if not op.exists(settings['output_dir']):
            os.makedirs(settings['output_dir'])

        derivatives = op.join(settings['output_dir'], 'derivatives')
        if not op.exists(derivatives):
            os.makedirs(derivatives)

        if not op.exists(settings['work_dir']):
            os.makedirs(settings['work_dir'])

        if not op.exists(log_dir):
            os.makedirs(log_dir)

    logger.addHandler(logging.FileHandler(op.join(log_dir,'run_workflow')))

    # Warn for default work/output directories
    if (opts.work_dir == parser.get_default('work_dir') or
          opts.output_dir == parser.get_default('output_dir')):
        logger.warning("work-dir and/or output-dir not specified. Using " +
                        opts.work_dir + " and " + opts.output_dir)

    # Set nipype config
    ncfg.update_config({
        'logging': {'log_directory': log_dir, 'log_to_file': True},
        'execution': {'crashdump_dir': log_dir}
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
    subject_list = opts.subject_id

    if not subject_list or len(subject_list) == 0:
        subject_list = [op.basename(subdir)[4:] for subdir in glob.glob(
            op.join(settings['bids_root'], 'sub-*'))]

    logger.info("subject list: {}", ', '.join(subject_list))

    # Build main workflow and run
    preproc_wf = fmriprep_single(subject_list, settings=settings)
    preproc_wf.base_dir = settings['work_dir']
    preproc_wf.run(**plugin_settings)

    if opts.write_graph:
        preproc_wf.write_graph()



if __name__ == '__main__':
    main()
