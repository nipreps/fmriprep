#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2017-03-03 17:57:53
"""
fMRI preprocessing workflow
=====
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path as op
import glob
import sys
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count

import logging

# set up logger
LOGGER = logging.getLogger('interfaces')

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

    # optional arguments
    parser.add_argument('--participant_label', action='store', nargs='+')
    parser.add_argument('-v', '--version', action='version',
                        version='fmriprep v{}'.format(__version__))

    # Other options
    g_input = parser.add_argument_group('fMRIprep specific arguments')
    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')
    g_input.add_argument('--mem_mb', action='store', default=0,
                         type=int, help='try to limit requested memory to this number')
    g_input.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')
    g_input.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')
    g_input.add_argument('-w', '--work-dir', action='store',
                         default=op.join(os.getcwd(), 'work'))

    #  ANTs options
    g_ants = parser.add_argument_group('specific settings for ANTs registrations')
    g_ants.add_argument('--ants-nthreads', action='store', type=int, default=0,
                        help='number of threads that will be set in ANTs processes')

    opts = parser.parse_args()
    create_workflow(opts)


def create_workflow(opts):
    from nipype import config as ncfg
    from fmriprep.utils import make_folder
    from fmriprep.viz.reports import run_reports
    from fmriprep.workflows.base import base_workflow_enumerator

    errno = 0

    settings = {
        'bids_root': op.abspath(opts.bids_dir),
        'write_graph': opts.write_graph,
        'nthreads': opts.nthreads,
        'mem_mb': opts.mem_mb,
        'ants_nthreads': opts.ants_nthreads,
        'output_dir': op.abspath(opts.output_dir),
        'work_dir': op.abspath(opts.work_dir),
    }

    log_dir = op.join(settings['output_dir'], 'log')
    derivatives = op.join(settings['output_dir'], 'derivatives')

    # Check and create output and working directories
    # Using make_folder to prevent https://github.com/poldracklab/mriqc/issues/111
    make_folder(settings['output_dir'])
    make_folder(settings['work_dir'])
    make_folder(derivatives)
    make_folder(log_dir)

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
            if settings['mem_mb']:
                plugin_settings['plugin_args']['memory_gb'] = settings['mem_mb']/1024

    if settings['ants_nthreads'] == 0:
        settings['ants_nthreads'] = cpu_count()

    # Determine subjects to be processed
    subject_list = opts.participant_label

    if subject_list is None or not subject_list:
        subject_list = [op.basename(subdir)[4:] for subdir in glob.glob(
            op.join(settings['bids_root'], 'sub-*'))]

    LOGGER.info('Subject list: %s', ', '.join(subject_list))

    # Build main workflow and run
    preproc_wf = fmap_enumerator(subject_list, settings=settings)
    preproc_wf.base_dir = settings['work_dir']
    try:
        preproc_wf.run(**plugin_settings)
    except RuntimeError:
        errno = 1

    if opts.write_graph:
        preproc_wf.write_graph(graph2use="colored", format='svg',
                               simple_form=True)

    # run_reports(settings['output_dir'])

    sys.exit(errno)


def fmap_enumerator(subject_list, settings):
    from nipype import logging
    from copy import deepcopy
    from time import strftime
    from nipype.pipeline import engine as pe

    workflow = pe.Workflow(name='workflow_enumerator')
    for subject_id in subject_list:
        fmap_wf = sbref_correct(subject_id, settings=settings)
        cur_time = strftime('%Y%m%d-%H%M%S')
        fmap_wf.config['execution']['crashdump_dir'] = (
            os.path.join(settings['output_dir'], 'log', subject_id, cur_time)
        )
        for node in fmap_wf._get_all_nodes():
            node.config = deepcopy(fmap_wf.config)
        workflow.add_nodes([fmap_wf])

    return workflow

def sbref_correct(subject_id, settings):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import fsl
    from fmriprep.utils.misc import collect_bids_data
    from fmriprep.interfaces.bids import BIDSDataGrabber, ReadSidecarJSON
    from fmriprep.interfaces.hmc import MotionCorrection
    from fmriprep.workflows.fieldmap import fmap_estimator
    from fmriprep.workflows.sbref import sbref_preprocess

    def _first(inlist):
        if isinstance(inlist, (list, tuple)):
            return inlist[0]
        return inlist

    subject_data = collect_bids_data(settings['bids_root'], subject_id)
    wfname = 'sbref_correct_%s' % subject_id.replace('sub-', '')
    wf = pe.Workflow(name=wfname)
    print(wfname)

    bidssrc = pe.Node(BIDSDataGrabber(subject_data=subject_data, sort=True),
                      name='BIDSDatasource')
    estimator = fmap_estimator(subject_data, settings=settings)
    sbrefprep = sbref_preprocess(settings=settings)

    wf.connect([
        (bidssrc, sbrefprep, [('sbref', 'sbref')]),
        (estimator, sbrefprep, [(('outputnode.fmap', _first), 'inputnode.fmap'),
                                (('outputnode.fmap_ref', _first), 'inputnode.fmap_ref'),
                                (('outputnode.fmap_mask', _first), 'inputnode.fmap_mask')])
    ])

    return wf


if __name__ == '__main__':
    main()
