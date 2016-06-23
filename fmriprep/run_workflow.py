#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-06-03 12:14:53
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
from multiprocessing import cpu_count
import os
import os.path as op
import matplotlib
matplotlib.use('Agg')

def preproc_and_reports(imaging_data, name='preproc_and_reports', settings=None):
    from nipype.pipeline import engine as pe
    from fmriprep.workflows import fmri_preprocess_single
    from fmriprep.viz.pipeline_reports import generate_report_workflow

    preproc_wf = fmri_preprocess_single(settings=settings)
#    report_wf = generate_report_workflow()

#    connector_wf = pe.Workflow(name=name)
#    connector_wf.connect([
#        (preproc_wf, report_wf, [
#            ('outputnode.t1_2_mni', 'inputnode.t1_2_mni'),
#            ('outputnode.stripped_t1', 'inputnode.t1_brain'),
#            ('outputnode.t1', 'inputnode.t1'),
#            ('outputnode.t1_segmentation', 'inputnode.t1_segmentation'),
#            ('outputnode.t1_wm_seg', 'inputnode.t1_wm_seg'),
#            ('outputnode.fieldmap', 'inputnode.fieldmap'),
#            ('outputnode.corrected_sbref', 'inputnode.corrected_sbref'),
#            ('outputnode.fmap_mag', 'inputnode.fmap_mag'),
#            ('outputnode.fmap_mag_brain', 'inputnode.fmap_mag_brain'),
#            ('outputnode.stripped_epi', 'inputnode.stripped_epi'),
#            ('outputnode.corrected_epi_mean', 'inputnode.corrected_epi_mean'),
#            ('outputnode.sbref_brain', 'inputnode.sbref_brain'),
#            ('inputnode.epi', 'inputnode.raw_epi'),
#            ('inputnode.sbref', 'inputnode.sbref')
#        ])
#    ])

    # Set inputnode of the full-workflow
    for key in imaging_data.keys():
        setattr(preproc_wf.inputs.inputnode, key, imaging_data[key])

    return preproc_wf
#    return connector_wf



def main():
    """Entry point"""
    from nipype import config as ncfg
    from nipype.pipeline import engine as pe
    from fmriprep import __version__
    from fmriprep.utils.misc import get_subject

    parser = ArgumentParser(description='fMRI Preprocessing workflow',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-B', '--bids-root', action='store', default=os.getcwd())
    g_input.add_argument('-S', '--subject-id', action='store', required=True)
    g_input.add_argument('-s', '--session-id', action='store', default='single_session')
    g_input.add_argument('-r', '--run-id', action='store', default='single_run')
    g_input.add_argument('-d', '--data-type', action='store', choices=['anat', 'func'])
    g_input.add_argument('-v', '--version', action='store_true', default=False,
                         help='Show current fmriprep version')
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

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output-dir', action='store')
    g_outputs.add_argument('-w', '--work-dir', action='store')

    opts = parser.parse_args()

    if opts.version:
        print('fmriprep version ' + __version__)
        exit(0)

    settings = {
        'bids_root': op.abspath(opts.bids_root),
        'output_dir': os.getcwd(),
        'write_graph': opts.write_graph,
        'nthreads': opts.nthreads,
        'debug': opts.debug,
        'skull_strip_ants': opts.skull_strip_ants
    }

    if opts.output_dir:
        settings['output_dir'] = op.abspath(opts.output_dir)

    if not op.exists(settings['output_dir']):
        os.makedirs(settings['output_dir'])

    # Logging and work directory
    settings['work_dir'] = (op.abspath(opts.work_dir)
                            if opts.work_dir
                            else op.abspath('.'))

    log_dir = op.join(settings['work_dir'], 'log')
    if not op.exists(log_dir):
        os.makedirs(log_dir)

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

    # Retrieve BIDS data
    imaging_data = get_subject(settings['bids_root'], opts.subject_id)

    # Build main workflow and run
    workflow = preproc_and_reports(imaging_data, settings=settings)
    workflow.base_dir = settings['work_dir']

    if opts.write_graph:
        workflow.write_graph()
    workflow.run(**plugin_settings)

# # This might be usefull in some future, but in principle we want single-subject runs.
# def fmri_preprocess_multiple(subject_list, plugin_settings, settings=None):
#     for subject in subject_list:
#         for session in subject_list[subject]:
#             imaging_data = subject_list[subject][session]
#             workflow = fmri_preprocess_single(imaging_data=imaging_data, settings=settings)
#             workflow.base_dir = settings['work_dir']
#             workflow.run(**plugin_settings)
#             return


if __name__ == '__main__':
    main()
