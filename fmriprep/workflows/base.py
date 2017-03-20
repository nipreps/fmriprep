#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base workflow: the workflow enumerator and the two orchestrators
(ds005 & ds054 -types of workflow)
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
from copy import deepcopy
from time import strftime

from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nipype.interfaces import utility as niu

from fmriprep.interfaces import BIDSDataGrabber, BIDSFreeSurferDir
from fmriprep.utils.misc import collect_bids_data, get_biggest_epi_file_size_gb
from fmriprep.workflows import confounds

from fmriprep.workflows.anatomical import t1w_preprocessing
from fmriprep.workflows.sbref import sbref_preprocess
from fmriprep.workflows.fieldmap import fmap_estimator
from fmriprep.workflows.epi import (epi_preprocess, ref_epi_t1_registration,
                                    epi_hmc, epi_sbref_registration, epi_mni_transformation)

def base_workflow_enumerator(subject_list, task_id, settings, run_uuid):
    workflow = pe.Workflow(name='workflow_enumerator')

    if settings['freesurfer']:
        fsdir = pe.Node(BIDSFreeSurferDir(), name='BIDSFreesurfer')
        fsdir.inputs.freesurfer_home = os.getenv('FREESURFER_HOME')
        fsdir.inputs.derivatives = os.path.join(settings['output_dir'])

    generated_list = []
    for subject in subject_list:
        generated_workflow = base_workflow_generator(subject, task_id=task_id,
                                                     settings=settings)
        if generated_workflow:
            generated_workflow.config['execution']['crashdump_dir'] = (
                os.path.join(settings['output_dir'], "fmriprep", "sub-" + subject, 'log', run_uuid)
            )
            for node in generated_workflow._get_all_nodes():
                node.config = deepcopy(generated_workflow.config)
            if settings['freesurfer']:
                workflow.connect(fsdir, 'subjects_dir',
                                 generated_workflow, 'inputnode.subjects_dir')
            else:
                workflow.add_nodes([generated_workflow])

    return workflow


def base_workflow_generator(subject_id, task_id, settings):
    subject_data = collect_bids_data(settings['bids_root'], subject_id, task_id)

    settings["biggest_epi_file_size_gb"] = get_biggest_epi_file_size_gb(subject_data['func'])

    if subject_data['t1w'] == []:
        raise Exception("No T1w images found for participant {}. "
                        "All workflows require T1w images.".format(subject_id))

    if all((subject_data['fmap'] != [],
            subject_data['sbref'] != [],
            "fieldmaps" not in settings['ignore'])):
        return basic_fmap_sbref_wf(subject_data, settings, name=subject_id)
    else:
        return basic_wf(subject_data, settings, name=subject_id)


def basic_fmap_sbref_wf(subject_data, settings, name='fMRI_prep'):
    """
    The main fmri preprocessing workflow, for the ds054-type of data:

      * [x] Has at least one T1w and at least one bold file (minimal reqs.)
      * [x] Has one or more SBRefs

    """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']),
                        name='inputnode')
    #  inputnode = pe.Node(niu.IdentityInterface(fields=['subject_id']),
    #                    name='inputnode')
    #  inputnode.iterables = [('subject_id', subject_list)]

    bidssrc = pe.Node(BIDSDataGrabber(subject_data=subject_data), name='BIDSDatasource')

    # Preprocessing of T1w (includes registration to MNI)
    t1w_pre = t1w_preprocessing(settings=settings)

    # Estimate fieldmap
    fmap_est = fmap_estimator(subject_data, settings=settings)

    # Correct SBRef
    sbref_pre = sbref_preprocess(settings=settings)

    # Register SBRef to T1
    sbref_t1 = ref_epi_t1_registration(reportlet_suffix='sbref_t1_bbr',
                                       inv_ds_suffix='target-sbref_affine',
                                       settings=settings)

    # HMC and SDC of EPI using SBRef as reference
    epi_pre = epi_preprocess(settings=settings)
    epi_pre.get_node('inputnode').iterables = ('epi', subject_data['func'])

    # get confounds
    confounds_wf = confounds.discover_wf(settings)
    confounds_wf.get_node('inputnode').inputs.t1_transform_flags = [False, True]

    # create list of transforms to resample t1 -> sbref -> epi
    epi_mni_trans_wf = epi_mni_transformation(settings=settings)

    workflow.connect([
        (bidssrc, t1w_pre, [('t1w', 'inputnode.t1w'),
                            ('t2w', 'inputnode.t2w')]),
        (bidssrc, sbref_pre, [('sbref', 'inputnode.sbref')]),
        (bidssrc, sbref_t1, [('sbref', 'inputnode.name_source'),
                             ('t1w', 'inputnode.t1w')]),
        (fmap_est, sbref_pre, [('outputnode.fmap', 'inputnode.fmap'),
                               ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
                               ('outputnode.fmap_mask', 'inputnode.fmap_mask')]),
        (fmap_est, epi_pre, [('outputnode.fmap', 'inputnode.fmap'),
                               ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
                               ('outputnode.fmap_mask', 'inputnode.fmap_mask')]),
        (sbref_pre, sbref_t1, [('outputnode.sbref_unwarped', 'inputnode.ref_epi'),
                               ('outputnode.sbref_unwarped_mask', 'inputnode.ref_epi_mask')]),
        (sbref_pre, epi_pre, [('outputnode.sbref_unwarped', 'inputnode.sbref')]),
        (epi_pre, sbref_t1, [('outputnode.out_warps', 'inputnode.hmc_scd_warps')]),
        (t1w_pre, sbref_t1, [
            ('outputnode.bias_corrected_t1', 'inputnode.bias_corrected_t1'),
            ('outputnode.t1_mask', 'inputnode.t1_mask'),
            ('outputnode.t1_brain', 'inputnode.t1_brain'),
            ('outputnode.t1_seg', 'inputnode.t1_seg')]),
        # (t1_to_epi_transforms, confounds_wf, [('out_file', 'inputnode.t1_transform')]),
        (t1w_pre, confounds_wf, [('outputnode.t1_tpms', 'inputnode.t1_tpms')]),

        (epi_pre, confounds_wf, [
            ('inputnode.epi', 'inputnode.source_file'),
            ('outputnode.epi_corr', 'inputnode.fmri_file'),
            # ('outputnode.epi_mean', 'inputnode.reference_image'),
            ('outputnode.epi_mask', 'inputnode.epi_mask'),
            ('outputnode.hmc_movpar', 'inputnode.movpar_file'),

        ]),
        (sbref_t1, epi_mni_trans_wf, [('outputnode.itk_epi_to_t1', 'inputnode.itk_epi_to_t1')]),
        (epi_pre, epi_mni_trans_wf, [('inputnode.epi', 'inputnode.name_source'),
                                     ('outputnode.out_warps', 'inputnode.hmc_scd_warps'),
                                     ('outputnode.epi_mask', 'inputnode.epi_mask'),
                                     ('outputnode.epi_corr_split', 'inputnode.epi_split')]),
        (t1w_pre, epi_mni_trans_wf, [('outputnode.bias_corrected_t1', 'inputnode.t1'),
                                     ('outputnode.t1_2_mni_forward_transform',
                                      'inputnode.t1_2_mni_forward_transform')])

    ])

    if settings['freesurfer']:
        workflow.connect([
            (inputnode, t1w_pre, [('subjects_dir', 'inputnode.subjects_dir')]),
            (inputnode, sbref_t1, [('subjects_dir', 'inputnode.subjects_dir')]),
            (t1w_pre, sbref_t1, [('outputnode.subject_id', 'inputnode.subject_id'),
                                 ('outputnode.fs_2_t1_transform', 'inputnode.fs_2_t1_transform')]),
            ])

    return workflow


def basic_wf(subject_data, settings, name='fMRI_prep'):
    """
    The main fmri preprocessing workflow, for the ds005-type of data:

      * [x] Has at least one T1w and at least one bold file (minimal reqs.)
      * [ ] No SBRefs

    """
    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']),
                        name='inputnode')
    #  inputnode = pe.Node(niu.IdentityInterface(fields=['subject_id']),
    #                      name='inputnode')
    #  inputnode.iterables = [('subject_id', subject_list)]

    bidssrc = pe.Node(BIDSDataGrabber(subject_data=subject_data),
                      name='BIDSDatasource')


    # Estimate fieldmap
    fmap_est = fmap_estimator(subject_data, settings=settings)
    if fmap_est is not None:
        raise NotImplementedError

    # Preprocessing of T1w (includes registration to MNI)
    t1w_pre = t1w_preprocessing(settings=settings)

    # HMC on the EPI
    hmcwf = epi_hmc(settings=settings)
    hmcwf.get_node('inputnode').iterables = ('epi', subject_data['func'])

    # mean EPI registration to T1w
    epi_2_t1 = ref_epi_t1_registration(reportlet_suffix='bbr',
                                       inv_ds_suffix='target-meanBOLD_affine',
                                       settings=settings)

    # get confounds
    confounds_wf = confounds.discover_wf(settings)
    confounds_wf.get_node('inputnode').inputs.t1_transform_flags = [False]

    # Apply transforms in 1 shot
    epi_mni_trans_wf = epi_mni_transformation(settings=settings)

    workflow.connect([
        (bidssrc, t1w_pre, [('t1w', 'inputnode.t1w'),
                            ('t2w', 'inputnode.t2w')]),
        (bidssrc, epi_2_t1, [('t1w', 'inputnode.t1w')]),
        (hmcwf, epi_2_t1, [('inputnode.epi', 'inputnode.name_source'),
                           ('outputnode.epi_mean', 'inputnode.ref_epi'),
                           ('outputnode.xforms', 'inputnode.hmc_scd_warps'),
                           ('outputnode.epi_mask', 'inputnode.ref_epi_mask'),
                           ('outputnode.epi_split', 'inputnode.epi_split')]),
        (t1w_pre, epi_2_t1, [('outputnode.bias_corrected_t1', 'inputnode.bias_corrected_t1'),
                             ('outputnode.t1_brain', 'inputnode.t1_brain'),
                             ('outputnode.t1_mask', 'inputnode.t1_mask'),
                             ('outputnode.t1_seg', 'inputnode.t1_seg')]),

        (t1w_pre, confounds_wf, [('outputnode.t1_tpms', 'inputnode.t1_tpms')]),
        (hmcwf, confounds_wf, [
            ('outputnode.movpar_file', 'inputnode.movpar_file'),
            ('outputnode.motion_confounds_file', 'inputnode.motion_confounds_file'),
            ('inputnode.epi', 'inputnode.source_file')]),
        (epi_2_t1, confounds_wf, [('outputnode.epi_t1', 'inputnode.fmri_file'),
                                  ('outputnode.epi_mask_t1', 'inputnode.epi_mask')]),

        (epi_2_t1, epi_mni_trans_wf, [('outputnode.itk_epi_to_t1', 'inputnode.itk_epi_to_t1')]),
        (hmcwf, epi_mni_trans_wf, [('inputnode.epi', 'inputnode.name_source'),
                                   ('outputnode.xforms', 'inputnode.hmc_scd_warps'),
                                   ('outputnode.epi_mask', 'inputnode.epi_mask'),
                                   ('outputnode.epi_split', 'inputnode.epi_split')]),
        (t1w_pre, epi_mni_trans_wf, [('outputnode.bias_corrected_t1', 'inputnode.t1'),
                                     ('outputnode.t1_2_mni_forward_transform',
                                      'inputnode.t1_2_mni_forward_transform')])
    ])

    if settings['freesurfer']:
        workflow.connect([
            (inputnode, t1w_pre, [('subjects_dir', 'inputnode.subjects_dir')]),
            (inputnode, epi_2_t1, [('subjects_dir', 'inputnode.subjects_dir')]),
            (t1w_pre, epi_2_t1, [('outputnode.subject_id', 'inputnode.subject_id'),
                                 ('outputnode.fs_2_t1_transform', 'inputnode.fs_2_t1_transform')]),
            ])

    return workflow


def _first(inlist):
    if isinstance(inlist, (list, tuple)):
        inlist = _first(inlist[0])
    return inlist
