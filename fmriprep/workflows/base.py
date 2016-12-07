#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Created on Wed Dec  2 17:35:40 2015

@author: craigmoodie
"""
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl

from fmriprep.interfaces import BIDSDataGrabber
from fmriprep.utils.misc import collect_bids_data, get_biggest_epi_file_size_gb
from fmriprep.workflows import confounds

from fmriprep.workflows.anatomical import t1w_preprocessing
from fmriprep.workflows.sbref import sbref_preprocess, sbref_t1_registration
from fmriprep.workflows.fieldmap import fmap_estimator
from fmriprep.workflows.epi import (
    epi_unwarp, epi_hmc, epi_sbref_registration,
    epi_mean_t1_registration, epi_mni_transformation)


def base_workflow_enumerator(subject_list, task_id, settings):
    workflow = pe.Workflow(name='workflow_enumerator')
    generated_list = []
    for subject in subject_list:
        generated_workflow = base_workflow_generator(subject, task_id=task_id,
                                                     settings=settings)
        if generated_workflow:
            generated_list.append(generated_workflow)
    workflow.add_nodes(generated_list)

    return workflow


def base_workflow_generator(subject_id, task_id, settings):
    subject_data = collect_bids_data(settings['bids_root'], subject_id, task_id)

    settings["biggest_epi_file_size_gb"] = get_biggest_epi_file_size_gb(subject_data['func'])

    if subject_data['t1w'] == []:
        raise Exception("No T1w images found for participant %s. All workflows require T1w images."%subject_id)

    if subject_data['sbref'] != [] or settings['workflow_type'] == "ds054":
        return wf_ds054_type(subject_data, settings, name=subject_id)
    elif subject_data['sbref'] == [] or settings['workflow_type'] == "ds005":
        return wf_ds005_type(subject_data, settings, name=subject_id)
    else:
        raise Exception("Could not figure out what kind of workflow to run for this dataset.")



def wf_ds054_type(subject_data, settings, name='fMRI_prep'):
    """
    The main fmri preprocessing workflow, for the ds054-type of data:

      * [x] Has at least one T1w and at least one bold file (minimal reqs.)
      * [x] Has one or more SBRefs

    """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)

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
    sbref_t1 = sbref_t1_registration(settings=settings)

    # HMC on the EPI
    hmcwf = epi_hmc(settings=settings)
    hmcwf.get_node('inputnode').iterables = ('epi', subject_data['func'])

    # EPI to SBRef
    epi2sbref = epi_sbref_registration(settings)

    # get confounds
    confounds_wf = confounds.discover_wf(settings)
    confounds_wf.get_node('inputnode').inputs.t1_transform_flags = [False, True]

    # create list of transforms to resample t1 -> sbref -> epi
    t1_to_epi_transforms = pe.Node(fsl.ConvertXFM(concat_xfm=True), name='T1ToEPITransforms')

    workflow.connect([
        (bidssrc, t1w_pre, [('t1w', 'inputnode.t1w')]),
        (bidssrc, sbref_pre, [('sbref', 'inputnode.sbref')]),
        (bidssrc, sbref_t1, [('sbref', 'inputnode.sbref')]),
        (sbref_pre, sbref_t1, [('outputnode.sbref_unwarped', 'inputnode.sbref_brain')]),
        (t1w_pre, sbref_t1, [
            ('outputnode.t1_brain', 'inputnode.t1_brain'),
            ('outputnode.t1_seg', 'inputnode.t1_seg')]),
        (sbref_pre, epi2sbref, [('outputnode.sbref_unwarped', 'inputnode.sbref_brain'),
                                ('outputnode.sbref_unwarped_mask', 'inputnode.sbref_brain_mask')]),
        (hmcwf, epi2sbref, [('outputnode.epi_brain', 'inputnode.epi_brain')]),
        (hmcwf, epi2sbref, [('outputnode.epi_mean', 'inputnode.epi_mean')]),
        (hmcwf, epi2sbref, [('inputnode.epi', 'inputnode.epi')]),
        (sbref_t1, t1_to_epi_transforms, [(('outputnode.mat_t1_to_sbr'), 'in_file')]),
        (epi2sbref, t1_to_epi_transforms, [('outputnode.out_mat_inv', 'in_file2')]),

        (t1_to_epi_transforms, confounds_wf, [('out_file', 'inputnode.t1_transform')]),

        (hmcwf, confounds_wf, [('outputnode.movpar_file', 'inputnode.movpar_file'),
                               ('outputnode.epi_mean', 'inputnode.reference_image'),
                               ('outputnode.motion_confounds_file',
                                'inputnode.motion_confounds_file'),
                               ('inputnode.epi', 'inputnode.source_file')]),
        (t1w_pre, confounds_wf, [('outputnode.t1_seg', 'inputnode.t1_seg')]),
    ])


    if fmap_est is not None:
        # EPI unwarp
        epiunwarp_wf = epi_unwarp(settings=settings)

        workflow.connect([
            (fmap_est, sbref_pre, [('outputnode.fmap', 'inputnode.fmap'),
                                   ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
                                   ('outputnode.fmap_mask', 'inputnode.fmap_mask')]),
            (fmap_est, epiunwarp_wf, [('outputnode.fmap', 'inputnode.fmap'),
                                      ('outputnode.fmap_mask', 'inputnode.fmap_mask'),
                                      ('outputnode.fmap_ref', 'inputnode.fmap_ref')]),
            (hmcwf, epiunwarp_wf, [('inputnode.epi', 'inputnode.epi')]),
            (epiunwarp_wf, confounds_wf, [('outputnode.epi_mask', 'inputnode.epi_mask'),
                                          ('outputnode.epi_unwarp', 'inputnode.fmri_file')]),
        ])
    else:
        raise NotImplementedError

    return workflow


def wf_ds005_type(subject_data, settings, name='fMRI_prep'):
    """
    The main fmri preprocessing workflow, for the ds005-type of data:

      * [x] Has at least one T1w and at least one bold file (minimal reqs.)
      * [ ] No SBRefs

    """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)

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
    epi_2_t1 = epi_mean_t1_registration(settings=settings)

    # get confounds
    confounds_wf = confounds.discover_wf(settings)
    confounds_wf.get_node('inputnode').inputs.t1_transform_flags = [False]

    # Apply transforms in 1 shot
    epi_mni_trans_wf = epi_mni_transformation(settings=settings)

    workflow.connect([
        (bidssrc, t1w_pre, [('t1w', 'inputnode.t1w')]),
        (bidssrc, epi_2_t1, [('t1w', 'inputnode.t1w')]),
        (hmcwf, epi_2_t1, [('inputnode.epi', 'inputnode.epi')]),
        (hmcwf, epi_2_t1, [('outputnode.epi_mean', 'inputnode.epi_mean')]),
        (t1w_pre, epi_2_t1, [('outputnode.t1_brain', 'inputnode.t1_brain'),
                             ('outputnode.t1_seg', 'inputnode.t1_seg')]),

        (t1w_pre, confounds_wf, [('outputnode.t1_seg', 'inputnode.t1_seg')]),
        (hmcwf, confounds_wf, [('outputnode.movpar_file', 'inputnode.movpar_file'),
                               ('outputnode.epi_brain', 'inputnode.fmri_file'),
                               ('outputnode.epi_mean', 'inputnode.reference_image'),
                               ('outputnode.epi_mask', 'inputnode.epi_mask'),
                               ('outputnode.motion_confounds_file', 'inputnode.motion_confounds_file')]),
        (epi_2_t1, confounds_wf, [('outputnode.mat_t1_to_epi', 'inputnode.t1_transform')]),
        (hmcwf, confounds_wf, [('inputnode.epi', 'inputnode.source_file')]),

        (hmcwf, epi_mni_trans_wf, [('inputnode.epi', 'inputnode.epi')]),
        (epi_2_t1, epi_mni_trans_wf, [('outputnode.itk_epi_to_t1', 'inputnode.itk_epi_to_t1')]),
        (hmcwf, epi_mni_trans_wf, [('outputnode.xforms', 'inputnode.hmc_xforms'),
                                   ('outputnode.epi_mask', 'inputnode.epi_mask')]),
        (t1w_pre, epi_mni_trans_wf, [('outputnode.t1_brain', 'inputnode.t1'),
                                     ('outputnode.t1_2_mni_forward_transform',
                                      'inputnode.t1_2_mni_forward_transform')])
    ])

    return workflow


def _first(inlist):
    if isinstance(inlist, (list, tuple)):
        inlist = _first(inlist[0])
    return inlist
