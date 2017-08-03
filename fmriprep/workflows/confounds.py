#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from niworkflows.nipype.interfaces import utility
from niworkflows.nipype.interfaces import fsl
from niworkflows.nipype.algorithms import confounds
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.interfaces.masks import ACompCorRPT, TCompCorRPT
from niworkflows.interfaces import segmentation as nws

from niworkflows.nipype.interfaces.nilearn import SignalExtraction
from fmriprep.interfaces.utils import prepare_roi_from_probtissue


def init_discover_wf(bold_file_size_gb, use_aroma, ignore_aroma_err, metadata,
                     name="discover_wf"):
    ''' All input fields are required.

    Calculates global regressor and tCompCor
        from motion-corrected fMRI ('inputnode.fmri_file').
    Calculates DVARS from the fMRI and an EPI brain mask ('inputnode.epi_mask')
    Calculates frame displacement from MCFLIRT movement parameters ('inputnode.movpar_file')
    Calculates segment regressors and aCompCor
        from the fMRI and a white matter/gray matter/CSF segmentation ('inputnode.t1_seg'), after
        applying the transform to the images. Transforms should be fsl-formatted.
    Calculates noise components identified from ICA-AROMA (if ``use_aroma=True``)
    Saves the confounds in a file ('outputnode.confounds_file')'''

    inputnode = pe.Node(utility.IdentityInterface(
        fields=['fmri_file', 'movpar_file', 't1_tpms', 'epi_mask', 'epi_mni', 'epi_mask_mni']),
        name='inputnode')
    outputnode = pe.Node(utility.IdentityInterface(
        fields=['confounds_file', 'confounds_list', 'acompcor_report', 'tcompcor_report',
                'ica_aroma_report', 'aroma_noise_ics', 'melodic_mix']),
        name='outputnode')

    # ICA-AROMA
    if use_aroma:
        ica_aroma_wf = init_ica_aroma_wf(name='ica_aroma_wf',
                                         ignore_aroma_err=ignore_aroma_err)

    # DVARS
    dvars = pe.Node(confounds.ComputeDVARS(save_all=True, remove_zerovariance=True),
                    name="dvars")
    dvars.interface.estimated_memory_gb = bold_file_size_gb * 3
    # Frame displacement
    frame_displace = pe.Node(confounds.FramewiseDisplacement(parameter_source="SPM"),
                             name="frame_displace")
    frame_displace.interface.estimated_memory_gb = bold_file_size_gb * 3
    # CompCor
    non_steady_state = pe.Node(confounds.NonSteadyStateDetector(), name='non_steady_state')
    tcompcor = pe.Node(TCompCorRPT(components_file='tcompcor.tsv',
                                   generate_report=True,
                                   pre_filter='cosine',
                                   save_pre_filter=True,
                                   percentile_threshold=.05),
                       name="tcompcor")
    tcompcor.interface.estimated_memory_gb = bold_file_size_gb * 3
    if 'RepetitionTime' in metadata:
        tcompcor.inputs.repetition_time = metadata['RepetitionTime']

    CSF_roi = pe.Node(utility.Function(function=prepare_roi_from_probtissue,
                                       output_names=['roi_file', 'eroded_mask']),
                      name='CSF_roi')
    CSF_roi.inputs.erosion_mm = 0
    CSF_roi.inputs.epi_mask_erosion_mm = 30

    WM_roi = pe.Node(utility.Function(function=prepare_roi_from_probtissue,
                                      output_names=['roi_file', 'eroded_mask']),
                     name='WM_roi')
    WM_roi.inputs.erosion_mm = 6
    WM_roi.inputs.epi_mask_erosion_mm = 10

    def concat_rois_func(in_WM, in_mask, ref_header):
        import os
        import nibabel as nb
        from nilearn.image import resample_to_img

        WM_nii = nb.load(in_WM)
        mask_nii = nb.load(in_mask)

        # we have to do this explicitly because of potential differences in
        # qform_code between the two files that prevent SignalExtraction to do
        # the concatenation
        concat_nii = nb.funcs.concat_images([resample_to_img(WM_nii,
                                                             mask_nii,
                                                             interpolation='nearest'),
                                             mask_nii])
        concat_nii = nb.Nifti1Image(concat_nii.get_data(),
                                    nb.load(ref_header).affine,
                                    nb.load(ref_header).header)
        concat_nii.to_filename("concat.nii.gz")
        return os.path.abspath("concat.nii.gz")

    concat_rois = pe.Node(utility.Function(function=concat_rois_func), name='concat_rois')

    # Global and segment regressors
    signals = pe.Node(SignalExtraction(detrend=True,
                                       class_labels=["WhiteMatter", "GlobalSignal"]),
                      name="signals")
    signals.interface.estimated_memory_gb = bold_file_size_gb * 3

    def combine_rois(in_CSF, in_WM, ref_header):
        import os
        import numpy as np
        import nibabel as nb

        CSF_nii = nb.load(in_CSF)
        CSF_data = CSF_nii.get_data()

        WM_nii = nb.load(in_WM)
        WM_data = WM_nii.get_data()

        combined = np.zeros_like(WM_data)

        combined[WM_data != 0] = 1
        combined[CSF_data != 0] = 1

        # we have to do this explicitly because of potential differences in
        # qform_code between the two files that prevent aCompCor to work
        new_nii = nb.Nifti1Image(combined, nb.load(ref_header).affine,
                                 nb.load(ref_header).header)
        new_nii.to_filename("logical_or.nii.gz")
        return os.path.abspath("logical_or.nii.gz")

    combine_rois = pe.Node(utility.Function(function=combine_rois), name='combine_rois')

    acompcor = pe.Node(ACompCorRPT(components_file='acompcor.tsv',
                                   pre_filter='cosine',
                                   save_pre_filter=True,
                                   generate_report=True),
                       name="acompcor")
    acompcor.interface.estimated_memory_gb = bold_file_size_gb * 3
    if 'RepetitionTime' in metadata:
        acompcor.inputs.repetition_time = metadata['RepetitionTime']

    # misc utilities
    concat = pe.Node(
        utility.Function(function=_gather_confounds,
                         output_names=['confounds_file', 'confounds_list']),
        name="concat")

    def pick_csf(files):
        return files[0]

    def pick_wm(files):
        return files[2]

    def add_header_func(in_file):
        import numpy as np
        import pandas as pd
        import os
        from sys import version_info
        PY3 = version_info[0] > 2

        data = np.loadtxt(in_file)

        df = pd.DataFrame(data, columns=["X", "Y", "Z", "RotX", "RotY", "RotZ"])
        df.to_csv("motion.tsv", sep="\t" if PY3 else '\t'.encode(), index=None)

        return os.path.abspath("motion.tsv")

    add_header = pe.Node(utility.Function(function=add_header_func), name="add_header")

    workflow = pe.Workflow(name=name)
    workflow.connect([
        # connect inputnode to each non-anatomical confound node
        (inputnode, dvars, [('fmri_file', 'in_file'),
                            ('epi_mask', 'in_mask')]),
        (inputnode, frame_displace, [('movpar_file', 'in_file')]),
        (inputnode, non_steady_state, [('fmri_file', 'in_file')]),
        (inputnode, tcompcor, [('fmri_file', 'realigned_file')]),

        (non_steady_state, tcompcor, [('n_volumes_to_discard', 'ignore_initial_volumes')]),
        (non_steady_state, acompcor, [('n_volumes_to_discard', 'ignore_initial_volumes')]),

        (inputnode, CSF_roi, [(('t1_tpms', pick_csf), 'in_file')]),
        (inputnode, CSF_roi, [('epi_mask', 'epi_mask')]),
        (CSF_roi, tcompcor, [('eroded_mask', 'mask_files')]),

        (inputnode, WM_roi, [(('t1_tpms', pick_wm), 'in_file')]),
        (inputnode, WM_roi, [('epi_mask', 'epi_mask')]),

        (CSF_roi, combine_rois, [('roi_file', 'in_CSF')]),
        (WM_roi, combine_rois, [('roi_file', 'in_WM')]),
        (inputnode, combine_rois, [('fmri_file', 'ref_header')]),

        # anatomical confound: aCompCor.
        (inputnode, acompcor, [('fmri_file', 'realigned_file')]),
        (combine_rois, acompcor, [('out', 'mask_files')]),

        (WM_roi, concat_rois, [('roi_file', 'in_WM')]),
        (inputnode, concat_rois, [('epi_mask', 'in_mask')]),
        (inputnode, concat_rois, [('fmri_file', 'ref_header')]),

        # anatomical confound: signal extraction
        (concat_rois, signals, [('out', 'label_files')]),
        (inputnode, signals, [('fmri_file', 'in_file')]),

        # connect the confound nodes to the concatenate node
        (signals, concat, [('out_file', 'signals')]),
        (dvars, concat, [('out_all', 'dvars')]),
        (frame_displace, concat, [('out_file', 'frame_displace')]),
        (tcompcor, concat, [('components_file', 'tcompcor'),
                            ('pre_filter_file', 'cosine_basis')]),
        (acompcor, concat, [('components_file', 'acompcor')]),
        (inputnode, add_header, [('movpar_file', 'in_file')]),
        (add_header, concat, [('out', 'motion')]),

        (concat, outputnode, [('confounds_file', 'confounds_file'),
                              ('confounds_list', 'confounds_list')]),
        (acompcor, outputnode, [('out_report', 'acompcor_report')]),
        (tcompcor, outputnode, [('out_report', 'tcompcor_report')]),
    ])
    if use_aroma:
        workflow.connect([
            (inputnode, ica_aroma_wf, [('epi_mni', 'inputnode.epi_mni'),
                                       ('epi_mask_mni', 'inputnode.epi_mask_mni'),
                                       ('movpar_file', 'inputnode.movpar_file')]),
            (ica_aroma_wf, concat,
                [('outputnode.aroma_confounds', 'aroma')]),
            (ica_aroma_wf, outputnode,
                [('outputnode.out_report', 'ica_aroma_report'),
                 ('outputnode.aroma_noise_ics', 'aroma_noise_ics'),
                 ('outputnode.melodic_mix', 'melodic_mix')])
        ])
    return workflow


def _gather_confounds(signals=None, dvars=None, frame_displace=None,
                      tcompcor=None, acompcor=None, cosine_basis=None,
                      motion=None, aroma=None):
    ''' load confounds from the filenames, concatenate together horizontally, and re-save '''
    import os
    import pandas as pd

    def less_breakable(a_string):
        ''' hardens the string to different envs (i.e. case insensitive, no whitespace, '#' '''
        return ''.join(a_string.split()).strip('#')

    def _adjust_indices(left_df, right_df):
        # This forces missing values to appear at the beggining of the DataFrame
        # instead of the end
        index_diff = len(left_df.index) - len(right_df.index)
        if index_diff > 0:
            right_df.index = range(index_diff,
                                   len(right_df.index) + index_diff)
        elif index_diff < 0:
            left_df.index = range(-index_diff,
                                  len(left_df.index) - index_diff)

    all_files = []
    confounds_list = []
    for confound, name in ((signals, 'Global signals'),
                           (dvars, 'DVARS'),
                           (frame_displace, 'Framewise displacement'),
                           (tcompcor, 'tCompCor'),
                           (acompcor, 'aCompCor'),
                           (cosine_basis, 'Cosine basis'),
                           (motion, 'Motion parameters'),
                           (aroma, 'ICA-AROMA')):
        if confound is not None:
            confounds_list.append(name)
            if os.path.exists(confound) and os.stat(confound).st_size > 0:
                all_files.append(confound)

    confounds_data = pd.DataFrame()
    for file_name in all_files:  # assumes they all have headings already
        new = pd.read_csv(file_name, sep="\t")
        for column_name in new.columns:
            new.rename(columns={column_name: less_breakable(column_name)},
                       inplace=True)

        _adjust_indices(confounds_data, new)
        confounds_data = pd.concat((confounds_data, new), axis=1)

    combined_out = os.path.abspath('confounds.tsv')
    confounds_data.to_csv(combined_out, sep=str("\t"), index=False,
                          na_rep="n/a")

    return combined_out, confounds_list


def reverse_order(inlist):
    ''' if a list, return the list in reversed order; else it is a single item, return it.'''
    if isinstance(inlist, list):
        inlist.reverse()
    return inlist


def get_ica_confounds(ica_out_dir, ignore_aroma_err):
    import os
    import shutil
    import numpy as np
    from niworkflows.nipype import logging

    # To catch edge cases when there are either no noise or signal components
    LOGGER = logging.getLogger('workflow')

    # Pass in numpy array and column base name to generate headers
    # modified from add_header_func
    def aroma_add_header_func(np_arr, col_base, comp_nums):
        import pandas as pd
        from sys import version_info
        PY3 = version_info[0] > 2

        df = pd.DataFrame(np_arr, columns=[str(col_base) + str(index) for index in comp_nums])
        df.to_csv(str(col_base) + "AROMAConfounds.tsv",
                  sep="\t" if PY3 else '\t'.encode(), index=None)

        return os.path.abspath(str(col_base) + "AROMAConfounds.tsv")

    # load the txt files from ICA-AROMA
    melodic_mix = os.path.join(ica_out_dir, 'melodic.ica/melodic_mix')
    motion_ics = os.path.join(ica_out_dir, 'classified_motion_ICs.txt')

    # Change names of motion_ics and melodic_mix for output
    melodic_mix_out = os.path.join(ica_out_dir, 'MELODICmix.tsv')
    motion_ics_out = os.path.join(ica_out_dir, 'AROMAnoiseICs.csv')

    # melodic_mix replace spaces with tabs
    with open(melodic_mix, 'r') as melodic_file:
        melodic_mix_out_char = melodic_file.read().replace('  ', '\t')
    # write to output file
    with open(melodic_mix_out, 'w+') as melodic_file_out:
        melodic_file_out.write(melodic_mix_out_char)

    # copy metion_ics file to derivatives name
    shutil.copyfile(motion_ics, motion_ics_out)

    # -1 since python lists start at index 0
    motion_ic_indices = np.loadtxt(motion_ics, dtype=int, delimiter=',') - 1
    melodic_mix_arr = np.loadtxt(melodic_mix, ndmin=2)

    # Return dummy list of ones if no noise compnents were found
    if motion_ic_indices.size == 0:
        if ignore_aroma_err:
            LOGGER.warn('WARNING: No noise components were classified')
            aroma_confounds = None
            return aroma_confounds, motion_ics_out, melodic_mix_out
        else:
            raise RuntimeError('ERROR: ICA-AROMA found no noise components!')

    # transpose melodic_mix_arr so x refers to the correct dimension
    aggr_confounds = np.asarray([melodic_mix_arr.T[x] for x in motion_ic_indices])

    # the "good" ics, (e.g. not motion related)
    good_ic_arr = np.delete(melodic_mix_arr, motion_ic_indices, 1).T

    # return dummy lists of zeros if no signal components were found
    if good_ic_arr.size == 0:
        if ignore_aroma_err:
            LOGGER.warn('WARNING: No signal components were classified')
            aroma_confounds = None
            return aroma_confounds, motion_ics_out, melodic_mix_out
        else:
            raise RuntimeError('ERROR: ICA-AROMA found no signal components!')

    # add one to motion_ic_indices to match melodic report.
    aggr_tsv = aroma_add_header_func(aggr_confounds.T, 'AROMAAggrComp',
                                     [str(x).zfill(2) for x in motion_ic_indices + 1])
    aroma_confounds = aggr_tsv

    return aroma_confounds, motion_ics_out, melodic_mix_out


def init_ica_aroma_wf(name='ica_aroma_wf', ignore_aroma_err=False):
    '''
    From: https://github.com/rhr-pruim/ICA-AROMA
    Description:
    ICA-AROMA (i.e. ‘ICA-based Automatic Removal Of Motion Artifacts’) concerns
    a data-driven method to identify and remove motion-related independent
    components from fMRI data.

    Preconditions/Assumptions:
    The input fmri bold file is in standard space
    (for ease of interfacing with the original ICA-AROMA code)

    Steps:
    1) smooth data using SUSAN
    2) run melodic outside of ICA-AROMA to generate the report
    3) run ICA-AROMA
    4) print identified motion components (aggressive) to tsv
    5) pass classified_motion_ICs and melodic_mix for user to complete nonaggr denoising
    '''
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(utility.IdentityInterface(fields=['epi_mni',
                                                          'movpar_file',
                                                          'epi_mask_mni']),
                        name='inputnode')

    outputnode = pe.Node(utility.IdentityInterface(
        fields=['aroma_confounds', 'out_report',
                'aroma_noise_ics', 'melodic_mix']), name='outputnode')

    # helper function to get
    # smoothing node (SUSAN)
    # functions to help set SUSAN
    def getbtthresh(medianval):
        return 0.75 * medianval

    def getusans_func(image, thresh):
        return [tuple([image, thresh])]

    calc_median_val = pe.Node(fsl.ImageStats(op_string='-k %s -p 50'),
                              name='calc_median_val')

    calc_epi_mean = pe.Node(fsl.MeanImage(),
                            name='calc_epi_mean')

    brightness_threshold = pe.Node(
        utility.Function(function=getbtthresh,
                         input_names=['medianval'],
                         output_names=['thresh']),
        name='brightness_threshold')

    getusans = pe.Node(
        utility.Function(function=getusans_func,
                         input_names=['image', 'thresh'],
                         output_names=['usans']),
        name='getusans')

    smooth = pe.Node(fsl.SUSAN(fwhm=6.0),
                     name='smooth')

    # melodic node
    melodic = pe.Node(
        nws.MELODICRPT(no_bet=True,
                       no_mm=True,
                       generate_report=True),
        name="melodic")

    # ica_aroma node
    ica_aroma = pe.Node(fsl.ICA_AROMA(denoise_type='no'), name='ica_aroma')

    # extract the confound ICs from the results
    ica_aroma_confound_extraction = pe.Node(
        utility.Function(function=get_ica_confounds,
                         input_names=['ica_out_dir', 'ignore_aroma_err'],
                         output_names=['aroma_confounds', 'aroma_noise_ics', 'melodic_mix']),
        name='ica_aroma_confound_extraction')
    ica_aroma_confound_extraction.inputs.ignore_aroma_err = ignore_aroma_err

    # connect the nodes
    workflow.connect([
        # Connect input nodes to complete smoothing
        (inputnode, calc_median_val, [('epi_mni', 'in_file'),
                                      ('epi_mask_mni', 'mask_file')]),
        (calc_median_val, brightness_threshold, [('out_stat', 'medianval')]),
        (inputnode, calc_epi_mean, [('epi_mni', 'in_file')]),
        (calc_epi_mean, getusans, [('out_file', 'image')]),
        (calc_median_val, getusans, [('out_stat', 'thresh')]),
        (inputnode, smooth, [('epi_mni', 'in_file')]),
        (getusans, smooth, [('usans', 'usans')]),
        (brightness_threshold, smooth, [('thresh', 'brightness_threshold')]),
        # connect smooth to melodic
        (smooth, melodic, [('smoothed_file', 'in_files')]),
        (inputnode, melodic, [('epi_mask_mni', 'report_mask'),
                              ('epi_mask_mni', 'mask')]),
        # connect nodes to ICA-AROMA
        (smooth, ica_aroma, [('smoothed_file', 'in_file')]),
        (inputnode, ica_aroma, [('movpar_file', 'motion_parameters')]),
        (melodic, ica_aroma, [('out_dir', 'melodic_dir')]),
        # generate tsvs from ICA-AROMA
        (ica_aroma, ica_aroma_confound_extraction, [('out_dir', 'ica_out_dir')]),
        # output for processing and reporting
        (ica_aroma_confound_extraction, outputnode, [('aroma_confounds', 'aroma_confounds'),
                                                     ('aroma_noise_ics', 'aroma_noise_ics'),
                                                     ('melodic_mix', 'melodic_mix')]),
        # TODO change melodic report to reflect noise and non-noise components
        (melodic, outputnode, [('out_report', 'out_report')]),
    ])

    return workflow
