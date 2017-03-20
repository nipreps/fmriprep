#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Anatomical Reference -processing workflows.

Originally coded by Craig Moodie. Refactored by the CRN Developers.

"""
import os.path as op

from nipype.interfaces import ants
from nipype.interfaces import freesurfer
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from niworkflows.interfaces.registration import RobustMNINormalizationRPT
from niworkflows.anat.skullstrip import afni_wf as skullstrip_wf
from niworkflows.data import get_mni_icbm152_nlin_asym_09c
from niworkflows.interfaces.masks import BrainExtractionRPT
from niworkflows.interfaces.segmentation import FASTRPT, ReconAllRPT

from fmriprep.interfaces import (DerivativesDataSink, IntraModalMerge)
from fmriprep.utils.misc import fix_multi_T1w_source_name

#  pylint: disable=R0914
def t1w_preprocessing(name='t1w_preprocessing', settings=None):
    """T1w images preprocessing pipeline"""

    if settings is None:
        raise RuntimeError('Workflow settings are missing')

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['t1w', 't2w', 'subjects_dir']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_seg', 't1_tpms', 'bias_corrected_t1', 't1_brain', 't1_mask',
                't1_2_mni', 't1_2_mni_forward_transform',
                't1_2_mni_reverse_transform', 'subject_id',
                'fs_2_t1_transform']), name='outputnode')

    # 0. Align, reorient to RAS and merge if several T1w images are provided
    t1wmrg = pe.Node(IntraModalMerge(to_ras=True, hmc=True), name='MergeT1s')


    # 2. T1 Bias Field Correction
    # Bias field correction is handled in skull strip workflows.

    # 3. Skull-stripping
    asw = skullstrip_wf()
    if settings.get('skull_strip_ants', False):
        asw = skullstrip_ants(settings=settings)

    # 4. Segmentation
    t1_seg = pe.Node(FASTRPT(generate_report=True, segments=True,
                             no_bias=True, probability_maps=True),
                     name='Segmentation')
    t1_seg.interface.estimated_memory_gb = 4

    # 5. Spatial normalization (T1w to MNI registration)
    t1_2_mni = pe.Node(
        RobustMNINormalizationRPT(
            generate_report=True,
            num_threads=settings['ants_nthreads'],
            testing=settings.get('debug', False),
            template='mni_icbm152_nlin_asym_09c'
        ),
        name='T1_2_MNI_Registration'
    )

    # should not be necesssary but does not hurt - make sure the multiproc
    # scheduler knows the resource limits
    t1_2_mni.interface.num_threads = settings['ants_nthreads']
    t1_2_mni.interface.estimated_memory_gb = 8

    # 6. FreeSurfer reconstruction
    if settings['freesurfer']:
        nthreads = settings['nthreads']

        def detect_inputs(t1w_list, t2w_list=[]):
            import os
            from nipype.interfaces.base import isdefined
            from nipype.utils.filemanip import filename_to_list
            from nipype.interfaces.traits_extension import Undefined
            import nibabel as nib
            t1w_list = filename_to_list(t1w_list)
            t2w_list = filename_to_list(t2w_list) if isdefined(t2w_list) else []
            t1w_ref = nib.load(t1w_list[0])
            # Use high resolution preprocessing if voxel size < 1.0mm
            # Tolerance of 0.05mm requires that rounds down to 0.9mm or lower
            hires = max(t1w_ref.header.get_zooms()) < 1 - 0.05
            t1w_outs = [t1w_list.pop(0)]
            for t1w in t1w_list:
                img = nib.load(t1w)
                if all((img.shape == t1w_ref.shape,
                        img.header.get_zooms() == t1w_ref.header.get_zooms())):
                    t1w_outs.append(t1w)

            t2w = Undefined
            if t2w_list and max(nib.load(t2w_list[0]).header.get_zooms()) < 1.2:
                t2w = t2w_list[0]

            # https://surfer.nmr.mgh.harvard.edu/fswiki/SubmillimeterRecon
            mris_inflate = '-n 50' if hires else Undefined
            return (t1w_outs, t2w, isdefined(t2w), hires, mris_inflate)

        recon_config = pe.Node(
            niu.Function(
                function=detect_inputs,
                input_names=['t1w_list', 't2w_list'],
                output_names=['t1w', 't2w', 'use_T2', 'hires', 'mris_inflate']),
            name='ReconConfig',
            run_without_submitting=True)

        def bidsinfo(in_file):
            from fmriprep.interfaces.bids import BIDS_NAME
            match = BIDS_NAME.search(in_file)
            params = match.groupdict() if match is not None else {}
            return tuple(map(params.get, ['subject_id', 'ses_id', 'task_id',
                                          'acq_id', 'rec_id', 'run_id']))

        bids_info = pe.Node(
            niu.Function(function=bidsinfo, input_names=['in_file'],
                         output_names=['subject_id', 'ses_id', 'task_id',
                                       'acq_id', 'rec_id', 'run_id']),
            name='BIDSInfo',
            run_without_submitting=True)

        autorecon1 = pe.Node(
            freesurfer.ReconAll(
                directive='autorecon1',
                flags='-noskullstrip',
                openmp=nthreads,
                parallel=True),
            name='Reconstruction')
        autorecon1.interface._can_resume = False
        autorecon1.interface.num_threads = nthreads

        def inject_skullstripped(subjects_dir, subject_id, skullstripped):
            import os
            import nibabel as nib
            from nilearn.image import resample_to_img, new_img_like
            from nipype.utils.filemanip import copyfile
            mridir = os.path.join(subjects_dir, subject_id, 'mri')
            t1 = os.path.join(mridir, 'T1.mgz')
            bm_auto = os.path.join(mridir, 'brainmask.auto.mgz')
            bm = os.path.join(mridir, 'brainmask.mgz')

            if not os.path.exists(bm_auto):
                img = nib.load(t1)
                mask = nib.load(skullstripped)
                bmask = new_img_like(mask, mask.get_data() > 0)
                resampled_mask = resample_to_img(bmask, img, 'nearest')
                masked_image = new_img_like(img, img.get_data() * resampled_mask.get_data())
                masked_image.to_filename(bm_auto)

            if not os.path.exists(bm):
                copyfile(bm_auto, bm, copy=True, use_hardlink=True)

            return subjects_dir, subject_id

        injector = pe.Node(
            niu.Function(
                function=inject_skullstripped,
                input_names=['subjects_dir', 'subject_id', 'skullstripped'],
                output_names=['subjects_dir', 'subject_id']),
            name='InjectSkullstrip')

        reconall = pe.Node(
            ReconAllRPT(
                flags='-noskullstrip',
                openmp=nthreads,
                parallel=True,
                out_report='reconall.svg',
                generate_report=True),
            name='Reconstruction2')
        reconall.interface.num_threads = nthreads

        fs_transform = pe.Node(
            freesurfer.utils.Tkregister2(fsl_out='freesurfer2subT1.mat',
                                         reg_header=True),
            name='FreeSurferTransform')

        recon_report = pe.Node(
            DerivativesDataSink(base_directory=settings['reportlets_dir'],
                                suffix='reconall'),
            name='ReconAll_Report'
        )

    # Resample the brain mask and the tissue probability maps into mni space
    bmask_mni = pe.Node(
        ants.ApplyTransforms(dimension=3, default_value=0,
                             interpolation='NearestNeighbor'),
        name='brain_mni_warp'
    )
    bmask_mni.inputs.reference_image = op.join(get_mni_icbm152_nlin_asym_09c(),
                                               '1mm_T1.nii.gz')
    tpms_mni = pe.MapNode(
        ants.ApplyTransforms(dimension=3, default_value=0,
                             interpolation='Linear'),
        iterfield=['input_image'],
        name='tpms_mni_warp'
    )
    tpms_mni.inputs.reference_image = op.join(get_mni_icbm152_nlin_asym_09c(),
                                              '1mm_T1.nii.gz')

    ds_t1_seg_report = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='t1_seg'),
        name='DS_T1_Seg_Report'
    )

    ds_t1_2_mni_report = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='t1_2_mni'),
        name='DS_T1_2_MNI_Report'
    )

    workflow.connect([
        (inputnode, t1wmrg, [('t1w', 'in_files')]),
        (t1wmrg, asw, [('out_avg', 'inputnode.in_file')]),
        (asw, t1_seg, [('outputnode.out_file', 'in_files')]),
        (asw, t1_2_mni, [('outputnode.bias_corrected', 'moving_image')]),
        (asw, t1_2_mni, [('outputnode.out_mask', 'moving_mask')]),
        (t1_seg, outputnode, [('tissue_class_map', 't1_seg')]),
        (asw, outputnode, [('outputnode.bias_corrected', 'bias_corrected_t1')]),
        (t1_seg, outputnode, [('probability_maps', 't1_tpms')]),
        (t1_2_mni, outputnode, [
            ('warped_image', 't1_2_mni'),
            ('forward_transforms', 't1_2_mni_forward_transform'),
            ('reverse_transforms', 't1_2_mni_reverse_transform')
        ]),
        (asw, bmask_mni, [('outputnode.out_mask', 'input_image')]),
        (t1_2_mni, bmask_mni, [('forward_transforms', 'transforms'),
                               ('forward_invert_flags',
                                'invert_transform_flags')]),
        (t1_seg, tpms_mni, [('probability_maps', 'input_image')]),
        (t1_2_mni, tpms_mni, [('forward_transforms', 'transforms'),
                              ('forward_invert_flags', 'invert_transform_flags')]),
        (asw, outputnode, [('outputnode.out_file', 't1_brain'),
                           ('outputnode.out_mask', 't1_mask')]),
        (inputnode, ds_t1_seg_report, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (t1_seg, ds_t1_seg_report, [('out_report', 'in_file')]),
        (inputnode, ds_t1_2_mni_report, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (t1_2_mni, ds_t1_2_mni_report, [('out_report', 'in_file')])
    ])

    if settings.get('skull_strip_ants', False):
        ds_t1_skull_strip_report = pe.Node(
            DerivativesDataSink(base_directory=settings['reportlets_dir'],
                                suffix='t1_skull_strip'),
            name='DS_Report'
        )
        workflow.connect([
            (inputnode, ds_t1_skull_strip_report, [
                (('t1w', fix_multi_T1w_source_name), 'source_file')]),
            (asw, ds_t1_skull_strip_report, [('outputnode.out_report', 'in_file')])
        ])

    if settings['freesurfer']:
        workflow.connect([
            (inputnode, recon_config, [('t1w', 't1w_list'),
                                       ('t2w', 't2w_list')]),
            (inputnode, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
            (inputnode, autorecon1, [('subjects_dir', 'subjects_dir')]),
            (recon_config, autorecon1, [('t1w', 'T1_files'),
                                        ('t2w', 'T2_file'),
                                        ('hires', 'hires'),
                                        # First run only (recon-all saves expert options)
                                        ('mris_inflate', 'mris_inflate')]),
            (bids_info, autorecon1, [('subject_id', 'subject_id')]),
            (autorecon1, injector, [('subjects_dir', 'subjects_dir'),
                                    ('subject_id', 'subject_id')]),
            (asw, injector, [('outputnode.out_file', 'skullstripped')]),
            (injector, reconall, [('subjects_dir', 'subjects_dir'),
                                  ('subject_id', 'subject_id')]),
            (recon_config, reconall, [('use_T2', 'use_T2')]),
            (t1wmrg, fs_transform, [('out_avg', 'target_image')]),
            (autorecon1, fs_transform, [('T1', 'moving_image')]),
            (recon_config, recon_report, [
                (('t1w', fix_multi_T1w_source_name), 'source_file')]),
            (reconall, recon_report, [('out_report', 'in_file')]),
            (reconall, outputnode, [('subject_id', 'subject_id')]),
            (fs_transform, outputnode, [('fsl_file', 'fs_2_t1_transform')]),
            ])

    # Write corrected file in the designated output dir
    ds_t1_bias = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='preproc'),
        name='DerivT1_inu'
    )
    ds_t1_seg = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='dtissue'),
        name='DerivT1_seg'
    )
    ds_mask = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='brainmask'),
        name='DerivT1_mask'
    )
    ds_t1_mni = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='space-MNI152NLin2009cAsym_preproc'),
        name='DerivT1w_MNI'
    )
    ds_t1_mni_aff = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='target-MNI152NLin2009cAsym_affine'),
        name='DerivT1w_MNI_affine'
    )
    ds_bmask_mni = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='space-MNI152NLin2009cAsym_brainmask'),
        name='DerivT1_Mask_MNI'
    )
    ds_tpms_mni = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='space-MNI152NLin2009cAsym_class-{extra_value}_probtissue'),
        name='DerivT1_TPMs_MNI'
    )
    ds_tpms_mni.inputs.extra_values = ['CSF', 'GM', 'WM']

    if settings.get('debug', False):
        workflow.connect([
            (t1_2_mni, ds_t1_mni_aff, [('forward_transforms', 'in_file')])
        ])
    else:
        ds_t1_mni_warp = pe.Node(
            DerivativesDataSink(base_directory=settings['output_dir'],
                                suffix='target-MNI152NLin2009cAsym_warp'), name='mni_warp')

        def _get_aff(inlist):
            return inlist[:-1]

        def _get_warp(inlist):
            return inlist[-1]

        workflow.connect([
            (inputnode, ds_t1_mni_warp, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
            (t1_2_mni, ds_t1_mni_aff, [
                (('forward_transforms', _get_aff), 'in_file')]),
            (t1_2_mni, ds_t1_mni_warp, [
                (('forward_transforms', _get_warp), 'in_file')])
        ])

    workflow.connect([
        (inputnode, ds_t1_bias, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_t1_seg, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_mask, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_t1_mni, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_t1_mni_aff, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_bmask_mni, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_tpms_mni, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (asw, ds_t1_bias, [('outputnode.bias_corrected', 'in_file')]),
        #  (inu_n4, ds_t1_bias, [('output_image', 'in_file')]),
        (t1_seg, ds_t1_seg, [('tissue_class_map', 'in_file')]),
        (asw, ds_mask, [('outputnode.out_mask', 'in_file')]),
        (t1_2_mni, ds_t1_mni, [('warped_image', 'in_file')]),
        (bmask_mni, ds_bmask_mni, [('output_image', 'in_file')]),
        (tpms_mni, ds_tpms_mni, [('output_image', 'in_file')])

    ])
    return workflow


def skullstrip_ants(name='ANTsBrainExtraction', settings=None):
    from niworkflows.data import get_ants_oasis_template_ras
    if settings is None:
        settings = {'debug': False}

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'source_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bias_corrected', 'out_file', 'out_mask', 'out_report']), name='outputnode')

    t1_skull_strip = pe.Node(BrainExtractionRPT(
        dimension=3, use_floatingpoint_precision=1,
        debug=settings['debug'], generate_report=True,
        num_threads=settings['ants_nthreads'], keep_temporary_files=1),
        name='Ants_T1_Brain_Extraction')

    # should not be necesssary byt does not hurt - make sure the multiproc
    # scheduler knows the resource limits
    t1_skull_strip.interface.num_threads = settings['ants_nthreads']

    t1_skull_strip.inputs.brain_template = op.join(
        get_ants_oasis_template_ras(),
        'T_template0.nii.gz'
    )
    t1_skull_strip.inputs.brain_probability_mask = op.join(
        get_ants_oasis_template_ras(),
        'T_template0_BrainCerebellumProbabilityMask.nii.gz'
    )
    t1_skull_strip.inputs.extraction_registration_mask = op.join(
        get_ants_oasis_template_ras(),
        'T_template0_BrainCerebellumRegistrationMask.nii.gz'
    )

    workflow.connect([
        (inputnode, t1_skull_strip, [('in_file', 'anatomical_image')]),
        (t1_skull_strip, outputnode, [('BrainExtractionMask', 'out_mask'),
                                      ('BrainExtractionBrain', 'out_file'),
                                      ('N4Corrected0', 'bias_corrected'),
                                      ('out_report', 'out_report')])
    ])

    return workflow
