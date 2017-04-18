#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Anatomical Reference -processing workflows.

Originally coded by Craig Moodie. Refactored by the CRN Developers.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os.path as op

from nipype.interfaces import ants
from nipype.interfaces import freesurfer
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nipype.pipeline import engine as pe

from niworkflows.interfaces.registration import RobustMNINormalizationRPT
from niworkflows.anat.skullstrip import afni_wf as skullstrip_wf
from niworkflows.data import get_mni_icbm152_nlin_asym_09c
from niworkflows.interfaces.masks import BrainExtractionRPT
from niworkflows.interfaces.segmentation import FASTRPT, ReconAllRPT

from fmriprep.interfaces import (DerivativesDataSink, IntraModalMerge)
from fmriprep.interfaces.images import reorient
from fmriprep.utils.misc import fix_multi_T1w_source_name


#  pylint: disable=R0914
def t1w_preprocessing(settings, name='t1w_preprocessing'):
    """T1w images preprocessing pipeline"""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1w', 't2w', 'subjects_dir']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_seg', 't1_tpms', 'bias_corrected_t1', 't1_brain', 't1_mask',
                't1_2_mni', 't1_2_mni_forward_transform',
                't1_2_mni_reverse_transform', 'subject_id',
                'fs_2_t1_transform']), name='outputnode')

    # 0. Align and merge if several T1w images are provided
    t1wmrg = pe.Node(IntraModalMerge(), name='MergeT1s')

    # 1. Reorient T1
    arw = pe.Node(niu.Function(input_names=['in_file'],
                               output_names=['out_file'],
                               function=reorient),
                  name='Reorient')

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

    # 6. FreeSurfer reconstruction
    if settings['freesurfer']:
        surface_recon = surface_reconstruction(settings=settings)

    # Resample the brain mask and the tissue probability maps into mni space
    bmask_mni = pe.Node(
        ants.ApplyTransforms(dimension=3, default_value=0, float=True,
                             interpolation='NearestNeighbor'),
        name='brain_mni_warp'
    )
    bmask_mni.inputs.reference_image = op.join(get_mni_icbm152_nlin_asym_09c(),
                                               '1mm_T1.nii.gz')
    tpms_mni = pe.MapNode(
        ants.ApplyTransforms(dimension=3, default_value=0, float=True,
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
        (t1wmrg, arw, [('out_avg', 'in_file')]),
        (arw, asw, [('out_file', 'inputnode.in_file')]),
        (asw, t1_seg, [('outputnode.out_file', 'in_files')]),
        (asw, t1_2_mni, [('outputnode.bias_corrected', 'moving_image')]),
        (asw, t1_2_mni, [('outputnode.out_mask', 'moving_mask')]),
        (t1_seg, outputnode, [('tissue_class_map', 't1_seg')]),
        (asw, outputnode, [('outputnode.bias_corrected', 'bias_corrected_t1')]),
        (t1_seg, outputnode, [('probability_maps', 't1_tpms')]),
        (t1_2_mni, outputnode, [
            ('warped_image', 't1_2_mni'),
            ('composite_transform', 't1_2_mni_forward_transform'),
            ('inverse_composite_transform', 't1_2_mni_reverse_transform')
        ]),
        (asw, bmask_mni, [('outputnode.out_mask', 'input_image')]),
        (t1_2_mni, bmask_mni, [('composite_transform', 'transforms')]),
        (t1_seg, tpms_mni, [('probability_maps', 'input_image')]),
        (t1_2_mni, tpms_mni, [('composite_transform', 'transforms')]),
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
            (inputnode, surface_recon, [('t1w', 'inputnode.t1w'),
                                        ('t2w', 'inputnode.t2w'),
                                        ('subjects_dir', 'inputnode.subjects_dir')]),
            (arw, surface_recon, [('out_file', 'inputnode.reoriented_t1')]),
            (asw, surface_recon, [('outputnode.out_file', 'inputnode.skullstripped_t1')]),
            (surface_recon, outputnode, [('outputnode.subjects_dir', 'subjects_dir'),
                                         ('outputnode.subject_id', 'subject_id'),
                                         ('outputnode.fs_2_t1_transform', 'fs_2_t1_transform')]),
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

    ds_t1_mni_warp = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='target-MNI152NLin2009cAsym_warp'), name='mni_warp')

    workflow.connect([
        (inputnode, ds_t1_mni_warp, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (t1_2_mni, ds_t1_mni_warp, [
            ('composite_transform', 'in_file')])
    ])

    workflow.connect([
        (inputnode, ds_t1_bias, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_t1_seg, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_mask, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (inputnode, ds_t1_mni, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
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


def surface_reconstruction(name='SurfaceReconstruction', settings=None):
    if settings is None:
        settings = {'debug': False}

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't2w', 'reoriented_t1', 'skullstripped_t1', 'subjects_dir']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['subjects_dir', 'subject_id', 'fs_2_t1_transform']), name='outputnode')

    nthreads = settings['nthreads']

    def detect_inputs(t1w_list, t2w_list=[], hires_enabled=True):
        from nipype.interfaces.base import isdefined
        from nipype.utils.filemanip import filename_to_list
        from nipype.interfaces.traits_extension import Undefined
        import nibabel as nib
        t1w_list = filename_to_list(t1w_list)
        t2w_list = filename_to_list(t2w_list) if isdefined(t2w_list) else []
        t1w_ref = nib.load(t1w_list[0])
        # Use high resolution preprocessing if voxel size < 1.0mm
        # Tolerance of 0.05mm requires that rounds down to 0.9mm or lower
        hires = hires_enabled and max(t1w_ref.header.get_zooms()) < 1 - 0.05
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
            input_names=['t1w_list', 't2w_list', 'hires_enabled'],
            output_names=['t1w', 't2w', 'use_T2', 'hires', 'mris_inflate']),
        name='ReconConfig',
        run_without_submitting=True)
    recon_config.inputs.hires_enabled = settings['hires']

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
        name='AutoRecon1')
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
        name='ReconAll')
    reconall.interface.num_threads = nthreads

    fs_transform = pe.Node(
        freesurfer.Tkregister2(fsl_out='freesurfer2subT1.mat',
                               reg_header=True),
        name='FreeSurferTransform')

    recon_report = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='reconall'),
        name='ReconAll_Report'
    )

    midthickness = pe.MapNode(
        freesurfer.MRIsExpand(thickness=True, distance=0.5,
                              out_name='midthickness'),
        iterfield='in_file',
        name='MidThickness')

    save_midthickness = pe.Node(nio.DataSink(parameterization=False),
                                name='SaveMidthickness')
    surface_list = pe.Node(niu.Merge(4), name='SurfaceList')
    gifticonv = pe.MapNode(freesurfer.MRIsConvert(out_datatype='gii'),
                           iterfield='in_file', name='GiftiSurfaces')

    def get_gifti_name(in_file):
        import os
        import re
        in_format = re.compile(r'(?P<LR>[lr])h.(?P<surf>.+)_converted.gii')
        name = os.path.basename(in_file)
        info = in_format.match(name).groupdict()
        info['LR'] = info['LR'].upper()
        return '{surf}.{LR}.surf'.format(**info)

    name_surfs = pe.MapNode(
        niu.Function(
            function=get_gifti_name,
            input_names=['in_file'],
            output_names=['normalized']),
        iterfield='in_file',
        name='NameSurfs'
        )

    def normalize_surfs(in_file):
        """ Re-center GIFTI coordinates to fit align to native T1 space

        For midthickness surfaces, add MidThickness metadata

        Coordinate update based on:
        https://github.com/Washington-University/workbench/blob/1b79e56/src/Algorithms/AlgorithmSurfaceApplyAffine.cxx#L73-L91
        and
        https://github.com/Washington-University/Pipelines/blob/ae69b9a/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh#L147
        """
        import os
        import numpy as np
        import nibabel as nib
        img = nib.load(in_file)
        pointset = img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]
        coords = pointset.data
        c_ras_keys = ('VolGeomC_R', 'VolGeomC_A', 'VolGeomC_S')
        ras = np.array([float(pointset.metadata[key])
                        for key in c_ras_keys])
        # Apply C_RAS translation to coordinates
        pointset.data = (coords + ras).astype(coords.dtype)

        secondary = nib.gifti.GiftiNVPairs('AnatomicalStructureSecondary',
                                           'MidThickness')
        geom_type = nib.gifti.GiftiNVPairs('GeometricType', 'Anatomical')
        has_ass = has_geo = False
        for nvpair in pointset.meta.data:
            # Remove C_RAS translation from metadata to avoid double-dipping in FreeSurfer
            if nvpair.name in c_ras_keys:
                nvpair.value = '0.000000'
            # Check for missing metadata
            elif nvpair.name == secondary.name:
                has_ass = True
            elif nvpair.name == geom_type.name:
                has_geo = True
        fname = os.path.basename(in_file)
        # Update metadata for MidThickness/graymid surfaces
        if 'midthickness' in fname.lower() or 'graymid' in fname.lower():
            if not has_ass:
                pointset.meta.data.insert(1, secondary)
            if not has_geo:
                pointset.meta.data.insert(2, geom_type)
        img.to_filename(fname)
        return os.path.abspath(fname)

    fix_surfs = pe.MapNode(
        niu.Function(
            function=normalize_surfs,
            input_names=['in_file'],
            output_names=['out_file']),
        iterfield='in_file',
        name='FixSurfs')

    ds_surfs = pe.MapNode(
        DerivativesDataSink(base_directory=settings['output_dir']),
        iterfield=['in_file', 'suffix'],
        name='DerivSurfs'
    )

    workflow.connect([
        # Configuration
        (inputnode, recon_config, [('t1w', 't1w_list'),
                                   ('t2w', 't2w_list')]),
        (inputnode, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
        # Passing subjects_dir / subject_id enforces serial order
        (inputnode, autorecon1, [('subjects_dir', 'subjects_dir')]),
        (bids_info, autorecon1, [('subject_id', 'subject_id')]),
        (autorecon1, injector, [('subjects_dir', 'subjects_dir'),
                                ('subject_id', 'subject_id')]),
        (injector, reconall, [('subjects_dir', 'subjects_dir'),
                              ('subject_id', 'subject_id')]),
        (reconall, outputnode, [('subjects_dir', 'subjects_dir'),
                                ('subject_id', 'subject_id')]),
        # Reconstruction phases
        (recon_config, autorecon1, [('t1w', 'T1_files'),
                                    ('t2w', 'T2_file'),
                                    ('hires', 'hires'),
                                    # First run only (recon-all saves expert options)
                                    ('mris_inflate', 'mris_inflate')]),
        (inputnode, injector, [('skullstripped_t1', 'skullstripped')]),
        (recon_config, reconall, [('use_T2', 'use_T2')]),
        # Display surface contours on structural image
        (recon_config, recon_report, [
            (('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (reconall, recon_report, [('out_report', 'in_file')]),
        # Construct transform from FreeSurfer conformed image to FMRIPREP
        # reoriented image
        (inputnode, fs_transform, [('reoriented_t1', 'target_image')]),
        (autorecon1, fs_transform, [('T1', 'moving_image')]),
        (fs_transform, outputnode, [('fsl_file', 'fs_2_t1_transform')]),
        # Generate midthickness surfaces and save to FreeSurfer derivatives
        (reconall, midthickness, [('smoothwm', 'in_file')]),
        (reconall, save_midthickness, [('subjects_dir', 'base_directory'),
                                       ('subject_id', 'container')]),
        (midthickness, save_midthickness, [('out_file', 'surf.@graymid')]),
        # Produce valid GIFTI surface files (dense mesh)
        (reconall, surface_list, [('smoothwm', 'in1'),
                                  ('pial', 'in2'),
                                  ('inflated', 'in3')]),
        (save_midthickness, surface_list, [('out_file', 'in4')]),
        (surface_list, gifticonv, [('out', 'in_file')]),
        (gifticonv, name_surfs, [('converted', 'in_file')]),
        (gifticonv, fix_surfs, [('converted', 'in_file')]),
        (inputnode, ds_surfs, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (name_surfs, ds_surfs, [('normalized', 'suffix')]),
        (fix_surfs, ds_surfs, [('out_file', 'in_file')]),
        ])

    return workflow
