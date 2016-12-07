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
from nipype.interfaces import fsl
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from niworkflows.interfaces.registration import RobustMNINormalizationRPT
from niworkflows.anat.skullstrip import afni_wf as skullstrip_wf
from niworkflows.data import get_mni_icbm152_nlin_asym_09c
from niworkflows.interfaces.masks import BrainExtractionRPT
from niworkflows.interfaces.segmentation import FASTRPT

from fmriprep.interfaces import (DerivativesDataSink, IntraModalMerge,
                                 ImageDataSink, RASReorient)

from fmriprep.viz import stripped_brain_overlay


#  pylint: disable=R0914
def t1w_preprocessing(name='t1w_preprocessing', settings=None):
    """T1w images preprocessing pipeline"""

    if settings is None:
        raise RuntimeError('Workflow settings are missing')

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['t1w']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_seg', 'bias_corrected_t1', 't1_brain', 't1_2_mni',
                't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                't1_segmentation']), name='outputnode')

    # 0. Align, reorient to RAS and merge if several T1w images are provided
    t1wmrg = pe.Node(IntraModalMerge(to_ras=True, hmc=True), name='MergeT1s')


    # 2. T1 Bias Field Correction
    inu_n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3),
                     name='CorrectINU')

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
    # should not be necesssary byt does not hurt - make sure the multiproc
    # scheduler knows the resource limits
    t1_2_mni.interface.num_threads = settings['ants_nthreads']

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
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='t1_seg', out_path_base='reports'),
        name='DS_T1_Seg_Report'
    )

    ds_t1_2_mni_report = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='t1_2_mni', out_path_base='reports'),
        name='DS_T1_2_MNI_Report'
    )

    workflow.connect([
        (inputnode, t1wmrg, [('t1w', 'in_files')]),
        (t1wmrg, inu_n4, [('out_avg', 'input_image')]),
        (inu_n4, asw, [('output_image', 'inputnode.in_file')]),
        (asw, t1_seg, [('outputnode.out_file', 'in_files')]),
        (inu_n4, t1_2_mni, [('output_image', 'moving_image')]),
        (asw, t1_2_mni, [('outputnode.out_mask', 'moving_mask')]),
        (t1_seg, outputnode, [('tissue_class_map', 't1_seg')]),
        (inu_n4, outputnode, [('output_image', 'bias_corrected_t1')]),
        (t1_seg, outputnode, [('tissue_class_map', 't1_segmentation')]),
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
        (asw, outputnode, [('outputnode.out_file', 't1_brain')]),
        (inputnode, ds_t1_seg_report, [('t1w', 'source_file')]),
        (t1_seg, ds_t1_seg_report, [('out_report', 'in_file')]),
        (inputnode, ds_t1_2_mni_report, [('t1w', 'source_file')]),
        (t1_2_mni, ds_t1_2_mni_report, [('out_report', 'in_file')])
    ])

    if settings.get('skull_strip_ants', False):
        ds_t1_skull_strip_report = pe.Node(
            DerivativesDataSink(base_directory=settings['output_dir'],
                                suffix='t1_skull_strip', out_path_base='reports'),
            name='DS_Report'
        )
        workflow.connect([
            (inputnode, ds_t1_skull_strip_report, [('t1w', 'source_file')]),
            (asw, ds_t1_skull_strip_report, [('outputnode.out_report', 'in_file')])
        ])

    # Connect reporting nodes
    t1_stripped_overlay = pe.Node(
        niu.Function(
            input_names=['in_file', 'overlay_file', 'out_file'],
            output_names=['out_file'],
            function=stripped_brain_overlay
        ),
        name='PNG_T1_SkullStrip'
    )
    t1_stripped_overlay.inputs.out_file = 't1_stripped_overlay.svg'

    t1_stripped_overlay_ds = pe.Node(
        ImageDataSink(base_directory=settings['output_dir']),
        name='T1StrippedOverlayDS'
    )

    #  t1 segmentation mapped onto t1 before registration into mni space
    t1_seg_native = pe.Node(
        niu.Function(
            input_names=['in_file', 'overlay_file', 'out_file'],
            output_names=['out_file'],
            function=stripped_brain_overlay
        ),
        name='T1SegNative'
    )
    t1_seg_native.inputs.out_file = 't1_seg_native.svg'

    t1_seg_native_ds = pe.Node(
        ImageDataSink(base_directory=settings['output_dir']),
        name='T1SegNativeDS'
    )

    #  The T1-to-MNI will be plotted using the segmentation.
    #  That's why we transform it first
    seg_2_mni = pe.Node(
        ants.ApplyTransforms(dimension=3, default_value=0,
                             interpolation='NearestNeighbor'),
        name='T1_2_MNI_warp'
    )
    seg_2_mni.inputs.reference_image = op.join(get_mni_icbm152_nlin_asym_09c(),
                                               '1mm_T1.nii.gz')

    t1_2_mni_overlay = pe.Node(
        niu.Function(
            input_names=['in_file', 'overlay_file', 'out_file'],
            output_names=['out_file'],
            function=stripped_brain_overlay
        ),
        name='T1ToMNI'
    )
    t1_2_mni_overlay.inputs.out_file = 't1_to_mni_overlay.svg'
    t1_2_mni_overlay.inputs.overlay_file = op.join(get_mni_icbm152_nlin_asym_09c(),
                                                   '1mm_T1.nii.gz')

    t1_2_mni_overlay_ds = pe.Node(
        ImageDataSink(base_directory=settings['output_dir']),
        name='T12MNIOverlayDS'
    )
    t1_2_mni_overlay_ds.inputs.overlay_file = op.join(get_mni_icbm152_nlin_asym_09c(),
                                                      '1mm_T1.nii.gz')

    datasink = pe.Node(
        interface=nio.DataSink(
            base_directory=op.join(settings['output_dir'], 'images')
        ),
        name='datasink',
        parameterization=False
    )


    workflow.connect([
        (inu_n4, t1_stripped_overlay, [('output_image', 'overlay_file')]),
        (asw, t1_stripped_overlay, [('outputnode.out_mask', 'in_file')]),
        (t1_stripped_overlay, t1_stripped_overlay_ds, [('out_file', 'in_file')]),
        (inu_n4, t1_stripped_overlay_ds, [('output_image', 'base_file')]),
        (asw, t1_stripped_overlay_ds, [('outputnode.out_mask', 'overlay_file')]),
        (inputnode, t1_stripped_overlay_ds, [('t1w', 'origin_file')]),

        (t1_seg, seg_2_mni, [('tissue_class_map', 'input_image')]),
        (t1_2_mni, seg_2_mni, [('forward_transforms', 'transforms'),
                               ('forward_invert_flags', 'invert_transform_flags')]),

        (seg_2_mni, t1_2_mni_overlay, [('output_image', 'in_file')]),
        (t1_2_mni_overlay, datasink, [('out_file', '@t1_2_mni_overlay')]),
        (seg_2_mni, t1_2_mni_overlay_ds, [('output_image', 'base_file')]),
        (t1_2_mni_overlay, t1_2_mni_overlay_ds, [('out_file', 'in_file')]),
        (inputnode, t1_2_mni_overlay_ds, [('t1w', 'origin_file')]),

        (t1_seg, t1_seg_native, [('tissue_class_map', 'in_file')]),
        (inu_n4, t1_seg_native, [('output_image', 'overlay_file')]),
        (t1_seg, t1_seg_native_ds, [('tissue_class_map', 'base_file')]),
        (inu_n4, t1_seg_native_ds, [('output_image', 'overlay_file')]),
        (inputnode, t1_seg_native_ds, [('t1w', 'origin_file')]),
        (t1_seg_native, t1_seg_native_ds, [('out_file', 'in_file')])
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
            (inputnode, ds_t1_mni_warp, [('t1w', 'source_file')]),
            (t1_2_mni, ds_t1_mni_aff, [
                (('forward_transforms', _get_aff), 'in_file')]),
            (t1_2_mni, ds_t1_mni_warp, [
                (('forward_transforms', _get_warp), 'in_file')])
        ])

    workflow.connect([
        (inputnode, ds_t1_bias, [('t1w', 'source_file')]),
        (inputnode, ds_t1_seg, [('t1w', 'source_file')]),
        (inputnode, ds_mask, [('t1w', 'source_file')]),
        (inputnode, ds_t1_mni, [('t1w', 'source_file')]),
        (inputnode, ds_t1_mni_aff, [('t1w', 'source_file')]),
        (inputnode, ds_bmask_mni, [('t1w', 'source_file')]),
        (inputnode, ds_tpms_mni, [('t1w', 'source_file')]),
        (inu_n4, ds_t1_bias, [('output_image', 'in_file')]),
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
        fields=['out_file', 'out_mask', 'out_report']), name='outputnode')

    t1_skull_strip = pe.Node(BrainExtractionRPT(
        dimension=3, use_floatingpoint_precision=1,
        debug=settings['debug'], generate_report=True,
        num_threads=settings['ants_nthreads']),
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
                                      ('out_report', 'out_report')])
    ])

    return workflow
