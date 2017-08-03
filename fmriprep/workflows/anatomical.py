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

from niworkflows.nipype.interfaces import ants
from niworkflows.nipype.interfaces import freesurfer as fs
from niworkflows.nipype.interfaces import utility as niu
from niworkflows.nipype.interfaces import io as nio
from niworkflows.nipype.pipeline import engine as pe

from niworkflows.interfaces.registration import RobustMNINormalizationRPT
from niworkflows.anat.skullstrip import afni_wf as init_skullstrip_afni_wf
import niworkflows.data as nid
from niworkflows.interfaces.masks import BrainExtractionRPT
from niworkflows.interfaces.segmentation import FASTRPT, ReconAllRPT

from ..interfaces import (
    DerivativesDataSink, StructuralReference, LTAConvert, MakeMidthickness, FSInjectBrainExtracted,
    FSDetectInputs, BIDSInfo, NormalizeSurf, GiftiNameSource, ConformSeries, AnatomicalSummary
)
from ..utils.misc import fix_multi_T1w_source_name, add_suffix


#  pylint: disable=R0914
def init_anat_preproc_wf(skull_strip_ants, output_spaces, template, debug, freesurfer,
                         longitudinal, omp_nthreads, hires, reportlets_dir, output_dir,
                         name='anat_preproc_wf'):
    """T1w images preprocessing pipeline"""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1w', 't2w', 'subjects_dir']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_tpms',
                't1_2_mni', 't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                'mni_mask', 'mni_seg', 'mni_tpms',
                'subjects_dir', 'subject_id', 'fs_2_t1_transform', 'surfaces']),
        name='outputnode')

    bids_info = pe.Node(BIDSInfo(), name='bids_info', run_without_submitting=True)

    summary = pe.Node(
        AnatomicalSummary(output_spaces=output_spaces, template=template),
        name='summary')

    # 0. Reorient T1w image(s) to RAS and resample to common voxel space
    t1_conform = pe.Node(ConformSeries(), name='t1_conform')

    # 1. Align and merge if several T1w images are provided
    t1_merge = pe.Node(
        # StructuralReference is fs.RobustTemplate if > 1 volume, copying otherwise
        StructuralReference(auto_detect_sensitivity=True,
                            initial_timepoint=1,      # For deterministic behavior
                            intensity_scaling=True,   # 7-DOF (rigid + intensity)
                            subsample_threshold=200,
                            transform_outputs=True,
                            ),
        name='t1_merge')

    convert_xfms = pe.MapNode(LTAConvert(out_fsl=True), name='convert_xfms', iterfield=['in_lta'])

    # 2. T1 Bias Field Correction
    # Bias field correction is handled in skull strip workflows.

    # 3. Skull-stripping
    skullstrip_wf = init_skullstrip_afni_wf(name='skullstrip_afni_wf')
    if skull_strip_ants:
        skullstrip_wf = init_skullstrip_ants_wf(name='skullstrip_ants_wf',
                                                debug=debug,
                                                omp_nthreads=omp_nthreads)

    # 4. Segmentation
    t1_seg = pe.Node(FASTRPT(generate_report=True, segments=True,
                             no_bias=True, probability_maps=True),
                     name='t1_seg', mem_gb=3)

    # 5. Spatial normalization (T1w to MNI registration)
    t1_2_mni = pe.Node(
        RobustMNINormalizationRPT(
            float=True,
            generate_report=True,
            num_threads=omp_nthreads,
            flavor='testing' if debug else 'precise',
        ),
        name='t1_2_mni',
        n_procs=omp_nthreads
    )

    # Resample the brain mask and the tissue probability maps into mni space
    mni_mask = pe.Node(
        ants.ApplyTransforms(dimension=3, default_value=0, float=True,
                             interpolation='NearestNeighbor'),
        name='mni_mask'
    )

    mni_seg = pe.Node(
        ants.ApplyTransforms(dimension=3, default_value=0, float=True,
                             interpolation='NearestNeighbor'),
        name='mni_seg'
    )

    mni_tpms = pe.MapNode(
        ants.ApplyTransforms(dimension=3, default_value=0, float=True,
                             interpolation='Linear'),
        iterfield=['input_image'],
        name='mni_tpms'
    )

    def set_threads(in_list, maximum):
        return min(len(in_list), maximum)

    def len_above_thresh(in_list, threshold, longitudinal):
        if longitudinal:
            return False
        return len(in_list) > threshold

    workflow.connect([
        (inputnode, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
        (inputnode, t1_conform, [('t1w', 't1w_list')]),
        (t1_conform, t1_merge, [
            ('t1w_list', 'in_files'),
            (('t1w_list', set_threads, omp_nthreads), 'num_threads'),
            (('t1w_list', len_above_thresh, 2, longitudinal), 'fixed_timepoint'),
            (('t1w_list', len_above_thresh, 2, longitudinal), 'no_iteration'),
            (('t1w_list', add_suffix, '_template'), 'out_file')]),
        (t1_merge, skullstrip_wf, [('out_file', 'inputnode.in_file')]),
        (t1_merge, convert_xfms, [('transform_outputs', 'in_lta')]),
        (skullstrip_wf, t1_seg, [('outputnode.out_file', 'in_files')]),
        (skullstrip_wf, outputnode, [('outputnode.bias_corrected', 't1_preproc'),
                                     ('outputnode.out_file', 't1_brain'),
                                     ('outputnode.out_mask', 't1_mask')]),
        (t1_seg, outputnode, [('tissue_class_map', 't1_seg'),
                              ('probability_maps', 't1_tpms')]),
        (inputnode, summary, [('t1w', 't1w')]),
    ])

    if 'template' in output_spaces:
        template_str = nid.TEMPLATE_MAP[template]
        ref_img = op.join(nid.get_dataset(template_str), '1mm_T1.nii.gz')

        t1_2_mni.inputs.template = template_str
        mni_mask.inputs.reference_image = ref_img
        mni_seg.inputs.reference_image = ref_img
        mni_tpms.inputs.reference_image = ref_img

        workflow.connect([
            (skullstrip_wf, t1_2_mni, [('outputnode.bias_corrected', 'moving_image')]),
            (skullstrip_wf, t1_2_mni, [('outputnode.out_mask', 'moving_mask')]),
            (skullstrip_wf, mni_mask, [('outputnode.out_mask', 'input_image')]),
            (t1_2_mni, mni_mask, [('composite_transform', 'transforms')]),
            (t1_seg, mni_seg, [('tissue_class_map', 'input_image')]),
            (t1_2_mni, mni_seg, [('composite_transform', 'transforms')]),
            (t1_seg, mni_tpms, [('probability_maps', 'input_image')]),
            (t1_2_mni, mni_tpms, [('composite_transform', 'transforms')]),
            (t1_2_mni, outputnode, [
                ('warped_image', 't1_2_mni'),
                ('composite_transform', 't1_2_mni_forward_transform'),
                ('inverse_composite_transform', 't1_2_mni_reverse_transform')]),
            (mni_mask, outputnode, [('output_image', 'mni_mask')]),
            (mni_seg, outputnode, [('output_image', 'mni_seg')]),
            (mni_tpms, outputnode, [('output_image', 'mni_tpms')]),
        ])

    # 6. FreeSurfer reconstruction
    if freesurfer:
        surface_recon_wf = init_surface_recon_wf(name='surface_recon_wf',
                                                 omp_nthreads=omp_nthreads, hires=hires)

        workflow.connect([
            (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
            (bids_info, summary, [('subject_id', 'subject_id')]),
            (inputnode, surface_recon_wf, [
                ('t2w', 'inputnode.t2w'),
                ('subjects_dir', 'inputnode.subjects_dir')]),
            (summary, surface_recon_wf, [('subject_id', 'inputnode.subject_id')]),
            (t1_merge, surface_recon_wf, [('out_file', 'inputnode.t1w')]),
            (skullstrip_wf, surface_recon_wf, [
                ('outputnode.out_file', 'inputnode.skullstripped_t1')]),
            (surface_recon_wf, outputnode, [
                ('outputnode.subjects_dir', 'subjects_dir'),
                ('outputnode.subject_id', 'subject_id'),
                ('outputnode.fs_2_t1_transform', 'fs_2_t1_transform'),
                ('outputnode.surfaces', 'surfaces')]),
        ])

    anat_reports_wf = init_anat_reports_wf(
        reportlets_dir=reportlets_dir, skull_strip_ants=skull_strip_ants,
        output_spaces=output_spaces, template=template, freesurfer=freesurfer)
    workflow.connect([
        (inputnode, anat_reports_wf, [
            (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file')]),
        (summary, anat_reports_wf, [('out_report', 'inputnode.summary_report')]),
        (t1_conform, anat_reports_wf, [('out_report', 'inputnode.t1_conform_report')]),
        (t1_seg, anat_reports_wf, [('out_report', 'inputnode.t1_seg_report')]),
    ])

    if skull_strip_ants:
        workflow.connect([
            (skullstrip_wf, anat_reports_wf, [
                ('outputnode.out_report', 'inputnode.t1_skull_strip_report')])
        ])
    if freesurfer:
        workflow.connect([
            (surface_recon_wf, anat_reports_wf, [
                ('outputnode.out_report', 'inputnode.recon_report')])
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (t1_2_mni, anat_reports_wf, [('out_report', 'inputnode.t1_2_mni_report')]),
        ])

    anat_derivatives_wf = init_anat_derivatives_wf(output_dir=output_dir,
                                                   output_spaces=output_spaces,
                                                   template=template,
                                                   freesurfer=freesurfer)

    workflow.connect([
        (inputnode, anat_derivatives_wf, [
            (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file')]),
        (outputnode, anat_derivatives_wf, [
            ('t1_preproc', 'inputnode.t1_preproc'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_tpms', 'inputnode.t1_tpms'),
            ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
            ('t1_2_mni', 'inputnode.t1_2_mni'),
            ('mni_mask', 'inputnode.mni_mask'),
            ('mni_seg', 'inputnode.mni_seg'),
            ('mni_tpms', 'inputnode.mni_tpms'),
            ('surfaces', 'inputnode.surfaces'),
        ]),
    ])

    return workflow


def init_skullstrip_ants_wf(debug, omp_nthreads, name='skullstrip_ants_wf'):
    from niworkflows.data import get_ants_oasis_template_ras

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'source_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bias_corrected', 'out_file', 'out_mask', 'out_report']), name='outputnode')

    t1_skull_strip = pe.Node(BrainExtractionRPT(
        dimension=3, use_floatingpoint_precision=1,
        debug=debug, generate_report=True,
        num_threads=omp_nthreads, keep_temporary_files=1),
        name='t1_skull_strip', n_procs=omp_nthreads)

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


def init_surface_recon_wf(omp_nthreads, hires, name='surface_recon_wf'):

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't2w', 'skullstripped_t1', 'subjects_dir', 'subject_id']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'fs_2_t1_transform', 'surfaces', 'out_report']),
        name='outputnode')

    recon_config = pe.Node(FSDetectInputs(hires_enabled=hires), name='recon_config',
                           run_without_submitting=True)

    autorecon1 = pe.Node(
        fs.ReconAll(
            directive='autorecon1',
            flags='-noskullstrip',
            openmp=omp_nthreads),
        name='autorecon1',
        n_procs=omp_nthreads)
    autorecon1.interface._can_resume = False

    skull_strip_extern = pe.Node(FSInjectBrainExtracted(), name='skull_strip_extern',
                                 run_without_submitting=True)

    fs_transform = pe.Node(
        fs.Tkregister2(fsl_out='freesurfer2subT1.mat', reg_header=True),
        name='fs_transform')

    autorecon_resume_wf = init_autorecon_resume_wf(omp_nthreads=omp_nthreads)
    gifti_surface_wf = init_gifti_surface_wf()

    workflow.connect([
        # Configuration
        (inputnode, recon_config, [('t1w', 't1w_list'),
                                   ('t2w', 't2w_list')]),
        # Passing subjects_dir / subject_id enforces serial order
        (inputnode, autorecon1, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id')]),
        (autorecon1, skull_strip_extern, [('subjects_dir', 'subjects_dir'),
                                          ('subject_id', 'subject_id')]),
        (skull_strip_extern, autorecon_resume_wf, [('subjects_dir', 'inputnode.subjects_dir'),
                                                   ('subject_id', 'inputnode.subject_id')]),
        (autorecon_resume_wf, gifti_surface_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        # Reconstruction phases
        (inputnode, autorecon1, [('t1w', 'T1_files')]),
        (recon_config, autorecon1, [('t2w', 'T2_file'),
                                    ('hires', 'hires'),
                                    # First run only (recon-all saves expert options)
                                    ('mris_inflate', 'mris_inflate')]),
        (inputnode, skull_strip_extern, [('skullstripped_t1', 'in_brain')]),
        (recon_config, autorecon_resume_wf, [('use_t2w', 'inputnode.use_T2')]),
        # Construct transform from FreeSurfer conformed image to FMRIPREP
        # reoriented image
        (inputnode, fs_transform, [('t1w', 'target_image')]),
        (autorecon1, fs_transform, [('T1', 'moving_image')]),
        # Output
        (autorecon_resume_wf, outputnode, [('outputnode.subjects_dir', 'subjects_dir'),
                                           ('outputnode.subject_id', 'subject_id'),
                                           ('outputnode.out_report', 'out_report')]),
        (gifti_surface_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
        (fs_transform, outputnode, [('fsl_file', 'fs_2_t1_transform')]),
    ])

    return workflow


def init_autorecon_resume_wf(omp_nthreads, name='autorecon_resume_wf'):
    """
    Resume broken recon-all execution
    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'use_T2']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'out_report']),
        name='outputnode')

    autorecon2_vol = pe.Node(
        fs.ReconAll(directive='autorecon2-volonly', openmp=omp_nthreads),
        n_procs=omp_nthreads,
        name='autorecon2_vol')

    autorecon_surfs = pe.MapNode(
        fs.ReconAll(
            directive='autorecon-hemi',
            flags=['-noparcstats', '-nocortparc2', '-noparcstats2',
                   '-nocortparc3', '-noparcstats3', '-nopctsurfcon',
                   '-nohyporelabel', '-noaparc2aseg', '-noapas2aseg',
                   '-nosegstats', '-nowmparc', '-nobalabels'],
            openmp=omp_nthreads),
        iterfield='hemi', n_procs=omp_nthreads,
        name='autorecon_surfs')
    autorecon_surfs.inputs.hemi = ['lh', 'rh']

    autorecon3 = pe.MapNode(
        fs.ReconAll(directive='autorecon3', openmp=omp_nthreads),
        iterfield='hemi', n_procs=omp_nthreads,
        name='autorecon3')
    autorecon3.inputs.hemi = ['lh', 'rh']

    # Only generate the report once; should be nothing to do
    recon_report = pe.Node(
        ReconAllRPT(directive='autorecon3', generate_report=True),
        name='recon_report')

    def _dedup(in_list):
        vals = set(in_list)
        if len(vals) > 1:
            raise ValueError(
                "Non-identical values can't be deduplicated:\n{!r}".format(in_list))
        return vals.pop()

    workflow.connect([
        (inputnode, autorecon3, [('use_T2', 'use_T2')]),
        (inputnode, autorecon2_vol, [('subjects_dir', 'subjects_dir'),
                                     ('subject_id', 'subject_id')]),
        (autorecon2_vol, autorecon_surfs, [('subjects_dir', 'subjects_dir'),
                                           ('subject_id', 'subject_id')]),
        (autorecon_surfs, autorecon3, [(('subjects_dir', _dedup), 'subjects_dir'),
                                       (('subject_id', _dedup), 'subject_id')]),
        (autorecon3, outputnode, [(('subjects_dir', _dedup), 'subjects_dir'),
                                  (('subject_id', _dedup), 'subject_id')]),
        (autorecon3, recon_report, [(('subjects_dir', _dedup), 'subjects_dir'),
                                    (('subject_id', _dedup), 'subject_id')]),
        (recon_report, outputnode, [('out_report', 'out_report')]),
    ])

    return workflow


def init_gifti_surface_wf(name='gifti_surface_wf'):
    """
    Extract surfaces from FreeSurfer derivatives folder and
    re-center GIFTI coordinates to align to native T1 space

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(['subjects_dir', 'subject_id']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['surfaces']), name='outputnode')

    get_surfaces = pe.Node(nio.FreeSurferSource(), name='get_surfaces')

    midthickness = pe.MapNode(
        MakeMidthickness(thickness=True, distance=0.5, out_name='midthickness'),
        iterfield='in_file',
        name='midthickness')

    save_midthickness = pe.Node(nio.DataSink(parameterization=False),
                                name='save_midthickness')

    surface_list = pe.Node(niu.Merge(4, ravel_inputs=True),
                           name='surface_list', run_without_submitting=True)
    fs_2_gii = pe.MapNode(fs.MRIsConvert(out_datatype='gii'),
                          iterfield='in_file', name='fs_2_gii')
    fix_surfs = pe.MapNode(NormalizeSurf(), iterfield='in_file', name='fix_surfs')

    workflow.connect([
        (inputnode, get_surfaces, [('subjects_dir', 'subjects_dir'),
                                   ('subject_id', 'subject_id')]),
        (inputnode, save_midthickness, [('subjects_dir', 'base_directory'),
                                        ('subject_id', 'container')]),
        # Generate midthickness surfaces and save to FreeSurfer derivatives
        (get_surfaces, midthickness, [('smoothwm', 'in_file'),
                                      ('graymid', 'graymid')]),
        (midthickness, save_midthickness, [('out_file', 'surf.@graymid')]),
        # Produce valid GIFTI surface files (dense mesh)
        (get_surfaces, surface_list, [('smoothwm', 'in1'),
                                      ('pial', 'in2'),
                                      ('inflated', 'in3')]),
        (save_midthickness, surface_list, [('out_file', 'in4')]),
        (surface_list, fs_2_gii, [('out', 'in_file')]),
        (fs_2_gii, fix_surfs, [('converted', 'in_file')]),
        (fix_surfs, outputnode, [('out_file', 'surfaces')]),
    ])
    return workflow


def init_anat_reports_wf(reportlets_dir, skull_strip_ants, output_spaces,
                         template, freesurfer, name='anat_reports_wf'):
    """
    Set up a battery of datasinks to store reports in the right location
    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 'summary_report', 't1_conform_report', 't1_seg_report',
                    't1_2_mni_report', 't1_skull_strip_report', 'recon_report']),
        name='inputnode')

    ds_summary_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='summary'),
        name='ds_summary_report', run_without_submitting=True)

    ds_t1_conform_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='conform'),
        name='ds_t1_conform_report', run_without_submitting=True)

    ds_t1_seg_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='t1_seg'),
        name='ds_t1_seg_report', run_without_submitting=True)

    ds_t1_2_mni_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='t1_2_mni'),
        name='ds_t1_2_mni_report', run_without_submitting=True)

    ds_t1_skull_strip_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='t1_skull_strip'),
        name='ds_t1_skull_strip_report', run_without_submitting=True)

    ds_recon_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='reconall'),
        name='ds_recon_report', run_without_submitting=True)

    workflow.connect([
        (inputnode, ds_summary_report, [('source_file', 'source_file'),
                                        ('summary_report', 'in_file')]),
        (inputnode, ds_t1_conform_report, [('source_file', 'source_file'),
                                           ('t1_conform_report', 'in_file')]),
        (inputnode, ds_t1_seg_report, [('source_file', 'source_file'),
                                       ('t1_seg_report', 'in_file')]),
    ])

    if skull_strip_ants:
        workflow.connect([
            (inputnode, ds_t1_skull_strip_report, [('source_file', 'source_file'),
                                                   ('t1_skull_strip_report', 'in_file')])
        ])
    if freesurfer:
        workflow.connect([
            (inputnode, ds_recon_report, [('source_file', 'source_file'),
                                          ('recon_report', 'in_file')])
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (inputnode, ds_t1_2_mni_report, [('source_file', 'source_file'),
                                             ('t1_2_mni_report', 'in_file')])
        ])

    return workflow


def init_anat_derivatives_wf(output_dir, output_spaces, template, freesurfer,
                             name='anat_derivatives_wf'):
    """
    Set up a battery of datasinks to store derivatives in the right location
    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 't1_preproc', 't1_mask', 't1_seg', 't1_tpms',
                    't1_2_mni_forward_transform', 't1_2_mni', 'mni_mask',
                    'mni_seg', 'mni_tpms', 'surfaces']),
        name='inputnode')

    ds_t1_preproc = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='preproc'),
        name='ds_t1_preproc', run_without_submitting=True)

    ds_t1_mask = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='brainmask'),
        name='ds_t1_mask', run_without_submitting=True)

    ds_t1_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='dtissue'),
        name='ds_t1_seg', run_without_submitting=True)

    ds_t1_tpms = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix='class-{extra_value}_probtissue'),
        name='ds_t1_tpms', run_without_submitting=True)
    ds_t1_tpms.inputs.extra_values = ['CSF', 'GM', 'WM']

    suffix_fmt = 'space-{}_{}'.format
    ds_t1_mni = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'preproc')),
        name='ds_t1_mni', run_without_submitting=True)

    ds_mni_mask = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'brainmask')),
        name='ds_mni_mask', run_without_submitting=True)

    ds_mni_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'dtissue')),
        name='ds_mni_seg', run_without_submitting=True)

    ds_mni_tpms = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'class-{extra_value}_probtissue')),
        name='ds_mni_tpms', run_without_submitting=True)
    ds_mni_tpms.inputs.extra_values = ['CSF', 'GM', 'WM']

    ds_t1_mni_warp = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt(template, 'warp')),
        name='ds_t1_mni_warp', run_without_submitting=True)

    name_surfs = pe.MapNode(GiftiNameSource(pattern=r'(?P<LR>[lr])h.(?P<surf>.+)_converted.gii',
                                            template='{surf}.{LR}.surf'),
                            iterfield='in_file',
                            name='name_surfs',
                            run_without_submitting=True)

    ds_surfs = pe.MapNode(
        DerivativesDataSink(base_directory=output_dir),
        iterfield=['in_file', 'suffix'], name='ds_surfs', run_without_submitting=True)

    workflow.connect([
        (inputnode, ds_t1_preproc, [('source_file', 'source_file'),
                                    ('t1_preproc', 'in_file')]),
        (inputnode, ds_t1_mask, [('source_file', 'source_file'),
                                 ('t1_mask', 'in_file')]),
        (inputnode, ds_t1_seg, [('source_file', 'source_file'),
                                ('t1_seg', 'in_file')]),
        (inputnode, ds_t1_tpms, [('source_file', 'source_file'),
                                 ('t1_tpms', 'in_file')]),
    ])

    if freesurfer:
        workflow.connect([
            (inputnode, name_surfs, [('surfaces', 'in_file')]),
            (inputnode, ds_surfs, [('source_file', 'source_file'),
                                   ('surfaces', 'in_file')]),
            (name_surfs, ds_surfs, [('out_name', 'suffix')]),
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (inputnode, ds_t1_mni_warp, [('source_file', 'source_file'),
                                         ('t1_2_mni_forward_transform', 'in_file')]),
            (inputnode, ds_t1_mni, [('source_file', 'source_file'),
                                    ('t1_2_mni', 'in_file')]),
            (inputnode, ds_mni_mask, [('source_file', 'source_file'),
                                      ('mni_mask', 'in_file')]),
            (inputnode, ds_mni_seg, [('source_file', 'source_file'),
                                     ('mni_seg', 'in_file')]),
            (inputnode, ds_mni_tpms, [('source_file', 'source_file'),
                                      ('mni_tpms', 'in_file')]),
        ])

    return workflow
