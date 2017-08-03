#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
EPI MRI -processing workflows.

Originally coded by Craig Moodie. Refactored by the CRN Developers.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import os.path as op

import pkg_resources as pkgr

from niworkflows.nipype import logging
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import ants, afni, c3, fsl
from niworkflows.nipype.interfaces import utility as niu
from niworkflows.nipype.interfaces import freesurfer as fs
from niworkflows.interfaces.registration import EstimateReferenceImage
import niworkflows.data as nid

from niworkflows.interfaces import SimpleBeforeAfter
from fmriprep.interfaces import DerivativesDataSink, InvertT1w, ValidateImage

from fmriprep.interfaces.images import GenerateSamplingReference, extract_wm
from fmriprep.interfaces.nilearn import Merge
from fmriprep.interfaces.reports import FunctionalSummary
from fmriprep.workflows import confounds
from niworkflows.nipype.utils.filemanip import split_filename
from fmriprep.workflows.fieldmap.unwarp import init_pepolar_unwarp_wf
from fmriprep.workflows.util import (
    init_enhance_and_skullstrip_epi_wf, init_skullstrip_epi_wf,
    init_bbreg_wf, init_fsl_bbr_wf)

LOGGER = logging.getLogger('workflow')


def init_func_preproc_wf(bold_file, ignore, freesurfer,
                         bold2t1w_dof, reportlets_dir,
                         output_spaces, template, output_dir, omp_nthreads,
                         fmap_bspline, fmap_demean, use_syn, force_syn,
                         use_aroma, ignore_aroma_err,
                         debug, output_grid_ref, layout=None):

    if bold_file == '/completely/made/up/path/sub-01_task-nback_bold.nii.gz':
        bold_file_size_gb = 1
    else:
        bold_file_size_gb = os.path.getsize(bold_file)/(1024*1024*1024)

    LOGGER.info('Creating bold processing workflow for "%s".', bold_file)
    fname = split_filename(bold_file)[1]
    fname_nosub = '_'.join(fname.split("_")[1:])
    name = "func_preproc_" + fname_nosub.replace(
        ".", "_").replace(" ", "").replace("-", "_").replace("_bold", "_wf")

    # For doc building purposes
    if layout is None or bold_file == 'bold_preprocesing':

        LOGGER.info('No valid layout: building empty workflow.')
        metadata = {"RepetitionTime": 2.0,
                    "SliceTiming": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        fmaps = [{
                  'type': 'phasediff',
                  'phasediff': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_phasediff.nii.gz',
                  'magnitude1': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude1.nii.gz',
                  'magnitude2': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude2.nii.gz'
                }]
    else:
        metadata = layout.get_metadata(bold_file)
        # Find fieldmaps. Options: (phase1|phase2|phasediff|epi|fieldmap)
        fmaps = layout.get_fieldmap(bold_file, return_list=True) \
            if 'fieldmaps' not in ignore else []

    # TODO: To be removed (supported fieldmaps):
    if not set([fmap['type'] for fmap in fmaps]).intersection(['phasediff', 'fieldmap', 'epi']):
        fmaps = None

    # Run SyN if forced or in the absence of fieldmap correction
    use_syn = force_syn or (use_syn and not fmaps)

    # Build workflow
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['epi', 't1_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_tpms',
                't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                'subjects_dir', 'subject_id', 'fs_2_t1_transform']),
        name='inputnode')
    inputnode.inputs.epi = bold_file

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['epi_t1', 'epi_mask_t1', 'epi_mni', 'epi_mask_mni', 'confounds', 'surfaces',
                'aroma_noise_ics', 'melodic_mix']),
        name='outputnode')

    summary = pe.Node(FunctionalSummary(output_spaces=output_spaces), name='summary')
    summary.inputs.slice_timing = "SliceTiming" in metadata and 'slicetiming' not in ignore
    summary.inputs.registration = 'bbregister' if freesurfer else 'FLIRT'

    func_reports_wf = init_func_reports_wf(reportlets_dir=reportlets_dir,
                                           freesurfer=freesurfer,
                                           use_aroma=use_aroma,
                                           use_syn=use_syn)

    func_derivatives_wf = init_func_derivatives_wf(output_dir=output_dir,
                                                   output_spaces=output_spaces,
                                                   template=template,
                                                   freesurfer=freesurfer,
                                                   use_aroma=use_aroma)

    workflow.connect([
        (inputnode, func_reports_wf, [('epi', 'inputnode.source_file')]),
        (inputnode, func_derivatives_wf, [('epi', 'inputnode.source_file')]),
        (outputnode, func_derivatives_wf, [('epi_t1', 'inputnode.epi_t1'),
                                           ('epi_mask_t1', 'inputnode.epi_mask_t1'),
                                           ('epi_mni', 'inputnode.epi_mni'),
                                           ('epi_mask_mni', 'inputnode.epi_mask_mni'),
                                           ('confounds', 'inputnode.confounds'),
                                           ('surfaces', 'inputnode.surfaces'),
                                           ('aroma_noise_ics', 'inputnode.aroma_noise_ics'),
                                           ('melodic_mix', 'inputnode.melodic_mix')
                                           ]),
        ])

    validate = pe.Node(ValidateImage(), name='validate')

    # HMC on the EPI
    epi_hmc_wf = init_epi_hmc_wf(name='epi_hmc_wf', metadata=metadata,
                                 bold_file_size_gb=bold_file_size_gb,
                                 ignore=ignore)

    # mean EPI registration to T1w
    epi_reg_wf = init_epi_reg_wf(name='epi_reg_wf',
                                 freesurfer=freesurfer,
                                 bold2t1w_dof=bold2t1w_dof,
                                 bold_file_size_gb=bold_file_size_gb,
                                 output_spaces=output_spaces,
                                 output_dir=output_dir,
                                 use_fieldwarp=(fmaps is not None or use_syn))

    # get confounds
    discover_wf = confounds.init_discover_wf(bold_file_size_gb=bold_file_size_gb,
                                             use_aroma=use_aroma,
                                             ignore_aroma_err=ignore_aroma_err,
                                             metadata=metadata,
                                             name='discover_wf')

    discover_wf.get_node('inputnode').inputs.t1_transform_flags = [False]

    workflow.connect([
        (inputnode, validate, [('epi', 'in_file')]),
        (validate, epi_hmc_wf, [('out_file', 'inputnode.epi')]),
        (inputnode, epi_reg_wf, [('epi', 'inputnode.name_source'),
                                 ('t1_preproc', 'inputnode.t1_preproc'),
                                 ('t1_brain', 'inputnode.t1_brain'),
                                 ('t1_mask', 'inputnode.t1_mask'),
                                 ('t1_seg', 'inputnode.t1_seg'),
                                 # Undefined if --no-freesurfer, but this is safe
                                 ('subjects_dir', 'inputnode.subjects_dir'),
                                 ('subject_id', 'inputnode.subject_id'),
                                 ('fs_2_t1_transform', 'inputnode.fs_2_t1_transform')
                                 ]),
        (inputnode, discover_wf, [('t1_tpms', 'inputnode.t1_tpms')]),
        (epi_hmc_wf, epi_reg_wf, [('outputnode.epi_split', 'inputnode.epi_split'),
                                  ('outputnode.xforms', 'inputnode.hmc_xforms')]),
        (epi_hmc_wf, discover_wf, [('outputnode.movpar_file', 'inputnode.movpar_file')]),
        (epi_reg_wf, discover_wf, [('outputnode.epi_t1', 'inputnode.fmri_file'),
                                   ('outputnode.epi_mask_t1', 'inputnode.epi_mask')]),
        (validate, func_reports_wf, [('out_report', 'inputnode.validation_report')]),
        (epi_reg_wf, func_reports_wf, [
            ('outputnode.out_report', 'inputnode.epi_reg_report'),
            ]),
        (discover_wf, outputnode, [('outputnode.confounds_file', 'confounds'),
                                   ('outputnode.aroma_noise_ics', 'aroma_noise_ics'),
                                   ('outputnode.melodic_mix', 'melodic_mix')]),
        (epi_reg_wf, outputnode, [('outputnode.epi_t1', 'epi_t1'),
                                  ('outputnode.epi_mask_t1', 'epi_mask_t1')]),
        (discover_wf, func_reports_wf, [
            ('outputnode.acompcor_report', 'inputnode.acompcor_report'),
            ('outputnode.tcompcor_report', 'inputnode.tcompcor_report'),
            ('outputnode.ica_aroma_report', 'inputnode.ica_aroma_report')]),
        (discover_wf, summary, [('outputnode.confounds_list', 'confounds')]),
        (summary, func_reports_wf, [('out_report', 'inputnode.summary_report')]),
        ])

    # Cases:
    # fmaps | use_syn | force_syn  |  ACTION
    # ----------------------------------------------
    #   T   |    *    |     T      | Fieldmaps + SyN
    #   T   |    *    |     F      | Fieldmaps
    #   F   |    *    |     T      | SyN
    #   F   |    T    |     F      | SyN
    #   F   |    F    |     F      | HMC only

    # Predefine to pacify the lintian checks about
    # "could be used before defined" - logic was tested to be sound
    nonlinear_sdc_wf = sdc_unwarp_wf = None

    if fmaps:
        # In case there are multiple fieldmaps prefer EPI
        fmaps.sort(key=lambda fmap: {'epi': 0, 'fieldmap': 1, 'phasediff': 2}[fmap['type']])
        fmap = fmaps[0]

        LOGGER.info('Fieldmap estimation: type "%s" found', fmap['type'])
        summary.inputs.distortion_correction = fmap['type']

        if fmap['type'] == 'epi':
            epi_fmaps = [fmap_['epi'] for fmap_ in fmaps if fmap_['type'] == 'epi']
            sdc_unwarp_wf = init_pepolar_unwarp_wf(fmaps=epi_fmaps,
                                                   layout=layout,
                                                   bold_file=bold_file,
                                                   omp_nthreads=omp_nthreads,
                                                   name='pepolar_unwarp_wf')
        else:
            # Import specific workflows here, so we don't brake everything with one
            # unused workflow.
            from fmriprep.workflows.fieldmap import init_fmap_estimator_wf, init_sdc_unwarp_wf
            fmap_estimator_wf = init_fmap_estimator_wf(fmap_bids=fmap,
                                                       reportlets_dir=reportlets_dir,
                                                       omp_nthreads=omp_nthreads,
                                                       fmap_bspline=fmap_bspline)
            sdc_unwarp_wf = init_sdc_unwarp_wf(reportlets_dir=reportlets_dir,
                                               omp_nthreads=omp_nthreads,
                                               fmap_bspline=fmap_bspline,
                                               fmap_demean=fmap_demean,
                                               debug=debug,
                                               name='sdc_unwarp_wf')
            workflow.connect([
                (fmap_estimator_wf, sdc_unwarp_wf, [
                    ('outputnode.fmap', 'inputnode.fmap'),
                    ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
                    ('outputnode.fmap_mask', 'inputnode.fmap_mask')]),
            ])

        # Connections and workflows common for all types of fieldmaps
        workflow.connect([
            (inputnode, sdc_unwarp_wf, [('epi', 'inputnode.name_source')]),
            (epi_hmc_wf, sdc_unwarp_wf, [
                ('outputnode.ref_image', 'inputnode.in_reference'),
                ('outputnode.ref_image_brain', 'inputnode.in_reference_brain'),
                ('outputnode.epi_mask', 'inputnode.in_mask')]),
            (sdc_unwarp_wf, epi_reg_wf, [
                ('outputnode.out_warp', 'inputnode.fieldwarp'),
                ('outputnode.out_reference_brain', 'inputnode.ref_epi_brain'),
                ('outputnode.out_mask', 'inputnode.ref_epi_mask')]),
            (sdc_unwarp_wf, func_reports_wf, [
                ('outputnode.out_mask_report', 'inputnode.epi_mask_report')])
        ])

        # Report on EPI correction
        fmap_unwarp_report_wf = init_fmap_unwarp_report_wf(reportlets_dir=reportlets_dir,
                                                           name='fmap_unwarp_report_wf')
        workflow.connect([(inputnode, fmap_unwarp_report_wf, [('t1_seg', 'inputnode.in_seg'),
                                                              ('epi', 'inputnode.name_source')]),
                          (epi_hmc_wf, fmap_unwarp_report_wf, [
                              ('outputnode.ref_image', 'inputnode.in_pre')]),
                          (sdc_unwarp_wf, fmap_unwarp_report_wf, [
                              ('outputnode.out_reference', 'inputnode.in_post')]),
                          (epi_reg_wf, fmap_unwarp_report_wf, [
                              ('outputnode.itk_t1_to_epi', 'inputnode.in_xfm')]),
                          ])
    elif not use_syn:
        LOGGER.warn('No fieldmaps found or they were ignored, building base workflow '
                    'for dataset %s.', bold_file)
        summary.inputs.distortion_correction = 'None'
        workflow.connect([
            (epi_hmc_wf, func_reports_wf, [
                ('outputnode.epi_mask_report', 'inputnode.epi_mask_report')]),
            (epi_hmc_wf, epi_reg_wf, [('outputnode.ref_image_brain', 'inputnode.ref_epi_brain'),
                                      ('outputnode.epi_mask', 'inputnode.ref_epi_mask')]),
        ])

    if use_syn:
        nonlinear_sdc_wf = init_nonlinear_sdc_wf(
            bold_file=bold_file, layout=layout, freesurfer=freesurfer, bold2t1w_dof=bold2t1w_dof,
            template=template, omp_nthreads=omp_nthreads)

        workflow.connect([
            (inputnode, nonlinear_sdc_wf, [
                 ('t1_brain', 'inputnode.t1_brain'),
                 ('t1_seg', 'inputnode.t1_seg'),
                 ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                 ('subjects_dir', 'inputnode.subjects_dir'),
                 ('subject_id', 'inputnode.subject_id')]),
            (epi_hmc_wf, nonlinear_sdc_wf, [('outputnode.ref_image_brain', 'inputnode.epi_ref')]),
            (nonlinear_sdc_wf, func_reports_wf, [
                ('outputnode.out_warp_report', 'inputnode.syn_sdc_report')]),
            ])

        # XXX Eliminate branch when forcing isn't an option
        if not fmaps:
            LOGGER.warn('No fieldmaps found or they were ignored. Using EXPERIMENTAL '
                        'nonlinear susceptibility correction for dataset %s.', bold_file)
            summary.inputs.distortion_correction = 'SyN'
            workflow.connect([
                (nonlinear_sdc_wf, func_reports_wf, [
                    ('outputnode.out_mask_report', 'inputnode.epi_mask_report')]),
                (nonlinear_sdc_wf, epi_reg_wf, [
                    ('outputnode.out_warp', 'inputnode.fieldwarp'),
                    ('outputnode.out_reference_brain', 'inputnode.ref_epi_brain'),
                    ('outputnode.out_mask', 'inputnode.ref_epi_mask')]),
            ])

    if 'template' in output_spaces:
        # Apply transforms in 1 shot
        epi_mni_trans_wf = init_epi_mni_trans_wf(output_dir=output_dir,
                                                 template=template,
                                                 bold_file_size_gb=bold_file_size_gb,
                                                 output_grid_ref=output_grid_ref,
                                                 name='epi_mni_trans_wf')
        workflow.connect([
            (inputnode, epi_mni_trans_wf, [
                ('epi', 'inputnode.name_source'),
                ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform')]),
            (epi_hmc_wf, epi_mni_trans_wf, [
                ('outputnode.epi_split', 'inputnode.epi_split'),
                ('outputnode.xforms', 'inputnode.hmc_xforms')]),
            (epi_reg_wf, epi_mni_trans_wf, [
                ('outputnode.itk_epi_to_t1', 'inputnode.itk_epi_to_t1')]),
            (epi_mni_trans_wf, outputnode, [('outputnode.epi_mni', 'epi_mni'),
                                            ('outputnode.epi_mask_mni', 'epi_mask_mni')]),
            (epi_mni_trans_wf, discover_wf, [('outputnode.epi_mask_mni', 'inputnode.epi_mask_mni'),
                                             ('outputnode.epi_mni', 'inputnode.epi_mni')])
        ])

        if fmaps:
            workflow.connect([
                (sdc_unwarp_wf, epi_mni_trans_wf, [
                    ('outputnode.out_warp', 'inputnode.fieldwarp'),
                    ('outputnode.out_mask', 'inputnode.epi_mask')]),
            ])
        elif use_syn:
            workflow.connect([
                (nonlinear_sdc_wf, epi_mni_trans_wf, [
                    ('outputnode.out_warp', 'inputnode.fieldwarp'),
                    ('outputnode.out_mask', 'inputnode.epi_mask')]),
            ])
        else:
            workflow.connect([
                (epi_hmc_wf, epi_mni_trans_wf, [
                    ('outputnode.epi_mask', 'inputnode.epi_mask')]),
            ])

    if freesurfer and any(space.startswith('fs') for space in output_spaces):
        LOGGER.info('Creating FreeSurfer processing flow.')
        epi_surf_wf = init_epi_surf_wf(output_spaces=output_spaces,
                                       name='epi_surf_wf')
        workflow.connect([
            (inputnode, epi_surf_wf, [('subjects_dir', 'inputnode.subjects_dir'),
                                      ('subject_id', 'inputnode.subject_id')]),
            (epi_reg_wf, epi_surf_wf, [('outputnode.epi_t1', 'inputnode.source_file')]),
            (epi_surf_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
        ])

    return workflow


# pylint: disable=R0914
def init_epi_hmc_wf(metadata, bold_file_size_gb, ignore,
                    name='epi_hmc_wf'):
    """
    Performs :abbr:`HMC (head motion correction)` over the input
    :abbr:`EPI (echo-planar imaging)` image.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['epi']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['xforms', 'epi_hmc', 'epi_split', 'epi_mask', 'ref_image',
                'ref_image_brain', 'movpar_file', 'n_volumes_to_discard',
                'epi_mask_report']), name='outputnode')

    def normalize_motion_func(in_file, format):
        import os
        import numpy as np
        from niworkflows.nipype.utils.misc import normalize_mc_params
        mpars = np.loadtxt(in_file)  # mpars is N_t x 6
        mpars = np.apply_along_axis(func1d=normalize_mc_params,
                                    axis=1, arr=mpars,
                                    source=format)
        np.savetxt("motion_params.txt", mpars)
        return os.path.abspath("motion_params.txt")

    normalize_motion = pe.Node(niu.Function(function=normalize_motion_func),
                               name="normalize_motion")
    normalize_motion.inputs.format = "FSL"

    # Head motion correction (hmc)
    hmc = pe.Node(fsl.MCFLIRT(
        save_mats=True, save_plots=True), name='EPI_hmc')
    hmc.interface.estimated_memory_gb = bold_file_size_gb * 3

    hcm2itk = pe.MapNode(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                         iterfield=['transform_file'], name='hcm2itk')

    enhance_and_skullstrip_epi_wf = init_enhance_and_skullstrip_epi_wf()

    gen_ref = pe.Node(EstimateReferenceImage(), name="gen_ref")

    workflow.connect([
        (inputnode, gen_ref, [('epi', 'in_file')]),
        (gen_ref, enhance_and_skullstrip_epi_wf, [('ref_image', 'inputnode.in_file')]),
        (gen_ref, hmc, [('ref_image', 'ref_file')]),
        (enhance_and_skullstrip_epi_wf, outputnode, [
            ('outputnode.bias_corrected_file', 'ref_image'),
            ('outputnode.mask_file', 'epi_mask'),
            ('outputnode.out_report', 'epi_mask_report'),
            ('outputnode.skull_stripped_file', 'ref_image_brain')]),
        ])

    split = pe.Node(fsl.Split(dimension='t'), name='split')
    split.interface.estimated_memory_gb = bold_file_size_gb * 3

    if "SliceTiming" in metadata and 'slicetiming' not in ignore:
        LOGGER.info('Slice-timing correction will be included.')

        def create_custom_slice_timing_file_func(metadata):
            import os
            slice_timings = metadata["SliceTiming"]
            slice_timings_ms = [str(t) for t in slice_timings]
            out_file = "timings.1D"
            with open("timings.1D", "w") as fp:
                fp.write("\t".join(slice_timings_ms))

            return os.path.abspath(out_file)

        create_custom_slice_timing_file = pe.Node(
            niu.Function(function=create_custom_slice_timing_file_func),
            name="create_custom_slice_timing_file")
        create_custom_slice_timing_file.inputs.metadata = metadata

        slice_timing_correction = pe.Node(interface=afni.TShift(),
                                          name='slice_timing_correction')
        slice_timing_correction.inputs.outputtype = 'NIFTI_GZ'
        slice_timing_correction.inputs.tr = str(metadata["RepetitionTime"]) + "s"

        def prefix_at(x):
            return "@" + x

        workflow.connect([
            (inputnode, slice_timing_correction, [('epi', 'in_file')]),
            (gen_ref, slice_timing_correction, [('n_volumes_to_discard', 'ignore')]),
            (create_custom_slice_timing_file, slice_timing_correction, [
                (('out', prefix_at), 'tpattern')]),
            (slice_timing_correction, hmc, [('out_file', 'in_file')])
        ])

    else:
        workflow.connect([
            (inputnode, hmc, [('epi', 'in_file')])
        ])

    workflow.connect([
        (hmc, hcm2itk, [('mat_file', 'transform_file')]),
        (gen_ref, hcm2itk, [('ref_image', 'source_file'),
                            ('ref_image', 'reference_file')]),
        (hcm2itk, outputnode, [('itk_transform', 'xforms')]),
        (hmc, normalize_motion, [('par_file', 'in_file')]),
        (normalize_motion, outputnode, [('out', 'movpar_file')]),
        (inputnode, split, [('epi', 'in_file')]),
        (split, outputnode, [('out_files', 'epi_split')]),
    ])

    return workflow


def init_epi_reg_wf(freesurfer, bold2t1w_dof,
                    bold_file_size_gb, output_spaces, output_dir,
                    name='epi_reg_wf', use_fieldwarp=False):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the T1-space
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['name_source', 'ref_epi_brain', 'ref_epi_mask',
                                      't1_preproc', 't1_brain', 't1_mask',
                                      't1_seg', 'epi_split', 'hmc_xforms',
                                      'subjects_dir', 'subject_id', 'fs_2_t1_transform',
                                      'fieldwarp']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['mat_epi_to_t1', 'mat_t1_to_epi',
                                      'itk_epi_to_t1', 'itk_t1_to_epi',
                                      'epi_t1', 'epi_mask_t1', 'fs_reg_file',
                                      'out_report']),
        name='outputnode'
    )

    if freesurfer:
        bbr_wf = init_bbreg_wf(bold2t1w_dof, report=True)
    else:
        bbr_wf = init_fsl_bbr_wf(bold2t1w_dof, report=True)

    # make equivalent warp fields
    invt_bbr = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='invt_bbr')

    #  EPI to T1 transform matrix is from fsl, using c3 tools to convert to
    #  something ANTs will like.
    fsl2itk_fwd = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_fwd')
    fsl2itk_inv = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_inv')

    workflow.connect([
        (inputnode, bbr_wf, [('ref_epi_brain', 'inputnode.in_file'),
                             ('fs_2_t1_transform', 'inputnode.fs_2_t1_transform'),
                             ('subjects_dir', 'inputnode.subjects_dir'),
                             ('subject_id', 'inputnode.subject_id'),
                             ('t1_seg', 'inputnode.t1_seg'),
                             ('t1_brain', 'inputnode.t1_brain')]),
        (inputnode, fsl2itk_fwd, [('t1_preproc', 'reference_file'),
                                  ('ref_epi_brain', 'source_file')]),
        (inputnode, fsl2itk_inv, [('ref_epi_brain', 'reference_file'),
                                  ('t1_preproc', 'source_file')]),
        (bbr_wf, invt_bbr, [('outputnode.out_matrix_file', 'in_file')]),
        (bbr_wf, fsl2itk_fwd, [('outputnode.out_matrix_file', 'transform_file')]),
        (invt_bbr, fsl2itk_inv, [('out_file', 'transform_file')]),
        (bbr_wf, outputnode, [('outputnode.out_matrix_file', 'mat_epi_to_t1'),
                              ('outputnode.out_reg_file', 'fs_reg_file'),
                              ('outputnode.out_report', 'out_report')]),
        (invt_bbr, outputnode, [('out_file', 'mat_t1_to_epi')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_epi_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_epi')]),
    ])

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref')

    mask_t1w_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='NearestNeighbor',
                             float=True),
        name='mask_t1w_tfm'
    )

    workflow.connect([
        (inputnode, gen_ref, [('ref_epi_brain', 'moving_image'),
                              ('t1_brain', 'fixed_image')]),
        (gen_ref, mask_t1w_tfm, [('out_file', 'reference_image')]),
        (fsl2itk_fwd, mask_t1w_tfm, [('itk_transform', 'transforms')]),
        (inputnode, mask_t1w_tfm, [('ref_epi_mask', 'input_image')]),
        (mask_t1w_tfm, outputnode, [('output_image', 'epi_mask_t1')])
    ])

    if use_fieldwarp:
        merge_transforms = pe.MapNode(niu.Merge(3), iterfield=['in3'],
                                      name='merge_transforms', run_without_submitting=True)
        workflow.connect([
            (inputnode, merge_transforms, [('fieldwarp', 'in2'),
                                           ('hmc_xforms', 'in3')])
            ])
    else:
        merge_transforms = pe.MapNode(niu.Merge(2), iterfield=['in2'],
                                      name='merge_transforms', run_without_submitting=True)
        workflow.connect([
            (inputnode, merge_transforms, [('hmc_xforms', 'in2')])
        ])

    merge = pe.Node(Merge(), name='merge')
    merge.interface.estimated_memory_gb = bold_file_size_gb * 3

    epi_to_t1w_transform = pe.MapNode(
        ants.ApplyTransforms(interpolation="LanczosWindowedSinc",
                             float=True),
        iterfield=['input_image', 'transforms'],
        name='epi_to_t1w_transform')
    epi_to_t1w_transform.terminal_output = 'file'

    workflow.connect([
        (fsl2itk_fwd, merge_transforms, [('itk_transform', 'in1')]),
        (merge_transforms, epi_to_t1w_transform, [('out', 'transforms')]),
        (epi_to_t1w_transform, merge, [('output_image', 'in_files')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (merge, outputnode, [('out_file', 'epi_t1')]),
        (inputnode, epi_to_t1w_transform, [('epi_split', 'input_image')]),
        (gen_ref, epi_to_t1w_transform, [('out_file', 'reference_image')]),
    ])

    return workflow


def init_epi_surf_wf(output_spaces, name='epi_surf_wf'):
    """ Sample functional images to FreeSurfer surfaces

    For each vertex, the cortical ribbon is sampled at six points (spaced 20% of thickness apart)
    and averaged.

    Outputs are in GIFTI format.

    output_spaces : set of structural spaces to sample functional series to
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 'subject_id', 'subjects_dir']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['surfaces']), name='outputnode')

    spaces = [space for space in output_spaces if space.startswith('fs')]

    def select_target(subject_id, space):
        """ Given a source subject ID and a target space, get the target subject ID """
        return subject_id if space == 'fsnative' else space

    targets = pe.MapNode(niu.Function(function=select_target),
                         iterfield=['space'], name='targets')
    targets.inputs.space = spaces

    # Rename the source file to the output space to simplify naming later
    rename_src = pe.MapNode(niu.Rename(format_string='%(subject)s', keep_ext=True),
                            iterfield='subject', name='rename_src', run_without_submitting=True)
    rename_src.inputs.subject = spaces

    sampler = pe.MapNode(
        fs.SampleToSurface(sampling_method='average', sampling_range=(0, 1, 0.2),
                           sampling_units='frac', reg_header=True,
                           interp_method='trilinear', cortex_mask=True,
                           out_type='gii'),
        iterfield=['source_file', 'target_subject'],
        iterables=('hemi', ['lh', 'rh']),
        name='sampler')

    merger = pe.JoinNode(niu.Merge(1, ravel_inputs=True), name='merger',
                         joinsource='sampler', joinfield=['in1'], run_without_submitting=True)

    def update_gifti_metadata(in_file):
        import os
        import nibabel as nib
        img = nib.load(in_file)
        fname = os.path.basename(in_file)
        if fname[:3] in ('lh.', 'rh.'):
            asp = 'CortexLeft' if fname[0] == 'l' else 'CortexRight'
        else:
            raise ValueError(
                "AnatomicalStructurePrimary cannot be derived from filename")
        primary = nib.gifti.GiftiNVPairs('AnatomicalStructurePrimary', asp)
        if not any(nvpair.name == primary.name for nvpair in img.meta.data):
            img.meta.data.insert(0, primary)
        img.to_filename(fname)
        return os.path.abspath(fname)

    update_metadata = pe.MapNode(niu.Function(function=update_gifti_metadata),
                                 iterfield='in_file', name='update_metadata')

    workflow.connect([
        (inputnode, targets, [('subject_id', 'subject_id')]),
        (inputnode, rename_src, [('source_file', 'in_file')]),
        (inputnode, sampler, [('subjects_dir', 'subjects_dir'),
                              ('subject_id', 'subject_id')]),
        (targets, sampler, [('out', 'target_subject')]),
        (rename_src, sampler, [('out_file', 'source_file')]),
        (sampler, merger, [('out_file', 'in1')]),
        (merger, update_metadata, [('out', 'in_file')]),
        (update_metadata, outputnode, [('out', 'surfaces')]),
        ])

    return workflow


def init_epi_mni_trans_wf(output_dir, template, bold_file_size_gb,
                          name='epi_mni_trans_wf',
                          output_grid_ref=None,
                          use_fieldwarp=False):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_epi_to_t1',
            't1_2_mni_forward_transform',
            'name_source',
            'epi_split',
            'epi_mask',
            'hmc_xforms',
            'fieldwarp'
        ]),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['epi_mni', 'epi_mask_mni']),
        name='outputnode')

    def _aslist(in_value):
        if isinstance(in_value, list):
            return in_value
        return [in_value]

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref')
    template_str = nid.TEMPLATE_MAP[template]
    gen_ref.inputs.fixed_image = op.join(nid.get_dataset(template_str), '1mm_T1.nii.gz')

    mask_mni_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='NearestNeighbor',
                             float=True),
        name='mask_mni_tfm'
    )

    # Write corrected file in the designated output dir
    mask_merge_tfms = pe.Node(niu.Merge(2), name='mask_merge_tfms', run_without_submitting=True)

    if use_fieldwarp:
        merge_transforms = pe.MapNode(niu.Merge(4), iterfield=['in4'],
                                      name='merge_transforms', run_without_submitting=True)
        workflow.connect([
            (inputnode, merge_transforms, [('fieldwarp', 'in3'),
                                           ('hmc_xforms', 'in4')])])

    else:
        merge_transforms = pe.MapNode(niu.Merge(3), iterfield=['in3'],
                                      name='merge_transforms', run_without_submitting=True)
        workflow.connect([
            (inputnode, merge_transforms, [('hmc_xforms', 'in3')])])

    workflow.connect([
        (inputnode, gen_ref, [('epi_mask', 'moving_image')]),
        (inputnode, mask_merge_tfms, [('t1_2_mni_forward_transform', 'in1'),
                                      (('itk_epi_to_t1', _aslist), 'in2')]),
        (mask_merge_tfms, mask_mni_tfm, [('out', 'transforms')]),
        (mask_mni_tfm, outputnode, [('output_image', 'epi_mask_mni')]),
        (inputnode, mask_mni_tfm, [('epi_mask', 'input_image')])
    ])

    merge = pe.Node(Merge(), name='merge')
    merge.interface.estimated_memory_gb = bold_file_size_gb * 3
    epi_to_mni_transform = pe.MapNode(
        ants.ApplyTransforms(interpolation="LanczosWindowedSinc",
                             float=True),
        iterfield=['input_image', 'transforms'],
        name='epi_to_mni_transform')
    epi_to_mni_transform.terminal_output = 'file'

    workflow.connect([
        (inputnode, merge_transforms, [('t1_2_mni_forward_transform', 'in1'),
                                       (('itk_epi_to_t1', _aslist), 'in2')]),
        (merge_transforms, epi_to_mni_transform, [('out', 'transforms')]),
        (epi_to_mni_transform, merge, [('output_image', 'in_files')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode, epi_to_mni_transform, [('epi_split', 'input_image')]),
        (merge, outputnode, [('out_file', 'epi_mni')]),
    ])

    if output_grid_ref is None:
        workflow.connect([
            (gen_ref, mask_mni_tfm, [('out_file', 'reference_image')]),
            (gen_ref, epi_to_mni_transform, [('out_file', 'reference_image')]),
        ])
    else:
        mask_mni_tfm.inputs.reference_image = output_grid_ref
        epi_to_mni_transform.inputs.reference_image = output_grid_ref
    return workflow


def init_nonlinear_sdc_wf(bold_file, layout, freesurfer, bold2t1w_dof,
                          template, omp_nthreads,
                          atlas_threshold=3, name='nonlinear_sdc_wf'):
    """
    This workflow takes a skull-stripped T1w image and reference EPI image and
    estimates a susceptibility distortion correction warp, using ANTs symmetric
    normalization (SyN) and the average fieldmap atlas described in
    [Treiber2016]_.

    If the phase-encoding (PE) direction is known, the SyN deformation is
    restricted to that direction; otherwise, deformation fields are calculated
    for both the right-left and anterior-posterior directions, and selected
    based on the unwarped file that can be aligned to the T1w image with the
    lowest boundary-based registration (BBR) cost.

    SyN deformation is also restricted to regions that are expected to have a
    >3mm (approximately 1 voxel) warp, based on the fieldmap atlas.

    This technique is a variation on those developed in [Huntenburg2014]_ and
    [Wang2017]_.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.epi import init_nonlinear_sdc_wf
        wf = init_nonlinear_sdc_wf(
            bold_file='/dataset/sub-01/func/sub-01_task-rest_bold.nii.gz',
            layout=None,
            freesurfer=True,
            bold2t1w_dof=9,
            template='MNI152NLin2009cAsym',
            omp_nthreads=8)

    Inputs

        t1_brain
            skull-stripped, bias-corrected structural image
        epi_ref
            skull-stripped reference image
        t1_seg
            FAST segmentation white and gray matter, in native T1w space
        t1_2_mni_reverse_transform
            inverse registration transform of T1w image to MNI template
        subjects_dir
            FreeSurfer subjects directory (if applicable)
        subject_id
            FreeSurfer subject_id (if applicable)

    Outputs

        out_reference_brain
            the ``epi_ref`` image after unwarping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_mask
            mask of the unwarped input file
        out_mask_report
            reportlet for the skullstripping

    .. [Huntenburg2014] Huntenburg, J. M. (2014) Evaluating Nonlinear
                        Coregistration of BOLD EPI and T1w Images. Berlin: Master
                        Thesis, Freie Universität. `PDF
                        <http://pubman.mpdl.mpg.de/pubman/item/escidoc:2327525:5/component/escidoc:2327523/master_thesis_huntenburg_4686947.pdf>`_.
    .. [Treiber2016] Treiber, J. M. et al. (2016) Characterization and Correction
                     of Geometric Distortions in 814 Diffusion Weighted Images,
                     PLoS ONE 11(3): e0152472. doi:`10.1371/journal.pone.0152472
                     <https://doi.org/10.1371/journal.pone.0152472>`_.
    .. [Wang2017] Wang S, et al. (2017) Evaluation of Field Map and Nonlinear
                  Registration Methods for Correction of Susceptibility Artifacts
                  in Diffusion MRI. Front. Neuroinform. 11:17.
                  doi:`10.3389/fninf.2017.00017
                  <https://doi.org/10.3389/fninf.2017.00017>`_.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(['t1_brain', 'epi_ref', 't1_2_mni_reverse_transform',
                               'subjects_dir', 'subject_id', 't1_seg']),  # BBR requirements
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['out_reference_brain', 'out_mask', 'out_warp',
                               'out_warp_report', 'out_mask_report']),
        name='outputnode')

    # Collect predefined data
    # Atlas image and registration affine
    atlas_img = pkgr.resource_filename('fmriprep', 'data/fmap_atlas.nii.gz')
    atlas_2_template_affine = pkgr.resource_filename(
        'fmriprep', 'data/fmap_atlas_2_{}_affine.mat'.format(template))
    # Registration specifications
    affine_transform = pkgr.resource_filename('fmriprep', 'data/affine.json')
    syn_transform = pkgr.resource_filename('fmriprep', 'data/susceptibility_syn.json')

    invert_t1w = pe.Node(InvertT1w(), name='invert_t1w')

    ref_2_t1 = pe.Node(ants.Registration(from_file=affine_transform, num_threads=omp_nthreads),
                       name='ref_2_t1', n_procs=omp_nthreads)
    t1_2_ref = pe.Node(ants.ApplyTransforms(invert_transform_flags=[True],
                                            num_threads=omp_nthreads),
                       name='t1_2_ref', n_procs=omp_nthreads)

    # 1) EPI -> T1; 2) MNI -> T1; 3) ATLAS -> MNI
    transform_list = pe.Node(niu.Merge(3), name='transform_list')
    transform_list.inputs.in3 = atlas_2_template_affine

    # Inverting (1), then applying in reverse order:
    #
    # ATLAS -> MNI -> T1 -> EPI
    atlas_2_ref = pe.Node(
        ants.ApplyTransforms(invert_transform_flags=[True, False, False],
                             num_threads=omp_nthreads),
        name='atlas_2_ref', n_procs=omp_nthreads)
    atlas_2_ref.inputs.input_image = atlas_img

    threshold_atlas = pe.Node(
        fsl.maths.MathsCommand(args='-thr {:.8g} -bin'.format(atlas_threshold),
                               output_datatype='char'),
        name='threshold_atlas')

    fixed_image_masks = pe.Node(niu.Merge(2), name='fixed_image_masks')
    fixed_image_masks.inputs.in1 = 'NULL'

    if layout is None:
        bold_pe = None
    else:
        bold_pe = layout.get_metadata(bold_file).get("PhaseEncodingDirection")

    restrict_i = [[1, 0, 0], [1, 0, 0]]
    restrict_j = [[0, 1, 0], [0, 1, 0]]

    syn_i = pe.Node(
        ants.Registration(from_file=syn_transform, num_threads=omp_nthreads,
                          restrict_deformation=restrict_i),
        name='syn_i', n_procs=omp_nthreads)
    syn_j = pe.Node(
        ants.Registration(from_file=syn_transform, num_threads=omp_nthreads,
                          restrict_deformation=restrict_j),
        name='syn_j', n_procs=omp_nthreads)

    seg_2_ref = pe.Node(
        ants.ApplyTransforms(interpolation='NearestNeighbor', float=True,
                             invert_transform_flags=[True], num_threads=omp_nthreads),
        name='seg_2_ref', n_procs=omp_nthreads)
    sel_wm = pe.Node(niu.Function(function=extract_wm), name='sel_wm')
    syn_rpt = pe.Node(SimpleBeforeAfter(), name='syn_rpt')

    skullstrip_epi_wf = init_skullstrip_epi_wf()

    workflow.connect([
        (inputnode, invert_t1w, [('t1_brain', 'in_file'),
                                 ('epi_ref', 'epi_ref')]),
        (inputnode, ref_2_t1, [('epi_ref', 'moving_image')]),
        (invert_t1w, ref_2_t1, [('out_file', 'fixed_image')]),
        (inputnode, t1_2_ref, [('epi_ref', 'reference_image')]),
        (invert_t1w, t1_2_ref, [('out_file', 'input_image')]),
        (ref_2_t1, t1_2_ref, [('forward_transforms', 'transforms')]),
        (ref_2_t1, transform_list, [('forward_transforms', 'in1')]),
        (inputnode, transform_list, [('t1_2_mni_reverse_transform', 'in2')]),
        (inputnode, atlas_2_ref, [('epi_ref', 'reference_image')]),
        (transform_list, atlas_2_ref, [('out', 'transforms')]),
        (atlas_2_ref, threshold_atlas, [('output_image', 'in_file')]),
        (threshold_atlas, fixed_image_masks, [('out_file', 'in2')]),
        ])

    if bold_pe is None:
        if freesurfer:
            bbr_i_wf = init_bbreg_wf(bold2t1w_dof, report=False, reregister=False, name='bbr_i_wf')
            bbr_j_wf = init_bbreg_wf(bold2t1w_dof, report=False, reregister=False, name='bbr_j_wf')
        else:
            bbr_i_wf = init_fsl_bbr_wf(bold2t1w_dof, report=False, name='bbr_i_wf')
            bbr_j_wf = init_fsl_bbr_wf(bold2t1w_dof, report=False, name='bbr_j_wf')

        def select_outputs(cost_i, warped_image_i, forward_transforms_i,
                           cost_j, warped_image_j, forward_transforms_j):
            if cost_i < cost_j:
                return warped_image_i, forward_transforms_i
            else:
                return warped_image_j, forward_transforms_j

        pe_chooser = pe.Node(
            niu.Function(function=select_outputs,
                         output_names=['warped_image', 'forward_transforms']),
            name='pe_chooser')

        workflow.connect([(inputnode, syn_i, [('epi_ref', 'moving_image')]),
                          (t1_2_ref, syn_i, [('output_image', 'fixed_image')]),
                          (fixed_image_masks, syn_i, [('out', 'fixed_image_masks')]),
                          (inputnode, syn_j, [('epi_ref', 'moving_image')]),
                          (t1_2_ref, syn_j, [('output_image', 'fixed_image')]),
                          (fixed_image_masks, syn_j, [('out', 'fixed_image_masks')]),
                          (inputnode, bbr_i_wf, [('subjects_dir', 'inputnode.subjects_dir'),
                                                 ('subject_id', 'inputnode.subject_id'),
                                                 ('t1_seg', 'inputnode.t1_seg'),
                                                 ('t1_brain', 'inputnode.t1_brain')]),
                          (inputnode, bbr_j_wf, [('subjects_dir', 'inputnode.subjects_dir'),
                                                 ('subject_id', 'inputnode.subject_id'),
                                                 ('t1_seg', 'inputnode.t1_seg'),
                                                 ('t1_brain', 'inputnode.t1_brain')]),
                          (syn_i, bbr_i_wf, [('warped_image', 'inputnode.in_file')]),
                          (syn_j, bbr_j_wf, [('warped_image', 'inputnode.in_file')]),
                          (bbr_i_wf, pe_chooser, [('outputnode.final_cost', 'cost_i')]),
                          (bbr_j_wf, pe_chooser, [('outputnode.final_cost', 'cost_j')]),
                          (syn_i, pe_chooser, [('warped_image', 'warped_image_i'),
                                               ('forward_transforms', 'forward_transforms_i')]),
                          (syn_j, pe_chooser, [('warped_image', 'warped_image_j'),
                                               ('forward_transforms', 'forward_transforms_j')]),
                          ])
        syn_out = pe_chooser
    elif bold_pe[0] == 'i':
        workflow.connect([(inputnode, syn_i, [('epi_ref', 'moving_image')]),
                          (t1_2_ref, syn_i, [('output_image', 'fixed_image')]),
                          (fixed_image_masks, syn_i, [('out', 'fixed_image_masks')]),
                          ])
        syn_out = syn_i
    elif bold_pe[0] == 'j':
        workflow.connect([(inputnode, syn_j, [('epi_ref', 'moving_image')]),
                          (t1_2_ref, syn_j, [('output_image', 'fixed_image')]),
                          (fixed_image_masks, syn_j, [('out', 'fixed_image_masks')]),
                          ])
        syn_out = syn_j

    workflow.connect([(inputnode, seg_2_ref, [('t1_seg', 'input_image')]),
                      (ref_2_t1, seg_2_ref, [('forward_transforms', 'transforms')]),
                      (syn_out, seg_2_ref, [('warped_image', 'reference_image')]),
                      (seg_2_ref, sel_wm, [('output_image', 'in_seg')]),
                      (inputnode, syn_rpt, [('epi_ref', 'before')]),
                      (syn_out, syn_rpt, [('warped_image', 'after')]),
                      (sel_wm, syn_rpt, [('out', 'wm_seg')]),
                      (syn_out, skullstrip_epi_wf, [('warped_image', 'inputnode.in_file')]),
                      (syn_out, outputnode, [('forward_transforms', 'out_warp')]),
                      (skullstrip_epi_wf, outputnode, [
                          ('outputnode.skull_stripped_file', 'out_reference_brain'),
                          ('outputnode.mask_file', 'out_mask'),
                          ('outputnode.out_report', 'out_mask_report')]),
                      (syn_rpt, outputnode, [('out_report', 'out_warp_report')])])

    return workflow


def init_fmap_unwarp_report_wf(reportlets_dir, name='fmap_unwarp_report_wf'):
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_pre', 'in_post', 'in_seg', 'in_xfm',
                'name_source']), name='inputnode')

    map_seg = pe.Node(ants.ApplyTransforms(
        dimension=3, float=True, interpolation='NearestNeighbor'),
        name='map_seg')

    sel_wm = pe.Node(niu.Function(function=extract_wm), name='sel_wm')

    epi_rpt = pe.Node(SimpleBeforeAfter(), name='epi_rpt')
    epi_rpt_ds = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='variant-hmcsdc_preproc'), name='epi_rpt_ds'
    )
    workflow.connect([
        (inputnode, epi_rpt, [('in_post', 'after'),
                              ('in_pre', 'before')]),
        (inputnode, epi_rpt_ds, [('name_source', 'source_file')]),
        (epi_rpt, epi_rpt_ds, [('out_report', 'in_file')]),
        (inputnode, map_seg, [('in_post', 'reference_image'),
                              ('in_seg', 'input_image'),
                              ('in_xfm', 'transforms')]),
        (map_seg, sel_wm, [('output_image', 'in_seg')]),
        (sel_wm, epi_rpt, [('out', 'wm_seg')]),
    ])

    return workflow


def init_func_reports_wf(reportlets_dir, freesurfer, use_aroma, use_syn, name='func_reports_wf'):
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 'summary_report', 'validation_report', 'epi_mask_report',
                    'epi_reg_report', 'acompcor_report', 'tcompcor_report', 'syn_sdc_report',
                    'ica_aroma_report']
            ),
        name='inputnode')

    ds_summary_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='summary'),
        name='ds_summary_report', run_without_submitting=True)

    ds_validation_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='validation'),
        name='ds_validation_report', run_without_submitting=True)

    ds_epi_mask_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='epi_mask'),
        name='ds_epi_mask_report', run_without_submitting=True)

    ds_syn_sdc_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='syn_sdc'),
        name='ds_syn_sdc_report', run_without_submitting=True)

    ds_epi_reg_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='bbr' if freesurfer else 'flt_bbr'),
        name='ds_epi_reg_report', run_without_submitting=True)

    ds_acompcor_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='acompcor'),
        name='ds_acompcor_report', run_without_submitting=True)

    ds_tcompcor_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='tcompcor'),
        name='ds_tcompcor_report', run_without_submitting=True)

    ds_ica_aroma_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='ica_aroma'),
        name='ds_ica_aroma_report', run_without_submitting=True)

    workflow.connect([
        (inputnode, ds_summary_report, [('source_file', 'source_file'),
                                        ('summary_report', 'in_file')]),
        (inputnode, ds_validation_report, [('source_file', 'source_file'),
                                           ('validation_report', 'in_file')]),
        (inputnode, ds_epi_mask_report, [('source_file', 'source_file'),
                                         ('epi_mask_report', 'in_file')]),
        (inputnode, ds_epi_reg_report, [('source_file', 'source_file'),
                                        ('epi_reg_report', 'in_file')]),
        (inputnode, ds_acompcor_report, [('source_file', 'source_file'),
                                         ('acompcor_report', 'in_file')]),
        (inputnode, ds_tcompcor_report, [('source_file', 'source_file'),
                                         ('tcompcor_report', 'in_file')]),
        ])

    if use_aroma:
        workflow.connect([
            (inputnode, ds_ica_aroma_report, [('source_file', 'source_file'),
                                              ('ica_aroma_report', 'in_file')]),
            ])

    if use_syn:
        workflow.connect([
            (inputnode, ds_syn_sdc_report, [('source_file', 'source_file'),
                                            ('syn_sdc_report', 'in_file')]),
            ])

    return workflow


def init_func_derivatives_wf(output_dir, output_spaces, template, freesurfer,
                             use_aroma, name='func_derivatives_wf'):
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 'epi_t1', 'epi_mask_t1', 'epi_mni', 'epi_mask_mni',
                    'confounds', 'surfaces', 'aroma_noise_ics', 'melodic_mix']
            ),
        name='inputnode')

    ds_epi_t1 = pe.Node(DerivativesDataSink(base_directory=output_dir, suffix='space-T1w_preproc'),
                        name='ds_epi_t1', run_without_submitting=True)

    ds_epi_mask_t1 = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                 suffix='space-T1w_brainmask'),
                             name='ds_epi_mask_t1', run_without_submitting=True)

    suffix_fmt = 'space-{}_{}'.format
    ds_epi_mni = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                             suffix=suffix_fmt(template, 'preproc')),
                         name='ds_epi_mni', run_without_submitting=True)
    ds_epi_mask_mni = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                  suffix=suffix_fmt(template, 'brainmask')),
                              name='ds_epi_mask_mni', run_without_submitting=True)

    ds_confounds = pe.Node(DerivativesDataSink(base_directory=output_dir, suffix='confounds'),
                           name="ds_confounds", run_without_submitting=True)

    ds_aroma_noise_ics = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                     suffix='AROMAnoiseICs'),
                                 name="ds_aroma_noise_ics", run_without_submitting=True)

    ds_melodic_mix = pe.Node(DerivativesDataSink(base_directory=output_dir, suffix='MELODICmix'),
                             name="ds_melodic_mix", run_without_submitting=True)

    if use_aroma:
        workflow.connect([
            (inputnode, ds_aroma_noise_ics, [('source_file', 'source_file'),
                                             ('aroma_noise_ics', 'in_file')]),
            (inputnode, ds_melodic_mix, [('source_file', 'source_file'),
                                         ('melodic_mix', 'in_file')]),
            ])

    def get_gifti_name(in_file):
        import os
        import re
        in_format = re.compile(r'(?P<LR>[lr])h.(?P<space>\w+).gii')
        info = in_format.match(os.path.basename(in_file)).groupdict()
        info['LR'] = info['LR'].upper()
        return 'space-{space}.{LR}.func'.format(**info)

    name_surfs = pe.MapNode(niu.Function(function=get_gifti_name),
                            iterfield='in_file', name='name_surfs')

    ds_bold_surfs = pe.MapNode(DerivativesDataSink(base_directory=output_dir),
                               iterfield=['in_file', 'suffix'], name='ds_bold_surfs',
                               run_without_submitting=True)

    workflow.connect([
        (inputnode, ds_confounds, [('source_file', 'source_file'),
                                   ('confounds', 'in_file')]),
        ])

    if 'T1w' in output_spaces:
        workflow.connect([
            (inputnode, ds_epi_t1, [('source_file', 'source_file'),
                                    ('epi_t1', 'in_file')]),
            (inputnode, ds_epi_mask_t1, [('source_file', 'source_file'),
                                         ('epi_mask_t1', 'in_file')]),
            ])
    if 'template' in output_spaces:
        workflow.connect([
            (inputnode, ds_epi_mni, [('source_file', 'source_file'),
                                     ('epi_mni', 'in_file')]),
            (inputnode, ds_epi_mask_mni, [('source_file', 'source_file'),
                                          ('epi_mask_mni', 'in_file')]),
            ])
    if freesurfer and any(space.startswith('fs') for space in output_spaces):
        workflow.connect([
            (inputnode, name_surfs, [('surfaces', 'in_file')]),
            (inputnode, ds_bold_surfs, [('source_file', 'source_file'),
                                        ('surfaces', 'in_file')]),
            (name_surfs, ds_bold_surfs, [('out', 'suffix')]),
            ])

    return workflow
