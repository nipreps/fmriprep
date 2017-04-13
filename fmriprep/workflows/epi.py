#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
EPI MRI -processing workflows.

Originally coded by Craig Moodie. Refactored by the CRN Developers.
"""

import os
import os.path as op

from nipype.pipeline import engine as pe
from nipype.interfaces import ants
from nipype.interfaces import afni
from nipype.interfaces import c3
from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.interfaces import freesurfer as fs
from niworkflows.interfaces.masks import ComputeEPIMask, BETRPT
from niworkflows.interfaces.registration import FLIRTRPT, BBRegisterRPT
from niworkflows.data import get_mni_icbm152_nlin_asym_09c

from fmriprep.interfaces import DerivativesDataSink
from fmriprep.interfaces.utils import nii_concat, EstimateReferenceImage
from fmriprep.utils.misc import _first
from fmriprep.workflows.fieldmap import sdc_unwarp
from fmriprep.workflows.sbref import _extract_wm
from fmriprep.workflows import confounds


def bold_preprocessing(name, metadata, settings):

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['epi',
                                                      'bias_corrected_t1',
                                                      't1_brain',
                                                      't1_mask',
                                                      't1_seg',
                                                      't1_tpms',
                                                      't1_2_mni_forward_transform',
                                                      'subjects_dir',
                                                      'subject_id',
                                                      'fs_2_t1_transform',
                                                      't1w']),
                        name='inputnode')

    # HMC on the EPI
    hmcwf = epi_hmc(metadata=metadata, settings=settings)

    # mean EPI registration to T1w
    epi_2_t1 = ref_epi_t1_registration(reportlet_suffix='bbr',
                                       settings=settings)

    # get confounds
    confounds_wf = confounds.discover_wf(settings)
    confounds_wf.get_node('inputnode').inputs.t1_transform_flags = [False]

    # Apply transforms in 1 shot
    epi_mni_trans_wf = epi_mni_transformation(settings=settings)

    workflow.connect([
        (inputnode, hmcwf, [('epi', 'inputnode.epi')]),
        (inputnode, epi_2_t1, [('t1w', 'inputnode.t1w')]),
        (hmcwf, epi_2_t1, [('outputnode.ref_image', 'inputnode.ref_epi'),
                           ('outputnode.xforms', 'inputnode.hmc_xforms'),
                           ('outputnode.epi_mask', 'inputnode.ref_epi_mask'),
                           ('outputnode.epi_split', 'inputnode.epi_split')]),
        (inputnode, epi_2_t1, [('epi', 'inputnode.name_source'),
                               ('bias_corrected_t1', 'inputnode.bias_corrected_t1'),
                               ('t1_brain', 'inputnode.t1_brain'),
                               ('t1_mask', 'inputnode.t1_mask'),
                               ('t1_seg', 'inputnode.t1_seg')]),

        (inputnode, confounds_wf, [('t1_tpms', 'inputnode.t1_tpms'),
                                   ('epi', 'inputnode.source_file')]),
        (hmcwf, confounds_wf, [
            ('outputnode.movpar_file', 'inputnode.movpar_file')]),
        (epi_2_t1, confounds_wf, [('outputnode.epi_t1', 'inputnode.fmri_file'),
                                  ('outputnode.epi_mask_t1', 'inputnode.epi_mask')]),

        (epi_2_t1, epi_mni_trans_wf, [('outputnode.itk_epi_to_t1', 'inputnode.itk_epi_to_t1')]),
        (hmcwf, epi_mni_trans_wf, [('outputnode.xforms', 'inputnode.hmc_xforms'),
                                   ('outputnode.epi_mask', 'inputnode.epi_mask'),
                                   ('outputnode.epi_split', 'inputnode.epi_split')]),
        (inputnode, epi_mni_trans_wf,
         [('epi', 'inputnode.name_source'),
          ('bias_corrected_t1', 'inputnode.t1'),
          ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform')])
    ])

    if settings['freesurfer']:
        epi_surf = epi_surf_sample(settings=settings)
        workflow.connect([
            (inputnode, epi_2_t1, [('subjects_dir', 'inputnode.subjects_dir'),
                                   ('subject_id', 'inputnode.subject_id'),
                                   ('fs_2_t1_transform', 'inputnode.fs_2_t1_transform')
                                   ]),
            (inputnode, epi_surf, [('subjects_dir', 'inputnode.subjects_dir'),
                                   ('subject_id', 'inputnode.subject_id'),
                                   ('epi', 'inputnode.name_source')]),
            (epi_2_t1, epi_surf, [('outputnode.epi_t1', 'inputnode.source_file')]),
            ])

    return workflow


# pylint: disable=R0914
def epi_hmc(metadata, name='EPI_HMC', settings=None):
    """
    Performs :abbr:`HMC (head motion correction)` over the input
    :abbr:`EPI (echo-planar imaging)` image.
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['epi']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['xforms', 'epi_hmc', 'epi_split', 'epi_mask', 'ref_image',
                'movpar_file', 'n_volumes_to_discard']), name='outputnode')

    def normalize_motion_func(in_file, format):
        import os
        import numpy as np
        from nipype.utils.misc import normalize_mc_params
        mpars = np.loadtxt(in_file)  # mpars is N_t x 6
        mpars = np.apply_along_axis(func1d=normalize_mc_params,
                                    axis=1, arr=mpars,
                                    source=format)
        np.savetxt("motion_params.txt", mpars)
        return os.path.abspath("motion_params.txt")

    normalize_motion = pe.Node(niu.Function(function=normalize_motion_func,
                                            input_names=["in_file", "format"],
                                            output_names=["out_file"]),
                               name="normalize_motion")
    normalize_motion.inputs.format = "FSL"

    # Head motion correction (hmc)
    hmc = pe.Node(fsl.MCFLIRT(
        save_mats=True, save_plots=True), name='EPI_hmc')
    hmc.interface.estimated_memory_gb = settings[
                                            "biggest_epi_file_size_gb"] * 3

    hcm2itk = pe.MapNode(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                         iterfield=['transform_file'], name='hcm2itk')

    inu = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='EPImeanBias')

    # Calculate EPI mask on the average after HMC
    skullstrip_epi = pe.Node(ComputeEPIMask(generate_report=True, dilation=1),
                             name='skullstrip_epi')

    ds_epi_mask = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='epi_mask'),
        name='ds_epi_mask'
    )

    gen_ref = pe.Node(EstimateReferenceImage(), name="gen_ref")

    workflow.connect([
        (inputnode, gen_ref, [('epi', 'in_file')]),
        (gen_ref, inu, [('ref_image', 'input_image')]),
        (inu, hmc, [('output_image', 'ref_file')]),
        (inu, skullstrip_epi, [('output_image', 'in_file')]),
        (inu, outputnode, [('output_image', 'ref_image')]),
    ])

    split = pe.Node(fsl.Split(dimension='t'), name='SplitEPI')
    split.interface.estimated_memory_gb = settings[
                                              "biggest_epi_file_size_gb"] * 3

    if "SliceTiming" in metadata and 'slicetiming' not in settings['ignore']:
        def create_custom_slice_timing_file_func(metadata):
            import os
            slice_timings = metadata["SliceTiming"]
            slice_timings_ms = [str(t) for t in slice_timings]
            out_file = "timings.1D"
            with open("timings.1D", "w") as fp:
                fp.write("\t".join(slice_timings_ms))

            return os.path.abspath(out_file)

        create_custom_slice_timing_file = pe.Node(
            niu.Function(function=create_custom_slice_timing_file_func,
                         input_names=["metadata"],
                         output_names=["out_file"]),
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
            (create_custom_slice_timing_file, slice_timing_correction, [(('out_file', prefix_at),
                                                                         'tpattern')]),
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
        (normalize_motion, outputnode, [('out_file', 'movpar_file')]),
        (skullstrip_epi, outputnode, [('mask_file', 'epi_mask')]),
        (skullstrip_epi, ds_epi_mask, [('out_report', 'in_file')]),
        (inputnode, ds_epi_mask, [('epi', 'source_file')]),
        (inputnode, split, [('epi', 'in_file')]),
        (split, outputnode, [('out_files', 'epi_split')]),
    ])

    return workflow


def ref_epi_t1_registration(reportlet_suffix, name='ref_epi_t1_registration',
                            settings=None):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the T1-space
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['name_source', 'ref_epi', 'ref_epi_mask',
                                      'bias_corrected_t1', 't1_brain', 't1_mask',
                                      't1_seg', 't1w', 'epi_split', 'hmc_xforms',
                                      'subjects_dir', 'subject_id', 'fs_2_t1_transform']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['mat_epi_to_t1', 'mat_t1_to_epi',
                                      'itk_epi_to_t1', 'itk_t1_to_epi',
                                      'epi_t1', 'epi_mask_t1', 'fs_reg_file']),
        name='outputnode'
    )

    # Extract wm mask from segmentation
    wm_mask = pe.Node(
        niu.Function(input_names=['in_file'], output_names=['out_file'],
                     function=_extract_wm),
        name='WM_mask'
    )

    explicit_mask_epi = pe.Node(fsl.ApplyMask(), name="explicit_mask_epi")

    if settings['freesurfer']:
        bbregister = pe.Node(
            BBRegisterRPT(
                contrast_type='t2',
                init='coreg',
                registered_file=True,
                out_fsl_file=True,
                generate_report=True),
            name='bbregister'
            )

        def apply_fs_transform(fs_2_t1_transform, bbreg_transform):
            import os
            import numpy as np
            out_file = os.path.abspath('transform.mat')
            fs_xfm = np.loadtxt(fs_2_t1_transform)
            bbrxfm = np.loadtxt(bbreg_transform)
            out_xfm = fs_xfm.dot(bbrxfm)
            assert np.allclose(out_xfm[3], [0, 0, 0, 1])
            out_xfm[3] = [0, 0, 0, 1]
            np.savetxt(out_file, out_xfm, fmt='%.12g')
            return out_file

        transformer = pe.Node(
            niu.Function(
                function=apply_fs_transform,
                input_names=['fs_2_t1_transform', 'bbreg_transform'],
                output_names=['out_file']),
            name='BBRegTransform')
    else:
        flt_bbr_init = pe.Node(
            FLIRTRPT(generate_report=True, dof=6),
            name='flt_bbr_init'
        )
        flt_bbr = pe.Node(
            FLIRTRPT(generate_report=True, dof=6, cost_func='bbr'),
            name='flt_bbr'
        )
        flt_bbr.inputs.schedule = op.join(os.getenv('FSLDIR'),
                                          'etc/flirtsch/bbr.sch')
        reportlet_suffix = reportlet_suffix.replace('bbr', 'flt_bbr')

    # make equivalent warp fields
    invt_bbr = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='Flirt_BBR_Inv')

    #  EPI to T1 transform matrix is from fsl, using c3 tools to convert to
    #  something ANTs will like.
    fsl2itk_fwd = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_fwd')
    fsl2itk_inv = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_inv')

    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix=reportlet_suffix),
        name='ds_report'
    )

    workflow.connect([
        (inputnode, wm_mask, [('t1_seg', 'in_file')]),
        (inputnode, explicit_mask_epi, [('ref_epi', 'in_file'),
                                        ('ref_epi_mask', 'mask_file')
                                        ]),
        (inputnode, fsl2itk_fwd, [('bias_corrected_t1', 'reference_file'),
                                  ('ref_epi', 'source_file')]),
        (inputnode, fsl2itk_inv, [('ref_epi', 'reference_file'),
                                  ('bias_corrected_t1', 'source_file')]),
        (invt_bbr, outputnode, [('out_file', 'mat_t1_to_epi')]),
        (invt_bbr, fsl2itk_inv, [('out_file', 'transform_file')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_epi_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_epi')]),
        (inputnode, ds_report, [(('name_source', _first), 'source_file')])
    ])

    gen_ref = pe.Node(niu.Function(
        input_names=['fixed_image', 'moving_image'], output_names=['out_file'],
        function=_gen_reference), name='GenNewT1wReference')
    gen_ref.inputs.fixed_image = op.join(get_mni_icbm152_nlin_asym_09c(),
                                         '1mm_T1.nii.gz')

    mask_t1w_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='NearestNeighbor',
                             float=True),
        name='MaskToT1w'
    )

    workflow.connect([
        (inputnode, gen_ref, [('ref_epi_mask', 'moving_image'),
                              ('t1_brain', 'fixed_image')]),
        (fsl2itk_fwd, mask_t1w_tfm, [('itk_transform', 'transforms')]),
        (gen_ref, mask_t1w_tfm, [('out_file', 'reference_image')]),
        (inputnode, mask_t1w_tfm, [('ref_epi_mask', 'input_image')]),
        (mask_t1w_tfm, outputnode, [('output_image', 'epi_mask_t1')]),
    ])

    merge_transforms = pe.MapNode(niu.Merge(2), iterfield=['in2'],
                                  name='MergeTransforms')
    merge = pe.Node(niu.Function(input_names=["in_files", "header_source"],
                                 output_names=["merged_file"],
                                 function=nii_concat), name='MergeEPI')
    merge.interface.estimated_memory_gb = settings[
                                              "biggest_epi_file_size_gb"] * 3

    epi_to_t1w_transform = pe.MapNode(
        ants.ApplyTransforms(interpolation="LanczosWindowedSinc",
                             float=True),
        iterfield=['input_image', 'transforms'],
        name='EPIToT1wTransform')
    epi_to_t1w_transform.terminal_output = 'file'

    workflow.connect([
        (fsl2itk_fwd, merge_transforms, [('itk_transform', 'in1')]),
        (inputnode, merge_transforms, [('hmc_xforms', 'in2')]),
        (merge_transforms, epi_to_t1w_transform, [('out', 'transforms')]),
        (epi_to_t1w_transform, merge, [('output_image', 'in_files')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (merge, outputnode, [('merged_file', 'epi_t1')]),
        (inputnode, epi_to_t1w_transform, [('epi_split', 'input_image')]),
        (gen_ref, epi_to_t1w_transform, [('out_file', 'reference_image')]),
    ])

    if not settings["skip_native"]:
        # Write corrected file in the designated output dir
        ds_t1w = pe.Node(
            DerivativesDataSink(base_directory=settings['output_dir'],
                                suffix='space-T1w_preproc'),
            name='DerivativesHMCT1w'
        )
        ds_t1w_mask = pe.Node(
            DerivativesDataSink(base_directory=settings['output_dir'],
                                suffix='space-T1w_brainmask'),
            name='DerivativesHMCT1wmask'
        )

        workflow.connect([
            (inputnode, ds_t1w, [(('name_source', _first), 'source_file')]),
            (inputnode, ds_t1w_mask,
             [(('name_source', _first), 'source_file')]),
            (outputnode, ds_t1w, [('epi_t1', 'in_file')]),
            (mask_t1w_tfm, ds_t1w_mask, [('output_image', 'in_file')]),
            ])

    if settings['freesurfer']:
        workflow.connect([
            (inputnode, bbregister, [('subjects_dir', 'subjects_dir'),
                                     ('subject_id', 'subject_id')]),
            (explicit_mask_epi, bbregister, [('out_file', 'source_file')]),
            (inputnode, transformer, [('fs_2_t1_transform', 'fs_2_t1_transform')]),
            (bbregister, transformer, [('out_fsl_file', 'bbreg_transform')]),
            (transformer, invt_bbr, [('out_file', 'in_file')]),
            (transformer, outputnode, [('out_file', 'mat_epi_to_t1')]),
            (transformer, fsl2itk_fwd, [('out_file', 'transform_file')]),
            (bbregister, ds_report, [('out_report', 'in_file')]),
            (bbregister, outputnode, [('out_reg_file', 'fs_reg_file')]),
        ])
    else:
        workflow.connect([
            (explicit_mask_epi, flt_bbr_init, [('out_file', 'in_file')]),
            (inputnode, flt_bbr_init, [('t1_brain', 'reference')]),
            (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
            (inputnode, flt_bbr, [('t1_brain', 'reference')]),
            (explicit_mask_epi, flt_bbr, [('out_file', 'in_file')]),
            (wm_mask, flt_bbr, [('out_file', 'wm_seg')]),
            (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
            (flt_bbr, outputnode, [('out_matrix_file', 'mat_epi_to_t1')]),
            (flt_bbr, fsl2itk_fwd, [('out_matrix_file', 'transform_file')]),
            (flt_bbr, ds_report, [('out_report', 'in_file')]),
        ])

    return workflow


def epi_sbref_registration(settings, name='EPI_SBrefRegistration'):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 'epi_name_source', 'sbref',
                                      'epi_mean', 'epi_mask',
                                      'sbref_mask']),
        name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['epi_registered', 'out_mat', 'out_mat_inv']), name='outputnode')

    epi_sbref = pe.Node(FLIRTRPT(generate_report=True, dof=6,
                                 out_matrix_file='init.mat',
                                 out_file='init.nii.gz'),
                        name='EPI2SBRefRegistration')
    # make equivalent inv
    sbref_epi = pe.Node(fsl.ConvertXFM(invert_xfm=True), name="SBRefEPI")

    epi_split = pe.Node(fsl.Split(dimension='t'), name='EPIsplit')
    epi_xfm = pe.MapNode(fsl.preprocess.ApplyXFM(), name='EPIapplyXFM',
                         iterfield=['in_file'])
    epi_merge = pe.Node(niu.Function(input_names=["in_files", "header_source"],
                                     output_names=["merged_file"],
                                     function=nii_concat), name='EPImergeback')

    ds_sbref = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='preproc'), name='DerivHMC_SBRef')

    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='epi_sbref'),
        name="DS_Report")

    workflow.connect([
        (inputnode, epi_split, [('epi', 'in_file')]),
        (inputnode, epi_sbref, [('sbref', 'reference'),
                                ('sbref_mask', 'ref_weight')]),
        (inputnode, epi_xfm, [('sbref', 'reference')]),
        (inputnode, epi_sbref, [('epi_mean', 'in_file'),
                                ('epi_mask', 'in_weight')]),

        (epi_split, epi_xfm, [('out_files', 'in_file')]),
        (epi_sbref, epi_xfm, [('out_matrix_file', 'in_matrix_file')]),
        (epi_xfm, epi_merge, [('out_file', 'in_files')]),
        (inputnode, epi_merge, [('epi_name_source', 'header_source')]),
        (epi_sbref, outputnode, [('out_matrix_file', 'out_mat')]),
        (epi_merge, outputnode, [('merged_file', 'epi_registered')]),

        (epi_sbref, sbref_epi, [('out_matrix_file', 'in_file')]),
        (sbref_epi, outputnode, [('out_file', 'out_mat_inv')]),

        (epi_merge, ds_sbref, [('merged_file', 'in_file')]),
        (inputnode, ds_sbref, [('epi_name_source', 'source_file')]),
        (inputnode, ds_report, [('epi_name_source', 'source_file')]),
        (epi_sbref, ds_report, [('out_report', 'in_file')])
    ])

    return workflow


def epi_surf_sample(name='SurfaceSample', settings=None):
    """ Sample functional images to FreeSurfer surfaces

    For each vertex, the cortical ribbon is sampled at six points (spaced 20% of thickness apart)
    and averaged.

    Outputs are in GIFTI format.

    Settings used:
        skip_native : sample only to fsaverage space, skipping subject native surface
        output_dir : directory to save derivatives to
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 'subject_id', 'subjects_dir', 'name_source']),
        name='inputnode')

    def select_targets(subject_id, template, skip_native):
        """ Select targets based on a provided template and whether to generate outputs in
        native space.

        Source target is defined from registration file to preclude mismatches

        Returns targets (FreeSurfer subject names) and spaces (FMRIPREP output space names)
        """
        targets = [template]
        spaces = [template]
        if not skip_native:
            targets.append(subject_id)
            spaces.append('fsnative')
        return targets, spaces

    targets = pe.Node(
        niu.Function(function=select_targets,
                     input_names=['subject_id', 'template', 'skip_native'],
                     output_names=['targets', 'spaces']),
        name='targets')
    # When study-specific templates are created, update this:
    targets.inputs.template = 'fsaverage'
    targets.inputs.skip_native = settings['skip_native']

    # Rename the source file to the output space to simplify naming later
    rename_src = pe.MapNode(niu.Rename(format_string='%(subject)s', keep_ext=True),
                            iterfield='subject', name='RenameFunc')

    sampler = pe.MapNode(
        fs.SampleToSurface(sampling_method='average', sampling_range=(0, 1, 0.2),
                           sampling_units='frac', reg_header=True,
                           interp_method='trilinear', cortex_mask=True,
                           out_type='gii'),
        iterfield=['source_file', 'target_subject'],
        iterables=('hemi', ['lh', 'rh']),
        name='sampler')

    merger = pe.JoinNode(niu.Merge(1, ravel_inputs=True), name='merger',
                         joinsource='sampler', joinfield=['in1'])

    def normalize_giftis(in_file):
        import os
        import re
        in_format = re.compile(r'(?P<LR>[lr])h.(?P<space>\w+).gii')
        info = in_format.match(os.path.basename(in_file)).groupdict()
        info['LR'] = info['LR'].upper()
        return 'space-{space}.{LR}.func'.format(**info)

    normalize = pe.MapNode(
        niu.Function(function=normalize_giftis, input_names=['in_file'],
                     output_names=['normalized']),
        iterfield='in_file',
        name='normalize')

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

    update_metadata = pe.MapNode(
        niu.Function(
            function=update_gifti_metadata,
            input_names=['in_file'],
            output_names=['out_file']),
        iterfield='in_file',
        name='UpdateMetadata')

    bold_surfaces = pe.MapNode(
         DerivativesDataSink(base_directory=settings['output_dir']),
         iterfield=['in_file', 'suffix'],
         name='BOLDSurfaces')

    workflow.connect([
        (inputnode, targets, [('subject_id', 'subject_id')]),
        (inputnode, rename_src, [('source_file', 'in_file')]),
        (targets, rename_src, [('spaces', 'subject')]),
        (inputnode, sampler, [('subjects_dir', 'subjects_dir'),
                              ('subject_id', 'subject_id')]),
        (targets, sampler, [('targets', 'target_subject')]),
        (rename_src, sampler, [('out_file', 'source_file')]),
        (sampler, merger, [('out_file', 'in1')]),
        (merger, normalize, [('out', 'in_file')]),
        (merger, update_metadata, [('out', 'in_file')]),
        (inputnode, bold_surfaces,
         [(('name_source', _first), 'source_file')]),
        (update_metadata, bold_surfaces, [('out_file', 'in_file')]),
        (normalize, bold_surfaces, [('normalized', 'suffix')]),
        ])

    return workflow


def epi_mni_transformation(name='EPIMNITransformation', settings=None):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_epi_to_t1',
            't1_2_mni_forward_transform',
            'name_source',
            'epi_split',
            'epi_mask',
            't1',
            'hmc_xforms'
        ]),
        name='inputnode'
    )

    def _aslist(in_value):
        if isinstance(in_value, list):
            return in_value
        return [in_value]

    gen_ref = pe.Node(niu.Function(
        input_names=['fixed_image', 'moving_image'], output_names=['out_file'],
        function=_gen_reference), name='GenNewMNIReference')
    gen_ref.inputs.fixed_image = op.join(get_mni_icbm152_nlin_asym_09c(),
                                         '1mm_T1.nii.gz')

    mask_merge_tfms = pe.Node(niu.Merge(2), name='MaskMergeTfms')
    mask_mni_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='NearestNeighbor',
                             float=True),
        name='MaskToMNI'
    )

    # Write corrected file in the designated output dir
    ds_mni = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='space-MNI152NLin2009cAsym_preproc'),
        name='DerivativesHMCMNI'
    )
    ds_mni_mask = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='space-MNI152NLin2009cAsym_brainmask'),
        name='DerivativesHMCMNImask'
    )

    workflow.connect([
        (inputnode, ds_mni, [('name_source', 'source_file')]),
        (inputnode, ds_mni_mask, [('name_source', 'source_file')]),
        (inputnode, gen_ref, [('epi_mask', 'moving_image')]),

        (inputnode, mask_merge_tfms, [('t1_2_mni_forward_transform', 'in1'),
                                      (('itk_epi_to_t1', _aslist), 'in2')]),
        (mask_merge_tfms, mask_mni_tfm, [('out', 'transforms')]),
        (gen_ref, mask_mni_tfm, [('out_file', 'reference_image')]),
        (inputnode, mask_mni_tfm, [('epi_mask', 'input_image')]),
        (mask_mni_tfm, ds_mni_mask, [('output_image', 'in_file')])
    ])

    merge_transforms = pe.MapNode(niu.Merge(3),
                                  iterfield=['in3'],
                                  name='MergeTransforms')
    merge = pe.Node(niu.Function(input_names=["in_files", "header_source"],
                                 output_names=["merged_file"],
                                 function=nii_concat), name='MergeEPI')
    merge.interface.estimated_memory_gb = settings[
                                              "biggest_epi_file_size_gb"] * 3

    epi_to_mni_transform = pe.MapNode(
        ants.ApplyTransforms(interpolation="LanczosWindowedSinc",
                             float=True),
        iterfield=['input_image', 'transforms'],
        name='EPIToMNITransform')
    epi_to_mni_transform.terminal_output = 'file'

    workflow.connect([
        (inputnode, merge_transforms, [('t1_2_mni_forward_transform', 'in1'),
                                       (('itk_epi_to_t1', _aslist), 'in2'),
                                       ('hmc_xforms', 'in3')]),
        (merge_transforms, epi_to_mni_transform, [('out', 'transforms')]),
        (epi_to_mni_transform, merge, [('output_image', 'in_files')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode, epi_to_mni_transform, [('epi_split', 'input_image')]),
        (gen_ref, epi_to_mni_transform, [('out_file', 'reference_image')]),
        (merge, ds_mni, [('merged_file', 'in_file')]),
    ])

    return workflow


# pylint: disable=R0914
def epi_unwarp(name='EPIUnwarpWorkflow', settings=None):
    """ A workflow to correct EPI images """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 'fmap', 'fmap_ref', 'fmap_mask',
                                      't1_seg']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['epi_unwarp', 'epi_mean', 'epi_mask']),
        name='outputnode'
    )

    unwarp = sdc_unwarp()
    unwarp.inputs.inputnode.hmc_movpar = ''

    # Compute outputs
    mean = pe.Node(fsl.MeanImage(dimension='T'), name='EPImean')
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                  name='EPIBET')

    ds_epi_unwarp = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='epi_unwarp'),
        name='DerivUnwarp_EPUnwarp_EPI'
    )

    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='epi_unwarp_bet'),
        name="DS_Report")

    workflow.connect([
        (inputnode, unwarp, [('fmap', 'inputnode.fmap'),
                             ('fmap_ref', 'inputnode.fmap_ref'),
                             ('fmap_mask', 'inputnode.fmap_mask'),
                             ('epi', 'inputnode.in_file')]),
        (inputnode, ds_epi_unwarp, [('epi', 'source_file')]),
        (unwarp, mean, [('outputnode.out_file', 'in_file')]),
        (mean, bet, [('out_file', 'in_file')]),
        (bet, outputnode, [('out_file', 'epi_mean'),
                           ('mask_file', 'epi_mask')]),
        (unwarp, outputnode, [('outputnode.out_file', 'epi_unwarp')]),
        (unwarp, ds_epi_unwarp, [('outputnode.out_file', 'in_file')]),
        (inputnode, ds_report, [('epi', 'source_file')]),
        (bet, ds_report, [('out_report', 'in_file')])
    ])

    return workflow


def _gen_reference(fixed_image, moving_image, out_file=None):
    import os.path as op
    import numpy
    from nilearn.image import resample_img, load_img

    if out_file is None:
        fname, ext = op.splitext(op.basename(fixed_image))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('%s_wm%s' % (fname, ext))

    new_zooms = load_img(moving_image).header.get_zooms()

    new_ref_im = resample_img(fixed_image, target_affine=numpy.diag(new_zooms),
                              interpolation='nearest')

    new_ref_im.to_filename(out_file)

    return out_file
