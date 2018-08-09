#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_pepolar :

Phase Encoding POLARity (*PEPOLAR*) techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import afni, ants, fsl, utility as niu
from niworkflows.interfaces import CopyHeader
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT

from ...engine import Workflow
from ...interfaces import StructuralReference
from ..bold.util import init_enhance_and_skullstrip_bold_wf


def init_pepolar_unwarp_wf(bold_meta, epi_fmaps, omp_nthreads=1,
                           name="pepolar_unwarp_wf"):
    """
    This workflow takes in a set of EPI files with opposite phase encoding
    direction than the target file and calculates a displacements field
    (in other words, an ANTs-compatible warp file).

    This procedure works if there is only one '_epi' file is present
    (as long as it has the opposite phase encoding direction to the target
    file). The target file will be used to estimate the field distortion.
    However, if there is another '_epi' file present with a matching
    phase encoding direction to the target it will be used instead.

    Currently, different phase encoding dimension in the target file and the
    '_epi' file(s) (for example 'i' and 'j') is not supported.

    The warp field correcting for the distortions is estimated using AFNI's
    3dQwarp, with displacement estimation limited to the target file phase
    encoding direction.

    It also calculates a new mask for the input dataset that takes into
    account the distortions.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.pepolar import init_pepolar_unwarp_wf
        wf = init_pepolar_unwarp_wf(
            bold_meta={'PhaseEncodingDirection': 'j'},
            epi_fmaps=[('/dataset/sub-01/fmap/sub-01_epi.nii.gz', 'j-')],
            omp_nthreads=8)


    Inputs

        in_reference
            the reference image
        in_reference_brain
            the reference image skullstripped
        in_mask
            a brain mask corresponding to ``in_reference``

    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_mask
            mask of the unwarped input file

    """
    bold_file_pe = bold_meta["PhaseEncodingDirection"]

    args = '-noXdis -noYdis -noZdis'
    rm_arg = {'i': '-noXdis',
              'j': '-noYdis',
              'k': '-noZdis'}[bold_file_pe[0]]
    args = args.replace(rm_arg, '')

    usable_fieldmaps_matching_pe = []
    usable_fieldmaps_opposite_pe = []
    for fmap, fmap_pe in epi_fmaps:
        if fmap_pe == bold_file_pe:
            usable_fieldmaps_matching_pe.append(fmap)
        elif fmap_pe[0] == bold_file_pe[0]:
            usable_fieldmaps_opposite_pe.append(fmap)

    if not usable_fieldmaps_opposite_pe:
        raise Exception("None of the discovered fieldmaps has the right "
                        "phase encoding direction. Possibly a problem with "
                        "metadata. If not, rerun with '--ignore fieldmaps' to "
                        "skip distortion correction step.")

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on two echo-planar imaging (EPI) references with opposing phase-encoding
directions, using `3dQwarp` @afni (AFNI {afni_ver}).
""".format(afni_ver=''.join(list(afni.QwarpPlusMinus().version or '<ver>')))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_reference', 'in_reference_brain', 'in_mask']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_warp', 'out_mask']),
        name='outputnode')

    prepare_epi_opposite_wf = init_prepare_epi_wf(omp_nthreads=omp_nthreads,
                                                  name="prepare_epi_opposite_wf")
    prepare_epi_opposite_wf.inputs.inputnode.fmaps = usable_fieldmaps_opposite_pe

    qwarp = pe.Node(afni.QwarpPlusMinus(pblur=[0.05, 0.05],
                                        blur=[-1, -1],
                                        noweight=True,
                                        minpatch=9,
                                        nopadWARP=True,
                                        environ={'OMP_NUM_THREADS': '%d' % omp_nthreads},
                                        args=args),
                    name='qwarp', n_procs=omp_nthreads)

    workflow.connect([
        (inputnode, prepare_epi_opposite_wf, [('in_reference_brain', 'inputnode.ref_brain')]),
        (prepare_epi_opposite_wf, qwarp, [('outputnode.out_file', 'base_file')]),
    ])

    if usable_fieldmaps_matching_pe:
        prepare_epi_matching_wf = init_prepare_epi_wf(omp_nthreads=omp_nthreads,
                                                      name="prepare_epi_matching_wf")
        prepare_epi_matching_wf.inputs.inputnode.fmaps = usable_fieldmaps_matching_pe

        workflow.connect([
            (inputnode, prepare_epi_matching_wf, [('in_reference_brain', 'inputnode.ref_brain')]),
            (prepare_epi_matching_wf, qwarp, [('outputnode.out_file', 'source_file')]),
        ])
    else:
        workflow.connect([(inputnode, qwarp, [('in_reference_brain', 'source_file')])])

    to_ants = pe.Node(niu.Function(function=_fix_hdr), name='to_ants',
                      mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name='cphdr_warp', mem_gb=0.01)

    unwarp_reference = pe.Node(ANTSApplyTransformsRPT(dimension=3,
                                                      generate_report=False,
                                                      float=True,
                                                      interpolation='LanczosWindowedSinc'),
                               name='unwarp_reference')

    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode, cphdr_warp, [('in_reference', 'hdr_file')]),
        (qwarp, cphdr_warp, [('source_warp', 'in_file')]),
        (cphdr_warp, to_ants, [('out_file', 'in_file')]),
        (to_ants, unwarp_reference, [('out', 'transforms')]),
        (inputnode, unwarp_reference, [('in_reference', 'reference_image'),
                                       ('in_reference', 'input_image')]),
        (unwarp_reference, enhance_and_skullstrip_bold_wf, [
            ('output_image', 'inputnode.in_file')]),
        (unwarp_reference, outputnode, [('output_image', 'out_reference')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.mask_file', 'out_mask'),
            ('outputnode.skull_stripped_file', 'out_reference_brain')]),
        (to_ants, outputnode, [('out', 'out_warp')]),
    ])

    return workflow


def init_prepare_epi_wf(omp_nthreads, name="prepare_epi_wf"):
    """
    This workflow takes in a set of EPI files with with the same phase
    encoding direction and returns a single 3D volume ready to be used in
    field distortion estimation.

    The procedure involves: estimating a robust template using FreeSurfer's
    'mri_robust_template', bias field correction using ANTs N4BiasFieldCorrection
    and AFNI 3dUnifize, skullstripping using FSL BET and AFNI 3dAutomask,
    and rigid coregistration to the reference using ANTs.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.pepolar import init_prepare_epi_wf
        wf = init_prepare_epi_wf(omp_nthreads=8)


    Inputs

        fmaps
            list of 3D or 4D NIfTI images
        ref_brain
            coregistration reference (skullstripped and bias field corrected)

    Outputs

        out_file
            single 3D NIfTI file

    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['fmaps', 'ref_brain']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    split = pe.MapNode(fsl.Split(dimension='t'), iterfield='in_file',
                       name='split')

    merge = pe.Node(
        StructuralReference(auto_detect_sensitivity=True,
                            initial_timepoint=1,
                            fixed_timepoint=True,  # Align to first image
                            intensity_scaling=True,
                            # 7-DOF (rigid + intensity)
                            no_iteration=True,
                            subsample_threshold=200,
                            out_file='template.nii.gz'),
        name='merge')

    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads)

    ants_settings = pkgr.resource_filename('fmriprep',
                                           'data/translation_rigid.json')
    fmap2ref_reg = pe.Node(ants.Registration(from_file=ants_settings,
                                             output_warped_image=True),
                           name='fmap2ref_reg', n_procs=omp_nthreads)

    workflow = Workflow(name=name)

    def _flatten(l):
        from nipype.utils.filemanip import filename_to_list
        return [item for sublist in l for item in filename_to_list(sublist)]

    workflow.connect([
        (inputnode, split, [('fmaps', 'in_file')]),
        (split, merge, [(('out_files', _flatten), 'in_files')]),
        (merge, enhance_and_skullstrip_bold_wf, [('out_file', 'inputnode.in_file')]),
        (enhance_and_skullstrip_bold_wf, fmap2ref_reg, [
            ('outputnode.skull_stripped_file', 'moving_image')]),
        (inputnode, fmap2ref_reg, [('ref_brain', 'fixed_image')]),
        (fmap2ref_reg, outputnode, [('warped_image', 'out_file')]),
    ])

    return workflow


def _fix_hdr(in_file, newpath=None):
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    hdr = nii.header.copy()
    hdr.set_data_dtype('<f4')
    hdr.set_intent('vector', (), '')
    out_file = fname_presuffix(in_file, "_warpfield", newpath=newpath)
    nb.Nifti1Image(nii.get_data().astype('<f4'), nii.affine, hdr).to_filename(
        out_file)
    return out_file
