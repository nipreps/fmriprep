# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Registration workflows
++++++++++++++++++++++

.. autofunction:: init_bold_reg_wf
.. autofunction:: init_bold_t1_trans_wf
.. autofunction:: init_bbreg_wf
.. autofunction:: init_fsl_bbr_wf

"""

import os
import os.path as op

import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl, c3
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
# See https://github.com/poldracklab/fmriprep/issues/768
from niworkflows.interfaces.freesurfer import (
    PatchedConcatenateLTA as ConcatenateLTA,
    PatchedBBRegisterRPT as BBRegisterRPT,
    PatchedMRICoregRPT as MRICoregRPT,
    PatchedLTAConvert as LTAConvert)
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from niworkflows.interfaces.images import extract_wm
from niworkflows.interfaces.itk import MultiApplyTransforms
from niworkflows.interfaces.registration import FLIRTRPT
from niworkflows.interfaces.utils import GenerateSamplingReference
from niworkflows.interfaces.nilearn import Merge

from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink
from ... import config


LOGGER = config.loggers.workflow


def init_bold_reg_wf(freesurfer, use_bbr, bold2t1w_dof, mem_gb, omp_nthreads,
                     use_compression=True, write_report=True, name='bold_reg_wf'):
    """
    Build a workflow to run same-subject, BOLD-to-T1w image-registration.

    Calculates the registration between a reference BOLD image and T1w-space
    using a boundary-based registration (BBR) cost function.
    If FreeSurfer-based preprocessing is enabled, the ``bbregister`` utility
    is used to align the BOLD images to the reconstructed subject, and the
    resulting transform is adjusted to target the T1 space.
    If FreeSurfer-based preprocessing is disabled, FSL FLIRT is used with the
    BBR cost function to directly target the T1 space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.registration import init_bold_reg_wf
            wf = init_bold_reg_wf(freesurfer=True,
                                  mem_gb=3,
                                  omp_nthreads=1,
                                  use_bbr=True,
                                  bold2t1w_dof=9)

    Parameters
    ----------
    freesurfer : :obj:`bool`
        Enable FreeSurfer functional registration (bbregister)
    use_bbr : :obj:`bool` or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
    bold2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for BOLD-T1w registration
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``bold_reg_wf``)
    use_compression : :obj:`bool`
        Save registered BOLD series as ``.nii.gz``
    use_fieldwarp : :obj:`bool`
        Include SDC warp in single-shot transform from BOLD to T1
    write_report : :obj:`bool`
        Whether a reportlet should be stored

    Inputs
    ------
    ref_bold_brain
        Reference image to which BOLD series is aligned
        If ``fieldwarp == True``, ``ref_bold_brain`` should be unwarped
    t1w_brain
        Skull-stripped ``t1w_preproc``
    t1w_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    Outputs
    -------
    itk_bold_to_t1
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    itk_t1_to_bold
        Affine transform from T1 space to BOLD space (ITK format)
    fallback
        Boolean indicating whether BBR was rejected (mri_coreg registration returned)

    See also
    --------
      * :py:func:`~fmriprep.workflows.bold.registration.init_bbreg_wf`
      * :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`

    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['ref_bold_brain', 't1w_brain', 't1w_dseg',
                    'subjects_dir', 'subject_id', 'fsnative2t1w_xfm']),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_bold_to_t1', 'itk_t1_to_bold', 'fallback']),
        name='outputnode'
    )

    if freesurfer:
        bbr_wf = init_bbreg_wf(use_bbr=use_bbr, bold2t1w_dof=bold2t1w_dof,
                               omp_nthreads=omp_nthreads)
    else:
        bbr_wf = init_fsl_bbr_wf(use_bbr=use_bbr, bold2t1w_dof=bold2t1w_dof)

    workflow.connect([
        (inputnode, bbr_wf, [
            ('ref_bold_brain', 'inputnode.in_file'),
            ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            ('t1w_brain', 'inputnode.t1w_brain')]),
        (bbr_wf, outputnode, [('outputnode.itk_bold_to_t1', 'itk_bold_to_t1'),
                              ('outputnode.itk_t1_to_bold', 'itk_t1_to_bold'),
                              ('outputnode.fallback', 'fallback')]),
    ])

    if write_report:
        ds_report_reg = pe.Node(
            DerivativesDataSink(keep_dtype=True),
            name='ds_report_reg', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        def _bold_reg_suffix(fallback, freesurfer):
            if fallback:
                return 'coreg' if freesurfer else 'flirtnobbr'
            return 'bbregister' if freesurfer else 'flirtbbr'

        workflow.connect([
            (bbr_wf, ds_report_reg, [
                ('outputnode.out_report', 'in_file'),
                (('outputnode.fallback', _bold_reg_suffix, freesurfer), 'desc')]),
        ])

    return workflow


def init_bold_t1_trans_wf(freesurfer, mem_gb, omp_nthreads, multiecho=False, use_fieldwarp=False,
                          use_compression=True, name='bold_t1_trans_wf'):
    """
    Co-register the reference BOLD image to T1w-space.

    The workflow uses :abbr:`BBR (boundary-based registration)`.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.registration import init_bold_t1_trans_wf
            wf = init_bold_t1_trans_wf(freesurfer=True,
                                       mem_gb=3,
                                       omp_nthreads=1)

    Parameters
    ----------
    freesurfer : :obj:`bool`
        Enable FreeSurfer functional registration (bbregister)
    use_fieldwarp : :obj:`bool`
        Include SDC warp in single-shot transform from BOLD to T1
    multiecho : :obj:`bool`
        If multiecho data was supplied, HMC already performed
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    use_compression : :obj:`bool`
        Save registered BOLD series as ``.nii.gz``
    name : :obj:`str`
        Name of workflow (default: ``bold_reg_wf``)

    Inputs
    ------
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing
    ref_bold_brain
        Reference image to which BOLD series is aligned
        If ``fieldwarp == True``, ``ref_bold_brain`` should be unwarped
    ref_bold_mask
        Skull-stripping mask of reference image
    t1w_brain
        Skull-stripped bias-corrected structural template image
    t1w_mask
        Mask of the skull-stripped template image
    t1w_aseg
        FreeSurfer's ``aseg.mgz`` atlas projected into the T1w reference
        (only if ``recon-all`` was run).
    t1w_aparc
        FreeSurfer's ``aparc+aseg.mgz`` atlas projected into the T1w reference
        (only if ``recon-all`` was run).
    bold_split
        Individual 3D BOLD volumes, not motion corrected
    hmc_xforms
        List of affine transforms aligning each volume to ``ref_image`` in ITK format
    itk_bold_to_t1
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    fieldwarp
        a :abbr:`DFM (displacements field map)` in ITK format

    Outputs
    -------
    bold_t1
        Motion-corrected BOLD series in T1 space
    bold_t1_ref
        Reference, contrast-enhanced summary of the motion-corrected BOLD series in T1w space
    bold_mask_t1
        BOLD mask in T1 space
    bold_aseg_t1
        FreeSurfer's ``aseg.mgz`` atlas, in T1w-space at the BOLD resolution
        (only if ``recon-all`` was run).
    bold_aparc_t1
        FreeSurfer's ``aparc+aseg.mgz`` atlas, in T1w-space at the BOLD resolution
        (only if ``recon-all`` was run).

    See also
    --------
      * :py:func:`~fmriprep.workflows.bold.registration.init_bbreg_wf`
      * :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`

    """
    from niworkflows.func.util import init_bold_reference_wf
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['name_source', 'ref_bold_brain', 'ref_bold_mask',
                    't1w_brain', 't1w_mask', 't1w_aseg', 't1w_aparc',
                    'bold_split', 'fieldwarp', 'hmc_xforms',
                    'itk_bold_to_t1']),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'bold_t1', 'bold_t1_ref', 'bold_mask_t1',
            'bold_aseg_t1', 'bold_aparc_t1']),
        name='outputnode'
    )

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref',
                      mem_gb=0.3)  # 256x256x256 * 64 / 8 ~ 150MB

    mask_t1w_tfm = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', float=True),
        name='mask_t1w_tfm', mem_gb=0.1
    )

    workflow.connect([
        (inputnode, gen_ref, [('ref_bold_brain', 'moving_image'),
                              ('t1w_brain', 'fixed_image'),
                              ('t1w_mask', 'fov_mask')]),
        (inputnode, mask_t1w_tfm, [('ref_bold_mask', 'input_image')]),
        (gen_ref, mask_t1w_tfm, [('out_file', 'reference_image')]),
        (inputnode, mask_t1w_tfm, [('itk_bold_to_t1', 'transforms')]),
        (mask_t1w_tfm, outputnode, [('output_image', 'bold_mask_t1')]),
    ])

    if freesurfer:
        # Resample aseg and aparc in T1w space (no transforms needed)
        aseg_t1w_tfm = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', transforms='identity', float=True),
            name='aseg_t1w_tfm', mem_gb=0.1)
        aparc_t1w_tfm = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', transforms='identity', float=True),
            name='aparc_t1w_tfm', mem_gb=0.1)

        workflow.connect([
            (inputnode, aseg_t1w_tfm, [('t1w_aseg', 'input_image')]),
            (inputnode, aparc_t1w_tfm, [('t1w_aparc', 'input_image')]),
            (gen_ref, aseg_t1w_tfm, [('out_file', 'reference_image')]),
            (gen_ref, aparc_t1w_tfm, [('out_file', 'reference_image')]),
            (aseg_t1w_tfm, outputnode, [('output_image', 'bold_aseg_t1')]),
            (aparc_t1w_tfm, outputnode, [('output_image', 'bold_aparc_t1')]),
        ])

    bold_to_t1w_transform = pe.Node(
        MultiApplyTransforms(interpolation="LanczosWindowedSinc", float=True, copy_dtype=True),
        name='bold_to_t1w_transform', mem_gb=mem_gb * 3 * omp_nthreads, n_procs=omp_nthreads)

    # merge 3D volumes into 4D timeseries
    merge = pe.Node(Merge(compress=use_compression), name='merge', mem_gb=mem_gb)

    # Generate a reference on the target T1w space
    gen_final_ref = init_bold_reference_wf(omp_nthreads, pre_mask=True)

    if not multiecho:
        # Merge transforms placing the head motion correction last
        nforms = 2 + int(use_fieldwarp)
        merge_xforms = pe.Node(niu.Merge(nforms), name='merge_xforms',
                               run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        if use_fieldwarp:
            workflow.connect([
                (inputnode, merge_xforms, [('fieldwarp', 'in2')])
            ])

        workflow.connect([
            # merge transforms
            (inputnode, merge_xforms, [
                ('hmc_xforms', 'in%d' % nforms),
                ('itk_bold_to_t1', 'in1')]),
            (merge_xforms, bold_to_t1w_transform, [('out', 'transforms')]),
            (inputnode, bold_to_t1w_transform, [('bold_split', 'input_image')]),
        ])

    else:
        from nipype.interfaces.fsl import Split as FSLSplit
        bold_split = pe.Node(FSLSplit(dimension='t'), name='bold_split',
                             mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, bold_split, [('bold_split', 'in_file')]),
            (bold_split, bold_to_t1w_transform, [('out_files', 'input_image')]),
            (inputnode, bold_to_t1w_transform, [('itk_bold_to_t1', 'transforms')]),
        ])

    workflow.connect([
        (inputnode, merge, [('name_source', 'header_source')]),
        (gen_ref, bold_to_t1w_transform, [('out_file', 'reference_image')]),
        (bold_to_t1w_transform, merge, [('out_files', 'in_files')]),
        (merge, gen_final_ref, [('out_file', 'inputnode.bold_file')]),
        (mask_t1w_tfm, gen_final_ref, [('output_image', 'inputnode.bold_mask')]),
        (merge, outputnode, [('out_file', 'bold_t1')]),
        (gen_final_ref, outputnode, [('outputnode.ref_image', 'bold_t1_ref')]),
    ])

    return workflow


def init_bbreg_wf(use_bbr, bold2t1w_dof, omp_nthreads, name='bbreg_wf'):
    """
    Build a workflow to run FreeSurfer's ``bbregister``.

    This workflow uses FreeSurfer's ``bbregister`` to register a BOLD image to
    a T1-weighted structural image.

    It is a counterpart to :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`,
    which performs the same task using FSL's FLIRT with a BBR cost function.
    The ``use_bbr`` option permits a high degree of control over registration.
    If ``False``, standard, affine coregistration will be performed using
    FreeSurfer's ``mri_coreg`` tool.
    If ``True``, ``bbregister`` will be seeded with the initial transform found
    by ``mri_coreg`` (equivalent to running ``bbregister --init-coreg``).
    If ``None``, after ``bbregister`` is run, the resulting affine transform
    will be compared to the initial transform found by ``mri_coreg``.
    Excessive deviation will result in rejecting the BBR refinement and
    accepting the original, affine registration.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.registration import init_bbreg_wf
            wf = init_bbreg_wf(use_bbr=True, bold2t1w_dof=9, omp_nthreads=1)


    Parameters
    ----------
    use_bbr : :obj:`bool` or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
    bold2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for BOLD-T1w registration
    name : :obj:`str`, optional
        Workflow name (default: bbreg_wf)

    Inputs
    ------
    in_file
        Reference BOLD image to be registered
    fsnative2t1w_xfm
        FSL-style affine matrix translating from FreeSurfer T1.mgz to T1w
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID (must have folder in SUBJECTS_DIR)
    t1w_brain
        Unused (see :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`)
    t1w_dseg
        Unused (see :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`)

    Outputs
    -------
    itk_bold_to_t1
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    itk_t1_to_bold
        Affine transform from T1 space to BOLD space (ITK format)
    out_report
        Reportlet for assessing registration quality
    fallback
        Boolean indicating whether BBR was rejected (mri_coreg registration returned)

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The BOLD reference was then co-registered to the T1w reference using
`bbregister` (FreeSurfer) which implements boundary-based registration [@bbr].
Co-registration was configured with {dof} degrees of freedom{reason}.
""".format(dof={6: 'six', 9: 'nine', 12: 'twelve'}[bold2t1w_dof],
           reason='' if bold2t1w_dof == 6 else
                  'to account for distortions remaining in the BOLD reference')

    inputnode = pe.Node(
        niu.IdentityInterface([
            'in_file',
            'fsnative2t1w_xfm', 'subjects_dir', 'subject_id',  # BBRegister
            't1w_dseg', 't1w_brain']),  # FLIRT BBR
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['itk_bold_to_t1', 'itk_t1_to_bold', 'out_report', 'fallback']),
        name='outputnode')

    mri_coreg = pe.Node(
        MRICoregRPT(dof=bold2t1w_dof, sep=[4], ftol=0.0001, linmintol=0.01,
                    generate_report=not use_bbr),
        name='mri_coreg', n_procs=omp_nthreads, mem_gb=5)

    lta_concat = pe.Node(ConcatenateLTA(out_file='out.lta'), name='lta_concat')
    # XXX LTA-FSL-ITK may ultimately be able to be replaced with a straightforward
    # LTA-ITK transform, but right now the translation parameters are off.
    lta2fsl_fwd = pe.Node(LTAConvert(out_fsl=True), name='lta2fsl_fwd')
    lta2fsl_inv = pe.Node(LTAConvert(out_fsl=True, invert=True), name='lta2fsl_inv')
    fsl2itk_fwd = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_fwd', mem_gb=DEFAULT_MEMORY_MIN_GB)
    fsl2itk_inv = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_inv', mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, mri_coreg, [('subjects_dir', 'subjects_dir'),
                                ('subject_id', 'subject_id'),
                                ('in_file', 'source_file')]),
        # Output ITK transforms
        (inputnode, lta_concat, [('fsnative2t1w_xfm', 'in_lta2')]),
        (lta_concat, lta2fsl_fwd, [('out_file', 'in_lta')]),
        (lta_concat, lta2fsl_inv, [('out_file', 'in_lta')]),
        (inputnode, fsl2itk_fwd, [('t1w_brain', 'reference_file'),
                                  ('in_file', 'source_file')]),
        (inputnode, fsl2itk_inv, [('in_file', 'reference_file'),
                                  ('t1w_brain', 'source_file')]),
        (lta2fsl_fwd, fsl2itk_fwd, [('out_fsl', 'transform_file')]),
        (lta2fsl_inv, fsl2itk_inv, [('out_fsl', 'transform_file')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_bold_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_bold')]),
    ])

    # Short-circuit workflow building, use initial registration
    if use_bbr is False:
        workflow.connect([
            (mri_coreg, outputnode, [('out_report', 'out_report')]),
            (mri_coreg, lta_concat, [('out_lta_file', 'in_lta1')])])
        outputnode.inputs.fallback = True

        return workflow

    bbregister = pe.Node(
        BBRegisterRPT(dof=bold2t1w_dof, contrast_type='t2', registered_file=True,
                      out_lta_file=True, generate_report=True),
        name='bbregister', mem_gb=12)

    workflow.connect([
        (inputnode, bbregister, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id'),
                                 ('in_file', 'source_file')]),
        (mri_coreg, bbregister, [('out_lta_file', 'init_reg_file')]),
    ])

    # Short-circuit workflow building, use boundary-based registration
    if use_bbr is True:
        workflow.connect([
            (bbregister, outputnode, [('out_report', 'out_report')]),
            (bbregister, lta_concat, [('out_lta_file', 'in_lta1')])])
        outputnode.inputs.fallback = False

        return workflow

    transforms = pe.Node(niu.Merge(2), run_without_submitting=True, name='transforms')
    reports = pe.Node(niu.Merge(2), run_without_submitting=True, name='reports')

    lta_ras2ras = pe.MapNode(LTAConvert(out_lta=True), iterfield=['in_lta'],
                             name='lta_ras2ras', mem_gb=2)
    compare_transforms = pe.Node(niu.Function(function=compare_xforms), name='compare_transforms')

    select_transform = pe.Node(niu.Select(), run_without_submitting=True, name='select_transform')
    select_report = pe.Node(niu.Select(), run_without_submitting=True, name='select_report')

    workflow.connect([
        (bbregister, transforms, [('out_lta_file', 'in1')]),
        (mri_coreg, transforms, [('out_lta_file', 'in2')]),
        # Normalize LTA transforms to RAS2RAS (inputs are VOX2VOX) and compare
        (transforms, lta_ras2ras, [('out', 'in_lta')]),
        (lta_ras2ras, compare_transforms, [('out_lta', 'lta_list')]),
        (compare_transforms, outputnode, [('out', 'fallback')]),
        # Select output transform
        (transforms, select_transform, [('out', 'inlist')]),
        (compare_transforms, select_transform, [('out', 'index')]),
        (select_transform, lta_concat, [('out', 'in_lta1')]),
        # Select output report
        (bbregister, reports, [('out_report', 'in1')]),
        (mri_coreg, reports, [('out_report', 'in2')]),
        (reports, select_report, [('out', 'inlist')]),
        (compare_transforms, select_report, [('out', 'index')]),
        (select_report, outputnode, [('out', 'out_report')]),
    ])

    return workflow


def init_fsl_bbr_wf(use_bbr, bold2t1w_dof, name='fsl_bbr_wf'):
    """
    Build a workflow to run FSL's ``flirt``.

    This workflow uses FSL FLIRT to register a BOLD image to a T1-weighted
    structural image, using a boundary-based registration (BBR) cost function.
    It is a counterpart to :py:func:`~fmriprep.workflows.bold.registration.init_bbreg_wf`,
    which performs the same task using FreeSurfer's ``bbregister``.

    The ``use_bbr`` option permits a high degree of control over registration.
    If ``False``, standard, rigid coregistration will be performed by FLIRT.
    If ``True``, FLIRT-BBR will be seeded with the initial transform found by
    the rigid coregistration.
    If ``None``, after FLIRT-BBR is run, the resulting affine transform
    will be compared to the initial transform found by FLIRT.
    Excessive deviation will result in rejecting the BBR refinement and
    accepting the original, affine registration.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.registration import init_fsl_bbr_wf
            wf = init_fsl_bbr_wf(use_bbr=True, bold2t1w_dof=9)


    Parameters
    ----------
    use_bbr : :obj:`bool` or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
    bold2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for BOLD-T1w registration
    name : :obj:`str`, optional
        Workflow name (default: fsl_bbr_wf)

    Inputs
    ------
    in_file
        Reference BOLD image to be registered
    t1w_brain
        Skull-stripped T1-weighted structural image
    t1w_dseg
        FAST segmentation of ``t1w_brain``
    fsnative2t1w_xfm
        Unused (see :py:func:`~fmriprep.workflows.bold.registration.init_bbreg_wf`)
    subjects_dir
        Unused (see :py:func:`~fmriprep.workflows.bold.registration.init_bbreg_wf`)
    subject_id
        Unused (see :py:func:`~fmriprep.workflows.bold.registration.init_bbreg_wf`)

    Outputs
    -------
    itk_bold_to_t1
        Affine transform from ``ref_bold_brain`` to T1w space (ITK format)
    itk_t1_to_bold
        Affine transform from T1 space to BOLD space (ITK format)
    out_report
        Reportlet for assessing registration quality
    fallback
        Boolean indicating whether BBR was rejected (rigid FLIRT registration returned)

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The BOLD reference was then co-registered to the T1w reference using
`flirt` [FSL {fsl_ver}, @flirt] with the boundary-based registration [@bbr]
cost-function.
Co-registration was configured with nine degrees of freedom to account
for distortions remaining in the BOLD reference.
""".format(fsl_ver=FLIRTRPT().version or '<ver>')

    inputnode = pe.Node(
        niu.IdentityInterface([
            'in_file',
            'fsnative2t1w_xfm', 'subjects_dir', 'subject_id',  # BBRegister
            't1w_dseg', 't1w_brain']),  # FLIRT BBR
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['itk_bold_to_t1', 'itk_t1_to_bold', 'out_report', 'fallback']),
        name='outputnode')

    wm_mask = pe.Node(niu.Function(function=extract_wm), name='wm_mask')
    flt_bbr_init = pe.Node(FLIRTRPT(dof=6, generate_report=not use_bbr,
                                    uses_qform=True), name='flt_bbr_init')

    invt_bbr = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='invt_bbr',
                       mem_gb=DEFAULT_MEMORY_MIN_GB)

    #  BOLD to T1 transform matrix is from fsl, using c3 tools to convert to
    #  something ANTs will like.
    fsl2itk_fwd = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_fwd', mem_gb=DEFAULT_MEMORY_MIN_GB)
    fsl2itk_inv = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                          name='fsl2itk_inv', mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, flt_bbr_init, [('in_file', 'in_file'),
                                   ('t1w_brain', 'reference')]),
        (inputnode, fsl2itk_fwd, [('t1w_brain', 'reference_file'),
                                  ('in_file', 'source_file')]),
        (inputnode, fsl2itk_inv, [('in_file', 'reference_file'),
                                  ('t1w_brain', 'source_file')]),
        (invt_bbr, fsl2itk_inv, [('out_file', 'transform_file')]),
        (fsl2itk_fwd, outputnode, [('itk_transform', 'itk_bold_to_t1')]),
        (fsl2itk_inv, outputnode, [('itk_transform', 'itk_t1_to_bold')]),
    ])

    # Short-circuit workflow building, use rigid registration
    if use_bbr is False:
        workflow.connect([
            (flt_bbr_init, invt_bbr, [('out_matrix_file', 'in_file')]),
            (flt_bbr_init, fsl2itk_fwd, [('out_matrix_file', 'transform_file')]),
            (flt_bbr_init, outputnode, [('out_report', 'out_report')]),
        ])
        outputnode.inputs.fallback = True

        return workflow

    flt_bbr = pe.Node(
        FLIRTRPT(cost_func='bbr', dof=bold2t1w_dof, generate_report=True),
        name='flt_bbr')

    FSLDIR = os.getenv('FSLDIR')
    if FSLDIR:
        flt_bbr.inputs.schedule = op.join(FSLDIR, 'etc/flirtsch/bbr.sch')
    else:
        # Should mostly be hit while building docs
        LOGGER.warning("FSLDIR unset - using packaged BBR schedule")
        flt_bbr.inputs.schedule = pkgr.resource_filename('fmriprep', 'data/flirtsch/bbr.sch')

    workflow.connect([
        (inputnode, wm_mask, [('t1w_dseg', 'in_seg')]),
        (inputnode, flt_bbr, [('in_file', 'in_file'),
                              ('t1w_brain', 'reference')]),
        (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (wm_mask, flt_bbr, [('out', 'wm_seg')]),
    ])

    # Short-circuit workflow building, use boundary-based registration
    if use_bbr is True:
        workflow.connect([
            (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
            (flt_bbr, fsl2itk_fwd, [('out_matrix_file', 'transform_file')]),
            (flt_bbr, outputnode, [('out_report', 'out_report')]),
        ])
        outputnode.inputs.fallback = False

        return workflow

    transforms = pe.Node(niu.Merge(2), run_without_submitting=True, name='transforms')
    reports = pe.Node(niu.Merge(2), run_without_submitting=True, name='reports')

    compare_transforms = pe.Node(niu.Function(function=compare_xforms), name='compare_transforms')

    select_transform = pe.Node(niu.Select(), run_without_submitting=True, name='select_transform')
    select_report = pe.Node(niu.Select(), run_without_submitting=True, name='select_report')

    fsl_to_lta = pe.MapNode(LTAConvert(out_lta=True), iterfield=['in_fsl'],
                            name='fsl_to_lta')

    workflow.connect([
        (flt_bbr, transforms, [('out_matrix_file', 'in1')]),
        (flt_bbr_init, transforms, [('out_matrix_file', 'in2')]),
        # Convert FSL transforms to LTA (RAS2RAS) transforms and compare
        (inputnode, fsl_to_lta, [('in_file', 'source_file'),
                                 ('t1w_brain', 'target_file')]),
        (transforms, fsl_to_lta, [('out', 'in_fsl')]),
        (fsl_to_lta, compare_transforms, [('out_lta', 'lta_list')]),
        (compare_transforms, outputnode, [('out', 'fallback')]),
        # Select output transform
        (transforms, select_transform, [('out', 'inlist')]),
        (compare_transforms, select_transform, [('out', 'index')]),
        (select_transform, invt_bbr, [('out', 'in_file')]),
        (select_transform, fsl2itk_fwd, [('out', 'transform_file')]),
        (flt_bbr, reports, [('out_report', 'in1')]),
        (flt_bbr_init, reports, [('out_report', 'in2')]),
        (reports, select_report, [('out', 'inlist')]),
        (compare_transforms, select_report, [('out', 'index')]),
        (select_report, outputnode, [('out', 'out_report')]),
    ])

    return workflow


def compare_xforms(lta_list, norm_threshold=15):
    """
    Computes a normalized displacement between two affine transforms as the
    maximum overall displacement of the midpoints of the faces of a cube, when
    each transform is applied to the cube.
    This combines displacement resulting from scaling, translation and rotation.

    Although the norm is in mm, in a scaling context, it is not necessarily
    equivalent to that distance in translation.

    We choose a default threshold of 15mm as a rough heuristic.
    Normalized displacement above 20mm showed clear signs of distortion, while
    "good" BBR refinements were frequently below 10mm displaced from the rigid
    transform.
    The 10-20mm range was more ambiguous, and 15mm chosen as a compromise.
    This is open to revisiting in either direction.

    See discussion in
    `GitHub issue #681`_ <https://github.com/poldracklab/fmriprep/issues/681>`_
    and the `underlying implementation
    <https://github.com/nipy/nipype/blob/56b7c81eedeeae884ba47c80096a5f66bd9f8116/nipype/algorithms/rapidart.py#L108-L159>`_.

    Parameters
    ----------

      lta_list : :obj:`list` or :obj:`tuple` of :obj:`str`
          the two given affines in LTA format
      norm_threshold : :obj:`float`
          the upper bound limit to the normalized displacement caused by the
          second transform relative to the first (default: `15`)

    """
    from niworkflows.interfaces.surf import load_transform
    from nipype.algorithms.rapidart import _calc_norm_affine

    bbr_affine = load_transform(lta_list[0])
    fallback_affine = load_transform(lta_list[1])

    norm, _ = _calc_norm_affine([fallback_affine, bbr_affine], use_differences=True)

    return norm[1] > norm_threshold
