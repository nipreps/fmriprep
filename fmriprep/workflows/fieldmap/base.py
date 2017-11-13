#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base fieldmap estimation
------------------------

* Subject can have phase-difference data.
* Subject can have a fieldmap acquisition.

"""
# from niworkflows.nipype import logging


def init_fmap_estimator_wf(fmap_bids, reportlets_dir, omp_nthreads,
                           fmap_bspline):
    """
    This workflow selects the fieldmap estimation data available for the subject and
    returns the estimated fieldmap in mm, along with a corresponding reference image.
    Current implementation applies the first type of fieldmap estimation found in the
    following order of precedence:

      * "Natural" fieldmaps
      * Phase-difference fieldmaps

    Outputs:

        fmap
          The estimated fieldmap itself IN UNITS OF Hz.
        fmap_ref
          the anatomical reference for the fieldmap (magnitude image, corrected SEm, etc.)
        fmap_mask
          a brain mask for the fieldmap


    """

    # pybids type options: (phase1|phase2|phasediff|epi|fieldmap)
    # https://github.com/INCF/pybids/blob/213c425d8ee820f4b7a7ae96e447a4193da2f359/bids/grabbids/bids_layout.py#L63
    if fmap_bids['type'] == 'fieldmap':
        from .fmap import init_fmap_wf
        fmap_wf = init_fmap_wf(reportlets_dir=reportlets_dir,
                               omp_nthreads=omp_nthreads,
                               fmap_bspline=fmap_bspline)
        # set inputs
        fmap_wf.inputs.inputnode.fieldmap = fmap_bids['fieldmap']
        fmap_wf.inputs.inputnode.magnitude = fmap_bids['magnitude']
        return fmap_wf

    if fmap_bids['type'] == 'phasediff':
        from .phdiff import init_phdiff_wf
        phdiff_wf = init_phdiff_wf(reportlets_dir=reportlets_dir,
                                   omp_nthreads=omp_nthreads)
        # set inputs
        phdiff_wf.inputs.inputnode.phasediff = fmap_bids['phasediff']
        phdiff_wf.inputs.inputnode.magnitude = [
            fmap_ for key, fmap_ in sorted(fmap_bids.items())
            if key.startswith("magnitude")
        ]
        return phdiff_wf

    if fmap_bids['type'] in ['phase1', 'phase2']:
        raise NotImplementedError


def init_fmap_unwarp_report_wf(reportlets_dir, name='fmap_unwarp_report_wf'):
    """
    This workflow generates and saves a reportlet showing the effect of fieldmap
    unwarping a BOLD image.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.base import init_fmap_unwarp_report_wf
        wf = init_fmap_unwarp_report_wf(reportlets_dir='.')

    **Parameters**

        reportlets_dir : str
            Directory in which to save reportlets
        name : str, optional
            Workflow name (default: fmap_unwarp_report_wf)

    **Inputs**

        in_pre
            Reference image, before unwarping
        in_post
            Reference image, after unwarping
        in_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        in_xfm
            Affine transform from T1 space to BOLD space (ITK format)
        name_source
            BOLD series NIfTI file
            Used to recover original information lost during processing

    """

    from niworkflows.nipype.pipeline import engine as pe
    from niworkflows.nipype.interfaces import utility as niu
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    from niworkflows.interfaces import SimpleBeforeAfter
    from ...interfaces.images import extract_wm
    from ...interfaces import DerivativesDataSink

    DEFAULT_MEMORY_MIN_GB = 0.01

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_pre', 'in_post', 'in_seg', 'in_xfm',
                'name_source']), name='inputnode')

    map_seg = pe.Node(ApplyTransforms(
        dimension=3, float=True, interpolation='NearestNeighbor'),
        name='map_seg', mem_gb=0.3)

    sel_wm = pe.Node(niu.Function(function=extract_wm), name='sel_wm',
                     mem_gb=DEFAULT_MEMORY_MIN_GB)

    bold_rpt = pe.Node(SimpleBeforeAfter(), name='bold_rpt',
                       mem_gb=0.1)
    bold_rpt_ds = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir,
                            suffix='variant-hmcsdc_preproc'), name='bold_rpt_ds',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True
    )
    workflow.connect([
        (inputnode, bold_rpt, [('in_post', 'after'),
                               ('in_pre', 'before')]),
        (inputnode, bold_rpt_ds, [('name_source', 'source_file')]),
        (bold_rpt, bold_rpt_ds, [('out_report', 'in_file')]),
        (inputnode, map_seg, [('in_post', 'reference_image'),
                              ('in_seg', 'input_image'),
                              ('in_xfm', 'transforms')]),
        (map_seg, sel_wm, [('output_image', 'in_seg')]),
        (sel_wm, bold_rpt, [('out', 'wm_seg')]),
    ])

    return workflow
