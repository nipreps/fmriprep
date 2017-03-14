#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Preprocessing workflows for :abbr:`SB (single-band)`-reference (SBRef)
images.

"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import ants
from niworkflows.interfaces.masks import ComputeEPIMask

from fmriprep.utils.misc import _first
from fmriprep.interfaces.images import FixAffine, SplitMerge
from fmriprep.interfaces import DerivativesDataSink
from fmriprep.workflows.fieldmap import sdc_unwarp
from fmriprep.interfaces.bids import ReadSidecarJSON
from fmriprep.interfaces.hmc import MotionCorrection

def sbref_preprocess(name='SBrefPreprocessing', settings=None):
    """
    SBref processing workflow

    This workflow runs a preliminary :abbr:`HMC (head motion correction)` on
    the input SBRefs (it can be a list of them).

    After :abbr:`HMC (head motion correction)`, the ``sdc_unwarp`` workflow
    will run :abbr:`SDC (susceptibility distortion correction)` followed by
    a second iteration of :abbr:`HMC (head motion correction)`.

    The output image is then removed the bias and skull-stripped.

    Input fields:
    ~~~~~~~~~~~~~

      inputnode.sbref - a list of target SBRefs to be processed
      inputnode.fmap - a fieldmap in Hz
      inputnode.fmap_ref - the fieldmap reference (generally, a *magnitude* image or the
                           resulting SE image)
      inputnode.fmap_mask - a brain mask in fieldmap-space

    Output fields:
    ~~~~~~~~~~~~~~

      outputnode.sbref_unwarped - the in_file after HMC and SDC corrections.
      outputnode.sbref_unwarped_mask - brain mask corresponding to the ``sbref_unwarped``


    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['sbref', 'fmap', 'fmap_ref', 'fmap_mask']
        ),
        name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['sbref_unwarped', 'sbref_unwarped_mask']),
                         name='outputnode')

    # Read metadata
    meta = pe.Node(ReadSidecarJSON(), name='metadata')

    # Remove the scanner affines
    hdr = pe.MapNode(FixAffine(), name='epi_hdr',
                     iterfield=['in_file'])

    # Split EPI
    split = pe.Node(SplitMerge(), name='split_merge')

    # Preliminary head motion correction
    pre_hmc = pe.Node(MotionCorrection(njobs=settings.get('ants_nthreads', 1)),
                      name='pre_hmc')

    # Unwarping
    unwarp = sdc_unwarp(settings=settings)

    # Remove bias field from SBRef for the sake of registrations
    inu = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='SBRefBias')

    # Get rid of skull
    skullstripping = pe.Node(ComputeEPIMask(generate_report=True,
                                            dilation=1), name='SBRefSkullstripping')


    workflow.connect([
        (inputnode, meta, [(('sbref', _first), 'in_file')]),
        (inputnode, hdr, [('sbref', 'in_file')]),
        (hdr, split, [('out_file', 'in_files')]),
        (split, pre_hmc, [('out_merged', 'in_merged'),
                          ('out_split', 'in_split')]),
        (inputnode, unwarp, [(('fmap', _first), 'inputnode.fmap'),
                             (('fmap_ref', _first), 'inputnode.fmap_ref'),
                             (('fmap_mask', _first), 'inputnode.fmap_mask')]),
        (split, unwarp, [('out_merged', 'inputnode.in_merged'),
                          ('out_split', 'inputnode.in_split')]),

        (meta, unwarp, [('out_dict', 'inputnode.in_meta')]),
        (pre_hmc, unwarp, [('out_avg', 'inputnode.in_reference'),
                           ('out_tfm', 'inputnode.in_hmcpar')]),
        (unwarp, inu, [('outputnode.out_mean', 'input_image')]),
        (inu, skullstripping, [('output_image', 'in_file')]),
        (skullstripping, outputnode, [('mask_file', 'sbref_unwarped_mask')]),
        (inu, outputnode, [('output_image', 'sbref_unwarped')])
    ])

    # Hook up reporting and push derivatives
    ds_report = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='brainmask'),
        name='DS_Report'
    )
    datasink = pe.Node(
        DerivativesDataSink(base_directory=settings['output_dir'],
                            suffix='sdc'),
        name='datasink'
    )

    workflow.connect([
        (inputnode, ds_report, [(('sbref', _first), 'source_file')]),
        (inputnode, datasink, [(('sbref', _first), 'source_file')]),
        (skullstripping, ds_report, [('out_report', 'in_file')]),
        (inu, datasink, [('output_image', 'in_file')])
    ])
    return workflow
