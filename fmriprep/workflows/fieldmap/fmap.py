#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Direct B0 mapping sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the fieldmap is directly measured with a prescribed sequence (such as
:abbr:`SE (spiral echo)`), we only need to calculate the corresponding B-Spline
coefficients to adapt the fieldmap to the TOPUP tool.
This procedure is described with more detail `here <https://cni.stanford.edu/\
wiki/GE_Processing#Fieldmaps>`_.

This corresponds to the section 8.9.3 --fieldmap image (and one magnitude image)--
of the BIDS specification.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import ants

from nipype.interfaces import fsl
from niworkflows.interfaces.masks import BETRPT
from fmriprep.interfaces import IntraModalMerge, CopyHeader
from fmriprep.interfaces.fmap import FieldEnhance

def fmap_workflow(name='FMAP_fmap', settings=None):
    """
    Fieldmap workflow - when we have a sequence that directly measures the fieldmap
    we just need to mask it (using the corresponding magnitude image) to remove the
    noise in the surrounding air region, and ensure that units are Hz.

    .. workflow ::

        from fmriprep.workflows.fieldmap.fmap import fmap_workflow
        wf = fmap_workflow()

    """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['input_images']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['fmap', 'fmap_ref', 'fmap_mask']),
                         name='outputnode')

    sortfmaps = pe.Node(niu.Function(function=_sort_fmaps,
                                     input_names=['input_images'],
                                     output_names=['magnitude', 'fieldmap']),
                        name='SortFmaps')

    # Merge input magnitude images
    magmrg = pe.Node(IntraModalMerge(), name='MagnitudeFuse')
    # Merge input fieldmap images
    fmapmrg = pe.Node(IntraModalMerge(zero_based_avg=False, hmc=False),
                      name='FieldmapFuse')

    # de-gradient the fields ("bias/illumination artifact")
    n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='MagnitudeBias')
    cphdr = pe.Node(CopyHeader(), name='FixHDR')
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                  name='MagnitudeBET')
    fmapenh = pe.Node(FieldEnhance(
        # despike_threshold=1.0, mask_erode=1),
        despike=False, njobs=settings.get('ants_nthreads', 4)), name='FieldmapMassage')

    workflow.connect([
        (inputnode, sortfmaps, [('input_images', 'input_images')]),
        (sortfmaps, magmrg, [('magnitude', 'in_files')]),
        (magmrg, n4, [('out_file', 'input_image')]),
        (n4, cphdr, [('output_image', 'in_file')]),
        (magmrg, cphdr, [('out_file', 'hdr_file')]),
        (cphdr, bet, [('out_file', 'in_file')]),
        (sortfmaps, fmapmrg, [('fieldmap', 'in_files')]),
        (bet, fmapenh, [('out_file', 'in_mask')]),
        (fmapmrg, fmapenh, [('out_file', 'in_file')]),
        (fmapenh, outputnode, [('out_file', 'fmap')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'fmap_ref')])
    ])
    return workflow

def _sort_fmaps(input_images):
    ''' just a little data massaging'''
    return (sorted([fname for fname in input_images if 'magnitude' in fname]),
            sorted([fname for fname in input_images if 'fieldmap' in fname]))
