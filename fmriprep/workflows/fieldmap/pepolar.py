#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-Encoding POLARity B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:abbr:`PE-polar (Phase-Encoding POLARity)` is the name coined by GE to the family of
methods to estimate the inhomogeneity of field B0 inside the scanner by using two
acquisitions with different (generally opposed) phase-encoding (PE) directions.
For more information, please see `this description <https://cni.stanford.edu/wiki/Data_Processing\
#Gradient-reversal_Unwarping_.28.27pepolar.27.29>`_

This corresponds to the section 8.9.4 --multiple phase encoded directions (topup)--
of the BIDS specification.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from pkg_resources import resource_filename as pkgrf
from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from nipype.pipeline import engine as pe
from niworkflows.interfaces.masks import BETRPT
from fmriprep.interfaces.topup import TopupInputs

# pylint: disable=R0914
def pepolar_workflow(name='FMAP_pepolar', settings=None):
    """
    Estimates the fieldmap using TOPUP on series of at least two images
    acquired with varying :abbr:`PE (phase encoding)` direction.
    Generally, the images are :abbr:`SE (Spin-Echo)` and the
    :abbr:`PE (phase encoding)` directions are opposed (they can also
    be orthogonal).

    .. workflow ::

        from fmriprep.workflows.fieldmap.pepolar import pepolar_workflow
        wf = pepolar_workflow()


    Outputs::

      outputnode.fmap_ref - The average magnitude image, skull-stripped
      outputnode.fmap_mask - The brain mask applied to the fieldmap
      outputnode.fmap - The estimated fieldmap in Hz

    """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['input_images']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_mask', 'fmap_ref']), name='outputnode')

    sortfmaps = pe.Node(TopupInputs(), name='TOPUP_inputs')

    # Run topup to estimate field distortions, do not estimate movement
    # since it is done in hmc_se
    topup = pe.Node(fsl.TOPUP(config=pkgrf('fmriprep.data', 'sbref2sbref.cnf')),
                    name='TOPUP_estimate')

    # Use the least-squares method to correct the dropout of the SE images
    unwarp_mag = pe.Node(fsl.ApplyTOPUP(method='lsr'), name='TOPUP_apply')

    # Remove bias
    inu_n4 = pe.Node(N4BiasFieldCorrection(dimension=3), name='SE_bias')

    # Skull strip corrected SE image to get reference brain and mask
    mag_bet = pe.Node(BETRPT(mask=True, robust=True), name='SE_brain')

    workflow.connect([
        (inputnode, sortfmaps, [('input_images', 'in_files')]),
        (sortfmaps, topup, [('out_file', 'in_file'),
                            ('out_encfile', 'encoding_file')]),
        (topup, unwarp_mag, [('out_fieldcoef', 'in_topup_fieldcoef'),
                             ('out_movpar', 'in_topup_movpar')]),
        (sortfmaps, unwarp_mag, [('out_filelist', 'in_files'),
                                 ('out_encfile', 'encoding_file')]),
        (unwarp_mag, inu_n4, [('out_corrected', 'input_image')]),
        (inu_n4, mag_bet, [('output_image', 'in_file')]),
        (topup, outputnode, [('out_field', 'fmap')]),
        (mag_bet, outputnode, [('out_file', 'fmap_ref'),
                               ('mask_file', 'fmap_mask')])
    ])
    return workflow


def _sort_fmaps(input_images):
    ''' just a little data massaging'''
    from fmriprep.workflows.fieldmap import PEPOLAR_MODALITIES

    result = []
    for mod in PEPOLAR_MODALITIES:
        result += sorted([fname for fname in input_images if mod in fname])
    return result
