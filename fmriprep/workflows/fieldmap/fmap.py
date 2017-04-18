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
from nipype.workflows.dmri.fsl.utils import demean_image, cleanup_edge_pipeline
from fmriprep.interfaces import IntraModalMerge, CopyHeader
from fmriprep.interfaces.bids import DerivativesDataSink
from fmriprep.interfaces.fmap import FieldEnhance
from fmriprep.interfaces.utils import ApplyMask

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
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['magnitude', 'fieldmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['fmap', 'fmap_ref', 'fmap_mask']),
                         name='outputnode')

    # Merge input magnitude images
    magmrg = pe.Node(IntraModalMerge(), name='MagnitudeFuse')
    # Merge input fieldmap images
    fmapmrg = pe.Node(IntraModalMerge(zero_based_avg=False, hmc=False),
                      name='FieldmapFuse')

    # de-gradient the fields ("bias/illumination artifact")
    mag_inu = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='MagnitudeBias')
    cphdr = pe.Node(CopyHeader(), name='FixHDR')
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                  name='MagnitudeBET')
    ds_fmap_mask = pe.Node(
        DerivativesDataSink(base_directory=settings['reportlets_dir'],
                            suffix='fmap_mask'), name='ds_fmap_mask')

    workflow.connect([
        (inputnode, magmrg, [('magnitude', 'in_files')]),
        (inputnode, fmapmrg, [('fieldmap', 'in_files')]),
        (magmrg, mag_inu, [('out_file', 'input_image')]),
        (mag_inu, cphdr, [('output_image', 'in_file')]),
        (magmrg, cphdr, [('out_file', 'hdr_file')]),
        (cphdr, bet, [('out_file', 'in_file')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'fmap_ref')]),
        (inputnode, ds_fmap_mask, [('fieldmap', 'source_file')]),
        (bet, ds_fmap_mask, [('out_report', 'in_file')]),
    ])

    if settings.get('fmap_bspline', False):
        # despike_threshold=1.0, mask_erode=1),
        fmapenh = pe.Node(FieldEnhance(
            unwrap=False, despike=False, njobs=settings.get('ants_nthreads', 4)),
            name='FieldmapMassage')
        fmapenh.interface.num_threads = settings.get('ants_nthreads', 4)
        fmapenh.interface.estimated_memory_gb = 4

        workflow.connect([
            (bet, fmapenh, [('mask_file', 'in_mask'),
                            ('out_file', 'in_magnitude')]),
            (fmapmrg, fmapenh, [('out_file', 'in_file')]),
            (fmapenh, outputnode, [('out_file', 'fmap')]),
        ])

    else:
        torads = pe.Node(niu.Function(
            input_names=['in_file'], output_names=['out_file', 'cutoff_hz'],
            function=_torads), name='PreUnwrap')
        prelude = pe.Node(fsl.PRELUDE(), name='PhaseUnwrap')
        tohz = pe.Node(niu.Function(
            input_names=['in_file', 'cutoff_hz'], output_names=['out_file'],
            function=_tohz), name='PostUnwrap')

        denoise = pe.Node(fsl.SpatialFilter(operation='median', kernel_shape='sphere',
                                            kernel_size=3), name='PhaseDenoise')
        demean = pe.Node(niu.Function(
            input_names=['in_file', 'in_mask'], output_names=['out_file'],
            function=demean_image), name='DemeanFmap')
        cleanup = cleanup_edge_pipeline()

        applymsk = pe.Node(ApplyMask(), name='PhaseMask')

        workflow.connect([
            (bet, prelude, [('mask_file', 'mask_file'),
                            ('out_file', 'magnitude_file')]),
            (fmapmrg, torads, [('out_file', 'in_file')]),
            (torads, tohz, [('cutoff_hz', 'cutoff_hz')]),
            (torads, prelude, [('out_file', 'phase_file')]),
            (prelude, tohz, [('unwrapped_phase_file', 'in_file')]),
            (tohz, denoise, [('out_file', 'in_file')]),
            (denoise, demean, [('out_file', 'in_file')]),
            (demean, cleanup, [('out_file', 'inputnode.in_file')]),
            (bet, cleanup, [('mask_file', 'inputnode.in_mask')]),
            (cleanup, applymsk, [('outputnode.out_file', 'in_file')]),
            (bet, applymsk, [('mask_file', 'in_mask')]),
            (applymsk, outputnode, [('out_file', 'fmap')]),
        ])

    return workflow


def _torads(in_file, out_file=None):
    from math import pi
    import nibabel as nb
    import numpy as np
    from fmriprep.utils.misc import genfname

    if out_file is None:
        out_file = genfname(in_file, suffix='rad')

    fmapnii = nb.load(in_file)
    fmapdata = fmapnii.get_data()
    cutoff = max(abs(fmapdata.min()), fmapdata.max())
    fmapdata *= (pi / cutoff)
    nb.Nifti1Image(fmapdata, fmapnii.affine, fmapnii.header).to_filename(
        out_file)
    return out_file, cutoff

def _tohz(in_file, cutoff_hz, out_file=None):
    from math import pi
    import nibabel as nb
    from fmriprep.utils.misc import genfname
    if out_file is None:
        out_file = genfname(in_file, suffix='hz')

    fmapnii = nb.load(in_file)
    fmapdata = fmapnii.get_data()
    fmapdata *= (cutoff_hz / pi)
    nb.Nifti1Image(fmapdata, fmapnii.affine, fmapnii.header).to_filename(
        out_file)
    return out_file
