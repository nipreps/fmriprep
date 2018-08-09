#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_base :

Automatic selection of the appropriate SDC method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the dataset metadata indicate tha more than one field map acquisition is
``IntendedFor`` (see BIDS Specification section 8.9) the following priority will
be used:

  1. :ref:`sdc_pepolar` (or **blip-up/blip-down**)

  2. :ref:`sdc_direct_b0`

  3. :ref:`sdc_phasediff`

  4. :ref:`sdc_fieldmapless`


Table of behavior (fieldmap use-cases):

=============== =========== ============= ===============
Fieldmaps found ``use_syn`` ``force_syn``     Action
=============== =========== ============= ===============
True            *           True          Fieldmaps + SyN
True            *           False         Fieldmaps
False           *           True          SyN
False           True        False         SyN
False           False       False         HMC only
=============== =========== ============= ===============


"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging

from ...engine import Workflow

# Fieldmap workflows
from .pepolar import init_pepolar_unwarp_wf
from .syn import init_syn_sdc_wf
from .unwarp import init_sdc_unwarp_wf

LOGGER = logging.getLogger('nipype.workflow')
FMAP_PRIORITY = {
    'epi': 0,
    'fieldmap': 1,
    'phasediff': 2,
    'syn': 3
}
DEFAULT_MEMORY_MIN_GB = 0.01


def init_sdc_wf(fmaps, bold_meta, omp_nthreads=1,
                debug=False, fmap_bspline=False, fmap_demean=True):
    """
    This workflow implements the heuristics to choose a
    :abbr:`SDC (susceptibility distortion correction)` strategy.
    When no field map information is present within the BIDS inputs,
    the EXPERIMENTAL "fieldmap-less SyN" can be performed, using
    the ``--use-syn`` argument. When ``--force-syn`` is specified,
    then the "fieldmap-less SyN" is always executed and reported
    despite of other fieldmaps available with higher priority.
    In the latter case (some sort of fieldmap(s) is available and
    ``--force-syn`` is requested), then the :abbr:`SDC (susceptibility
    distortion correction)` method applied is that with the
    highest priority.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap import init_sdc_wf
        wf = init_sdc_wf(
            fmaps=[{
                'type': 'phasediff',
                'phasediff': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_phasediff.nii.gz',
                'magnitude1': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude1.nii.gz',
                'magnitude2': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude2.nii.gz',
            }],
            bold_meta={
                'RepetitionTime': 2.0,
                'SliceTiming': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'PhaseEncodingDirection': 'j',
            },
        )

    **Parameters**

        fmaps : list of pybids dicts
            A list of dictionaries with the available fieldmaps
            (and their metadata using the key ``'metadata'`` for the
            case of *epi* fieldmaps)
        bold_meta : dict
            BIDS metadata dictionary corresponding to the BOLD run
        omp_nthreads : int
            Maximum number of threads an individual process may use
        fmap_bspline : bool
            **Experimental**: Fit B-Spline field using least-squares
        fmap_demean : bool
            Demean voxel-shift map during unwarp
        debug : bool
            Enable debugging outputs

    **Inputs**
        bold_ref
            A BOLD reference calculated at a previous stage
        bold_ref_brain
            Same as above, but brain-masked
        bold_mask
            Brain mask for the BOLD run
        t1_brain
            T1w image, brain-masked, for the fieldmap-less SyN method
        t1_2_mni_reverse_transform
            MNI-to-T1w transform to map prior knowledge to the T1w
            fo the fieldmap-less SyN method
        template : str
            Name of template targeted by ``template`` output space


    **Outputs**
        bold_ref
            An unwarped BOLD reference
        bold_mask
            The corresponding new mask after unwarping
        bold_ref_brain
            Brain-extracted, unwarped BOLD reference
        out_warp
            The deformation field to unwarp the susceptibility distortions
        syn_bold_ref
            If ``--force-syn``, an unwarped BOLD reference with this
            method (for reporting purposes)

    """

    # TODO: To be removed (filter out unsupported fieldmaps):
    fmaps = [fmap for fmap in fmaps if fmap['type'] in FMAP_PRIORITY]

    workflow = Workflow(name='sdc_wf' if fmaps else 'sdc_bypass_wf')
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_ref', 'bold_ref_brain', 'bold_mask',
                't1_brain', 't1_2_mni_reverse_transform', 'template']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_ref', 'bold_mask', 'bold_ref_brain',
                'out_warp', 'syn_bold_ref', 'method']),
        name='outputnode')

    # No fieldmaps - forward inputs to outputs
    if not fmaps:
        outputnode.inputs.method = 'None'
        workflow.connect([
            (inputnode, outputnode, [('bold_ref', 'bold_ref'),
                                     ('bold_mask', 'bold_mask'),
                                     ('bold_ref_brain', 'bold_ref_brain')]),
        ])
        return workflow

    workflow.__postdesc__ = """\
Based on the estimated susceptibility distortion, an
unwarped BOLD reference was calculated for a more accurate
co-registration with the anatomical reference.
"""

    # In case there are multiple fieldmaps prefer EPI
    fmaps.sort(key=lambda fmap: FMAP_PRIORITY[fmap['type']])
    fmap = fmaps[0]

    # PEPOLAR path
    if fmap['type'] == 'epi':
        outputnode.inputs.method = 'PEB/PEPOLAR (phase-encoding based / PE-POLARity)'
        # Get EPI polarities and their metadata
        epi_fmaps = [(fmap_['epi'], fmap_['metadata']["PhaseEncodingDirection"])
                     for fmap_ in fmaps if fmap_['type'] == 'epi']
        sdc_unwarp_wf = init_pepolar_unwarp_wf(
            bold_meta=bold_meta,
            epi_fmaps=epi_fmaps,
            omp_nthreads=omp_nthreads,
            name='pepolar_unwarp_wf')

        workflow.connect([
            (inputnode, sdc_unwarp_wf, [
                ('bold_ref', 'inputnode.in_reference'),
                ('bold_mask', 'inputnode.in_mask'),
                ('bold_ref_brain', 'inputnode.in_reference_brain')]),
        ])

    # FIELDMAP path
    if fmap['type'] in ['fieldmap', 'phasediff']:
        outputnode.inputs.method = 'FMB (%s-based)' % fmap['type']
        # Import specific workflows here, so we don't break everything with one
        # unused workflow.
        if fmap['type'] == 'fieldmap':
            from .fmap import init_fmap_wf
            fmap_estimator_wf = init_fmap_wf(
                omp_nthreads=omp_nthreads,
                fmap_bspline=fmap_bspline)
            # set inputs
            fmap_estimator_wf.inputs.inputnode.fieldmap = fmap['fieldmap']
            fmap_estimator_wf.inputs.inputnode.magnitude = fmap['magnitude']

        if fmap['type'] == 'phasediff':
            from .phdiff import init_phdiff_wf
            fmap_estimator_wf = init_phdiff_wf(omp_nthreads=omp_nthreads)
            # set inputs
            fmap_estimator_wf.inputs.inputnode.phasediff = fmap['phasediff']
            fmap_estimator_wf.inputs.inputnode.magnitude = [
                fmap_ for key, fmap_ in sorted(fmap.items())
                if key.startswith("magnitude")
            ]

        sdc_unwarp_wf = init_sdc_unwarp_wf(
            omp_nthreads=omp_nthreads,
            fmap_demean=fmap_demean,
            debug=debug,
            name='sdc_unwarp_wf')
        sdc_unwarp_wf.inputs.inputnode.metadata = bold_meta

        workflow.connect([
            (inputnode, sdc_unwarp_wf, [
                ('bold_ref', 'inputnode.in_reference'),
                ('bold_ref_brain', 'inputnode.in_reference_brain'),
                ('bold_mask', 'inputnode.in_mask')]),
            (fmap_estimator_wf, sdc_unwarp_wf, [
                ('outputnode.fmap', 'inputnode.fmap'),
                ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
                ('outputnode.fmap_mask', 'inputnode.fmap_mask')]),
        ])

    # FIELDMAP-less path
    if any(fm['type'] == 'syn' for fm in fmaps):
        syn_sdc_wf = init_syn_sdc_wf(
            bold_pe=bold_meta.get('PhaseEncodingDirection', None),
            omp_nthreads=omp_nthreads)

        workflow.connect([
            (inputnode, syn_sdc_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                ('bold_ref', 'inputnode.bold_ref'),
                ('bold_ref_brain', 'inputnode.bold_ref_brain'),
                ('template', 'inputnode.template')]),
        ])

        # XXX Eliminate branch when forcing isn't an option
        if fmap['type'] == 'syn':  # No fieldmaps, but --use-syn
            outputnode.inputs.method = 'FLB ("fieldmap-less", SyN-based)'
            sdc_unwarp_wf = syn_sdc_wf
        else:  # --force-syn was called when other fieldmap was present
            sdc_unwarp_wf.__desc__ = None
            workflow.connect([
                (syn_sdc_wf, outputnode, [
                    ('outputnode.out_reference', 'syn_bold_ref')]),
            ])

    workflow.connect([
        (sdc_unwarp_wf, outputnode, [
            ('outputnode.out_warp', 'out_warp'),
            ('outputnode.out_reference', 'bold_ref'),
            ('outputnode.out_reference_brain', 'bold_ref_brain'),
            ('outputnode.out_mask', 'bold_mask')]),
    ])

    return workflow
