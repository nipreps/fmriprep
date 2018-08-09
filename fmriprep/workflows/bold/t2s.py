# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generate T2* map from multi-echo BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_t2s_wf

"""
from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...engine import Workflow
from ...interfaces.multiecho import (T2SMap, MaskT2SMap)
from .resampling import init_bold_preproc_trans_wf

from .util import init_skullstrip_bold_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


# pylint: disable=R0914
def init_bold_t2s_wf(bold_echos, echo_times, mem_gb, omp_nthreads,
                     name='bold_t2s_wf',
                     use_compression=True,
                     use_fieldwarp=False):
    """
    This workflow performs :abbr:`HMC (head motion correction)`
    on individual echo_files, uses T2SMap to generate a T2* image
    for coregistration instead of mean BOLD EPI.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_t2s_wf
        wf = init_bold_t2s_wf(
            bold_echos=['echo1', 'echo2', 'echo3'],
            echo_times=[13.6, 29.79, 46.59],
            mem_gb=3,
            omp_nthreads=1)

    **Parameters**

        bold_echos
            list of ME-BOLD files
        echo_times
            list of TEs associated with each echo
        mem_gb : float
            Size of BOLD file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_t2s_wf``)
        use_compression : bool
            Save registered BOLD series as ``.nii.gz``
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from BOLD to MNI

    **Inputs**

        name_source
            (one echo of) the original BOLD series NIfTI file
            Used to recover original information lost during processing
        hmc_xforms
            ITKTransform file aligning each volume to ``ref_image``

    **Outputs**

        t2s_map
            the T2* map for the EPI run
        oc_mask
            the skull-stripped optimal combination mask
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_echos', 'name_source', 'hmc_xforms']),
        name='inputnode')
    inputnode.iterables = ('bold_echos', bold_echos)

    outputnode = pe.Node(niu.IdentityInterface(fields=['t2s_map', 'oc_mask']),
                         name='outputnode')

    LOGGER.log(25, 'Generating T2* map.')

    # Apply transforms in 1 shot
    bold_bold_trans_wf = init_bold_preproc_trans_wf(
        mem_gb=mem_gb,
        omp_nthreads=omp_nthreads,
        use_compression=use_compression,
        use_fieldwarp=use_fieldwarp,
        name='bold_bold_trans_wf',
        split_file=True,
        interpolation='NearestNeighbor'
    )

    t2s_map = pe.JoinNode(T2SMap(
        te_list=echo_times), joinsource='inputnode', joinfield=['in_files'],
        name='t2s_map')

    skullstrip_bold_wf = init_skullstrip_bold_wf()

    mask_t2s = pe.Node(MaskT2SMap(), name='mask_t2s')

    workflow.connect([
        (inputnode, bold_bold_trans_wf, [
            ('bold_echos', 'inputnode.bold_file'),
            ('name_source', 'inputnode.name_source'),
            ('hmc_xforms', 'inputnode.hmc_xforms')]),
        (bold_bold_trans_wf, t2s_map, [('outputnode.bold', 'in_files')]),
        (t2s_map, skullstrip_bold_wf, [('opt_comb', 'inputnode.in_file')]),
        (t2s_map, mask_t2s, [('t2s_vol', 'image')]),
        (skullstrip_bold_wf, outputnode, [('outputnode.mask_file', 'oc_mask')]),
        (skullstrip_bold_wf, mask_t2s, [('outputnode.mask_file', 'mask')]),
        (mask_t2s, outputnode, [('masked_t2s', 't2s_map')])
    ])

    return workflow


def _first(inlist):
    return inlist[0][0]
