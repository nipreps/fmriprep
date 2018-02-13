# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generate T2* map from multi-echo BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_t2s_wf

"""
from niworkflows.nipype import logging
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import utility as niu

from ...interfaces.nilearn import Merge
from ...interfaces.multiecho import T2SMap
from ...interfaces import MultiApplyTransforms

from .util import init_skullstrip_bold_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('workflow')


# pylint: disable=R0914
def init_bold_t2s_wf(echo_times, mem_gb, omp_nthreads, name='bold_t2s_wf'):
    """
    This workflow performs :abbr:`HMC (head motion correction)`
    on individual echo_files, uses T2SMap to generate a T2* image
    for coregistration instead of mean BOLD EPI.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_t2s_wf
        wf = init_bold_t2s_wf(echo_times=[13.6, 29.79, 46.59],
                              mem_gb=3,
                              omp_nthreads=1)

    **Parameters**

        echo_times
            list of TEs associated with each echo
        mem_gb : float
            Size of BOLD file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_t2s_wf``)

    **Inputs**

        xforms
            ITKTransform file aligning each volume to ``ref_image``
        echo_split
            3D volumes of multi-echo BOLD EPI

    **Outputs**

        t2s_map
            the T2* map for the EPI run
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['echo_split', 'xforms']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['t2s_map', 't2s_mask']),
                         name='outputnode')

    LOGGER.log(25, 'Generating T2* map.')

    apply_hmc = pe.Node(
        MultiApplyTransforms(interpolation='MultiLabel', float=True, copy_dtype=True),
        mem_gb=(mem_gb * 3 * omp_nthreads), n_procs=omp_nthreads, name='apply_hmc')

    merge = pe.Node(Merge(compress=True), name='merge', mem_gb=mem_gb)

    t2s_map = pe.JoinNode(T2SMap(te_list=echo_times),
                          joinfield='in_files', joinsource='inputnode',
                          name='t2s_map')

    skullstrip_bold_wf = init_skullstrip_bold_wf()

    workflow.connect([
        (inputnode, apply_hmc, [('hmc_xforms', 'transforms'),
                                ('echo_split', 'input_image')]),
        (apply_hmc, merge, [('out_files', 'in_files')]),
        (merge, t2s_map, [('out_file', 'in_files')]),
        (t2s_map, skullstrip_bold_wf, [('output_image', 'inputnode.in_file')]),
        (skullstrip_bold_wf, outputnode, [('outputnode.skull_stripped_file', 't2s_map'),
                                          ('outputnode.mask_file', 't2s_mask')])
    ])

    return workflow
