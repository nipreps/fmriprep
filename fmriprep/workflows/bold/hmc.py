# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Head-Motion Estimation and Correction (HMC) of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_hmc_wf

"""

from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import utility as niu, fsl
from niworkflows.interfaces import NormalizeMotionParams
from ...interfaces import MCFLIRT2ITK

DEFAULT_MEMORY_MIN_GB = 0.01


# pylint: disable=R0914
def init_bold_hmc_wf(mem_gb, omp_nthreads, name='bold_hmc_wf'):
    """
    This workflow performs :abbr:`HMC (head motion correction)` over the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_hmc_wf
        wf = init_bold_hmc_wf(
            mem_gb=3,
            omp_nthreads=1)

    **Parameters**

        mem_gb : float
            Size of BOLD file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_hmc_wf``)

    **Inputs**

        bold_file
            BOLD series NIfTI file
        raw_ref_image
            Reference image to which BOLD series is motion corrected

    **Outputs**

        bold_split
            Individual 3D volumes, not motion corrected
        xforms
            ITKTransform file aligning each volume to ``ref_image``
        movpar_file
            MCFLIRT motion parameters, normalized to SPM format (X, Y, Z, Rx, Ry, Rz)

    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file', 'raw_ref_image']),
                        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_split', 'xforms', 'movpar_file']),
        name='outputnode')

    # Head motion correction (hmc)
    mcflirt = pe.Node(fsl.MCFLIRT(save_mats=True, save_plots=True),
                      name='mcflirt', mem_gb=mem_gb * 3)

    fsl2itk = pe.Node(MCFLIRT2ITK(), name='fsl2itk',
                      mem_gb=0.05, n_procs=omp_nthreads)

    normalize_motion = pe.Node(NormalizeMotionParams(format='FSL'),
                               name="normalize_motion",
                               mem_gb=DEFAULT_MEMORY_MIN_GB)

    split = pe.Node(fsl.Split(dimension='t'), name='split',
                    mem_gb=mem_gb * 3)

    workflow.connect([
        (inputnode, split, [('bold_file', 'in_file')]),
        (inputnode, mcflirt, [('raw_ref_image', 'ref_file'),
                              ('bold_file', 'in_file')]),
        (inputnode, fsl2itk, [('raw_ref_image', 'in_source'),
                              ('raw_ref_image', 'in_reference')]),
        (mcflirt, fsl2itk, [('mat_file', 'in_files')]),
        (mcflirt, normalize_motion, [('par_file', 'in_file')]),
        (fsl2itk, outputnode, [('out_file', 'xforms')]),
        (normalize_motion, outputnode, [('out_file', 'movpar_file')]),
        (split, outputnode, [('out_files', 'bold_split')]),
    ])

    return workflow
