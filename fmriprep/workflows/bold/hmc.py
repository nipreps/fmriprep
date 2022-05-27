# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Head-Motion Estimation and Correction (HMC) of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_hmc_wf

"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl

from ...config import DEFAULT_MEMORY_MIN_GB


def init_bold_hmc_wf(mem_gb, omp_nthreads, name='bold_hmc_wf'):
    """
    Build a workflow to estimate head-motion parameters.

    This workflow estimates the motion parameters to perform
    :abbr:`HMC (head motion correction)` over the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_hmc_wf
            wf = init_bold_hmc_wf(
                mem_gb=3,
                omp_nthreads=1)

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``bold_hmc_wf``)

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    raw_ref_image
        Reference image to which BOLD series is motion corrected

    Outputs
    -------
    xforms
        ITKTransform file aligning each volume to ``ref_image``
    movpar_file
        MCFLIRT motion parameters, normalized to SPM format (X, Y, Z, Rx, Ry, Rz)
    rms_file
        Framewise displacement as measured by ``fsl_motion_outliers`` [Jenkinson2002]_.

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.confounds import NormalizeMotionParams
    from niworkflows.interfaces.itk import MCFLIRT2ITK

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Head-motion parameters with respect to the BOLD reference
(transformation matrices, and six corresponding rotation and translation
parameters) are estimated before any spatiotemporal filtering using
`mcflirt` [FSL {fsl_ver}, @mcflirt].
""".format(fsl_ver=fsl.Info().version() or '<ver>')

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'raw_ref_image']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['xforms', 'movpar_file', 'rmsd_file']),
        name='outputnode')

    # Head motion correction (hmc)
    mcflirt = pe.Node(
        fsl.MCFLIRT(save_mats=True, save_plots=True, save_rms=True),
        name='mcflirt', mem_gb=mem_gb * 3)

    fsl2itk = pe.Node(MCFLIRT2ITK(), name='fsl2itk',
                      mem_gb=0.05, n_procs=omp_nthreads)

    normalize_motion = pe.Node(NormalizeMotionParams(format='FSL'),
                               name="normalize_motion",
                               mem_gb=DEFAULT_MEMORY_MIN_GB)

    def _pick_rel(rms_files):
        return rms_files[-1]

    workflow.connect([
        (inputnode, mcflirt, [('raw_ref_image', 'ref_file'),
                              ('bold_file', 'in_file')]),
        (inputnode, fsl2itk, [('raw_ref_image', 'in_source'),
                              ('raw_ref_image', 'in_reference')]),
        (mcflirt, fsl2itk, [('mat_file', 'in_files')]),
        (mcflirt, normalize_motion, [('par_file', 'in_file')]),
        (mcflirt, outputnode, [(('rms_files', _pick_rel), 'rmsd_file')]),
        (fsl2itk, outputnode, [('out_file', 'xforms')]),
        (normalize_motion, outputnode, [('out_file', 'movpar_file')]),
    ])

    return workflow
