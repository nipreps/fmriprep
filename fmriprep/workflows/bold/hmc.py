# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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

from nipype.interfaces import afni, fsl
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe


def init_bold_hmc_wf(
    mem_gb: float,
    omp_nthreads: int,
    hmc_method: str = 'mcflirt',
    name: str = 'bold_hmc_wf',
):
    """
    Build a workflow to estimate head-motion parameters.

    This workflow estimates the motion parameters to perform
    :abbr:`HMC (head motion correction)` over the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.
    With ``hmc_method='mcflirt'``, FSL's ``mcflirt`` estimates the motion,
    and its matrices are converted to ITK transforms.
    With ``hmc_method='3dvolreg'``, AFNI's ``3dvolreg`` estimates it
    (``-cubic -zpad 1``), and the transforms are built from its
    ``-1Dfile`` motion parameters, cross-checked against ``-1Dmatrix_save``.

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
    hmc_method : :obj:`str`
        Head-motion estimator: ``'mcflirt'`` (default) or ``'3dvolreg'``
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

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.itk import MCFLIRT2ITK

    from ...interfaces.nitransforms import VolregToITK
    from ...interfaces.patches import Volreg

    workflow = Workflow(name=name)
    match hmc_method:
        case 'mcflirt':
            package, version, citation = 'FSL', fsl.Info.version(), '@mcflirt'
        case '3dvolreg':
            package, citation = 'AFNI', '@afni; @volreg'
            version = ''.join(f'{v:02d}' for v in afni.Info.version() or [])
        case _:
            raise ValueError(f"Unknown HMC method: {hmc_method}. Must be 'mcflirt' or '3dvolreg'.")
    workflow.__desc__ = f"""\
Head-motion parameters with respect to the BOLD reference
(transformation matrices, and six corresponding rotation and translation
parameters) are estimated before any spatiotemporal filtering using
`{hmc_method}` [{package} {version or '<ver>'}, {citation}].
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'raw_ref_image']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['xforms']), name='outputnode')

    motion2itk = pe.Node(
        MCFLIRT2ITK() if hmc_method == 'mcflirt' else VolregToITK(),
        name=f'{hmc_method}2itk',
        mem_gb=0.05,
    )

    workflow.connect([
        (inputnode, motion2itk, [('raw_ref_image', 'in_reference')]),
        (motion2itk, outputnode, [('out_file', 'xforms')]),
    ])  # fmt:skip

    if hmc_method == '3dvolreg':
        volreg = pe.Node(
            Volreg(interp='cubic', zpad=1, out_file='NULL'), name='volreg', mem_gb=mem_gb * 3
        )
        workflow.connect([
            (inputnode, volreg, [('raw_ref_image', 'basefile'),
                                 ('bold_file', 'in_file')]),
            (volreg, motion2itk, [('oned_file', 'in_params'),
                                  ('oned_matrix_save', 'in_matrices')]),
        ])  # fmt:skip
    else:
        mcflirt = pe.Node(fsl.MCFLIRT(save_mats=True), name='mcflirt', mem_gb=mem_gb * 3)

        workflow.connect([
            (inputnode, mcflirt, [('raw_ref_image', 'ref_file'),
                                  ('bold_file', 'in_file')]),
            (inputnode, motion2itk, [('raw_ref_image', 'in_source')]),
            (mcflirt, motion2itk, [('mat_file', 'in_files')]),
        ])  # fmt:skip

    return workflow
