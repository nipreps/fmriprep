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
Registration workflows
++++++++++++++++++++++

.. autofunction:: init_pet_reg_wf

"""

import typing as ty

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

AffineDOF = ty.Literal[6, 9, 12]
RegistrationInit = ty.Literal['t1w', 't2w', 'header']


def init_pet_reg_wf(
    *,
    freesurfer: bool,
    use_bbr: bool,
    pet2anat_dof: AffineDOF,
    pet2anat_init: RegistrationInit,
    mem_gb: float,
    omp_nthreads: int,
    name: str = 'pet_reg_wf',
    sloppy: bool = False,
):
    """
    Build a workflow to run same-subject, PET-to-T1w image-registration.

    Calculates the registration between a reference PET image and T1w-space
    using a boundary-based registration (BBR) cost function.
    If FreeSurfer-based preprocessing is enabled, the ``bbregister`` utility
    is used to align the PET images to the reconstructed subject, and the
    resulting transform is adjusted to target the T1 space.
    If FreeSurfer-based preprocessing is disabled, FSL FLIRT is used with the
    BBR cost function to directly target the T1 space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.pet.registration import init_pet_reg_wf
            wf = init_pet_reg_wf(freesurfer=True,
                                  mem_gb=3,
                                  omp_nthreads=1,
                                  use_bbr=True,
                                  pet2anat_dof=9,
                                  pet2anat_init='t2w')

    Parameters
    ----------
    freesurfer : :obj:`bool`
        Enable FreeSurfer functional registration (bbregister)
    use_bbr : :obj:`bool` or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
    pet2anat_dof : 6, 9 or 12
        Degrees-of-freedom for PET-anatomical registration
    pet2anat_init : str, 't1w', 't2w' or 'header'
        If ``'header'``, use header information for initialization of PET and T1 images.
        If ``'t1w'``, align PET to T1w by their centers.
        If ``'t2w'``, align PET to T1w using the T2w as an intermediate.
    mem_gb : :obj:`float`
        Size of PET file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``pet_reg_wf``)

    Inputs
    ------
    ref_pet_brain
        Reference image to which PET series is aligned
        If ``fieldwarp == True``, ``ref_pet_brain`` should be unwarped
    t1w_brain
        Skull-stripped ``t1w_preproc``
    t1w_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    Outputs
    -------
    itk_pet_to_t1
        Affine transform from ``ref_pet_brain`` to T1 space (ITK format)
    itk_t1_to_pet
        Affine transform from T1 space to PET space (ITK format)
    fallback
        Boolean indicating whether BBR was rejected (mri_coreg registration returned)

    See Also
    --------
      * :py:func:`~fmriprep.workflows.pet.registration.init_bbreg_wf`
      * :py:func:`~fmriprep.workflows.pet.registration.init_fsl_bbr_wf`

    """
    from nipype.interfaces.freesurfer import MRICoreg
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.nibabel import ApplyMask
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['ref_pet_brain', 't1w_preproc', 't1w_mask']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['itk_pet_to_t1', 'itk_t1_to_pet']),
        name='outputnode',
    )

    mask_t1w_brain = pe.Node(ApplyMask(), name='mask_t1w_brain')
    mri_coreg = pe.Node(
        MRICoreg(dof=pet2anat_dof, sep=[4], ftol=0.0001, linmintol=0.01),
        name='mri_coreg',
        n_procs=omp_nthreads,
        mem_gb=5,
    )
    convert_xfm = pe.Node(ConcatenateXFMs(inverse=True), name='convert_xfm')

    workflow.connect([
        (inputnode, mask_t1w_brain, [('t1w_preproc', 'in_file'),
                                     ('t1w_mask', 'in_mask')]),
        (inputnode, mri_coreg, [('ref_pet_brain', 'source_file')]),
        (mask_t1w_brain, mri_coreg, [('out_file', 'reference_file')]),
        (mri_coreg, convert_xfm, [('out_lta_file', 'in_xfms')]),
        (convert_xfm, outputnode, [
            ('out_xfm', 'itk_pet_to_t1'),
            ('out_inv', 'itk_t1_to_pet'),
        ]),
    ])  # fmt:skip

    return workflow
