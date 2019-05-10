# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Anatomical reference preprocessing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_anat_preproc_wf

Surface preprocessing
+++++++++++++++++++++

``fmriprep`` uses FreeSurfer_ to reconstruct surfaces from T1w/T2w
structural images.

.. autofunction:: init_surface_recon_wf
.. autofunction:: init_autorecon_resume_wf
.. autofunction:: init_gifti_surface_wf

"""
from smriprep.workflows.anatomical import (
    init_anat_preproc_wf,
)
from smriprep.workflows.surfaces import (
    init_surface_recon_wf,
    init_autorecon_resume_wf,
    init_gifti_surface_wf,
)

__all__ = [
    'init_anat_preproc_wf',
    'init_surface_recon_wf',
    'init_autorecon_resume_wf',
    'init_gifti_surface_wf',
]
