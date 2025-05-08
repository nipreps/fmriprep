# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

Pre-processing fMRI - BOLD signal workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fmriprep.workflows.pet.base
.. automodule:: fmriprep.workflows.pet.hmc
.. automodule:: fmriprep.workflows.pet.stc
.. automodule:: fmriprep.workflows.pet.registration
.. automodule:: fmriprep.workflows.pet.resampling
.. automodule:: fmriprep.workflows.pet.confounds


"""

from .confounds import init_pet_confs_wf
from .hmc import init_pet_hmc_wf
from .registration import init_pet_reg_wf
from .resampling import init_pet_surf_wf

__all__ = [
    'init_pet_confs_wf',
    'init_pet_hmc_wf',
    'init_pet_reg_wf',
    'init_pet_surf_wf',
]
