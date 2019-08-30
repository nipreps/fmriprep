# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_reference_wf
.. autofunction:: init_enhance_and_skullstrip_bold_wf
.. autofunction:: init_skullstrip_bold_wf

"""
from niworkflows.func.util import (
    init_bold_reference_wf,
    init_enhance_and_skullstrip_bold_wf,
    init_skullstrip_bold_wf
)

__all__ = [
    'init_bold_reference_wf',
    'init_enhance_and_skullstrip_bold_wf',
    'init_skullstrip_bold_wf'
]
