#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

.. _sdc_estimation :

Fieldmap estimation and unwarping workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. automodule:: fmriprep.workflows.fieldmap.base
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fmriprep.workflows.fieldmap.fmap
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fmriprep.workflows.fieldmap.phdiff
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fmriprep.workflows.fieldmap.pepolar
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fmriprep.workflows.fieldmap.syn
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fmriprep.workflows.fieldmap.unwarp
    :members:
    :undoc-members:
    :show-inheritance:


"""

from .base import init_sdc_wf
from .unwarp import init_sdc_unwarp_wf, init_fmap_unwarp_report_wf
from .pepolar import init_pepolar_unwarp_wf
from .syn import init_syn_sdc_wf
