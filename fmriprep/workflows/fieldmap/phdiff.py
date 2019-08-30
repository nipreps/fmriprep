# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_phasediff :

Phase-difference B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The field inhomogeneity inside the scanner (fieldmap) is proportional to the
phase drift between two subsequent :abbr:`GRE (gradient recall echo)`
sequence.


Fieldmap preprocessing workflow for fieldmap data structure
8.9.1 in BIDS 1.0.0: one phase diff and at least one magnitude image

"""
from sdcflows.workflows.phdiff import init_phdiff_wf

__all__ = ['init_phdiff_wf']
