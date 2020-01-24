# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

.. _sdc_direct_b0 :

Direct B0 mapping sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the fieldmap is directly measured with a prescribed sequence (such as
:abbr:`SE (spiral echo)`), we only need to calculate the corresponding B-Spline
coefficients to adapt the fieldmap to the TOPUP tool.
This procedure is described with more detail `here <https://cni.stanford.edu/\
wiki/GE_Processing#Fieldmaps>`__.

This corresponds to the section 8.9.3 --fieldmap image (and one magnitude image)--
of the BIDS specification.

"""
from sdcflows.workflows.fmap import init_fmap_wf

__all__ = ['init_fmap_wf']
