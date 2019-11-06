# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_fieldmapless :

Fieldmap-less estimation (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the absence of direct measurements of fieldmap data, we provide an (experimental)
option to estimate the susceptibility distortion based on the ANTs symmetric
normalization (SyN) technique.
This feature may be enabled, using the ``--use-syn-sdc`` flag, and will only be
applied if fieldmaps are unavailable.

During the evaluation phase, the ``--force-syn`` flag will cause this estimation to
be performed *in addition to* fieldmap-based estimation, to permit the direct
comparison of the results of each technique.
Note that, even if ``--force-syn`` is given, the functional outputs of FMRIPREP will
be corrected using the fieldmap-based estimates.

Feedback will be enthusiastically received.


"""
from sdcflows.workflows.syn import init_syn_sdc_wf

__all__ = ['init_syn_sdc_wf']
