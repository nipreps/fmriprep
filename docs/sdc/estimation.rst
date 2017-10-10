:orphan:

.. _sdc-estimation:

Fieldmap estimation
-------------------

.. automodule:: fmriprep.workflows.fieldmap.fmap
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fmriprep.workflows.fieldmap.phdiff
    :members:
    :undoc-members:
    :show-inheritance:


.. _fieldmapless_estimation:

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

.. autofunction:: fmriprep.workflows.bold.init_nonlinear_sdc_wf
