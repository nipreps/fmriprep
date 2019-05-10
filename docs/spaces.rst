

.. _output-spaces:

Defining standard and nonstandard spaces where data will be resampled
=====================================================================

The command line interface of fMRIPrep allows resampling the preprocessed data
onto other output spaces.
That is achieved using the ``--output-spaces`` argument, where standard and
nonstandard spaces can be inserted.

Standard spaces
"""""""""""""""

When using fMRIPrep in a workflow that will investigate effects that span across
analytical groupings, neuroimagers typically resample their data on to a standard,
stereotactic coordinate system.
The most extended standard space for fMRI analyses is generally referred to MNI.
For instance, to instruct fMRIPrep to use the MNI template brain distributed with
FSL as coordinate reference the option will read as follows: ``--output-spaces MNI152NLin6Asym``.
By default, fMRIPrep uses ``MNI152NLin2009cAsym`` as spatial-standardization reference.
Valid template identifiers (``MNI152NLin6Asym``, ``MNI152NLin2009cAsym``, etc.) come from
the `TemplateFlow project <https://github.com/templateflow/templateflow>`__.

Therefore, fMRIPrep will run nonlinear registration processes against the template
T1w image corresponding to all the standard spaces supplied with the argument
``--output-spaces``.
By default, fMRIPrep will resample the preprocessed data on those spaces (labeling the
corresponding outputs with the `space-<template-identifier>` BIDS entity) but keeping
the original resolution of the BOLD data to produce smaller files, more consistent with
the original data gridding.
However, many users will be interested in utilizing a coarse gridding (typically 2mm isotropic)
of the target template.
Such a behavior can be achieved applying modifiers to the template identifier, separated by
a ``:`` character.
For instance, ``--output-spaces MNI152NLin6Asym:res-2 MNI152NLin2009cAsym`` will generate
preprocessed BOLD 4D files on two standard spaces (``MNI152NLin6Asym``,
and ``MNI152NLin2009cAsym``) with the template's 2mm isotropic resolution for
the data on ``MNI152NLin6Asym`` space and the original BOLD resolution
(say, e.g., 2x2x2.5 [mm]) for the case of ``MNI152NLin2009cAsym``.

Other possible modifiers are, for instance, the ``cohort`` selector.
Although currently there is no template in TemplateFlow with several cohorts,
very soon we will integrate pediatric templates, for which ``cohort`` will
function to select the appropriate age range.
Therefore, in upcoming versions of fMRIPrep, it will be possible to run it with
``--output-spaces MNIPediatricAsym:res-2:cohort-2`` where ``cohort-2`` would select
the template instance for the, say, 24-48 months old range.

When specifying surface spaces (e.g., ``fsaverage``), the legacy identifiers from
FreeSurfer will be supported (e.g., ``fsaverage5``) although the use of the density
modifier would be preferred (i.e., ``fsaverage:den-10k`` for ``fsaverage5``).

Custom standard spaces
""""""""""""""""""""""

Although the functionality is not available yet, the interface of the
``--output-spaces`` permits providing paths to custom templates that
follow TemplateFlow's naming conventions
(e.g., ``/path/to/custom/templates/tpl-MyCustom:res-2``).
Following the example, at least the following files
must be found under under ``/path/to/custom/templates/tpl-MyCustom``: ::

  tpl-MyCustom/
      template_description.json
      tpl-MyCustom_res-1_T1w.nii.gz
      tpl-MyCustom_res-1_desc-brain_mask.nii.gz
      tpl-MyCustom_res-2_T1w.nii.gz
      tpl-MyCustom_res-2_desc-brain_mask.nii.gz

Although a more comprehensive coverage of standard files would be advised.

Nonstandard spaces
""""""""""""""""""

Additionally, ``--output-spaces`` accepts identifiers of spatial references
that do not generate *standardized* coordinate spaces:

  * ``T1w`` or ``anat``: data are resampled into the individual's anatomical
    reference generated with the T1w and T2w images available within the
    BIDS structure.
  * ``fsnative``: similarly to the ``anat`` space for volumetric references,
    including the ``fsnative`` space will instruct fMRIPrep to sample the
    original BOLD data onto FreeSurfer's reconstructed surfaces for this
    individual.
  * **Additional nonstandard spaces** which are being discussed
    `here <https://github.com/poldracklab/fmriprep/issues/1604>`__.

Modifiers are not allowed when providing nonstandard spaces.

Preprocessing blocks depending on standard templates
""""""""""""""""""""""""""""""""""""""""""""""""""""

Some modules of the pipeline (e.g., the ICA-AROMA denoising, the generation of
HCP compatible *grayordinates* files, or the *fieldmap-less* distortion correction)
operate in specific template spaces.
When selecting those modules to be included (using any of the following flags:
``--use-aroma``, ``--cifti-outputs``, ``--use-syn-sdc``) will modify the list of
output spaces to include the space identifiers they require, should the
identifier not be found within the ``--output-spaces`` list already.
In other words, running fMRIPrep with ``--output-spaces MNI152NLin6Asym:res-2
--use-syn-sdc`` will expand the list of output spaces to be
``MNI152NLin6Asym:res-2 MNI152NLin2009cAsym``.
