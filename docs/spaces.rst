.. include:: links.rst

.. _output-spaces:

Defining standard and nonstandard spaces where data will be resampled
=====================================================================
The command line interface of *fMRIPrep* allows resampling the preprocessed data
onto other output spaces.
That is achieved using the ``--output-spaces`` argument, where standard and
nonstandard spaces can be inserted.

.. important::
   *fMRIPrep* will reduce the amount of output spaces to just spaces listed in ``--output-spaces``,
   even if other options require resampling the preprocessed data into intermediary spaces.


.. _TemplateFlow:

*TemplateFlow*
""""""""""""""
*TemplateFlow* is a software library and a repository of neuroimaging templates
that allows end-user applications such as *fMRIPrep* to flexibly query and pull
template and atlas information.
In other words, *TemplateFlow* enables *fMRIPrep* to access a wide range
of templates (and also custom templates, see below).
Therefore, *TemplateFlow* is central to define *fMRIPrep*'s interface regarding
template and atlas prior-knowledge.
For more general information about *TemplateFlow*, visit
`TemplateFlow.org <https://www.templateflow.org>`__.


Standard spaces
"""""""""""""""
When using *fMRIPrep* in a workflow that will investigate effects that span across
analytical groupings, neuroimagers typically resample their data on to a standard,
stereotactic coordinate system.
The most extended standard space for fMRI analyses is generally referred to MNI.
For instance, to instruct *fMRIPrep* to use the MNI template brain distributed with
FSL as coordinate reference the option will read as follows: ``--output-spaces MNI152NLin6Asym``.
By default, *fMRIPrep* uses ``MNI152NLin2009cAsym`` as spatial-standardization reference.
Valid template identifiers (``MNI152NLin6Asym``, ``MNI152NLin2009cAsym``, etc.) come from
the `TemplateFlow repository <https://github.com/templateflow/templateflow>`__.

Therefore, *fMRIPrep* will run nonlinear registration processes against the template
T1w image corresponding to all the standard spaces supplied with the argument
``--output-spaces``.
By default, *fMRIPrep* will resample the preprocessed data on those spaces (labeling the
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
This is equivalent to saying
``--output-spaces MNI152NLin6Asym:res-2 MNI152NLin2009cAsym:res-native``.

.. danger::

   Please remember that the ``resolution`` entity of *TemplateFlow* is an **index**,
   and therefore, ``res-2`` does not necessarily mean 2mm\ :sup:`3` - although, it
   coincidentally does in the example above.
   However, it may not be the case.
   For instance, ``MNI152NLin6Asym:res-3`` contains a template with
   isotropic voxels of 0.5mm\ :sup:`3`.

Other possible modifiers are, for instance, the ``cohort`` selector.
For instance, ``--output-spaces MNIPediatricAsym:res-1:cohort-2`` selects
the resolution ``1`` of ``cohort-2`` which, for the ``MNIPediatricAsym``
template, corresponds to the `prepuberty phase
<https://github.com/templateflow/tpl-MNIPediatricAsym/blob/bcf77616f547f327ee53c01dadf689ab6518a097/template_description.json#L22-L26>`__
(4.5--8.5 years old).

Space modifiers such as ``res`` are combinatorial:
``--output-spaces MNIPediatricAsym:cohort-1:cohort-2:res-native:res-1`` will
generate conversions for the following combinations:

* cohort ``1`` and "native" resolution (meaning, the original BOLD resolution),
* cohort ``1`` and resolution ``1`` of the template,
* cohort ``2`` and "native" resolution (meaning, the original BOLD resolution), and
* cohort ``2`` and resolution ``1`` of the template.

Please mind that the selected resolutions specified must exist within TemplateFlow.

When specifying surface spaces (e.g., ``fsaverage``), the legacy identifiers from
FreeSurfer will be supported (e.g., ``fsaverage5``) although the use of the density
modifier would be preferred (i.e., ``fsaverage:den-10k`` for ``fsaverage5``).

Custom standard spaces
""""""""""""""""""""""
To make your custom templates visible by *fMRIPrep*, and usable via
the ``--output-spaces`` argument, please store your template under
*TemplateFlow*'s home directory.
The default *TemplateFlow*'s home directory is ``$HOME/.cache/templateflow``
and that can path can be arbitrarily changed by setting
the ``$TEMPLATEFLOW_HOME`` environment variable.
A minimal example of the necessary files for a template called
``MyCustom`` (and therefore callable via, e.g., ``--output-spaces MyCustom``)
follows::

  $TEMPLATEFLOW_HOME/
      tpl-MyCustom/
          template_description.json
          tpl-MyCustom_res-1_T1w.nii.gz
          tpl-MyCustom_res-1_desc-brain_mask.nii.gz
          tpl-MyCustom_res-2_T1w.nii.gz
          tpl-MyCustom_res-2_desc-brain_mask.nii.gz

For further information about how custom templates must be organized and
corresponding naming, please check `the TemplateFlow tutorials
<https://www.templateflow.org/python-client/tutorials.html>`__.

Nonstandard spaces
""""""""""""""""""
Additionally, ``--output-spaces`` accepts identifiers of spatial references
that do not generate *standardized* coordinate spaces:

* ``T1w`` or ``anat``: data are resampled into the individual's anatomical
  reference generated with the T1w and T2w images available within the
  BIDS structure.
* ``fsnative``: similarly to the ``anat`` space for volumetric references,
  including the ``fsnative`` space will instruct *fMRIPrep* to sample the
  original BOLD data onto FreeSurfer's reconstructed surfaces for this
  individual.
* ``func``, ``bold``, ``run``, ``boldref`` or ``sbref`` can be used to
  generate BOLD data in their original grid, after slice-timing,
  head-motion, and susceptibility-distortion corrections.
  These keywords are experimental, and expected to change because
  **additional nonstandard spaces** are currently being discussed
  `here <https://github.com/nipreps/fmriprep/issues/1604>`__.

Modifiers are not allowed when providing nonstandard spaces.

Preprocessing blocks depending on standard templates
""""""""""""""""""""""""""""""""""""""""""""""""""""
Some modules of the pipeline (e.g., the generation of HCP compatible
*grayordinates* files, or the *fieldmap-less* distortion correction)
operate in specific template spaces.
When selecting those modules to be included (using any of the following flags:
``--cifti-outputs``, ``--use-syn-sdc``) will modify the list of
*internal* spaces to include the space identifiers they require, should the
identifier not be found within the ``--output-spaces`` list already.
In other words, running *fMRIPrep* with ``--output-spaces MNI152NLin6Asym:res-2
--use-syn-sdc`` will expand the list of resampling spaces to be
``MNI152NLin6Asym:res-2 MNI152NLin2009cAsym``.
However, these spaces that are added implicitly will not be saved to
the derivatives directory.
