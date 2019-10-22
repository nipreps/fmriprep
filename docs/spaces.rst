.. include:: links.rst

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
  * ``func``, ``bold``, ``run``, ``boldref`` or ``sbref`` can be used to
    generate BOLD data in their original grid, after slice-timing,
    head-motion, and susceptibility-distortion corrections.
    These keywords are experimental, and expected to change because
    **additional nonstandard spaces** are currently being discussed
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

.. _TemplateFlow:

*TemplateFlow*
""""""""""""""
Group inference and reporting of neuroimaging studies require that individual's
features are spatially aligned into a common frame where their location can be
called *standard*.
To that end, a multiplicity of brain templates with anatomical annotations
(i.e., atlases) have been published.
However, a centralized resource that allows programmatic access to templates
was lacking.
*TemplateFlow* is a modular, version-controlled resource that allows researchers
to use templates "off-the-shelf" and share new ones.

In addition to the repository from which neuroimaging templates are redistributed,
*TemplateFlow* also comprehends a Python client tool to access them programmatically
when used as a library by other software, or interactively by humans.
Therefore *TemplateFlow* is the software module that allows *fMRIPrep* to flexibly
change, and dynamically pull down, new standardized template information.

.. _tf_no_internet:

**How do you use TemplateFlow in the absence of access to the Internet?**.
This is a fairly common situation in :abbr:`HPCs (high-performance computing)`
systems, where the so-called login nodes have access to the Internet but
compute nodes are isolated, or in PC/laptop enviroments if you are travelling.
*TemplateFlow* will require Internet access the first time it receives a
query for a template resource that has not been previously accessed.
If you know what are the templates you are planning to use, you could
prefetch them using the Python client.
To do so, follow the next steps.

  1. By default, a mirror of *TemplateFlow* to store the resources will be
     created in ``$HOME/.cache/templateflow``.
     You can modify such a configuration with the ``TEMPLATEFLOW_HOME``
     environment variable, e.g.:
     ::
  
       $ export TEMPLATEFLOW_HOME=$HOME/.templateflow
  
  2. Install the client within your favorite Python 3 environment (this can
     be done in your login-node, or in a host with Internet access, 
     without need for Docker/Singularity):
     ::
       
       $ python -m pip install -U templateflow
  
  3. Use the ``get()`` utility of the client to pull down all the templates you'll
     want to use. For example:
     ::
  
       $ python -c "from templateflow.api import get; get(['MNI152NLin2009cAsym', 'MNI152NLin6Asym', 'OASIS30ANTs', 'MNIPediatricAsym', 'MNIInfant'])"

After pulling down the resources you'll need, you will just need to make sure your
runtime environment is able to access the filesystem, at the location of your
*TemplateFlow home* directory.
If you are a Singularity user, please check out :ref:`singularity_tf`.
