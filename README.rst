*fMRIPrep*: A Robust Preprocessing Pipeline for fMRI Data
=========================================================

This pipeline is developed by the `Poldrack lab at Stanford University
<https://poldracklab.stanford.edu/>`_ for use at the `Center for Reproducible
Neuroscience (CRN) <http://reproducibility.stanford.edu/>`_, as well as for
open-source software distribution.

.. image:: https://img.shields.io/badge/docker-poldracklab/fmriprep-brightgreen.svg?logo=docker&style=flat
  :target: https://hub.docker.com/r/poldracklab/fmriprep/tags/
  :alt: Docker image available!

.. image:: https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg
  :target: https://doi.org/10.24433/CO.ed5ddfef-76a3-4996-b298-e3200f69141b
  :alt: Available in CodeOcean!

.. image:: https://circleci.com/gh/poldracklab/fmriprep/tree/master.svg?style=shield
  :target: https://circleci.com/gh/poldracklab/fmriprep/tree/master

.. image:: https://readthedocs.org/projects/fmriprep/badge/?version=latest
  :target: http://fmriprep.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/fmriprep.svg
  :target: https://pypi.python.org/pypi/fmriprep/
  :alt: Latest Version

.. image:: https://img.shields.io/badge/doi-10.1038%2Fs41592--018--0235--4-blue.svg
  :target: https://doi.org/10.1038/s41592-018-0235-4
  :alt: Published in Nature Methods

.. image:: https://img.shields.io/badge/RRID-SCR__016216-blue.svg
  :target: https://doi.org/10.1038/s41592-018-0235-4
  :alt: RRID:SCR_016216

About
-----

.. image:: https://github.com/oesteban/fmriprep/raw/38a63e9504ab67812b63813c5fe9af882109408e/docs/_static/fmriprep-workflow-all.png


*fMRIPrep* is a functional magnetic resonance imaging (fMRI) data
preprocessing pipeline that is designed to provide an easily accessible,
state-of-the-art interface that is robust to variations in scan acquisition
protocols and that requires minimal user input, while providing easily
interpretable and comprehensive error and output reporting.
It performs basic processing steps (coregistration, normalization, unwarping,
noise component extraction, segmentation, skullstripping etc.) providing
outputs that can be easily submitted to a variety of group level analyses,
including task-based or resting-state fMRI, graph theory measures, surface or
volume-based statistics, etc.

.. note::

   *fMRIPrep* performs minimal preprocessing.
   Here we define 'minimal preprocessing'  as motion correction, field
   unwarping, normalization, bias field correction, and brain extraction.
   See the `workflows section of our documentation
   <https://fmriprep.readthedocs.io/en/latest/workflows.html>`__ for more details.

The *fMRIPrep* pipeline uses a combination of tools from well-known software
packages, including FSL_, ANTs_, FreeSurfer_ and AFNI_.
This pipeline was designed to provide the best software implementation for each
state of preprocessing, and will be updated as newer and better neuroimaging
software become available.

This tool allows you to easily do the following:

- Take fMRI data from raw to fully preprocessed form.
- Implement tools from different software packages.
- Achieve optimal data processing quality by using the best tools available.
- Generate preprocessing quality reports, with which the user can easily
  identify outliers.
- Receive verbose output concerning the stage of preprocessing for each
  subject, including meaningful errors.
- Automate and parallelize processing steps, which provides a significant
  speed-up from typical linear, manual processing.

More information and documentation can be found at
https://fmriprep.readthedocs.io/


Principles
----------

*fMRIPrep* is built around three principles:

1. **Robustness** - The pipeline adapts the preprocessing steps depending on
   the input dataset and should provide results as good as possible
   independently of scanner make, scanning parameters or presence of additional
   correction scans (such as fieldmaps).
2. **Ease of use** - Thanks to dependence on the BIDS standard, manual
   parameter input is reduced to a minimum, allowing the pipeline to run in an
   automatic fashion.
3. **"Glass box"** philosophy - Automation should not mean that one should not
   visually inspect the results or understand the methods.
   Thus, *fMRIPrep* provides visual reports for each subject, detailing the
   accuracy of the most important processing steps.
   This, combined with the documentation, can help researchers to understand
   the process and decide which subjects should be kept for the group level
   analysis.


Limitations and reasons not to use *fMRIPrep*
---------------------------------------------

1. Very narrow :abbr:`FoV (field-of-view)` images oftentimes do not contain
   enough information for standard image registration methods to work correctly.
   Also, problems may arise when extracting the brain from these data.
   Supporting these particular images is already a future line of the development
   road-map.
2. *fMRIPrep* may also underperform for particular populations (e.g., infants) and
   non-human brains, although appropriate templates can be provided to overcome the
   issue.
3. The "EPInorm" approach is currently not supported, although we plan to implement
   this feature (see `#620 <https://github.com/poldracklab/fmriprep/issues/620>`_).
4. If you really want unlimited flexibility (which is obviously a double-edged sword).
5. If you want students to suffer through implementing each step for didactic purposes,
   or to learn shell-scripting or Python along the way.
6. If you are trying to reproduce some *in-house* lab pipeline.

(Reasons 4-6 were kindly provided by S. Nastase in his
`open review <https://pubpeer.com/publications/6B3E024EAEBF2C80085FDF644C2085>`__
of our `pre-print <https://doi.org/10.1101/306951>`__).


Acknowledgements
----------------

Please acknowledge this work using the citation boilerplate that *fMRIPrep* includes
in the visual report generated for every subject processed.
For an illustration of how the citation boilerplate generally reads, please
check `our documentation <https://fmriprep.readthedocs.io/en/latest/citing.html>`__.
