.. include:: links.rst

fmriprep: A Robust Preprocessing Pipeline for fMRI Data
=======================================================

This pipeline is developed by the `Poldrack lab at Stanford University
<https://poldracklab.stanford.edu/>`_ for use at the `Center for Reproducible
Neuroscience (CRN) <http://reproducibility.stanford.edu/>`_, as well as for
open-source software distribution.

.. image:: https://circleci.com/gh/poldracklab/fmriprep/tree/master.svg?style=shield
  :target: https://circleci.com/gh/poldracklab/fmriprep/tree/master

.. image:: https://readthedocs.org/projects/fmriprep/badge/?version=latest
  :target: http://fmriprep.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/fmriprep.svg
  :target: https://pypi.python.org/pypi/fmriprep/
  :alt: Latest Version


About
-----

``fmriprep`` is a functional magnetic resonance imaging (fMRI) data
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

   fmriprep performs minimal preprocessing.
   Here we define 'minimal preprocessing'  as motion correction, field
   unwarping, normalization, bias field correction, and brain extraction.
   See the workflows_ for more details.

The ``fmriprep`` pipeline uses a combination of tools from well-known software
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

``fmriprep`` is built around three principles:

1. **Robustness** - The pipeline adapts the preprocessing steps depending on
   the input dataset and should provide results as good as possible
   independently of scanner make, scanning parameters or presence of additional
   correction scans (such as fieldmaps).
2. **Ease of use** - Thanks to dependence on the BIDS standard, manual
   parameter input is reduced to a minimum, allowing the pipeline to run in an
   automatic fashion.
3. **"Glass box"** philosophy - Automation should not mean that one should not
   visually inspect the results or understand the methods.
   Thus, ``fmriprep`` provides visual reports for each subject, detailing the
   accuracy of the most important processing steps.
   This, combined with the documentation, can help researchers to understand
   the process and decide which subjects should be kept for the group level
   analysis.

Acknowledgements
----------------

Please acknowledge this work by mentioning explicitly the name of this software
(fmriprep) and the version, along with a link to the `GitHub repository
<https://github.com/poldracklab/fmriprep>`_ or the Zenodo reference.
For more details, please see :ref:`citation`.

.. include:: license.rst


.. image:: https://badges.gitter.im/poldracklab/fmriprep.svg
   :alt: Join the chat at https://gitter.im/poldracklab/fmriprep
   :target: https://gitter.im/poldracklab/fmriprep?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
