*fMRIPrep*: A Robust Preprocessing Pipeline for fMRI Data
=========================================================
*fMRIPrep* is a *NiPreps (NeuroImaging PREProcessing toolS)* application
(`www.nipreps.org <https://www.nipreps.org>`__) for the preprocessing of
task-based and resting-state functional MRI (fMRI).

.. image:: https://img.shields.io/badge/docker-nipreps/fmriprep-brightgreen.svg?logo=docker&style=flat
  :target: https://hub.docker.com/r/nipreps/fmriprep/tags/
  :alt: Docker image available!

.. image:: https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg
  :target: https://doi.org/10.24433/CO.ed5ddfef-76a3-4996-b298-e3200f69141b
  :alt: Available in CodeOcean!

.. image:: https://circleci.com/gh/nipreps/fmriprep/tree/master.svg?style=shield
  :target: https://circleci.com/gh/nipreps/fmriprep/tree/master

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
.. image:: https://github.com/oesteban/fmriprep/raw/f4c7a9804be26c912b24ef4dccba54bdd72fa1fd/docs/_static/fmriprep-21.0.0.svg


*fMRIPrep* is a functional magnetic resonance imaging (fMRI) data
preprocessing pipeline that is designed to provide an easily accessible,
state-of-the-art interface that is robust to variations in scan acquisition
protocols and that requires minimal user input, while providing easily
interpretable and comprehensive error and output reporting.
It performs basic processing steps (coregistration, normalization, unwarping,
noise component extraction, segmentation, skull-stripping, etc.) providing
outputs that can be easily submitted to a variety of group level analyses,
including task-based or resting-state fMRI, graph theory measures, and surface
or volume-based statistics.

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
  speed-up from manual processing or shell-scripted pipelines.

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

Citation
--------
**Citation boilerplate**.
Please acknowledge this work using the citation boilerplate that *fMRIPrep* includes
in the visual report generated for every subject processed.
For a more detailed description of the citation boilerplate and its relevance,
please check out the
`NiPreps documentation <https://www.nipreps.org/intro/transparency/#citation-boilerplates>`__.

**Plagiarism disclaimer**.
The boilerplate text is public domain, distributed under the
`CC0 license <https://creativecommons.org/publicdomain/zero/1.0/>`__,
and we recommend *fMRIPrep* users to reproduce it verbatim in their works.
Therefore, if reviewers and/or editors raise concerns because the text is flagged by automated
plagiarism detection, please refer them to the *NiPreps* community and/or the note to this
effect in the `boilerplate documentation page <https://www.nipreps.org/intro/transparency/#citation-boilerplates>`__.

**Papers**.
*fMRIPrep* contributors have published two relevant papers:
`Esteban et al. (2019) <https://doi.org/10.1038/s41592-018-0235-4>`__
[`preprint <https://doi.org/10.1101/306951>`__], and
`Esteban et al. (2020) <https://doi.org/10.1038/s41596-020-0327-3>`__
[`preprint <https://doi.org/10.1101/694364>`__].

**Other**.
Other materials that have been generated over time include the
`OHBM 2018 software demonstration <https://effigies.github.io/fmriprep-demo/>`__
and some conference posters:

* Organization for Human Brain Mapping 2018
  (`Abstract <https://ww5.aievolution.com/hbm1801/index.cfm?do=abs.viewAbs&abs=1321>`__;
  `PDF <https://files.aievolution.com/hbm1801/abstracts/31779/2035_Markiewicz.pdf>`__)

.. image:: _static/OHBM2018-poster_thumb.png
   :target: _static/OHBM2018-poster.png

* Organization for Human Brain Mapping 2017
  (`Abstract <https://ww5.aievolution.com/hbm1701/index.cfm?do=abs.viewAbs&abs=4111>`__;
  `PDF <https://f1000research.com/posters/6-1129>`__)

.. image:: _static/OHBM2017-poster_thumb.png
   :target: _static/OHBM2017-poster.png

License information
-------------------
*fMRIPrep* adheres to the
`general licensing guidelines <https://www.nipreps.org/community/licensing/>`__
of the *NiPreps framework*.

License
~~~~~~~
Copyright (c) 2023, the *NiPreps* Developers.

As of the 21.0.x pre-release and release series, *fMRIPrep* is
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
`http://www.apache.org/licenses/LICENSE-2.0
<http://www.apache.org/licenses/LICENSE-2.0>`__.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Acknowledgements
----------------
This work is steered and maintained by the `NiPreps Community <https://www.nipreps.org>`__.
This work was supported by the Laura and John Arnold Foundation,
the NIH (grant NBIB R01EB020740, PI: Ghosh),
and NIMH (R24MH114705, R24MH117179, R01MH121867, PI: Poldrack)
