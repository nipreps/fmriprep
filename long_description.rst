Preprocessing of functional MRI (fMRI) involves numerous steps to clean and standardize
the data before statistical analysis.
Generally, researchers create ad hoc preprocessing workflows for each dataset,
building upon a large inventory of available tools.
The complexity of these workflows has snowballed with rapid advances in
acquisition and processing.
fMRIPrep is an analysis-agnostic tool that addresses the challenge of robust and
reproducible preprocessing for task-based and resting fMRI data.
fMRIPrep automatically adapts a best-in-breed workflow to the idiosyncrasies of
virtually any dataset, ensuring high-quality preprocessing without manual intervention.
fMRIPrep robustly produces high-quality results on diverse fMRI data.
Additionally, fMRIPrep introduces less uncontrolled spatial smoothness than observed
with commonly used preprocessing tools.
fMRIPrep equips neuroscientists with an easy-to-use and transparent preprocessing
workflow, which can help ensure the validity of inference and the interpretability
of results.

The workflow is based on `Nipype <https://nipype.readthedocs.io>`_ and encompasses a large
set of tools from well-known neuroimaging packages, including
`FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`_,
`ANTs <https://stnava.github.io/ANTs/>`_,
`FreeSurfer <https://surfer.nmr.mgh.harvard.edu/>`_,
`AFNI <https://afni.nimh.nih.gov/>`_,
and `Nilearn <https://nilearn.github.io/>`_.
This pipeline was designed to provide the best software implementation for each state of
preprocessing, and will be updated as newer and better neuroimaging software becomes
available.

fMRIPrep performs basic preprocessing steps (coregistration, normalization, unwarping, noise
component extraction, segmentation, skullstripping etc.) providing outputs that can be
easily submitted to a variety of group level analyses, including task-based or resting-state
fMRI, graph theory measures, surface or volume-based statistics, etc.
fMRIPrep allows you to easily do the following:

  * Take fMRI data from *unprocessed* (only reconstructed) to ready for analysis.
  * Implement tools from different software packages.
  * Achieve optimal data processing quality by using the best tools available.
  * Generate preprocessing-assessment reports, with which the user can easily identify problems.
  * Receive verbose output concerning the stage of preprocessing for each subject, including
    meaningful errors.
  * Automate and parallelize processing steps, which provides a significant speed-up from
    typical linear, manual processing.

[Nat Meth doi:`10.1038/s41592-018-0235-4 <https://doi.org/10.1038/s41592-018-0235-4>`_]
[Documentation `fmriprep.org <https://fmriprep.readthedocs.io>`_]
[Software doi:`10.5281/zenodo.852659 <https://doi.org/10.5281/zenodo.852659>`_]
[Support `neurostars.org <https://neurostars.org/tags/fmriprep>`_]

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
