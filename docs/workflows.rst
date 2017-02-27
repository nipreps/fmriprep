.. include:: links.rst

=========
Workflows
=========

Basic workflow (no fieldmaps)
=============================

``fmriprep``'s basic pipeline is used on datasets for which there are only t1ws
and at least one functional (EPI) file, but no SBRefs or fieldmaps.
To force using this pipeline on datasets that do include fieldmaps and SBRefs
use the ``--ignore fieldmaps`` flag.

What It Does
------------
High-level view of the basic pipeline:

.. image:: ds005.dot.png
    :scale: 100%

BIDSDatasource
~~~~~~~~~~~~~~

This node reads the BIDS_-formatted T1 data.

t1w_preprocessing
~~~~~~~~~~~~~~~~~

.. image:: t1w_preprocessing.dot.png
    :scale: 100%

The ``t1w_preprocessing`` sub-workflow finds the skull stripping mask and the
white matter/gray matter/cerebrospinal fluid segments and finds a non-linear
warp to the MNI space.

.. figure:: _static/brainextraction_t1.svg
    :scale: 100%

    Brain extraction (ANTs).

.. figure:: _static/segmentation.svg
    :scale: 100%

    Segmentation (FAST).

.. figure:: _static/T1MNINormalization.svg
    :scale: 100%

    Animation showing T1 to MNI normalization (ANTs)

Additionally, FreeSurfer surfaces are reconstructed from T1-weighted structural
image(s), using the ANTs-extracted brain mask.
This feature may be disabled with the ``--no-freesurfer`` flag.

.. figure:: _static/reconall.svg
    :scale: 100%

    Surface reconstruction (FreeSurfer)

EPI_HMC
~~~~~~~

.. image:: EPI_HMC.dot.png
    :scale: 100%

The EPI_HMC sub-workflow collects BIDS_-formatted EPI files, performs head
motion correction, and skullstripping. FSL MCFLIRT is used to estimate motion
transformations and ANTs is used to apply them using Lanczos interpolation. Nilearn
is used to perform skullstripping of the mean EPI image.

.. figure:: _static/brainextraction.svg
    :scale: 100%

    Brain extraction (nilearn).

ref_epi_t1_registration
~~~~~~~~~~~~~~~~~~~~

.. image:: ref_epi_t1_registration.dot.png
    :scale: 100%

The ref_epi_t1_registration sub-workflow uses FSL FLIRT with the BBR cost
function to find the transform that maps the EPI space into the T1-space.

.. figure:: _static/EPIT1Normalization.svg
    :scale: 100%

    Animation showing EPI to T1 registration (FSL FLIRT with BBR)

EPIMNITransformation
~~~~~~~~~~~~~~~~~~~~

.. image:: EPIMNITransformation.dot.png
    :scale: 100%

The EPIMNITransformation sub-workflow uses the transform from
`EPIMeanNormalization`_ and a t1-to-MNI transform from `t1w_preprocessing`_ to
map the EPI image to standardized MNI space.
It also maps the t1w-based mask to MNI space.

Transforms are concatenated and applied all at once, with one interpolation
step, so as little information is lost as possible.

ConfoundDiscoverer
~~~~~~~~~~~~~~~~~~

.. image:: ConfoundDiscoverer.dot.png
    :scale: 100%

Given a motion-corrected fMRI, a brain mask, MCFLIRT movement parameters and a
segmentation, the ConfoundDiscoverer sub-workflow calculates potential
confounds per volume.

Calculated confounds include the mean global signal, mean tissue class signal,
tCompCor, aCompCor, Framewise Displacement, 6 motion parameters and DVARS.


Reports
-------

``fmriprep`` outputs summary reports, outputted to ``<output dir>/fmriprep/sub-<subject_label>.html``.
These reports provide a quick way to make visual inspection of the results easy.
Each report is self contained and thus can be easily shared with collaborators (for example via email).
`View a sample report. <_static/sample_report.html>`_

Derivatives
-----------

There are additional files, called "Derivatives", outputted to ``<output dir>/fmriprep/sub-<subject_label>/``.
See the BIDS_ spec for more information.

Derivatives related to t1w files are in the ``anat`` subfolder:

- ``*T1w_brainmask.nii.gz`` Brain mask derived using ANTS or AFNI, depending on the command flag ``--skull-strip-ants``
- ``*T1w_space-MNI152NLin2009cAsym_brainmask.nii.gz`` Same as above, but in MNI space.
- ``*T1w_dtissue.nii.gz`` Tissue class map derived using FAST.
- ``*T1w_preproc.nii.gz`` Bias field corrected t1w file, using ANTS' N4BiasFieldCorrection
- ``*T1w_space-MNI152NLin2009cAsym_preproc.nii.gz`` Same as above, but in MNI space
- ``*T1w_target-MNI152NLin2009cAsym_affine.mat`` The affine matrix to transform T1w into MNI space
- ``*T1w_space-MNI152NLin2009cAsym_class-CSF_probtissue.nii.gz``
- ``*T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz``
- ``*T1w_space-MNI152NLin2009cAsym_class-WM_probtissue.nii.gz`` Probability tissue maps, transformed into MNI space
- ``*T1w_target-MNI152NLin2009cAsym_warp.nii.gz`` Warp transform to transform t1w into MNI space

Derivatives related to EPI files are in the ``func`` subfolder:

- ``*bold_space-T1w_brainmask.nii.gz`` Brain mask for EPI files, calculated by nilearn on the average EPI volume, post-motion correction, in T1w space
- ``*bold_space-MNI152NLin2009cAsym_brainmask.nii.gz`` Same as above, but in MNI space
- ``*bold_confounds.tsv`` A tab-separated value file with one column per calculated confound and one row per timepoint/volume
- ``*bold_space-T1w_preproc.nii.gz`` Motion-corrected (using MCFLIRT for estimation and ANTs for interpolation) EPI file in T1w space
- ``*bold_space-MNI152NLin2009cAsym_preproc.nii.gz`` Same as above, but in MNI space

If FreeSurfer reconstruction is performed, the reconstructed subject is placed in
``<output dir>/freesurfer/sub-<subject_label>/``.

