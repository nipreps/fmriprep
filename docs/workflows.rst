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

Several steps are added or modified if `Surface preprocessing`_ is enabled.

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

If enabled, FreeSurfer surfaces are reconstructed from T1-weighted structural
image(s), using the ANTs-extracted brain mask.
See Reconstruction_ for details.

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
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ref_epi_t1_registration.dot.png
    :scale: 100%

The ref_epi_t1_registration sub-workflow uses FSL FLIRT with the BBR cost
function to find the transform that maps the EPI space into the T1-space.

.. figure:: _static/EPIT1Normalization.svg
    :scale: 100%

    Animation showing EPI to T1 registration (FSL FLIRT with BBR)

If surface processing is enabled, ``bbregister`` is used instead.
See `Boundary-based Registration (BBR)`_ for details.

EPIMNITransformation
~~~~~~~~~~~~~~~~~~~~

.. image:: EPIMNITransformation.dot.png
    :scale: 100%

The EPIMNITransformation sub-workflow uses the transform from
`EPIMeanNormalization`_ and a T1-to-MNI transform from `t1w_preprocessing`_ to
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


Surface preprocessing
=====================

``fmriprep`` uses FreeSurfer_ to reconstruct surfaces from T1/T2-weighted
structural images.
If enabled, several steps in the ``fmriprep`` pipeline are added or replaced.
All surface preprocessing may be disabled with the ``--no-freesurfer`` flag.

Reconstruction
--------------
If FreeSurfer reconstruction is performed, the reconstructed subject is placed in
``<output dir>/freesurfer/sub-<subject_label>/`` (see `FreeSurfer Derivatives`_).

Surface reconstruction is performed in three phases.
The first phase initializes the subject with T1- and T2-weighted (if available)
structural images and performs basic reconstruction (``autorecon1``) with the
exception of skull-stripping.
For example, a subject with only one session with T1 and T2-weighted images
would be processed by the following command::

    $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
        -i <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T1w.nii.gz \
        -T2 <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T2w.nii.gz \
        -autorecon1 \
        -noskullstrip

The second phase imports the brainmask calculated in the t1w_preprocessing_
sub-workflow.
The final phase resumes reconstruction, using the T2-weighted image to assist
in finding the pial surface, if available::

    $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
        -all -T2pial

Reconstructed white and pial surfaces are included in the report.

.. figure:: _static/reconall.svg
    :scale: 100%

    Surface reconstruction (FreeSurfer)

If T1-weighted voxel sizes are less 1mm in all dimensions (rounding to nearest
.1mm), `submillimeter reconstruction`_ is used.

In order to bypass reconstruction in ``fmriprep``, place existing reconstructed
subjects in ``<output dir>/freesurfer`` prior to the run.
``fmriprep`` will perform any missing ``recon-all`` steps, but will not perform
any steps whose outputs already exist.

Boundary-based Registration (BBR)
---------------------------------
The mean EPI image of each run is aligned to the reconstructed subject using
the gray/white matter boundary (FreeSurfer's ``?h.white`` surfaces).

If FreeSurfer processing is disabled, FLIRT is performed with the BBR cost
function, using the FAST segmentation to establish the gray/white matter
boundary.

FreeSurfer Derivatives
----------------------

A FreeSurfer subjects directory is created in ``<output dir>/freesurfer``.

::

    freesurfer/
        fsaverage/
            mri/
            surf/
            ...
        sub-<subject_label>/
            mri/
            surf/
            ...
        ...

A copy of the ``fsaverage`` subject distributed with the running version of
FreeSurfer is copied into this subjects directory.
