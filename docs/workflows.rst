.. include:: links.rst

===========================
Processing pipeline details
===========================

``fmriprep`` adapts its pipeline depending on what data and metadata is
available is used as the input. For example, slice timing correction will be
performed only if the ``SliceTiming`` metadata field is found for the input
dataset.

High-level view of the basic pipeline (for single-band datasets, without
slice-timing information and no fieldmap acquisitions):

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.base import basic_wf
    wf = basic_wf(
        {'func': ['bold_preprocessing']},
        settings={'ants_nthreads': 1,
                  'nthreads': 1,
                  'freesurfer': True,
                  'reportlets_dir': '.',
                  'output_dir': '.',
                  'bids_root': '.',
                  'biggest_epi_file_size_gb': 3,
                  'skull_strip_ants': True,
                  'output_spaces': ['T1w', 'fsnative',
                                    'MNI152NLin2009cAsym', 'fsaverage5'],
                  'ignore': [],
                  'debug': False,
                  'hires': True,
                  'bold2t1w_dof': 9}
    )


T1w/T2w preprocessing
---------------------
:mod:`fmriprep.workflows.anatomical.t1w_preprocessing`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.anatomical import t1w_preprocessing
    wf = t1w_preprocessing(settings={'ants_nthreads': 1,
                                     'nthreads': 1,
                                     'freesurfer': True,
                                     'reportlets_dir': '.',
                                     'output_dir': '.',
                                     'skull_strip_ants': True,
                                     'debug': False,
                                     'hires': True})

This sub-workflow finds the skull stripping mask and the
white matter/gray matter/cerebrospinal fluid segments and finds a non-linear
warp to the MNI space.

.. figure:: _static/brainextraction_t1.svg
    :scale: 100%

    Brain extraction (ANTs).

.. figure:: _static/segmentation.svg
    :scale: 100%

    Brain tissue segmentation (FAST).

.. figure:: _static/T1MNINormalization.svg
    :scale: 100%

    Animation showing T1w to MNI normalization (ANTs)

Surface preprocessing
~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.anatomical.surface_reconstruction`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.anatomical import surface_reconstruction
    wf = surface_reconstruction(
        settings={'nthreads': 1,
                  'freesurfer': True,
                  'reportlets_dir': '.',
                  'output_dir': '.',
                  'output_spaces': ['T1w', 'fsnative',
                                    'MNI152NLin2009cAsym', 'fsaverage5'],
                  'hires': True})

``fmriprep`` uses FreeSurfer_ to reconstruct surfaces from T1w/T2w
structural images.
If enabled, several steps in the ``fmriprep`` pipeline are added or replaced.
All surface preprocessing may be disabled with the ``--no-freesurfer`` flag.

If FreeSurfer reconstruction is performed, the reconstructed subject is placed in
``<output dir>/freesurfer/sub-<subject_label>/`` (see `FreeSurfer Derivatives`_).

Surface reconstruction is performed in three phases.
The first phase initializes the subject with T1w and T2w (if available)
structural images and performs basic reconstruction (``autorecon1``) with the
exception of skull-stripping.
For example, a subject with only one session with T1w and T2w images
would be processed by the following command::

    $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
        -i <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T1w.nii.gz \
        -T2 <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T2w.nii.gz \
        -autorecon1 \
        -noskullstrip

The second phase imports the brainmask calculated in the `T1w/T2w preprocessing`_
sub-workflow.
The final phase resumes reconstruction, using the T2w image to assist
in finding the pial surface, if available::

    $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
        -all -T2pial

Reconstructed white and pial surfaces are included in the report.

.. figure:: _static/reconall.svg
    :scale: 100%

    Surface reconstruction (FreeSurfer)

If T1w voxel sizes are less than 1mm in all dimensions (rounding to nearest
.1mm), `submillimeter reconstruction`_ is used, unless disabled with
``--no-submm-recon``.

In order to bypass reconstruction in ``fmriprep``, place existing reconstructed
subjects in ``<output dir>/freesurfer`` prior to the run.
``fmriprep`` will perform any missing ``recon-all`` steps, but will not perform
any steps whose outputs already exist.

``lh.midthickness`` and ``rh.midthickness`` surfaces are created in the subject
``surf/`` directory, corresponding to the surface half-way between the gray/white
boundary and the pial surface.
The ``smoothwm``, ``midthickness``, ``pial`` and ``inflated`` surfaces are also
converted to GIFTI_ format and adjusted to be compatible with multiple software
packages, including FreeSurfer and the `Connectome Workbench`_.


BOLD preprocessing
------------------
:mod:`fmriprep.workflows.epi.bold_preprocessing`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmriprep.workflows.epi import bold_preprocessing
    wf = bold_preprocessing(
        "bold_preprocessing",
        settings={'ants_nthreads': 1,
                  'ignore':[],
                  'nthreads': 1,
                  'freesurfer': True,
                  'reportlets_dir': '.',
                  'output_dir': '.',
                  'bids_root': '.',
                  'biggest_epi_file_size_gb': 3,
                  'skull_strip_ants': True,
                  'output_spaces': ['T1w', 'fsnative',
                                    'MNI152NLin2009cAsym', 'fsaverage5'],
                  'debug': False,
                  'bold2t1w_dof': 9})

Preprocessing of BOLD files is split into multiple sub-workflows decribed below.

.. epi_hmc :

Head-motion estimation and slice time correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.epi.epi_hmc`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.epi import epi_hmc
    wf = epi_hmc(
        metadata={"RepetitionTime": 2.0,
                  "SliceTiming": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        settings={'ants_nthreads': 1,
                  'ignore':[],
                  'nthreads': 1,
                  'freesurfer': True,
                  'reportlets_dir': '.',
                  'output_dir': '.',
                  'bids_root': '.',
                  'biggest_epi_file_size_gb': 3,
                  'skull_strip_ants': True,
                  'output_spaces': ['T1w', 'fsnative',
                                    'MNI152NLin2009cAsym', 'fsaverage5'],
                  'debug': False})

This workflow performs slice time
correction (if ``SliceTiming`` field is present in the input dataset metadata), head
motion estimation, and skullstripping.

Slice time correction is performed
using AFNI 3dTShift. All slices are realigned in time to the middle of each
TR. Slice time correction can be disabled with ``--ignore slicetiming`` command
line argument.

FSL MCFLIRT is used to estimate motion
transformations using an automatically estimated reference scan. If T1-saturation effects
("dummy scans" or non-steady state volumes) are detected they are used as reference due to
their superior tissue contrast. Otherwise a median of motion corrected subset of volumes is used.

Skullstripping of the reference image is performed using Nilearn.

.. figure:: _static/brainextraction.svg
    :scale: 100%

    Brain extraction (nilearn).

Susceptibility Distortion Correction (SDC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fmriprep.workflows.fieldmap
    :members:
    :undoc-members:
    :show-inheritance:

EPI to T1w registration
~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.epi.ref_epi_t1_registration`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.epi import ref_epi_t1_registration
    wf = ref_epi_t1_registration(
        "ref_epi_t1_registration",
        settings={'ants_nthreads': 1,
                  'ignore':[],
                  'nthreads': 1,
                  'freesurfer': True,
                  'reportlets_dir': '.',
                  'output_dir': '.',
                  'bids_root': '.',
                  'biggest_epi_file_size_gb': 3,
                  'skull_strip_ants': True,
                  'output_spaces': ['T1w', 'fsnative',
                                    'MNI152NLin2009cAsym', 'fsaverage5'],
                  'debug': False,
                  'bold2t1w_dof': 9})

The reference EPI image of each run is aligned by the ``bbregister`` routine to the
reconstructed subject using
the gray/white matter boundary (FreeSurfer's ``?h.white`` surfaces).

.. figure:: _static/EPIT1Normalization.svg
    :scale: 100%

    Animation showing EPI to T1w registration (FreeSurfer bbregister)

If FreeSurfer processing is disabled, FLIRT is performed with the BBR cost
function, using the FAST segmentation to establish the gray/white matter
boundary.

EPI to MNI transformation
~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.epi.epi_mni_transformation`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.epi import epi_mni_transformation
    wf = epi_mni_transformation(
        "epi_mni_transformation",
        settings={'ants_nthreads': 1,
                  'ignore':[],
                  'nthreads': 1,
                  'freesurfer': True,
                  'reportlets_dir': '.',
                  'output_dir': '.',
                  'bids_root': '.',
                  'biggest_epi_file_size_gb': 3,
                  'skull_strip_ants': True,
                  'debug': False})

This sub-workflow uses the transform from
`Head-motion estimation and slice time correction`_,
`Susceptibility Distortion Correction (SDC)`_ (if fieldmaps are available),
`EPI to T1w registration`_, and a T1w-to-MNI transform from
`T1w/T2w preprocessing`_ to map the EPI image to standardized MNI space.
It also maps the T1w-based mask to MNI space.

Transforms are concatenated and applied all at once, with one interpolation (Lanczos)
step, so as little information is lost as possible.

EPI sampled to FreeSurfer surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.epi.epi_surf_sample`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.epi import epi_surf_sample
    wf = epi_surf_sample(
        "epi_surf_sample",
        settings={'output_dir': '.',
                  'output_spaces': ['T1w', 'fsnative',
                                    'MNI152NLin2009cAsym', 'fsaverage5'],
                  })

If FreeSurfer processing is enabled, the motion-corrected functional series
(after single shot resampling to T1w space) is sampled to the
surface by averaging across the cortical ribbon.
Specifically, at each vertex, the segment normal to the white-matter surface, extending to the pial
surface, is sampled at 6 intervals and averaged.

Surfaces are generated for the "subject native" surface, as well as transformed to the
``fsaverage`` template space.
All surface outputs are in GIFTI format.

Confounds estimation
~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.confounds.discover_wf`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.confounds import discover_wf
    wf = discover_wf(name="discover_wf",
                 settings={'ants_nthreads': 1,
                           'ignore':[],
                                     'nthreads': 1,
                                     'freesurfer': True,
                                     'reportlets_dir': '.',
                                     'output_dir': '.',
                                     'bids_root': '.',
                                     'biggest_epi_file_size_gb': 3,
                                     'skull_strip_ants': True,
                                     'debug': False})

Given a motion-corrected fMRI, a brain mask, MCFLIRT movement parameters and a
segmentation, the `discover_wf` sub-workflow calculates potential
confounds per volume.

Calculated confounds include the mean global signal, mean tissue class signal,
tCompCor, aCompCor, Framewise Displacement, 6 motion parameters and DVARS.


Reports
-------

FMRIPREP outputs summary reports, outputted to ``<output dir>/fmriprep/sub-<subject_label>.html``.
These reports provide a quick way to make visual inspection of the results easy.
Each report is self contained and thus can be easily shared with collaborators (for example via email).
`View a sample report. <_static/sample_report.html>`_

Derivatives
-----------

There are additional files, called "Derivatives", outputted to ``<output dir>/fmriprep/sub-<subject_label>/``.
See the `BIDS Derivatives`_ spec for more information.

Derivatives related to t1w files are in the ``anat`` subfolder:

- ``*T1w_brainmask.nii.gz`` Brain mask derived using ANTS or AFNI, depending on the command flag ``--skull-strip-ants``
- ``*T1w_space-MNI152NLin2009cAsym_brainmask.nii.gz`` Same as above, but in MNI space.
- ``*T1w_dtissue.nii.gz`` Tissue class map derived using FAST.
- ``*T1w_preproc.nii.gz`` Bias field corrected t1w file, using ANTS' N4BiasFieldCorrection
- ``*T1w_smoothwm.[LR].surf.gii`` Smoothed GrayWhite surfaces
- ``*T1w_pial.[LR].surf.gii`` Pial surfaces
- ``*T1w_midthickness.[LR].surf.gii`` MidThickness surfaces
- ``*T1w_inflated.[LR].surf.gii`` FreeSurfer inflated surfaces for visualization
- ``*T1w_space-MNI152NLin2009cAsym_preproc.nii.gz`` Same as above, but in MNI space
- ``*T1w_space-MNI152NLin2009cAsym_class-CSF_probtissue.nii.gz``
- ``*T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz``
- ``*T1w_space-MNI152NLin2009cAsym_class-WM_probtissue.nii.gz`` Probability tissue maps, transformed into MNI space
- ``*T1w_target-MNI152NLin2009cAsym_warp.h5`` Composite (warp and affine) transform to transform t1w into MNI space

Derivatives related to EPI files are in the ``func`` subfolder.

- ``*bold_confounds.tsv`` A tab-separated value file with one column per calculated confound and one row per timepoint/volume

Volumetric output spaces include ``T1w`` and ``MNI152NLin2009cAsym`` (default).

- ``*bold_space-<space>_brainmask.nii.gz`` Brain mask for EPI files, calculated by nilearn on the average EPI volume, post-motion correction
- ``*bold_space-<space>_preproc.nii.gz`` Motion-corrected (using MCFLIRT for estimation and ANTs for interpolation) EPI file

Surface output spaces include ``fsnative`` (full density subject-specific mesh),
``fsaverage`` and the down-sampled meshes ``fsaverage6`` (41k vertices) and
``fsaverage5`` (10k vertices, default).

- ``*bold_space-<space>.[LR].func.gii`` Motion-corrected EPI file sampled to surface ``<space>``


FreeSurfer Derivatives
----------------------

A FreeSurfer subjects directory is created in ``<output dir>/freesurfer``.

::

    freesurfer/
        fsaverage{,5,6}/
            mri/
            surf/
            ...
        sub-<subject_label>/
            mri/
            surf/
            ...
        ...

Copies of the ``fsaverage`` subjects distributed with the running version of
FreeSurfer are copied into this subjects directory, if any functional data are
sampled to those subject spaces.

