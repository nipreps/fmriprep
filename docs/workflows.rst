.. include:: links.rst

===========================
Processing pipeline details
===========================

*fMRIPrep* adapts its pipeline depending on what data and metadata are
available and are used as the input.
For example, slice timing correction will be
performed only if the ``SliceTiming`` metadata field is found for the input
dataset.

A (very) high-level view of the simplest pipeline (for a single-band dataset with only
one task, single-run, with no slice-timing information nor fieldmap acquisitions)
is presented below:

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmriprep.workflows.base import init_single_subject_wf
    from collections import namedtuple, OrderedDict
    BIDSLayout = namedtuple('BIDSLayout', ['root'])
    wf = init_single_subject_wf(
        anat_only=False,
        aroma_melodic_dim=-200,
        bold2t1w_dof=9,
        cifti_output=False,
        debug=False,
        dummy_scans=None,
        echo_idx=None,
        err_on_aroma_warn=False,
        fmap_bspline=False,
        fmap_demean=True,
        force_syn=True,
        freesurfer=True,
        hires=True,
        ignore=[],
        layout=BIDSLayout('.'),
        longitudinal=False,
        low_mem=False,
        medial_surface_nan=False,
        name='single_subject_wf',
        omp_nthreads=1,
        output_dir='.',
        reportlets_dir='.',
        regressors_all_comps=False,
        regressors_dvars_th=1.5,
        regressors_fd_th=0.5,
        skull_strip_fixed_seed=False,
        skull_strip_template=('OASIS30ANTs', {}),
        subject_id='test',
        t2s_coreg=False,
        target_spaces=OrderedDict([
            ('MNI152Lin', {}), ('fsaverage', {'density': '10k'}),
            ('T1w', {}), ('fsnative', {})]),
        task_id='',
        use_aroma=False,
        use_bbr=True,
        use_syn=True,
    )

T1w/T2w preprocessing
---------------------
:mod:`fmriprep.workflows.anatomical.init_anat_preproc_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from collections import OrderedDict
    from fmriprep.workflows.anatomical import init_anat_preproc_wf
    wf = init_anat_preproc_wf(
        bids_root='.',
        freesurfer=True,
        hires=True,
        longitudinal=False,
        num_t1w=1,
        omp_nthreads=1,
        output_dir='.',
        output_spaces=OrderedDict([
            ('MNI152NLin2009cAsym', {}), ('fsaverage5', {})]),
        reportlets_dir='.',
        skull_strip_template=('MNI152NLin2009cAsym', {}),
        skull_strip_fixed_seed=False,
    )

The anatomical sub-workflow begins by constructing an average image by
conforming all found T1w images to RAS orientation and
a common voxel size, and, in the case of multiple images, averages them into a
single reference template (see `Longitudinal processing`_).

.. _t1preproc_steps:

Brain extraction, brain tissue segmentation and spatial normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then, the T1w reference is skull-stripped using a Nipype implementation of
the ``antsBrainExtraction.sh`` tool (ANTs), which is an atlas-based
brain extraction workflow:

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from niworkflows.anat.ants import init_brain_extraction_wf
    wf = init_brain_extraction_wf()


An example of brain extraction is shown below:

.. figure:: _static/brainextraction_t1.svg
    :scale: 100%

    Brain extraction


Once the brain mask is computed, FSL ``fast`` is utilized for brain tissue segmentation.

.. figure:: _static/segmentation.svg
    :scale: 100%

    Brain tissue segmentation.


Finally, spatial normalization to standard spaces is performed using ANTs' ``antsRegistration``
in a multiscale, mutual-information based, nonlinear registration scheme.
See :ref:`output-spaces` for information about how standard and nonstandard spaces can
be set to resample the preprocessed data onto the final output spaces.


.. figure:: _static/T1MNINormalization.svg
    :scale: 100%

    Animation showing spatial normalization of T1w onto the ``MNI152NLin2009cAsym`` template.

Cost function masking during spatial normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When processing images from patients with focal brain lesions (e.g., stroke, tumor
resection), it is possible to provide a lesion mask to be used during spatial
normalization to standard space [Brett2001]_.
ANTs will use this mask to minimize warping of healthy tissue into damaged
areas (or vice-versa).
Lesion masks should be binary NIfTI images (damaged areas = 1, everywhere else = 0)
in the same space and resolution as the T1 image, and follow the naming convention specified in
`BIDS Extension Proposal 3: Common Derivatives <https://docs.google.com/document/d/1Wwc4A6Mow4ZPPszDIWfCUCRNstn7d_zzaWPcfcHmgI4/edit#heading=h.9146wuepclkt>`_
(e.g., ``sub-001_T1w_label-lesion_roi.nii.gz``).
This file should be placed in the ``sub-*/anat`` directory of the BIDS dataset
to be run through *fMRIPrep*.
Because lesion masks are not currently part of the BIDS specification, it is also necessary to
include a ``.bidsignore`` file in the root of your dataset directory. This will prevent
`bids-validator <https://github.com/bids-standard/bids-validator#bidsignore>`_ from complaining
that your dataset is not valid BIDS, which prevents *fMRIPrep* from running.
Your ``.bidsignore`` file should include the following line::

*lesion_roi.nii.gz


Longitudinal processing
~~~~~~~~~~~~~~~~~~~~~~~
In the case of multiple T1w images (across sessions and/or runs), T1w images are
merged into a single template image using FreeSurfer's `mri_robust_template`_.
This template may be *unbiased*, or equidistant from all source images, or
aligned to the first image (determined lexicographically by session label).
For two images, the additional cost of estimating an unbiased template is
trivial and is the default behavior, but, for greater than two images, the cost
can be a slowdown of an order of magnitude.
Therefore, in the case of three or more images, *fMRIPrep* constructs
templates aligned to the first image, unless passed the ``--longitudinal``
flag, which forces the estimation of an unbiased template.

.. note::

    The preprocessed T1w image defines the ``T1w`` space.
    In the case of multiple T1w images, this space may not be precisely aligned
    with any of the original images.
    Reconstructed surfaces and functional datasets will be registered to the
    ``T1w`` space, and not to the input images.


.. _workflows_surface:

Surface preprocessing
~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.anatomical.init_surface_recon_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmriprep.workflows.anatomical import init_surface_recon_wf
    wf = init_surface_recon_wf(omp_nthreads=1,
                               hires=True)

*fMRIPrep* uses FreeSurfer_ to reconstruct surfaces from T1w/T2w
structural images.
If enabled, several steps in the *fMRIPrep* pipeline are added or replaced.
All surface preprocessing may be disabled with the ``--fs-no-reconall`` flag.

.. note::
    Surface processing will be skipped if the outputs already exist.

    In order to bypass reconstruction in *fMRIPrep*, place existing reconstructed
    subjects in ``<output dir>/freesurfer`` prior to the run, or specify an external
    subjects directory with the ``--fs-subjects-dir`` flag.
    *fMRIPrep* will perform any missing ``recon-all`` steps, but will not perform
    any steps whose outputs already exist.


If FreeSurfer reconstruction is performed, the reconstructed subject is placed in
``<output dir>/freesurfer/sub-<subject_label>/`` (see :ref:`fsderivs`).

Surface reconstruction is performed in three phases.
The first phase initializes the subject with T1w and T2w (if available)
structural images and performs basic reconstruction (``autorecon1``) with the
exception of skull-stripping.
Skull-stripping is skipped since the brain mask :ref:`calculated previously
<t1preproc_steps>` is injected into the appropriate location for FreeSurfer.
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
in finding the pial surface, if available.
See :py:meth:`~fmriprep.workflows.anatomical.init_autorecon_resume_wf` for
details.

Reconstructed white and pial surfaces are included in the report.

.. figure:: _static/reconall.svg
    :scale: 100%

    Surface reconstruction (FreeSurfer)

If T1w voxel sizes are less than 1mm in all dimensions (rounding to nearest
.1mm), `submillimeter reconstruction`_ is used, unless disabled with
``--no-submm-recon``.

``lh.midthickness`` and ``rh.midthickness`` surfaces are created in the subject
``surf/`` directory, corresponding to the surface half-way between the gray/white
boundary and the pial surface.
The ``smoothwm``, ``midthickness``, ``pial`` and ``inflated`` surfaces are also
converted to GIFTI_ format and adjusted to be compatible with multiple software
packages, including FreeSurfer and the `Connectome Workbench`_.

.. note::
    GIFTI surface outputs are aligned to the FreeSurfer T1.mgz image, which
    may differ from the T1w space in some cases, to maintain compatibility
    with the FreeSurfer directory.
    Any measures sampled to the surface take into account any difference in
    these images.

Refinement of the brain mask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typically, the original brain mask calculated with ``antsBrainExtraction.sh``
will contain some innaccuracies including small amounts of MR signal from
outside the brain.
Based on the tissue segmentation of FreeSurfer (located in ``mri/aseg.mgz``)
and only when the :ref:`Surface Processing <workflows_surface>` step has been
executed, *fMRIPrep* replaces the brain mask with a refined one that derives
from the ``aseg.mgz`` file as described in
:mod:`fmriprep.interfaces.freesurfer.grow_mask`.


BOLD preprocessing
------------------
:mod:`fmriprep.workflows.bold.base.init_func_preproc_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from collections import namedtuple, OrderedDict
    from fmriprep.workflows.bold.base import init_func_preproc_wf
    BIDSLayout = namedtuple('BIDSLayout', ['root'])
    wf = init_func_preproc_wf(
        aroma_melodic_dim=-200,
        bold2t1w_dof=9,
        bold_file='/completely/made/up/path/sub-01_task-nback_bold.nii.gz',
        cifti_output=False,
        debug=False,
        dummy_scans=None,
        err_on_aroma_warn=False,
        fmap_bspline=True,
        fmap_demean=True,
        force_syn=True,
        freesurfer=True,
        ignore=[],
        low_mem=False,
        medial_surface_nan=False,
        omp_nthreads=1,
        output_dir='.',
        regressors_all_comps=False,
        regressors_fd_th=0.5,
        regressors_dvars_th=1.5,
        reportlets_dir='.',
        t2s_coreg=False,
        target_spaces=OrderedDict([
            ('MNI152Lin', {}), ('fsaverage', {'density': '10k'}),
            ('T1w', {}), ('fsnative', {})]),
        use_aroma=False,
        use_bbr=True,
        use_syn=True,
        layout=BIDSLayout('.'),
        num_bold=1,
    )

Preprocessing of :abbr:`BOLD (blood-oxygen level-dependent)` files is
split into multiple sub-workflows described below.

.. _bold_ref:

BOLD reference image estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.util.init_bold_reference_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmriprep.workflows.bold import init_bold_reference_wf
    wf = init_bold_reference_wf(omp_nthreads=1)

This workflow estimates a reference image for a
:abbr:`BOLD (blood-oxygen level-dependent)` series.
If a single-band reference ("sbref") image associated with the BOLD series is
available, then it is used directly.
If not, a reference image is estimated from the BOLD series as follows:
When T1-saturation effects ("dummy scans" or non-steady state volumes) are
detected, they are averaged and used as reference due to their
superior tissue contrast.
Otherwise, a median of motion corrected subset of volumes is used.

The reference image is then used to calculate a brain mask for the
:abbr:`BOLD (blood-oxygen level-dependent)` signal using the
:mod:`fmriprep.workflows.bold.util.init_enhance_and_skullstrip_bold_wf`.
Further, the reference is fed to the :ref:`head-motion estimation
workflow <bold_hmc>` and the :ref:`registration workflow to map
BOLD series into the T1w image of the same subject <bold_reg>`.

.. figure:: _static/brainextraction.svg
    :scale: 100%

    Calculation of a brain mask from the BOLD series.

.. _bold_hmc:

Head-motion estimation
~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.hmc.init_bold_hmc_wf`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.bold import init_bold_hmc_wf
    wf = init_bold_hmc_wf(
        mem_gb=1,
        omp_nthreads=1)

Using the previously :ref:`estimated reference scan <bold_ref>`,
FSL ``mcflirt`` is used to estimate head-motion.
As a result, one rigid-body transform with respect to
the reference image is written for each :abbr:`BOLD (blood-oxygen level-dependent)`
time-step.
Additionally, a list of 6-parameters (three rotations,
three translations) per time-step is written and fed to the
:ref:`confounds workflow <bold_confounds>`.
For a more accurate estimation of head-motion, we calculate its parameters
before any time-domain filtering (i.e., :ref:`slice-timing correction <bold_stc>`),
as recommended in [Power2017]_.

.. _bold_stc:

Slice time correction
~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.stc.init_bold_stc_wf`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.bold import init_bold_stc_wf
    wf = init_bold_stc_wf(
        metadata={'RepetitionTime': 2.0,
                  'SliceTiming': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        )

If the ``SliceTiming`` field is available within the input dataset metadata,
this workflow performs slice time correction prior to other signal resampling
processes.
Slice time correction is performed using AFNI ``3dTShift``.
All slices are realigned in time to the middle of each TR.

Slice time correction can be disabled with the ``--ignore slicetiming``
command line argument.
If a :abbr:`BOLD (blood-oxygen level-dependent)` series has fewer than
5 usable (steady-state) volumes, slice time correction will be disabled
for that run.


Susceptibility Distortion Correction (SDC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.fieldmap.base.init_sdc_wf`

.. figure:: _static/unwarping.svg
    :scale: 100%

    Applying susceptibility-derived distortion correction, based on
    fieldmap estimation.

One of the major problems that affects :abbr:`EPI (echo planar imaging)` data
is the spatial distortion caused by the inhomogeneity of the field inside
the scanner.
Please refer to :ref:`sdc` for details on the
available workflows.


Pre-processed BOLD in native space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.resampling.init_bold_preproc_trans_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmriprep.workflows.bold import init_bold_preproc_trans_wf
    wf = init_bold_preproc_trans_wf(mem_gb=3, omp_nthreads=1)

A new *preproc* :abbr:`BOLD (blood-oxygen level-dependent)` series is generated
from the slice-timing corrected or the original data (if
:abbr:`STC (slice-timing correction)` was not applied) in the
original space.
All volumes in the :abbr:`BOLD (blood-oxygen level-dependent)` series are
resampled in their native space by concatenating the mappings found in previous
correction workflows (:abbr:`HMC (head-motion correction)` and
:abbr:`SDC (susceptibility-derived distortion correction)` if excecuted)
for a one-shot interpolation process.
Interpolation uses a Lanczos kernel.

.. _bold_reg:

EPI to T1w registration
~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.registration.init_bold_reg_wf`

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmriprep.workflows.bold import init_bold_reg_wf
    wf = init_bold_reg_wf(
        freesurfer=True,
        mem_gb=1,
        omp_nthreads=1,
        use_bbr=True,
        bold2t1w_dof=9)

The alignment between the reference :abbr:`EPI (echo-planar imaging)` image
of each run and the reconstructed subject using the gray/white matter boundary
(FreeSurfer's ``?h.white`` surfaces) is calculated by the ``bbregister`` routine.

.. figure:: _static/EPIT1Normalization.svg
    :scale: 100%

    Animation showing :abbr:`EPI (echo-planar imaging)` to T1w registration (FreeSurfer ``bbregister``)

If FreeSurfer processing is disabled, FSL ``flirt`` is run with the
:abbr:`BBR (boundary-based registration)` cost function, using the
``fast`` segmentation to establish the gray/white matter boundary.
After :abbr:`BBR (boundary-based registration)` is run, the resulting affine transform will be compared to the initial transform found by FLIRT.
Excessive deviation will result in rejecting the BBR refinement and accepting the original, affine registration.

Resampling BOLD runs onto standard spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.resampling.init_bold_std_trans_wf`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from collections import OrderedDict
    from fmriprep.workflows.bold import init_bold_std_trans_wf
    wf = init_bold_std_trans_wf(
        freesurfer=True,
        mem_gb=3,
        omp_nthreads=1,
        standard_spaces=OrderedDict([('MNI152Lin', {}),
                                     ('fsaverage', {'density': '10k'})]),
    )

This sub-workflow concatenates the transforms calculated upstream (see
`Head-motion estimation`_, `Susceptibility Distortion Correction (SDC)`_ --if
fieldmaps are available--, `EPI to T1w registration`_, and an anatomical-to-standard
transform from `T1w/T2w preprocessing`_) to map the :abbr:`EPI (echo-planar imaging)`
image to the standard spaces given by the ``--output-spaces`` argument
(see :ref:`output-spaces`).
It also maps the T1w-based mask to each of those standard spaces.

Transforms are concatenated and applied all at once, with one interpolation (Lanczos)
step, so as little information is lost as possible.

The output space grid can be specified using modifiers to the ``--output-spaces``
argument.


EPI sampled to FreeSurfer surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.resampling.init_bold_surf_wf`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.bold import init_bold_surf_wf
    wf = init_bold_surf_wf(
        mem_gb=1,
        target_spaces=['T1w', 'fsnative',
                       'template', 'fsaverage5'],
        medial_surface_nan=False)

If FreeSurfer processing is enabled, the motion-corrected functional series
(after single shot resampling to T1w space) is sampled to the
surface by averaging across the cortical ribbon.
Specifically, at each vertex, the segment normal to the white-matter surface, extending to the pial
surface, is sampled at 6 intervals and averaged.

Surfaces are generated for the "subject native" surface, as well as transformed to the
``fsaverage`` template space.
All surface outputs are in GIFTI format.

HCP Grayordinates
~~~~~~~~~~~~~~~~~
If CIFTI output is enabled, the motion-corrected functional timeseries (in T1w space) is first
sampled to the high resolution 164k vertex (per hemisphere) ``fsaverage``. Following that,
the resampled timeseries is sampled to `HCP Pipelines_`'s ``fsLR`` mesh (with the left and
right hemisphere aligned) using `Connectome Workbench`_'s ``-metric-resample`` to generate a
surface timeseries for each hemisphere. These surfaces are then combined with corresponding
volumetric timeseries to create a CIFTI2 file.

.. _bold_confounds:

Confounds estimation
~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.confounds.init_bold_confs_wf`

.. workflow::
    :graph2use: colored
    :simple_form: yes

    from fmriprep.workflows.bold.confounds import init_bold_confs_wf
    wf = init_bold_confs_wf(
        name="discover_wf",
        mem_gb=1,
        metadata={"RepetitionTime": 2.0,
                  "SliceTiming": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        regressors_all_comps=False,
        regressors_dvars_th=1.5,
        regressors_fd_th=0.5,
    )

Given a motion-corrected fMRI, a brain mask, ``mcflirt`` movement parameters and a
segmentation, the `discover_wf` sub-workflow calculates potential
confounds per volume.

Calculated confounds include the mean global signal, mean tissue class signal,
tCompCor, aCompCor, Frame-wise Displacement, 6 motion parameters, DVARS, spike regressors,
and, if the ``--use-aroma`` flag is enabled, the noise components identified by ICA-AROMA
(those to be removed by the "aggressive" denoising strategy).
Particular details about ICA-AROMA are given below.


ICA-AROMA
~~~~~~~~~
:mod:`fmriprep.workflows.bold.confounds.init_ica_aroma_wf`

ICA-AROMA denoising is performed in ``MNI152NLin6Asym`` space, which is automatically
added to the list of ``--output-spaces`` if it was not already requested by the user.
The number of ICA-AROMA components depends on a dimensionality estimate made by
FSL MELODIC.
For datasets with a very short TR and a large number of timepoints, this may result
in an unusually high number of components.
By default, dimensionality is limited to a maximum of 200 components.
To override this upper limit one may specify the number of components to be extracted
with ``--aroma-melodic-dimensionality``.
Further details on the implementation are given within the workflow generation
function (:mod:`fmriprep.workflows.bold.confounds.init_ica_aroma_wf`).

*Note*: *non*-aggressive AROMA denoising is a fundamentally different procedure
from its "aggressive" counterpart and cannot be performed only by using a set of noise
regressors (a separate GLM with both noise and signal regressors needs to be used).
Therefore instead of regressors, *fMRIPrep* produces *non*-aggressive denoised 4D NIFTI
files in the MNI space:

``*space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz``

Additionally, the MELODIC mix and noise component indices will
be generated, so non-aggressive denoising can be manually performed in the T1w space with ``fsl_regfilt``, *e.g.*::

    fsl_regfilt -i sub-<subject_label>_task-<task_id>_space-T1w_desc-preproc_bold.nii.gz \
        -f $(cat sub-<subject_label>_task-<task_id>_AROMAnoiseICs.csv) \
        -d sub-<subject_label>_task-<task_id>_desc-MELODIC_mixing.tsv \
        -o sub-<subject_label>_task-<task_id>_space-T1w_desc-AROMAnonaggr_bold.nii.gz

*Note*: The non-steady state volumes are removed for the determination of components in melodic.
Therefore ``*MELODIC_mixing.tsv`` may have zero padded rows to account for the volumes not used in
melodic's estimation of components.

A visualization of the AROMA component classification is also included in the HTML reports.

.. figure:: _static/aroma.svg
    :scale: 100%

    Maps created with maximum intensity projection (glass brain) with a black
    brain outline.
    Right hand side of each map: time series (top in seconds),
    frequency spectrum (bottom in Hertz).
    Components classified as signal in green; noise in red.


.. _bold_t2s:

T2* Driven Coregistration
~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`fmriprep.workflows.bold.t2s.init_bold_t2s_wf`

If multi-echo :abbr:`BOLD (blood-oxygen level-dependent)` data is supplied,
this workflow uses the `tedana`_ `T2* workflow`_ to generate an adaptive T2* map
and optimally weighted combination of all supplied single echo time series.
This optimaly combined time series is then carried forward for all subsequent
preprocessing steps.
Optionally, if the ``--t2s-coreg`` flag is supplied, the T2* map is then used
in place of the :ref:`BOLD reference image <bold_ref>` to
:ref:`register the BOLD series to the T1w image <bold_reg>` of the same subject.

.. _tedana: https://github.com/me-ica/tedana
.. _`T2* workflow`: https://tedana.readthedocs.io/en/latest/generated/tedana.workflows.t2smap_workflow.html#tedana.workflows.t2smap_workflow  # noqa
