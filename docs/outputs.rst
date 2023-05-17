.. include:: links.rst

.. _outputs:

---------------------
Outputs of *fMRIPrep*
---------------------
*fMRIPrep* outputs conform to the :abbr:`BIDS (brain imaging data structure)`
Derivatives specification (see `BIDS Derivatives`_, along with the
upcoming `BEP 011`_ and `BEP 012`_).
*fMRIPrep* generates three broad classes of outcomes:

1. **Visual QA (quality assessment) reports**:
   one :abbr:`HTML (hypertext markup language)` per subject,
   that allows the user a thorough visual assessment of the quality
   of processing and ensures the transparency of *fMRIPrep* operation.

2. **Derivatives (preprocessed data)** the input fMRI data ready for
   analysis, i.e., after the various preparation procedures
   have been applied.
   For example, :abbr:`INU (intensity non-uniformity)`-corrected versions
   of the T1-weighted image (per subject), the brain mask,
   or :abbr:`BOLD (blood-oxygen level dependent)`
   images after head-motion correction, slice-timing correction and aligned into
   the same-subject's T1w space or in some standard space.

3. **Confounds**: this is a special family of derivatives that can be utilized
   to inform subsequent denoising steps.

   .. important::
       In order to remain agnostic to any possible subsequent analysis,
       *fMRIPrep* does not perform any denoising (e.g., spatial smoothing) itself.
       There are exceptions to this principle (described in its corresponding
       section below):

       - CompCor regressors, which are calculated after temporal high-pass filtering.

Layout
------
Assuming fMRIPrep is invoked with::

    fmriprep <input_dir>/ <output_dir>/ participant [OPTIONS]

The outputs will be a `BIDS Derivatives`_ dataset of the form::

    <output_dir>/
      logs/
      sub-<label>/
      sub-<label>.html
      dataset_description.json
      .bidsignore

For each participant in the dataset,
a directory of derivatives (``sub-<label>/``)
and a visual report (``sub-<label>.html``) are generated.
The log directory contains `citation boilerplate`_ text.
``dataset_description.json`` is a metadata file in which fMRIPrep
records metadata recommended by the BIDS standard.

This layout, now the default, may be explicitly specified with the
``--output-layout bids`` command-line option.
For compatibility with versions of fMRIPrep prior to 21.0, the
`legacy layout`_ is available via ``--output-layout legacy``.

Visual Reports
--------------
*fMRIPrep* outputs summary reports, written to ``<output dir>/fmriprep/sub-<subject_label>.html``.
These reports provide a quick way to make visual inspection of the results easy.
`View a sample report. <_static/SampleReport/sample_report.html>`_

Derivatives of *fMRIPrep* (preprocessed data)
---------------------------------------------
Preprocessed, or derivative, data are written to
``<output dir>/sub-<subject_label>/``.
The `BIDS Derivatives`_ specification describes the naming and metadata conventions we follow.

Anatomical derivatives
~~~~~~~~~~~~~~~~~~~~~~
Anatomical derivatives are placed in each subject's ``anat`` subfolder::

  sub-<subject_label>/
    anat/
      sub-<subject_label>[_space-<space_label>]_desc-preproc_T1w.nii.gz
      sub-<subject_label>[_space-<space_label>]_desc-preproc_T2w.nii.gz
      sub-<subject_label>[_space-<space_label>]_desc-brain_mask.nii.gz
      sub-<subject_label>[_space-<space_label>]_dseg.nii.gz
      sub-<subject_label>[_space-<space_label>]_label-CSF_probseg.nii.gz
      sub-<subject_label>[_space-<space_label>]_label-GM_probseg.nii.gz
      sub-<subject_label>[_space-<space_label>]_label-WM_probseg.nii.gz

Spatially-standardized derivatives are denoted with a space label,
such as ``MNI152NLin2009cAsym``, while derivatives in
the original ``T1w`` space omit the ``space-`` keyword.

T2w images are aligned to the anatomical (``T1w``) space, if found.

.. note::

   T2w derivatives are only generated if FreeSurfer processing is enabled.

Additionally, the following transforms are saved::

  sub-<subject_label>/
    anat/
      sub-<subject_label>_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
      sub-<subject_label>_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5

If FreeSurfer reconstructions are used, the following surface files are generated::

  sub-<subject_label>/
    anat/
      sub-<subject_label>_hemi-[LR]_smoothwm.surf.gii
      sub-<subject_label>_hemi-[LR]_pial.surf.gii
      sub-<subject_label>_hemi-[LR]_midthickness.surf.gii
      sub-<subject_label>_hemi-[LR]_inflated.surf.gii

And the affine translation (and inverse) between the original T1w sampling and FreeSurfer's
conformed space for surface reconstruction (``fsnative``) is stored in::

  sub-<subject_label>/
    anat/
      sub-<subject_label>_from-fsnative_to-T1w_mode-image_xfm.txt
      sub-<subject_label>_from-T1w_to-fsnative_mode-image_xfm.txt

.. _fsderivs:

FreeSurfer derivatives
~~~~~~~~~~~~~~~~~~~~~~
If FreeSurfer is run, then a FreeSurfer subjects directory is created in
``<output dir>/sourcedata/freesurfer`` or the directory indicated with the
``--fs-subjects-dir`` flag.
Additionally, FreeSurfer segmentations are resampled into the BOLD space,
and lookup tables are provided. ::

    <output_dir>/
      sourcedata/
        freesurfer/
          fsaverage{,5,6}/
              mri/
              surf/
              ...
          sub-<label>/
              mri/
              surf/
              ...
          ...
      desc-aparc_dseg.tsv
      desc-aparcaseg_dseg.tsv

Copies of the ``fsaverage`` subjects distributed with the running version of
FreeSurfer are copied into this subjects directory, if any functional data are
sampled to those subject spaces.

Note that the use of ``sourcedata/`` recognizes FreeSurfer derivatives as an input to
the fMRIPrep workflow.
This is strictly true when pre-computed FreeSurfer derivatives are provided either in
the ``sourcedata/`` directory or passed via the ``--fs-subjects-dir`` flag;
if fMRIPrep runs FreeSurfer, then there is a mutual dependency.

Functional derivatives
~~~~~~~~~~~~~~~~~~~~~~
Functional derivatives are stored in the ``func/`` subfolder.
All derivatives contain ``task-<task_label>`` (mandatory) and ``run-<run_index>`` (optional), and
these will be indicated with ``[specifiers]``::

  sub-<subject_label>/
    func/
      sub-<subject_label>_[specifiers]_space-<space_label>_boldref.nii.gz
      sub-<subject_label>_[specifiers]_space-<space_label>_desc-brain_mask.nii.gz
      sub-<subject_label>_[specifiers]_space-<space_label>_desc-preproc_bold.nii.gz

Additionally, the following transforms are saved::

  sub-<subject_label>/
    func/
      sub-<subject_label>_[specifiers]_from-scanner_to-T1w_mode-image_xfm.txt
      sub-<subject_label>_[specifiers]_from-T1w_to-scanner_mode-image_xfm.txt

**Regularly gridded outputs (images)**.
Volumetric output spaces labels (``<space_label>`` above, and in the following) include
``T1w`` and ``MNI152NLin2009cAsym`` (default).

**Surfaces, segmentations and parcellations from FreeSurfer**.
If FreeSurfer reconstructions are used, the ``(aparc+)aseg`` segmentations are aligned to the
subject's T1w space and resampled to the BOLD grid, and the BOLD series are resampled to the
mid-thickness surface mesh::

  sub-<subject_label>/
    func/
      sub-<subject_label>_[specifiers]_space-T1w_desc-aparcaseg_dseg.nii.gz
      sub-<subject_label>_[specifiers]_space-T1w_desc-aseg_dseg.nii.gz
      sub-<subject_label>_[specifiers]_space-<space_label>_hemi-[LR]_bold.func.gii

Surface output spaces include ``fsnative`` (full density subject-specific mesh),
``fsaverage`` and the down-sampled meshes ``fsaverage6`` (41k vertices) and
``fsaverage5`` (10k vertices, default).

**Grayordinates files**.
`CIFTI <https://www.nitrc.org/forum/attachment.php?attachid=333&group_id=454&forum_id=1955>`_ is
a container format that holds both volumetric (regularly sampled in a grid) and surface
(sampled on a triangular mesh) samples.
Sub-cortical time series are sampled on a regular grid derived from one MNI template, while
cortical time series are sampled on surfaces projected from the [Glasser2016]_ template.
If CIFTI outputs are requested (with the ``--cifti-outputs`` argument), the BOLD series are also
saved as ``dtseries.nii`` CIFTI2 files::

  sub-<subject_label>/
    func/
      sub-<subject_label>_[specifiers]_bold.dtseries.nii

CIFTI output resolution can be specified as an optional parameter after ``--cifti-output``.
By default, '91k' outputs are produced and match up to the standard `HCP Pipelines`_ CIFTI
output (91282 grayordinates @ 2mm). However, '170k' outputs are also possible, and produce
higher resolution CIFTI output (170494 grayordinates @ 1.6mm).

**Extracted confounding time series**.
For each :abbr:`BOLD (blood-oxygen level dependent)` run processed with *fMRIPrep*, an
accompanying *confounds* file will be generated.
Confounds_ are saved as a :abbr:`TSV (tab-separated value)` file::

  sub-<subject_label>/
    func/
      sub-<subject_label>_[specifiers]_desc-confounds_timeseries.tsv
      sub-<subject_label>_[specifiers]_desc-confounds_timeseries.json

These :abbr:`TSV (tab-separated values)` tables look like the example below,
where each row of the file corresponds to one time point found in the
corresponding :abbr:`BOLD (blood-oxygen level dependent)` time series::

  csf white_matter  global_signal std_dvars dvars framewise_displacement  t_comp_cor_00 t_comp_cor_01 t_comp_cor_02 t_comp_cor_03 t_comp_cor_04 t_comp_cor_05 a_comp_cor_00 a_comp_cor_01 a_comp_cor_02 a_comp_cor_03 a_comp_cor_04 a_comp_cor_05 non_steady_state_outlier00  trans_x trans_y trans_z rot_x rot_y rot_z
  682.75275 0.0 491.64752000000004  n/a n/a n/a 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 -0.00017029 -0.0  0.0
  669.14166 0.0 489.4421  1.168398  17.575331 0.07211929999999998 -0.4506846719 0.1191909139  -0.0945884724 0.1542023065  -0.2302324641 0.0838194238  -0.032426848599999995 0.4284323184  -0.5809158299 0.1382414008  -0.1203486637 0.3783661265  0.0 0.0 0.0207752 0.0463124 -0.000270924  -0.0  0.0
  665.3969  0.0 488.03  1.085204  16.323903999999995  0.0348966 0.010819676200000001  0.0651895837  -0.09556632150000001  -0.033148835  -0.4768871111 0.20641088559999998 0.2818768463  0.4303863764  0.41323714850000004 -0.2115232212 -0.0037154909000000004  0.10636180070000001 0.0 0.0 0.0 0.0457372 0.0 -0.0  0.0
  662.82715 0.0 487.37302 1.01591 15.281561 0.0333937 0.3328022893  -0.2220965269 -0.0912891436 0.2326688125  0.279138129 -0.111878887  0.16901660629999998 0.0550480212  0.1798747037  -0.25383302620000003  0.1646403629  0.3953613889  0.0 0.010164  -0.0103568  0.0424513 0.0 -0.0  0.00019174

**Multi-echo derivatives**.
For multi-echo datasets, the output ``_bold`` series are "optimally combined" by
`tedana`_ to better estimate the BOLD signal.
This process also generates a T2\* map, which is resampled into every requested output
space.

::

  sub-<subject_label>/
    func/
      sub-<subject_label>_[specifiers]_T2starmap.nii.gz

If the ``--me-output-echos`` flag is specified, then the distortion-corrected (STC, HMC, SDC)
per-echo time series are output. For example, if the inputs are of the form::

  sub-01/
    func/
      sub-01_task-rest_echo-1_bold.nii.gz
      sub-01_task-rest_echo-2_bold.nii.gz
      sub-01_task-rest_echo-3_bold.nii.gz

Then the output will include::

  sub-01/
    func/
      sub-01_task-rest_boldref.nii.gz
      sub-01_task-rest_desc-brain_mask.nii.gz
      sub-01_task-rest_echo-1_desc-preproc_bold.nii.gz
      sub-01_task-rest_echo-2_desc-preproc_bold.nii.gz
      sub-01_task-rest_echo-3_desc-preproc_bold.nii.gz

These may then be used independently with multi-echo tools, such as `tedana`_,
to perform more advanced denoising or alternative combination strategies.

.. danger::
   Slice timing correction in *fMRIPrep* is referenced to the middle slice by default,
   which leads to a time shift in the volume onsets by 0.5 TR (repetition time).
   For example, assuming a TR of 2s, original onsets of 0, 2, and 4s would be shifted
   to 1, 3, and 5s, respectively.
   In case you did execute slice timing correction, you must check that subsequent
   analyses (e.g., general linear modeling) consider the right onset shifts.
   For example, when specifying a first-level model, you should set parameters in your
   software package or first-level model function accordingly (e.g., select the middle
   slice as reference).
   Alternatively, you could manually adjust the volume onsets (e.g. as mentioned in
   the example above from [0, 2, 4] to [1, 3, 5]) or the event onsets accordingly.

   Further information on this issue is found at
   `this blog post (with thanks to Russell Poldrack and Jeanette Mumford)
   <https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/>`__.

Confounds
---------
The :abbr:`BOLD (blood-oxygen level dependent)` signal measured with fMRI is a mixture of fluctuations
of both neuronal and non-neuronal origin.
Neuronal signals are measured indirectly as changes in the local concentration of oxygenated hemoglobin.
Non-neuronal fluctuations in fMRI data may appear as a result of head motion, scanner noise,
or physiological fluctuations (related to cardiac or respiratory effects).
For a detailed review of the possible sources of noise in the BOLD signal, refer to [Greve2013]_.

*Confounds* (or nuisance regressors) are variables representing fluctuations with a potential
non-neuronal origin.
Such non-neuronal fluctuations may drive spurious results in fMRI data analysis,
including standard activation :abbr:`GLM (General Linear Model)` and functional connectivity analyses.
It is possible to minimize confounding effects of non-neuronal signals by including
them as nuisance regressors in the GLM design matrix or regressing them out from
the fMRI data - a procedure known as *denoising*.
There is currently no consensus on an optimal denoising strategy in the fMRI community.
Rather, different strategies have been proposed, which achieve different compromises between
how much of the non-neuronal fluctuations are effectively removed, and how much of neuronal fluctuations
are damaged in the process.
The *fMRIPrep* pipeline generates a large array of possible confounds.

The most well established confounding variables in neuroimaging are the six head-motion parameters
(three rotations and three translations) - the common output of the head-motion correction
(also known as *realignment*) of popular fMRI preprocessing software
such as SPM_ or FSL_.
Beyond the standard head-motion parameters, the fMRIPrep pipeline generates a large array
of possible confounds, which enable researchers to choose the most suitable denoising
strategy for their downstream analyses.

Confounding variables calculated in *fMRIPrep* are stored separately for each subject,
session and run in :abbr:`TSV (tab-separated value)` files - one column for each confound variable.
Such tabular files may include over 100 columns of potential confound regressors.

.. danger::
   Do not include all columns of ``~_desc-confounds_timeseries.tsv`` table
   into your design matrix or denoising procedure.
   Filter the table first, to include only the confounds (or components thereof)
   you want to remove from your fMRI signal.
   The choice of confounding variables may depend on the analysis you want to perform,
   and may be not straightforward as no gold standard procedure exists.
   For a detailed description of various denoising strategies and their performance,
   see [Parkes2018]_ and [Ciric2017]_.

Confound regressors description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Basic confounds**. The most commonly used confounding time series:

- Estimated head-motion parameters:
  ``trans_x``, ``trans_y``, ``trans_z``, ``rot_x``, ``rot_y``, ``rot_z`` - the 6 rigid-body motion
  parameters (3 translations and 3 rotation), estimated relative to a reference image;

- Global signals:

  - ``csf`` - the average signal within anatomically-derived eroded :abbr:`CSF (cerebro-spinal fluid)` mask;
  - ``white_matter`` - the average signal within  the anatomically-derived eroded :abbr:`WM (white matter)` masks;
  - ``global_signal`` -  the average signal within the brain mask.

**Parameter expansion of basic confounds**.
The standard six-motion parameters may not account for all the variance related
to head-motion.
[Friston1996]_ and [Satterthwaite2013]_ proposed an expansion of the six fundamental
head-motion parameters.
To make this technique more accessible, *fMRIPrep* automatically calculates motion parameter
expansion [Satterthwaite2013]_, providing time series corresponding to the first
*temporal derivatives* of the six base motion parameters, together with their
*quadratic terms*, resulting in the total 24 head motion parameters
(six base motion parameters + six temporal derivatives of six motion parameters +
12 quadratic terms of six motion parameters and their six temporal derivatives).
Additionally, *fMRIPrep* returns temporal derivatives and quadratic terms for the
three global signals (``csf``, ``white_matter`` and ``global_signal``)
to enable applying the 36-parameter denoising strategy proposed by [Satterthwaite2013]_.

Derivatives and quadratic terms are stored under column names with
suffixes: ``_derivative1`` and powers ``_power2``.
These are calculated for head-motion estimates (``trans_`` and ``rot_``) and global signals
(``white_matter``, ``csf``, and ``global_signal``).

**Outlier detection**.
These confounds can be used to detect potential outlier time points -
frames with sudden and large motion or intensity spikes.

- ``framewise_displacement`` - is a quantification of the estimated bulk-head motion calculated using
  formula proposed by [Power2012]_;
- ``rmsd`` - is a quantification of the estimated relative (frame-to-frame) bulk head motion
  calculated using the :abbr:`RMS (root mean square)` approach of [Jenkinson2002]_;
- ``dvars`` - the derivative of RMS variance over voxels (or :abbr:`DVARS (derivative of
  RMS variance over voxels)`) [Power2012]_;
- ``std_dvars`` - standardized :abbr:`DVARS (derivative of RMS variance over voxels)`;
- ``non_steady_state_outlier_XX`` - columns indicate non-steady state volumes with a single
  ``1`` value and ``0`` elsewhere (*i.e.*, there is one ``non_steady_state_outlier_XX`` column per
  outlier/volume).

Detected outliers can be further removed from time series using methods such as:
volume *censoring* - entirely discarding problematic time points [Power2012]_,
regressing signal from outlier points in denoising procedure, or
including outlier points in the subsequent first-level analysis when building
the design matrix.
Averaged value of confound (for example, mean ``framewise_displacement``)
can also be added as regressors in group level analysis [Yan2013]_.
*Regressors of motion spikes* for outlier censoring are generated from within *fMRIPrep*,
and their calculation may be adjusted with the command line options ``--fd-spike-threshold``
and ``--dvars-spike-threshold`` (defaults are FD > 0.5 mm or DVARS > 1.5).
Regressors of motion spikes are stored in separate ``motion_outlier_XX`` columns.

**Discrete cosine-basis regressors**.
Physiological and instrumental (scanner) noise sources are generally present in fMRI
data, typically taking the form of low-frequency signal drifts.
To account for these drifts, temporal high-pass filtering is the immediate option.
Alternatively, low-frequency regressors can be included in the statistical model to account
for these confounding signals.
Using the :abbr:`DCT (discrete cosine transform)` basis functions, *fMRIPrep* generates
these low-frequency predictors:

- ``cosine_XX`` - DCT-basis regressors.

One characteristic of the cosine regressors is that they are identical for two different
datasets with the same :abbr:`TR (repetition time)` and the same *effective* number of
time points (*effective* length).
It is relevant to mention *effective* because initial time points identified as *nonsteady
states* are removed before generating the cosine regressors.

.. caution::
    If your analysis includes separate high-pass filtering, do not include
    ``cosine_XX`` regressors in your design matrix.

.. admonition:: See also

    - A detailed explanation about temporal high-pass filtering is provided with
      the `BrainVoyager User Guide
      <https://www.brainvoyager.com/bvqx/doc/UsersGuide/Preprocessing/TemporalHighPassFiltering.html>`_.
    - `This comment
      <https://github.com/nipreps/fmriprep/issues/1899#issuecomment-561687460>`__
      on an issue regarding CompCor regressors.

**CompCor confounds**.
:abbr:`CompCor (Component Based Noise Correction)` is a :abbr:`PCA (principal component analysis)`,
hence component-based, noise pattern recognition method.
In the method, principal components are calculated within an :abbr:`ROI (Region of Interest)`
that is unlikely to include signal related to neuronal activity, such as :abbr:`CSF (cerebro-spinal fluid)`
and :abbr:`WM (white matter)` masks.
Signals extracted from CompCor components can be further regressed out from the fMRI data with a
denoising procedure [Behzadi2007]_.

- ``a_comp_cor_XX`` - additional noise components are calculated using anatomical :abbr:`CompCor
  (Component Based Noise Correction)`;
- ``t_comp_cor_XX`` - additional noise components are calculated using temporal :abbr:`CompCor
  (Component Based Noise Correction)`.

Four separate CompCor decompositions are performed to compute noise components: one temporal
decomposition (``t_comp_cor_XX``) and three anatomical decompositions (``a_comp_cor_XX``) across
three different noise ROIs: an eroded white matter mask, an eroded CSF mask, and a combined mask derived
from the union of these.

Each confounds data file will also have a corresponding metadata file
(``~desc-confounds_regressors.json``).
Metadata files contain additional information about columns in the confounds TSV file:

.. code-block:: json

    {
      "a_comp_cor_00": {
        "CumulativeVarianceExplained": 0.1081970825,
        "Mask": "combined",
        "Method": "aCompCor",
        "Retained": true,
        "SingularValue": 25.8270209974,
        "VarianceExplained": 0.1081970825
      },
      "dropped_0": {
        "CumulativeVarianceExplained": 0.5965809597,
        "Mask": "combined",
        "Method": "aCompCor",
        "Retained": false,
        "SingularValue": 20.7955177198,
        "VarianceExplained": 0.0701465624
      }
    }

For CompCor decompositions, entries include:

  - ``Method``: anatomical or temporal CompCor.
  - ``Mask``: denotes the :abbr:`ROI (region of interest)` where the decomposition that generated
    the component was performed: ``CSF``, ``WM``, or ``combined`` for anatomical CompCor.
  - ``SingularValue``: singular value of the component.
  - ``VarianceExplained``: the fraction of variance explained by the component across the decomposition ROI mask.
  - ``CumulativeVarianceExplained``: the total fraction of variance explained by this particular component
    and all preceding components.
  - ``Retained``: Indicates whether the component was saved in ``desc-confounds_timeseries.tsv``
    for use in denoising.
    Entries that are not saved in the data file for denoising are still stored in metadata with the
    ``dropped`` prefix.

.. caution::
    Only a subset of these CompCor decompositions should be used for further denoising.
    The original Behzadi aCompCor implementation [Behzadi2007]_ can be applied using
    components from the combined masks, while the more recent Muschelli implementation
    [Muschelli2014]_ can be applied using
    the :abbr:`WM (white matter)` and :abbr:`CSF (cerebro-spinal fluid)` masks.
    To determine the provenance of each component, consult the metadata file (described above).

    There are many valid ways of selecting CompCor components for further denoising.
    In general, the components with the largest singular values (i.e., those that
    explain the largest fraction of variance in the data) should be selected.
    *fMRIPrep* outputs components in descending order of singular value.
    Common approaches include selecting a fixed number of components (e.g., the
    first 5 or 6), using a quantitative or qualitative criterion (e.g., elbow, broken
    stick, or condition number), or using sufficiently many components that a minimum
    cumulative fraction of variance is explained (e.g., 50%).

.. caution::
    Similarly, if you are using anatomical or temporal CompCor it may not make sense
    to use the ``csf``, or ``white_matter`` global regressors -
    see `#1049 <https://github.com/nipreps/fmriprep/issues/1049>`_.
    Conversely, using the overall ``global_signal`` confound in addition to CompCor's
    regressors can be beneficial (see [Parkes2018]_).

.. danger::
    *fMRIPrep* does high-pass filtering before running anatomical or temporal CompCor.
    Therefore, when using CompCor regressors, the corresponding ``cosine_XX`` regressors
    should also be included in the design matrix.

.. admonition:: See also

    This didactic `discussion on NeuroStars.org
    <https://neurostars.org/t/fmrirep-outputs-very-high-number-of-acompcors-up-to-1000/5451>`__
    where Patrick Sadil gets into details about PCA and how that base technique applies
    to CompCor in general and *fMRIPrep*'s implementation in particular.

**Confounds estimated from the brain's outer edge**.
Reusing the implementation of aCompCor, *fMRIPrep* generates regressors corresponding to the
24 first principal components extracted with PCA using the voxel time-series delineated by
the brain's outer edge (*crown*) mask.
The procedure essentially follows the initial proposal of the approach by Patriat et al.
[Patriat2017]_ and is described in our ISMRM abstract [Provins2022]_.

Confounds and "carpet"-plot on the visual reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The visual reports provide several sections per task and run to aid designing
a denoising strategy for subsequent analysis.
Some of the estimated confounds are plotted with a "carpet" visualization of the
:abbr:`BOLD (blood-oxygen level-dependent)` time series [Power2016]_.
An example of these plots follows:

.. figure:: _static/sub-405_ses-01_task-rest_run-01_desc-carpetplot_bold.svg

    The figure shows on top several confounds estimated for the BOLD series:
    global signals ('GS', 'GSCSF', 'GSWM'), DVARS,
    and framewise-displacement ('FD').
    At the bottom, a 'carpetplot' summarizing the BOLD series [Power2016]_.
    The carpet plot rows correspond to voxelwise time series,
    and are separated into regions: cortical gray matter, deep
    gray matter, white matter and cerebrospinal fluid, cerebellum
    and the brain-edge or “crown” [Provins2022]_.
    The crown corresponds to the voxels located on a
    closed band around the brain [Patriat2015]_.

Noise components computed during each CompCor decomposition are evaluated according
to the fraction of variance that they explain across the nuisance ROI.
This is used by *fMRIPrep* to determine whether each component should be saved for
use in denoising operations: a component is saved if it contributes to explaining
the top 50 percent of variance in the nuisance ROI.
*fMRIPrep* can be configured to save all components instead using the command line
option ``--return-all-components``.
*fMRIPrep* reports include a plot of the cumulative variance explained by each
component, ordered by descending singular value.

.. figure:: _static/sub-01_task-rest_compcor.svg

    The figure displays the cumulative variance explained by components for each
    of four CompCor decompositions (left to right: anatomical CSF mask, anatomical
    white matter mask, anatomical combined mask, temporal).
    The number of components is plotted on the abscissa and
    the cumulative variance explained on the ordinate.
    Dotted lines indicate the minimum number of components necessary
    to explain 50%, 70%, and 90% of the variance in the nuisance mask.
    By default, only the components that explain the top 50% of the variance
    are saved.

Also included is a plot of correlations among confound regressors.
This can be used to guide selection of a confound model or to assess the extent
to which tissue-specific regressors correlate with global signal.

.. figure:: _static/sub-01_task-mixedgamblestask_run-01_confounds_correlation.svg

    The left-hand panel shows the matrix of correlations among selected confound
    time series as a heat-map.
    Note the zero-correlation blocks near the diagonal; these correspond to each
    CompCor decomposition.
    The right-hand panel displays the correlation of selected confound time series
    with the mean global signal computed across the whole brain; the regressors shown
    are those with greatest correlation with the global signal.
    This information can be used to diagnose partial volume effects.

See implementation on :mod:`~fmriprep.workflows.bold.confounds.init_bold_confs_wf`.

Legacy layout
-------------

Prior to fMRIPrep 21.0, the following organizational structure was used::

    <output_dir>/
      fmriprep/
      freesurfer/

Although this has the advantage of keeping all outputs together,
it ensured that the output of fMRIPrep could not itself be a BIDS derivative dataset,
only contain one.

To restore this behavior, use the ``--output-layout legacy`` command-line option.

The BIDS and legacy layouts are otherwise the same in all respects.
It is thus possible to achieve identical results with the BIDS layout by using
the following invocation::

    fmriprep <input_dir>/ <output_dir>/fmriprep/ participant \
        --fs-subjects-dir <output_dir>/freesurfer/ [OPTIONS]


.. topic:: References

  .. [Behzadi2007] Behzadi Y, Restom K, Liau J, Liu TT, A component-based noise correction method
     (CompCor) for BOLD and perfusion-based fMRI. NeuroImage. 2007.
     doi:`10.1016/j.neuroimage.2007.04.042 <http://doi.org/10.1016/j.neuroimage.2007.04.042>`_

  .. [Ciric2017] Ciric R, Wolf DH, Power JD, Roalf DR, Baum GL, Ruparel K, Shinohara RT, Elliott MA,
     Eickhoff SB, Davatzikos C., Gur RC, Gur RE, Bassett DS, Satterthwaite TD.
     Benchmarking of participant-level confound regression strategies for the control of motion
     artifact in studies of functional connectivity. Neuroimage. 2017.
     doi:`10.1016/j.neuroimage.2017.03.020 <https://doi.org/10.1016/j.neuroimage.2017.03.020>`_

  .. [Greve2013] Greve DN, Brown GG, Mueller BA, Glover G, Liu TT,
     A Survey of the Sources of Noise in fMRI. Psychometrika. 2013.
     doi:`10.1007/s11336-013-9344-2 <http://dx.doi.org/10.1007/s11336-013-9344-2>`_

  .. [Friston1996] Friston KJ1, Williams S, Howard R, Frackowiak RS, Turner R,
     Movement‐Related effects in fMRI time‐series. Magnetic Resonance in Medicine. 1996.
     doi:`10.1002/mrm.191035031 <https://doi.org/10.1002/mrm.1910350312>`_

  .. [Glasser2016] Glasser MF, Coalson TS Robinson EC, Hacker CD, Harwell J, Yacoub E, Ugurbil K,
     Andersson J, Beckmann CF, Jenkinson M, Smith SM, Van Essen DC.
     A multi-modal parcellation of human cerebral cortex. Nature. 2016.
     doi:`10.1038/nature18933 <https://doi.org/10.1038/nature18933>`_

  .. [Hallquist2013] Hallquist MN, Hwang K, Luna B. The Nuisance of Nuisance Regression.
     NeuroImage. 2013. doi:`10.1016/j.neuroimage.2013.05.116
     <https://doi.org/10.1016/j.neuroimage.2013.05.116>`_

  .. [Jenkinson2002] Jenkinson M, Bannister P, Brady M, Smith S. Improved optimization for the
     robust and accurate linear registration and motion correction of brain images. Neuroimage.
     2002. doi:`10.1016/s1053-8119(02)91132-8 <https://doi.org/10.1016/s1053-8119(02)91132-8>`__.

  .. [Lindquist2019] Lindquist, MA, Geuter, S, and Wager, TD, Caffo, BS,
     Modular preprocessing pipelines can reintroduce artifacts into fMRI data.
     Human Brain Mapping. 2019. doi: `10.1002/hbm.24528
     <https://doi.org/10.1002/hbm.24528>`_

  .. [Muschelli2014] Muschelli J, Nebel MB, Caffo BS, Barber AD, Pekar JJ, Mostofsky SH,
     Reduction of motion-related artifacts in resting state fMRI using aCompCor. NeuroImage. 2014.
     doi:`10.1016/j.neuroimage.2014.03.028 <http://doi.org/10.1016/j.neuroimage.2014.03.028>`_

  .. [Parkes2018] Parkes L, Fulcher B, Yücel M, Fornito A, An evaluation of the efficacy, reliability,
     and sensitivity of motion correction strategies for resting-state functional MRI. NeuroImage. 2018.
     doi:`10.1016/j.neuroimage.2017.12.073 <https://doi.org/10.1016/j.neuroimage.2017.12.073>`_

  .. [Patriat2015] Patriat R, EK Molloy, RM Birn, T. Guitchev, and A. Popov. ,Using Edge Voxel Information to
     Improve Motion Regression for Rs-FMRI Connectivity Studies. Brain Connectivity. 2015.
     doi:`10.1089/brain.2014.0321 <https://doi.org/10.1089/brain.2014.0321>`_.

  .. [Patriat2017] Patriat R, Reynolds RC, Birn RM, An improved model of motion-related signal
     changes in fMRI. NeuroImage. 2017.
     doi:`10.1016/j.neuroimage.2016.08.051 <https://doi.org/10.1016/j.neuroimage.2016.08.051>`_.

  .. [Power2012] Power JD, Barnes KA, Snyder AZ, Schlaggar BL, Petersen, SA, Spurious but systematic
     correlations in functional connectivity MRI networks arise from subject motion. NeuroImage. 2012.
     doi:`10.1016/j.neuroimage.2011.10.018 <https://doi.org/10.1016/j.neuroimage.2011.10.018>`_

  .. [Power2016] Power JD, A simple but useful way to assess fMRI scan qualities. NeuroImage. 2016.
     doi:`10.1016/j.neuroimage.2016.08.009 <http://doi.org/10.1016/j.neuroimage.2016.08.009>`_

  .. [Provins2022] Provins C et al., Quality control and nuisance regression of fMRI, looking out
     where signal should not be found. Proc. Intl. Soc. Mag. Reson. Med. 31, London (UK). 2022
     doi:`10.31219/osf.io/hz52v <https://doi.org/10.31219/osf.io/hz52v>`_.

  .. [Satterthwaite2013] Satterthwaite TD, Elliott MA, Gerraty RT, Ruparel K, Loughead J, Calkins ME,
     Eickhoff SB, Hakonarson H, Gur RC, Gur RE, Wolf DH,
     An improved framework for confound regression and filtering for control of motion artifact
     in the preprocessing of resting-state functional connectivity data. NeuroImage. 2013. doi:`10.1016/j.neuroimage.2012.08.052 <https://doi.org/10.1016/j.neuroimage.2012.08.052>`_

  .. [Yan2013] Yan CG, Cheung B, Kelly C, Colcombe S, Craddock RC, Di Martino A, Li Q, Zuo XN,
     Castellanos FX, Milham MP, A comprehensive assessment of regional variation in the impact of
     head micromovements on functional connectomics. NeuroImage. 2013.
     doi:`10.1016/j.neuroimage.2013.03.004 <https://doi.org/10.1016/j.neuroimage.2013.03.004>`_
