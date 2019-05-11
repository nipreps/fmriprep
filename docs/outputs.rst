.. include:: links.rst

.. _outputs:

-------------------
Outputs of fMRIPrep
-------------------

FMRIPrep generates three broad classes of outcomes:

  1. **Visual QA (quality assessment) reports**:
     one :abbr:`HTML (hypertext markup language)` per subject,
     that allows the user a thorough visual assessment of the quality
     of processing and ensures the transparency of fMRIPrep operation.

  2. **Pre-processed imaging data** which are derivatives of the original
     anatomical and functional images after various preparation procedures
     have been applied. For example,
     :abbr:`INU (intensity non-uniformity)`-corrected versions of the T1-weighted
     image (per subject), the brain mask, or :abbr:`BOLD (blood-oxygen level dependent)`
     images after head-motion correction, slice-timing correction and aligned into
     the same-subject's T1w space or into MNI space.

  3. **Additional data for subsequent analysis**, for instance the transformations
     between different spaces or the estimated confounds.


fMRIPrep outputs conform to the :abbr:`BIDS (brain imaging data structure)`
Derivatives specification (see `BIDS Derivatives RC1`_).

Visual Reports
--------------

FMRIPrep outputs summary reports, written to ``<output dir>/fmriprep/sub-<subject_label>.html``.
These reports provide a quick way to make visual inspection of the results easy.
Each report is self contained and thus can be easily shared with collaborators (for example via email).
`View a sample report. <_static/sample_report.html>`_


Preprocessed data (fMRIPrep *derivatives*)
------------------------------------------

Preprocessed, or derivative, data are written to
``<output dir>/fmriprep/sub-<subject_label>/``.
The `BIDS Derivatives RC1`_ specification describes the naming and metadata conventions we follow.

Anatomical derivatives are placed in each subject's ``anat`` subfolder:

- ``anat/sub-<subject_label>_[space-<space_label>_]desc-preproc_T1w.nii.gz``
- ``anat/sub-<subject_label>_[space-<space_label>_]desc-brain_mask.nii.gz``
- ``anat/sub-<subject_label>_[space-<space_label>_]dseg.nii.gz``
- ``anat/sub-<subject_label>_[space-<space_label>_]label-CSF_probseg.nii.gz``
- ``anat/sub-<subject_label>_[space-<space_label>_]label-GM_probseg.nii.gz``
- ``anat/sub-<subject_label>_[space-<space_label>_]label-WM_probseg.nii.gz``

Template-normalized derivatives use the space label ``MNI152NLin2009cAsym``, while derivatives in
the original ``T1w`` space omit the ``space-`` keyword.

Additionally, the following transforms are saved:

- ``anat/sub-<subject_label>_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5`` 
- ``anat/sub-<subject_label>_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5`` 

If FreeSurfer reconstructions are used, the following surface files are generated:

- ``anat/sub-<subject_label>_hemi-[LR]_smoothwm.surf.gii``
- ``anat/sub-<subject_label>_hemi-[LR]_pial.surf.gii``
- ``anat/sub-<subject_label>_hemi-[LR]_midthickness.surf.gii``
- ``anat/sub-<subject_label>_hemi-[LR]_inflated.surf.gii``

And the affine translation between ``T1w`` space and FreeSurfer's reconstruction (``fsnative``) is
stored in:

- ``anat/sub-<subject_label>_from-T1w_to-fsnative_mode-image_xfm.txt``

Functional derivatives are stored in the ``func`` subfolder.
All derivatives contain ``task-<task_label>`` (mandatory) and ``run-<run_index>`` (optional), and
these will be indicated with ``[specifiers]``.

- ``func/sub-<subject_label>_[specifiers]_space-<space_label>_boldref.nii.gz``
- ``func/sub-<subject_label>_[specifiers]_space-<space_label>_desc-brain_mask.nii.gz``
- ``func/sub-<subject_label>_[specifiers]_space-<space_label>_desc-preproc_bold.nii.gz``

Volumetric output spaces include ``T1w`` and ``MNI152NLin2009cAsym`` (default).

Confounds are saved as a :abbr:`TSV (tab-separated value)` file:

- ``func/sub-<subject_label>_[specifiers]_desc-confounds_regressors.nii.gz``

If FreeSurfer reconstructions are used, the ``(aparc+)aseg`` segmentations are aligned to the
subject's T1w space and resampled to the BOLD grid, and the BOLD series are resampled to the
midthickness surface mesh:

- ``func/sub-<subject_label>_[specifiers]_space-T1w_desc-aparcaseg_dseg.nii.gz``
- ``func/sub-<subject_label>_[specifiers]_space-T1w_desc-aseg_dseg.nii.gz``
- ``func/sub-<subject_label>_[specifiers]_space-<space_label>_hemi-[LR].func.gii``

Surface output spaces include ``fsnative`` (full density subject-specific mesh),
``fsaverage`` and the down-sampled meshes ``fsaverage6`` (41k vertices) and
``fsaverage5`` (10k vertices, default).

If CIFTI outputs are requested, the BOLD series is also saved as ``dtseries.nii`` CIFTI2 files:

- ``func/sub-<subject_label>_[specifiers]_bold.dtseries.nii``

Sub-cortical time series are volumetric (supported spaces: ``MNI152NLin2009cAsym``), while cortical
time series are sampled to surface (supported spaces: ``fsaverage5``, ``fsaverage6``)

Finally, if ICA-AROMA is used, the MELODIC mixing matrix and the components classified as noise
are saved:

- ``func/sub-<subject_label>_[specifiers]_AROMAnoiseICs.csv``
- ``func/sub-<subject_label>_[specifiers]_desc-MELODIC_mixing.tsv``


.. _fsderivs:

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



Confounds
---------

See implementation on :mod:`~fmriprep.workflows.bold.confounds.init_bold_confs_wf`.


For each :abbr:`BOLD (blood-oxygen level dependent)` run processed with fMRIPrep, a
``<output_folder>/fmriprep/sub-<sub_id>/func/sub-<sub_id>_task-<task_id>_run-<run_id>_desc-confounds_regressors.tsv``
file will be generated.
These are :abbr:`TSV (tab-separated values)` tables, which look like the example below: ::

  csf	white_matter	global_signal	std_dvars	dvars	framewise_displacement	t_comp_cor_00	t_comp_cor_01	t_comp_cor_02	t_comp_cor_03	t_comp_cor_04	t_comp_cor_05	a_comp_cor_00	a_comp_cor_01	a_comp_cor_02	a_comp_cor_03	a_comp_cor_04	a_comp_cor_05	non_steady_state_outlier00	trans_x	trans_y	trans_z	rot_x	rot_y	rot_z	aroma_motion_02	aroma_motion_04
  682.75275	0.0	491.64752000000004	n/a	n/a	n/a	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	-0.00017029	-0.0	0.0	0.0	0.0
  669.14166	0.0	489.4421	1.168398	17.575331	0.07211929999999998	-0.4506846719	0.1191909139	-0.0945884724	0.1542023065	-0.2302324641	0.0838194238	-0.032426848599999995	0.4284323184	-0.5809158299	0.1382414008	-0.1203486637	0.3783661265	0.0	0.0	0.0207752	0.0463124	-0.000270924	-0.0	0.0	-2.402958171	-0.7574011893
  665.3969	0.0	488.03	1.085204	16.323903999999995	0.0348966	0.010819676200000001	0.0651895837	-0.09556632150000001	-0.033148835	-0.4768871111	0.20641088559999998	0.2818768463	0.4303863764	0.41323714850000004	-0.2115232212	-0.0037154909000000004	0.10636180070000001	0.0	0.0	0.0	0.0457372	0.0	-0.0	0.0	-1.341359143	0.1636017242
  662.82715	0.0	487.37302	1.01591	15.281561	0.0333937	0.3328022893	-0.2220965269	-0.0912891436	0.2326688125	0.279138129	-0.111878887	0.16901660629999998	0.0550480212	0.1798747037	-0.25383302620000003	0.1646403629	0.3953613889	0.0	0.010164	-0.0103568	0.0424513	0.0	-0.0	0.00019174	-0.1554834655	0.6451987913


Each row of the file corresponds to one time point found in the
corresponding :abbr:`BOLD (blood-oxygen level dependent)` time-series
(stored in ``<output_folder>/fmriprep/sub-<sub_id>/func/sub-<sub_id>_task-<task_id>_run-<run_id>_desc-preproc_bold.nii.gz``).

Columns represent the different confounds: ``csf`` and ``white_matter`` are the average signal
inside the anatomically-derived :abbr:`CSF (cerebro-spinal fluid)` and :abbr:`WM (white matter)`
masks across time;
``global_signal`` corresponds to the mean time series within the brain mask; two columns relate to
the derivative of RMS variance over voxels (or :abbr:`DVARS (defined in Power, et al. 2012)`), and
both the original (``dvars``) and standardized (``std_dvars``) are provided;
``framewise_displacement`` is a quantification of the estimated bulk-head motion;
``trans_x``, ``trans_y``, ``trans_z``, ``rot_x``, ``rot_y``, ``rot_z`` are the 6 rigid-body
motion-correction parameters estimated by fMRIPrep;
if present, ``non_steady_state_outlier_XX`` columns indicate non-steady state volumes with a single
``1`` value and ``0`` elsewhere (*i.e.*, there is one ``non_steady_state_outlier_XX`` column per
outlier/volume);
additional noise components are calculated using :abbr:`CompCor (Component Based Noise Correction)`,
according to both the anatomical (``a_comp_cor_XX``) and temporal (``t_comp_cor_XX``) variants;
and the motion-related components identified by
:abbr:`ICA (independent components analysis)`-:abbr:`AROMA (Automatic Removal Of Motion Artifacts)`
(if enabled) are indicated with ``aroma_motion_XX``.

Four separate CompCor decompositions are performed to compute noise components: one temporal
decomposition and three anatomical decompositions across three different noise ROIs: an eroded
white matter compartment, an eroded CSF compartment, and a combined mask derived from the union
of these.
In general, only a subset of these decompositions should be used for further denoising.
The original Behzadi aCompCor implementation ([Behzadi2007]_) can be applied using
components from the combined ROI, while the more recent Muschelli implementation
([Muschelli2014]_) can be applied using the WM and CSF ROIs.
To determine the provenance of each component, consult the metadata file (see below).

All these confounds can be used to perform *scrubbing* and *censoring* of outliers,
in the subsequent first-level analysis when building the design matrix,
and in group level analysis.
*Spike regressors* for outlier censoring can also be generated from within fMRIPrep using
the command line options ``--fd-spike-threshold`` and ``--dvars-spike-threshold``.
Spike regressors are stored in separate ``motion_outlier_XX`` columns.

Each confounds data file will also have a corresponding metadata file (``~desc-confounds_regressors.json``).
Metadata files contain additional information about columns in the confounds TSV file: ::

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
  - ``Mask``: denotes the ROI where the decomposition that generated the component
    was performed: ``CSF``, ``WM``, or ``combined`` for anatomical CompCor.
  - ``SingularValue``: singular value of the component.
  - ``VarianceExplained``: the fraction of variance explained by the component across the decomposition ROI mask.
  - ``CumulativeVarianceExplained``: the total fraction of variance explained by this particular component
    and all preceding components.
  - ``Retained``: Indicates whether the component was saved in ``desc-confounds_regressors.tsv``
    for use in denoising.
    Entries that are not saved in the data file for denoising are still stored in metadata with the
    ``dropped`` prefix.

Confounds and "carpet"-plot on the visual reports
-------------------------------------------------

Some of the estimated confounds, as well as a "carpet" visualization of the
:abbr:`BOLD (blood-oxygen level-dependent)` time-series (see [Power2016]_).
This plot is included for each run within the corresponding visual report.
An example of these plots follows:


.. figure:: _static/sub-01_task-mixedgamblestask_run-01_bold_carpetplot.svg
    :scale: 100%

    The figure shows on top several confounds estimated for the BOLD series:
    global signals ('GlobalSignal', 'WM', 'GM'), standardized DVARS ('stdDVARS'),
    and framewise-displacement ('FramewiseDisplacement').
    At the bottom, a 'carpetplot' summarizing the BOLD series.
    The colormap on the left-side of the carpetplot denotes signals located
    in cortical gray matter regions (blue), subcortical gray matter (orange),
    cerebellum (green) and the union of white-matter and CSF compartments (red).

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
    :scale: 100%

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
    :scale: 100%

    The left-hand panel shows the matrix of correlations among selected confound
    time series as a heatmap.
    Note the zero-correlation blocks near the diagonal; these correspond to each
    CompCor decomposition.
    The right-hand panel displays the correlation of selected confound time series
    with the mean global signal computed across the whole brain; the regressors shown
    are those with greatest correlation with the global signal.
    This information can be used to diagnose partial volume effects.

.. topic:: References

  .. [Behzadi2007] Behzadi Y, Restom K, Liau J, Liu TT,
     A component-based noise correction method (CompCor) for BOLD and perfusion-based fMRI.
     NeuroImage. 2007. doi: `10.1016/j.neuroimage.2007.04.042 <http://doi.org/10.1016/j.neuroimage.2007.04.042>`_

  .. [Muschelli2014] Muschelli J, Nebel MB, Caffo BS, Barber AD, Pekar JJ, Mostofsky SH,
     Reduction of motion-related artifacts in resting state fMRI using aCompCor.
     NeuroImage. 2014. doi: `10.1016/j.neuroimage.2014.03.028 <http://doi.org/10.1016/j.neuroimage.2014.03.028>`_

  .. [Power2016] Power JD, A simple but useful way to assess fMRI scan qualities.
     NeuroImage. 2016. doi: `10.1016/j.neuroimage.2016.08.009 <http://doi.org/10.1016/j.neuroimage.2016.08.009>`_
