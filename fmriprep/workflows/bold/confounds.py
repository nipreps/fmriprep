# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Calculate BOLD confounds
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_confs_wf
.. autofunction:: init_ica_aroma_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl
from nipype.algorithms import confounds as nac

from templateflow.api import get as get_template
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.confounds import ExpandModel, SpikeRegressors
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from niworkflows.interfaces.images import SignalExtraction
from niworkflows.interfaces.masks import ROIsPlot
from niworkflows.interfaces.utility import KeySelect
from niworkflows.interfaces.patches import (
    RobustACompCor as ACompCor,
    RobustTCompCor as TCompCor,
)
from niworkflows.interfaces.plotting import (
    CompCorVariancePlot, ConfoundsCorrelationPlot
)
from niworkflows.interfaces.segmentation import ICA_AROMARPT
from niworkflows.interfaces.utils import (
    TPM2ROI, AddTPMs, AddTSVHeader, TSV2JSON, DictMerge
)

from ...interfaces import (
    GatherConfounds, ICAConfounds,
    FMRISummary, DerivativesDataSink
)

DEFAULT_MEMORY_MIN_GB = 0.01


def init_bold_confs_wf(
    mem_gb,
    metadata,
    regressors_all_comps,
    regressors_dvars_th,
    regressors_fd_th,
    name="bold_confs_wf",
):
    """
    This workflow calculates confounds for a BOLD series, and aggregates them
    into a :abbr:`TSV (tab-separated value)` file, for use as nuisance
    regressors in a :abbr:`GLM (general linear model)`.

    The following confounds are calculated, with column headings in parentheses:

    #. Region-wise average signal (``csf``, ``white_matter``, ``global_signal``)
    #. DVARS - original and standardized variants (``dvars``, ``std_dvars``)
    #. Framewise displacement, based on head-motion parameters
       (``framewise_displacement``)
    #. Temporal CompCor (``t_comp_cor_XX``)
    #. Anatomical CompCor (``a_comp_cor_XX``)
    #. Cosine basis set for high-pass filtering w/ 0.008 Hz cut-off
       (``cosine_XX``)
    #. Non-steady-state volumes (``non_steady_state_XX``)
    #. Estimated head-motion parameters, in mm and rad
       (``trans_x``, ``trans_y``, ``trans_z``, ``rot_x``, ``rot_y``, ``rot_z``)


    Prior to estimating aCompCor and tCompCor, non-steady-state volumes are
    censored and high-pass filtered using a :abbr:`DCT (discrete cosine
    transform)` basis.
    The cosine basis, as well as one regressor per censored volume, are included
    for convenience.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold.confounds import init_bold_confs_wf
        wf = init_bold_confs_wf(
            mem_gb=1,
            metadata={},
            regressors_all_comps=False,
            regressors_dvars_th=1.5,
            regressors_fd_th=0.5,
        )

    **Parameters**

        mem_gb : float
            Size of BOLD file in GB - please note that this size
            should be calculated after resamplings that may extend
            the FoV
        metadata : dict
            BIDS metadata for BOLD file
        name : str
            Name of workflow (default: ``bold_confs_wf``)
        regressors_all_comps: bool
            Indicates whether CompCor decompositions should return all
            components instead of the minimal number of components necessary
            to explain 50 percent of the variance in the decomposition mask.
        regressors_dvars_th
            Criterion for flagging DVARS outliers
        regressors_fd_th
            Criterion for flagging framewise displacement outliers


    **Inputs**

        bold
            BOLD image, after the prescribed corrections (STC, HMC and SDC)
            when available.
        bold_mask
            BOLD series mask
        movpar_file
            SPM-formatted motion parameters file
        skip_vols
            number of non steady state volumes
        t1_mask
            Mask of the skull-stripped template image
        t1_tpms
            List of tissue probability maps in T1w space
        t1_bold_xform
            Affine matrix that maps the T1w space into alignment with
            the native BOLD space

    **Outputs**

        confounds_file
            TSV of all aggregated confounds
        rois_report
            Reportlet visualizing white-matter/CSF mask used for aCompCor,
            the ROI for tCompCor and the BOLD brain mask.
        confounds_metadata
            Confounds metadata dictionary.

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Several confounding time-series were calculated based on the
*preprocessed BOLD*: framewise displacement (FD), DVARS and
three region-wise global signals.
FD and DVARS are calculated for each functional run, both using their
implementations in *Nipype* [following the definitions by @power_fd_dvars].
The three global signals are extracted within the CSF, the WM, and
the whole-brain masks.
Additionally, a set of physiological regressors were extracted to
allow for component-based noise correction [*CompCor*, @compcor].
Principal components are estimated after high-pass filtering the
*preprocessed BOLD* time-series (using a discrete cosine filter with
128s cut-off) for the two *CompCor* variants: temporal (tCompCor)
and anatomical (aCompCor).
tCompCor components are then calculated from the top 5% variable
voxels within a mask covering the subcortical regions.
This subcortical mask is obtained by heavily eroding the brain mask,
which ensures it does not include cortical GM regions.
For aCompCor, components are calculated within the intersection of
the aforementioned mask and the union of CSF and WM masks calculated
in T1w space, after their projection to the native space of each
functional run (using the inverse BOLD-to-T1w transformation). Components
are also calculated separately within the WM and CSF masks.
For each CompCor decomposition, the *k* components with the largest singular
values are retained, such that the retained components' time series are
sufficient to explain 50 percent of variance across the nuisance mask (CSF,
WM, combined, or temporal). The remaining components are dropped from
consideration.
The head-motion estimates calculated in the correction step were also
placed within the corresponding confounds file.
The confound time series derived from head motion estimates and global
signals were expanded with the inclusion of temporal derivatives and
quadratic terms for each [@confounds_satterthwaite_2013].
Frames that exceeded a threshold of {fd} mm FD or {dv} standardised DVARS
were annotated as motion outliers.
""".format(fd=regressors_fd_th, dv=regressors_dvars_th)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold', 'bold_mask', 'movpar_file', 'skip_vols',
                't1_mask', 't1_tpms', 't1_bold_xform']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['confounds_file', 'confounds_metadata']),
        name='outputnode')

    # Get masks ready in T1w space
    acc_tpm = pe.Node(AddTPMs(indices=[0, 2]), name='tpms_add_csf_wm')  # acc stands for aCompCor
    csf_roi = pe.Node(TPM2ROI(erode_mm=0, mask_erode_mm=30), name='csf_roi')
    wm_roi = pe.Node(TPM2ROI(
        erode_prop=0.6, mask_erode_prop=0.6**3),  # 0.6 = radius; 0.6^3 = volume
        name='wm_roi')
    acc_roi = pe.Node(TPM2ROI(
        erode_prop=0.6, mask_erode_prop=0.6**3),  # 0.6 = radius; 0.6^3 = volume
        name='acc_roi')

    # Map ROIs in T1w space into BOLD space
    csf_tfm = pe.Node(ApplyTransforms(interpolation='NearestNeighbor', float=True),
                      name='csf_tfm', mem_gb=0.1)
    wm_tfm = pe.Node(ApplyTransforms(interpolation='NearestNeighbor', float=True),
                     name='wm_tfm', mem_gb=0.1)
    acc_tfm = pe.Node(ApplyTransforms(interpolation='NearestNeighbor', float=True),
                      name='acc_tfm', mem_gb=0.1)
    tcc_tfm = pe.Node(ApplyTransforms(interpolation='NearestNeighbor', float=True),
                      name='tcc_tfm', mem_gb=0.1)

    # Ensure ROIs don't go off-limits (reduced FoV)
    csf_msk = pe.Node(niu.Function(function=_maskroi), name='csf_msk')
    wm_msk = pe.Node(niu.Function(function=_maskroi), name='wm_msk')
    acc_msk = pe.Node(niu.Function(function=_maskroi), name='acc_msk')
    tcc_msk = pe.Node(niu.Function(function=_maskroi), name='tcc_msk')

    # DVARS
    dvars = pe.Node(nac.ComputeDVARS(save_nstd=True, save_std=True, remove_zerovariance=True),
                    name="dvars", mem_gb=mem_gb)

    # Frame displacement
    fdisp = pe.Node(nac.FramewiseDisplacement(parameter_source="SPM"),
                    name="fdisp", mem_gb=mem_gb)

    # a/t-CompCor
    mrg_lbl_cc = pe.Node(niu.Merge(3), name='merge_rois_cc', run_without_submitting=True)

    tcompcor = pe.Node(
        TCompCor(components_file='tcompcor.tsv', header_prefix='t_comp_cor_', pre_filter='cosine',
                 save_pre_filter=True, save_metadata=True, percentile_threshold=.05,
                 failure_mode='NaN'),
        name="tcompcor", mem_gb=mem_gb)

    acompcor = pe.Node(
        ACompCor(components_file='acompcor.tsv', header_prefix='a_comp_cor_', pre_filter='cosine',
                 save_pre_filter=True, save_metadata=True, mask_names=['combined', 'CSF', 'WM'],
                 merge_method='none', failure_mode='NaN'),
        name="acompcor", mem_gb=mem_gb)

    # Set number of components
    if regressors_all_comps:
        acompcor.inputs.num_components = 'all'
        tcompcor.inputs.num_components = 'all'
    else:
        acompcor.inputs.variance_threshold = 0.5
        tcompcor.inputs.variance_threshold = 0.5

    # Set TR if present
    if 'RepetitionTime' in metadata:
        tcompcor.inputs.repetition_time = metadata['RepetitionTime']
        acompcor.inputs.repetition_time = metadata['RepetitionTime']

    # Global and segment regressors
    mrg_lbl = pe.Node(niu.Merge(3), name='merge_rois', run_without_submitting=True)
    signals = pe.Node(SignalExtraction(class_labels=["csf", "white_matter", "global_signal"]),
                      name="signals", mem_gb=mem_gb)

    # Arrange confounds
    add_dvars_header = pe.Node(
        AddTSVHeader(columns=["dvars"]),
        name="add_dvars_header", mem_gb=0.01, run_without_submitting=True)
    add_std_dvars_header = pe.Node(
        AddTSVHeader(columns=["std_dvars"]),
        name="add_std_dvars_header", mem_gb=0.01, run_without_submitting=True)
    add_motion_headers = pe.Node(
        AddTSVHeader(columns=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]),
        name="add_motion_headers", mem_gb=0.01, run_without_submitting=True)
    concat = pe.Node(GatherConfounds(), name="concat", mem_gb=0.01, run_without_submitting=True)

    # CompCor metadata
    tcc_metadata_fmt = pe.Node(
        TSV2JSON(index_column='component', drop_columns=['mask'], output=None,
                 additional_metadata={'Method': 'tCompCor'}, enforce_case=True),
        name='tcc_metadata_fmt')
    acc_metadata_fmt = pe.Node(
        TSV2JSON(index_column='component', output=None,
                 additional_metadata={'Method': 'aCompCor'}, enforce_case=True),
        name='acc_metadata_fmt')
    mrg_conf_metadata = pe.Node(niu.Merge(2), name='merge_confound_metadata',
                                run_without_submitting=True)
    mrg_conf_metadata2 = pe.Node(DictMerge(), name='merge_confound_metadata2',
                                 run_without_submitting=True)

    # Expand model to include derivatives and quadratics
    model_expand = pe.Node(ExpandModel(
        model_formula='(dd1(rps + wm + csf + gsr))^^2 + others'),
        name='model_expansion')

    # Add spike regressors
    spike_regress = pe.Node(SpikeRegressors(
        fd_thresh=regressors_fd_th,
        dvars_thresh=regressors_dvars_th),
        name='spike_regressors')

    # Generate reportlet (ROIs)
    mrg_compcor = pe.Node(niu.Merge(2), name='merge_compcor', run_without_submitting=True)
    rois_plot = pe.Node(ROIsPlot(colors=['b', 'magenta'], generate_report=True),
                        name='rois_plot', mem_gb=mem_gb)

    ds_report_bold_rois = pe.Node(
        DerivativesDataSink(desc='rois', keep_dtype=True),
        name='ds_report_bold_rois', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    # Generate reportlet (CompCor)
    mrg_cc_metadata = pe.Node(niu.Merge(2), name='merge_compcor_metadata',
                              run_without_submitting=True)
    compcor_plot = pe.Node(
        CompCorVariancePlot(variance_thresholds=(0.5, 0.7, 0.9),
                            metadata_sources=['tCompCor', 'aCompCor']),
        name='compcor_plot')
    ds_report_compcor = pe.Node(
        DerivativesDataSink(desc='compcorvar', keep_dtype=True),
        name='ds_report_compcor', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    # Generate reportlet (Confound correlation)
    conf_corr_plot = pe.Node(
        ConfoundsCorrelationPlot(reference_column='global_signal', max_dim=70),
        name='conf_corr_plot')
    ds_report_conf_corr = pe.Node(
        DerivativesDataSink(desc='confoundcorr', keep_dtype=True),
        name='ds_report_conf_corr', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    def _pick_csf(files):
        return files[0]

    def _pick_wm(files):
        return files[-1]

    workflow.connect([
        # Massage ROIs (in T1w space)
        (inputnode, acc_tpm, [('t1_tpms', 'in_files')]),
        (inputnode, csf_roi, [(('t1_tpms', _pick_csf), 'in_tpm'),
                              ('t1_mask', 'in_mask')]),
        (inputnode, wm_roi, [(('t1_tpms', _pick_wm), 'in_tpm'),
                             ('t1_mask', 'in_mask')]),
        (inputnode, acc_roi, [('t1_mask', 'in_mask')]),
        (acc_tpm, acc_roi, [('out_file', 'in_tpm')]),
        # Map ROIs to BOLD
        (inputnode, csf_tfm, [('bold_mask', 'reference_image'),
                              ('t1_bold_xform', 'transforms')]),
        (csf_roi, csf_tfm, [('roi_file', 'input_image')]),
        (inputnode, wm_tfm, [('bold_mask', 'reference_image'),
                             ('t1_bold_xform', 'transforms')]),
        (wm_roi, wm_tfm, [('roi_file', 'input_image')]),
        (inputnode, acc_tfm, [('bold_mask', 'reference_image'),
                              ('t1_bold_xform', 'transforms')]),
        (acc_roi, acc_tfm, [('roi_file', 'input_image')]),
        (inputnode, tcc_tfm, [('bold_mask', 'reference_image'),
                              ('t1_bold_xform', 'transforms')]),
        (csf_roi, tcc_tfm, [('eroded_mask', 'input_image')]),
        # Mask ROIs with bold_mask
        (inputnode, csf_msk, [('bold_mask', 'in_mask')]),
        (inputnode, wm_msk, [('bold_mask', 'in_mask')]),
        (inputnode, acc_msk, [('bold_mask', 'in_mask')]),
        (inputnode, tcc_msk, [('bold_mask', 'in_mask')]),
        # connect inputnode to each non-anatomical confound node
        (inputnode, dvars, [('bold', 'in_file'),
                            ('bold_mask', 'in_mask')]),
        (inputnode, fdisp, [('movpar_file', 'in_file')]),

        # tCompCor
        (inputnode, tcompcor, [('bold', 'realigned_file')]),
        (inputnode, tcompcor, [('skip_vols', 'ignore_initial_volumes')]),
        (tcc_tfm, tcc_msk, [('output_image', 'roi_file')]),
        (tcc_msk, tcompcor, [('out', 'mask_files')]),

        # aCompCor
        (inputnode, acompcor, [('bold', 'realigned_file')]),
        (inputnode, acompcor, [('skip_vols', 'ignore_initial_volumes')]),
        (acc_tfm, acc_msk, [('output_image', 'roi_file')]),
        (acc_msk, mrg_lbl_cc, [('out', 'in1')]),
        (csf_msk, mrg_lbl_cc, [('out', 'in2')]),
        (wm_msk, mrg_lbl_cc, [('out', 'in3')]),
        (mrg_lbl_cc, acompcor, [('out', 'mask_files')]),

        # Global signals extraction (constrained by anatomy)
        (inputnode, signals, [('bold', 'in_file')]),
        (csf_tfm, csf_msk, [('output_image', 'roi_file')]),
        (csf_msk, mrg_lbl, [('out', 'in1')]),
        (wm_tfm, wm_msk, [('output_image', 'roi_file')]),
        (wm_msk, mrg_lbl, [('out', 'in2')]),
        (inputnode, mrg_lbl, [('bold_mask', 'in3')]),
        (mrg_lbl, signals, [('out', 'label_files')]),

        # Collate computed confounds together
        (inputnode, add_motion_headers, [('movpar_file', 'in_file')]),
        (dvars, add_dvars_header, [('out_nstd', 'in_file')]),
        (dvars, add_std_dvars_header, [('out_std', 'in_file')]),
        (signals, concat, [('out_file', 'signals')]),
        (fdisp, concat, [('out_file', 'fd')]),
        (tcompcor, concat, [('components_file', 'tcompcor'),
                            ('pre_filter_file', 'cos_basis')]),
        (acompcor, concat, [('components_file', 'acompcor')]),
        (add_motion_headers, concat, [('out_file', 'motion')]),
        (add_dvars_header, concat, [('out_file', 'dvars')]),
        (add_std_dvars_header, concat, [('out_file', 'std_dvars')]),

        # Confounds metadata
        (tcompcor, tcc_metadata_fmt, [('metadata_file', 'in_file')]),
        (acompcor, acc_metadata_fmt, [('metadata_file', 'in_file')]),
        (tcc_metadata_fmt, mrg_conf_metadata, [('output', 'in1')]),
        (acc_metadata_fmt, mrg_conf_metadata, [('output', 'in2')]),
        (mrg_conf_metadata, mrg_conf_metadata2, [('out', 'in_dicts')]),

        # Expand the model with derivatives, quadratics, and spikes
        (concat, model_expand, [('confounds_file', 'confounds_file')]),
        (model_expand, spike_regress, [('confounds_file', 'confounds_file')]),

        # Set outputs
        (spike_regress, outputnode, [('confounds_file', 'confounds_file')]),
        (mrg_conf_metadata2, outputnode, [('out_dict', 'confounds_metadata')]),
        (inputnode, rois_plot, [('bold', 'in_file'),
                                ('bold_mask', 'in_mask')]),
        (tcompcor, mrg_compcor, [('high_variance_masks', 'in1')]),
        (acc_msk, mrg_compcor, [('out', 'in2')]),
        (mrg_compcor, rois_plot, [('out', 'in_rois')]),
        (rois_plot, ds_report_bold_rois, [('out_report', 'in_file')]),
        (tcompcor, mrg_cc_metadata, [('metadata_file', 'in1')]),
        (acompcor, mrg_cc_metadata, [('metadata_file', 'in2')]),
        (mrg_cc_metadata, compcor_plot, [('out', 'metadata_files')]),
        (compcor_plot, ds_report_compcor, [('out_file', 'in_file')]),
        (concat, conf_corr_plot, [('confounds_file', 'confounds_file')]),
        (conf_corr_plot, ds_report_conf_corr, [('out_file', 'in_file')]),
    ])

    return workflow


def init_carpetplot_wf(standard_spaces, mem_gb, metadata, name="bold_carpet_wf"):
    """

    Resamples the MNI parcellation (ad-hoc parcellation derived from the
    Harvard-Oxford template and others).

    **Parameters**

        mem_gb : float
            Size of BOLD file in GB - please note that this size
            should be calculated after resamplings that may extend
            the FoV
        metadata : dict
            BIDS metadata for BOLD file
        name : str
            Name of workflow (default: ``bold_carpet_wf``)

    **Inputs**

        bold
            BOLD image, after the prescribed corrections (STC, HMC and SDC)
            when available.
        bold_mask
            BOLD series mask
        confounds_file
            TSV of all aggregated confounds
        t1_bold_xform
            Affine matrix that maps the T1w space into alignment with
            the native BOLD space
        std2anat_xfm
            ANTs-compatible affine-and-warp transform file

    **Outputs**

        out_carpetplot
            Path of the generated SVG file

    """
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold', 'bold_mask', 'confounds_file',
                't1_bold_xform', 'std2anat_xfm']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_carpetplot']), name='outputnode')

    select_std = pe.Node(KeySelect(
        keys=list(standard_spaces.keys()), fields=['std2anat_xfm']),
        name='select_std', run_without_submitting=True)
    select_std.inputs.key = 'MNI152NLin2009cAsym'

    # List transforms
    mrg_xfms = pe.Node(niu.Merge(2), name='mrg_xfms')

    # Warp segmentation into EPI space
    resample_parc = pe.Node(ApplyTransforms(
        float=True,
        input_image=str(get_template(
            'MNI152NLin2009cAsym', resolution=1, desc='carpet',
            suffix='dseg', extensions=['.nii', '.nii.gz'])),
        dimension=3, default_value=0, interpolation='MultiLabel'),
        name='resample_parc')

    # Carpetplot and confounds plot
    conf_plot = pe.Node(FMRISummary(
        tr=metadata['RepetitionTime'],
        confounds_list=[
            ('global_signal', None, 'GS'),
            ('csf', None, 'GSCSF'),
            ('white_matter', None, 'GSWM'),
            ('std_dvars', None, 'DVARS'),
            ('framewise_displacement', 'mm', 'FD')]),
        name='conf_plot', mem_gb=mem_gb)
    ds_report_bold_conf = pe.Node(
        DerivativesDataSink(desc='carpetplot', keep_dtype=True),
        name='ds_report_bold_conf', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, select_std, [('std2anat_xfm', 'std2anat_xfm')]),
        (inputnode, mrg_xfms, [('t1_bold_xform', 'in1')]),
        (inputnode, resample_parc, [('bold_mask', 'reference_image')]),
        (select_std, mrg_xfms, [('std2anat_xfm', 'in2')]),
        (mrg_xfms, resample_parc, [('out', 'transforms')]),
        # Carpetplot
        (inputnode, conf_plot, [
            ('bold', 'in_func'),
            ('bold_mask', 'in_mask'),
            ('confounds_file', 'confounds_file')]),
        (resample_parc, conf_plot, [('output_image', 'in_segm')]),
        (conf_plot, ds_report_bold_conf, [('out_file', 'in_file')]),
        (conf_plot, outputnode, [('out_file', 'out_carpetplot')]),
    ])
    return workflow


def init_ica_aroma_wf(metadata, mem_gb, omp_nthreads,
                      name='ica_aroma_wf',
                      susan_fwhm=6.0,
                      err_on_aroma_warn=False,
                      aroma_melodic_dim=-200,
                      use_fieldwarp=True):
    """
    This workflow wraps `ICA-AROMA`_ to identify and remove motion-related
    independent components from a BOLD time series.

    The following steps are performed:

    #. Remove non-steady state volumes from the bold series.
    #. Smooth data using FSL `susan`, with a kernel width FWHM=6.0mm.
    #. Run FSL `melodic` outside of ICA-AROMA to generate the report
    #. Run ICA-AROMA
    #. Aggregate identified motion components (aggressive) to TSV
    #. Return ``classified_motion_ICs`` and ``melodic_mix`` for user to complete
       non-aggressive denoising in T1w space
    #. Calculate ICA-AROMA-identified noise components
       (columns named ``AROMAAggrCompXX``)

    Additionally, non-aggressive denoising is performed on the BOLD series
    resampled into MNI space.

    There is a current discussion on whether other confounds should be extracted
    before or after denoising `here <http://nbviewer.jupyter.org/github/poldracklab/\
    fmriprep-notebooks/blob/922e436429b879271fa13e76767a6e73443e74d9/issue-817_\
    aroma_confounds.ipynb>`__.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold.confounds import init_ica_aroma_wf
        wf = init_ica_aroma_wf(metadata={'RepetitionTime': 1.0},
                               mem_gb=3,
                               omp_nthreads=1)

    **Parameters**

        standard_spaces : str
            Spatial normalization template used as target when that
            registration step was previously calculated with
            :py:func:`~fmriprep.workflows.bold.registration.init_bold_reg_wf`.
            The template must be one of the MNI templates (fMRIPrep uses
            ``MNI152NLin2009cAsym`` by default).
        metadata : dict
            BIDS metadata for BOLD file
        mem_gb : float
            Size of BOLD file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_tpl_trans_wf``)
        susan_fwhm : float
            Kernel width (FWHM in mm) for the smoothing step with
            FSL ``susan`` (default: 6.0mm)
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from BOLD to MNI
        err_on_aroma_warn : bool
            Do not fail on ICA-AROMA errors
        aroma_melodic_dim: int
            Set the dimensionality of the MELODIC ICA decomposition.
            Negative numbers set a maximum on automatic dimensionality estimation.
            Positive numbers set an exact number of components to extract.
            (default: -200, i.e., estimate <=200 components)

    **Inputs**

        itk_bold_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        anat2std_xfm
            ANTs-compatible affine-and-warp transform file
        name_source
            BOLD series NIfTI file
            Used to recover original information lost during processing
        skip_vols
            number of non steady state volumes
        bold_split
            Individual 3D BOLD volumes, not motion corrected
        bold_mask
            BOLD series mask in template space
        hmc_xforms
            List of affine transforms aligning each volume to ``ref_image`` in ITK format
        fieldwarp
            a :abbr:`DFM (displacements field map)` in ITK format
        movpar_file
            SPM-formatted motion parameters file

    **Outputs**

        aroma_confounds
            TSV of confounds identified as noise by ICA-AROMA
        aroma_noise_ics
            CSV of noise components identified by ICA-AROMA
        melodic_mix
            FSL MELODIC mixing matrix
        nonaggr_denoised_file
            BOLD series with non-aggressive ICA-AROMA denoising applied

    .. _ICA-AROMA: https://github.com/maartenmennes/ICA-AROMA

    """
    workflow = Workflow(name=name)
    workflow.__postdesc__ = """\
Automatic removal of motion artifacts using independent component analysis
[ICA-AROMA, @aroma] was performed on the *preprocessed BOLD on MNI space*
time-series after removal of non-steady state volumes and spatial smoothing
with an isotropic, Gaussian kernel of 6mm FWHM (full-width half-maximum).
Corresponding "non-aggresively" denoised runs were produced after such
smoothing.
Additionally, the "aggressive" noise-regressors were collected and placed
in the corresponding confounds file.
"""

    inputnode = pe.Node(niu.IdentityInterface(
        fields=[
            'bold_std',
            'bold_mask_std',
            'movpar_file',
            'name_source',
            'skip_vols',
            'templates',
        ]), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['aroma_confounds', 'aroma_noise_ics', 'melodic_mix',
                'nonaggr_denoised_file']), name='outputnode')

    select_std = pe.Node(KeySelect(
        fields=['bold_mask_std', 'bold_std']),
        name='select_std', run_without_submitting=True)
    select_std.inputs.key = 'MNI152NLin6Asym'

    rm_non_steady_state = pe.Node(niu.Function(function=_remove_volumes,
                                               output_names=['bold_cut']),
                                  name='rm_nonsteady')

    calc_median_val = pe.Node(fsl.ImageStats(op_string='-k %s -p 50'), name='calc_median_val')
    calc_bold_mean = pe.Node(fsl.MeanImage(), name='calc_bold_mean')

    def _getusans_func(image, thresh):
        return [tuple([image, thresh])]
    getusans = pe.Node(niu.Function(function=_getusans_func, output_names=['usans']),
                       name='getusans', mem_gb=0.01)

    smooth = pe.Node(fsl.SUSAN(fwhm=susan_fwhm), name='smooth')

    # melodic node
    melodic = pe.Node(fsl.MELODIC(
        no_bet=True, tr_sec=float(metadata['RepetitionTime']), mm_thresh=0.5, out_stats=True,
        dim=aroma_melodic_dim), name="melodic")

    # ica_aroma node
    ica_aroma = pe.Node(ICA_AROMARPT(
        denoise_type='nonaggr', generate_report=True, TR=metadata['RepetitionTime']),
        name='ica_aroma')

    add_non_steady_state = pe.Node(niu.Function(function=_add_volumes,
                                                output_names=['bold_add']),
                                   name='add_nonsteady')

    # extract the confound ICs from the results
    ica_aroma_confound_extraction = pe.Node(ICAConfounds(err_on_aroma_warn=err_on_aroma_warn),
                                            name='ica_aroma_confound_extraction')

    ds_report_ica_aroma = pe.Node(
        DerivativesDataSink(desc='aroma', keep_dtype=True),
        name='ds_report_ica_aroma', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    def _getbtthresh(medianval):
        return 0.75 * medianval

    # connect the nodes
    workflow.connect([
        (inputnode, select_std, [('templates', 'keys'),
                                 ('bold_std', 'bold_std'),
                                 ('bold_mask_std', 'bold_mask_std')]),
        (inputnode, ica_aroma, [('movpar_file', 'motion_parameters')]),
        (inputnode, rm_non_steady_state, [
            ('skip_vols', 'skip_vols')]),
        (select_std, rm_non_steady_state, [
            ('bold_std', 'bold_file')]),
        (select_std, calc_median_val, [
            ('bold_mask_std', 'mask_file')]),
        (rm_non_steady_state, calc_median_val, [
            ('bold_cut', 'in_file')]),
        (rm_non_steady_state, calc_bold_mean, [
            ('bold_cut', 'in_file')]),
        (calc_bold_mean, getusans, [('out_file', 'image')]),
        (calc_median_val, getusans, [('out_stat', 'thresh')]),
        # Connect input nodes to complete smoothing
        (rm_non_steady_state, smooth, [
            ('bold_cut', 'in_file')]),
        (getusans, smooth, [('usans', 'usans')]),
        (calc_median_val, smooth, [(('out_stat', _getbtthresh), 'brightness_threshold')]),
        # connect smooth to melodic
        (smooth, melodic, [('smoothed_file', 'in_files')]),
        (select_std, melodic, [
            ('bold_mask_std', 'mask')]),
        # connect nodes to ICA-AROMA
        (smooth, ica_aroma, [('smoothed_file', 'in_file')]),
        (select_std, ica_aroma, [
            ('bold_mask_std', 'report_mask'),
            ('bold_mask_std', 'mask')]),
        (melodic, ica_aroma, [('out_dir', 'melodic_dir')]),
        # generate tsvs from ICA-AROMA
        (ica_aroma, ica_aroma_confound_extraction, [('out_dir', 'in_directory')]),
        (inputnode, ica_aroma_confound_extraction, [
            ('skip_vols', 'skip_vols')]),
        # output for processing and reporting
        (ica_aroma_confound_extraction, outputnode, [('aroma_confounds', 'aroma_confounds'),
                                                     ('aroma_noise_ics', 'aroma_noise_ics'),
                                                     ('melodic_mix', 'melodic_mix')]),
        (ica_aroma, add_non_steady_state, [
            ('nonaggr_denoised_file', 'bold_cut_file')]),
        (select_std, add_non_steady_state, [
            ('bold_std', 'bold_file')]),
        (inputnode, add_non_steady_state, [
            ('skip_vols', 'skip_vols')]),
        (add_non_steady_state, outputnode, [('bold_add', 'nonaggr_denoised_file')]),
        (ica_aroma, ds_report_ica_aroma, [('out_report', 'in_file')]),
    ])

    return workflow


def _remove_volumes(bold_file, skip_vols):
    """remove skip_vols from bold_file"""
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    if skip_vols == 0:
        return bold_file

    out = fname_presuffix(bold_file, suffix='_cut')
    bold_img = nb.load(bold_file)
    bold_img.__class__(bold_img.dataobj[..., skip_vols:],
                       bold_img.affine, bold_img.header).to_filename(out)

    return out


def _add_volumes(bold_file, bold_cut_file, skip_vols):
    """prepend skip_vols from bold_file onto bold_cut_file"""
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    if skip_vols == 0:
        return bold_cut_file

    bold_img = nb.load(bold_file)
    bold_cut_img = nb.load(bold_cut_file)

    bold_data = np.concatenate((bold_img.dataobj[..., :skip_vols],
                                bold_cut_img.dataobj), axis=3)

    out = fname_presuffix(bold_cut_file, suffix='_addnonsteady')
    bold_img.__class__(bold_data, bold_img.affine, bold_img.header).to_filename(out)

    return out


def _maskroi(in_mask, roi_file):
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    roi = nb.load(roi_file)
    roidata = roi.get_data().astype(np.uint8)
    msk = nb.load(in_mask).get_data().astype(bool)
    roidata[~msk] = 0
    roi.set_data_dtype(np.uint8)

    out = fname_presuffix(roi_file, suffix='_boldmsk')
    roi.__class__(roidata, roi.affine, roi.header).to_filename(out)
    return out
