# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Calculate BOLD confounds
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_confs_wf
.. autofunction:: init_ica_aroma_wf

"""
import os
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl
from nipype.interfaces.nilearn import SignalExtraction
from nipype.algorithms import confounds as nac

from niworkflows.data import get_mni_icbm152_linear, get_mni_icbm152_nlin_asym_09c
from niworkflows.interfaces.segmentation import ICA_AROMARPT
from niworkflows.interfaces.masks import ROIsPlot
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

from ...engine import Workflow
from ...interfaces import (
    TPM2ROI, AddTPMs, AddTSVHeader, GatherConfounds, ICAConfounds,
    FMRISummary, DerivativesDataSink
)
from ...interfaces.patches import (
    RobustACompCor as ACompCor,
    RobustTCompCor as TCompCor
)

from .resampling import init_bold_mni_trans_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_bold_confs_wf(mem_gb, metadata, name="bold_confs_wf"):
    """
    This workflow calculates confounds for a BOLD series, and aggregates them
    into a :abbr:`TSV (tab-separated value)` file, for use as nuisance
    regressors in a :abbr:`GLM (general linear model)`.

    The following confounds are calculated, with column headings in parentheses:

    #. Region-wise average signal (``CSF``, ``WhiteMatter``, ``GlobalSignal``)
    #. DVARS - standard, nonstandard, and voxel-wise standard variants
       (``stdDVARS``, ``non-stdDVARS``, ``vx-wisestdDVARS``)
    #. Framewise displacement, based on MCFLIRT motion parameters
       (``FramewiseDisplacement``)
    #. Temporal CompCor (``tCompCorXX``)
    #. Anatomical CompCor (``aCompCorXX``)
    #. Cosine basis set for high-pass filtering w/ 0.008 Hz cut-off
       (``CosineXX``)
    #. Non-steady-state volumes (``NonSteadyStateXX``)
    #. Estimated head-motion parameters, in mm and rad
       (``X``, ``Y``, ``Z``, ``RotX``, ``RotY``, ``RotZ``)


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
            metadata={})

    **Parameters**

        mem_gb : float
            Size of BOLD file in GB - please note that this size
            should be calculated after resamplings that may extend
            the FoV
        metadata : dict
            BIDS metadata for BOLD file
        name : str
            Name of workflow (default: ``bold_confs_wf``)

    **Inputs**

        bold
            BOLD image, after the prescribed corrections (STC, HMC and SDC)
            when available.
        bold_mask
            BOLD series mask
        movpar_file
            SPM-formatted motion parameters file
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
Six tCompCor components are then calculated from the top 5% variable
voxels within a mask covering the subcortical regions.
This subcortical mask is obtained by heavily eroding the brain mask,
which ensures it does not include cortical GM regions.
For aCompCor, six components are calculated within the intersection of
the aforementioned mask and the union of CSF and WM masks calculated
in T1w space, after their projection to the native space of each
functional run (using the inverse BOLD-to-T1w transformation).
The head-motion estimates calculated in the correction step were also
placed within the corresponding confounds file.
"""
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold', 'bold_mask', 'movpar_file', 't1_mask', 't1_tpms',
                't1_bold_xform']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['confounds_file']),
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
    dvars = pe.Node(nac.ComputeDVARS(save_all=True, remove_zerovariance=True),
                    name="dvars", mem_gb=mem_gb)

    # Frame displacement
    fdisp = pe.Node(nac.FramewiseDisplacement(parameter_source="SPM"),
                    name="fdisp", mem_gb=mem_gb)

    # a/t-CompCor
    non_steady_state = pe.Node(nac.NonSteadyStateDetector(), name='non_steady_state')
    tcompcor = pe.Node(TCompCor(
        components_file='tcompcor.tsv', pre_filter='cosine', save_pre_filter=True,
        percentile_threshold=.05), name="tcompcor", mem_gb=mem_gb)

    acompcor = pe.Node(ACompCor(
        components_file='acompcor.tsv', pre_filter='cosine', save_pre_filter=True),
        name="acompcor", mem_gb=mem_gb)

    # Set TR if present
    if 'RepetitionTime' in metadata:
        tcompcor.inputs.repetition_time = metadata['RepetitionTime']
        acompcor.inputs.repetition_time = metadata['RepetitionTime']

    # Global and segment regressors
    mrg_lbl = pe.Node(niu.Merge(3), name='merge_rois', run_without_submitting=True)
    signals = pe.Node(SignalExtraction(class_labels=["CSF", "WhiteMatter", "GlobalSignal"]),
                      name="signals", mem_gb=mem_gb)

    # Arrange confounds
    add_header = pe.Node(AddTSVHeader(columns=["X", "Y", "Z", "RotX", "RotY", "RotZ"]),
                         name="add_header", mem_gb=0.01, run_without_submitting=True)
    concat = pe.Node(GatherConfounds(), name="concat", mem_gb=0.01, run_without_submitting=True)

    # Generate reportlet
    mrg_compcor = pe.Node(niu.Merge(2), name='merge_compcor', run_without_submitting=True)
    rois_plot = pe.Node(ROIsPlot(colors=['r', 'b', 'magenta'], generate_report=True),
                        name='rois_plot')

    ds_report_bold_rois = pe.Node(
        DerivativesDataSink(suffix='rois'),
        name='ds_report_bold_rois', run_without_submitting=True,
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

        # Calculate nonsteady state
        (inputnode, non_steady_state, [('bold', 'in_file')]),

        # tCompCor
        (inputnode, tcompcor, [('bold', 'realigned_file')]),
        (non_steady_state, tcompcor, [('n_volumes_to_discard', 'ignore_initial_volumes')]),
        (tcc_tfm, tcc_msk, [('output_image', 'roi_file')]),
        (tcc_msk, tcompcor, [('out', 'mask_files')]),

        # aCompCor
        (inputnode, acompcor, [('bold', 'realigned_file')]),
        (non_steady_state, acompcor, [('n_volumes_to_discard', 'ignore_initial_volumes')]),
        (acc_tfm, acc_msk, [('output_image', 'roi_file')]),
        (acc_msk, acompcor, [('out', 'mask_files')]),

        # Global signals extraction (constrained by anatomy)
        (inputnode, signals, [('bold', 'in_file')]),
        (csf_tfm, csf_msk, [('output_image', 'roi_file')]),
        (csf_msk, mrg_lbl, [('out', 'in1')]),
        (wm_tfm, wm_msk, [('output_image', 'roi_file')]),
        (wm_msk, mrg_lbl, [('out', 'in2')]),
        (inputnode, mrg_lbl, [('bold_mask', 'in3')]),
        (mrg_lbl, signals, [('out', 'label_files')]),

        # Collate computed confounds together
        (inputnode, add_header, [('movpar_file', 'in_file')]),
        (signals, concat, [('out_file', 'signals')]),
        (dvars, concat, [('out_all', 'dvars')]),
        (fdisp, concat, [('out_file', 'fd')]),
        (tcompcor, concat, [('components_file', 'tcompcor'),
                            ('pre_filter_file', 'cos_basis')]),
        (acompcor, concat, [('components_file', 'acompcor')]),
        (add_header, concat, [('out_file', 'motion')]),

        # Set outputs
        (concat, outputnode, [('confounds_file', 'confounds_file')]),
        (inputnode, rois_plot, [('bold', 'in_file'),
                                ('bold_mask', 'in_mask')]),
        (tcompcor, mrg_compcor, [('high_variance_masks', 'in1')]),
        (acc_msk, mrg_compcor, [('out', 'in2')]),
        (mrg_compcor, rois_plot, [('out', 'in_rois')]),
        (rois_plot, ds_report_bold_rois, [('out_report', 'in_file')]),
    ])

    return workflow


def init_carpetplot_wf(mem_gb, metadata, name="bold_carpet_wf"):
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
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file

    **Outputs**

        out_carpetplot
            Path of the generated SVG file

    """
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold', 'bold_mask', 'confounds_file',
                't1_bold_xform', 't1_2_mni_reverse_transform']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_carpetplot']), name='outputnode')

    # List transforms
    mrg_xfms = pe.Node(niu.Merge(2), name='mrg_xfms')

    # Warp segmentation into EPI space
    resample_parc = pe.Node(ApplyTransforms(
        float=True,
        input_image=os.path.join(
            get_mni_icbm152_nlin_asym_09c(), '1mm_parc.nii.gz'),
        dimension=3, default_value=0, interpolation='MultiLabel'),
        name='resample_parc')

    # Carpetplot and confounds plot
    conf_plot = pe.Node(FMRISummary(
        tr=metadata['RepetitionTime'],
        confounds_list=[
            ('GlobalSignal', None, 'GS'),
            ('CSF', None, 'GSCSF'),
            ('WhiteMatter', None, 'GSWM'),
            ('stdDVARS', None, 'DVARS'),
            ('FramewiseDisplacement', 'mm', 'FD')]),
        name='conf_plot', mem_gb=mem_gb)
    ds_report_bold_conf = pe.Node(
        DerivativesDataSink(suffix='carpetplot'),
        name='ds_report_bold_conf', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, mrg_xfms, [('t1_bold_xform', 'in1'),
                               ('t1_2_mni_reverse_transform', 'in2')]),
        (inputnode, resample_parc, [('bold_mask', 'reference_image')]),
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


def init_ica_aroma_wf(template, metadata, mem_gb, omp_nthreads,
                      name='ica_aroma_wf',
                      susan_fwhm=6.0,
                      ignore_aroma_err=False,
                      aroma_melodic_dim=None,
                      use_fieldwarp=True):
    """
    This workflow wraps `ICA-AROMA`_ to identify and remove motion-related
    independent components from a BOLD time series.

    The following steps are performed:

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
        wf = init_ica_aroma_wf(template='MNI152NLin2009cAsym',
                               metadata={'RepetitionTime': 1.0},
                               mem_gb=3,
                               omp_nthreads=1)

    **Parameters**

        template : str
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
            Name of workflow (default: ``bold_mni_trans_wf``)
        susan_fwhm : float
            Kernel width (FWHM in mm) for the smoothing step with
            FSL ``susan`` (default: 6.0mm)
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from BOLD to MNI
        ignore_aroma_err : bool
            Do not fail on ICA-AROMA errors
        aroma_melodic_dim: int or None
            Set the dimensionality of the Melodic ICA decomposition
            If None, MELODIC automatically estimates dimensionality.


    **Inputs**

        bold_mni
            BOLD series, resampled to template space
        movpar_file
            SPM-formatted motion parameters file
        bold_mask_mni
            BOLD series mask in template space

    **Outputs**

        aroma_confounds
            TSV of confounds identified as noise by ICA-AROMA
        aroma_noise_ics
            CSV of noise components identified by ICA-AROMA
        melodic_mix
            FSL MELODIC mixing matrix
        nonaggr_denoised_file
            BOLD series with non-aggressive ICA-AROMA denoising applied

    .. _ICA-AROMA: https://github.com/rhr-pruim/ICA-AROMA

    """
    workflow = Workflow(name=name)
    workflow.__postdesc__ = """\
Automatic removal of motion artifacts using independent component analysis
[ICA-AROMA, @aroma] was performed on the *preprocessed BOLD on MNI space*
time-series after a spatial smoothing with an isotropic, Gaussian kernel
of 6mm FWHM (full-width half-maximum).
Corresponding "non-aggresively" denoised runs were produced after such
smoothing.
Additionally, the "aggressive" noise-regressors were collected and placed
in the corresponding confounds file.
"""

    inputnode = pe.Node(niu.IdentityInterface(
        fields=[
            'itk_bold_to_t1',
            't1_2_mni_forward_transform',
            'name_source',
            'bold_split',
            'bold_mask',
            'hmc_xforms',
            'fieldwarp',
            'movpar_file']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['aroma_confounds', 'aroma_noise_ics', 'melodic_mix',
                'nonaggr_denoised_file']), name='outputnode')

    bold_mni_trans_wf = init_bold_mni_trans_wf(
        template=template,
        mem_gb=mem_gb,
        omp_nthreads=omp_nthreads,
        template_out_grid=os.path.join(get_mni_icbm152_linear(),
                                       '2mm_T1.nii.gz'),
        use_compression=False,
        use_fieldwarp=use_fieldwarp,
        name='bold_mni_trans_wf'
    )
    bold_mni_trans_wf.__desc__ = None

    calc_median_val = pe.Node(fsl.ImageStats(op_string='-k %s -p 50'), name='calc_median_val')
    calc_bold_mean = pe.Node(fsl.MeanImage(), name='calc_bold_mean')

    def _getusans_func(image, thresh):
        return [tuple([image, thresh])]
    getusans = pe.Node(niu.Function(function=_getusans_func, output_names=['usans']),
                       name='getusans', mem_gb=0.01)

    smooth = pe.Node(fsl.SUSAN(fwhm=susan_fwhm), name='smooth')

    # melodic node
    melodic = pe.Node(fsl.MELODIC(
        no_bet=True, tr_sec=float(metadata['RepetitionTime']), mm_thresh=0.5, out_stats=True),
        name="melodic")

    if aroma_melodic_dim is not None:
        melodic.inputs.dim = aroma_melodic_dim

    # ica_aroma node
    ica_aroma = pe.Node(ICA_AROMARPT(
        denoise_type='nonaggr', generate_report=True, TR=metadata['RepetitionTime']),
        name='ica_aroma')

    # extract the confound ICs from the results
    ica_aroma_confound_extraction = pe.Node(ICAConfounds(ignore_aroma_err=ignore_aroma_err),
                                            name='ica_aroma_confound_extraction')

    ds_report_ica_aroma = pe.Node(
        DerivativesDataSink(suffix='ica_aroma'),
        name='ds_report_ica_aroma', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    def _getbtthresh(medianval):
        return 0.75 * medianval

    # connect the nodes
    workflow.connect([
        (inputnode, bold_mni_trans_wf, [
            ('name_source', 'inputnode.name_source'),
            ('bold_split', 'inputnode.bold_split'),
            ('bold_mask', 'inputnode.bold_mask'),
            ('hmc_xforms', 'inputnode.hmc_xforms'),
            ('itk_bold_to_t1', 'inputnode.itk_bold_to_t1'),
            ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
            ('fieldwarp', 'inputnode.fieldwarp')]),
        (inputnode, ica_aroma, [('movpar_file', 'motion_parameters')]),
        (bold_mni_trans_wf, calc_median_val, [
            ('outputnode.bold_mni', 'in_file'),
            ('outputnode.bold_mask_mni', 'mask_file')]),
        (bold_mni_trans_wf, calc_bold_mean, [
            ('outputnode.bold_mni', 'in_file')]),
        (calc_bold_mean, getusans, [('out_file', 'image')]),
        (calc_median_val, getusans, [('out_stat', 'thresh')]),
        # Connect input nodes to complete smoothing
        (bold_mni_trans_wf, smooth, [
            ('outputnode.bold_mni', 'in_file')]),
        (getusans, smooth, [('usans', 'usans')]),
        (calc_median_val, smooth, [(('out_stat', _getbtthresh), 'brightness_threshold')]),
        # connect smooth to melodic
        (smooth, melodic, [('smoothed_file', 'in_files')]),
        (bold_mni_trans_wf, melodic, [
            ('outputnode.bold_mask_mni', 'mask')]),
        # connect nodes to ICA-AROMA
        (smooth, ica_aroma, [('smoothed_file', 'in_file')]),
        (bold_mni_trans_wf, ica_aroma, [
            ('outputnode.bold_mask_mni', 'report_mask')]),
        (melodic, ica_aroma, [('out_dir', 'melodic_dir')]),
        # generate tsvs from ICA-AROMA
        (ica_aroma, ica_aroma_confound_extraction, [('out_dir', 'in_directory')]),
        # output for processing and reporting
        (ica_aroma_confound_extraction, outputnode, [('aroma_confounds', 'aroma_confounds'),
                                                     ('aroma_noise_ics', 'aroma_noise_ics'),
                                                     ('melodic_mix', 'melodic_mix')]),
        # TODO change melodic report to reflect noise and non-noise components
        (ica_aroma, outputnode, [('nonaggr_denoised_file', 'nonaggr_denoised_file')]),
        (ica_aroma, ds_report_ica_aroma, [('out_report', 'in_file')]),
    ])

    return workflow


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
