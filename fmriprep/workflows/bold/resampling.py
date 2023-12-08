# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Resampling workflows
++++++++++++++++++++

.. autofunction:: init_bold_surf_wf
.. autofunction:: init_bold_fsLR_resampling_wf
.. autofunction:: init_bold_grayords_wf
.. autofunction:: init_goodvoxels_bold_mask_wf

"""
from __future__ import annotations

import typing as ty

from nipype import Function
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from niworkflows.interfaces.freesurfer import MedialNaNs

from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces.workbench import MetricDilate, MetricMask, MetricResample
from ...utils.bids import dismiss_echo
from .outputs import prepare_timing_parameters


def init_bold_surf_wf(
    *,
    mem_gb: float,
    surface_spaces: ty.List[str],
    medial_surface_nan: bool,
    metadata: dict,
    output_dir: str,
    name: str = "bold_surf_wf",
):
    """
    Sample functional images to FreeSurfer surfaces.

    For each vertex, the cortical ribbon is sampled at six points (spaced 20% of thickness apart)
    and averaged.

    Outputs are in GIFTI format.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_surf_wf
            wf = init_bold_surf_wf(mem_gb=0.1,
                                   surface_spaces=["fsnative", "fsaverage5"],
                                   medial_surface_nan=False,
                                   metadata={},
                                   output_dir='.',
                                   )

    Parameters
    ----------
    surface_spaces : :obj:`list`
        List of FreeSurfer surface-spaces (either ``fsaverage{3,4,5,6,}`` or ``fsnative``)
        the functional images are to be resampled to.
        For ``fsnative``, images will be resampled to the individual subject's
        native surface.
    medial_surface_nan : :obj:`bool`
        Replace medial wall values with NaNs on functional GIFTI files

    Inputs
    ------
    source_file
        Original BOLD series
    bold_t1w
        Motion-corrected BOLD series in T1 space
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        ITK-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    Outputs
    -------
    surfaces
        BOLD series, resampled to FreeSurfer surfaces

    """
    from nipype.interfaces.io import FreeSurferSource
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs
    from niworkflows.interfaces.surf import GiftiSetAnatomicalStructure

    from fmriprep.interfaces import DerivativesDataSink

    timing_parameters = prepare_timing_parameters(metadata)

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The BOLD time-series were resampled onto the following surfaces
(FreeSurfer reconstruction nomenclature):
{out_spaces}.
""".format(
        out_spaces=", ".join(["*%s*" % s for s in surface_spaces])
    )

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "source_file",
                "bold_t1w",
                "subject_id",
                "subjects_dir",
                "fsnative2t1w_xfm",
            ]
        ),
        name="inputnode",
    )
    itersource = pe.Node(niu.IdentityInterface(fields=["target"]), name="itersource")
    itersource.iterables = [("target", surface_spaces)]

    get_fsnative = pe.Node(FreeSurferSource(), name="get_fsnative", run_without_submitting=True)

    def select_target(subject_id, space):
        """Get the target subject ID, given a source subject ID and a target space."""
        return subject_id if space == "fsnative" else space

    targets = pe.Node(
        niu.Function(function=select_target),
        name="targets",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    itk2lta = pe.Node(
        ConcatenateXFMs(out_fmt="fs", inverse=True), name="itk2lta", run_without_submitting=True
    )
    sampler = pe.MapNode(
        fs.SampleToSurface(
            interp_method="trilinear",
            out_type="gii",
            override_reg_subj=True,
            sampling_method="average",
            sampling_range=(0, 1, 0.2),
            sampling_units="frac",
        ),
        iterfield=["hemi"],
        name="sampler",
        mem_gb=mem_gb * 3,
    )
    sampler.inputs.hemi = ["lh", "rh"]

    update_metadata = pe.MapNode(
        GiftiSetAnatomicalStructure(),
        iterfield=["in_file"],
        name="update_metadata",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    ds_bold_surfs = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            extension=".func.gii",
            dismiss_entities=dismiss_echo(),
            TaskName=metadata.get('TaskName'),
            **timing_parameters,
        ),
        iterfield=['in_file', 'hemi'],
        name='ds_bold_surfs',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_bold_surfs.inputs.hemi = ["L", "R"]

    workflow.connect([
        (inputnode, get_fsnative, [
            ("subject_id", "subject_id"),
            ("subjects_dir", "subjects_dir")
        ]),
        (inputnode, targets, [("subject_id", "subject_id")]),
        (inputnode, itk2lta, [
            ("bold_t1w", "moving"),
            ("fsnative2t1w_xfm", "in_xfms"),
        ]),
        (get_fsnative, itk2lta, [("T1", "reference")]),
        (inputnode, sampler, [
            ("subjects_dir", "subjects_dir"),
            ("subject_id", "subject_id"),
            ("bold_t1w", "source_file"),
        ]),
        (itersource, targets, [("target", "space")]),
        (itk2lta, sampler, [("out_inv", "reg_file")]),
        (targets, sampler, [("out", "target_subject")]),
        (inputnode, ds_bold_surfs, [("source_file", "source_file")]),
        (itersource, ds_bold_surfs, [("target", "space")]),
        (update_metadata, ds_bold_surfs, [("out_file", "in_file")]),
    ])  # fmt:skip

    # Refine if medial vertices should be NaNs
    medial_nans = pe.MapNode(
        MedialNaNs(), iterfield=["in_file"], name="medial_nans", mem_gb=DEFAULT_MEMORY_MIN_GB
    )

    if medial_surface_nan:
        # fmt: off
        workflow.connect([
            (inputnode, medial_nans, [("subjects_dir", "subjects_dir")]),
            (sampler, medial_nans, [("out_file", "in_file")]),
            (medial_nans, update_metadata, [("out_file", "in_file")]),
        ])
        # fmt: on
    else:
        workflow.connect([(sampler, update_metadata, [("out_file", "in_file")])])

    return workflow


def init_goodvoxels_bold_mask_wf(mem_gb: float, name: str = "goodvoxels_bold_mask_wf"):
    """Calculate a mask of a BOLD series excluding high variance voxels.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.bold.resampling import init_goodvoxels_bold_mask_wf
            wf = init_goodvoxels_bold_mask_wf(mem_gb=0.1)

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    name : :obj:`str`
        Name of workflow (default: ``goodvoxels_bold_mask_wf``)

    Inputs
    ------
    anat_ribbon
        Cortical ribbon in T1w space
    bold_file
        Motion-corrected BOLD series in T1w space

    Outputs
    -------
    masked_bold
        BOLD series after masking outlier voxels with locally high COV
    goodvoxels_ribbon
        Cortical ribbon mask excluding voxels with locally high COV
    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "anat_ribbon",
                "bold_file",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "goodvoxels_mask",
                "goodvoxels_ribbon",
            ]
        ),
        name="outputnode",
    )
    ribbon_boldsrc_xfm = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', transforms='identity'),
        name="ribbon_boldsrc_xfm",
        mem_gb=mem_gb,
    )

    stdev_volume = pe.Node(
        fsl.maths.StdImage(dimension='T'),
        name="stdev_volume",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    mean_volume = pe.Node(
        fsl.maths.MeanImage(dimension='T'),
        name="mean_volume",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    cov_volume = pe.Node(
        fsl.maths.BinaryMaths(operation='div'),
        name="cov_volume",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    cov_ribbon = pe.Node(
        fsl.ApplyMask(),
        name="cov_ribbon",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    cov_ribbon_mean = pe.Node(
        fsl.ImageStats(op_string='-M'),
        name="cov_ribbon_mean",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    cov_ribbon_std = pe.Node(
        fsl.ImageStats(op_string='-S'),
        name="cov_ribbon_std",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    cov_ribbon_norm = pe.Node(
        fsl.maths.BinaryMaths(operation='div'),
        name="cov_ribbon_norm",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    smooth_norm = pe.Node(
        fsl.maths.MathsCommand(args="-bin -s 5"),
        name="smooth_norm",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    merge_smooth_norm = pe.Node(
        niu.Merge(1),
        name="merge_smooth_norm",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )

    cov_ribbon_norm_smooth = pe.Node(
        fsl.maths.MultiImageMaths(op_string='-s 5 -div %s -dilD'),
        name="cov_ribbon_norm_smooth",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    cov_norm = pe.Node(
        fsl.maths.BinaryMaths(operation='div'),
        name="cov_norm",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    cov_norm_modulate = pe.Node(
        fsl.maths.BinaryMaths(operation='div'),
        name="cov_norm_modulate",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    cov_norm_modulate_ribbon = pe.Node(
        fsl.ApplyMask(),
        name="cov_norm_modulate_ribbon",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    def _calc_upper_thr(in_stats):
        return in_stats[0] + (in_stats[1] * 0.5)

    upper_thr_val = pe.Node(
        Function(
            input_names=["in_stats"], output_names=["upper_thresh"], function=_calc_upper_thr
        ),
        name="upper_thr_val",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    def _calc_lower_thr(in_stats):
        return in_stats[1] - (in_stats[0] * 0.5)

    lower_thr_val = pe.Node(
        Function(
            input_names=["in_stats"], output_names=["lower_thresh"], function=_calc_lower_thr
        ),
        name="lower_thr_val",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    mod_ribbon_mean = pe.Node(
        fsl.ImageStats(op_string='-M'),
        name="mod_ribbon_mean",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    mod_ribbon_std = pe.Node(
        fsl.ImageStats(op_string='-S'),
        name="mod_ribbon_std",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    merge_mod_ribbon_stats = pe.Node(
        niu.Merge(2),
        name="merge_mod_ribbon_stats",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )

    bin_mean_volume = pe.Node(
        fsl.maths.UnaryMaths(operation="bin"),
        name="bin_mean_volume",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    merge_goodvoxels_operands = pe.Node(
        niu.Merge(2),
        name="merge_goodvoxels_operands",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )

    goodvoxels_thr = pe.Node(
        fsl.maths.Threshold(),
        name="goodvoxels_thr",
        mem_gb=mem_gb,
    )

    goodvoxels_mask = pe.Node(
        fsl.maths.MultiImageMaths(op_string='-bin -sub %s -mul -1'),
        name="goodvoxels_mask",
        mem_gb=mem_gb,
    )

    # make HCP-style "goodvoxels" mask in t1w space for filtering outlier voxels
    # in bold timeseries, based on modulated normalized covariance
    workflow.connect(
        [
            (inputnode, ribbon_boldsrc_xfm, [("anat_ribbon", "input_image")]),
            (inputnode, stdev_volume, [("bold_file", "in_file")]),
            (inputnode, mean_volume, [("bold_file", "in_file")]),
            (mean_volume, ribbon_boldsrc_xfm, [("out_file", "reference_image")]),
            (stdev_volume, cov_volume, [("out_file", "in_file")]),
            (mean_volume, cov_volume, [("out_file", "operand_file")]),
            (cov_volume, cov_ribbon, [("out_file", "in_file")]),
            (ribbon_boldsrc_xfm, cov_ribbon, [("output_image", "mask_file")]),
            (cov_ribbon, cov_ribbon_mean, [("out_file", "in_file")]),
            (cov_ribbon, cov_ribbon_std, [("out_file", "in_file")]),
            (cov_ribbon, cov_ribbon_norm, [("out_file", "in_file")]),
            (cov_ribbon_mean, cov_ribbon_norm, [("out_stat", "operand_value")]),
            (cov_ribbon_norm, smooth_norm, [("out_file", "in_file")]),
            (smooth_norm, merge_smooth_norm, [("out_file", "in1")]),
            (cov_ribbon_norm, cov_ribbon_norm_smooth, [("out_file", "in_file")]),
            (merge_smooth_norm, cov_ribbon_norm_smooth, [("out", "operand_files")]),
            (cov_ribbon_mean, cov_norm, [("out_stat", "operand_value")]),
            (cov_volume, cov_norm, [("out_file", "in_file")]),
            (cov_norm, cov_norm_modulate, [("out_file", "in_file")]),
            (cov_ribbon_norm_smooth, cov_norm_modulate, [("out_file", "operand_file")]),
            (cov_norm_modulate, cov_norm_modulate_ribbon, [("out_file", "in_file")]),
            (ribbon_boldsrc_xfm, cov_norm_modulate_ribbon, [("output_image", "mask_file")]),
            (cov_norm_modulate_ribbon, mod_ribbon_mean, [("out_file", "in_file")]),
            (cov_norm_modulate_ribbon, mod_ribbon_std, [("out_file", "in_file")]),
            (mod_ribbon_mean, merge_mod_ribbon_stats, [("out_stat", "in1")]),
            (mod_ribbon_std, merge_mod_ribbon_stats, [("out_stat", "in2")]),
            (merge_mod_ribbon_stats, upper_thr_val, [("out", "in_stats")]),
            (merge_mod_ribbon_stats, lower_thr_val, [("out", "in_stats")]),
            (mean_volume, bin_mean_volume, [("out_file", "in_file")]),
            (upper_thr_val, goodvoxels_thr, [("upper_thresh", "thresh")]),
            (cov_norm_modulate, goodvoxels_thr, [("out_file", "in_file")]),
            (bin_mean_volume, merge_goodvoxels_operands, [("out_file", "in1")]),
            (goodvoxels_thr, goodvoxels_mask, [("out_file", "in_file")]),
            (merge_goodvoxels_operands, goodvoxels_mask, [("out", "operand_files")]),
        ]
    )

    goodvoxels_ribbon_mask = pe.Node(
        fsl.ApplyMask(),
        name_source=['in_file'],
        keep_extension=True,
        name="goodvoxels_ribbon_mask",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # apply goodvoxels ribbon mask to bold
    workflow.connect(
        [
            (goodvoxels_mask, goodvoxels_ribbon_mask, [("out_file", "in_file")]),
            (ribbon_boldsrc_xfm, goodvoxels_ribbon_mask, [("output_image", "mask_file")]),
            (goodvoxels_mask, outputnode, [("out_file", "goodvoxels_mask")]),
            (goodvoxels_ribbon_mask, outputnode, [("out_file", "goodvoxels_ribbon")]),
        ]
    )

    return workflow


def init_bold_fsLR_resampling_wf(
    grayord_density: ty.Literal['91k', '170k'],
    omp_nthreads: int,
    mem_gb: float,
    name: str = "bold_fsLR_resampling_wf",
):
    """Resample BOLD time series to fsLR surface.

    This workflow is derived heavily from three scripts within the DCAN-HCP pipelines scripts

    Line numbers correspond to the locations of the code in the original scripts, found at:
    https://github.com/DCAN-Labs/DCAN-HCP/tree/9291324/

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.bold.resampling import init_bold_fsLR_resampling_wf
            wf = init_bold_fsLR_resampling_wf(
                grayord_density='92k',
                omp_nthreads=1,
                mem_gb=1,
            )

    Parameters
    ----------
    grayord_density : :class:`str`
        Either ``"91k"`` or ``"170k"``, representing the total *grayordinates*.
    omp_nthreads : :class:`int`
        Maximum number of threads an individual process may use
    mem_gb : :class:`float`
        Size of BOLD file in GB
    name : :class:`str`
        Name of workflow (default: ``bold_fsLR_resampling_wf``)

    Inputs
    ------
    bold_file : :class:`str`
        Path to BOLD file resampled into T1 space
    white : :class:`list` of :class:`str`
        Path to left and right hemisphere white matter GIFTI surfaces.
    pial : :class:`list` of :class:`str`
        Path to left and right hemisphere pial GIFTI surfaces.
    midthickness : :class:`list` of :class:`str`
        Path to left and right hemisphere midthickness GIFTI surfaces.
    midthickness_fsLR : :class:`list` of :class:`str`
        Path to left and right hemisphere midthickness GIFTI surfaces in fsLR space.
    sphere_reg_fsLR : :class:`list` of :class:`str`
        Path to left and right hemisphere sphere.reg GIFTI surfaces, mapping from subject to fsLR
    cortex_mask : :class:`list` of :class:`str`
        Path to left and right hemisphere cortical masks.
    volume_roi : :class:`str` or Undefined
        Pre-calculated goodvoxels mask. Not required.

    Outputs
    -------
    bold_fsLR : :class:`list` of :class:`str`
        Path to BOLD series resampled as functional GIFTI files in fsLR space

    """
    import templateflow.api as tf
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.utility import KeySelect
    from smriprep import data as smriprep_data

    from fmriprep.interfaces.workbench import VolumeToSurfaceMapping

    fslr_density = "32k" if grayord_density == "91k" else "59k"

    workflow = Workflow(name=name)

    workflow.__desc__ = """\
The BOLD time-series were resampled onto the left/right-symmetric template
"fsLR" using the Connectome Workbench [@hcppipelines].
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                'white',
                'pial',
                'midthickness',
                'midthickness_fsLR',
                'sphere_reg_fsLR',
                'cortex_mask',
                'volume_roi',
            ]
        ),
        name='inputnode',
    )

    hemisource = pe.Node(
        niu.IdentityInterface(fields=['hemi']),
        name='hemisource',
        iterables=[('hemi', ['L', 'R'])],
    )

    joinnode = pe.JoinNode(
        niu.IdentityInterface(fields=['bold_fsLR']),
        name='joinnode',
        joinsource='hemisource',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_fsLR']),
        name='outputnode',
    )

    # select white, midthickness and pial surfaces based on hemi
    select_surfaces = pe.Node(
        KeySelect(
            fields=[
                "white",
                "pial",
                "midthickness",
                'midthickness_fsLR',
                "sphere_reg_fsLR",
                "template_sphere",
                'cortex_mask',
                "template_roi",
            ],
            keys=["L", "R"],
        ),
        name="select_surfaces",
        run_without_submitting=True,
    )
    select_surfaces.inputs.template_sphere = [
        str(sphere)
        for sphere in tf.get(
            template='fsLR',
            density=fslr_density,
            suffix='sphere',
            space=None,
            extension='.surf.gii',
        )
    ]
    atlases = smriprep_data.load_resource('atlases')
    select_surfaces.inputs.template_roi = [
        str(atlases / f'L.atlasroi.{fslr_density}_fs_LR.shape.gii'),
        str(atlases / f'R.atlasroi.{fslr_density}_fs_LR.shape.gii'),
    ]

    # RibbonVolumeToSurfaceMapping.sh
    # Line 85 thru ...
    volume_to_surface = pe.Node(
        VolumeToSurfaceMapping(method="ribbon-constrained"),
        name="volume_to_surface",
        mem_gb=mem_gb * 3,
        n_procs=omp_nthreads,
    )
    metric_dilate = pe.Node(
        MetricDilate(distance=10, nearest=True),
        name="metric_dilate",
        mem_gb=1,
        n_procs=omp_nthreads,
    )
    mask_native = pe.Node(MetricMask(), name="mask_native")
    resample_to_fsLR = pe.Node(
        MetricResample(method='ADAP_BARY_AREA', area_surfs=True),
        name="resample_to_fsLR",
        mem_gb=1,
        n_procs=omp_nthreads,
    )
    # ... line 89
    mask_fsLR = pe.Node(MetricMask(), name="mask_fsLR")

    workflow.connect([
        (inputnode, select_surfaces, [
            ('white', 'white'),
            ('pial', 'pial'),
            ('midthickness', 'midthickness'),
            ('midthickness_fsLR', 'midthickness_fsLR'),
            ('sphere_reg_fsLR', 'sphere_reg_fsLR'),
            ('cortex_mask', 'cortex_mask'),
        ]),
        (hemisource, select_surfaces, [('hemi', 'key')]),
        # Resample BOLD to native surface, dilate and mask
        (inputnode, volume_to_surface, [
            ('bold_file', 'volume_file'),
            ('volume_roi', 'volume_roi'),
        ]),
        (select_surfaces, volume_to_surface, [
            ('midthickness', 'surface_file'),
            ('white', 'inner_surface'),
            ('pial', 'outer_surface'),
        ]),
        (select_surfaces, metric_dilate, [('midthickness', 'surf_file')]),
        (select_surfaces, mask_native, [('cortex_mask', 'mask')]),
        (volume_to_surface, metric_dilate, [('out_file', 'in_file')]),
        (metric_dilate, mask_native, [('out_file', 'in_file')]),
        # Resample BOLD to fsLR and mask
        (select_surfaces, resample_to_fsLR, [
            ('sphere_reg_fsLR', 'current_sphere'),
            ('template_sphere', 'new_sphere'),
            ('midthickness', 'current_area'),
            ('midthickness_fsLR', 'new_area'),
            ('cortex_mask', 'roi_metric'),
        ]),
        (mask_native, resample_to_fsLR, [('out_file', 'in_file')]),
        (select_surfaces, mask_fsLR, [('template_roi', 'mask')]),
        (resample_to_fsLR, mask_fsLR, [('out_file', 'in_file')]),
        # Output
        (mask_fsLR, joinnode, [('out_file', 'bold_fsLR')]),
        (joinnode, outputnode, [('bold_fsLR', 'bold_fsLR')]),
    ])  # fmt:skip

    return workflow


def init_bold_grayords_wf(
    grayord_density: ty.Literal['91k', '170k'],
    mem_gb: float,
    repetition_time: float,
    name: str = "bold_grayords_wf",
):
    """
    Sample Grayordinates files onto the fsLR atlas.

    Outputs are in CIFTI2 format.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.bold.resampling import init_bold_grayords_wf
            wf = init_bold_grayords_wf(mem_gb=0.1, grayord_density="91k", repetition_time=2)

    Parameters
    ----------
    grayord_density : :class:`str`
        Either ``"91k"`` or ``"170k"``, representing the total *grayordinates*.
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    repetition_time : :obj:`float`
        Repetition time in seconds
    name : :obj:`str`
        Unique name for the subworkflow (default: ``"bold_grayords_wf"``)

    Inputs
    ------
    bold_fsLR : :obj:`str`
        List of paths to BOLD series resampled as functional GIFTI files in fsLR space
    bold_std : :obj:`str`
        List of BOLD conversions to standard spaces.
    spatial_reference : :obj:`str`
        List of unique identifiers corresponding to the BOLD standard-conversions.


    Outputs
    -------
    cifti_bold : :obj:`str`
        BOLD CIFTI dtseries.
    cifti_metadata : :obj:`str`
        BIDS metadata file corresponding to ``cifti_bold``.

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.cifti import GenerateCifti

    workflow = Workflow(name=name)

    mni_density = "2" if grayord_density == "91k" else "1"

    workflow.__desc__ = f"""\
*Grayordinates* files [@hcppipelines] containing {grayord_density} samples were also
generated with surface data transformed directly to fsLR space and subcortical
data transformed to {mni_density} mm resolution MNI152NLin6Asym space.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["bold_std", "bold_fsLR"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["cifti_bold", "cifti_metadata"]),
        name="outputnode",
    )

    gen_cifti = pe.Node(
        GenerateCifti(
            TR=repetition_time,
            grayordinates=grayord_density,
        ),
        name="gen_cifti",
        mem_gb=mem_gb,
    )

    workflow.connect([
        (inputnode, gen_cifti, [
            ("bold_fsLR", "surface_bolds"),
            ("bold_std", "bold_file"),
        ]),
        (gen_cifti, outputnode, [
            ("out_file", "cifti_bold"),
            ("out_metadata", "cifti_metadata"),
        ]),
    ])  # fmt:skip
    return workflow
