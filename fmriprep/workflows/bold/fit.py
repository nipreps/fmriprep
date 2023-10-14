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
Orchestrating the BOLD-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_func_preproc_wf
.. autofunction:: init_func_derivatives_wf

"""
import os
import typing as ty

import bids
import nibabel as nb
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from niworkflows.interfaces.utility import KeySelect
from niworkflows.utils.connections import listify
from sdcflows.workflows.apply.correction import init_unwarp_wf
from sdcflows.workflows.apply.registration import init_coeff2epi_wf

from ... import config
from ...interfaces.reports import FunctionalSummary
from ...interfaces.resampling import (
    DistortionParameters,
    ReconstructFieldmap,
    ResampleSeries,
)
from ...utils.bids import extract_entities
from ...utils.misc import estimate_bold_mem_usage

# BOLD workflows
from .hmc import init_bold_hmc_wf
from .outputs import (
    init_ds_boldref_wf,
    init_ds_hmc_wf,
    init_ds_registration_wf,
    init_func_fit_reports_wf,
)
from .reference import init_raw_boldref_wf
from .registration import init_bold_reg_wf
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf


def get_sbrefs(
    bold_files: ty.List[str],
    entity_overrides: ty.Dict[str, ty.Any],
    layout: bids.BIDSLayout,
) -> ty.List[str]:
    """Find single-band reference(s) associated with BOLD file(s)

    Parameters
    ----------
    bold_files
        List of absolute paths to BOLD files
    entity_overrides
        Query parameters to override defaults
    layout
        :class:`~bids.layout.BIDSLayout` to query

    Returns
    -------
    sbref_files
        List of absolute paths to sbref files associated with input BOLD files,
        sorted by EchoTime
    """
    entities = extract_entities(bold_files)
    entities.pop("echo", None)
    entities.update(suffix="sbref", extension=[".nii", ".nii.gz"], **entity_overrides)

    return sorted(
        layout.get(return_type="file", **entities),
        key=lambda fname: layout.get_metadata(fname).get("EchoTime"),
    )


def init_bold_fit_wf(
    *,
    bold_series: ty.List[str],
    precomputed: dict,
    fieldmap_id: ty.Optional[str] = None,
    omp_nthreads: int = 1,
    name: str = "bold_fit_wf",
) -> pe.Workflow:
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmriprep.utils.misc import estimate_bold_mem_usage

    layout = config.execution.layout

    # Collect bold and sbref files, sorted by EchoTime
    bold_files = sorted(bold_series, key=lambda fname: layout.get_metadata(fname).get("EchoTime"))
    sbref_files = get_sbrefs(
        bold_files,
        entity_overrides=config.execution.get().get('bids_filters', {}).get('sbref', {}),
        layout=layout,
    )

    # Fitting operates on the shortest echo
    # This could become more complicated in the future
    bold_file = bold_files[0]

    # Get metadata from BOLD file(s)
    entities = extract_entities(bold_files)
    metadata = layout.get_metadata(bold_file)
    orientation = "".join(nb.aff2axcodes(nb.load(bold_file).affine))

    bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

    # Boolean used to update workflow self-descriptions
    multiecho = len(bold_files) > 1

    have_hmcref = "hmc_boldref" in precomputed
    have_coregref = "coreg_boldref" in precomputed
    # Can contain
    #  1) boldref2fmap
    #  2) boldref2anat
    #  3) hmc
    transforms = precomputed.get("transforms", {})
    hmc_xforms = transforms.get("hmc")
    boldref2fmap_xform = transforms.get("boldref2fmap")
    boldref2anat_xform = transforms.get("boldref2anat")

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                # Fieldmap registration
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "fmap_id",
                "sdc_method",
                # Anatomical coregistration
                "t1w_preproc",
                "t1w_mask",
                "t1w_dseg",
                "subjects_dir",
                "subject_id",
                "fsnative2t1w_xfm",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.bold_file = bold_series

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dummy_scans",
                "hmc_boldref",
                "coreg_boldref",
                "bold_mask",
                "motion_xfm",
                "boldref2anat_xfm",
                "boldref2fmap_xfm",
            ],
        ),
        name="outputnode",
    )

    # If all derivatives exist, inputnode could go unconnected, so add explicitly
    workflow.add_nodes([inputnode])

    hmcref_buffer = pe.Node(
        niu.IdentityInterface(fields=["boldref", "bold_file"]), name="hmcref_buffer"
    )
    fmapref_buffer = pe.Node(niu.Function(function=_select_ref), name="fmapref_buffer")
    hmc_buffer = pe.Node(niu.IdentityInterface(fields=["hmc_xforms"]), name="hmc_buffer")
    fmapreg_buffer = pe.Node(
        niu.IdentityInterface(fields=["boldref2fmap_xfm"]), name="fmapreg_buffer"
    )
    regref_buffer = pe.Node(
        niu.IdentityInterface(fields=["boldref", "boldmask"]), name="regref_buffer"
    )

    summary = pe.Node(
        FunctionalSummary(
            distortion_correction="None",  # Can override with connection
            registration=("FSL", "FreeSurfer")[config.workflow.run_reconall],
            registration_dof=config.workflow.bold2t1w_dof,
            registration_init=config.workflow.bold2t1w_init,
            pe_direction=metadata.get("PhaseEncodingDirection"),
            echo_idx=entities.get("echo", []),
            tr=metadata["RepetitionTime"],
            orientation=orientation,
        ),
        name="summary",
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )
    summary.inputs.dummy_scans = config.workflow.dummy_scans

    func_fit_reports_wf = init_func_fit_reports_wf(
        # TODO: Enable sdc report even if we find coregref
        sdc_correction=not (have_coregref or fieldmap_id is None),
        freesurfer=config.workflow.run_reconall,
        output_dir=config.execution.fmriprep_dir,
    )

    # fmt:off
    workflow.connect([
        (hmcref_buffer, outputnode, [("boldref", "hmc_boldref")]),
        (regref_buffer, outputnode, [("boldref", "coreg_boldref"),
                                     ("boldmask", "bold_mask")]),
        (fmapreg_buffer, outputnode, [("boldref2fmap_xfm", "boldref2fmap_xfm")]),
        (hmc_buffer, outputnode, [("hmc_xforms", "motion_xfm")]),
        (inputnode, func_fit_reports_wf, [
            ("bold_file", "inputnode.source_file"),
            ("t1w_preproc", "inputnode.t1w_preproc"),
            # May not need all of these
            ("t1w_mask", "inputnode.t1w_mask"),
            ("t1w_dseg", "inputnode.t1w_dseg"),
            ("subjects_dir", "inputnode.subjects_dir"),
            ("subject_id", "inputnode.subject_id"),
        ]),
        (outputnode, func_fit_reports_wf, [
            ("coreg_boldref", "inputnode.coreg_boldref"),
            ("boldref2anat_xfm", "inputnode.boldref2anat_xfm"),
        ]),
        (summary, func_fit_reports_wf, [("out_report", "inputnode.summary_report")]),
    ])
    # fmt:on

    # Stage 1: Generate motion correction boldref
    if not have_hmcref:
        config.loggers.workflow.info("Stage 1: Adding HMC boldref workflow")
        hmc_boldref_wf = init_raw_boldref_wf(
            name="hmc_boldref_wf",
            bold_file=bold_file,
            multiecho=multiecho,
        )
        hmc_boldref_wf.inputs.inputnode.dummy_scans = config.workflow.dummy_scans

        ds_hmc_boldref_wf = init_ds_boldref_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            desc='hmc',
            name='ds_hmc_boldref_wf',
        )
        ds_hmc_boldref_wf.inputs.inputnode.source_files = [bold_file]

        # fmt:off
        workflow.connect([
            (hmc_boldref_wf, hmcref_buffer, [
                ("outputnode.bold_file", "bold_file"),
                ("outputnode.boldref", "boldref"),
            ]),
            (hmcref_buffer, ds_hmc_boldref_wf, [("boldref", "inputnode.boldref")]),
            (hmc_boldref_wf, summary, [("outputnode.algo_dummy_scans", "algo_dummy_scans")]),
            (hmc_boldref_wf, func_fit_reports_wf, [
                ("outputnode.validation_report", "inputnode.validation_report"),
            ]),
        ])
        # fmt:on
    else:
        config.loggers.workflow.info("Found HMC boldref - skipping Stage 1")

        validate_bold = pe.Node(ValidateImage(), name="validate_bold")
        validate_bold.inputs.in_file = bold_file

        hmcref_buffer.inputs.boldref = precomputed["hmc_boldref"]

        # fmt:off
        workflow.connect([
            (validate_bold, hmcref_buffer, [("out_file", "bold_file")]),
            (validate_bold, func_fit_reports_wf, [("out_report", "inputnode.validation_report")]),
        ])
        # fmt:on

    # Stage 2: Estimate head motion
    if not hmc_xforms:
        config.loggers.workflow.info("Stage 2: Adding motion correction workflow")
        bold_hmc_wf = init_bold_hmc_wf(
            name="bold_hmc_wf", mem_gb=mem_gb["filesize"], omp_nthreads=omp_nthreads
        )

        ds_hmc_wf = init_ds_hmc_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
        )
        ds_hmc_wf.inputs.inputnode.source_files = [bold_file]

        # fmt:off
        workflow.connect([
            (hmcref_buffer, bold_hmc_wf, [
                ("boldref", "inputnode.raw_ref_image"),
                ("bold_file", "inputnode.bold_file"),
            ]),
            (bold_hmc_wf, ds_hmc_wf, [("outputnode.xforms", "inputnode.xforms")]),
            (ds_hmc_wf, hmc_buffer, [("outputnode.xforms", "hmc_xforms")]),
        ])
        # fmt:on
    else:
        config.loggers.workflow.info("Found motion correction transforms - skipping Stage 2")
        hmc_buffer.inputs.hmc_xforms = hmc_xforms

    # Stage 3: Create coregistration reference
    # Fieldmap correction only happens during fit if this stage is needed
    if not have_coregref:
        config.loggers.workflow.info("Stage 3: Adding coregistration boldref workflow")

        # Select initial boldref, enhance contrast, and generate mask
        fmapref_buffer.inputs.sbref_files = sbref_files
        enhance_boldref_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

        ds_coreg_boldref_wf = init_ds_boldref_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            desc='coreg',
            name='ds_coreg_boldref_wf',
        )

        # fmt:off
        workflow.connect([
            (hmcref_buffer, fmapref_buffer, [("boldref", "boldref_files")]),
            (fmapref_buffer, enhance_boldref_wf, [("out", "inputnode.in_file")]),
            (fmapref_buffer, ds_coreg_boldref_wf, [("out", "inputnode.source_files")]),
            (ds_coreg_boldref_wf, regref_buffer, [("outputnode.boldref", "boldref")]),
            (fmapref_buffer, func_fit_reports_wf, [("out", "inputnode.sdc_boldref")]),
        ])
        # fmt:on

        if fieldmap_id:
            fmap_select = pe.Node(
                KeySelect(
                    fields=["fmap_ref", "fmap_coeff", "fmap_mask", "sdc_method"],
                    key=fieldmap_id,
                ),
                name="fmap_select",
                run_without_submitting=True,
            )

            if not boldref2fmap_xform:
                fmapreg_wf = init_coeff2epi_wf(
                    debug="fieldmaps" in config.execution.debug,
                    omp_nthreads=config.nipype.omp_nthreads,
                    sloppy=config.execution.sloppy,
                    name="fmapreg_wf",
                )

                itk_mat2txt = pe.Node(ConcatenateXFMs(out_fmt="itk"), name="itk_mat2txt")

                ds_fmapreg_wf = init_ds_registration_wf(
                    bids_root=layout.root,
                    output_dir=config.execution.fmriprep_dir,
                    source="boldref",
                    dest=fieldmap_id.replace('_', ''),
                    name="ds_fmapreg_wf",
                )

                # fmt:off
                workflow.connect([
                    (enhance_boldref_wf, fmapreg_wf, [
                        ('outputnode.bias_corrected_file', 'inputnode.target_ref'),
                        ('outputnode.mask_file', 'inputnode.target_mask'),
                    ]),
                    (fmap_select, fmapreg_wf, [
                        ("fmap_ref", "inputnode.fmap_ref"),
                        ("fmap_mask", "inputnode.fmap_mask"),
                    ]),
                    (fmapreg_wf, itk_mat2txt, [('outputnode.target2fmap_xfm', 'in_xfms')]),
                    (itk_mat2txt, ds_fmapreg_wf, [('out_xfm', 'inputnode.xform')]),
                    (fmapref_buffer, ds_fmapreg_wf, [('out', 'inputnode.source_files')]),
                    (ds_fmapreg_wf, fmapreg_buffer, [('outputnode.xform', 'boldref2fmap_xfm')]),
                ])
                # fmt:on
            else:
                fmapreg_buffer.inputs.boldref2fmap_xfm = boldref2fmap_xform

            unwarp_wf = init_unwarp_wf(
                free_mem=config.environment.free_mem,
                debug="fieldmaps" in config.execution.debug,
                omp_nthreads=config.nipype.omp_nthreads,
            )
            unwarp_wf.inputs.inputnode.metadata = layout.get_metadata(bold_file)

            # fmt:off
            workflow.connect([
                (inputnode, fmap_select, [
                    ("fmap_ref", "fmap_ref"),
                    ("fmap_coeff", "fmap_coeff"),
                    ("fmap_mask", "fmap_mask"),
                    ("sdc_method", "sdc_method"),
                    ("fmap_id", "keys"),
                ]),
                (fmap_select, unwarp_wf, [
                    ("fmap_coeff", "inputnode.fmap_coeff"),
                ]),
                (fmapreg_buffer, unwarp_wf, [
                    # This looks backwards, but unwarp_wf describes transforms in
                    # terms of points while we (and init_coeff2epi_wf) describe them
                    # in terms of images. Mapping fieldmap coordinates into boldref
                    # coordinates maps the boldref image onto the fieldmap image.
                    ("boldref2fmap_xfm", "inputnode.fmap2data_xfm"),
                ]),
                (enhance_boldref_wf, unwarp_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.distorted'),
                ]),
                (unwarp_wf, ds_coreg_boldref_wf, [
                    ('outputnode.corrected', 'inputnode.boldref'),
                ]),
                (unwarp_wf, regref_buffer, [
                    ('outputnode.corrected_mask', 'boldmask'),
                ]),
                (fmap_select, func_fit_reports_wf, [("fmap_ref", "inputnode.fmap_ref")]),
                (fmap_select, summary, [("sdc_method", "distortion_correction")]),
                (fmapreg_buffer, func_fit_reports_wf, [
                    ("boldref2fmap_xfm", "inputnode.boldref2fmap_xfm"),
                ]),
                (unwarp_wf, func_fit_reports_wf, [("outputnode.fieldmap", "inputnode.fieldmap")]),
            ])
            # fmt:on
        else:
            # fmt:off
            workflow.connect([
                (enhance_boldref_wf, ds_coreg_boldref_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.boldref'),
                ]),
                (enhance_boldref_wf, regref_buffer, [
                    ('outputnode.mask_file', 'boldmask'),
                ]),
            ])
            # fmt:on
    else:
        config.loggers.workflow.info("Found coregistration reference - skipping Stage 3")
        regref_buffer.inputs.boldref = precomputed["coreg_boldref"]

    if not boldref2anat_xform:
        # calculate BOLD registration to T1w
        bold_reg_wf = init_bold_reg_wf(
            bold2t1w_dof=config.workflow.bold2t1w_dof,
            bold2t1w_init=config.workflow.bold2t1w_init,
            freesurfer=config.workflow.run_reconall,
            mem_gb=mem_gb["resampled"],
            name="bold_reg_wf",
            omp_nthreads=omp_nthreads,
            sloppy=config.execution.sloppy,
            use_bbr=config.workflow.use_bbr,
            use_compression=False,
            write_report=False,
        )

        ds_boldreg_wf = init_ds_registration_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            source="boldref",
            dest="T1w",
            name="ds_boldreg_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, bold_reg_wf, [
                ("t1w_preproc", "inputnode.t1w_preproc"),
                ("t1w_mask", "inputnode.t1w_mask"),
                ("t1w_dseg", "inputnode.t1w_dseg"),
                # Undefined if --fs-no-reconall, but this is safe
                ("subjects_dir", "inputnode.subjects_dir"),
                ("subject_id", "inputnode.subject_id"),
                ("fsnative2t1w_xfm", "inputnode.fsnative2t1w_xfm"),
            ]),
            (regref_buffer, bold_reg_wf, [("boldref", "inputnode.ref_bold_brain")]),
            # Incomplete sources
            (regref_buffer, ds_boldreg_wf, [("boldref", "inputnode.source_files")]),
            (bold_reg_wf, ds_boldreg_wf, [("outputnode.itk_bold_to_t1", "inputnode.xform")]),
            (ds_boldreg_wf, outputnode, [("outputnode.xform", "boldref2anat_xfm")]),
            (bold_reg_wf, summary, [("outputnode.fallback", "fallback")]),
        ])
        # fmt:on
    else:
        outputnode.inputs.boldref2anat_xfm = boldref2anat_xform

    return workflow


def init_bold_native_wf(
    *,
    bold_series: ty.Union[str, ty.List[str]],
    fieldmap_id: ty.Optional[str] = None,
    name: str = "bold_native_wf",
) -> pe.Workflow:
    """First derivatives for fMRIPrep

    This workflow performs slice-timing correction, and resamples to boldref space
    with head motion and susceptibility distortion correction. It also handles
    multi-echo processing and selects the transforms needed to perform further
    resampling.
    """

    layout = config.execution.layout

    # Shortest echo first
    all_metadata, bold_files, echo_times = zip(
        *sorted(
            (
                (md := layout.get_metadata(bold_file), bold_file, md.get("EchoTime"))
                for bold_file in listify(bold_series)
            ),
            key=lambda x: x[2],
        )
    )
    multiecho = len(bold_files) > 1

    bold_file = bold_files[0]
    metadata = all_metadata[0]

    bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

    if multiecho:
        shapes = [nb.load(echo).shape for echo in bold_files]
        if len(set(shapes)) != 1:
            diagnostic = "\n".join(
                f"{os.path.basename(echo)}: {shape}" for echo, shape in zip(bold_files, shapes)
            )
            raise RuntimeError(f"Multi-echo images found with mismatching shapes\n{diagnostic}")
        if len(shapes) == 2:
            raise RuntimeError(
                "Multi-echo processing requires at least three different echos (found two)."
            )

    run_stc = bool(metadata.get("SliceTiming")) and "slicetiming" not in config.workflow.ignore

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # BOLD fit
                "boldref",
                "motion_xfm",
                "fmapreg_xfm",
                "dummy_scans",
                # Fieldmap fit
                "fmap_ref",
                "fmap_coeff",
                "fmap_id",
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_minimal",  # Single echo: STC; Multi-echo: optimal combination
                "bold_native",   # STC + HMC + SDC; Multi-echo: optimal combination
                "motion_xfm",    # motion_xfms to apply to bold_minimal (none for ME)
                "fieldmap_id",   # fieldmap to apply to bold_minimal (none for ME)
                # Multiecho outputs
                "bold_echos",    # Individual corrected echos
                "t2star_map",    # T2* map
            ],  # fmt:skip
        ),
        name="outputnode",
    )

    boldbuffer = pe.Node(
        niu.IdentityInterface(fields=["bold_file", "ro_time", "pe_dir"]), name="boldbuffer"
    )

    # Track echo index - this allows us to treat multi- and single-echo workflows
    # almost identically
    echo_index = pe.Node(niu.IdentityInterface(fields=["echoidx"]), name="echo_index")
    if multiecho:
        echo_index.iterables = [("echoidx", range(len(bold_files)))]
    else:
        echo_index.inputs.echoidx = 0

    # BOLD source: track original BOLD file(s)
    bold_source = pe.Node(niu.Select(inlist=bold_files), name="bold_source")
    validate_bold = pe.Node(ValidateImage(), name="validate_bold")
    workflow.connect([
        (echo_index, bold_source, [("echoidx", "index")]),
        (bold_source, validate_bold, [("out", "in_file")]),
    ])  # fmt:skip

    # Slice-timing correction
    if run_stc:
        bold_stc_wf = init_bold_stc_wf(name="bold_stc_wf", metadata=metadata)
        workflow.connect([
            (inputnode, bold_stc_wf, [("dummy_scans", "inputnode.skip_vols")]),
            (validate_bold, bold_stc_wf, [("out_file", "inputnode.bold_file")]),
            (bold_stc_wf, boldbuffer, [("outputnode.stc_file", "bold_file")]),
        ])  # fmt:skip
    else:
        workflow.connect([(validate_bold, boldbuffer, [("out_file", "bold_file")])])

    # Prepare fieldmap metadata
    if fieldmap_id:
        fmap_select = pe.Node(
            KeySelect(fields=["fmap_ref", "fmap_coeff"], key=fieldmap_id),
            name="fmap_select",
            run_without_submitting=True,
        )

        distortion_params = pe.Node(
            DistortionParameters(metadata=metadata, in_file=bold_file),
            name="distortion_params",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, fmap_select, [
                ("fmap_ref", "fmap_ref"),
                ("fmap_coeff", "fmap_coeff"),
                ("fmap_id", "keys"),
            ]),
            (distortion_params, boldbuffer, [
                ("readout_time", "ro_time"),
                ("pe_direction", "pe_dir"),
            ]),
        ])  # fmt:skip

    # Resample to boldref
    boldref_bold = pe.Node(
        ResampleSeries(), name="boldref_bold", n_procs=config.nipype.omp_nthreads
    )

    workflow.connect([
        (inputnode, boldref_bold, [
            ("boldref", "ref_file"),
            ("motion_xfm", "transforms"),
        ]),
        (boldbuffer, boldref_bold, [
            ("bold_file", "in_file"),
            ("ro_time", "ro_time"),
            ("pe_dir", "pe_dir"),
        ]),
    ])  # fmt:skip

    if fieldmap_id:
        boldref_fmap = pe.Node(ReconstructFieldmap(inverse=[True]), name="boldref_fmap")
        workflow.connect([
            (inputnode, boldref_fmap, [
                ("boldref", "target_ref_file"),
                ("fmapreg_xfm", "transforms"),
            ]),
            (fmap_select, boldref_fmap, [
                ("fmap_coeff", "in_coeffs"),
                ("fmap_ref", "fmap_ref_file"),
            ]),
            (boldref_fmap, boldref_bold, [("out_file", "fieldmap")]),
        ])  # fmt:skip

    if multiecho:
        join_echos = pe.JoinNode(
            niu.IdentityInterface(fields=["bold_files"]),
            joinsource="echo_index",
            joinfield=["bold_files"],
            name="join_echos",
        )

        # create optimal combination, adaptive T2* map
        bold_t2s_wf = init_bold_t2s_wf(
            echo_times=echo_times,
            mem_gb=mem_gb["filesize"],
            omp_nthreads=config.nipype.omp_nthreads,
            name="bold_t2smap_wf",
        )

        # Do NOT set motion_xfm or fieldmap_id on outputnode
        # This prevents downstream resamplers from double-dipping
        workflow.connect([
            (boldref_bold, join_echos, [("out_file", "bold_files")]),
            (join_echos, bold_t2s_wf, [("bold_files", "inputnode.bold_file")]),
            (join_echos, outputnode, [("bold_files", "bold_echos")]),
            (bold_t2s_wf, outputnode, [
                ("outputnode.bold", "bold_minimal"),
                ("outputnode.bold", "bold_native"),
                ("outputnode.t2star", "t2star_map"),
            ]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (inputnode, outputnode, [("motion_xfm", "motion_xfm")]),
            (boldbuffer, outputnode, [("bold_file", "bold_minimal")]),
            (boldref_bold, outputnode, [("out_file", "bold_native")]),
        ])  # fmt:skip

        if fieldmap_id:
            outputnode.inputs.fieldmap_id = fieldmap_id

    return workflow


def _select_ref(sbref_files, boldref_files):
    """Select first sbref or boldref file, preferring sbref if available"""
    from niworkflows.utils.connections import listify

    refs = sbref_files or boldref_files
    return listify(refs)[0]
