from __future__ import annotations

import os
import typing as ty

import nibabel as nb
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.utility import KeySelect
from niworkflows.utils.connections import listify

from fmriprep import config

from ..interfaces.resampling import (
    DistortionParameters,
    ReconstructFieldmap,
    ResampleSeries,
)
from ..utils.misc import estimate_bold_mem_usage
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf, init_t2s_reporting_wf

if ty.TYPE_CHECKING:
    from niworkflows.utils.spaces import SpatialReferences


def init_bold_native_wf(
    *,
    bold_series: ty.Union[str, ty.List[str]],
    fieldmap_id: ty.Optional[str] = None,
    name: str = 'bold_apply_wf',
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
            (md := layout.get_metadata(bold_file), bold_file, md.get("EchoTime"))
            for bold_file in listify(bold_series)
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

    run_stc = bool(metadata.get("SliceTiming")) and "slicetiming" not in config.workflow.ignore

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "coreg_boldref",
                "motion_xfm",
                "fmapreg_xfm",
                "dummy_scans",
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
            KeySelect(
                fields=["fmap", "fmap_ref", "fmap_coeff", "fmap_mask", "sdc_method"],
                key=fieldmap_id,
            ),
            name="fmap_select",
            run_without_submitting=True,
        )

        distortion_params = pe.Node(
            DistortionParameters(metadata=metadata, in_file=bold_file),
            name="distortion_params",
            run_without_submitting=True,
        )
        workflow.connect([
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
            ("coreg_boldref", "ref_file"),
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
                ("coreg_boldref", "target_ref_file"),
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


def init_bold_apply_wf(
    *,
    spaces: SpatialReferences,
    name: str = 'bold_apply_wf',
) -> pe.Workflow:
    """TODO: Docstring"""
    from smriprep.workflows.outputs import init_template_iterator_wf

    workflow = pe.Workflow(name=name)

    if getattr(spaces, "_cached") is not None and spaces.cached.references:
        template_iterator_wf = init_template_iterator_wf(spaces=spaces)
        # TODO: Refactor bold_std_trans_wf
        # bold_std_trans_wf = init_bold_std_trans_wf(
        #     freesurfer=config.workflow.run_reconall,
        #     mem_gb=mem_gb["resampled"],
        #     omp_nthreads=config.nipype.omp_nthreads,
        #     spaces=spaces,
        #     multiecho=multiecho,
        #     use_compression=not config.execution.low_mem,
        #     name="bold_std_trans_wf",
        # )

    return workflow
