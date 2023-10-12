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
import numpy as np
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from niworkflows.utils.connections import listify
from sdcflows.workflows.apply.correction import init_unwarp_wf
from sdcflows.workflows.apply.registration import init_coeff2epi_wf

from ... import config
from ...interfaces.reports import FunctionalSummary
from ...utils.bids import extract_entities

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
    bold_series: ty.Union[str, ty.List[str]],
    precomputed: dict,
    fieldmap_id: ty.Optional[str] = None,
    omp_nthreads: int = 1,
) -> pe.Workflow:
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.utility import KeySelect

    layout = config.execution.layout

    # Collect bold and sbref files, sorted by EchoTime
    bold_files = sorted(
        listify(bold_series),
        key=lambda fname: layout.get_metadata(fname).get("EchoTime"),
    )
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

    if os.path.isfile(bold_file):
        bold_tlen, mem_gb = _create_mem_gb(bold_file)

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

    workflow = Workflow(name=_get_wf_name(bold_file))

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
                "hmc_boldref",
                "coreg_boldref",
                "bold_mask",
                "motion_xfm",
                "boldref2anat_xfm",
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
        niu.IdentityInterface(fields=["boldref2fmap_xform"]), name="fmapreg_buffer"
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
            (hmc_boldref_wf, hmcref_buffer, [("outputnode.bold_file", "bold_file")]),
            (hmc_boldref_wf, ds_hmc_boldref_wf, [("outputnode.boldref", "inputnode.boldref")]),
            (ds_hmc_boldref_wf, hmcref_buffer, [("outputnode.boldref", "boldref")]),
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
                    fields=["fmap", "fmap_ref", "fmap_coeff", "fmap_mask", "sdc_method"],
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
                    ("fmap", "fmap"),
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


def _create_mem_gb(bold_fname):
    img = nb.load(bold_fname)
    nvox = int(np.prod(img.shape, dtype='u8'))
    # Assume tools will coerce to 8-byte floats to be safe
    bold_size_gb = 8 * nvox / (1024**3)
    bold_tlen = img.shape[-1]
    mem_gb = {
        "filesize": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "largemem": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return bold_tlen, mem_gb


def _get_wf_name(bold_fname):
    """
    Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz")
    'func_preproc_task_nback_wf'
    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz")
    'func_preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = "_".join(fname.split("_")[1:])
    name = "func_preproc_" + fname_nosub.replace(".", "_").replace(" ", "").replace(
        "-", "_"
    ).replace("_bold", "_wf")

    return name


def _to_join(in_file, join_file):
    """Join two tsv files if the join_file is not ``None``."""
    from niworkflows.interfaces.utility import JoinTSVColumns

    if join_file is None:
        return in_file
    res = JoinTSVColumns(in_file=in_file, join_file=join_file).run()
    return res.outputs.out_file


def _select_ref(sbref_files, boldref_files):
    """Select first sbref or boldref file, preferring sbref if available"""
    from niworkflows.utils.connections import listify

    refs = sbref_files or boldref_files
    return listify(refs)[0]
