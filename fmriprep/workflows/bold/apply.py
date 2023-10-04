import os
import typing as ty

import nibabel as nb
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe

from fmriprep import config

if ty.TYPE_CHECKING:
    from niworkflows.utils.spaces import SpatialReferences


def init_bold_apply_wf(
    *,
    bold_series: ty.Union[str, ty.List[str]],
    spaces: SpatialReferences,
    has_fieldmap: bool,
    name: str = 'bold_apply_wf',
) -> pe.Workflow:
    """TODO: Docstring"""
    from niworkflows.utils.connections import listify
    from smriprep.workflows.outputs import init_template_iterator_wf

    from fmriprep.utils.bids import extract_entities
    from fmriprep.utils.misc import estimate_bold_mem_usage

    from .resampling import init_bold_std_trans_wf
    from .stc import init_bold_stc_wf

    layout = config.execution.layout

    bold_files = sorted(
        listify(bold_series),
        key=lambda fname: layout.get_metadata(fname).get("EchoTime"),
    )
    bold_file = bold_files[0]

    # Get metadata from BOLD file(s)
    entities = extract_entities(bold_files)
    metadata = layout.get_metadata(bold_file)

    if not os.path.isfile(bold_file):
        raise IOError("Invalid BOLD file")
    bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)
    orientation = "".join(nb.aff2axcodes(nb.load(bold_file).affine))

    # Boolean used to update workflow self-descriptions
    multiecho = len(bold_files) > 1

    if multiecho:
        # Drop echo entity for future queries, have a boolean shorthand
        entities.pop("echo", None)
        # reorder echoes from shortest to largest
        tes, bold_file = zip(
            *sorted([(layout.get_metadata(bf)["EchoTime"], bf) for bf in bold_file])
        )
        ref_file = bold_file[0]  # Reset reference to be the shortest TE
        shapes = [nb.load(echo).shape for echo in bold_file]
        if len(set(shapes)) != 1:
            diagnostic = "\n".join(
                f"{os.path.basename(echo)}: {shape}" for echo, shape in zip(bold_file, shapes)
            )
            raise RuntimeError(f"Multi-echo images found with mismatching shapes\n{diagnostic}")

    run_stc = bool(metadata.get("SliceTiming")) and "slicetiming" not in config.workflow.ignore

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                # fit workflow
                "coreg_boldref",
                "bold_mask",
                "motion_xfm",
                "boldref2anat_xfm",
                "coreg_xfm",
                "fmapreg_xfm",
                "dummy_scans",
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.bold_file = bold_file

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_native",
                "bold_ref_native",
                "bold_mask_native",
                "bold_t1",
                "bold_ref_t1",
                "bold_mask_t1",
                "bold_aseg_t1",
                "bold_aparc_t1",
                "bold_std",
                "bold_ref_std",
                "bold_mask_std",
                "bold_aseg_std",
                "bold_aparc_std",
                "bold_cifti",
                "bold_cifti_metadata",
                "t2star_bold",
                "t2star_t1",
                "t2star_std",
                # confounds?
                # fsLR weights?
            ],
        ),
        name="outputnode",
    )

    # Track echo index - this allows us to treat multi- and single-echo workflows
    # almost identically
    echo_index = pe.Node(niu.IdentityInterface(fields=["echoidx"]), name="echo_index")
    if multiecho:
        echo_index.iterables = [("echoidx", range(len(bold_file)))]
    else:
        echo_index.inputs.echoidx = 0

    # BOLD source: track original BOLD file(s)
    bold_source = pe.Node(niu.Select(inlist=bold_file), name="bold_source")

    # BOLD buffer: an identity used as a pointer to either the original BOLD
    # or the STC'ed one for further use.
    boldbuffer = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="boldbuffer")

    # Slice-timing correction
    if run_stc:
        bold_stc_wf = init_bold_stc_wf(name="bold_stc_wf", metadata=metadata)
        # fmt:off
        workflow.connect([
            (inputnode, bold_stc_wf, [("dummy_scans", "inputnode.skip_vols")]),
            # (select_bold, bold_stc_wf, [("out", "inputnode.bold_file")]),  # Validated BOLD files
            (bold_stc_wf, boldbuffer, [("outputnode.stc_file", "bold_file")]),
        ])
        # fmt:on

    # bypass STC from original BOLD in both SE and ME cases
    else:
        # workflow.connect([(select_bold, boldbuffer, [("out", "bold_file")])])
        ...

    if getattr(spaces, "_cached") is not None and spaces.cached.references:
        # TODO: Refactor bold_std_trans_wf
        bold_std_trans_wf = init_bold_std_trans_wf(
            freesurfer=config.workflow.run_reconall,
            mem_gb=mem_gb["resampled"],
            omp_nthreads=config.nipype.omp_nthreads,
            spaces=spaces,
            multiecho=multiecho,
            use_compression=not config.execution.low_mem,
            name="bold_std_trans_wf",
        )

    return workflow
