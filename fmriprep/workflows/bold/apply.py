from __future__ import annotations

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.interfaces.nibabel import GenerateSamplingReference

from ...interfaces.resampling import (
    DistortionParameters,
    ReconstructFieldmap,
    ResampleSeries,
)


def init_bold_volumetric_resample_wf(
    *,
    metadata: dict,
    mem_gb: dict[str, float],
    jacobian: bool,
    fallback_total_readout_time: str | float | None = None,
    fieldmap_id: str | None = None,
    omp_nthreads: int = 1,
    name: str = 'bold_volumetric_resample_wf',
) -> pe.Workflow:
    """Resample a BOLD series to a volumetric target space.

    This workflow collates a sequence of transforms to resample a BOLD series
    in a single shot, including motion correction and fieldmap correction, if
    requested.

    .. workflow::

        from fmriprep.workflows.bold.resampling import init_bold_volumetric_resample_wf
        wf = init_bold_volumetric_resample_wf(
            metadata={
                'RepetitionTime': 2.0,
                'PhaseEncodingDirection': 'j-',
                'TotalReadoutTime': 0.03
            },
            mem_gb={'resampled': 1},
            jacobian=True,
            fieldmap_id='my_fieldmap',
        )

    Parameters
    ----------
    metadata
        BIDS metadata for BOLD file.
    fieldmap_id
        Fieldmap identifier, if fieldmap correction is to be applied.
    omp_nthreads
        Maximum number of threads an individual process may use.
    name
        Name of workflow (default: ``bold_volumetric_resample_wf``)

    Inputs
    ------
    bold_file
        BOLD series to resample.
    bold_ref_file
        Reference image to which BOLD series is aligned.
    target_ref_file
        Reference image defining the target space.
    target_mask
        Brain mask corresponding to ``target_ref_file``.
        This is used to define the field of view for the resampled BOLD series.
    motion_xfm
        List of affine transforms aligning each volume to ``bold_ref_file``.
        If undefined, no motion correction is performed.
    run2fmap_xfm
        Affine transform from ``bold_ref_file`` to the fieldmap reference image.
    fmap_ref
        Fieldmap reference image defining the valid field of view for the fieldmap.
    fmap_coeff
        B-Spline coefficients for the fieldmap.
    run2boldref_xfm
        Affine transform from the run-level BOLD reference to the boldref
        template. Identity when ``--bold-coreg-level run``.
    boldref2anat_xfm
        Affine transform from ``bold_ref_file`` to the anatomical reference image.
    anat2std_xfm
        Affine transform from the anatomical reference image to standard space.
        Leave undefined to resample to anatomical reference space.

    Outputs
    -------
    bold_file
        The ``bold_file`` input, resampled to ``target_ref_file`` space.
    resampling_reference
        An empty reference image with the correct affine and header for resampling
        further images into the BOLD series' space.

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                'bold_ref_file',
                'target_ref_file',
                'target_mask',
                # HMC
                'motion_xfm',
                # SDC
                'run2fmap_xfm',
                'fmap_ref',
                'fmap_coeff',
                # boldref template
                'run2boldref_xfm',
                # Anatomical
                'boldref2anat_xfm',
                # Template
                'anat2std_xfm',
                # Entity for selecting target resolution
                'resolution',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'resampling_reference']),
        name='outputnode',
    )

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref', mem_gb=0.3)

    run2target = pe.Node(niu.Merge(3), name='run2target', run_without_submitting=True)
    bold2target = pe.Node(niu.Merge(2), name='bold2target', run_without_submitting=True)
    resample = pe.Node(
        ResampleSeries(jacobian=jacobian),
        name='resample',
        n_procs=omp_nthreads,
        mem_gb=mem_gb['resampled'],
    )

    workflow.connect([
        (inputnode, gen_ref, [
            ('bold_ref_file', 'moving_image'),
            ('target_ref_file', 'fixed_image'),
            ('target_mask', 'fov_mask'),
            (('resolution', _is_native), 'keep_native'),
        ]),
        (inputnode, run2target, [
            ('run2boldref_xfm', 'in1'),
            ('boldref2anat_xfm', 'in2'),
            ('anat2std_xfm', 'in3'),
        ]),
        (inputnode, bold2target, [('motion_xfm', 'in1')]),
        (inputnode, resample, [('bold_file', 'in_file')]),
        (gen_ref, resample, [('out_file', 'ref_file')]),
        (run2target, bold2target, [('out', 'in2')]),
        (bold2target, resample, [('out', 'transforms')]),
        (gen_ref, outputnode, [('out_file', 'resampling_reference')]),
        (resample, outputnode, [('out_file', 'bold_file')]),
    ])  # fmt:skip

    if not fieldmap_id:
        return workflow

    distortion_params = pe.Node(
        DistortionParameters(
            metadata=metadata,
            fallback=fallback_total_readout_time,
        ),
        name='distortion_params',
        run_without_submitting=True,
    )
    fmap2target = pe.Node(niu.Merge(2), name='fmap2target', run_without_submitting=True)
    inverses = pe.Node(
        niu.Function(function=_gen_inverses),
        name='inverses',
        run_without_submitting=True,
    )

    fmap_recon = pe.Node(ReconstructFieldmap(), name='fmap_recon', mem_gb=1)

    workflow.connect([
        (inputnode, distortion_params, [('bold_file', 'in_file')]),
        (inputnode, fmap2target, [('run2fmap_xfm', 'in1')]),
        (gen_ref, fmap_recon, [('out_file', 'target_ref_file')]),
        (run2target, fmap2target, [('out', 'in2')]),
        (run2target, inverses, [('out', 'inlist')]),
        (inputnode, fmap_recon, [
            ('fmap_coeff', 'in_coeffs'),
            ('fmap_ref', 'fmap_ref_file'),
        ]),
        (fmap2target, fmap_recon, [('out', 'transforms')]),
        (inverses, fmap_recon, [('out', 'inverse')]),
        # Inject fieldmap correction into resample node
        (distortion_params, resample, [
            ('readout_time', 'ro_time'),
            ('pe_direction', 'pe_dir'),
        ]),
        (fmap_recon, resample, [('out_file', 'fieldmap')]),
    ])  # fmt:skip

    return workflow


def _gen_inverses(inlist: list) -> list[bool]:
    """Create a list indicating the first transform should be inverted.

    The input list is the collection of transforms that follow the
    inverted one.
    """
    from niworkflows.utils.connections import listify

    if not inlist:
        return [True]
    return [True] + [False] * len(listify(inlist))


def _is_native(value):
    return value == 'native'
