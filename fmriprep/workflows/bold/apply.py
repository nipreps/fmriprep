from __future__ import annotations

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.interfaces.nibabel import GenerateSamplingReference

from ...interfaces.resampling import ResampleSeries


def init_bold_volumetric_resample_wf(
    *,
    metadata: dict,
    mem_gb: dict[str, float],
    omp_nthreads: int = 1,
    name: str = 'bold_volumetric_resample_wf',
) -> pe.Workflow:
    """Resample a BOLD series to a volumetric target space.

    This workflow collates a sequence of transforms to resample a BOLD series
    in a single shot, including motion correction.

    .. workflow::

        from fmriprep.workflows.bold.resampling import init_bold_volumetric_resample_wf
        wf = init_bold_volumetric_resample_wf(
            metadata={
                'RepetitionTime': 2.0,
            },
            mem_gb={'resampled': 1},
        )

    Parameters
    ----------
    metadata
        BIDS metadata for BOLD file.
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

    boldref2target = pe.Node(niu.Merge(2), name='boldref2target', run_without_submitting=True)
    bold2target = pe.Node(niu.Merge(2), name='bold2target', run_without_submitting=True)
    resample = pe.Node(
        ResampleSeries(),
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
        (inputnode, boldref2target, [
            ('boldref2anat_xfm', 'in1'),
            ('anat2std_xfm', 'in2'),
        ]),
        (inputnode, bold2target, [('motion_xfm', 'in1')]),
        (inputnode, resample, [('bold_file', 'in_file')]),
        (gen_ref, resample, [('out_file', 'ref_file')]),
        (boldref2target, bold2target, [('out', 'in2')]),
        (bold2target, resample, [('out', 'transforms')]),
        (gen_ref, outputnode, [('out_file', 'resampling_reference')]),
        (resample, outputnode, [('out_file', 'bold_file')]),
    ])  # fmt:skip

    return workflow


def _is_native(value):
    return value == 'native'
