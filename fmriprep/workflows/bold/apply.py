from __future__ import annotations

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.interfaces.nibabel import GenerateSamplingReference

from ...interfaces.resampling import ResampleSeries


def init_pet_volumetric_resample_wf(
    *,
    mem_gb: dict[str, float],
    omp_nthreads: int = 1,
    name: str = 'pet_volumetric_resample_wf',
) -> pe.Workflow:
    """Resample a PET series to a volumetric target space.

    This workflow collates a sequence of transforms to resample a PET series
    in a single shot, including motion correction.

    .. workflow::

        from fmriprep.workflows.pet.resampling import init_pet_volumetric_resample_wf
        wf = init_pet_volumetric_resample_wf(
            mem_gb={'resampled': 1},
        )

    Parameters
    ----------
    omp_nthreads
        Maximum number of threads an individual process may use.
    name
        Name of workflow (default: ``pet_volumetric_resample_wf``)

    Inputs
    ------
    pet_file
        PET series to resample.
    pet_ref_file
        Reference image to which PET series is aligned.
    target_ref_file
        Reference image defining the target space.
    target_mask
        Brain mask corresponding to ``target_ref_file``.
        This is used to define the field of view for the resampled PET series.
    motion_xfm
        List of affine transforms aligning each volume to ``pet_ref_file``.
        If undefined, no motion correction is performed.
    petref2anat_xfm
        Affine transform from ``pet_ref_file`` to the anatomical reference image.
    anat2std_xfm
        Affine transform from the anatomical reference image to standard space.
        Leave undefined to resample to anatomical reference space.

    Outputs
    -------
    pet_file
        The ``pet_file`` input, resampled to ``target_ref_file`` space.
    resampling_reference
        An empty reference image with the correct affine and header for resampling
        further images into the PET series' space.

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet_file',
                'pet_ref_file',
                'target_ref_file',
                'target_mask',
                # HMC
                'motion_xfm',
                # Anatomical
                'petref2anat_xfm',
                # Template
                'anat2std_xfm',
                # Entity for selecting target resolution
                'resolution',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'resampling_reference']),
        name='outputnode',
    )

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref', mem_gb=0.3)

    petref2target = pe.Node(niu.Merge(2), name='petref2target', run_without_submitting=True)
    pet2target = pe.Node(niu.Merge(2), name='pet2target', run_without_submitting=True)
    resample = pe.Node(
        ResampleSeries(),
        name='resample',
        n_procs=omp_nthreads,
        mem_gb=mem_gb['resampled'],
    )

    workflow.connect([
        (inputnode, gen_ref, [
            ('pet_ref_file', 'moving_image'),
            ('target_ref_file', 'fixed_image'),
            ('target_mask', 'fov_mask'),
            (('resolution', _is_native), 'keep_native'),
        ]),
        (inputnode, petref2target, [
            ('petref2anat_xfm', 'in1'),
            ('anat2std_xfm', 'in2'),
        ]),
        (inputnode, pet2target, [('motion_xfm', 'in1')]),
        (inputnode, resample, [('pet_file', 'in_file')]),
        (gen_ref, resample, [('out_file', 'ref_file')]),
        (petref2target, pet2target, [('out', 'in2')]),
        (pet2target, resample, [('out', 'transforms')]),
        (gen_ref, outputnode, [('out_file', 'resampling_reference')]),
        (resample, outputnode, [('out_file', 'pet_file')]),
    ])  # fmt:skip

    return workflow


def _is_native(value):
    return value == 'native'
