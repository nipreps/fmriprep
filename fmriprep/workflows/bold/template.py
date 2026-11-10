# Copyright The NiPreps Developers <nipreps@gmail.com>
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
BOLD template creation workflow.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_template_wf
.. autofunction:: init_register_excluded_wf

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_bold_template_wf(
    *,
    num_bold_runs: int,
    reference: int | None = None,
    template_runs: list[bool] | None = None,
    upsample: tuple[float, float, float] | None = None,
    omp_nthreads: int = 1,
    name: str = 'bold_template_wf',
) -> Workflow:
    """
    Register all BOLD runs to a common template reference.

    Parameters
    ----------
    num_bold_runs : :obj:`int`
        Number of BOLD runs.
    reference : :obj:`int` or None
        Index of the run whose reference defines the template space, to which the
        other template runs are registered.
        When ``None`` (default), an unbiased (iterative) mean-shape template is
        constructed from the template runs.
    template_runs : :obj:`list` of :obj:`bool` or None
        Whether each run's reference is used to construct the template, one per run,
        or ``None`` (default) to use all runs. The remaining runs are registered to
        the finished template.
    upsample : :obj:`tuple` or None
        Isotropic zooms to upsample the run references to before registration,
        or ``None`` to register them at the acquired resolution.
    omp_nthreads : :obj:`int`
        Number of threads.

    Inputs
    ------
    boldref_files
        List of BOLD reference files to be coregistered.

    Outputs
    -------
    boldref
        The computed BOLD template reference.
    run2template_xfms
        Transforms from each run's original space to the boldref template

    """
    from niworkflows.interfaces.freesurfer import StructuralReference
    from niworkflows.interfaces.nitransforms import ConvertAffine

    from fmriprep.interfaces.resampling import ResampleTemplate, UpsampleToZooms

    if template_runs is None:
        template_runs = [True] * num_bold_runs
    if len(template_runs) != num_bold_runs or not any(template_runs):
        raise ValueError(f'Expected {num_bold_runs} template runs, with at least one selected.')
    template_indices = [i for i, keep in enumerate(template_runs) if keep]
    excluded_indices = [i for i, keep in enumerate(template_runs) if not keep]
    if reference is not None and reference not in template_indices:
        raise ValueError(f'The reference run ({reference}) must be one of the template runs.')
    n_template = len(template_indices)
    unbiased = reference is None

    workflow = Workflow(name=name)
    workflow.__desc__ = 'All BOLD runs were coregistered to '
    if unbiased:
        workflow.__desc__ += 'an unbiased BOLD reference'
        if excluded_indices:
            workflow.__desc__ += f', constructed from {n_template} of the {num_bold_runs} runs'
        workflow.__desc__ += ' using an iterative template construction strategy.'
    else:
        workflow.__desc__ += (
            'the BOLD reference of a selected run.'
            if reference or excluded_indices
            else "the first run's BOLD reference."
        )

    if upsample:
        workflow.__desc__ += (
            f' The run references were upsampled to {upsample[0]:g}mm isotropic resolution '
            'prior to registration.'
        )
    if n_template == 1:
        workflow.__desc__ += (
            " The template image was the selected run's reference, resampled into the "
            'template space.'
        )
    else:
        selected = f'{n_template} selected ' if excluded_indices else ''
        workflow.__desc__ += (
            f' The template image was computed as the voxelwise median of the {selected}'
            'intensity-normalized run references, each resampled once into the template space.'
        )

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['boldref_files']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'run2template_xfms']),
        name='outputnode',
    )

    # A single reference is copied with an identity transform
    boldref_template = pe.Node(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1 if unbiased else template_indices.index(reference) + 1,
            intensity_scaling=True,
            subsample_threshold=200,
            fixed_timepoint=not unbiased,
            no_iteration=not unbiased,
            transform_outputs=True,
            out_file='boldref_template.nii.gz',
        ),
        mem_gb=2 * n_template - 1,
        name='boldref_template',
        n_procs=omp_nthreads,
    )

    to_itk = pe.MapNode(
        ConvertAffine(in_fmt='fs', out_fmt='itk'),
        iterfield=['in_xfm'],
        name='to_itk',
    )

    # Only the grid, transforms and intensity scales of mri_robust_template are kept
    resample_template = pe.Node(
        ResampleTemplate(),
        mem_gb=0.2 * n_template,
        name='resample_template',
    )

    boldref_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref_files']), name='boldref_buffer'
    )
    if upsample:
        upsample_boldrefs = pe.Node(
            UpsampleToZooms(target_zooms=upsample), name='upsample_boldrefs'
        )

        workflow.connect([
            (inputnode, upsample_boldrefs, [('boldref_files', 'in_files')]),
            (upsample_boldrefs, boldref_buffer, [('out_files', 'boldref_files')]),
        ])  # fmt:skip
    else:
        workflow.connect(inputnode, 'boldref_files', boldref_buffer, 'boldref_files')

    # single reference writes no intensity scale
    if n_template > 1:
        boldref_template.inputs.scaled_intensity_outputs = True
        workflow.connect(
            boldref_template, 'scaled_intensity_outputs', resample_template, 'intensity_scales'
        )

    workflow.connect([
        (boldref_template, resample_template, [('out_file', 'reference')]),
        (boldref_template, to_itk, [('transform_outputs', 'in_xfm')]),
        (to_itk, resample_template, [('out_xfm', 'transforms')]),
        (resample_template, outputnode, [('out_file', 'boldref')]),
    ])  # fmt:skip

    # If no runs are excluded, connect the template directly to the output
    if not excluded_indices:
        workflow.connect([
            (inputnode, resample_template, [('boldref_files', 'in_files')]),
            (boldref_buffer, boldref_template, [('boldref_files', 'in_files')]),
            (to_itk, outputnode, [('out_xfm', 'run2template_xfms')]),
        ])  # fmt:skip
        return workflow

    # Excluded runs do not shape the template, but are registered to it from the same
    # (upsampled) references as the template runs, for consistency
    select_template = pe.Node(
        niu.Select(index=template_indices), name='select_template', run_without_submitting=True
    )
    # The template image is resampled from the original references, not the upsampled ones
    select_resample = pe.Node(
        niu.Select(index=template_indices), name='select_resample', run_without_submitting=True
    )
    select_excluded = pe.Node(
        niu.Select(index=excluded_indices), name='select_excluded', run_without_submitting=True
    )
    register_excluded_wf = init_register_excluded_wf(
        num_bold_runs=len(excluded_indices), omp_nthreads=omp_nthreads
    )
    merge_xfms = pe.Node(niu.Merge(2), name='merge_xfms', run_without_submitting=True)
    # Restore the run order of the template and excluded transforms
    merged_runs = template_indices + excluded_indices
    sort_xfms = pe.Node(
        niu.Select(index=[merged_runs.index(i) for i in range(num_bold_runs)]),
        name='sort_xfms',
        run_without_submitting=True,
    )

    workflow.connect([
        (boldref_buffer, select_template, [('boldref_files', 'inlist')]),
        (inputnode, select_resample, [('boldref_files', 'inlist')]),
        (boldref_buffer, select_excluded, [('boldref_files', 'inlist')]),
        (select_template, boldref_template, [('out', 'in_files')]),
        (select_resample, resample_template, [('out', 'in_files')]),
        (select_excluded, register_excluded_wf, [('out', 'inputnode.boldref_files')]),
        (resample_template, register_excluded_wf, [('out_file', 'inputnode.boldref')]),
        (to_itk, merge_xfms, [('out_xfm', 'in1')]),
        (register_excluded_wf, merge_xfms, [('outputnode.run2template_xfms', 'in2')]),
        (merge_xfms, sort_xfms, [('out', 'inlist')]),
        (sort_xfms, outputnode, [('out', 'run2template_xfms')]),
    ])  # fmt:skip

    return workflow


def init_register_excluded_wf(
    *,
    num_bold_runs: int,
    omp_nthreads: int = 1,
    name: str = 'register_excluded_wf',
) -> Workflow:
    """
    Register the references of runs excluded from a BOLD template to it.

    Parameters
    ----------
    num_bold_runs : :obj:`int`
        Number of excluded BOLD runs.
    omp_nthreads : :obj:`int`
        Number of threads.

    Inputs
    ------
    boldref_files
        List of BOLD reference files of the excluded runs.
    boldref
        The BOLD template reference.

    Outputs
    -------
    run2template_xfms
        Transforms from each excluded run's original space to the boldref template.

    """
    from nipype.interfaces.freesurfer import RobustRegister
    from niworkflows.interfaces.nitransforms import ConvertAffine

    workflow = Workflow(name=name)
    workflow.__desc__ = (
        ' The remaining run was registered to the resulting template.'
        if num_bold_runs == 1
        else ' The remaining runs were registered to the resulting template.'
    )

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['boldref_files', 'boldref']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['run2template_xfms']),
        name='outputnode',
    )

    register_runs = pe.MapNode(
        RobustRegister(auto_sens=True, est_int_scale=True, out_reg_file=True),
        iterfield=['source_file'],
        name='register_runs',
        n_procs=omp_nthreads,
    )
    to_itk = pe.MapNode(
        ConvertAffine(in_fmt='fs', out_fmt='itk'),
        iterfield=['in_xfm'],
        name='to_itk',
    )

    workflow.connect([
        (inputnode, register_runs, [
            ('boldref_files', 'source_file'),
            ('boldref', 'target_file'),
        ]),
        (register_runs, to_itk, [('out_reg_file', 'in_xfm')]),
        (to_itk, outputnode, [('out_xfm', 'run2template_xfms')]),
    ])  # fmt:skip

    return workflow
