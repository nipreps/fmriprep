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

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def select_by_mask(inlist, mask, invert=False):
    """Select the items of ``inlist`` where ``mask`` is true (or false, if ``invert``)."""
    return [item for item, keep in zip(inlist, mask, strict=True) if bool(keep) != invert]


def _interleave(included, excluded, mask):
    included, excluded = iter(included), iter(excluded)
    return [next(included) if keep else next(excluded) for keep in mask]


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
        Index of the run whose reference defines the template space, to which all
        other runs are registered.
        When ``None`` (default), an unbiased (iterative) mean-shape template is
        constructed from the runs in ``template_runs``, and any remaining runs are
        registered to it afterwards.
    template_runs : :obj:`list` of :obj:`bool` or None
        Whether each run's reference is averaged into the template image, one per run,
        or ``None`` (default) to average all runs.
    upsample : :obj:`tuple` or None
        Isotropic zooms to upsample the run references to before constructing the
        template, or ``None`` to construct it at the acquired resolution.
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
    from nipype.interfaces.freesurfer import RobustRegister
    from niworkflows.interfaces.freesurfer import StructuralReference
    from niworkflows.interfaces.nitransforms import ConvertAffine

    from fmriprep.interfaces.resampling import ResampleTemplate, UpsampleToZooms

    if template_runs is None:
        template_runs = [True] * num_bold_runs
    if len(template_runs) != num_bold_runs or not any(template_runs):
        raise ValueError(f'Expected {num_bold_runs} template runs, with at least one selected.')
    n_template = sum(template_runs)
    unbiased = reference is None
    register_excluded = unbiased and n_template < num_bold_runs

    workflow = Workflow(name=name)
    workflow.__desc__ = 'All BOLD runs were coregistered to '
    if unbiased:
        workflow.__desc__ += (
            'an unbiased BOLD reference using an iterative template construction strategy'
        )
        if register_excluded:
            workflow.__desc__ += (
                f' over {n_template} of the {num_bold_runs} runs, and the remaining runs '
                'were registered to the resulting template'
            )
        workflow.__desc__ += '.'
    else:
        workflow.__desc__ += (
            'the BOLD reference of a selected run.'
            if reference or n_template < num_bold_runs
            else "the first run's BOLD reference."
        )

    if upsample:
        workflow.__desc__ += (
            f' The run references were upsampled to {upsample[0]:g}mm isotropic resolution '
            'prior to template construction.'
        )
    if n_template == 1:
        workflow.__desc__ += (
            " The template image was the selected run's intensity-normalized reference, "
            'resampled once into the template space.'
        )
    else:
        selected = f'{n_template} selected ' if n_template < num_bold_runs else ''
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

    boldref_template = pe.Node(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1 if unbiased else reference + 1,
            intensity_scaling=True,
            subsample_threshold=200,
            fixed_timepoint=not unbiased,
            no_iteration=not unbiased,
            transform_outputs=True,
            scaled_intensity_outputs=True,
            out_file='boldref_template.nii.gz',
        ),
        mem_gb=2 * (n_template if register_excluded else num_bold_runs) - 1,
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
        mem_gb=0.2 * num_bold_runs,
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

    workflow.connect([
        (boldref_template, resample_template, [
            ('out_file', 'reference'),
            ('scaled_intensity_outputs', 'intensity_scales'),
        ]),
        (boldref_template, to_itk, [('transform_outputs', 'in_xfm')]),
        (to_itk, resample_template, [('out_xfm', 'transforms')]),
        (resample_template, outputnode, [('out_file', 'boldref')]),
    ])  # fmt:skip

    if not register_excluded:
        resample_template.inputs.indices = template_runs
        workflow.connect([
            (inputnode, resample_template, [('boldref_files', 'in_files')]),
            (boldref_buffer, boldref_template, [('boldref_files', 'in_files')]),
            (to_itk, outputnode, [('out_xfm', 'run2template_xfms')]),
        ])  # fmt:skip
        return workflow

    # Unbiased template over the selected runs, to which the remaining runs are registered
    select_template = pe.Node(
        niu.Function(function=select_by_mask), name='select_template', run_without_submitting=True
    )
    select_template.inputs.mask = template_runs
    select_resample = pe.Node(
        niu.Function(function=select_by_mask), name='select_resample', run_without_submitting=True
    )
    select_resample.inputs.mask = template_runs
    select_excluded = pe.Node(
        niu.Function(function=select_by_mask), name='select_excluded', run_without_submitting=True
    )
    select_excluded.inputs.mask = template_runs
    select_excluded.inputs.invert = True

    excluded_to_itk = pe.MapNode(
        ConvertAffine(in_fmt='fs', out_fmt='itk'),
        iterfield=['in_xfm'],
        name='excluded_to_itk',
    )
    register_excluded_runs = pe.MapNode(
        RobustRegister(auto_sens=True, est_int_scale=True, out_reg_file=True),
        iterfield=['source_file'],
        name='register_excluded_runs',
        n_procs=omp_nthreads,
    )
    merge_xfms = pe.Node(
        niu.Function(function=_interleave), name='merge_xfms', run_without_submitting=True
    )
    merge_xfms.inputs.mask = template_runs

    workflow.connect([
        (boldref_buffer, select_template, [('boldref_files', 'inlist')]),
        (inputnode, select_resample, [('boldref_files', 'inlist')]),
        (boldref_buffer, select_excluded, [('boldref_files', 'inlist')]),
        (select_template, boldref_template, [('out', 'in_files')]),
        (select_resample, resample_template, [('out', 'in_files')]),
        (select_excluded, register_excluded_runs, [('out', 'source_file')]),
        (resample_template, register_excluded_runs, [('out_file', 'target_file')]),
        (register_excluded_runs, excluded_to_itk, [('out_reg_file', 'in_xfm')]),
        (to_itk, merge_xfms, [('out_xfm', 'included')]),
        (excluded_to_itk, merge_xfms, [('out_xfm', 'excluded')]),
        (merge_xfms, outputnode, [('out', 'run2template_xfms')]),
    ])  # fmt:skip

    return workflow
