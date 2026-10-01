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


def init_bold_template_wf(
    *,
    num_bold_runs: int,
    unbiased: bool | None = None,
    upsample: tuple[float, float, float] | None = None,
    template_runs: list[bool] | None = None,
    omp_nthreads: int = 1,
    name: str = 'bold_template_wf',
) -> Workflow:
    """
    Register all BOLD runs to a common template reference.

    Parameters
    ----------
    num_bold_runs : :obj:`int`
        Number of BOLD runs.
    omp_nthreads : :obj:`int`
        Number of threads.
    unbiased : :obj:`bool` or None
        Whether to use an unbiased (iterative) registration strategy.
        When ``None`` (default), the strategy is chosen automatically:
        ``False`` (fixed first run as reference) for 2 runs, ``True``
        (iterative mean-shape template) for 3 or more runs.
    upsample : :obj:`tuple` or None
        Zooms to upsample the run references to before constructing the template,
        or ``None`` to construct it at the acquired resolution.
    template_runs : :obj:`list` of :obj:`bool` or None
        Whether each run's reference is averaged into the template image, one per run,
        or ``None`` (default) to average all runs. All runs define the template space.

    Inputs
    ------
    boldref_files
        List of BOLD reference files to be coregistered.

    Outputs
    -------
    boldref
        The computed BOLD template reference.
    boldref_files
        List of BOLD reference files (same as input).
    run2template_xfms
        Transforms from each run's original space to the boldref template

    """
    from niworkflows.interfaces.freesurfer import StructuralReference
    from niworkflows.interfaces.nitransforms import ConvertAffine

    from fmriprep.interfaces.resampling import ResampleTemplate, UpsampleToZooms

    if unbiased is None:
        unbiased = num_bold_runs >= 3

    workflow = Workflow(name=name)
    workflow.__desc__ = 'All BOLD runs were coregistered to '
    if unbiased:
        workflow.__desc__ += 'an unbiased session-level BOLD reference using an iterative template construction strategy.'
    else:
        workflow.__desc__ += "the first run's BOLD reference."

    if upsample:
        workflow.__desc__ += (
            f' The run references were upsampled to {upsample[0]:g}mm isotropic resolution '
            'prior to template construction, to reduce the interpolation error incurred '
            'when resampling.'
        )
    workflow.__desc__ += (
        ' The template image was computed as the voxelwise median of the '
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
            initial_timepoint=1,
            intensity_scaling=True,
            subsample_threshold=200,
            fixed_timepoint=not unbiased,
            no_iteration=not unbiased,
            transform_outputs=True,
            scaled_intensity_outputs=True,
            out_file='boldref_template.nii.gz',
        ),
        mem_gb=2 * num_bold_runs - 1,
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
    if template_runs is not None:
        resample_template.inputs.indices = template_runs

    if upsample:
        upsample_boldrefs = pe.Node(
            UpsampleToZooms(target_zooms=upsample), name='upsample_boldrefs'
        )

        workflow.connect([
            (inputnode, upsample_boldrefs, [('boldref_files', 'in_files')]),
            (upsample_boldrefs, boldref_template, [('out_files', 'in_files')]),
        ])  # fmt:skip
    else:
        workflow.connect(inputnode, 'boldref_files', boldref_template, 'in_files')

    workflow.connect([
        (inputnode, resample_template, [('boldref_files', 'in_files')]),
        (boldref_template, resample_template, [
            ('out_file', 'reference'),
            ('scaled_intensity_outputs', 'intensity_scales'),
        ]),
        (boldref_template, to_itk, [
            ('transform_outputs', 'in_xfm'),
        ]),
        (to_itk, resample_template, [('out_xfm', 'transforms')]),
        (resample_template, outputnode, [('out_file', 'boldref')]),
        (to_itk, outputnode, [
            ('out_xfm', 'run2template_xfms'),
        ]),
    ])  # fmt:skip

    return workflow
