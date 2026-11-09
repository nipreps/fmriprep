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
"""BOLD anatomical coregistration workflow."""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.base import Undefined
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.utils.connections import pop_file

from fmriprep import config
from fmriprep.interfaces import DerivativesDataSink
from fmriprep.interfaces.bids import BIDSURI

logger = config.loggers.workflow

INPUT_FIELDS = [
    'run_boldrefs',
    't1w_preproc',
    't1w_mask',
    't1w_dseg',
    'subjects_dir',
    'subject_id',
    'fsnative2t1w_xfm',
]
OUTPUT_FIELDS = [
    'template_boldrefs',
    'run2template_xfms',
    'template2anat_xfms',
    'run2anat_xfms',
    'fallbacks',
]
ANAT_REG_INPUTS = [
    ('t1w_preproc', 'inputnode.t1w_preproc'),
    ('t1w_mask', 'inputnode.t1w_mask'),
    ('t1w_dseg', 'inputnode.t1w_dseg'),
    ('subjects_dir', 'inputnode.subjects_dir'),
    ('subject_id', 'inputnode.subject_id'),
    ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm'),
]


def _expand(value, n):
    return [value] * n


def init_bold_anat_coreg_wf(
    *,
    bold_files: list[str],
    coreg_space: str,
    bold2anat_dof: int,
    bold2anat_init: str,
    use_bbr: bool | None,
    freesurfer: bool,
    omp_nthreads: int,
    mem_gb: float,
    sloppy: bool,
    output_dir: str,
    reference_anat: str,
    precomputed: dict | None = None,
    template_reference: int | None = None,
    template_runs: list[bool] | None = None,
    name: str = 'bold_anat_coreg_wf',
) -> Workflow:
    """
    Build a workflow to coregister BOLD run references to anatomical space.

    Behavior is controlled by ``coreg_space``. At ``session`` or ``subject``
    level, a common BOLD template is built from all run references and
    registered to the anatomical, composing per-run ``run->template->anat``
    transforms (:func:`init_bold_template_coreg_wf`). At ``run`` level, each run
    reference is registered to the anatomical independently
    (:func:`init_bold_run_coreg_wf`).

    Either way, per-run lists are returned so downstream workflows can be wired uniformly.

    Writes coregistration derivatives (template boldref, template mask,
    ``run2template_xfms`` and ``template2anat_xfm``). When a derivative is supplied
    via ``precomputed``, the corresponding computation and datasink are skipped and
    the precomputed path is reused.

    Parameters
    ----------
    precomputed
        Dictionary of precomputed coregistration derivatives to reuse. Recognized
        keys:

        ``template2anat_xfm``
            Run-level: list of per-run boldref->anat transforms (``None`` where
            absent). Template-level: a single template->anat transform. Where a
            transform is present, registration is skipped for that boldref.
        ``run2template_xfms``
            Template-level: list of per-run run->template transforms. When all runs
            are present, the template workflow is skipped and these transforms are
            applied to the run references to reconstruct the template space.
        ``boldref_template``
            Template-level: the precomputed template reference. When supplied
            alongside a complete set of ``run2template_xfms``, it is reused directly
            instead of reconstructing the reference from the template runs. The
            template brain mask is always derived from the run references.
    template_reference
        Index of the run defining the template space, or ``None`` for an unbiased
        template (see :func:`~fmriprep.workflows.bold.template.init_bold_template_wf`).
    template_runs
        Whether each run is averaged into the template image.

    Inputs
    ------
    run_boldrefs
        List of per-run SDC-corrected BOLD references.
    t1w_preproc
        Bias-corrected anatomical image.
    t1w_mask
        Skull-strip mask.
    t1w_dseg
        Tissue segmentation image.
    subjects_dir
        FreeSurfer SUBJECTS_DIR (may be undefined).
    subject_id
        FreeSurfer subject ID (may be undefined).
    fsnative2t1w_xfm
        Transform from FreeSurfer native to anatomical space (may be undefined).

    Outputs
    -------
    template_boldrefs
        Per-run boldref in coregistration target space: template repeated
        n times (template-level) or each run's own boldref (run-level).
    run2template_xfms
        Per-run transform from run space to the coregistration template.
        Identity transforms for run-level coregistration.
    template2anat_xfms
        Per-run transform from coregistration target to anatomical space.
    run2anat_xfms
        Per-run composed run->anat transform.
    fallbacks
        Per-run fallback flags from registration.
    """

    if coreg_space == 'run':
        init_coreg_wf = init_bold_run_coreg_wf
        template_kwargs = {}
    else:
        init_coreg_wf = init_bold_template_coreg_wf
        template_kwargs = {
            'template_reference': template_reference,
            'template_runs': template_runs,
        }
    return init_coreg_wf(
        **template_kwargs,
        bold_files=bold_files,
        coreg_space=coreg_space,
        bold2anat_dof=bold2anat_dof,
        bold2anat_init=bold2anat_init,
        use_bbr=use_bbr,
        freesurfer=freesurfer,
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
        sloppy=sloppy,
        output_dir=output_dir,
        reference_anat=reference_anat,
        precomputed=precomputed,
        name=name,
    )


def init_bold_run_coreg_wf(
    *,
    bold_files: list[str],
    coreg_space: str,
    bold2anat_dof: int,
    bold2anat_init: str,
    use_bbr: bool | None,
    freesurfer: bool,
    omp_nthreads: int,
    mem_gb: float,
    sloppy: bool,
    output_dir: str,
    reference_anat: str,
    precomputed: dict | None = None,
    name: str = 'bold_run_coreg_wf',
) -> Workflow:
    """Register each BOLD run reference to the anatomical independently.

    Shares the input/output signature of :func:`init_bold_anat_coreg_wf`.
    ``run2template_xfms`` are returned as identity transforms and ``run2anat_xfms``
    equal the per-run ``template2anat_xfms``. Runs whose ``template2anat_xfm`` is
    supplied via ``precomputed`` skip registration.
    """
    from fmriprep.utils.misc import get_wf_name
    from fmriprep.workflows.bold.outputs import init_ds_registration_wf
    from fmriprep.workflows.bold.registration import init_bold_reg_wf

    precomputed = precomputed or {}
    bids_root = str(config.execution.bids_dir)
    n_runs = len(bold_files)
    bold_ids = [get_wf_name(bold_file, None).removesuffix('_wf') for bold_file in bold_files]
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=OUTPUT_FIELDS), name='outputnode')

    template2anat_xfm = precomputed.get('template2anat_xfm') or [None] * n_runs
    outputnode.inputs.run2template_xfms = [Undefined] * n_runs

    merge_template2anat = pe.Node(
        niu.Merge(n_runs), name='merge_template2anat', run_without_submitting=True
    )
    merge_fallbacks = pe.Node(
        niu.Merge(n_runs), name='merge_fallbacks', run_without_submitting=True
    )

    for i, (bold_file, bold_id) in enumerate(zip(bold_files, bold_ids, strict=True)):
        select_boldref = pe.Node(
            niu.Select(index=i), name=f'select_boldref_{bold_id}', run_without_submitting=True
        )
        workflow.connect(inputnode, 'run_boldrefs', select_boldref, 'inlist')

        if template2anat_xfm[i]:
            setattr(merge_template2anat.inputs, f'in{i + 1}', template2anat_xfm[i])
            setattr(merge_fallbacks.inputs, f'in{i + 1}', False)
            continue

        reg_wf = init_bold_reg_wf(
            name=f'boldref_reg_{bold_id}_wf',
            bold2anat_dof=bold2anat_dof,
            bold2anat_init=bold2anat_init,
            use_bbr=use_bbr,
            freesurfer=freesurfer,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            sloppy=sloppy,
        )
        ds_template2anat_wf = init_ds_registration_wf(
            bids_root=bids_root,
            source_file=bold_file,
            output_dir=output_dir,
            source=coreg_space,
            dest=reference_anat,
            desc='coreg',
            name=f'ds_template2anat_{bold_id}_wf',
        )

        workflow.connect([
            (inputnode, reg_wf, list(ANAT_REG_INPUTS)),
            (select_boldref, reg_wf, [('out', 'inputnode.ref_bold_brain')]),
            (select_boldref, ds_template2anat_wf, [('out', 'inputnode.source_files')]),
            (reg_wf, ds_template2anat_wf, [
                ('outputnode.itk_bold_to_t1', 'inputnode.xform'),
                ('outputnode.metadata', 'inputnode.metadata'),
            ]),
            (ds_template2anat_wf, merge_template2anat, [('outputnode.xform', f'in{i + 1}')]),
            (reg_wf, merge_fallbacks, [('outputnode.fallback', f'in{i + 1}')]),
        ])  # fmt:skip

    workflow.connect([
        (inputnode, outputnode, [
            ('run_boldrefs', 'template_boldrefs'),
        ]),
        (merge_template2anat, outputnode, [
            ('out', 'template2anat_xfms'),
            ('out', 'run2anat_xfms'),
        ]),
        (merge_fallbacks, outputnode, [('out', 'fallbacks')]),
    ])  # fmt:skip

    return workflow


def init_bold_template_coreg_wf(
    *,
    bold_files: list[str],
    coreg_space: str,
    bold2anat_dof: int,
    bold2anat_init: str,
    use_bbr: bool | None,
    freesurfer: bool,
    omp_nthreads: int,
    mem_gb: float,
    sloppy: bool,
    output_dir: str,
    reference_anat: str,
    precomputed: dict | None = None,
    template_reference: int | None = None,
    template_runs: list[bool] | None = None,
    name: str = 'bold_template_coreg_wf',
) -> Workflow:
    """Coregister BOLD runs through a common template to the anatomical.

    Shares the input/output signature of :func:`init_bold_anat_coreg_wf`. The run
    references are combined into a ``coreg_space`` (``session`` or ``subject``)
    template, that template is registered to the anatomical, and per-run
    ``run->template->anat`` transforms are composed.

    The runs defining the template space and those averaged into the template image
    are recorded in the ``TemplateSpace`` and ``TemplateData`` metadata of the template.
    When neither ``template_reference`` nor ``template_runs`` is given, the strategy
    is chosen from the number of runs.
    """
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs

    from fmriprep.utils.bids import (
        GROUP_DISMISS_ENTITIES,
        bold_template_zooms,
        resolve_selection,
    )
    from fmriprep.utils.misc import get_wf_name
    from fmriprep.workflows.bold.outputs import init_ds_registration_wf
    from fmriprep.workflows.bold.registration import init_bold_reg_wf
    from fmriprep.workflows.bold.template import init_bold_template_wf, select_by_mask

    precomputed = precomputed or {}
    bids_root = str(config.execution.bids_dir)
    n_runs = len(bold_files)
    if template_reference is None and template_runs is None:
        template_reference, template_runs = resolve_selection(['n/a'] * n_runs)
    template_runs = template_runs or [True] * n_runs
    bold_ids = [get_wf_name(bold_file, None).removesuffix('_wf') for bold_file in bold_files]

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=OUTPUT_FIELDS), name='outputnode')

    _dismiss = list(GROUP_DISMISS_ENTITIES)
    if coreg_space == 'subject':
        _dismiss.append('session')

    boldref_template = precomputed.get('boldref_template')
    run2template_xfms = precomputed.get('run2template_xfms') or [None] * n_runs
    template2anat_xfm = precomputed.get('template2anat_xfm')
    if isinstance(template2anat_xfm, (list, tuple)):
        template2anat_xfm = template2anat_xfm[0] if template2anat_xfm else None

    template_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'run2template_xfms']),
        name='template_buffer',
    )
    reg_buffer = pe.Node(
        niu.IdentityInterface(fields=['template2anat_xfm', 'fallback']),
        name='reg_buffer',
    )
    # Connected only when something is computed, but the caller connects to it regardless
    workflow.add_nodes([inputnode])

    # Populate buffers; these can be overridden by connections.
    if boldref_template:
        template_buffer.inputs.boldref = boldref_template
    if have_run2template_xfms := all(run2template_xfms):
        template_buffer.inputs.run2template_xfms = run2template_xfms
    else:
        if any(run2template_xfms):
            logger.warning(
                f'Only some run-to-{coreg_space} transforms were found; '
                'ignoring and recomputing the template.'
            )
        if template2anat_xfm:
            template2anat_xfm = None
            logger.warning(
                f'A precomputed {coreg_space}-to-anat transform was found without a '
                f'complete set of run-to-{coreg_space} transforms; '
                'ignoring it and recomputing registration against the rebuilt template.'
            )
    if template2anat_xfm:
        logger.info(f'Found precomputed {coreg_space}-to-anat transform; skipping coregistration.')
        reg_buffer.inputs.template2anat_xfm = template2anat_xfm
        reg_buffer.inputs.fallback = False

    # Collation nodes
    expand_template2anat = pe.Node(niu.Function(function=_expand), name='expand_template2anat')
    expand_boldref = pe.Node(niu.Function(function=_expand), name='expand_boldref')
    expand_fallback = pe.Node(niu.Function(function=_expand), name='expand_fallback')
    for node in (expand_template2anat, expand_boldref, expand_fallback):
        node.inputs.n = n_runs
        node.run_without_submitting = True

    merge_run2anat = pe.MapNode(
        niu.Merge(2), iterfield=['in1'], name='merge_run2anat', run_without_submitting=True
    )
    run2anat_xfms = pe.MapNode(ConcatenateXFMs(), iterfield=['in_xfms'], name='run2anat_xfms')

    workflow.connect([
        # Create run2anat transforms
        (template_buffer, merge_run2anat, [('run2template_xfms', 'in1')]),
        (reg_buffer, merge_run2anat, [('template2anat_xfm', 'in2')]),
        (merge_run2anat, run2anat_xfms, [('out', 'in_xfms')]),
        # Broadcast single values to per-run lists
        (template_buffer, expand_boldref, [('boldref', 'value')]),
        (reg_buffer, expand_template2anat, [('template2anat_xfm', 'value')]),
        (reg_buffer, expand_fallback, [('fallback', 'value')]),
        # Connect outputs
        (expand_boldref, outputnode, [('out', 'template_boldrefs')]),
        (expand_fallback, outputnode, [('out', 'fallbacks')]),
        (expand_template2anat, outputnode, [('out', 'template2anat_xfms')]),
        (template_buffer, outputnode, [('run2template_xfms', 'run2template_xfms')]),
        (run2anat_xfms, outputnode, [('out_xfm', 'run2anat_xfms')]),
    ])  # fmt:skip

    upsample = None
    if config.workflow.bold_coreg_upsample and not (boldref_template and have_run2template_xfms):
        upsample = bold_template_zooms(bold_files)

    template_sources = pe.Node(
        BIDSURI(
            numinputs=1, dataset_links=config.execution.dataset_links, out_dir=str(output_dir)
        ),
        name='template_sources',
        run_without_submitting=True,
    )
    template_data = pe.Node(
        niu.Function(function=select_by_mask), name='template_data', run_without_submitting=True
    )
    template_data.inputs.mask = template_runs

    # Single template is written
    ds_boldref_template = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_files[0],
            space=coreg_space,
            suffix='boldref',
            compress=True,
            dismiss_entities=_dismiss,
        ),
        name='ds_boldref_template',
        run_without_submitting=True,
    )

    if not (boldref_template and have_run2template_xfms):
        workflow.connect([
            (inputnode, template_sources, [('run_boldrefs', 'in1')]),
            (template_sources, template_data, [('out', 'inlist')]),
            (template_sources, ds_boldref_template, [('out', 'Sources')]),
            (template_data, ds_boldref_template, [('out', 'TemplateData')]),
            (ds_boldref_template, template_buffer, [('out_file', 'boldref')]),
        ])  # fmt:skip

    if not have_run2template_xfms:
        logger.info('Constructing BOLD template from run references.')

        bold_template_wf = init_bold_template_wf(
            num_bold_runs=n_runs,
            reference=template_reference,
            template_runs=template_runs,
            upsample=upsample,
            omp_nthreads=omp_nthreads,
        )

        template_space = pe.Node(
            niu.Function(function=select_by_mask),
            name='template_space',
            run_without_submitting=True,
        )
        template_space.inputs.mask = (
            template_runs
            if template_reference is None
            else [i == template_reference for i in range(n_runs)]
        )

        merge_run2template = pe.Node(
            niu.Merge(n_runs), name='merge_run2template', run_without_submitting=True
        )

        workflow.connect([
            (inputnode, bold_template_wf, [('run_boldrefs', 'inputnode.boldref_files')]),
            (template_sources, template_space, [('out', 'inlist')]),
            (template_space, ds_boldref_template, [('out', 'TemplateSpace')]),
            (bold_template_wf, ds_boldref_template, [('outputnode.boldref', 'in_file')]),
            (merge_run2template, template_buffer, [('out', 'run2template_xfms')]),
        ])  # fmt:skip

        for i, (bold_file, bold_id) in enumerate(zip(bold_files, bold_ids, strict=True)):
            select_run2template = pe.Node(
                niu.Select(index=i),
                name=f'select_run2template_{bold_id}',
                run_without_submitting=True,
            )
            ds_run2template_wf = init_ds_registration_wf(
                bids_root=bids_root,
                source_file=bold_file,
                output_dir=output_dir,
                source='run',
                dest=coreg_space,
                desc='coreg',
                name=f'ds_run2template_{bold_id}_wf',
            )
            workflow.connect([
                (inputnode, ds_run2template_wf, [('run_boldrefs', 'inputnode.source_files')]),
                (bold_template_wf, select_run2template, [('outputnode.run2template_xfms', 'inlist')]),
                (select_run2template, ds_run2template_wf, [('out', 'inputnode.xform')]),
                (ds_run2template_wf, merge_run2template, [('outputnode.xform', f'in{i + 1}')]),
            ])  # fmt:skip

    elif boldref_template:
        logger.info('Reusing precomputed boldref template; skipping reconstruction.')
    else:
        from fmriprep.interfaces.resampling import ResampleTemplate, UpsampleToZooms

        logger.info(
            f'Found precomputed run2{coreg_space} transforms; '
            'reconstructing boldref template from run references.'
        )
        select_reference = pe.Node(
            niu.Select(
                index=template_runs.index(True)
                if template_reference is None
                else template_reference
            ),
            name='select_reference',
            run_without_submitting=True,
        )
        reconstruct_template = pe.Node(
            ResampleTemplate(transforms=run2template_xfms, indices=template_runs),
            mem_gb=0.2 * n_runs,
            name='reconstruct_template',
        )
        workflow.connect([
            (inputnode, select_reference, [('run_boldrefs', 'inlist')]),
            (inputnode, reconstruct_template, [('run_boldrefs', 'in_files')]),
            (reconstruct_template, ds_boldref_template, [('out_file', 'in_file')]),
        ])  # fmt:skip

        if upsample:
            upsample_reference = pe.Node(
                UpsampleToZooms(target_zooms=upsample), name='upsample_reference'
            )
            workflow.connect([
                (select_reference, upsample_reference, [('out', 'in_files')]),
                (upsample_reference, reconstruct_template, [(('out_files', pop_file), 'reference')]),
            ])  # fmt:skip
        else:
            workflow.connect(select_reference, 'out', reconstruct_template, 'reference')

    if not template2anat_xfm:
        logger.info('Registering BOLD template to anatomical.')
        boldref_reg_wf = init_bold_reg_wf(
            name='boldref_reg_wf',
            bold2anat_dof=bold2anat_dof,
            bold2anat_init=bold2anat_init,
            use_bbr=use_bbr,
            freesurfer=freesurfer,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            sloppy=sloppy,
        )

        # Single template->anat transform is written
        ds_template2anat_wf = init_ds_registration_wf(
            bids_root=bids_root,
            source_file=bold_files[0],
            output_dir=output_dir,
            source=coreg_space,
            dest=reference_anat,
            desc='coreg',
            dismiss_entities=_dismiss,
            name='ds_template2anat_wf',
        )
        workflow.connect([
            (inputnode, boldref_reg_wf, list(ANAT_REG_INPUTS)),
            (inputnode, ds_template2anat_wf, [('run_boldrefs', 'inputnode.source_files')]),
            (template_buffer, boldref_reg_wf, [('boldref', 'inputnode.ref_bold_brain')]),
            (boldref_reg_wf, ds_template2anat_wf, [
                ('outputnode.itk_bold_to_t1', 'inputnode.xform'),
                ('outputnode.metadata', 'inputnode.metadata'),
            ]),
            (boldref_reg_wf, reg_buffer, [('outputnode.fallback', 'fallback')]),
            (ds_template2anat_wf, reg_buffer, [('outputnode.xform', 'template2anat_xfm')]),
        ])  # fmt:skip

    return workflow
