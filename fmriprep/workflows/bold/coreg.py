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

import logging

from nipype.interfaces import utility as niu
from nipype.interfaces.base import Undefined
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from fmriprep import config
from fmriprep.interfaces import DerivativesDataSink
from fmriprep.interfaces.bids import BIDSURI

logger = logging.getLogger('nipype.workflow')

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
            absent). Session-level: a single template->anat transform. Where a
            transform is present, registration is skipped for that boldref.
        ``run2template_xfms``
            Session-level: list of per-run run->template transforms. When all runs
            are present, the template workflow is skipped and these transforms are
            applied to the run references to reconstruct the template space.
        ``boldref_template``
            Session-level: the precomputed template reference. When supplied
            alongside a complete set of ``run2template_xfms``, it is reused directly
            instead of reconstructing the reference from run 0. The template brain
            mask is always derived from the run references.

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
        Per-run boldref in coregistration target space: session template repeated
        n times (session-level) or each run's own boldref (run-level).
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

    init_coreg_wf = init_bold_run_coreg_wf if coreg_space == 'run' else init_bold_template_coreg_wf
    return init_coreg_wf(
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
            (inputnode, reg_wf, ANAT_REG_INPUTS),
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
    name: str = 'bold_template_coreg_wf',
) -> Workflow:
    """Coregister BOLD runs through a common template to the anatomical.

    Shares the input/output signature of :func:`init_bold_anat_coreg_wf`. All run
    references are combined into a ``coreg_space`` (``session`` or ``subject``)
    template, that template is registered to the anatomical, and per-run
    ``run->template->anat`` transforms are composed.
    """
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs

    from fmriprep.utils.bids import GROUP_DISMISS_ENTITIES
    from fmriprep.utils.misc import get_wf_name
    from fmriprep.workflows.bold.outputs import init_ds_registration_wf
    from fmriprep.workflows.bold.registration import init_bold_reg_wf
    from fmriprep.workflows.bold.template import init_bold_template_wf

    precomputed = precomputed or {}
    bids_root = str(config.execution.bids_dir)
    n_runs = len(bold_files)
    bold_ids = [get_wf_name(bold_file, None).removesuffix('_wf') for bold_file in bold_files]
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=OUTPUT_FIELDS), name='outputnode')

    _dismiss = list(GROUP_DISMISS_ENTITIES)
    if coreg_space == 'subject':
        _dismiss.append('session')

    run2template_xfms = precomputed.get('run2template_xfms') or [None] * n_runs
    template2anat_xfm = precomputed.get('template2anat_xfm')
    if isinstance(template2anat_xfm, (list, tuple)):
        template2anat_xfm = template2anat_xfm[0] if template2anat_xfm else None

    # Only skip if ALL run -> boldref transforms are present.
    skip_template = all(run2template_xfms)
    # If template needs to be recomputed, redo coregistration to anat
    skip_reg = bool(template2anat_xfm) and skip_template
    if template2anat_xfm and not skip_template:
        logger.warning(
            'A precomputed template2anat transform was found without a complete set '
            'of run2template transforms; ignoring it and recomputing registration '
            'against the rebuilt template.'
        )

    template_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'run2template_xfms']),
        name='template_buffer',
    )
    boldref_template = precomputed.get('boldref_template')
    if skip_template:
        from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

        template_buffer.inputs.run2template_xfms = list(run2template_xfms)

        if boldref_template:
            logger.info('Reusing precomputed boldref template; skipping reconstruction.')
            template_buffer.inputs.boldref = boldref_template
        else:
            logger.info(
                'Found precomputed run2template transforms; '
                'reconstructing boldref template from run references.'
            )
            select_boldref0 = pe.Node(
                niu.Select(index=0), name='select_boldref0', run_without_submitting=True
            )
            warp_template_boldref = pe.Node(
                ApplyTransforms(
                    transforms=[run2template_xfms[0]], interpolation='LanczosWindowedSinc'
                ),
                name='warp_template_boldref',
            )
            workflow.connect([
                (inputnode, select_boldref0, [('run_boldrefs', 'inlist')]),
                (select_boldref0, warp_template_boldref, [
                    ('out', 'input_image'),
                    ('out', 'reference_image'),
                ]),
                (warp_template_boldref, template_buffer, [('output_image', 'boldref')]),
            ])  # fmt:skip
    else:
        if any(run2template_xfms):
            logger.warning(
                'Only some run2template transforms were found - ignoring and recomputing the template.'
            )
        bold_template_wf = init_bold_template_wf(num_bold_runs=n_runs, omp_nthreads=omp_nthreads)

        template_sources = pe.Node(
            BIDSURI(
                numinputs=1, dataset_links=config.execution.dataset_links, out_dir=str(output_dir)
            ),
            name='template_sources',
            run_without_submitting=True,
        )

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
        workflow.connect([
            (inputnode, bold_template_wf, [('run_boldrefs', 'inputnode.boldref_files')]),
            (inputnode, template_sources, [('run_boldrefs', 'in1')]),
            (template_sources, ds_boldref_template, [('out', 'Sources')]),
            (bold_template_wf, ds_boldref_template, [
                ('outputnode.boldref', 'in_file'),
            ]),
            (bold_template_wf, template_buffer, [
                ('outputnode.run2template_xfms', 'run2template_xfms'),
            ]),
            (ds_boldref_template, template_buffer, [('out_file', 'boldref')]),
        ])  # fmt:skip

    reg_buffer = pe.Node(
        niu.IdentityInterface(fields=['template2anat', 'fallback']),
        name='reg_buffer',
    )
    if skip_reg:
        logger.info('Found precomputed template2anat transform; skipping coregistration.')
        reg_buffer.inputs.template2anat = template2anat_xfm
        reg_buffer.inputs.fallback = False
    else:
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
            (inputnode, boldref_reg_wf, ANAT_REG_INPUTS),
            (inputnode, ds_template2anat_wf, [('run_boldrefs', 'inputnode.source_files')]),
            (template_buffer, boldref_reg_wf, [('boldref', 'inputnode.ref_bold_brain')]),
            (boldref_reg_wf, ds_template2anat_wf, [
                ('outputnode.itk_bold_to_t1', 'xform'),
            ]),
            (boldref_reg_wf, reg_buffer, [('outputnode.fallback', 'fallback')]),
            (ds_template2anat_wf, reg_buffer, [('outputnode.xform', 'template2anat')]),
        ])  # fmt:skip

    merge_run2template = pe.Node(
        niu.Merge(n_runs), name='merge_run2template', run_without_submitting=True
    )
    merge_template2anat = pe.Node(
        niu.Merge(n_runs), name='merge_template2anat', run_without_submitting=True
    )
    merge_run2anat_xfms = pe.Node(
        niu.Merge(n_runs), name='merge_run2anat_xfms', run_without_submitting=True
    )

    for i, (bold_file, bold_id) in enumerate(zip(bold_files, bold_ids, strict=True)):
        select_run2template = pe.Node(
            niu.Select(index=i),
            name=f'select_run2template_{bold_id}',
            run_without_submitting=True,
        )
        workflow.connect(template_buffer, 'run2template_xfms', select_run2template, 'inlist')

        merge_run2anat = pe.Node(
            niu.Merge(2), name=f'merge_run2anat_{bold_id}', run_without_submitting=True
        )
        concat = pe.Node(ConcatenateXFMs(), name=f'concat_run2anat_{bold_id}')

        if skip_template:
            workflow.connect([
                (select_run2template, merge_run2template, [('out', f'in{i + 1}')]),
                (select_run2template, merge_run2anat, [('out', 'in1')]),
            ])  # fmt:skip
        else:
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
                (select_run2template, ds_run2template_wf, [('out', 'inputnode.xform')]),
                (ds_run2template_wf, merge_run2template, [('outputnode.xform', f'in{i + 1}')]),
                (ds_run2template_wf, merge_run2anat, [('outputnode.xform', 'in1')]),
            ])  # fmt:skip

        if skip_reg:
            setattr(merge_template2anat.inputs, f'in{i + 1}', template2anat_xfm)
            merge_run2anat.inputs.in2 = template2anat_xfm
        else:
            # Pass single template->anat transform to all runs
            workflow.connect([
                (reg_buffer, merge_template2anat, [('template2anat', f'in{i + 1}')]),
                (reg_buffer, merge_run2anat, [('template2anat', 'in2')]),
            ])  # fmt:skip

        workflow.connect([
            (merge_run2anat, concat, [('out', 'in_xfms')]),
            (concat, merge_run2anat_xfms, [('out_xfm', f'in{i + 1}')]),
        ])  # fmt:skip

    # Broadcast the session template image/mask/fallback into per-run lists
    expand_boldref = pe.Node(niu.Function(function=_expand), name='expand_boldref')
    expand_fallback = pe.Node(niu.Function(function=_expand), name='expand_fallback')
    for node in (expand_boldref, expand_fallback):
        node.inputs.n = n_runs
        node.run_without_submitting = True

    workflow.connect([
        (template_buffer, expand_boldref, [('boldref', 'value')]),
        (reg_buffer, expand_fallback, [('fallback', 'value')]),
        (expand_boldref, outputnode, [('out', 'template_boldrefs')]),
        (expand_fallback, outputnode, [('out', 'fallbacks')]),
        (merge_run2template, outputnode, [('out', 'run2template_xfms')]),
        (merge_template2anat, outputnode, [('out', 'template2anat_xfms')]),
        (merge_run2anat_xfms, outputnode, [('out', 'run2anat_xfms')]),
    ])  # fmt:skip

    return workflow
