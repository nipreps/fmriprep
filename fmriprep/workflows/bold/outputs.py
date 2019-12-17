# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Writing out derivative files."""
from collections import OrderedDict

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.cifti import CiftiNameSource
from niworkflows.interfaces.surf import GiftiNameSource
from niworkflows.interfaces.utility import KeySelect

from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink


def init_func_derivatives_wf(
    bids_root,
    cifti_output,
    freesurfer,
    metadata,
    output_dir,
    output_spaces,
    standard_spaces,
    use_aroma,
    fslr_density=None,
    name='func_derivatives_wf',
):
    """
    Set up a battery of datasinks to store derivatives in the right location.

    Parameters
    ----------
    bids_root : str
        Original BIDS dataset path.
    cifti_output : bool
        Whether the ``--cifti-output`` flag was set.
    freesurfer : bool
        Whether FreeSurfer anatomical processing was run.
    metadata : dict
        Metadata dictionary associated to the BOLD run.
    output_dir : str
        Where derivatives should be written out to.
    output_spaces : OrderedDict
        List of selected ``--output-spaces``.
    use_aroma : bool
        Whether ``--use-aroma`` flag was set.
    fslr_density : str, optional
        Density of fsLR surface (32k or 59k)
    name : str
        This workflow's identifier (default: ``func_derivatives_wf``).

    """
    from smriprep.workflows.outputs import _bids_relative
    workflow = Workflow(name=name)

    # remove intermediate spaces
    output_spaces = _remove_internal_spaces(output_spaces)

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'aroma_noise_ics', 'bold_aparc_std', 'bold_aparc_t1', 'bold_aseg_std',
        'bold_aseg_t1', 'bold_cifti', 'bold_mask_std', 'bold_mask_t1', 'bold_std',
        'bold_std_ref', 'bold_t1', 'bold_t1_ref', 'bold_native', 'bold_native_ref',
        'bold_mask_native', 'cifti_variant', 'cifti_metadata', 'cifti_density',
        'confounds', 'confounds_metadata', 'melodic_mix', 'nonaggr_denoised_file',
        'source_file', 'surfaces', 'template']),
        name='inputnode')

    raw_sources = pe.Node(niu.Function(function=_bids_relative), name='raw_sources')
    raw_sources.inputs.bids_root = bids_root

    ds_confounds = pe.Node(DerivativesDataSink(
        base_directory=output_dir, desc='confounds', suffix='regressors'),
        name="ds_confounds", run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, raw_sources, [('source_file', 'in_files')]),
        (inputnode, ds_confounds, [('source_file', 'source_file'),
                                   ('confounds', 'in_file'),
                                   ('confounds_metadata', 'meta_dict')]),
    ])

    if set(('func', 'run', 'bold', 'boldref', 'sbref')).intersection(output_spaces):
        ds_bold_native = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='preproc',
                                keep_dtype=True, compress=True, SkullStripped=False,
                                RepetitionTime=metadata.get('RepetitionTime'),
                                TaskName=metadata.get('TaskName')),
            name='ds_bold_native', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_native_ref = pe.Node(
            DerivativesDataSink(base_directory=output_dir, suffix='boldref', compress=True),
            name='ds_bold_native_ref', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_mask_native = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='brain',
                                suffix='mask', compress=True),
            name='ds_bold_mask_native', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, ds_bold_native, [('source_file', 'source_file'),
                                         ('bold_native', 'in_file')]),
            (inputnode, ds_bold_native_ref, [('source_file', 'source_file'),
                                             ('bold_native_ref', 'in_file')]),
            (inputnode, ds_bold_mask_native, [('source_file', 'source_file'),
                                              ('bold_mask_native', 'in_file')]),
            (raw_sources, ds_bold_mask_native, [('out', 'RawSources')]),
        ])

    # Resample to T1w space
    if 'T1w' in output_spaces or 'anat' in output_spaces:
        ds_bold_t1 = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='T1w', desc='preproc',
                                keep_dtype=True, compress=True, SkullStripped=False,
                                RepetitionTime=metadata.get('RepetitionTime'),
                                TaskName=metadata.get('TaskName')),
            name='ds_bold_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_t1_ref = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='T1w',
                                suffix='boldref', compress=True),
            name='ds_bold_t1_ref', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        ds_bold_mask_t1 = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='T1w', desc='brain',
                                suffix='mask', compress=True),
            name='ds_bold_mask_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_bold_t1, [('source_file', 'source_file'),
                                     ('bold_t1', 'in_file')]),
            (inputnode, ds_bold_t1_ref, [('source_file', 'source_file'),
                                         ('bold_t1_ref', 'in_file')]),
            (inputnode, ds_bold_mask_t1, [('source_file', 'source_file'),
                                          ('bold_mask_t1', 'in_file')]),
            (raw_sources, ds_bold_mask_t1, [('out', 'RawSources')]),
        ])
        if freesurfer:
            ds_bold_aseg_t1 = pe.Node(DerivativesDataSink(
                base_directory=output_dir, space='T1w', desc='aseg', suffix='dseg'),
                name='ds_bold_aseg_t1', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            ds_bold_aparc_t1 = pe.Node(DerivativesDataSink(
                base_directory=output_dir, space='T1w', desc='aparcaseg', suffix='dseg'),
                name='ds_bold_aparc_t1', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([
                (inputnode, ds_bold_aseg_t1, [('source_file', 'source_file'),
                                              ('bold_aseg_t1', 'in_file')]),
                (inputnode, ds_bold_aparc_t1, [('source_file', 'source_file'),
                                               ('bold_aparc_t1', 'in_file')]),
            ])

    # Resample to template (default: MNI)
    volume_std_spaces = [space for space in standard_spaces if not space.startswith('fs')]
    if volume_std_spaces:

        select_std = pe.MapNode(KeySelect(fields=['bold_std', 'bold_std_ref', 'bold_mask_std']),
                                iterfield=['key'], name='select_std', run_without_submitting=True,
                                mem_gb=DEFAULT_MEMORY_MIN_GB)
        select_std.inputs.key = standard_spaces

        ds_bold_std = pe.MapNode(
            DerivativesDataSink(base_directory=output_dir, desc='preproc',
                                keep_dtype=True, compress=True, SkullStripped=False,
                                RepetitionTime=metadata.get('RepetitionTime'),
                                TaskName=metadata.get('TaskName')),
            name='ds_bold_std', run_without_submitting=True,
            iterfield=['space', 'in_file'], mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_std_ref = pe.MapNode(
            DerivativesDataSink(base_directory=output_dir, suffix='boldref'),
            name='ds_bold_std_ref', run_without_submitting=True,
            iterfield=['space', 'in_file'], mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_mask_std = pe.MapNode(
            DerivativesDataSink(base_directory=output_dir, desc='brain',
                                suffix='mask'),
            name='ds_bold_mask_std', run_without_submitting=True,
            iterfield=['space', 'in_file'], mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, select_std, [('bold_std', 'bold_std'),
                                     ('bold_std_ref', 'bold_std_ref'),
                                     ('bold_mask_std', 'bold_mask_std'),
                                     ('template', 'keys')]),
            (select_std, ds_bold_std, [('bold_std', 'in_file'),
                                       ('key', 'space')]),
            (select_std, ds_bold_std_ref, [('bold_std_ref', 'in_file'),
                                           ('key', 'space')]),
            (select_std, ds_bold_mask_std, [('bold_mask_std', 'in_file'),
                                            ('key', 'space')]),
            (inputnode, ds_bold_std, [('source_file', 'source_file')]),
            (inputnode, ds_bold_std_ref, [('source_file', 'source_file')]),
            (inputnode, ds_bold_mask_std, [('source_file', 'source_file')]),
            (raw_sources, ds_bold_mask_std, [('out', 'RawSources')]),
        ])

        if freesurfer:
            select_fs_std = pe.MapNode(KeySelect(fields=['bold_aseg_std', 'bold_aparc_std']),
                                       iterfield=['key'], name='select_fs_std',
                                       run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
            select_fs_std.inputs.key = standard_spaces

            ds_bold_aseg_std = pe.MapNode(DerivativesDataSink(
                base_directory=output_dir, desc='aseg', suffix='dseg'),
                name='ds_bold_aseg_std', run_without_submitting=True,
                iterfield=['space', 'in_file'], mem_gb=DEFAULT_MEMORY_MIN_GB)
            ds_bold_aparc_std = pe.MapNode(DerivativesDataSink(
                base_directory=output_dir, desc='aparcaseg', suffix='dseg'),
                name='ds_bold_aparc_std', run_without_submitting=True,
                iterfield=['space', 'in_file'], mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([
                (inputnode, select_fs_std, [('bold_aseg_std', 'bold_aseg_std'),
                                            ('bold_aparc_std', 'bold_aparc_std'),
                                            ('template', 'keys')]),
                (select_fs_std, ds_bold_aseg_std, [('bold_aseg_std', 'in_file'),
                                                   ('key', 'space')]),
                (select_fs_std, ds_bold_aparc_std, [('bold_aparc_std', 'in_file'),
                                                    ('key', 'space')]),
                (inputnode, ds_bold_aseg_std, [('source_file', 'source_file')]),
                (inputnode, ds_bold_aparc_std, [('source_file', 'source_file')])
            ])

    # fsaverage space
    surface_spaces = [space for space in output_spaces.keys() if space.startswith('fs')]
    if freesurfer and surface_spaces:

        filter_surfaces = pe.Node(niu.Function(function=_filter_surfaces),
                                  name='filter_surfaces', mem_gb=DEFAULT_MEMORY_MIN_GB,
                                  run_with_submitting=True)
        filter_surfaces.inputs.surface_spaces = surface_spaces

        extract_surf_info = pe.MapNode(niu.Function(function=_extract_surf_info),
                                       iterfield=['in_file'], name='extract_surf_info',
                                       mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True)
        extract_surf_info.inputs.density = fslr_density

        name_surfs = pe.MapNode(GiftiNameSource(
            pattern=r'(?P<LR>[lr])h.(?P<space>\w+).gii',
            template='space-{space}{den}_hemi-{LR}.func'),
            iterfield=['in_file', 'template_kwargs'], name='name_surfs',
            mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True)

        ds_bold_surfs = pe.MapNode(DerivativesDataSink(base_directory=output_dir),
                                   iterfield=['in_file', 'suffix'], name='ds_bold_surfs',
                                   run_without_submitting=True,
                                   mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, filter_surfaces, [('surfaces', 'surface_files')]),
            (filter_surfaces, extract_surf_info, [('out', 'in_file')]),
            (filter_surfaces, name_surfs, [('out', 'in_file')]),
            (extract_surf_info, name_surfs, [('out', 'template_kwargs')]),
            (inputnode, ds_bold_surfs, [('source_file', 'source_file')]),
            (filter_surfaces, ds_bold_surfs, [('out', 'in_file')]),
            (name_surfs, ds_bold_surfs, [('out_name', 'suffix')]),
        ])

    # CIFTI output
    if cifti_output:
        name_cifti = pe.MapNode(
            CiftiNameSource(), iterfield=['variant', 'density'], name='name_cifti',
            mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True)
        cifti_bolds = pe.MapNode(
            DerivativesDataSink(base_directory=output_dir, compress=False),
            iterfield=['in_file', 'suffix'], name='cifti_bolds',
            run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        cifti_key = pe.MapNode(DerivativesDataSink(
            base_directory=output_dir), iterfield=['in_file', 'suffix'],
            name='cifti_key', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, name_cifti, [('cifti_variant', 'variant'),
                                     ('cifti_density', 'density')]),
            (inputnode, cifti_bolds, [('bold_cifti', 'in_file'),
                                      ('source_file', 'source_file')]),
            (name_cifti, cifti_bolds, [('out_name', 'suffix')]),
            (name_cifti, cifti_key, [('out_name', 'suffix')]),
            (inputnode, cifti_key, [('source_file', 'source_file'),
                                    ('cifti_metadata', 'in_file')]),
        ])

    if use_aroma:
        ds_aroma_noise_ics = pe.Node(DerivativesDataSink(
            base_directory=output_dir, suffix='AROMAnoiseICs'),
            name="ds_aroma_noise_ics", run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_melodic_mix = pe.Node(DerivativesDataSink(
            base_directory=output_dir, desc='MELODIC', suffix='mixing'),
            name="ds_melodic_mix", run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_aroma_std = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='MNI152NLin6Asym',
                                desc='smoothAROMAnonaggr', keep_dtype=True),
            name='ds_aroma_std', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, ds_aroma_noise_ics, [('source_file', 'source_file'),
                                             ('aroma_noise_ics', 'in_file')]),
            (inputnode, ds_melodic_mix, [('source_file', 'source_file'),
                                         ('melodic_mix', 'in_file')]),
            (inputnode, ds_aroma_std, [('source_file', 'source_file'),
                                       ('nonaggr_denoised_file', 'in_file')]),
        ])

    return workflow


def _extract_surf_info(in_file, density):
    import os
    info = {'den': ''}
    if 'fsLR' in os.path.basename(in_file):
        info['den'] = '_den-{}'.format(density)
    return info


def _filter_surfaces(surface_files, surface_spaces):
    import os
    output_surfaces = []
    for fl in surface_files:
        if any(
            "{}.".format(space) in os.path.basename(fl) for space in surface_spaces
        ):
            output_surfaces.append(fl)
    return output_surfaces


def _remove_internal_spaces(spaces):
    return OrderedDict([
        (key, modifiers) for key, modifiers in spaces.items()
        if not modifiers.get('no-output')]
    )
