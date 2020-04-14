# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Writing out derivative files."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.cifti import CiftiNameSource
from niworkflows.interfaces.surf import GiftiNameSource
from niworkflows.interfaces.utility import KeySelect
from niworkflows.utils.spaces import format_reference as _fmt_space

from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink


def init_func_derivatives_wf(
    bids_root,
    cifti_output,
    freesurfer,
    metadata,
    output_dir,
    spaces,
    use_aroma,
    name='func_derivatives_wf',
):
    """
    Set up a battery of datasinks to store derivatives in the right location.

    Parameters
    ----------
    bids_root : :obj:`str`
        Original BIDS dataset path.
    cifti_output : :obj:`bool`
        Whether the ``--cifti-output`` flag was set.
    freesurfer : :obj:`bool`
        Whether FreeSurfer anatomical processing was run.
    metadata : :obj:`dict`
        Metadata dictionary associated to the BOLD run.
    output_dir : :obj:`str`
        Where derivatives should be written out to.
    spaces : :py:class:`~niworkflows.utils.spaces.SpatialReferences`
        A container for storing, organizing, and parsing spatial normalizations. Composed of
        :py:class:`~niworkflows.utils.spaces.Reference` objects representing spatial references.
        Each ``Reference`` contains a space, which is a string of either TemplateFlow template IDs
        (e.g., ``MNI152Lin``, ``MNI152NLin6Asym``, ``MNIPediatricAsym``), nonstandard references
        (e.g., ``T1w`` or ``anat``, ``sbref``, ``run``, etc.), or a custom template located in
        the TemplateFlow root directory. Each ``Reference`` may also contain a spec, which is a
        dictionary with template specifications (e.g., a specification of ``{'resolution': 2}``
        would lead to resampling on a 2mm resolution of the space).
    use_aroma : :obj:`bool`
        Whether ``--use-aroma`` flag was set.
    name : :obj:`str`
        This workflow's identifier (default: ``func_derivatives_wf``).

    """
    from smriprep.workflows.outputs import _bids_relative
    nonstd_spaces = set(spaces.get_nonstandard())
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'aroma_noise_ics', 'bold_aparc_std', 'bold_aparc_t1', 'bold_aseg_std',
        'bold_aseg_t1', 'bold_cifti', 'bold_mask_std', 'bold_mask_t1', 'bold_std',
        'bold_std_ref', 'bold_t1', 'bold_t1_ref', 'bold_native', 'bold_native_ref',
        'bold_mask_native', 'cifti_variant', 'cifti_metadata', 'cifti_density',
        'confounds', 'confounds_metadata', 'melodic_mix', 'nonaggr_denoised_file',
        'source_file', 'surf_files', 'surf_refs', 'template', 'spatial_reference']),
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

    if nonstd_spaces.intersection(('func', 'run', 'bold', 'boldref', 'sbref')):
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
    if nonstd_spaces.intersection(('T1w', 'anat')):
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

    if getattr(spaces, '_cached') is None:
        return workflow

    # Store resamplings in standard spaces when listed in --output-spaces
    if spaces.cached.references:
        itersource = pe.Node(niu.IdentityInterface(fields=['space_definition']),
                             name='itersource')

        itersource.iterables = (
            'space_definition',
            [(s.fullname, s.spec) for s in spaces.cached.get_standard(dim=(3,))]
        )

        select_std = pe.Node(KeySelect(
            fields=['template', 'bold_std', 'bold_std_ref', 'bold_mask_std']),
            name='select_std', run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)

        ds_bold_std = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='preproc',
                                keep_dtype=True, compress=True, SkullStripped=False,
                                RepetitionTime=metadata.get('RepetitionTime'),
                                TaskName=metadata.get('TaskName')),
            name='ds_bold_std', run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_std_ref = pe.Node(
            DerivativesDataSink(base_directory=output_dir, suffix='boldref'),
            name='ds_bold_std_ref', run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_mask_std = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='brain',
                                suffix='mask'),
            name='ds_bold_mask_std', run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, ds_bold_std, [('source_file', 'source_file')]),
            (inputnode, ds_bold_std_ref, [('source_file', 'source_file')]),
            (inputnode, ds_bold_mask_std, [('source_file', 'source_file')]),
            (inputnode, select_std, [('bold_std', 'bold_std'),
                                     ('bold_std_ref', 'bold_std_ref'),
                                     ('bold_mask_std', 'bold_mask_std'),
                                     ('template', 'template'),
                                     ('spatial_reference', 'keys')]),
            (itersource, select_std, [(('space_definition', _fmt_space), 'key')]),
            (select_std, ds_bold_std, [('bold_std', 'in_file'),
                                       ('key', 'space')]),
            (select_std, ds_bold_std_ref, [('bold_std_ref', 'in_file'),
                                           ('key', 'space')]),
            (select_std, ds_bold_mask_std, [('bold_mask_std', 'in_file'),
                                            ('key', 'space')]),
            (raw_sources, ds_bold_mask_std, [('out', 'RawSources')]),
        ])

        if freesurfer:
            select_fs_std = pe.Node(KeySelect(
                fields=['bold_aseg_std', 'bold_aparc_std', 'template']),
                name='select_fs_std', run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
            ds_bold_aseg_std = pe.Node(DerivativesDataSink(
                base_directory=output_dir, desc='aseg', suffix='dseg'),
                name='ds_bold_aseg_std', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            ds_bold_aparc_std = pe.Node(DerivativesDataSink(
                base_directory=output_dir, desc='aparcaseg', suffix='dseg'),
                name='ds_bold_aparc_std', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([
                (itersource, select_fs_std, [
                    (('space_definition', _fmt_space), 'key')]),
                (inputnode, select_fs_std, [('bold_aseg_std', 'bold_aseg_std'),
                                            ('bold_aparc_std', 'bold_aparc_std'),
                                            ('template', 'template'),
                                            ('spatial_reference', 'keys')]),
                (select_fs_std, ds_bold_aseg_std, [('bold_aseg_std', 'in_file'),
                                                   ('key', 'space')]),
                (select_fs_std, ds_bold_aparc_std, [('bold_aparc_std', 'in_file'),
                                                    ('key', 'space')]),
                (inputnode, ds_bold_aseg_std, [('source_file', 'source_file')]),
                (inputnode, ds_bold_aparc_std, [('source_file', 'source_file')])
            ])

    fs_outputs = spaces.cached.get_fs_spaces()
    if freesurfer and fs_outputs:
        select_fs_surf = pe.Node(KeySelect(
            fields=['surfaces', 'surf_kwargs']), name='select_fs_surf',
            run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        select_fs_surf.iterables = [('key', fs_outputs)]
        select_fs_surf.inputs.surf_kwargs = [{'space': s} for s in fs_outputs]

        name_surfs = pe.MapNode(GiftiNameSource(
            pattern=r'(?P<LR>[lr])h.\w+',
            template='space-{space}_hemi-{LR}.func'),
            iterfield=['in_file'], name='name_surfs',
            mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True)

        ds_bold_surfs = pe.MapNode(DerivativesDataSink(base_directory=output_dir),
                                   iterfield=['in_file', 'suffix'], name='ds_bold_surfs',
                                   run_without_submitting=True,
                                   mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, select_fs_surf, [
                ('surf_files', 'surfaces'),
                ('surf_refs', 'keys')]),
            (select_fs_surf, name_surfs, [('surfaces', 'in_file'),
                                          ('surf_kwargs', 'template_kwargs')]),
            (inputnode, ds_bold_surfs, [('source_file', 'source_file')]),
            (select_fs_surf, ds_bold_surfs, [('surfaces', 'in_file')]),
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

    return workflow
