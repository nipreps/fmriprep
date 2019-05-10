
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.cifti import CiftiNameSource
from niworkflows.interfaces.surf import GiftiNameSource

from ...interfaces import DerivativesDataSink


DEFAULT_MEMORY_MIN_GB = 0.01


def init_func_derivatives_wf(
    bids_root,
    cifti_output,
    freesurfer,
    metadata,
    output_dir,
    output_spaces,
    standard_spaces,
    use_aroma,
    name='func_derivatives_wf',
):
    """
    Set up a battery of datasinks to store derivatives in the right location

    **Parameters**

    bids_root : str
    cifti_output : bool
    freesurfer : bool
    metadata : dict
    output_dir : str
    output_spaces : OrderedDict
    use_aroma : bool
    name : str

    """
    from smriprep.workflows.outputs import _bids_relative
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'aroma_noise_ics', 'bold_aparc_std', 'bold_aparc_t1', 'bold_aseg_std',
        'bold_aseg_t1', 'bold_cifti', 'bold_mask_std', 'bold_mask_t1', 'bold_std',
        'bold_std_ref', 'bold_t1', 'bold_t1_ref', 'cifti_variant', 'cifti_variant_key',
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

    # Resample to T1w space
    if 'T1w' in output_spaces or 'anat' in output_spaces:
        ds_bold_t1 = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='T1w', desc='preproc',
                                keep_dtype=True, compress=True, SkullStripped=False,
                                RepetitionTime=metadata.get('RepetitionTime')),
            name='ds_bold_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_t1_ref = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='T1w', suffix='boldref'),
            name='ds_bold_t1_ref', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        ds_bold_mask_t1 = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='T1w', desc='brain',
                                suffix='mask'),
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
    if standard_spaces:
        ds_bold_std = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='preproc',
                                keep_dtype=True, compress=True, SkullStripped=False,
                                RepetitionTime=metadata.get('RepetitionTime')),
            name='ds_bold_std', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_std_ref = pe.Node(
            DerivativesDataSink(base_directory=output_dir, suffix='boldref'),
            name='ds_bold_std_ref', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        ds_bold_mask_std = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='brain',
                                suffix='mask'),
            name='ds_bold_mask_std', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_bold_std, [('source_file', 'source_file'),
                                      ('bold_std', 'in_file'),
                                      ('template', 'space')]),
            (inputnode, ds_bold_std_ref, [('source_file', 'source_file'),
                                          ('bold_std_ref', 'in_file'),
                                          ('template', 'space')]),
            (inputnode, ds_bold_mask_std, [('source_file', 'source_file'),
                                           ('bold_mask_std', 'in_file'),
                                           ('template', 'space')]),
            (raw_sources, ds_bold_mask_std, [('out', 'RawSources')]),
        ])

        if freesurfer:
            ds_bold_aseg_std = pe.Node(DerivativesDataSink(
                base_directory=output_dir, desc='aseg', suffix='dseg'),
                name='ds_bold_aseg_std', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            ds_bold_aparc_std = pe.Node(DerivativesDataSink(
                base_directory=output_dir, desc='aparcaseg', suffix='dseg'),
                name='ds_bold_aparc_std', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([
                (inputnode, ds_bold_aseg_std, [('source_file', 'source_file'),
                                               ('bold_aseg_std', 'in_file'),
                                               ('template', 'space')]),
                (inputnode, ds_bold_aparc_std, [('source_file', 'source_file'),
                                                ('bold_aparc_std', 'in_file'),
                                                ('template', 'space')]),
            ])

    # fsaverage space
    if freesurfer and any(space.startswith('fs') for space in output_spaces.keys()):
        name_surfs = pe.MapNode(GiftiNameSource(
            pattern=r'(?P<LR>[lr])h.(?P<space>\w+).gii', template='space-{space}_hemi-{LR}.func'),
            iterfield='in_file', name='name_surfs', mem_gb=DEFAULT_MEMORY_MIN_GB,
            run_without_submitting=True)
        ds_bold_surfs = pe.MapNode(DerivativesDataSink(base_directory=output_dir),
                                   iterfield=['in_file', 'suffix'], name='ds_bold_surfs',
                                   run_without_submitting=True,
                                   mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, name_surfs, [('surfaces', 'in_file')]),
            (inputnode, ds_bold_surfs, [('source_file', 'source_file'),
                                        ('surfaces', 'in_file')]),
            (name_surfs, ds_bold_surfs, [('out_name', 'suffix')]),
        ])

        # CIFTI output
        if cifti_output and 'MNI152NLin2009cAsym' in output_spaces:
            name_cifti = pe.MapNode(
                CiftiNameSource(), iterfield=['variant'], name='name_cifti',
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
                (inputnode, name_cifti, [('cifti_variant', 'variant')]),
                (inputnode, cifti_bolds, [('bold_cifti', 'in_file'),
                                          ('source_file', 'source_file')]),
                (name_cifti, cifti_bolds, [('out_name', 'suffix')]),
                (name_cifti, cifti_key, [('out_name', 'suffix')]),
                (inputnode, cifti_key, [('source_file', 'source_file'),
                                        ('cifti_variant_key', 'in_file')]),
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
            DerivativesDataSink(base_directory=output_dir,
                                desc='smoothAROMAnonaggr', keep_dtype=True),
            name='ds_aroma_std', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, ds_aroma_noise_ics, [('source_file', 'source_file'),
                                             ('aroma_noise_ics', 'in_file')]),
            (inputnode, ds_melodic_mix, [('source_file', 'source_file'),
                                         ('melodic_mix', 'in_file')]),
            (inputnode, ds_aroma_std, [('source_file', 'source_file'),
                                       ('nonaggr_denoised_file', 'in_file'),
                                       ('template', 'space')]),
        ])

    return workflow
