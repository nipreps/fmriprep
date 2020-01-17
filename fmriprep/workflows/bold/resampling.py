# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Resampling workflows
++++++++++++++++++++

.. autofunction:: init_bold_surf_wf
.. autofunction:: init_bold_std_trans_wf
.. autofunction:: init_bold_preproc_trans_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, freesurfer as fs
from nipype.interfaces.fsl import Split as FSLSplit
import nipype.interfaces.workbench as wb

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from niworkflows.interfaces.freesurfer import (
    MedialNaNs,
    # See https://github.com/poldracklab/fmriprep/issues/768
    PatchedConcatenateLTA as ConcatenateLTA,
    PatchedLTAConvert as LTAConvert
)
from niworkflows.interfaces.itk import MultiApplyTransforms
from niworkflows.interfaces.utils import GenerateSamplingReference
from niworkflows.interfaces.utility import KeySelect
from niworkflows.interfaces.surf import GiftiSetAnatomicalStructure
from niworkflows.interfaces.nilearn import Merge

from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink

from .util import init_bold_reference_wf


def init_bold_surf_wf(
    mem_gb,
    surface_spaces,
    medial_surface_nan,
    name='bold_surf_wf'
):
    """
    Sample functional images to FreeSurfer surfaces.

    For each vertex, the cortical ribbon is sampled at six points (spaced 20% of thickness apart)
    and averaged.
    Outputs are in GIFTI format.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_surf_wf
            wf = init_bold_surf_wf(mem_gb=0.1,
                                   surface_spaces=['T1w', 'fsnative',
                                                  'template', 'fsaverage5'],
                                   medial_surface_nan=False)

    Parameters
    ----------
    surface_spaces : list
        List of (`name`, `space specs`) tuples of spaces functional images
        are to be resampled to.
        Target spaces with name beginning with ``fs`` will be selected for resampling,
        such as ``fsaverage`` or related template spaces
        For ``fsnative`` space, images will be resampled to the individual
        subject's native surface
        For ``fsLR`` space, images will be resampled twice;
        first to ``fsaverage`` and then to ``fsLR``.
    medial_surface_nan : bool
        Replace medial wall values with NaNs on functional GIFTI files


    Inputs
    ------
    source_file
        Motion-corrected BOLD series in T1 space
    t1w_preproc
        Bias-corrected structural template image
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    t1w2fsnative_xfm
        LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space

    Outputs
    -------
    surfaces
        BOLD series, resampled to FreeSurfer surfaces

    """
    # Ensure volumetric spaces do not sneak into this workflow
    spaces = [space for space, specs in surface_spaces if space.startswith('fs')]
    workflow = Workflow(name=name)

    if spaces:
        workflow.__desc__ = """\
The BOLD time-series, were resampled to surfaces on the following
spaces: {out_spaces}.
""".format(out_spaces=', '.join(['*%s*' % s for s in spaces]))
    spaces, fslr_density = _remove_fslr(surface_spaces)
    to_fslr = 'fsaverage' in spaces and fslr_density

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 't1w_preproc', 'subject_id', 'subjects_dir',
                                      't1w2fsnative_xfm']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['surfaces', 'fslr_density']),
                         name='outputnode')
    outputnode.inputs.fslr_density = fslr_density

    def select_target(subject_id, space):
        """Get the target subject ID, given a source subject ID and a target space."""
        return subject_id if space == 'fsnative' else space

    targets = pe.MapNode(niu.Function(function=select_target),
                         iterfield=['space'], name='targets',
                         mem_gb=DEFAULT_MEMORY_MIN_GB)
    targets.inputs.space = spaces

    # Rename the source file to the output space to simplify naming later
    rename_src = pe.MapNode(niu.Rename(format_string='%(subject)s', keep_ext=True),
                            iterfield='subject', name='rename_src', run_without_submitting=True,
                            mem_gb=DEFAULT_MEMORY_MIN_GB)
    rename_src.inputs.subject = spaces

    resampling_xfm = pe.Node(LTAConvert(in_lta='identity.nofile', out_lta=True),
                             name='resampling_xfm')
    set_xfm_source = pe.Node(ConcatenateLTA(out_type='RAS2RAS'), name='set_xfm_source')

    sampler = pe.MapNode(
        fs.SampleToSurface(sampling_method='average', sampling_range=(0, 1, 0.2),
                           sampling_units='frac', interp_method='trilinear', cortex_mask=True,
                           override_reg_subj=True, out_type='gii'),
        iterfield=['source_file', 'target_subject'],
        iterables=('hemi', ['lh', 'rh']),
        name='sampler', mem_gb=mem_gb * 3)

    if to_fslr:
        filter_fsavg = pe.Node(niu.Function(function=_select_fsaverage_hemi,
                                            output_names=['fsaverage_bold', 'hemi']),
                               name='filter_fsavg', mem_gb=DEFAULT_MEMORY_MIN_GB,
                               run_without_submitting=True)

        rename_fslr = pe.MapNode(niu.Rename(format_string="%(hemi)s.fsLR.%(density)s",
                                            keep_ext=True,
                                            parse_string=r'^(?P<hemi>\w+)'),
                                 name='rename_fslr', mem_gb=DEFAULT_MEMORY_MIN_GB,
                                 iterfield=['density'], run_without_submitting=True)
        rename_fslr.inputs.density = list(fslr_density)

        fetch_fsaverage_tpls = pe.Node(niu.Function(function=_fetch_fsaverage_templates,
                                                    output_names=['fsaverage_sphere',
                                                                  'fsaverage_midthick']),
                                       name='fetch_fsaverage_tpls', mem_gb=DEFAULT_MEMORY_MIN_GB,
                                       overwrite=True)

        fetch_fslr_tpls = pe.MapNode(niu.Function(function=_fetch_fslr_templates,
                                                  output_names=['fslr_sphere',
                                                                'fslr_midthick']),
                                     name='fetch_fslr_tpls', mem_gb=DEFAULT_MEMORY_MIN_GB,
                                     iterfield=['density'], overwrite=True)
        fetch_fslr_tpls.inputs.density = list(fslr_density)

        resample_fslr = pe.MapNode(wb.MetricResample(method='ADAP_BARY_AREA', area_metrics=True),
                                   iterfield=['in_file', 'out_file', 'new_sphere', 'new_area'],
                                   name='resample_fslr')

        merge_fslr = pe.Node(niu.Merge(2), name='merge_fslr', mem_gb=DEFAULT_MEMORY_MIN_GB,
                             run_without_submitting=True)

        def _basename(in_files):
            import os
            return list(map(os.path.basename, in_files))

        workflow.connect([
            (sampler, filter_fsavg, [('out_file', 'in_files')]),
            (filter_fsavg, fetch_fslr_tpls, [('hemi', 'hemi')]),
            (filter_fsavg, fetch_fsaverage_tpls, [('hemi', 'hemi')]),
            (filter_fsavg, rename_fslr, [('fsaverage_bold', 'in_file')]),
            (rename_fslr, resample_fslr, [('out_file', 'in_file')]),
            (rename_fslr, resample_fslr, [(('out_file', _basename), 'out_file')]),
            (fetch_fsaverage_tpls, resample_fslr, [('fsaverage_sphere', 'current_sphere'),
                                                   ('fsaverage_midthick', 'current_area')]),
            (fetch_fslr_tpls, resample_fslr, [('fslr_sphere', 'new_sphere'),
                                              ('fslr_midthick', 'new_area')]),
            (sampler, merge_fslr, [('out_file', 'in1')]),
            (resample_fslr, merge_fslr, [('out_file', 'in2')]),
        ])

    merger = pe.JoinNode(niu.Merge(1, ravel_inputs=True), name='merger',
                         joinsource='sampler', joinfield=['in1'], run_without_submitting=True,
                         mem_gb=DEFAULT_MEMORY_MIN_GB)

    if medial_surface_nan:
        medial_nans = pe.MapNode(MedialNaNs(), iterfield=['in_file'],
                                 name='medial_nans', mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, medial_nans, [('subjects_dir', 'subjects_dir')]),
            (medial_nans, merger, [('out_file', 'in1')]),
        ])

    update_metadata = pe.MapNode(GiftiSetAnatomicalStructure(), iterfield='in_file',
                                 name='update_metadata', mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, targets, [('subject_id', 'subject_id')]),
        (inputnode, rename_src, [('source_file', 'in_file')]),
        (inputnode, resampling_xfm, [('source_file', 'source_file'),
                                     ('t1w_preproc', 'target_file')]),
        (inputnode, set_xfm_source, [('t1w2fsnative_xfm', 'in_lta2')]),
        (resampling_xfm, set_xfm_source, [('out_lta', 'in_lta1')]),
        (inputnode, sampler, [('subjects_dir', 'subjects_dir'),
                              ('subject_id', 'subject_id')]),
        (set_xfm_source, sampler, [('out_file', 'reg_file')]),
        (targets, sampler, [('out', 'target_subject')]),
        (rename_src, sampler, [('out_file', 'source_file')]),
        (merger, update_metadata, [('out', 'in_file')]),
        (update_metadata, outputnode, [('out_file', 'surfaces')]),
    ])

    if to_fslr and medial_surface_nan:
        medial_nans.inputs.density = fslr_density
        workflow.connect(merge_fslr, 'out', medial_nans, 'in_file')
    elif to_fslr:
        workflow.connect(merge_fslr, 'out', merger, 'in1')
    elif medial_surface_nan:
        workflow.connect(sampler, 'out_file', medial_nans, 'in_file')
    else:
        workflow.connect(sampler, 'out_file', merger, 'in1')

    return workflow


def init_bold_std_trans_wf(
    freesurfer,
    mem_gb,
    omp_nthreads,
    standard_spaces,
    name='bold_std_trans_wf',
    use_compression=True,
    use_fieldwarp=False
):
    """
    Sample fMRI into standard space with a single-step resampling of the original BOLD series.

    .. important::
        This workflow provides two outputnodes.
        One output node (with name ``poutputnode``) will be parameterized in a Nipype sense
        (see `Nipype iterables
        <https://miykael.github.io/nipype_tutorial/notebooks/basic_iteration.html>`__), and a
        second node (``outputnode``) will collapse the parameterized outputs into synchronous
        lists of the output fields listed below.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from collections import OrderedDict
            from fmriprep.workflows.bold import init_bold_std_trans_wf
            wf = init_bold_std_trans_wf(
                freesurfer=True,
                mem_gb=3,
                omp_nthreads=1,
                standard_spaces=OrderedDict([('MNI152Lin', {}),
                                             ('fsaverage', {'density': '10k'})]),
            )

    Parameters
    ----------
    freesurfer : bool
        Whether to generate FreeSurfer's aseg/aparc segmentations on BOLD space.
    mem_gb : float
        Size of BOLD file in GB
    omp_nthreads : int
        Maximum number of threads an individual process may use
    standard_spaces : OrderedDict
        Ordered dictionary where keys are TemplateFlow ID strings (e.g.,
        ``MNI152Lin``, ``MNI152NLin6Asym``, ``MNI152NLin2009cAsym``, or ``fsLR``),
        or paths pointing to custom templates organized in a TemplateFlow-like structure.
        Values of the dictionary aggregate modifiers (e.g., the value for the key ``MNI152Lin``
        could be ``{'resolution': 2}`` if one wants the resampling to be done on the 2mm
        resolution version of the selected template).
    name : str
        Name of workflow (default: ``bold_std_trans_wf``)
    use_compression : bool
        Save registered BOLD series as ``.nii.gz``
    use_fieldwarp : bool
        Include SDC warp in single-shot transform from BOLD to MNI

    Inputs
    ------
    anat2std_xfm
        List of anatomical-to-standard space transforms generated during
        spatial normalization.
    bold_aparc
        FreeSurfer's ``aparc+aseg.mgz`` atlas projected into the T1w reference
        (only if ``recon-all`` was run).
    bold_aseg
        FreeSurfer's ``aseg.mgz`` atlas projected into the T1w reference
        (only if ``recon-all`` was run).
    bold_mask
        Skull-stripping mask of reference image
    bold_split
        Individual 3D volumes, not motion corrected
    fieldwarp
        a :abbr:`DFM (displacements field map)` in ITK format
    hmc_xforms
        List of affine transforms aligning each volume to ``ref_image`` in ITK format
    itk_bold_to_t1
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing
    templates
        List of templates that were applied as targets during
        spatial normalization.

    Outputs
    -------
    bold_std
        BOLD series, resampled to template space
    bold_std_ref
        Reference, contrast-enhanced summary of the BOLD series, resampled to template space
    bold_mask_std
        BOLD series mask in template space
    bold_aseg_std
        FreeSurfer's ``aseg.mgz`` atlas, in template space at the BOLD resolution
        (only if ``recon-all`` was run)
    bold_aparc_std
        FreeSurfer's ``aparc+aseg.mgz`` atlas, in template space at the BOLD resolution
        (only if ``recon-all`` was run)
    templates
        Template identifiers synchronized correspondingly to previously
        described outputs.

    """
    # Filter ``standard_spaces``
    vol_std_spaces = [space[0] for space in standard_spaces]
    workflow = Workflow(name=name)

    if len(vol_std_spaces) == 1:
        workflow.__desc__ = """\
The BOLD time-series were resampled into standard space,
generating a *preprocessed BOLD run in {tpl} space*.
""".format(tpl=vol_std_spaces)
    else:
        workflow.__desc__ = """\
The BOLD time-series were resampled into several standard spaces,
correspondingly generating the following *spatially-normalized,
preprocessed BOLD runs*: {tpl}.
""".format(tpl=', '.join(vol_std_spaces))

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'anat2std_xfm',
            'bold_aparc',
            'bold_aseg',
            'bold_mask',
            'bold_split',
            'fieldwarp',
            'hmc_xforms',
            'itk_bold_to_t1',
            'name_source',
            'templates',
        ]),
        name='inputnode'
    )

    select_std = pe.Node(KeySelect(
        fields=['anat2std_xfm', 'resolution'], no_hash=True),
        name='select_std', run_without_submitting=True)
    select_std.inputs.resolution = [
        specs.get('resolution') or specs.get('res') or 'native' for tpl, specs in standard_spaces
    ]
    select_std.iterables = ('key', standard_spaces)

    select_tpl = pe.Node(niu.Function(function=_select_template),
                         name='select_tpl', run_without_submitting=True)

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref',
                      mem_gb=0.3)  # 256x256x256 * 64 / 8 ~ 150MB)

    mask_std_tfm = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', float=True),
        name='mask_std_tfm',
        mem_gb=1
    )

    # Write corrected file in the designated output dir
    mask_merge_tfms = pe.Node(niu.Merge(2), name='mask_merge_tfms', run_without_submitting=True,
                              mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, select_std, [('templates', 'keys'),
                                 ('anat2std_xfm', 'anat2std_xfm')]),
        (inputnode, mask_std_tfm, [('bold_mask', 'input_image')]),
        (inputnode, gen_ref, [(('bold_split', _first), 'moving_image')]),
        (inputnode, mask_merge_tfms, [(('itk_bold_to_t1', _aslist), 'in2')]),
        (select_std, select_tpl, [('key', 'template')]),
        (select_std, mask_merge_tfms, [('anat2std_xfm', 'in1')]),
        (select_std, gen_ref, [(('resolution', _is_native), 'keep_native')]),
        (select_tpl, gen_ref, [('out', 'fixed_image')]),
        (mask_merge_tfms, mask_std_tfm, [('out', 'transforms')]),
        (gen_ref, mask_std_tfm, [('out_file', 'reference_image')]),
    ])

    nxforms = 4 if use_fieldwarp else 3
    merge_xforms = pe.Node(niu.Merge(nxforms), name='merge_xforms',
                           run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([(inputnode, merge_xforms, [('hmc_xforms', 'in%d' % nxforms)])])

    if use_fieldwarp:
        workflow.connect([(inputnode, merge_xforms, [('fieldwarp', 'in3')])])

    bold_to_std_transform = pe.Node(
        MultiApplyTransforms(interpolation="LanczosWindowedSinc", float=True, copy_dtype=True),
        name='bold_to_std_transform', mem_gb=mem_gb * 3 * omp_nthreads, n_procs=omp_nthreads)

    merge = pe.Node(Merge(compress=use_compression), name='merge',
                    mem_gb=mem_gb * 3)

    # Generate a reference on the target T1w space
    gen_final_ref = init_bold_reference_wf(
        omp_nthreads=omp_nthreads, pre_mask=True)

    workflow.connect([
        (inputnode, merge_xforms, [
            (('itk_bold_to_t1', _aslist), 'in2')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode, bold_to_std_transform, [('bold_split', 'input_image')]),
        (select_std, merge_xforms, [('anat2std_xfm', 'in1')]),
        (merge_xforms, bold_to_std_transform, [('out', 'transforms')]),
        (gen_ref, bold_to_std_transform, [('out_file', 'reference_image')]),
        (bold_to_std_transform, merge, [('out_files', 'in_files')]),
        (merge, gen_final_ref, [('out_file', 'inputnode.bold_file')]),
        (mask_std_tfm, gen_final_ref, [('output_image', 'inputnode.bold_mask')]),
    ])

    # Connect output nodes
    output_names = ['bold_std', 'bold_std_ref', 'bold_mask_std', 'templates']
    if freesurfer:
        output_names += ['bold_aseg_std', 'bold_aparc_std']

    # poutputnode - parametric output node
    poutputnode = pe.Node(niu.IdentityInterface(fields=output_names),
                          name='poutputnode')

    workflow.connect([
        (gen_final_ref, poutputnode, [('outputnode.ref_image', 'bold_std_ref')]),
        (merge, poutputnode, [('out_file', 'bold_std')]),
        (mask_std_tfm, poutputnode, [('output_image', 'bold_mask_std')]),
        (select_std, poutputnode, [('key', 'templates')]),
    ])

    if freesurfer:
        # Sample the parcellation files to functional space
        aseg_std_tfm = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', float=True),
            name='aseg_std_tfm', mem_gb=1)
        aparc_std_tfm = pe.Node(
            ApplyTransforms(interpolation='MultiLabel', float=True),
            name='aparc_std_tfm', mem_gb=1)

        workflow.connect([
            (inputnode, aseg_std_tfm, [('bold_aseg', 'input_image')]),
            (inputnode, aparc_std_tfm, [('bold_aparc', 'input_image')]),
            (select_std, aseg_std_tfm, [('anat2std_xfm', 'transforms')]),
            (select_std, aparc_std_tfm, [('anat2std_xfm', 'transforms')]),
            (gen_ref, aseg_std_tfm, [('out_file', 'reference_image')]),
            (gen_ref, aparc_std_tfm, [('out_file', 'reference_image')]),
            (aseg_std_tfm, poutputnode, [('output_image', 'bold_aseg_std')]),
            (aparc_std_tfm, poutputnode, [('output_image', 'bold_aparc_std')]),
        ])

    # Connect outputnode to the parameterized outputnode
    outputnode = pe.JoinNode(niu.IdentityInterface(fields=output_names),
                             name='outputnode', joinsource='select_std')
    workflow.connect([
        (poutputnode, outputnode, [(f, f) for f in output_names])
    ])

    return workflow


def init_bold_preproc_trans_wf(mem_gb, omp_nthreads,
                               name='bold_preproc_trans_wf',
                               use_compression=True,
                               use_fieldwarp=False,
                               split_file=False,
                               interpolation='LanczosWindowedSinc'):
    """
    Resample in native (original) space.

    This workflow resamples the input fMRI in its native (original)
    space in a "single shot" from the original BOLD series.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_preproc_trans_wf
            wf = init_bold_preproc_trans_wf(mem_gb=3, omp_nthreads=1)

    Parameters
    ----------
    mem_gb : float
        Size of BOLD file in GB
    omp_nthreads : int
        Maximum number of threads an individual process may use
    name : str
        Name of workflow (default: ``bold_std_trans_wf``)
    use_compression : bool
        Save registered BOLD series as ``.nii.gz``
    use_fieldwarp : bool
        Include SDC warp in single-shot transform from BOLD to MNI
    split_file : bool
        Whether the input file should be splitted (it is a 4D file)
        or it is a list of 3D files (default ``False``, do not split)
    interpolation : str
        Interpolation type to be used by ANTs' ``applyTransforms``
        (default ``'LanczosWindowedSinc'``)

    Inputs
    ------
    bold_file
        Individual 3D volumes, not motion corrected
    bold_mask
        Skull-stripping mask of reference image
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing
    hmc_xforms
        List of affine transforms aligning each volume to ``ref_image`` in ITK format
    fieldwarp
        a :abbr:`DFM (displacements field map)` in ITK format

    Outputs
    -------
    bold
        BOLD series, resampled in native space, including all preprocessing
    bold_mask
        BOLD series mask calculated with the new time-series
    bold_ref
        BOLD reference image: an average-like 3D image of the time-series
    bold_ref_brain
        Same as ``bold_ref``, but once the brain mask has been applied

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The BOLD time-series (including slice-timing correction when applied)
were resampled onto their original, native space by applying
{transforms}.
These resampled BOLD time-series will be referred to as *preprocessed
BOLD in original space*, or just *preprocessed BOLD*.
""".format(transforms="""\
a single, composite transform to correct for head-motion and
susceptibility distortions""" if use_fieldwarp else """\
the transforms to correct for head-motion""")

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'name_source', 'bold_file', 'bold_mask', 'hmc_xforms', 'fieldwarp']),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold', 'bold_mask', 'bold_ref', 'bold_ref_brain']),
        name='outputnode')

    bold_transform = pe.Node(
        MultiApplyTransforms(interpolation=interpolation, float=True, copy_dtype=True),
        name='bold_transform', mem_gb=mem_gb * 3 * omp_nthreads, n_procs=omp_nthreads)

    merge = pe.Node(Merge(compress=use_compression), name='merge',
                    mem_gb=mem_gb * 3)

    # Generate a new BOLD reference
    bold_reference_wf = init_bold_reference_wf(omp_nthreads=omp_nthreads)
    bold_reference_wf.__desc__ = None  # Unset description to avoid second appearance

    workflow.connect([
        (inputnode, merge, [('name_source', 'header_source')]),
        (bold_transform, merge, [('out_files', 'in_files')]),
        (merge, bold_reference_wf, [('out_file', 'inputnode.bold_file')]),
        (merge, outputnode, [('out_file', 'bold')]),
        (bold_reference_wf, outputnode, [
            ('outputnode.ref_image', 'bold_ref'),
            ('outputnode.ref_image_brain', 'bold_ref_brain'),
            ('outputnode.bold_mask', 'bold_mask')]),
    ])

    # Input file is not splitted
    if split_file:
        bold_split = pe.Node(FSLSplit(dimension='t'), name='bold_split',
                             mem_gb=mem_gb * 3)
        workflow.connect([
            (inputnode, bold_split, [('bold_file', 'in_file')]),
            (bold_split, bold_transform, [
                ('out_files', 'input_image'),
                (('out_files', _first), 'reference_image'),
            ])
        ])
    else:
        workflow.connect([
            (inputnode, bold_transform, [('bold_file', 'input_image'),
                                         (('bold_file', _first), 'reference_image')]),
        ])

    if use_fieldwarp:
        merge_xforms = pe.Node(niu.Merge(2), name='merge_xforms',
                               run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, merge_xforms, [('fieldwarp', 'in1'),
                                       ('hmc_xforms', 'in2')]),
            (merge_xforms, bold_transform, [('out', 'transforms')]),
        ])
    else:
        def _aslist(val):
            return [val]
        workflow.connect([
            (inputnode, bold_transform, [(('hmc_xforms', _aslist), 'transforms')]),
        ])

    # Code ready to generate a pre/post processing report
    # bold_bold_report_wf = init_bold_preproc_report_wf(
    #     mem_gb=mem_gb['resampled'],
    #     reportlets_dir=reportlets_dir
    # )
    # workflow.connect([
    #     (inputnode, bold_bold_report_wf, [
    #         ('bold_file', 'inputnode.name_source'),
    #         ('bold_file', 'inputnode.in_pre')]),  # This should be after STC
    #     (bold_bold_trans_wf, bold_bold_report_wf, [
    #         ('outputnode.bold', 'inputnode.in_post')]),
    # ])

    return workflow


def init_bold_preproc_report_wf(mem_gb, reportlets_dir, name='bold_preproc_report_wf'):
    """
    Generate a visual report.

    This workflow generates and saves a reportlet showing the effect of resampling
    the BOLD signal using the standard deviation maps.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.resampling import init_bold_preproc_report_wf
            wf = init_bold_preproc_report_wf(mem_gb=1, reportlets_dir='.')

    Parameters
    ----------
    mem_gb : float
        Size of BOLD file in GB
    reportlets_dir : str
        Directory in which to save reportlets
    name : str, optional
        Workflow name (default: bold_preproc_report_wf)

    Inputs
    ------
    in_pre
        BOLD time-series, before resampling
    in_post
        BOLD time-series, after resampling
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing

    """
    from nipype.algorithms.confounds import TSNR
    from niworkflows.interfaces import SimpleBeforeAfter

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_pre', 'in_post', 'name_source']), name='inputnode')

    pre_tsnr = pe.Node(TSNR(), name='pre_tsnr', mem_gb=mem_gb * 4.5)
    pos_tsnr = pe.Node(TSNR(), name='pos_tsnr', mem_gb=mem_gb * 4.5)

    bold_rpt = pe.Node(SimpleBeforeAfter(), name='bold_rpt',
                       mem_gb=0.1)
    ds_report_bold = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, desc='preproc',
                            keep_dtype=True), name='ds_report_bold',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True
    )

    workflow.connect([
        (inputnode, ds_report_bold, [('name_source', 'source_file')]),
        (inputnode, pre_tsnr, [('in_pre', 'in_file')]),
        (inputnode, pos_tsnr, [('in_post', 'in_file')]),
        (pre_tsnr, bold_rpt, [('stddev_file', 'before')]),
        (pos_tsnr, bold_rpt, [('stddev_file', 'after')]),
        (bold_rpt, ds_report_bold, [('out_report', 'in_file')]),
    ])

    return workflow


def _select_template(template):
    from niworkflows.utils.misc import get_template_specs
    template, specs = template
    specs['suffix'] = specs.get('suffix', 'T1w')
    return get_template_specs(template, template_spec=specs)[0]


def _first(inlist):
    return inlist[0]


def _aslist(in_value):
    if isinstance(in_value, list):
        return in_value
    return [in_value]


def _is_native(in_value):
    return in_value == 'native'


def _tpl_res(in_value):
    if in_value == 'native':
        return 2
    return in_value


def _select_fsaverage_hemi(in_files):
    """Select the fsaverage surface from available surfaces and identify hemisphere"""
    import os
    for fl in in_files:
        if fl.endswith('fsaverage.gii'):
            hemi = 'R' if os.path.basename(fl).startswith('rh') else 'L'
            return fl, hemi
    raise OSError


def _fetch_fsaverage_templates(hemi):
    import templateflow.api as tf

    fsaverage_sphere = str(tf.get(
        'fsLR', space='fsaverage', suffix='sphere', hemi=hemi, density='164k'
    ))
    fsaverage_midthick = str(tf.get(
        'fsLR', space='fsaverage', suffix='midthickness', hemi=hemi, density='164k'
    ))

    return fsaverage_sphere, fsaverage_midthick


def _fetch_fslr_templates(hemi, density):
    """Fetch the necessary fsLR templates for fsaverage to fsLR transform"""
    import templateflow.api as tf

    fslr_sphere = str(tf.get('fsLR', space=None, suffix='sphere', hemi=hemi, density=density))
    fslr_midthick = str(
        tf.get('fsLR', space=None, suffix='midthickness', hemi=hemi, density=density)
    )
    return fslr_sphere, fslr_midthick


def _remove_fslr(spaces):
    nspaces = []
    dens = set()
    for space, specs in spaces:
        if space == 'fsLR':
            dens.add(specs.get('den', '32k'))
        else:
            nspaces.append(space)
    return nspaces, sorted(list(dens))
