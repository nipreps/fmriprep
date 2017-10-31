# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Resampling workflows
++++++++++++++++++++

.. autofunction:: init_bold_surf_wf
.. autofunction:: init_bold_mni_trans_wf

"""
import os.path as op


from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import utility as niu, freesurfer as fs

from niworkflows import data as nid
from niworkflows.interfaces.utils import GenerateSamplingReference
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

from ...interfaces import GiftiSetAnatomicalStructure, MultiApplyTransforms
from ...interfaces.nilearn import Merge
# See https://github.com/poldracklab/fmriprep/issues/768
from ...interfaces.freesurfer import PatchedConcatenateLTA as ConcatenateLTA

DEFAULT_MEMORY_MIN_GB = 0.01


def init_bold_surf_wf(bold_file_size_gb, output_spaces, medial_surface_nan, name='bold_surf_wf'):
    """
    This workflow samples functional images to FreeSurfer surfaces

    For each vertex, the cortical ribbon is sampled at six points (spaced 20% of thickness apart)
    and averaged.

    Outputs are in GIFTI format.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_surf_wf
        wf = init_bold_surf_wf(bold_file_size_gb=0.1,
                               output_spaces=['T1w', 'fsnative',
                                             'template', 'fsaverage5'],
                               medial_surface_nan=False)

    **Parameters**

        output_spaces : list
            List of output spaces functional images are to be resampled to
            Target spaces beginning with ``fs`` will be selected for resampling,
            such as ``fsaverage`` or related template spaces
            If the list contains ``fsnative``, images will be resampled to the
            individual subject's native surface
        medial_surface_nan : bool
            Replace medial wall values with NaNs on functional GIFTI files

    **Inputs**

        source_file
            Motion-corrected BOLD series in T1 space
        t1_preproc
            Bias-corrected structural template image
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space

    **Outputs**

        surfaces
            BOLD series, resampled to FreeSurfer surfaces

    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 't1_preproc', 'subject_id', 'subjects_dir',
                                      't1_2_fsnative_forward_transform']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['surfaces']), name='outputnode')

    spaces = [space for space in output_spaces if space.startswith('fs')]

    def select_target(subject_id, space):
        """ Given a source subject ID and a target space, get the target subject ID """
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

    resampling_xfm = pe.Node(fs.utils.LTAConvert(in_lta='identity.nofile', out_lta=True),
                             name='resampling_xfm')
    set_xfm_source = pe.Node(ConcatenateLTA(out_type='RAS2RAS'), name='set_xfm_source')

    sampler = pe.MapNode(
        fs.SampleToSurface(sampling_method='average', sampling_range=(0, 1, 0.2),
                           sampling_units='frac', interp_method='trilinear', cortex_mask=True,
                           override_reg_subj=True, out_type='gii'),
        iterfield=['source_file', 'target_subject'],
        iterables=('hemi', ['lh', 'rh']),
        name='sampler', mem_gb=bold_file_size_gb * 3)

    def medial_wall_to_nan(in_file, subjects_dir, target_subject):
        """ Convert values on medial wall to NaNs
        """
        import nibabel as nb
        import numpy as np
        import os

        fn = os.path.basename(in_file)
        if not target_subject.startswith('fs'):
            return in_file

        cortex = nb.freesurfer.read_label(os.path.join(
            subjects_dir, target_subject, 'label', '{}.cortex.label'.format(fn[:2])))
        func = nb.load(in_file)
        medial = np.delete(np.arange(len(func.darrays[0].data)), cortex)
        for darray in func.darrays:
            darray.data[medial] = np.nan

        out_file = os.path.join(os.getcwd(), fn)
        func.to_filename(out_file)
        return out_file

    medial_nans = pe.MapNode(niu.Function(function=medial_wall_to_nan),
                             iterfield=['in_file', 'target_subject'], name='medial_nans',
                             mem_gb=DEFAULT_MEMORY_MIN_GB)

    merger = pe.JoinNode(niu.Merge(1, ravel_inputs=True), name='merger',
                         joinsource='sampler', joinfield=['in1'], run_without_submitting=True,
                         mem_gb=DEFAULT_MEMORY_MIN_GB)

    update_metadata = pe.MapNode(GiftiSetAnatomicalStructure(), iterfield='in_file',
                                 name='update_metadata', mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, targets, [('subject_id', 'subject_id')]),
        (inputnode, rename_src, [('source_file', 'in_file')]),
        (inputnode, resampling_xfm, [('source_file', 'source_file'),
                                     ('t1_preproc', 'target_file')]),
        (inputnode, set_xfm_source, [('t1_2_fsnative_forward_transform', 'in_lta2')]),
        (resampling_xfm, set_xfm_source, [('out_lta', 'in_lta1')]),
        (inputnode, sampler, [('subjects_dir', 'subjects_dir'),
                              ('subject_id', 'subject_id')]),
        (set_xfm_source, sampler, [('out_file', 'reg_file')]),
        (targets, sampler, [('out', 'target_subject')]),
        (rename_src, sampler, [('out_file', 'source_file')]),
        (merger, update_metadata, [('out', 'in_file')]),
        (update_metadata, outputnode, [('out_file', 'surfaces')]),
    ])

    if medial_surface_nan:
        workflow.connect([
            (inputnode, medial_nans, [('subjects_dir', 'subjects_dir')]),
            (sampler, medial_nans, [('out_file', 'in_file')]),
            (targets, medial_nans, [('out', 'target_subject')]),
            (medial_nans, merger, [('out', 'in1')]),
        ])
    else:
        workflow.connect(sampler, 'out_file', merger, 'in1')

    return workflow


def init_bold_mni_trans_wf(template, bold_file_size_gb, omp_nthreads,
                           name='bold_mni_trans_wf',
                           output_grid_ref=None, use_compression=True,
                           use_fieldwarp=False):
    """
    This workflow samples functional images to the MNI template in a "single shot"
    from the original BOLD series.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_mni_trans_wf
        wf = init_bold_mni_trans_wf(template='MNI152NLin2009cAsym',
                                    bold_file_size_gb=3,
                                    omp_nthreads=1,
                                    output_grid_ref=None)

    **Parameters**

        template : str
            Name of template targeted by `'template'` output space
        bold_file_size_gb : float
            Size of BOLD file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_mni_trans_wf``)
        output_grid_ref : str or None
            Path of custom reference image for normalization
        use_compression : bool
            Save registered BOLD series as ``.nii.gz``
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from BOLD to MNI

    **Inputs**

        itk_bold_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        bold_split
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

    **Outputs**

        bold_mni
            BOLD series, resampled to template space
        bold_mask_mni
            BOLD series mask in template space

    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'itk_bold_to_t1',
            't1_2_mni_forward_transform',
            'name_source',
            'bold_split',
            'bold_mask',
            'hmc_xforms',
            'fieldwarp'
        ]),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_mni', 'bold_mask_mni']),
        name='outputnode')

    def _aslist(in_value):
        if isinstance(in_value, list):
            return in_value
        return [in_value]

    gen_ref = pe.Node(GenerateSamplingReference(), name='gen_ref',
                      mem_gb=0.3)  # 256x256x256 * 64 / 8 ~ 150MB)
    template_str = nid.TEMPLATE_MAP[template]
    gen_ref.inputs.fixed_image = op.join(nid.get_dataset(template_str), '1mm_T1.nii.gz')

    mask_mni_tfm = pe.Node(
        ApplyTransforms(interpolation='NearestNeighbor', float=True),
        name='mask_mni_tfm',
        mem_gb=bold_file_size_gb * 3
    )

    # Write corrected file in the designated output dir
    mask_merge_tfms = pe.Node(niu.Merge(2), name='mask_merge_tfms', run_without_submitting=True,
                              mem_gb=DEFAULT_MEMORY_MIN_GB)

    nxforms = 4 if use_fieldwarp else 3
    merge_xforms = pe.Node(niu.Merge(nxforms), name='merge_xforms',
                           run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([(inputnode, merge_xforms, [('hmc_xforms', 'in%d' % nxforms)])])

    if use_fieldwarp:
        workflow.connect([(inputnode, merge_xforms, [('fieldwarp', 'in3')])])

    workflow.connect([
        (inputnode, gen_ref, [('bold_mask', 'moving_image')]),
        (inputnode, mask_merge_tfms, [('t1_2_mni_forward_transform', 'in1'),
                                      (('itk_bold_to_t1', _aslist), 'in2')]),
        (mask_merge_tfms, mask_mni_tfm, [('out', 'transforms')]),
        (mask_mni_tfm, outputnode, [('output_image', 'bold_mask_mni')]),
        (inputnode, mask_mni_tfm, [('bold_mask', 'input_image')])
    ])

    bold_to_mni_transform = pe.Node(
        MultiApplyTransforms(interpolation="LanczosWindowedSinc", float=True, copy_dtype=True),
        name='bold_to_mni_transform', mem_gb=bold_file_size_gb * 3, n_procs=omp_nthreads)

    merge = pe.Node(Merge(compress=use_compression), name='merge',
                    mem_gb=bold_file_size_gb * 3)

    workflow.connect([
        (inputnode, merge_xforms, [('t1_2_mni_forward_transform', 'in1'),
                                   (('itk_bold_to_t1', _aslist), 'in2')]),
        (merge_xforms, bold_to_mni_transform, [('out', 'transforms')]),
        (inputnode, merge, [('name_source', 'header_source')]),
        (inputnode, bold_to_mni_transform, [('bold_split', 'input_image')]),
        (bold_to_mni_transform, merge, [('out_files', 'in_files')]),
        (merge, outputnode, [('out_file', 'bold_mni')]),
    ])

    if output_grid_ref is None:
        workflow.connect([
            (gen_ref, mask_mni_tfm, [('out_file', 'reference_image')]),
            (gen_ref, bold_to_mni_transform, [('out_file', 'reference_image')]),
        ])
    else:
        mask_mni_tfm.inputs.reference_image = output_grid_ref
        bold_to_mni_transform.inputs.reference_image = output_grid_ref
    return workflow
