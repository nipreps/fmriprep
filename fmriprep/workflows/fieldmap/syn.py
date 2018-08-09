# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_fieldmapless :

Fieldmap-less estimation (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the absence of direct measurements of fieldmap data, we provide an (experimental)
option to estimate the susceptibility distortion based on the ANTs symmetric
normalization (SyN) technique.
This feature may be enabled, using the ``--use-syn-sdc`` flag, and will only be
applied if fieldmaps are unavailable.

During the evaluation phase, the ``--force-syn`` flag will cause this estimation to
be performed *in addition to* fieldmap-based estimation, to permit the direct
comparison of the results of each technique.
Note that, even if ``--force-syn`` is given, the functional outputs of FMRIPREP will
be corrected using the fieldmap-based estimates.

Feedback will be enthusiastically received.


"""
import pkg_resources as pkgr

from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, utility as niu
from nipype.interfaces.image import Rescale
from niworkflows.interfaces.fixes import (FixHeaderApplyTransforms as ApplyTransforms,
                                          FixHeaderRegistration as Registration)
from ...engine import Workflow
from ..bold.util import init_skullstrip_bold_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_syn_sdc_wf(omp_nthreads, bold_pe=None,
                    atlas_threshold=3, name='syn_sdc_wf'):
    """
    This workflow takes a skull-stripped T1w image and reference BOLD image and
    estimates a susceptibility distortion correction warp, using ANTs symmetric
    normalization (SyN) and the average fieldmap atlas described in
    [Treiber2016]_.

    SyN deformation is restricted to the phase-encoding (PE) direction.
    If no PE direction is specified, anterior-posterior PE is assumed.

    SyN deformation is also restricted to regions that are expected to have a
    >3mm (approximately 1 voxel) warp, based on the fieldmap atlas.

    This technique is a variation on those developed in [Huntenburg2014]_ and
    [Wang2017]_.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.fieldmap.syn import init_syn_sdc_wf
        wf = init_syn_sdc_wf(
            bold_pe='j',
            omp_nthreads=8)

    **Inputs**

        bold_ref
            reference image
        bold_ref_brain
            skull-stripped reference image
        template : str
            Name of template targeted by ``template`` output space
        t1_brain
            skull-stripped, bias-corrected structural image
        t1_2_mni_reverse_transform
            inverse registration transform of T1w image to MNI template

    **Outputs**

        out_reference
            the ``bold_ref`` image after unwarping
        out_reference_brain
            the ``bold_ref_brain`` image after unwarping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_mask
            mask of the unwarped input file

    """

    if bold_pe is None or bold_pe[0] not in ['i', 'j']:
        LOGGER.warning('Incorrect phase-encoding direction, assuming PA (posterior-to-anterior).')
        bold_pe = 'j'

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on *fMRIPrep*'s *fieldmap-less* approach.
The deformation field is that resulting from co-registering the BOLD reference
to the same-subject T1w-reference with its intensity inverted [@fieldmapless1;
@fieldmapless2].
Registration is performed with `antsRegistration` (ANTs {ants_ver}), and
the process regularized by constraining deformation to be nonzero only
along the phase-encoding direction, and modulated with an average fieldmap
template [@fieldmapless3].
""".format(ants_ver=Registration().version or '<ver>')
    inputnode = pe.Node(
        niu.IdentityInterface(['bold_ref', 'bold_ref_brain', 'template',
                               't1_brain', 't1_2_mni_reverse_transform']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['out_reference', 'out_reference_brain',
                               'out_mask', 'out_warp']),
        name='outputnode')

    # Collect predefined data
    # Atlas image and registration affine
    atlas_img = pkgr.resource_filename('fmriprep', 'data/fmap_atlas.nii.gz')
    # Registration specifications
    affine_transform = pkgr.resource_filename('fmriprep', 'data/affine.json')
    syn_transform = pkgr.resource_filename('fmriprep', 'data/susceptibility_syn.json')

    invert_t1w = pe.Node(Rescale(invert=True), name='invert_t1w',
                         mem_gb=0.3)

    ref_2_t1 = pe.Node(Registration(from_file=affine_transform),
                       name='ref_2_t1', n_procs=omp_nthreads)
    t1_2_ref = pe.Node(ApplyTransforms(invert_transform_flags=[True]),
                       name='t1_2_ref', n_procs=omp_nthreads)

    # 1) BOLD -> T1; 2) MNI -> T1; 3) ATLAS -> MNI
    transform_list = pe.Node(niu.Merge(3), name='transform_list',
                             mem_gb=DEFAULT_MEMORY_MIN_GB)

    # Inverting (1), then applying in reverse order:
    #
    # ATLAS -> MNI -> T1 -> BOLD
    atlas_2_ref = pe.Node(
        ApplyTransforms(invert_transform_flags=[True, False, False]),
        name='atlas_2_ref', n_procs=omp_nthreads,
        mem_gb=0.3)
    atlas_2_ref.inputs.input_image = atlas_img

    threshold_atlas = pe.Node(
        fsl.maths.MathsCommand(args='-thr {:.8g} -bin'.format(atlas_threshold),
                               output_datatype='char'),
        name='threshold_atlas', mem_gb=0.3)

    fixed_image_masks = pe.Node(niu.Merge(2), name='fixed_image_masks',
                                mem_gb=DEFAULT_MEMORY_MIN_GB)
    fixed_image_masks.inputs.in1 = 'NULL'

    restrict = [[int(bold_pe[0] == 'i'), int(bold_pe[0] == 'j'), 0]] * 2
    syn = pe.Node(
        Registration(from_file=syn_transform, restrict_deformation=restrict),
        name='syn', n_procs=omp_nthreads)

    unwarp_ref = pe.Node(ApplyTransforms(
        dimension=3, float=True, interpolation='LanczosWindowedSinc'),
        name='unwarp_ref')

    skullstrip_bold_wf = init_skullstrip_bold_wf()

    workflow.connect([
        (inputnode, invert_t1w, [('t1_brain', 'in_file'),
                                 ('bold_ref', 'ref_file')]),
        (inputnode, ref_2_t1, [('bold_ref_brain', 'moving_image')]),
        (invert_t1w, ref_2_t1, [('out_file', 'fixed_image')]),
        (inputnode, t1_2_ref, [('bold_ref', 'reference_image')]),
        (invert_t1w, t1_2_ref, [('out_file', 'input_image')]),
        (ref_2_t1, t1_2_ref, [('forward_transforms', 'transforms')]),
        (ref_2_t1, transform_list, [('forward_transforms', 'in1')]),
        (inputnode, transform_list, [
            ('t1_2_mni_reverse_transform', 'in2'),
            (('template', _prior_path), 'in3')]),
        (inputnode, atlas_2_ref, [('bold_ref', 'reference_image')]),
        (transform_list, atlas_2_ref, [('out', 'transforms')]),
        (atlas_2_ref, threshold_atlas, [('output_image', 'in_file')]),
        (threshold_atlas, fixed_image_masks, [('out_file', 'in2')]),
        (inputnode, syn, [('bold_ref_brain', 'moving_image')]),
        (t1_2_ref, syn, [('output_image', 'fixed_image')]),
        (fixed_image_masks, syn, [('out', 'fixed_image_masks')]),
        (syn, outputnode, [('forward_transforms', 'out_warp')]),
        (syn, unwarp_ref, [('forward_transforms', 'transforms')]),
        (inputnode, unwarp_ref, [('bold_ref', 'reference_image'),
                                 ('bold_ref', 'input_image')]),
        (unwarp_ref, skullstrip_bold_wf, [
            ('output_image', 'inputnode.in_file')]),
        (unwarp_ref, outputnode, [('output_image', 'out_reference')]),
        (skullstrip_bold_wf, outputnode, [
            ('outputnode.skull_stripped_file', 'out_reference_brain'),
            ('outputnode.mask_file', 'out_mask')]),
    ])

    return workflow


def _prior_path(template):
    from pkg_resources import resource_filename
    return resource_filename(
        'fmriprep', 'data/fmap_atlas_2_{}_affine.mat'.format(template))
