# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
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
Resampling workflows
++++++++++++++++++++

.. autofunction:: init_pet_surf_wf
.. autofunction:: init_pet_fsLR_resampling_wf
.. autofunction:: init_pet_grayords_wf

"""

from __future__ import annotations

import typing as ty

from nipype.interfaces import freesurfer as fs
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.interfaces.freesurfer import MedialNaNs

from ... import config
from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces.bids import BIDSURI
from ...interfaces.workbench import MetricDilate, MetricMask, MetricResample
from .outputs import prepare_timing_parameters


def init_pet_surf_wf(
    *,
    mem_gb: float,
    surface_spaces: list[str],
    medial_surface_nan: bool,
    metadata: dict,
    output_dir: str,
    name: str = 'pet_surf_wf',
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

            from fmriprep.workflows.pet import init_pet_surf_wf
            wf = init_pet_surf_wf(mem_gb=0.1,
                                   surface_spaces=["fsnative", "fsaverage5"],
                                   medial_surface_nan=False,
                                   metadata={},
                                   output_dir='.',
                                   )

    Parameters
    ----------
    surface_spaces : :obj:`list`
        List of FreeSurfer surface-spaces (either ``fsaverage{3,4,5,6,}`` or ``fsnative``)
        the functional images are to be resampled to.
        For ``fsnative``, images will be resampled to the individual subject's
        native surface.
    medial_surface_nan : :obj:`bool`
        Replace medial wall values with NaNs on functional GIFTI files

    Inputs
    ------
    source_file
        Original PET series
    sources
        List of files used to create the output files.
    pet_t1w
        Motion-corrected PET series in T1 space
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        ITK-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    Outputs
    -------
    surfaces
        PET series, resampled to FreeSurfer surfaces

    """
    from nipype.interfaces.io import FreeSurferSource
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs
    from niworkflows.interfaces.surf import GiftiSetAnatomicalStructure

    from fmriprep.interfaces import DerivativesDataSink

    timing_parameters = prepare_timing_parameters(metadata)

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The PET time-series were resampled onto the following surfaces
(FreeSurfer reconstruction nomenclature):
{out_spaces}.
""".format(out_spaces=', '.join([f'*{s}*' for s in surface_spaces]))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',
                'sources',
                'pet_t1w',
                'subject_id',
                'subjects_dir',
                'fsnative2t1w_xfm',
            ]
        ),
        name='inputnode',
    )
    itersource = pe.Node(niu.IdentityInterface(fields=['target']), name='itersource')
    itersource.iterables = [('target', surface_spaces)]

    surfs_sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='surfs_sources',
    )

    get_fsnative = pe.Node(FreeSurferSource(), name='get_fsnative', run_without_submitting=True)

    def select_target(subject_id, space):
        """Get the target subject ID, given a source subject ID and a target space."""
        return subject_id if space == 'fsnative' else space

    targets = pe.Node(
        niu.Function(function=select_target),
        name='targets',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    itk2lta = pe.Node(
        ConcatenateXFMs(out_fmt='fs', inverse=True), name='itk2lta', run_without_submitting=True
    )
    sampler = pe.MapNode(
        fs.SampleToSurface(
            interp_method='trilinear',
            out_type='gii',
            override_reg_subj=True,
            sampling_method='average',
            sampling_range=(0, 1, 0.2),
            sampling_units='frac',
        ),
        iterfield=['hemi'],
        name='sampler',
        mem_gb=mem_gb * 3,
    )
    sampler.inputs.hemi = ['lh', 'rh']

    update_metadata = pe.MapNode(
        GiftiSetAnatomicalStructure(),
        iterfield=['in_file'],
        name='update_metadata',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    ds_pet_surfs = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            extension='.func.gii',
            TaskName=metadata.get('TaskName'),
            **timing_parameters,
        ),
        iterfield=['in_file', 'hemi'],
        name='ds_pet_surfs',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_pet_surfs.inputs.hemi = ['L', 'R']

    workflow.connect([
        (inputnode, get_fsnative, [
            ('subject_id', 'subject_id'),
            ('subjects_dir', 'subjects_dir')
        ]),
        (inputnode, targets, [('subject_id', 'subject_id')]),
        (inputnode, itk2lta, [
            ('pet_t1w', 'moving'),
            ('fsnative2t1w_xfm', 'in_xfms'),
        ]),
        (get_fsnative, itk2lta, [('T1', 'reference')]),
        (inputnode, sampler, [
            ('subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id'),
            ('pet_t1w', 'source_file'),
        ]),
        (itersource, targets, [('target', 'space')]),
        (itk2lta, sampler, [('out_inv', 'reg_file')]),
        (targets, sampler, [('out', 'target_subject')]),
        (inputnode, ds_pet_surfs, [('source_file', 'source_file')]),
        (inputnode, surfs_sources, [('sources', 'in1')]),
        (surfs_sources, ds_pet_surfs, [('out', 'Sources')]),
        (itersource, ds_pet_surfs, [('target', 'space')]),
        (update_metadata, ds_pet_surfs, [('out_file', 'in_file')]),
    ])  # fmt:skip

    # Refine if medial vertices should be NaNs
    medial_nans = pe.MapNode(
        MedialNaNs(), iterfield=['in_file'], name='medial_nans', mem_gb=DEFAULT_MEMORY_MIN_GB
    )

    if medial_surface_nan:
        # fmt: off
        workflow.connect([
            (inputnode, medial_nans, [('subjects_dir', 'subjects_dir')]),
            (sampler, medial_nans, [('out_file', 'in_file')]),
            (medial_nans, update_metadata, [('out_file', 'in_file')]),
        ])
        # fmt: on
    else:
        workflow.connect([(sampler, update_metadata, [('out_file', 'in_file')])])

    return workflow


def init_pet_fsLR_resampling_wf(
    grayord_density: ty.Literal['91k', '170k'],
    omp_nthreads: int,
    mem_gb: float,
    name: str = 'pet_fsLR_resampling_wf',
):
    """Resample PET time series to fsLR surface.

    This workflow is derived heavily from three scripts within the DCAN-HCP pipelines scripts

    Line numbers correspond to the locations of the code in the original scripts, found at:
    https://github.com/DCAN-Labs/DCAN-HCP/tree/9291324/

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.pet.resampling import init_pet_fsLR_resampling_wf
            wf = init_pet_fsLR_resampling_wf(
                grayord_density='92k',
                omp_nthreads=1,
                mem_gb=1,
            )

    Parameters
    ----------
    grayord_density : :class:`str`
        Either ``"91k"`` or ``"170k"``, representing the total *grayordinates*.
    omp_nthreads : :class:`int`
        Maximum number of threads an individual process may use
    mem_gb : :class:`float`
        Size of PET file in GB
    name : :class:`str`
        Name of workflow (default: ``pet_fsLR_resampling_wf``)

    Inputs
    ------
    pet_file : :class:`str`
        Path to PET file resampled into T1 space
    white : :class:`list` of :class:`str`
        Path to left and right hemisphere white matter GIFTI surfaces.
    pial : :class:`list` of :class:`str`
        Path to left and right hemisphere pial GIFTI surfaces.
    midthickness : :class:`list` of :class:`str`
        Path to left and right hemisphere midthickness GIFTI surfaces.
    midthickness_fsLR : :class:`list` of :class:`str`
        Path to left and right hemisphere midthickness GIFTI surfaces in fsLR space.
    sphere_reg_fsLR : :class:`list` of :class:`str`
        Path to left and right hemisphere sphere.reg GIFTI surfaces, mapping from subject to fsLR
    cortex_mask : :class:`list` of :class:`str`
        Path to left and right hemisphere cortical masks.
    volume_roi : :class:`str` or Undefined
        Pre-calculated mask. Not required.

    Outputs
    -------
    pet_fsLR : :class:`list` of :class:`str`
        Path to PET series resampled as functional GIFTI files in fsLR space

    """
    import smriprep.data
    import templateflow.api as tf
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.utility import KeySelect

    from fmriprep.interfaces.workbench import VolumeToSurfaceMapping

    fslr_density = '32k' if grayord_density == '91k' else '59k'

    workflow = Workflow(name=name)

    workflow.__desc__ = """\
The PET time-series were resampled onto the left/right-symmetric template
"fsLR" using the Connectome Workbench [@hcppipelines].
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet_file',
                'white',
                'pial',
                'midthickness',
                'midthickness_fsLR',
                'sphere_reg_fsLR',
                'cortex_mask',
                'volume_roi',
            ]
        ),
        name='inputnode',
    )

    hemisource = pe.Node(
        niu.IdentityInterface(fields=['hemi']),
        name='hemisource',
        iterables=[('hemi', ['L', 'R'])],
    )

    joinnode = pe.JoinNode(
        niu.IdentityInterface(fields=['pet_fsLR']),
        name='joinnode',
        joinsource='hemisource',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_fsLR']),
        name='outputnode',
    )

    # select white, midthickness and pial surfaces based on hemi
    select_surfaces = pe.Node(
        KeySelect(
            fields=[
                'white',
                'pial',
                'midthickness',
                'midthickness_fsLR',
                'sphere_reg_fsLR',
                'template_sphere',
                'cortex_mask',
                'template_roi',
            ],
            keys=['L', 'R'],
        ),
        name='select_surfaces',
        run_without_submitting=True,
    )
    select_surfaces.inputs.template_sphere = [
        str(sphere)
        for sphere in tf.get(
            template='fsLR',
            density=fslr_density,
            suffix='sphere',
            space=None,
            extension='.surf.gii',
        )
    ]
    atlases = smriprep.data.load('atlases')
    select_surfaces.inputs.template_roi = [
        str(atlases / f'L.atlasroi.{fslr_density}_fs_LR.shape.gii'),
        str(atlases / f'R.atlasroi.{fslr_density}_fs_LR.shape.gii'),
    ]

    # RibbonVolumeToSurfaceMapping.sh
    # Line 85 thru ...
    volume_to_surface = pe.Node(
        VolumeToSurfaceMapping(method='ribbon-constrained'),
        name='volume_to_surface',
        mem_gb=mem_gb * 3,
        n_procs=omp_nthreads,
    )
    metric_dilate = pe.Node(
        MetricDilate(distance=10, nearest=True),
        name='metric_dilate',
        mem_gb=1,
        n_procs=omp_nthreads,
    )
    mask_native = pe.Node(MetricMask(), name='mask_native')
    resample_to_fsLR = pe.Node(
        MetricResample(method='ADAP_BARY_AREA', area_surfs=True),
        name='resample_to_fsLR',
        mem_gb=1,
        n_procs=omp_nthreads,
    )
    # ... line 89
    mask_fsLR = pe.Node(MetricMask(), name='mask_fsLR')

    workflow.connect([
        (inputnode, select_surfaces, [
            ('white', 'white'),
            ('pial', 'pial'),
            ('midthickness', 'midthickness'),
            ('midthickness_fsLR', 'midthickness_fsLR'),
            ('sphere_reg_fsLR', 'sphere_reg_fsLR'),
            ('cortex_mask', 'cortex_mask'),
        ]),
        (hemisource, select_surfaces, [('hemi', 'key')]),
        # Resample PET to native surface, dilate and mask
        (inputnode, volume_to_surface, [
            ('pet_file', 'volume_file'),
            ('volume_roi', 'volume_roi'),
        ]),
        (select_surfaces, volume_to_surface, [
            ('midthickness', 'surface_file'),
            ('white', 'inner_surface'),
            ('pial', 'outer_surface'),
        ]),
        (select_surfaces, metric_dilate, [('midthickness', 'surf_file')]),
        (select_surfaces, mask_native, [('cortex_mask', 'mask')]),
        (volume_to_surface, metric_dilate, [('out_file', 'in_file')]),
        (metric_dilate, mask_native, [('out_file', 'in_file')]),
        # Resample PET to fsLR and mask
        (select_surfaces, resample_to_fsLR, [
            ('sphere_reg_fsLR', 'current_sphere'),
            ('template_sphere', 'new_sphere'),
            ('midthickness', 'current_area'),
            ('midthickness_fsLR', 'new_area'),
            ('cortex_mask', 'roi_metric'),
        ]),
        (mask_native, resample_to_fsLR, [('out_file', 'in_file')]),
        (select_surfaces, mask_fsLR, [('template_roi', 'mask')]),
        (resample_to_fsLR, mask_fsLR, [('out_file', 'in_file')]),
        # Output
        (mask_fsLR, joinnode, [('out_file', 'pet_fsLR')]),
        (joinnode, outputnode, [('pet_fsLR', 'pet_fsLR')]),
    ])  # fmt:skip

    return workflow


def init_pet_grayords_wf(
    grayord_density: ty.Literal['91k', '170k'],
    mem_gb: float,
    repetition_time: float,
    name: str = 'pet_grayords_wf',
):
    """
    Sample Grayordinates files onto the fsLR atlas.

    Outputs are in CIFTI2 format.

    Workflow Graph
        .. workflow::
            :graph2use: colored
            :simple_form: yes

            from fmriprep.workflows.pet.resampling import init_pet_grayords_wf
            wf = init_pet_grayords_wf(mem_gb=0.1, grayord_density="91k", repetition_time=2)

    Parameters
    ----------
    grayord_density : :class:`str`
        Either ``"91k"`` or ``"170k"``, representing the total *grayordinates*.
    mem_gb : :obj:`float`
        Size of PET file in GB
    repetition_time : :obj:`float`
        Repetition time in seconds
    name : :obj:`str`
        Unique name for the subworkflow (default: ``"pet_grayords_wf"``)

    Inputs
    ------
    pet_fsLR : :obj:`str`
        List of paths to PET series resampled as functional GIFTI files in fsLR space
    pet_std : :obj:`str`
        List of PET conversions to standard spaces.
    spatial_reference : :obj:`str`
        List of unique identifiers corresponding to the PET standard-conversions.


    Outputs
    -------
    cifti_bold : :obj:`str`
        PET CIFTI dtseries.
    cifti_metadata : :obj:`str`
        BIDS metadata file corresponding to ``cifti_bold``.

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.cifti import GenerateCifti

    workflow = Workflow(name=name)

    mni_density = '2' if grayord_density == '91k' else '1'

    workflow.__desc__ = f"""\
*Grayordinates* files [@hcppipelines] containing {grayord_density} samples were also
generated with surface data transformed directly to fsLR space and subcortical
data transformed to {mni_density} mm resolution MNI152NLin6Asym space.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_std', 'pet_fsLR']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['cifti_bold', 'cifti_metadata']),
        name='outputnode',
    )

    gen_cifti = pe.Node(
        GenerateCifti(
            TR=repetition_time,
            grayordinates=grayord_density,
        ),
        name='gen_cifti',
        mem_gb=mem_gb,
    )

    workflow.connect([
        (inputnode, gen_cifti, [
            ('pet_fsLR', 'surface_bolds'),
            ('pet_std', 'pet_file'),
        ]),
        (gen_cifti, outputnode, [
            ('out_file', 'cifti_bold'),
            ('out_metadata', 'cifti_metadata'),
        ]),
    ])  # fmt:skip
    return workflow
