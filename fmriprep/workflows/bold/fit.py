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
import nibabel as nb
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf, init_skullstrip_bold_wf
from niworkflows.interfaces.header import ValidateImage

from ... import config
from ...interfaces.reports import FunctionalSummary
from ...interfaces.resampling import ResampleSeries
from ...utils.bids import extract_entities
from ...utils.misc import estimate_bold_mem_usage

# BOLD workflows
from .hmc import init_pet_hmc_wf
from .outputs import (
    init_ds_petmask_wf,
    init_ds_petref_wf,
    init_ds_hmc_wf,
    init_ds_registration_wf,
    init_func_fit_reports_wf,
)
from .reference import init_petref_wf, init_validation_and_dummies_wf
from .registration import init_pet_reg_wf


def init_pet_fit_wf(
    *,
    pet_file: str,
    precomputed: dict = None,
    omp_nthreads: int = 1,
    name: str = 'bold_fit_wf',
) -> pe.Workflow:
    """
    This workflow controls the minimal estimation steps for functional preprocessing.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.pet.fit import init_pet_fit_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_pet_fit_wf(bold_series=[str(bold_file)])

    Parameters
    ----------
    bold_series
        List of paths to NIfTI files
    precomputed
        Dictionary containing precomputed derivatives to reuse, if possible.

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    t1w_preproc
        Bias-corrected structural template image
    t1w_mask
        Mask of the skull-stripped template image
    t1w_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    anat2std_xfm
        List of transform files, collated with templates
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    Outputs
    -------
    petref
        PET reference image used for head motion correction.
    pet_mask
        Mask of ``petref``.
    motion_xfm
        Affine transforms from each PET volume to ``petref``, written
        as concatenated ITK affine transforms.
    petref2anat_xfm
        Affine transform mapping from PET reference space to the anatomical
        space.
    dummy_scans
        The number of dummy scans declared or detected at the beginning of the series.

    See Also
    --------

    * :py:func:`~fmriprep.workflows.pet.reference.init_petref_wf`
    * :py:func:`~fmriprep.workflows.pet.hmc.init_pet_hmc_wf`
    * :py:func:`~niworkflows.func.utils.init_enhance_and_skullstrip_pet_wf`
    * :py:func:`~fmriprep.workflows.pet.registration.init_pet_reg_wf`
    * :py:func:`~fmriprep.workflows.pet.outputs.init_ds_petref_wf`
    * :py:func:`~fmriprep.workflows.pet.outputs.init_ds_hmc_wf`
    * :py:func:`~fmriprep.workflows.pet.outputs.init_ds_registration_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmriprep.utils.misc import estimate_pet_mem_usage

    if precomputed is None:
        precomputed = {}
    layout = config.execution.layout

    # Get metadata from PET file(s)
    entities = extract_entities(pet_file)
    metadata = layout.get_metadata(pet_file)
    orientation = ''.join(nb.aff2axcodes(nb.load(pet_file).affine))

    pet_tlen, mem_gb = estimate_pet_mem_usage(pet_file)

    petref = precomputed.get('petref')
    # Can contain
    #  1) petref2anat
    #  2) hmc
    transforms = precomputed.get('transforms', {})
    hmc_xforms = transforms.get('hmc')
    petref2anat_xform = transforms.get('petref2anat')

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet_file',
                # Anatomical coregistration
                't1w_preproc',
                't1w_mask',
                't1w_dseg',
                'subjects_dir',
                'subject_id',
                'fsnative2t1w_xfm',
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.pet_file = pet_file

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dummy_scans',
                'petref',
                'pet_mask',
                'motion_xfm',
                'petref2anat_xfm',
            ],
        ),
        name='outputnode',
    )

    # If all derivatives exist, inputnode could go unconnected, so add explicitly
    workflow.add_nodes([inputnode])

    petref_buffer = pe.Node(
        niu.IdentityInterface(fields=['petref', 'pet_file', 'pet_mask']),
        name='petref_buffer',
    )
    hmc_buffer = pe.Node(niu.IdentityInterface(fields=['hmc_xforms']), name='hmc_buffer')

    if petref:
        petref_buffer.inputs.petref = petref
        config.loggers.workflow.debug('Reusing motion correction reference: %s', petref)
    if hmc_xforms:
        hmc_buffer.inputs.hmc_xforms = hmc_xforms
        config.loggers.workflow.debug('Reusing motion correction transforms: %s', hmc_xforms)

    summary = pe.Node(
        FunctionalSummary(
            distortion_correction='None',  # Can override with connection
            registration=(
                'Precomputed'
                if petref2anat_xform
                else 'FreeSurfer'
                if config.workflow.run_reconall
                else 'FSL'
            ),
            registration_dof=config.workflow.pet2anat_dof,
            registration_init=config.workflow.pet2anat_init,
            tr=metadata['RepetitionTime'],
            orientation=orientation,
        ),
        name='summary',
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )
    summary.inputs.dummy_scans = config.workflow.dummy_scans
    if config.workflow.level == 'full':
        # Hack. More pain than it's worth to connect this up at a higher level.
        # We can consider separating out fit and transform summaries,
        # or connect a bunch a bunch of summary parameters to outputnodes
        # to make available to the base workflow.
        summary.inputs.slice_timing = (
            bool(metadata.get('SliceTiming')) and 'slicetiming' not in config.workflow.ignore
        )

    func_fit_reports_wf = init_func_fit_reports_wf(
        sdc_correction=False,
        freesurfer=config.workflow.run_reconall,
        output_dir=config.execution.fmriprep_dir,
    )

    workflow.connect([
        (petref_buffer, outputnode, [
            ('petref', 'petref'),
            ('petmask', 'petmask'),
        ]),
        (hmc_buffer, outputnode, [
            ('hmc_xforms', 'motion_xfm'),
        ]),
        (inputnode, func_fit_reports_wf, [
            ('pet_file', 'inputnode.source_file'),
            ('t1w_preproc', 'inputnode.t1w_preproc'),
            # May not need all of these
            ('t1w_mask', 'inputnode.t1w_mask'),
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
        ]),
        (outputnode, func_fit_reports_wf, [
            ('pet_mask', 'inputnode.pet_mask'),
            ('petref2anat_xfm', 'inputnode.petref2anat_xfm'),
        ]),
        (summary, func_fit_reports_wf, [('out_report', 'inputnode.summary_report')]),
    ])  # fmt:skip

    # Stage 1: Generate motion correction petref
    petref_source_buffer = pe.Node(
        niu.IdentityInterface(fields=['in_file']),
        name='petref_source_buffer',
    )
    if not petref:
        config.loggers.workflow.info('Stage 1: Adding PET reference workflow')
        petref_wf = init_petref_wf(
            name='petref_wf',
            pet_file=pet_file,
        )
        petref_wf.inputs.inputnode.dummy_scans = config.workflow.dummy_scans

        ds_petref_wf = init_ds_petref_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            desc='hmc',
            name='ds_petref_wf',
        )
        ds_petref_wf.inputs.inputnode.source_files = [pet_file]

        workflow.connect([
            (petref_wf, petref_buffer, [
                ('outputnode.bold_file', 'bold_file'),
                ('outputnode.boldref', 'boldref'),
                ('outputnode.skip_vols', 'dummy_scans'),
            ]),
            (petref_buffer, ds_petref_wf, [('boldref', 'inputnode.boldref')]),
            (petref_wf, summary, [('outputnode.algo_dummy_scans', 'algo_dummy_scans')]),
            (petref_wf, func_fit_reports_wf, [
                ('outputnode.validation_report', 'inputnode.validation_report'),
            ]),
            (ds_petref_wf, petref_source_buffer, [
                ('outputnode.boldref', 'in_file'),
            ]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found HMC boldref - skipping Stage 1')

        validation_and_dummies_wf = init_validation_and_dummies_wf(bold_file=bold_file)

        workflow.connect([
            (validation_and_dummies_wf, petref_buffer, [
                ('outputnode.bold_file', 'bold_file'),
                ('outputnode.skip_vols', 'dummy_scans'),
            ]),
            (validation_and_dummies_wf, func_fit_reports_wf, [
                ('outputnode.validation_report', 'inputnode.validation_report'),
            ]),
            (petref_buffer, petref_source_buffer, [('boldref', 'in_file')]),
        ])  # fmt:skip

    # Stage 2: Estimate head motion
    if not hmc_xforms:
        config.loggers.workflow.info('Stage 2: Adding motion correction workflow')
        pet_hmc_wf = init_pet_hmc_wf(
            name='pet_hmc_wf', mem_gb=mem_gb['filesize'], omp_nthreads=omp_nthreads
        )

        ds_hmc_wf = init_ds_hmc_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
        )
        ds_hmc_wf.inputs.inputnode.source_files = [bold_file]

        workflow.connect([
            (petref_buffer, pet_hmc_wf, [
                ('boldref', 'inputnode.raw_ref_image'),
                ('bold_file', 'inputnode.bold_file'),
            ]),
            (pet_hmc_wf, ds_hmc_wf, [('outputnode.xforms', 'inputnode.xforms')]),
            (ds_hmc_wf, hmc_buffer, [('outputnode.xforms', 'hmc_xforms')]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found motion correction transforms - skipping Stage 2')

    # Stage 3: Create coregistration reference
    # Fieldmap correction only happens during fit if this stage is needed
    if not coreg_boldref:
        config.loggers.workflow.info('Stage 3: Adding coregistration boldref workflow')

        enhance_boldref_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

        ds_coreg_boldref_wf = init_ds_petref_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            desc='coreg',
            name='ds_coreg_boldref_wf',
        )
        ds_petmask_wf = init_ds_petmask_wf(
            output_dir=config.execution.fmriprep_dir,
            desc='brain',
            name='ds_petmask_wf',
        )
        ds_boldmask_wf.inputs.inputnode.source_files = [bold_file]

        workflow.connect([
            (petref_buffer, enhance_boldref_wf, [('boldref', 'inputnode.in_file')]),
            (petref_source_buffer, ds_coreg_boldref_wf, [
                ('in_file', 'inputnode.source_files'),
            ]),
            (ds_coreg_boldref_wf, regref_buffer, [('outputnode.boldref', 'boldref')]),
            (ds_boldmask_wf, regref_buffer, [('outputnode.boldmask', 'boldmask')]),
        ])  # fmt:skip

        if True:  # No fieldmap... check if these connections are PET relevant
            workflow.connect([
                (enhance_boldref_wf, ds_coreg_boldref_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.boldref'),
                ]),
                (enhance_boldref_wf, ds_boldmask_wf, [
                    ('outputnode.mask_file', 'inputnode.boldmask'),
                ]),
            ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found coregistration reference - skipping Stage 3')

        # TODO: Allow precomputed bold masks to be passed
        # Also needs consideration for how it interacts above
        skullstrip_precomp_ref_wf = init_skullstrip_bold_wf(name='skullstrip_precomp_ref_wf')
        skullstrip_precomp_ref_wf.inputs.inputnode.in_file = coreg_boldref
        workflow.connect([
            (skullstrip_precomp_ref_wf, regref_buffer, [('outputnode.mask_file', 'boldmask')])
        ])  # fmt:skip

    if not petref2anat_xform:
        use_bbr = False

        # calculate PET registration to T1w
        bold_reg_wf = init_pet_reg_wf(
            pet2anat_dof=config.workflow.pet2anat_dof,
            pet2anat_init=config.workflow.pet2anat_init,
            use_bbr=use_bbr,
            freesurfer=config.workflow.run_reconall,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb['resampled'],
            sloppy=config.execution.sloppy,
        )

        ds_boldreg_wf = init_ds_registration_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            source='boldref',
            dest='T1w',
            name='ds_boldreg_wf',
        )

        # fmt:off
        workflow.connect([
            (inputnode, bold_reg_wf, [
                ('t1w_preproc', 'inputnode.t1w_preproc'),
                ('t1w_mask', 'inputnode.t1w_mask'),
                ('t1w_dseg', 'inputnode.t1w_dseg'),
                # Undefined if --fs-no-reconall, but this is safe
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id'),
                ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm'),
            ]),
            (regref_buffer, bold_reg_wf, [('boldref', 'inputnode.ref_pet_brain')]),
            # Incomplete sources
            (regref_buffer, ds_boldreg_wf, [('boldref', 'inputnode.source_files')]),
            (bold_reg_wf, ds_boldreg_wf, [('outputnode.itk_bold_to_t1', 'inputnode.xform')]),
            (ds_boldreg_wf, outputnode, [('outputnode.xform', 'petref2anat_xfm')]),
            (bold_reg_wf, summary, [('outputnode.fallback', 'fallback')]),
        ])
        # fmt:on
    else:
        outputnode.inputs.petref2anat_xfm = petref2anat_xform

    return workflow


def init_pet_native_wf(
    *,
    pet_file: str,
    omp_nthreads: int = 1,
    name: str = 'bold_native_wf',
) -> pe.Workflow:
    r"""
    Minimal resampling workflow.

    This workflow performs slice-timing correction, and resamples to boldref space
    with head motion and susceptibility distortion correction. It also selects
    the transforms needed to perform further resampling.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.pet.fit import init_pet_native_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_pet_native_wf(bold_series=[str(bold_file)])

    Parameters
    ----------
    pet_file
        Path to NIfTI file.

    Inputs
    ------
    petref
        PET reference file
    pet_mask
        Mask of pet reference file
    motion_xfm
        Affine transforms from each PET volume to ``petref``, written
        as concatenated ITK affine transforms.

    Outputs
    -------
    pet_minimal
        BOLD series ready for further resampling.
    pet_native
        BOLD series resampled into BOLD reference space. Head motion correction
        will be applied to each file.
    metadata
        Metadata dictionary of BOLD series
    motion_xfm
        Motion correction transforms for further correcting bold_minimal.

    """

    layout = config.execution.layout

    metadata = layout.get_metadata(pet_file)

    _, mem_gb = estimate_bold_mem_usage(pet_file)

    run_stc = bool(metadata.get('SliceTiming')) and 'slicetiming' not in config.workflow.ignore

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # BOLD fit
                'boldref',
                'bold_mask',
                'motion_xfm',
                'dummy_scans',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_minimal',
                'bold_native',
                'metadata',
                # Transforms
                'motion_xfm',
            ],  # fmt:skip
        ),
        name='outputnode',
    )
    outputnode.inputs.metadata = metadata

    boldbuffer = pe.Node(
        niu.IdentityInterface(fields=['bold_file']), name='boldbuffer'
    )

    # BOLD source: track original BOLD file(s)
    bold_source = pe.Node(niu.Select(inlist=bold_series), name='bold_source')
    validate_bold = pe.Node(ValidateImage(), name='validate_bold')
    workflow.connect([
        (bold_source, validate_bold, [('out', 'in_file')]),
    ])  # fmt:skip

    workflow.connect([(validate_bold, boldbuffer, [('out_file', 'bold_file')])])

    # Resample to boldref
    boldref_bold = pe.Node(
        ResampleSeries(),
        name='boldref_bold',
        n_procs=omp_nthreads,
        mem_gb=mem_gb['resampled'],
    )

    workflow.connect([
        (inputnode, boldref_bold, [
            ('boldref', 'ref_file'),
            ('motion_xfm', 'transforms'),
        ]),
        (boldbuffer, boldref_bold, [
            ('bold_file', 'in_file'),
        ]),
    ])  # fmt:skip

    workflow.connect([
        (inputnode, outputnode, [('motion_xfm', 'motion_xfm')]),
        (boldbuffer, outputnode, [('bold_file', 'bold_minimal')]),
        (boldref_bold, outputnode, [('out_file', 'bold_native')]),
    ])  # fmt:skip

    return workflow
