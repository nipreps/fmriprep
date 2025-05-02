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
from .hmc import init_bold_hmc_wf
from .outputs import (
    init_ds_boldmask_wf,
    init_ds_boldref_wf,
    init_ds_hmc_wf,
    init_ds_registration_wf,
    init_func_fit_reports_wf,
)
from .reference import init_raw_boldref_wf, init_validation_and_dummies_wf
from .registration import init_bold_reg_wf
from .stc import init_bold_stc_wf


def init_bold_fit_wf(
    *,
    bold_series: list[str],
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
            from fmriprep.workflows.bold.fit import init_bold_fit_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_bold_fit_wf(bold_series=[str(bold_file)])

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
    hmc_boldref
        BOLD reference image used for head motion correction.
        Minimally processed to ensure consistent contrast with BOLD series.
    coreg_boldref
        BOLD reference image used for coregistration. Contrast-enhanced
        for greater anatomical fidelity, and aligned with ``hmc_boldref``.
    bold_mask
        Mask of ``coreg_boldref``.
    motion_xfm
        Affine transforms from each BOLD volume to ``hmc_boldref``, written
        as concatenated ITK affine transforms.
    boldref2anat_xfm
        Affine transform mapping from BOLD reference space to the anatomical
        space.
    dummy_scans
        The number of dummy scans declared or detected at the beginning of the series.

    See Also
    --------

    * :py:func:`~fmriprep.workflows.bold.reference.init_raw_boldref_wf`
    * :py:func:`~fmriprep.workflows.bold.hmc.init_bold_hmc_wf`
    * :py:func:`~niworkflows.func.utils.init_enhance_and_skullstrip_bold_wf`
    * :py:func:`~fmriprep.workflows.bold.registration.init_bold_reg_wf`
    * :py:func:`~fmriprep.workflows.bold.outputs.init_ds_boldref_wf`
    * :py:func:`~fmriprep.workflows.bold.outputs.init_ds_hmc_wf`
    * :py:func:`~fmriprep.workflows.bold.outputs.init_ds_registration_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmriprep.utils.misc import estimate_bold_mem_usage

    if precomputed is None:
        precomputed = {}
    layout = config.execution.layout
    bids_filters = config.execution.get().get('bids_filters', {})

    bold_file = bold_series[0]

    # Get metadata from BOLD file(s)
    entities = extract_entities(bold_series)
    metadata = layout.get_metadata(bold_file)
    orientation = ''.join(nb.aff2axcodes(nb.load(bold_file).affine))

    bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

    hmc_boldref = precomputed.get('hmc_boldref')
    coreg_boldref = precomputed.get('coreg_boldref')
    # Can contain
    #  1) boldref2anat
    #  2) hmc
    transforms = precomputed.get('transforms', {})
    hmc_xforms = transforms.get('hmc')
    boldref2anat_xform = transforms.get('boldref2anat')

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
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
    inputnode.inputs.bold_file = bold_series

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dummy_scans',
                'hmc_boldref',
                'coreg_boldref',
                'bold_mask',
                'motion_xfm',
                'boldref2anat_xfm',
            ],
        ),
        name='outputnode',
    )

    # If all derivatives exist, inputnode could go unconnected, so add explicitly
    workflow.add_nodes([inputnode])

    hmcref_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'bold_file', 'dummy_scans']),
        name='hmcref_buffer',
    )
    hmc_buffer = pe.Node(niu.IdentityInterface(fields=['hmc_xforms']), name='hmc_buffer')
    regref_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'boldmask']), name='regref_buffer'
    )

    if hmc_boldref:
        hmcref_buffer.inputs.boldref = hmc_boldref
        config.loggers.workflow.debug('Reusing motion correction reference: %s', hmc_boldref)
    if hmc_xforms:
        hmc_buffer.inputs.hmc_xforms = hmc_xforms
        config.loggers.workflow.debug('Reusing motion correction transforms: %s', hmc_xforms)
    if coreg_boldref:
        regref_buffer.inputs.boldref = coreg_boldref
        config.loggers.workflow.debug('Reusing coregistration reference: %s', coreg_boldref)

    summary = pe.Node(
        FunctionalSummary(
            distortion_correction='None',  # Can override with connection
            registration=(
                'Precomputed'
                if boldref2anat_xform
                else 'FreeSurfer'
                if config.workflow.run_reconall
                else 'FSL'
            ),
            registration_dof=config.workflow.bold2anat_dof,
            registration_init=config.workflow.bold2anat_init,
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
        # TODO: Enable sdc report even if we find coregref
        sdc_correction=False,
        freesurfer=config.workflow.run_reconall,
        output_dir=config.execution.fmriprep_dir,
    )

    workflow.connect([
        (hmcref_buffer, outputnode, [
            ('boldref', 'hmc_boldref'),
            ('dummy_scans', 'dummy_scans'),
        ]),
        (regref_buffer, outputnode, [
            ('boldref', 'coreg_boldref'),
            ('boldmask', 'bold_mask'),
        ]),
        (hmc_buffer, outputnode, [
            ('hmc_xforms', 'motion_xfm'),
        ]),
        (inputnode, func_fit_reports_wf, [
            ('bold_file', 'inputnode.source_file'),
            ('t1w_preproc', 'inputnode.t1w_preproc'),
            # May not need all of these
            ('t1w_mask', 'inputnode.t1w_mask'),
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
        ]),
        (outputnode, func_fit_reports_wf, [
            ('coreg_boldref', 'inputnode.coreg_boldref'),
            ('bold_mask', 'inputnode.bold_mask'),
            ('boldref2anat_xfm', 'inputnode.boldref2anat_xfm'),
        ]),
        (summary, func_fit_reports_wf, [('out_report', 'inputnode.summary_report')]),
    ])  # fmt:skip

    # Stage 1: Generate motion correction boldref
    hmc_boldref_source_buffer = pe.Node(
        niu.IdentityInterface(fields=['in_file']),
        name='hmc_boldref_source_buffer',
    )
    if not hmc_boldref:
        config.loggers.workflow.info('Stage 1: Adding HMC boldref workflow')
        hmc_boldref_wf = init_raw_boldref_wf(
            name='hmc_boldref_wf',
            bold_file=bold_file,
        )
        hmc_boldref_wf.inputs.inputnode.dummy_scans = config.workflow.dummy_scans

        ds_hmc_boldref_wf = init_ds_boldref_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            desc='hmc',
            name='ds_hmc_boldref_wf',
        )
        ds_hmc_boldref_wf.inputs.inputnode.source_files = [bold_file]

        workflow.connect([
            (hmc_boldref_wf, hmcref_buffer, [
                ('outputnode.bold_file', 'bold_file'),
                ('outputnode.boldref', 'boldref'),
                ('outputnode.skip_vols', 'dummy_scans'),
            ]),
            (hmcref_buffer, ds_hmc_boldref_wf, [('boldref', 'inputnode.boldref')]),
            (hmc_boldref_wf, summary, [('outputnode.algo_dummy_scans', 'algo_dummy_scans')]),
            (hmc_boldref_wf, func_fit_reports_wf, [
                ('outputnode.validation_report', 'inputnode.validation_report'),
            ]),
            (ds_hmc_boldref_wf, hmc_boldref_source_buffer, [
                ('outputnode.boldref', 'in_file'),
            ]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found HMC boldref - skipping Stage 1')

        validation_and_dummies_wf = init_validation_and_dummies_wf(bold_file=bold_file)

        workflow.connect([
            (validation_and_dummies_wf, hmcref_buffer, [
                ('outputnode.bold_file', 'bold_file'),
                ('outputnode.skip_vols', 'dummy_scans'),
            ]),
            (validation_and_dummies_wf, func_fit_reports_wf, [
                ('outputnode.validation_report', 'inputnode.validation_report'),
            ]),
            (hmcref_buffer, hmc_boldref_source_buffer, [('boldref', 'in_file')]),
        ])  # fmt:skip

    # Stage 2: Estimate head motion
    if not hmc_xforms:
        config.loggers.workflow.info('Stage 2: Adding motion correction workflow')
        bold_hmc_wf = init_bold_hmc_wf(
            name='bold_hmc_wf', mem_gb=mem_gb['filesize'], omp_nthreads=omp_nthreads
        )

        ds_hmc_wf = init_ds_hmc_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
        )
        ds_hmc_wf.inputs.inputnode.source_files = [bold_file]

        workflow.connect([
            (hmcref_buffer, bold_hmc_wf, [
                ('boldref', 'inputnode.raw_ref_image'),
                ('bold_file', 'inputnode.bold_file'),
            ]),
            (bold_hmc_wf, ds_hmc_wf, [('outputnode.xforms', 'inputnode.xforms')]),
            (ds_hmc_wf, hmc_buffer, [('outputnode.xforms', 'hmc_xforms')]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found motion correction transforms - skipping Stage 2')

    # Stage 3: Create coregistration reference
    # Fieldmap correction only happens during fit if this stage is needed
    if not coreg_boldref:
        config.loggers.workflow.info('Stage 3: Adding coregistration boldref workflow')

        enhance_boldref_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

        ds_coreg_boldref_wf = init_ds_boldref_wf(
            bids_root=layout.root,
            output_dir=config.execution.fmriprep_dir,
            desc='coreg',
            name='ds_coreg_boldref_wf',
        )
        ds_boldmask_wf = init_ds_boldmask_wf(
            output_dir=config.execution.fmriprep_dir,
            desc='brain',
            name='ds_boldmask_wf',
        )
        ds_boldmask_wf.inputs.inputnode.source_files = [bold_file]

        workflow.connect([
            (hmcref_buffer, enhance_boldref_wf, [('boldref', 'inputnode.in_file')]),
            (hmc_boldref_source_buffer, ds_coreg_boldref_wf, [
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

    if not boldref2anat_xform:
        use_bbr = False

        # calculate BOLD registration to T1w
        bold_reg_wf = init_bold_reg_wf(
            bold2anat_dof=config.workflow.bold2anat_dof,
            bold2anat_init=config.workflow.bold2anat_init,
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
            (regref_buffer, bold_reg_wf, [('boldref', 'inputnode.ref_bold_brain')]),
            # Incomplete sources
            (regref_buffer, ds_boldreg_wf, [('boldref', 'inputnode.source_files')]),
            (bold_reg_wf, ds_boldreg_wf, [('outputnode.itk_bold_to_t1', 'inputnode.xform')]),
            (ds_boldreg_wf, outputnode, [('outputnode.xform', 'boldref2anat_xfm')]),
            (bold_reg_wf, summary, [('outputnode.fallback', 'fallback')]),
        ])
        # fmt:on
    else:
        outputnode.inputs.boldref2anat_xfm = boldref2anat_xform

    return workflow


def init_bold_native_wf(
    *,
    bold_series: list[str],
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
            from fmriprep.workflows.bold.fit import init_bold_native_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_bold_native_wf(bold_series=[str(bold_file)])

    Parameters
    ----------
    bold_series
        List of paths to NIfTI files.

    Inputs
    ------
    boldref
        BOLD reference file
    bold_mask
        Mask of BOLD reference file
    motion_xfm
        Affine transforms from each BOLD volume to ``hmc_boldref``, written
        as concatenated ITK affine transforms.

    Outputs
    -------
    bold_minimal
        BOLD series ready for further resampling.
        Only slice-timing correction (STC) may have been applied.
    bold_native
        BOLD series resampled into BOLD reference space. Slice-timing,
        head motion and susceptibility distortion correction (STC, HMC, SDC)
        will all be applied to each file.
    metadata
        Metadata dictionary of BOLD series
    motion_xfm
        Motion correction transforms for further correcting bold_minimal.

    See Also
    --------

    * :py:func:`~fmriprep.workflows.bold.stc.init_bold_stc_wf`
    * :py:func:`~fmriprep.workflows.bold.t2s.init_bold_t2s_wf`

    .. _optimal combination: https://tedana.readthedocs.io/en/stable/approach.html#optimal-combination

    """

    layout = config.execution.layout

    all_metadata = [layout.get_metadata(bold_file) for bold_file in bold_series]

    bold_file = bold_series[0]
    metadata = all_metadata[0]

    bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

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

    # Slice-timing correction
    if run_stc:
        bold_stc_wf = init_bold_stc_wf(metadata=metadata, mem_gb=mem_gb)
        workflow.connect([
            (inputnode, bold_stc_wf, [('dummy_scans', 'inputnode.skip_vols')]),
            (validate_bold, bold_stc_wf, [('out_file', 'inputnode.bold_file')]),
            (bold_stc_wf, boldbuffer, [('outputnode.stc_file', 'bold_file')]),
        ])  # fmt:skip
    else:
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
