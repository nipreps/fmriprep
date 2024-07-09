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
import os
import typing as ty

import bids
import nibabel as nb
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from niworkflows.interfaces.utility import KeySelect
from sdcflows.workflows.apply.correction import init_unwarp_wf
from sdcflows.workflows.apply.registration import init_coeff2epi_wf

from ... import config
from ...interfaces.reports import FunctionalSummary
from ...interfaces.resampling import (
    DistortionParameters,
    ReconstructFieldmap,
    ResampleSeries,
)
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
from .reference import init_raw_boldref_wf
from .registration import init_bold_reg_wf
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf


def get_sbrefs(
    bold_files: list[str],
    entity_overrides: dict[str, ty.Any],
    layout: bids.BIDSLayout,
) -> list[str]:
    """Find single-band reference(s) associated with BOLD file(s)

    Parameters
    ----------
    bold_files
        List of absolute paths to BOLD files
    entity_overrides
        Query parameters to override defaults
    layout
        :class:`~bids.layout.BIDSLayout` to query

    Returns
    -------
    sbref_files
        List of absolute paths to sbref files associated with input BOLD files,
        sorted by EchoTime
    """
    entities = extract_entities(bold_files)
    entities.pop('echo', None)
    entities.update(suffix='sbref', extension=['.nii', '.nii.gz'])
    entities.update(entity_overrides)

    return sorted(
        layout.get(return_type='file', **entities),
        key=lambda fname: layout.get_metadata(fname).get('EchoTime'),
    )


def init_bold_fit_wf(
    *,
    bold_series: list[str],
    precomputed: dict = None,
    fieldmap_id: str | None = None,
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
        List of paths to NIfTI files, sorted by echo time.
    precomputed
        Dictionary containing precomputed derivatives to reuse, if possible.
    fieldmap_id
        ID of the fieldmap to use to correct this BOLD series. If :obj:`None`,
        no correction will be applied.

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
    fmap_id
        Unique identifiers to select fieldmap files
    fmap
        List of estimated fieldmaps (collated with fmap_id)
    fmap_ref
        List of fieldmap reference files (collated with fmap_id)
    fmap_coeff
        List of lists of spline coefficient files (collated with fmap_id)
    fmap_mask
        List of fieldmap masks (collated with fmap_id)
    sdc_method
        List of fieldmap correction method names (collated with fmap_id)

    Outputs
    -------
    hmc_boldref
        BOLD reference image used for head motion correction.
        Minimally processed to ensure consistent contrast with BOLD series.
    coreg_boldref
        BOLD reference image used for coregistration. Contrast-enhanced
        and fieldmap-corrected for greater anatomical fidelity, and aligned
        with ``hmc_boldref``.
    bold_mask
        Mask of ``coreg_boldref``.
    motion_xfm
        Affine transforms from each BOLD volume to ``hmc_boldref``, written
        as concatenated ITK affine transforms.
    boldref2anat_xfm
        Affine transform mapping from BOLD reference space to the anatomical
        space.
    boldref2fmap_xfm
        Affine transform mapping from BOLD reference space to the fieldmap
        space, if applicable.
    movpar_file
        MCFLIRT motion parameters, normalized to SPM format (X, Y, Z, Rx, Ry, Rz)
    rmsd_file
        Root mean squared deviation as measured by ``fsl_motion_outliers`` [Jenkinson2002]_.
    dummy_scans
        The number of dummy scans declared or detected at the beginning of the series.

    See Also
    --------

    * :py:func:`~fmriprep.workflows.bold.reference.init_raw_boldref_wf`
    * :py:func:`~fmriprep.workflows.bold.hmc.init_bold_hmc_wf`
    * :py:func:`~niworkflows.func.utils.init_enhance_and_skullstrip_bold_wf`
    * :py:func:`~sdcflows.workflows.apply.registration.init_coeff2epi_wf`
    * :py:func:`~sdcflows.workflows.apply.correction.init_unwarp_wf`
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

    # Fitting operates on the shortest echo
    # This could become more complicated in the future
    bold_file = bold_series[0]

    # Collect sbref files, sorted by EchoTime
    sbref_files = get_sbrefs(
        bold_series,
        entity_overrides=bids_filters.get('sbref', {}),
        layout=layout,
    )

    basename = os.path.basename(bold_file)
    sbref_msg = f'No single-band-reference found for {basename}.'
    if sbref_files and 'sbref' in config.workflow.ignore:
        sbref_msg = f'Single-band reference file(s) found for {basename} and ignored.'
        sbref_files = []
    elif sbref_files:
        sbref_msg = 'Using single-band reference file(s) {}.'.format(
            ','.join([os.path.basename(sbf) for sbf in sbref_files])
        )
    config.loggers.workflow.info(sbref_msg)

    # Get metadata from BOLD file(s)
    entities = extract_entities(bold_series)
    metadata = layout.get_metadata(bold_file)
    orientation = ''.join(nb.aff2axcodes(nb.load(bold_file).affine))

    bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

    # Boolean used to update workflow self-descriptions
    multiecho = len(bold_series) > 1

    have_hmcref = 'hmc_boldref' in precomputed
    have_coregref = 'coreg_boldref' in precomputed
    # Can contain
    #  1) boldref2fmap
    #  2) boldref2anat
    #  3) hmc
    transforms = precomputed.get('transforms', {})
    hmc_xforms = transforms.get('hmc')
    boldref2fmap_xform = transforms.get('boldref2fmap')
    boldref2anat_xform = transforms.get('boldref2anat')

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                # Fieldmap registration
                'fmap',
                'fmap_ref',
                'fmap_coeff',
                'fmap_mask',
                'fmap_id',
                'sdc_method',
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
                'boldref2fmap_xfm',
                'movpar_file',
                'rmsd_file',
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
    fmapref_buffer = pe.Node(niu.Function(function=_select_ref), name='fmapref_buffer')
    hmc_buffer = pe.Node(
        niu.IdentityInterface(fields=['hmc_xforms', 'movpar_file', 'rmsd_file']), name='hmc_buffer'
    )
    fmapreg_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref2fmap_xfm']), name='fmapreg_buffer'
    )
    regref_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'boldmask']), name='regref_buffer'
    )

    summary = pe.Node(
        FunctionalSummary(
            distortion_correction='None',  # Can override with connection
            registration=('FSL', 'FreeSurfer')[config.workflow.run_reconall],
            registration_dof=config.workflow.bold2anat_dof,
            registration_init=config.workflow.bold2anat_init,
            pe_direction=metadata.get('PhaseEncodingDirection'),
            echo_idx=entities.get('echo', []),
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
        sdc_correction=not (have_coregref or fieldmap_id is None),
        freesurfer=config.workflow.run_reconall,
        output_dir=config.execution.fmriprep_dir,
    )

    # fmt:off
    workflow.connect([
        (hmcref_buffer, outputnode, [
            ('boldref', 'hmc_boldref'),
            ('dummy_scans', 'dummy_scans'),
        ]),
        (regref_buffer, outputnode, [
            ('boldref', 'coreg_boldref'),
            ('boldmask', 'bold_mask'),
        ]),
        (fmapreg_buffer, outputnode, [('boldref2fmap_xfm', 'boldref2fmap_xfm')]),
        (hmc_buffer, outputnode, [
            ('hmc_xforms', 'motion_xfm'),
            ('movpar_file', 'movpar_file'),
            ('rmsd_file', 'rmsd_file'),
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
    ])
    # fmt:on

    # Stage 1: Generate motion correction boldref
    hmc_boldref_source_buffer = pe.Node(
        niu.IdentityInterface(fields=['in_file']),
        name='hmc_boldref_source_buffer',
    )
    if not have_hmcref:
        config.loggers.workflow.info('Stage 1: Adding HMC boldref workflow')
        hmc_boldref_wf = init_raw_boldref_wf(
            name='hmc_boldref_wf',
            bold_file=bold_file,
            multiecho=multiecho,
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

        validate_bold = pe.Node(ValidateImage(), name='validate_bold')
        validate_bold.inputs.in_file = bold_file

        hmcref_buffer.inputs.boldref = precomputed['hmc_boldref']

        workflow.connect([
            (validate_bold, hmcref_buffer, [('out_file', 'bold_file')]),
            (validate_bold, func_fit_reports_wf, [('out_report', 'inputnode.validation_report')]),
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
            (bold_hmc_wf, hmc_buffer, [
                ('outputnode.movpar_file', 'movpar_file'),
                ('outputnode.rmsd_file', 'rmsd_file'),
            ]),
            (ds_hmc_wf, hmc_buffer, [('outputnode.xforms', 'hmc_xforms')]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info('Found motion correction transforms - skipping Stage 2')
        hmc_buffer.inputs.hmc_xforms = hmc_xforms

    # Stage 3: Create coregistration reference
    # Fieldmap correction only happens during fit if this stage is needed
    if not have_coregref:
        config.loggers.workflow.info('Stage 3: Adding coregistration boldref workflow')

        # Select initial boldref, enhance contrast, and generate mask
        fmapref_buffer.inputs.sbref_files = sbref_files
        if sbref_files and nb.load(sbref_files[0]).ndim > 3:
            raw_sbref_wf = init_raw_boldref_wf(
                name='raw_sbref_wf',
                bold_file=sbref_files[0],
                multiecho=len(sbref_files) > 1,
            )
            workflow.connect(raw_sbref_wf, 'outputnode.boldref', fmapref_buffer, 'sbref_files')

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
            (hmcref_buffer, fmapref_buffer, [('boldref', 'boldref_files')]),
            (fmapref_buffer, enhance_boldref_wf, [('out', 'inputnode.in_file')]),
            (hmc_boldref_source_buffer, ds_coreg_boldref_wf, [
                ('in_file', 'inputnode.source_files'),
            ]),
            (ds_coreg_boldref_wf, regref_buffer, [('outputnode.boldref', 'boldref')]),
            (ds_boldmask_wf, regref_buffer, [('outputnode.boldmask', 'boldmask')]),
            (fmapref_buffer, func_fit_reports_wf, [('out', 'inputnode.sdc_boldref')]),
        ])  # fmt:skip

        if fieldmap_id:
            fmap_select = pe.Node(
                KeySelect(
                    fields=['fmap_ref', 'fmap_coeff', 'fmap_mask', 'sdc_method'],
                    key=fieldmap_id,
                ),
                name='fmap_select',
                run_without_submitting=True,
            )

            if not boldref2fmap_xform:
                fmapreg_wf = init_coeff2epi_wf(
                    debug='fieldmaps' in config.execution.debug,
                    omp_nthreads=config.nipype.omp_nthreads,
                    sloppy=config.execution.sloppy,
                    name='fmapreg_wf',
                )

                itk_mat2txt = pe.Node(ConcatenateXFMs(out_fmt='itk'), name='itk_mat2txt')

                ds_fmapreg_wf = init_ds_registration_wf(
                    bids_root=layout.root,
                    output_dir=config.execution.fmriprep_dir,
                    source='boldref',
                    dest=fieldmap_id.replace('_', ''),
                    name='ds_fmapreg_wf',
                )
                ds_fmapreg_wf.inputs.inputnode.source_files = [bold_file]

                # fmt:off
                workflow.connect([
                    (enhance_boldref_wf, fmapreg_wf, [
                        ('outputnode.bias_corrected_file', 'inputnode.target_ref'),
                        ('outputnode.mask_file', 'inputnode.target_mask'),
                    ]),
                    (fmap_select, fmapreg_wf, [
                        ('fmap_ref', 'inputnode.fmap_ref'),
                        ('fmap_mask', 'inputnode.fmap_mask'),
                    ]),
                    (fmapreg_wf, itk_mat2txt, [('outputnode.target2fmap_xfm', 'in_xfms')]),
                    (itk_mat2txt, ds_fmapreg_wf, [('out_xfm', 'inputnode.xform')]),
                    (ds_fmapreg_wf, fmapreg_buffer, [('outputnode.xform', 'boldref2fmap_xfm')]),
                ])
                # fmt:on
            else:
                fmapreg_buffer.inputs.boldref2fmap_xfm = boldref2fmap_xform

            unwarp_wf = init_unwarp_wf(
                free_mem=config.environment.free_mem,
                debug='fieldmaps' in config.execution.debug,
                omp_nthreads=config.nipype.omp_nthreads,
            )
            unwarp_wf.inputs.inputnode.metadata = layout.get_metadata(bold_file)

            # fmt:off
            workflow.connect([
                (inputnode, fmap_select, [
                    ('fmap_ref', 'fmap_ref'),
                    ('fmap_coeff', 'fmap_coeff'),
                    ('fmap_mask', 'fmap_mask'),
                    ('sdc_method', 'sdc_method'),
                    ('fmap_id', 'keys'),
                ]),
                (fmap_select, unwarp_wf, [
                    ('fmap_coeff', 'inputnode.fmap_coeff'),
                ]),
                (fmapreg_buffer, unwarp_wf, [
                    # This looks backwards, but unwarp_wf describes transforms in
                    # terms of points while we (and init_coeff2epi_wf) describe them
                    # in terms of images. Mapping fieldmap coordinates into boldref
                    # coordinates maps the boldref image onto the fieldmap image.
                    ('boldref2fmap_xfm', 'inputnode.fmap2data_xfm'),
                ]),
                (enhance_boldref_wf, unwarp_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.distorted'),
                ]),
                (unwarp_wf, ds_coreg_boldref_wf, [
                    ('outputnode.corrected', 'inputnode.boldref'),
                ]),
                (unwarp_wf, ds_boldmask_wf, [
                    ('outputnode.corrected_mask', 'inputnode.boldmask'),
                ]),
                (fmap_select, func_fit_reports_wf, [('fmap_ref', 'inputnode.fmap_ref')]),
                (fmap_select, summary, [('sdc_method', 'distortion_correction')]),
                (fmapreg_buffer, func_fit_reports_wf, [
                    ('boldref2fmap_xfm', 'inputnode.boldref2fmap_xfm'),
                ]),
                (unwarp_wf, func_fit_reports_wf, [('outputnode.fieldmap', 'inputnode.fieldmap')]),
            ])
            # fmt:on
        else:
            # fmt:off
            workflow.connect([
                (enhance_boldref_wf, ds_coreg_boldref_wf, [
                    ('outputnode.bias_corrected_file', 'inputnode.boldref'),
                ]),
                (enhance_boldref_wf, ds_boldmask_wf, [
                    ('outputnode.mask_file', 'inputnode.boldmask'),
                ]),
            ])
            # fmt:on
    else:
        config.loggers.workflow.info('Found coregistration reference - skipping Stage 3')
        regref_buffer.inputs.boldref = precomputed['coreg_boldref']

    if not boldref2anat_xform:
        # calculate BOLD registration to T1w
        bold_reg_wf = init_bold_reg_wf(
            bold2anat_dof=config.workflow.bold2anat_dof,
            bold2anat_init=config.workflow.bold2anat_init,
            use_bbr=config.workflow.use_bbr,
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
    fieldmap_id: str | None = None,
    omp_nthreads: int = 1,
    name: str = 'bold_native_wf',
) -> pe.Workflow:
    r"""
    Minimal resampling workflow.

    This workflow performs slice-timing correction, and resamples to boldref space
    with head motion and susceptibility distortion correction. It also handles
    multi-echo processing and selects the transforms needed to perform further
    resampling.

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
    fieldmap_id
        ID of the fieldmap to use to correct this BOLD series. If :obj:`None`,
        no correction will be applied.

    Inputs
    ------
    boldref
        BOLD reference file
    bold_mask
        Mask of BOLD reference file
    motion_xfm
        Affine transforms from each BOLD volume to ``hmc_boldref``, written
        as concatenated ITK affine transforms.
    boldref2fmap_xfm
        Affine transform mapping from BOLD reference space to the fieldmap
        space, if applicable.
    fmap_id
        Unique identifiers to select fieldmap files
    fmap_ref
        List of fieldmap reference files (collated with fmap_id)
    fmap_coeff
        List of lists of spline coefficient files (collated with fmap_id)

    Outputs
    -------
    bold_minimal
        BOLD series ready for further resampling. For single-echo data, only
        slice-timing correction (STC) may have been applied. For multi-echo
        data, this is identical to bold_native.
    bold_native
        BOLD series resampled into BOLD reference space. Slice-timing,
        head motion and susceptibility distortion correction (STC, HMC, SDC)
        will all be applied to each file. For multi-echo data, the echos
        are combined to form an `optimal combination`_.
    metadata
        Metadata dictionary of BOLD series with the shortest echo
    motion_xfm
        Motion correction transforms for further correcting bold_minimal.
        For multi-echo data, motion correction has already been applied, so
        this will be undefined.
    bold_echos
        The individual, corrected echos, suitable for use in Tedana.
        (Multi-echo only.)
    t2star_map
        The T2\* map estimated by Tedana when calculating the optimal combination.
        (Multi-echo only.)

    See Also
    --------

    * :py:func:`~fmriprep.workflows.bold.stc.init_bold_stc_wf`
    * :py:func:`~fmriprep.workflows.bold.t2s.init_bold_t2s_wf`

    .. _optimal combination: https://tedana.readthedocs.io/en/stable/approach.html#optimal-combination

    """

    layout = config.execution.layout

    # Shortest echo first
    all_metadata = [layout.get_metadata(bold_file) for bold_file in bold_series]
    echo_times = [md.get('EchoTime') for md in all_metadata]
    multiecho = len(bold_series) > 1

    bold_file = bold_series[0]
    metadata = all_metadata[0]

    bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

    if multiecho:
        shapes = [nb.load(echo).shape for echo in bold_series]
        if len(set(shapes)) != 1:
            diagnostic = '\n'.join(
                f'{os.path.basename(echo)}: {shape}'
                for echo, shape in zip(bold_series, shapes, strict=False)
            )
            raise RuntimeError(f'Multi-echo images found with mismatching shapes\n{diagnostic}')
        if len(shapes) == 2:
            raise RuntimeError(
                'Multi-echo processing requires at least three different echos (found two).'
            )

    run_stc = bool(metadata.get('SliceTiming')) and 'slicetiming' not in config.workflow.ignore

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # BOLD fit
                'boldref',
                'bold_mask',
                'motion_xfm',
                'boldref2fmap_xfm',
                'dummy_scans',
                # Fieldmap fit
                'fmap_ref',
                'fmap_coeff',
                'fmap_id',
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
                # Multiecho outputs
                'bold_echos',  # Individual corrected echos
                't2star_map',  # T2* map
            ],  # fmt:skip
        ),
        name='outputnode',
    )
    outputnode.inputs.metadata = metadata

    boldbuffer = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'ro_time', 'pe_dir']), name='boldbuffer'
    )

    # Track echo index - this allows us to treat multi- and single-echo workflows
    # almost identically
    echo_index = pe.Node(niu.IdentityInterface(fields=['echoidx']), name='echo_index')
    if multiecho:
        echo_index.iterables = [('echoidx', range(len(bold_series)))]
    else:
        echo_index.inputs.echoidx = 0

    # BOLD source: track original BOLD file(s)
    bold_source = pe.Node(niu.Select(inlist=bold_series), name='bold_source')
    validate_bold = pe.Node(ValidateImage(), name='validate_bold')
    workflow.connect([
        (echo_index, bold_source, [('echoidx', 'index')]),
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

    # Prepare fieldmap metadata
    if fieldmap_id:
        fmap_select = pe.Node(
            KeySelect(fields=['fmap_ref', 'fmap_coeff'], key=fieldmap_id),
            name='fmap_select',
            run_without_submitting=True,
        )

        distortion_params = pe.Node(
            DistortionParameters(metadata=metadata, in_file=bold_file),
            name='distortion_params',
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, fmap_select, [
                ('fmap_ref', 'fmap_ref'),
                ('fmap_coeff', 'fmap_coeff'),
                ('fmap_id', 'keys'),
            ]),
            (distortion_params, boldbuffer, [
                ('readout_time', 'ro_time'),
                ('pe_direction', 'pe_dir'),
            ]),
        ])  # fmt:skip

    # Resample to boldref
    boldref_bold = pe.Node(
        ResampleSeries(jacobian='fmap-jacobian' not in config.workflow.ignore),
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
            ('ro_time', 'ro_time'),
            ('pe_dir', 'pe_dir'),
        ]),
    ])  # fmt:skip

    if fieldmap_id:
        boldref_fmap = pe.Node(ReconstructFieldmap(inverse=[True]), name='boldref_fmap', mem_gb=1)
        workflow.connect([
            (inputnode, boldref_fmap, [
                ('boldref', 'target_ref_file'),
                ('boldref2fmap_xfm', 'transforms'),
            ]),
            (fmap_select, boldref_fmap, [
                ('fmap_coeff', 'in_coeffs'),
                ('fmap_ref', 'fmap_ref_file'),
            ]),
            (boldref_fmap, boldref_bold, [('out_file', 'fieldmap')]),
        ])  # fmt:skip

    if multiecho:
        join_echos = pe.JoinNode(
            niu.IdentityInterface(fields=['bold_files']),
            joinsource='echo_index',
            joinfield=['bold_files'],
            name='join_echos',
            run_without_submitting=True,
        )

        # create optimal combination, adaptive T2* map
        bold_t2s_wf = init_bold_t2s_wf(
            echo_times=echo_times,
            mem_gb=mem_gb['filesize'],
            omp_nthreads=config.nipype.omp_nthreads,
            name='bold_t2smap_wf',
        )

        # Do NOT set motion_xfm on outputnode
        # This prevents downstream resamplers from double-dipping
        workflow.connect([
            (inputnode, bold_t2s_wf, [('bold_mask', 'inputnode.bold_mask')]),
            (boldref_bold, join_echos, [('out_file', 'bold_files')]),
            (join_echos, bold_t2s_wf, [('bold_files', 'inputnode.bold_file')]),
            (join_echos, outputnode, [('bold_files', 'bold_echos')]),
            (bold_t2s_wf, outputnode, [
                ('outputnode.bold', 'bold_minimal'),
                ('outputnode.bold', 'bold_native'),
                ('outputnode.t2star_map', 't2star_map'),
            ]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (inputnode, outputnode, [('motion_xfm', 'motion_xfm')]),
            (boldbuffer, outputnode, [('bold_file', 'bold_minimal')]),
            (boldref_bold, outputnode, [('out_file', 'bold_native')]),
        ])  # fmt:skip

    return workflow


def _select_ref(sbref_files, boldref_files):
    """Select first sbref or boldref file, preferring sbref if available"""
    from niworkflows.utils.connections import listify

    refs = sbref_files or boldref_files
    return listify(refs)[0]
