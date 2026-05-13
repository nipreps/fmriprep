from pathlib import Path
from unittest.mock import patch

import bids
import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton
from sdcflows.fieldmaps import clear_registry
from sdcflows.utils.wrangler import find_estimators

from ... import config
from ..base import get_estimator, init_fmriprep_wf
from ..tests import mock_config
from .layouts import get_layout


@pytest.fixture(scope='module', autouse=True)
def _quiet_logger():
    import logging

    logger = logging.getLogger('nipype.workflow')
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(old_level)


@pytest.fixture(autouse=True)
def _reset_sdcflows_registry():
    yield
    clear_registry()


@pytest.fixture(scope='module')
def _layout_cache():
    """Cache layouts across tests to avoid regenerating BIDS directories."""
    return {}


@pytest.fixture(scope='module')
def bids_root(tmp_path_factory, _layout_cache):
    """Default fixture using the "no_session" layout."""
    return _make_bids_root(tmp_path_factory, 'no_session', _layout_cache)


def _make_bids_root(tmp_path_factory, layout_id: str, cache: dict):
    """Helper to create a BIDS directory from a layout spec."""
    if layout_id in cache:
        return cache[layout_id]

    base = tmp_path_factory.mktemp('base')
    bids_dir = base / layout_id

    layout = get_layout(layout_id)
    generate_bids_skeleton(bids_dir, layout)
    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))
    for bold_path in bids_dir.glob('sub-01/**/*.nii.gz'):
        img.to_filename(bold_path)
    cache[layout_id] = bids_dir
    return bids_dir


@pytest.fixture(scope='module')
def bids_root_factory(tmp_path_factory, _layout_cache):
    """Factory fixture — call with any layout to get a BIDS root."""

    def _factory(layout):
        return _make_bids_root(tmp_path_factory, layout, _layout_cache)

    return _factory


def _make_params(
    bold2anat_init: str = 'auto',
    dummy_scans: int | None = None,
    me_output_echos: bool = False,
    medial_surface_nan: bool = False,
    project_goodvoxels: bool = False,
    cifti_output: bool | str = False,
    run_msmsulc: bool = True,
    skull_strip_t1w: str = 'auto',
    use_syn_sdc: str | bool = False,
    freesurfer: bool = True,
    ignore: list[str] | None = None,
    force: list[str] | None = None,
    bids_filters: dict | None = None,
):
    if ignore is None:
        ignore = []
    if force is None:
        force = []
    if bids_filters is None:
        bids_filters = {}
    return (
        bold2anat_init,
        dummy_scans,
        me_output_echos,
        medial_surface_nan,
        project_goodvoxels,
        cifti_output,
        run_msmsulc,
        skull_strip_t1w,
        use_syn_sdc,
        freesurfer,
        ignore,
        force,
        bids_filters,
    )


@pytest.mark.parametrize('level', ['minimal', 'resampling', 'full'])
@pytest.mark.parametrize('anat_only', [False, True])
@pytest.mark.parametrize(
    (
        'bold2anat_init',
        'dummy_scans',
        'me_output_echos',
        'medial_surface_nan',
        'project_goodvoxels',
        'cifti_output',
        'run_msmsulc',
        'skull_strip_t1w',
        'use_syn_sdc',
        'freesurfer',
        'ignore',
        'force',
        'bids_filters',
    ),
    [
        _make_params(),
        _make_params(bold2anat_init='t1w'),
        _make_params(bold2anat_init='t2w'),
        _make_params(bold2anat_init='header'),
        _make_params(force=['bbr']),
        _make_params(force=['no-bbr']),
        _make_params(bold2anat_init='header', force=['bbr']),
        # Currently disabled
        # _make_params(bold2anat_init="header", force=['no-bbr']),
        _make_params(dummy_scans=2),
        _make_params(me_output_echos=True),
        _make_params(medial_surface_nan=True),
        _make_params(cifti_output='91k'),
        _make_params(cifti_output='91k', project_goodvoxels=True),
        _make_params(cifti_output='91k', project_goodvoxels=True, run_msmsulc=False),
        _make_params(cifti_output='91k', run_msmsulc=False),
        _make_params(skull_strip_t1w='force'),
        _make_params(skull_strip_t1w='skip'),
        _make_params(use_syn_sdc='warn', ignore=['fieldmaps'], force=['syn-sdc']),
        _make_params(freesurfer=False),
        _make_params(freesurfer=False, force=['bbr']),
        _make_params(freesurfer=False, force=['no-bbr']),
        # Currently unsupported:
        # _make_params(freesurfer=False, bold2anat_init="header"),
        # _make_params(freesurfer=False, bold2anat_init="header", force=['bbr']),
        # _make_params(freesurfer=False, bold2anat_init="header", force=['no-bbr']),
        # Regression test for gh-3154:
        _make_params(bids_filters={'sbref': {'suffix': 'sbref'}}),
    ],
)
def test_init_fmriprep_wf(
    bids_root: Path,
    level: str,
    anat_only: bool,
    bold2anat_init: str,
    dummy_scans: int | None,
    me_output_echos: bool,
    medial_surface_nan: bool,
    project_goodvoxels: bool,
    cifti_output: bool | str,
    run_msmsulc: bool,
    skull_strip_t1w: str,
    use_syn_sdc: str | bool,
    freesurfer: bool,
    ignore: list[str],
    force: list[str],
    bids_filters: dict,
):
    with mock_config(bids_dir=bids_root):
        config.workflow.level = level
        config.workflow.anat_only = anat_only
        config.workflow.bold2anat_init = bold2anat_init
        config.workflow.dummy_scans = dummy_scans
        config.execution.me_output_echos = me_output_echos
        config.workflow.medial_surface_nan = medial_surface_nan
        config.workflow.project_goodvoxels = project_goodvoxels
        config.workflow.run_msmsulc = run_msmsulc
        config.workflow.skull_strip_t1w = skull_strip_t1w
        config.workflow.cifti_output = cifti_output
        config.workflow.run_reconall = freesurfer
        config.workflow.ignore = ignore
        config.workflow.force = force
        config.workflow.use_syn_sdc = use_syn_sdc
        with patch.dict('fmriprep.config.execution.bids_filters', bids_filters):
            wf = init_fmriprep_wf()

    generate_expanded_graph(wf._create_flat_graph())


def test_init_fmriprep_wf_sanitize_fmaps(tmp_path):
    bids_dir = tmp_path / 'bids'

    spec = get_layout('no_session')
    spec['01']['func'][0]['metadata']['B0FieldSource'] = 'epi<<run1>>'
    spec['01']['fmap'][2]['metadata']['B0FieldIdentifier'] = 'epi<<run1>>'
    spec['01']['fmap'][3]['metadata']['B0FieldIdentifier'] = 'epi<<run1>>'
    del spec['01']['func'][4:]

    generate_bids_skeleton(bids_dir, spec)
    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))
    for img_path in bids_dir.glob('sub-01/*/*.nii.gz'):
        img.to_filename(img_path)

    with mock_config(bids_dir=bids_dir):
        wf = init_fmriprep_wf()
    generate_expanded_graph(wf._create_flat_graph())


def test_init_fmriprep_wf_sanitize_plus(tmp_path):
    bids_dir = tmp_path / 'bids'

    spec = get_layout('no_session')
    spec['01']['func'][0]['acquisition'] = 'mb4+pf68th'
    spec['01']['anat'][0]['acquisition'] = 'memprage+rms'
    spec['01']['anat'][1]['acquisition'] = 'memprage'
    spec['01']['fmap'][2]['acquisition'] = 'sbref+pf68th'
    spec['01']['fmap'][3]['acquisition'] = 'sbref+pf68th'
    del spec['01']['func'][4:]

    generate_bids_skeleton(bids_dir, spec)
    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))
    for img_path in bids_dir.glob('sub-01/*/*.nii.gz'):
        img.to_filename(img_path)

    with mock_config(bids_dir=bids_dir):
        wf = init_fmriprep_wf()
    generate_expanded_graph(wf._create_flat_graph())


def test_get_estimator_none(tmp_path):
    bids_dir = tmp_path / 'bids'

    # No IntendedFors/B0Fields
    generate_bids_skeleton(bids_dir, get_layout('no_session'))
    layout = bids.BIDSLayout(bids_dir)
    bold_files = sorted(
        layout.get(suffix='bold', task='rest', extension='.nii.gz', return_type='file')
    )

    assert get_estimator(layout, bold_files[0]) == ()
    assert get_estimator(layout, bold_files[1]) == ()


def test_get_estimator_b0field_and_intendedfor(tmp_path):
    bids_dir = tmp_path / 'bids'

    # Set B0FieldSource for run 1
    spec = get_layout('no_session')
    spec['01']['func'][0]['metadata']['B0FieldSource'] = 'epi'
    spec['01']['fmap'][2]['metadata']['B0FieldIdentifier'] = 'epi'
    spec['01']['fmap'][3]['metadata']['B0FieldIdentifier'] = 'epi'

    # Set IntendedFor for run 2
    spec['01']['fmap'][0]['metadata']['IntendedFor'] = 'func/sub-01_task-rest_run-2_bold.nii.gz'

    generate_bids_skeleton(bids_dir, spec)
    layout = bids.BIDSLayout(bids_dir)
    _ = find_estimators(layout=layout, subject='01')

    bold_files = sorted(
        layout.get(suffix='bold', task='rest', extension='.nii.gz', return_type='file')
    )

    assert get_estimator(layout, bold_files[0]) == ('epi',)
    # if B0FieldIdentifiers are found, IntendedFor will not be used
    assert get_estimator(layout, bold_files[1]) == ()


def test_get_estimator_intendedfor(tmp_path):
    bids_dir = tmp_path / 'bids'

    # Set B0FieldSource for run 1
    spec = get_layout('no_session')
    spec['01']['fmap'][0]['metadata']['IntendedFor'] = 'func/sub-01_task-rest_run-2_bold.nii.gz'

    generate_bids_skeleton(bids_dir, spec)
    layout = bids.BIDSLayout(bids_dir)
    _ = find_estimators(layout=layout, subject='01')

    bold_files = sorted(
        layout.get(suffix='bold', task='rest', extension='.nii.gz', return_type='file')
    )

    assert get_estimator(layout, bold_files[1]) == ('auto_00000',)


def test_get_estimator_overlapping_specs(tmp_path):
    bids_dir = tmp_path / 'bids'

    # Set B0FieldSource for both runs
    spec = get_layout('no_session')
    spec['01']['func'][0]['metadata']['B0FieldSource'] = 'epi'
    spec['01']['func'][1]['metadata']['B0FieldSource'] = 'epi'
    spec['01']['fmap'][2]['metadata']['B0FieldIdentifier'] = 'epi'
    spec['01']['fmap'][3]['metadata']['B0FieldIdentifier'] = 'epi'

    # Set IntendedFor for both runs
    spec['01']['fmap'][0]['metadata']['IntendedFor'] = [
        'func/sub-01_task-rest_run-1_bold.nii.gz',
        'func/sub-01_task-rest_run-2_bold.nii.gz',
    ]

    generate_bids_skeleton(bids_dir, spec)
    layout = bids.BIDSLayout(bids_dir)
    _ = find_estimators(layout=layout, subject='01')

    bold_files = sorted(
        layout.get(suffix='bold', task='rest', extension='.nii.gz', return_type='file')
    )

    # B0Fields take precedence
    assert get_estimator(layout, bold_files[0]) == ('epi',)
    assert get_estimator(layout, bold_files[1]) == ('epi',)


def test_get_estimator_multiple_b0fields(tmp_path):
    bids_dir = tmp_path / 'bids'

    # Set B0FieldSource for both runs
    spec = get_layout('no_session')
    spec['01']['func'][0]['metadata']['B0FieldSource'] = ('epi', 'phasediff')
    spec['01']['func'][1]['metadata']['B0FieldSource'] = 'epi'
    spec['01']['fmap'][0]['metadata']['B0FieldIdentifier'] = 'phasediff'
    spec['01']['fmap'][1]['metadata']['B0FieldIdentifier'] = 'phasediff'
    spec['01']['fmap'][2]['metadata']['B0FieldIdentifier'] = 'epi'
    spec['01']['fmap'][3]['metadata']['B0FieldIdentifier'] = 'epi'

    generate_bids_skeleton(bids_dir, spec)
    layout = bids.BIDSLayout(bids_dir)
    _ = find_estimators(layout=layout, subject='01')

    bold_files = sorted(
        layout.get(suffix='bold', task='rest', extension='.nii.gz', return_type='file')
    )

    # Always get an iterable; don't care if it's a list or tuple
    assert get_estimator(layout, bold_files[0]) == ['epi', 'phasediff']
    assert get_estimator(layout, bold_files[1]) == ('epi',)


@pytest.mark.parametrize('layout_id', ['no_session', 'single_session', 'homogeneous_sessions'])
@pytest.mark.parametrize('subject_anatomical_reference', ['first-lex', 'sessionwise', 'unbiased'])
def test_fmriprep_wf_builds(bids_root_factory, layout_id, subject_anatomical_reference):
    bids_dir = bids_root_factory(layout_id)
    with mock_config(bids_dir=bids_dir):
        config.workflow.subject_anatomical_reference = subject_anatomical_reference
        config._create_processing_groups()
        assert init_fmriprep_wf()


def test_fmriprep_wf_heterogeneous_sessions(bids_root_factory):
    """Test on a heterogeneous sessions layout, and test track_sessions behavior"""
    bids_dir = bids_root_factory('heterogeneous_sessions')
    with mock_config(bids_dir=bids_dir):
        config.workflow.bold2anat_init = 't1w'
        config.workflow.track_sessions = True
        procs = config._create_processing_groups()
        assert procs == [('01', ['anat', 'fmri'])]
        wf = init_fmriprep_wf()
        assert wf.get_node('sub_01_ses_anat-fmri_wf')

        config.workflow.track_sessions = False
        procs = config._create_processing_groups()
        assert procs == [('01', None)]
        wf = init_fmriprep_wf()
        assert wf.get_node('sub_01_wf')
