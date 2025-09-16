from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..fit import init_bold_fit_wf, init_bold_native_wf


@pytest.fixture(scope='module', autouse=True)
def _quiet_logger():
    import logging

    logger = logging.getLogger('nipype.workflow')
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(old_level)


@pytest.fixture(scope='module')
def bids_root(tmp_path_factory):
    base = tmp_path_factory.mktemp('boldfit')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


def _make_params(
    have_hmcref: bool = True,
    have_coregref: bool = True,
    have_hmc_xfms: bool = True,
    have_boldref2fmap_xfm: bool = True,
    have_boldref2anat_xfm: bool = True,
):
    return (
        have_hmcref,
        have_coregref,
        have_hmc_xfms,
        have_boldref2anat_xfm,
        have_boldref2fmap_xfm,
    )


@pytest.mark.parametrize('task', ['rest', 'nback'])
@pytest.mark.parametrize('fieldmap_id', ['phasediff', None])
@pytest.mark.parametrize(
    (
        'have_hmcref',
        'have_coregref',
        'have_hmc_xfms',
        'have_boldref2fmap_xfm',
        'have_boldref2anat_xfm',
    ),
    [
        (True, True, True, True, True),
        (False, False, False, False, False),
        _make_params(have_hmcref=False),
        _make_params(have_hmc_xfms=False),
        _make_params(have_coregref=False),
        _make_params(have_coregref=False, have_boldref2fmap_xfm=False),
        _make_params(have_boldref2anat_xfm=False),
    ],
)
def test_bold_fit_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    fieldmap_id: str | None,
    have_hmcref: bool,
    have_coregref: bool,
    have_hmc_xfms: bool,
    have_boldref2fmap_xfm: bool,
    have_boldref2anat_xfm: bool,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        bold_series = [
            str(bids_root / 'sub-01' / 'func' / 'sub-01_task-rest_run-1_bold.nii.gz'),
        ]
        sbref = str(bids_root / 'sub-01' / 'func' / 'sub-01_task-rest_run-1_sbref.nii.gz')
    elif task == 'nback':
        bold_series = [
            str(bids_root / 'sub-01' / 'func' / f'sub-01_task-nback_echo-{i}_bold.nii.gz')
            for i in range(1, 4)
        ]
        sbref = str(bids_root / 'sub-01' / 'func' / 'sub-01_task-nback_echo-1_sbref.nii.gz')

    # The workflow will attempt to read file headers
    for path in bold_series:
        img.to_filename(path)
    # Single volume sbref; multi-volume tested in test_base
    img.slicer[:, :, :, 0].to_filename(sbref)

    dummy_nifti = str(tmp_path / 'dummy.nii')
    dummy_affine = str(tmp_path / 'dummy.txt')
    img.to_filename(dummy_nifti)
    np.savetxt(dummy_affine, np.eye(4))

    # Construct precomputed files
    precomputed = {'transforms': {}}
    if have_hmcref:
        precomputed['hmc_boldref'] = dummy_nifti
    if have_coregref:
        precomputed['coreg_boldref'] = dummy_nifti
    if have_hmc_xfms:
        precomputed['transforms']['hmc'] = dummy_affine
    if have_boldref2anat_xfm:
        precomputed['transforms']['boldref2anat'] = dummy_affine
    if have_boldref2fmap_xfm:
        precomputed['transforms']['boldref2fmap'] = dummy_affine

    with mock_config(bids_dir=bids_root):
        wf = init_bold_fit_wf(
            bold_series=bold_series,
            precomputed=precomputed,
            fieldmap_id=fieldmap_id,
            omp_nthreads=1,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


@pytest.mark.parametrize('task', ['rest', 'nback'])
@pytest.mark.parametrize('fieldmap_id', ['phasediff', None])
@pytest.mark.parametrize('run_stc', [True, False])
def test_bold_native_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    fieldmap_id: str | None,
    run_stc: bool,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        bold_series = [
            str(bids_root / 'sub-01' / 'func' / 'sub-01_task-rest_run-1_bold.nii.gz'),
        ]
    elif task == 'nback':
        bold_series = [
            str(bids_root / 'sub-01' / 'func' / f'sub-01_task-nback_echo-{i}_bold.nii.gz')
            for i in range(1, 4)
        ]

    # The workflow will attempt to read file headers
    for path in bold_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        config.workflow.ignore = ['slicetiming'] if not run_stc else []
        wf = init_bold_native_wf(
            bold_series=bold_series,
            fieldmap_id=fieldmap_id,
            omp_nthreads=1,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)
