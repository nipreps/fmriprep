from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..fit import init_pet_fit_wf, init_pet_native_wf


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
    base = tmp_path_factory.mktemp('petfit')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


def _make_params(
    have_petref: bool = True,
    have_hmc_xfms: bool = True,
    have_petref2anat_xfm: bool = True,
):
    return (
        have_petref,
        have_hmc_xfms,
        have_petref2anat_xfm,
    )


@pytest.mark.parametrize('task', ['rest'])
@pytest.mark.parametrize(
    (
        'have_petref',
        'have_hmc_xfms',
        'have_petref2anat_xfm',
    ),
    [
        (True, True, True, True, True),
        (False, False, False, False, False),
        _make_params(have_petref=False),
        _make_params(have_hmc_xfms=False),
        _make_params(have_petref2anat_xfm=False),
    ],
)
def test_pet_fit_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    have_petref: bool,
    have_hmc_xfms: bool,
    have_petref2anat_xfm: bool,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        pet_file = str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')

    # The workflow will attempt to read file headers
    img.to_filename(pet_file)

    dummy_nifti = str(tmp_path / 'dummy.nii')
    dummy_affine = str(tmp_path / 'dummy.txt')
    img.to_filename(dummy_nifti)
    np.savetxt(dummy_affine, np.eye(4))

    # Construct precomputed files
    precomputed = {'transforms': {}}
    if have_petref:
        precomputed['petref'] = dummy_nifti
    if have_hmc_xfms:
        precomputed['transforms']['hmc'] = dummy_affine
    if have_petref2anat_xfm:
        precomputed['transforms']['petref2anat'] = dummy_affine

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(
            pet_file=pet_file,
            precomputed=precomputed,
            omp_nthreads=1,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


@pytest.mark.parametrize('task', ['rest', 'nback'])
@pytest.mark.parametrize('run_stc', [True, False])
def test_pet_native_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    run_stc: bool,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        pet_file = str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')

    # The workflow will attempt to read file headers
    img.to_filename(pet_file)

    with mock_config(bids_dir=bids_root):
        config.workflow.ignore = ['slicetiming'] if not run_stc else []
        wf = init_pet_native_wf(
            pet_file=pet_file,
            omp_nthreads=1,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)
