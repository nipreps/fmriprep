from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_pet_wf


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
    base = tmp_path_factory.mktemp('petbase')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


@pytest.mark.parametrize('task', ['rest'])
@pytest.mark.parametrize('level', ['minimal', 'resampling', 'full'])
@pytest.mark.parametrize('pet2anat_init', ['t1w', 't2w'])
def test_pet_wf(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    level: str,
    pet2anat_init: str,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        pet_series = [
            str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz'),
        ]

    # The workflow will attempt to read file headers
    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        config.workflow.pet2anat_init = pet2anat_init
        config.workflow.level = level
        config.workflow.run_reconall = freesurfer
        wf = init_pet_wf(
            pet_series=pet_series,
            precomputed={},
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)
