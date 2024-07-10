from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_bold_wf


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
    base = tmp_path_factory.mktemp('boldbase')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


@pytest.mark.parametrize('task', ['rest', 'nback'])
@pytest.mark.parametrize('fieldmap_id', ['phasediff', None])
@pytest.mark.parametrize('freesurfer', [False, True])
@pytest.mark.parametrize('level', ['minimal', 'resampling', 'full'])
@pytest.mark.parametrize('bold2anat_init', ['t1w', 't2w'])
def test_bold_wf(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    fieldmap_id: str | None,
    freesurfer: bool,
    level: str,
    bold2anat_init: str,
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
    img.to_filename(sbref)

    with mock_config(bids_dir=bids_root):
        config.workflow.bold2anat_init = bold2anat_init
        config.workflow.level = level
        config.workflow.run_reconall = freesurfer
        wf = init_bold_wf(
            bold_series=bold_series,
            fieldmap_id=fieldmap_id,
            precomputed={},
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)
