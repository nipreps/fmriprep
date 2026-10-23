from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.layouts import get_layout
from ..base import init_bold_apply_wf


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
    generate_bids_skeleton(bids_dir, get_layout('no_session'))
    return bids_dir


@pytest.mark.parametrize('task', ['rest', 'nback'])
@pytest.mark.parametrize('fieldmap_id', ['phasediff', None])
@pytest.mark.parametrize('freesurfer', [False, True])
@pytest.mark.parametrize('level', ['resampling', 'full'])
@pytest.mark.parametrize('bold2anat_init', ['t1w', 't2w'])
def test_bold_apply_wf(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    fieldmap_id: str | None,
    freesurfer: bool,
    level: str,
    bold2anat_init: str,
):
    """Test as many combinations of input configurations as possible."""
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
        wf = init_bold_apply_wf(
            bold_series=bold_series,
            fieldmap_id=fieldmap_id,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


@pytest.fixture
def rest_bold(bids_root):
    bold_file = bids_root / 'sub-01' / 'func' / 'sub-01_task-rest_run-1_bold.nii.gz'
    nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4)).to_filename(bold_file)
    return [str(bold_file)]


@pytest.mark.parametrize('medial_surface_nan', [False, True])
def test_bold_apply_wf_fs_spaces_workbench(bids_root, rest_bold, monkeypatch, medial_surface_nan):
    """Without a FreeSurfer subject, fsnative and fsaverage are sampled with Workbench."""
    with mock_config(bids_dir=bids_root):
        monkeypatch.setattr(config.workflow, 'level', 'full')
        monkeypatch.setattr(config.workflow, 'medial_surface_nan', medial_surface_nan)
        monkeypatch.setattr(
            config.execution, 'output_spaces', 'MNI152NLin2009cAsym fsnative fsaverage5 fsaverage4'
        )
        config.init_spaces()
        wf = init_bold_apply_wf(bold_series=rest_bold, freesurfer=False)

    nodes = wf.list_node_names()
    assert not any(node.startswith('bold_surf_wf.') for node in nodes)
    assert any(node.startswith('wb_vol_surf_wf.') for node in nodes)
    assert 'ds_bold_fsnative_wb' in nodes
    # fsaverage4 is TemplateFlow's den-3k
    for space, density in (('fsaverage5', '10k'), ('fsaverage4', '3k')):
        assert f'ds_bold_{space}_wb' in nodes
        assert any(node.startswith(f'wb_surf_native_fsaverage_{density}_wf.') for node in nodes)
        assert (f'medial_nans_{space}' in nodes) is medial_surface_nan

    # The subject's registration to fsaverage drives the resampling
    resample = wf.get_node('resample_surfaces_wf_fsaverage5')
    edges = wf._graph.get_edge_data(wf.get_node('inputnode'), resample)['connect']
    assert ('sphere_reg', 'inputnode.sphere_reg_fsaverage') in edges

    # Surfaces, as the anatomical workflow provides them
    for surf in ('white', 'pial', 'midthickness', 'space-fsaverage_desc-reg_sphere'):
        field = 'sphere_reg' if surf.endswith('sphere') else surf
        setattr(
            wf.inputs.inputnode, field, [f'sub-01_hemi-{hemi}_{surf}.surf.gii' for hemi in 'LR']
        )
    generate_expanded_graph(wf._create_flat_graph())


def test_bold_apply_wf_fs_spaces_freesurfer(bids_root, rest_bold, monkeypatch):
    """With a FreeSurfer subject, fsnative and fsaverage are sampled with mri_vol2surf."""
    with mock_config(bids_dir=bids_root):
        monkeypatch.setattr(config.workflow, 'level', 'full')
        monkeypatch.setattr(config.execution, 'output_spaces', 'fsnative fsaverage5')
        config.init_spaces()
        wf = init_bold_apply_wf(bold_series=rest_bold, freesurfer=True)

    nodes = wf.list_node_names()
    assert any(node.startswith('bold_surf_wf.') for node in nodes)
    assert not any(node.startswith('wb_vol_surf_wf.') for node in nodes)


def test_bold_apply_wf_fsaverage3_needs_subject(bids_root, rest_bold, monkeypatch):
    with mock_config(bids_dir=bids_root):
        monkeypatch.setattr(config.workflow, 'level', 'full')
        monkeypatch.setattr(config.execution, 'output_spaces', 'fsaverage3')
        config.init_spaces()
        with pytest.raises(RuntimeError, match='fsaverage3 requires a FreeSurfer subject'):
            init_bold_apply_wf(bold_series=rest_bold, freesurfer=False)
