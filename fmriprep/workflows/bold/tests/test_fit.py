import logging
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ....utils.bids import collect_func_derivatives, dismiss_echo, extract_entities
from ....utils.testing import write_derivatives
from ...tests import mock_config
from ...tests.layouts import get_layout
from ..fit import get_sbrefs, init_bold_fit_wf, init_bold_native_wf


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
    generate_bids_skeleton(bids_dir, get_layout('no_session'))
    return bids_dir


def test_get_sbrefs_rejects_missing_echo_time(caplog):
    """SBRefs without EchoTime metadata should be dropped with a warning."""
    bold_files = [
        '/bids/sub-01/func/sub-01_task-rest_run-01_echo-1_bold.nii.gz',
        '/bids/sub-01/func/sub-01_task-rest_run-01_echo-2_bold.nii.gz',
    ]
    sbref_files = [
        '/bids/sub-01/func/sub-01_task-rest_run-01_echo-2_sbref.nii.gz',
        '/bids/sub-01/func/sub-01_task-rest_run-01_echo-1_sbref.nii.gz',
    ]

    class Layout:
        def get(self, **_entities):
            return list(sbref_files)

        def get_metadata(self, fname):
            return {'EchoTime': 0.01} if fname.endswith('echo-1_sbref.nii.gz') else {}

    logger = logging.getLogger('nipype.workflow')
    old_propagate = logger.propagate
    logger.propagate = True
    with caplog.at_level(logging.WARNING, logger='nipype.workflow'):
        found = get_sbrefs(bold_files, {}, Layout())
    logger.propagate = old_propagate

    assert found == ['/bids/sub-01/func/sub-01_task-rest_run-01_echo-1_sbref.nii.gz']
    assert 'Dropping SBRef without EchoTime metadata' in caplog.text


def test_get_sbrefs_preserves_single_missing_echo_time():
    """A single SBRef without EchoTime should still be returned."""
    bold_files = ['/bids/sub-01/func/sub-01_task-rest_run-01_bold.nii.gz']
    sbref_file = '/bids/sub-01/func/sub-01_task-rest_run-01_sbref.nii.gz'

    class Layout:
        def get(self, **_entities):
            return [sbref_file]

        def get_metadata(self, _fname):
            return {}

    found = get_sbrefs(bold_files, {}, Layout())

    assert found == [sbref_file]


DERIV_GROUPS = ['hmc_boldref', 'run_boldref', 'hmc', 'run2fmap']


@pytest.mark.parametrize('task', ['rest', 'nback'])
@pytest.mark.parametrize('fieldmap_id', ['phasediff', None])
@pytest.mark.parametrize('group', [None, *DERIV_GROUPS])
@pytest.mark.parametrize('mode', ['include', 'omit'])
def test_bold_fit_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    fieldmap_id: str | None,
    group: str | None,
    mode: str,
):
    """Test precomputed inputs one-by-one with a few configurations."""
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

    # Collect precomputed files from a derivatives dataset
    if group is None:
        include = [] if mode == 'include' else DERIV_GROUPS
    else:
        include = [group] if mode == 'include' else [g for g in DERIV_GROUPS if g != group]

    deriv_dir = write_derivatives(
        tmp_path / 'derivatives',
        include,
        task=task,
        run=1 if task == 'rest' else None,
        fmapid=fieldmap_id or 'auto00000',
    )
    for path in deriv_dir.rglob('*.nii.gz'):
        img.to_filename(path)
    for path in deriv_dir.rglob('*.txt'):
        np.savetxt(path, np.eye(4))

    # Mirrors how init_single_subject_wf selects entities before collecting derivatives
    entities = extract_entities(bold_series)
    entities = {k: v for k, v in entities.items() if k not in dismiss_echo(['part'])}
    precomputed = collect_func_derivatives([deriv_dir], entities, fieldmap_id=fieldmap_id)

    with mock_config(bids_dir=bids_root):
        config.workflow.bold2anat_init = 't1w'
        wf = init_bold_fit_wf(
            bold_series=bold_series,
            precomputed=precomputed,
            fieldmap_id=fieldmap_id,
            omp_nthreads=1,
        )

    # Precomputed derivatives skip the stages that would produce them
    assert (wf.get_node('hmc_boldref_wf') is None) == ('hmc_boldref' in include)
    assert (wf.get_node('bold_hmc_wf') is None) == ('hmc' in include)
    assert (wf.get_node('ds_run_boldref_wf') is None) == ('run_boldref' in include)
    assert (wf.get_node('fmapreg_wf') is None) == (fieldmap_id is None or 'run2fmap' in include)

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


@pytest.mark.parametrize('task', ['rest', 'nback'])
@pytest.mark.parametrize('fieldmap_id', ['phasediff', None])
@pytest.mark.parametrize('run_stc', [True, False])
def test_bold_native(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    fieldmap_id: str | None,
    run_stc: bool,
):
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
