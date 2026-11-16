"""Tests for the svht_denoise BOLD denoising workflow."""

import json
import re

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces.base import isdefined

from .... import config
from ....utils.testing import write_bold
from ...tests import mock_config
from ..denoise import init_bold_denoise_wf


def _write_run(bids_dir, name, phase_units=None, n_volumes=10, seed=0):
    """Write a BOLD file and its sidecar; with ``phase_units``, a part-phase companion too."""
    func = bids_dir / 'sub-01' / 'func'
    bold_file = write_bold(func / f'sub-01_{name}_part-mag_bold.nii.gz', n_volumes, seed)
    (func / f'sub-01_{name}_part-mag_bold.json').write_text(
        json.dumps({'RepetitionTime': 2.0, 'TaskName': 'rest'})
    )
    if phase_units is None:
        return bold_file, None

    phase_file = write_bold(func / f'sub-01_{name}_part-phase_bold.nii.gz', n_volumes, seed + 1)
    (func / f'sub-01_{name}_part-phase_bold.json').write_text(
        json.dumps({'RepetitionTime': 2.0, 'TaskName': 'rest', 'Units': phase_units})
    )
    return bold_file, phase_file


@pytest.fixture
def bids_dir(tmp_path):
    bids_dir = tmp_path / 'bids'
    bids_dir.mkdir()
    (bids_dir / 'dataset_description.json').write_text(
        json.dumps({'Name': 'test', 'BIDSVersion': '1.10.0'})
    )
    return bids_dir


def _build(bold_series, phase_series=None, **kwargs):
    return init_bold_denoise_wf(
        bold_series=bold_series,
        phase_series=phase_series,
        output_dir=str(config.execution.fmriprep_dir),
        **kwargs,
    )


def test_magnitude_only_graph(bids_dir):
    bold_file, _ = _write_run(bids_dir, 'task-rest')
    with mock_config(bids_dir=bids_dir):
        wf = _build([bold_file])

    names = set(wf.list_node_names())
    assert {'denoiser_echo1', 'ds_noise_echo1', 'ds_rank_echo1'} <= names
    assert 'ds_report_denoising_echo1' in names
    assert 'absolute_echo1' not in names
    assert 'validate_phase_echo1' not in names

    denoiser = wf.get_node('denoiser_echo1')
    assert not isdefined(denoiser.inputs.phase_file)
    assert denoiser.inputs.rank_image == 'rank.nii.gz'

    noise_meta = wf.get_node('ds_noise_echo1').interface._metadata
    assert noise_meta['UsedPhase'] is False
    assert 'PhaseUnits' not in noise_meta
    assert noise_meta['Units'] == 'arbitrary'
    assert wf.get_node('ds_rank_echo1').interface._metadata['Units'] == 'count'


@pytest.mark.parametrize(
    ('units', 'expected'),
    [('rad', 'radians'), ('arbitrary', None)],
)
def test_phase_graph(bids_dir, units, expected):
    bold_file, phase_file = _write_run(bids_dir, 'task-rest', phase_units=units)
    with mock_config(bids_dir=bids_dir):
        wf = _build([bold_file], [phase_file])

    assert wf.get_node('validate_phase_echo1').inputs.in_file == phase_file
    assert 'absolute_echo1' in wf.list_node_names()

    denoiser = wf.get_node('denoiser_echo1')
    if expected is None:
        assert not isdefined(denoiser.inputs.phase_units)
    else:
        assert denoiser.inputs.phase_units == expected

    noise_meta = wf.get_node('ds_noise_echo1').interface._metadata
    assert noise_meta['UsedPhase'] is True
    assert noise_meta['PhaseUnits'] == (expected or 'detected from the data range')


def test_multi_echo_graph(bids_dir):
    runs = [_write_run(bids_dir, f'task-rest_echo-{echo}', 'rad', seed=echo) for echo in (1, 2)]
    bold_series = [bold for bold, _ in runs]
    phase_series = [phase for _, phase in runs]
    with mock_config(bids_dir=bids_dir):
        wf = _build(bold_series, phase_series)

    for i, (bold_file, phase_file) in enumerate(runs, start=1):
        assert wf.get_node(f'validate_bold_echo{i}').inputs.in_file == bold_file
        assert wf.get_node(f'validate_phase_echo{i}').inputs.in_file == phase_file
        assert wf.get_node(f'ds_noise_echo{i}').inputs.source_file == [bold_file]

    # Each echo's denoised series lands at its own position of the merged list
    edges = wf._graph.get_edge_data(wf.get_node('absolute_echo2'), wf.get_node('merge_bold'))
    assert ('out_file', 'in2') in edges['connect']

    # Every echo's reportlet shows the same volume
    volumes = {wf.get_node(f'denoiser_echo{i}').inputs.report_volume for i in (1, 2)}
    assert len(volumes) == 1
    assert 0 <= volumes.pop() < 10


def test_build_errors(bids_dir):
    bold_file, phase_file = _write_run(bids_dir, 'task-rest', phase_units='rad')
    short_file, _ = _write_run(bids_dir, 'task-short', n_volumes=2)
    long_file, _ = _write_run(bids_dir, 'task-long', n_volumes=30)
    with mock_config(bids_dir=bids_dir):
        with pytest.raises(ValueError, match='phase files for a series of 1 echoes'):
            _build([bold_file], [phase_file, phase_file])

        with pytest.raises(ValueError, match='too few'):
            _build([short_file])

        with pytest.raises(ValueError, match='cannot apply vst'):
            _build([bold_file], [phase_file], denoise_params={'vst': True})

        # 3**3 = 27 patch voxels cannot hold 30 volumes
        with pytest.raises(ValueError, match='"extent" of 3'):
            _build([long_file], denoise_params={'extent': 3})

        # A 13-voxel cube does not fit the 12-slice image
        with pytest.raises(ValueError, match='too small for the 13-voxel cubic patch'):
            _build([bold_file], denoise_params={'shape': 'cube', 'extent': 13})


def test_build_rejects_mismatched_echoes(bids_dir):
    """Test that every echo is checked, so a later echo cannot fail inside svht_denoise."""
    echo1, _ = _write_run(bids_dir, 'task-rest_echo-1', n_volumes=10)
    echo2, _ = _write_run(bids_dir, 'task-rest_echo-2', n_volumes=12)
    with mock_config(bids_dir=bids_dir), pytest.raises(ValueError, match='differ in shape'):
        _build([echo1, echo2])


def _node_output(graph, name, output):
    node = next(node for node in graph.nodes() if node.name == name)
    return getattr(node.result.outputs, output)


@pytest.mark.usefixtures('fake_svht')
@pytest.mark.parametrize('use_phase', [False, True])
def test_run_writes_derivatives(bids_dir, use_phase):
    runs = [
        _write_run(bids_dir, f'task-rest_echo-{echo}', 'rad' if use_phase else None, seed=echo)
        for echo in (1, 2)
    ]
    bold_series = [bold for bold, _ in runs]
    phase_series = [phase for _, phase in runs] if use_phase else None
    with mock_config(bids_dir=bids_dir):
        out_dir = config.execution.fmriprep_dir
        wf = _build(bold_series, phase_series)
        wf.base_dir = str(config.execution.work_dir)
        graph = wf.run()

        func = out_dir / 'sub-01' / 'func'
        for echo in (1, 2):
            # part- is dropped; echo- is kept, since each echo has its own maps
            stem = f'sub-01_task-rest_echo-{echo}'
            noise_file = func / f'{stem}_desc-noise_boldmap.nii.gz'
            rank_file = func / f'{stem}_desc-rank_boldmap.nii.gz'
            assert noise_file.is_file()
            assert rank_file.is_file()
            assert (out_dir / 'sub-01' / 'figures' / f'{stem}_desc-denoising_bold.svg').is_file()

            meta = json.loads(noise_file.with_name(f'{stem}_desc-noise_boldmap.json').read_text())
            assert meta['SoftwareVersion'] == '0.0.test'
            assert meta['UsedPhase'] is use_phase
            assert len(meta['Sources']) == (2 if use_phase else 1)
            if use_phase:
                assert meta['PhaseUnits'] == 'radians'

        # Both echoes' figures show the same volume
        labels = {
            re.search(
                r'Before denoising: volume \d+',
                (
                    out_dir
                    / 'sub-01'
                    / 'figures'
                    / f'sub-01_task-rest_echo-{echo}_desc-denoising_bold.svg'
                ).read_text(),
            ).group()
            for echo in (1, 2)
        }
        assert len(labels) == 1

        noise_files = _node_output(graph, 'merge_noise', 'out')
        assert [f.split('/')[-1] for f in noise_files] == [
            f'sub-01_task-rest_echo-{echo}_desc-noise_boldmap.nii.gz' for echo in (1, 2)
        ]

        # The fake svht_denoise negates its output when given phase; the series handed on
        # must still be positive
        for denoised in _node_output(graph, 'merge_bold', 'out'):
            assert np.asanyarray(nb.load(denoised).dataobj).min() >= 0
