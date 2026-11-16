"""Tests for collecting the phase companions of BOLD series."""

import json
import logging

import nibabel as nb
import numpy as np
import pytest
from bids.layout import BIDSLayout

from fmriprep.utils.bids import collect_bold_phase

SHAPE = (4, 4, 3, 5)


def _write_nifti(path, shape=SHAPE, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4) if affine is None else affine
    nb.Nifti1Image(np.zeros(shape, dtype=np.float32), affine).to_filename(str(path))
    path.with_name(path.name.replace('.nii.gz', '.json')).write_text(
        json.dumps({'RepetitionTime': 2.0, 'TaskName': 'rest'})
    )
    return str(path)


def _func_file(bids_dir, name):
    return bids_dir / 'sub-01' / 'func' / f'sub-01_{name}.nii.gz'


def _layout(bids_dir):
    (bids_dir / 'dataset_description.json').write_text(
        json.dumps({'Name': 'test', 'BIDSVersion': '1.10.0'})
    )
    return BIDSLayout(str(bids_dir), validate=False)


@pytest.fixture(autouse=True)
def _workflow_warnings():
    """Let the workflow logger's warnings reach caplog, whatever other tests set."""
    logger = logging.getLogger('nipype.workflow')
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.WARNING)
    logger.propagate = True
    yield
    logger.setLevel(old_level)
    logger.propagate = old_propagate


@pytest.fixture
def bids_dir(tmp_path):
    return tmp_path / 'bids'


def test_single_echo_with_phase(bids_dir):
    mag = _write_nifti(_func_file(bids_dir, 'task-rest_part-mag_bold'))
    phase = _write_nifti(_func_file(bids_dir, 'task-rest_part-phase_bold'))
    assert collect_bold_phase(_layout(bids_dir), [mag]) == [phase]


def test_single_echo_without_phase(bids_dir):
    mag = _write_nifti(_func_file(bids_dir, 'task-rest_bold'))
    assert collect_bold_phase(_layout(bids_dir), [mag]) is None


def test_unlabeled_magnitude_pairs_with_phase(bids_dir):
    """Test that a magnitude without a part entity still finds its part-phase companion."""
    mag = _write_nifti(_func_file(bids_dir, 'task-rest_bold'))
    phase = _write_nifti(_func_file(bids_dir, 'task-rest_part-phase_bold'))
    assert collect_bold_phase(_layout(bids_dir), mag) == [phase]


def test_phase_pairs_by_run_and_rejects_extra_entities(bids_dir):
    """Test that each run gets its own phase, and a phase with an extra entity is ignored."""
    mags = [
        _write_nifti(_func_file(bids_dir, f'task-rest_run-{run}_part-mag_bold')) for run in (1, 2)
    ]
    phases = [
        _write_nifti(_func_file(bids_dir, f'task-rest_run-{run}_part-phase_bold'))
        for run in (1, 2)
    ]
    # Same run, but an acquisition the magnitude does not have
    _write_nifti(_func_file(bids_dir, 'task-rest_acq-other_run-1_part-phase_bold'))

    layout = _layout(bids_dir)
    assert collect_bold_phase(layout, [mags[0]]) == [phases[0]]
    assert collect_bold_phase(layout, [mags[1]]) == [phases[1]]


def test_multi_echo_complete(bids_dir):
    mags = [
        _write_nifti(_func_file(bids_dir, f'task-rest_echo-{echo}_part-mag_bold'))
        for echo in (1, 2, 3)
    ]
    phases = [
        _write_nifti(_func_file(bids_dir, f'task-rest_echo-{echo}_part-phase_bold'))
        for echo in (1, 2, 3)
    ]
    # The phase list follows the order of the magnitude list
    assert collect_bold_phase(_layout(bids_dir), mags[::-1]) == phases[::-1]


def test_multi_echo_partial(bids_dir, caplog):
    mags = [
        _write_nifti(_func_file(bids_dir, f'task-rest_echo-{echo}_part-mag_bold'))
        for echo in (1, 2, 3)
    ]
    for echo in (1, 2):
        _write_nifti(_func_file(bids_dir, f'task-rest_echo-{echo}_part-phase_bold'))

    with caplog.at_level(logging.WARNING):
        assert collect_bold_phase(_layout(bids_dir), mags) is None

    assert 'only some echoes' in caplog.text
    assert mags[2] in caplog.text


@pytest.mark.parametrize(
    ('phase_kwargs', 'message'),
    [
        ({'shape': SHAPE[:3] + (4,)}, 'shape'),
        ({'affine': np.diag([2.0, 2.0, 2.0, 1.0])}, 'affine'),
    ],
)
def test_mismatched_phase_is_not_used(bids_dir, caplog, phase_kwargs, message):
    mag = _write_nifti(_func_file(bids_dir, 'task-rest_part-mag_bold'))
    phase = _write_nifti(_func_file(bids_dir, 'task-rest_part-phase_bold'), **phase_kwargs)

    with caplog.at_level(logging.WARNING):
        assert collect_bold_phase(_layout(bids_dir), [mag]) is None

    assert phase in caplog.text
    assert message in caplog.text
