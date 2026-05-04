from pathlib import Path

import pytest
from bids.layout import BIDSLayout

from fmriprep.utils import bids


@pytest.mark.parametrize('desc', ['hmc', 'coreg'])
def test_baseline_found_as_str(tmp_path: Path, desc: str):
    subject = '0'
    task = 'rest'

    to_find = tmp_path.joinpath(
        f'sub-{subject}', 'func', f'sub-{subject}_task-{task}_desc-{desc}_boldref.nii.gz'
    )
    to_find.parent.mkdir(parents=True)
    to_find.touch()

    entities = {
        'subject': subject,
        'task': task,
        'suffix': 'bold',
        'extension': '.nii.gz',
    }

    derivs = bids.collect_derivatives(derivatives_dir=tmp_path, entities=entities)
    assert dict(derivs) == {f'{desc}_boldref': str(to_find), 'transforms': {}}


@pytest.mark.parametrize('xfm', ['boldref2fmap', 'boldref2anat', 'hmc'])
def test_transforms_found_as_str(tmp_path: Path, xfm: str):
    subject = '0'
    task = 'rest'
    fromto = {
        'hmc': 'from-orig_to-boldref',
        'boldref2fmap': 'from-boldref_to-auto00000',
        'boldref2anat': 'from-boldref_to-anat',
    }[xfm]

    to_find = tmp_path.joinpath(
        f'sub-{subject}', 'func', f'sub-{subject}_task-{task}_{fromto}_mode-image_xfm.txt'
    )
    to_find.parent.mkdir(parents=True)
    to_find.touch()

    entities = {
        'subject': subject,
        'task': task,
        'suffix': 'bold',
        'extension': '.nii.gz',
    }

    derivs = bids.collect_derivatives(
        derivatives_dir=tmp_path,
        entities=entities,
        fieldmap_id='auto_00000',
    )
    assert derivs == {'transforms': {xfm: str(to_find)}}


def test_select_bold_magnitude_files():
    files = [
        'sub-01/func/sub-01_task-rest_echo-1_part-mag_bold.nii.gz',
        'sub-01/func/sub-01_task-rest_echo-1_part-phase_bold.nii.gz',
        'sub-01/func/sub-01_task-rest_echo-2_part-mag_bold.nii.gz',
    ]
    assert bids.select_bold_magnitude_files(files) == [files[0], files[2]]


def test_collect_bold_part_files(tmp_path: Path):
    (tmp_path / 'dataset_description.json').write_text('{"Name": "test", "BIDSVersion": "1.8.0"}')
    func_dir = tmp_path / 'sub-01' / 'func'
    func_dir.mkdir(parents=True)

    magnitude_files = []
    phase_files = []
    for echo in range(1, 4):
        mag = func_dir / f'sub-01_task-rest_echo-{echo}_part-mag_bold.nii.gz'
        phase = func_dir / f'sub-01_task-rest_echo-{echo}_part-phase_bold.nii.gz'
        mag.touch()
        phase.touch()
        magnitude_files.append(str(mag))
        phase_files.append(str(phase))

    layout = BIDSLayout(tmp_path, validate=False)
    assert bids.collect_bold_part_files(layout, magnitude_files, part='phase') == phase_files
