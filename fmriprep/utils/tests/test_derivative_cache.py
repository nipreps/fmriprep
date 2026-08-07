from pathlib import Path

import pytest

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
    expected = {f'{desc}_boldref': str(to_find), 'transforms': {}}
    if desc == 'coreg':
        # coreg_boldref is aliased to the run_boldref key for legacy inputs
        expected['run_boldref'] = str(to_find)
    assert dict(derivs) == expected


@pytest.mark.parametrize('xfm', ['run2fmap', 'boldref2anat', 'hmc'])
def test_transforms_found_as_str(tmp_path: Path, xfm: str):
    subject = '0'
    task = 'rest'
    fromto = {
        'hmc': 'from-orig_to-boldref',
        'run2fmap': 'from-run_to-auto00000',
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
