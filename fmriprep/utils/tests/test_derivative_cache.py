from pathlib import Path

import pytest

from fmriprep.utils import bids


@pytest.mark.parametrize(
    ('key', 'ents'),
    [
        ('hmc', 'space-orig_desc-hmc'),
        ('run', 'space-run'),
        # legacy space-less names
        ('hmc', 'desc-hmc'),
        ('run', 'desc-coreg'),
    ],
)
def test_baseline_found_as_str(tmp_path: Path, key: str, ents: str):
    subject = '0'
    task = 'rest'

    to_find = tmp_path.joinpath(
        f'sub-{subject}', 'func', f'sub-{subject}_task-{task}_{ents}_boldref.nii.gz'
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
    assert dict(derivs) == {f'{key}_boldref': str(to_find), 'transforms': {}}


@pytest.mark.parametrize(
    ('xfm', 'fromto'),
    [
        ('hmc', 'from-orig_to-boldref'),
        ('run2fmap', 'from-run_to-auto00000'),
        ('boldref2anat', 'from-run_to-anat'),
        # legacy from-boldref names
        ('run2fmap', 'from-boldref_to-auto00000'),
        ('boldref2anat', 'from-boldref_to-anat'),
    ],
)
def test_transforms_found_as_str(tmp_path: Path, xfm: str, fromto: str):
    subject = '0'
    task = 'rest'

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


@pytest.mark.parametrize(('coreg_space', 'ses_ents'), [('session', '_ses-A'), ('subject', '')])
def test_group_boldref2anat_found_for_run(tmp_path: Path, coreg_space: str, ses_ents: str):
    subject = '0'
    task = 'rest'

    func = tmp_path.joinpath(f'sub-{subject}', *(['ses-A'] if ses_ents else []), 'func')
    func.mkdir(parents=True)
    to_find = func / (
        f'sub-{subject}{ses_ents}_from-{coreg_space}_to-anat_mode-image_desc-coreg_xfm.txt'
    )
    to_find.touch()

    entities = {
        'subject': subject,
        'session': 'A',
        'task': task,
        'run': '01',
        'suffix': 'bold',
        'extension': '.nii.gz',
    }

    derivs = bids.collect_derivatives(derivatives_dir=tmp_path, entities=entities)
    assert derivs == {'transforms': {'boldref2anat': str(to_find)}}
