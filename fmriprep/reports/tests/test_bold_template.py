# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
import json
import re
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from fmriprep.cli.bold_template import main
from fmriprep.reports.bold_template import REPORT_FILE, SELECTION_FILE, find_runs
from fmriprep.utils.bids import (
    collect_selection,
    resolve_selection,
    write_derivative_description,
)


def _first_pass(tmp_path: Path, sessions=('A', 'B'), runs=(1, 2, 3), echoes=('',)) -> Path:
    """Write a raw dataset, and the run BOLD references of its fMRIPrep output."""
    fmriprep_dir, raw_dir = tmp_path / 'fmriprep', tmp_path / 'raw'

    rng = np.random.default_rng(0)
    brain = np.zeros((12, 12, 10), dtype='uint8')
    brain[3:9, 3:9, 2:8] = 1
    for ses in sessions:
        raw_func = raw_dir / f'sub-01/ses-{ses}/func'
        func = fmriprep_dir / f'sub-01/ses-{ses}/func'
        raw_func.mkdir(parents=True)
        func.mkdir(parents=True)
        for run in runs:
            prefix = f'sub-01_ses-{ses}_task-rest_run-{run}'
            for echo in echoes:
                (raw_func / f'{prefix}{echo}_bold.nii.gz').touch()
            boldref = 100 * brain + rng.normal(10, 5, brain.shape)
            nb.Nifti1Image(boldref.astype('float32'), np.eye(4)).to_filename(
                func / f'{prefix}_space-run_boldref.nii.gz'
            )
            nb.Nifti1Image(brain, np.eye(4)).to_filename(
                func / f'{prefix}_space-run_desc-brain_mask.nii.gz'
            )
    return fmriprep_dir


def _viz(tmp_path: Path, *args: str, coreg_level: str = 'session') -> Path:
    output_dir = tmp_path / 'selection'
    raw_dir, fmriprep_dir = tmp_path / 'raw', tmp_path / 'fmriprep'
    main(
        [str(raw_dir), str(fmriprep_dir), str(output_dir), '--bold-coreg-level', coreg_level]
        + list(args)
    )
    return output_dir


def _save(output_dir: Path, selection: dict) -> Path:
    """Write a selection as saved from the page."""
    selection_file = output_dir / SELECTION_FILE
    selection_file.write_text(json.dumps(selection))
    return selection_file


def _page(output_dir: Path) -> dict:
    html = (output_dir / REPORT_FILE).read_text()
    return json.loads(re.search(r'<script id="page"[^>]*>(.*?)</script>', html, re.DOTALL)[1])


def test_find_runs_multiecho(tmp_path: Path):
    fmriprep_dir = _first_pass(tmp_path, sessions=('A',), runs=(1,), echoes=('_echo-2', '_echo-1'))

    (run,) = find_runs(tmp_path / 'raw', fmriprep_dir)
    assert run.scans == Path('sub-01/ses-A/sub-01_ses-A_scans.tsv')
    assert run.filename == 'func/sub-01_ses-A_task-rest_run-1_echo-1_bold.nii.gz'
    assert run.mask.name == 'sub-01_ses-A_task-rest_run-1_space-run_desc-brain_mask.nii.gz'


def test_selection_round_trip(tmp_path: Path):
    fmriprep_dir = _first_pass(tmp_path)
    output_dir = _viz(tmp_path)

    # Described like an fMRIPrep output, linked to its inputs
    write_derivative_description(tmp_path / 'raw', tmp_path)
    expected = json.loads((tmp_path / 'dataset_description.json').read_text())
    description = json.loads((output_dir / 'dataset_description.json').read_text())
    assert description['GeneratedBy'] == expected['GeneratedBy']
    assert description['DatasetLinks'] == {
        'raw': str(tmp_path / 'raw'),
        'fmriprep': str(fmriprep_dir),
    }

    page = _page(output_dir)
    assert [g['scans'] for g in page['groups']] == [
        ['sub-01/ses-A/sub-01_ses-A_scans.tsv'],
        ['sub-01/ses-B/sub-01_ses-B_scans.tsv'],
    ]
    for group in page['groups']:
        assert all(run['figure'].startswith('data:image/png;base64,') for run in group['runs'])
        assert {run['value'] for run in group['runs']} == {'n/a'}
        assert (output_dir / group['scans'][0]).is_file()

    selection_file = _save(
        output_dir,
        {
            'sub-01/ses-A/sub-01_ses-A_scans.tsv': {
                'func/sub-01_ses-A_task-rest_run-1_bold.nii.gz': 'include',
                'func/sub-01_ses-A_task-rest_run-2_bold.nii.gz': 'target',
                'func/sub-01_ses-A_task-rest_run-3_bold.nii.gz': 'n/a',
            },
            'sub-01/ses-B/sub-01_ses-B_scans.tsv': {
                'func/sub-01_ses-B_task-rest_run-1_bold.nii.gz': 'include',
                'func/sub-01_ses-B_task-rest_run-3_bold.nii.gz': 'include',
            },
        },
    )
    _viz(tmp_path)
    assert not selection_file.exists()

    raw_dir = tmp_path / 'raw'
    bold_files = [
        str(raw_dir / f'sub-01/ses-{ses}/func/sub-01_ses-{ses}_task-rest_run-{run}_bold.nii.gz')
        for ses in 'AB'
        for run in (1, 2, 3)
    ]
    labels = collect_selection(bold_files, raw_dir, [fmriprep_dir, output_dir])
    assert resolve_selection(labels[:3]) == (1, [True, True, False])
    assert resolve_selection(labels[3:]) == (None, [True, False, True])

    # The page is regenerated from the applied selection
    values = [run['value'] for g in _page(output_dir)['groups'] for run in g['runs']]
    assert values == labels


@pytest.mark.parametrize(
    ('selection', 'match'),
    [
        ({'func/sub-01_ses-A_task-rest_run-1_bold.nii.gz': 'best'}, 'invalid values'),
        ({'func/sub-01_ses-A_task-rest_run-9_bold.nii.gz': 'include'}, 'unknown runs'),
        (
            {
                'func/sub-01_ses-A_task-rest_run-1_bold.nii.gz': 'target',
                'func/sub-01_ses-A_task-rest_run-2_bold.nii.gz': 'target',
            },
            'more than one',
        ),
    ],
)
def test_apply_selection_invalid(tmp_path: Path, caplog, selection, match):
    _first_pass(tmp_path)
    output_dir = _viz(tmp_path)
    scans = {p: p.read_text() for p in output_dir.glob('sub-01/ses-*/*_scans.tsv')}

    selection_file = _save(
        output_dir,
        {
            'sub-01/ses-B/sub-01_ses-B_scans.tsv': {
                'func/sub-01_ses-B_task-rest_run-1_bold.nii.gz': 'target',
            },
            'sub-01/ses-A/sub-01_ses-A_scans.tsv': selection,
        },
    )
    with pytest.raises(SystemExit) as exc:
        _viz(tmp_path)
    assert exc.value.code == 1
    assert match in caplog.text
    assert {p: p.read_text() for p in scans} == scans
    assert selection_file.exists()


@pytest.mark.parametrize(
    ('coreg_level', 'n_groups', 'code'), [('session', 2, None), ('subject', 1, 1)]
)
def test_coreg_level(tmp_path: Path, caplog, coreg_level, n_groups, code):
    _first_pass(tmp_path)
    output_dir = _viz(tmp_path, coreg_level=coreg_level)
    assert len(_page(output_dir)['groups']) == n_groups

    _save(
        output_dir,
        {
            'sub-01/ses-A/sub-01_ses-A_scans.tsv': {
                'func/sub-01_ses-A_task-rest_run-1_bold.nii.gz': 'target',
            }
        },
    )
    _viz(tmp_path, coreg_level=coreg_level)

    # The second target is in another session's scans file
    _save(
        output_dir,
        {
            'sub-01/ses-B/sub-01_ses-B_scans.tsv': {
                'func/sub-01_ses-B_task-rest_run-2_bold.nii.gz': 'target',
            }
        },
    )
    if code is None:
        _viz(tmp_path, coreg_level=coreg_level)
        return
    with pytest.raises(SystemExit) as exc:
        _viz(tmp_path, coreg_level=coreg_level)
    assert exc.value.code == code
    assert 'more than one "target" run in sub-01' in caplog.text


@pytest.mark.parametrize('label', ['B', 'ses-B'])
def test_session_label(tmp_path: Path, label):
    _first_pass(tmp_path)
    output_dir = _viz(tmp_path, '--session-label', label)
    assert [g['scans'] for g in _page(output_dir)['groups']] == [
        ['sub-01/ses-B/sub-01_ses-B_scans.tsv']
    ]


def test_raw_scans_carried_over(tmp_path: Path, caplog):
    _first_pass(tmp_path, sessions=('A',))
    raw_scans = tmp_path / 'raw/sub-01/ses-A/sub-01_ses-A_scans.tsv'
    raw_scans.parent.mkdir(parents=True, exist_ok=True)
    raw_scans.write_text(
        'filename\tacq_time\tboldref_template\n'
        'anat/sub-01_ses-A_T1w.nii.gz\t2020-01-01T10:00:00\t\n'
        'func/sub-01_ses-A_task-rest_run-3_bold.nii.gz\t2020-01-01T10:30:00\ttarget\n'
    )

    output_dir = _viz(tmp_path)

    assert 'fMRIPrep reads instead of this selection' in caplog.text
    assert (output_dir / 'sub-01/ses-A/sub-01_ses-A_scans.tsv').read_text() == (
        'filename\tacq_time\tboldref_template\n'
        'anat/sub-01_ses-A_T1w.nii.gz\t2020-01-01T10:00:00\t\n'
        'func/sub-01_ses-A_task-rest_run-3_bold.nii.gz\t2020-01-01T10:30:00\ttarget\n'
        'func/sub-01_ses-A_task-rest_run-1_bold.nii.gz\tn/a\tn/a\n'
        'func/sub-01_ses-A_task-rest_run-2_bold.nii.gz\tn/a\tn/a\n'
    )
    (group,) = _page(output_dir)['groups']
    assert [run['value'] for run in group['runs']] == ['n/a', 'n/a', 'target']


def test_existing_scans_extended(tmp_path: Path):
    _first_pass(tmp_path, sessions=('A',))
    scans = tmp_path / 'selection/sub-01/ses-A/sub-01_ses-A_scans.tsv'
    scans.parent.mkdir(parents=True)
    scans.write_text(
        'filename\tboldref_template\nfunc/sub-01_ses-A_task-rest_run-1_bold.nii.gz\tinclude\n'
    )

    _viz(tmp_path)

    assert scans.read_text() == (
        'filename\tboldref_template\n'
        'func/sub-01_ses-A_task-rest_run-1_bold.nii.gz\tinclude\n'
        'func/sub-01_ses-A_task-rest_run-2_bold.nii.gz\tn/a\n'
        'func/sub-01_ses-A_task-rest_run-3_bold.nii.gz\tn/a\n'
    )
