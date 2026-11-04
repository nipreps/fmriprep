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
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from fmriprep.utils.bids import (
    bold_template_zooms,
    collect_selection,
    is_valid_bold_template,
    resolve_selection,
)


class _MockLayout:
    """Minimal BIDSLayout stub for is_valid_bold_template tests."""

    def __init__(self, pe_map: dict):
        self._pe_map = pe_map

    def get_metadata(self, f):
        pe = self._pe_map.get(f)
        return {'PhaseEncodingDirection': pe} if pe is not None else {}


TEST_ZOOMS = (2.4, 2.4, 2.4)


# Each run is a (fieldmap, phase-encoding direction, voxel size) tuple
@pytest.mark.parametrize(
    ('runs', 'expected'),
    [
        pytest.param([], False, id='no_runs'),
        pytest.param([('fmap1', 'j', TEST_ZOOMS)], False, id='one_run'),
        pytest.param([('fmap1', 'j', TEST_ZOOMS), ('fmap2', 'j', TEST_ZOOMS)], True, id='all_sdc'),
        pytest.param([('fmap1', 'j', TEST_ZOOMS), (None, 'j', TEST_ZOOMS)], False, id='mixed_sdc'),
        pytest.param(
            [(None, 'j', TEST_ZOOMS), (None, 'j', TEST_ZOOMS)], True, id='no_sdc_single_pe'
        ),
        pytest.param(
            [(None, 'j', TEST_ZOOMS), (None, 'j-', TEST_ZOOMS)], False, id='no_sdc_opposing_pe'
        ),
        pytest.param(
            [(None, None, TEST_ZOOMS), (None, None, TEST_ZOOMS)], True, id='no_sdc_no_pe'
        ),
        pytest.param(
            [(None, 'j', TEST_ZOOMS), (None, None, TEST_ZOOMS)], False, id='no_sdc_missing_pe'
        ),
        pytest.param(
            [('fmap1', 'j', TEST_ZOOMS), ('fmap2', 'j', (0.8, 0.8, 0.8))],
            False,
            id='mixed_resolution',
        ),
        pytest.param(
            [('fmap1', 'j', TEST_ZOOMS), ('fmap2', 'j', (2.4, 2.4, 3.0))],
            False,
            id='mixed_slice_thickness',
        ),
        pytest.param(
            [('fmap1', 'j', TEST_ZOOMS), ('fmap2', 'j', (2.4002, 2.4002, 2.4002))],
            True,
            id='negligible_zoom_difference',
        ),
    ],
)
def test_is_valid_bold_template(tmp_path: Path, runs, expected):
    bold_runs, estimator_map, pe_map = [], {}, {}
    for i, (fmapid, pe_dir, zooms) in enumerate(runs):
        path = str(tmp_path / f'run{i}.nii.gz')
        nb.Nifti1Image(np.zeros((2, 2, 2), dtype='uint8'), np.diag((*zooms, 1.0))).to_filename(
            path
        )
        bold_runs.append([path])
        estimator_map[path] = fmapid
        pe_map[path] = pe_dir

    layout = _MockLayout(pe_map)
    assert is_valid_bold_template(bold_runs, estimator_map, layout) is expected


@pytest.mark.parametrize(
    ('zooms', 'expected'),
    [
        pytest.param((2.4, 2.4, 2.4), (1.2, 1.2, 1.2), id='coarse'),
        pytest.param((4.0, 4.0, 4.0), (1.2, 1.2, 1.2), id='very_coarse'),
        pytest.param((3.0, 3.0, 4.0), (1.2, 1.2, 1.2), id='anisotropic'),
        pytest.param((1.0, 1.0, 2.0), (1.0, 1.0, 1.0), id='finest_axis_is_finer'),
        pytest.param((2.0, 2.0, 0.8), (0.8, 0.8, 0.8), id='finest_axis_is_much_finer'),
        pytest.param((1.2, 1.2, 1.2), None, id='at_target'),
        pytest.param((0.8, 0.8, 0.8), None, id='finer_than_target'),
    ],
)
def test_bold_template_zooms(tmp_path: Path, zooms, expected):
    bold_files = []
    for i in range(2):
        path = str(tmp_path / f'run{i}.nii.gz')
        nb.Nifti1Image(np.zeros((2, 2, 2), dtype='uint8'), np.diag((*zooms, 1.0))).to_filename(
            path
        )
        bold_files.append(path)

    assert bold_template_zooms(bold_files) == expected


@pytest.mark.parametrize(
    ('labels', 'reference', 'template_runs'),
    [
        (['n/a', 'n/a'], 0, [True, True]),
        (['n/a', 'n/a', 'n/a'], None, [True, True, True]),
        (['n/a', 'target', 'n/a'], 1, [False, True, False]),
        (['include', 'n/a', 'target'], 2, [True, False, True]),
        (['n/a', 'include', 'n/a'], 1, [False, True, False]),
        (['include', 'n/a', 'include'], None, [True, False, True]),
    ],
)
def test_resolve_selection(labels, reference, template_runs):
    assert resolve_selection(labels) == (reference, template_runs)


def test_resolve_selection_multiple_targets():
    with pytest.raises(RuntimeError, match='target'):
        resolve_selection(['target', 'include', 'target'])


def _write_scans(path: Path, rows: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['filename\tboldref_template'] + [f'{f}\t{v}' for f, v in rows.items()]
    path.write_text('\n'.join(lines) + '\n')


def _bold(bids_dir: Path, ses: str, run: int) -> str:
    return str(
        bids_dir / f'sub-01/ses-{ses}/func/sub-01_ses-{ses}_task-rest_run-{run}_bold.nii.gz'
    )


def test_collect_selection(tmp_path: Path):
    """Raw scans files win over derivatives, which fill in sessions raw does not annotate."""
    bids_dir, deriv_dir = tmp_path / 'bids', tmp_path / 'deriv'
    bold_files = [_bold(bids_dir, ses, run) for ses in 'AB' for run in (1, 2)]

    _write_scans(
        bids_dir / 'sub-01/ses-A/sub-01_ses-A_scans.tsv',
        {'func/sub-01_ses-A_task-rest_run-1_bold.nii.gz': 'target'},
    )
    _write_scans(
        deriv_dir / 'sub-01/ses-A/sub-01_ses-A_scans.tsv',
        {'func/sub-01_ses-A_task-rest_run-2_bold.nii.gz': 'target'},
    )
    (bids_dir / 'sub-01/ses-B').mkdir(parents=True)
    (bids_dir / 'sub-01/ses-B/sub-01_ses-B_scans.tsv').write_text(
        'filename\tacq_time\nfunc/sub-01_ses-B_task-rest_run-1_bold.nii.gz\tn/a\n'
    )
    _write_scans(
        deriv_dir / 'sub-01/ses-B/sub-01_ses-B_scans.tsv',
        {
            'func/sub-01_ses-B_task-rest_run-1_bold.nii.gz': 'include',
            'func/sub-01_ses-B_task-rest_run-2_bold.nii.gz': 'n/a',
        },
    )

    labels = collect_selection(bold_files, bids_dir, [deriv_dir])
    assert labels == ['target', 'n/a', 'include', 'n/a']


def test_collect_selection_without_sessions(tmp_path: Path):
    bold_files = [
        str(tmp_path / f'sub-01/func/sub-01_task-rest_run-{run}_bold.nii.gz') for run in (1, 2)
    ]
    _write_scans(
        tmp_path / 'sub-01/sub-01_scans.tsv',
        {'func/sub-01_task-rest_run-2_bold.nii.gz': 'include'},
    )

    assert collect_selection(bold_files, tmp_path, []) == ['n/a', 'include']


@pytest.mark.parametrize(
    ('content', 'match'),
    [
        (
            'filename\tboldref_template\nfunc/sub-01_ses-A_task-rest_run-1_bold.nii.gz\tbest\n',
            'best',
        ),
        ('boldref_template\ntarget\n', 'filename'),
    ],
)
def test_collect_selection_invalid(tmp_path: Path, content, match):
    scans = tmp_path / 'sub-01/ses-A/sub-01_ses-A_scans.tsv'
    scans.parent.mkdir(parents=True)
    scans.write_text(content)

    with pytest.raises(ValueError, match=match):
        collect_selection([_bold(tmp_path, 'A', 1)], tmp_path, [])
