from pathlib import Path

import pytest

from fmriprep.utils.testing import deriv_skeleton, write_derivatives


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())


def test_deriv_skeleton_defaults(tmp_path: Path):
    root = write_derivatives(tmp_path / 'deriv', session='A', run=1)

    assert root == tmp_path / 'deriv'
    ses = 'sub-01/ses-A'
    run = 'sub-01_ses-A_task-rest_run-1'
    assert _files(root) == sorted(
        [
            'dataset_description.json',
            f'{ses}/func/{run}_space-orig_desc-hmc_boldref.nii.gz',
            f'{ses}/func/{run}_space-run_boldref.nii.gz',
            f'{ses}/func/sub-01_ses-A_space-session_boldref.nii.gz',
            'sub-01/func/sub-01_space-subject_boldref.nii.gz',
            f'{ses}/func/{run}_from-orig_to-run_mode-image_desc-hmc_xfm.txt',
            f'{ses}/func/{run}_from-run_to-T1w_mode-image_desc-coreg_xfm.txt',
            f'{ses}/func/{run}_from-run_to-auto00000_mode-image_desc-fmap_xfm.txt',
            f'{ses}/func/{run}_from-run_to-session_mode-image_desc-coreg_xfm.txt',
            f'{ses}/func/sub-01_ses-A_from-session_to-T1w_mode-image_desc-coreg_xfm.txt',
            'sub-01/func/sub-01_from-subject_to-T1w_mode-image_desc-coreg_xfm.txt',
            f'{ses}/fmap/sub-01_ses-A_fmapid-auto00000_desc-preproc_fieldmap.nii.gz',
            f'{ses}/fmap/sub-01_ses-A_fmapid-auto00000_desc-coeff_fieldmap.nii.gz',
            f'{ses}/fmap/sub-01_ses-A_fmapid-auto00000_desc-magnitude_fieldmap.nii.gz',
        ]
    )


def test_deriv_skeleton_legacy_and_alternatives(tmp_path: Path):
    groups = [
        'hmc_boldref_legacy',
        'run_boldref_legacy',
        'hmc_legacy',
        'run2anat_legacy',
        'run2fmap_legacy',
        'coeffs_split',
        'magnitude_epi',
    ]
    root = write_derivatives(tmp_path / 'deriv', groups, run=1)

    run = 'sub-01_task-rest_run-1'
    assert _files(root) == sorted(
        [
            'dataset_description.json',
            f'sub-01/func/{run}_desc-hmc_boldref.nii.gz',
            f'sub-01/func/{run}_desc-coreg_boldref.nii.gz',
            f'sub-01/func/{run}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt',
            f'sub-01/func/{run}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt',
            f'sub-01/func/{run}_from-boldref_to-auto00000_mode-image_xfm.txt',
            'sub-01/fmap/sub-01_fmapid-auto00000_desc-coeff0_fieldmap.nii.gz',
            'sub-01/fmap/sub-01_fmapid-auto00000_desc-coeff1_fieldmap.nii.gz',
            'sub-01/fmap/sub-01_fmapid-auto00000_desc-epi_fieldmap.nii.gz',
        ]
    )


def test_deriv_skeleton_placeholders():
    skeleton = deriv_skeleton(['run2fmap'], subject='02', task='nback', fmapid='phasediff')

    assert skeleton['dataset_description']['DatasetType'] == 'derivative'
    assert skeleton['02'] == [
        {
            'session': None,
            'func': [
                {
                    'task': 'nback',
                    'from': 'run',
                    'to': 'phasediff',
                    'mode': 'image',
                    'desc': 'fmap',
                    'suffix': 'xfm',
                    'extension': '.txt',
                },
            ],
        },
    ]


def test_deriv_skeleton_empty(tmp_path: Path):
    assert deriv_skeleton([])['01'] == []
    assert _files(write_derivatives(tmp_path / 'deriv', [])) == ['dataset_description.json']


def test_deriv_skeleton_unknown_group():
    with pytest.raises(KeyError, match='nonexistent'):
        deriv_skeleton(['nonexistent'])
