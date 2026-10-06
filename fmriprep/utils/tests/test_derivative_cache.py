from pathlib import Path

import pytest

from fmriprep.utils import bids
from fmriprep.utils.testing import write_derivatives

ENTITIES = {
    'subject': '01',
    'task': 'rest',
    'run': 1,
    'suffix': 'bold',
    'extension': '.nii.gz',
}


def _files(root: Path, datatype: str = 'func') -> list[str]:
    return sorted(str(p) for p in root.rglob(f'{datatype}/*') if p.is_file())


def _entities(session):
    return {**ENTITIES, 'session': session} if session else ENTITIES


BOLDREFS = [
    'hmc_boldref',
    'hmc_boldref_legacy',
    'run_boldref',
    'run_boldref_legacy',
    'session_boldref',
    'subject_boldref',
]
TRANSFORMS = [
    'hmc',
    'hmc_legacy',
    'run2anat',
    'run2anat_legacy',
    'run2fmap',
    'run2fmap_legacy',
    'run2template',
    'session2anat',
    'subject2anat',
]


@pytest.mark.parametrize('session', [None, 'A'])
@pytest.mark.parametrize('group', BOLDREFS)
def test_boldref_found_as_str(tmp_path: Path, group: str, session):
    """Generate a single boldref and verify it's found.

    `<x>_legacy` is defined in data/tests/derivatives.yml to generate files,
    but the collector loads `<x>` and `<x>_legacy` as `<x>`.
    """
    root = write_derivatives(tmp_path / 'deriv', [group], run=1, session=session)
    [found] = _files(root)

    key = group.removesuffix('_legacy')
    derivs = bids.collect_derivatives(root, _entities(session), fieldmap_id='auto_00000')
    assert dict(derivs) == {key: found, 'transforms': {}}


@pytest.mark.parametrize('session', [None, 'A'])
@pytest.mark.parametrize('group', TRANSFORMS)
def test_transform_found_as_str(tmp_path: Path, group: str, session):
    """Generate a single transform and verify it's found.

    See above RE legacy derivatives.
    """
    root = write_derivatives(tmp_path / 'deriv', [group], run=1, session=session)
    [found] = _files(root)

    key = group.removesuffix('_legacy')
    derivs = bids.collect_derivatives(root, _entities(session), fieldmap_id='auto_00000')
    assert dict(derivs) == {'transforms': {key: found}}

    # Collector function normalizes fieldmap IDs internally
    if key == 'run2fmap':
        assert '_to-auto00000_' in found


@pytest.mark.parametrize('session', [None, 'A'])
def test_defaults_collected_once_each(tmp_path: Path, session):
    """Default (None) includes loads all keys."""
    root = write_derivatives(tmp_path / 'deriv', None, run=1, session=session)

    derivs = bids.collect_derivatives(root, _entities(session), fieldmap_id='auto_00000')
    transforms = derivs.pop('transforms')
    assert derivs.keys() == {'hmc_boldref', 'run_boldref', 'session_boldref', 'subject_boldref'}
    assert transforms.keys() == {group.removesuffix('_legacy') for group in TRANSFORMS}
    assert {*derivs.values(), *transforms.values()} == set(_files(root))


@pytest.mark.xfail(reason='existing collector returns both the current and the legacy file')
@pytest.mark.parametrize('group', ['hmc_boldref', 'run_boldref', 'hmc', 'run2anat', 'run2fmap'])
def test_current_name_preferred_over_legacy(tmp_path: Path, group: str):
    """Generate current and legacy files and verify only current is returned."""
    # Each current-name group is collected under its own name
    [current] = _files(write_derivatives(tmp_path / 'current', [group], run=1))
    root = write_derivatives(tmp_path / 'deriv', [group, f'{group}_legacy'], run=1)

    derivs = bids.collect_derivatives(root, ENTITIES, fieldmap_id='auto_00000')
    found = derivs['transforms'][group] if group in derivs['transforms'] else derivs[group]
    assert found == str(root / 'sub-01' / 'func' / Path(current).name)


@pytest.mark.xfail(reason='existing run2fmap query matches any transform from the run')
def test_no_run2fmap_without_fieldmap(tmp_path: Path):
    # Note that this is probably harmless, because no fieldmap_id means we won't look for run2fmap
    # However, it should be fixed in nipost, so here's a regression test
    root = write_derivatives(tmp_path / 'deriv', ['run2anat'], run=1)

    derivs = bids.collect_derivatives(root, ENTITIES)
    assert 'run2fmap' not in derivs['transforms']


@pytest.mark.parametrize(
    ('include', 'expected'),
    [
        (['fieldmap', 'coeffs', 'magnitude'], ('preproc', 'coeff', 'magnitude')),
        (['fieldmap', 'coeffs', 'magnitude_epi'], ('preproc', 'coeff', 'epi')),
        (
            ['fieldmap', 'coeffs_split', 'magnitude'],
            ('preproc', ['coeff0', 'coeff1'], 'magnitude'),
        ),
    ],
    ids=['magnitude', 'epi', 'split-coeffs'],
)
def test_fieldmaps_found(tmp_path: Path, include, expected):
    root = write_derivatives(tmp_path / 'deriv', include, run=1)
    fmap = root / 'sub-01' / 'fmap'

    def _path(desc):
        if isinstance(desc, list):
            return [_path(d) for d in desc]
        return str(fmap / f'sub-01_fmapid-auto00000_desc-{desc}_fieldmap.nii.gz')

    fmaps = bids.collect_fieldmaps(root, {'subject': '01'})
    assert dict(fmaps) == {
        'auto00000': dict(
            zip(('fieldmap', 'coeffs', 'magnitude'), map(_path, expected), strict=True)
        ),
    }


def test_aggregate_coreg_precomputed_run():
    caches = [
        {'transforms': {'run2anat': '/ra1', 'session2anat': '/sa', 'run2template': '/rb1'}},
        {'transforms': {'run2anat': '/ra2'}},
    ]
    assert bids.aggregate_coreg_precomputed(caches, 'run') == {
        'template2anat_xfm': ['/ra1', '/ra2'],
    }


def test_aggregate_coreg_precomputed_group():
    caches = [
        {
            'transforms': {'session2anat': '/sa', 'run2template': '/rb1'},
            'session_boldref': '/tpl',
        },
        {'transforms': {'session2anat': '/sa', 'run2template': '/rb2'}},
    ]
    assert bids.aggregate_coreg_precomputed(caches, 'session') == {
        'template2anat_xfm': ['/sa', '/sa'],
        'run2template_xfms': ['/rb1', '/rb2'],
        'boldref_template': '/tpl',
    }
