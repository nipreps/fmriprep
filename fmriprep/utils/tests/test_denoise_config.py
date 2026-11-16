"""Tests for the svht_denoise configuration and build-time checks."""

import json
import os
import sys

import pytest

from fmriprep.utils import denoise
from fmriprep.utils.denoise import (
    SVHT_MAX_EXTENT,
    SVHT_MAX_VOLUMES,
    can_svht_denoise,
    check_svht_phase,
    check_svht_volumes,
    describe_svht,
    load_svht_config,
    svht_auto_extent,
    svht_denoise_version,
)


def _write_config(tmp_path, content):
    path = tmp_path / 'denoise.json'
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def test_load_svht_config_valid(tmp_path):
    cfg = {
        'aggregator': 'gaussian',
        'aggregator_fwhm': 1.5,
        'extent': 9,
        'filter_method': 'optthresh',
        'noise_dof': 8,
        'preserve_noise_bias': True,
        'shape': 'cube',
        'stride': 2,
        'vst': True,
    }
    assert load_svht_config(_write_config(tmp_path, cfg)) == cfg
    assert load_svht_config(_write_config(tmp_path, {})) == {}


@pytest.mark.parametrize(
    ('content', 'match'),
    [
        ('{"extent": 7, "extent": 9}', 'repeats key'),
        ('{"extent": ', 'not valid JSON'),
        ('[1, 2]', 'JSON object'),
        ({'demean': 'shells'}, 'unknown key'),
        ({'aggregator': 'median'}, 'must be one of'),
        ({'shape': 3}, 'must be one of'),
        ({'vst': 'yes'}, 'true or false'),
        ({'noise_dof': 0}, 'from 1 to 64'),
        ({'noise_dof': True}, 'from 1 to 64'),
        ({'stride': 3}, 'must be 1 or 2'),
        ({'extent': 8}, f'odd integer from 3 to {SVHT_MAX_EXTENT}'),
        ({'extent': -1}, 'odd integer from 3'),
        # svht_denoise's smallest patch is 3 voxels across
        ({'extent': 1}, 'odd integer from 3'),
        ({'extent': SVHT_MAX_EXTENT + 2}, f'odd integer from 3 to {SVHT_MAX_EXTENT}'),
        ({'aggregator_fwhm': 0}, 'number > 0'),
        ({'aggregator': 'uniform', 'aggregator_fwhm': 1.0}, 'needs "aggregator" "gaussian"'),
        ({'aggregator': 'exclusive', 'stride': 2}, 'centered on every voxel'),
        ({'vst': False, 'noise_dof': 4}, 'with "vst" false'),
    ],
)
def test_load_svht_config_invalid(tmp_path, content, match):
    with pytest.raises(ValueError, match=match):
        load_svht_config(_write_config(tmp_path, content))


def test_load_svht_config_missing(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        load_svht_config(tmp_path / 'missing.json')


def test_check_svht_phase():
    # Anything goes without phase
    check_svht_phase({'vst': True, 'noise_dof': 4}, use_phase=False)
    check_svht_phase({'vst': False}, use_phase=True)
    check_svht_phase({}, use_phase=True)

    with pytest.raises(ValueError, match='vst and noise_dof'):
        check_svht_phase({'vst': True, 'noise_dof': 4}, use_phase=True)

    with pytest.raises(ValueError, match='preserve_noise_bias'):
        check_svht_phase({'preserve_noise_bias': True}, use_phase=True)


def test_can_svht_denoise():
    assert not can_svht_denoise(2)
    assert can_svht_denoise(3)


def test_check_svht_volumes():
    check_svht_volumes({}, SVHT_MAX_VOLUMES)
    check_svht_volumes({'extent': 7}, 342)

    with pytest.raises(ValueError, match=f'at most {SVHT_MAX_VOLUMES}'):
        check_svht_volumes({}, SVHT_MAX_VOLUMES + 1, bold_file='sub-01_task-rest_bold.nii.gz')

    with pytest.raises(ValueError, match='"extent" of 7'):
        check_svht_volumes({'extent': 7}, 343)


@pytest.mark.parametrize(
    ('params', 'n_volumes', 'spatial_shape', 'fits'),
    [
        # A 9-voxel cube on a 64x64x20 image fits; a 27-voxel one does not
        ({'shape': 'cube', 'extent': 9}, 300, (64, 64, 20), True),
        ({'shape': 'cube', 'extent': 27}, 300, (64, 64, 20), False),
        # Without an extent, svht_denoise's default (here 7, the smallest odd k with
        # k**3 > 300) must fit
        ({'shape': 'cube'}, 300, (64, 64, 7), True),
        ({'shape': 'cube'}, 300, (64, 64, 6), False),
        # Spherical patches are sized in mm, and not checked against the grid
        ({'shape': 'sphere', 'extent': 27}, 300, (64, 64, 20), True),
        ({'extent': 27}, 300, (64, 64, 20), True),
        # Without the image's dimensions, nothing is checked
        ({'shape': 'cube', 'extent': 27}, 300, None, True),
    ],
)
def test_check_svht_volumes_cube_fits(params, n_volumes, spatial_shape, fits):
    if fits:
        check_svht_volumes(params, n_volumes, spatial_shape=spatial_shape)
        return

    with pytest.raises(ValueError, match=r'too small for the \d+-voxel cubic patch'):
        check_svht_volumes(params, n_volumes, spatial_shape=spatial_shape)


@pytest.mark.parametrize(
    ('n_volumes', 'extent'),
    [(2, 3), (26, 3), (27, 5), (124, 5), (125, 7), (243, 7), (4096, 17)],
)
def test_svht_auto_extent(n_volumes, extent):
    """Test the default extent against svht_denoise's rule: smallest odd k >= 3, k**3 > n."""
    assert svht_auto_extent(n_volumes) == extent


@pytest.fixture
def clear_version_cache():
    svht_denoise_version.cache_clear()
    yield
    svht_denoise_version.cache_clear()


@pytest.mark.usefixtures('clear_version_cache')
def test_svht_denoise_version(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'svht_denoise'
    script.write_text(f'#!{sys.executable}\nprint("svht_denoise 0.1.20261001")\n')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    assert svht_denoise_version() == '0.1.20261001'


@pytest.mark.usefixtures('clear_version_cache')
def test_svht_denoise_version_missing(monkeypatch):
    def _missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(denoise.subprocess, 'run', _missing)
    assert svht_denoise_version() is None


@pytest.mark.parametrize(
    ('params', 'use_phase', 'multiecho', 'present', 'absent'),
    [
        (
            {},
            False,
            False,
            ['variance-stabilized', 'sized automatically', 'spherical', '@gavish2017'],
            ['rotated onto the real axis', 'Each echo', 'receive channels'],
        ),
        (
            {},
            True,
            True,
            ['rotated onto the real axis [@nordic]', 'Each echo', 'absolute value'],
            ['variance-stabilized', 'noise-floor bias'],
        ),
        (
            {'shape': 'cube', 'extent': 9, 'filter_method': 'optthresh', 'noise_dof': 8},
            False,
            False,
            ['9x9x9 voxels', 'optimal hard threshold', 'assuming 8 receive channels'],
            ['@gavish2017'],
        ),
        (
            {'vst': False},
            False,
            False,
            ['The magnitude data were denoised'],
            ['variance-stabiliz'],
        ),
        (
            {'preserve_noise_bias': True},
            False,
            False,
            ['variance-stabilized'],
            ['noise-floor bias'],
        ),
        (
            {'extent': 7, 'aggregator': 'exclusive'},
            False,
            False,
            ['holding at least 343 voxels', 'solely from the patch centered on it'],
            ['@manjon2013'],
        ),
    ],
)
def test_describe_svht(params, use_phase, multiecho, present, absent):
    desc = describe_svht(params, use_phase=use_phase, multiecho=multiecho, version='1.2.3')
    assert desc.startswith('Before any other processing')
    assert '`svht_denoise` 1.2.3 [@svht_denoise]' in desc
    for text in present:
        assert text in desc

    for text in absent:
        assert text not in desc


def test_describe_svht_citations_exist():
    """Test that every citation the description can make is in the boilerplate bib file."""
    import re

    from fmriprep import data

    bib = data.load.readable('boilerplate.bib').read_text()
    bib_keys = set(re.findall(r'^@\w+\{([^,]+),', bib, flags=re.MULTILINE))

    cited = set()
    for filter_method in ('optshrink', 'optthresh', 'truncate'):
        for aggregator in ('gaussian', 'uniform', 'exclusive'):
            for use_phase in (False, True):
                params = {'filter_method': filter_method, 'aggregator': aggregator}
                desc = describe_svht(params, use_phase=use_phase)
                cited |= set(re.findall(r'@([A-Za-z0-9_]+)', desc))

    assert cited
    assert cited <= bib_keys, cited - bib_keys
    # Without a version, the software is named alone
    assert '`svht_denoise` [@svht_denoise]' in describe_svht({}, use_phase=False)
