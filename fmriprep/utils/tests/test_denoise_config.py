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
    load_svht_config,
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
        ({'extent': 8}, 'odd positive integer'),
        ({'extent': -1}, 'odd positive integer'),
        ({'extent': SVHT_MAX_EXTENT + 2}, f'no greater than {SVHT_MAX_EXTENT}'),
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
