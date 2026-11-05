"""Tests for the svht_denoise interface and its report."""

import os
import sys
import textwrap
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline import engine as pe

from fmriprep.interfaces.denoise import (
    SVHTDenoise,
    noise_contour_levels,
    select_report_volume,
)


def _write_image(path, shape=(4, 4, 4, 3)):
    nb.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)).to_filename(str(path))
    return str(path)


def _write_bold(path, n_volumes=10, seed=0):
    """Write a small 4D series: a bright ellipsoid on a dark, noisy background."""
    rng = np.random.default_rng(seed)
    shape = (24, 24, 12)
    ijk = np.indices(shape).astype(float)
    center = (np.array(shape) - 1) / 2
    radii = np.array(shape) / 3
    inside = (((ijk - center[:, None, None, None]) / radii[:, None, None, None]) ** 2).sum(0) <= 1
    data = np.where(inside, 1000.0, 20.0)[..., None] + rng.normal(0, 10, shape + (n_volumes,))
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    nb.Nifti1Image(np.abs(data).astype(np.float32), affine).to_filename(str(path))
    return str(path)


def _options(cmdline):
    """Split a command line into its option tokens, after the input and output."""
    return cmdline.split()[3:]


@pytest.fixture
def fake_svht(tmp_path, monkeypatch):
    """Put a stand-in ``svht_denoise`` on the PATH.

    It writes the outputs svht_denoise would: the "denoised" series (the input, signed when
    ``-phase`` is given), a 3D noise map, and a rank map when ``-rank`` is given.
    """
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'svht_denoise'
    script.write_text(
        f'#!{sys.executable}\n'
        + textwrap.dedent(
            """
            import sys

            import nibabel as nb
            import numpy as np

            args = sys.argv[1:]
            in_file, out_file = args[0], args[1]
            opts = dict(zip(args[2::2], args[3::2]))
            img = nb.load(in_file)
            data = img.get_fdata(dtype=np.float32)
            if '-phase' in opts:
                data = -data
            nb.Nifti1Image(data, img.affine).to_filename(out_file)
            noise = np.linspace(1, 2, data[..., 0].size, dtype=np.float32)
            noise = noise.reshape(data.shape[:3])
            nb.Nifti1Image(noise, img.affine).to_filename(opts['-noise'])
            if '-rank' in opts:
                rank = np.full(data.shape[:3], 2, dtype=np.float32)
                nb.Nifti1Image(rank, img.affine).to_filename(opts['-rank'])
            """
        )
    )
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return script


def test_svht_denoise_cmdline(tmp_path):
    in_file = _write_image(tmp_path / 'bold.nii.gz')
    phase_file = _write_image(tmp_path / 'phase.nii.gz')

    denoise = SVHTDenoise(in_file=in_file, nthreads=2)
    assert denoise.cmdline == (
        f'svht_denoise {in_file} bold_denoised.nii.gz -noise bold_noise.nii.gz -nthreads 2'
    )

    denoise = SVHTDenoise(in_file=in_file, phase_file=phase_file, phase_units='radians', extent=7)
    cmdline = denoise.cmdline.split()
    assert cmdline[:3] == ['svht_denoise', in_file, 'bold_denoised.nii.gz']
    assert {'-phase', phase_file, '-phaseunits', 'radians', '-extent', '7'} <= set(cmdline)


def test_svht_denoise_phase_units_need_phase(tmp_path):
    """Test that phase_units without phase_file is refused."""
    in_file = _write_image(tmp_path / 'bold.nii.gz')
    denoise = SVHTDenoise(in_file=in_file, phase_units='radians')
    with pytest.raises(ValueError, match='requires'):
        denoise.cmdline  # noqa: B018


def test_svht_denoise_writes_the_rank_only_when_asked(tmp_path, monkeypatch):
    """Test that -rank is passed and its output listed only when rank_image is set."""
    monkeypatch.chdir(tmp_path)
    in_file = _write_image(tmp_path / 'bold.nii.gz')

    assert '-rank' not in SVHTDenoise(in_file=in_file).cmdline

    denoise = SVHTDenoise(in_file=in_file, rank_image='rank.nii.gz')
    assert '-rank rank.nii.gz' in denoise.cmdline
    assert denoise._list_outputs()['rank_image'] == str(tmp_path / 'rank.nii.gz')


def test_svht_denoise_cmdline_config_options(tmp_path):
    in_file = _write_image(tmp_path / 'bold.nii.gz')

    denoise = SVHTDenoise(
        in_file=in_file,
        filter_method='optthresh',
        shape='cube',
        aggregator='gaussian',
        aggregator_fwhm=1.5,
        stride=1,
        vst=True,
        noise_dof=4,
        preserve_noise_bias=True,
    )
    cmdline = ' '.join(_options(denoise.cmdline))
    for option in (
        '-filter optthresh',
        '-shape cube',
        '-aggregator gaussian',
        '-aggregator_fwhm 1.5',
        '-stride 1',
        '-vst y',
        '-noise_dof 4',
        '-preserve_noise_bias',
    ):
        assert option in cmdline

    # svht_denoise spells false as n, and a false bare flag is left out
    denoise = SVHTDenoise(in_file=in_file, vst=False, preserve_noise_bias=False)
    cmdline = ' '.join(_options(denoise.cmdline))
    assert '-vst n' in cmdline
    assert '-preserve_noise_bias' not in cmdline


def test_select_report_volume_is_reproducible():
    """Test that the report volume is fixed per file name, in range, and varies by name."""
    names = [f'/data/sub-01_task-rest_run-{run}_bold.nii.gz' for run in range(20)]
    picks = [select_report_volume(name, 300) for name in names]

    assert all(0 <= pick < 300 for pick in picks)
    assert picks == [select_report_volume(name, 300) for name in names]
    # Only the base name matters, so a moved file shows the same volume
    assert select_report_volume('/elsewhere/sub-01_task-rest_run-0_bold.nii.gz', 300) == picks[0]
    assert len(set(picks)) > 1
    assert select_report_volume(names[0], 1) == 0


def test_noise_contour_levels():
    noise = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.ones_like(noise, dtype=bool)
    levels = noise_contour_levels(noise, mask)
    assert levels == pytest.approx(np.percentile(noise, (25, 50, 75)))
    assert levels == sorted(levels)

    # Out-of-mask and non-finite values are ignored
    noise_nan = noise.copy()
    noise_nan[0, 0] = np.nan
    noise_nan[9, 9] = 1e6
    mask_partial = mask.copy()
    mask_partial[9, 9] = False
    expected = np.percentile(noise[mask_partial][1:], (25, 50, 75))
    assert noise_contour_levels(noise_nan, mask_partial) == pytest.approx(expected)

    # Nothing to contour in a constant map or an empty mask
    assert noise_contour_levels(np.ones((4, 4)), np.ones((4, 4), dtype=bool)) == []
    assert noise_contour_levels(noise, np.zeros_like(mask)) == []


@pytest.mark.parametrize('use_phase', [False, True])
def test_svht_denoise_run_and_report(tmp_path, fake_svht, use_phase):
    """Test that a run lists its outputs and writes a before/after report."""
    in_file = _write_bold(tmp_path / 'sub-01_task-rest_bold.nii.gz')
    inputs = {'in_file': in_file, 'rank_image': 'rank.nii.gz', 'nthreads': 1}
    if use_phase:
        inputs['phase_file'] = _write_bold(tmp_path / 'phase.nii.gz', seed=1)
        inputs['phase_units'] = 'radians'

    node = pe.Node(SVHTDenoise(**inputs), name='denoise', base_dir=str(tmp_path))
    result = node.run()

    outputs = result.outputs
    for name in ('out_file', 'noise_image', 'rank_image', 'out_report'):
        assert os.path.isfile(getattr(outputs, name)), name

    index = select_report_volume(in_file, 10)
    report = Path(outputs.out_report).read_text()
    assert f'Before denoising: volume {index}' in report
    assert f'After denoising: volume {index}' in report
    # The two frames of the flicker
    assert 'moving-image' in report
    assert 'fixed-image' in report
