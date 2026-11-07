"""Tests for the svht_denoise interface and its report."""

import os
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline import engine as pe
from nipype.pipeline.engine.nodes import NodeExecutionError

from fmriprep.interfaces.denoise import (
    AbsoluteValue,
    SVHTDenoise,
    noise_contour_levels,
    select_report_volume,
)
from fmriprep.utils.testing import write_bold as _write_bold


def _write_image(path, shape=(4, 4, 4, 3)):
    nb.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)).to_filename(str(path))
    return str(path)


def _options(cmdline):
    """Split a command line into its option tokens, after the input and output."""
    return cmdline.split()[3:]


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


def test_select_report_volume_ignores_echo():
    """Test that every echo of a run gets the same volume, and other entities still count."""
    for run in range(10):
        echoes = [
            f'sub-01_task-rest_run-{run}_echo-{echo}_part-mag_bold.nii.gz' for echo in (1, 2, 5)
        ]
        assert len({select_report_volume(echo, 300) for echo in echoes}) == 1

    runs = [f'sub-01_task-rest_run-{run}_echo-1_bold.nii.gz' for run in range(20)]
    assert len({select_report_volume(run, 300) for run in runs}) > 1


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


def test_svht_denoise_report_volume(tmp_path, fake_svht):
    """Test that an explicit report_volume is shown, and that one out of range is refused."""
    in_file = _write_bold(tmp_path / 'sub-01_task-rest_bold.nii.gz')
    default = select_report_volume(in_file, 10)
    chosen = (default + 3) % 10

    node = pe.Node(
        SVHTDenoise(in_file=in_file, nthreads=1, report_volume=chosen),
        name='denoise',
        base_dir=str(tmp_path),
    )
    report = Path(node.run().outputs.out_report).read_text()
    assert f'Before denoising: volume {chosen}' in report
    assert f'volume {default}' not in report

    out_of_range = pe.Node(
        SVHTDenoise(in_file=in_file, nthreads=1, report_volume=10),
        name='denoise_out_of_range',
        base_dir=str(tmp_path),
    )
    with pytest.raises(NodeExecutionError, match='out of range for 10 volumes'):
        out_of_range.run()


def test_absolute_value(tmp_path):
    in_file = str(tmp_path / 'signed.nii.gz')
    data = np.array([[[[-2.0, 1.0], [0.5, -0.25]]]], dtype=np.float32)
    nb.Nifti1Image(data, np.eye(4)).to_filename(in_file)

    node = pe.Node(AbsoluteValue(in_file=in_file), name='absolute', base_dir=str(tmp_path))
    result = node.run()

    out_img = nb.load(result.outputs.out_file)
    assert out_img.get_data_dtype() == np.float32
    assert np.array_equal(out_img.get_fdata(), np.abs(data))
