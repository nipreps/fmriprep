# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2026 The NiPreps Developers <nipreps@gmail.com>
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
"""Tests for the CAT-Surface interfaces."""

import sys
import types

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest

from ..catsurf import CATBBReg, CATVolRealign, _write_itk


def _translation(x):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix


def test_write_itk_pulls(tmp_path):
    """The BOLD-to-T1w ITK transform maps T1w points into the BOLD reference."""
    bold_to_t1 = _translation(5.0)  # moving (BOLD) RAS -> fixed (T1w) RAS
    out = _write_itk(np.linalg.inv(bold_to_t1), tmp_path / 'xfm.txt')

    point = np.array([[10.0, -20.0, 30.0]])
    mapped = nt.linear.load(out, fmt='itk').map(point)
    assert np.allclose(mapped, [[5.0, -20.0, 30.0]])


@pytest.fixture
def bbreg_inputs(tmp_path):
    img = nb.Nifti1Image(np.zeros((4, 4, 4), dtype='float32'), np.eye(4))
    for name in ('bold', 't1w'):
        img.to_filename(tmp_path / f'{name}.nii.gz')
    coords = np.eye(3, dtype='float32')
    faces = np.array([[0, 1, 2]], dtype='int32')
    files = {'white': [], 'cortex_mask': [], 'thickness': []}
    for hemi in 'LR':
        surf = nb.GiftiImage(
            darrays=[
                nb.gifti.GiftiDataArray(coords, intent='NIFTI_INTENT_POINTSET'),
                nb.gifti.GiftiDataArray(faces, intent='NIFTI_INTENT_TRIANGLE'),
            ]
        )
        files['white'].append(str(tmp_path / f'hemi-{hemi}_white.surf.gii'))
        surf.to_filename(files['white'][-1])
        for field, values in (('cortex_mask', [1, 1, 0]), ('thickness', [2.5, 2.5, 2.5])):
            files[field].append(str(tmp_path / f'hemi-{hemi}_{field}.gii'))
            nb.GiftiImage(
                darrays=[nb.gifti.GiftiDataArray(np.array(values, dtype='float32'))]
            ).to_filename(files[field][-1])
    return {
        'in_file': str(tmp_path / 'bold.nii.gz'),
        't1w_preproc': str(tmp_path / 't1w.nii.gz'),
        **files,
    }


@pytest.mark.parametrize(
    ('use_bbr', 'bbr_shift', 'fallback'),
    [
        (True, 2.0, False),
        (False, 2.0, True),
        (None, 2.0, False),
        (None, 40.0, True),  # BBR moved too far from the initialization
    ],
)
def test_cat_bbreg_fallback(tmp_path, monkeypatch, bbreg_inputs, use_bbr, bbr_shift, fallback):
    """BBR is returned unless disabled or too far from the volume registration."""
    calls = []

    def bbreg(volume_file, lh_surface=None, **kwargs):
        calls.append(kwargs)
        if lh_surface is None:  # volume registration only
            return _translation(1.0), 0.0
        return _translation(bbr_shift), 0.1

    monkeypatch.setitem(sys.modules, 'cat_surf', types.SimpleNamespace(bbreg=bbreg))
    monkeypatch.chdir(tmp_path)
    result = CATBBReg(use_bbr=use_bbr, **bbreg_inputs).run()

    assert result.outputs.fallback is fallback
    expected = 1.0 if fallback else bbr_shift
    bold_to_t1 = np.linalg.inv(nt.linear.load(result.outputs.itk_bold_to_t1, fmt='itk').matrix)
    assert np.allclose(bold_to_t1, _translation(expected))
    t1_to_bold = nt.linear.load(result.outputs.itk_t1_to_bold, fmt='itk').matrix
    assert np.allclose(t1_to_bold, _translation(expected))

    bbr_calls = [call for call in calls if 'lh_mask' in call]
    assert len(bbr_calls) == (use_bbr is not False)
    for call in bbr_calls:
        # bbregister's sampling for BOLD contrast
        assert call['invert_contrast'] == 1
        assert call['wm_dist'] == 2.0
        assert call['gm_proj_frac'] == 0.5


@pytest.mark.parametrize('dof', [6, 9, 12])
@pytest.mark.parametrize('init', ['t1w', 't2w', 'header'])
def test_cat_bbreg_init_dof(tmp_path, monkeypatch, bbreg_inputs, init, dof):
    """The initialization registers to the chosen image, and dof reaches BBR."""
    calls = []

    def bbreg(volume_file, lh_surface=None, **kwargs):
        calls.append(kwargs)
        return _translation(1.0), 0.0

    t2w = tmp_path / 't2w.nii.gz'
    nb.Nifti1Image(np.zeros((4, 4, 4), dtype='float32'), np.eye(4)).to_filename(t2w)
    monkeypatch.setitem(sys.modules, 'cat_surf', types.SimpleNamespace(bbreg=bbreg))
    monkeypatch.chdir(tmp_path)
    CATBBReg(init=init, dof=dof, t2w_preproc=str(t2w), **bbreg_inputs).run()

    reference = {'t1w': bbreg_inputs['t1w_preproc'], 't2w': str(t2w), 'header': None}[init]
    assert [call['ref_file'] for call in calls] == [reference] * (1 if init == 'header' else 2)
    # an older cat-surf without affine BBR does not take dof
    assert calls[-1].get('dof') == (None if dof == 6 else dof)


def test_cat_bbreg_t2w_required(tmp_path, monkeypatch, bbreg_inputs):
    monkeypatch.setitem(
        sys.modules, 'cat_surf', types.SimpleNamespace(bbreg=lambda *a, **k: (np.eye(4), 0.0))
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='t2w_preproc'):
        CATBBReg(init='t2w', **bbreg_inputs).run()


def test_cat_vol_realign(tmp_path, monkeypatch):
    """Outputs are named and counted as mcflirt's."""
    pytest.importorskip('cat_surf')

    # Asymmetric, so that every rotation is determined
    grid = np.indices((32, 32, 32)).astype('float32')
    blobs = [
        ((12, 14, 16), (20, 30, 40), 100.0),
        ((20, 18, 12), (8, 12, 6), 60.0),
        ((16, 22, 20), (5, 5, 15), 40.0),
    ]
    vols = []
    for shift in (0.0, 0.0, 1.0, 0.0):
        vol = np.full(grid.shape[1:], 10.0, dtype='float32')
        for (cx, cy, cz), (sx, sy, sz), amp in blobs:
            vol += amp * np.exp(
                -(
                    (grid[0] - cx - shift) ** 2 / sx
                    + (grid[1] - cy) ** 2 / sy
                    + (grid[2] - cz) ** 2 / sz
                )
            )
        vols.append(vol)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nb.Nifti1Image(np.stack(vols, axis=-1), affine).to_filename(tmp_path / 'bold.nii.gz')
    nb.Nifti1Image(vols[0], affine).to_filename(tmp_path / 'ref.nii.gz')

    monkeypatch.chdir(tmp_path)
    result = CATVolRealign(
        in_file=str(tmp_path / 'bold.nii.gz'),
        ref_file=str(tmp_path / 'ref.nii.gz'),
        save_mats=True,
        save_plots=True,
        save_rms=True,
    ).run()

    assert len(result.outputs.mat_file) == 4
    par = np.loadtxt(result.outputs.par_file)
    assert par.shape == (4, 6)
    # Only the third volume is displaced, by one voxel (2 mm)
    assert np.abs(par[[0, 1, 3], 3:]).max() < 0.2
    assert np.abs(par[2, 3:]).max() > 1.0
    assert len(result.outputs.rms_files) == 2
