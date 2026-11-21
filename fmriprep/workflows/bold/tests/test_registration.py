# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
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
import re
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from scipy.spatial.transform import Rotation

from fmriprep.interfaces.nitransforms import ConvertAffine
from fmriprep.workflows.bold.registration import (
    _fs_init_lta,
    init_bbreg_wf,
    init_bold_reg_wf,
)


def _rigid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler('xyz', rng.uniform(-0.2, 0.2, 3)).as_matrix()
    affine[:3, 3] = rng.uniform(-10, 10, 3)
    return affine


def _lta_ras2ras(path: Path) -> np.ndarray:
    text = path.read_text()
    rows = re.search(r'1 4 4\n(.*?)\nsrc volume', text, re.DOTALL).group(1).split('\n')
    return np.array([[float(value) for value in row.split()] for row in rows[:4]])


def _lta_geometry(path: Path, which: str) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Shape, voxel size and vox2ras of the ``src`` or ``dst`` volume of an LTA file."""
    block = path.read_text().split(f'{which} volume info\n')[1]
    fields = dict(re.findall(r'^(\w+)[ \t]*=[ \t]*(.*)$', block, re.MULTILINE)[:8])
    shape = np.array(fields['volume'].split(), dtype=float)
    zooms = np.array(fields['voxelsize'].split(), dtype=float)
    axes = np.column_stack(
        [np.array(fields[key].split(), dtype=float) for key in ('xras', 'yras', 'zras')]
    )
    vox2ras = np.eye(4)
    vox2ras[:3, :3] = axes * zooms
    vox2ras[:3, 3] = np.array(fields['cras'].split(), dtype=float) - vox2ras[:3, :3] @ (shape / 2)
    return tuple(int(n) for n in shape), zooms, vox2ras


def _session_header_lta(tmp_path, bold, fsnative, bold2t1w, fsnative2t1w, subject_id='sub-01'):
    """Run the nodes of the session-header initialization on files; return the final LTA."""
    t1w = tmp_path / 't1w.nii.gz'
    nb.load(fsnative).to_filename(t1w)

    session_xfm = tmp_path / 'from-orig_to-T1w_mode-image_xfm.txt'
    nt.linear.Affine(np.linalg.inv(bold2t1w), reference=t1w).to_filename(session_xfm, fmt='itk')
    fsnative2t1w_xfm = tmp_path / 'from-fsnative_to-T1w_mode-image_xfm.txt'
    nt.linear.Affine(np.linalg.inv(fsnative2t1w), reference=t1w).to_filename(
        fsnative2t1w_xfm, fmt='itk'
    )

    inverse = ConvertAffine(in_xfm=str(fsnative2t1w_xfm), inverse=True).run(cwd=tmp_path)
    init_lta = ConcatenateXFMs(
        in_xfms=[inverse.outputs.out_inv, str(session_xfm)],
        out_fmt='fs',
        reference=str(fsnative),
        moving=str(bold),
    ).run(cwd=tmp_path)
    return Path(init_lta.outputs.out_xfm), Path(_fs_init_lta(init_lta.outputs.out_xfm, subject_id))


@pytest.mark.parametrize('freesurfer', [True, False])
def test_bold_reg_wf_session_header(freesurfer):
    kwargs = {
        'freesurfer': freesurfer,
        'use_bbr': None,
        'bold2anat_dof': 6,
        'bold2anat_init': 'session-header',
        'mem_gb': 1,
        'omp_nthreads': 1,
    }
    if freesurfer:
        wf = init_bold_reg_wf(**kwargs)
        bbreg_wf = wf.get_node('bbreg_wf')
        assert bbreg_wf.get_node('init_lta') is not None
        # bbregister must be initialized from the FreeSurfer-ready LTA, not the raw one
        edge = bbreg_wf._graph.get_edge_data(
            bbreg_wf.get_node('fs_init_lta'), bbreg_wf.get_node('bbregister')
        )
        assert ('out', 'init_reg_file') in edge['connect']
        assert not bbreg_wf._graph.has_edge(
            bbreg_wf.get_node('init_lta'), bbreg_wf.get_node('bbregister')
        )
    else:
        with pytest.raises(NotImplementedError):
            init_bold_reg_wf(**kwargs)


def test_bbreg_wf_session_header_requires_bbr():
    with pytest.raises(ValueError, match='Cannot disable BBR'):
        init_bbreg_wf(
            use_bbr=False,
            bold2anat_dof=6,
            bold2anat_init='session-header',
            omp_nthreads=1,
        )


def _write_images(tmp_path):
    """An oblique BOLD and a conformed (LIA) FreeSurfer-like volume with different grids."""
    bold_affine = _rigid(3)
    bold_affine[:3, :3] = bold_affine[:3, :3] @ np.diag([2.5, 2.5, 3.0])
    bold = nb.Nifti1Image(np.zeros((24, 24, 16), dtype=np.float32), bold_affine)
    fsnative_affine = np.array([[-1.0, 0, 0, 32], [0, 0, 1, -32], [0, -1, 0, 32], [0, 0, 0, 1]])
    fsnative = nb.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), fsnative_affine)
    paths = tmp_path / 'bold.nii.gz', tmp_path / 'fsnative.nii.gz'
    bold.to_filename(paths[0])
    fsnative.to_filename(paths[1])
    return paths


def test_session_header_init_lta(tmp_path, monkeypatch):
    """The initialization maps BOLD to FreeSurfer native space via the session transform."""
    monkeypatch.chdir(tmp_path)
    bold, fsnative = _write_images(tmp_path)
    bold2t1w = _rigid(1)
    fsnative2t1w = _rigid(2)

    raw_lta, fs_lta = _session_header_lta(tmp_path, bold, fsnative, bold2t1w, fsnative2t1w)
    bold2fsnative = np.linalg.inv(fsnative2t1w) @ bold2t1w

    # NiTransforms composes the right matrix (BOLD -> fsnative coordinates)...
    np.testing.assert_allclose(_lta_ras2ras(raw_lta), bold2fsnative, atol=1e-4)
    # ...and the FreeSurfer LTA keeps it, with BOLD as src and fsnative as dst
    np.testing.assert_allclose(_lta_ras2ras(fs_lta), bold2fsnative, atol=1e-4)
    src_shape, _, src_affine = _lta_geometry(fs_lta, 'src')
    dst_shape, _, dst_affine = _lta_geometry(fs_lta, 'dst')
    assert src_shape == nb.load(bold).shape
    assert dst_shape == nb.load(fsnative).shape
    np.testing.assert_allclose(src_affine, nb.load(bold).affine, atol=1e-4)
    np.testing.assert_allclose(dst_affine, nb.load(fsnative).affine, atol=1e-4)
    # bbregister reads the subject from the LTA (subject-unknown if unset), not from --s
    assert 'subject sub-01' in fs_lta.read_text().splitlines()
