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
import os
import shutil
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nitransforms.resampling import apply
from scipy.spatial.transform import Rotation

from ..template import init_bold_template_wf


def _freesurfer_available():
    fs_home = os.getenv('FREESURFER_HOME')
    licenses = [os.getenv('FS_LICENSE')]
    if fs_home:
        licenses += [Path(fs_home) / 'license.txt', Path(fs_home) / '.license']
    has_license = any(path and Path(path).is_file() for path in licenses)
    return bool(shutil.which('mri_robust_template')) and has_license


def _boldrefs(path, gains):
    """Resample a BOLD reference at 2.4mm under random head motion and receive gains."""
    from templateflow.api import get

    truth = nb.load(get('MNI152NLin2009cAsym', resolution=2, desc='fMRIPrep', suffix='boldref'))
    zooms = np.array(truth.header.get_zooms()[:3])
    affine = truth.affine.copy()
    affine[:3, :3] *= 2.4 / zooms
    grid = nb.Nifti1Image(np.zeros(np.ceil(truth.shape * zooms / 2.4).astype(int)), affine)

    rng = np.random.default_rng(42)
    boldrefs = []
    for i, gain in enumerate(gains):
        motion = np.eye(4)
        motion[:3, :3] = Rotation.from_euler(
            'xyz', rng.uniform(-4, 4, 3), degrees=True
        ).as_matrix()
        motion[:3, 3] = rng.uniform(-3, 3, 3)
        run = apply(nt.linear.Affine(motion, reference=grid), truth, reference=grid)
        data = np.asanyarray(run.dataobj) * gain
        data += rng.normal(0, 0.02 * data.max(), data.shape)
        boldrefs.append(str(path / f'boldref{i}.nii.gz'))
        nb.Nifti1Image(np.clip(data, 0, None).astype('f4'), affine).to_filename(boldrefs[-1])
    return boldrefs


@pytest.mark.skipif(not _freesurfer_available(), reason='FreeSurfer and a license required')
@pytest.mark.parametrize(('upsample', 'tolerance'), [(None, 1e-3), ((1.2, 1.2, 1.2), 5e-3)])
def test_template_matches_freesurfer(tmp_path, upsample, tolerance):
    """The template reproduces the median of mri_robust_template, resampling each run once."""
    wf = init_bold_template_wf(num_bold_runs=3, upsample=upsample)
    wf.base_dir = str(tmp_path)
    wf.inputs.inputnode.boldref_files = _boldrefs(tmp_path, gains=[0.8, 1.0, 1.3])
    wf.run()

    freesurfer = nb.load(tmp_path / 'bold_template_wf/boldref_template/boldref_template.nii.gz')
    template = nb.load(next((tmp_path / 'bold_template_wf/resample_template').glob('*.nii.gz')))
    assert np.allclose(template.affine, freesurfer.affine)

    expected, actual = freesurfer.get_fdata(), template.get_fdata()
    brain = expected > 0.2 * np.percentile(expected, 99)
    nrmse = np.sqrt(np.mean((actual - expected)[brain] ** 2)) / expected[brain].mean()
    assert nrmse < tolerance


@pytest.mark.parametrize('upsample', [(1.2, 1.2, 1.2), (0.8, 0.8, 0.8)])
def test_upsampling_zooms(upsample):
    wf = init_bold_template_wf(num_bold_runs=2, upsample=upsample)

    assert wf.get_node('upsample_boldrefs').inputs.target_zooms == upsample
    assert f'upsampled to {upsample[0]:g}mm isotropic resolution' in wf.__desc__


def test_upsampling_can_be_disabled():
    """Without zooms, no upsampling happens and the description does not claim it."""
    wf = init_bold_template_wf(num_bold_runs=2, upsample=None)

    assert wf.get_node('upsample_boldrefs') is None
    assert 'upsampled' not in wf.__desc__
    assert wf.get_node('boldref_template')


@pytest.mark.parametrize(
    ('reference', 'template_runs', 'fixed', 'registered'),
    [
        (0, None, True, False),
        (None, None, False, False),
        (2, [False, True, True], True, False),
        (None, [True, False, True], False, True),
    ],
)
def test_template_selection(reference, template_runs, fixed, registered):
    """The reference fixes the template space; excluded runs are registered to unbiased ones."""
    wf = init_bold_template_wf(num_bold_runs=3, reference=reference, template_runs=template_runs)

    robust_template = wf.get_node('boldref_template').inputs
    assert robust_template.fixed_timepoint is fixed
    assert robust_template.no_iteration is fixed
    if fixed:
        assert robust_template.initial_timepoint == reference + 1
    assert (wf.get_node('register_excluded_runs') is not None) is registered
    if template_runs and not registered:
        assert wf.get_node('resample_template').inputs.indices == template_runs
    if template_runs:
        assert 'selected' in wf.__desc__


def test_merge_xfms_order():
    """Transforms of registered runs are returned in the original run order."""
    from ..template import _interleave

    assert _interleave(['b', 'd'], ['a', 'c'], [False, True, False, True]) == ['a', 'b', 'c', 'd']
