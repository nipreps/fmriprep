import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fmriprep.utils.transforms import ras_to_volreg, volreg_to_ras, write_itk_affines


def _grid(oblique=False, shift=(0.0, 0.0, 0.0)):
    """A 48x52x30 LAS image with anisotropic voxels, optionally oblique or shifted."""
    affine = nb.affines.from_matvec(np.diag([-2.5, 2.5, 3.0]), np.add([60.0, -70.0, -40.0], shift))
    if oblique:
        tilt = Rotation.from_rotvec([0.15, -0.1, 0.05]).as_matrix()
        affine = nb.affines.from_matvec(tilt) @ affine
    return nb.Nifti1Image(np.zeros((48, 52, 30), dtype='float32'), affine)


def _random_params(n=20, seed=0):
    """Rows of roll pitch yaw (deg, up to 6) and dS dL dP (mm, up to 4)."""
    rng = np.random.default_rng(seed)
    return np.hstack([rng.uniform(-6, 6, (n, 3)), rng.uniform(-4, 4, (n, 3))])


@pytest.mark.parametrize('oblique', [False, True], ids=['cardinal', 'oblique'])
@pytest.mark.parametrize('shifted_base', [False, True], ids=['same-base', 'shifted-base'])
def test_volreg_roundtrip(oblique, shifted_base):
    reference = _grid(oblique)
    base = _grid(oblique, shift=(4.0, -3.0, 2.0)) if shifted_base else None
    params = _random_params()

    ras = volreg_to_ras(params, reference, base)

    assert ras.shape == (len(params), 4, 4)
    assert np.allclose(ras_to_volreg(ras, reference, base), params, atol=1e-10)


@pytest.mark.parametrize('oblique', [False, True], ids=['cardinal', 'oblique'])
def test_volreg_zero_motion_is_identity(oblique):
    ras = volreg_to_ras(np.zeros((3, 6)), _grid(oblique))
    assert np.allclose(ras, np.eye(4), atol=1e-12)


def test_volreg_single_volume():
    """One row of parameters, or one affine, is accepted without a leading axis."""
    reference = _grid()
    params = _random_params(n=1)

    ras = volreg_to_ras(params[0], reference)

    assert ras.shape == (1, 4, 4)
    assert np.allclose(ras_to_volreg(ras[0], reference), params, atol=1e-10)


def test_volreg_storage_orientation():
    """Reversing a voxel axis in storage changes neither the world grid nor the transforms."""
    reference = _grid(oblique=True)
    flipped = reference.as_reoriented([[0, -1], [1, 1], [2, 1]])
    params = _random_params()

    assert np.allclose(
        volreg_to_ras(params, flipped), volreg_to_ras(params, reference), atol=1e-10
    )


def _itk_parameters(path):
    return np.array(
        [
            line.split(':', 1)[1].split()
            for line in path.read_text().splitlines()
            if line.startswith('Parameters:')
        ],
        dtype=float,
    )


def test_write_itk_affines(tmp_path):
    ras = volreg_to_ras(_random_params(), _grid(oblique=True))
    out_file = tmp_path / 'hmc.txt'

    write_itk_affines(ras, out_file)

    assert out_file.read_text().count('Transform: AffineTransform_double_3_3') == len(ras)
    assert np.allclose(nt.linear.load(out_file).matrix, ras, rtol=1e-6, atol=1e-6)

    # The same on-disk convention as nitransforms' own (6-digit) ITK writer
    nt_file = tmp_path / 'nitransforms.txt'
    nt.linear.LinearTransformsMapping(ras).to_filename(nt_file, fmt='itk')
    assert np.allclose(_itk_parameters(out_file), _itk_parameters(nt_file), atol=1e-4)
