from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nibabel.tmpdirs import InGivenDirectory
from scipy.spatial.transform import Rotation

from fmriprep.interfaces import nitransforms as fin
from fmriprep.utils.transforms import ras_to_volreg

BOLDREF = 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'


def test_ConvertAffine(tmp_path, data_dir):
    bold = data_dir / 'sub-pixar008_task-pixar_desc-coreg_boldref.nii.gz'
    anat = data_dir / 'sub-pixar008_desc-preproc_T1w.nii.gz'

    # lta_affine = nt.linear.load(data_dir / 'mri_coreg.lta')
    lta_convert_fsl = nt.linear.load(
        data_dir / 'mri_coreg-lta_convert.mat', moving=bold, reference=anat, fmt='fsl'
    )
    lta_convert_itk = nt.linear.load(data_dir / 'mri_coreg-lta_convert.txt')
    c3d_itk = nt.linear.load(data_dir / 'mri_coreg-c3d.txt')
    lta_convert_itk_inv = nt.linear.load(data_dir / 'mri_coreg-lta_convert-invert.txt')
    c3d_itk_inv = nt.linear.load(data_dir / 'mri_coreg-c3d-invert.txt')

    with InGivenDirectory(tmp_path):
        lta_to_fsl = fin.ConvertAffine(
            in_xfm=data_dir / 'mri_coreg.lta',
            reference=anat,
            moving=bold,
            out_fmt='fsl',
        ).run()
        assert lta_to_fsl.outputs.out_xfm == str(tmp_path / 'mri_coreg_fwd.mat')
        assert not lta_to_fsl.outputs.out_inv

        nitransforms_fsl = nt.linear.load(
            lta_to_fsl.outputs.out_xfm, moving=bold, reference=anat, fmt='fsl'
        )
        assert np.allclose(nitransforms_fsl.matrix, lta_convert_fsl.matrix, atol=1e-4)

        lta_to_itk = fin.ConvertAffine(in_xfm=data_dir / 'mri_coreg.lta', inverse=True).run()
        assert lta_to_itk.outputs.out_xfm == str(tmp_path / 'mri_coreg_fwd.txt')
        assert lta_to_itk.outputs.out_inv == str(tmp_path / 'mri_coreg_inv.txt')

        nitransforms_itk = nt.linear.load(lta_to_itk.outputs.out_xfm)
        assert np.allclose(nitransforms_itk.matrix, lta_convert_itk.matrix, atol=1e-4)
        assert np.allclose(nitransforms_itk.matrix, c3d_itk.matrix, atol=1e-4)

        nitransforms_itk_inv = nt.linear.load(lta_to_itk.outputs.out_inv)
        assert np.allclose(nitransforms_itk_inv.matrix, lta_convert_itk_inv.matrix, atol=1e-4)
        assert np.allclose(nitransforms_itk_inv.matrix, c3d_itk_inv.matrix, atol=1e-4)

        fsl_to_itk = fin.ConvertAffine(
            in_xfm=data_dir / 'mri_coreg-lta_convert.mat',
            reference=anat,
            moving=bold,
            out_fmt='itk',
            inverse=True,
        ).run()

        nitransforms_itk = nt.linear.load(fsl_to_itk.outputs.out_xfm)
        assert np.allclose(nitransforms_itk.matrix, lta_convert_itk.matrix, atol=1e-4)
        assert np.allclose(nitransforms_itk.matrix, c3d_itk.matrix, atol=1e-4)

        nitransforms_itk_inv = nt.linear.load(fsl_to_itk.outputs.out_inv)
        assert np.allclose(nitransforms_itk_inv.matrix, lta_convert_itk_inv.matrix, atol=1e-4)
        assert np.allclose(nitransforms_itk_inv.matrix, c3d_itk_inv.matrix, atol=1e-4)


def _as_ail(img):
    """The image stored as AIL: L/R on the last voxel axis, with a positive determinant."""
    return img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt('AIL')))


def _as_oblique(img):
    """The image with its affine tilted by 0.2 rad about the RAS x axis, through its center."""
    center = (img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3]
    R = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()
    tilt = nb.affines.from_matvec(R, center - R @ center)
    return nb.Nifti1Image(np.asanyarray(img.dataobj), tilt @ img.affine, img.header)


VOLREG_CASES = [
    pytest.param('volreg_large_motion', None, id='las'),
    pytest.param('volreg_large_motion_ail', _as_ail, id='ail'),
    pytest.param('volreg_large_motion_oblique', _as_oblique, id='oblique'),
]


def _boldref(data_dir, tmp_path, transform):
    """The test boldref, or a variant of it written to ``tmp_path``."""
    if transform is None:
        return data_dir / BOLDREF
    boldref = tmp_path / 'boldref.nii.gz'
    transform(nb.load(data_dir / BOLDREF)).to_filename(boldref)
    return boldref


@pytest.mark.parametrize(('volreg_dir', 'transform'), VOLREG_CASES)
def test_VolregToITK(tmp_path, data_dir, volreg_dir, transform):
    """The stored transforms reproduce 3dvolreg's -1Dfile, and agree with -1Dmatrix_save."""
    boldref = _boldref(data_dir, tmp_path, transform)
    volreg_out = data_dir / volreg_dir

    result = fin.VolregToITK(
        in_params=volreg_out / 'volreg.1D',
        in_matrices=volreg_out / 'volreg.aff12.1D',
        reference=boldref,
    ).run(cwd=tmp_path)

    out_file = Path(result.outputs.out_file)
    assert out_file.parent == tmp_path
    assert 'AffineTransform_double_3_3' in out_file.read_text()
    params = ras_to_volreg(nt.linear.load(out_file).matrix, nb.load(boldref))
    assert np.allclose(params, np.loadtxt(volreg_out / 'volreg.1D'), atol=1e-5)


def test_VolregToITK_single_volume(tmp_path, data_dir):
    """A one-volume run gives a one-transform file."""
    volreg_out = data_dir / 'volreg_large_motion'
    in_params = tmp_path / 'one.1D'
    in_params.write_text((volreg_out / 'volreg.1D').read_text().splitlines()[1] + '\n')
    header, _, second, *_ = (volreg_out / 'volreg.aff12.1D').read_text().splitlines()
    in_matrices = tmp_path / 'one.aff12.1D'
    in_matrices.write_text(f'{header}\n{second}\n')

    result = fin.VolregToITK(
        in_params=in_params, in_matrices=in_matrices, reference=data_dir / BOLDREF
    ).run(cwd=tmp_path)

    params = ras_to_volreg(
        nt.linear.load(result.outputs.out_file).matrix, nb.load(data_dir / BOLDREF)
    )
    assert np.allclose(params, np.loadtxt(in_params, ndmin=2), atol=1e-5)


def test_VolregToITK_crosscheck_fails(tmp_path, data_dir):
    """Matrices that disagree with the parameters are an error, naming the volume."""
    volreg_out = data_dir / 'volreg_large_motion'
    lines = (volreg_out / 'volreg.aff12.1D').read_text().splitlines()
    # Line 0 is a comment, so volume 3 is line 4; move its first translation by 0.01 mm
    values = lines[4].split()
    values[3] = f'{float(values[3]) + 0.01:.6g}'
    lines[4] = ' '.join(values)
    in_matrices = tmp_path / 'perturbed.aff12.1D'
    in_matrices.write_text('\n'.join(lines) + '\n')

    with pytest.raises(RuntimeError, match='at volume 3'):
        fin.VolregToITK(
            in_params=volreg_out / 'volreg.1D',
            in_matrices=in_matrices,
            reference=data_dir / BOLDREF,
        ).run(cwd=tmp_path)


def test_VolregToITK_rejects_mismatched_inputs(tmp_path, data_dir):
    volreg_out = data_dir / 'volreg_large_motion'
    params = np.loadtxt(volreg_out / 'volreg.1D')

    short = tmp_path / 'short.1D'
    np.savetxt(short, params[:-1], fmt='%8.4f')
    with pytest.raises(ValueError, match=r'9 rows, but .* 10 matrices'):
        fin.VolregToITK(
            in_params=short,
            in_matrices=volreg_out / 'volreg.aff12.1D',
            reference=data_dir / BOLDREF,
        ).run(cwd=tmp_path)

    five_columns = tmp_path / 'five.1D'
    np.savetxt(five_columns, params[:, :5], fmt='%8.4f')
    with pytest.raises(ValueError, match='5 columns'):
        fin.VolregToITK(in_params=five_columns, reference=data_dir / BOLDREF).run(cwd=tmp_path)
