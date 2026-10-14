from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import pandas as pd
import pytest
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nipype.pipeline import engine as pe
from niworkflows.interfaces.itk import MCFLIRT2ITK
from scipy.spatial.transform import Rotation

from fmriprep.interfaces import confounds
from fmriprep.utils.transforms import volreg_to_ras


def test_RenameACompCor(tmp_path, data_dir):
    renamer = pe.Node(confounds.RenameACompCor(), name='renamer', base_dir=str(tmp_path))
    renamer.inputs.components_file = data_dir / 'acompcor_truncated.tsv'
    renamer.inputs.metadata_file = data_dir / 'component_metadata_truncated.tsv'

    res = renamer.run()

    target_components = Path.read_text(data_dir / 'acompcor_renamed.tsv')
    target_meta = Path.read_text(data_dir / 'component_metadata_renamed.tsv')
    renamed_components = Path(res.outputs.components_file).read_text()
    renamed_meta = Path(res.outputs.metadata_file).read_text()
    assert renamed_components == target_components
    assert renamed_meta == target_meta


def test_FilterDropped(tmp_path, data_dir):
    filt = pe.Node(confounds.FilterDropped(), name='filt', base_dir=str(tmp_path))
    filt.inputs.in_file = data_dir / 'component_metadata_truncated.tsv'

    res = filt.run()

    target_meta = Path.read_text(data_dir / 'component_metadata_filtered.tsv')
    filtered_meta = Path(res.outputs.out_file).read_text()
    assert filtered_meta == target_meta


def _as_oblique(img):
    """The image with its affine tilted by 0.2 rad about the RAS x axis, through its center."""
    center = (img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3]
    R = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()
    tilt = nb.affines.from_matvec(R, center - R @ center)
    return nb.Nifti1Image(np.asanyarray(img.dataobj), tilt @ img.affine, img.header)


def _volreg_columns(motion_tsv):
    """AFNIMotionParams output in 3dvolreg's -1Dfile order and units."""
    derived = pd.read_csv(motion_tsv, sep='\t')
    assert list(derived.columns) == ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    rot = np.rad2deg(derived[['rot_z', 'rot_x', 'rot_y']].to_numpy())
    return np.hstack((rot, derived[['trans_z', 'trans_x', 'trans_y']].to_numpy()))


def _as_ail(img):
    """The image stored as AIL: L/R on the last voxel axis, with a positive determinant."""
    return img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt('AIL')))


@pytest.mark.parametrize(
    ('volreg_dir', 'transform'),
    [
        pytest.param('volreg_large_motion', None, id='las'),
        pytest.param('volreg_large_motion_ail', _as_ail, id='ail'),
        pytest.param('volreg_large_motion_oblique', _as_oblique, id='oblique'),
    ],
)
def test_AFNIMotionParams(tmp_path, data_dir, volreg_dir, transform):
    """AFNIMotionParams reproduces 3dvolreg's -1Dfile from 3dvolreg's -1Dmatrix_save.

    The transform comes from 3dvolreg's own matrices through nitransforms, not
    from VolregToITK, so this does not depend on fMRIPrep's conversion code.
    """
    boldref_file = data_dir / 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'
    if transform is not None:
        boldref = transform(nb.load(boldref_file))
        boldref_file = tmp_path / 'boldref.nii.gz'
        boldref.to_filename(boldref_file)

    volreg_out = data_dir / volreg_dir
    xfm_file = tmp_path / 'hmc.txt'
    nt.linear.load(
        volreg_out / 'volreg.aff12.1D', fmt='afni', reference=boldref_file, moving=boldref_file
    ).to_filename(xfm_file, fmt='itk')

    motion = pe.Node(
        confounds.AFNIMotionParams(xfm_file=str(xfm_file), boldref_file=str(boldref_file)),
        name='afni_motion',
        base_dir=str(tmp_path),
    )
    derived = _volreg_columns(motion.run().outputs.out_file)

    # 3dvolreg's -1Dfile columns: roll pitch yaw (deg), dS dL dP (mm)
    oned = np.loadtxt(volreg_out / 'volreg.1D')
    # -1Dmatrix_save has 6 significant digits, which leaves about 5e-5 deg and 7e-5 mm
    assert np.allclose(derived[:, :3], oned[:, :3], atol=2e-4)
    assert np.allclose(derived[:, 3:], oned[:, 3:], atol=1e-3)


def test_AFNIMotionParams_mcflirt_transforms(tmp_path, data_dir):
    """With transforms from MCFLIRT, as when --hmc 3dvolreg reuses precomputed ones,
    the parameters still describe the transforms."""
    boldref_file = data_dir / 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'
    mcflirt_out = data_dir / 'mcflirt_large_motion'
    fsl2itk = pe.Node(
        MCFLIRT2ITK(
            in_files=[str(mat) for mat in sorted(mcflirt_out.glob('MAT_*'))],
            in_reference=str(boldref_file),
            in_source=str(boldref_file),
        ),
        name='fsl2itk',
        base_dir=str(tmp_path),
    )
    xfm_file = fsl2itk.run().outputs.out_file

    motion = pe.Node(
        confounds.AFNIMotionParams(xfm_file=xfm_file, boldref_file=str(boldref_file)),
        name='afni_motion',
        base_dir=str(tmp_path),
    )
    oned = _volreg_columns(motion.run().outputs.out_file)

    # MCFLIRT's matrices are rigid only to their 6 printed decimals
    assert np.allclose(
        volreg_to_ras(oned, nb.load(boldref_file)), nt.linear.load(xfm_file).matrix, atol=1e-3
    )
