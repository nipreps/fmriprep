from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nipype.pipeline import engine as pe
from niworkflows.interfaces.itk import MCFLIRT2ITK

from fmriprep.interfaces import confounds


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


def test_FSLRMSDeviation(tmp_path, data_dir):
    base = 'sub-01_task-mixedgamblestask_run-01'
    xfms = data_dir / f'{base}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
    boldref = data_dir / f'{base}_desc-hmc_boldref.nii.gz'
    timeseries = data_dir / f'{base}_desc-motion_timeseries.tsv'

    rmsd = pe.Node(
        confounds.FSLRMSDeviation(xfm_file=str(xfms), boldref_file=str(boldref)),
        name='rmsd',
        base_dir=str(tmp_path),
    )
    res = rmsd.run()

    orig = pd.read_csv(timeseries, sep='\t')['rmsd']
    derived = pd.read_csv(res.outputs.out_file, sep='\t')['rmsd']

    # RMSD is nominally in mm, so 0.1um is an acceptable deviation
    assert np.allclose(orig.values, derived.values, equal_nan=True, atol=1e-4)


def test_FSLMotionParams(tmp_path, data_dir):
    base = 'sub-01_task-mixedgamblestask_run-01'
    xfms = data_dir / f'{base}_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
    boldref = data_dir / f'{base}_desc-hmc_boldref.nii.gz'
    orig_timeseries = data_dir / f'{base}_desc-motion_timeseries.tsv'

    motion = pe.Node(
        confounds.FSLMotionParams(xfm_file=str(xfms), boldref_file=str(boldref)),
        name='fsl_motion',
        base_dir=str(tmp_path),
    )
    res = motion.run()

    derived_params = pd.read_csv(res.outputs.out_file, sep='\t')
    # orig_timeseries includes framewise_displacement
    orig_params = pd.read_csv(orig_timeseries, sep='\t')[derived_params.columns]

    # Motion parameters are in mm and rad
    # These are empirically determined bounds, but they seem reasonable
    # for the units
    limits = pd.DataFrame(
        {
            'trans_x': [1e-4],
            'trans_y': [1e-4],
            'trans_z': [1e-4],
            'rot_x': [1e-6],
            'rot_y': [1e-6],
            'rot_z': [1e-6],
        }
    )
    max_diff = (orig_params - derived_params).abs().max()
    assert np.all(max_diff < limits)


def _add_offset(img):
    """The image with 200 added to every voxel, as float32."""
    data = np.float32(img.dataobj) + 200
    return nb.Nifti1Image(data, img.affine, img.header)


def _as_ail(img):
    """The image stored as AIL: L/R on the last voxel axis, with a positive determinant."""
    return img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt('AIL')))


@pytest.mark.parametrize(
    ('mcflirt_dir', 'transform'),
    [
        pytest.param('mcflirt_large_motion', None, id='las'),
        pytest.param('mcflirt_large_motion', nb.as_closest_canonical, id='ras'),
        pytest.param('mcflirt_large_motion_offset', _add_offset, id='offset'),
        pytest.param('mcflirt_large_motion_ail', _as_ail, id='ail'),
    ],
)
def test_FSLMotionParams_mcflirt(tmp_path, data_dir, mcflirt_dir, transform):
    """FSLMotionParams reproduces MCFLIRT's .par from MCFLIRT's matrices.

    The fixtures are MCFLIRT outputs for a synthetic series with rotations
    up to 0.1 rad, registered to the test boldref (or a variant of it).
    """
    boldref_file = data_dir / 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'
    if transform is not None:
        boldref = transform(nb.load(boldref_file))
        boldref_file = tmp_path / 'boldref.nii.gz'
        boldref.to_filename(boldref_file)

    mcflirt_out = data_dir / mcflirt_dir
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
        confounds.FSLMotionParams(xfm_file=xfm_file, boldref_file=str(boldref_file)),
        name='fsl_motion',
        base_dir=str(tmp_path),
    )
    derived = pd.read_csv(motion.run().outputs.out_file, sep='\t')

    # MCFLIRT's .par columns: rot_x rot_y rot_z (rad), trans_x trans_y trans_z (mm)
    par = np.loadtxt(mcflirt_out / 'mcf.par')
    rot = derived[['rot_x', 'rot_y', 'rot_z']].to_numpy()
    trans = derived[['trans_x', 'trans_y', 'trans_z']].to_numpy()

    # MCFLIRT writes its matrices to 6 decimal places, which leaves about 1e-4 mm
    assert np.allclose(rot, par[:, :3], atol=1e-5)
    assert np.allclose(trans, par[:, 3:], atol=1e-3)


def test_FramewiseDisplacement(tmp_path, data_dir):
    timeseries = data_dir / 'sub-01_task-mixedgamblestask_run-01_desc-motion_timeseries.tsv'

    framewise_displacement = pe.Node(
        confounds.FramewiseDisplacement(in_file=str(timeseries)),
        name='framewise_displacement',
        base_dir=str(tmp_path),
    )
    res = framewise_displacement.run()

    orig = pd.read_csv(timeseries, sep='\t')['framewise_displacement']
    derived = pd.read_csv(res.outputs.out_file, sep='\t')['FramewiseDisplacement']

    assert np.allclose(orig.values, derived.values, equal_nan=True)
