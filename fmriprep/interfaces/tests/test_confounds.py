from pathlib import Path

import numpy as np
import pandas as pd
from nipype.pipeline import engine as pe

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
