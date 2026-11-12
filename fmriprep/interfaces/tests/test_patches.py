import shutil

import nibabel as nb
import numpy as np
import pytest
from nipype.interfaces.base import isdefined
from nipype.pipeline import engine as pe

from fmriprep.interfaces.patches import Volreg


@pytest.mark.skipif(shutil.which('3dvolreg') is None, reason='3dvolreg required')
@pytest.mark.parametrize('out_file', ['NULL', None])
def test_Volreg_run(tmp_path, out_file):
    """3dvolreg runs with the patched options, and writes a series only if asked."""
    rng = np.random.default_rng(0)
    data = rng.uniform(100, 200, (12, 12, 8, 3)).astype('float32')
    bold = tmp_path / 'bold.nii.gz'
    boldref = tmp_path / 'boldref.nii.gz'
    nb.Nifti1Image(data, np.diag([3.0, 3.0, 3.0, 1.0])).to_filename(bold)
    nb.Nifti1Image(data[..., 0], np.diag([3.0, 3.0, 3.0, 1.0])).to_filename(boldref)

    volreg = Volreg(
        in_file=str(bold), basefile=str(boldref), interp='heptic', twopass=True, zpad=4
    )
    if out_file is not None:
        volreg.inputs.out_file = out_file
    node = pe.Node(volreg, name='volreg', base_dir=str(tmp_path))
    outputs = node.run().outputs

    assert np.loadtxt(outputs.oned_file).shape == (3, 6)
    assert np.loadtxt(outputs.oned_matrix_save).shape == (3, 12)
    assert not isdefined(outputs.md1d_file)
    assert isdefined(outputs.out_file) == (out_file is None)
