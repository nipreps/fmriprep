import json
import os
import sys
import textwrap
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

os.environ['NO_ET'] = '1'

#: A stand-in for svht_denoise. It writes the outputs svht_denoise would: the "denoised"
#: series (the input, negated when -phase is given, so the output is signed), a 3D noise
#: map, and a rank map when -rank is given.
_FAKE_SVHT = textwrap.dedent(
    """
    import sys

    import nibabel as nb
    import numpy as np

    args = sys.argv[1:]
    if args == ['--version']:
        print('svht_denoise 0.0.test')
        sys.exit(0)

    in_file, out_file = args[0], args[1]
    opts = dict(zip(args[2::2], args[3::2]))
    img = nb.load(in_file)
    data = img.get_fdata(dtype=np.float32)
    if '-phase' in opts:
        data = -data

    nb.Nifti1Image(data, img.affine).to_filename(out_file)
    noise = np.linspace(1, 2, data[..., 0].size, dtype=np.float32).reshape(data.shape[:3])
    nb.Nifti1Image(noise, img.affine).to_filename(opts['-noise'])
    if '-rank' in opts:
        rank = np.full(data.shape[:3], 2, dtype=np.float32)
        nb.Nifti1Image(rank, img.affine).to_filename(opts['-rank'])
    """
)


@pytest.fixture
def fake_svht(tmp_path, monkeypatch):
    """Put a stand-in ``svht_denoise`` on the PATH, and clear the cached version."""
    from fmriprep.utils.denoise import svht_denoise_version

    bin_dir = tmp_path / 'fake_svht_bin'
    bin_dir.mkdir()
    script = bin_dir / 'svht_denoise'
    script.write_text(f'#!{sys.executable}\n{_FAKE_SVHT}')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    svht_denoise_version.cache_clear()
    yield script
    svht_denoise_version.cache_clear()


@pytest.fixture(scope='session', autouse=True)
def _legacy_printoptions():
    np.set_printoptions(legacy='1.21')


@pytest.fixture
def minimal_bids(tmp_path):
    bids = tmp_path / 'bids'
    bids.mkdir()
    Path.write_text(
        bids / 'dataset_description.json', json.dumps({'Name': 'Test DS', 'BIDSVersion': '1.8.0'})
    )
    T1w = bids / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz'
    T1w.parent.mkdir(parents=True)
    nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4)).to_filename(T1w)
    return bids
