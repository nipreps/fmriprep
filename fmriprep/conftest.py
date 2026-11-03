import json
import os
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

os.environ['NO_ET'] = '1'


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
