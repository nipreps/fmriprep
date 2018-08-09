import time
import numpy as np
import nibabel as nb
from nipype.interfaces import nilearn as nl
from .. import images as im

import pytest


@pytest.mark.parametrize('nvols, nmasks, ext, factor', [
    (500, 10, '.nii', 2),
    (500, 10, '.nii.gz', 5),
    (200, 3, '.nii', 1.1),
    (200, 3, '.nii.gz', 2),
    (200, 10, '.nii', 1.1),
    (200, 10, '.nii.gz', 2),
    ])
def test_signal_extraction_equivalence(tmpdir, nvols, nmasks, ext, factor):
    tmpdir.chdir()

    vol_shape = (64, 64, 40)

    img_fname = 'img' + ext
    masks_fname = 'masks' + ext

    random_data = np.random.random(size=vol_shape + (nvols,)) * 2000
    random_mask_data = np.random.random(size=vol_shape + (nmasks,)) < 0.2

    nb.Nifti1Image(random_data, np.eye(4)).to_filename(img_fname)
    nb.Nifti1Image(random_mask_data.astype(np.uint8), np.eye(4)).to_filename(masks_fname)

    se1 = nl.SignalExtraction(in_file=img_fname, label_files=masks_fname,
                              class_labels=['a%d' % i for i in range(nmasks)],
                              out_file='nlsignals.tsv')
    se2 = im.SignalExtraction(in_file=img_fname, label_files=masks_fname,
                              class_labels=['a%d' % i for i in range(nmasks)],
                              out_file='imsignals.tsv')

    tic = time.time()
    se1.run()
    toc = time.time()
    se2.run()
    toc2 = time.time()

    tab1 = np.loadtxt('nlsignals.tsv', skiprows=1)
    tab2 = np.loadtxt('imsignals.tsv', skiprows=1)

    assert np.allclose(tab1, tab2)

    t1 = toc - tic
    t2 = toc2 - toc

    assert t2 < t1 / factor
