#!/usr/bin/env python
"""
Generate the 3dvolreg outputs that the ``VolregToITK`` and ``AFNIMotionParams`` tests use.

The test boldref is moved by known rigid motions (rotations up to 0.1 rad,
translations up to 3 mm) to make a synthetic series, which ``3dvolreg``
registers back to the boldref with the options fMRIPrep's HMC workflow uses.
The ``-1Dfile`` and ``-1Dmatrix_save`` outputs of each run are written to
``fmriprep/interfaces/tests/data/volreg_large_motion*/`` as ``volreg.1D`` and
``volreg.aff12.1D``. Three variants are generated:

* ``volreg_large_motion``: the boldref as stored (LAS).
* ``volreg_large_motion_ail``: series and boldref stored as AIL.
* ``volreg_large_motion_oblique``: series and boldref with their affines tilted
  by 0.2 rad about the RAS x axis, through the image center.

The script checks that 3dvolreg's estimates include rotations about two axes
at once that are large enough for the tests to tell a correct decomposition
from a wrong one, and raises ``ValueError`` if not; pass another ``--seed``
to draw different motions.

``3dvolreg`` must be on the ``PATH``. The committed fixtures were made with
AFNI_26.3.03.
"""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nb
import nitransforms as nt
import nitransforms.resampling
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nipype.pipeline import engine as pe
from scipy.spatial.transform import Rotation

from fmriprep.interfaces.patches import Volreg

DATA_DIR = Path(__file__).resolve().parents[1] / 'fmriprep' / 'interfaces' / 'tests' / 'data'
BOLDREF = DATA_DIR / 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'

# Decomposing in the wrong Euler order errs by about the product of the two
# largest angles, so some volume needs a product well above the tests'
# tolerances (2e-4 deg, about 3.5e-6 rad).
MIN_ANGLE_PRODUCT = 1e-3


def make_series(boldref, n_vols=10, max_rot=0.1, max_trans=3.0, seed=0):
    """Move ``boldref`` by random rigid motions, about its center; volume 0 is unmoved."""
    rng = np.random.default_rng(seed)
    center = (boldref.affine @ np.r_[(np.array(boldref.shape[:3]) - 1) / 2, 1])[:3]
    vols = [boldref]
    for r, t in rng.uniform(-1, 1, (n_vols - 1, 2, 3)):
        R = Rotation.from_rotvec(r * max_rot).as_matrix()
        aff = nb.affines.from_matvec(R, center - R @ center + t * max_trans)
        vols.append(
            nt.resampling.apply(nt.linear.Affine(aff, reference=boldref), boldref, order=3)
        )
    series = nb.funcs.concat_images(vols)
    series.set_data_dtype('float32')
    return series


def as_ail(img):
    """The image stored as AIL: L/R on the last voxel axis, with a positive determinant."""
    return img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt('AIL')))


def as_oblique(img):
    """The image with its affine tilted by 0.2 rad about the RAS x axis, through its center."""
    center = (img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3]
    R = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()
    tilt = nb.affines.from_matvec(R, center - R @ center)
    return nb.Nifti1Image(np.asanyarray(img.dataobj), tilt @ img.affine, img.header)


VARIANTS = {
    'volreg_large_motion': lambda x: x,
    'volreg_large_motion_ail': as_ail,
    'volreg_large_motion_oblique': as_oblique,
}


def run_volreg(series, boldref, base_dir):
    """Register ``series`` to ``boldref`` with 3dvolreg, working in ``base_dir``."""
    base_dir.mkdir(parents=True)
    series.to_filename(base_dir / 'series.nii.gz')
    boldref.to_filename(base_dir / 'boldref.nii.gz')
    volreg = pe.Node(
        Volreg(
            in_file=str(base_dir / 'series.nii.gz'),
            basefile=str(base_dir / 'boldref.nii.gz'),
            interp='Fourier',
            twopass=True,
            zpad=4,
            out_file='NULL',
        ),
        name='volreg',
        base_dir=str(base_dir),
    )
    return volreg.run().outputs


def check_motion(oned_file, seed):
    """Raise ``ValueError`` unless some volume rotates enough about two axes at once."""
    rot = np.deg2rad(np.abs(np.loadtxt(oned_file)[:, :3]))
    largest = np.sort(rot, axis=1)[:, 1:].prod(axis=1).max()
    if largest < MIN_ANGLE_PRODUCT:
        raise ValueError(
            f'With seed {seed}, the largest product of two rotation angles that 3dvolreg '
            f'estimated in one volume is {largest:.2g} rad^2, below {MIN_ANGLE_PRODUCT:g}. '
            'The tests could miss errors in the Euler decomposition; try another --seed.'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        '--out-dir', type=Path, default=DATA_DIR, help='where to write the fixture directories'
    )
    parser.add_argument('--seed', type=int, default=0, help='seed for the random motions')
    opts = parser.parse_args()

    boldref = nb.load(BOLDREF)
    series = make_series(boldref, seed=opts.seed)
    with TemporaryDirectory() as tmpdir:
        results = {
            name: run_volreg(transform(series), transform(boldref), Path(tmpdir) / name)
            for name, transform in VARIANTS.items()
        }
        for outputs in results.values():
            check_motion(outputs.oned_file, opts.seed)
        for name, outputs in results.items():
            out_dir = opts.out_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(outputs.oned_file, out_dir / 'volreg.1D')
            shutil.copy(outputs.oned_matrix_save, out_dir / 'volreg.aff12.1D')


if __name__ == '__main__':
    main()
