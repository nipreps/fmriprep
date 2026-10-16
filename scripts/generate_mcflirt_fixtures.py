#!/usr/bin/env python
"""
Generate the MCFLIRT outputs that ``test_FSLMotionParams_mcflirt`` uses as its oracle.

The test boldref is moved by known rigid motions (rotations up to 0.1 rad,
translations up to 3 mm) to make a synthetic series, which ``mcflirt``
registers back to the boldref. The MAT files and ``.par`` from each run are
written to ``fmriprep/interfaces/tests/data/mcflirt_large_motion*/``.
Three variants are generated:

* ``mcflirt_large_motion``: the boldref as stored (LAS).
* ``mcflirt_large_motion_offset``: series and boldref with 200 added to every voxel.
* ``mcflirt_large_motion_ail``: series and boldref stored as AIL.

The script checks that MCFLIRT's estimates include rotations about two axes
at once that are large enough for the tests to tell a correct decomposition
from a wrong one, and raises ``ValueError`` if not; pass another ``--seed``
to draw different motions.

``mcflirt`` must be on the ``PATH``. The committed fixtures were made with
``fsl-mcflirt 2111.0``, for example::

    pixi exec -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public -c conda-forge \\
        --spec 'fsl-mcflirt=2111.0' -- python scripts/generate_mcflirt_fixtures.py
"""

import argparse
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nb
import nitransforms as nt
import nitransforms.resampling
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nipype.interfaces import fsl
from nipype.pipeline import engine as pe
from scipy.spatial.transform import Rotation

DATA_DIR = Path(__file__).resolve().parents[1] / 'fmriprep' / 'interfaces' / 'tests' / 'data'
BOLDREF = DATA_DIR / 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'

# Decomposing in the wrong Euler order errs by about the product of the two
# largest angles (measured: 0.98-1.05 times it), so some volume needs a product
# well above the tests' 1e-5 rad tolerance. Angles this large also make the
# first-order center-of-gravity errors exceed the 1e-3 mm tolerance for the
# boldref we use.
MIN_ANGLE_PRODUCT = 1e-3


def make_series(boldref, n_vols=10, max_rot=0.1, max_trans=3.0, seed=0):
    """Move ``boldref`` by random rigid motions, about its center; volume 0 is unmoved."""
    rng = np.random.default_rng(seed)
    center = (boldref.affine @ np.r_[(np.array(boldref.shape[:3]) - 1) / 2, 1])[:3]
    vols = [boldref]
    # Note that we do not record the specific parameters we use.
    # The important thing is that MCFLIRT generates parameters we can round-trip
    # through ITK-style affines and back.
    for r, t in rng.uniform(-1, 1, (n_vols - 1, 2, 3)):
        R = Rotation.from_rotvec(r * max_rot).as_matrix()
        aff = nb.affines.from_matvec(R, center - R @ center + t * max_trans)
        vols.append(
            nt.resampling.apply(nt.linear.Affine(aff, reference=boldref), boldref, order=3)
        )
    series = nb.funcs.concat_images(vols)
    series.set_data_dtype('float32')
    return series


def add_offset(img):
    """The image with 200 added to every voxel, as float32."""
    data = np.float32(img.dataobj) + 200
    return nb.Nifti1Image(data, img.affine, img.header)


def as_ail(img):
    """The image stored as AIL: L/R on the last voxel axis, with a positive determinant."""
    return img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt('AIL')))


VARIANTS = {
    'mcflirt_large_motion': lambda x: x,
    'mcflirt_large_motion_offset': add_offset,
    'mcflirt_large_motion_ail': as_ail,
}


def run_mcflirt(series, boldref, base_dir):
    """Register ``series`` to ``boldref`` with MCFLIRT, working in ``base_dir``."""
    base_dir.mkdir(parents=True)
    series.to_filename(in_file := base_dir / 'series.nii.gz')
    boldref.to_filename(ref_file := base_dir / 'boldref.nii.gz')
    mcflirt = pe.Node(
        fsl.MCFLIRT(in_file=str(in_file), ref_file=str(ref_file), save_mats=True, save_plots=True),
        name='mcflirt',
        base_dir=str(base_dir),
    )
    return mcflirt.run().outputs


def check_motion(par_file, seed):
    """Raise ``ValueError`` unless some volume rotates enough about two axes at once."""
    rot = np.abs(np.loadtxt(par_file)[:, :3])
    largest = np.sort(rot, axis=1)[:, 1:].prod(axis=1).max()
    if largest < MIN_ANGLE_PRODUCT:
        raise ValueError(
            f'With seed {seed}, the largest product of two rotation angles that MCFLIRT '
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

    # nipype's FSL interfaces warn if this is unset
    os.environ.setdefault('FSLOUTPUTTYPE', 'NIFTI_GZ')
    boldref = nb.load(BOLDREF)
    series = make_series(boldref, seed=opts.seed)
    with TemporaryDirectory() as tmpdir:
        results = {
            name: run_mcflirt(transform(series), transform(boldref), Path(tmpdir) / name)
            for name, transform in VARIANTS.items()
        }
        for outputs in results.values():
            check_motion(outputs.par_file, opts.seed)
        for name, outputs in results.items():
            out_dir = opts.out_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for mat in outputs.mat_file:
                shutil.copy(mat, out_dir)
            shutil.copy(outputs.par_file, out_dir / 'mcf.par')


if __name__ == '__main__':
    main()
