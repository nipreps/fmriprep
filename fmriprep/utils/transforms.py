"""Utilities for loading transforms for resampling"""
from pathlib import Path

import h5py
import nibabel as nb
import nitransforms as nt
import numpy as np
from nitransforms.io.itk import ITKCompositeH5


def load_transforms(xfm_paths: list[Path], inverse: list[bool]) -> nt.base.TransformBase:
    """Load a series of transforms as a nitransforms TransformChain

    An empty list will return an identity transform
    """
    if len(inverse) == 1:
        inverse = inverse * len(xfm_paths)
    elif len(inverse) != len(xfm_paths):
        raise ValueError("Mismatched number of transforms and inverses")

    chain = None
    for path, inv in zip(xfm_paths[::-1], inverse[::-1]):
        path = Path(path)
        if path.suffix == '.h5':
            xfm = load_ants_h5(path)
        else:
            xfm = nt.linear.load(path)
        if inv:
            xfm = ~xfm
        if chain is None:
            chain = xfm
        else:
            chain += xfm
    if chain is None:
        chain = nt.base.TransformBase()
    return chain


def load_ants_h5(filename: Path) -> nt.TransformChain:
    """Load ANTs H5 files as a nitransforms TransformChain"""
    affine, warp, warp_affine = parse_combined_hdf5(filename)
    warp_transform = nt.DenseFieldTransform(nb.Nifti1Image(warp, warp_affine))
    return nt.TransformChain([warp_transform, nt.Affine(affine)])


def parse_combined_hdf5(h5_fn, to_ras=True):
    # Borrowed from https://github.com/feilong/process
    # process.resample.parse_combined_hdf5()
    h = h5py.File(h5_fn)
    xform = ITKCompositeH5.from_h5obj(h)
    affine = xform[0].to_ras()
    # Confirm these transformations are applicable
    assert (
        h['TransformGroup']['2']['TransformType'][:][0] == b'DisplacementFieldTransform_float_3_3'
    )
    assert np.array_equal(
        h['TransformGroup']['2']['TransformFixedParameters'][:],
        np.array(
            [
                193.0,
                229.0,
                193.0,
                96.0,
                132.0,
                -78.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ),
    )
    warp = h['TransformGroup']['2']['TransformParameters'][:]
    warp = warp.reshape((193, 229, 193, 3)).transpose(2, 1, 0, 3)
    warp *= np.array([-1, -1, 1])
    warp_affine = np.array(
        [
            [1.0, 0.0, 0.0, -96.0],
            [0.0, 1.0, 0.0, -132.0],
            [0.0, 0.0, 1.0, -78.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return affine, warp, warp_affine
