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
        inverse *= len(xfm_paths)
    elif len(inverse) != len(xfm_paths):
        raise ValueError('Mismatched number of transforms and inverses')

    chain = None
    for path, inv in zip(xfm_paths[::-1], inverse[::-1], strict=False):
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


def load_ants_h5(filename: Path) -> nt.base.TransformBase:
    """Load ANTs H5 files as a nitransforms TransformChain"""
    # Borrowed from https://github.com/feilong/process
    # process.resample.parse_combined_hdf5()
    #
    # Changes:
    #   * Tolerate a missing displacement field
    #   * Return the original affine without a round-trip
    #   * Always return a nitransforms TransformChain
    #
    # This should be upstreamed into nitransforms
    h = h5py.File(filename)
    xform = ITKCompositeH5.from_h5obj(h)

    # nt.Affine
    transforms = [nt.Affine(xform[0].to_ras())]

    if '2' not in h['TransformGroup']:
        return transforms[0]

    transform2 = h['TransformGroup']['2']

    # Confirm these transformations are applicable
    if transform2['TransformType'][:][0] not in (
        b'DisplacementFieldTransform_float_3_3',
        b'DisplacementFieldTransform_double_3_3',
    ):
        msg = 'Unknown transform type [2]\n'
        for i in h['TransformGroup'].keys():
            msg += f'[{i}]: {h["TransformGroup"][i]["TransformType"][:][0]}\n'
        raise ValueError(msg)

    fixed_params = transform2['TransformFixedParameters'][:]

    spacing = fixed_params[6:9]
    direction = fixed_params[9:]

    # Supported spacing
    if not np.array_equal(spacing, np.array([1.0, 1.0, 1.0])):
        raise ValueError(f'Unexpected spacing: {spacing}')

    if not np.array_equal(direction, np.diag([-1, -1, -1])):
        raise ValueError(f'Asymmetric direction matrix: {direction}')

    shape = tuple(fixed_params[:3].astype(int))
    # ITK stores warps in Fortran-order, where the vector components change fastest
    # Nitransforms expects 3 volumes, not a volume of three-vectors, so transpose
    warp = np.reshape(
        transform2['TransformParameters'],
        (3, *shape),
        order='F',
    ).transpose(1, 2, 3, 0)
    warp *= np.array([-1, -1, 1])

    warp_affine = np.eye(4)
    warp_affine[:3, :3] = fixed_params[9:].reshape((3, 3))
    warp_affine[:3, 3] = fixed_params[3:6]
    lps_to_ras = np.eye(4) * np.array([-1, -1, 1, 1])
    warp_affine = lps_to_ras @ warp_affine
    transforms.insert(0, nt.DenseFieldTransform(nb.Nifti1Image(warp, warp_affine)))
    return nt.TransformChain(transforms)
