"""Utilities for loading transforms for resampling"""

from pathlib import Path

import h5py
import nibabel as nb
import nitransforms as nt
import numpy as np
from nitransforms.io.itk import ITKCompositeH5
from transforms3d.affines import compose as compose_affine


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
    #   * Always return a nitransforms TransformBase
    #   * Construct warp affine from fixed parameters
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

    # Warp field fixed parameters as defined in
    # https://itk.org/Doxygen/html/classitk_1_1DisplacementFieldTransform.html
    shape = transform2['TransformFixedParameters'][:3]
    origin = transform2['TransformFixedParameters'][3:6]
    spacing = transform2['TransformFixedParameters'][6:9]
    direction = transform2['TransformFixedParameters'][9:].reshape((3, 3))

    # We are not yet confident that we handle non-unit spacing
    # or direction cosine ordering correctly.
    # If we confirm or fix, we can remove these checks.
    if not np.allclose(spacing, 1):
        raise ValueError(f'Unexpected spacing: {spacing}')
    if not np.allclose(direction, direction.T):
        raise ValueError(f'Asymmetric direction matrix: {direction}')

    # ITK uses LPS affines
    lps_affine = compose_affine(T=origin, R=direction, Z=spacing)
    ras_affine = np.diag([-1, -1, 1, 1]) @ lps_affine

    # ITK stores warps in Fortran-order, where the vector components change fastest
    # Vectors are in mm LPS
    itk_warp = np.reshape(
        transform2['TransformParameters'],
        (3, *shape.astype(int)),
        order='F',
    )

    # Nitransforms warps are in RAS, with the vector components changing slowest
    nt_warp = itk_warp.transpose(1, 2, 3, 0) * np.array([-1, -1, 1])

    transforms.insert(0, nt.DenseFieldTransform(nb.Nifti1Image(nt_warp, ras_affine)))
    return nt.TransformChain(transforms)
