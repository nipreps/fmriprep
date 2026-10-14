"""Utilities for loading, converting and writing transforms.

Loads transform chains for resampling, converts AFNI 3dvolreg motion
parameters to and from RAS+ affines, and writes ITK transform files
at full precision.
"""

import os
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


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
            # Load as a TransformChain
            xfm = nt.manip.load(path, fmt='h5')
        else:
            xfm = nt.linear.load(path)
        if inv:
            xfm = ~xfm
        if chain is None:
            chain = xfm
        else:
            chain += xfm
    if chain is None:
        chain = nt.Affine()  # Identity
    return chain


def volreg_to_ras(
    params: np.typing.ArrayLike,
    reference: nb.spatialimages.SpatialImage,
    base: nb.spatialimages.SpatialImage | None = None,
) -> np.ndarray:
    """Convert AFNI 3dvolreg motion parameters to RAS+ affines.

    Parameters
    ----------
    params
        ``(T, 6)`` rows of ``roll pitch yaw dS dL dP`` (degrees and mm),
        as written by ``3dvolreg -1Dfile``. A single row may be passed as ``(6,)``.
    reference
        An image on the grid of the registered BOLD series.
    base
        3dvolreg's ``-base`` image. Defaults to ``reference``.

    Returns
    -------
    ``(T, 4, 4)`` RAS+ affines in nitransforms' convention: each maps
    coordinates in ``base`` to the corresponding coordinates in that BOLD volume.

    Examples
    --------
    >>> img = nb.Nifti1Image(np.zeros((10, 10, 10)), np.diag([2.0, 2.0, 2.0, 1.0]))
    >>> np.allclose(volreg_to_ras(np.zeros(6), img), np.eye(4))
    True

    """
    raise NotImplementedError


def ras_to_volreg(
    matrices: np.typing.ArrayLike,
    reference: nb.spatialimages.SpatialImage,
    base: nb.spatialimages.SpatialImage | None = None,
    degrees: bool = True,
) -> np.ndarray:
    """Convert RAS+ affines to AFNI 3dvolreg motion parameters.

    This is the inverse of :func:`volreg_to_ras`.

    Parameters
    ----------
    matrices
        ``(T, 4, 4)`` RAS+ affines in nitransforms' convention, mapping
        coordinates in ``base`` to coordinates in each BOLD volume.
        A single affine may be passed as ``(4, 4)``.
    reference
        An image on the grid of the BOLD series.
    base
        The image the affines map from. Defaults to ``reference``.
    degrees
        Whether to return angles in degrees (default, matching AFNI) or radians.

    Returns
    -------
    ``(T, 6)`` rows of ``roll pitch yaw dS dL dP`` (degrees and mm),
    as ``3dvolreg -1Dfile`` writes them.

    """
    raise NotImplementedError


def write_itk_affines(matrices: np.typing.ArrayLike, out_file: str | os.PathLike) -> None:
    """Write RAS+ affines to an ITK text transform file at full double precision.

    nitransforms' ITK writer prints 6 significant digits, which moves the
    stored transforms by up to about 5e-5 mm. This writer prints 17, so the
    file reproduces ``matrices`` to the precision of the reader.

    Parameters
    ----------
    matrices
        ``(T, 4, 4)`` RAS+ affines in nitransforms' convention, or one ``(4, 4)`` affine.
    out_file
        The file to write. :func:`nitransforms.linear.load` reads it back.

    """
    raise NotImplementedError
