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
from nitransforms.io.afni import AFNILinearTransform, AFNILinearTransformArray
from scipy.spatial import transform as sst

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
    if base is None:
        base = reference

    params = np.asarray(params, dtype=float).reshape(-1, 6)
    roll, pitch, yaw, dS, dL, dP = params.T
    c_inp, c_bas = _afni_center(reference), _afni_center(base)

    # The parameters describe the input -> base correction, in AFNI's DICOM frame.
    # 3dvolreg composes R = Ry(yaw) @ Rx(pitch) @ Rz(roll) (3dvolreg.c VOLREG_MATVEC)
    correction = np.tile(np.eye(4), (len(params), 1, 1))
    correction[:, :3, :3] = R = sst.Rotation.from_euler(
        'YXZ', np.c_[yaw, pitch, roll], degrees=True
    ).as_matrix()
    correction[:, :3, 3] = c_bas + np.c_[dL, dP, dS] - R @ c_inp

    # -1Dmatrix_save stores the inverse, base -> input; convert it as nitransforms
    # converts that file, including AFNI's de-obliquing
    afni = AFNILinearTransformArray()
    afni.xforms = [AFNILinearTransform() for _ in params]
    for xfm, matrix in zip(afni.xforms, np.linalg.inv(correction), strict=True):
        xfm.structarr['parameters'] = matrix
    return afni.to_ras(reference=base, moving=reference)


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
    if base is None:
        base = reference

    matrices = np.asarray(matrices, dtype=float).reshape(-1, 4, 4)
    c_inp, c_bas = _afni_center(reference), _afni_center(base)

    afni = AFNILinearTransformArray.from_ras(matrices, reference=base, moving=reference)
    # -1Dmatrix_save holds base -> input; the parameters describe its inverse,
    # the input -> base correction
    correction = np.linalg.inv(np.stack([xfm.structarr['parameters'] for xfm in afni.xforms]))
    R = correction[:, :3, :3]
    # 3dvolreg composes R = Ry(yaw) @ Rx(pitch) @ Rz(roll) (3dvolreg.c VOLREG_MATVEC)
    yaw, pitch, roll = sst.Rotation.from_matrix(R).as_euler('YXZ', degrees=degrees).T
    # Translations relative to the dataset centers
    dL, dP, dS = (correction[:, :3, 3] + R @ c_inp - c_bas).T
    return np.c_[roll, pitch, yaw, dS, dL, dP]


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
    matrices = np.asarray(matrices, dtype=float).reshape(-1, 4, 4)
    lines = ['#Insight Transform File V1.0']
    for i, ras in enumerate(matrices):
        # ITK transforms are LPS
        lps = LPS @ ras @ LPS
        values = [*lps[:3, :3].ravel(), *lps[:3, 3]]
        lines += [
            f'#Transform {i}',
            'Transform: AffineTransform_double_3_3',
            'Parameters: ' + ' '.join(f'{value:.17g}' for value in values),
            'FixedParameters: 0 0 0',
        ]
    Path(out_file).write_text('\n'.join(lines) + '\n')


def _deoblique(oblique: np.ndarray) -> np.ndarray:
    """Return a copy of ``affine`` with obliquity removed, in AFNI's DICOM frame.

    AFNI drops an image's obliquity by snapping its direction cosines to the
    nearest axes and rounding its voxel sizes to 4 decimals, keeping the origin.
    """
    # Rely on nibabel to stably identify the orientation
    axes, flips = nb.orientations.io_orientation(oblique).astype(int).T
    # Voxel axis i points along world axis axes[i], so axis indices permute columns
    # of an identity rotation matrix, and flips are applied per-column
    cosines = np.eye(3)[:, axes] * flips
    zooms = np.round(nb.affines.voxel_sizes(oblique), decimals=4)
    cardinal = nb.affines.from_matvec(zooms * cosines, oblique[:3, 3])
    return LPS @ cardinal


def _afni_center(img: nb.spatialimages.SpatialImage) -> np.ndarray:
    """Return the center of ``img``'s grid after AFNI de-obliques it, in AFNI's DICOM frame.

    3dvolreg rotates about this point (``THD_dataset_center``). AFNI drops an
    image's obliquity by snapping its direction cosines to the nearest axes and
    rounding its voxel sizes to 4 decimals, keeping the origin. This reproduces
    ``nitransforms.io.afni._dicom_real_to_card``, which is private API.
    """
    oblique = img.affine
    cardinal = np.eye(4)
    cardinal[:3, 3] = oblique[:3, 3]
    cosines = oblique[:3, :3] / np.abs(oblique[:3, :3]).max(0)
    cosines[np.abs(cosines) < 1.0] = 0
    cardinal[:3, :3] = np.round(nb.affines.voxel_sizes(oblique), decimals=4) * cosines
    center_ijk = np.r_[(np.array(img.shape[:3]) - 1) / 2, 1]
    return (LPS @ cardinal @ center_ijk)[:3]
