"""Interfaces for resampling images in a single shot"""

import asyncio
import os
from functools import partial

import nibabel as nb
import nitransforms as nt
import nitransforms.resampling
import numpy as np
from nipype.interfaces.base import (
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from scipy import ndimage as ndi

from ..utils.asynctools import worker
from ..utils.transforms import load_transforms


class ResampleSeriesInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='3D or 4D image file to resample')
    ref_file = File(exists=True, mandatory=True, desc='File to resample in_file to')
    transforms = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='Transform files, from in_file to ref_file (image mode)',
    )
    inverse = InputMultiObject(
        traits.Bool,
        value=[False],
        usedefault=True,
        desc='Whether to invert each file in transforms',
    )
    num_threads = traits.Int(1, usedefault=True, desc='Number of threads to use for resampling')
    output_data_type = traits.Str('float32', usedefault=True, desc='Data type of output image')
    order = traits.Int(3, usedefault=True, desc='Order of interpolation (0=nearest, 3=cubic)')
    mode = traits.Enum(
        'nearest',
        'constant',
        'mirror',
        'reflect',
        'wrap',
        'grid-constant',
        'grid-mirror',
        'grid-wrap',
        usedefault=True,
        desc='How data is extended beyond its boundaries. '
        'See scipy.ndimage.map_coordinates for more details.',
    )
    cval = traits.Float(0.0, usedefault=True, desc='Value to fill past edges of data')
    prefilter = traits.Bool(True, usedefault=True, desc='Spline-prefilter data if order > 1')


class ResampleSeriesOutputSpec(TraitedSpec):
    out_file = File(desc='Resampled image or series')


class ResampleSeries(SimpleInterface):
    """Resample a time series, applying susceptibility and motion correction
    simultaneously.
    """

    input_spec = ResampleSeriesInputSpec
    output_spec = ResampleSeriesOutputSpec

    def _run_interface(self, runtime):
        out_path = fname_presuffix(self.inputs.in_file, suffix='resampled', newpath=runtime.cwd)

        source = nb.load(self.inputs.in_file)
        target = nb.load(self.inputs.ref_file)

        transforms = load_transforms(self.inputs.transforms, self.inputs.inverse)

        resampled = resample_image(
            source=source,
            target=target,
            transforms=transforms,
            nthreads=self.inputs.num_threads,
            output_dtype=self.inputs.output_data_type,
            order=self.inputs.order,
            mode=self.inputs.mode,
            cval=self.inputs.cval,
            prefilter=self.inputs.prefilter,
        )
        resampled.to_filename(out_path)

        self._results['out_file'] = out_path
        return runtime


def resample_vol(
    data: np.ndarray,
    coordinates: np.ndarray,
    hmc_xfm: np.ndarray | None,
    output: np.dtype | np.ndarray | None = None,
    order: int = 3,
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True,
) -> np.ndarray:
    """Resample a volume at specified coordinates

    This function implements simultaneous head-motion correction and
    susceptibility-distortion correction. It accepts coordinates in
    the source voxel space. It is the responsibility of the caller to
    transform coordinates from any other target space.

    Parameters
    ----------
    data
        The data array to resample
    coordinates
        The first-approximation voxel coordinates to sample from ``data``
        The first dimension should have length ``data.ndim``. The further
        dimensions have the shape of the target array.
    hmc_xfm
        Affine transformation accounting for head motion from the individual
        volume into the BOLD reference space. This affine must be in VOX2VOX
        form.
    output
        The dtype or a pre-allocated array for sampling into the target space.
        If pre-allocated, ``output.shape == coordinates.shape[1:]``.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.

    Returns
    -------
    resampled_array
        The resampled array, with shape ``coordinates.shape[1:]``.
    """
    if hmc_xfm is not None:
        # Move image with the head
        coords_shape = coordinates.shape
        coordinates = nb.affines.apply_affine(
            hmc_xfm, coordinates.reshape(coords_shape[0], -1).T
        ).T.reshape(coords_shape)
    else:
        # Copy coordinates to avoid interfering with other calls
        coordinates = coordinates.copy()

    result = ndi.map_coordinates(
        data,
        coordinates,
        output=output,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    return result


async def resample_series_async(
    data: np.ndarray,
    coordinates: np.ndarray,
    hmc_xfms: list[np.ndarray] | None,
    output_dtype: np.dtype | None = None,
    order: int = 3,
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True,
    max_concurrent: int = min(os.cpu_count(), 12),
) -> np.ndarray:
    """Resample a 4D time series at specified coordinates

    This function implements simultaneous head-motion correction and
    susceptibility-distortion correction. It accepts coordinates in
    the source voxel space. It is the responsibility of the caller to
    transform coordinates from any other target space.

    Parameters
    ----------
    data
        The data array to resample
    coordinates
        The first-approximation voxel coordinates to sample from ``data``.
        The first dimension should have length 3.
        The further dimensions determine the shape of the target array.
    hmc_xfm
        A sequence of affine transformations accounting for head motion from
        the individual volume into the BOLD reference space.
        These affines must be in VOX2VOX form.
    output_dtype
        The dtype of the output array.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.
    max_concurrent
        Maximum number of volumes to resample concurrently

    Returns
    -------
    resampled_array
        The resampled array, with shape ``coordinates.shape[1:] + (N,)``,
        where N is the number of volumes in ``data``.
    """
    if data.ndim == 3:
        return resample_vol(
            data,
            coordinates,
            hmc_xfms[0] if hmc_xfms else None,
            output_dtype,
            order,
            mode,
            cval,
            prefilter,
        )

    semaphore = asyncio.Semaphore(max_concurrent)

    # Order F ensures individual volumes are contiguous in memory
    # Also matches NIfTI, making final save more efficient
    out_array = np.zeros(coordinates.shape[1:] + data.shape[-1:], dtype=output_dtype, order='F')

    tasks = [
        asyncio.create_task(
            worker(
                partial(
                    resample_vol,
                    data=volume,
                    coordinates=coordinates,
                    hmc_xfm=hmc_xfms[volid] if hmc_xfms else None,
                    output=out_array[..., volid],
                    order=order,
                    mode=mode,
                    cval=cval,
                    prefilter=prefilter,
                ),
                semaphore,
            )
        )
        for volid, volume in enumerate(np.rollaxis(data, -1, 0))
    ]

    await asyncio.gather(*tasks)

    return out_array


def resample_series(
    data: np.ndarray,
    coordinates: np.ndarray,
    hmc_xfms: list[np.ndarray] | None,
    output_dtype: np.dtype | None = None,
    order: int = 3,
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True,
    nthreads: int = 1,
) -> np.ndarray:
    """Resample a 4D time series at specified coordinates

    This function implements simultaneous head-motion correction and
    susceptibility-distortion correction. It accepts coordinates in
    the source voxel space. It is the responsibility of the caller to
    transform coordinates from any other target space.

    Parameters
    ----------
    data
        The data array to resample
    coordinates
        The first-approximation voxel coordinates to sample from ``data``.
        The first dimension should have length 3.
        The further dimensions determine the shape of the target array.
    hmc_xfm
        A sequence of affine transformations accounting for head motion from
        the individual volume into the BOLD reference space.
        These affines must be in VOX2VOX form.
    output_dtype
        The dtype of the output array.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.
    nthreads
        Number of threads to use for parallel resampling

    Returns
    -------
    resampled_array
        The resampled array, with shape ``coordinates.shape[1:] + (N,)``,
        where N is the number of volumes in ``data``.
    """
    return asyncio.run(
        resample_series_async(
            data=data,
            coordinates=coordinates,
            hmc_xfms=hmc_xfms,
            output_dtype=output_dtype,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
            max_concurrent=nthreads,
        )
    )


def resample_image(
    source: nb.Nifti1Image,
    target: nb.Nifti1Image,
    transforms: nt.TransformChain,
    nthreads: int = 1,
    output_dtype: np.dtype | str | None = 'f4',
    order: int = 3,
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True,
) -> nb.Nifti1Image:
    """Resample a 3- or 4D image into a target space, applying head-motion
    and susceptibility-distortion correction simultaneously.

    Parameters
    ----------
    source
        The 3D bold image or 4D bold series to resample.
    target
        An image sampled in the target space.
    transforms
        A nitransforms TransformChain that maps images from the individual
        BOLD volume space into the target space.
    nthreads
        Number of threads to use for parallel resampling
    output_dtype
        The dtype of the output array.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.

    Returns
    -------
    resampled_bold
        The BOLD series resampled into the target space
    """
    if not isinstance(transforms, nt.TransformChain):
        transforms = nt.TransformChain([transforms])
    if isinstance(transforms[-1], nt.linear.LinearTransformsMapping):
        transform_list, hmc = transforms[:-1], transforms[-1]
    else:
        if any(isinstance(xfm, nt.linear.LinearTransformsMapping) for xfm in transforms):
            classes = [xfm.__class__.__name__ for xfm in transforms]
            raise ValueError(f'HMC transforms must come last. Found sequence: {classes}')
        transform_list: list = transforms.transforms
        hmc = []

    # Retrieve the RAS coordinates of the target space
    coordinates = nt.base.SpatialReference.factory(target).ndcoords.astype('f4').T

    # We will operate in voxel space, so get the source affine
    vox2ras = source.affine
    ras2vox = np.linalg.inv(vox2ras)
    # Transform RAS2RAS head motion transforms to VOX2VOX
    hmc_xfms = [ras2vox @ xfm.matrix @ vox2ras for xfm in hmc]

    # After removing the head-motion transforms, add a mapping from boldref
    # world space to voxels. This new transform maps from world coordinates
    # in the target space to voxel coordinates in the source space.
    ref2vox = nt.TransformChain(transform_list + [nt.Affine(ras2vox)])
    mapped_coordinates = ref2vox.map(coordinates)

    resampled_data = resample_series(
        data=source.get_fdata(dtype='f4'),
        coordinates=mapped_coordinates.T.reshape((3, *target.shape[:3])),
        hmc_xfms=hmc_xfms,
        output_dtype=output_dtype,
        nthreads=nthreads,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )
    resampled_img = nb.Nifti1Image(resampled_data, target.affine, target.header)
    resampled_img.set_data_dtype('f4')
    # Preserve zooms of additional dimensions
    resampled_img.header.set_zooms(target.header.get_zooms()[:3] + source.header.get_zooms()[3:])

    return resampled_img
