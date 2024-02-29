"""Interfaces for resampling images in a single shot"""

import asyncio
import os
from functools import partial

import nibabel as nb
import nitransforms as nt
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
from scipy.sparse import hstack as sparse_hstack
from sdcflows.transform import grid_bspline_weights
from sdcflows.utils.tools import ensure_positive_cosines

from ..utils.asynctools import worker
from ..utils.transforms import load_transforms


class ResampleSeriesInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="3D or 4D image file to resample")
    ref_file = File(exists=True, mandatory=True, desc="File to resample in_file to")
    transforms = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="Transform files, from in_file to ref_file (image mode)",
    )
    inverse = InputMultiObject(
        traits.Bool,
        value=[False],
        usedefault=True,
        desc="Whether to invert each file in transforms",
    )
    fieldmap = File(exists=True, desc="Fieldmap file resampled into reference space")
    ro_time = traits.Float(desc="EPI readout time (s).")
    pe_dir = traits.Enum(
        "i",
        "i-",
        "j",
        "j-",
        "k",
        "k-",
        desc="the phase-encoding direction corresponding to in_data",
    )
    jacobian = traits.Bool(mandatory=True, desc="Whether to apply Jacobian correction")
    num_threads = traits.Int(1, usedefault=True, desc="Number of threads to use for resampling")
    output_data_type = traits.Str("float32", usedefault=True, desc="Data type of output image")
    order = traits.Int(3, usedefault=True, desc="Order of interpolation (0=nearest, 3=cubic)")
    mode = traits.Str(
        'constant',
        usedefault=True,
        desc="How data is extended beyond its boundaries. "
        "See scipy.ndimage.map_coordinates for more details.",
    )
    cval = traits.Float(0.0, usedefault=True, desc="Value to fill past edges of data")
    prefilter = traits.Bool(True, usedefault=True, desc="Spline-prefilter data if order > 1")


class ResampleSeriesOutputSpec(TraitedSpec):
    out_file = File(desc="Resampled image or series")


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
        fieldmap = nb.load(self.inputs.fieldmap) if self.inputs.fieldmap else None

        nvols = source.shape[3] if source.ndim > 3 else 1

        transforms = load_transforms(self.inputs.transforms, self.inputs.inverse)

        pe_dir = self.inputs.pe_dir
        ro_time = self.inputs.ro_time
        pe_info = None

        if pe_dir and ro_time:
            pe_axis = "ijk".index(pe_dir[0])
            pe_flip = pe_dir.endswith("-")

            # Nitransforms displacements are positive
            source, axcodes = ensure_positive_cosines(source)
            axis_flip = axcodes[pe_axis] in "LPI"

            pe_info = [(pe_axis, -ro_time if (axis_flip ^ pe_flip) else ro_time)] * nvols

        resampled = resample_image(
            source=source,
            target=target,
            transforms=transforms,
            fieldmap=fieldmap,
            pe_info=pe_info,
            jacobian=self.inputs.jacobian,
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


class ReconstructFieldmapInputSpec(TraitedSpec):
    in_coeffs = InputMultiObject(
        File(exists=True), mandatory=True, desc="SDCflows-style spline coefficient files"
    )
    target_ref_file = File(
        exists=True, mandatory=True, desc="Image to reconstruct the field in alignment with"
    )
    fmap_ref_file = File(
        exists=True, mandatory=True, desc="Reference file aligned with coefficients"
    )
    transforms = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="Transform files, from in_file to ref_file (image mode)",
    )
    inverse = InputMultiObject(
        traits.Bool,
        value=[False],
        usedefault=True,
        desc="Whether to invert each file in transforms",
    )


class ReconstructFieldmapOutputSpec(TraitedSpec):
    out_file = File(desc="Fieldmap reconstructed in target_ref_file space")


class ReconstructFieldmap(SimpleInterface):
    """Reconstruct a fieldmap from B-spline coefficients in a target space.

    If the target reference does not have an aligned grid (guaranteed if
    transforms include a warp), then a reference file describing the space
    where it is valid to extrapolate the field will be used as an intermediate
    step.
    """

    input_spec = ReconstructFieldmapInputSpec
    output_spec = ReconstructFieldmapOutputSpec

    def _run_interface(self, runtime):
        out_path = fname_presuffix(self.inputs.in_coeffs[-1], suffix='rec', newpath=runtime.cwd)

        coefficients = [nb.load(coeff_file) for coeff_file in self.inputs.in_coeffs]
        target = nb.load(self.inputs.target_ref_file)
        fmapref = nb.load(self.inputs.fmap_ref_file)

        transforms = load_transforms(self.inputs.transforms, self.inputs.inverse)

        fieldmap = reconstruct_fieldmap(
            coefficients=coefficients,
            fmap_reference=fmapref,
            target=target,
            transforms=transforms,
        )
        fieldmap.to_filename(out_path)

        self._results['out_file'] = out_path
        return runtime


class DistortionParametersInputSpec(TraitedSpec):
    in_file = File(exists=True, desc="EPI image corresponding to the metadata")
    metadata = traits.Dict(mandatory=True, desc="metadata corresponding to the inputs")


class DistortionParametersOutputSpec(TraitedSpec):
    readout_time = traits.Float
    pe_direction = traits.Enum("i", "i-", "j", "j-", "k", "k-")


class DistortionParameters(SimpleInterface):
    """Retrieve PhaseEncodingDirection and TotalReadoutTime from available metadata.

    One or both parameters may be missing; downstream interfaces should be prepared
    to handle this.
    """

    input_spec = DistortionParametersInputSpec
    output_spec = DistortionParametersOutputSpec

    def _run_interface(self, runtime):
        from sdcflows.utils.epimanip import get_trt

        try:
            self._results["readout_time"] = get_trt(
                self.inputs.metadata,
                self.inputs.in_file or None,
            )
            self._results["pe_direction"] = self.inputs.metadata["PhaseEncodingDirection"]
        except (KeyError, ValueError):
            pass

        return runtime


def resample_vol(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: tuple[int, float],
    jacobian: bool,
    hmc_xfm: np.ndarray | None,
    fmap_hz: np.ndarray,
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
    pe_info
        The readout vector in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
    hmc_xfm
        Affine transformation accounting for head motion from the individual
        volume into the BOLD reference space. This affine must be in VOX2VOX
        form.
    fmap_hz
        The fieldmap, sampled to the target space, in Hz
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

    vsm = fmap_hz * pe_info[1]
    coordinates[pe_info[0], ...] += vsm

    result = ndi.map_coordinates(
        data,
        coordinates,
        output=output,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    if jacobian:
        result *= 1 + np.gradient(vsm, axis=pe_info[0])

    return result


async def resample_series_async(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: list[tuple[int, float]],
    jacobian: bool,
    hmc_xfms: list[np.ndarray] | None,
    fmap_hz: np.ndarray,
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
    pe_info
        A list of readout vectors in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
    hmc_xfm
        A sequence of affine transformations accounting for head motion from
        the individual volume into the BOLD reference space.
        These affines must be in VOX2VOX form.
    fmap_hz
        The fieldmap, sampled to the target space, in Hz
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
            pe_info[0],
            jacobian,
            hmc_xfms[0] if hmc_xfms else None,
            fmap_hz,
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
                    pe_info=pe_info[volid],
                    jacobian=jacobian,
                    hmc_xfm=hmc_xfms[volid] if hmc_xfms else None,
                    fmap_hz=fmap_hz,
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
    pe_info: list[tuple[int, float]],
    jacobian: bool,
    hmc_xfms: list[np.ndarray] | None,
    fmap_hz: np.ndarray,
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
    pe_info
        A list of readout vectors in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
    hmc_xfm
        A sequence of affine transformations accounting for head motion from
        the individual volume into the BOLD reference space.
        These affines must be in VOX2VOX form.
    fmap_hz
        The fieldmap, sampled to the target space, in Hz
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
            pe_info=pe_info,
            jacobian=jacobian,
            hmc_xfms=hmc_xfms,
            fmap_hz=fmap_hz,
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
    fieldmap: nb.Nifti1Image | None,
    pe_info: list[tuple[int, float]] | None,
    jacobian: bool = True,
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
    fieldmap
        The fieldmap, in Hz, sampled in the target space
    pe_info
        A list of readout vectors in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
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
            raise ValueError(f"HMC transforms must come last. Found sequence: {classes}")
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

    # Some identities to reduce special casing downstream
    if fieldmap is None:
        fieldmap = nb.Nifti1Image(np.zeros(target.shape[:3], dtype='f4'), target.affine)
    if pe_info is None:
        pe_info = [[0, 0] for _ in range(source.shape[-1])]

    resampled_data = resample_series(
        data=source.get_fdata(dtype='f4'),
        coordinates=mapped_coordinates.T.reshape((3, *target.shape[:3])),
        pe_info=pe_info,
        jacobian=jacobian,
        hmc_xfms=hmc_xfms,
        fmap_hz=fieldmap.get_fdata(dtype='f4'),
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


def aligned(aff1: np.ndarray, aff2: np.ndarray) -> bool:
    """Determine if two affines have aligned grids"""
    return np.allclose(
        np.linalg.norm(np.cross(aff1[:-1, :-1].T, aff2[:-1, :-1].T), axis=1),
        0,
        atol=1e-3,
    )


def as_affine(xfm: nt.base.TransformBase) -> nt.Affine | None:
    # Identity transform
    if type(xfm) is nt.base.TransformBase:
        return nt.Affine()

    if isinstance(xfm, nt.Affine):
        return xfm

    if isinstance(xfm, nt.TransformChain) and all(isinstance(x, nt.Affine) for x in xfm):
        return xfm.asaffine()

    return None


def reconstruct_fieldmap(
    coefficients: list[nb.Nifti1Image],
    fmap_reference: nb.Nifti1Image,
    target: nb.Nifti1Image,
    transforms: nt.TransformChain,
) -> nb.Nifti1Image:
    """Resample a fieldmap from B-Spline coefficients into a target space

    If the coefficients and target are aligned, the field is reconstructed
    directly in the target space.
    If not, then the field is reconstructed to the ``fmap_reference``
    resolution, and then resampled according to transforms.

    The former method only applies if the transform chain can be
    collapsed to a single affine transform.

    Parameters
    ----------
    coefficients
        list of B-spline coefficient files. The affine matrices are used
        to reconstruct the knot locations.
    fmap_reference
        The intermediate reference to reconstruct the fieldmap in, if
        it cannot be reconstructed directly in the target space.
    target
        The target space to to resample the fieldmap into.
    transforms
        A nitransforms TransformChain that maps images from the fieldmap
        space into the target space.

    Returns
    -------
    fieldmap
        The fieldmap encoded in ``coefficients``, resampled in the same
        space as ``target``
    """

    direct = False
    affine_xfm = as_affine(transforms)
    if affine_xfm is not None:
        # Transforms maps RAS coordinates in the target to RAS coordinates in
        # the fieldmap space. Composed with target.affine, we have a target voxel
        # to fieldmap RAS affine. Hence, this is projected into fieldmap space.
        projected_affine = affine_xfm.matrix @ target.affine
        # If the coordinates have the same rotation from voxels, we can construct
        # bspline weights efficiently.
        direct = aligned(projected_affine, coefficients[-1].affine)

    if direct:
        reference, _ = ensure_positive_cosines(
            target.__class__(target.dataobj, projected_affine, target.header),
        )
    else:
        if not aligned(fmap_reference.affine, coefficients[-1].affine):
            raise ValueError('Reference passed is not aligned with spline grids')
        reference, _ = ensure_positive_cosines(fmap_reference)

    # Generate tensor-product B-Spline weights
    colmat = sparse_hstack(
        [grid_bspline_weights(reference, level) for level in coefficients]
    ).tocsr()
    coefficients = np.hstack(
        [level.get_fdata(dtype='float32').reshape(-1) for level in coefficients]
    )

    # Reconstruct the fieldmap (in Hz) from coefficients
    fmap_img = nb.Nifti1Image(
        np.reshape(colmat @ coefficients, reference.shape[:3]),
        reference.affine,
    )

    if not direct:
        fmap_img = transforms.apply(fmap_img, reference=target)

    fmap_img.header.set_intent('estimate', name='fieldmap Hz')
    fmap_img.header.set_data_dtype('float32')
    fmap_img.header['cal_max'] = max((abs(fmap_img.dataobj.min()), fmap_img.dataobj.max()))
    fmap_img.header['cal_min'] = -fmap_img.header['cal_max']

    return fmap_img
