#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals


import numpy as np
import nibabel as nb
from scipy.interpolate import interpn
from datetime import datetime as dt

from builtins import object, str, bytes

from nipype import logging
LOGGER = logging.getLogger('interfaces')


class BSplineFieldmap(object):
    """
    A fieldmap representation object using BSpline basis
    """

    def __init__(self, fmapnii, weights=None, knots_zooms=[20., 20., 12.]):
        if not isinstance(knots_zooms, (list, tuple)):
            knots_zooms = [knots_zooms] * 3

        self._knots_zooms = np.array(knots_zooms)

        if isinstance(fmapnii, (str, bytes)):
            fmapnii = nb.as_closest_canonical(nb.load(fmapnii))

        self._fmapnii = fmapnii
        mapshape = self._fmapnii.get_data().shape

        # Compose a RAS affine mat, since the affine of the image might not be
        # RAS
        self._fmapaff = np.eye(
            4) * (list(fmapnii.header.get_zooms()[:3]) + [1])
        self._fmapaff[:3, 3] -= self._fmapaff[:3,
                                              :3].dot((np.array(mapshape[:3], dtype=float) - 1.0) * 0.5)
        self._extent = self._fmapaff[:3, :3].dot(mapshape[:3])

        # The list of ijk coordinates
        self._fmapijk = np.mgrid[
            0:mapshape[0], 0:mapshape[1], 0:mapshape[2]].reshape(3, -1).T
        ijk_h = np.hstack((self._fmapijk, np.array(
            [1.0] * len(self._fmapijk))[..., np.newaxis]))  # In homogeneous coords

        # The list of xyz coordinates
        self._fmapxyz = self._fmapaff.dot(ijk_h.T)[:3, :].T

        self._knots_shape = (
            np.ceil((self._extent - self._knots_zooms) / self._knots_zooms) + 3).astype(int)
        self._knots_grid = np.zeros(tuple(self._knots_shape), dtype=np.float32)
        self._knots_aff = np.eye(
            4) * np.array(self._knots_zooms.tolist() + [1.0])
        self._knots_aff[:3, 3] -= self._knots_aff[:3,
                                                  :3].dot((np.array(self._knots_shape) - 1) * 0.5)

        self._knots_ijk = np.mgrid[0:self._knots_shape[0], 0:self._knots_shape[
            1], 0:self._knots_shape[2]].reshape(3, -1).T
        knots_ijk_h = np.hstack((self._knots_ijk, np.array(
            [1.0] * len(self._knots_ijk))[..., np.newaxis]))  # In homogeneous coords

        # The list of xyz coordinates
        self._knots_xyz = self._knots_aff.dot(knots_ijk_h.T)[:3, :].T

        self._weights = np.ones((len(self._fmapxyz)))
        self._boundary = None
        if weights is not None:
            boundary = np.zeros_like(weights, dtype=np.uint8)
            boundary[:3, :, :] = 1
            boundary[:, :3, :] = 1
            boundary[:, :, :3] = 1
            boundary[-3:, :, :] = 1
            boundary[:, -3:, :] = 1
            boundary[:, :, -3:] = 1
            self._boundary = boundary[tuple(self._fmapijk.T)]
            self._weights = weights[tuple(self._fmapijk.T)]
            self._weights[self._boundary > 0] = 1

        self._pedir = 1
        self._X = None
        self._coeff = None
        self._smoothed = None

        self._Xinv = None
        self._inverted = None
        self._invcoeff = None

    def _evaluate_bspline(self):
        """ Calculates the design matrix """
        print('[%s] Evaluating tensor-product cubic BSpline on %d points, %d control points' %
              (dt.now(), len(self._fmapxyz), len(self._knots_xyz)))
        self._X = tbspl_eval(self._fmapxyz, self._knots_xyz, self._knots_zooms)
        print('[%s] Finished BSpline evaluation, %s' %
              (dt.now(), str(self._X.shape)))

    def evaluate(self):
        self._evaluate_bspline()

        fieldata = self._fmapnii.get_data()[tuple(self._fmapijk.T)]

        if self._boundary is not None:
            fieldata[self._boundary > 0] = 0

        print('[%s] Starting least-squares fitting using %d unmasked points' %
              (dt.now(), len(fieldata[self._weights > 0.0])))
        self._coeff = np.linalg.lstsq(
            self._X[self._weights > 0.0, ...], fieldata[self._weights > 0.0])[0]
        print('[%s] Finished least-squares fitting' % dt.now())

    def get_coeffmap(self):
        self._knots_grid[tuple(self._knots_ijk.T)] = self._coeff
        return nb.Nifti1Image(self._knots_grid, self._knots_aff, None)

    def get_smoothed(self):
        self._smoothed = np.zeros_like(self._fmapnii.get_data())
        self._smoothed[tuple(self._fmapijk.T)] = self._X.dot(self._coeff)
        return nb.Nifti1Image(self._smoothed, self._fmapnii.affine, self._fmapnii.header)

    def invert(self):
        targets = self._fmapxyz.copy()
        targets[:, self._pedir] += self._smoothed[tuple(self._fmapijk.T)]
        print('[%s] Inverting transform :: evaluating tensor-product cubic BSpline on %d points, %d control points' %
              (dt.now(), len(targets), len(self._knots_xyz)))
        self._Xinv = tbspl_eval(targets, self._knots_xyz, self._knots_zooms)
        print('[%s] Finished BSpline evaluation, %s' %
              (dt.now(), str(self._X.shape)))

        print('[%s] Starting least-squares fitting using %d unmasked points' %
              (dt.now(), len(targets)))
        self._invcoeff = np.linalg.lstsq(
            self._Xinv, self._fmapxyz[:, self._pedir] - targets[:, self._pedir])[0]
        print('[%s] Finished least-squares fitting' % dt.now())

    def get_inverted(self):
        self._inverted = np.zeros_like(self._fmapnii.get_data())
        self._inverted[tuple(self._fmapijk.T)] = self._X.dot(self._invcoeff)
        return nb.Nifti1Image(self._inverted, self._fmapnii.affine, self._fmapnii.header)

    def interp(self, in_data, inverse=False, fwd_pe=True):
        dshape = tuple(in_data.shape)
        gridxyz = self._fmapxyz.reshape((dshape[0], dshape[1], dshape[2], -1))

        x = gridxyz[:, 0, 0, 0]
        y = gridxyz[0, :, 0, 1]
        z = gridxyz[0, 0, :, 2]

        xyzmin = (x.min(), y.min(), z.min())
        xyzmax = (x.max(), y.max(), z.max())

        targets = self._fmapxyz.copy()

        if inverse:
            factor = 1.0 if fwd_pe else -1.0
            targets[:, self._pedir] += factor * \
                self._inverted[tuple(self._fmapijk.T)]
        else:
            targets[:, self._pedir] += self._smoothed[tuple(self._fmapijk.T)]

        interpolated = np.zeros_like(self._fmapnii.get_data())
        interpolated[tuple(self._fmapijk.T)] = interpn(
            (x, y, z), in_data, [tuple(v) for v in targets],
            bounds_error=False, fill_value=0)

        return nb.Nifti1Image(interpolated, self._fmapnii.affine, self._fmapnii.header)

#     def xfm_coords(self, in_coord):
#         X = fif.tbspl_eval(np.array([in_coord]), self._knots_xyz, self._knots_zooms)
#         new_coord = in_coord + X.dot(self._coeff if not inverse else self._invcoeff)


def _approx(fmapnii, s=14.):
    """
    Slice-wise approximation of a smooth 2D bspline
    credits: http://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-\
    with-scipyinterpolaterectbivariatespline/

    """
    from scipy.interpolate import RectBivariateSpline
    from builtins import str, bytes

    if isinstance(fmapnii, (str, bytes)):
        fmapnii = nb.load(fmapnii)

    if not isinstance(s, (tuple, list)):
        s = np.array([s] * 2)

    data = fmapnii.get_data()
    zooms = fmapnii.header.get_zooms()

    knot_decimate = np.floor(s / np.array(zooms)[:2]).astype(np.uint8)
    knot_space = np.array(zooms)[:2] * knot_decimate

    xmax = 0.5 * data.shape[0] * zooms[0]
    ymax = 0.5 * data.shape[1] * zooms[1]

    x = np.arange(-xmax, xmax, knot_space[0])
    y = np.arange(-ymax, ymax, knot_space[1])

    x2 = np.arange(-xmax, xmax, zooms[0])
    y2 = np.arange(-ymax, ymax, zooms[1])

    coeffs = []
    nslices = data.shape[-1]
    for k in range(nslices):
        data2d = data[..., k]
        data2dsubs = data2d[::knot_decimate[0], ::knot_decimate[1]]
        interp_spline = RectBivariateSpline(x, y, data2dsubs)

        data[..., k] = interp_spline(x2, y2)
        coeffs.append(interp_spline.get_coeffs().reshape(data2dsubs.shape))

    # Save smoothed data
    hdr = fmapnii.header.copy()
    caff = fmapnii.affine
    datanii = nb.Nifti1Image(data.astype(np.float32), caff, hdr)

    # Save bspline coeffs
    caff[0, 0] = knot_space[0]
    caff[1, 1] = knot_space[1]
    coeffnii = nb.Nifti1Image(np.stack(coeffs, axis=2), caff, hdr)

    return datanii, coeffnii


def bspl_smoothing(fmapnii, masknii=None, knot_space=[18., 18., 20.]):
    """
    A 3D BSpline smoothing of the fieldmap
    """
    from datetime import datetime as dt
    from builtins import str, bytes
    from scipy.linalg import pinv2

    if not isinstance(knot_space, (list, tuple)):
        knot_space = [knot_space] * 3
    knot_space = np.array(knot_space)

    if isinstance(fmapnii, (str, bytes)):
        fmapnii = nb.load(fmapnii)

    data = fmapnii.get_data()
    zooms = fmapnii.header.get_zooms()

    # Calculate hi-res i
    ijk = np.where(data < np.inf)
    xyz = np.array(ijk).T * np.array(zooms)[np.newaxis, :3]

    # Calculate control points
    xyz_max = xyz.max(axis=0)
    knot_dims = np.ceil(xyz_max / knot_space) + 2
    bspl_grid = np.zeros(tuple(knot_dims.astype(int)))
    bspl_ijk = np.where(bspl_grid == 0)
    bspl_xyz = np.array(bspl_ijk).T * knot_space[np.newaxis, ...]
    bspl_max = bspl_xyz.max(axis=0)
    bspl_xyz -= 0.5 * (bspl_max - xyz_max)[np.newaxis, ...]

    points_ijk = ijk
    points_xyz = xyz

    # Mask if provided
    if masknii is not None:
        if isinstance(masknii, (str, bytes)):
            masknii = nb.load(masknii)
        data[masknii.get_data() <= 0] = 0
        points_ijk = np.where(masknii.get_data() > 0)
        points_xyz = np.array(points_ijk).T * np.array(zooms)[np.newaxis, :3]

    print('[%s] Evaluating tensor-product cubic-bspline on %d points' %
          (dt.now(), len(points_xyz)))
    # Calculate design matrix
    X = tbspl_eval(points_xyz, bspl_xyz, knot_space)
    print('[%s] Finished, bspline grid has %d control points' %
          (dt.now(), len(bspl_xyz)))
    Y = data[points_ijk]

    # Fit coefficients
    print('[%s] Starting least-squares fitting' % dt.now())
    # coeff = (pinv2(X.T.dot(X)).dot(X.T)).dot(Y) # manual way (seems equally
    # slow)
    coeff = np.linalg.lstsq(X, Y)[0]
    print('[%s] Finished least-squares fitting' % dt.now())
    bspl_grid[bspl_ijk] = coeff
    aff = np.eye(4)
    aff[:3, :3] = aff[:3, :3] * knot_space[..., np.newaxis]
    coeffnii = nb.Nifti1Image(bspl_grid, aff, None)

    # Calculate hi-res design matrix:
    # print('[%s] Evaluating tensor-product cubic-bspline on %d points' % (dt.now(), len(xyz)))
    # Xinterp = tbspl_eval(xyz, bspl_xyz, knot_space)
    # print('[%s] Finished, start interpolation' % dt.now())

    # And interpolate
    newdata = np.zeros_like(data)
    newdata[points_ijk] = X.dot(coeff)
    newnii = nb.Nifti1Image(newdata, fmapnii.affine, fmapnii.header)

    return newnii, coeffnii


def tbspl_eval(points, knots, zooms):
    from fmriprep.utils.maths import bspl
    points = np.array(points)
    knots = np.array(knots)
    vbspl = np.vectorize(bspl)

    coeffs = []
    ushape = (knots.shape[0], 3)
    for p in points:
        u_vec = (knots - p[np.newaxis, ...]) / zooms[np.newaxis, ...]
        c = vbspl(u_vec.reshape(-1)).reshape(ushape).prod(axis=1)
        coeffs.append(c)

    return np.array(coeffs)
