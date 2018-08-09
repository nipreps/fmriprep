#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import numpy as np
import nibabel as nb
from scipy.interpolate import interpn
from datetime import datetime as dt

from nipype import logging
LOGGER = logging.getLogger('nipype.interfaces')


class BSplineFieldmap(object):
    """
    A fieldmap representation object using BSpline basis
    """

    def __init__(self, fmapnii, weights=None, knots_zooms=None, padding=3,
                 pe_dir=1, njobs=-1):

        self._pedir = pe_dir
        if knots_zooms is None:
            knots_zooms = [40., 40., 18.]
            knots_zooms[pe_dir] = 60.

        if not isinstance(knots_zooms, (list, tuple)):
            knots_zooms = [knots_zooms] * 3

        self._knots_zooms = np.array(knots_zooms)

        if isinstance(fmapnii, str):
            fmapnii = nb.as_closest_canonical(nb.load(fmapnii))

        self._fmapnii = fmapnii
        self._padding = padding

        # Pad data with zeros
        self._data = np.zeros(tuple(np.array(
            self._fmapnii.get_data().shape) + 2 * padding))

        # The list of ijk coordinates
        self._fmapijk = get_ijk(self._data)

        # Find padding coordinates
        self._data[padding:-padding,
                   padding:-padding,
                   padding:-padding] = 1
        self._frameijk = self._data[tuple(self._fmapijk.T)] > 0

        # Set data
        self._data[padding:-padding,
                   padding:-padding,
                   padding:-padding] = fmapnii.get_data()

        # Get ijk in homogeneous coords
        ijk_h = np.hstack((self._fmapijk, np.array([1.0] * len(self._fmapijk))[..., np.newaxis]))

        # The list of xyz coordinates
        self._fmapaff = compute_affine(self._data, self._fmapnii.header.get_zooms())
        self._fmapxyz = self._fmapaff.dot(ijk_h.T)[:3, :].T

        # Mask coordinates
        self._weights = self._set_weights(weights)

        # Generate control points
        self._generate_knots()

        self._X = None
        self._coeff = None
        self._smoothed = None

        self._Xinv = None
        self._inverted = None
        self._invcoeff = None

        self._njobs = njobs

    def _generate_knots(self):
        extent = self._fmapaff[:3, :3].dot(self._data.shape[:3])
        self._knots_shape = (np.ceil(
            (extent - self._knots_zooms) / self._knots_zooms) + 3).astype(int)
        self._knots_grid = np.zeros(tuple(self._knots_shape), dtype=np.float32)
        self._knots_aff = compute_affine(self._knots_grid, self._knots_zooms)

        self._knots_ijk = get_ijk(self._knots_grid)
        knots_ijk_h = np.hstack((self._knots_ijk, np.array(
            [1.0] * len(self._knots_ijk))[..., np.newaxis]))  # In homogeneous coords

        # The list of xyz coordinates
        self._knots_xyz = self._knots_aff.dot(knots_ijk_h.T)[:3, :].T

    def _set_weights(self, weights=None):
        if weights is not None:
            extweights = np.ones_like(self._data)
            extweights[self._padding:-self._padding,
                       self._padding:-self._padding,
                       self._padding:-self._padding] = weights

            return extweights[tuple(self._fmapijk.T)]
        return np.ones((len(self._fmapxyz)))

    def _evaluate_bspline(self):
        """ Calculates the design matrix """
        print('[%s] Evaluating tensor-product cubic BSpline on %d points, %d control points' %
              (dt.now(), len(self._fmapxyz), len(self._knots_xyz)))
        self._X = tbspl_eval(self._fmapxyz, self._knots_xyz, self._knots_zooms, njobs=self._njobs)
        print('[%s] Finished BSpline evaluation' % dt.now())

    def fit(self):
        self._evaluate_bspline()

        fieldata = self._data[tuple(self._fmapijk.T)]

        print('[%s] Starting least-squares fitting using %d unmasked points' %
              (dt.now(), len(fieldata[self._weights > 0.0])))
        self._coeff = np.linalg.lstsq(
            self._X[self._weights > 0.0, ...].toarray(),
            fieldata[self._weights > 0.0])[0]
        print('[%s] Finished least-squares fitting' % dt.now())

    def get_coeffmap(self):
        self._knots_grid[tuple(self._knots_ijk.T)] = self._coeff
        return nb.Nifti1Image(self._knots_grid, self._knots_aff, None)

    def get_smoothed(self):
        self._smoothed = np.zeros_like(self._data)
        coords = tuple(self._fmapijk[self._frameijk].T)
        self._smoothed[coords] = self._X[self._frameijk].dot(self._coeff)

        output_image = self._smoothed[self._padding:-self._padding,
                                      self._padding:-self._padding,
                                      self._padding:-self._padding]
        return nb.Nifti1Image(output_image, self._fmapnii.affine, self._fmapnii.header)

    def invert(self):
        targets = self._fmapxyz.copy()
        targets[:, self._pedir] += self._smoothed[tuple(self._fmapijk.T)]
        print('[%s] Inverting transform :: evaluating tensor-product cubic BSpline on %d points, '
              '%d control points' % (dt.now(), len(targets), len(self._knots_xyz)))
        self._Xinv = tbspl_eval(targets, self._knots_xyz, self._knots_zooms, self._njobs)
        print('[%s] Finished BSpline evaluation, %s' %
              (dt.now(), str(self._Xinv.shape)))

        print('[%s] Starting least-squares fitting using %d unmasked points' %
              (dt.now(), len(targets)))
        self._invcoeff = np.linalg.lstsq(
            self._Xinv, self._fmapxyz[:, self._pedir] - targets[:, self._pedir])[0]
        print('[%s] Finished least-squares fitting' % dt.now())

    def get_inverted(self):
        self._inverted = np.zeros_like(self._data)
        self._inverted[tuple(self._fmapijk.T)] = self._X.dot(self._invcoeff)
        return nb.Nifti1Image(self._inverted, self._fmapnii.affine, self._fmapnii.header)

    def interp(self, in_data, inverse=False, fwd_pe=True):
        dshape = tuple(in_data.shape)
        gridxyz = self._fmapxyz.reshape((dshape[0], dshape[1], dshape[2], -1))

        x = gridxyz[:, 0, 0, 0]
        y = gridxyz[0, :, 0, 1]
        z = gridxyz[0, 0, :, 2]

        targets = self._fmapxyz.copy()

        if inverse:
            factor = 1.0 if fwd_pe else -1.0
            targets[:, self._pedir] += factor * \
                self._inverted[tuple(self._fmapijk.T)]
        else:
            targets[:, self._pedir] += self._smoothed[tuple(self._fmapijk.T)]

        interpolated = np.zeros_like(self._data)
        interpolated[tuple(self._fmapijk.T)] = interpn(
            (x, y, z), in_data, [tuple(v) for v in targets],
            bounds_error=False, fill_value=0)

        return nb.Nifti1Image(interpolated, self._fmapnii.affine, self._fmapnii.header)


def get_ijk(data, offset=0):
    """
    Calculates voxel coordinates from data
    """
    from numpy import mgrid

    if not isinstance(offset, (list, tuple)):
        offset = [offset] * 3

    grid = mgrid[offset[0]:(offset[0] + data.shape[0]),
                 offset[1]:(offset[1] + data.shape[1]),
                 offset[2]:(offset[2] + data.shape[2])]
    return grid.reshape(3, -1).T


def compute_affine(data, zooms):
    """
    Compose a RAS affine mat, since the affine of the image might not be RAS
    """
    aff = np.eye(4) * (list(zooms) + [1])
    aff[:3, 3] -= aff[:3, :3].dot(np.array(data.shape[:3], dtype=float) - 1.0) * 0.5
    return aff


def tbspl_eval(points, knots, zooms, njobs=None):
    """
    Evaluate tensor product BSpline
    """
    from scipy.sparse import vstack
    from fmriprep.utils.maths import bspl

    points = np.array(points, dtype=float)
    knots = np.array(knots, dtype=float)
    vbspl = np.vectorize(bspl)

    if njobs is not None and njobs < 1:
        njobs = None

    if njobs == 1:
        coeffs = [_evalp((p, knots, vbspl, zooms)) for p in points]
    else:
        from multiprocessing import Pool
        pool = Pool(processes=njobs, maxtasksperchild=100)
        coeffs = pool.map(
            _evalp, [(p, knots, vbspl, zooms) for p in points])
        pool.close()
        pool.join()

    return vstack(coeffs)


def _evalp(args):
    import numpy as np
    from scipy.sparse import csr_matrix

    point, knots, vbspl, zooms = args
    u_vec = (knots - point[np.newaxis, ...]) / zooms[np.newaxis, ...]
    c = vbspl(u_vec.reshape(-1)).reshape((knots.shape[0], 3)).prod(axis=1)
    return csr_matrix(c)
