#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to deal with the various types of fieldmap sources

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os.path as op
from shutil import copy
from builtins import range
import numpy as np
import nibabel as nb
from nipype import logging
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
                                    File, isdefined, traits, InputMultiPath)
from nipype.interfaces import fsl
from fmriprep.utils.misc import genfname

LOGGER = logging.getLogger('interfaces')


class FieldCoefficientsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_ref = File(exists=True, mandatory=True, desc='reference file')
    in_movpar = File(exists=True, desc='input head motion parameters')

class FieldCoefficientsOutputSpec(TraitedSpec):
    out_fieldcoef = File(desc='the calculated BSpline coefficients')
    out_movpar = File(desc='the calculated head motion coefficients')

class FieldCoefficients(BaseInterface):
    """
    The FieldCoefficients interface wraps a workflow to compute the BSpline coefficients
    corresponding to the input fieldmap (in Hz). It also sets the appropriate nifti headers
    to be digested by ApplyTOPUP.
    """
    input_spec = FieldCoefficientsInputSpec
    output_spec = FieldCoefficientsOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(FieldCoefficients, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        movpar = None
        if isdefined(self.inputs.in_movpar):
            movpar = self.inputs.in_movpar

        self._results['out_fieldcoef'], self._results['out_movpar'] = _gen_coeff(
            self.inputs.in_file, self.inputs.in_ref, movpar)
        return runtime

class GenerateMovParamsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input - a mcflirt output file')
    in_mats = InputMultiPath(File(exists=True), mandatory=True,
                             desc='matrices - mcflirt output matrices')


class GenerateMovParamsOutputSpec(TraitedSpec):
    out_movpar = File(desc='the calculated head motion coefficients')

class GenerateMovParams(BaseInterface):
    """
    The GenerateMovParams interface generates TopUp compatible movpar files
    """
    input_spec = GenerateMovParamsInputSpec
    output_spec = GenerateMovParamsOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(GenerateMovParams, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        # For some reason, MCFLIRT's parameters
        # are not compatible, fill with zeroes for now
        # see https://github.com/poldracklab/fmriprep/issues/218
        # ntsteps = nb.load(self.inputs.in_file).get_shape()[-1]
        ntsteps = len(self.inputs.in_mats)
        self._results['out_movpar'] = genfname(
            self.inputs.in_file, suffix='movpar')

        np.savetxt(self._results['out_movpar'], np.zeros((ntsteps, 6)))
        return runtime


class FieldEnhanceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_mask = File(exists=True, desc='brain mask')
    despike = traits.Bool(True, usedefault=True, desc='run despike filter')
    bspline_smooth = traits.Bool(True, usedefault=True, desc='run 3D bspline smoother')
    mask_erode = traits.Int(1, usedefault=True, desc='mask erosion iterations')
    despike_threshold = traits.Float(0.2, usedefault=True, desc='mask erosion iterations')


class FieldEnhanceOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')
    out_coeff = File(desc='write bspline coefficients')

class FieldEnhance(BaseInterface):
    """
    The FieldEnhance interface wraps a workflow to massage the input fieldmap
    and return it masked, despiked, etc.
    """
    input_spec = FieldEnhanceInputSpec
    output_spec = FieldEnhanceOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(FieldEnhance, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        from scipy import ndimage as sim

        fmap_nii = nb.load(self.inputs.in_file)
        data = np.squeeze(fmap_nii.get_data().astype(np.float32))

        # Despike / denoise (no-mask)
        if self.inputs.despike:
            data = _despike2d(data, self.inputs.despike_threshold)

        masknii = None
        if isdefined(self.inputs.in_mask):
            masknii = nb.load(self.inputs.in_mask)
            mask = masknii.get_data().astype(np.uint8)

            # Dilate mask
            if self.inputs.mask_erode > 0:
                struc = sim.iterate_structure(sim.generate_binary_structure(3, 2), 1)
                mask = sim.binary_erosion(
                    mask, struc,
                    iterations=self.inputs.mask_erode).astype(np.uint8)  # pylint: disable=no-member

            # Apply mask
            data[mask == 0] = 0.0

        self._results['out_file'] = genfname(self.inputs.in_file, suffix='enh')
        self._results['out_coeff'] = genfname(self.inputs.in_file, suffix='coeff')

        datanii = nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header)
        # data interpolation
        if self.inputs.bspline_smooth:
            datanii, coefnii = bspl_smoothing(datanii, masknii)
            coefnii.to_filename(self._results['out_coeff'])

        datanii.to_filename(self._results['out_file'])
        return runtime


class FieldMatchHistogramInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input enhanced fieldmap')
    in_reference = File(exists=True, mandatory=True, desc='input fieldmap')
    in_mask = File(exists=True, mandatory=True, desc='brain mask')

class FieldMatchHistogramOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')

class FieldMatchHistogram(BaseInterface):
    """
    The FieldMatchHistogram interface wraps a workflow to massage the input fieldmap
    and return it masked, despiked, etc.
    """
    input_spec = FieldMatchHistogramInputSpec
    output_spec = FieldMatchHistogramOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(FieldMatchHistogram, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        from scipy import ndimage as sim

        fmap_nii = nb.load(self.inputs.in_file)
        data = np.squeeze(fmap_nii.get_data().astype(np.float32))

        refdata = nb.load(self.inputs.in_reference).get_data()
        maskdata = nb.load(self.inputs.in_mask).get_data()

        ref85 = np.percentile(refdata[maskdata > 0], 85.0)
        ref15 = np.percentile(refdata[maskdata > 0], 15.0)

        fmap85 = np.percentile(data[maskdata > 0], 85.0)
        fmap15 = np.percentile(data[maskdata > 0], 15.0)


        data[data > 0] *= ref85 / fmap85
        data[data < 0] *= ref15 / fmap15

        self._results['out_file'] = genfname(self.inputs.in_file, suffix='histmatch')
        datanii = nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header)
        datanii.to_filename(self._results['out_file'])
        return runtime

def _despike2d(data, thres, neigh=None):
    """
    despiking as done in FSL fugue
    """

    if neigh is None:
        neigh = [-1, 0, 1]
    nslices = data.shape[-1]

    for k in range(nslices):
        data2d = data[..., k]

        for i in range(data2d.shape[0]):
            for j in range(data2d.shape[1]):
                vals = []
                thisval = data2d[i, j]
                for ii in neigh:
                    for jj in neigh:
                        try:
                            vals.append(data2d[i + ii, j + jj])
                        except IndexError:
                            pass
                vals = np.array(vals)
                patch_range = vals.max() - vals.min()
                patch_med = np.median(vals)

                if (patch_range > 1e-6 and
                        (abs(thisval - patch_med) / patch_range) > thres):
                    data[i, j, k] = patch_med
    return data


def _gen_coeff(in_file, in_ref, in_movpar=None):
    """Convert to a valid fieldcoeff"""


    def _get_fname(in_file):
        import os.path as op
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        return op.abspath(fname)

    out_topup = _get_fname(in_file)

    # 1. Add one dimension (4D image) of 3D coordinates
    #    so that this is a 3D deformation field
    im0 = nb.load(in_file)
    data = np.zeros_like(im0.get_data())
    sizes = data.shape[:3]
    spacings = im0.header.get_zooms()[:3]
    im1 = nb.Nifti1Image(data, im0.affine, im0.header)
    im4d = nb.concat_images([im0, im1, im1])
    im4d_fname = '{}_{}'.format(out_topup, 'field4D.nii.gz')
    im4d.to_filename(im4d_fname)

    # 2. Warputils to compute bspline coefficients
    to_coeff = fsl.WarpUtils(out_format='spline', knot_space=(2, 2, 2))
    to_coeff.inputs.in_file = im4d_fname
    to_coeff.inputs.reference = in_ref

    # 3. Remove unnecessary dims (Y and Z)
    get_first = fsl.ExtractROI(t_min=0, t_size=1)
    get_first.inputs.in_file = to_coeff.run().outputs.out_file

    # 4. Set correct header
    # see https://github.com/poldracklab/preprocessing-workflow/issues/92
    img = nb.load(get_first.run().outputs.roi_file)
    hdr = img.header.copy()
    hdr['intent_p1'] = spacings[0]
    hdr['intent_p2'] = spacings[1]
    hdr['intent_p3'] = spacings[2]
    hdr['intent_code'] = 2016

    sform = np.eye(4)
    sform[:3, 3] = sizes
    hdr.set_sform(sform, code='scanner')
    hdr['qform_code'] = 1

    out_movpar = '{}_movpar.txt'.format(out_topup)
    copy(in_movpar, out_movpar)

    out_fieldcoef = '{}_fieldcoef.nii.gz'.format(out_topup)
    nb.Nifti1Image(img.get_data(), None, hdr).to_filename(out_fieldcoef)

    return out_fieldcoef, out_movpar

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
    bspl_grid = np.zeros(tuple(knot_dims))
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


    print('[%s] Evaluating tensor-product cubic-bspline on %d points' % (dt.now(), len(points_xyz)))
    # Calculate design matrix
    X = tbspl_eval(points_xyz, bspl_xyz, knot_space)
    print('[%s] Finished, bspline grid has %d control points' % (dt.now(), len(bspl_xyz)))
    Y = data[points_ijk]


    # Fit coefficients
    print('[%s] Starting least-squares fitting' % dt.now())
    # coeff = (pinv2(X.T.dot(X)).dot(X.T)).dot(Y) # manual way (seems equally slow)
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
