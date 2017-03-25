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
                                    File, isdefined, traits, InputMultiPath, Str)
from nipype.interfaces import fsl
from niworkflows.interfaces.registration import FUGUERPT
from fmriprep.utils.misc import genfname

LOGGER = logging.getLogger('interfaces')


class WarpReferenceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_mask = File(exists=True, mandatory=True, desc='mask of fieldmap')
    fmap_ref = File(exists=True, mandatory=True, desc='reference file')
    echospacing = traits.Float(mandatory=True,
                               desc='effective echo spacing or dwell time')
    pe_dir = traits.Enum('y', 'x', 'y-', 'x-', mandatory=True,
                         desc='phase encoding direction')

class WarpReferenceOutputSpec(TraitedSpec):
    out_warped = File(desc='the "fmap_ref" image after warping')
    out_mask = File(desc='the corresponding fieldmap mask warped')

class WarpReference(BaseInterface):
    """
    The WarpReference interface wraps a workflow to generate
    a warped version of the reference (generally magnitude) image.
    This way, the reference image is geometrically closer to the
    target EPI and thus, easier to register.
    """
    input_spec = WarpReferenceInputSpec
    output_spec = WarpReferenceOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(WarpReference, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        eec = self.inputs.echospacing
        ped = self.inputs.pe_dir

        vsm_file = genfname(self.inputs.fmap_ref, 'vsm')
        gen_vsm = fsl.FUGUE(
            fmap_in_file=self.inputs.in_file, dwell_time=eec,
            unwarp_direction=ped, shift_out_file=vsm_file)
        gen_vsm.run()

        fugue = fsl.FUGUE(
            in_file=self.inputs.fmap_ref, shift_in_file=vsm_file, nokspace=True,
            forward_warping=True, unwarp_direction=ped, icorr=False,
            warped_file=genfname(self.inputs.fmap_ref, 'warped%s' % ped)).run()

        fuguemask = fsl.FUGUE(
            in_file=self.inputs.in_mask, shift_in_file=vsm_file, nokspace=True,
            forward_warping=True, unwarp_direction=ped, icorr=False,
            warped_file=genfname(self.inputs.in_mask, 'maskwarped%s' % ped)).run()

        masknii = nb.load(fuguemask.outputs.warped_file)
        mask = masknii.get_data()
        mask[mask > 0] = 1
        mask[mask < 1] = 0
        out_mask = genfname(self.inputs.in_mask, 'warp_bin')
        nb.Nifti1Image(mask.astype(np.uint8), masknii.affine,
                       masknii.header).to_filename(out_mask)

        self._results['out_warped'] = fugue.outputs.warped_file
        self._results['out_mask'] = out_mask

        return runtime

class ApplyFieldmapInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='unwarping target file')
    in_vsm = File(exists=True, mandatory=True, desc='input voxel shift map')
    pe_dir = traits.Enum('y', 'x', 'y-', 'x-', mandatory=True,
                         desc='phase encoding direction')
    generate_report = traits.Bool(False, usedefault=True, desc='generate report')

class ApplyFieldmapOutputSpec(TraitedSpec):
    out_corrected = File(desc='the "fmap_ref" image after warping')

class ApplyFieldmap(BaseInterface):
    """
    The ApplyFieldmap interface simplifies the FUGUE interface
    for unwarping, including the (optional) application of head
    motion parameters to the input vsm (voxel shift map)
    """
    input_spec = ApplyFieldmapInputSpec
    output_spec = ApplyFieldmapOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ApplyFieldmap, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        ped = self.inputs.pe_dir

        in_file = self.inputs.in_file
        im = nb.as_closest_canonical(nb.load(in_file))
        oldaff = im.affine
        newaff = np.eye(4)
        newaff[:3, :3] = np.eye(3) * im.header.get_zooms()[:3]
        newaff[:3, 3] -= 0.5 * newaff[:3, :3].dot(im.shape[:3])

        in_file = genfname(in_file, suffix='nosform')
        nb.Nifti1Image(im.get_data(), newaff, None).to_filename(
            in_file)

        fugue = FUGUERPT(
            in_file=in_file, shift_in_file=self.inputs.in_vsm,
            icorr=True, unwarp_direction=ped,
            generate_report=self.inputs.generate_report,
            unwarped_file=genfname(in_file, 'unwarped')).run()

        # Restore headers
        out_file = genfname(self.inputs.in_file, 'unwarped')
        nb.Nifti1Image(nb.load(fugue.outputs.unwarped_file).get_data(),
                       oldaff, im.header).to_filename(out_file)

        self._results['out_corrected'] = out_file
        return runtime


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
    njobs = traits.Int(1, usedefault=True, nohash=True, desc='number of jobs')

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

        mask = None
        if isdefined(self.inputs.in_mask):
            masknii = nb.load(self.inputs.in_mask)
            mask = masknii.get_data().astype(np.uint8)

            # Dilate mask
            if self.inputs.mask_erode > 0:
                struc = sim.iterate_structure(sim.generate_binary_structure(3, 2), 1)
                mask = sim.binary_erosion(
                    mask, struc,
                    iterations=self.inputs.mask_erode).astype(np.uint8)  # pylint: disable=no-member

        self._results['out_file'] = genfname(self.inputs.in_file, suffix='enh')
        self._results['out_coeff'] = genfname(self.inputs.in_file, suffix='coeff')

        datanii = nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header)
        # data interpolation
        datanii.to_filename(self._results['out_file'])

        if self.inputs.bspline_smooth:
            from fmriprep.utils import bspline as fbsp
            from statsmodels.robust.scale import mad

            # Fit BSplines (coarse)
            bspobj = fbsp.BSplineFieldmap(datanii, weights=mask, njobs=self.inputs.njobs)
            bspobj.fit()
            smoothed1 = bspobj.get_smoothed()

            # Manipulate the difference map
            diffmap = data - smoothed1.get_data()
            sderror = mad(diffmap[mask > 0])
            errormask = np.zeros_like(diffmap)
            errormask[np.abs(diffmap) > (10 * sderror)] = 1
            errormask *= mask
            errorslice = np.squeeze(np.argwhere(errormask.sum(0).sum(0) > 0))

            if (errorslice[-1] - errorslice[0]) > 1:
                diffmapmsk = mask[..., errorslice[0]:errorslice[-1]]
                diffmapnii = nb.Nifti1Image(
                    diffmap[..., errorslice[0]:errorslice[-1]] * diffmapmsk,
                    datanii.affine, datanii.header)


                bspobj2 = fbsp.BSplineFieldmap(diffmapnii, knots_zooms=[24., 24., 4.],
                                               njobs=self.inputs.njobs)
                bspobj2.fit()
                smoothed2 = bspobj2.get_smoothed().get_data()

                final = smoothed1.get_data().copy()
                final[..., errorslice[0]:errorslice[-1]] += smoothed2
            else:
                final = smoothed1

            nb.Nifti1Image(final, datanii.affine, datanii.header).to_filename(
                self._results['out_file'])

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
