#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to deal with the various types of fieldmap sources

"""

import numpy as np
import nibabel as nb
from niworkflows.nipype import logging
from niworkflows.nipype.utils.filemanip import fname_presuffix
from niworkflows.nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, isdefined, traits,
    SimpleInterface)
from ...due import Doi

LOGGER = logging.getLogger('interface')


class FieldEnhanceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_mask = File(exists=True, desc='brain mask')
    in_magnitude = File(exists=True, desc='input magnitude')
    unwrap = traits.Bool(False, usedefault=True, desc='run phase unwrap')
    despike = traits.Bool(True, usedefault=True, desc='run despike filter')
    bspline_smooth = traits.Bool(True, usedefault=True, desc='run 3D bspline smoother')
    mask_erode = traits.Int(1, usedefault=True, desc='mask erosion iterations')
    despike_threshold = traits.Float(0.2, usedefault=True, desc='mask erosion iterations')
    njobs = traits.Int(1, usedefault=True, nohash=True, desc='number of jobs')


class FieldEnhanceOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')
    out_unwrapped = File(desc='unwrapped fieldmap')


class FieldEnhance(SimpleInterface):
    """
    The FieldEnhance interface wraps a workflow to massage the input fieldmap
    and return it masked, despiked, etc.
    """
    input_spec = FieldEnhanceInputSpec
    output_spec = FieldEnhanceOutputSpec

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
                    iterations=self.inputs.mask_erode
                ).astype(np.uint8)  # pylint: disable=no-member

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_enh', newpath=runtime.cwd)
        datanii = nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header)

        if self.inputs.unwrap:
            data = _unwrap(data, self.inputs.in_magnitude, mask)
            self._results['out_unwrapped'] = fname_presuffix(
                self.inputs.in_file, suffix='_unwrap', newpath=runtime.cwd)
            nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header).to_filename(
                self._results['out_unwrapped'])

        if not self.inputs.bspline_smooth:
            datanii.to_filename(self._results['out_file'])
            return runtime
        else:
            from ..utils import bspline as fbsp
            from statsmodels.robust.scale import mad

            # Fit BSplines (coarse)
            bspobj = fbsp.BSplineFieldmap(datanii, weights=mask,
                                          njobs=self.inputs.njobs)
            bspobj.fit()
            smoothed1 = bspobj.get_smoothed()

            # Manipulate the difference map
            diffmap = data - smoothed1.get_data()
            sderror = mad(diffmap[mask > 0])
            LOGGER.info('SD of error after B-Spline fitting is %f', sderror)
            errormask = np.zeros_like(diffmap)
            errormask[np.abs(diffmap) > (10 * sderror)] = 1
            errormask *= mask

            nslices = 0
            try:
                errorslice = np.squeeze(np.argwhere(errormask.sum(0).sum(0) > 0))
                nslices = errorslice[-1] - errorslice[0]
            except IndexError:  # mask is empty, do not refine
                pass

            if nslices > 1:
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
                final = smoothed1.get_data()

            nb.Nifti1Image(final, datanii.affine, datanii.header).to_filename(
                self._results['out_file'])

        return runtime


class PhaseDiff2FieldmapInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True, mandatory=True, desc='Input phdiff file')
    delta_te = traits.Float(mandatory=True, desc='Echo time (TE) difference')


class PhaseDiff2FieldmapOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True)


class PhaseDiff2Fieldmap(SimpleInterface):
    r"""
    Converts the input phase-difference map into a fieldmap in Hz,
    using the eq. (1) of [Hutton2002]_:

    .. math::

        \Delta B_0 (\text{T}^{-1}) = \frac{\Delta \Theta}{2\pi\gamma \Delta\text{TE}}


    In this case, we do not take into account the gyromagnetic ratio of the
    proton (:math:`\gamma`), since it will be applied inside TOPUP:

    .. math::

        \Delta B_0 (\text{Hz}) = \frac{\Delta \Theta}{2\pi \Delta\text{TE}}


    .. [Hutton2002] Hutton et al., Image Distortion Correction in fMRI: A Quantitative
                    Evaluation, NeuroImage 16(1):217-240, 2002. doi:`10.1006/nimg.2001.1054
                    <http://dx.doi.org/10.1006/nimg.2001.1054>`_.

    """
    input_spec = PhaseDiff2FieldmapInputSpec
    output_spec = PhaseDiff2FieldmapOutputSpec
    references_ = [{
        'entry': Doi('10.1006/nimg.2001.1054'),
        'description': "Conversion of the input phase-difference map into a fieldmap in Hz",
        'tags': ['edu']}]

    def _run_interface(self, runtime):
        self.outputs['out_file'] = phdiff2fmap(self.inputs.in_file, self.inputs.delta_te)
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


def _unwrap(fmap_data, mag_file, mask=None):
    from math import pi
    from niworkflows.nipype.interfaces.fsl import PRELUDE
    magnii = nb.load(mag_file)

    if mask is None:
        mask = np.ones_like(fmap_data, dtype=np.uint8)

    fmapmax = max(abs(fmap_data[mask > 0].min()), fmap_data[mask > 0].max())
    fmap_data *= pi / fmapmax

    nb.Nifti1Image(fmap_data, magnii.affine).to_filename('fmap_rad.nii.gz')
    nb.Nifti1Image(mask, magnii.affine).to_filename('fmap_mask.nii.gz')
    nb.Nifti1Image(magnii.get_data(), magnii.affine).to_filename('fmap_mag.nii.gz')

    # Run prelude
    res = PRELUDE(phase_file='fmap_rad.nii.gz',
                  magnitude_file='fmap_mag.nii.gz',
                  mask_file='fmap_mask.nii.gz').run()

    unwrapped = nb.load(res.outputs.unwrapped_phase_file).get_data() * (fmapmax / pi)
    return unwrapped


# The phdiff2fmap interface is equivalent to:
# rad2rsec (using rads2radsec from nipype.workflows.dmri.fsl.utils)
# pre_fugue = pe.Node(fsl.FUGUE(save_fmap=True), name='ComputeFieldmapFUGUE')
# rsec2hz (divide by 2pi)
def phdiff2fmap(in_file, delta_te, out_file=None):
    r"""
    Converts the input phase-difference map into a fieldmap in Hz,
    using the eq. (1) of [Hutton2002]_:

    .. math::

        \Delta B_0 (\text{T}^{-1}) = \frac{\Delta \Theta}{2\pi\gamma \Delta\text{TE}}


    In this case, we do not take into account the gyromagnetic ratio of the
    proton (:math:`\gamma`), since it will be applied inside TOPUP:

    .. math::

        \Delta B_0 (\text{Hz}) = \frac{\Delta \Theta}{2\pi \Delta\text{TE}}


    .. [Hutton2002] Hutton et al., Image Distortion Correction in fMRI: A Quantitative
                    Evaluation, NeuroImage 16(1):217-240, 2002. doi:`10.1006/nimg.2001.1054
                    <http://dx.doi.org/10.1006/nimg.2001.1054>`_.

    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    import math

    #  GYROMAG_RATIO_H_PROTON_MHZ = 42.576

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_fmap.nii.gz' % fname)

    image = nb.load(in_file)
    data = (image.get_data().astype(np.float32) / (2. * math.pi * delta_te))

    nb.Nifti1Image(data, image.affine, image.header).to_filename(out_file)
    return out_file
