#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals

import nibabel as nb
from niworkflows.nipype.interfaces.base import TraitedSpec, BaseInterfaceInputSpec, File
from niworkflows.interfaces.base import SimpleInterface

from fmriprep.utils.misc import genfname


class ApplyMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    in_mask = File(exists=True, mandatory=True, desc='input mask')


class ApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')


class ApplyMask(SimpleInterface):
    input_spec = ApplyMaskInputSpec
    output_spec = ApplyMaskOutputSpec

    def _run_interface(self, runtime):
        out_file = genfname(self.inputs.in_file, 'brainmask')
        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()
        data[nb.load(self.inputs.in_mask).get_data() <= 0] = 0
        nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


def prepare_roi_from_probtissue(in_file, epi_mask, epi_mask_erosion_mm=0,
                                erosion_mm=0):
    import os
    import nibabel as nb
    import scipy.ndimage as nd
    from nilearn.image import resample_to_img

    probability_map = resample_to_img(in_file, epi_mask)
    max_zoom = max(probability_map.header.get_zooms())

    epi_mask_nii = nb.load(epi_mask)
    epi_mask_data = epi_mask_nii.get_data()
    if epi_mask_erosion_mm:
        epi_mask_data = nd.binary_erosion(
            epi_mask_data, iterations=(epi_mask_erosion_mm // max_zoom)).astype('u1')
        eroded_mask_file = os.path.abspath("eroded_mask.nii.gz")
        img = nb.Nifti1Image(epi_mask_data, epi_mask_nii.affine, epi_mask_nii.header)
        img.set_data_dtype('u1')
        img.to_filename(eroded_mask_file)
    else:
        eroded_mask_file = epi_mask

    probability_mask = epi_mask_data * (probability_map.get_data() >= 0.95)

    # shrinking
    if erosion_mm:
        probability_mask = nd.binary_erosion(probability_mask,
                                             iterations=erosion_mm // max_zoom).astype('u1')

    new_nii = nb.Nifti1Image(probability_mask, probability_map.affine, probability_map.header)
    new_nii.set_data_dtype('u1')
    new_nii.to_filename("roi.nii.gz")
    return os.path.abspath("roi.nii.gz"), eroded_mask_file
