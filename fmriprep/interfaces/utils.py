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

    probability_map_nii = resample_to_img(in_file, epi_mask)
    probability_map_data = probability_map_nii.get_data()

    # thresholding
    probability_map_data[probability_map_data < 0.95] = 0
    probability_map_data[probability_map_data != 0] = 1

    epi_mask_nii = nb.load(epi_mask)
    epi_mask_data = epi_mask_nii.get_data()
    if epi_mask_erosion_mm:
        epi_mask_data = nd.binary_erosion(
            epi_mask_data,
            iterations=int(epi_mask_erosion_mm /
                           max(probability_map_nii.header.get_zooms()))).astype(int)
        eroded_mask_file = os.path.abspath("erodd_mask.nii.gz")
        nb.Nifti1Image(epi_mask_data, epi_mask_nii.affine,
                       epi_mask_nii.header).to_filename(eroded_mask_file)
    else:
        eroded_mask_file = epi_mask
    probability_map_data[epi_mask_data != 1] = 0

    # shrinking
    if erosion_mm:
        iter_n = int(erosion_mm/max(probability_map_nii.header.get_zooms()))
        probability_map_data = nd.binary_erosion(probability_map_data,
                                                 iterations=iter_n).astype(int)

    new_nii = nb.Nifti1Image(probability_map_data, probability_map_nii.affine,
                             probability_map_nii.header)
    new_nii.to_filename("roi.nii.gz")
    return os.path.abspath("roi.nii.gz"), eroded_mask_file
