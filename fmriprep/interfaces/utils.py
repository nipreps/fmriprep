#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals

import os.path as op
import numpy as np
import nibabel as nb
from nilearn.image import mean_img
from nipype.interfaces.base import (traits, TraitedSpec, BaseInterface,
                                    BaseInterfaceInputSpec, File, InputMultiPath)
from fmriprep.utils.misc import genfname


class MeanTimeseriesInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input file or list of files')


class MeanTimeseriesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')

class MeanTimeseries(BaseInterface):
    input_spec = MeanTimeseriesInputSpec
    output_spec = MeanTimeseriesOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(MeanTimeseries, self).__init__(**inputs)

    def _run_interface(self, runtime):
        nii = mean_img([nb.load(fname) for fname in self.inputs.in_files])

        self._results['out_file'] = genfname(
            self.inputs.in_files[0], suffix='avg')
        nii.to_filename(self._results['out_file'])
        return runtime

    def _list_outputs(self):
        return self._results

class ApplyMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    in_mask = File(exists=True, mandatory=True, desc='input mask')


class ApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')

class ApplyMask(BaseInterface):
    input_spec = ApplyMaskInputSpec
    output_spec = ApplyMaskOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ApplyMask, self).__init__(**inputs)

    def _run_interface(self, runtime):
        out_file = genfname(self.inputs.in_file, 'brainmask')
        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()
        data[nb.load(self.inputs.in_mask).get_data() <= 0] = 0
        nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime

    def _list_outputs(self):
        return self._results


class FormatHMCParamInputSpec(BaseInterfaceInputSpec):
    translations = traits.List(traits.Tuple(traits.Float, traits.Float, traits.Float),
                               mandatory=True, desc='three translations in mm')
    rot_angles = traits.List(traits.Tuple(traits.Float, traits.Float, traits.Float),
                             mandatory=True, desc='three rotations in rad')
    fmt = traits.Enum('confounds', 'movpar_file', usedefault=True,
                      desc='type of resulting file')


class FormatHMCParamOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')

class FormatHMCParam(BaseInterface):
    input_spec = FormatHMCParamInputSpec
    output_spec = FormatHMCParamOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(FormatHMCParam, self).__init__(**inputs)

    def _run_interface(self, runtime):
        self._results['out_file'] = _tsv_format(
            self.inputs.translations, self.inputs.rot_angles,
            fmt=self.inputs.fmt)
        return runtime

    def _list_outputs(self):
        return self._results


def _tsv_format(translations, rot_angles, fmt='confounds'):
    parameters = np.hstack((translations, rot_angles)).astype(np.float32)

    if fmt == 'movpar_file':
        out_file = op.abspath('movpar.txt')
        np.savetxt(out_file, parameters)
    elif fmt == 'confounds':
        out_file = op.abspath('movpar.tsv')
        np.savetxt(out_file, parameters,
                   header='X\tY\tZ\tRotX\tRotY\tRotZ',
                   delimiter='\t')
    else:
        raise NotImplementedError

    return out_file

def nii_concat(in_files, header_source=None):
    from nibabel.funcs import concat_images
    import nibabel as nb
    import os
    from nibabel.funcs import concat_images
    new_nii = concat_images(in_files, check_affines=False)

    if header_source:
        header_nii = nb.load(header_source)
        new_nii.header.set_xyzt_units(t=header_nii.header.get_xyzt_units()[-1])
        new_nii.header.set_zooms(list(new_nii.header.get_zooms()[:3]) + [header_nii.header.get_zooms()[3]])

    new_nii.to_filename("merged.nii.gz")

    return os.path.abspath("merged.nii.gz")

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
        epi_mask_data = nd.binary_erosion(epi_mask_data,
                                      iterations=int(epi_mask_erosion_mm/max(probability_map_nii.header.get_zooms()))).astype(int)
        eroded_mask_file = os.path.abspath("erodd_mask.nii.gz")
        nb.Nifti1Image(epi_mask_data, epi_mask_nii.affine, epi_mask_nii.header).to_filename(eroded_mask_file)
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
