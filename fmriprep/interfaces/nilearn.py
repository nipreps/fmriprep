#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import nibabel as nb
import numpy as np
from skimage import morphology as sim
from scipy.ndimage.morphology import binary_fill_holes

from nilearn.masking import compute_epi_mask
from nilearn.image import concat_imgs

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec,
    File, InputMultiPath, SimpleInterface
)

LOGGER = logging.getLogger('nipype.interface')


class MaskEPIInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input EPI or list of files')
    lower_cutoff = traits.Float(0.2, usedefault=True)
    upper_cutoff = traits.Float(0.85, usedefault=True)
    connected = traits.Bool(True, usedefault=True)
    enhance_t2 = traits.Bool(False, usedefault=True,
                             desc='enhance T2 contrast on image')
    opening = traits.Int(2, usedefault=True)
    closing = traits.Bool(True, usedefault=True)
    fill_holes = traits.Bool(True, usedefault=True)
    exclude_zeros = traits.Bool(False, usedefault=True)
    ensure_finite = traits.Bool(True, usedefault=True)
    target_affine = traits.Either(None, traits.File(exists=True),
                                  default=None, usedefault=True)
    target_shape = traits.Either(None, traits.File(exists=True),
                                 default=None, usedefault=True)
    no_sanitize = traits.Bool(False, usedefault=True)


class MaskEPIOutputSpec(TraitedSpec):
    out_mask = File(exists=True, desc='output mask')


class MaskEPI(SimpleInterface):
    input_spec = MaskEPIInputSpec
    output_spec = MaskEPIOutputSpec

    def _run_interface(self, runtime):

        in_files = self.inputs.in_files

        if self.inputs.enhance_t2:
            in_files = [_enhance_t2_contrast(f, newpath=runtime.cwd)
                        for f in in_files]

        masknii = compute_epi_mask(
            in_files,
            lower_cutoff=self.inputs.lower_cutoff,
            upper_cutoff=self.inputs.upper_cutoff,
            connected=self.inputs.connected,
            opening=self.inputs.opening,
            exclude_zeros=self.inputs.exclude_zeros,
            ensure_finite=self.inputs.ensure_finite,
            target_affine=self.inputs.target_affine,
            target_shape=self.inputs.target_shape
        )

        if self.inputs.closing:
            closed = sim.binary_closing(masknii.get_data().astype(
                np.uint8), sim.ball(1)).astype(np.uint8)
            masknii = masknii.__class__(closed, masknii.affine,
                                        masknii.header)

        if self.inputs.fill_holes:
            filled = binary_fill_holes(masknii.get_data().astype(
                np.uint8), sim.ball(6)).astype(np.uint8)
            masknii = masknii.__class__(filled, masknii.affine,
                                        masknii.header)

        if self.inputs.no_sanitize:
            in_file = self.inputs.in_files
            if isinstance(in_file, list):
                in_file = in_file[0]
            nii = nb.load(in_file)
            qform, code = nii.get_qform(coded=True)
            masknii.set_qform(qform, int(code))
            sform, code = nii.get_sform(coded=True)
            masknii.set_sform(sform, int(code))

        self._results['out_mask'] = fname_presuffix(
            self.inputs.in_files[0], suffix='_mask', newpath=runtime.cwd)
        masknii.to_filename(self._results['out_mask'])
        return runtime


class MergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input list of files to merge')
    dtype = traits.Enum('f4', 'f8', 'u1', 'u2', 'u4', 'i2', 'i4',
                        usedefault=True, desc='numpy dtype of output image')
    header_source = File(exists=True, desc='a Nifti file from which the header should be copied')
    compress = traits.Bool(True, usedefault=True, desc='Use gzip compression on .nii output')


class MergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output merged file')


class Merge(SimpleInterface):
    input_spec = MergeInputSpec
    output_spec = MergeOutputSpec

    def _run_interface(self, runtime):
        ext = '.nii.gz' if self.inputs.compress else '.nii'
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_files[0], suffix='_merged' + ext, newpath=runtime.cwd, use_ext=False)
        new_nii = concat_imgs(self.inputs.in_files, dtype=self.inputs.dtype)

        if isdefined(self.inputs.header_source):
            src_hdr = nb.load(self.inputs.header_source).header
            new_nii.header.set_xyzt_units(t=src_hdr.get_xyzt_units()[-1])
            new_nii.header.set_zooms(list(new_nii.header.get_zooms()[:3]) +
                                     [src_hdr.get_zooms()[3]])

        new_nii.to_filename(self._results['out_file'])

        return runtime


def _enhance_t2_contrast(in_file, newpath=None, offset=0.5):
    """
    Performs a logarithmic transformation of intensity that
    effectively splits brain and background and makes the
    overall distribution more Gaussian.
    """
    out_file = fname_presuffix(in_file, suffix='_t1enh',
                               newpath=newpath)
    nii = nb.load(in_file)
    data = nii.get_data()
    maxd = data.max()
    newdata = np.log(offset + data / maxd)
    newdata -= newdata.min()
    newdata *= maxd / newdata.max()
    nii = nii.__class__(newdata, nii.affine, nii.header)
    nii.to_filename(out_file)
    return out_file
