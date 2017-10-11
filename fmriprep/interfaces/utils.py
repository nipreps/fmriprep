#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utilities
^^^^^^^^^^^^^^^^^^^^^^^

"""

import os
import numpy as np
import nibabel as nb
import scipy.ndimage as nd
from nilearn.image import resample_to_img

from niworkflows.nipype.utils.filemanip import fname_presuffix
from niworkflows.nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiPath,
    SimpleInterface
)


class TPM2ROIInputSpec(BaseInterfaceInputSpec):
    t1_tpm = File(exists=True, mandatory=True, desc='Tissue probability map file in T1 space')
    t1_mask = File(exists=True, mandatory=True, desc='Binary mask of skull-stripped T1w image')
    bold_mask = File(exists=True, mandatory=True, desc='Binary mask of skull-stripped BOLD image')
    mask_erode_mm = traits.Float(xor=['mask_erode_prop'],
                                 desc='erode input mask (kernel width in mm)')
    erode_mm = traits.Float(xor=['erode_prop'],
                            desc='erode output mask (kernel width in mm)')
    mask_erode_prop = traits.Float(xor=['mask_erode_mm'],
                                   desc='erode input mask (target volume ratio)')
    erode_prop = traits.Float(xor=['erode_mm'],
                              desc='erode output mask (target volume ratio)')
    prob_thresh = traits.Float(0.95, usedefault=True,
                               desc='threshold for the tissue probability maps')


class TPM2ROIOutputSpec(TraitedSpec):
    roi_file = File(exists=True, desc='output ROI file')
    eroded_mask = File(exists=True, desc='resulting eroded mask')


class TPM2ROI(SimpleInterface):
    """Convert tissue probability maps (TPMs) into ROIs

    This interface follows the following logic:

    #. Erode ``t1_mask`` by ``mask_erode_mm`` and apply to ``t1_tpm``
    #. Threshold masked TPM at ``prob_thresh``
    #. Erode resulting mask by ``erode_mm``

    Both the eroded brain mask and eroded ROI mask are then resampled to BOLD
    resolution, and masked by ``bold_mask``.

    """

    input_spec = TPM2ROIInputSpec
    output_spec = TPM2ROIOutputSpec

    def _run_interface(self, runtime):
        mask_erode_mm = self.inputs.mask_erode_mm
        if not isdefined(mask_erode_mm):
            mask_erode_mm = None
        erode_mm = self.inputs.erode_mm
        if not isdefined(erode_mm):
            erode_mm = None
        mask_erode_prop = self.inputs.mask_erode_prop
        if not isdefined(mask_erode_prop):
            mask_erode_prop = None
        erode_prop = self.inputs.erode_prop
        if not isdefined(erode_prop):
            erode_prop = None
        roi_file, eroded_mask = _tpm2roi(
            self.inputs.t1_tpm,
            self.inputs.t1_mask,
            self.inputs.bold_mask,
            mask_erode_mm,
            erode_mm,
            mask_erode_prop,
            erode_prop,
            self.inputs.prob_thresh
        )
        self._results['roi_file'] = roi_file
        self._results['eroded_mask'] = eroded_mask
        return runtime


class CombineROIsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='input list of ROIs')
    ref_header = File(exists=True, mandatory=True,
                      desc='reference NIfTI file with desired output header/affine')


class CombineROIsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='union of binarized input files')


class CombineROIs(SimpleInterface):
    """Generate the union (logical or) of a series of ROI masks"""
    input_spec = CombineROIsInputSpec
    output_spec = CombineROIsOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _combine_rois(self.inputs.in_files, self.inputs.ref_header)
        return runtime


class ConcatROIsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    in_mask = File(exists=True, mandatory=True, desc='input mask')
    ref_header = File(exists=True, mandatory=True,
                      desc='reference NIfTI file with desired output header/affine')


class ConcatROIsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')


class ConcatROIs(SimpleInterface):
    """Concatenate two ROI files along time axis

    ``in_file`` is resampled to match ``in_mask``, and the two concatenated
    datasets are saved with the affine and header of ``ref_header``.
    """
    input_spec = ConcatROIsInputSpec
    output_spec = ConcatROIsOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _concat_rois(
            self.inputs.in_file, self.inputs.in_mask, self.inputs.ref_header)
        return runtime


class AddTSVHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    columns = traits.List(traits.Str, mandatory=True, desc='header for columns')


class AddTSVHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')


class AddTSVHeader(SimpleInterface):
    """Add a header row to a TSV file

    .. testsetup::

    >>> import os
    >>> import pandas as pd
    >>> import numpy as np
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)

    .. doctest::

    An example TSV:

    >>> np.savetxt('data.tsv', np.arange(30).reshape((6, 5)), delimiter='\t')

    Add headers:

    >>> from fmriprep.interfaces import AddTSVHeader
    >>> addheader = AddTSVHeader()
    >>> addheader.inputs.in_file = 'data.tsv'
    >>> addheader.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = addheader.run()
    >>> pd.read_csv(res.outputs.out_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
          a     b     c     d     e
    0   0.0   1.0   2.0   3.0   4.0
    1   5.0   6.0   7.0   8.0   9.0
    2  10.0  11.0  12.0  13.0  14.0
    3  15.0  16.0  17.0  18.0  19.0
    4  20.0  21.0  22.0  23.0  24.0
    5  25.0  26.0  27.0  28.0  29.0

    .. testcleanup::

    >>> tmpdir.cleanup()

    """
    input_spec = AddTSVHeaderInputSpec
    output_spec = AddTSVHeaderOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(self.inputs.in_file, suffix='_motion.tsv', newpath=runtime.cwd,
                                   use_ext=False)
        data = np.loadtxt(self.inputs.in_file)
        np.savetxt(out_file, data, delimiter='\t', header='\t'.join(self.inputs.columns),
                   comments='')

        self._results['out_file'] = out_file
        return runtime


def _tpm2roi(t1_tpm, t1_mask, bold_mask, mask_erosion_mm=None, erosion_mm=None,
             mask_erosion_prop=None, erosion_prop=None, pthres=0.95):
    """
    Generate a mask from a tissue probability map
    """
    tpm_img = nb.load(t1_tpm)
    bold_mask_img = nb.load(bold_mask)

    probability_map_data = (tpm_img.get_data() >= pthres).astype(np.uint8)

    eroded_mask_file = bold_mask
    erode_in = (mask_erosion_mm is not None and mask_erosion_mm > 0 or
                mask_erosion_prop is not None and mask_erosion_prop < 1)
    if erode_in:
        eroded_mask_file = os.path.abspath("eroded_mask.nii.gz")

        mask_img = nb.load(t1_mask)
        if mask_erosion_mm:
            iter_n = max(int(mask_erosion_mm / max(mask_img.header.get_zooms())), 1)
            mask_data = nd.binary_erosion(mask_img.get_data().astype(np.uint8), iterations=iter_n)
        else:
            mask_data = mask_img.get_data().astype(np.uint8)
            orig_vol = np.sum(mask_data > 0)
            while np.sum(mask_data > 0) / orig_vol > mask_erosion_prop:
                mask_data = nd.binary_erosion(mask_data, iterations=1)

        # Resample to BOLD and save
        eroded_t1 = nb.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
        eroded_t1.set_data_dtype(np.uint8)
        eroded_bold = _resample_and_mask(eroded_t1, bold_mask_img, interpolation='nearest')
        eroded_bold.to_filename(eroded_mask_file)

        # Mask TPM data (no effect if not eroded)
        probability_map_data[~mask_data] = 0

    # shrinking
    erode_out = (erosion_mm is not None and erosion_mm > 0 or
                 erosion_prop is not None and erosion_prop < 1)
    if erode_out:
        if erosion_mm:
            iter_n = max(int(erosion_mm / max(tpm_img.header.get_zooms())), 1)
            iter_n = int(erosion_mm / max(tpm_img.header.get_zooms()))
            probability_map_data = nd.binary_erosion(probability_map_data, iterations=iter_n)
        else:
            orig_vol = np.sum(probability_map_data > 0)
            while np.sum(probability_map_data > 0) / orig_vol > erosion_prop:
                probability_map_data = nd.binary_erosion(probability_map_data, iterations=1)

    # Create image to resample
    img_t1 = nb.Nifti1Image(probability_map_data, tpm_img.affine, tpm_img.header)
    img_t1.set_data_dtype(np.uint8)
    roi_img = _resample_and_mask(img_t1, bold_mask_img, interpolation='nearest')

    roi_img.to_filename("roi.nii.gz")
    return os.path.abspath("roi.nii.gz"), eroded_mask_file


def _resample_and_mask(source_img, mask_img, interpolation='continuous'):
    resampled_img = resample_to_img(source_img, mask_img, interpolation=interpolation)
    masked_data = resampled_img.get_data() * mask_img.get_data().astype(bool)

    out_img = mask_img.__class__(masked_data, mask_img.affine, mask_img.header)
    out_img.set_data_dtype(source_img.get_data_dtype())
    return out_img


def _combine_rois(in_files, ref_header):
    if len(in_files) < 2:
        raise RuntimeError('Combining ROIs requires at least two inputs')

    ref = nb.load(ref_header)

    nii = nb.concat_images([nb.load(f) for f in in_files])
    combined = nii.get_data().any(3).astype(np.uint8)

    # we have to do this explicitly because of potential differences in
    # qform_code between the two files that prevent aCompCor to work
    combined_nii = nb.Nifti1Image(combined, ref.affine, ref.header)
    combined_nii.set_data_dtype(np.uint8)

    combined_nii.to_filename("logical_or.nii.gz")
    return os.path.abspath("logical_or.nii.gz")


def _concat_rois(in_file, in_mask, ref_header):
    nii = nb.load(in_file)
    mask_nii = nb.load(in_mask)
    ref = nb.load(ref_header)

    # we have to do this explicitly because of potential differences in
    # qform_code between the two files that prevent SignalExtraction to do
    # the concatenation
    concat_nii = nb.concat_images([resample_to_img(nii, mask_nii, interpolation='nearest'),
                                   mask_nii])
    concat_nii = nb.Nifti1Image(concat_nii.get_data().astype(np.uint8), ref.affine, ref.header)
    concat_nii.set_data_dtype(np.uint8)

    concat_nii.to_filename("concat.nii.gz")
    return os.path.abspath("concat.nii.gz")
