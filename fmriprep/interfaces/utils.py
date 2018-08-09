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

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, isdefined, File, InputMultiPath,
    TraitedSpec, DynamicTraitedSpec, BaseInterfaceInputSpec, SimpleInterface
)
from nipype.interfaces.io import add_traits

IFLOGGER = logging.getLogger('nipype.interfaces')


class TPM2ROIInputSpec(BaseInterfaceInputSpec):
    in_tpm = File(exists=True, mandatory=True, desc='Tissue probability map file in T1 space')
    in_mask = File(exists=True, mandatory=True, desc='Binary mask of skull-stripped T1w image')
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

    #. Erode ``in_mask`` by ``mask_erode_mm`` and apply to ``in_tpm``
    #. Threshold masked TPM at ``prob_thresh``
    #. Erode resulting mask by ``erode_mm``

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
            self.inputs.in_tpm,
            self.inputs.in_mask,
            mask_erode_mm,
            erode_mm,
            mask_erode_prop,
            erode_prop,
            self.inputs.prob_thresh,
            newpath=runtime.cwd,
        )
        self._results['roi_file'] = roi_file
        self._results['eroded_mask'] = eroded_mask
        return runtime


class AddTPMsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='input list of ROIs')
    indices = traits.List(traits.Int, desc='select specific maps')


class AddTPMsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='union of binarized input files')


class AddTPMs(SimpleInterface):
    """Calculate the union of several :abbr:`TPMs (tissue-probability map)`"""
    input_spec = AddTPMsInputSpec
    output_spec = AddTPMsOutputSpec

    def _run_interface(self, runtime):
        in_files = self.inputs.in_files

        indices = list(range(len(in_files)))
        if isdefined(self.inputs.indices):
            indices = self.inputs.indices

        if len(self.inputs.in_files) < 2:
            self._results['out_file'] = in_files[0]
            return runtime

        first_fname = in_files[indices[0]]
        if len(indices) == 1:
            self._results['out_file'] = first_fname
            return runtime

        im = nb.concat_images([in_files[i] for i in indices])
        data = im.get_data().astype(float).sum(axis=3)
        data = np.clip(data, a_min=0.0, a_max=1.0)

        out_file = fname_presuffix(first_fname, suffix='_tpmsum',
                                   newpath=runtime.cwd)
        newnii = im.__class__(data, im.affine, im.header)
        newnii.set_data_dtype(np.float32)

        # Set visualization thresholds
        newnii.header['cal_max'] = 1.0
        newnii.header['cal_min'] = 0.0
        newnii.to_filename(out_file)
        self._results['out_file'] = out_file

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


class JoinTSVColumnsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    join_file = File(exists=True, mandatory=True, desc='file to be adjoined')
    side = traits.Enum('right', 'left', usedefault=True, desc='where to join')
    columns = traits.List(traits.Str, desc='header for columns')


class JoinTSVColumnsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output TSV file')


class JoinTSVColumns(SimpleInterface):
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

    >>> data = np.arange(30).reshape((6, 5))
    >>> np.savetxt('data.tsv', data[:, :3], delimiter='\t', fmt='%.1f')
    >>> np.savetxt('add.tsv', data[:, 3:], delimiter='\t', fmt='%.1f')

    Add headers:

    >>> from fmriprep.interfaces import JoinTSVColumns
    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> join.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = join.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '...data_joined.tsv'
    >>> pd.read_csv(res.outputs.out_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
          a     b     c     d     e
    0   0.0   1.0   2.0   3.0   4.0
    1   5.0   6.0   7.0   8.0   9.0
    2  10.0  11.0  12.0  13.0  14.0
    3  15.0  16.0  17.0  18.0  19.0
    4  20.0  21.0  22.0  23.0  24.0
    5  25.0  26.0  27.0  28.0  29.0

    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> res = join.run()
    >>> pd.read_csv(res.outputs.out_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
        0.0   1.0   2.0   3.0   4.0
    0   5.0   6.0   7.0   8.0   9.0
    1  10.0  11.0  12.0  13.0  14.0
    2  15.0  16.0  17.0  18.0  19.0
    3  20.0  21.0  22.0  23.0  24.0
    4  25.0  26.0  27.0  28.0  29.0

    >>> join = JoinTSVColumns()
    >>> join.inputs.in_file = 'data.tsv'
    >>> join.inputs.join_file = 'add.tsv'
    >>> join.inputs.side = 'left'
    >>> join.inputs.columns = ['a', 'b', 'c', 'd', 'e']
    >>> res = join.run()
    >>> pd.read_csv(res.outputs.out_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
          a     b     c    d     e
    0   3.0   4.0  0.0   1.0   2.0
    1   8.0   9.0  5.0   6.0   7.0
    2  13.0  14.0 10.0  11.0  12.0
    3  18.0  19.0 15.0  16.0  17.0
    4  23.0  24.0 20.0  21.0  22.0
    5  28.0  29.0 25.0  26.0  27.0

    .. testcleanup::

    >>> tmpdir.cleanup()

    """
    input_spec = JoinTSVColumnsInputSpec
    output_spec = JoinTSVColumnsOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(
            self.inputs.in_file, suffix='_joined.tsv', newpath=runtime.cwd,
            use_ext=False)

        header = ''
        if isdefined(self.inputs.columns) and self.inputs.columns:
            header = '\t'.join(self.inputs.columns)

        with open(self.inputs.in_file) as ifh:
            data = ifh.read().splitlines(keepends=False)

        with open(self.inputs.join_file) as ifh:
            join = ifh.read().splitlines(keepends=False)

        assert len(data) == len(join)

        merged = []
        for d, j in zip(data, join):
            line = '%s\t%s' % ((j, d) if self.inputs.side == 'left' else (d, j))
            merged.append(line)

        if header:
            merged.insert(0, header)

        with open(out_file, 'w') as ofh:
            ofh.write('\n'.join(merged))

        self._results['out_file'] = out_file
        return runtime


class ConcatAffinesInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    invert = traits.Bool(False, usedefault=True, desc='Invert output transform')


class ConcatAffinesOutputSpec(TraitedSpec):
    out_mat = File(exists=True, desc='Output transform')


class ConcatAffines(SimpleInterface):
    input_spec = ConcatAffinesInputSpec
    output_spec = ConcatAffinesOutputSpec

    def __init__(self, num_affines=0, *args, **kwargs):
        super(ConcatAffines, self).__init__(*args, **kwargs)
        self._num_affines = num_affines
        trait_type = File(exists=True)
        if num_affines == 0:
            add_traits(self.inputs, ['mat_list'], trait_type)
        elif num_affines < 26:
            add_traits(self.inputs, self._get_names(num_affines), trait_type)

    @staticmethod
    def _get_names(num_affines):
        A = ord('A') - 1
        return ['mat_{}to{}'.format(chr(X), chr(X + 1))
                for X in range(A + num_affines, A, -1)]

    def _run_interface(self, runtime):
        out_mat = os.path.join(runtime.cwd, 'concat.mat')
        in_list = [self.inputs.get()[name] for name in self._get_names(self._num_affines)]

        out_xfm = _concat_xfms(in_list, invert=self.inputs.invert)
        np.savetxt(out_mat, out_xfm, fmt=str('%.12g'))

        self._results['out_mat'] = out_mat
        return runtime


def _tpm2roi(in_tpm, in_mask, mask_erosion_mm=None, erosion_mm=None,
             mask_erosion_prop=None, erosion_prop=None, pthres=0.95,
             newpath=None):
    """
    Generate a mask from a tissue probability map
    """
    tpm_img = nb.load(in_tpm)
    roi_mask = (tpm_img.get_data() >= pthres).astype(np.uint8)

    eroded_mask_file = None
    erode_in = (mask_erosion_mm is not None and mask_erosion_mm > 0 or
                mask_erosion_prop is not None and mask_erosion_prop < 1)
    if erode_in:
        eroded_mask_file = fname_presuffix(in_mask, suffix='_eroded',
                                           newpath=newpath)
        mask_img = nb.load(in_mask)
        mask_data = mask_img.get_data().astype(np.uint8)
        if mask_erosion_mm:
            iter_n = max(int(mask_erosion_mm / max(mask_img.header.get_zooms())), 1)
            mask_data = nd.binary_erosion(mask_data, iterations=iter_n)
        else:
            orig_vol = np.sum(mask_data > 0)
            while np.sum(mask_data > 0) / orig_vol > mask_erosion_prop:
                mask_data = nd.binary_erosion(mask_data, iterations=1)

        # Store mask
        eroded = nb.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
        eroded.set_data_dtype(np.uint8)
        eroded.to_filename(eroded_mask_file)

        # Mask TPM data (no effect if not eroded)
        roi_mask[~mask_data] = 0

    # shrinking
    erode_out = (erosion_mm is not None and erosion_mm > 0 or
                 erosion_prop is not None and erosion_prop < 1)
    if erode_out:
        if erosion_mm:
            iter_n = max(int(erosion_mm / max(tpm_img.header.get_zooms())), 1)
            iter_n = int(erosion_mm / max(tpm_img.header.get_zooms()))
            roi_mask = nd.binary_erosion(roi_mask, iterations=iter_n)
        else:
            orig_vol = np.sum(roi_mask > 0)
            while np.sum(roi_mask > 0) / orig_vol > erosion_prop:
                roi_mask = nd.binary_erosion(roi_mask, iterations=1)

    # Create image to resample
    roi_fname = fname_presuffix(in_tpm, suffix='_roi', newpath=newpath)
    roi_img = nb.Nifti1Image(roi_mask, tpm_img.affine, tpm_img.header)
    roi_img.set_data_dtype(np.uint8)
    roi_img.to_filename(roi_fname)
    return roi_fname, eroded_mask_file or in_mask


def _concat_xfms(in_list, invert):
    transforms = [np.loadtxt(in_mat) for in_mat in in_list]
    out_xfm = transforms.pop(0)
    for xfm in transforms:
        out_xfm = out_xfm.dot(xfm)

    if invert:
        out_xfm = np.linalg.inv(out_xfm)

    return out_xfm
