#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals

import json
import re
import os
import os.path as op
from shutil import copy
import numpy as np
import nibabel as nb

from io import open

from nipype import logging
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    File, InputMultiPath, OutputMultiPath, traits
)

from fmriprep.interfaces.bids import _splitext
from fmriprep.utils.misc import make_folder, genfname

LOGGER = logging.getLogger('interface')


class GenerateSamplingReferenceInputSpec(BaseInterfaceInputSpec):
    fixed_image = File(exists=True, mandatory=True, desc='the reference file')
    moving_image = File(exists=True, mandatory=True, desc='the pixel size reference')

class GenerateSamplingReferenceOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='one file with all inputs flattened')

class GenerateSamplingReference(BaseInterface):
    """
    Generates a reference grid for resampling one image keeping original resolution,
    but moving data to a different space (e.g. MNI)
    """

    input_spec = GenerateSamplingReferenceInputSpec
    output_spec = GenerateSamplingReferenceOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(GenerateSamplingReference, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        self._results['out_file'] = _gen_reference(self.inputs.fixed_image,
                                                   self.inputs.moving_image)
        return runtime


class IntraModalMergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input files')
    hmc = traits.Bool(True, usedefault=True)
    zero_based_avg = traits.Bool(True, usedefault=True)
    to_ras = traits.Bool(True, usedefault=True)


class IntraModalMergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='merged image')
    out_avg = File(exists=True, desc='average image')
    out_mats = OutputMultiPath(exists=True, desc='output matrices')
    out_movpar = OutputMultiPath(exists=True, desc='output movement parameters')

class IntraModalMerge(BaseInterface):
    input_spec = IntraModalMergeInputSpec
    output_spec = IntraModalMergeOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(IntraModalMerge, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        from nipype.interfaces import fsl
        in_files = self.inputs.in_files
        if not isinstance(in_files, list):
            in_files = [self.inputs.in_files]

        # Generate output average name early
        self._results['out_avg'] = genfname(self.inputs.in_files[0],
                                            suffix='avg')

        if self.inputs.to_ras:
            in_files = [reorient(inf) for inf in in_files]

        if len(in_files) == 1:
            filenii = nb.load(in_files[0])
            filedata = filenii.get_data()

            # magnitude files can have an extra dimension empty
            if filedata.ndim == 5:
                sqdata = np.squeeze(filedata)
                if sqdata.ndim == 5:
                    raise RuntimeError('Input image (%s) is 5D' % in_files[0])
                else:
                    in_files = [genfname(in_files[0], suffix='squeezed')]
                    nb.Nifti1Image(sqdata, filenii.get_affine(),
                                   filenii.get_header()).to_filename(in_files[0])

            if np.squeeze(nb.load(in_files[0]).get_data()).ndim < 4:
                self._results['out_file'] = in_files[0]
                self._results['out_avg'] = in_files[0]
                # TODO: generate identity out_mats and zero-filled out_movpar
                return runtime
            in_files = in_files[0]
        else:
            magmrg = fsl.Merge(dimension='t', in_files=self.inputs.in_files)
            in_files = magmrg.run().outputs.merged_file
        mcflirt = fsl.MCFLIRT(cost='normcorr', save_mats=True, save_plots=True,
                              ref_vol=0, in_file=in_files)
        mcres = mcflirt.run()
        self._results['out_mats'] = mcres.outputs.mat_file
        self._results['out_movpar'] = mcres.outputs.par_file
        self._results['out_file'] = mcres.outputs.out_file

        hmcnii = nb.load(mcres.outputs.out_file)
        hmcdat = hmcnii.get_data().mean(axis=3)
        if self.inputs.zero_based_avg:
            hmcdat -= hmcdat.min()

        nb.Nifti1Image(
            hmcdat, hmcnii.get_affine(), hmcnii.get_header()).to_filename(
            self._results['out_avg'])

        return runtime


class CopyHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the file we get the data from')
    hdr_file = File(exists=True, mandatory=True, desc='the file we get the header from')

class CopyHeaderOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True, desc='written file path'))

class CopyHeader(BaseInterface):
    input_spec = CopyHeaderInputSpec
    output_spec = CopyHeaderOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(CopyHeader, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):

        hdr = nb.load(self.inputs.hdr_file).get_header().copy()
        aff = nb.load(self.inputs.hdr_file).get_affine()
        data = nb.load(self.inputs.in_file).get_data()

        fname, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext

        out_name = op.abspath('{}_fixhdr{}'.format(fname, ext))
        nb.Nifti1Image(data, aff, hdr).to_filename(out_name)
        self._results['out_file'] = out_name
        return runtime


def reorient(in_file, out_file=None):
    import nibabel as nb
    from fmriprep.utils.misc import genfname
    from builtins import (str, bytes)

    if out_file is None:
        out_file = genfname(in_file, suffix='ras')

    if isinstance(in_file, (str, bytes)):
        nii = nb.load(in_file)
    nii = nb.as_closest_canonical(nii)
    nii.to_filename(out_file)
    return out_file


def _flatten_split_merge(in_files):
    from builtins import bytes, str

    if isinstance(in_files, (bytes, str)):
        in_files = [in_files]

    nfiles = len(in_files)

    all_nii = []
    for fname in in_files:
        nii = nb.squeeze_image(nb.load(fname))

        if nii.get_data().ndim > 3:
            all_nii += nb.four_to_three(nii)
        else:
            all_nii.append(nii)


    if len(all_nii) == 1:
        LOGGER.warn('File %s cannot be split', all_nii[0])
        return in_files[0], in_files

    if len(all_nii) == nfiles:
        flat_split = in_files
    else:
        splitname = genfname(in_files[0], suffix='split%04d')
        flat_split = []
        for i, nii in enumerate(all_nii):
            flat_split.append(splitname % i)
            nii.to_filename(flat_split[-1])

    # Only one 4D file was supplied
    if nfiles == 1:
        merged = in_files[0]
    else:
        # More that one in_files - need merge
        merged = genfname(in_files[0], suffix='merged')
        nb.concat_images(all_nii).to_filename(merged)

    return merged, flat_split


def _gen_reference(fixed_image, moving_image, out_file=None):
    import os.path as op
    import numpy
    from nilearn.image import resample_img, load_img

    if out_file is None:
        out_file = genfname(fixed_image, suffix='reference')
    new_zooms = load_img(moving_image).header.get_zooms()
    resample_img(fixed_image, target_affine=numpy.diag(new_zooms),
                 interpolation='nearest').to_filename(out_file)
    return out_file
