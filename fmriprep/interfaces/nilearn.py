#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import os.path as op
import numpy as np
import nibabel as nb
from nilearn.masking import compute_epi_mask

from nipype import logging
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    File, InputMultiPath, OutputMultiPath, traits
)
from fmriprep.utils.misc import genfname
LOGGER = logging.getLogger('interface')


class MaskEPIInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input EPI or list of files')
    lower_cutoff = traits.Float(0.2, usedefault=True)
    upper_cutoff = traits.Float(0.85, usedefault=True)
    connected = traits.Bool(True, usedefault=True)
    opening = traits.Int(2, usedefault=True)
    exclude_zeros = traits.Bool(False, usedefault=True)
    ensure_finite = traits.Bool(True, usedefault=True)
    target_affine = traits.File()
    target_shape = traits.File()


class MaskEPIOutputSpec(TraitedSpec):
    out_mask = File(exists=True, desc='output mask')

class MaskEPI(BaseInterface):
    input_spec = MaskEPIInputSpec
    output_spec = MaskEPIOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(MaskEPI, self).__init__(**inputs)

    def _run_interface(self, runtime):
        target_affine = None
        target_shape = None

        if isdefined(self.inputs.target_affine):
            target_affine = self.inputs.target_affine
        if isdefined(self.inputs.target_shape):
            target_shape = self.inputs.target_shape

        masknii = compute_epi_mask(
            self.inputs.in_files,
            lower_cutoff=self.inputs.lower_cutoff,
            upper_cutoff=self.inputs.upper_cutoff,
            connected=self.inputs.connected,
            opening=self.inputs.opening,
            exclude_zeros=self.inputs.exclude_zeros,
            ensure_finite=self.inputs.ensure_finite,
            target_affine=target_affine,
            target_shape=target_shape
        )

        self._results['out_mask'] = genfname(
            self.inputs.in_files[0], suffix='mask')
        masknii.to_filename(self._results['out_mask'])
        return runtime

    def _list_outputs(self):
        return self._results
