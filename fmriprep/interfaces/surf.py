#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling surfaces
-----------------

"""
import os
import re

import numpy as np
import nibabel as nb

from niworkflows.nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, File

from niworkflows.interfaces.base import SimpleInterface


class NormalizeSurfInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc='input file, part of a BIDS tree')


class NormalizeSurfOutputSpec(TraitedSpec):
    out_file = File(desc='output file with re-centered GIFTI coordinates')


class NormalizeSurf(SimpleInterface):
    input_spec = NormalizeSurfInputSpec
    output_spec = NormalizeSurfOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = normalize_surfs(self.inputs.in_file)
        return runtime


class GiftiNameSourceInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc='input file, part of a BIDS tree')


class GiftiNameSourceOutputSpec(TraitedSpec):
    out_file = File(desc='output file with re-centered GIFTI coordinates')


class GiftiNameSource(SimpleInterface):
    input_spec = GiftiNameSourceInputSpec
    output_spec = GiftiNameSourceOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = get_gifti_name(self.inputs.in_file)
        return runtime


def normalize_surfs(in_file):
    """ Re-center GIFTI coordinates to fit align to native T1 space

    For midthickness surfaces, add MidThickness metadata

    Coordinate update based on:
    https://github.com/Washington-University/workbench/blob/1b79e56/src/Algorithms/AlgorithmSurfaceApplyAffine.cxx#L73-L91
    and
    https://github.com/Washington-University/Pipelines/blob/ae69b9a/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh#L147
    """

    img = nb.load(in_file)
    pointset = img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]
    coords = pointset.data
    c_ras_keys = ('VolGeomC_R', 'VolGeomC_A', 'VolGeomC_S')
    ras = np.array([float(pointset.metadata[key])
                    for key in c_ras_keys])
    # Apply C_RAS translation to coordinates
    pointset.data = (coords + ras).astype(coords.dtype)

    secondary = nb.gifti.GiftiNVPairs('AnatomicalStructureSecondary',
                                      'MidThickness')
    geom_type = nb.gifti.GiftiNVPairs('GeometricType', 'Anatomical')
    has_ass = has_geo = False
    for nvpair in pointset.meta.data:
        # Remove C_RAS translation from metadata to avoid double-dipping in FreeSurfer
        if nvpair.name in c_ras_keys:
            nvpair.value = '0.000000'
        # Check for missing metadata
        elif nvpair.name == secondary.name:
            has_ass = True
        elif nvpair.name == geom_type.name:
            has_geo = True
    fname = os.path.basename(in_file)
    # Update metadata for MidThickness/graymid surfaces
    if 'midthickness' in fname.lower() or 'graymid' in fname.lower():
        if not has_ass:
            pointset.meta.data.insert(1, secondary)
        if not has_geo:
            pointset.meta.data.insert(2, geom_type)
    img.to_filename(fname)
    return os.path.abspath(fname)


def get_gifti_name(in_file):
    in_format = re.compile(r'(?P<LR>[lr])h.(?P<surf>.+)_converted.gii')
    name = os.path.basename(in_file)
    info = in_format.match(name).groupdict()
    info['LR'] = info['LR'].upper()
    return '{surf}.{LR}.surf'.format(**info)
