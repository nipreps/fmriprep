#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import numpy as np
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File,
    InputMultiPath, OutputMultiPath, isdefined
)

from io import open
from fmriprep.utils.misc import genfname

ITK_TFM_HEADER = "#Insight Transform File V1.0"
ITK_TFM_TPL = """\
#Transform {tf_id}
Transform: {tf_type}
Parameters: {tf_params}
FixedParameters: {fixed_params}""".format


class SplitITKTransformInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')

class SplitITKTransformOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(
        File(exists=True), desc='list of output files')

class SplitITKTransform(BaseInterface):

    """
    This interface splits an ITK Transform file, generating open
    individual text file per transform in it.

    """
    input_spec = SplitITKTransformInputSpec
    output_spec = SplitITKTransformOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(SplitITKTransform, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        with open(self.inputs.in_file) as infh:
            lines = infh.readlines()

        lines.append('#Transform') # forces flushing last transform
        tfm_list = []
        tfm_prefix = [ITK_TFM_HEADER, '#Transform 0']
        tfm = []
        for line in lines[1:]:
            if line.startswith('#Transform'):
                if tfm:
                    tfm_list.append('\n'.join(
                        tfm_prefix + tfm + ['']))
                    tfm = []
            else:
                tfm.append(line.replace('\n', ''))

        out_files = []
        for i, tfm in enumerate(tfm_list):
            out_files.append(genfname(
                self.inputs.in_file, suffix=('%04d' % i), ext='tfm'))
            print(out_files[-1])
            with open(out_files[-1], 'w') as ofh:
                ofh.write(tfm)

        self._results['out_files'] = out_files

        return runtime

class IdentityITKTransformInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='input file')

class IdentityITKTransformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='list of output files')

class IdentityITKTransform(BaseInterface):

    """
    This interface generates an identity transform if the input
    is not set.

    """
    input_spec = IdentityITKTransformInputSpec
    output_spec = IdentityITKTransformOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(IdentityITKTransform, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        if isdefined(self.inputs.in_file):
            self._results['out_file'] = self.inputs.in_file
        else:
            out_file = op.abspath('identity.tfm')

            ident = np.eye(3).reshape(-1).tolist() + [0.0] * 3

            tfmstr = '%s\n' % ITK_TFM_HEADER
            tfmstr += ITK_TFM_TPL(
                tf_id=0, tf_type='AffineTransform_double_3_3',
                tf_params=' '.join(['%.1f' % i for i in ident]),
                fixed_params=' '.join(['0.0'] * 3))
            tfmstr += '\n'

            with open(out_file, 'w') as ofh:
                ofh.write(tfmstr)

            self._results['out_file'] = out_file

        return runtime

class ConcatANTsTransformInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='input file')
    in_file_invert = traits.Bool(False, usedefault=True)
    position = traits.Int(-1, usedefault=True)
    transforms = InputMultiPath(File(exists=True),
                                mandatory=True, desc='input file')
    invert_transform_flags = traits.List(
        traits.Bool(), mandatory=True, desc='invert transforms')

class ConcatANTsTransformOutputSpec(TraitedSpec):
    transforms = OutputMultiPath(File(exists=True),
                                 desc='list of output files')
    invert_transform_flags = traits.List(
        traits.Bool(), desc='invert transforms')

class ConcatANTsTransform(BaseInterface):

    """
    This interface generates an identity transform if the input
    is not set.

    """
    input_spec = ConcatANTsTransformInputSpec
    output_spec = ConcatANTsTransformOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ConcatANTsTransform, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        self._results['transforms'] = self.inputs.transforms
        self._results['invert_transform_flags'] = self.inputs.invert_transform_flags

        if isdefined(self.inputs.in_file) and self.inputs.in_file is not None:
            flag = self.inputs.in_file_invert
            in_file = self.inputs.in_file
            pos = self.inputs.position
            if pos == -1:
                self._results['transforms'] += [in_file]
                self._results['invert_transform_flags'] += [flag]
            else:
                self._results['transforms'].insert(pos, in_file)
                self._results['invert_transform_flags'].insert(pos, flag)

        return runtime
