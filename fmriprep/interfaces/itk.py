#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals

from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File,
    OutputMultiPath
)

from io import open
from fmriprep.utils.misc import genfname

ITK_TFM_HEADER = "#Insight Transform File V1.0"
ITK_TFM_TPL = """\
#Transform {tf_id}
Transform: {tf_type}
Parameters: {tf_params}
FixedParameters: {fixed_params}""".format


class SplitITKTranformInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')

class SplitITKTranformOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(
        File(exists=True), desc='list of output files')

class SplitITKTranform(BaseInterface):

    """
    This interface splits an ITK Transform file, generating open
    individual text file per transform in it.

    """
    input_spec = SplitITKTranformInputSpec
    output_spec = SplitITKTranformOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(SplitITKTranform, self).__init__(**inputs)

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
