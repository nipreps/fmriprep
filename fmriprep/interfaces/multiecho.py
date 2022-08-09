# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Multi-echo EPI
~~~~~~~~~~~~~~

For using multi-echo EPI data.

Change directory to provide relative paths for doctests
>>> import os
>>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
>>> datadir = os.path.realpath(os.path.join(filepath, '../data/'))
>>> os.chdir(datadir)

"""
import os

from nipype import logging
from nipype.interfaces.base import (
    traits, TraitedSpec, File,
    CommandLine, CommandLineInputSpec)

LOGGER = logging.getLogger('nipype.interface')


class T2SMapInputSpec(CommandLineInputSpec):
    in_files = traits.List(File(exists=True),
                           argstr='-d %s',
                           position=1,
                           mandatory=True,
                           minlen=3,
                           desc='multi-echo BOLD EPIs')
    echo_times = traits.List(traits.Float,
                             argstr='-e %s',
                             position=2,
                             mandatory=True,
                             minlen=3,
                             desc='echo times')
    mask_file = File(argstr='--mask %s',
                     position=3,
                     desc='mask file',
                     exists=True)
    fittype = traits.Enum('curvefit', 'loglin',
                          argstr='--fittype %s',
                          position=4,
                          usedefault=True,
                          desc=('Desired fitting method: '
                                '"loglin" means that a linear model is fit '
                                'to the log of the data. '
                                '"curvefit" means that a more computationally '
                                'demanding monoexponential model is fit '
                                'to the raw data.'))


class T2SMapOutputSpec(TraitedSpec):
    t2star_map = File(exists=True, desc='limited T2* map')
    s0_map = File(exists=True, desc='limited S0 map')
    optimal_comb = File(exists=True, desc='optimally combined ME-EPI time series')


class T2SMap(CommandLine):
    """
    Runs the tedana T2* workflow to generate an adaptive T2* map and create
    an optimally combined ME-EPI time series.

    Example
    =======
    >>> from fmriprep.interfaces import multiecho
    >>> t2smap = multiecho.T2SMap()
    >>> t2smap.inputs.in_files = ['sub-01_run-01_echo-1_bold.nii.gz', \
                                  'sub-01_run-01_echo-2_bold.nii.gz', \
                                  'sub-01_run-01_echo-3_bold.nii.gz']
    >>> t2smap.inputs.echo_times = [0.013, 0.027, 0.043]
    >>> t2smap.cmdline  # doctest: +ELLIPSIS
    't2smap -d sub-01_run-01_echo-1_bold.nii.gz sub-01_run-01_echo-2_bold.nii.gz \
sub-01_run-01_echo-3_bold.nii.gz -e 13.0 27.0 43.0 --fittype curvefit'

    """
    _cmd = 't2smap'
    input_spec = T2SMapInputSpec
    output_spec = T2SMapOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'echo_times':
            value = [te * 1000 for te in value]
        return super(T2SMap, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = os.getcwd()
        outputs['t2star_map'] = os.path.join(out_dir, 'T2starmap.nii.gz')
        outputs['s0_map'] = os.path.join(out_dir, 'S0map.nii.gz')
        outputs['optimal_comb'] = os.path.join(out_dir, 'desc-optcom_bold.nii.gz')
        return outputs
