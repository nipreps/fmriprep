#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
from nibabel.filename_parser import splitext_addext

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


class T2SMapOutputSpec(TraitedSpec):
    t2star_map = File(exists=True, desc='limited T2* map')
    s0_map = File(exists=True, desc='limited s0 map')
    t2star_adaptive_map = File(exists=True, desc='adaptive T2* map')
    s0_adaptive_map = File(exists=True, desc='adaptive s0 map')
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
sub-01_run-01_echo-3_bold.nii.gz -e 13.0 27.0 43.0'
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
        filename = splitext_addext(os.path.basename(self.inputs.in_files[0]))[0]
        out_dir = os.path.abspath('TED.{}'.format(filename))

        outputs['t2star_map'] = os.path.join(out_dir, 't2sv.nii')
        outputs['s0_map'] = os.path.join(out_dir, 's0v.nii')
        outputs['t2star_adaptive_map'] = os.path.join(out_dir, 't2svG.nii')
        outputs['s0_adaptive_map'] = os.path.join(out_dir, 's0vG.nii')
        outputs['optimal_comb'] = os.path.join(out_dir, 'ts_OC.nii')

        return outputs
