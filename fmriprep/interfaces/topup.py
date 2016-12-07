#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
TopUp helpers
~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import nibabel as nb
from nilearn.image import mean_img
from nipype import logging
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces import fsl
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    File, InputMultiPath, traits
)
from fmriprep.utils.misc import genfname
from .images import reorient

LOGGER = logging.getLogger('interface')

class ConformTopupInputsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input files')
    in_mats = InputMultiPath(File(exists=True), desc='input files')
    in_ref = traits.Int(-1, usedefault=True, desc='reference volume')


class ConformTopupInputsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='merged image')
    out_reference = File(exists=True, desc='reference image')
    out_mask = File(exists=True, desc='out mask')
    out_brain = File(exists=True, desc='reference image, masked')
    out_movpar = File(exists=True, desc='output movement parameters')

class ConformTopupInputs(BaseInterface):

    """
    This function interprets that we are dealing with a
    multiple PE (phase encoding) input if it finds several
    files in in_files.

    If we have several images with various PE directions,
    it will compute the HMC parameters between them using
    an embedded workflow.

    It just forwards the two inputs otherwise.
    """
    input_spec = ConformTopupInputsInputSpec
    output_spec = ConformTopupInputsOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ConformTopupInputs, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _one_flirt_file(self, in_files):
        """
        Input is one FLIRT'ed file (typically: _bold),
        must come along with the corresponding HMC matrices.

        """
        nmats = len(self.inputs.in_mats)
        if len(in_files) > 1:
            raise RuntimeError('Only one file, aligned for head motion '
                               'expected. Got %s' % in_files)
        ntrs = nb.load(in_files[0]).get_data().shape[-1]

        if nmats != ntrs:
            raise RuntimeError('Number of TRs (%s) and input HMC matrices (%s) do not '
                               ' match' % (ntrs, nmats))

        self._results['out_file'] = in_files[0]
        self._results['out_movpar'] = genfname(in_files[0], suffix='movpar', ext='txt')
        movpar = _generate_topup_movpar(self.inputs.in_mats)
        np.savetxt(self._results['out_movpar'], movpar)

    def _run_flirt(self, in_files):
        # Head motion correction
        fslmerge = fsl.Merge(dimension='t', in_files=in_files)
        hmc = fsl.MCFLIRT(cost='normcorr', save_mats=True)
        if self.inputs.in_ref >= 0:
            hmc.inputs.ref_vol = self.inputs.in_ref
        hmc.inputs.in_file = fslmerge.run().outputs.merged_file
        hmc_res = hmc.run()
        self._results['out_file'] = hmc_res.outputs.out_file
        self._results['out_movpar'] = genfname(in_files[0], suffix='movpar', ext='txt')
        movpar = _generate_topup_movpar(hmc_res.outputs.par_file)
        np.savetxt(self._results['out_movpar'], movpar)


    def _run_interface(self, runtime):
        from builtins import (str, bytes)
        in_files = self.inputs.in_files

        ref_vol = self.inputs.in_ref
        if isinstance(in_files, (str, bytes)):
            in_files = [in_files]

        in_files = [reorient(f) for f in in_files]

        ntsteps = 0
        for fname in in_files:
            try:
                nii = nb.four_to_three(nb.load(fname))
                ntsteps += len(nii)
            except ValueError:
                ntsteps += 1

        # If HMC mats are present, we expect only
        # one file, aligned with flirt.
        if isdefined(self.inputs.in_mats):
            self._one_flirt_file(in_files)
        elif ntsteps > 1:
            self._run_flirt(in_files)
        else:
            self._results['out_file'] = in_files[0]
            self._results['out_reference'] = in_files[0]
            self._results['out_movpar'] = genfname(in_files[0], suffix='movpar', ext='txt')
            np.savetxt(self._results['out_movpar'], np.zeros((1, 6)))

        if ntsteps > 1:
            if ref_vol > -1:
                self._results['out_reference'] = genfname(
                    in_files[0], suffix='vol%02d' % ref_vol)
                nii_list = nb.four_to_three(nb.load(self._results['out_file']))
                nii_list[ref_vol].to_filename(self._results['out_reference'])
            else:
                self._results['out_reference'] = genfname(in_files[0], suffix='avg')
                nii = mean_img(nb.load(self._results['out_file']))
                nii.to_filename(self._results['out_reference'])

        inu = N4BiasFieldCorrection(
            dimension=3, input_image=self._results['out_reference']).run()
        bet = fsl.BET(in_file=inu.outputs.output_image,
                      frac=0.6, mask=True).run().outputs
        self._results['out_mask'] = bet.mask_file
        self._results['out_brain'] = bet.out_file

        # Generate Encoding file

        return runtime

def _generate_topup_movpar(in_mats):
    # TODO!
    return np.zeros((len(in_mats), 6))
