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
from nilearn.image import mean_img, concat_imgs
from nilearn.masking import (compute_epi_mask,
                             apply_mask)
from nipype import logging
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces import fsl
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    File, InputMultiPath, traits, OutputMultiPath
)
from fmriprep.utils.misc import genfname
from .images import reorient
from .bids import get_metadata_for_nifti

LOGGER = logging.getLogger('interface')

class TopupInputsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True),
                              desc='input file for topup field estimation')
    to_ras = traits.Bool(True, usedefault=True,
                         desc='reorient all input images to RAS')
    mask_inputs = traits.Bool(True, usedefault=True,
                              desc='do mask of inputs')

class TopupInputsOutputSpec(TraitedSpec):
    enc_file = File(exists=True,
                    desc='the topup encoding parameters file corresponding to'
                         '  the inputs')
    out_file = File(exists=True,
                    desc='all input files in a single 4D volume')
    out_files = OutputMultiPath(File(exists=True),
                                desc='input files as a list')


class TopupInputs(BaseInterface):

    """
    This interface generates the input files required by FSL topup:

      * A 4D file with all input images
      * The topup-encoding parameters file corresponding to the 4D file.


    """
    input_spec = TopupInputsInputSpec
    output_spec = TopupInputsOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(TopupInputs, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):

        in_files = [fname for fname in self.inputs.in_files
                    if 'epi' in fname or 'sbref' in fname]

        self._results['enc_file'] = genfname(in_files[0], suffix='encfile', ext='txt')
        self._results['out_file'] = genfname(in_files[0], suffix='datain')

        # Check input files.
        if not isinstance(in_files, (list, tuple)):
            raise RuntimeError('in_files should be a list of files')
        if len(in_files) < 2:
            raise RuntimeError('in_files should be a list of 2 or more input files')

        # Check metadata of inputs and retrieve the pe dir and ro time.
        pe_params = []
        for fname in in_files:
            line = [0.0] * 4
            pe_dir, line[3] = get_pe_params(fname)
            line[int(abs(pe_dir))] = 1.0 if pe_dir > 0.0 else -1.0

            # check number of images in dataset
            ntsteps = 1
            nii = nb.squeeze_image(nb.load(fname))
            if len(nii.shape) == 4:
                ntsteps = nii.shape[-1]

            pe_params += [line] * ntsteps

        np.savetxt(self._results['enc_file'], pe_params,
                   fmt=[b'%0.1f', b'%0.1f', b'%0.1f', b'%0.20f'])

        # to RAS
        if self.inputs.to_ras:
            in_files = [reorient(fname) for fname in in_files]

        # split 4D volumes
        self._results['out_files'] = []
        for i, fname in enumerate(in_files):
            nii = nb.load(fname)
            if len(nii.shape) == 3:
                self._results['out_files'].append(fname)
                continue

            nii = nb.squeeze_image(nii)
            if len(nii.shape) == 3:
                nii_list = [nii]
            else:
                nii_list = nb.four_to_three(nii)

            for j, frame in enumerate(nii_list):
                newfname = genfname(fname, suffix='sq%02d_%03d' % (i, j))
                self._results['out_files'].append(newfname)
                frame.to_filename(newfname)

        if self.inputs.mask_inputs:
            self._results['out_files'] = [apply_epi_mask(fname)
                                          for fname in self._results['out_files']]
        # merge
        nii = concat_imgs(self._results['out_files'], ensure_ndim=4)
        nii.to_filename(self._results['out_file'])

        return runtime

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

def get_pe_params(in_file):
    """
    Checks on the BIDS metadata associated with the file and
    extracts the two parameters of interest: PE direction and
    RO time.
    """
    meta = get_metadata_for_nifti(in_file)

    # Process PE direction
    pe_meta = meta.get('PhaseEncodingDirection')
    if pe_meta is None:
        raise RuntimeError('PhaseEncodingDirection metadata not found for '
                           ' %s' % in_file)

    if pe_meta[0] == 'i':
        pe_dir = 0
    elif pe_meta[0] == 'j':
        pe_dir = 1
    elif pe_meta[0] == 'k':
        LOGGER.warn('Detected phase-encoding direction perpendicular '
                    'to the axial plane of the brain.')
        pe_dir = 2

    if pe_meta.endswith('-'):
        pe_dir *= -1.0

    # Option 1: we find the RO time label
    ro_time = meta.get('TotalReadoutTime', None)

    # Option 2: we find the effective echo spacing label
    eff_ec = meta.get('EffectiveEchoSpacing', None)
    if ro_time is None and eff_ec is not None:
        pe_echoes = nb.load(in_file).shape[int(abs(pe_dir))] - 1
        ro_time = eff_ec * pe_echoes

    # Option 3: we find echo time label
    ect = meta.get('EchoTime', None)
    if ro_time is None and ect is not None:
        LOGGER.warn('Total readout time was estimated from the '
                    'EchoTime metadata, please be aware that acceleration '
                    'factors do modify the effective echo time that is '
                    'necessary for fieldmap correction.')
        pe_echoes = nb.load(in_file).shape[int(abs(pe_dir))] - 1
        ro_time = ect * pe_echoes

    # Option 4: using the bandwidth parameter
    pebw = meta.get('BandwidthPerPixelPhaseEncode', None)
    if ro_time is None and pebw is not None:
        LOGGER.warn('Total readout time was estimated from the '
                    'BandwidthPerPixelPhaseEncode metadata, please be aware '
                    'that acceleration factors do modify the effective echo '
                    'time that is necessary for fieldmap correction.')
        pe_echoes = nb.load(in_file).shape[int(abs(pe_dir))]
        ro_time = 1.0 / (pebw * pe_echoes)

    if ro_time is None:
        raise RuntimeError('Readout time could not be set')

    return pe_dir, ro_time

def apply_epi_mask(in_file, mask=None):
    out_file = genfname(in_file, suffix='brainmask')
    nii = nb.load(in_file)
    data = nii.get_data()
    if mask is None:
        mask = compute_epi_mask(
            nii, lower_cutoff=0.05, upper_cutoff=0.95,
            opening=False, exclude_zeros=True).get_data()

    data[np.where(mask <= 0)] = 0
    nb.Nifti1Image(data, nii.get_affine(),
                   nii.get_header()).to_filename(out_file)
    return out_file

def _generate_topup_movpar(in_mats):
    # TODO!
    return np.zeros((len(in_mats), 6))
