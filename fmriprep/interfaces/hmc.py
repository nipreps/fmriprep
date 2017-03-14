#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Motion correction helpers
~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os, os.path as op
import numpy as np
import nibabel as nb
from scipy.ndimage.measurements import center_of_mass
from nipype import logging
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    File, InputMultiPath, traits, OutputMultiPath, CommandLine
)

from nilearn.image import concat_imgs, mean_img

from io import open
from ..utils.misc import genfname
from .itk import ITK_TFM_HEADER, ITK_TFM_TPL

LOGGER = logging.getLogger('interface')


class MotionCorrectionInputSpec(BaseInterfaceInputSpec):
    in_split = InputMultiPath(File(exists=True), mandatory=True, desc='input file')
    in_merged = File(exists=True, mandatory=True, desc='input file')
    ref_vol = traits.Float(0.5, usedefault=True,
                           desc='location of the reference volume (0.0 is first, 1.0 is last)')
    reference_image = File(exists=True, desc='use reference image')
    nthreads = traits.Int(-1, usedefault=True, nohash=True,
                          desc='number of threads to use within ANTs registrations')

class MotionCorrectionOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(
        File(exists=True), desc='list of output files')
    out_avg = File(exists=True, desc='average across time')
    out_movpar = File(exists=True, desc='output motion parameters')
    out_confounds = File(exists=True, desc='output confounds file')
    out_tfm = File(exists=True, desc='list of transform matrices')

    inputs_merged = File(exists=True, desc='input file(s) flattened and merged')
    inputs_split = OutputMultiPath(File(exists=True),
                                   desc='input file(s) flattened and split')


class MotionCorrection(BaseInterface):

    """
    This interface generates the input files required by FSL topup:

      * A 4D file with all input images
      * The topup-encoding parameters file corresponding to the 4D file.


    """
    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(MotionCorrection, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):

        # Set number of threads
        nthreads = self.inputs.nthreads
        if nthreads < 1:
            from multiprocessing import cpu_count
            nthreads = cpu_count()

        # Check input files.
        if len(self.inputs.in_split) < 2:
            raise RuntimeError('in_split should be a list of 2 or more files')

        ref_image = None
        if isdefined(self.inputs.reference_image):
            ref_image = self.inputs.reference_image

        out_files, movpar = motion_correction(
            self.inputs.in_split, self.inputs.in_merged,
            reference_image=ref_image, ref_vol=self.inputs.ref_vol, nthreads=nthreads)

        # Save average image
        out_avg = genfname(self.inputs.in_split[0], suffix='average')
        mean_img(out_files, n_jobs=nthreads).to_filename(out_avg)

        # Set outputs
        self._results['out_avg'] = out_avg
        self._results['out_files'] = out_files
        self._results['out_movpar'] = op.abspath(movpar)
        self._results['out_tfm'] = moco2itk(op.abspath(movpar), out_avg)
        return runtime

def motion_correction(in_files, merged_files, reference_image=None,
                      ref_vol=0.5, nthreads=None):
    """
    A very simple motion correction workflow including two
    passes of antsMotionCorr
    """

    nfiles = len(in_files)
    if reference_image is None:
        reference_image = in_files[int(ref_vol * (nfiles - 0.5))]

    out_prefix = op.basename(genfname(reference_image,
                             suffix='motcor0', ext=''))
    out_file = _run_antsMotionCorr(out_prefix, reference_image, merged_files,
                                   only_rigid=(nfiles == 2), nthreads=nthreads)

    if nfiles > 2:
        out_prefix = op.basename(genfname(reference_image, suffix='motcor1', ext=''))
        out_file = _run_antsMotionCorr(out_prefix, out_file, merged_files,
                                       nthreads=nthreads, iteration=1)

    out_movpar = out_prefix + 'MOCOparams.csv'
    return out_file, out_movpar


def _run_antsMotionCorr(out_prefix, ref_image, all_images, return_avg=True,
                        only_rigid=False, nthreads=None, iteration=0):
    env = os.environ.copy()
    if nthreads is not None and nthreads > 0:
        env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '%d' % nthreads


    npoints = nb.load(all_images).get_data().shape[-1]
    if iteration == 0:
        args = ['-d', '3', '-o', '[{0},{0}.nii.gz,{0}_avg.nii.gz]'.format(out_prefix),
                '-m', 'MI[%s, %s, 1, 32, Regular, 0.2]' % (ref_image, all_images),
                '-t', '%s[ 0.5 ]' % ('Rigid' if only_rigid else 'Affine'),
                '-u', '1', '-e', '1', '-l', '1',
                '-i', '40x20', '-s', '4x1', '-f', '2x1']
    else:
        args = ['-d', '3', '-o', '[{0},{0}.nii.gz,{0}_avg.nii.gz]'.format(out_prefix),
                '-m', 'gc[%s, %s, 1, 5, Random, 0.2]' % (ref_image, all_images),
                '-t', '%s[ 0.1 ]' % ('Rigid' if only_rigid else 'Affine'),
                '-u', '1', '-e', '1', '-l', '1',
                '-i', '15x3', '-s', '2x0', '-f', '2x1']

    # Add number of volumes to average
    args += ['-n', str(min(npoints, 10))]

    cmd = CommandLine(command='antsMotionCorr', args=' '.join(args), environ=env)
    LOGGER.info('Running antsMotionCorr: %s', cmd.cmdline)
    result = cmd.run()
    returncode = getattr(result.runtime, 'returncode', 1)

    if returncode not in [0, None]:
        raise RuntimeError('antsMotionCorr failed to run (exit code %d)\n%s' % (
            returncode, cmd.cmdline))

    if return_avg:
        return op.abspath(out_prefix + '_avg.nii.gz')
    else:
        return op.abspath(out_prefix + '.nii.gz')


def moco2itk(in_csv, in_reference, out_file=None):
    movpar = np.loadtxt(in_csv, dtype=float, skiprows=1,
                        delimiter=',')[:, 2:]

    nii = nb.load(in_reference)

    # Convert first to RAS, then to LPS
    cmfixed = np.diag([-1, -1, 1, 1]).dot(nii.affine).dot(
        list(center_of_mass(nii.get_data())) + [1])

    tf_type = ('AffineTransform_double_3_3' if movpar.shape[1] == 12
               else 'Euler3DTransform_double_3_3')

    xfm_file = [ITK_TFM_HEADER]
    for i, param in enumerate(movpar):
        xfm_file.append(ITK_TFM_TPL(
            tf_id=i, tf_type=tf_type,
            tf_params=' '.join(['%.8f' % p for p in param]),
            fixed_params=' '.join(['%.8f' % p for p in cmfixed[:3]])
        ))
    xfm_file += ['']

    if out_file is None:
        out_file = genfname(in_reference, suffix='itk', ext='tfm')

    with open(out_file, 'w') as outfh:
        outfh.write("\n".join(xfm_file))

    return out_file
