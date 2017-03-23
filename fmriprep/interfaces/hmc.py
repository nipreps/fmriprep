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
from pkg_resources import resource_filename as pkgrf


from scipy.ndimage.measurements import center_of_mass
from nipype import logging
from nipype.utils.filemanip import load_json
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    File, InputMultiPath, traits, OutputMultiPath, CommandLine, Directory
)

from nipype.interfaces.ants import Registration
from nilearn.image import mean_img

from io import open
from ..utils.misc import genfname
from .itk import ITK_TFM_HEADER, ITK_TFM_TPL

LOGGER = logging.getLogger('interface')


class MotionCorrectionInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='input file')
    reference_image = File(exists=True, desc='use reference image')
    interp = traits.Enum(
        'LanczosWindowedSinc', 'Linear', 'NearestNeighbor', 'CosineWindowedSinc',
        'WelchWindowedSinc', 'HammingWindowedSinc', 'BSpline', 'MultiLabel', 'Gaussian',
        usedefault=True, desc='interpolation of the final resampling step')
    ref_vol = traits.Float(0.5, usedefault=True,
                           desc='location of the reference volume (0.0 is first, 1.0 is last)')
    njobs = traits.Int(-1, usedefault=True, nohash=True,
                       desc='number of threads to use within ANTs registrations')
    cache_dir = Directory(desc='use cache directory', nohash=True)

class MotionCorrectionOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(
        File(exists=True), desc='list of output files')
    out_avg = File(exists=True, desc='average across time')
    out_tfm = OutputMultiPath(File(exists=True),
                              desc='list of affine matrices')


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
        njobs = self.inputs.njobs
        if njobs < 1:
            from multiprocessing import cpu_count
            njobs = cpu_count()

        # Check input files.
        if len(self.inputs.in_files) < 2:
            LOGGER.warn('Tried to run HMC on a single 3D file')
            self._results['out_avg'] = self.inputs.in_files[0]
            self._results['out_files'] = self.inputs.in_files
            self._results['out_tfm'] = pkgrf('fmriprep', 'data/itk_identity.tfm')
            return runtime

        ref_image = None
        if isdefined(self.inputs.reference_image):
            ref_image = self.inputs.reference_image

        cache_dir = None
        if isdefined(self.inputs.cache_dir):
            cache_dir = self.inputs.cache_dir

        out_files, out_avg, out_tfm = motion_correction(
            self.inputs.in_files,
            interp=self.inputs.interp,
            reference_image=ref_image,
            ref_vol=self.inputs.ref_vol,
            njobs=njobs,
            cache_dir=cache_dir)

        # Set outputs
        self._results['out_avg'] = out_avg
        self._results['out_files'] = out_files
        self._results['out_tfm'] = out_tfm
        return runtime

def motion_correction(in_files, interp=None, reference_image=None,
                      ref_vol=0.5, njobs=None, cache_dir=None):
    """
    A very simple motion correction workflow including two
    passes of antsMotionCorr
    """
    from nipype.caching import Memory

    if cache_dir is None:
        cache_dir = os.getcwd()

    mem = Memory(cache_dir)

    nfiles = len(in_files)
    if reference_image is None:
        reference_image = in_files[int(ref_vol * (nfiles - 0.5))]

    out_files_0 = []
    if nfiles > 2:
        argsdict = load_json(pkgrf('fmriprep', 'data/moco_level0.json'))
        argsdict['fixed_image'] = reference_image

        if njobs is not None:
            argsdict['num_threads'] = njobs
        if interp is not None:
            argsdict['interpolation'] = interp

        LOGGER.info('HMC [level 0] of %d timepoints', nfiles)
        for in_file in in_files:
            if in_file == reference_image:
                out_files_0.append(in_file)
                continue

            args = argsdict.copy()
            args['moving_image'] = in_file

            ants_0 = mem.cache(Registration)(**args)
            out_files_0.append(ants_0.outputs.warped_image)

        reference_image = genfname(reference_image, suffix='motcor0_avg')
        mean_img(out_files_0, n_jobs=njobs).to_filename(reference_image)


    out_files = []
    out_tfms = []

    if nfiles == 2:
        reference_image = in_files[0]
        out_tfms = [pkgrf('fmriprep', 'data/itk_identity.tfm')]

    argsdict = load_json(pkgrf('fmriprep', 'data/moco_level1.json'))
    argsdict['fixed_image'] = reference_image
    if njobs is not None:
        argsdict['num_threads'] = njobs
    if interp is not None:
        argsdict['interpolation'] = interp

    LOGGER.info('HMC [level 1] of %d timepoints', nfiles)
    for in_file in in_files:
        if in_file == reference_image:
            out_files.append(in_file)
            continue

        args = argsdict.copy()
        args['moving_image'] = in_file

        ants_1 = mem.cache(Registration)(**args)
        out_files.append(ants_1.outputs.warped_image)
        out_tfms.append(ants_1.outputs.forward_transforms[0])


    out_avg = genfname(reference_image, suffix='motcor1_avg')
    mean_img(out_files, n_jobs=njobs).to_filename(out_avg)
    return out_files, out_avg, out_tfms


def _run_antsMotionCorr(image4d, ref_image, suffix='motcor',
                        only_rigid=False, njobs=None, iteration=0):
    env = os.environ.copy()
    if njobs is not None and njobs > 0:
        env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '%d' % njobs

    out_prefix = op.basename(genfname(
        ref_image, suffix='%s%d' % (suffix, iteration), ext=''))


    npoints = nb.load(image4d).get_data().shape[-1]
    if iteration == 0:
        args = ['-d', '3', '-o', '[{0},{0}.nii.gz,{0}_avg.nii.gz]'.format(out_prefix),
                '-m', 'MI[%s, %s, 1, 32, Regular, 0.2]' % (ref_image, image4d),
                '-t', '%s[ 0.5 ]' % ('Rigid' if only_rigid else 'Affine'),
                '-u', '1', '-e', '1', '-l', '1',
                '-i', '40x20', '-s', '4x1', '-f', '2x1']
    else:
        args = ['-d', '3', '-o', '[{0},{0}.nii.gz,{0}_avg.nii.gz]'.format(out_prefix),
                '-m', 'gc[%s, %s, 1, 5, Random, 0.2]' % (ref_image, image4d),
                '-t', '%s[ 0.1 ]' % ('Rigid' if only_rigid else 'Affine'),
                '-u', '1', '-e', '1', '-l', '1',
                '-i', '15x3', '-s', '2x0', '-f', '2x1']

    # Add number of volumes to average
    args += ['-n', str(min(npoints, 10))]

    cmd = CommandLine(command='antsMotionCorr', args=' '.join(args), environ=env)
    LOGGER.info('Running antsMotionCorr [level %d]: %s', iteration, cmd.cmdline)
    result = cmd.run()
    returncode = getattr(result.runtime, 'returncode', 1)

    if returncode not in [0, None]:
        raise RuntimeError('antsMotionCorr failed to run (exit code %d)\n%s' % (
            returncode, cmd.cmdline))

    return out_prefix


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

def itk2moco(in_files, out_par=None, out_confounds=None):
    import re
    import numpy as np
    from nibabel.eulerangles import mat2euler
    from nipype.interfaces.base import CommandLine
    from builtins import str, bytes
    from fmriprep.utils.misc import genfname


    if isinstance(in_files, (str, bytes)):
        in_files = [in_files]

    expr_mat = re.compile('Matrix:\n(?P<matrix>[0-9\.\ -]+\n[0-9\.\ -]+\n[0-9\.\ -]+)\n')
    expr_tra = re.compile(
        'Translation:\s+\[(?P<translation>[0-9\.-]+,\s[0-9\.-]+,\s[0-9\.-]+)\]')

    moco = []
    for in_file in in_files:
        cmd = CommandLine(command='antsTransformInfo', args=in_file,
                          terminal_output='file')
        stdout = cmd.run().runtime.stdout

        if 'IdentityTransform' in stdout:
            moco.append(np.eye(3).reshape(-1).tolist() + [0.0] * 3)
            continue

        mat = [float(v) for v in re.split(
            ' |\n', expr_mat.search(stdout).group('matrix'))]
        trans = [float(v) for v in expr_tra.search(
            stdout).group('translation').split(', ')]

        param_z, param_y, param_x = mat2euler(np.array(mat).reshape(3, 3))
        moco.append([param_x, param_y, param_z] + trans)

    moco = np.array(moco, dtype=np.float32)

    if out_par is None:
        out_par = genfname(in_files[0], suffix='moco', ext='par')
    np.savetxt(out_par, moco)


    if out_confounds is None:
        out_confounds = genfname(in_files[0], suffix='confounds', ext='tsv')
    np.savetxt(out_confounds, moco, delimiter='\t')

    return out_par, out_confounds
