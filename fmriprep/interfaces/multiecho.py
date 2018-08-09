#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
T2* map generation
~~~~~~~~~~~~~~~~~~~~~~

Using multi-echo EPI data, generates a T2*-map
for use in T2*-driven EPI->T1 coregistration

Change directory to provide relative paths for doctests
>>> import os
>>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
>>> datadir = os.path.realpath(os.path.join(filepath, '../data/'))
>>> os.chdir(datadir)

"""
import os
import numpy as np
import nibabel as nb
from nilearn.masking import (apply_mask, unmask)

from nipype import logging
from nipype.utils.filemanip import (split_filename, fname_presuffix)
from nipype.interfaces.base import (
    traits, TraitedSpec, File, InputMultiPath, SimpleInterface,
    BaseInterfaceInputSpec)

LOGGER = logging.getLogger('nipype.interface')


class FirstEchoInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, minlen=2,
                              desc='multi-echo BOLD EPIs')
    ref_imgs = InputMultiPath(File(exists=True), mandatory=True, minlen=2,
                              desc='generated reference image for each '
                              'multi-echo BOLD EPI')
    te_list = traits.List(traits.Float, mandatory=True, desc='echo times')


class FirstEchoOutputSpec(TraitedSpec):
    first_image = File(exists=True,
                       desc='BOLD EPI series for the first echo')
    first_ref_image = File(exists=True, desc='generated reference image for '
                                             'the first echo')


class FirstEcho(SimpleInterface):
    """
    Finds the first echo in a multi-echo series and its associated reference
    image.

    Example
    =======
    >>> from fmriprep.interfaces import multiecho
    >>> first_echo = multiecho.FirstEcho()
    >>> first_echo.inputs.in_files = ['sub-01_run-01_echo-1_bold.nii.gz', \
                                      'sub-01_run-01_echo-2_bold.nii.gz', \
                                      'sub-01_run-01_echo-3_bold.nii.gz']
    >>> first_echo.inputs.ref_imgs = ['sub-01_run-01_echo-1_bold.nii.gz', \
                                      'sub-01_run-01_echo-2_bold.nii.gz', \
                                      'sub-01_run-01_echo-3_bold.nii.gz']
    >>> first_echo.inputs.te_list = [0.013, 0.027, 0.043]
    >>> res = first_echo.run()
    >>> res.outputs.first_image
    'sub-01_run-01_echo-1_bold.nii.gz'
    >>> res.outputs.first_ref_image
    'sub-01_run-01_echo-1_bold.nii.gz'
    """
    input_spec = FirstEchoInputSpec
    output_spec = FirstEchoOutputSpec

    def _run_interface(self, runtime):
        self._results['first_image'] = self.inputs.in_files[np.argmin(self.inputs.te_list)]
        self._results['first_ref_image'] = self.inputs.ref_imgs[np.argmin(self.inputs.te_list)]

        return runtime


class MaskT2SMapInputSpec(BaseInterfaceInputSpec):
    image = File(mandatory=True, exists=True, desc='T2* volume to mask')
    mask = File(mandatory=True, exists=True,
                desc='skull-stripped mean optimal combination volume')
    compress = traits.Bool(True, usedefault=True,
                           desc='use gzip compression on .nii output')


class MaskT2SMapOutputSpec(TraitedSpec):
    masked_t2s = File(exists=True, desc='masked T2* map')


class MaskT2SMap(SimpleInterface):
    """
    Masks an existing T2* map using the skull-stripped optimally combined
    volume (i.e., a weighted combination of multi-echo data).

    Example
    =======
    >>> from fmriprep.interfaces import multiecho
    >>> mask_t2s = multiecho.MaskT2SMap()
    >>> mask_t2s.inputs.image = 'sub-01_run-01_t2smap.nii.gz'
    >>> mask_t2s.inputs.mask = 'sub-01_run-01_optcomb.nii.gz'
    >>> res = mask_t2s.run() # doctest: +SKIP
    """

    input_spec = MaskT2SMapInputSpec
    output_spec = MaskT2SMapOutputSpec

    def _run_interface(self, runtime):
        ext = '.nii.gz' if self.inputs.compress else '.nii'
        flat_mask = apply_mask(self.inputs.image, self.inputs.mask)
        masked_image = unmask(flat_mask, self.inputs.mask)
        self._results['masked_t2s'] = fname_presuffix(
            self.inputs.image, suffix='_masked' + ext, newpath=runtime.cwd, use_ext=False)
        masked_image.to_filename(self._results['masked_t2s'])
        return runtime


class T2SMapInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='multi-echo BOLD EPIs')
    te_list = traits.List(traits.Float, mandatory=True, desc='echo times')
    compress = traits.Bool(True, usedefault=True,
                           desc='use gzip compression on .nii output')


class T2SMapOutputSpec(TraitedSpec):
    t2s_vol = File(exists=True, desc='T2* map')
    opt_comb = File(exists=True, desc='optimal combination of echos')


class T2SMap(SimpleInterface):
    """
    Generates a T2* map and optimally combined average volume (i.e., a weighted
    combination) for multi-echo data, for use in coregistration.

    Example
    =======
    >>> from fmriprep.interfaces import multiecho
    >>> t2s_map = multiecho.T2SMap()
    >>> t2s_map.inputs.in_files = ['sub-01_run-01_echo-1_bold.nii.gz', \
                                   'sub-01_run-01_echo-2_bold.nii.gz', \
                                   'sub-01_run-01_echo-3_bold.nii.gz']
    >>> t2s_map.inputs.te_list = [0.013, 0.027, 0.040]
    >>> res = t2s_map.run() # doctest: +SKIP
    """
    input_spec = T2SMapInputSpec
    output_spec = T2SMapOutputSpec

    def _run_interface(self, runtime):
        ext = '.nii.gz' if self.inputs.compress else '.nii'
        e1_nii = nb.load(self.inputs.in_files[0])

        last_emask, two_emask = echo_sampling_mask(self.inputs.in_files)
        t2s_map = define_t2s_map(self.inputs.in_files, self.inputs.te_list,
                                 last_emask, two_emask)
        opt_comb = get_opt_comb(self.inputs.in_files, self.inputs.te_list,
                                t2s_map, last_emask)
        _, fname, _ = split_filename(self.inputs.in_files[0])
        fname_preecho = fname.split('_echo-')[0]
        self._results['t2s_vol'] = os.path.join(runtime.cwd, fname_preecho + '_t2smap' + ext)
        self._results['opt_comb'] = os.path.join(runtime.cwd, fname_preecho + '_optcomb' + ext)
        nb.Nifti1Image(t2s_map, e1_nii.affine, e1_nii.header).to_filename(
            self._results['t2s_vol'])
        nb.Nifti1Image(opt_comb, e1_nii.affine, e1_nii.header).to_filename(
            self._results['opt_comb'])
        return runtime


def echo_sampling_mask(echo_list):
    """
    Make a map of longest echo that a voxel can be sampled with,
    with minimum value of map as X value of voxel that has median
    value in the 1st echo. N.B. larger factor leads to bias to lower TEs

    **Inputs**

        echo_list
            List of file names for all echos

    **Outputs**

        last_emask
            numpy array whose values correspond to which
            echo a voxel can last be sampled with
        two_emask
            boolean array of voxels that can be sampled
            with at least two echos

    """
    # First, load each echo and average over time
    echos = [np.mean(nb.load(e).get_data(), axis=-1) for e in echo_list]

    # In the first echo, find the 33rd percentile and the voxel(s)
    # whose average activity is equal to that value
    perc33 = np.percentile(echos[0][echos[0].nonzero()],
                           33, interpolation="higher")
    med_vox = (echos[0] == perc33)

    # For each (averaged) echo, extract the max signal in the
    # identified voxels and divide it by 3-- save as a threshold
    thrs = np.hstack([np.max(echo[med_vox]) / 3 for echo in echos])

    # Let's stack the arrays to make this next bit easier
    emeans = np.stack(echos, axis=-1)

    # Now, we want to find all voxels (in each echo) that show
    # absolute signal greater than our echo-specific threshold
    mthr = np.ones_like(emeans)
    mthr *= thrs[np.newaxis, np.newaxis, np.newaxis, :]
    voxels = np.abs(emeans) > mthr

    # We save those voxel indices out to an array
    last_emask = np.array(voxels, dtype=np.int).sum(-1)
    # Save a mask of voxels sampled by at least two echos
    two_emask = (last_emask != 0)

    return last_emask, two_emask


def define_t2s_map(echo_list, tes, last_emask, two_emask):
    """
    Computes the quantiative T2* mapping according to
    :math:`ΔS/S = ΔS0/S0 − ΔR2 * TE`.

    **Inputs**

        echo_list
            list of file names for all echos
        tes
            echo times for the multi-echo EPI run
        last_emask
            numpy array where voxel values correspond to which
            echo a voxel can last be sampled with
        two_emask
            boolean array of voxels that can be sampled
            with at least two echos

    **Outputs**

        t2s_map
            the T2* map for the EPI run
    """
    # get some basic shape information
    echo_stack = np.stack([nb.load(echo).get_data() for echo in echo_list],
                          axis=-2)
    nx, ny, nz, necho, nt = echo_stack.shape

    # create empty arrays to fill later
    t2ss = np.zeros([nx, ny, nz, necho - 1])
    s0vs = t2ss.copy()

    # consider only those voxels sampled by at least two echos
    two_edata = echo_stack[two_emask.nonzero()]
    two_echo_nvox = two_edata.shape[0]

    # for the second echo on, do log linear fit
    for echo in range(2, necho + 1):

        # multiply by 1000, so in ms rather than s
        # ΔS/S = ΔS0/S0 − ΔR2 * TE, so take neg TEs
        neg_tes = [-1000 * te for te in tes[:echo]]

        # Create coefficient matrix
        a = np.array([np.ones(echo), neg_tes])
        A = np.tile(a, (1, nt))
        A = np.sort(A)[:, ::-1].transpose()

        # Create log-scale dependent-var matrix
        B = np.reshape(np.abs(two_edata[:, :echo, :]) + 1,
                       (two_echo_nvox, echo * nt)).transpose()
        B = np.log(B)

        # find the least squares solution for the echo
        X, res, rank, sing = np.linalg.lstsq(A, B)

        # scale the echo-coefficients (ΔR2), intercept (s0)
        r2 = 1 / X[1, :].transpose()
        s0 = np.exp(X[0, :]).transpose()

        # fix any non-numerical values
        r2[np.isinf(r2)] = 500.
        s0[np.isnan(s0)] = 0.

        # reshape into arrays for mapping
        t2ss[:, :, :, echo - 2] = _unmask(r2, two_emask)
        s0vs[:, :, :, echo - 2] = _unmask(s0, two_emask)

    # limited T2* and S0 maps
    fl = np.zeros([nx, ny, nz, necho - 1])
    for echo in range(necho - 1):
        fl_ = fl[:, :, :, echo]
        fl_[last_emask == echo + 2] = True
        fl[:, :, :, echo] = fl_

    fl = np.array(fl, dtype=bool)
    t2s_map = np.squeeze(_unmask(t2ss[fl], last_emask > 1))
    t2s_map[np.logical_or(np.isnan(t2s_map), t2s_map < 0)] = 0

    return t2s_map


def get_opt_comb(echo_list, tes, t2s_map, last_emask):
    """
    Returns the optimal combination of all supplied echos,
    averaged over time.

    **Inputs**

        echo_list
            list of file names for all echos
        tes
            echo times for the multi-echo EPI run
        last_emask
            numpy array where voxel values correspond to which
            echo a voxel can last be sampled with

    **Outputs**

        opt_comb
            the optimal combination of echos
    """
    # get some basic shape information
    echo_stack = np.stack([nb.load(echo).get_data() for echo in echo_list],
                          axis=-2)
    nx, ny, nz, necho, nt = echo_stack.shape

    fdat = _fmask(echo_stack, last_emask)
    ft2s = _fmask(t2s_map, last_emask)

    ft2s = ft2s[:, np.newaxis]

    alpha = fdat.mean(-1) * tes
    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, nt))

    fout = np.average(fdat, axis=1, weights=alpha)
    opt_comb = _unmask(fout, last_emask)
    return np.average(opt_comb, axis=-1)


def _fmask(data, mask):
    """
    Masks `data` using non-zero entries of `mask`

    **Inputs**

    data
        Masked array of shape (nx*ny*nz[, Ne[, nt]])
    mask
        Boolean array of shape (nx, ny, nz)

    **Outputs**

    ndarray
        Array of shape (nx, ny, nz[, Ne[, nt]])
    """
    new_s = tuple([np.prod(data.shape[:3])] + list(data.shape[3:]))

    tmp1 = np.reshape(data, new_s)
    fdata = tmp1.compress((mask > 0).ravel(), axis=0)

    return fdata.squeeze()


def _unmask(data, mask):
    """
    Unmasks `data` using non-zero entries of `mask`

    **Inputs**

    data
        Masked array of shape (nx*ny*nz[, Ne[, nt]])
    mask
        Boolean array of shape (nx, ny, nz)

    **Outputs**

    ndarray
        Array of shape (nx, ny, nz[, Ne[, nt]])

    """

    M = (mask != 0).ravel()
    Nm = M.sum()

    nx, ny, nz = mask.shape

    if len(data.shape) > 1:
        nt = data.shape[1]
    else:
        nt = 1

    out = np.zeros((nx * ny * nz, nt), dtype=data.dtype)
    out[M, :] = np.reshape(data, (Nm, nt))

    return np.squeeze(np.reshape(out, (nx, ny, nz, nt)))
