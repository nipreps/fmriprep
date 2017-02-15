#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Some tooling for handling fieldmaps (potentially to be used inside
nipype Function nodes)

"""
from __future__ import print_function, division, absolute_import, unicode_literals


def create_encoding_file(input_images, in_dict):
    """
    Creates a valid encoding file for topup
    """
    import nibabel as nb
    import numpy as np
    import os

    if not isinstance(input_images, list):
        input_images = [input_images]
    if not isinstance(in_dict, list):
        in_dict = [in_dict]

    pe_dirs = {'i': 0, 'j': 1, 'k': 2}
    enc_table = []
    for fmap, meta in zip(input_images, in_dict):
        readout_time = meta.get('TotalReadoutTime')
        eff_echo = meta.get('EffectiveEchoSpacing')
        echo_time = meta.get('EchoTime')
        if not any((readout_time, eff_echo, echo_time)):
            raise RuntimeError(
                'One of "TotalReadoutTime", "EffectiveEchoSpacing" or "EchoTime"'
                ' should be found in the sidecar JSON corresponding to '
                '"%s".', fmap)

        meta_pe = meta.get('PhaseEncodingDirection')
        if meta_pe is None:
            raise RuntimeError('PhaseEncodingDirection not found among '
                               'metadata of file "%s"', fmap)

        pe_axis = pe_dirs[meta_pe[0]]
        if readout_time is None:
            if eff_echo is None:
                raise NotImplementedError

            # See http://support.brainvoyager.com/functional-analysis-preparation/27-pre-
            #     processing/459-epi-distortion-correction-echo-spacing.html
            # See http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP/ExampleTopupFollowedByApplytopup
            # Particularly "the final column is the time (in seconds) between the readout of the
            # centre of the first echo and the centre of the last echo (equal to dwell-time
            # multiplied by # of phase-encode steps minus one)"
            pe_size = nb.load(fmap).get_data().shape[pe_axis]
            readout_time = eff_echo * (pe_size - 1)

        line_values = [0, 0, 0, readout_time]
        line_values[pe_axis] = 1 + (-2*(len(meta['PhaseEncodingDirection']) == 2))

        nvols = 1
        if len(nb.load(fmap).shape) > 3:
            nvols = nb.load(fmap).shape[3]

        enc_table += [line_values] * nvols

    np.savetxt(os.path.abspath('parameters.txt'), enc_table,
               fmt=['%0.1f', '%0.1f', '%0.1f', '%0.20f'])
    return os.path.abspath('parameters.txt')


def mcflirt2topup(in_files, in_mats, out_movpar=None):
    """
    Converts a list of matrices from MCFLIRT to the movpar input
    of TOPUP (a row per file with 6 parameters - 3 translations and 3 rotations
    in this particular order).

    """

    import os.path as op
    import numpy as np
    params = np.zeros((len(in_files), 6))

    if in_mats:
        if len(in_mats) != len(in_files):
            raise RuntimeError('Number of input matrices and files do not match')
        else:
            raise NotImplementedError

    if out_movpar is None:
        fname, fext = op.splitext(op.basename(in_files[0]))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_movpar = op.abspath('./%s_movpar.txt' % fname)

    np.savetxt(out_movpar, params)
    return out_movpar
