#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FreeSurfer tools interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os.path as op
import nibabel as nb

from nilearn.image import resample_to_img, new_img_like

from niworkflows.nipype.utils.filemanip import copyfile, filename_to_list
from niworkflows.nipype.interfaces.base import (
    isdefined, InputMultiPath, BaseInterfaceInputSpec, TraitedSpec, File, traits, Directory
)
from niworkflows.nipype.interfaces import freesurfer as fs

from niworkflows.interfaces.base import SimpleInterface


class StructuralReference(fs.RobustTemplate):
    """ Variation on RobustTemplate that simply copies the source if a single
    volume is provided. """
    @property
    def cmdline(self):
        cmd = super(StructuralReference, self).cmdline
        if len(self.inputs.in_files) > 1:
            return cmd

        img = nb.load(self.inputs.in_files[0])
        if len(img.shape) > 3 and img.shape[3] > 1:
            return cmd

        out_file = self._list_outputs()['out_file']
        copyfile(self.inputs.in_files[0], out_file)
        return "echo Only one time point!"


class MakeMidthicknessInputSpec(fs.utils.MRIsExpandInputSpec):
    graymid = InputMultiPath(desc='Existing graymid/midthickness file')


class MakeMidthickness(fs.MRIsExpand):
    """ Variation on RobustTemplate that simply copies the source if a single
    volume is provided. """
    input_spec = MakeMidthicknessInputSpec

    @property
    def cmdline(self):
        cmd = super(MakeMidthickness, self).cmdline
        if not isdefined(self.inputs.graymid) or len(self.inputs.graymid) < 1:
            return cmd

        # Possible graymid values inclue {l,r}h.{graymid,midthickness}
        # Prefer midthickness to graymid, require to be of the same hemisphere
        # as input
        source = None
        in_base = op.basename(self.inputs.in_file)
        mt = self._associated_file(in_base, 'midthickness')
        gm = self._associated_file(in_base, 'graymid')

        for surf in self.inputs.graymid:
            if op.basename(surf) == mt:
                source = surf
                break
            if op.basename(surf) == gm:
                source = surf

        if source is None:
            return cmd

        return "cp {} {}".format(source, self._list_outputs()['out_file'])


class FSInjectBrainExtractedInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(mandatory=True, desc='FreeSurfer SUBJECTS_DIR')
    subject_id = traits.Str(mandatory=True, desc='Subject ID')
    in_brain = File(mandatory=True, exists=True, desc='input file, part of a BIDS tree')


class FSInjectBrainExtractedOutputSpec(TraitedSpec):
    subjects_dir = Directory(desc='FreeSurfer SUBJECTS_DIR')
    subject_id = traits.Str(desc='Subject ID')


class FSInjectBrainExtracted(SimpleInterface):
    input_spec = FSInjectBrainExtractedInputSpec
    output_spec = FSInjectBrainExtractedOutputSpec

    def _run_interface(self, runtime):
        subjects_dir, subject_id = inject_skullstripped(
            self.inputs.subjects_dir,
            self.inputs.subject_id,
            self.inputs.in_brain)
        self._results['subjects_dir'] = subjects_dir
        self._results['subject_id'] = subject_id
        return runtime


class FSDetectInputsInputSpec(BaseInterfaceInputSpec):
    t1w_list = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input file, part of a BIDS tree')
    t2w_list = InputMultiPath(File(exists=True), desc='input file, part of a BIDS tree')
    hires_enabled = traits.Bool(True, usedefault=True, desc='enable hi-resolution processing')


class FSDetectInputsOutputSpec(TraitedSpec):
    t2w = File(desc='reference T2w image')
    use_t2w = traits.Bool(desc='enable use of T2w downstream computation')
    hires = traits.Bool(desc='enable hi-res processing')
    mris_inflate = traits.Str(desc='mris_inflate argument')


class FSDetectInputs(SimpleInterface):
    input_spec = FSDetectInputsInputSpec
    output_spec = FSDetectInputsOutputSpec

    def _run_interface(self, runtime):
        t2w, self._results['hires'], mris_inflate = detect_inputs(
            self.inputs.t1w_list,
            hires_enabled=self.inputs.hires_enabled,
            t2w_list=self.inputs.t2w_list if isdefined(self.inputs.t2w_list) else None)

        self._results['use_t2w'] = t2w is not None
        if self._results['use_t2w']:
            self._results['t2w'] = t2w

        if self._results['hires']:
            self._results['mris_inflate'] = mris_inflate

        return runtime


def inject_skullstripped(subjects_dir, subject_id, skullstripped):
    mridir = op.join(subjects_dir, subject_id, 'mri')
    t1 = op.join(mridir, 'T1.mgz')
    bm_auto = op.join(mridir, 'brainmask.auto.mgz')
    bm = op.join(mridir, 'brainmask.mgz')

    if not op.exists(bm_auto):
        img = nb.load(t1)
        mask = nb.load(skullstripped)
        bmask = new_img_like(mask, mask.get_data() > 0)
        resampled_mask = resample_to_img(bmask, img, 'nearest')
        masked_image = new_img_like(img, img.get_data() * resampled_mask.get_data())
        masked_image.to_filename(bm_auto)

    if not op.exists(bm):
        copyfile(bm_auto, bm, copy=True, use_hardlink=True)

    return subjects_dir, subject_id


def detect_inputs(t1w_list, t2w_list=None, hires_enabled=True):
    t1w_list = filename_to_list(t1w_list)
    t2w_list = filename_to_list(t2w_list) if t2w_list is not None else []
    t1w_ref = nb.load(t1w_list[0])
    # Use high resolution preprocessing if voxel size < 1.0mm
    # Tolerance of 0.05mm requires that rounds down to 0.9mm or lower
    hires = hires_enabled and max(t1w_ref.header.get_zooms()) < 1 - 0.05

    t2w = None
    if t2w_list and max(nb.load(t2w_list[0]).header.get_zooms()) < 1.2:
        t2w = t2w_list[0]

    # https://surfer.nmr.mgh.harvard.edu/fswiki/SubmillimeterRecon
    mris_inflate = '-n 50' if hires else None
    return (t2w, hires, mris_inflate)
