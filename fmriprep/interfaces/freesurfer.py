#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FreeSurfer tools interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetch some example data:

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

Disable warnings:

    >>> import niworkflows.nipype as nn
    >>> nn.logging.getLogger('interface').setLevel('ERROR')

"""

import os.path as op
import nibabel as nb

from nilearn.image import resample_to_img, new_img_like

from niworkflows.nipype.utils.filemanip import copyfile, filename_to_list
from niworkflows.nipype.interfaces.base import (
    isdefined, InputMultiPath, BaseInterfaceInputSpec, TraitedSpec, File, traits, Directory
)
from niworkflows.nipype.interfaces import freesurfer as fs
from niworkflows.nipype.interfaces.base import SimpleInterface
from niworkflows.nipype.interfaces.freesurfer.preprocess import ConcatenateLTA


class StructuralReference(fs.RobustTemplate):
    """ Variation on RobustTemplate that simply copies the source if a single
    volume is provided.

    >>> from fmriprep.utils.bids import collect_data
    >>> t1w = collect_data('ds114', '01')[0]['t1w']
    >>> template = StructuralReference()
    >>> template.inputs.in_files = t1w
    >>> template.inputs.auto_detect_sensitivity = True
    >>> template.cmdline  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    'mri_robust_template --satit --mov .../sub-01_ses-retest_T1w.nii.gz
        .../sub-01_ses-test_T1w.nii.gz --template mri_robust_template_out.mgz'

    """

    def _num_vols(self):
        n_files = len(self.inputs.in_files)
        if n_files != 1:
            return n_files

        img = nb.load(self.inputs.in_files[0])
        if len(img.shape) == 3:
            return 1

        return img.shape[3]

    @property
    def cmdline(self):
        if self._num_vols() == 1:
            return "echo Only one time point!"
        return super(StructuralReference, self).cmdline

    def _list_outputs(self):
        outputs = super(StructuralReference, self)._list_outputs()
        if self._num_vols() == 1:
            in_file = self.inputs.in_files[0]
            transform_file = outputs['transform_outputs'][0]
            outputs['out_file'] = in_file
            fs.utils.LTAConvert(in_lta='identity.nofile', source_file=in_file,
                                target_file=in_file, out_lta=transform_file).run()
        return outputs


class MakeMidthicknessInputSpec(fs.utils.MRIsExpandInputSpec):
    graymid = InputMultiPath(desc='Existing graymid/midthickness file')


class MakeMidthickness(fs.MRIsExpand):
    """ Variation on MRIsExpand that checks for an existing midthickness/graymid
    surface, and copies if available.

    mris_expand is an expensive operation, so this avoids re-running it when the
    working directory is lost.
    If users provide their own midthickness/graymid file, we assume they have
    created it correctly.
    """
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


class PatchedConcatenateLTA(ConcatenateLTA):
    """
    A temporarily patched version of ``fs.ConcatenateLTA`` to recover from
    `this bug <https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg55520.html>`_
    in FreeSurfer, that was
    `fixed here <https://github.com/freesurfer/freesurfer/pull/180>`_.

    The original FMRIPREP's issue is found
    `here <https://github.com/poldracklab/fmriprep/issues/768>`_.
    """

    def _list_outputs(self):
        outputs = super(ConcatenateLTA, self)._list_outputs()

        with open(outputs['out_file'], 'r') as f:
            lines = f.readlines()

        fixed = False
        newfile = []
        for line in lines:
            if line.startswith('filename = ') and len(line.strip("\n")) >= 255:
                fixed = True
                newfile.append('filename = path_too_long\n')
            else:
                newfile.append(line)

        if fixed:
            with open(outputs['out_file'], 'w') as f:
                f.write(''.join(newfile))
        return outputs


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
