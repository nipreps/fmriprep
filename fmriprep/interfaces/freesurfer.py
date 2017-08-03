#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FreeSurfer tools interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import os.path as op
import nibabel as nb

from nilearn.image import resample_to_img, new_img_like

from niworkflows.nipype.utils.filemanip import copyfile, filename_to_list, fname_presuffix
from niworkflows.nipype.interfaces.base import (
    traits, isdefined, InputMultiPath, File, Directory,
    BaseInterfaceInputSpec, TraitedSpec, CommandLineInputSpec, CommandLine
)
from niworkflows.nipype.interfaces import freesurfer as fs

from niworkflows.interfaces.base import SimpleInterface


class StructuralReferenceInputSpec(fs.longitudinal.RobustTemplateInputSpec):
    transform_outputs = traits.Either(
        True, InputMultiPath(File(exists=False)),
        argstr='--lta %s', desc='output xforms to template (for each input)')


class StructuralReference(fs.RobustTemplate):
    """ Variation on RobustTemplate that simply copies the source if a single
    volume is provided. """
    input_spec = StructuralReferenceInputSpec

    @property
    def cmdline(self):
        cmd = super(StructuralReference, self).cmdline
        if len(self.inputs.in_files) > 1:
            return cmd

        img = nb.load(self.inputs.in_files[0])
        if len(img.shape) > 3 and img.shape[3] > 1:
            return cmd

        in_file = self.inputs.in_files[0]
        out_file = self._list_outputs()['out_file']
        copyfile(in_file, out_file)

        if isdefined(self.inputs.transform_outputs):
            transform_outputs = self._list_outputs()['transform_outputs']
            lta = LTAConvert(in_lta='identity.nofile', source=in_file,
                             target=out_file, out_lta=transform_outputs[0])
            lta.run()

        return "echo Only one time point!"

    def _format_arg(self, name, spec, value):
        if name == 'transform_outputs':
            value = self._list_outputs()['transform_outputs']
        return super(StructuralReference, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(
            self.inputs.out_file)
        if isdefined(self.inputs.transform_outputs):
            fnames = self.inputs.transform_outputs
            if fnames is True:
                fnames = [fname_presuffix(in_file, suffix='.lta', newpath=os.getcwd(),
                                          use_ext=False)
                          for in_file in self.inputs.in_files]
            outputs['transform_outputs'] = list(map(os.path.abspath, fnames))
        if isdefined(self.inputs.scaled_intensity_outputs):
            outputs['scaled_intensity_outputs'] = [os.path.abspath(
                x) for x in self.inputs.scaled_intensity_outputs]
        return outputs


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


class LTAConvertInputSpec(CommandLineInputSpec):
    # Inputs
    _in_xor = ('in_lta', 'in_fsl', 'in_mni', 'in_reg', 'in_niftyreg')
    in_lta = traits.Either(File(exists=True), 'identity.nofile', argstr='--inlta %s',
                           xor=_in_xor, desc='input transform of LTA type')
    in_fsl = File(exists=True, argstr='--infsl %s',
                  xor=_in_xor, desc='input transform of FSL type')
    in_mni = File(exists=True, argstr='--inmni %s',
                  xor=_in_xor, desc='input transform of MNI/XFM type')
    in_reg = File(exists=True, argstr='--inreg %s',
                  xor=_in_xor, desc='input transform of TK REG type (deprecated format)')
    in_niftyreg = File(exists=True, argstr='--inniftyreg %s',
                       xor=_in_xor, desc='input transform of Nifty Reg type (inverse RAS2RAS)')
    # Outputs
    out_lta = traits.Either(traits.Bool, File, argstr='--outlta %s',
                            desc='output linear transform (LTA Freesurfer format)')
    out_fsl = traits.Either(traits.Bool, File, argstr='--outfsl %s',
                            desc='output transform in FSL format')
    out_mni = traits.Either(traits.Bool, File, argstr='--outmni %s',
                            desc='output transform in MNI/XFM format')
    out_reg = traits.Either(traits.Bool, File, argstr='--outreg %s',
                            desc='output transform in reg dat format')
    # Optional flags
    invert = traits.Bool(argstr='--invert')
    ltavox2vox = traits.Bool(argstr='--ltavox2vox', requires=['out_lta'])
    source_file = File(exists=True, argstr='--src %s')
    target_file = File(exists=True, argstr='--trg %s')
    target_conform = traits.Bool(argstr='--trgconform')


class LTAConvertOutputSpec(TraitedSpec):
    out_lta = File(exists=True, desc='output linear transform (LTA Freesurfer format)')
    out_fsl = File(exists=True, desc='output transform in FSL format')
    out_mni = File(exists=True, desc='output transform in MNI/XFM format')
    out_reg = File(exists=True, desc='output transform in reg dat format')


class LTAConvert(CommandLine):
    input_spec = LTAConvertInputSpec
    output_spec = LTAConvertOutputSpec
    _cmd = 'lta_convert'

    def _format_arg(self, name, spec, value):
        if name.startswith('out_') and value is True:
            value = self._list_outputs()[name]
        return super(LTAConvert, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.out_lta:
            fname = 'out.lta' if self.inputs.out_lta is True else self.inputs.out_lta
            outputs['out_lta'] = os.path.abspath(fname)
        if self.inputs.out_fsl:
            fname = 'out.mat' if self.inputs.out_fsl is True else self.inputs.out_fsl
            outputs['out_fsl'] = os.path.abspath(fname)
        if self.inputs.out_mni:
            fname = 'out.xfm' if self.inputs.out_mni is True else self.inputs.out_mni
            outputs['out_mni'] = os.path.abspath(fname)
        if self.inputs.out_reg:
            fname = 'out.dat' if self.inputs.out_reg is True else self.inputs.out_reg
            outputs['out_reg'] = os.path.abspath(fname)
        return outputs


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
