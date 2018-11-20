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

    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')

"""

import os.path as op
import nibabel as nb
import numpy as np

from skimage import morphology as sim
from scipy.ndimage.morphology import binary_fill_holes
from nilearn.image import resample_to_img, new_img_like

from nipype.utils.filemanip import copyfile, filename_to_list, fname_presuffix
from nipype.interfaces.base import (
    isdefined, InputMultiPath, BaseInterfaceInputSpec, TraitedSpec, File, traits, Directory
)
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import SimpleInterface
from nipype.interfaces.freesurfer.preprocess import ConcatenateLTA, RobustRegister
from nipype.interfaces.freesurfer.utils import LTAConvert
from niworkflows.interfaces.registration import BBRegisterRPT, MRICoregRPT


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
            outputs['out_file'] = in_file
            if isdefined(outputs['transform_outputs']):
                transform_file = outputs['transform_outputs'][0]
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
    _always_run = True

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
    flair_list = InputMultiPath(File(exists=True), desc='input file, part of a BIDS tree')
    hires_enabled = traits.Bool(True, usedefault=True, desc='enable hi-resolution processing')


class FSDetectInputsOutputSpec(TraitedSpec):
    t2w = File(desc='reference T2w image')
    use_t2w = traits.Bool(desc='enable use of T2w downstream computation')
    flair = File(desc='reference FLAIR image')
    use_flair = traits.Bool(desc='enable use of FLAIR downstream computation')
    hires = traits.Bool(desc='enable hi-res processing')
    mris_inflate = traits.Str(desc='mris_inflate argument')


class FSDetectInputs(SimpleInterface):
    input_spec = FSDetectInputsInputSpec
    output_spec = FSDetectInputsOutputSpec

    def _run_interface(self, runtime):
        t2w, flair, self._results['hires'], mris_inflate = detect_inputs(
            self.inputs.t1w_list,
            t2w_list=self.inputs.t2w_list if isdefined(self.inputs.t2w_list) else None,
            flair_list=self.inputs.flair_list if isdefined(self.inputs.flair_list) else None,
            hires_enabled=self.inputs.hires_enabled)

        self._results['use_t2w'] = t2w is not None
        if self._results['use_t2w']:
            self._results['t2w'] = t2w

        self._results['use_flair'] = flair is not None
        if self._results['use_flair']:
            self._results['flair'] = flair

        if self._results['hires']:
            self._results['mris_inflate'] = mris_inflate

        return runtime


class TruncateLTA(object):
    """Mixin to ensure that LTA files do not store overly long paths,
    which lead to segmentation faults when read by FreeSurfer tools.

    See the following issues for discussion:

    * https://github.com/freesurfer/freesurfer/pull/180
    * https://github.com/poldracklab/fmriprep/issues/768
    * https://github.com/poldracklab/fmriprep/pull/778
    * https://github.com/poldracklab/fmriprep/issues/1268
    * https://github.com/poldracklab/fmriprep/pull/1274
    """

    # Use a tuple in case some object produces multiple transforms
    lta_outputs = ('out_lta_file',)

    def _post_run_hook(self, runtime):

        outputs = self._list_outputs()

        for lta_name in self.lta_outputs:
            lta_file = outputs[lta_name]
            if not isdefined(lta_file):
                continue

            with open(lta_file, 'r') as f:
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
                with open(lta_file, 'w') as f:
                    f.write(''.join(newfile))

        runtime = super(TruncateLTA, self)._post_run_hook(runtime)

        return runtime


class PatchedConcatenateLTA(TruncateLTA, ConcatenateLTA):
    """
    A temporarily patched version of ``fs.ConcatenateLTA`` to recover from
    `this bug <https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg55520.html>`_
    in FreeSurfer, that was
    `fixed here <https://github.com/freesurfer/freesurfer/pull/180>`__.

    The original FMRIPREP's issue is found
    `here <https://github.com/poldracklab/fmriprep/issues/768>`__.

    the fix is now done through mixin with TruncateLTA
    """
    lta_outputs = ['out_file']


class PatchedLTAConvert(TruncateLTA, LTAConvert):
    """
    LTAconvert is producing a lta file refer as out_lta
    truncate filename through mixin TruncateLTA
    """
    lta_outputs = ('out_lta',)


class PatchedBBRegisterRPT(TruncateLTA, BBRegisterRPT):
    pass


class PatchedMRICoregRPT(TruncateLTA, MRICoregRPT):
    pass


class PatchedRobustRegister(TruncateLTA, RobustRegister):
    lta_outputs = ('out_reg_file', 'half_source_xfm', 'half_targ_xfm')


class RefineBrainMaskInputSpec(BaseInterfaceInputSpec):
    in_anat = File(exists=True, mandatory=True,
                   desc='input anatomical reference (INU corrected)')
    in_aseg = File(exists=True, mandatory=True,
                   desc='input ``aseg`` file, in NifTi format.')
    in_ants = File(exists=True, mandatory=True,
                   desc='brain tissue segmentation generated with antsBrainExtraction.sh')


class RefineBrainMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='new mask')


class RefineBrainMask(SimpleInterface):
    """
    Refine the brain mask implicit in the ``aseg.mgz``
    file to include possibly missing gray-matter voxels
    and deep, wide sulci.
    """

    input_spec = RefineBrainMaskInputSpec
    output_spec = RefineBrainMaskOutputSpec

    def _run_interface(self, runtime):

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_anat, suffix='_rbrainmask', newpath=runtime.cwd)

        anatnii = nb.load(self.inputs.in_anat)
        msknii = nb.Nifti1Image(
            grow_mask(anatnii.get_data(),
                      nb.load(self.inputs.in_aseg).get_data(),
                      nb.load(self.inputs.in_ants).get_data()),
            anatnii.affine,
            anatnii.header
        )
        msknii.set_data_dtype(np.uint8)
        msknii.to_filename(self._results['out_file'])

        return runtime


class MedialNaNsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input surface file')
    target_subject = traits.Str(mandatory=True, desc='target subject ID')
    subjects_dir = Directory(mandatory=True, desc='FreeSurfer SUBJECTS_DIR')


class MedialNaNsOutputSpec(TraitedSpec):
    out_file = File(desc='the output surface file')


class MedialNaNs(SimpleInterface):
    """
    The MedialNaNs converts from arbitrary units to rad/s
    """
    input_spec = MedialNaNsInputSpec
    output_spec = MedialNaNsOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = medial_wall_to_nan(
            self.inputs.in_file,
            self.inputs.subjects_dir,
            self.inputs.target_subject,
            newpath=runtime.cwd)
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


def detect_inputs(t1w_list, t2w_list=None, flair_list=None, hires_enabled=True):
    t1w_list = filename_to_list(t1w_list)
    t2w_list = filename_to_list(t2w_list) if t2w_list is not None else []
    flair_list = filename_to_list(flair_list) if flair_list is not None else []
    t1w_ref = nb.load(t1w_list[0])
    # Use high resolution preprocessing if voxel size < 1.0mm
    # Tolerance of 0.05mm requires that rounds down to 0.9mm or lower
    hires = hires_enabled and max(t1w_ref.header.get_zooms()) < 1 - 0.05

    t2w = None
    if t2w_list and max(nb.load(t2w_list[0]).header.get_zooms()) < 1.2:
        t2w = t2w_list[0]

    # Prefer T2w to FLAIR if both present and T2w satisfies
    flair = None
    if flair_list and not t2w and max(nb.load(flair_list[0]).header.get_zooms()) < 1.2:
        flair = flair_list[0]

    # https://surfer.nmr.mgh.harvard.edu/fswiki/SubmillimeterRecon
    mris_inflate = '-n 50' if hires else None
    return (t2w, flair, hires, mris_inflate)


def refine_aseg(aseg, ball_size=4):
    """
    First step to reconcile ANTs' and FreeSurfer's brain masks.

    Here, the ``aseg.mgz`` mask from FreeSurfer is refined in two
    steps, using binary morphological operations:

      1. With a binary closing operation the sulci are included
         into the mask. This results in a smoother brain mask
         that does not exclude deep, wide sulci.

      2. Fill any holes (typically, there could be a hole next to
         the pineal gland and the corpora quadrigemina if the great
         cerebral brain is segmented out).


    """
    # Read aseg data
    bmask = aseg.copy()
    bmask[bmask > 0] = 1
    bmask = bmask.astype(np.uint8)

    # Morphological operations
    selem = sim.ball(ball_size)
    newmask = sim.binary_closing(bmask, selem)
    newmask = binary_fill_holes(newmask.astype(np.uint8), selem).astype(np.uint8)

    return newmask.astype(np.uint8)


def grow_mask(anat, aseg, ants_segs=None, ww=7, zval=2.0, bw=4):
    """
    Grow mask including pixels that have a high likelihood.
    GM tissue parameters are sampled in image patches of ``ww`` size.

    This is inspired on mindboggle's solution to the problem:
    https://github.com/nipy/mindboggle/blob/master/mindboggle/guts/segment.py#L1660

    """
    selem = sim.ball(bw)

    if ants_segs is None:
        ants_segs = np.zeros_like(aseg, dtype=np.uint8)

    aseg[aseg == 42] = 3  # Collapse both hemispheres
    gm = anat.copy()
    gm[aseg != 3] = 0

    refined = refine_aseg(aseg)
    newrefmask = sim.binary_dilation(refined, selem) - refined
    indices = np.argwhere(newrefmask > 0)
    for pixel in indices:
        # When ATROPOS identified the pixel as GM, set and carry on
        if ants_segs[tuple(pixel)] == 2:
            refined[tuple(pixel)] = 1
            continue

        window = gm[
            pixel[0] - ww:pixel[0] + ww,
            pixel[1] - ww:pixel[1] + ww,
            pixel[2] - ww:pixel[2] + ww
        ]
        if np.any(window > 0):
            mu = window[window > 0].mean()
            sigma = max(window[window > 0].std(), 1.e-5)
            zstat = abs(anat[tuple(pixel)] - mu) / sigma
            refined[tuple(pixel)] = int(zstat < zval)

    refined = sim.binary_opening(refined, selem)
    return refined


def medial_wall_to_nan(in_file, subjects_dir, target_subject, newpath=None):
    """ Convert values on medial wall to NaNs
    """
    import nibabel as nb
    import numpy as np
    import os

    fn = os.path.basename(in_file)
    if not target_subject.startswith('fs'):
        return in_file

    cortex = nb.freesurfer.read_label(os.path.join(
        subjects_dir, target_subject, 'label', '{}.cortex.label'.format(fn[:2])))
    func = nb.load(in_file)
    medial = np.delete(np.arange(len(func.darrays[0].data)), cortex)
    for darray in func.darrays:
        darray.data[medial] = np.nan

    out_file = os.path.join(newpath or os.getcwd(), fn)
    func.to_filename(out_file)
    return out_file
