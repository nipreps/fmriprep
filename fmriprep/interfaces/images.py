#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

import os
import numpy as np
import nibabel as nb
import nilearn.image as nli

from niworkflows.nipype import logging
from niworkflows.nipype.utils.filemanip import fname_presuffix
from niworkflows.nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, InputMultiPath, OutputMultiPath)
from niworkflows.nipype.interfaces import fsl
from niworkflows.nipype.interfaces.base import SimpleInterface

LOGGER = logging.getLogger('interface')


class IntraModalMergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input files')
    hmc = traits.Bool(True, usedefault=True)
    zero_based_avg = traits.Bool(True, usedefault=True)
    to_ras = traits.Bool(True, usedefault=True)


class IntraModalMergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='merged image')
    out_avg = File(exists=True, desc='average image')
    out_mats = OutputMultiPath(File(exists=True), desc='output matrices')
    out_movpar = OutputMultiPath(File(exists=True), desc='output movement parameters')


class IntraModalMerge(SimpleInterface):
    input_spec = IntraModalMergeInputSpec
    output_spec = IntraModalMergeOutputSpec

    def _run_interface(self, runtime):
        in_files = self.inputs.in_files
        if not isinstance(in_files, list):
            in_files = [self.inputs.in_files]

        # Generate output average name early
        self._results['out_avg'] = fname_presuffix(self.inputs.in_files[0],
                                                   suffix='_avg', newpath=runtime.cwd)

        if self.inputs.to_ras:
            in_files = [reorient(inf) for inf in in_files]

        if len(in_files) == 1:
            filenii = nb.load(in_files[0])
            filedata = filenii.get_data()

            # magnitude files can have an extra dimension empty
            if filedata.ndim == 5:
                sqdata = np.squeeze(filedata)
                if sqdata.ndim == 5:
                    raise RuntimeError('Input image (%s) is 5D' % in_files[0])
                else:
                    in_files = [fname_presuffix(in_files[0], suffix='_squeezed',
                                                newpath=runtime.cwd)]
                    nb.Nifti1Image(sqdata, filenii.get_affine(),
                                   filenii.get_header()).to_filename(in_files[0])

            if np.squeeze(nb.load(in_files[0]).get_data()).ndim < 4:
                self._results['out_file'] = in_files[0]
                self._results['out_avg'] = in_files[0]
                # TODO: generate identity out_mats and zero-filled out_movpar
                return runtime
            in_files = in_files[0]
        else:
            magmrg = fsl.Merge(dimension='t', in_files=self.inputs.in_files)
            in_files = magmrg.run().outputs.merged_file
        mcflirt = fsl.MCFLIRT(cost='normcorr', save_mats=True, save_plots=True,
                              ref_vol=0, in_file=in_files)
        mcres = mcflirt.run()
        self._results['out_mats'] = mcres.outputs.mat_file
        self._results['out_movpar'] = mcres.outputs.par_file
        self._results['out_file'] = mcres.outputs.out_file

        hmcnii = nb.load(mcres.outputs.out_file)
        hmcdat = hmcnii.get_data().mean(axis=3)
        if self.inputs.zero_based_avg:
            hmcdat -= hmcdat.min()

        nb.Nifti1Image(
            hmcdat, hmcnii.get_affine(), hmcnii.get_header()).to_filename(
            self._results['out_avg'])

        return runtime


CONFORMATION_TEMPLATE = """\t\t<h3 class="elem-title">Anatomical Conformation</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Input T1w images: {n_t1w}</li>
\t\t\t<li>Output orientation: RAS</li>
\t\t\t<li>Output dimensions: {dims}</li>
\t\t\t<li>Output voxel size: {zooms}</li>
\t\t\t<li>Discarded images: {n_discards}</li>
{discard_list}
\t\t</ul>
"""

DISCARD_TEMPLATE = """\t\t\t\t<li><abbr title="{path}">{basename}</abbr></li>"""


class TemplateDimensionsInputSpec(BaseInterfaceInputSpec):
    t1w_list = InputMultiPath(File(exists=True), mandatory=True, desc='input T1w images')
    max_scale = traits.Float(3.0, usedefault=True,
                             desc='Maximum scaling factor in images to accept')


class TemplateDimensionsOutputSpec(TraitedSpec):
    t1w_valid_list = OutputMultiPath(exists=True, desc='valid T1w images')
    target_zooms = traits.Tuple(traits.Float, traits.Float, traits.Float,
                                desc='Target zoom information')
    target_shape = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                desc='Target shape information')
    out_report = File(exists=True, desc='conformation report')


class TemplateDimensions(SimpleInterface):
    """
    Finds template target dimensions for a series of T1w images, filtering low-resolution images,
    if necessary.

    Along each axis, the minimum voxel size (zoom) and the maximum number of voxels (shape) are
    found across images.

    The ``max_scale`` parameter sets a bound on the degree of up-sampling performed.
    By default, an image with a voxel size greater than 3x the smallest voxel size
    (calculated separately for each dimension) will be discarded.

    To select images that require no scaling (i.e. all have smallest voxel sizes),
    set ``max_scale=1``.
    """
    input_spec = TemplateDimensionsInputSpec
    output_spec = TemplateDimensionsOutputSpec

    def _generate_segment(self, discards, dims, zooms):
        items = [DISCARD_TEMPLATE.format(path=path, basename=os.path.basename(path))
                 for path in discards]
        discard_list = '\n'.join(["\t\t\t<ul>"] + items + ['\t\t\t</ul>']) if items else ''
        zoom_fmt = '{:.02g}mm x {:.02g}mm x {:.02g}mm'.format(*zooms)
        return CONFORMATION_TEMPLATE.format(n_t1w=len(self.inputs.t1w_list),
                                            dims='x'.join(map(str, dims)),
                                            zooms=zoom_fmt,
                                            n_discards=len(discards),
                                            discard_list=discard_list)

    def _run_interface(self, runtime):
        # Load images, orient as RAS, collect shape and zoom data
        in_names = np.array(self.inputs.t1w_list)
        orig_imgs = np.vectorize(nb.load)(in_names)
        reoriented = np.vectorize(nb.as_closest_canonical)(orig_imgs)
        all_zooms = np.array([img.header.get_zooms()[:3] for img in reoriented])
        all_shapes = np.array([img.shape[:3] for img in reoriented])

        # Identify images that would require excessive up-sampling
        valid = np.ones(all_zooms.shape[0], dtype=bool)
        while valid.any():
            target_zooms = all_zooms[valid].min(axis=0)
            scales = all_zooms[valid] / target_zooms
            if np.all(scales < self.inputs.max_scale):
                break
            valid[valid] ^= np.any(scales == scales.max(), axis=1)

        # Ignore dropped images
        valid_fnames = in_names[valid]
        self._results['t1w_valid_list'] = valid_fnames.tolist()

        # Set target shape information
        target_zooms = all_zooms[valid].min(axis=0)
        target_shape = all_shapes[valid].max(axis=0)

        self._results['target_zooms'] = tuple(target_zooms.tolist())
        self._results['target_shape'] = tuple(target_shape.tolist())

        # Create report
        dropped_images = in_names[~valid]
        segment = self._generate_segment(dropped_images, target_shape, target_zooms)
        out_report = os.path.join(runtime.cwd, 'report.html')
        with open(out_report, 'w') as fobj:
            fobj.write(segment)

        self._results['out_report'] = out_report

        return runtime


class ConformInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Input T1w image')
    target_zooms = traits.Tuple(traits.Float, traits.Float, traits.Float,
                                desc='Target zoom information')
    target_shape = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                desc='Target shape information')


class ConformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Conformed T1w image')


class Conform(SimpleInterface):
    """Conform a series of T1w images to enable merging.

    Performs two basic functions:

    #. Orient to RAS (left-right, posterior-anterior, inferior-superior)
    #. Resample to target zooms (voxel sizes) and shape (number of voxels)
    """
    input_spec = ConformInputSpec
    output_spec = ConformOutputSpec

    def _run_interface(self, runtime):
        # Load image, orient as RAS
        fname = self.inputs.in_file
        orig_img = nb.load(fname)
        reoriented = nb.as_closest_canonical(orig_img)

        # Set target shape information
        target_zooms = np.array(self.inputs.target_zooms)
        target_shape = np.array(self.inputs.target_shape)
        target_span = target_shape * target_zooms

        zooms = np.array(reoriented.header.get_zooms()[:3])
        shape = np.array(reoriented.shape[:3])

        xyz_unit = reoriented.header.get_xyzt_units()[0]
        if xyz_unit == 'unknown':
            # Common assumption; if we're wrong, unlikely to be the only thing that breaks
            xyz_unit = 'mm'

        # Set a 0.05mm threshold to performing rescaling
        atol = {'meter': 1e-5, 'mm': 0.01, 'micron': 10}[xyz_unit]

        # Rescale => change zooms
        # Resize => update image dimensions
        rescale = not np.allclose(zooms, target_zooms, atol=atol)
        resize = not np.all(shape == target_shape)
        if rescale or resize:
            target_affine = np.eye(4, dtype=reoriented.affine.dtype)
            if rescale:
                scale_factor = target_zooms / zooms
                target_affine[:3, :3] = reoriented.affine[:3, :3].dot(np.diag(scale_factor))
            else:
                target_affine[:3, :3] = reoriented.affine[:3, :3]

            if resize:
                # The shift is applied after scaling.
                # Use a proportional shift to maintain relative position in dataset
                size_factor = target_span / (zooms * shape)
                # Use integer shifts to avoid unnecessary interpolation
                offset = (reoriented.affine[:3, 3] * size_factor - reoriented.affine[:3, 3])
                target_affine[:3, 3] = reoriented.affine[:3, 3] + offset.astype(int)
            else:
                target_affine[:3, 3] = reoriented.affine[:3, 3]

            data = nli.resample_img(reoriented, target_affine, target_shape).get_data()
            reoriented = reoriented.__class__(data, target_affine, reoriented.header)

        # Image may be reoriented, rescaled, and/or resized
        if reoriented is not orig_img:
            out_name = fname_presuffix(fname, suffix='_ras', newpath=runtime.cwd)
            reoriented.to_filename(out_name)
        else:
            out_name = fname

        self._results['out_file'] = out_name

        return runtime


class ReorientInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='Input T1w image')


class ReorientOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Reoriented T1w image')


class Reorient(SimpleInterface):
    """Reorient a T1w image to RAS (left-right, posterior-anterior, inferior-superior)"""
    input_spec = ReorientInputSpec
    output_spec = ReorientOutputSpec

    def _run_interface(self, runtime):
        # Load image, orient as RAS
        fname = self.inputs.in_file
        orig_img = nb.load(fname)
        reoriented = nb.as_closest_canonical(orig_img)

        # Image may be reoriented
        if reoriented is not orig_img:
            out_name = fname_presuffix(fname, suffix='_ras', newpath=runtime.cwd)
            reoriented.to_filename(out_name)
        else:
            out_name = fname

        self._results['out_file'] = out_name

        return runtime


class ValidateImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input image')


class ValidateImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='validated image')
    out_report = File(exists=True, desc='HTML segment containing warning')


class ValidateImage(SimpleInterface):
    input_spec = ValidateImageInputSpec
    output_spec = ValidateImageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        out_report = os.path.abspath('report.html')

        qform_code = img.header._structarr['qform_code']
        sform_code = img.header._structarr['sform_code']

        # Valid affine information
        if (qform_code, sform_code) != (0, 0):
            self._results['out_file'] = self.inputs.in_file
            open(out_report, 'w').close()
            self._results['out_report'] = out_report
            return runtime

        out_fname = fname_presuffix(self.inputs.in_file, suffix='_valid', newpath=runtime.cwd)

        # Nibabel derives a default LAS affine from the shape and zooms
        # Use scanner xform code to indicate no alignment has been done
        img.set_sform(img.affine, nb.nifti1.xform_codes['scanner'])

        img.to_filename(out_fname)
        self._results['out_file'] = out_fname

        snippet = (r'<h3 class="elem-title">WARNING - Invalid header</h3>',
                   r'<p class="elem-desc">Input file does not have valid qform or sform matrix.',
                   r'A default, LAS-oriented affine has been constructed.',
                   r'A left-right flip may have occurred.',
                   r'Analyses of this dataset MAY BE INVALID.</p>')

        with open(out_report, 'w') as fobj:
            fobj.write('\n'.join('\t' * 3 + line for line in snippet))
            fobj.write('\n')

        self._results['out_report'] = out_report
        return runtime


class InvertT1wInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='Skull-stripped T1w structural image')
    ref_file = File(exists=True, mandatory=True,
                    desc='Skull-stripped reference image')


class InvertT1wOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Inverted T1w structural image')


class InvertT1w(SimpleInterface):
    input_spec = InvertT1wInputSpec
    output_spec = InvertT1wOutputSpec

    def _run_interface(self, runtime):
        t1_img = nli.load_img(self.inputs.in_file)
        t1_data = t1_img.get_data()
        epi_data = nli.load_img(self.inputs.ref_file).get_data()

        # We assume the image is already masked
        mask = t1_data > 0

        t1_min, t1_max = np.unique(t1_data)[[1, -1]]
        epi_min, epi_max = np.unique(epi_data)[[1, -1]]
        scale_factor = (epi_max - epi_min) / (t1_max - t1_min)

        inv_data = mask * ((t1_max - t1_data) * scale_factor + epi_min)

        out_file = fname_presuffix(self.inputs.in_file, suffix='_inv', newpath=runtime.cwd)
        nli.new_img_like(t1_img, inv_data, copy_header=True).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


def reorient(in_file, out_file=None):
    """Reorient Nifti files to RAS"""
    if out_file is None:
        out_file = fname_presuffix(in_file, suffix='_ras', newpath=os.getcwd())
    nb.as_closest_canonical(nb.load(in_file)).to_filename(out_file)
    return out_file


def _flatten_split_merge(in_files):
    if isinstance(in_files, str):
        in_files = [in_files]

    nfiles = len(in_files)

    all_nii = []
    for fname in in_files:
        nii = nb.squeeze_image(nb.load(fname))

        if nii.get_data().ndim > 3:
            all_nii += nb.four_to_three(nii)
        else:
            all_nii.append(nii)

    if len(all_nii) == 1:
        LOGGER.warning('File %s cannot be split', all_nii[0])
        return in_files[0], in_files

    if len(all_nii) == nfiles:
        flat_split = in_files
    else:
        splitname = fname_presuffix(in_files[0], suffix='_split%04d', newpath=os.getcwd())
        flat_split = []
        for i, nii in enumerate(all_nii):
            flat_split.append(splitname % i)
            nii.to_filename(flat_split[-1])

    # Only one 4D file was supplied
    if nfiles == 1:
        merged = in_files[0]
    else:
        # More that one in_files - need merge
        merged = fname_presuffix(in_files[0], suffix='_merged', newpath=os.getcwd())
        nb.concat_images(all_nii).to_filename(merged)

    return merged, flat_split


def extract_wm(in_seg, wm_label=3):
    import os.path as op
    import nibabel as nb
    import numpy as np

    nii = nb.load(in_seg)
    data = np.zeros(nii.shape, dtype=np.uint8)
    data[nii.get_data() == wm_label] = 1
    hdr = nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    nb.Nifti1Image(data, nii.affine, hdr).to_filename('wm.nii.gz')
    return op.abspath('wm.nii.gz')
