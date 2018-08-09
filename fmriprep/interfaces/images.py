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
from textwrap import indent

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, SimpleInterface,
    File, InputMultiPath, OutputMultiPath)
from nipype.interfaces import fsl

LOGGER = logging.getLogger('nipype.interface')


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
            in_files = [reorient(inf, newpath=runtime.cwd)
                        for inf in in_files]

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
        valid_fnames = np.atleast_1d(in_names[valid]).tolist()
        self._results['t1w_valid_list'] = valid_fnames

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
    in_file = File(exists=True, mandatory=True, desc='Input image')
    target_zooms = traits.Tuple(traits.Float, traits.Float, traits.Float,
                                desc='Target zoom information')
    target_shape = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                desc='Target shape information')


class ConformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Conformed image')
    transform = File(exists=True, desc='Conformation transform')


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

        # Reconstruct transform from orig to reoriented image
        ornt_xfm = nb.orientations.inv_ornt_aff(
            nb.io_orientation(orig_img.affine), orig_img.shape)
        # Identity unless proven otherwise
        target_affine = reoriented.affine.copy()
        conform_xfm = np.eye(4)

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
            if rescale:
                scale_factor = target_zooms / zooms
                target_affine[:3, :3] = reoriented.affine[:3, :3].dot(np.diag(scale_factor))

            if resize:
                # The shift is applied after scaling.
                # Use a proportional shift to maintain relative position in dataset
                size_factor = target_span / (zooms * shape)
                # Use integer shifts to avoid unnecessary interpolation
                offset = (reoriented.affine[:3, 3] * size_factor - reoriented.affine[:3, 3])
                target_affine[:3, 3] = reoriented.affine[:3, 3] + offset.astype(int)

            data = nli.resample_img(reoriented, target_affine, target_shape).get_data()
            conform_xfm = np.linalg.inv(reoriented.affine).dot(target_affine)
            reoriented = reoriented.__class__(data, target_affine, reoriented.header)

        # Image may be reoriented, rescaled, and/or resized
        if reoriented is not orig_img:
            out_name = fname_presuffix(fname, suffix='_ras', newpath=runtime.cwd)
            reoriented.to_filename(out_name)
        else:
            out_name = fname

        transform = ornt_xfm.dot(conform_xfm)
        assert np.allclose(orig_img.affine.dot(transform), target_affine)

        mat_name = fname_presuffix(fname, suffix='.mat', newpath=runtime.cwd, use_ext=False)
        np.savetxt(mat_name, transform, fmt='%.08f')

        self._results['out_file'] = out_name
        self._results['transform'] = mat_name

        return runtime


class ValidateImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input image')


class ValidateImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='validated image')
    out_report = File(exists=True, desc='HTML segment containing warning')


class ValidateImage(SimpleInterface):
    """
    Check the correctness of x-form headers (matrix and code)

    This interface implements the `following logic
    <https://github.com/poldracklab/fmriprep/issues/873#issuecomment-349394544>`_:

    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | valid quaternions | `qform_code > 0` | `sform_code > 0` | `qform == sform` \
| actions                                        |
    +===================+==================+==================+==================\
+================================================+
    | True              | True             | True             | True             \
| None                                           |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | True              | True             | False            | *                \
| sform, scode <- qform, qcode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | *                | True             | False            \
| qform, qcode <- sform, scode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | False            | True             | *                \
| qform, qcode <- sform, scode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | False            | False            | *                \
| sform, qform <- best affine; scode, qcode <- 1 |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | False             | *                | False            | *                \
| sform, qform <- best affine; scode, qcode <- 1 |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    """
    input_spec = ValidateImageInputSpec
    output_spec = ValidateImageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        out_report = os.path.join(runtime.cwd, 'report.html')

        # Retrieve xform codes
        sform_code = int(img.header._structarr['sform_code'])
        qform_code = int(img.header._structarr['qform_code'])

        # Check qform is valid
        valid_qform = False
        try:
            qform = img.get_qform()
            valid_qform = True
        except ValueError:
            pass

        sform = img.get_sform()
        if np.linalg.det(sform) == 0:
            valid_sform = False
        else:
            RZS = sform[:3, :3]
            zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
            valid_sform = np.allclose(zooms, img.header.get_zooms()[:3])

        # Matching affines
        matching_affines = valid_qform and np.allclose(qform, sform)

        # Both match, qform valid (implicit with match), codes okay -> do nothing, empty report
        if matching_affines and qform_code > 0 and sform_code > 0:
            self._results['out_file'] = self.inputs.in_file
            open(out_report, 'w').close()
            self._results['out_report'] = out_report
            return runtime

        # A new file will be written
        out_fname = fname_presuffix(self.inputs.in_file, suffix='_valid', newpath=runtime.cwd)
        self._results['out_file'] = out_fname

        # Row 2:
        if valid_qform and qform_code > 0 and (sform_code == 0 or not valid_sform):
            img.set_sform(qform, qform_code)
            warning_txt = 'Note on orientation: sform matrix set'
            description = """\
<p class="elem-desc">The sform has been copied from qform.</p>
"""
        # Rows 3-4:
        # Note: if qform is not valid, matching_affines is False
        elif (valid_sform and sform_code > 0) and (not matching_affines or qform_code == 0):
            img.set_qform(img.get_sform(), sform_code)
            warning_txt = 'Note on orientation: qform matrix overwritten'
            description = """\
<p class="elem-desc">The qform has been copied from sform.</p>
"""
            if not valid_qform and qform_code > 0:
                warning_txt = 'WARNING - Invalid qform information'
                description = """\
<p class="elem-desc">
    The qform matrix found in the file header is invalid.
    The qform has been copied from sform.
    Checking the original qform information from the data produced
    by the scanner is advised.
</p>
"""
        # Rows 5-6:
        else:
            affine = img.header.get_base_affine()
            img.set_sform(affine, nb.nifti1.xform_codes['scanner'])
            img.set_qform(affine, nb.nifti1.xform_codes['scanner'])
            warning_txt = 'WARNING - Missing orientation information'
            description = """\
<p class="elem-desc">
    FMRIPREP could not retrieve orientation information from the image header.
    The qform and sform matrices have been set to a default, LAS-oriented affine.
    Analyses of this dataset MAY BE INVALID.
</p>
"""
        snippet = '<h3 class="elem-title">%s</h3>\n%s\n' % (warning_txt, description)
        # Store new file and report
        img.to_filename(out_fname)
        with open(out_report, 'w') as fobj:
            fobj.write(indent(snippet, '\t' * 3))

        self._results['out_report'] = out_report
        return runtime


class DemeanImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='image to be demeaned')
    in_mask = File(exists=True, mandatory=True,
                   desc='mask where median will be calculated')
    only_mask = traits.Bool(False, usedefault=True,
                            desc='demean only within mask')


class DemeanImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='demeaned image')


class DemeanImage(SimpleInterface):
    input_spec = DemeanImageInputSpec
    output_spec = DemeanImageOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = demean(
            self.inputs.in_file,
            self.inputs.in_mask,
            only_mask=self.inputs.only_mask,
            newpath=runtime.cwd)
        return runtime


class FilledImageLikeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='image to be demeaned')
    fill_value = traits.Float(1.0, usedefault=True,
                              desc='value to fill')
    dtype = traits.Enum('float32', 'uint8', usedefault=True,
                        desc='force output data type')


class FilledImageLikeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='demeaned image')


class FilledImageLike(SimpleInterface):
    input_spec = FilledImageLikeInputSpec
    output_spec = FilledImageLikeOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = nii_ones_like(
            self.inputs.in_file,
            self.inputs.fill_value,
            self.inputs.dtype,
            newpath=runtime.cwd)
        return runtime


def reorient(in_file, newpath=None):
    """Reorient Nifti files to RAS"""
    out_file = fname_presuffix(in_file, suffix='_ras', newpath=newpath)
    nb.as_closest_canonical(nb.load(in_file)).to_filename(out_file)
    return out_file


def extract_wm(in_seg, wm_label=3, newpath=None):
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_seg)
    data = np.zeros(nii.shape, dtype=np.uint8)
    data[nii.get_data() == wm_label] = 1

    out_file = fname_presuffix(in_seg, suffix='_wm', newpath=newpath)
    new = nb.Nifti1Image(data, nii.affine, nii.header)
    new.set_data_dtype(np.uint8)
    new.to_filename(out_file)
    return out_file


def normalize_xform(img):
    """ Set identical, valid qform and sform matrices in an image

    Selects the best available affine (sform > qform > shape-based), and
    coerces it to be qform-compatible (no shears).

    The resulting image represents this same affine as both qform and sform,
    and is marked as NIFTI_XFORM_ALIGNED_ANAT, indicating that it is valid,
    not aligned to template, and not necessarily preserving the original
    coordinates.

    If header would be unchanged, returns input image.
    """
    # Let nibabel convert from affine to quaternions, and recover xform
    tmp_header = img.header.copy()
    tmp_header.set_qform(img.affine)
    xform = tmp_header.get_qform()
    xform_code = 2

    # Check desired codes
    qform, qform_code = img.get_qform(coded=True)
    sform, sform_code = img.get_sform(coded=True)
    if all((qform is not None and np.allclose(qform, xform),
            sform is not None and np.allclose(sform, xform),
            int(qform_code) == xform_code, int(sform_code) == xform_code)):
        return img

    new_img = img.__class__(img.get_data(), xform, img.header)
    # Unconditionally set sform/qform
    new_img.set_sform(xform, xform_code)
    new_img.set_qform(xform, xform_code)
    return new_img


def demean(in_file, in_mask, only_mask=False, newpath=None):
    """Demean ``in_file`` within the mask defined by ``in_mask``"""
    import os
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(in_file, suffix='_demeaned',
                               newpath=os.getcwd())
    nii = nb.load(in_file)
    msk = nb.load(in_mask).get_data()
    data = nii.get_data()
    if only_mask:
        data[msk > 0] -= np.median(data[msk > 0])
    else:
        data -= np.median(data[msk > 0])
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(
        out_file)
    return out_file


def nii_ones_like(in_file, value, dtype, newpath=None):
    """Create a NIfTI file filled with ``value``, matching properties of ``in_file``"""
    import os
    import numpy as np
    import nibabel as nb

    nii = nb.load(in_file)
    data = np.ones(nii.shape, dtype=float) * value

    out_file = os.path.join(newpath or os.getcwd(), "filled.nii.gz")
    nii = nb.Nifti1Image(data, nii.affine, nii.header)
    nii.set_data_dtype(dtype)
    nii.to_filename(out_file)

    return out_file


class SignalExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='4-D fMRI nii file')
    label_files = InputMultiPath(
        File(exists=True),
        mandatory=True,
        desc='a 3-D label image, with 0 denoting '
        'background, or a list of 3-D probability '
        'maps (one per label) or the equivalent 4D '
        'file.')
    class_labels = traits.List(
        mandatory=True,
        desc='Human-readable labels for each segment '
        'in the label file, in order. The length of '
        'class_labels must be equal to the number of '
        'segments (background excluded). This list '
        'corresponds to the class labels in label_file '
        'in ascending order')
    out_file = File(
        'signals.tsv',
        usedefault=True,
        exists=False,
        desc='The name of the file to output to. '
        'signals.tsv by default')


class SignalExtractionOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='tsv file containing the computed '
        'signals, with as many columns as there are labels and as '
        'many rows as there are timepoints in in_file, plus a '
        'header row with values from class_labels')


class SignalExtraction(SimpleInterface):
    """ Extract mean signals from a time series within a set of ROIs

    This interface is intended to be a memory-efficient alternative to
    nipype.interfaces.nilearn.SignalExtraction.
    Not all features of nilearn.SignalExtraction are implemented at
    this time.
    """
    input_spec = SignalExtractionInputSpec
    output_spec = SignalExtractionOutputSpec

    def _run_interface(self, runtime):
        mask_imgs = [nb.load(fname) for fname in self.inputs.label_files]
        if len(mask_imgs) == 1:
            mask_imgs = nb.four_to_three(mask_imgs[0])

        masks = [mask_img.get_data().astype(np.bool) for mask_img in mask_imgs]

        n_masks = len(masks)

        if n_masks != len(self.inputs.class_labels):
            raise ValueError("Number of masks must match number of labels")

        img = nb.load(self.inputs.in_file)

        series = np.zeros((img.shape[3], n_masks))

        data = img.get_data()
        for j in range(n_masks):
            series[:, j] = data[masks[j], :].mean(axis=0)

        output = np.vstack((self.inputs.class_labels, series.astype(str)))
        self._results['out_file'] = os.path.join(runtime.cwd,
                                                 self.inputs.out_file)
        np.savetxt(
            self._results['out_file'], output, fmt=b'%s', delimiter='\t')

        return runtime
