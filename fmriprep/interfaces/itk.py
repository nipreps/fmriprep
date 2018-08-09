#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
import os
from mimetypes import guess_type
from tempfile import TemporaryDirectory
import numpy as np
import nibabel as nb

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiPath, OutputMultiPath,
    SimpleInterface)
from nipype.interfaces.ants.resampling import ApplyTransformsInputSpec

LOGGER = logging.getLogger('nipype.interface')


class MCFLIRT2ITKInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='list of MAT files from MCFLIRT')
    in_reference = File(exists=True, mandatory=True,
                        desc='input image for spatial reference')
    in_source = File(exists=True, mandatory=True,
                     desc='input image for spatial source')
    num_threads = traits.Int(1, usedefault=True, nohash=True,
                             desc='number of parallel processes')


class MCFLIRT2ITKOutputSpec(TraitedSpec):
    out_file = File(desc='the output ITKTransform file')


class MCFLIRT2ITK(SimpleInterface):

    """
    Convert a list of MAT files from MCFLIRT into an ITK Transform file.
    """
    input_spec = MCFLIRT2ITKInputSpec
    output_spec = MCFLIRT2ITKOutputSpec

    def _run_interface(self, runtime):
        num_threads = self.inputs.num_threads
        if num_threads < 1:
            num_threads = None

        with TemporaryDirectory(prefix='tmp-', dir=runtime.cwd) as tmp_folder:
            # Inputs are ready to run in parallel
            if num_threads is None or num_threads > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=num_threads) as pool:
                    itk_outs = list(pool.map(_mat2itk, [
                        (in_mat, self.inputs.in_reference, self.inputs.in_source, i, tmp_folder)
                        for i, in_mat in enumerate(self.inputs.in_files)]
                    ))
            else:
                itk_outs = [_mat2itk((
                    in_mat, self.inputs.in_reference, self.inputs.in_source, i, tmp_folder))
                    for i, in_mat in enumerate(self.inputs.in_files)
                ]

        # Compose the collated ITK transform file and write
        tfms = '#Insight Transform File V1.0\n' + ''.join(
            [el[1] for el in sorted(itk_outs)])

        self._results['out_file'] = os.path.join(runtime.cwd, 'mat2itk.txt')
        with open(self._results['out_file'], 'w') as f:
            f.write(tfms)

        return runtime


class MultiApplyTransformsInputSpec(ApplyTransformsInputSpec):
    input_image = InputMultiPath(File(exists=True), mandatory=True,
                                 desc='input time-series as a list of volumes after splitting'
                                      ' through the fourth dimension')
    num_threads = traits.Int(1, usedefault=True, nohash=True,
                             desc='number of parallel processes')
    save_cmd = traits.Bool(True, usedefault=True,
                           desc='write a log of command lines that were applied')
    copy_dtype = traits.Bool(False, usedefault=True,
                             desc='copy dtype from inputs to outputs')


class MultiApplyTransformsOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(File(), desc='the output ITKTransform file')
    log_cmdline = File(desc='a list of command lines used to apply transforms')


class MultiApplyTransforms(SimpleInterface):

    """
    Apply the corresponding list of input transforms
    """
    input_spec = MultiApplyTransformsInputSpec
    output_spec = MultiApplyTransformsOutputSpec

    def _run_interface(self, runtime):
        # Get all inputs from the ApplyTransforms object
        ifargs = self.inputs.get()

        # Extract number of input images and transforms
        in_files = ifargs.pop('input_image')
        num_files = len(in_files)
        transforms = ifargs.pop('transforms')
        # Get number of parallel jobs
        num_threads = ifargs.pop('num_threads')
        save_cmd = ifargs.pop('save_cmd')

        # Remove certain keys
        for key in ['environ', 'ignore_exception',
                    'terminal_output', 'output_image']:
            ifargs.pop(key, None)

        # Get a temp folder ready
        tmp_folder = TemporaryDirectory(prefix='tmp-', dir=runtime.cwd)

        xfms_list = _arrange_xfms(transforms, num_files, tmp_folder)
        assert len(xfms_list) == num_files

        # Inputs are ready to run in parallel
        if num_threads < 1:
            num_threads = None

        if num_threads == 1:
            out_files = [_applytfms((
                in_file, in_xfm, ifargs, i, runtime.cwd))
                for i, (in_file, in_xfm) in enumerate(zip(in_files, xfms_list))
            ]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                out_files = list(pool.map(_applytfms, [
                    (in_file, in_xfm, ifargs, i, runtime.cwd)
                    for i, (in_file, in_xfm) in enumerate(zip(in_files, xfms_list))]
                ))
        tmp_folder.cleanup()

        # Collect output file names, after sorting by index
        self._results['out_files'] = [el[0] for el in out_files]

        if save_cmd:
            self._results['log_cmdline'] = os.path.join(runtime.cwd, 'command.txt')
            with open(self._results['log_cmdline'], 'w') as cmdfile:
                print('\n-------\n'.join([el[1] for el in out_files]),
                      file=cmdfile)
        return runtime


class FUGUEvsm2ANTSwarpInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='input displacements field map')
    pe_dir = traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-',
                         desc='phase-encoding axis')


class FUGUEvsm2ANTSwarpOutputSpec(TraitedSpec):
    out_file = File(desc='the output warp field')


class FUGUEvsm2ANTSwarp(SimpleInterface):

    """
    Convert a voxel-shift-map to ants warp

    """
    input_spec = FUGUEvsm2ANTSwarpInputSpec
    output_spec = FUGUEvsm2ANTSwarpOutputSpec

    def _run_interface(self, runtime):

        nii = nb.load(self.inputs.in_file)

        phaseEncDim = {'i': 0, 'j': 1, 'k': 2}[self.inputs.pe_dir[0]]

        if len(self.inputs.pe_dir) == 2:
            phaseEncSign = 1.0
        else:
            phaseEncSign = -1.0

        # Fix header
        hdr = nii.header.copy()
        hdr.set_data_dtype(np.dtype('<f4'))
        hdr.set_intent('vector', (), '')

        # Get data, convert to mm
        data = nii.get_data()

        aff = np.diag([1.0, 1.0, -1.0])
        if np.linalg.det(aff) < 0 and phaseEncDim != 0:
            # Reverse direction since ITK is LPS
            aff *= -1.0

        aff = aff.dot(nii.affine[:3, :3])

        data *= phaseEncSign * nii.header.get_zooms()[phaseEncDim]

        # Add missing dimensions
        zeros = np.zeros_like(data)
        field = [zeros, zeros]
        field.insert(phaseEncDim, data)
        field = np.stack(field, -1)
        # Add empty axis
        field = field[:, :, :, np.newaxis, :]

        # Write out
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_antswarp', newpath=runtime.cwd)
        nb.Nifti1Image(
            field.astype(np.dtype('<f4')), nii.affine, hdr).to_filename(
                self._results['out_file'])

        return runtime


def _mat2itk(args):
    from nipype.interfaces.c3 import C3dAffineTool
    from nipype.utils.filemanip import fname_presuffix

    in_file, in_ref, in_src, index, newpath = args
    # Generate a temporal file name
    out_file = fname_presuffix(in_file, suffix='_itk-%05d.txt' % index,
                               newpath=newpath)

    # Run c3d_affine_tool
    C3dAffineTool(transform_file=in_file, reference_file=in_ref, source_file=in_src,
                  fsl2ras=True, itk_transform=out_file, resource_monitor=False).run()
    transform = '#Transform %d\n' % index
    with open(out_file) as itkfh:
        transform += ''.join(itkfh.readlines()[2:])

    return (index, transform)


def _applytfms(args):
    """
    Applies ANTs' antsApplyTransforms to the input image.
    All inputs are zipped in one tuple to make it digestible by
    multiprocessing's map
    """
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    in_file, in_xform, ifargs, index, newpath = args
    out_file = fname_presuffix(in_file, suffix='_xform-%05d' % index,
                               newpath=newpath, use_ext=True)

    copy_dtype = ifargs.pop('copy_dtype', False)
    xfm = ApplyTransforms(
        input_image=in_file, transforms=in_xform, output_image=out_file, **ifargs)
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    runtime = xfm.run().runtime

    if copy_dtype:
        nii = nb.load(out_file)
        in_dtype = nb.load(in_file).get_data_dtype()

        # Overwrite only iff dtypes don't match
        if in_dtype != nii.get_data_dtype():
            nii.set_data_dtype(in_dtype)
            nii.to_filename(out_file)

    return (out_file, runtime.cmdline)


def _arrange_xfms(transforms, num_files, tmp_folder):
    """
    Convenience method to arrange the list of transforms that should be applied
    to each input file
    """
    base_xform = ['#Insight Transform File V1.0', '#Transform 0']
    # Initialize the transforms matrix
    xfms_T = []
    for i, tf_file in enumerate(transforms):
        # If it is a deformation field, copy to the tfs_matrix directly
        if guess_type(tf_file)[0] != 'text/plain':
            xfms_T.append([tf_file] * num_files)
            continue

        with open(tf_file) as tf_fh:
            tfdata = tf_fh.read().strip()

        # If it is not an ITK transform file, copy to the tfs_matrix directly
        if not tfdata.startswith('#Insight Transform File'):
            xfms_T.append([tf_file] * num_files)
            continue

        # Count number of transforms in ITK transform file
        nxforms = tfdata.count('#Transform')

        # Remove first line
        tfdata = tfdata.split('\n')[1:]

        # If it is a ITK transform file with only 1 xform, copy to the tfs_matrix directly
        if nxforms == 1:
            xfms_T.append([tf_file] * num_files)
            continue

        if nxforms != num_files:
            raise RuntimeError('Number of transforms (%d) found in the ITK file does not match'
                               ' the number of input image files (%d).' % (nxforms, num_files))

        # At this point splitting transforms will be necessary, generate a base name
        out_base = fname_presuffix(tf_file, suffix='_pos-%03d_xfm-{:05d}' % i,
                                   newpath=tmp_folder.name).format
        # Split combined ITK transforms file
        split_xfms = []
        for xform_i in range(nxforms):
            # Find start token to extract
            startidx = tfdata.index('#Transform %d' % xform_i)
            next_xform = base_xform + tfdata[startidx + 1:startidx + 4] + ['']
            xfm_file = out_base(xform_i)
            with open(xfm_file, 'w') as out_xfm:
                out_xfm.write('\n'.join(next_xform))
            split_xfms.append(xfm_file)
        xfms_T.append(split_xfms)

    # Transpose back (only Python 3)
    return list(map(list, zip(*xfms_T)))
