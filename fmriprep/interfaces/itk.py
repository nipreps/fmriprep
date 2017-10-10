#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
import os
import numpy as np
import nibabel as nb

from joblib import Parallel, delayed
from mimetypes import guess_type
from tempfile import TemporaryDirectory

from niworkflows.nipype.utils.filemanip import fname_presuffix
from niworkflows.nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiPath, OutputMultiPath)

from niworkflows.nipype.interfaces.ants.resampling import ApplyTransformsInputSpec
from niworkflows.interfaces.base import SimpleInterface


class MCFLIRT2ITKInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='list of MAT files from MCFLIRT')
    in_reference = File(exists=True, mandatory=True,
                        desc='input image for spatial reference')
    in_source = File(exists=True, mandatory=True,
                     desc='input image for spatial source')
    nprocs = traits.Int(1, usedefault=True, nohash=True,
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
        parallel = Parallel(n_jobs=self.inputs.nprocs)
        with TemporaryDirectory() as tmp_folder:
            itk_outs = parallel(delayed(_mat2itk)(
                in_mat, self.inputs.in_reference, self.inputs.in_source, i,
                tmp_folder) for i, in_mat in enumerate(self.inputs.in_files))

        # Compose the collated ITK transform file and write
        tfms = '#Insight Transform File V1.0\n' + ''.join(
            [el[1] for el in sorted(itk_outs)])

        self._results['out_file'] = os.path.abspath('mat2itk.txt')
        with open(self._results['out_file'], 'w') as f:
            f.write(tfms)

        return runtime


class MultiApplyTransformsInputSpec(ApplyTransformsInputSpec):
    input_image = InputMultiPath(File(exists=True), mandatory=True,
                                 desc='input time-series as a list of volumes after splitting'
                                      ' through the fourth dimension')
    nprocs = traits.Int(1, usedefault=True, nohash=True,
                        desc='number of parallel processes')


class MultiApplyTransformsOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(File(), desc='the output ITKTransform file')


class MultiApplyTransforms(SimpleInterface):

    """
    Apply the corresponding list of input transforms
    """
    input_spec = MultiApplyTransformsInputSpec
    output_spec = MultiApplyTransformsOutputSpec

    def _run_interface(self, runtime):
        # Get all inputs from the ApplyTransforms object
        ifargs = self.inputs.get()

        # Get number of parallel jobs
        nprocs = ifargs.pop('nprocs')

        # Remove certain keys
        for key in ['environ', 'ignore_exception', 'num_threads',
                    'terminal_output', 'output_image']:
            ifargs.pop(key)

        # Extract number of input images and transforms
        in_files = ifargs.pop('input_image')
        num_files = len(in_files)
        transforms = ifargs.pop('transforms')

        # Get a temp folder ready
        tmp_folder = TemporaryDirectory()

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
        xfms_list = list(map(list, zip(*xfms_T)))
        assert len(xfms_list) == num_files

        # Inputs are ready to run in parallel
        parallel = Parallel(n_jobs=nprocs)
        out_files = parallel(delayed(_applytfms)(
            in_file, in_xfm, ifargs, i,
            runtime.cwd) for i, (in_file, in_xfm) in enumerate(zip(in_files, xfms_list)))

        tmp_folder.cleanup()
        # Collect output file names, after sorting by index
        self._results['out_files'] = [el[1] for el in sorted(out_files)]
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


def _mat2itk(in_file, in_ref, in_src, index=0, newpath=None):
    from niworkflows.nipype.interfaces.c3 import C3dAffineTool
    from niworkflows.nipype.utils.filemanip import fname_presuffix

    # Generate a temporal file name
    out_file = fname_presuffix(in_file, suffix='_itk-%05d.txt' % index,
                               newpath=newpath)

    # Run c3d_affine_tool
    C3dAffineTool(transform_file=in_file, reference_file=in_ref, source_file=in_src,
                  fsl2ras=True, itk_transform=out_file).run()
    transform = '#Transform %d\n' % index
    with open(out_file) as itkfh:
        transform += ''.join(itkfh.readlines()[2:])

    return (index, transform)


def _applytfms(in_file, in_xform, ifargs, index=0, newpath=None):
    from niworkflows.nipype.utils.filemanip import fname_presuffix
    from niworkflows.nipype.interfaces.ants import ApplyTransforms

    out_file = fname_presuffix(in_file, suffix='_xform-%05d' % index,
                               newpath=newpath, use_ext=True)

    ApplyTransforms(
        input_image=in_file, transforms=in_xform, output_image=out_file, **ifargs).run()
    return (index, out_file)
