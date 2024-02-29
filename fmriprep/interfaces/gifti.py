# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces for manipulating GIFTI files."""

import os

import nibabel as nb
import numpy as np
from nipype.interfaces.base import File, SimpleInterface, TraitedSpec, isdefined, traits


class CreateROIInputSpec(TraitedSpec):
    subject_id = traits.Str(desc='subject ID')
    hemisphere = traits.Enum(
        'L',
        'R',
        mandatory=True,
        desc='hemisphere',
    )
    thickness_file = File(exists=True, mandatory=True, desc='input GIFTI file')


class CreateROIOutputSpec(TraitedSpec):
    roi_file = File(desc='output GIFTI file')


class CreateROI(SimpleInterface):
    """Prepare GIFTI thickness file for use as a cortical ROI"""

    input_spec = CreateROIInputSpec
    output_spec = CreateROIOutputSpec

    def _run_interface(self, runtime):
        subject, hemi = self.inputs.subject_id, self.inputs.hemisphere
        if not isdefined(subject):
            subject = 'sub-XYZ'
        img = nb.GiftiImage.from_filename(self.inputs.thickness_file)
        # wb_command -set-structure
        img.meta['AnatomicalStructurePrimary'] = {'L': 'CortexLeft', 'R': 'CortexRight'}[hemi]
        darray = img.darrays[0]
        # wb_command -set-map-names
        meta = darray.meta
        meta['Name'] = f'{subject}_{hemi}_ROI'
        # wb_command -metric-palette calls have no effect on ROI files

        # Compiling an odd sequence of math operations that works out to:
        # wb_command -metric-math "abs(var * -1) > 0"
        roi = np.abs(darray.data) > 0

        # Divergence: Set datatype to uint8, since the values are boolean
        # wb_command sets datatype to float32
        darray = nb.gifti.GiftiDataArray(
            roi,
            intent=darray.intent,
            datatype='uint8',
            encoding=darray.encoding,
            endian=darray.endian,
            coordsys=darray.coordsys,
            ordering=darray.ind_ord,
            meta=meta,
        )

        img.darrays[0] = darray

        out_filename = os.path.join(runtime.cwd, f'{subject}.{hemi}.roi.native.shape.gii')
        img.to_filename(out_filename)
        self._results['roi_file'] = out_filename
        return runtime
