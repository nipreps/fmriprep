import os
import numpy as np
from nipype.interfaces.base import SimpleInterface, TraitedSpec, traits, File
from nipype.utils.filemanip import fname_presuffix


class ClipInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Input imaging file")
    out_file = File(desc="Output file name")
    minimum = traits.Float(-np.inf, usedefault=True,
                           desc="Values under minimum are set to minimum")
    maximum = traits.Float(np.inf, usedefault=True,
                           desc="Values over maximum are set to maximum")


class ClipOutputSpec(TraitedSpec):
    out_file = File(desc="Output file name")


class Clip(SimpleInterface):
    """ Simple clipping interface that clips values to specified minimum/maximum

    If no values are outside the bounds, nothing is done and the in_file is passed
    as the out_file without copying.
    """
    input_spec = ClipInputSpec
    output_spec = ClipOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        img = nb.load(self.inputs.in_file)
        data = img.get_fdata()

        out_file = self.inputs.out_file
        if out_file:
            out_file = os.path.join(runtime.cwd, out_file)

        if np.any((data < self.inputs.minimum) | (data > self.inputs.maximum)):
            if not out_file:
                out_file = fname_presuffix(self.inputs.in_file, suffix="_clipped",
                                           newpath=runtime.cwd)
            np.clip(data, self.inputs.minimum, self.inputs.maximum, out=data)
            img.__class__(data, img.affine, img.header).to_filename(out_file)
        elif not out_file:
            out_file = self.inputs.in_file

        self._results["out_file"] = out_file
        return runtime


class Label2MaskInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Input label file")
    label_val = traits.Int(mandatory=True, dec="Label value to create mask from")


class Label2MaskOutputSpec(TraitedSpec):
    out_file = File(desc="Output file name")


class Label2Mask(SimpleInterface):
    """ Create mask file for a label from a multi-label segmentation
    """
    input_spec = Label2MaskInputSpec
    output_spec = Label2MaskOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        img = nb.load(self.inputs.in_file)

        mask = np.uint16(img.dataobj) == self.inputs.label_val
        out_img = img.__class__(mask, img.affine, img.header)
        out_img.set_data_dtype(np.uint8)

        out_file = fname_presuffix(self.inputs.in_file, suffix="_mask",
                                   newpath=runtime.cwd)

        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime
