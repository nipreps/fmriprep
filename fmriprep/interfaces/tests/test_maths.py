import nibabel as nb
import numpy as np
from nipype.pipeline import engine as pe
from fmriprep.interfaces.maths import Clip


def test_Clip(tmp_path):
    in_file = str(tmp_path / "input.nii")
    data = np.array([[[-1., 1.], [-2., 2.]]])
    nb.Nifti1Image(data, np.eye(4)).to_filename(in_file)

    threshold = pe.Node(Clip(in_file=in_file, minimum=0), name="threshold", base_dir=tmp_path)

    ret = threshold.run()

    assert ret.outputs.out_file == str(tmp_path / "threshold/input_clipped.nii")
    out_img = nb.load(ret.outputs.out_file)
    assert np.allclose(out_img.get_fdata(), [[[0., 1.], [0., 2.]]])

    threshold2 = pe.Node(
        Clip(in_file=in_file, minimum=-3),
        name="threshold2",
        base_dir=tmp_path)

    ret = threshold2.run()

    assert ret.outputs.out_file == in_file
    out_img = nb.load(ret.outputs.out_file)
    assert np.allclose(out_img.get_fdata(), [[[-1., 1.], [-2., 2.]]])

    clip = pe.Node(
        Clip(in_file=in_file, minimum=-1, maximum=1),
        name="clip",
        base_dir=tmp_path)

    ret = clip.run()

    assert ret.outputs.out_file == str(tmp_path / "clip/input_clipped.nii")
    out_img = nb.load(ret.outputs.out_file)
    assert np.allclose(out_img.get_fdata(), [[[-1., 1.], [-1., 1.]]])

    nonpositive = pe.Node(
        Clip(in_file=in_file, maximum=0),
        name="nonpositive",
        base_dir=tmp_path)

    ret = nonpositive.run()

    assert ret.outputs.out_file == str(tmp_path / "nonpositive/input_clipped.nii")
    out_img = nb.load(ret.outputs.out_file)
    assert np.allclose(out_img.get_fdata(), [[[-1., 0.], [-2., 0.]]])
