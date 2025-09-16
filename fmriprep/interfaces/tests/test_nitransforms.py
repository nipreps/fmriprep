import nitransforms as nt
import numpy as np
from nibabel.tmpdirs import InGivenDirectory

from fmriprep.interfaces import nitransforms as fin


def test_ConvertAffine(tmp_path, data_dir):
    bold = data_dir / 'sub-pixar008_task-pixar_desc-coreg_boldref.nii.gz'
    anat = data_dir / 'sub-pixar008_desc-preproc_T1w.nii.gz'

    # lta_affine = nt.linear.load(data_dir / 'mri_coreg.lta')
    lta_convert_fsl = nt.linear.load(
        data_dir / 'mri_coreg-lta_convert.mat', moving=bold, reference=anat, fmt='fsl'
    )
    lta_convert_itk = nt.linear.load(data_dir / 'mri_coreg-lta_convert.txt')
    c3d_itk = nt.linear.load(data_dir / 'mri_coreg-c3d.txt')
    lta_convert_itk_inv = nt.linear.load(data_dir / 'mri_coreg-lta_convert-invert.txt')
    c3d_itk_inv = nt.linear.load(data_dir / 'mri_coreg-c3d-invert.txt')

    with InGivenDirectory(tmp_path):
        lta_to_fsl = fin.ConvertAffine(
            in_xfm=data_dir / 'mri_coreg.lta',
            reference=anat,
            moving=bold,
            out_fmt='fsl',
        ).run()
        assert lta_to_fsl.outputs.out_xfm == str(tmp_path / 'mri_coreg_fwd.mat')
        assert not lta_to_fsl.outputs.out_inv

        nitransforms_fsl = nt.linear.load(
            lta_to_fsl.outputs.out_xfm, moving=bold, reference=anat, fmt='fsl'
        )
        assert np.allclose(nitransforms_fsl.matrix, lta_convert_fsl.matrix, atol=1e-4)

        lta_to_itk = fin.ConvertAffine(in_xfm=data_dir / 'mri_coreg.lta', inverse=True).run()
        assert lta_to_itk.outputs.out_xfm == str(tmp_path / 'mri_coreg_fwd.txt')
        assert lta_to_itk.outputs.out_inv == str(tmp_path / 'mri_coreg_inv.txt')

        nitransforms_itk = nt.linear.load(lta_to_itk.outputs.out_xfm)
        assert np.allclose(nitransforms_itk.matrix, lta_convert_itk.matrix, atol=1e-4)
        assert np.allclose(nitransforms_itk.matrix, c3d_itk.matrix, atol=1e-4)

        nitransforms_itk_inv = nt.linear.load(lta_to_itk.outputs.out_inv)
        assert np.allclose(nitransforms_itk_inv.matrix, lta_convert_itk_inv.matrix, atol=1e-4)
        assert np.allclose(nitransforms_itk_inv.matrix, c3d_itk_inv.matrix, atol=1e-4)

        fsl_to_itk = fin.ConvertAffine(
            in_xfm=data_dir / 'mri_coreg-lta_convert.mat',
            reference=anat,
            moving=bold,
            out_fmt='itk',
            inverse=True,
        ).run()

        nitransforms_itk = nt.linear.load(fsl_to_itk.outputs.out_xfm)
        assert np.allclose(nitransforms_itk.matrix, lta_convert_itk.matrix, atol=1e-4)
        assert np.allclose(nitransforms_itk.matrix, c3d_itk.matrix, atol=1e-4)

        nitransforms_itk_inv = nt.linear.load(fsl_to_itk.outputs.out_inv)
        assert np.allclose(nitransforms_itk_inv.matrix, lta_convert_itk_inv.matrix, atol=1e-4)
        assert np.allclose(nitransforms_itk_inv.matrix, c3d_itk_inv.matrix, atol=1e-4)
