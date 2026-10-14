import shutil

import nibabel as nb
import nitransforms as nt
import nitransforms.resampling
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ....interfaces.tests.data import load as load_test_data
from ..hmc import init_bold_hmc_wf

BOLDREF = 'sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'


@pytest.mark.parametrize(
    ('hmc_method', 'nodes', 'tool'),
    [
        ('mcflirt', ['mcflirt', 'fsl2itk'], '`mcflirt`'),
        ('3dvolreg', ['volreg', 'volreg2itk'], '`3dvolreg`'),
    ],
)
def test_bold_hmc_wf(hmc_method, nodes, tool):
    wf = init_bold_hmc_wf(mem_gb=1, omp_nthreads=1, hmc_method=hmc_method)

    assert sorted(wf.list_node_names()) == sorted(['inputnode', 'outputnode', *nodes])
    assert tool in wf.__desc__


def _as_oblique(img):
    """The image with its affine tilted by 0.2 rad about the RAS x axis, through its center."""
    center = (img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3]
    R = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()
    tilt = nb.affines.from_matvec(R, center - R @ center)
    return nb.Nifti1Image(np.asanyarray(img.dataobj), tilt @ img.affine, img.header)


# Rotation vectors (rad) and translations (mm) of volumes 1-3; volume 0 is unmoved
MOTIONS = [
    ([0.05, -0.03, 0.02], [1.5, -2.0, 0.5]),
    ([-0.04, 0.06, -0.05], [-1.0, 2.5, -2.0]),
    ([0.08, 0.04, -0.06], [2.0, 1.0, 3.0]),
]


@pytest.mark.skipif(shutil.which('3dvolreg') is None, reason='3dvolreg required')
@pytest.mark.parametrize('oblique', [False, True], ids=['las', 'oblique'])
def test_bold_hmc_wf_3dvolreg_run(tmp_path, oblique):
    """The 3dvolreg workflow recovers known motion from a synthetic series."""
    boldref = nb.load(load_test_data(BOLDREF))
    if oblique:
        boldref = _as_oblique(boldref)
    center = (boldref.affine @ np.r_[(np.array(boldref.shape[:3]) - 1) / 2, 1])[:3]

    # Volume t samples the boldref at motion_t(x), so the transform that
    # resamples it back onto the boldref grid is inv(motion_t)
    vols, truth = [boldref], [np.eye(4)]
    for rotvec, translation in MOTIONS:
        R = Rotation.from_rotvec(rotvec).as_matrix()
        motion = nb.affines.from_matvec(R, center - R @ center + translation)
        vols.append(
            nt.resampling.apply(nt.linear.Affine(motion, reference=boldref), boldref, order=3)
        )
        truth.append(np.linalg.inv(motion))
    bold_file = tmp_path / 'bold.nii.gz'
    boldref_file = tmp_path / 'boldref.nii.gz'
    nb.funcs.concat_images(vols).to_filename(bold_file)
    boldref.to_filename(boldref_file)

    wf = init_bold_hmc_wf(mem_gb=1, omp_nthreads=1, hmc_method='3dvolreg')
    wf.base_dir = str(tmp_path / 'work')
    wf.inputs.inputnode.bold_file = str(bold_file)
    wf.inputs.inputnode.raw_ref_image = str(boldref_file)
    result = wf.run()

    (volreg2itk,) = [node for node in result.nodes() if node.name == 'volreg2itk']
    xforms = nt.linear.load(volreg2itk.result.outputs.out_file).matrix

    # Displacement of the field-of-view corners; 3dvolreg gets within about 0.2 mm
    corners = np.array(
        [[i, j, k, 1] for i in (0, 49) for j in (0, 49) for k in (0, 33)], dtype=float
    ).T
    world = boldref.affine @ corners
    displacement = np.linalg.norm((xforms - np.stack(truth)) @ world, axis=1)
    assert displacement.max() < 0.5
