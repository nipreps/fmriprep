import logging

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nipype.pipeline import engine as pe

from fmriprep.interfaces.resampling import ResampleTemplate, UpsampleToZooms


def _image(path, zooms):
    shape = tuple(round(48 * 2.4 / zoom) for zoom in zooms)
    nb.Nifti1Image(
        np.random.default_rng(0).random(shape, dtype='float32'),
        np.diag((*zooms, 1.0)),
    ).to_filename(path)
    return str(path)


@pytest.mark.parametrize(
    ('natives', 'requested', 'expected'),
    [
        pytest.param([(2.4, 2.4, 2.4)] * 2, (1.2, 1.2, 1.2), (1.2, 1.2, 1.2), id='every_axis'),
        pytest.param([(3.0, 3.0, 4.0)] * 2, (1.2, 1.2, 1.2), (1.2, 1.2, 1.2), id='anisotropic'),
        pytest.param([(1.0, 1.0, 2.0)] * 2, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), id='coarse_axis'),
        pytest.param([(1.0, 1.0, 2.0)] * 2, (1.0, 1.0, 1.2), (1.0, 1.0, 1.2), id='aniso_target'),
        # Axes finer than requested are held, as is the finest image of the group
        pytest.param([(2.0, 2.0, 0.8)] * 2, (1.2, 1.2, 1.2), (1.2, 1.2, 0.8), id='one_axis_held'),
        pytest.param([(1.0, 1.0, 2.0)] * 2, (1.2, 1.2, 1.2), (1.0, 1.0, 1.2), id='two_axes_held'),
        pytest.param(
            [(2.4, 2.4, 2.4), (0.8, 0.8, 0.8)],
            (1.2, 1.2, 1.2),
            (0.8, 0.8, 0.8),
            id='finest_image_held',
        ),
    ],
)
def test_UpsampleToZooms(tmp_path, natives, requested, expected):
    """Images are regridded toward the target, covering the acquired field of view."""
    in_files = [_image(tmp_path / f'input{i}.nii.gz', native) for i, native in enumerate(natives)]

    upsample = pe.Node(
        UpsampleToZooms(in_files=in_files, target_zooms=requested),
        name='upsample',
        base_dir=tmp_path,
    )
    ret = upsample.run()

    assert ret.outputs.out_files == [
        str(tmp_path / f'upsample/input{i}_upsampled.nii.gz') for i in range(len(natives))
    ]
    extent = np.array(natives[0]) * nb.load(in_files[0]).shape
    for out_file in ret.outputs.out_files:
        out_img = nb.load(out_file)
        zooms = np.array(out_img.header.get_zooms()[:3])
        assert np.allclose(zooms, expected)
        # The field of view is covered, to within a voxel of the finer grid
        assert np.allclose(zooms * out_img.shape, extent, atol=zooms)


@pytest.mark.parametrize(
    'natives',
    [
        pytest.param([(0.8, 0.8, 0.8)] * 2, id='every_axis_finer'),
        pytest.param([(1.2, 1.2, 1.2)] * 2, id='every_axis_equal'),
    ],
)
def test_UpsampleToZooms_passthrough(tmp_path, caplog, natives):
    """Images with no axis to refine are passed through as a group."""
    in_files = [_image(tmp_path / f'input{i}.nii.gz', native) for i, native in enumerate(natives)]

    upsample = pe.Node(
        UpsampleToZooms(in_files=in_files, target_zooms=(1.2, 1.2, 1.2)),
        name='upsample',
        base_dir=tmp_path,
    )
    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        ret = upsample.run()

    assert ret.outputs.out_files == in_files
    assert 'Skipping upsampling' in caplog.text


@pytest.mark.parametrize(('tolerance', 'upsampled'), [(1e-3, False), (1e-5, True)])
def test_UpsampleToZooms_tolerance(tmp_path, tolerance, upsampled):
    """``tolerance`` sets how much coarser than the target an image must be to be upsampled."""
    in_files = [_image(tmp_path / f'input{i}.nii.gz', (1.2005, 1.2005, 1.2005)) for i in range(2)]

    upsample = pe.Node(
        UpsampleToZooms(in_files=in_files, target_zooms=(1.2, 1.2, 1.2), tolerance=tolerance),
        name='upsample',
        base_dir=tmp_path,
    )

    assert (upsample.run().outputs.out_files != in_files) is upsampled


def _to_itk(path, translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    nt.linear.Affine(matrix).to_filename(path, fmt='itk')
    return str(path)


@pytest.mark.parametrize(
    'indices',
    [
        pytest.param(None, id='all'),
        pytest.param([True, False, True, False], id='subset'),
        pytest.param([False, True, False, False], id='single'),
    ],
)
def test_ResampleTemplate(tmp_path, indices):
    """Images are scaled, then their voxelwise median is taken on the reference grid."""
    rng = np.random.default_rng(0)
    data = rng.random((4, 16, 16, 16), dtype='float32')
    scales = [2.0, 1.0, 0.5, 4.0]

    in_files, transforms, scale_files = [], [], []
    for i, (volume, scale) in enumerate(zip(data, scales, strict=True)):
        in_files.append(str(tmp_path / f'input{i}.nii.gz'))
        nb.Nifti1Image(volume, np.eye(4)).to_filename(in_files[-1])
        transforms.append(_to_itk(tmp_path / f'xfm{i}.txt'))
        scale_files.append(str(tmp_path / f'is{i}.txt'))
        np.savetxt(scale_files[-1], [scale])

    median = ResampleTemplate(
        in_files=in_files,
        transforms=transforms,
        reference=in_files[0],
        intensity_scales=scale_files,
    )
    if indices is not None:
        median.inputs.indices = indices
    out_file = median.run(cwd=tmp_path).outputs.out_file

    selected = np.flatnonzero(indices) if indices is not None else range(len(data))
    expected = np.median([data[i] * scales[i] for i in selected], axis=0)
    assert np.allclose(nb.load(out_file).get_fdata(), expected, atol=1e-4)


def test_ResampleTemplate_transform(tmp_path):
    """Transforms map each image onto the reference, as run-to-template transforms do."""
    data = np.random.default_rng(0).random((16, 16, 16), dtype='float32')
    reference = str(tmp_path / 'reference.nii.gz')
    nb.Nifti1Image(data, np.eye(4)).to_filename(reference)

    # The same brain, displaced by 4mm along x
    shifted = np.eye(4)
    shifted[0, 3] = 4.0
    moved = str(tmp_path / 'moved.nii.gz')
    nb.Nifti1Image(data, shifted).to_filename(moved)

    out_file = (
        ResampleTemplate(
            in_files=[moved],
            transforms=[_to_itk(tmp_path / 'xfm.txt', translation=(4.0, 0.0, 0.0))],
            reference=reference,
        )
        .run(cwd=tmp_path)
        .outputs.out_file
    )

    assert np.allclose(nb.load(out_file).get_fdata(), data, atol=1e-4)
