import numpy as np

from fmriprep.interfaces.resampling import resample_series


def test_resample_series_uses_volume_specific_fieldmaps():
    data = np.zeros((3, 1, 1, 2), dtype='f4')
    data[:, 0, 0, 0] = [10.0, 20.0, 30.0]
    data[:, 0, 0, 1] = [100.0, 200.0, 300.0]

    coordinates = np.zeros((3, 1, 1, 1), dtype='f4')
    pe_info = [(0, 1.0), (0, 1.0)]
    fmap_hz = np.zeros((1, 1, 1, 2), dtype='f4')
    fmap_hz[..., 1] = 1.0

    resampled = resample_series(
        data=data,
        coordinates=coordinates,
        pe_info=pe_info,
        jacobian=False,
        hmc_xfms=None,
        fmap_hz=fmap_hz,
        output_dtype='f4',
        order=0,
        mode='nearest',
        prefilter=False,
    )

    assert np.allclose(resampled[0, 0, 0, 0], 10.0)
    assert np.allclose(resampled[0, 0, 0, 1], 200.0)
