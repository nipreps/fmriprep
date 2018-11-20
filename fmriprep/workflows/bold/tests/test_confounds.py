''' Testing module for fmriprep.workflows.bold.confounds '''
import pytest
import os
import nibabel as nib

from ..confounds import _add_volumes, _remove_volumes


skip_pytest = pytest.mark.skipif(not os.getenv('FMRIPREP_REGRESSION_SOURCE') or
                                 not os.getenv('FMRIPREP_REGRESSION_TARGETS'),
                                 reason='FMRIPREP_REGRESSION_{SOURCE,TARGETS} env vars not set')


@skip_pytest
def test_remove_volumes():
    bold_file = os.path.join(os.getenv('FMRIPREP_REGRESSION_SOURCE'),
                             'ds001362/sub-01_task-taskname_run-01_bold.nii.gz')
    n_volumes = nib.load(bold_file).shape[3]
    skip_vols = 3

    expected_volumes = n_volumes - skip_vols

    cut_file = _remove_volumes(bold_file, skip_vols)
    out_volumes = nib.load(cut_file).shape[3]
    # cleanup output file
    os.remove(cut_file)

    assert out_volumes == expected_volumes


@skip_pytest
def test_add_volumes():
    bold_file = os.path.join(os.getenv('FMRIPREP_REGRESSION_SOURCE'),
                             'ds001362/sub-01_task-taskname_run-01_bold.nii.gz')
    n_volumes = nib.load(bold_file).shape[3]
    add_vols = 3

    expected_volumes = n_volumes + add_vols

    add_file = _add_volumes(bold_file, bold_file, add_vols)
    out_volumes = nib.load(add_file).shape[3]
    # cleanup output file
    os.remove(add_file)

    assert out_volumes == expected_volumes
