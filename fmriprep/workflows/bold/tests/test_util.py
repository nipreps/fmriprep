''' Testing module for fmriprep.workflows.bold.util '''
import pytest
import os

import numpy as np
from nipype.utils.filemanip import fname_presuffix
from nilearn.image import load_img
from ..util import init_bold_reference_wf


def symmetric_overlap(img1, img2):
    mask1 = load_img(img1).get_data() > 0
    mask2 = load_img(img2).get_data() > 0

    total1 = np.sum(mask1)
    total2 = np.sum(mask2)
    overlap = np.sum(mask1 & mask2)
    return overlap / np.sqrt(total1 * total2)


@pytest.mark.skipif(not os.getenv('FMRIPREP_REGRESSION_SOURCE') or
                    not os.getenv('FMRIPREP_REGRESSION_TARGETS'),
                    reason='FMRIPREP_REGRESSION_{SOURCE,TARGETS} env vars not set')
@pytest.mark.parametrize('input_fname,expected_fname', [
    (os.path.join(os.getenv('FMRIPREP_REGRESSION_SOURCE', ''),
                  base_fname),
     fname_presuffix(base_fname, suffix='_mask', use_ext=True,
                     newpath=os.path.join(
                         os.getenv('FMRIPREP_REGRESSION_TARGETS', ''),
                         os.path.dirname(base_fname))))
    for base_fname in (
        'ds000116/sub-12_task-visualoddballwithbuttonresponsetotargetstimuli_run-02_bold.nii.gz',
        'ds000133/sub-06_ses-post_task-rest_run-01_bold.nii.gz',
        'ds000140/sub-32_task-heatpainwithregulationandratings_run-02_bold.nii.gz',
        'ds000157/sub-23_task-passiveimageviewing_bold.nii.gz',
        'ds000210/sub-06_task-rest_run-01_echo-1_bold.nii.gz',
        'ds000210/sub-06_task-rest_run-01_echo-2_bold.nii.gz',
        'ds000210/sub-06_task-rest_run-01_echo-3_bold.nii.gz',
        'ds000216/sub-03_task-rest_echo-1_bold.nii.gz',
        'ds000216/sub-03_task-rest_echo-2_bold.nii.gz',
        'ds000216/sub-03_task-rest_echo-3_bold.nii.gz',
        'ds000216/sub-03_task-rest_echo-4_bold.nii.gz',
        'ds000237/sub-03_task-MemorySpan_acq-multiband_run-01_bold.nii.gz',
        'ds000237/sub-06_task-MemorySpan_acq-multiband_run-01_bold.nii.gz',
        )
    ])
def test_masking(input_fname, expected_fname):
    bold_reference_wf = init_bold_reference_wf(omp_nthreads=1, enhance_t2=True)
    bold_reference_wf.inputs.inputnode.bold_file = input_fname
    res = bold_reference_wf.run(plugin='MultiProc')

    combine_masks = [node for node in res.nodes if node.name.endswith('combine_masks')][0]
    overlap = symmetric_overlap(expected_fname,
                                combine_masks.result.outputs.out_file)

    assert overlap > 0.95, input_fname
