# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
from pathlib import Path

import pytest

from .... import config
from ...tests import mock_config
from ..coreg import init_bold_template_coreg_wf


def test_template_reconstruct(tmp_path: Path):
    """A template is rebuilt from precomputed transforms on the reference run's grid."""
    xfms = [str(tmp_path / f'run{i}.txt') for i in (1, 2)]
    template2anat = str(tmp_path / 'template2anat.txt')
    for path in (*xfms, template2anat):
        Path(path).touch()

    bold_files = [
        f'/bids/sub-01/ses-A/func/sub-01_ses-A_task-rest_run-{i}_bold.nii.gz' for i in (1, 2)
    ]
    with mock_config():
        config.workflow.bold_coreg_upsample = False
        wf = init_bold_template_coreg_wf(
            bold_files=bold_files,
            coreg_space='session',
            bold2anat_dof=6,
            bold2anat_init='t1w',
            use_bbr=None,
            freesurfer=False,
            omp_nthreads=1,
            mem_gb=1,
            sloppy=True,
            output_dir=str(tmp_path),
            reference_anat='T1w',
            precomputed={'run2template_xfms': xfms, 'template2anat_xfm': template2anat},
            template_reference=1,
            template_runs=[True, True],
        )

    reconstruct_template_wf = wf.get_node('reconstruct_template_wf')
    assert reconstruct_template_wf.get_node('reconstruct_template').inputs.indices == [True, True]
    assert reconstruct_template_wf.get_node('select_reference').inputs.index == [1]
    assert wf.get_node('template_space') is None
    assert wf.get_node('template_data').inputs.index == [0, 1]


@pytest.mark.parametrize(
    ('reference', 'template_runs', 'space', 'data'),
    [
        (None, None, [0, 1, 2], [0, 1, 2]),
        (1, [True, True, False], [1], [0, 1]),
        (None, [True, False, True], [0, 2], [0, 2]),
    ],
)
def test_template_provenance(tmp_path: Path, reference, template_runs, space, data):
    """The runs defining the template space and image are recorded in its metadata."""
    bold_files = [
        f'/bids/sub-01/ses-A/func/sub-01_ses-A_task-rest_run-{i}_bold.nii.gz' for i in (1, 2, 3)
    ]
    with mock_config():
        config.workflow.bold_coreg_upsample = False
        wf = init_bold_template_coreg_wf(
            bold_files=bold_files,
            coreg_space='session',
            bold2anat_dof=6,
            bold2anat_init='t1w',
            use_bbr=None,
            freesurfer=False,
            omp_nthreads=1,
            mem_gb=1,
            sloppy=True,
            output_dir=str(tmp_path),
            reference_anat='T1w',
            template_reference=reference,
            template_runs=template_runs,
        )

    assert wf.get_node('template_space').inputs.index == space
    assert wf.get_node('template_data').inputs.index == data
