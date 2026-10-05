# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
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
from ....utils.misc import get_wf_name
from ...tests import mock_config
from ..coreg import init_bold_run_coreg_wf, init_bold_template_coreg_wf


def test_run_coreg_skips_precomputed(tmp_path: Path):
    """Runs with a precomputed template2anat transform skip registration."""
    bold_files = [f'/bids/sub-01/func/sub-01_task-rest_run-{i}_bold.nii.gz' for i in (1, 2)]
    with mock_config():
        wf = init_bold_run_coreg_wf(
            bold_files=bold_files,
            coreg_space='run',
            bold2anat_dof=6,
            bold2anat_init='t1w',
            use_bbr=None,
            freesurfer=False,
            omp_nthreads=1,
            mem_gb=1,
            sloppy=True,
            output_dir=str(tmp_path),
            reference_anat='T1w',
            precomputed={'template2anat_xfm': ['/a.txt', None]},
        )
    bold_ids = [get_wf_name(f, None).removesuffix('_wf') for f in bold_files]
    # run-1 is precomputed (no registration node); run-2 is registered.
    assert wf.get_node(f'boldref_reg_{bold_ids[0]}_wf') is None
    assert wf.get_node(f'boldref_reg_{bold_ids[1]}_wf') is not None


@pytest.mark.parametrize('coreg_space', ['session', 'subject'])
@pytest.mark.parametrize('have_template', [True, False])
def test_template_reuse(tmp_path: Path, coreg_space: str, have_template: bool):
    """A precomputed group template boldref is reused instead of reconstructed."""
    xfms = [str(tmp_path / f'run{i}.txt') for i in (1, 2)]
    template2anat = str(tmp_path / 'template2anat.txt')
    boldref_template = str(tmp_path / 'boldref.nii.gz')
    for path in (*xfms, template2anat, boldref_template):
        Path(path).touch()

    bold_files = [
        f'/bids/sub-01/ses-A/func/sub-01_ses-A_task-rest_run-{i}_bold.nii.gz' for i in (1, 2)
    ]
    precomputed = {'run2template_xfms': xfms, 'template2anat_xfm': template2anat}
    if have_template:
        precomputed['boldref_template'] = boldref_template

    with mock_config():
        config.workflow.bold_coreg_upsample = False
        wf = init_bold_template_coreg_wf(
            bold_files=bold_files,
            coreg_space=coreg_space,
            bold2anat_dof=6,
            bold2anat_init='t1w',
            use_bbr=None,
            freesurfer=False,
            omp_nthreads=1,
            mem_gb=1,
            sloppy=True,
            output_dir=str(tmp_path),
            reference_anat='T1w',
            precomputed=precomputed,
        )

    template_buffer = wf.get_node('template_buffer')
    reconstructed = wf.get_node('resample_template')
    if have_template:
        assert template_buffer.inputs.boldref == boldref_template
        assert reconstructed is None
        assert wf.get_node('ds_boldref_template') is None
    else:
        assert reconstructed is not None
        assert wf.get_node('template_space') is None
        assert wf.get_node('template_data') is not None


@pytest.mark.parametrize(
    ('reference', 'template_runs', 'space', 'data'),
    [
        (None, None, [True, True, True], [True, True, True]),
        (1, [True, True, False], [False, True, False], [True, True, False]),
        (None, [True, False, True], [True, False, True], [True, False, True]),
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

    assert wf.get_node('template_space').inputs.mask == space
    assert wf.get_node('template_data').inputs.mask == data
    template_wf = wf.get_node('bold_template_wf')
    assert template_wf.get_node('boldref_template').inputs.fixed_timepoint is (
        reference is not None
    )
