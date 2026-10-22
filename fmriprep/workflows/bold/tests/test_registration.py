# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2026 The NiPreps Developers <nipreps@gmail.com>
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
"""Selection of the BOLD-to-anatomical registration and head-motion correction methods."""

import pytest

from ..hmc import init_bold_hmc_wf
from ..registration import init_bold_reg_wf


@pytest.mark.parametrize('use_bbr', [True, False, None])
@pytest.mark.parametrize(
    ('freesurfer', 'bbr_method', 'expected'),
    [
        (True, 'auto', 'bbreg_wf'),
        (False, 'auto', 'fsl_bbr_wf'),
        (True, 'fsl', 'fsl_bbr_wf'),
        (True, 'cat', 'cat_bbr_wf'),
        (False, 'cat', 'cat_bbr_wf'),
    ],
)
def test_bold_reg_wf_method(freesurfer, bbr_method, expected, use_bbr):
    wf = init_bold_reg_wf(
        freesurfer=freesurfer,
        use_bbr=use_bbr,
        bold2anat_dof=6,
        bold2anat_init='t1w',
        mem_gb=1,
        omp_nthreads=1,
        bbr_method=bbr_method,
    )
    subworkflows = {name.split('.')[0] for name in wf.list_node_names()}
    assert subworkflows == {'inputnode', 'outputnode', expected}


def test_cat_bbr_wf_is_rigid():
    with pytest.raises(ValueError, match='rigid'):
        init_bold_reg_wf(
            freesurfer=False,
            use_bbr=None,
            bold2anat_dof=9,
            bold2anat_init='t1w',
            mem_gb=1,
            omp_nthreads=1,
            bbr_method='cat',
        )


@pytest.mark.parametrize(('method', 'node'), [('mcflirt', 'mcflirt'), ('cat', 'cat_realign')])
def test_bold_hmc_wf_method(method, node):
    wf = init_bold_hmc_wf(mem_gb=1, omp_nthreads=1, method=method)
    assert set(wf.list_node_names()) == {'inputnode', 'outputnode', 'fsl2itk', node}
