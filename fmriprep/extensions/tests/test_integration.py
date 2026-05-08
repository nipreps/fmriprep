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
"""Integration tests: dispatch wired into fmriprep workflows."""

from __future__ import annotations

import inspect


def test_init_single_subject_wf_routes_anat_fit_through_dispatch():
    """``init_single_subject_wf`` must hand ``init_anat_fit_wf`` to the
    dispatcher as the call-site default rather than calling it directly."""
    from fmriprep.workflows import base as wf_base

    src = inspect.getsource(wf_base.init_single_subject_wf)

    # Routes through dispatch with the right hook name.
    assert '_build_extension(' in src
    assert "'anat_fit'" in src or '"anat_fit"' in src

    # The function passes ``init_anat_fit_wf`` as the default builder, so
    # the bare name must appear -- but it must not be invoked directly
    # (i.e. ``init_anat_fit_wf(`` should not appear as a call site).
    assert 'init_anat_fit_wf' in src
    assert 'init_anat_fit_wf(' not in src, (
        'init_anat_fit_wf should be passed to _build_extension as the '
        'default builder, not called directly.'
    )
