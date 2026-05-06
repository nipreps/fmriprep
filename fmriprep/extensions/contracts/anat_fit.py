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
"""Contract for the ``anat_fit`` hook."""

from __future__ import annotations

from pathlib import Path

from fmriprep.extensions.contracts import Contract, ContractField


class AnatFitContract(Contract):
    """Contract hook for `init_anat_fit_wf`.

    A class-based implementation of the inputnode andoutputnode of
    ``smriprep.workflows.anatomical.init_anat_fit_wf``. Output field
    names retain sMRIPrep's historical T1w-centric naming (``t1w_preproc``
    etc.); they are slot identifiers, not semantic statements about source
    contrast. An extension whose primary contrast is T2w may still claim
    this contract by producing those slot names — downstream consumers wire
    by name, not by contrast.

    A future fmriprep release will introduce contrast-agnostic
    (``anat_preproc``-style) naming, with a corresponding refactor of
    fmriprep's downstream wiring. That change will be a breaking contract
    update tied to a major version bump.
    """

    name = 'anat_fit'
    validated_through = {'smriprep': '0.19.2'}

    inputs = [
        ContractField('subjects_dir', Path, description='FreeSurfer SUBJECTS_DIR'),
        ContractField('subject_id', str, description='FreeSurfer subject ID'),
        ContractField('t1w', list, description='raw T1-weighted structural images'),
        ContractField('t2w', list, description='raw T2-weighted structural images'),
        ContractField('flair', list, description='raw FLAIR images'),
        ContractField(
            'roi', Path, description='mask of regions to exclude during standardization'
        ),
    ]

    outputs = [
        ContractField(
            't1w_preproc',
            Path,
            description='anatomical reference, bias-corrected',
        ),
        ContractField('t1w_mask', Path, description='binary brain mask'),
        ContractField(
            't1w_dseg',
            Path,
            description='discrete tissue segmentation (GM, WM, CSF)',
        ),
        ContractField(
            't1w_tpms',
            list,
            description='tissue probability maps (GM, WM, CSF)',
        ),
        ContractField(
            'anat2std_xfm',
            list,
            description='anat-to-template forward nonlinear transforms',
        ),
        ContractField(
            'std2anat_xfm',
            list,
            description='template-to-anat inverse nonlinear transforms',
        ),
        ContractField('template', list, description='full template names'),
        ContractField(
            'fsnative2t1w_xfm',
            Path,
            description='ITK affine, FreeSurfer-conformed to anat',
        ),
        ContractField(
            'subjects_dir',
            Path,
            description='FreeSurfer SUBJECTS_DIR',
        ),
        ContractField(
            'subject_id',
            str,
            description='FreeSurfer subject ID',
        ),
        ContractField(
            'cortex_mask',
            Path,
            description='cortical ribbon binary mask',
        ),
        ContractField(
            'anat_ribbon',
            Path,
            description='cortical ribbon volume (gray-matter band)',
        ),
        ContractField(
            'white',
            list,
            description='white-matter surface meshes (L/R)',
        ),
        ContractField(
            'pial',
            list,
            description='pial surface meshes (L/R)',
        ),
        ContractField(
            'midthickness',
            list,
            description='mid-thickness surface meshes (L/R)',
        ),
        ContractField(
            'sphere_reg',
            list,
            description='spherically registered surfaces (L/R)',
        ),
        ContractField(
            'sphere_reg_fsLR',
            list,
            description='spheres registered to fsLR (L/R)',
        ),
        ContractField(
            'thickness',
            list,
            description='cortical thickness scalar maps (L/R)',
        ),
        ContractField(
            'sulc',
            list,
            description='sulcal depth scalar maps (L/R)',
        ),
        ContractField(
            'curv',
            list,
            description='curvature scalar maps (L/R)',
        ),
    ]
