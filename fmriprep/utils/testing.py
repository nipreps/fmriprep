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
"""Helpers for writing test datasets."""

import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from niworkflows.utils.testing import generate_bids_skeleton

from .. import data

_PLACEHOLDER = re.compile(r'\{(\w+)\}')


def _fill(value, params):
    if isinstance(value, str) and (match := _PLACEHOLDER.fullmatch(value)):
        return params[match.group(1)]
    return value


def deriv_skeleton(
    include=None, *, subject='01', session=None, task='rest', run=None, fmapid='auto00000'
):
    """Describe a skeleton of fMRIPrep derivatives.

    Parameters
    ----------
    include : :obj:`list` of :obj:`str` or :obj:`None`
        Groups from ``fmriprep/data/tests/derivatives.yml`` to write.
        By default, the groups listed under ``defaults``.
    subject : :obj:`str`
        Subject label, without ``sub-``.
    session, task, run, fmapid
        Values for the ``{name}`` placeholders in the groups.
        Entities whose value is ``None`` are left out.

    Returns
    -------
    :obj:`dict`
        A dataset description for :func:`niworkflows.utils.testing.generate_bids_skeleton`.

    """
    spec = yaml.safe_load(data.load.readable('tests/derivatives.yml').read_text())
    params = {'session': session, 'task': task, 'run': run, 'fmapid': fmapid}

    sessions = {}
    for name in spec['defaults'] if include is None else include:
        for file in spec['groups'][name]:
            entities = {
                key: interp
                for key, value in file.items()
                if (interp := _fill(value, params)) is not None
            }
            datatype = entities.pop('datatype')
            ses = entities.pop('session', None)
            sessions.setdefault(ses, {}).setdefault(datatype, []).append(entities)

    return {
        'dataset_description': spec['dataset_description'],
        subject: [{'session': ses, **datatypes} for ses, datatypes in sessions.items()],
    }


def write_derivatives(path, include=None, **kwargs) -> Path:
    """Write a skeleton of fMRIPrep derivatives.

    Parameters
    ----------
    path : :obj:`os.PathLike`
        Root of the dataset. If it exists, the files are added to it,
        so that a dataset can hold several runs.
    include, **kwargs
        Passed to :func:`deriv_skeleton`.

    Returns
    -------
    :obj:`~pathlib.Path`
        ``path``.

    """
    path = Path(path)
    skeleton = deriv_skeleton(include, **kwargs)
    if not path.exists():
        generate_bids_skeleton(path, skeleton)
        return path

    # generate_bids_skeleton() only writes new datasets
    with TemporaryDirectory() as tmpdir:
        generate_bids_skeleton(Path(tmpdir) / 'deriv', skeleton)
        shutil.copytree(Path(tmpdir) / 'deriv', path, dirs_exist_ok=True)
    return path
