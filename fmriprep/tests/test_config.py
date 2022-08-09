# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""Check the configuration module and file."""
import os
from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from unittest.mock import patch

import pytest
from toml import loads
from niworkflows.utils.spaces import format_reference

from .. import config


def _reset_config():
    """
    Forcibly reload the configuration module to restore defaults.

    .. caution::
      `importlib.reload` creates new sets of objects, but will not remove
      previous references to those objects."""
    import importlib
    importlib.reload(config)


def test_reset_config():
    execution = config.execution
    setattr(execution, 'bids_dir', 'TESTING')
    assert config.execution.bids_dir == 'TESTING'
    _reset_config()
    assert config.execution.bids_dir is None
    # Even though the config module was reset,
    # previous references to config classes
    # have not been touched.
    assert execution.bids_dir == 'TESTING'


def test_config_spaces():
    """Check that all necessary spaces are recorded in the config."""
    filename = Path(pkgrf('fmriprep', 'data/tests/config.toml'))
    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != 'environment':
            section = getattr(config, sectionname)
            section.load(configs, init=False)
    config.nipype.init()
    config.loggers.init()
    config.init_spaces()

    spaces = config.workflow.spaces
    assert "MNI152NLin6Asym:res-2" not in [
        str(s) for s in spaces.get_standard(full_spec=True)]

    assert "MNI152NLin6Asym_res-2" not in [
        format_reference((s.fullname, s.spec))
        for s in spaces.references if s.standard and s.dim == 3
    ]

    config.workflow.use_aroma = True
    config.init_spaces()
    spaces = config.workflow.spaces

    assert "MNI152NLin6Asym:res-2" in [
        str(s) for s in spaces.get_standard(full_spec=True)]

    assert "MNI152NLin6Asym_res-2" in [
        format_reference((s.fullname, s.spec))
        for s in spaces.references if s.standard and s.dim == 3
    ]

    config.execution.output_spaces = None
    config.workflow.use_aroma = False
    config.init_spaces()
    spaces = config.workflow.spaces

    assert [str(s) for s in spaces.get_standard(full_spec=True)] == []

    assert [
        format_reference((s.fullname, s.spec))
        for s in spaces.references if s.standard and s.dim == 3
    ] == ['MNI152NLin2009cAsym']
    _reset_config()


@pytest.mark.parametrize("master_seed,ants_seed,numpy_seed", [
    (1, 17612, 8272), (100, 19094, 60232)
])
def test_prng_seed(master_seed, ants_seed, numpy_seed):
    """Ensure seeds are properly tracked"""
    seeds = config.seeds
    with patch.dict(os.environ, {}):
        seeds.load({'_random_seed': master_seed}, init=True)
        assert getattr(seeds, 'master') == master_seed
        assert seeds.ants == ants_seed
        assert seeds.numpy == numpy_seed
        assert os.getenv("ANTS_RANDOM_SEED") == str(ants_seed)

    _reset_config()
    for seed in ('_random_seed', 'master', 'ants', 'numpy'):
        assert getattr(config.seeds, seed) is None
