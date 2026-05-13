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
"""Tests for config.extensions sub-namespace serialisation."""

from fmriprep import config


def setup_function():
    config.extensions._namespaces.clear()


def teardown_function():
    config.extensions._namespaces.clear()


def test_namespace_survives_toml_round_trip():
    config.extensions.set_namespace('nibabies', {'age_months': 6})
    serialised = config.extensions.get()
    config.extensions._namespaces.clear()
    config.extensions.load(serialised, init=False)
    assert config.extensions.get_namespace('nibabies')['age_months'] == 6
