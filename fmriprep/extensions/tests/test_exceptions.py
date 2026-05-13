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
"""Tests for extension framework exception types."""

from fmriprep.extensions.exceptions import (
    ExtensionActivationError,
    ExtensionConfigError,
    ExtensionContractError,
    ExtensionError,
    ExtensionRegistrationError,
)


def test_extension_errors_subclasses():
    for cls in (
        ExtensionRegistrationError,
        ExtensionContractError,
        ExtensionActivationError,
        ExtensionConfigError,
    ):
        assert issubclass(cls, ExtensionError)
