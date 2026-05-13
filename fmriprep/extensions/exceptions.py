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
"""Typed errors raised by the fMRIPrep extension framework."""

from __future__ import annotations


class ExtensionError(Exception):
    """Base for all extension framework errors."""

    def __init__(self, extension_name: str, message: str) -> None:
        super().__init__(f'[extension={extension_name}] {message}')
        self.extension_name = extension_name
        self.message = message


class ExtensionRegistrationError(ExtensionError):
    """Raised when a descriptor is malformed or fails registration validation."""


class ExtensionContractError(ExtensionError):
    """Raised when an extension claims an unknown hook, or its produced
    workflow does not match the declared contract schema."""


class ExtensionActivationError(ExtensionError):
    """Raised when activation cannot be unambiguously determined.

    Covers the case where more than one extension is installed in the same
    environment: the design assumes a single extension per installation, so
    this is treated as a packaging error.
    """


class ExtensionConfigError(ExtensionError):
    """Raised by an extension's ``init_config`` when it cannot compute
    its required dynamic config (e.g., a required path is missing).

    Extensions should raise this with a message that tells the user
    what to fix (e.g., 'set MCRIBS_HOME or pass --nibabies-mcribs-dir').
    """
