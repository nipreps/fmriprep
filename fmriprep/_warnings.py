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
"""Manipulate Python warnings."""
import warnings
import logging

_wlog = logging.getLogger("py.warnings")
_wlog.addHandler(logging.NullHandler())


def _warn(message, category=None, stacklevel=1, source=None):
    """Redefine the warning function."""
    if category is not None:
        category = type(category).__name__
        category = category.replace("type", "WARNING")

    logging.getLogger("py.warnings").warning(
        f"{category or 'WARNING'}: {message}"
    )


def _showwarning(message, category, filename, lineno, file=None, line=None):
    _warn(message, category=category)


warnings.warn = _warn
warnings.showwarning = _showwarning
