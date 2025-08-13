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
"""Miscellaneous utilities."""

from functools import cache


def check_deps(workflow):
    """Make sure dependencies are present in this system."""
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and which(node.interface._cmd.split()[0]) is None)
    )


def fips_enabled():
    """
    Check if FIPS is enabled on the system.

    For more information, see:
    https://github.com/nipreps/fmriprep/issues/2480#issuecomment-891199276
    """
    from pathlib import Path

    fips = Path('/proc/sys/crypto/fips_enabled')
    return fips.exists() and fips.read_text()[0] != '0'


@cache
def estimate_bold_mem_usage(bold_fname: str) -> tuple[int, dict]:
    import nibabel as nb
    import numpy as np

    img = nb.load(bold_fname)
    nvox = int(np.prod(img.shape, dtype='u8'))
    # Assume tools will coerce to 8-byte floats to be safe
    bold_size_gb = 8 * nvox / (1024**3)
    bold_tlen = img.shape[-1]
    mem_gb = {
        'filesize': bold_size_gb,
        'resampled': bold_size_gb * 4,
        'largemem': bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return bold_tlen, mem_gb


def fmt_subjects_sessions(subses: list[tuple[str]], concat_limit: int = 1):
    """
    Format a list of subjects and sessions to be printed.

    Example
    -------
    >>> fmt_subjects_sessions([('01', 'A'), ('02', ['A', 'B']), ('03', None), ('04', ['A'])])
    'sub-01 ses-A, sub-02 (2 sessions), sub-03, sub-04 ses-A'
    """
    output = []
    for subject, session in subses:
        if isinstance(session, list):
            if len(session) > concat_limit:
                output.append(f'sub-{subject} ({len(session)} sessions)')
                continue
            session = session[0]

        if session is None:
            output.append(f'sub-{subject}')
        else:
            output.append(f'sub-{subject} ses-{session}')

    return ', '.join(output)
