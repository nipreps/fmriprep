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
"""Select the runs that build a session- or subject-level BOLD template."""

from argparse import ArgumentParser
from pathlib import Path


def get_parser() -> ArgumentParser:
    from .. import __version__

    parser = ArgumentParser(
        prog='fmriprep-bold-template',
        description=(
            'Plot the run BOLD references of an fMRIPrep output (--bold-coreg-level run), '
            'and write a page to select the runs that build the BOLD template. '
            'Running the command again writes the selection saved from the page '
            '(bold_template.json) into the scans files.'
        ),
    )
    parser.add_argument('bids_dir', type=Path, help='The input BIDS dataset')
    parser.add_argument(
        'fmriprep_dir',
        type=Path,
        help='The fMRIPrep output, processed with --bold-coreg-level run',
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Directory for the selection page and the scans files',
    )
    parser.add_argument(
        '--bold-coreg-level',
        required=True,
        choices=['session', 'subject'],
        help='The --bold-coreg-level of the fMRIPrep run that builds the templates',
    )
    parser.add_argument(
        '--participant-label',
        '--participant_label',
        nargs='+',
        type=lambda label: label.removeprefix('sub-'),
        help='A space delimited list of participant identifiers or a single '
        'identifier (the sub- prefix can be removed)',
    )
    parser.add_argument(
        '--session-label',
        '--session_label',
        nargs='+',
        type=lambda label: label.removeprefix('ses-'),
        help='A space delimited list of session identifiers or a single '
        'identifier (the ses- prefix can be removed)',
    )
    parser.add_argument('--version', action='version', version=f'fMRIPrep v{__version__}')
    return parser


def main(argv=None):
    import logging
    import sys

    from ..reports.bold_template import LOGGER, find_runs, generate

    opts = get_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        runs = find_runs(
            opts.bids_dir.absolute(),
            opts.fmriprep_dir.absolute(),
            opts.participant_label,
            opts.session_label,
        )
        if not runs:
            raise RuntimeError(
                f'No run BOLD references (*_space-run_boldref.nii.gz) in {opts.fmriprep_dir}.'
            )
        report, written = generate(
            runs,
            opts.bids_dir.absolute(),
            opts.fmriprep_dir.absolute(),
            opts.output_dir.absolute(),
            opts.bold_coreg_level,
        )
    except (RuntimeError, ValueError) as err:
        LOGGER.error(err)
        sys.exit(1)

    for scans_file in written:
        LOGGER.info('Wrote %s', scans_file)
    LOGGER.info('Open %s to select the runs of each BOLD template.', report)
