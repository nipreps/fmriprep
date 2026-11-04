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

from __future__ import annotations

import json
import logging
from base64 import b64encode
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from itertools import groupby
from pathlib import Path

from .. import data
from ..utils.bids import (
    TEMPLATE_SELECTION_COLUMN,
    TEMPLATE_SELECTION_VALUES,
    read_selection,
    write_derivative_description,
)

LOGGER = logging.getLogger('fmriprep.bold_template')
SELECTION_FILE = 'bold_template.json'
REPORT_FILE = 'bold_template.html'


@dataclass(frozen=True)
class BoldrefRun:
    boldref: Path
    mask: Path
    bold: Path
    """Raw BOLD file, relative to the input dataset."""

    @property
    def scans(self) -> Path:
        """Scans file, relative to the input dataset."""
        scans_dir = self.bold.parent.parent
        return scans_dir / f'{"_".join(scans_dir.parts)}_scans.tsv'

    @property
    def filename(self) -> str:
        """Raw BOLD file, relative to the scans file."""
        return self.bold.relative_to(self.scans.parent).as_posix()


def find_runs(
    bids_dir: Path,
    fmriprep_dir: Path,
    participant_label: list[str] | None = None,
    session_label: list[str] | None = None,
) -> list[BoldrefRun]:
    """Pair the run BOLD references of a ``--bold-coreg-level run`` output with their BOLD.

    Multi-echo runs are paired with their first echo.
    """
    from bids.layout import BIDSLayout

    raw = BIDSLayout(bids_dir, validate=False)
    derivatives = BIDSLayout(fmriprep_dir, config=['bids', 'derivatives'], validate=False)
    filters = {'subject': participant_label, 'session': session_label}
    filters = {key: value for key, value in filters.items() if value}

    runs = []
    for boldref in derivatives.get(space='run', suffix='boldref', extension='.nii.gz', **filters):
        entities = {
            key: value
            for key, value in boldref.entities.items()
            if key not in ('space', 'suffix', 'extension')
        }
        bold = raw.get(
            return_type='filename', suffix='bold', extension=['.nii', '.nii.gz'], **entities
        )
        mask = derivatives.get(
            return_type='filename', space='run', desc='brain', suffix='mask', **entities
        )
        if not bold or not mask:
            raise RuntimeError(f'Could not find the BOLD file or brain mask of {boldref.path}.')
        runs.append(
            BoldrefRun(
                boldref=Path(boldref.path),
                mask=Path(mask[0]),
                bold=Path(sorted(bold)[0]).relative_to(bids_dir),
            )
        )
    return sorted(runs, key=lambda run: run.bold)


def group_key(scans: Path, coreg_level: str) -> Path:
    """The runs of a scans file build a template per session, or per subject."""
    return scans.parent if coreg_level == 'session' else Path(scans.parts[0])


def plot_boldref(boldref: Path, mask: Path) -> bytes:
    """Plot axial, sagittal and coronal slices of a BOLD reference in its native space.

    Intensities are scaled to the 98th percentile within the brain, so that the background
    noise of each run appears in proportion to its signal.
    """
    import nibabel as nb
    import numpy as np
    from matplotlib.figure import Figure

    img = nb.as_closest_canonical(nb.load(boldref))
    values = img.get_fdata(dtype='float32')
    brain = np.asanyarray(nb.as_closest_canonical(nb.load(mask)).dataobj) > 0
    zooms = img.header.get_zooms()[:3]
    vmax = np.percentile(values[brain], 98)

    ijk = np.argwhere(brain)
    lower, upper = ijk.min(axis=0), ijk.max(axis=0)
    center = (lower + upper) // 2
    axial = np.linspace(lower[2], upper[2], 8)[1:-1].round().astype(int)

    slices = [(values[:, :, z].T, zooms[1] / zooms[0]) for z in axial]
    slices.append((values[center[0], :, :].T, zooms[2] / zooms[1]))
    slices.append((values[:, center[1], :].T, zooms[2] / zooms[0]))

    fig = Figure(figsize=(2 * len(slices), 2.4), facecolor='black')
    for ax, (view, aspect) in zip(fig.subplots(1, len(slices)), slices, strict=True):
        ax.imshow(
            view,
            cmap='gray',
            vmin=0,
            vmax=vmax,
            origin='lower',
            aspect=aspect,
            interpolation='nearest',
        )
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    png = BytesIO()
    fig.savefig(png, format='png', dpi=100, facecolor='black')
    return png.getvalue()


def init_scans(runs: list[BoldrefRun], output_dir: Path, bids_dir: Path) -> None:
    """Write a scans file per session that lists every run.

    The scans file starts from the existing one in the output directory, or else from the
    raw dataset's, keeping all its rows and columns. Runs it does not list are added as
    ``n/a``.
    """
    import pandas as pd

    for scans, group in groupby(runs, key=lambda run: run.scans):
        raw_file = bids_dir / scans
        out_file = output_dir / scans
        if read_selection(raw_file) is not None:
            LOGGER.warning(
                '%s has a "%s" column, which fMRIPrep reads instead of this selection.',
                raw_file,
                TEMPLATE_SELECTION_COLUMN,
            )

        source = out_file if out_file.is_file() else raw_file
        if source.is_file():
            table = pd.read_csv(source, sep='\t', dtype=str, keep_default_na=False)
        else:
            table = pd.DataFrame({'filename': []}, dtype=str)
        if TEMPLATE_SELECTION_COLUMN not in table:
            table[TEMPLATE_SELECTION_COLUMN] = 'n/a'
        missing = [run.filename for run in group if run.filename not in set(table['filename'])]
        if source == out_file and not missing:
            continue

        if missing:
            table = pd.concat(
                [table, pd.DataFrame({'filename': missing, TEMPLATE_SELECTION_COLUMN: 'n/a'})],
                ignore_index=True,
            ).fillna('n/a')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_file, sep='\t', index=False)


def apply_selection(runs: list[BoldrefRun], output_dir: Path, coreg_level: str) -> list[Path]:
    """Write the selection saved from the page into the scans files, and remove it.

    The selection is validated in full before any scans file is written.
    """
    import pandas as pd

    selection_file = output_dir / SELECTION_FILE
    selection = {
        (Path(scans), filename): label
        for scans, labels in json.loads(selection_file.read_text()).items()
        for filename, label in labels.items()
    }
    if unknown := sorted(set(selection) - {(run.scans, run.filename) for run in runs}):
        names = ', '.join(f'{scans}: {filename}' for scans, filename in unknown)
        raise ValueError(f'{SELECTION_FILE}: unknown runs {names}.')
    if invalid := sorted(set(selection.values()) - set(TEMPLATE_SELECTION_VALUES)):
        raise ValueError(
            f'{SELECTION_FILE}: invalid values {", ".join(invalid)} '
            f'(expected one of {", ".join(TEMPLATE_SELECTION_VALUES)}).'
        )

    tables = {
        scans: pd.read_csv(output_dir / scans, sep='\t', dtype=str, keep_default_na=False)
        for scans in {run.scans for run in runs}
    }
    for (scans, filename), label in selection.items():
        table = tables[scans]
        table.loc[table['filename'] == filename, TEMPLATE_SELECTION_COLUMN] = label

    targets = Counter()
    for scans, table in tables.items():
        targets[group_key(scans, coreg_level)] += (
            table[TEMPLATE_SELECTION_COLUMN] == 'target'
        ).sum()
    if conflicts := [key for key, count in targets.items() if count > 1]:
        raise ValueError(
            f'{SELECTION_FILE}: more than one "target" run in {", ".join(map(str, conflicts))}.'
        )

    written = sorted({scans for scans, _ in selection})
    for scans in written:
        tables[scans].to_csv(output_dir / scans, sep='\t', index=False)
    selection_file.unlink()
    return [output_dir / scans for scans in written]


def write_report(runs: list[BoldrefRun], output_dir: Path, coreg_level: str) -> Path:
    """Write the selection page, with the figures embedded and the scans files' values."""
    values = {
        scans: read_selection(output_dir / scans) or {} for scans in {run.scans for run in runs}
    }

    groups = []
    for key, group in groupby(runs, key=lambda run: group_key(run.scans, coreg_level)):
        group = list(group)
        groups.append(
            {
                'label': ' '.join(key.parts),
                'scans': sorted({run.scans.as_posix() for run in group}),
                'runs': [
                    {
                        'scans': run.scans.as_posix(),
                        'filename': run.filename,
                        'figure': 'data:image/png;base64,'
                        + b64encode(plot_boldref(run.boldref, run.mask)).decode(),
                        'value': values[run.scans].get(run.filename, 'n/a'),
                    }
                    for run in group
                ],
            }
        )

    page = {
        'selectionFile': SELECTION_FILE,
        'coregLevel': coreg_level,
        'values': TEMPLATE_SELECTION_VALUES,
        'groups': groups,
    }
    template = data.load.readable('bold_template.html').read_text()
    out_file = output_dir / REPORT_FILE
    out_file.write_text(
        template.replace('{{ page }}', json.dumps(page, indent=1).replace('</', '<\\/'))
    )
    return out_file


def generate(
    runs: list[BoldrefRun],
    bids_dir: Path,
    fmriprep_dir: Path,
    output_dir: Path,
    coreg_level: str,
) -> tuple[Path, list[Path]]:
    """Write the scans files, apply a selection saved from the page, and write the page.

    Returns the page and the scans files written from the saved selection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    write_derivative_description(
        bids_dir,
        output_dir,
        dataset_links={'raw': bids_dir, 'fmriprep': fmriprep_dir},
        name='fMRIPrep - BOLD template selection',
    )
    init_scans(runs, output_dir, bids_dir)
    written = (
        apply_selection(runs, output_dir, coreg_level)
        if (output_dir / SELECTION_FILE).is_file()
        else []
    )
    return write_report(runs, output_dir, coreg_level), written
