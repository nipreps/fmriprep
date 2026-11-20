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
"""Test the entry point."""

import json
import sys

import pytest
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ... import config
from ...tests.test_config import _reset_config
from ...utils.provenance import PROV_FILENAME, get_run_id
from .. import run


def _double(value):
    return 2 * value


def _build_workflow(config_file, retval):
    """Stand in for the construction of the workflow, with a tiny one laid out alike."""
    log_dir = config.execution.fmriprep_dir / 'sub-01' / 'log' / config.execution.run_uuid
    workflow = pe.Workflow(name='fmriprep_wf', base_dir=str(config.execution.work_dir))
    subject_wf = pe.Workflow(name='sub_01_wf')
    double = pe.Node(
        niu.Function(function=_double, input_names=['value'], output_names=['out']),
        name='double',
    )
    double.inputs.value = 21
    double.config = {'execution': {'crashdump_dir': str(log_dir)}}
    subject_wf.add_nodes([double])
    workflow.add_nodes([subject_wf])
    retval.update(return_code=0, workflow=workflow)
    return retval


def _build_boilerplate(config_file, workflow):
    """Stand in for the generation of the boilerplate (run in another process)."""


@pytest.mark.parametrize('write_provenance', [True, False])
def test_write_provenance(tmp_path, minimal_bids, monkeypatch, write_provenance):
    """``--write-provenance`` leaves the records of the run with the logs of the participant."""
    out_dir = tmp_path / 'out'
    argv = [
        'fmriprep',
        str(minimal_bids),
        str(out_dir),
        'participant',
        '-w',
        str(tmp_path / 'work'),
        '--skip-bids-validation',
        '--notrack',
        '--md-only-boilerplate',
        '--debug',
        'pdb',  # build the workflow in this process, and run it serially
        *(['--write-provenance'] if write_provenance else []),
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(sys, 'excepthook', sys.excepthook)  # --debug pdb sets its own
    monkeypatch.setattr('fmriprep.cli.workflow.build_workflow', _build_workflow)
    monkeypatch.setattr('fmriprep.cli.workflow.build_boilerplate', _build_boilerplate)
    monkeypatch.setattr('fmriprep.reports.core.generate_reports', lambda *args, **kwargs: [])

    try:
        with pytest.raises(SystemExit) as exit_info:
            run.main()
        assert exit_info.value.code == 0
        run_uuid = config.execution.run_uuid
        version = config.environment.version
    finally:
        _reset_config()

    prov_file = out_dir / 'sub-01' / 'log' / run_uuid / PROV_FILENAME
    assert prov_file.exists() is write_provenance
    if not write_provenance:
        return

    # The records are those of this run: its identifier, command line and version
    records = json.loads(prov_file.read_text())['@graph']
    nodes = {node['@id']: node for node in records if '@graph' not in node}
    run_id = f'bidsprov:{get_run_id(run_uuid)}'
    assert nodes[run_id]['dcterms:identifier'] == run_uuid
    assert nodes[run_id]['nipype:command'] == ' '.join(argv)
    agent = nodes[nodes[run_id]['prov:wasAssociatedWith']['@id']]
    assert (agent['rdfs:label'], agent['nipype:version']) == ('fMRIPrep', version)
    # ... its workflows and nodes
    assert nodes[f'{run_id}/fmriprep_wf.sub_01_wf']['dcterms:isPartOf'] == {
        '@id': f'{run_id}/fmriprep_wf'
    }
    assert nodes[f'{run_id}/fmriprep_wf.sub_01_wf.double']['rdfs:label'] == 'double'
    # ... and what the node has done
    (bundle,) = [node for node in records if '@graph' in node]
    assert bundle['@id'] == f'{run_id}/fmriprep_wf.sub_01_wf.double/record'
    assert 42 in [entity.get('prov:value') for entity in bundle['@graph']]
