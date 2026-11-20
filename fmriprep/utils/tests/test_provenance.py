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
"""Tests of provenance records."""

import getpass
import json
import os
import pwd
import socket
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
import rdflib
from nipype import config as ncfg
from nipype.interfaces import utility as niu
from nipype.interfaces.base import CommandLine
from nipype.pipeline import engine as pe
from nipype.pipeline.engine import Node
from nipype.utils import provenance as nprov
from rdflib.compare import isomorphic

from fmriprep.interfaces import DerivativesDataSink
from fmriprep.interfaces.resampling import ResampleSeries
from fmriprep.utils.provenance import (
    PROV_NAMESPACE,
    get_run_id,
    group_nodes,
    write_provenance,
)

PROV = rdflib.Namespace('http://www.w3.org/ns/prov#')
NIPYPE = rdflib.Namespace('http://nipy.org/nipype/terms/')
DCTERMS = rdflib.Namespace('http://purl.org/dc/terms/')
CRYPTO = rdflib.Namespace('http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions/')
RDFS_LABEL = rdflib.RDFS.label
RDF_TYPE = rdflib.RDF.type

RUN_UUID = '20200306-105302_d365772b-fd60-4741-a722-372c2f558b50'
RUN_ID = 'fmriprep-20200306T105302-d365772b'
MARKER = 'correct-horse-battery-staple'


def _id(local=''):
    return rdflib.URIRef(f'{PROV_NAMESPACE}{RUN_ID}{local}')


def _describe(files):
    return ', '.join(sorted(files))


def _add(a, b):
    from pathlib import Path

    out_file = Path('sum.txt').absolute()
    out_file.write_text(f'{a + b}')
    return a + b, str(out_file)


def _init_wf(base_dir, log_dir, subjects):
    wf = pe.Workflow(name='fmriprep_wf', base_dir=str(base_dir))
    shared = pe.Node(niu.Function(function=_add, output_names=['out', 'out_file']), name='shared')
    shared.inputs.a = 0
    shared.inputs.b = 1
    for subject in subjects:
        sub_wf = pe.Workflow(name=f'sub_{subject}_wf')
        first = pe.Node(
            niu.Function(function=_add, output_names=['out', 'out_file']), name='first'
        )
        first.inputs.b = 2
        second = pe.MapNode(
            niu.Function(function=_add, output_names=['out', 'out_file']),
            name='second',
            iterfield=['a'],
        )
        second.inputs.a = [3, 4]
        echo = pe.Node(
            CommandLine('echo', args=MARKER, environ={'ANTS_RANDOM_SEED': '42'}), name='echo'
        )
        # A dictionary of files, which Nipype records as text
        describe = pe.Node(niu.Function(function=_describe), name='describe')
        describe.inputs.files = {'self': __file__}
        sub_wf.connect(first, 'out', second, 'b')
        sub_wf.add_nodes([echo, describe])
        # Mimic fmriprep.workflows.base.init_fmriprep_wf
        sub_wf.config['execution']['crashdump_dir'] = str(log_dir / f'sub-{subject}' / 'log')
        for node in sub_wf._get_all_nodes():
            node.config = {'execution': dict(sub_wf.config['execution'])}
        wf.connect(shared, 'out', sub_wf, 'first.a')
    return wf


@pytest.fixture(scope='module')
def workflow_run(tmp_path_factory):
    """Run the workflow once: tests only read its results."""

    tmp_path = tmp_path_factory.mktemp('provenance')
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setitem(
            ncfg._sections['execution'], 'crashdump_dir', str(tmp_path / 'out' / 'logs')
        )
        wf = _init_wf(tmp_path / 'work', tmp_path / 'out', ['01', '02'])
        # Processes started afresh, as fMRIPrep does, do not inherit Nipype's configuration
        execgraph = wf.run(
            plugin='MultiProc', plugin_args={'n_procs': 2, 'mp_context': 'forkserver'}
        )
    return execgraph, tmp_path


@pytest.fixture
def execgraph(workflow_run):
    return workflow_run[0]


@pytest.fixture
def tmp_path(workflow_run):
    """Where the workflow was run; provenance files are removed after each test."""
    yield workflow_run[1]
    for path in (workflow_run[1] / 'out').rglob('*provenance*'):
        path.unlink()


def _write(execgraph, tmp_path):
    out_files, errors = write_provenance(
        execgraph,
        tmp_path / 'out' / 'logs',
        RUN_UUID,
        version='26.0.0+test',
        command=['fmriprep', 'my bids', 'out', 'participant'],
    )
    assert errors == {}
    return out_files


def _load(out_file):
    dataset = rdflib.Dataset()
    dataset.parse(out_file, format='json-ld')
    return dataset


def _named(graph):
    """The collections of inputs, outputs and environment of an interface."""
    return {
        subject
        for kind in (NIPYPE.Inputs, NIPYPE.Outputs, NIPYPE.Environment)
        for subject in graph.subjects(RDF_TYPE, kind)
    }


def _values(dataset, predicate):
    """Collect the objects of a predicate across all the graphs of a dataset."""
    return {str(quad[2]) for quad in dataset.quads((None, predicate, None))}


@pytest.mark.parametrize(
    ('run_uuid', 'expected'),
    [
        (RUN_UUID, RUN_ID),
        ('20200306-105302_D365772B', RUN_ID),
        ('custom', 'fmriprep-run-custom'),
        ('my run #1', 'fmriprep-run-my%20run%20%231'),
        ('', 'fmriprep-run-'),
    ],
)
def test_get_run_id(run_uuid, expected):
    assert get_run_id(run_uuid) == expected


def test_group_nodes(tmp_path, execgraph):
    log_dir = tmp_path / 'out' / 'logs'
    groups = group_nodes(execgraph, log_dir)
    assert sorted(groups) == [tmp_path / 'out' / f'sub-{s}' / 'log' for s in ('01', '02')]
    for path, nodes in groups.items():
        subject = path.parent.name.removeprefix('sub-')
        assert sorted(node.fullname for node in nodes) == [
            'fmriprep_wf.shared',
            f'fmriprep_wf.sub_{subject}_wf.describe',
            f'fmriprep_wf.sub_{subject}_wf.echo',
            f'fmriprep_wf.sub_{subject}_wf.first',
            f'fmriprep_wf.sub_{subject}_wf.second',
        ]

    # Without participant-specific nodes, everything goes to the default directory
    assert list(group_nodes(execgraph.subgraph(groups[path][:1]), log_dir)) == [log_dir]


def test_write_provenance(tmp_path, execgraph):
    out_files = _write(execgraph, tmp_path)
    assert sorted(out_files) == [
        tmp_path / 'out' / f'sub-{s}' / 'log' / 'provenance.jsonld' for s in ('01', '02')
    ]
    # Nothing else is left behind
    assert sorted(path.name for path in tmp_path.iterdir()) == ['out', 'work']
    assert not list(tmp_path.rglob('.*.tmp'))
    assert [path.name for path in (tmp_path / 'work').iterdir()] == ['fmriprep_wf']
    assert [path.name for path in out_files[0].parent.iterdir()] == ['provenance.jsonld']

    shared_graphs = []
    for out_file in out_files:
        subject = out_file.parent.parent.name.removeprefix('sub-')
        other = {'01': '02', '02': '01'}[subject]

        dataset = _load(out_file)
        default = dataset.default_graph

        # The run, the workflows and the nodes are activities with informative identifiers
        sub_wf = f'/fmriprep_wf.sub_{subject}_wf'
        nodes = {
            _id('/fmriprep_wf.shared'): _id('/fmriprep_wf'),
            _id(f'{sub_wf}.describe'): _id(sub_wf),
            _id(f'{sub_wf}.echo'): _id(sub_wf),
            _id(f'{sub_wf}.first'): _id(sub_wf),
            _id(f'{sub_wf}.second'): _id(sub_wf),
        }
        workflows = {_id('/fmriprep_wf'): _id(), _id(sub_wf): _id('/fmriprep_wf')}
        assert set(default.subjects(RDF_TYPE, PROV.Activity)) == {_id(), *workflows, *nodes}
        assert set(default.subjects(RDF_TYPE, NIPYPE.Workflow)) == set(workflows)
        assert set(default.subject_objects(DCTERMS.isPartOf)) == set((nodes | workflows).items())

        # The run
        assert str(default.value(_id(), RDFS_LABEL)) == RUN_ID
        assert str(default.value(_id(), DCTERMS.identifier)) == RUN_UUID
        assert str(default.value(_id(), NIPYPE.command)) == "fmriprep 'my bids' out participant"
        started = default.value(_id(), PROV.startedAtTime).toPython()
        ended = default.value(_id(), PROV.endedAtTime).toPython()
        assert started.utcoffset() is not None
        assert started < ended
        agent = default.value(_id(), PROV.wasAssociatedWith)
        assert agent == rdflib.URIRef(f'{PROV_NAMESPACE}fmriprep-26.0.0%2Btest')
        assert (agent, RDF_TYPE, PROV.SoftwareAgent) in default
        assert str(default.value(agent, NIPYPE.version)) == '26.0.0+test'

        # The nodes
        assert str(default.value(_id(f'{sub_wf}.second'), RDFS_LABEL)) == 'second'
        assert (_id(f'{sub_wf}.second'), RDF_TYPE, NIPYPE.MapNode) in default
        assert (_id(f'{sub_wf}.first'), RDF_TYPE, NIPYPE.Node) in default
        assert (_id(f'{sub_wf}.first'), RDF_TYPE, NIPYPE.Function) in default
        assert str(default.value(_id(f'{sub_wf}.echo'), NIPYPE.command)) == f'echo {MARKER}'
        for node in nodes:
            assert default.value(node, NIPYPE.hashval)
            assert (
                started
                < default.value(node, PROV.startedAtTime).toPython()
                <= default.value(node, PROV.endedAtTime).toPython()
                < ended
            )
        assert set(default.subject_objects(PROV.wasInformedBy)) == {
            (_id(f'{sub_wf}.first'), _id('/fmriprep_wf.shared')),
            (_id(f'{sub_wf}.second'), _id(f'{sub_wf}.first')),
        }

        # One bundle per execution of an interface
        records = {
            _id('/fmriprep_wf.shared/record'): _id('/fmriprep_wf.shared'),
            _id(f'{sub_wf}.describe/record'): _id(f'{sub_wf}.describe'),
            _id(f'{sub_wf}.echo/record'): _id(f'{sub_wf}.echo'),
            _id(f'{sub_wf}.first/record'): _id(f'{sub_wf}.first'),
            _id(f'{sub_wf}.second/record-0'): _id(f'{sub_wf}.second'),
            _id(f'{sub_wf}.second/record-1'): _id(f'{sub_wf}.second'),
        }
        assert set(default.subjects(RDF_TYPE, PROV.Bundle)) == set(records)
        assert set(records) == {g.identifier for g in dataset.graphs()} - {default.identifier}
        for record, node in records.items():
            assert (record, PROV.wasGeneratedBy, node) in default
            # ... where the execution of the interface is linked to its node
            interface = rdflib.URIRef(record.replace('/record', '/interface'))
            graph = dataset.graph(record)
            assert set(graph.subjects(RDF_TYPE, PROV.Activity)) == {interface}
            assert (interface, DCTERMS.isPartOf, node) in graph
            inputs = rdflib.URIRef(record.replace('/record', '/inputs'))
            assert (inputs, RDF_TYPE, NIPYPE.Inputs) in graph

        # Inputs and outputs of an interface, by name
        graph = dataset.graph(_id(f'{sub_wf}.second/record-1'))
        ports = {}
        for usage in graph.objects(_id(f'{sub_wf}.second/interface-1'), PROV.qualifiedUsage):
            entity = graph.value(usage, PROV.entity)
            ports[str(graph.value(usage, NIPYPE.inPort))] = graph.value(entity, PROV.value)
        assert ports['a'].toPython() == 4
        assert ports['b'].toPython() == 3
        (location,) = graph.objects(None, PROV.atLocation)
        assert str(location) == (
            f'file://{tmp_path}/work/fmriprep_wf/sub_{subject}_wf/second/mapflow/_second1/sum.txt'
        )

        # The details of what was run do not leak across participants
        workdirs = _values(dataset, NIPYPE.workingDirectory)
        assert len(workdirs) == 6
        assert sum(f'sub_{subject}_wf' in workdir for workdir in workdirs) == 5
        assert not any(f'sub_{other}_wf' in workdir for workdir in workdirs)
        # Checksums of the files that still exist: 3 + 3, and 3 + 4
        # (Nipype removes the outputs that no other node needs)
        assert len(_values(dataset, CRYPTO.sha512)) == 2

        shared_graphs.append(
            (
                {triple for triple in default if 'sub_' not in ''.join(map(str, triple))},
                dataset.graph(_id('/fmriprep_wf.shared/record')),
            )
        )

    # What is not specific to a participant is described identically in all files
    (default_1, record_1), (default_2, record_2) = shared_graphs
    assert default_1 == default_2
    assert (_id('/fmriprep_wf.shared'), RDF_TYPE, PROV.Activity) in default_1
    assert len(record_1) > 20
    assert isomorphic(record_1, record_2)


def test_jsonld_default_graph(tmp_path, execgraph):
    """The default graph must not be wrapped into an anonymous named graph."""
    out_file = _write(execgraph, tmp_path)[0]
    jsonld = json.loads(out_file.read_text())
    assert set(jsonld) == {'@context', '@graph'}
    assert all('@id' in item for item in jsonld['@graph'])
    named = [item for item in jsonld['@graph'] if '@graph' in item]
    assert len(named) == 6
    assert all(item['@id'].startswith(f'bidsprov:{RUN_ID}/') for item in named)


def test_privacy(tmp_path, execgraph, monkeypatch):
    """User, host and output streams are not recorded."""

    monkeypatch.setenv('LOGNAME', 'jane.doe')
    assert getpass.getuser() == 'jane.doe'

    for out_file in _write(execgraph, tmp_path):
        text = out_file.read_text()
        assert 'jane.doe' not in text
        assert 'foaf:host' not in text
        assert f'file://{socket.gethostname()}' not in text
        assert 'file:///' in text
        assert f'file://{__file__}' in text  # within a dictionary
        for stream in ('stdout', 'stderr', 'merged', 'LoggedInUser', 'prov:Person'):
            assert stream not in text
        # What the command printed is not recorded: the marker only shows as an input
        # (``args``) and in the command line of the node and of the interface
        assert text.count(MARKER) == 3
        dataset = _load(out_file)
        assert _values(dataset, NIPYPE.environmentVariable) >= {'PATH'}
        assert not _values(dataset, NIPYPE.environmentVariable) & {'USER', 'LOGNAME'}
        # No dangling references to what was left out
        for graph in dataset.graphs():
            subjects = set(graph.subjects())
            for obj in graph.objects(None, PROV.agent):
                assert obj in subjects


def test_reproducibility_envvars(tmp_path, execgraph):
    """Seeds are recorded with the interfaces they were set for."""
    dataset = _load(_write(execgraph, tmp_path)[0])
    seeds = {
        (graph.identifier, str(graph.value(entity, PROV.value)))
        for graph in dataset.graphs()
        for entity in graph.subjects(
            NIPYPE.environmentVariable, rdflib.Literal('ANTS_RANDOM_SEED')
        )
    }
    # Other interfaces may have got one from the environment the tests are run in
    assert (_id('/fmriprep_wf.sub_01_wf.echo/record'), '42') in seeds
    assert not {record for record, seed in seeds if seed == '42'} - {
        _id('/fmriprep_wf.sub_01_wf.echo/record')
    }


def test_unknown_user(tmp_path, execgraph, monkeypatch):
    """Users without a name (e.g., ``docker run -u <uid>``) get their provenance too."""

    for envvar in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
        monkeypatch.delenv(envvar, raising=False)

    def _getpwuid(uid):
        raise KeyError(f'getpwuid(): uid not found: {uid}')

    monkeypatch.setattr(pwd, 'getpwuid', _getpwuid)
    with pytest.raises(OSError, match='No username'):
        getpass.getuser()

    assert len(_write(execgraph, tmp_path)) == 2
    assert 'LOGNAME' not in os.environ


def test_checksums_once(tmp_path, execgraph, monkeypatch):
    """Files are read once, however many nodes they are involved in."""

    calls = []
    hash_infile = nprov.hash_infile

    def _hash_infile(afile, *args, **kwargs):
        calls.append(afile)
        return hash_infile(afile, *args, **kwargs)

    monkeypatch.setattr(nprov, 'hash_infile', _hash_infile)
    _write(execgraph, tmp_path)
    # The two outputs of ``second``, which are used and generated, for each participant
    assert len(calls) == len(set(calls)) == 4
    assert nprov.hash_infile is _hash_infile


def test_write_provenance_errors(tmp_path, execgraph):
    """A participant whose records cannot be written does not affect the others."""
    log_dir = tmp_path / 'out' / 'sub-01' / 'log'
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    if log_dir.exists():
        log_dir.rmdir()  # Emptied after earlier tests
    log_dir.write_text('not a directory')
    try:
        out_files, errors = write_provenance(execgraph, tmp_path / 'out' / 'logs', RUN_UUID)
    finally:
        log_dir.unlink()

    assert out_files == [tmp_path / 'out' / 'sub-02' / 'log' / 'provenance.jsonld']
    assert list(errors) == [log_dir]
    assert isinstance(errors[log_dir], OSError)
    # No partial file is left behind
    assert list((tmp_path / 'out').rglob('*provenance*')) == out_files


def test_node_without_result(tmp_path, execgraph, monkeypatch, caplog):
    """A node that cannot be described does not affect the others."""

    def _result(self):
        if self.fullname == 'fmriprep_wf.sub_01_wf.first':
            raise FileNotFoundError('no result file')
        return self._load_results()

    monkeypatch.setattr(Node, 'result', property(_result))
    out_files = _write(execgraph, tmp_path)
    assert len(out_files) == 2
    assert 'No provenance record for node "fmriprep_wf.sub_01_wf.first"' in caplog.text

    dataset = _load(tmp_path / 'out' / 'sub-01' / 'log' / 'provenance.jsonld')
    default = dataset.default_graph
    node = _id('/fmriprep_wf.sub_01_wf.first')
    assert (node, RDF_TYPE, PROV.Activity) in default
    assert (node, DCTERMS.isPartOf, _id('/fmriprep_wf.sub_01_wf')) in default
    assert (_id('/fmriprep_wf.sub_01_wf.second'), PROV.wasInformedBy, node) in default
    assert not list(default.subjects(PROV.wasGeneratedBy, node))
    assert len(list(default.subjects(RDF_TYPE, PROV.Bundle))) == 5


def test_list_members(tmp_path_factory, monkeypatch):
    """Files of an input that is a list (e.g., a chain of transforms) are described."""

    tmp_path = tmp_path_factory.mktemp('redo')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(ncfg._sections['execution'], 'crashdump_dir', str(tmp_path / 'logs'))
    rng = np.random.default_rng(2026)
    nb.Nifti1Image(rng.random((8, 8, 8, 3), dtype='f4'), np.eye(4)).to_filename('bold.nii.gz')
    nb.Nifti1Image(np.zeros((6, 6, 6), dtype='f4'), np.eye(4)).to_filename('ref.nii.gz')
    xfms = []
    for idx, shift in enumerate((0.5, -1.0)):
        matrix = np.eye(4)
        matrix[idx, 3] = shift
        xfms.append(str(tmp_path / f'xfm{idx}.txt'))
        nt.linear.Affine(matrix).to_filename(xfms[-1], fmt='itk')

    wf = pe.Workflow(name='fmriprep_wf', base_dir=str(tmp_path / 'work'))
    resample = pe.Node(
        ResampleSeries(
            in_file=str(tmp_path / 'bold.nii.gz'),
            ref_file=str(tmp_path / 'ref.nii.gz'),
            transforms=xfms,
            inverse=[False, True],
        ),
        name='resample',
    )
    wf.add_nodes([resample])
    execgraph = wf.run(plugin='Linear')
    (out_file,), errors = write_provenance(execgraph, tmp_path / 'logs', RUN_UUID)
    assert errors == {}
    assert out_file == tmp_path / 'logs' / 'provenance.jsonld'

    # The transforms are a list, whose files are described with their checksums
    dataset = _load(out_file)
    graph = dataset.graph(_id('/fmriprep_wf.resample/record'))
    (chain,) = set(graph.subjects(RDF_TYPE, PROV.Collection)) - _named(graph)
    assert json.loads(graph.value(chain, PROV.value)) == xfms
    members = set(graph.objects(chain, PROV.hadMember))
    assert {str(graph.value(member, PROV.atLocation)) for member in members} == {
        f'file://{xfm}' for xfm in xfms
    }
    assert all(graph.value(member, CRYPTO.sha512) for member in members)


def test_sink_metadata(tmp_path_factory, monkeypatch):
    """Metadata given to a data sink when it is created are recorded as its inputs."""

    tmp_path = tmp_path_factory.mktemp('target')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(ncfg._sections['execution'], 'crashdump_dir', str(tmp_path / 'logs'))
    bold = tmp_path / 'raw' / 'sub-01' / 'func' / 'sub-01_task-rest_bold.nii.gz'
    bold.parent.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    nb.Nifti1Image(rng.random((8, 8, 8, 3), dtype='f4'), np.eye(4)).to_filename(bold)
    nb.Nifti1Image(np.zeros((6, 6, 6), dtype='f4'), np.eye(4)).to_filename('ref.nii.gz')
    matrix = np.eye(4)
    matrix[0, 3] = 0.5
    nt.linear.Affine(matrix).to_filename('xfm.txt', fmt='itk')

    wf = pe.Workflow(name='fmriprep_wf', base_dir=str(tmp_path / 'work'))
    inputnode = pe.Node(niu.IdentityInterface(fields=['metadata']), name='inputnode')
    inputnode.inputs.metadata = {'RepetitionTime': 2.0}
    resample = pe.Node(
        ResampleSeries(
            in_file=str(bold),
            ref_file=str(tmp_path / 'ref.nii.gz'),
            transforms=[str(tmp_path / 'xfm.txt')],
        ),
        name='resample',
    )
    ds_bold = pe.Node(
        DerivativesDataSink(
            base_directory=str(tmp_path / 'out'),
            source_file=str(bold),
            space='MNI152NLin6Asym',
            desc='preproc',
            compress=True,
            TaskName='rest',
        ),
        name='ds_bold',
    )
    wf.connect([
        (resample, ds_bold, [('out_file', 'in_file')]),
        (inputnode, ds_bold, [('metadata', 'meta_dict')]),
    ])  # fmt:skip
    execgraph = wf.run(plugin='Linear')
    (out_file,), errors = write_provenance(execgraph, tmp_path / 'logs', RUN_UUID)
    assert errors == {}

    name = 'sub-01/func/sub-01_task-rest_space-MNI152NLin6Asym_desc-preproc_bold'
    original = tmp_path / 'out' / f'{name}.nii.gz'
    sidecar = json.loads(original.with_name(original.name[:-7] + '.json').read_text())
    assert sidecar['TaskName'] == 'rest'

    # Metadata given to the data sink when created are recorded as its inputs
    graph = _load(out_file).graph(_id('/fmriprep_wf.ds_bold/record'))
    assert {str(port) for port in graph.objects(None, NIPYPE.inPort)} >= {'TaskName', 'meta_dict'}


def _mixed_values(flag, links):
    import numpy as np

    return not flag, (np.float32(1.5), np.float32(2.0)), links


def test_encoding(tmp_path, monkeypatch, caplog):
    """Booleans, path objects and NumPy scalars are recorded as what they are."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(ncfg._sections['execution'], 'crashdump_dir', str(tmp_path / 'logs'))
    wf = pe.Workflow(name='fmriprep_wf', base_dir=str(tmp_path / 'work'))
    values = pe.Node(
        niu.Function(
            function=_mixed_values,
            input_names=['flag', 'links'],
            output_names=['flag', 'zooms', 'links'],
        ),
        name='values',
    )
    values.inputs.flag = True
    values.inputs.links = {'raw': Path('/data'), 'known': [Path('/out'), 1]}
    wf.add_nodes([values])
    execgraph = wf.run(plugin='Linear')
    (out_file,), errors = write_provenance(execgraph, tmp_path / 'logs', RUN_UUID)
    assert errors == {}
    assert 'Could not encode' not in caplog.text
    assert 'No provenance record' not in caplog.text

    graph = _load(out_file).graph(_id('/fmriprep_wf.values/record'))
    recorded = {'in': {}, 'out': {}}
    for usage in graph.subjects(NIPYPE.inPort, None):
        port = str(graph.value(usage, NIPYPE.inPort))
        recorded['in'][port] = graph.value(graph.value(usage, PROV.entity), PROV.value)
    for generation in graph.subjects(NIPYPE.outPort, None):
        port = str(graph.value(generation, NIPYPE.outPort))
        (entity,) = graph.subjects(PROV.qualifiedGeneration, generation)
        recorded['out'][port] = graph.value(entity, PROV.value)

    assert recorded['in']['flag'].toPython() is True
    assert recorded['out']['flag'].toPython() is False
    assert recorded['in']['flag'].datatype == rdflib.XSD.boolean
    links = json.loads(recorded['in']['links'])
    assert links['raw'] == '/data'
    assert json.loads(links['known']) == ['/out', 1]  # Nipype nests JSON within JSON
    assert json.loads(recorded['out']['zooms']) == [1.5, 2.0]
