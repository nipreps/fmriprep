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
"""Write out the provenance of an *fMRIPrep* execution.

After a workflow has been run, the results of all its nodes hold what is necessary to
describe what was done: the interface that was run with its inputs and outputs,
command line, environment, and timing.
The utilities in this module describe them following the
`W3C PROV <https://www.w3.org/TR/prov-overview/>`__ data model, relying on *Nipype*
for the description of each interface, split the description by participant
(and session, if processed separately), and serialize it as PROV-O in
`JSON-LD <https://www.w3.org/TR/json-ld11/>`__.

.. note::
    *Nipype*'s own ``write_provenance`` execution option is deliberately **not** used:

    * every interface run in a process that carries the option (the main process, or
      workers that were forked from it, but not the ``forkserver`` workers *fMRIPrep*
      starts) writes ``provenance.provn`` and ``provenance.ttl`` into the directory
      the workflow was launched from, each overwriting those of the previous one;
    * at the end of the run it writes ``workflow_provenance_<date>.provn`` and ``.trig``
      into the base of the working directory (i.e., not with the derivatives),
      for all participants together, with random identifiers, and the names of the user
      and of the host;
    * a failure to write the latter (e.g., when the user is unknown to the system, as is
      common in containers) makes a successful run of *fMRIPrep* fail; and
    * enabling it with ``nipype.config.enable_provenance()`` also changes how the inputs
      of nodes are hashed.

    The first, third and fourth are reported as
    `nipy/nipype#3818 <https://github.com/nipy/nipype/issues/3818>`__.
    Its description of each interface is reused, working around how it encodes some
    values (see :func:`_nipype_patches`).

**Identifiers**.
The run, and the workflows and nodes executed in it, are given identifiers
that are informative and reproducible, rather than random:

* ``<run-id>`` (see :func:`get_run_id`), for the execution of *fMRIPrep* as a whole;
* ``<run-id>/<workflow fullname>``, for each (nested) workflow;
* ``<run-id>/<node fullname>``, for each node;
* ``<run-id>/<node fullname>/record[-<index>]``, for the bundle with the detailed record
  of a node (one per iteration, for a ``MapNode``); and
* ``<run-id>/<node fullname>/interface[-<index>]``, for the execution of the interface
  that is described in that bundle, and likewise ``inputs``, ``outputs`` and ``environment``
  for the collections of what it used and generated.

These are local names within :data:`PROV_NAMESPACE`, the namespace that
`BEP028 <https://github.com/bids-standard/bids-specification/pull/2099>`__ (a draft,
at the time of writing) proposes for provenance records described within a dataset.
Values (including files, by their checksum) keep the content-based identifiers
*Nipype* gives them, so the same value is the same entity wherever it is used.

**Privacy**.
Derivatives are often shared, so the records are stripped of the name of the user,
the name of the host, the ``USER`` and ``LOGNAME`` environment variables, and
the standard output and error of the interfaces.

**Reproducibility**.
Besides the environment variables *Nipype* records (such as ``PATH`` or
``OMP_NUM_THREADS``), those that seed random number generators or restrict the
CPU features in use are recorded for each interface
(see :data:`REPRODUCIBILITY_ENVVARS`).
"""

import getpass
import json
import logging
import os
import re
import shlex
from contextlib import ExitStack, contextmanager, suppress
from copy import copy
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock
from urllib.parse import quote

import numpy as np
import prov.model as pm
from nipype.interfaces.base import isdefined
from nipype.interfaces.base.support import InterfaceResult
from nipype.utils import provenance as nprov
from nipype.utils.provenance import ProvStore, dcterms, foaf, nipype_ns, prov_encode

LOGGER = logging.getLogger('nipype.workflow')

PROV_FILENAME = 'provenance.jsonld'

PROV_NAMESPACE = 'bids::prov#'
"""Namespace of the identifiers of the run, its workflows and its nodes."""

JSONLD_CONTEXT = {
    'prov': 'http://www.w3.org/ns/prov#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
    'foaf': 'http://xmlns.com/foaf/0.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'nipype': 'http://nipy.org/nipype/terms/',
    'niiri': 'http://iri.nidash.org/',
    'crypto': 'http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions/',
    'bidsprov': PROV_NAMESPACE,
}
"""Prefixes used to keep the JSON-LD serialization compact."""

_RUN_UUID = re.compile(r'(?P<date>\d{8})-(?P<time>\d{6})_(?P<uid>[0-9a-fA-F]{8})')
_FILE_HOST = re.compile(r'file://[^/\s"\']+/')
_PRIVATE_ENVVARS = ('USER', 'LOGNAME')
REPRODUCIBILITY_ENVVARS = (
    # Seeds of random number generators
    'ANTS_RANDOM_SEED',
    'AFNI_RANDOM_SEEDVAL',
    'FREESURFER_SEED',
    'GSL_RNG_SEED',
    'GSL_RNG_TYPE',
    'MVPA_SEED',
    'PYTHONHASHSEED',
    'REPROSEED',
    # CPU features numerical libraries are restricted to
    'REPROSEED_CPU',
    'NPY_DISABLE_CPU_FEATURES',
    'NPY_ENABLE_CPU_FEATURES',
    'OPENBLAS_CORETYPE',
    'MKL_CBWR',
    'ATEN_CPU_CAPABILITY',
    'ONEDNN_MAX_CPU_ISA',
    'DNNL_MAX_CPU_ISA',
    # Multithreading
    'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)
"""Environment variables recorded in addition to those *Nipype* records, as they
determine whether results can be replicated (see, for instance,
`reproseed <https://github.com/ReproNim/reproseed>`__)."""
_OUTPUT_STREAMS = ('stdout', 'stderr', 'merged')


def get_run_id(run_uuid):
    """Derive a short, informative identifier of a run from its ``run_uuid``.

    The identifier is composed of the name of the tool, the time the run was
    started and the first eight characters of its UUID, which makes it readable
    while keeping collisions unlikely, even for runs started within the same second:

    >>> get_run_id('20200306-105302_d365772b-fd60-4741-a722-372c2f558b50')
    'fmriprep-20200306T105302-d365772b'

    Identifiers of a different form (which may only come from an edited configuration
    file) are kept, percent-encoding characters that are unsafe in an IRI:

    >>> get_run_id('my run #1')
    'fmriprep-run-my%20run%20%231'

    """
    if match := _RUN_UUID.match(str(run_uuid)):
        return f'fmriprep-{match["date"]}T{match["time"]}-{match["uid"].lower()}'
    return f'fmriprep-run-{quote(str(run_uuid), safe="")}'


def get_run_start(run_uuid):
    """Extract the time a run was started from its ``run_uuid``, if possible.

    >>> get_run_start('20200306-105302_d365772b-fd60-4741-a722-372c2f558b50') is not None
    True
    >>> get_run_start('my run #1') is None
    True

    """

    if match := _RUN_UUID.match(str(run_uuid)):
        try:
            started = datetime.strptime(f'{match["date"]}{match["time"]}', '%Y%m%d%H%M%S')  # noqa: DTZ007
        except ValueError:
            return None
        # The run_uuid is stamped with the local time; Nipype reports times in UTC
        return started.astimezone(UTC)
    return None


def group_nodes(execgraph, default_dir):
    """Split the nodes of an execution graph by the log directory they report to.

    *fMRIPrep* sets the ``crashdump_dir`` of all the nodes of each
    single-subject workflow to ``<output_dir>/sub-<label>[/ses-<label>]/log/<run_uuid>``.
    Nodes that do not belong to any of those workflows (for example, the node
    setting up the *FreeSurfer* subjects directory) report to ``default_dir``.
    Such nodes are shared prerequisites, and are therefore included in all groups.

    Parameters
    ----------
    execgraph : :obj:`networkx.DiGraph`
        The execution graph, as returned by :meth:`nipype.pipeline.engine.Workflow.run`.
    default_dir : :obj:`os.PathLike`
        The log directory of nodes that are not specific to one participant.

    Returns
    -------
    groups : :obj:`dict`
        A mapping of log directories to the list of nodes that report to them.
        ``default_dir`` is only a key if no node reports to a different directory.

    """
    default_dir = Path(default_dir)
    shared = []
    groups = {}
    for node in execgraph.nodes():
        log_dir = Path((node.config or {}).get('execution', {}).get('crashdump_dir', default_dir))
        if log_dir == default_dir:
            shared.append(node)
        else:
            groups.setdefault(log_dir, []).append(node)

    if not groups:
        return {default_dir: shared}
    return {log_dir: shared + nodes for log_dir, nodes in groups.items()}


@contextmanager
def _known_user():
    """Make sure :func:`getpass.getuser` works, as *Nipype* needs it to describe results.

    It fails for users without an entry in the password database, which is common
    within containers (``docker run -u $(id -u)``).
    The name of the user is not recorded in any case.
    """

    try:
        getpass.getuser()
    except (OSError, KeyError):
        with mock.patch.dict(os.environ, {'LOGNAME': 'unknown'}):
            yield
    else:
        yield


@contextmanager
def _nipype_patches():
    """Adjust how *Nipype* describes results, for which it offers no options.

    * The checksum of each file is calculated only once. The same files (for instance,
      a BOLD series) are inputs or outputs of many nodes, and *Nipype* would read
      them in full every time one is described.
    * The :data:`REPRODUCIBILITY_ENVVARS` are recorded too.
    * Values *Nipype* does not encode well are converted first: booleans would be recorded
      as integers (`nipy/nipype#3817 <https://github.com/nipy/nipype/issues/3817>`__),
      a dictionary or list holding path objects as an error message
      (`nipy/nipype#3815 <https://github.com/nipy/nipype/issues/3815>`__),
      and a sequence of *NumPy* scalars would fail the description of the node
      (`nipy/nipype#3816 <https://github.com/nipy/nipype/issues/3816>`__).

    Should *Nipype* rename what is patched here, records are just built without these.
    """

    hash_infile = getattr(nprov, 'hash_infile', None)
    cache = {}

    def _hash_infile(afile, *args, **kwargs):
        try:
            stat = os.stat(afile)
        except OSError:
            return hash_infile(afile, *args, **kwargs)
        key = (os.fspath(afile), stat.st_mtime_ns, stat.st_size, args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = hash_infile(afile, *args, **kwargs)
        return cache[key]

    safe_encode = getattr(nprov, 'safe_encode', None)

    def _plain(x):
        if isinstance(x, np.generic | np.ndarray):
            return x.tolist()
        if isinstance(x, os.PathLike):
            return os.fspath(x)
        if isinstance(x, list | tuple):
            return type(x)(_plain(item) for item in x)
        if isinstance(x, dict):
            return {key: _plain(value) for key, value in x.items()}
        return x

    def _safe_encode(x, as_literal=True):
        x = _plain(x)
        if isinstance(x, bool):
            return pm.Literal(x, pm.XSD['boolean']) if as_literal else x
        return safe_encode(x, as_literal=as_literal)

    with ExitStack() as stack:
        if safe_encode is not None:
            stack.enter_context(mock.patch.object(nprov, 'safe_encode', _safe_encode))
        if hash_infile is not None:
            stack.enter_context(mock.patch.object(nprov, 'hash_infile', _hash_infile))
        if (envvars := getattr(nprov, 'PROV_ENVVARS', None)) is not None:
            stack.enter_context(
                mock.patch.object(nprov, 'PROV_ENVVARS', [*envvars, *REPRODUCIBILITY_ENVVARS])
            )
        yield


def _results(node):
    """Split the result of a node into the results of the interfaces it has run.

    Data sinks are given metadata when they are created, which they do not hold
    as inputs: they are described along with their inputs.
    """

    result = node.result
    metadata = getattr(node.interface, '_metadata', None)
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    if not isinstance(result.runtime, list):
        if metadata:
            result = copy(result)
            result.inputs = {**metadata, **(result.inputs or {})}
        return {'': result}

    # MapNode: one result per iteration
    results = {}
    for idx, runtime in enumerate(result.runtime):
        results[f'-{idx}'] = subresult = InterfaceResult(
            result.interface[idx], runtime, outputs={}
        )
        if result.inputs and idx < len(result.inputs):
            subresult.inputs = {**metadata, **result.inputs[idx]}
        for key, values in result.outputs.items() if result.outputs else ():
            if isdefined(values) and idx < len(values):
                subresult.outputs[key] = values[idx]
    return results


def _add_record(doc, records, node_id, suffix=''):
    """Add the records describing one interface result as a bundle of ``doc``.

    The description generated by *Nipype* is copied over, giving informative identifiers
    to what would otherwise have random ones, and leaving private information out.

    """

    def _qname(name):
        return doc.valid_qualified_name(f'bidsprov:{node_id}/{name}{suffix}')

    def _has(record, name, value=None):
        return any(key == name and value in (None, val) for key, val in record.attributes)

    # Decide what is renamed (random identifiers) and what is left out
    renamed = {}
    dropped = set()
    for record in records:
        if record.get_type() == pm.PROV_ACTIVITY:
            renamed[record.identifier] = _qname('interface')
        elif (
            _has(record, pm.PROV['type'], nipype_ns['Runtime'])
            or _has(record, pm.PROV['type'], pm.PROV['Person'])
            or any(_has(record, nipype_ns[stream]) for stream in _OUTPUT_STREAMS)
            or any(
                _has(record, nipype_ns['environmentVariable'], envvar)
                for envvar in _PRIVATE_ENVVARS
            )
        ):
            dropped.add(record.identifier)
        else:
            for kind in ('Inputs', 'Outputs', 'Environment'):
                if _has(record, pm.PROV['type'], nipype_ns[kind]):
                    renamed[record.identifier] = _qname(kind.lower())

    def _value(value):
        if isinstance(value, pm.QualifiedName):
            return renamed.get(value, value)
        if isinstance(value, pm.Identifier):  # How prov < 3.3 holds xsd:anyURI values
            return pm.Identifier(_FILE_HOST.sub('file:///', value.uri))
        # file://<hostname>/path -> file:///path, also within lists and dictionaries
        # of files, which Nipype records as text
        if isinstance(value, pm.Literal) and isinstance(value.value, str):
            return pm.Literal(
                _FILE_HOST.sub('file:///', value.value), value.datatype, value.langtag
            )
        if isinstance(value, str):
            return _FILE_HOST.sub('file:///', value)
        return value

    bundle = doc.bundle(_qname('record'))
    for record in records:
        if record.identifier in dropped or any(val in dropped for _, val in record.attributes):
            continue
        attributes = [(key, _value(val)) for key, val in record.attributes if key != foaf['host']]
        if record.get_type() == pm.PROV_ACTIVITY:
            attributes.append(
                (dcterms['isPartOf'], doc.valid_qualified_name(f'bidsprov:{node_id}'))
            )
        bundle.new_record(
            record.get_type(), renamed.get(record.identifier, record.identifier), attributes
        )
        if record.get_type() == pm.PROV_ENTITY:
            _add_members(bundle, record, _value)
    return bundle


def _listed_files(value):
    """Find the files within a list that *Nipype* recorded as text."""
    if not isinstance(value, str) or not value.lstrip().startswith('['):
        return []
    try:
        items = json.loads(value)
    except ValueError:
        return []

    def _flatten(item):
        if isinstance(item, list):
            for subitem in item:
                yield from _flatten(subitem)
        elif isinstance(item, str) and os.path.isfile(item):
            yield item

    return list(_flatten(items))


def _add_members(bundle, record, _value):
    """Describe the files of a list (e.g., a chain of transforms) as its members.

    *Nipype* records lists of several files as text, hence without their checksums.
    """

    files = [
        afile
        for key, val in record.attributes
        if key == pm.PROV_VALUE
        for afile in _listed_files(val)
    ]
    if not files:
        return
    bundle.collection(record.identifier)
    for afile in files:
        member = prov_encode(pm.ProvDocument(), afile)
        bundle.new_record(
            member.get_type(),
            member.identifier,
            [(key, _value(val)) for key, val in member.attributes],
        )
        bundle.hadMember(record.identifier, member.identifier)


def build_provenance(execgraph, run_uuid, *, version=None, command=None, ended=None):
    """Describe the execution of a workflow as a W3C PROV document.

    The document has one ``prov:Activity`` for the run, one for each (nested)
    workflow and one for each node, where:

    * workflows and nodes are related to the workflow (or run) they belong to
      with ``dcterms:isPartOf``;
    * nodes are related to the nodes they take inputs from with ``prov:wasInformedBy``;
    * the detailed record of each node (the interface that was run with its inputs,
      outputs, environment...) is kept in a ``prov:Bundle`` (that is, a named graph)
      which ``prov:wasGeneratedBy`` the node, and where the execution of the interface
      is an activity that is ``dcterms:isPartOf`` the node.

    Parameters
    ----------
    execgraph : :obj:`networkx.DiGraph`
        The execution graph (or a subgraph of it), as returned by
        :meth:`nipype.pipeline.engine.Workflow.run`.
    run_uuid : :obj:`str`
        The unique identifier of the run (``config.execution.run_uuid``).
    version : :obj:`str`
        Version of *fMRIPrep*.
    command : :obj:`list` of :obj:`str`
        Command line that started the run.
    ended : :obj:`datetime.datetime`
        Time the run ended (default: now).

    Returns
    -------
    doc : :obj:`prov.model.ProvDocument`
        The provenance document.

    """

    doc = ProvStore().g  # Comes with the namespaces Nipype's records use
    doc.add_namespace('bidsprov', PROV_NAMESPACE)

    def _qname(local):
        return doc.valid_qualified_name(f'bidsprov:{local}')

    # The run itself
    run_id = get_run_id(run_uuid)
    run_attrs = {pm.PROV['label']: run_id, dcterms['identifier']: str(run_uuid)}
    if command:
        run_attrs[nipype_ns['command']] = shlex.join(command)
    run = doc.activity(
        _qname(run_id), get_run_start(run_uuid), ended or datetime.now(UTC), run_attrs
    )
    agent_attrs = {pm.PROV['type']: pm.PROV['SoftwareAgent'], pm.PROV['label']: 'fMRIPrep'}
    if version:
        agent_attrs[nipype_ns['version']] = str(version)
    agent_id = f'fmriprep-{quote(str(version or "unknown"), safe="")}'
    doc.wasAssociatedWith(run, doc.agent(_qname(agent_id), agent_attrs))

    # Nested workflows
    workflows = {}

    def _workflow(hierarchy):
        if not hierarchy:
            return run.identifier
        if hierarchy not in workflows:
            attrs = {
                pm.PROV['type']: nipype_ns['Workflow'],
                pm.PROV['label']: hierarchy[-1],
                dcterms['isPartOf']: _workflow(hierarchy[:-1]),
            }
            wf_id = _qname(f'{run_id}/{quote(".".join(hierarchy), safe="")}')
            workflows[hierarchy] = doc.activity(wf_id, other_attributes=attrs).identifier
        return workflows[hierarchy]

    # Nodes
    activities = {}
    for node in execgraph.nodes():
        # The itername tells apart the expansions of iterables, unlike the fullname
        node_id = f'{run_id}/{quote(node.itername, safe="")}'
        *hierarchy, _ = node.fullname.split('.')
        attrs = [
            (pm.PROV['type'], nipype_ns[node.interface.__class__.__name__]),
            (pm.PROV['label'], node.name),
            (dcterms['isPartOf'], _workflow(tuple(hierarchy))),
        ]
        try:
            results = _results(node)
            with _known_user():
                records = {
                    suffix: ProvStore().add_results(result).get_records()
                    for suffix, result in results.items()
                }
            attrs.append((nipype_ns['hashval'], node.hash_exists()[1]))
        except Exception as exc:  # noqa: BLE001 - describe the other nodes anyway
            LOGGER.warning('No provenance record for node "%s": %s', node.fullname, exc)
            activities[node] = doc.activity(_qname(node_id), other_attributes=attrs)
            continue

        runtimes = [result.runtime for result in results.values()]
        attrs.append((pm.PROV['type'], nipype_ns['Node' if '' in results else 'MapNode']))
        attrs += [
            (nipype_ns['command'], runtime.cmdline)
            for runtime in runtimes
            if getattr(runtime, 'cmdline', None)
        ]
        start_times = [t for t in (getattr(r, 'startTime', None) for r in runtimes) if t]
        end_times = [t for t in (getattr(r, 'endTime', None) for r in runtimes) if t]
        activities[node] = doc.activity(
            _qname(node_id), min(start_times, default=None), max(end_times, default=None), attrs
        )
        for suffix, node_records in records.items():
            bundle = _add_record(doc, node_records, node_id, suffix)
            entity = doc.entity(bundle.identifier, {pm.PROV['type']: pm.PROV_BUNDLE})
            doc.wasGeneratedBy(entity, activities[node])

    # Dependencies
    for source, target in execgraph.edges():
        doc.wasInformedBy(activities[target], activities[source])

    return doc


def to_jsonld(doc):
    """Serialize a provenance document as PROV-O in JSON-LD.

    Returns
    -------
    jsonld : :obj:`dict`
        A JSON-LD document where the top-level ``@graph`` holds the default graph
        (the run, the workflows and the nodes) and one named graph per bundle.

    """
    jsonld = json.loads(
        doc.serialize(
            format='rdf', rdf_format='json-ld', context=JSONLD_CONTEXT, auto_compact=True
        )
    )
    # rdflib wraps the default graph in an object with ``@graph`` but without ``@id``.
    # Conforming JSON-LD processors take that as a graph named by a blank node,
    # so the default graph would be empty for them.
    graph = []
    for item in jsonld.get('@graph', []):
        if '@graph' in item and '@id' not in item:
            graph += item['@graph']
        else:
            graph.append(item)
    jsonld['@graph'] = graph
    return jsonld


def write_provenance(execgraph, default_dir, run_uuid, *, version=None, command=None):
    """Write the provenance of an executed workflow, one JSON-LD file per log directory.

    Each file (named ``provenance.jsonld``) describes the nodes that report to that
    log directory (see :func:`group_nodes` and :func:`build_provenance`).
    Because identifiers of the run, workflows and nodes are derived from their names,
    what is shared across the participants of a run is identified consistently
    across their files.

    Parameters
    ----------
    execgraph : :obj:`networkx.DiGraph`
        The execution graph, as returned by :meth:`nipype.pipeline.engine.Workflow.run`.
    default_dir : :obj:`os.PathLike`
        The log directory of nodes that are not specific to one participant.
    run_uuid : :obj:`str`
        The unique identifier of the run (``config.execution.run_uuid``).
    version : :obj:`str`
        Version of *fMRIPrep*.
    command : :obj:`list` of :obj:`str`
        Command line that started the run.

    Returns
    -------
    out_files : :obj:`list` of :obj:`pathlib.Path`
        The files that were written.
    errors : :obj:`dict`
        The exceptions raised for the log directories whose file could not be written.

    """

    ended = datetime.now(UTC)
    out_files = []
    errors = {}
    with _nipype_patches():
        for log_dir, nodes in group_nodes(execgraph, default_dir).items():
            out_file = log_dir / PROV_FILENAME
            tmp_file = out_file.with_name(f'.{PROV_FILENAME}.tmp')
            try:
                doc = build_provenance(
                    execgraph.subgraph(nodes),
                    run_uuid,
                    version=version,
                    command=command,
                    ended=ended,
                )
                log_dir.mkdir(exist_ok=True, parents=True)
                tmp_file.write_text(json.dumps(to_jsonld(doc), indent=1) + '\n')
                tmp_file.replace(out_file)
            except Exception as exc:  # noqa: BLE001 - one participant must not affect the others
                with suppress(OSError):
                    tmp_file.unlink()
                errors[log_dir] = exc
            else:
                out_files.append(out_file)
    return out_files, errors
