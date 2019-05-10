# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Stripped out routines for Sentry"""
import os
from pathlib import Path
import re
from niworkflows.utils.misc import read_crashfile
import sentry_sdk

CHUNK_SIZE = 16384
# Group common events with pre specified fingerprints
KNOWN_ERRORS = {
    'permission-denied': [
        "PermissionError: [Errno 13] Permission denied"
    ],
    'memory-error': [
        "MemoryError",
        "Cannot allocate memory",
        "Return code: 134",
    ],
    'reconall-already-running': [
        "ERROR: it appears that recon-all is already running"
    ],
    'no-disk-space': [
        "[Errno 28] No space left on device",
        "[Errno 122] Disk quota exceeded"
    ],
    'segfault': [
        "Segmentation Fault",
        "Segfault",
        "Return code: 139",
    ],
    'potential-race-condition': [
        "[Errno 39] Directory not empty",
        "_unfinished.json",
    ],
    'keyboard-interrupt': [
        "KeyboardInterrupt",
    ],
}


def start_ping(run_uuid, npart):
    with sentry_sdk.configure_scope() as scope:
        if run_uuid:
            scope.set_tag('run_uuid', run_uuid)
        scope.set_tag('npart', npart)
    sentry_sdk.add_breadcrumb(message='fMRIPrep started', level='info')
    sentry_sdk.capture_message('fMRIPrep started', level='info')


def sentry_setup(opts, exec_env):
    from os import cpu_count
    import psutil
    import hashlib
    from ..__about__ import __version__

    environment = "prod"
    release = __version__
    if not __version__:
        environment = "dev"
        release = "dev"
    elif int(os.getenv('FMRIPREP_DEV', '0')) or ('+' in __version__):
        environment = "dev"

    sentry_sdk.init("https://d5a16b0c38d84d1584dfc93b9fb1ade6@sentry.io/1137693",
                    release=release,
                    environment=environment,
                    before_send=before_send)
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag('exec_env', exec_env)

        if exec_env == 'fmriprep-docker':
            scope.set_tag('docker_version', os.getenv('DOCKER_VERSION_8395080871'))

        dset_desc_path = opts.bids_dir / 'dataset_description.json'
        if dset_desc_path.exists():
            desc_content = dset_desc_path.read_bytes()
            scope.set_tag('dset_desc_sha256', hashlib.sha256(desc_content).hexdigest())

        free_mem_at_start = round(psutil.virtual_memory().free / 1024**3, 1)
        scope.set_tag('free_mem_at_start', free_mem_at_start)
        scope.set_tag('cpu_count', cpu_count())

        # Memory policy may have a large effect on types of errors experienced
        overcommit_memory = Path('/proc/sys/vm/overcommit_memory')
        if overcommit_memory.exists():
            policy = {'0': 'heuristic',
                      '1': 'always',
                      '2': 'never'}.get(overcommit_memory.read_text().strip(), 'unknown')
            scope.set_tag('overcommit_memory', policy)
            if policy == 'never':
                overcommit_kbytes = Path('/proc/sys/vm/overcommit_memory')
                kb = overcommit_kbytes.read_text().strip()
                if kb != '0':
                    limit = '{}kB'.format(kb)
                else:
                    overcommit_ratio = Path('/proc/sys/vm/overcommit_ratio')
                    limit = '{}%'.format(overcommit_ratio.read_text().strip())
                scope.set_tag('overcommit_limit', limit)
            else:
                scope.set_tag('overcommit_limit', 'n/a')
        else:
            scope.set_tag('overcommit_memory', 'n/a')
            scope.set_tag('overcommit_limit', 'n/a')

        for k, v in vars(opts).items():
            scope.set_tag(k, v)


def process_crashfile(crashfile):
    """Parse the contents of a crashfile and submit sentry messages"""
    crash_info = read_crashfile(str(crashfile))
    with sentry_sdk.push_scope() as scope:
        scope.level = 'fatal'

        # Extract node name
        node_name = crash_info.pop('node').split('.')[-1]
        scope.set_tag("node_name", node_name)

        # Massage the traceback, extract the gist
        traceback = crash_info.pop('traceback')
        # last line is probably most informative summary
        gist = traceback.splitlines()[-1]
        exception_text_start = 1
        for line in traceback.splitlines()[1:]:
            if not line[0].isspace():
                break
            exception_text_start += 1

        exception_text = '\n'.join(
            traceback.splitlines()[exception_text_start:])

        # Extract inputs, if present
        inputs = crash_info.pop('inputs', None)
        if inputs:
            scope.set_extra('inputs', dict(inputs))

        # Extract any other possible metadata in the crash file
        for k, v in crash_info.items():
            strv = list(_chunks(str(v)))
            if len(strv) == 1:
                scope.set_extra(k, strv[0])
            else:
                for i, chunk in enumerate(strv):
                    scope.set_extra('%s_%02d' % (k, i), chunk)

        fingerprint = ''
        issue_title = '{}: {}'.format(node_name, gist)
        for new_fingerprint, error_snippets in KNOWN_ERRORS.items():
            for error_snippet in error_snippets:
                if error_snippet in traceback:
                    fingerprint = new_fingerprint
                    issue_title = new_fingerprint
                    break
            if fingerprint:
                break

        message = issue_title + '\n\n'
        message += exception_text[-(8192 - len(message)):]
        if fingerprint:
            sentry_sdk.add_breadcrumb(message=fingerprint, level='fatal')
        else:
            # remove file paths
            fingerprint = re.sub(r"(/[^/ ]*)+/?", '', message)
            # remove words containing numbers
            fingerprint = re.sub(r"([a-zA-Z]*[0-9]+[a-zA-Z]*)+", '', fingerprint)
            # adding the return code if it exists
            for line in message.splitlines():
                if line.startswith("Return code"):
                    fingerprint += line
                    break

        scope.fingerprint = [fingerprint]
        sentry_sdk.capture_message(message, 'fatal')


def before_send(event, hints):
    # Filtering log messages about crashed nodes
    if 'logentry' in event and 'message' in event['logentry']:
        msg = event['logentry']['message']
        if msg.startswith("could not run node:"):
            return None
        if msg.startswith("Saving crash info to "):
            return None
        if re.match("Node .+ failed to run on host .+", msg):
            return None

    if 'breadcrumbs' in event and isinstance(event['breadcrumbs'], list):
        fingerprints_to_propagate = ['no-disk-space', 'memory-error', 'permission-denied',
                                     'keyboard-interrupt']
        for bc in event['breadcrumbs']:
            msg = bc.get('message', 'empty-msg')
            if msg in fingerprints_to_propagate:
                event['fingerprint'] = [msg]
                break

    return event


def _chunks(string, length=CHUNK_SIZE):
    """
    Splits a string into smaller chunks

    >>> list(_chunks('some longer string.', length=3))
    ['som', 'e l', 'ong', 'er ', 'str', 'ing', '.']

    """
    return (string[i:i + length]
            for i in range(0, len(string), length))
