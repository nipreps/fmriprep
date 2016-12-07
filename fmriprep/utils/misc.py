import copy
from glob import glob
import itertools
import json
import os
from os import path as op
from errno import EEXIST
import re

from bids.grabbids import BIDSLayout

INPUTS_SPEC = {'fieldmaps': [], 'func': [], 't1': [], 'sbref': []}


def genfname(in_file, suffix=None, path=None, ext=None):
    from os import getcwd
    import os.path as op

    fname, fext = op.splitext(op.basename(in_file))
    if fext == '.gz':
        fname, fext2 = op.splitext(fname)
        fext = fext2 + fext

    if path is None:
        path = getcwd()

    if ext is None:
        ext = fext

    if ext.startswith('.'):
        ext = ext[1:]

    if suffix is None:
        suffix = 'mod'

    return op.join(path, '{}_{}.{}'.format(fname, suffix, ext))

def _first(inlist):
    if not isinstance(inlist, (list, tuple)):
        inlist = [inlist]

    return sorted(inlist)[0]

def make_folder(folder):
    try:
        os.makedirs(folder)
    except OSError as exc:
        if exc.errno == EEXIST:
            pass
    return folder

def gen_list(inlist, base=1):
    return range(base, len(inlist) + base)

def _walk_dir_for_prefix(target_dir, prefix):
    return [x for x in next(os.walk(target_dir))[1]
            if x.startswith(prefix)]

def is_fieldmap_file(string):
    is_fieldmap_file = False
    for suffix in fieldmap_suffixes.values():
        if re.search(suffix, string):
            is_fieldmap_file = True
    return is_fieldmap_file


fieldmap_suffixes = {
    'phasediff': r"phasediff[0-9]*\.nii(\.gz)?",
    'magnitude': r"magnitude[0-9]*\.nii(\.gz)?",
    'phase': r"phase[0-9]+\.nii(\.gz)?",
    'fieldmap': r"fieldmap\.nii(\.gz)?",
    'topup': r"epi\.nii(\.gz)?"
}


def collect_bids_data(dataset, subject, task=None, session=None, run=None):
    subject = str(subject)
    if subject.startswith('sub-'):
        subject = subject[4:]

    layout = BIDSLayout(dataset)

    if session:
        session_list = [session]
    else:
        session_list = layout.unique('session')
        if session_list == []:
            session_list = [None]

    if run:
        run_list = [run]
    else:
        run_list = layout.unique('run')
        if run_list == []:
            run_list = [None]

    queries = {
        'fmap': {'modality': 'fmap', 'ext': 'nii'},
        'epi': {'modality': 'func', 'type': 'bold', 'ext': 'nii'},
        'sbref': {'modality': 'func', 'type': 'sbref', 'ext': 'nii'},
        't1w': {'type': 'T1w', 'ext': 'nii'}
    }

    if task:
        queries['epi']['task'] = task

    #  Add a subject key pair to each query we make so that we only deal with
    #  files related to this workflows specific subject. Could be made opt...
    for key in queries.keys():
        queries[key]['subject'] = subject

    imaging_data = copy.deepcopy(INPUTS_SPEC)
    fieldmap_files = [x.filename for x in layout.get(**queries['fmap'])]
    imaging_data['fmap'] = fieldmap_files
    t1_files = [x.filename for x in layout.get(**queries['t1w'])]
    imaging_data['t1w'] = t1_files
    sbref_files = [x.filename for x in layout.get(**queries['sbref'])]
    imaging_data['sbref'] = sbref_files
    epi_files = [x.filename for x in layout.get(**queries['epi'])]
    imaging_data['func'] = epi_files

    '''
    loop_on = ['session', 'run', 'acquisition', 'task']
    get_kwargs = {}

    for key in loop_on:
        unique_list = layout.unique(key)
        if unique_list:
            get_kwargs[key] = unique_list

    query_kwargs = []
    for key in get_kwargs:
        query_kwargs.append([(key, x) for x in get_kwargs[key]])

    query_kwargs = itertools.product(*query_kwargs)

    for elem in query_kwargs:
        epi_files = [x.filename for x
                     in layout.get(**dict(dict(elem), **queries['epi']))]
        if epi_files:
            imaging_data['func'] += epi_files
    '''

    return imaging_data


def get_biggest_epi_file_size_gb(files):
    max_size = 0
    for file in files:
        size = os.path.getsize(file)/(1024*1024*1024)
        if size > max_size:
            max_size = size
    return max_size

if __name__ == '__main__':
    pass
