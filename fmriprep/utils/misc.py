#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals

import copy
from glob import glob
import itertools
import json
import os
from os import path as op
from errno import EEXIST
import re

from bids.grabbids import BIDSLayout

INPUTS_SPEC = {'fieldmaps': [], 'func': [], 't1': [], 'sbref': [], 't2w': []}


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

    if ext != '' and not ext.startswith('.'):
        ext = '.' + ext

    if suffix is None:
        suffix = 'mod'

    return op.join(path, '{}_{}{}'.format(fname, suffix, ext))

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
        'fmap': {'modality': 'fmap', 'extensions': ['nii', 'nii.gz']},
        'epi': {'modality': 'func', 'type': 'bold', 'extensions': ['nii', 'nii.gz']},
        'sbref': {'modality': 'func', 'type': 'sbref', 'extensions': ['nii', 'nii.gz']},
        't1w': {'type': 'T1w', 'extensions': ['nii', 'nii.gz']},
        't2w': {'type': 'T2w', 'extensions': ['nii', 'nii.gz']},
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
    t2_files = [x.filename for x in layout.get(**queries['t2w'])]
    imaging_data['t2w'] = t2_files

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

def fix_multi_T1w_source_name(in_files):
    import os
    # in case there are multiple T1s we make up a generic source name
    if isinstance(in_files, list):
        subject_label = in_files[0].split(os.sep)[-1].split("_")[0].split("-")[
            -1]
        base, _ = os.path.split(in_files[0])
        return os.path.join(base, "sub-%s_T1w.nii.gz" % subject_label)
    else:
        return in_files


def _extract_wm(in_file):
    import os.path as op
    import nibabel as nb
    import numpy as np

    image = nb.load(in_file)
    data = image.get_data().astype(np.uint8)
    data[data != 3] = 0
    data[data > 0] = 1

    out_file = op.abspath('wm_mask.nii.gz')
    nb.Nifti1Image(data, image.get_affine(), image.get_header()).to_filename(out_file)
    return out_file


if __name__ == '__main__':
    pass
