#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to handle BIDS inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch some test data

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

"""
import os
import re
import os.path as op
import warnings
from itertools import groupby
from bids.grabbids import BIDSLayout


class BIDSError(ValueError):
    def __init__(self, message, bids_root):
        indent = 10
        header = '{sep} BIDS root folder: "{bids_root}" {sep}'.format(
            bids_root=bids_root, sep=''.join(['-'] * indent))
        self.msg = '\n{header}\n{indent}{message}\n{footer}'.format(
            header=header, indent=''.join([' '] * (indent + 1)),
            message=message, footer=''.join(['-'] * len(header))
        )
        super(BIDSError, self).__init__(self.msg)
        self.bids_root = bids_root


class BIDSWarning(RuntimeWarning):
    pass


def collect_participants(bids_dir, participant_label=None, strict=False):
    """
    List the participants under the BIDS root and checks that participants
    designated with the participant_label argument exist in that folder.

    Returns the list of participants to be finally processed.

    Requesting all subjects in a BIDS directory root:

    >>> collect_participants('ds114')
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    Requesting two subjects, given their IDs:

    >>> collect_participants('ds114', participant_label=['02', '04'])
    ['02', '04']

    Requesting two subjects, given their IDs (works with 'sub-' prefixes):

    >>> collect_participants('ds114', participant_label=['sub-02', 'sub-04'])
    ['02', '04']

    Requesting two subjects, but one does not exist:

    >>> collect_participants('ds114', participant_label=['02', '14'])
    ['02']

    >>> collect_participants('ds114', participant_label=['02', '14'],
    ...                      strict=True)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    fmriprep.utils.bids.BIDSError:
    ...


    """
    bids_dir = op.abspath(bids_dir)
    all_participants = sorted(
        [subdir[4:] for subdir in os.listdir(bids_dir)
         if op.isdir(op.join(bids_dir, subdir)) and subdir.startswith('sub-')])

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            'Could not find participants. Please make sure the BIDS data '
            'structure is present and correct. Datasets can be validated online '
            'using the BIDS Validator (http://incf.github.io/bids-validator/).\n'
            'If you are using Docker for Mac or Docker for Windows, you '
            'may need to adjust your "File sharing" preferences.', bids_dir)

    # No --participant-label was set, return all
    if participant_label is None or not participant_label:
        return all_participants

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [sub[4:] if sub.startswith('sub-') else sub for sub in participant_label]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & set(all_participants))
    if not found_label:
        raise BIDSError('Could not find participants [{}]'.format(
            ', '.join(participant_label)), bids_dir)

    # Warn if some IDs were not found
    notfound_label = sorted(set(participant_label) - set(all_participants))
    if notfound_label:
        exc = BIDSError('Some participants were not found: {}'.format(
            ', '.join(notfound_label)), bids_dir)
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning)

    return found_label


def collect_data(dataset, participant_label, task=None):
    """
    Uses grabbids to retrieve the input data for a given participant

    >>> bids_root, _ = collect_data('ds054', '100185')
    >>> bids_root['fmap']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/fmap/sub-100185_magnitude1.nii.gz', \
'.../ds054/sub-100185/fmap/sub-100185_magnitude2.nii.gz', \
'.../ds054/sub-100185/fmap/sub-100185_phasediff.nii.gz']

    >>> bids_root['bold']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/func/sub-100185_task-machinegame_run-01_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-02_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-03_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-04_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-05_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-06_bold.nii.gz']

    >>> bids_root['sbref']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/func/sub-100185_task-machinegame_run-01_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-02_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-03_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-04_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-05_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-06_sbref.nii.gz']

    >>> bids_root['t1w']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/anat/sub-100185_T1w.nii.gz']

    >>> bids_root['t2w']  # doctest: +ELLIPSIS
    []


    """
    layout = BIDSLayout(dataset, exclude=['derivatives', 'sourcedata'])
    queries = {
        'fmap': {'subject': participant_label, 'modality': 'fmap',
                 'extensions': ['nii', 'nii.gz']},
        'bold': {'subject': participant_label, 'modality': 'func', 'type': 'bold',
                 'extensions': ['nii', 'nii.gz']},
        'sbref': {'subject': participant_label, 'modality': 'func', 'type': 'sbref',
                  'extensions': ['nii', 'nii.gz']},
        'flair': {'subject': participant_label, 'modality': 'anat', 'type': 'FLAIR',
                  'extensions': ['nii', 'nii.gz']},
        't2w': {'subject': participant_label, 'modality': 'anat', 'type': 'T2w',
                'extensions': ['nii', 'nii.gz']},
        't1w': {'subject': participant_label, 'modality': 'anat', 'type': 'T1w',
                'extensions': ['nii', 'nii.gz']},
        'roi': {'subject': participant_label, 'modality': 'anat', 'type': 'roi',
                'extensions': ['nii', 'nii.gz']},
    }

    if task:
        queries['bold']['task'] = task

    subj_data = {modality: [x.filename for x in layout.get(**query)]
                 for modality, query in queries.items()}

    def _grp_echos(x):
        if '_echo-' not in x:
            return x
        echo = re.search("_echo-\\d*", x).group(0)
        return x.replace(echo, "_echo-?")

    if subj_data["bold"] is not []:
        bold_sess = subj_data["bold"]

        if any(['_echo-' in bold for bold in bold_sess]):
            ses_uids = [list(bold) for _, bold in groupby(bold_sess, key=_grp_echos)]
            ses_uids = [x[0] if len(x) == 1 else x for x in ses_uids]
        else:
            ses_uids = bold_sess

    subj_data.update({"bold": ses_uids})

    return subj_data, layout
