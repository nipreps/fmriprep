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
import json
from pathlib import Path


def write_derivative_description(bids_dir, deriv_dir):
    from ..__about__ import __version__, __url__, DOWNLOAD_URL

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'Name': 'fMRIPrep - fMRI PREProcessing workflow',
        'BIDSVersion': '1.1.1',
        'PipelineDescription': {
            'Name': 'fMRIPrep',
            'Version': __version__,
            'CodeURL': DOWNLOAD_URL,
        },
        'CodeURL': __url__,
        'HowToAcknowledge':
            'Please cite our paper (https://doi.org/10.1038/s41592-018-0235-4), '
            'and include the generated citation boilerplate within the Methods '
            'section of the text.',
    }

    # Keys that can only be set by environment
    if 'FMRIPREP_DOCKER_TAG' in os.environ:
        desc['DockerHubContainerTag'] = os.environ['FMRIPREP_DOCKER_TAG']
    if 'FMRIPREP_SINGULARITY_URL' in os.environ:
        singularity_url = os.environ['FMRIPREP_SINGULARITY_URL']
        desc['SingularityContainerURL'] = singularity_url

        singularity_md5 = _get_shub_version(singularity_url)
        if singularity_md5 and singularity_md5 is not NotImplemented:
            desc['SingularityContainerMD5'] = _get_shub_version(singularity_url)

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        with fname.open() as fobj:
            orig_desc = json.load(fobj)

    if 'DatasetDOI' in orig_desc:
        desc['SourceDatasetsURLs'] = ['https://doi.org/{}'.format(
            orig_desc['DatasetDOI'])]
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    with (deriv_dir / 'dataset_description.json').open('w') as fobj:
        json.dump(desc, fobj, indent=4)


def _get_shub_version(singularity_url):
    return NotImplemented
