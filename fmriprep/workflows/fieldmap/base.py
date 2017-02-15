#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base fieldmap handling utilities.

Possible fieldmap specifications according to BIDS 1.0.0
--------------------------------------------------------

8.9.1 one phase diff image, at least one magnitude image
8.9.2 two phase images, two magnitude images
8.9.3 fieldmap image (and one magnitude image)
8.9.4 multiple phase encoded directions (topup)

"""
from __future__ import print_function, division, absolute_import, unicode_literals
from future.utils import raise_from

from nipype import logging

LOGGER = logging.getLogger('workflow')


def is_fmap_type(fmap_type, filename):
    from fmriprep.utils import misc
    import re
    return re.search(misc.fieldmap_suffixes[fmap_type], filename)


def sort_fmaps(fieldmaps):  # i.e. filenames
    from fmriprep.utils import misc
    from fmriprep.workflows.fieldmap.base import is_fmap_type
    fmaps = {}
    for fmap_type in misc.fieldmap_suffixes.keys():
        fmaps[fmap_type] = []
        fmaps[fmap_type] = [doc for doc in fieldmaps
                            if is_fmap_type(fmap_type, doc)]
    return fmaps


def fieldmap_decider(fieldmap_data, settings):
    ''' Run fieldmap_decider to automatically find a
    Fieldmap preprocessing workflow '''
    # inputs = { 'fieldmaps': None }
    # outputs = { 'fieldmaps': None,
    #             'outputnode.mag_brain': None,
    #             'outputnode.fmap_mask': None,
    #             'outputnode.fmap_fieldcoef': None,
    #             'outputnode.fmap_movpar': None}

    try:
        fieldmap_data[0]
    except IndexError as e:
        raise_from(NotImplementedError("No fieldmap data found"), e)

    for filename in subject_data['fieldmaps']:
        if is_fmap_type('phasediff', filename):  # 8.9.1
            from fmriprep.workflows.fieldmap.phase_diff_and_magnitudes import phase_diff_and_magnitudes
            return phase_diff_and_magnitudes(settings)
        elif is_fmap_type('phase', filename):  # 8.9.2
            raise NotImplementedError("No workflow for phase fieldmap data")
        elif is_fmap_type('fieldmap', filename):  # 8.9.3
            from fmriprep.workflows.fieldmap.fieldmap_to_phasediff import fieldmap_to_phasediff
            return fieldmap_to_phasediff(settings=settings)
        elif is_fmap_type('topup', filename):  # 8.0.4
            from fmriprep.workflows.fieldmap.se_fmap_workflow import se_fmap_workflow
            return se_fmap_workflow(settings=settings)

    raise IOError("Unrecognized fieldmap structure")
