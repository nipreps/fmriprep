from copy import deepcopy

import bids
from niworkflows.utils.testing import generate_bids_skeleton
from sdcflows.fieldmaps import clear_registry
from sdcflows.utils.wrangler import find_estimators

from ..base import get_estimator

BASE_LAYOUT = {
    "01": {
        "anat": [{"suffix": "T1w"}],
        "func": [
            {
                "task": "rest",
                "run": i,
                "suffix": "bold",
                "metadata": {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.6},
            }
            for i in range(1, 3)
        ],
        "fmap": [
            {"suffix": "phasediff", "metadata": {"EchoTime1": 0.005, "EchoTime2": 0.007}},
            {"suffix": "magnitude1", "metadata": {"EchoTime": 0.005}},
            {
                "suffix": "epi",
                "direction": "PA",
                "metadata": {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.6},
            },
            {
                "suffix": "epi",
                "direction": "AP",
                "metadata": {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.6},
            },
        ],
    },
}


def test_get_estimator_none(tmp_path):
    bids_dir = tmp_path / "bids"

    # No IntendedFors/B0Fields
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    layout = bids.BIDSLayout(bids_dir)
    bold_files = sorted(layout.get(suffix='bold', extension='.nii.gz', return_type='file'))

    assert get_estimator(layout, bold_files[0]) == ()
    assert get_estimator(layout, bold_files[1]) == ()


def test_get_estimator_b0field_and_intendedfor(tmp_path):
    bids_dir = tmp_path / "bids"

    # Set B0FieldSource for run 1
    spec = deepcopy(BASE_LAYOUT)
    spec['01']['func'][0]['metadata']['B0FieldSource'] = 'epi'
    spec['01']['fmap'][2]['metadata']['B0FieldIdentifier'] = 'epi'
    spec['01']['fmap'][3]['metadata']['B0FieldIdentifier'] = 'epi'

    # Set IntendedFor for run 2
    spec['01']['fmap'][0]['metadata']['IntendedFor'] = 'func/sub-01_task-rest_run-2_bold.nii.gz'

    generate_bids_skeleton(bids_dir, spec)
    layout = bids.BIDSLayout(bids_dir)
    _ = find_estimators(layout=layout, subject='01')

    bold_files = sorted(layout.get(suffix='bold', extension='.nii.gz', return_type='file'))

    assert get_estimator(layout, bold_files[0]) == ('epi',)
    assert get_estimator(layout, bold_files[1]) == ('auto_00000',)
    clear_registry()


def test_get_estimator_overlapping_specs(tmp_path):
    bids_dir = tmp_path / "bids"

    # Set B0FieldSource for both runs
    spec = deepcopy(BASE_LAYOUT)
    spec['01']['func'][0]['metadata']['B0FieldSource'] = 'epi'
    spec['01']['func'][1]['metadata']['B0FieldSource'] = 'epi'
    spec['01']['fmap'][2]['metadata']['B0FieldIdentifier'] = 'epi'
    spec['01']['fmap'][3]['metadata']['B0FieldIdentifier'] = 'epi'

    # Set IntendedFor for both runs
    spec['01']['fmap'][0]['metadata']['IntendedFor'] = [
        'func/sub-01_task-rest_run-1_bold.nii.gz',
        'func/sub-01_task-rest_run-2_bold.nii.gz',
    ]

    generate_bids_skeleton(bids_dir, spec)
    layout = bids.BIDSLayout(bids_dir)
    _ = find_estimators(layout=layout, subject='01')

    bold_files = sorted(layout.get(suffix='bold', extension='.nii.gz', return_type='file'))

    # B0Fields take precedence
    assert get_estimator(layout, bold_files[0]) == ('epi',)
    assert get_estimator(layout, bold_files[1]) == ('epi',)
    clear_registry()


def test_get_estimator_multiple_b0fields(tmp_path):
    bids_dir = tmp_path / "bids"

    # Set B0FieldSource for both runs
    spec = deepcopy(BASE_LAYOUT)
    spec['01']['func'][0]['metadata']['B0FieldSource'] = ('epi', 'phasediff')
    spec['01']['func'][1]['metadata']['B0FieldSource'] = 'epi'
    spec['01']['fmap'][0]['metadata']['B0FieldIdentifier'] = 'phasediff'
    spec['01']['fmap'][1]['metadata']['B0FieldIdentifier'] = 'phasediff'
    spec['01']['fmap'][2]['metadata']['B0FieldIdentifier'] = 'epi'
    spec['01']['fmap'][3]['metadata']['B0FieldIdentifier'] = 'epi'

    generate_bids_skeleton(bids_dir, spec)
    layout = bids.BIDSLayout(bids_dir)
    _ = find_estimators(layout=layout, subject='01')

    bold_files = sorted(layout.get(suffix='bold', extension='.nii.gz', return_type='file'))

    # Always get an iterable; don't care if it's a list or tuple
    assert get_estimator(layout, bold_files[0]) == ['epi', 'phasediff']
    assert get_estimator(layout, bold_files[1]) == ('epi',)
    clear_registry()
