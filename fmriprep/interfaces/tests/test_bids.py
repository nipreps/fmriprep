"""Tests for fmriprep.interfaces.bids."""

import pytest


def test_BIDSURI():
    """Test the BIDSURI interface."""
    from fmriprep.interfaces.bids import BIDSURI

    dataset_links = {
        'raw': '/data',
        'deriv-0': '/data/derivatives/source-1',
    }
    out_dir = '/data/derivatives/fmriprep'

    # A single element as a string
    interface = BIDSURI(
        numinputs=1,
        dataset_links=dataset_links,
        out_dir=out_dir,
    )
    interface.inputs.in1 = '/data/sub-01/func/sub-01_task-rest_bold.nii.gz'
    results = interface.run()
    assert results.outputs.out == ['bids:raw:sub-01/func/sub-01_task-rest_bold.nii.gz']

    # A single element as a list
    interface = BIDSURI(
        numinputs=1,
        dataset_links=dataset_links,
        out_dir=out_dir,
    )
    interface.inputs.in1 = ['/data/sub-01/func/sub-01_task-rest_bold.nii.gz']
    results = interface.run()
    assert results.outputs.out == ['bids:raw:sub-01/func/sub-01_task-rest_bold.nii.gz']

    # Two inputs: a string and a list
    interface = BIDSURI(
        numinputs=2,
        dataset_links=dataset_links,
        out_dir=out_dir,
    )
    interface.inputs.in1 = '/data/sub-01/func/sub-01_task-rest_bold.nii.gz'
    interface.inputs.in2 = [
        '/data/derivatives/source-1/sub-01/func/sub-01_task-rest_bold.nii.gz',
        '/out/sub-01/func/sub-01_task-rest_bold.nii.gz',
    ]
    results = interface.run()
    assert results.outputs.out == [
        'bids:raw:sub-01/func/sub-01_task-rest_bold.nii.gz',
        'bids:deriv-0:sub-01/func/sub-01_task-rest_bold.nii.gz',
        '/out/sub-01/func/sub-01_task-rest_bold.nii.gz',  # No change
    ]

    # Two inputs as lists
    interface = BIDSURI(
        numinputs=2,
        dataset_links=dataset_links,
        out_dir=out_dir,
    )
    interface.inputs.in1 = [
        '/data/sub-01/func/sub-01_task-rest_bold.nii.gz',
        'bids:raw:sub-01/func/sub-01_task-rest_boldref.nii.gz',
    ]
    interface.inputs.in2 = [
        '/data/derivatives/source-1/sub-01/func/sub-01_task-rest_bold.nii.gz',
        '/out/sub-01/func/sub-01_task-rest_bold.nii.gz',
    ]
    results = interface.run()
    assert results.outputs.out == [
        'bids:raw:sub-01/func/sub-01_task-rest_bold.nii.gz',
        'bids:raw:sub-01/func/sub-01_task-rest_boldref.nii.gz',  # No change
        'bids:deriv-0:sub-01/func/sub-01_task-rest_bold.nii.gz',
        '/out/sub-01/func/sub-01_task-rest_bold.nii.gz',  # No change
    ]


bids_infos_anat = [
    [{'t1w': ['sub-01/anat/sub-01_T1w.nii.gz']}, {}],
    [{'t2w': ['sub-01/anat/sub-01_T2w.nii.gz']}, {}],
    [
        {'t1w': []},
        {'t1w_preproc': ['sourcedata/smriprep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz']},
    ],
    [
        {'t2w': []},
        {'t2w_preproc': ['sourcedata/smriprep/sub-01/anat/sub-01_desc-preproc_T2w.nii.gz']},
    ],
]

bids_infos_func = {
    'single-echo': {
        'bold': ['sub-01/func/sub-01_task-rest_bold.nii.gz'],
    },
    'multi-echo': {
        'bold': [
            [
                'sub-01/func/sub-01_task-rest_echo-1_bold.nii.gz',
                'sub-01/func/sub-01_task-rest_echo-2_bold.nii.gz',
                'sub-01/func/sub-01_task-rest_echo-3_bold.nii.gz',
            ],
        ],
    },
}


@pytest.mark.parametrize(
    ('bids_info_anat', 'bids_info_func', 'precomputed_infos'),
    [
        (bids_info_anat, bids_info_func, precomputed_infos)
        for bids_info_anat, precomputed_infos in bids_infos_anat
        for _, bids_info_func in bids_infos_func.items()
    ],
)
def test_BIDSSourceFile(bids_info_anat, bids_info_func, precomputed_infos):
    """Test the BIDSSourceFile interface"""
    from fmriprep.interfaces.bids import BIDSSourceFile

    interface = BIDSSourceFile()
    anat_type = next(iter(bids_info_anat))
    interface.inputs.anat_type = anat_type
    interface.inputs.bids_info = {**bids_info_anat, **bids_info_func}
    interface.inputs.precomputed = precomputed_infos
    results = interface.run()

    if precomputed_infos:
        assert results.outputs.source_file == 'sub-01/func/sub-01_task-rest_bold.nii.gz'
    else:
        assert results.outputs.source_file == bids_info_anat[anat_type][0]
