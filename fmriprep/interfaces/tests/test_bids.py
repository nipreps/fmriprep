"""Tests for fmriprep.interfaces.bids."""


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
