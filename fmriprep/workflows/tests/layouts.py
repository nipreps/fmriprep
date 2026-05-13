def _make_anat():
    return [
        {'run': 1, 'suffix': 'T1w'},
        {'run': 2, 'suffix': 'T1w'},
        {'suffix': 'T2w'},
    ]


def _make_func():
    return [
        *(
            {
                'task': 'rest',
                'run': i,
                'suffix': suffix,
                'metadata': {
                    'RepetitionTime': 2.0,
                    'PhaseEncodingDirection': 'j',
                    'TotalReadoutTime': 0.6,
                    'EchoTime': 0.03,
                    'SliceTiming': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
                },
            }
            for suffix in ('bold', 'sbref')
            for i in range(1, 3)
        ),
        *(
            {
                'task': 'nback',
                'echo': i,
                'suffix': 'bold',
                'metadata': {
                    'RepetitionTime': 2.0,
                    'PhaseEncodingDirection': 'j',
                    'TotalReadoutTime': 0.6,
                    'EchoTime': 0.015 * i,
                    'SliceTiming': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
                },
            }
            for i in range(1, 4)
        ),
    ]


def _make_fmap():
    return [
        {'suffix': 'phasediff', 'metadata': {'EchoTime1': 0.005, 'EchoTime2': 0.007}},
        {'suffix': 'magnitude1', 'metadata': {'EchoTime': 0.005}},
        {
            'suffix': 'epi',
            'dir': 'PA',
            'metadata': {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.6},
        },
        {
            'suffix': 'epi',
            'dir': 'AP',
            'metadata': {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.6},
        },
    ]


def _make_no_session():
    return {
        '01': {
            'anat': _make_anat(),
            'func': _make_func(),
            'fmap': _make_fmap(),
        },
    }


def _make_single_session():
    return {
        '01': [
            {
                'session': 'pre',
                'anat': _make_anat(),
                'func': _make_func(),
                'fmap': _make_fmap(),
            }
        ],
    }


def _make_homogeneous_sessions():
    return {
        '01': [
            {
                'session': 'pre',
                'anat': _make_anat(),
                'func': _make_func(),
                'fmap': _make_fmap(),
            },
            {
                'session': 'post',
                'anat': _make_anat(),
                'func': _make_func(),
                'fmap': _make_fmap(),
            },
        ]
    }


def _make_heterogeneous_sessions():
    return {
        '01': [
            {
                'session': 'anat',
                'anat': {'suffix': 'T1w'},
            },
            {
                'session': 'fmri',
                'func': [
                    {
                        'task': 'rest',
                        'suffix': 'bold',
                        'metadata': {
                            'RepetitionTime': 2.0,
                            'PhaseEncodingDirection': 'j',
                            'TotalReadoutTime': 0.6,
                        },
                    }
                ],
            },
        ]
    }


_LAYOUT_FACTORIES = {
    'no_session': _make_no_session,
    'single_session': _make_single_session,
    'homogeneous_sessions': _make_homogeneous_sessions,
    'heterogeneous_sessions': _make_heterogeneous_sessions,
}


def get_layout(layout_id: str):
    """Get a fresh layout spec by ID."""
    try:
        return _LAYOUT_FACTORIES[layout_id]()
    except KeyError as exc:
        raise ValueError(
            f'Unknown layout: {layout_id!r}. Choose from {list(_LAYOUT_FACTORIES)}'
        ) from exc


__all__ = ['get_layout']
