import json
import fmriprep.workflows.base as base
import re
import unittest
import mock

class TestBase(unittest.TestCase):

    @mock.patch('fmriprep.workflows.sbref.sbref_t1_registration')
    def test_fmriprep_single(self, mock_registration):
        ''' Tests fmriprep_single for code errors, not correctness '''
        # NOT a test for correctness
        # SET UP INPUTS
        test_settings = {
            'output_dir': '.',
            'work_dir': '.',
            'bids_root': '.'
        }

        # SET UP EXPECTATIONS

        # RUN
        base.fmriprep_single(mock.MagicMock(), settings=test_settings)

        # ASSERT
