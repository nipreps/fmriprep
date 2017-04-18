''' Testing module for fmriprep.workflows.base '''
import mock

from fmriprep.workflows.base import basic_wf
from test.workflows.utilities import TestWorkflow

@mock.patch('fmriprep.interfaces.BIDSDataGrabber') # no actual BIDS dir necessary
class TestBase(TestWorkflow):

    def test_wf_basic(self, _):
        # set up
        mock_subject_data = {'func': ''}
        mock_settings = {'output_dir': '.', 'ants_nthreads': 1, 'reportlets_dir': '.',
                         'biggest_epi_file_size_gb': 1,
                         'bids_root': '.',
                         'skip_native': False, 'freesurfer': False}

        # run
        wfbasic = basic_wf(mock_subject_data, mock_settings)
        wfbasic.write_graph()

        self._assert_mandatory_inputs_set(wfbasic)

    def _assert_mandatory_inputs_set(self, workflow):
        self.assert_inputs_set(workflow, {
            'BIDSDatasource': ['subject_data']
        })
