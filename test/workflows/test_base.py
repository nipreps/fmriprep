''' Testing module for fmriprep.workflows.base '''
import mock

from fmriprep.workflows.base import basic_fmap_sbref_wf, basic_wf
from test.workflows.utilities import TestWorkflow

@mock.patch('fmriprep.interfaces.BIDSDataGrabber') # no actual BIDS dir necessary
class TestBase(TestWorkflow):

    # Skip until fieldmaps are done
    # def test_wf_basic_fmap_sbref(self, _):
    #     # set up
    #     mock_subject_data = {'t1w': ['um'], 'sbref': ['um'], 'func': 'um'}
    #     mock_settings = {'output_dir': '.', 'work_dir': '.', 'reportlets_dir': '.',
    #                      'ants_nthreads': 1, 'biggest_epi_file_size_gb': 1,
    #                      'skip_native': False, 'freesurfer': False}
    #
    #     # run
    #     wfbasic_fmap_sbref = basic_fmap_sbref(mock_subject_data, mock_settings)
    #     wfbasic_fmap_sbref .write_graph()
    #
    #     # assert
    #
    #     # check some key paths
    #     self.assert_circular(wfbasic_fmap_sbref, [
    #         ('ref_epi_t1_registration', 'BIDSDatasource',
    #          [('outputnode.mat_epi_to_t1', 'subject_data')]),
    #         ('EPIUnwarpWorkflow', 'BIDSDatasource', [('outputnode.epi_mean', 'subject_data')]),
    #         ('ConfoundDiscoverer', 'BIDSDatasource',
    #          [('outputnode.confounds_file', 'subject_data')]),
    #         ('EPI_SBrefRegistration', 'BIDSDatasource', [('outputnode.out_mat', 'subject_data')]),
    #         ('EPIUnwarpWorkflow', 'EPI_HMC', [('outputnode.epi_mean', 'inputnode.epi')]),
    #         ('ConfoundDiscoverer', 'EPI_HMC', [('outputnode.confounds_file', 'inputnode.epi')]),
    #         ('EPI_SBrefRegistration', 'EPI_HMC', [('outputnode.out_mat', 'inputnode.epi')])
    #     ])
    #
    #     # Make sure mandatory inputs are set/connected
    #     self._assert_mandatory_inputs_set(wfbasic_fmap_sbref)

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
