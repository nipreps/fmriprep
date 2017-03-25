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
    #
    #     # Make sure mandatory inputs are set/connected
    #     self._assert_mandatory_inputs_set(wfbasic_fmap_sbref)

    def test_wf_basic(self, _):
        # set up
        mock_subject_data = {'func': ''}
        mock_settings = {'output_dir': '.', 'ants_nthreads': 1, 'reportlets_dir': '.',
                         'biggest_epi_file_size_gb': 1,
                         'skip_native': False, 'freesurfer': False}

        # run
        wfbasic = basic_wf(mock_subject_data, mock_settings)
        wfbasic.write_graph()

        # assert
        self.assert_circular(wfbasic, [
            ('EPIMNITransformation', 'BIDSDatasource',
             [('DerivativesHMCMNI.out_file', 'subject_data')]),
            ('EPIMNITransformation', 'EPI_HMC', [('DerivativesHMCMNI.out_file', 'inputnode.epi')]),
            ('ref_epi_t1_registration', 'EPI_HMC', [('outputnode.mat_epi_to_t1', 'inputnode.epi')]),
        ])

        self._assert_mandatory_inputs_set(wfbasic)

    def _assert_mandatory_inputs_set(self, workflow):
        self.assert_inputs_set(workflow, {
            'BIDSDatasource': ['subject_data'],
            'ConfoundDiscoverer': ['inputnode.fmri_file', 'inputnode.movpar_file',
                                   'inputnode.t1_tpms', 'inputnode.epi_mask']
        })
