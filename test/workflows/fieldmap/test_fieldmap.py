import json
from fmriprep.workflows.fieldmap import (se_pair_workflow,
                                         fieldmap_to_phasediff)
import re
import mock
from test.workflows.utilities import TestWorkflow

class TestFieldMap(TestWorkflow):

    SOME_INT = 3

    def test_fieldmap_to_phasediff(self):
        # SET UP EXPECTATIONS
        expected_interfaces = ['UnaryMaths', 'BinaryMaths', 'BinaryMaths',
                               'FUGUE', 'IdentityInterface',
                               'IdentityInterface']
        expected_inputs = []
        expected_outputs = []

        # RUN
        result = fieldmap_to_phasediff.fieldmap_to_phasediff()

        # ASSERT
        self.assertIsAlmostExpectedWorkflow(fieldmap_to_phasediff.WORKFLOW_NAME,
                                            expected_interfaces,
                                            expected_inputs,
                                            expected_outputs,
                                            result)

    def test_se_pair_workflow(self):
        # SET UP INPUTS
        mock_settings = {
            'work_dir': '.'
        }

        # SET UP EXPECTATIONS
        expected_interfaces = ['Function', 'N4BiasFieldCorrection', 'BET',
                               'MCFLIRT', 'Merge', 'Split', 'TOPUP',
                               'ApplyTOPUP', 'Function', 'DataSink',
                               'IdentityInterface', 'ReadSidecarJSON',
                               'IdentityInterface']
        expected_outputs = ['outputnode.fieldmap', 'outputnode.mag_brain',
                            'outputnode.fmap_mask']
        expected_inputs = []

        # RUN
        result = se_pair_workflow.se_pair_workflow(settings=mock_settings)

        # ASSERT
        self.assertIsAlmostExpectedWorkflow(se_pair_workflow.WORKFLOW_NAME,
                                            expected_interfaces,
                                            expected_inputs,
                                            expected_outputs,
                                            result)

    @mock.patch('nibabel.load')
    @mock.patch('numpy.savetxt')
    def test_create_encoding_file(self, mock_savetxt, mock_load):
        # SET UP INPUTS
        fieldmaps = 'some_file.nii.gz'
        in_dict = { 'TotalReadoutTime': 'a_time',
                    'PhaseEncodingDirection': ['i']
        }

        # SET UP EXPECTATIONS
        mock_load(fieldmaps).shape = ['', '', '', self.SOME_INT]
        expected_enc_table = ([[1, 0, 0, in_dict['TotalReadoutTime']]] *
                              self.SOME_INT)

        # RUN
        out_file = se_pair_workflow.create_encoding_file(fieldmaps, in_dict)

        # ASSERT
        # the output file is called parameters.txt
        self.assertRegexpMatches(out_file, '/parameters.txt$')
        # nibabel.load was called with fieldmaps
        mock_load.assert_called_with(fieldmaps)
        # numpy.savetxt was called once. It was called with expected_enc_table
        mock_savetxt.assert_called_once_with(mock.ANY, expected_enc_table, 
                                             fmt=mock.ANY)

