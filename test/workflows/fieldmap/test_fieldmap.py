import json
import fmriprep.workflows.fieldmap as fieldmap
import re
import unittest
import mock

class TestFieldMap(unittest.TestCase):

    SOME_INT = 3

    def test_se_pair_workflow(self):
        pass

    @mock.patch('nibabel.load')
    @mock.patch('numpy.savetxt')
    def test_create_encoding_file(self, mock_savetxt, mock_load):
        # SETUP INPUTS
        fieldmaps = 'some_file.nii.gz'
        in_dict = { 'TotalReadoutTime': 'a_time',
                    'PhaseEncodingDirection': ['i']
        }

        # SET UP EXPECTATIONS
        mock_load(fieldmaps).shape = ['', '', '', self.SOME_INT]
        expected_enc_table = ([[1, 0, 0, in_dict['TotalReadoutTime']]] *
                              self.SOME_INT)

        # RUN
        out_file = fieldmap.se_pair_workflow.create_encoding_file(fieldmaps, in_dict)

        # ASSERT
        # the output file is called parameters.txt
        self.assertRegexpMatches(out_file, '/parameters.txt$')
        # nibabel.load was called with fieldmaps
        mock_load.assert_called_with(fieldmaps)
        # numpy.savetxt was called once. It was called with expected_enc_table
        mock_savetxt.assert_called_once_with(mock.ANY, expected_enc_table, 
                                             fmt=mock.ANY)

    def test_fieldmap_to_phasediff(self):
        pass
