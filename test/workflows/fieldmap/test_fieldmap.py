import json
from fmriprep.workflows.fieldmap.utils import create_encoding_file
from fmriprep.workflows.fieldmap import phdiff
from fmriprep.workflows.fieldmap import pepolar
import re
import mock
from test.workflows.utilities import TestWorkflow

class TestFieldMap(TestWorkflow):

    SOME_INT = 3

    def test_phasediff_workflow(self):
        # SET UP INPUTS
        mock_settings = {
            'work_dir': '.',
            'output_dir': '.'
        }

        # SET UP EXPECTATIONS
        expected_interfaces = [
            'DataSink', 'MultiImageMaths', 'ApplyMask', 'FUGUE', 'Merge', 'MathsCommand',
            'MultiImageMaths', 'IdentityInterface', 'IdentityInterface', 'Function', 'Function',
            'Function', 'BETRPT', 'N4BiasFieldCorrection', 'IntraModalMerge', 'SpatialFilter',
            'PRELUDE', 'Function', 'Function', 'DataSink', 'Function', 'IdentityInterface',
            'ReadSidecarJSON', 'IdentityInterface'
        ]

        expected_outputs = ['outputnode.fmap', 'outputnode.fmap_mask',
                            'outputnode.fmap_ref']
        expected_inputs = ['inputnode.input_images']

        # RUN
        result = phdiff.phdiff_workflow(mock_settings)

        # ASSERT
        self.assertIsAlmostExpectedWorkflow(phdiff.WORKFLOW_NAME,
                                            expected_interfaces,
                                            expected_inputs,
                                            expected_outputs,
                                            result)

    def test_pepolar_workflow(self):
        # SET UP INPUTS
        mock_settings = {
            'work_dir': '.',
            'output_dir': '.'
        }

        # SET UP EXPECTATIONS
        expected_interfaces = ['Function', 'N4BiasFieldCorrection', 'BETRPT',
                               'MCFLIRT', 'Merge', 'Split', 'TOPUP',
                               'ApplyTOPUP', 'Function', 'ImageDataSink',
                               'IdentityInterface', 'ReadSidecarJSON',
                               'IdentityInterface']
        expected_outputs = ['outputnode.fmap', 'outputnode.fmap_mask',
                            'outputnode.fmap_ref']
        expected_inputs = ['inputnode.input_images']

        # RUN
        result = pepolar.pepolar_workflow(settings=mock_settings)

        # ASSERT
        self.assertIsAlmostExpectedWorkflow(pepolar.WORKFLOW_NAME,
                                            expected_interfaces,
                                            expected_inputs,
                                            expected_outputs,
                                            result)

