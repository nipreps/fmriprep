#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from fmriprep.workflows.fieldmap import phdiff
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
        expected_interfaces = ['Function', 'N4BiasFieldCorrection', 'BETRPT',
                               'PRELUDE', 'IdentityInterface', 'ReadSidecarJSON',
                               'IdentityInterface', 'DataSink', 'MultiImageMaths',
                               'ApplyMask', 'FUGUE', 'Merge', 'MathsCommand',
                               'IntraModalMerge', 'SpatialFilter']
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

    # def test_pepolar_workflow(self):
    #     # SET UP INPUTS
    #     mock_settings = {
    #         'work_dir': '.',
    #         'output_dir': '.'
    #     }

    #     # SET UP EXPECTATIONS
    #     expected_interfaces = ['Function', 'N4BiasFieldCorrection', 'BETRPT',
    #                            'MCFLIRT', 'Merge', 'Split', 'TOPUP',
    #                            'ApplyTOPUP', 'Function', 'ImageDataSink',
    #                            'IdentityInterface', 'ReadSidecarJSON',
    #                            'IdentityInterface']
    #     expected_outputs = ['outputnode.fmap', 'outputnode.fmap_mask',
    #                         'outputnode.fmap_ref']
    #     expected_inputs = ['inputnode.input_images']

    #     # RUN
    #     result = pepolar.pepolar_workflow(settings=mock_settings)

    #     # ASSERT
    #     self.assertIsAlmostExpectedWorkflow(pepolar.WORKFLOW_NAME,
    #                                         expected_interfaces,
    #                                         expected_inputs,
    #                                         expected_outputs,
    #                                         result)

