import unittest
from multiprocessing import Process

import numpy as np
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.utils.tmpdirs import InTemporaryDirectory
from nipype.testing.utils import save_toy_nii

from fmriprep.utils.validation import validate, is_4d_nifti, is_3d_nifti

class TestValidation(unittest.TestCase):

    nifti_4d_file = 'nifti_4d_file.nii'
    nifti_3d_file = 'nifti_3d_file.nii'

    @classmethod
    def setUp(self):
        ''' make fake little 4d and 3d nifti files '''
        array_4d = np.ones((2, 3, 4, 5)) # 2-by-3-by-4-by-5
        save_toy_nii(array_4d, self.nifti_4d_file)
        save_toy_nii(array_4d[:, :, :, 0], self.nifti_3d_file)

    def test_is_4d_nifti(self):
        ''' implicity tests `is_nd_nifti` and `validate` '''
        result = validate(self.nifti_4d_file, is_4d_nifti)
        self.assertEqual(result, self.nifti_4d_file)

        with self.assertRaisesRegexp(ValueError, 'is_4d_nifti'):
            validate(self.nifti_3d_file, is_4d_nifti)

    def test_in_workflow(self):
        ''' check return value of workflow makes sense.
        not zero on failure, zero on success '''
        def dummy_fun(in_nifti):
            return in_nifti

        def make_dummy_node(name):
            return pe.Node(niu.Function(function=dummy_fun, input_names=['in_nifti'],
                                        output_names=['out_4d_nifti']),
                           name=name)

        def workflow_exit_code(workflow):
            workflow_process = Process(target=pe.Workflow.run, args=(workflow,))
            workflow_process.start()
            workflow_process.join() # wait for process to finish
            return workflow_process.exitcode

        workflow = pe.Workflow(name='test_workflow')
        dummynode = make_dummy_node('dummynode')
        outputnode = make_dummy_node('outputnode')

        workflow.connect([
            (dummynode, outputnode, [(('out_4d_nifti', validate, is_4d_nifti,
                                       'in_nifti must be 4-D'),
                                      'in_nifti')])
        ])

        # success
        dummynode.inputs.in_nifti=self.nifti_4d_file
        self.assertEqual(workflow_exit_code(workflow), 0)

        # fail
        dummynode.inputs.in_nifti=self.nifti_3d_file
        self.assertNotEqual(workflow_exit_code(workflow), 0)
        
