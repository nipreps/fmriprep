''' Testing module for fmriprep.workflows.confounds '''
import logging
from ..bold.confounds import init_bold_confs_wf
from ...utils.testing import TestWorkflow

logging.disable(logging.INFO)  # don't print unnecessary msgs


class TestConfounds(TestWorkflow):
    ''' Testing class for fmriprep.workflows.confounds '''

    def test_confounds_wf(self):
        # run
        workflow = init_bold_confs_wf(
            mem_gb=1, use_aroma=False, ignore_aroma_err=False,
            metadata={"RepetitionTime": 2.0,
                      "SliceTiming": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
        workflow.write_hierarchical_dotfile()

        # assert

        # check some key paths
        self.assert_circular(workflow, [
            ('outputnode', 'inputnode', [('confounds_file', 'fmri_file')]),
        ])

        # Make sure mandatory inputs are set
        self.assert_inputs_set(workflow, {'outputnode': ['confounds_file'],
                                          'concat': ['signals', 'dvars', 'fd',
                                                     # 'acompcor', See confounds.py
                                                     'tcompcor'],
                                          # 'aCompCor': ['components_file', 'mask_file'], }) see ^^
                                          'tcompcor': ['components_file']})
