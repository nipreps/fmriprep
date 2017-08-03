''' Testing module for fmriprep.workflows.base '''
import mock

from fmriprep.workflows.base import init_single_subject_wf
from test.workflows.utilities import TestWorkflow


@mock.patch('fmriprep.interfaces.BIDSDataGrabber')  # no actual BIDS dir necessary
class TestBase(TestWorkflow):

    def test_single_subject_wf(self, _):

        # run
        wfbasic = init_single_subject_wf(subject_id='test',
                                         name='single_subject_wf',
                                         task_id='',
                                         ignore=[],
                                         debug=False,
                                         anat_only=False,
                                         longitudinal=False,
                                         omp_nthreads=1,
                                         skull_strip_ants=False,
                                         reportlets_dir='.',
                                         output_dir='.',
                                         bids_dir='.',
                                         freesurfer=False,
                                         output_spaces=['T1w'],
                                         template='MNI152NLin2009cAsym',
                                         hires=False,
                                         bold2t1w_dof=9,
                                         fmap_bspline=True,
                                         fmap_demean=True,
                                         use_aroma=False,
                                         ignore_aroma_err=False,
                                         use_syn=True,
                                         force_syn=True,
                                         output_grid_ref=None)
        wfbasic.write_graph()

        self._assert_mandatory_inputs_set(wfbasic)

    def _assert_mandatory_inputs_set(self, workflow):
        self.assert_inputs_set(workflow, {
            'bidssrc': ['subject_data']
        })
