''' Testing module for fmriprep.workflows.base '''
import mock

from ...utils.testing import TestWorkflow
from ..base import init_single_subject_wf


@mock.patch('fmriprep.interfaces.BIDSDataGrabber')  # no actual BIDS dir necessary
class TestBase(TestWorkflow):

    def test_single_subject_wf(self, _):

        # run
        wfbasic = init_single_subject_wf(subject_id='test',
                                         name='single_subject_wf',
                                         task_id='',
                                         echo_idx=None,
                                         ignore=[],
                                         debug=False,
                                         low_mem=False,
                                         anat_only=False,
                                         longitudinal=False,
                                         t2s_coreg=False,
                                         omp_nthreads=1,
                                         skull_strip_template='OASIS',
                                         skull_strip_fixed_seed=False,
                                         reportlets_dir='.',
                                         output_dir='.',
                                         bids_dir='.',
                                         freesurfer=False,
                                         output_spaces=['T1w'],
                                         template='MNI152NLin2009cAsym',
                                         medial_surface_nan=False,
                                         cifti_output=False,
                                         hires=False,
                                         use_bbr=None,
                                         bold2t1w_dof=9,
                                         fmap_bspline=True,
                                         fmap_demean=True,
                                         use_aroma=False,
                                         aroma_melodic_dim=-200,
                                         ignore_aroma_err=False,
                                         use_syn=True,
                                         force_syn=True,
                                         template_out_grid='native')
        wfbasic.write_graph()
        self._assert_mandatory_inputs_set(wfbasic)

    def _assert_mandatory_inputs_set(self, workflow):
        self.assert_inputs_set(workflow, {
            'bidssrc': ['subject_data']
        })
