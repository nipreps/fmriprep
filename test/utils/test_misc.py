import os.path as op
import json
import fmriprep.utils.misc as misc
import re
import unittest
import test.constant as c
from future.utils import raise_from

class TestCollectBids(unittest.TestCase):
    subject_id = 'sub-S5271NYO'

    @classmethod
    def setUp(self):
        try:
            subject_data = misc.collect_bids_data(c.DATASET, self.subject_id,
                                                  c.BIDS_SPEC)
            self.imaging_data = {
                self.subject_id: subject_data
            }
        except IOError as e:
            url = "http://googledrive.com/host/0BxI12kyv2olZbl9GN3BIOVVoelE"
            raise_from(Exception("Couldn't find data at " + c.DATASET + 
                                 ". Download from " + url), e)

    def test_collect_bids_data(self):
        ''' test data has at least one subject with at least one session '''
        self.assertNotEqual(0, len(self.imaging_data))
        self.assertNotEqual(0, len(next(iter(self.imaging_data.values()))))

    def test_epi(self):
        epi_template = op.join(c.DATASET, "{subject}/func/"
                               "{subject}_task-rest_acq-RL_run-1_bold.nii.gz")
        self.assert_key_exists(epi_template, 'func')

    def test_sbref(self):
        sbref_template = op.join(c.DATASET, "{subject}/func/"
                          "{subject}_task-rest_acq-LR_run-1_sbref.nii.gz")
        self.assert_key_exists(sbref_template, 'sbref')

    def test_t1w(self):
        t1_template = op.join(c.DATASET,
                              "{subject}/anat/{subject}_run-1_T1w.nii.gz")
        self.assert_key_exists(t1_template, 't1w')

    def test_fieldmaps(self):
        fieldmap_pattern = r"{0}\/fmap\/{0}_dir-[0-9]+_run-[0-9]+_epi\.nii\.gz"
        self.assert_fieldmap_files_exist(fieldmap_pattern, 'fieldmaps')
    
    # HELPER ASSERTIONS

    def assert_fieldmap_files_exist(self, pattern, key):
        for subject in self.imaging_data:
            search_pattern = pattern.format(subject)
            for session in self.imaging_data[subject]:
                for fieldmap in self.imaging_data[subject][session][key]:
                    match = re.search(search_pattern, fieldmap)
                    self.assertTrue(match)

    def assert_key_exists(self, template, key):
        for subject in self.imaging_data:
            for session in self.imaging_data[subject]:
                self.assertIn(template.format(subject=subject),
                              self.imaging_data[subject][session][key])
        
if __name__ == '__main__':
    unittest.main() 
    #dataset = "../../test_data/aa_conn"
    #imaging_data = misc.collect_bids_data(dataset)
