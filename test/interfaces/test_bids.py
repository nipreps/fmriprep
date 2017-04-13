import os
import shutil
import tempfile
import unittest

from nipype.pipeline import engine as pe

from fmriprep.interfaces.bids import DerivativesDataSink
import fmriprep.utils.misc as misc


class TestDerivativesDatasink(unittest.TestCase):
    prefix = 'sub-01_ses-01_'
    def setUp(self):
        _, self.infile_path = tempfile.mkstemp(prefix=self.prefix)
        self.infile_contents = 'test text\n'
        fp = open(self.infile_path, 'w')
        fp.write(self.infile_contents)
        fp.close()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.remove(self.infile_path)
        shutil.rmtree(self.out_dir)
        
    def test_known_file_out(self):
        suffix = 'sfx'
        deriv_ds = pe.Node(
            DerivativesDataSink(
                base_directory=self.out_dir,
                source_file=self.infile_path,
                suffix=suffix,
                in_file=self.infile_path
            ),
            name='TestDerivatives'
        )
        deriv_ds.run()
        deriv_filename = '{}_{}'.format(os.path.basename(self.infile_path), suffix)
        deriv_path = os.path.join(
            self.out_dir, 
            'fmriprep/sub-01/ses-01/func/',
            deriv_filename
        )
        self.assertTrue(os.path.isfile(deriv_path))
