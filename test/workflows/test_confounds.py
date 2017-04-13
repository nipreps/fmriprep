''' Testing module for fmriprep.workflows.confounds '''
import logging
import os
import mock

import pandas as pd

from fmriprep.workflows.confounds import discover_wf, _gather_confounds

from test.workflows.utilities import TestWorkflow
from test.workflows import stub

logging.disable(logging.INFO) # don't print unnecessary msgs

class TestConfounds(TestWorkflow):
    ''' Testing class for fmriprep.workflows.confounds '''

    def test_discover_wf(self):
        # run
        workflow = discover_wf(stub.settings({'biggest_epi_file_size_gb': 1,
                                              'skip_native': False,
                                              'reportlets_dir': '.'}))
        workflow.write_hierarchical_dotfile()

        # assert

        # check some key paths
        self.assert_circular(workflow, [
            ('outputnode', 'inputnode', [('confounds_file', 'fmri_file')]),
            ('DerivConfounds', 'inputnode', [('out_file', 'fmri_file')])
        ])

        # Make sure mandatory inputs are set
        self.assert_inputs_set(workflow, {'outputnode': ['confounds_file'],
                                          'ConcatConfounds': ['signals', 'dvars', 'frame_displace',
                                                              #'acompcor', See confounds.py
                                                              'tcompcor'],
                                          'tCompCor': ['components_file']})
                                          # 'aCompCor': ['components_file', 'mask_file'], }) see ^^

    @mock.patch('pandas.read_csv')
    @mock.patch.object(pd.DataFrame, 'to_csv', autospec=True)
    @mock.patch.object(pd.DataFrame, '__eq__', autospec=True,
                       side_effect=lambda me, them: me.equals(them))
    def test_gather_confounds(self, df_equality, mock_df, mock_csv_reader):
        ''' asserts that the function for node ConcatConfounds reads and writes
        the confounds properly '''

        # set up
        signals = "signals.tsv"
        dvars = "dvars.tsv"

        mock_csv_reader.side_effect = [pd.DataFrame({'a': [0.1]}), pd.DataFrame({'b': [0.2]})]

        # run
        _gather_confounds(signals, dvars)

        # assert
        calls = [mock.call(confounds, sep="\t") for confounds in [signals, dvars]]
        mock_csv_reader.assert_has_calls(calls)

        confounds = pd.DataFrame({'a': [0.1], 'b': [0.2]})

        mock_df.assert_called_once_with(confounds, os.path.abspath("confounds.tsv"),
                                        na_rep='n/a', index=False, sep="\t")
