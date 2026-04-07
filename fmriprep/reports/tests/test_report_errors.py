from unittest.mock import patch

from fmriprep.reports.core import run_reports


def test_run_reports_error_handling(tmp_path):
    with patch('fmriprep.reports.core.Report') as MockReport:
        MockReport.return_value.generate_report.side_effect = Exception('Test Exception')

        res = run_reports(
            output_dir=str(tmp_path),
            subject_label='01',
            run_uuid='test_uuid',
            errorname='report.err',
        )

        assert res == '01'
        error_file = tmp_path / 'logs' / 'report.err'
        assert error_file.is_file()

        content = error_file.read_text()
        assert 'Traceback' in content
        assert 'Test Exception' in content
