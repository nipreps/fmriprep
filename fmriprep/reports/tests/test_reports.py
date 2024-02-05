import shutil
from pathlib import Path

import pytest
from bids.layout import BIDSLayout

from fmriprep.reports.core import generate_reports

from ... import config, data

data_dir = data.load('tests')


# Test with and without sessions' aggregation
@pytest.mark.parametrize(
    ('aggr_ses_reports', 'expected_files'),
    [
        (
            3,
            [
                'sub-001_anat.html',
                'sub-001_ses-001_func.html',
                'sub-001_ses-003_func.html',
                'sub-001_ses-004_func.html',
                'sub-001_ses-005_func.html',
            ],
        ),
        (4, ['sub-001.html']),
    ],
)
# Test with and without crash file
@pytest.mark.parametrize('error', [True, False])
# Test with and without boilerplate
@pytest.mark.parametrize('boilerplate', [True, False])
# Test ses- prefix stripping
@pytest.mark.parametrize(
    'session_list', [['001', '003', '004', '005'], ['ses-001', 'ses-003', 'ses-004', 'ses-005']]
)
# Test sub- prefix stripping
@pytest.mark.parametrize('subject_label', ['001', 'sub-001'])
@pytest.mark.skipif(
    not Path.exists(data_dir / 'work'),
    reason='Package installed - large test data directory excluded from wheel',
)
def test_ReportSeparation(
    tmp_path,
    monkeypatch,
    aggr_ses_reports,
    expected_files,
    error,
    boilerplate,
    session_list,
    subject_label,
):
    fake_uuid = 'fake_uuid'

    sub_dir = tmp_path / 'sub-001'
    shutil.copytree(data_dir / 'work/reportlets/fmriprep/sub-001', sub_dir)

    # Test report generation with and without crash file
    if error:
        crash_file = next(data_dir.glob('crash_files/crash*.txt'))
        run_log_dir = sub_dir / 'log' / fake_uuid
        run_log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(crash_file, run_log_dir / crash_file.name)

    # Test report generation with and without boilerplate
    if boilerplate:
        log_dir = tmp_path / 'logs'
        log_dir.mkdir()
        shutil.copy2(data_dir / 'logs/CITATION.html', log_dir / 'CITATION.html')

    monkeypatch.setattr(config.execution, 'aggr_ses_reports', aggr_ses_reports)

    def mock_session_list(*args, **kwargs):
        return session_list

    config.execution.layout = BIDSLayout(data_dir / 'ds000005')
    monkeypatch.setattr(config.execution.layout, 'get_sessions', mock_session_list)
    monkeypatch.setattr(
        config.execution, 'bids_filters', {'bold': {'session': ['001', '003', '004', '005']}}
    )

    # Generate report
    failed_reports = generate_reports([subject_label], tmp_path, fake_uuid)

    # Verify that report generation was successful
    assert not failed_reports

    # Check that all expected files were generated
    for expected_file in expected_files:
        file_path = tmp_path / expected_file
        assert file_path.is_file(), f'Expected file {expected_file} is missing'

    # Check if there are no unexpected HTML files
    unexpected_files = {file.name for file in tmp_path.glob('*.html')} - set(expected_files)
    assert not unexpected_files, f'Unexpected HTML files found: {unexpected_files}'

    if not (boilerplate or error):
        return

    html_content = Path.read_text(tmp_path / expected_files[0])
    if boilerplate:
        assert (
            'The boilerplate text was automatically generated' in html_content
        ), f'The file {expected_files[0]} did not contain the reported error.'

    if error:
        assert (
            'One or more execution steps failed' in html_content
        ), f'The file {expected_files[0]} did not contain the reported error.'
