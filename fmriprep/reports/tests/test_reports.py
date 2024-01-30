import os
import shutil
from pathlib import Path

import pytest
from bids.layout import BIDSLayout

from fmriprep.reports.core import generate_reports

from ... import config


@pytest.fixture(scope="module")
def data_dir():
    return Path(__file__).parents[2] / "data" / "tests"


# Test with and without sessions' aggregation
@pytest.mark.parametrize(
    "aggr_ses_reports, expected_files",
    [
        (
            3,
            [
                "sub-001_anat.html",
                "sub-001_ses-001_func.html",
                "sub-001_ses-003_func.html",
                "sub-001_ses-004_func.html",
                "sub-001_ses-005_func.html",
            ],
        ),
        (4, ["sub-001.html"]),
    ],
)
# Test with and without crash file
@pytest.mark.parametrize("error", (True, False))

# Test with and without boilerplate
@pytest.mark.parametrize("boilerplate", (True, False))

# Test ses- prefix stripping
@pytest.mark.parametrize(
    "session_list", (["001", "003", "004", "005"], ["ses-001", "ses-003", "ses-004", "ses-005"])
)
# Test sub- prefix stripping
@pytest.mark.parametrize("subject_label", ("001", "sub-001"))
def test_ReportSeparation(
    monkeypatch,
    data_dir,
    aggr_ses_reports,
    expected_files,
    error,
    boilerplate,
    session_list,
    subject_label,
):
    fake_uuid = "fake_uuid"

    # Test report generation with and without crash file
    if error:
        # Copy the test crash file under the subject folder
        dst_path_e = data_dir / f"work/reportlets/fmriprep/sub-001/log/{fake_uuid}/"
        crash_file = "crash-20170905-182839-root-dvars-b78e9ea8-e295-48a1-af71-2d36afd9cebf.txt"
        os.makedirs(dst_path_e, exist_ok=True)
        shutil.copy2(data_dir / f"crash_files/{crash_file}", dst_path_e / crash_file)

    # Test report generation with and without boilerplate
    if boilerplate:
        # Copy the CITATION.html under the logs folder
        dst_path_c = data_dir / f"work/reportlets/fmriprep/logs"
        citation_file = "CITATION.html"
        os.makedirs(dst_path_c, exist_ok=True)
        shutil.copy2(data_dir / f"logs/{citation_file}", dst_path_c / citation_file)

    # Patching
    monkeypatch.setattr(config.execution, 'aggr_ses_reports', aggr_ses_reports)

    def mock_session_list(*args, **kwargs):
        return session_list

    config.execution.layout = BIDSLayout(data_dir / "ds000005")
    monkeypatch.setattr(config.execution.layout, "get_sessions", mock_session_list)
    monkeypatch.setattr(
        config.execution, "bids_filters", {'bold': {'session': ['001', '003', '004', '005']}}
    )
    output_dir = data_dir / "work/reportlets/fmriprep"

    # Generate report
    failed_reports = generate_reports([subject_label], output_dir, fake_uuid)

    # Verify that report generation was successful
    assert not failed_reports

    # Check that all expected files were generated
    for expected_file in expected_files:
        file_path = output_dir / expected_file
        assert file_path.is_file(), f"Expected file {expected_file} is missing"

    # Check if there are no unexpected HTML files
    unexpected_files = {
        file.name for file in output_dir.iterdir() if file.suffix == '.html'
    } - set(expected_files)
    assert not unexpected_files, f"Unexpected HTML files found: {unexpected_files}"

    if boilerplate:
        # Verify that the keywords indicating the boilerplate is reported are present in the HTML
        with open(output_dir / expected_files[0], 'r', encoding='utf-8') as file:
            html_content = file.read()
            assert (
                "The boilerplate text was automatically generated" in html_content
            ), f"The file {file} did not contain the reported error."

        # Delete copied citation file
        os.remove(dst_path_c / citation_file)
        os.rmdir(dst_path_c)

    if error:
        # Verify that the keywords indicating a reported error are present in the HTML
        with open(output_dir / expected_files[0], 'r', encoding='utf-8') as file:
            html_content = file.read()
            assert (
                "One or more execution steps failed" in html_content
            ), f"The file {file} did not contain the reported error."

        # Delete copied crash file
        os.remove(dst_path_e / crash_file)
        os.rmdir(dst_path_e)

    # Delete generated HTML files
    for file in output_dir.iterdir():
        if file.suffix == '.html':
            os.remove(file)
