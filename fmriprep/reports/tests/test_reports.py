import os
import pytest
import shutil

from ... import config
from fmriprep.reports.core import generate_reports
from pathlib import Path
from bids.layout import BIDSLayout


@pytest.fixture(scope="module")
def data_dir():
    return Path(__file__).parents[2] / "data" / "tests"


@pytest.mark.parametrize(
    "max_ses_agr, expected_files, crash_txt",
    [
        (
            4,
            [
                "sub-001_anat.html",
                "sub-001_ses-001_func.html",
                "sub-001_ses-003_func.html",
                "sub-001_ses-004_func.html",
                "sub-001_ses-005_func.html",
            ],
            False,
        ),
        (
            4,
            [
                "sub-001_anat.html",
                "sub-001_ses-001_func.html",
                "sub-001_ses-003_func.html",
                "sub-001_ses-004_func.html",
                "sub-001_ses-005_func.html",
            ],
            True,
        ),
        (8, ["sub-001.html"], False),
        (8, ["sub-001.html"], True),
    ],
)
def test_ReportSeparation(monkeypatch, data_dir, max_ses_agr, expected_files, crash_txt):
    fake_uuid = "fake_uuid"

    # Test report generation with and without crash file
    if crash_txt:
        # Copy the test crash file under the subject folder
        dst_path = data_dir / f"work/reportlets/fmriprep/sub-001/{fake_uuid}/"
        crash_file = "crash-20170905-182839-root-dvars-b78e9ea8-e295-48a1-af71-2d36afd9cebf.txt"
        os.makedirs(dst_path, exist_ok=True)
        shutil.copy2(data_dir / f"crash_files/{crash_file}", dst_path / crash_file)

    monkeypatch.setattr(config.execution, 'max_ses_agr', max_ses_agr)

    def mock_session_list(subject):
        return ['001', '003', '004', '005']

    config.execution.layout = BIDSLayout(data_dir / "ds000005")
    monkeypatch.setattr(config.execution.layout, "get_sessions", mock_session_list)

    output_dir = data_dir / "work/reportlets/fmriprep"

    failed_reports = generate_reports(["sub-001"], output_dir, fake_uuid)

    # Verify that report generation was successfull
    assert not failed_reports

    for expected_file in expected_files:
        file_path = output_dir / expected_file
        assert file_path.is_file(), f"Expected file {expected_file} is missing"

    # Check if there are no unexpected HTML files
    unexpected_files = {
        file.name for file in output_dir.iterdir() if file.suffix == '.html'
    } - set(expected_files)

    assert not unexpected_files, f"Unexpected HTML files found: {unexpected_files}"

    # Delete copied crash file
    if crash_txt:
        import pdb; pdb.set_trace()
        os.remove(dst_path / crash_file)
        os.rmdir(dst_path)

    # Delete generated HTML files
    for file in output_dir.iterdir():
        if file.suffix == '.html':
            os.remove(file)


@pytest.mark.parametrize(
    "subject_label, session_list",
    [
        ("sub-001", ["001", "003", "004", "005"]),
        ("sub-001", ["ses-001", "ses-003", "ses-004", "ses-005"]),
        ("001", ["001", "003", "004", "005"]),
        ("001", ["ses-001", "ses-003", "ses-004", "ses-005"]),
    ],
)
def test_PrefixStripping(monkeypatch, data_dir, subject_label, session_list):
    monkeypatch.setattr(config.execution, 'max_ses_agr', 4)

    def mock_session_list(subject):
        return session_list

    config.execution.layout = BIDSLayout(data_dir / "ds000005")
    monkeypatch.setattr(config.execution.layout, "get_sessions", mock_session_list)

    output_dir = data_dir / "work/reportlets/fmriprep"

    failed_reports = generate_reports([subject_label], output_dir, "fake_uuid")

    # Verify that report generation was successfull
    assert not failed_reports

    # Drop ses- prefixes
    session_list = [ses[4:] if ses.startswith("ses-") else ses for ses in session_list]
    # Drop sub- prefix
    subject_label = subject_label.lstrip("sub-")

    expected_files = [f"sub-{subject_label}_anat.html"] + [
        f"sub-{subject_label}_ses-{session}_func.html" for session in session_list
    ]

    for expected_file in expected_files:
        file_path = output_dir / expected_file
        assert file_path.is_file(), f"Expected file {expected_file} is missing"

    # Check if there are no unexpected HTML files
    unexpected_files = {
        file.name for file in output_dir.iterdir() if file.suffix == '.html'
    } - set(expected_files)
    assert not unexpected_files, f"Unexpected HTML files found: {unexpected_files}"

    # Delete generated HTML files
    for file in output_dir.iterdir():
        if file.suffix == '.html':
            os.remove(file)
