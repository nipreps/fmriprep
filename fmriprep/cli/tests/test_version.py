# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Test version checks."""
from os import getenv
from datetime import datetime
from pathlib import Path
from packaging.version import Version
import pytest
from .. import version as _version
from ..version import check_latest, DATE_FMT, requests, is_flagged


class MockResponse:
    """Mocks the requests module so that Pypi is not actually queried."""

    status_code = 200
    _json = {
        "releases": {"1.0.0": None, "1.0.1": None, "1.1.0": None, "1.1.1rc1": None}
    }

    def __init__(self, code=200, json=None):
        """Allow setting different response codes."""
        self.status_code = code
        if json is not None:
            self._json = json

    def json(self):
        """Redefine the response object."""
        return self._json


def test_check_latest1(tmpdir, monkeypatch):
    """Test latest version check."""
    tmpdir.chdir()
    monkeypatch.setenv("HOME", str(tmpdir))
    assert str(Path.home()) == str(tmpdir)

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    # Initially, cache should not exist
    cachefile = Path.home() / ".cache" / "fmriprep" / "latest"
    assert not cachefile.exists()

    # First check actually fetches from pypi
    v = check_latest()
    assert cachefile.exists()
    assert isinstance(v, Version)
    assert v == Version("1.1.0")
    assert cachefile.read_text().split("|") == [
        str(v),
        datetime.now().strftime(DATE_FMT),
    ]

    # Second check - test the cache file is read
    cachefile.write_text("|".join(("1.0.0", cachefile.read_text().split("|")[1])))
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version("1.0.0")

    # Third check - forced oudating of cache
    cachefile.write_text("2.0.0|20180121")
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version("1.1.0")

    # Mock timeouts
    def mock_get(*args, **kwargs):
        raise requests.exceptions.Timeout

    monkeypatch.setattr(requests, "get", mock_get)

    cachefile.write_text("|".join(("1.0.0", cachefile.read_text().split("|")[1])))
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version("1.0.0")

    cachefile.write_text("2.0.0|20180121")
    v = check_latest()
    assert v is None

    cachefile.unlink()
    v = check_latest()
    assert v is None


@pytest.mark.parametrize(
    ("result", "code", "json"),
    [
        (None, 404, None),
        (None, 200, {"releases": {"1.0.0rc1": None}}),
        (Version("1.1.0"), 200, None),
        (Version("1.0.0"), 200, {"releases": {"1.0.0": None}}),
    ],
)
def test_check_latest2(tmpdir, monkeypatch, result, code, json):
    """Test latest version check with varying server responses."""
    tmpdir.chdir()
    monkeypatch.setenv("HOME", str(tmpdir))
    assert str(Path.home()) == str(tmpdir)

    def mock_get(*args, **kwargs):
        return MockResponse(code=code, json=json)

    monkeypatch.setattr(requests, "get", mock_get)

    v = check_latest()
    if result is None:
        assert v is None
    else:
        assert isinstance(v, Version)
        assert v == result


@pytest.mark.parametrize(
    "bad_cache",
    [
        "3laj#r???d|3akajdf#",
        "2.0.0|3akajdf#",
        "|".join(("2.0.0", datetime.now().strftime(DATE_FMT), "")),
        "",
    ],
)
def test_check_latest3(tmpdir, monkeypatch, bad_cache):
    """Test latest version check when the cache file is corrupted."""
    tmpdir.chdir()
    monkeypatch.setenv("HOME", str(tmpdir))
    assert str(Path.home()) == str(tmpdir)

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    # Initially, cache should not exist
    cachefile = Path.home() / ".cache" / "fmriprep" / "latest"
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    assert not cachefile.exists()

    cachefile.write_text(bad_cache)
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version("1.1.0")


@pytest.mark.parametrize(
    ("result", "version", "code", "json"),
    [
        (False, "1.2.1", 200, {"flagged": {"1.0.0": None}}),
        (True, "1.2.1", 200, {"flagged": {"1.2.1": None}}),
        (True, "1.2.1", 200, {"flagged": {"1.2.1": "FATAL Bug!"}}),
        (False, "1.2.1", 404, {"flagged": {"1.0.0": None}}),
        (False, "1.2.1", 200, {"flagged": []}),
        (False, "1.2.1", 200, {}),
    ],
)
def test_is_flagged(monkeypatch, result, version, code, json):
    """Test that the flagged-versions check is correct."""
    monkeypatch.setattr(_version, "__version__", version)

    def mock_get(*args, **kwargs):
        return MockResponse(code=code, json=json)

    monkeypatch.setattr(requests, "get", mock_get)

    val, reason = is_flagged()
    assert val is result

    test_reason = None
    if val:
        test_reason = json.get("flagged", {}).get(version, None)

    if test_reason is not None:
        assert reason == test_reason
    else:
        assert reason is None


def test_readonly(tmp_path, monkeypatch):
    """Test behavior when $HOME/.cache/fmriprep/latest can't be written out."""
    home_path = (
        Path("/home/readonly") if getenv("TEST_READONLY_FILESYSTEM") else tmp_path
    )
    monkeypatch.setenv("HOME", str(home_path))
    cachedir = home_path / ".cache"

    if getenv("TEST_READONLY_FILESYSTEM") is None:
        cachedir.mkdir(mode=0o555, exist_ok=True)

    # Make sure creating the folder will raise the exception.
    with pytest.raises(OSError):
        (cachedir / "fmriprep").mkdir(parents=True)

    # Should not raise
    check_latest()
