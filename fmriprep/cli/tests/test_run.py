"""Test CLI."""
from packaging.version import Version
import pytest
from .. import version as _version
from ... import __about__
from ..run import get_parser


@pytest.mark.parametrize(('current', 'latest'), [
    ('1.0.0', '1.3.2'),
    ('1.3.2', '1.3.2')
])
def test_get_parser_update(monkeypatch, capsys, current, latest):
    """Make sure the out-of-date banner is shown."""
    expectation = Version(current) < Version(latest)

    def _mock_check_latest(*args, **kwargs):
        return Version(latest)

    monkeypatch.setattr(__about__, '__version__', current)
    monkeypatch.setattr(_version, 'check_latest', _mock_check_latest)

    get_parser()
    captured = capsys.readouterr().err

    msg = """\
You are using fMRIPrep-%s, and a newer version of fMRIPrep is available: %s.
Please check out our documentation about how and when to upgrade:
https://fmriprep.readthedocs.io/en/latest/faq.html#upgrading""" % (current, latest)

    assert (msg in captured) is expectation


@pytest.mark.parametrize('flagged', [
    (True, None),
    (True, 'random reason'),
    (False, None),
])
def test_get_parser_blacklist(monkeypatch, capsys, flagged):
    """Make sure the blacklisting banner is shown."""
    def _mock_is_bl(*args, **kwargs):
        return flagged

    monkeypatch.setattr(_version, 'is_flagged', _mock_is_bl)

    get_parser()
    captured = capsys.readouterr().err

    assert ('FLAGGED' in captured) is flagged[0]
    if flagged[0]:
        assert ((flagged[1] or 'reason: unknown') in captured)
