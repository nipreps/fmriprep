"""Test parser."""
from packaging.version import Version
import pytest
from ..parser import _build_parser
from .. import version as _version
from ... import config

MIN_ARGS = ['data/', 'out/', 'participant']


@pytest.mark.parametrize('args,code', [
    ([], 2),
    (MIN_ARGS, 2),  # bids_dir does not exist
    (MIN_ARGS + ['--fs-license-file'], 2),
    (MIN_ARGS + ['--fs-license-file', 'fslicense.txt'], 2),
])
def test_parser_errors(args, code):
    """Check behavior of the parser."""
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(args)

    assert error.value.code == code


@pytest.mark.parametrize('args', [
    MIN_ARGS,
    MIN_ARGS + ['--fs-license-file'],
])
def test_parser_valid(tmp_path, args):
    """Check valid arguments."""
    datapath = (tmp_path / 'data')
    datapath.mkdir(exist_ok=True)
    args[0] = str(datapath)

    if '--fs-license-file' in args:
        _fs_file = tmp_path / 'license.txt'
        _fs_file.write_text('')
        args.insert(args.index('--fs-license-file') + 1,
                    str(_fs_file.absolute()))

    opts = _build_parser().parse_args(args)

    assert opts.bids_dir == datapath


@pytest.mark.parametrize('argval,gb', [
    ('1G', 1),
    ('1GB', 1),
    ('1000', 1),    # Default units are MB
    ('32000', 32),  # Default units are MB
    ('4000', 4),    # Default units are MB
    ('1000M', 1),
    ('1000MB', 1),
    ('1T', 1000),
    ('1TB', 1000),
    ('%dK' % 1e6, 1),
    ('%dKB' % 1e6, 1),
    ('%dB' % 1e9, 1),
])
def test_memory_arg(tmp_path, argval, gb):
    """Check the correct parsing of the memory argument."""
    datapath = (tmp_path / 'data')
    datapath.mkdir(exist_ok=True)
    _fs_file = tmp_path / 'license.txt'
    _fs_file.write_text('')

    args = MIN_ARGS + ['--fs-license-file', str(_fs_file)] \
        + ['--mem', argval]
    opts = _build_parser().parse_args(args)

    assert opts.memory_gb == gb


@pytest.mark.parametrize('current,latest', [
    ('1.0.0', '1.3.2'),
    ('1.3.2', '1.3.2')
])
def test_get_parser_update(monkeypatch, capsys, current, latest):
    """Make sure the out-of-date banner is shown."""
    expectation = Version(current) < Version(latest)

    def _mock_check_latest(*args, **kwargs):
        return Version(latest)

    monkeypatch.setattr(config.environment, 'version', current)
    monkeypatch.setattr(_version, 'check_latest', _mock_check_latest)

    _build_parser()
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

    _build_parser()
    captured = capsys.readouterr().err

    assert ('FLAGGED' in captured) is flagged[0]
    if flagged[0]:
        assert ((flagged[1] or 'reason: unknown') in captured)
