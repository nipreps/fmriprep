"""Utilities and mocks for testing and documentation building."""
import os
from contextlib import contextmanager
from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from toml import loads
from tempfile import mkdtemp


@contextmanager
def mock_config():
    """Create a mock config for documentation and testing purposes."""
    from .. import config
    _old_fs = os.getenv('FREESURFER_HOME')
    if not _old_fs:
        os.environ['FREESURFER_HOME'] = mkdtemp()

    filename = Path(pkgrf('fmriprep', 'data/tests/config.toml'))
    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != 'environment':
            section = getattr(config, sectionname)
            section.load(configs, init=False)
    config.nipype.init()
    config.loggers.init()
    config.init_spaces()

    config.execution.work_dir = Path(mkdtemp())
    config.execution.bids_dir = Path(pkgrf('fmriprep', 'data/tests/ds000005')).absolute()
    config.execution.init()

    yield

    if not _old_fs:
        del os.environ["FREESURFER_HOME"]
