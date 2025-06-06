from shutil import copytree

import pytest

from .tests.data import load

try:
    from contextlib import chdir as _chdir
except ImportError:  # PY310
    import os
    from contextlib import contextmanager

    @contextmanager  # type: ignore
    def _chdir(path):
        cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(cwd)


@pytest.fixture
def data_dir():
    with load.as_path() as data_dir:
        yield data_dir


@pytest.fixture(autouse=True)
def _docdir(request, tmp_path):
    # Trigger ONLY for the doctests.
    doctest_plugin = request.config.pluginmanager.getplugin('doctest')
    if isinstance(request.node, doctest_plugin.DoctestItem):
        with load.as_path() as data_dir:
            copytree(data_dir, tmp_path, dirs_exist_ok=True)

        # Chdir only for the duration of the test.
        with _chdir(tmp_path):
            yield

    else:
        # For normal tests, we have to yield, since this is a yield-fixture.
        yield
