import os
from shutil import copytree

import pytest

try:
    from importlib.resources import files as ir_files
except ImportError:  # PY<3.9
    from importlib_resources import files as ir_files

os.environ['NO_ET'] = '1'


def chdir_or_skip():
    data_dir = ir_files('fmriprep') / 'data'
    try:
        os.chdir(data_dir)
    except OSError:
        pytest.skip(f"Cannot chdir into {data_dir!r}. Probably in a zipped distribution.")


def copytree_or_skip(source, target):
    data_dir = ir_files('fmriprep') / source
    if not data_dir.exists():
        pytest.skip(f"Cannot chdir into {data_dir!r}. Probably in a zipped distribution.")

    try:
        copytree(data_dir, target / data_dir.name)
    except Exception:
        pytest.skip(f"Cannot copy {data_dir!r} into {target / data_dir.name}. Probably in a zip.")


@pytest.fixture(autouse=True)
def populate_namespace(doctest_namespace, tmp_path):
    doctest_namespace['chdir_or_skip'] = chdir_or_skip
    doctest_namespace['copytree_or_skip'] = copytree_or_skip
    doctest_namespace['testdir'] = tmp_path
