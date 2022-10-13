from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_dir():
    return Path(__file__).parent / "data"
