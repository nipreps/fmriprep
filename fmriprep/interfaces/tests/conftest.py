import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def data_dir():
    return Path(__file__).parent / "data"
