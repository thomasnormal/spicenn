from __future__ import annotations

import shutil

import pytest


@pytest.fixture
def ngspice_path() -> str:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")
    return ngspice
