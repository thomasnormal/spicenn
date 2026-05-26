from __future__ import annotations

from functools import lru_cache
import shutil
from typing import Optional

import pytest


@lru_cache(maxsize=1)
def _find_ngspice() -> Optional[str]:
    return shutil.which("ngspice")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "ngspice: test requires the ngspice executable",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _find_ngspice() is not None:
        return
    skip_ngspice = pytest.mark.skip(reason="ngspice is not installed")
    for item in items:
        if "ngspice" in item.keywords or "ngspice_path" in getattr(item, "fixturenames", ()):
            item.add_marker(skip_ngspice)


@pytest.fixture(scope="session")
def ngspice_path() -> str:
    ngspice = _find_ngspice()
    if ngspice is None:
        pytest.skip("ngspice is not installed")
    return ngspice
