from __future__ import annotations

from typing import Any

import conftest


class _FakeItem:
    def __init__(self, *, keywords: tuple[str, ...] = (), fixturenames: tuple[str, ...] = ()) -> None:
        self.keywords = {keyword: True for keyword in keywords}
        self.fixturenames = fixturenames
        self.markers: list[Any] = []

    def add_marker(self, marker: Any) -> None:
        self.markers.append(marker)


def test_ngspice_collection_skip_applies_to_fixture_users(monkeypatch) -> None:
    monkeypatch.setattr(conftest, "_find_ngspice", lambda: None)
    by_fixture = _FakeItem(fixturenames=("tmp_path", "ngspice_path"))
    by_marker = _FakeItem(keywords=("ngspice",))
    plain = _FakeItem(fixturenames=("tmp_path",))

    conftest.pytest_collection_modifyitems(None, [by_fixture, by_marker, plain])

    assert len(by_fixture.markers) == 1
    assert by_fixture.markers[0].mark.name == "skip"
    assert len(by_marker.markers) == 1
    assert by_marker.markers[0].mark.name == "skip"
    assert plain.markers == []


def test_ngspice_collection_skip_is_not_added_when_executable_exists(monkeypatch) -> None:
    monkeypatch.setattr(conftest, "_find_ngspice", lambda: "/usr/bin/ngspice")
    item = _FakeItem(fixturenames=("ngspice_path",))

    conftest.pytest_collection_modifyitems(None, [item])

    assert item.markers == []
