from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import datasets as spice_datasets  # noqa: E402


def test_counted_mnist_dataset_records_threads_download_flag(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_mnist_records(sample_count, seed, frontend, class_count, *, root, download=False):
        calls.append(
            {
                "sample_count": sample_count,
                "seed": seed,
                "frontend": frontend,
                "class_count": class_count,
                "root": root,
                "download": download,
            }
        )
        return [{"label": 0, "inputs": {"x0": 0.0}}]

    monkeypatch.setattr(spice_datasets, "mnist_records", fake_mnist_records)

    records = spice_datasets.dataset_records("mnist3fixed8_6", 11, root=tmp_path, download=True)

    assert records == [{"label": 0, "inputs": {"x0": 0.0}}]
    assert calls == [
        {
            "sample_count": 6,
            "seed": 11,
            "frontend": "fixed8",
            "class_count": 3,
            "root": tmp_path,
            "download": True,
        }
    ]


def test_mnist01_dataset_records_threads_download_flag(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_mnist01_records(sample_count, seed, frontend, *, root, download=False):
        calls.append(
            {
                "sample_count": sample_count,
                "seed": seed,
                "frontend": frontend,
                "root": root,
                "download": download,
            }
        )
        return [{"label": 1, "inputs": {"x0": 1.0}}]

    monkeypatch.setattr(spice_datasets, "mnist01_records", fake_mnist01_records)

    records = spice_datasets.dataset_records("mnist01fixed8_4", 5, root=tmp_path, download=True)

    assert records == [{"label": 1, "inputs": {"x0": 1.0}}]
    assert calls == [
        {
            "sample_count": 4,
            "seed": 5,
            "frontend": "fixed8",
            "root": tmp_path,
            "download": True,
        }
    ]
