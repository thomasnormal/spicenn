import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "experiments" / "18_full_spice_goal_audit.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("full_spice_goal_audit", AUDIT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_json_falls_back_from_spice_results_to_results_tables(tmp_path):
    audit = load_audit_module()
    summary = tmp_path / "results" / "tables" / "spice_mnist_demo_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"heldout_test_accuracy": 0.91}))

    assert audit.load_json("spice/results/spice_mnist_demo_summary.json", tmp_path) == {
        "heldout_test_accuracy": 0.91
    }


def test_iter_spice_mnist_summaries_finds_results_tables_and_dedupes(tmp_path):
    audit = load_audit_module()
    tables = tmp_path / "results" / "tables"
    spice_results = tmp_path / "spice" / "results"
    tables.mkdir(parents=True)
    spice_results.mkdir(parents=True)
    (tables / "spice_mnist_demo_summary.json").write_text("{}")
    (spice_results / "spice_mnist_demo_summary.json").write_text("{}")
    (tables / "fast_mnist_demo_summary.json").write_text("{}")
    (tables / "spice_digits_demo_summary.json").write_text("{}")

    paths = audit.iter_spice_mnist_summary_paths(tmp_path)

    assert [path.name for path in paths] == ["spice_mnist_demo_summary.json"]
