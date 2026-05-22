from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_TOPOLOGY = {
    "image_size": 10,
    "block_size": 4,
    "stride": 2,
    "channels": 2,
}


FIELDS = [
    "tag",
    "topology",
    "simulator",
    "updates",
    "eval_samples",
    "mnist_index_order",
    "batch_size",
    "update_mode",
    "reference_mode",
    "eval_backend",
    "output_mode",
    "local_activation",
    "hidden_synapse_mode",
    "readout_synapse_mode",
    "fully_on_device_execution_contract_met",
    "strict_fully_on_device_contract_met",
    "strict_fully_on_device_requested",
    "random_init_used",
    "initial_weights_source",
    "continuous_transient_contract_met",
    "direction_matches_batch_op_reference",
    "eval_accuracy_matches_batch_op_reference",
    "python_weight_updates_between_samples",
    "python_checkpointing_between_samples",
    "initial_eval_accuracy",
    "phase_eval_accuracy",
    "phase_eval_improvement",
    "spice_phase_eval_accuracy",
    "numpy_phase_eval_accuracy",
    "phase_eval_backend_abs_diff",
    "phase_update_l2",
    "state_update_direction_cosine",
    "state_update_sign_alignment_fraction",
    "phase_wall_time_s",
    "eval_wall_time_s",
    "summary_path",
]


def is_phase_transient_summary(data: dict[str, Any]) -> bool:
    status = str(data.get("status") or "")
    return (
        status.startswith("continuous_phase_train")
        or data.get("architecture") == "phase_resolved_transient_local_feature_readout"
        or "fully_on_device_execution_contract_met" in data
        or "single_phase_training_transient" in data
    )


def topology_label(data: dict[str, Any]) -> str:
    image_size = data.get("image_size")
    block_size = data.get("block_size")
    stride = data.get("stride")
    channels = data.get("channels")
    return f"{image_size}x{image_size} b{block_size} s{stride} c{channels}"


def row_from_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    tag = path.name.removesuffix("_summary.json")
    return {
        "tag": tag,
        "topology": topology_label(data),
        "image_size": data.get("image_size"),
        "block_size": data.get("block_size"),
        "stride": data.get("stride"),
        "channels": data.get("channels"),
        "simulator": data.get("simulator_selector") or data.get("simulator"),
        "updates": data.get("updates"),
        "eval_samples": data.get("eval_samples"),
        "mnist_index_order": data.get("mnist_index_order"),
        "batch_size": data.get("batch_size"),
        "update_mode": data.get("update_mode"),
        "reference_mode": data.get("reference_mode"),
        "eval_backend": data.get("eval_backend", "spice"),
        "output_mode": data.get("output_mode"),
        "local_activation": data.get("local_activation"),
        "hidden_synapse_mode": data.get("hidden_synapse_mode"),
        "readout_synapse_mode": data.get("readout_synapse_mode"),
        "fully_on_device_execution_contract_met": data.get("fully_on_device_execution_contract_met"),
        "strict_fully_on_device_contract_met": data.get("strict_fully_on_device_contract_met"),
        "strict_fully_on_device_requested": data.get("strict_fully_on_device_requested"),
        "random_init_used": data.get("random_init_used"),
        "initial_weights_source": data.get("initial_weights_source"),
        "continuous_transient_contract_met": data.get("continuous_transient_contract_met"),
        "direction_matches_batch_op_reference": data.get("direction_matches_batch_op_reference"),
        "eval_accuracy_matches_batch_op_reference": data.get("eval_accuracy_matches_batch_op_reference"),
        "python_weight_updates_between_samples": data.get("python_weight_updates_between_samples"),
        "python_checkpointing_between_samples": data.get("python_checkpointing_between_samples"),
        "initial_eval_accuracy": data.get("initial_eval_accuracy"),
        "phase_eval_accuracy": data.get("phase_eval_accuracy"),
        "phase_eval_improvement": data.get("phase_eval_improvement"),
        "spice_phase_eval_accuracy": data.get("spice_phase_eval_accuracy"),
        "numpy_phase_eval_accuracy": data.get("numpy_phase_eval_accuracy"),
        "phase_eval_backend_abs_diff": data.get("phase_eval_backend_abs_diff"),
        "phase_update_l2": data.get("phase_update_l2"),
        "state_update_direction_cosine": data.get("state_update_direction_cosine"),
        "state_update_sign_alignment_fraction": data.get("state_update_sign_alignment_fraction"),
        "phase_wall_time_s": data.get("phase_wall_time_s"),
        "eval_wall_time_s": data.get("eval_wall_time_s"),
        "summary_mtime_s": path.stat().st_mtime,
        "summary_path": str(path),
    }


def discover_summary_paths(root: Path = ROOT) -> list[Path]:
    paths_by_name: dict[str, Path] = {}
    for summary_dir in (root / "results" / "tables", root / "spice" / "results"):
        if not summary_dir.exists():
            continue
        for path in sorted(summary_dir.glob("spice_mnist_local_feature_phase*_summary.json")):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if is_phase_transient_summary(data):
                paths_by_name.setdefault(path.name, path)
    return sorted(paths_by_name.values(), key=lambda path: path.name)


def as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def same_target_topology(row: dict[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in TARGET_TOPOLOGY.items())


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    target_topology: bool = False,
    contract_only: bool = False,
    strict_contract_only: bool = False,
    min_updates: int | None = None,
) -> list[dict[str, Any]]:
    selected = rows
    if target_topology:
        selected = [row for row in selected if same_target_topology(row)]
    if contract_only:
        selected = [
            row
            for row in selected
            if row.get("fully_on_device_execution_contract_met") is True
        ]
    if strict_contract_only:
        selected = [
            row
            for row in selected
            if row.get("strict_fully_on_device_contract_met") is True
        ]
    if min_updates is not None:
        selected = [row for row in selected if as_int(row.get("updates")) >= min_updates]
    return selected


def sort_rows(rows: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    if sort_key == "updates":
        return sorted(
            rows,
            key=lambda row: (as_int(row.get("updates")), row.get("tag") or ""),
            reverse=True,
        )
    if sort_key == "improvement":
        return sorted(
            rows,
            key=lambda row: (
                float(row.get("phase_eval_improvement") or -999.0),
                as_int(row.get("updates")),
            ),
            reverse=True,
        )
    return sorted(rows, key=lambda row: float(row.get("summary_mtime_s") or 0.0), reverse=True)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "tag",
        "topology",
        "updates",
        "eval_samples",
        "eval_backend",
        "fully_on_device_execution_contract_met",
        "strict_fully_on_device_contract_met",
        "random_init_used",
        "reference_mode",
        "initial_eval_accuracy",
        "phase_eval_accuracy",
        "phase_eval_improvement",
        "phase_eval_backend_abs_diff",
        "phase_update_l2",
        "phase_wall_time_s",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")


def resolve_input_paths(paths: list[str]) -> list[Path]:
    resolved = [Path(path) if Path(path).is_absolute() else ROOT / path for path in paths]
    missing = [path for path in resolved if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing summary path(s):\n{joined}")
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize fully-on-device local-feature phase-transient result JSONs."
    )
    ap.add_argument(
        "summaries",
        nargs="*",
        help="Summary JSON paths. Defaults to discovered phase summaries.",
    )
    ap.add_argument("--target-topology", action="store_true", help="Keep only 10x10 b4 stride2 c2 runs.")
    ap.add_argument("--contract-only", action="store_true", help="Keep only runs meeting the execution contract.")
    ap.add_argument(
        "--strict-contract-only",
        action="store_true",
        help="Keep only random-init, batch_size=1, no-reference fully-on-device runs.",
    )
    ap.add_argument("--min-updates", type=int, help="Keep only runs with at least this many online updates.")
    ap.add_argument("--limit", type=int, help="Maximum rows to print or write after sorting.")
    ap.add_argument(
        "--sort",
        choices=["latest", "updates", "improvement"],
        default="latest",
        help="Row ordering before applying --limit.",
    )
    ap.add_argument("--out", type=Path, help="Optional CSV output path.")
    ap.add_argument("--json-out", type=Path, help="Optional JSON output path.")
    ap.add_argument("--markdown", action="store_true", help="Print a compact Markdown table.")
    args = ap.parse_args()

    paths = resolve_input_paths(args.summaries) if args.summaries else discover_summary_paths()
    rows = [row_from_summary(path) for path in paths]
    rows = filter_rows(
        rows,
        target_topology=args.target_topology,
        contract_only=args.contract_only,
        strict_contract_only=args.strict_contract_only,
        min_updates=args.min_updates,
    )
    rows = sort_rows(rows, args.sort)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.out:
        out_path = args.out if args.out.is_absolute() else ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=FIELDS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
    if args.json_out:
        out_path = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2) + "\n")
    if args.markdown or (not args.out and not args.json_out):
        print_markdown(rows)


if __name__ == "__main__":
    main()
