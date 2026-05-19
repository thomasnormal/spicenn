from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


FIELDS = [
    "tag",
    "dataset",
    "input_frontend_key",
    "hidden_cells",
    "hidden_init",
    "flow_pre_store",
    "flow_hidden_write",
    "hidden_update_width_u",
    "epochs",
    "final_eval_accuracy",
    "best_final_transient_accuracy",
    "best_final_transient_min_margin_v",
    "input_feature_separable",
    "initial_hidden_separable",
    "final_hidden_separable",
    "final_hidden_min_margin",
    "max_abs_total_readout_signed_delta_v",
    "max_abs_total_hidden_signed_delta_v",
    "wall_time_s",
    "summary_path",
]


def nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def row_from_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    tag = path.name.removesuffix("_summary.json")
    return {
        "tag": tag,
        "dataset": data.get("dataset") or data.get("benchmark"),
        "input_frontend_key": data.get("input_frontend_key"),
        "hidden_cells": data.get("hidden_cells"),
        "hidden_init": data.get("hidden_init"),
        "flow_pre_store": data.get("flow_pre_store"),
        "flow_hidden_write": data.get("flow_hidden_write"),
        "hidden_update_width_u": data.get("hidden_update_width_u"),
        "epochs": data.get("epochs"),
        "final_eval_accuracy": data.get("final_eval_accuracy"),
        "best_final_transient_accuracy": data.get("best_final_transient_accuracy"),
        "best_final_transient_min_margin_v": data.get("best_final_transient_min_margin_v"),
        "input_feature_separable": nested(data, "input_feature_separability", "linearly_separable"),
        "initial_hidden_separable": nested(data, "initial_hidden_feature_separability", "linearly_separable"),
        "final_hidden_separable": nested(data, "final_hidden_feature_separability", "linearly_separable"),
        "final_hidden_min_margin": nested(data, "final_hidden_feature_separability", "min_margin"),
        "max_abs_total_readout_signed_delta_v": data.get("max_abs_total_readout_signed_delta_v"),
        "max_abs_total_hidden_signed_delta_v": data.get("max_abs_total_hidden_signed_delta_v"),
        "wall_time_s": data.get("wall_time_s"),
        "summary_path": str(path),
    }


def resolve_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = glob.glob(str(ROOT / pattern)) if not Path(pattern).is_absolute() else glob.glob(pattern)
        if expanded:
            paths.extend(Path(item) for item in expanded)
        else:
            paths.append(ROOT / pattern if not Path(pattern).is_absolute() else Path(pattern))
    unique = sorted({path.resolve() for path in paths})
    missing = [path for path in unique if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing summary path(s):\n{joined}")
    return unique


def print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "tag",
        "dataset",
        "flow_hidden_write",
        "hidden_update_width_u",
        "final_eval_accuracy",
        "best_final_transient_min_margin_v",
        "final_hidden_separable",
        "max_abs_total_hidden_signed_delta_v",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize direct-flow random-hidden SPICE JSON results.")
    ap.add_argument(
        "summaries",
        nargs="*",
        help="Summary JSON paths or glob patterns. Defaults to recent device_mnist01 random-hidden summaries.",
    )
    ap.add_argument("--out", type=Path, help="Optional CSV output path.")
    ap.add_argument("--markdown", action="store_true", help="Print a compact Markdown table.")
    args = ap.parse_args()

    patterns = args.summaries or ["spice/results/device_mnist01*_random_hidden*_summary.json"]
    rows = [row_from_summary(path) for path in resolve_paths(patterns)]
    rows.sort(key=lambda row: (str(row.get("dataset")), str(row.get("tag"))))

    if args.out:
        out_path = args.out if args.out.is_absolute() else ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    if args.markdown or not args.out:
        print_markdown(rows)


if __name__ == "__main__":
    main()
