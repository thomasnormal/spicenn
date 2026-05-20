from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_device_readout_mobility_sweep import (
    ROOT,
    WRITE_MODES,
    branch_pair_signed_mobility_table,
    summarize_branch_pair_mobility,
)


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_branch_pair_mobility")
    ap.add_argument("--positive-csv", type=Path, required=True)
    ap.add_argument("--negative-csv", type=Path, required=True)
    ap.add_argument("--write-mode", choices=WRITE_MODES, default="bounded_discharge")
    ap.add_argument("--pos-low-v", type=float, required=True)
    ap.add_argument("--pos-high-v", type=float, required=True)
    ap.add_argument("--neg-low-v", type=float, required=True)
    ap.add_argument("--neg-high-v", type=float, required=True)
    ap.add_argument(
        "--summary-act-v",
        type=float,
        default=0.5,
        help="Activation voltage used for compact branch-pair summary statistics.",
    )
    args = ap.parse_args()
    if not 0.0 <= args.pos_low_v < args.pos_high_v <= 1.2:
        raise SystemExit("--pos-low-v/--pos-high-v must be ordered within 0..1.2 V.")
    if not 0.0 <= args.neg_low_v < args.neg_high_v <= 1.2:
        raise SystemExit("--neg-low-v/--neg-high-v must be ordered within 0..1.2 V.")

    positive_df = pd.read_csv(args.positive_csv)
    negative_df = pd.read_csv(args.negative_csv)
    pair = branch_pair_signed_mobility_table(positive_df, negative_df, args.write_mode)
    summary = summarize_branch_pair_mobility(
        pair,
        args.pos_low_v,
        args.pos_high_v,
        args.neg_low_v,
        args.neg_high_v,
        args.summary_act_v,
    )
    summary.update(
        {
            "tag": args.tag,
            "positive_csv": str(args.positive_csv),
            "negative_csv": str(args.negative_csv),
            "write_mode": args.write_mode,
        }
    )

    safe_tag = sanitize_tag(args.tag)
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    results.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    summary_path = results / f"{safe_tag}_summary.json"
    pair.to_csv(csv_path, index=False)
    pair.to_csv(table_path, index=False)
    summary["branch_pair_mobility_csv"] = str(csv_path)
    summary["branch_pair_mobility_table_csv"] = str(table_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
