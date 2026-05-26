from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


SCORE_CASES = ("positive", "negative", "neutral")


def score_values(case: str, *, center: float, delta: float) -> tuple[float, float]:
    if case == "positive":
        return center + delta / 2.0, center - delta / 2.0
    if case == "negative":
        return center - delta / 2.0, center + delta / 2.0
    if case == "neutral":
        return center, center
    raise ValueError(f"score case must be one of {SCORE_CASES}")


def generate_netlist(
    *,
    score_case: str,
    score_center: float = 0.10,
    score_delta: float = 0.04,
    pullup_width: float = 8.0,
    pulldown_width: float = 12.0,
) -> str:
    if score_case not in SCORE_CASES:
        raise ValueError(f"score_case must be one of {SCORE_CASES}")
    for name, value in {"pullup_width": pullup_width, "pulldown_width": pulldown_width}.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    score, scoren = score_values(score_case, center=score_center, delta=score_delta)
    if min(score, scoren) < 0.0 or max(score, scoren) > 1.2:
        raise ValueError("score rails must stay within supply rails")
    lines = [
        "* Score differential precharged decision primitive smoke.",
        "* Tests direct transistor sensing of score/scoren without output-latch indirection.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vscore score 0 {score:.12g}",
        f"Vscoren scoren 0 {scoren:.12g}",
        "Vrstfn rstfn 0 PULSE(0 1.2 0.8n 10p 10p 8n 10n)",
        "Vdec dec 0 PULSE(0 1.2 1.0n 10p 10p 3n 10n)",
        "Cdecision decision 0 20f IC=0",
        "Cdecisionn decisionn 0 20f IC=0",
        "Rdecision decision 0 1G",
        "Rdecisionn decisionn 0 1G",
        "Mprecharge_decision decision rstfn vdd vdd PMOS W=4u L=180n",
        "Mprecharge_decisionn decisionn rstfn vdd vdd PMOS W=4u L=180n",
        f"Mdec_scorepc_p decision decisionn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
        f"Mdecn_scorepc_p decisionn decision vdd vdd PMOS W={pullup_width:.6g}u L=180n",
        f"Mdec_scorepc_n decision scoren dec_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
        f"Mdecn_scorepc_n decisionn score dec_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
        f"Mdec_scorepc_tail dec_src dec 0 0 NMOS W={pulldown_width:.6g}u L=180n",
        ".meas tran decision_after FIND V(decision) AT=4.5n",
        ".meas tran decisionn_after FIND V(decisionn) AT=4.5n",
        ".meas tran decision_diff PARAM='decision_after-decisionn_after'",
        ".tran 2p 5n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def classify_sign(actual: float, expected: float, *, min_abs_margin: float) -> str:
    if abs(expected) < 1e-15:
        return "dead_zone" if abs(actual) < min_abs_margin else "biased"
    if expected > 0.0 and actual >= min_abs_margin:
        return "aligned"
    if expected < 0.0 and actual <= -min_abs_margin:
        return "aligned"
    return "wrong_sign"


def classify_row(row: dict[str, Any], *, min_abs_margin: float) -> dict[str, str]:
    case = str(row["score_case"])
    if case == "neutral":
        margin = abs(float(row.get("decision_diff", 0.0)))
        return {"decision_classification": "resolved" if margin >= min_abs_margin else "dead_zone"}
    expected = 1.0 if case == "positive" else -1.0 if case == "negative" else 0.0
    return {
        "decision_classification": classify_sign(
            float(row.get("decision_diff", 0.0)),
            expected,
            min_abs_margin=min_abs_margin,
        )
    }


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for case in SCORE_CASES:
        path = generated / f"{tag}_{sanitize_tag(case)}.cir"
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(
                score_case=case,
                score_center=args.score_center,
                score_delta=args.score_delta,
                pullup_width=args.pullup_width,
                pulldown_width=args.pulldown_width,
            ),
            timeout=args.timeout,
        )
        row = {"score_case": case, **measures}
        row.update(classify_row(row, min_abs_margin=args.min_abs_margin))
        rows.append(row)
    csv_path = tables / f"{tag}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    classification_counts: dict[str, int] = {}
    for row in rows:
        cls = str(row["decision_classification"])
        classification_counts[cls] = classification_counts.get(cls, 0) + 1
    passed = all(str(row["decision_classification"]) in {"aligned", "dead_zone", "resolved"} for row in rows)
    summary = {
        "simulator": version,
        "architecture": "score_diff_precharged_decision_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "cases": len(rows),
        "passed": passed,
        "classification_counts": {"decision_classification": classification_counts},
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="score_decision_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--score-center", type=float, default=0.10)
    ap.add_argument("--score-delta", type=float, default=0.04)
    ap.add_argument("--pullup-width", type=float, default=8.0)
    ap.add_argument("--pulldown-width", type=float, default=12.0)
    ap.add_argument("--min-abs-margin", type=float, default=50e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    for name in ["pullup_width", "pulldown_width"]:
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.min_abs_margin < 0.0:
        raise ValueError("min-abs-margin must be nonnegative")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_cases(args), indent=2))


if __name__ == "__main__":
    main()
