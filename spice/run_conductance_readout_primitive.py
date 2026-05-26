from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


READOUT_CASES = ("positive", "negative", "neutral", "inactive")
SUM_CASES = ("single_positive", "two_positive", "mixed_cancel", "inactive_extra")
SUM_ISOLATION_MODES = ("direct", "diode")


def readout_values(case: str, *, positive_weight: float, negative_weight: float) -> tuple[float, float, float]:
    if case == "positive":
        return 0.85, positive_weight, negative_weight
    if case == "negative":
        return 0.85, negative_weight, positive_weight
    if case == "neutral":
        neutral = 0.5 * (positive_weight + negative_weight)
        return 0.85, neutral, neutral
    if case == "inactive":
        return 0.0, positive_weight, negative_weight
    raise ValueError(f"readout case must be one of {READOUT_CASES}")


def generate_netlist(
    *,
    readout_case: str,
    positive_weight: float = 0.50,
    negative_weight: float = 0.34,
    readout_width: float = 64.0,
    score_capacitance: float = 10e-15,
) -> str:
    if readout_case not in READOUT_CASES:
        raise ValueError(f"readout_case must be one of {READOUT_CASES}")
    for name, value in {"readout_width": readout_width, "score_capacitance": score_capacitance}.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    act, vwp, vwn = readout_values(readout_case, positive_weight=positive_weight, negative_weight=negative_weight)
    lines = [
        "* Conductance-row readout primitive smoke.",
        "* Tests act row -> stored conductance pair -> score/scoren capacitors.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(0 1.2 0.0n 10p 10p 0.55n 20n)",
        "Vfwd fwd 0 PULSE(0 1.2 1.0n 10p 10p 3.0n 20n)",
        "Vfwdn fwdn 0 PULSE(1.2 0 1.0n 10p 10p 3.0n 20n)",
        f"Vact act 0 {act:.12g}",
        f"Cvwp vwp 0 20f IC={vwp:.12g}",
        f"Cvwn vwn 0 20f IC={vwn:.12g}",
        "Rvwp vwp 0 1e15",
        "Rvwn vwn 0 1e15",
        f"Cscore score 0 {score_capacitance:.12g} IC=0",
        f"Cscoren scoren 0 {score_capacitance:.12g} IC=0",
        "Cactrow actrow 0 1f IC=0",
        "Rscore score 0 1G",
        "Rscoren scoren 0 1G",
        "Ractrow actrow 0 1e12",
        f"Mactrow_n actrow fwd act 0 NMOS W={max(1.0, readout_width / 4.0):.6g}u L=180n",
        f"Mactrow_p actrow fwdn act vdd PMOS W={max(2.0, readout_width / 2.0):.6g}u L=180n",
        "Mactrow_rst actrow rst 0 0 NMOS W=4u L=180n",
        f"Movpos_cond actrow vwp score 0 NMOS W={readout_width:.6g}u L=180n",
        f"Movneg_cond actrow vwn scoren 0 NMOS W={0.75 * readout_width:.6g}u L=180n",
        ".meas tran actrow_after FIND V(actrow) AT=4.5n",
        ".meas tran score_after FIND V(score) AT=4.5n",
        ".meas tran scoren_after FIND V(scoren) AT=4.5n",
        ".meas tran score_margin PARAM='score_after-scoren_after'",
        ".meas tran score_common PARAM='0.5*(score_after+scoren_after)'",
        ".tran 5p 8n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def sum_features(case: str, *, positive_weight: float, negative_weight: float) -> list[tuple[float, float, float]]:
    positive = (0.85, positive_weight, negative_weight)
    negative = (0.85, negative_weight, positive_weight)
    inactive = (0.0, positive_weight, negative_weight)
    if case == "single_positive":
        return [positive]
    if case == "two_positive":
        return [positive, positive]
    if case == "mixed_cancel":
        return [positive, negative]
    if case == "inactive_extra":
        return [positive, inactive]
    raise ValueError(f"sum case must be one of {SUM_CASES}")


def generate_sum_netlist(
    *,
    sum_case: str,
    positive_weight: float = 0.50,
    negative_weight: float = 0.34,
    readout_width: float = 64.0,
    score_capacitance: float = 10e-15,
    isolation: str = "direct",
    score_load_resistance: float = 1e9,
) -> str:
    if sum_case not in SUM_CASES:
        raise ValueError(f"sum_case must be one of {SUM_CASES}")
    if isolation not in SUM_ISOLATION_MODES:
        raise ValueError(f"isolation must be one of {SUM_ISOLATION_MODES}")
    for name, value in {
        "readout_width": readout_width,
        "score_capacitance": score_capacitance,
        "score_load_resistance": score_load_resistance,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    features = sum_features(sum_case, positive_weight=positive_weight, negative_weight=negative_weight)
    lines = [
        "* Multi-feature conductance-row readout summation primitive smoke.",
        "* Tests several actrow -> conductance branches sharing score/scoren capacitors.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(0 1.2 0.0n 10p 10p 0.55n 20n)",
        "Vfwd fwd 0 PULSE(0 1.2 1.0n 10p 10p 3.0n 20n)",
        "Vfwdn fwdn 0 PULSE(1.2 0 1.0n 10p 10p 3.0n 20n)",
        f"Cscore score 0 {score_capacitance:.12g} IC=0",
        f"Cscoren scoren 0 {score_capacitance:.12g} IC=0",
        f"Rscore score 0 {score_load_resistance:.12g}",
        f"Rscoren scoren 0 {score_load_resistance:.12g}",
    ]
    for index, (act, vwp, vwn) in enumerate(features):
        lines += [
            f"Vact{index} act{index} 0 {act:.12g}",
            f"Cvwp{index} vwp{index} 0 20f IC={vwp:.12g}",
            f"Cvwn{index} vwn{index} 0 20f IC={vwn:.12g}",
            f"Rvwp{index} vwp{index} 0 1e15",
            f"Rvwn{index} vwn{index} 0 1e15",
            f"Cactrow{index} actrow{index} 0 1f IC=0",
            f"Ractrow{index} actrow{index} 0 1e12",
            f"Mactrow{index}_n actrow{index} fwd act{index} 0 NMOS W={max(1.0, readout_width / 4.0):.6g}u L=180n",
            f"Mactrow{index}_p actrow{index} fwdn act{index} vdd PMOS W={max(2.0, readout_width / 2.0):.6g}u L=180n",
            f"Mactrow{index}_rst actrow{index} rst 0 0 NMOS W=4u L=180n",
        ]
        if isolation == "direct":
            lines += [
                f"Movpos{index}_cond actrow{index} vwp{index} score 0 NMOS W={readout_width:.6g}u L=180n",
                f"Movneg{index}_cond actrow{index} vwn{index} scoren 0 NMOS W={0.75 * readout_width:.6g}u L=180n",
            ]
        else:
            lines += [
                f"Cmidp{index} midp{index} 0 0.1f IC=0",
                f"Cmidn{index} midn{index} 0 0.1f IC=0",
                f"Rmidp{index} midp{index} 0 1G",
                f"Rmidn{index} midn{index} 0 1G",
                f"Movpos{index}_cond actrow{index} vwp{index} midp{index} 0 NMOS W={readout_width:.6g}u L=180n",
                f"Movpos{index}_diode midp{index} midp{index} score 0 NSENSE W={readout_width:.6g}u L=180n",
                f"Movneg{index}_cond actrow{index} vwn{index} midn{index} 0 NMOS W={0.75 * readout_width:.6g}u L=180n",
                f"Movneg{index}_diode midn{index} midn{index} scoren 0 NSENSE W={0.75 * readout_width:.6g}u L=180n",
            ]
    lines += [
        ".meas tran score_after FIND V(score) AT=4.5n",
        ".meas tran scoren_after FIND V(scoren) AT=4.5n",
        ".meas tran score_margin PARAM='score_after-scoren_after'",
        ".meas tran score_common PARAM='0.5*(score_after+scoren_after)'",
        ".tran 5p 8n uic",
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
    case = str(row["readout_case"])
    expected = 1.0 if case == "positive" else -1.0 if case == "negative" else 0.0
    return {
        "score_classification": classify_sign(
            float(row.get("score_margin", 0.0)),
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
    for case in READOUT_CASES:
        path = generated / f"{tag}_{sanitize_tag(case)}.cir"
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(
                readout_case=case,
                positive_weight=args.positive_weight,
                negative_weight=args.negative_weight,
                readout_width=args.readout_width,
                score_capacitance=args.score_capacitance,
            ),
            timeout=args.timeout,
        )
        row = {"readout_case": case, **measures}
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
        cls = str(row["score_classification"])
        classification_counts[cls] = classification_counts.get(cls, 0) + 1
    passed = all(str(row["score_classification"]) in {"aligned", "dead_zone"} for row in rows)
    summary = {
        "simulator": version,
        "architecture": "conductance_row_readout_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "cases": len(rows),
        "passed": passed,
        "classification_counts": {"score_classification": classification_counts},
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="conductance_readout_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--positive-weight", type=float, default=0.50)
    ap.add_argument("--negative-weight", type=float, default=0.34)
    ap.add_argument("--readout-width", type=float, default=64.0)
    ap.add_argument("--score-capacitance", type=float, default=10e-15)
    ap.add_argument("--min-abs-margin", type=float, default=1e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    for name in ["readout_width", "score_capacitance"]:
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
