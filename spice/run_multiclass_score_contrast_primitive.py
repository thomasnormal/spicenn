from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


CONTRAST_CASES = (
    "ordered",
    "flat",
    "small_gap",
    "large_gap",
    "low_common",
)
INPUT_STAGES = ("direct", "low-gain")


def case_scores(case: str) -> tuple[float, float, float]:
    if case == "ordered":
        return (0.75, 0.45, 0.15)
    if case == "flat":
        return (0.45, 0.45, 0.45)
    if case == "small_gap":
        return (0.55, 0.45, 0.35)
    if case == "large_gap":
        return (0.85, 0.45, 0.05)
    if case == "low_common":
        return (0.0075, 0.0045, 0.0015)
    raise ValueError(f"case must be one of {CONTRAST_CASES}")


def generate_netlist(
    *,
    case: str,
    score_values: tuple[float, ...] | None = None,
    class_count: int = 3,
    input_stage: str = "direct",
    common_resistance_ohm: float | None = None,
    common_capacitance_f: float = 4.0,
    contrast_capacitance_f: float = 10.0,
    contrast_initial_v: float = 0.60,
    pullup_width_u: float = 192.0,
    pulldown_width_u: float = 24.0,
    gain_capacitance_f: float = 8.0,
    gain_input_width_u: float = 1.0,
    gain_tail_width_u: float = 8.0,
) -> str:
    if class_count != 3:
        raise ValueError("class_count must currently be 3")
    if case not in CONTRAST_CASES:
        raise ValueError(f"case must be one of {CONTRAST_CASES}")
    if input_stage not in INPUT_STAGES:
        raise ValueError(f"input_stage must be one of {INPUT_STAGES}")
    if common_resistance_ohm is None:
        # The low-gain preamp produces millivolt-scale class separation. A
        # 20 kOhm common-average network, useful for the already-amplified
        # direct-score case, loads those caps enough to erase the ordering.
        common_resistance_ohm = 10000000.0 if input_stage == "low-gain" else 20000.0
    scores = case_scores(case) if score_values is None else tuple(float(value) for value in score_values)
    if len(scores) != class_count:
        raise ValueError("score_values must have class_count entries")
    if min(scores) < 0.0 or max(scores) > 1.2:
        raise ValueError("score rails must stay within supply rails")
    for name, value in {
        "common_resistance_ohm": common_resistance_ohm,
        "common_capacitance_f": common_capacitance_f,
        "contrast_capacitance_f": contrast_capacitance_f,
        "pullup_width_u": pullup_width_u,
        "pulldown_width_u": pulldown_width_u,
        "gain_capacitance_f": gain_capacitance_f,
        "gain_input_width_u": gain_input_width_u,
        "gain_tail_width_u": gain_tail_width_u,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if contrast_initial_v < 0.0 or contrast_initial_v > 1.2:
        raise ValueError("contrast_initial_v must stay within supply rails")

    if input_stage == "direct":
        cmp_start = "1.0n"
        score_common_measure = "0.9n"
        contrast_measure = "3.2n"
        stop_time = "4n"
    else:
        cmp_start = "2.6n"
        score_common_measure = "2.55n"
        contrast_measure = "4.9n"
        stop_time = "5.5n"

    lines = [
        "* Multiclass analog score-contrast primitive smoke.",
        "* Physical common score average plus class-local charge/discharge contrast caps.",
        "* This is a pre-writer score-normalization probe; no behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vcmp cmp 0 PULSE(0 1.2 {cmp_start} 10p 10p 2.0n 10n)",
        f"Cscore_common score_common 0 {common_capacitance_f:.12g}f IC=0",
        "Rscore_common_leak score_common 0 1G",
    ]
    if input_stage == "low-gain":
        lines += [
            "Vrstfn rstfn 0 PULSE(0 1.2 0.8n 10p 10p 8n 10n)",
            "Vamp amp 0 PULSE(0 1.2 1.0n 10p 10p 1.2n 10n)",
        ]
    for class_idx, score in enumerate(scores):
        if input_stage == "direct":
            score_node = f"score{class_idx}"
            score_lines = [f"Vscore{class_idx} score{class_idx} 0 {score:.12g}"]
        else:
            score_node = f"score{class_idx}"
            score_lines = [
                f"Vscore_raw{class_idx} score_raw{class_idx} 0 {score:.12g}",
                f"Cscore{class_idx} score{class_idx} 0 {gain_capacitance_f:.12g}f IC=1.2",
                f"Rscore{class_idx} score{class_idx} 0 1G",
                f"Mprecharge_score{class_idx} score{class_idx} rstfn vdd vdd PMOS W=4u L=180n",
                f"Rscore{class_idx}_amp_i score{class_idx}_amp_i 0 1G",
                (
                    f"Mscore{class_idx}_amp_p score{class_idx} score_raw{class_idx} "
                    f"score{class_idx}_amp_i vdd PMOS W={gain_input_width_u:.6g}u L=180n"
                ),
                (
                    f"Mscore{class_idx}_amp_tail score{class_idx}_amp_i amp 0 0 "
                    f"NMOS W={gain_tail_width_u:.6g}u L=180n"
                ),
            ]
        lines += [
            *score_lines,
            f"Rscore_common_c{class_idx} score_common {score_node} {common_resistance_ohm:.12g}",
            f"Ccontrast{class_idx} contrast{class_idx} 0 {contrast_capacitance_f:.12g}f IC={contrast_initial_v:.12g}",
            f"Rcontrast{class_idx} contrast{class_idx} 0 1G",
            f"Rcontrast{class_idx}_up contrast{class_idx}_up 0 1G",
            f"Rcontrast{class_idx}_dn contrast{class_idx}_dn 0 1G",
            f"Mcontrast{class_idx}_up_v vdd {score_node} contrast{class_idx}_up 0 NREL W={pullup_width_u:.6g}u L=180n",
            f"Mcontrast{class_idx}_up_t contrast{class_idx}_up cmp contrast{class_idx} 0 NSENSE W={pullup_width_u:.6g}u L=180n",
            f"Mcontrast{class_idx}_dn_v contrast{class_idx} score_common contrast{class_idx}_dn 0 NREL W={pulldown_width_u:.6g}u L=180n",
            f"Mcontrast{class_idx}_dn_t contrast{class_idx}_dn cmp 0 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
            f".meas tran score{class_idx}_norm_after FIND V({score_node}) AT={score_common_measure}",
            f".meas tran contrast{class_idx}_after FIND V(contrast{class_idx}) AT={contrast_measure}",
        ]
    lines += [
        f".meas tran score_common_after FIND V(score_common) AT={score_common_measure}",
        ".meas tran contrast_0_1_margin PARAM='contrast0_after-contrast1_after'",
        ".meas tran contrast_1_2_margin PARAM='contrast1_after-contrast2_after'",
        ".meas tran contrast_spread PARAM='contrast0_after-contrast2_after'",
        f".tran 2p {stop_time} uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def classify_case(row: dict[str, Any], *, min_margin: float) -> str:
    case = str(row["case"])
    input_stage = str(row.get("input_stage", "direct"))
    spread = abs(float(row["contrast_spread"]))
    if case == "flat":
        return "dead_zone" if spread < min_margin else "active"
    if case == "low_common" and input_stage == "direct":
        return "quiet" if spread < min_margin else "active"
    if float(row["contrast_0_1_margin"]) > min_margin and float(row["contrast_1_2_margin"]) > min_margin:
        return "ordered"
    return "weak"


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for case in CONTRAST_CASES:
        path = generated / f"{tag}_{sanitize_tag(case)}.cir"
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(
                case=case,
                input_stage=args.input_stage,
                common_resistance_ohm=args.common_resistance,
                common_capacitance_f=args.common_capacitance_f,
                contrast_capacitance_f=args.contrast_capacitance_f,
                pullup_width_u=args.pullup_width,
                pulldown_width_u=args.pulldown_width,
                gain_capacitance_f=args.gain_capacitance_f,
                gain_input_width_u=args.gain_input_width,
                gain_tail_width_u=args.gain_tail_width,
            ),
            timeout=args.timeout,
        )
        row = {"case": case, "input_stage": args.input_stage, **measures}
        row["classification"] = classify_case(row, min_margin=args.min_margin)
        rows.append(row)
    csv_path = tables / f"{tag}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        cls = str(row["classification"])
        counts[cls] = counts.get(cls, 0) + 1
    passed = all(str(row["classification"]) in {"ordered", "dead_zone", "quiet"} for row in rows)
    summary = {
        "simulator": version,
        "architecture": "multiclass_score_contrast_primitive",
        "input_stage": args.input_stage,
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "cases": len(rows),
        "passed": passed,
        "classification_counts": counts,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="multiclass_score_contrast_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--input-stage", choices=INPUT_STAGES, default="direct")
    ap.add_argument("--common-resistance", type=float, default=None)
    ap.add_argument("--common-capacitance-f", type=float, default=4.0)
    ap.add_argument("--contrast-capacitance-f", type=float, default=10.0)
    ap.add_argument("--pullup-width", type=float, default=192.0)
    ap.add_argument("--pulldown-width", type=float, default=24.0)
    ap.add_argument("--gain-capacitance-f", type=float, default=8.0)
    ap.add_argument("--gain-input-width", type=float, default=1.0)
    ap.add_argument("--gain-tail-width", type=float, default=8.0)
    ap.add_argument("--min-margin", type=float, default=0.005)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    for name in (
        "common_capacitance_f",
        "contrast_capacitance_f",
        "pullup_width",
        "pulldown_width",
        "gain_capacitance_f",
        "gain_input_width",
        "gain_tail_width",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.common_resistance is not None and args.common_resistance <= 0.0:
        raise ValueError("common-resistance must be positive")
    if args.min_margin < 0.0:
        raise ValueError("min-margin must be nonnegative")


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
