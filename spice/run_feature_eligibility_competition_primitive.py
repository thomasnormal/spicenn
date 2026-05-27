from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


CASES: dict[str, tuple[float, ...]] = {
    "one_hot0": (0.90, 0.06, 0.03),
    "one_hot1": (0.05, 0.86, 0.04),
    "unique_dense0": (0.541, 0.313, 0.080, 0.360, 0.247),
    "unique_dense4": (0.314, 0.541, 0.518, 0.489, 0.760, 0.488),
    "flat_dense": (0.541, 0.541, 0.541, 0.541, 0.541),
    "fixed8_like0": (0.541, 0.541, 0.541, 0.541, 0.080, 0.541, 0.247, 0.541, 0.541),
    "fixed8_like1": (0.313, 0.080, 0.080, 0.360, 0.541, 0.080, 0.080, 0.080, 0.541),
    "fixed8_like2": (0.314, 0.541, 0.518, 0.489, 0.541, 0.488, 0.541, 0.541, 0.541),
}


def elig_node(feature_idx: int) -> str:
    return f"elig{feature_idx}"


def decision_node(winner_idx: int, loser_idx: int) -> str:
    return f"e{winner_idx}_gt_e{loser_idx}_decision"


def gate_node(feature_idx: int) -> str:
    return f"egate{feature_idx}"


def pairwise_feature_winner_lines(
    *,
    feature_a: int,
    feature_b: int,
    decision_clock_node: str = "eligdec",
    reset_node: str = "rst",
    width_u: float = 32.0,
) -> list[str]:
    prefix = f"e{feature_a}_gt_e{feature_b}_"
    node_ab = decision_node(feature_a, feature_b)
    node_ba = decision_node(feature_b, feature_a)
    keeper_width_u = max(1.0, width_u / 64.0)
    return [
        f"C{node_ab} {node_ab} 0 12f IC=1.2",
        f"C{node_ba} {node_ba} 0 12f IC=1.2",
        f"R{node_ab} {node_ab} 0 1G",
        f"R{node_ba} {node_ba} 0 1G",
        f"Mprecharge_{node_ab} {node_ab} {reset_node} vdd vdd PMOS W=4u L=180n",
        f"Mprecharge_{node_ba} {node_ba} {reset_node} vdd vdd PMOS W=4u L=180n",
        f"M{prefix}ab_dis_s {node_ab} {elig_node(feature_b)} {prefix}ab_dn 0 NMOS W={width_u:.6g}u L=180n",
        f"M{prefix}ab_dis_e {prefix}ab_dn {decision_clock_node} 0 0 NMOS W={width_u:.6g}u L=180n",
        f"M{prefix}ba_dis_s {node_ba} {elig_node(feature_a)} {prefix}ba_dn 0 NMOS W={width_u:.6g}u L=180n",
        f"M{prefix}ba_dis_e {prefix}ba_dn {decision_clock_node} 0 0 NMOS W={width_u:.6g}u L=180n",
        f"M{prefix}ab_keep_p {node_ab} {node_ba} vdd vdd PMOS W={keeper_width_u:.6g}u L=180n",
        f"M{prefix}ba_keep_p {node_ba} {node_ab} vdd vdd PMOS W={keeper_width_u:.6g}u L=180n",
        f"M{prefix}ab_keep_n {node_ab} {node_ba} 0 0 NMOS W={keeper_width_u:.6g}u L=180n",
        f"M{prefix}ba_keep_n {node_ba} {node_ab} 0 0 NMOS W={keeper_width_u:.6g}u L=180n",
    ]


def feature_loss_suppression_lines(
    *,
    feature_count: int,
    gate_clock_node: str = "eliggate",
    reset_node: str = "rst",
    gate_capacitance_f: float = 8.0,
    loss_width_u: float = 32.0,
) -> list[str]:
    lines: list[str] = []
    for feature_idx in range(feature_count):
        gate = gate_node(feature_idx)
        lines += [
            f"C{gate} {gate} 0 {gate_capacitance_f:.12g}f IC=0",
            f"R{gate} {gate} 0 1G",
            f"Mprecharge_{gate} {gate} {reset_node} vdd vdd PMOS W=4u L=180n",
        ]
        for opponent_idx in range(feature_count):
            if opponent_idx == feature_idx:
                continue
            loss_mid = f"e{feature_idx}_loss_to_e{opponent_idx}_mid"
            loss_decision = decision_node(opponent_idx, feature_idx)
            lines += [
                f"M{loss_mid}_dec {gate} {loss_decision} {loss_mid} 0 NHIGH W={loss_width_u:.6g}u L=180n",
                f"M{loss_mid}_clk {loss_mid} {gate_clock_node} 0 0 NMOS W={loss_width_u:.6g}u L=180n",
            ]
    return lines


def values_for_case(case: str, eligibility_values: tuple[float, ...] | None = None) -> tuple[float, ...]:
    if eligibility_values is not None:
        values = tuple(float(value) for value in eligibility_values)
    elif case in CASES:
        values = CASES[case]
    else:
        raise ValueError(f"case must be one of {tuple(CASES)} unless eligibility_values is supplied")
    if len(values) < 2:
        raise ValueError("eligibility_values must contain at least two features")
    if min(values) < 0.0 or max(values) > 1.2:
        raise ValueError("eligibility values must stay within supply rails")
    return values


def generate_netlist(
    *,
    case: str = "one_hot0",
    eligibility_values: tuple[float, ...] | None = None,
    pairwise_width_u: float = 64.0,
    gate_loss_width_u: float = 32.0,
    gate_capacitance_f: float = 8.0,
) -> str:
    if min(pairwise_width_u, gate_loss_width_u, gate_capacitance_f) <= 0.0:
        raise ValueError("widths and capacitances must be positive")
    values = values_for_case(case, eligibility_values)
    feature_count = len(values)
    lines = [
        "* Feature eligibility competition primitive.",
        "* Pairwise transistor score decisions suppress every feature beaten by another feature.",
        "* Python supplies forced eligibility rails and clocks only; no behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".model NHIGH NMOS LEVEL=1 VTO=0.75 KP=220u LAMBDA=0.03 GAMMA=0.20 PHI=0.60",
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(0 1.2 0.50n 10p 10p 8n 10n)",
        "Veligdec eligdec 0 PULSE(0 1.2 0.90n 10p 10p 1.60n 10n)",
        "Veliggate eliggate 0 PULSE(0 1.2 3.50n 10p 10p 1.00n 10n)",
    ]
    for idx, value in enumerate(values):
        lines.append(f"V{elig_node(idx)} {elig_node(idx)} 0 {value:.12g}")
    for feature_a in range(feature_count):
        for feature_b in range(feature_a + 1, feature_count):
            lines += pairwise_feature_winner_lines(
                feature_a=feature_a,
                feature_b=feature_b,
                width_u=pairwise_width_u,
            )
    lines += feature_loss_suppression_lines(
        feature_count=feature_count,
        gate_capacitance_f=gate_capacitance_f,
        loss_width_u=gate_loss_width_u,
    )
    for idx in range(feature_count):
        lines.append(f".meas tran {gate_node(idx)}_after FIND V({gate_node(idx)}) AT=4.80n")
        for opponent_idx in range(feature_count):
            if opponent_idx == idx:
                continue
            lines.append(
                f".meas tran e{idx}_gt_e{opponent_idx}_after FIND V({decision_node(idx, opponent_idx)}) AT=3.40n"
            )
    lines += [
        ".tran 2p 5.0n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def summarize_case(measures: dict[str, Any], *, feature_count: int, active_threshold: float) -> dict[str, Any]:
    gates = [float(measures[f"{gate_node(idx)}_after"]) for idx in range(feature_count)]
    active = [idx for idx, value in enumerate(gates) if value >= active_threshold]
    return {
        "gate_values": gates,
        "active_features": active,
        "active_count": len(active),
        "max_gate": max(gates),
        "min_gate": min(gates),
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
    for case, values in CASES.items():
        path = generated / f"{tag}_{sanitize_tag(case)}.cir"
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(
                case=case,
                pairwise_width_u=args.pairwise_width,
                gate_loss_width_u=args.gate_loss_width,
                gate_capacitance_f=args.gate_capacitance_f,
            ),
            timeout=args.timeout,
        )
        summary = summarize_case(measures, feature_count=len(values), active_threshold=args.active_threshold)
        rows.append({"case": case, "feature_count": len(values), **measures, **summary})
    csv_path = tables / f"{tag}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    passed = all(int(row["active_count"]) <= args.max_active for row in rows)
    summary = {
        "tag": tag,
        "spice": version,
        "cases": [row["case"] for row in rows],
        "passed": passed,
        "elapsed_s": time.perf_counter() - start,
        "csv_path": str(csv_path),
        "active_threshold": args.active_threshold,
        "max_active": args.max_active,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="feature_eligibility_competition")
    parser.add_argument("--spice-bin", default=None)
    parser.add_argument("--pairwise-width", type=float, default=64.0)
    parser.add_argument("--gate-loss-width", type=float, default=32.0)
    parser.add_argument("--gate-capacitance-f", type=float, default=8.0)
    parser.add_argument("--active-threshold", type=float, default=0.6)
    parser.add_argument("--max-active", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main_for_test(argv: list[str]) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if min(args.pairwise_width, args.gate_loss_width, args.gate_capacitance_f, args.active_threshold) <= 0.0:
        raise ValueError("widths, capacitances, and thresholds must be positive")
    if args.max_active < 0:
        raise ValueError("max-active must be nonnegative")
    return run_cases(args)


def main() -> None:
    args = build_arg_parser().parse_args()
    if min(args.pairwise_width, args.gate_loss_width, args.gate_capacitance_f, args.active_threshold) <= 0.0:
        raise ValueError("widths, capacitances, and thresholds must be positive")
    if args.max_active < 0:
        raise ValueError("max-active must be nonnegative")
    run_cases(args)


if __name__ == "__main__":
    main()
