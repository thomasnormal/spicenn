from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


def class_node(class_idx: int, name: str) -> str:
    return f"c{class_idx}_{name}"


def signed_store_lines(
    *,
    positive_node: str,
    negative_node: str,
    positive_ic: float,
    negative_ic: float,
    capacitance_f: float = 20.0,
) -> list[str]:
    return [
        f"C{positive_node} {positive_node} 0 {capacitance_f:.12g}f IC={positive_ic:.12g}",
        f"C{negative_node} {negative_node} 0 {capacitance_f:.12g}f IC={negative_ic:.12g}",
        f"R{positive_node} {positive_node} 0 1e15",
        f"R{negative_node} {negative_node} 0 1e15",
    ]


def class_local_readout_forward_lines(
    *,
    class_idx: int,
    feature_idx: int,
    width_u: float = 64.0,
    negative_width_u: float = 48.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    actrow = f"actrow{feature_idx}"
    return [
        f"M{prefix}pos_cond {actrow} {class_node(class_idx, f'vwp{feature_idx}')} {class_node(class_idx, 'score')} 0 NMOS W={width_u:.6g}u L=180n",
        f"M{prefix}neg_cond {actrow} {class_node(class_idx, f'vwn{feature_idx}')} {class_node(class_idx, 'scoren')} 0 NMOS W={negative_width_u:.6g}u L=180n",
    ]


def class_local_label_descent_gradient_lines(
    *,
    class_idx: int,
    feature_idx: int,
    activation_node: str,
    width_u: float = 24.0,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    return [
        f"M{prefix}gvp_a vdd {activation_node} {prefix}gvp_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_d {prefix}gvp_a {class_node(class_idx, 'targetp')} {prefix}gvp_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvp_g {prefix}gvp_d acc {class_node(class_idx, f'gvp{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_a vdd {activation_node} {prefix}gvn_a 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_d {prefix}gvn_a {class_node(class_idx, 'targetn')} {prefix}gvn_d 0 NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}gvn_g {prefix}gvn_d acc {class_node(class_idx, f'gvn{feature_idx}')} 0 NREL W={width_u:.6g}u L=180n",
        f"M{prefix}rgp_pd {class_node(class_idx, f'rgp{feature_idx}')} {class_node(class_idx, f'gvp{feature_idx}')} 0 0 NSENSE W=16u L=180n",
        f"M{prefix}rgn_pd {class_node(class_idx, f'rgn{feature_idx}')} {class_node(class_idx, f'gvn{feature_idx}')} 0 0 NSENSE W=16u L=180n",
    ]


def class_local_bounded_update_lines(
    *,
    class_idx: int,
    feature_idx: int,
    pmos_width_u: float = 2.8,
    nmos_width_u: float = 0.7,
) -> list[str]:
    prefix = f"c{class_idx}_f{feature_idx}_"
    vwp = class_node(class_idx, f"vwp{feature_idx}")
    vwn = class_node(class_idx, f"vwn{feature_idx}")
    gvp = class_node(class_idx, f"gvp{feature_idx}")
    gvn = class_node(class_idx, f"gvn{feature_idx}")
    rgp = class_node(class_idx, f"rgp{feature_idx}")
    rgn = class_node(class_idx, f"rgn{feature_idx}")
    return [
        f"M{prefix}vwp_up_p0 {prefix}vwp_up {rgp} vwhi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
        f"M{prefix}vwp_up_p1 {vwp} applyn {prefix}vwp_up vdd PMOS W={pmos_width_u:.6g}u L=180n",
        f"M{prefix}vwn_dn_a {vwn} apply {prefix}vwn_dn 0 NREL W={nmos_width_u:.6g}u L=180n",
        f"M{prefix}vwn_dn_g {prefix}vwn_dn {gvp} vwlo_ref 0 NSENSE W={nmos_width_u:.6g}u L=180n",
        f"M{prefix}vwn_up_p0 {prefix}vwn_up {rgn} vwhi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
        f"M{prefix}vwn_up_p1 {vwn} applyn {prefix}vwn_up vdd PMOS W={pmos_width_u:.6g}u L=180n",
        f"M{prefix}vwp_dn_a {vwp} apply {prefix}vwp_dn 0 NREL W={nmos_width_u:.6g}u L=180n",
        f"M{prefix}vwp_dn_g {prefix}vwp_dn {gvn} vwlo_ref 0 NSENSE W={nmos_width_u:.6g}u L=180n",
    ]


def generate_netlist(
    *,
    class_count: int = 2,
    target_class: int = 0,
    initial_positive: float = 0.36,
    initial_negative: float = 0.34,
    target_high: float = 1.1,
    activation_v: float = 0.85,
) -> str:
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    if target_class < 0 or target_class >= class_count:
        raise ValueError("target_class must be a valid class index")
    if min(initial_positive, initial_negative, target_high, activation_v) < 0.0:
        raise ValueError("voltages must be nonnegative")
    if max(initial_positive, initial_negative, target_high, activation_v) > 1.2:
        raise ValueError("voltages must stay within supply rails")

    lines = [
        "* Class-local differential output-head primitive.",
        "* One-hot label-descent rails write independent class readout weights.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        f"Vact act0 0 {activation_v:.12g}",
        "Vactrow actrow0 0 PULSE(0 0.85 7n 10p 10p 3n 20n)",
        "Vacc acc 0 PULSE(0 1.2 1n 10p 10p 2n 20n)",
        "Vapply apply 0 PULSE(0 1.2 4n 10p 10p 2n 20n)",
        "Vapplyn applyn 0 PULSE(1.2 0 4n 10p 10p 2n 20n)",
    ]
    for class_idx in range(class_count):
        targetp = target_high if class_idx == target_class else 0.0
        targetn = 0.0 if class_idx == target_class else target_high
        lines += [
            f"V{class_node(class_idx, 'targetp')} {class_node(class_idx, 'targetp')} 0 PULSE(0 {targetp:.12g} 1n 10p 10p 2n 20n)",
            f"V{class_node(class_idx, 'targetn')} {class_node(class_idx, 'targetn')} 0 PULSE(0 {targetn:.12g} 1n 10p 10p 2n 20n)",
            f"C{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 10f IC=0",
            f"C{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 10f IC=0",
            f"R{class_node(class_idx, 'score')} {class_node(class_idx, 'score')} 0 30k",
            f"R{class_node(class_idx, 'scoren')} {class_node(class_idx, 'scoren')} 0 30k",
            f"C{class_node(class_idx, 'gvp0')} {class_node(class_idx, 'gvp0')} 0 2f IC=0",
            f"C{class_node(class_idx, 'gvn0')} {class_node(class_idx, 'gvn0')} 0 2f IC=0",
            f"C{class_node(class_idx, 'rgp0')} {class_node(class_idx, 'rgp0')} 0 4f IC=1.2",
            f"C{class_node(class_idx, 'rgn0')} {class_node(class_idx, 'rgn0')} 0 4f IC=1.2",
            f"R{class_node(class_idx, 'gvp0')} {class_node(class_idx, 'gvp0')} 0 1G",
            f"R{class_node(class_idx, 'gvn0')} {class_node(class_idx, 'gvn0')} 0 1G",
            f"R{class_node(class_idx, 'rgp0')} {class_node(class_idx, 'rgp0')} vdd 50k",
            f"R{class_node(class_idx, 'rgn0')} {class_node(class_idx, 'rgn0')} vdd 50k",
            *signed_store_lines(
                positive_node=class_node(class_idx, "vwp0"),
                negative_node=class_node(class_idx, "vwn0"),
                positive_ic=initial_positive,
                negative_ic=initial_negative,
            ),
            *class_local_label_descent_gradient_lines(
                class_idx=class_idx,
                feature_idx=0,
                activation_node="act0",
            ),
            *class_local_bounded_update_lines(class_idx=class_idx, feature_idx=0),
            *class_local_readout_forward_lines(class_idx=class_idx, feature_idx=0),
            f".meas tran c{class_idx}_signed_before PARAM='{initial_positive:.12g}-{initial_negative:.12g}'",
            f".meas tran c{class_idx}_vwp_after FIND V({class_node(class_idx, 'vwp0')}) AT=6.5n",
            f".meas tran c{class_idx}_vwn_after FIND V({class_node(class_idx, 'vwn0')}) AT=6.5n",
            f".meas tran c{class_idx}_signed_after PARAM='c{class_idx}_vwp_after-c{class_idx}_vwn_after'",
            f".meas tran c{class_idx}_signed_delta PARAM='c{class_idx}_signed_after-c{class_idx}_signed_before'",
            f".meas tran c{class_idx}_score_after FIND V({class_node(class_idx, 'score')}) AT=8.5n",
            f".meas tran c{class_idx}_scoren_after FIND V({class_node(class_idx, 'scoren')}) AT=8.5n",
            f".meas tran c{class_idx}_score_net PARAM='c{class_idx}_score_after-c{class_idx}_scoren_after'",
        ]
    lines += [
        ".tran 2p 12n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def classify_rows(rows: list[dict[str, Any]], *, target_class: int, min_signed_delta: float) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        class_idx = int(row["class"])
        delta = float(row["signed_delta_v"])
        if class_idx == target_class:
            label = "target_aligned" if delta > min_signed_delta else "target_weak_or_wrong"
        else:
            label = "nontarget_aligned" if delta < -min_signed_delta else "nontarget_weak_or_wrong"
        counts[label] = counts.get(label, 0) + 1
    return counts


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    start = time.perf_counter()
    deck = generate_netlist(class_count=args.class_count, target_class=args.target_class)
    path = generated / f"{tag}.cir"
    measures = run_netlist(spice_bin, path, deck, timeout=args.timeout)
    rows = []
    for class_idx in range(args.class_count):
        rows.append(
            {
                "class": class_idx,
                "target": class_idx == args.target_class,
                "signed_delta_v": measures[f"c{class_idx}_signed_delta"],
                "score_net_v": measures[f"c{class_idx}_score_net"],
            }
        )
    counts = classify_rows(rows, target_class=args.target_class, min_signed_delta=args.min_signed_delta)
    target_score = float(rows[args.target_class]["score_net_v"])
    nontarget_scores = [float(row["score_net_v"]) for row in rows if int(row["class"]) != args.target_class]
    best_nontarget_score = max(nontarget_scores)
    score_winner_margin = target_score - best_nontarget_score
    passed = (
        counts.get("target_aligned", 0) == 1
        and counts.get("nontarget_aligned", 0) == args.class_count - 1
        and score_winner_margin > args.min_score_margin
    )
    csv_path = tables / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "target", "signed_delta_v", "score_net_v"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "simulator": version,
        "architecture": "class_local_differential_output_head_primitive",
        "class_count": args.class_count,
        "target_class": args.target_class,
        "passed": passed,
        "classification_counts": counts,
        "target_score_net_v": target_score,
        "best_nontarget_score_net_v": best_nontarget_score,
        "score_winner_margin_v": score_winner_margin,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="multiclass_output_head_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--class-count", type=int, default=3)
    ap.add_argument("--target-class", type=int, default=1)
    ap.add_argument("--min-signed-delta", type=float, default=5e-3)
    ap.add_argument("--min-score-margin", type=float, default=5e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.class_count < 2:
        raise ValueError("class-count must be at least 2")
    if args.target_class < 0 or args.target_class >= args.class_count:
        raise ValueError("target-class must be a valid class index")
    if args.min_signed_delta < 0.0:
        raise ValueError("min-signed-delta must be nonnegative")
    if args.min_score_margin < 0.0:
        raise ValueError("min-score-margin must be nonnegative")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_case(args), indent=2))


if __name__ == "__main__":
    main()
