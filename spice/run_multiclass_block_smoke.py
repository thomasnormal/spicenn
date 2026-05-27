from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_multiclass_output_head_primitive import (
    class_local_bounded_update_lines,
    class_local_label_descent_gradient_lines,
    class_local_readout_forward_lines,
    class_node,
    signed_store_lines,
)
from run_spice_sweep import ROOT, detect_spice


DEFAULT_CLASS_READOUTS = (
    (0.40, 0.30),
    (0.30, 0.40),
    (0.34, 0.34),
)


def class_readout_initials(class_count: int) -> list[tuple[float, float]]:
    if class_count <= len(DEFAULT_CLASS_READOUTS):
        return list(DEFAULT_CLASS_READOUTS[:class_count])
    initials = list(DEFAULT_CLASS_READOUTS)
    initials.extend([(0.34, 0.34)] * (class_count - len(initials)))
    return initials


def generate_netlist(
    *,
    class_count: int = 3,
    target_class: int = 0,
    input_v: float = 0.85,
    hidden_positive: float = 1.00,
    hidden_negative: float = 0.20,
    hidden_width_u: float = 1.0,
    readout_width_u: float = 64.0,
    target_high: float = 1.1,
) -> str:
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    if target_class < 0 or target_class >= class_count:
        raise ValueError("target_class must be a valid class index")
    for name, value in {
        "input_v": input_v,
        "hidden_positive": hidden_positive,
        "hidden_negative": hidden_negative,
        "hidden_width_u": hidden_width_u,
        "readout_width_u": readout_width_u,
        "target_high": target_high,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if max(input_v, hidden_positive, hidden_negative, target_high) > 1.2:
        raise ValueError("voltages must stay within supply rails")

    lines = [
        "* Multiclass block smoke: row-pulsed split-rail hidden feature, class-local scores, local writes.",
        "* This is a transistor/passive low-level check, not a behavioral training shortcut.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        "Vrst rst 0 PULSE(0 1.2 0.0n 10p 10p 0.55n 20n)",
        f"Vrow row0 0 PULSE(0 {input_v:.12g} 1.0n 10p 10p 3.0n 20n)",
        "Vsamp samp 0 PULSE(0 1.2 2.5n 10p 10p 1.0n 20n)",
        "Vsampn sampn 0 PULSE(1.2 0 2.5n 10p 10p 1.0n 20n)",
        "Vout out 0 PULSE(0 1.2 5.0n 10p 10p 3.0n 20n)",
        "Voutn outn 0 PULSE(1.2 0 5.0n 10p 10p 3.0n 20n)",
        "Vacc acc 0 PULSE(0 1.2 9.0n 10p 10p 2.0n 20n)",
        "Vapply apply 0 PULSE(0 1.2 12.0n 10p 10p 2.0n 20n)",
        "Vapplyn applyn 0 PULSE(1.2 0 12.0n 10p 10p 2.0n 20n)",
        f"Cwhp whp 0 20f IC={hidden_positive:.12g}",
        f"Cwhn whn 0 20f IC={hidden_negative:.12g}",
        "Rwhp whp 0 1e15",
        "Rwhn whn 0 1e15",
        "Cpre_p pre_p 0 20f IC=0",
        "Cpre_n pre_n 0 20f IC=0",
        "Cact_raw act_raw 0 20f IC=0",
        "Cact_store act0 0 20f IC=0",
        "Cact_grad actg0 0 20f IC=0",
        "Celig elig0 0 20f IC=0",
        "Cactrow actrow0 0 1f IC=0",
        "Rpre_p pre_p 0 1G",
        "Rpre_n pre_n 0 1G",
        "Ract_raw act_raw 0 1G",
        "Ract_store act0 0 1G",
        "Ract_grad actg0 0 1G",
        "Relig elig0 0 1G",
        "Ractrow actrow0 0 1e12",
        "Mpre_p_rst pre_p rst 0 0 NMOS W=4u L=180n",
        "Mpre_n_rst pre_n rst 0 0 NMOS W=4u L=180n",
        "Mact_raw_rst act_raw rst 0 0 NMOS W=4u L=180n",
        "Mact_store_rst act0 rst 0 0 NMOS W=4u L=180n",
        "Mact_grad_rst actg0 rst 0 0 NMOS W=4u L=180n",
        "Melig_rst elig0 rst 0 0 NMOS W=4u L=180n",
        "Mactrow_rst actrow0 rst 0 0 NMOS W=4u L=180n",
        f"Mhidden_pos row0 whp pre_p 0 NMOS W={hidden_width_u:.6g}u L=180n",
        f"Mhidden_neg row0 whn pre_n 0 NMOS W={hidden_width_u:.6g}u L=180n",
        "Mact_p vdd pre_p act_raw 0 NREL W=24u L=180n",
        "Mact_n act_raw pre_n 0 0 NREL W=24u L=180n",
        "Mact_store_n act0 samp act_raw 0 NMOS W=16u L=180n",
        "Mact_store_p act0 sampn act_raw vdd PMOS W=32u L=180n",
        "Mact_grad_n actg0 samp act_raw 0 NMOS W=16u L=180n",
        "Mact_grad_p actg0 sampn act_raw vdd PMOS W=32u L=180n",
        "Melig_n elig0 samp pre_p 0 NMOS W=16u L=180n",
        "Melig_p elig0 sampn pre_p vdd PMOS W=32u L=180n",
        "Mactrow_n actrow0 out act0 0 NMOS W=16u L=180n",
        "Mactrow_p actrow0 outn act0 vdd PMOS W=32u L=180n",
        ".meas tran pre_p_after FIND V(pre_p) AT=3.2n",
        ".meas tran pre_n_after FIND V(pre_n) AT=3.2n",
        ".meas tran pre_margin PARAM='pre_p_after-pre_n_after'",
        ".meas tran act_raw_after FIND V(act_raw) AT=3.2n",
        ".meas tran act_after FIND V(actg0) AT=4.5n",
        ".meas tran act_update_after FIND V(actg0) AT=9.5n",
        ".meas tran eligibility_after FIND V(elig0) AT=4.5n",
        ".meas tran actrow_after FIND V(actrow0) AT=8.5n",
    ]
    for class_idx, (initial_positive, initial_negative) in enumerate(class_readout_initials(class_count)):
        targetp = target_high if class_idx == target_class else 0.0
        targetn = 0.0 if class_idx == target_class else target_high
        lines += [
            f"V{class_node(class_idx, 'targetp')} {class_node(class_idx, 'targetp')} 0 PULSE(0 {targetp:.12g} 9.0n 10p 10p 2.0n 20n)",
            f"V{class_node(class_idx, 'targetn')} {class_node(class_idx, 'targetn')} 0 PULSE(0 {targetn:.12g} 9.0n 10p 10p 2.0n 20n)",
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
            *class_local_readout_forward_lines(
                class_idx=class_idx,
                feature_idx=0,
                width_u=readout_width_u,
                negative_width_u=0.75 * readout_width_u,
            ),
            *class_local_label_descent_gradient_lines(
                class_idx=class_idx,
                feature_idx=0,
                activation_node="elig0",
            ),
            *class_local_bounded_update_lines(class_idx=class_idx, feature_idx=0),
            f".meas tran c{class_idx}_signed_before PARAM='{initial_positive:.12g}-{initial_negative:.12g}'",
            f".meas tran c{class_idx}_score_after FIND V({class_node(class_idx, 'score')}) AT=8.5n",
            f".meas tran c{class_idx}_scoren_after FIND V({class_node(class_idx, 'scoren')}) AT=8.5n",
            f".meas tran c{class_idx}_score_net PARAM='c{class_idx}_score_after-c{class_idx}_scoren_after'",
            f".meas tran c{class_idx}_vwp_after FIND V({class_node(class_idx, 'vwp0')}) AT=15.0n",
            f".meas tran c{class_idx}_vwn_after FIND V({class_node(class_idx, 'vwn0')}) AT=15.0n",
            f".meas tran c{class_idx}_signed_after PARAM='c{class_idx}_vwp_after-c{class_idx}_vwn_after'",
            f".meas tran c{class_idx}_signed_delta PARAM='c{class_idx}_signed_after-c{class_idx}_signed_before'",
        ]
    lines += [
        ".tran 2p 16n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    start = time.perf_counter()
    path = generated / f"{tag}.cir"
    measures = run_netlist(
        spice_bin,
        path,
        generate_netlist(class_count=args.class_count, target_class=args.target_class),
        timeout=args.timeout,
    )
    rows = []
    for class_idx in range(args.class_count):
        rows.append(
            {
                "class": class_idx,
                "target": class_idx == args.target_class,
                "score_net_v": measures[f"c{class_idx}_score_net"],
                "signed_delta_v": measures[f"c{class_idx}_signed_delta"],
            }
        )
    target_score = float(rows[args.target_class]["score_net_v"])
    best_nontarget_score = max(float(row["score_net_v"]) for row in rows if int(row["class"]) != args.target_class)
    target_delta = float(rows[args.target_class]["signed_delta_v"])
    nontarget_deltas = [float(row["signed_delta_v"]) for row in rows if int(row["class"]) != args.target_class]
    score_winner_margin = target_score - best_nontarget_score
    passed = (
        float(measures["pre_margin"]) > args.min_pre_margin
        and float(measures["act_after"]) > args.min_activation
        and score_winner_margin > args.min_score_margin
        and target_delta > args.min_signed_delta
        and all(delta < -args.min_signed_delta for delta in nontarget_deltas)
    )
    csv_path = tables / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "target", "score_net_v", "signed_delta_v"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "simulator": version,
        "architecture": "multiclass_row_pulsed_split_rail_block_smoke",
        "class_count": args.class_count,
        "target_class": args.target_class,
        "passed": passed,
        "pre_margin_v": measures["pre_margin"],
        "act_after_v": measures["act_after"],
        "actrow_after_v": measures["actrow_after"],
        "target_score_net_v": target_score,
        "best_nontarget_score_net_v": best_nontarget_score,
        "score_winner_margin_v": score_winner_margin,
        "target_signed_delta_v": target_delta,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="multiclass_block_smoke")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--class-count", type=int, default=3)
    ap.add_argument("--target-class", type=int, default=0)
    ap.add_argument("--min-pre-margin", type=float, default=20e-3)
    ap.add_argument("--min-activation", type=float, default=20e-3)
    ap.add_argument("--min-score-margin", type=float, default=2e-3)
    ap.add_argument("--min-signed-delta", type=float, default=1e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.class_count < 2:
        raise ValueError("class-count must be at least 2")
    if args.target_class < 0 or args.target_class >= args.class_count:
        raise ValueError("target-class must be a valid class index")
    for name in ["min_pre_margin", "min_activation", "min_score_margin", "min_signed_delta"]:
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")


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
