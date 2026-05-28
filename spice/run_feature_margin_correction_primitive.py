from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_multiclass_output_head_primitive import signed_store_lines
from run_multiclass_output_head_sequence import class_local_live_label_descent_update_lines, class_node
from run_score_decision_primitive import low_gain_preamp_lines, low_gain_ref_state_lines
from run_spice_sweep import ROOT, detect_spice


CASES = (
    "wrong_feature_active",
    "target_feature_active",
    "inactive_feature",
    "rotated_wrong_feature_active",
)


def case_config(case: str) -> dict[str, float | int]:
    if case == "wrong_feature_active":
        return {
            "target_class": 1,
            "wrong_class": 0,
            "act_v": 0.85,
            "target_wp": 0.38,
            "target_wn": 0.32,
            "wrong_wp": 0.62,
            "wrong_wn": 0.32,
        }
    if case == "target_feature_active":
        return {
            "target_class": 1,
            "wrong_class": 0,
            "act_v": 0.85,
            "target_wp": 0.62,
            "target_wn": 0.32,
            "wrong_wp": 0.38,
            "wrong_wn": 0.32,
        }
    if case == "inactive_feature":
        return {
            "target_class": 1,
            "wrong_class": 0,
            "act_v": 0.0,
            "target_wp": 0.38,
            "target_wn": 0.32,
            "wrong_wp": 0.62,
            "wrong_wn": 0.32,
        }
    if case == "rotated_wrong_feature_active":
        return {
            "target_class": 2,
            "wrong_class": 1,
            "act_v": 0.85,
            "target_wp": 0.38,
            "target_wn": 0.32,
            "wrong_wp": 0.62,
            "wrong_wn": 0.32,
        }
    raise ValueError(f"case must be one of {CASES}")


def _feature_contribution_decision_lines(
    *,
    prefix: str,
    wrong_score: str,
    target_score: str,
    wrong_gt_target: str,
    target_ge_wrong: str,
    reset_node: str,
    amp_clock_node: str,
    decision_clock_node: str,
    pullup_width_u: float = 16.0,
    pulldown_width_u: float = 96.0,
) -> list[str]:
    dec_src = f"{prefix}dec_src"
    return [
        f"C{wrong_gt_target} {wrong_gt_target} 0 4f IC=0",
        f"C{target_ge_wrong} {target_ge_wrong} 0 4f IC=0",
        f"R{wrong_gt_target} {wrong_gt_target} 0 1G",
        f"R{target_ge_wrong} {target_ge_wrong} 0 1G",
        f"Mprecharge_{wrong_gt_target} {wrong_gt_target} {reset_node} vdd vdd PMOS W=4u L=180n",
        f"Mprecharge_{target_ge_wrong} {target_ge_wrong} {reset_node} vdd vdd PMOS W=4u L=180n",
        *low_gain_ref_state_lines(prefix=prefix, reset_node=reset_node, gain_capacitance_f=8.0),
        *low_gain_preamp_lines(
            prefix=prefix,
            score_node=wrong_score,
            scoren_node=target_score,
            amp_clock_node=amp_clock_node,
            gain_input_width=1.0,
            gain_tail_width=8.0,
        ),
        f"M{prefix}dec_pair_p {wrong_gt_target} {target_ge_wrong} vdd vdd PMOS W={pullup_width_u:.6g}u L=180n",
        f"M{prefix}decn_pair_p {target_ge_wrong} {wrong_gt_target} vdd vdd PMOS W={pullup_width_u:.6g}u L=180n",
        f"M{prefix}dec_pair_n {wrong_gt_target} {prefix}scoren_amp {dec_src} 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
        f"M{prefix}decn_pair_n {target_ge_wrong} {prefix}score_amp {dec_src} 0 NSENSE W={pulldown_width_u:.6g}u L=180n",
        f"M{prefix}dec_pair_tail {dec_src} {decision_clock_node} 0 0 NMOS W={pulldown_width_u:.6g}u L=180n",
    ]


def generate_netlist(
    *,
    case: str,
    feature_idx: int = 0,
    readout_width_u: float = 64.0,
    writer_width_u: float = 0.75,
) -> str:
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    if feature_idx < 0:
        raise ValueError("feature_idx must be nonnegative")
    if min(readout_width_u, writer_width_u) <= 0.0:
        raise ValueError("device widths must be positive")
    cfg = case_config(case)
    target_class = int(cfg["target_class"])
    wrong_class = int(cfg["wrong_class"])
    target_wp = float(cfg["target_wp"])
    target_wn = float(cfg["target_wn"])
    wrong_wp = float(cfg["wrong_wp"])
    wrong_wn = float(cfg["wrong_wn"])
    target_vwp = class_node(target_class, f"vwp{feature_idx}")
    target_vwn = class_node(target_class, f"vwn{feature_idx}")
    wrong_vwp = class_node(wrong_class, f"vwp{feature_idx}")
    wrong_vwn = class_node(wrong_class, f"vwn{feature_idx}")
    wrong_score = f"f{feature_idx}_wrong_contrib"
    target_score = f"f{feature_idx}_target_contrib"
    wrong_gt_target = f"f{feature_idx}_wrong_gt_target"
    target_ge_wrong = f"f{feature_idx}_target_ge_wrong"
    writer_gate = f"f{feature_idx}_correction_gate"
    lines = [
        "* Feature-specific live margin-correction primitive.",
        "* The feature contributes through target/wrong conductances, a local comparator",
        "* enables the live writer only when the wrong-class contribution is larger.",
        "* No behavioral sources, no per-weight gradient capacitors, no apply phase.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.48",
        "Vvwlo_ref vwlo_ref 0 0.22",
        "Vrst rst 0 PULSE(1.2 0 0.20n 10p 10p 8n 12n)",
        "Vscorepre scorepre 0 PULSE(0 1.2 0.20n 10p 10p 8n 12n)",
        "Vfwd fwd 0 PULSE(0 1.2 1.00n 10p 10p 1.20n 12n)",
        "Vfwdn fwdn 0 PULSE(1.2 0 1.00n 10p 10p 1.20n 12n)",
        "Vscoreamp scoreamp 0 PULSE(0 1.2 2.45n 10p 10p 0.90n 12n)",
        "Vscoredec scoredec 0 PULSE(0 1.2 3.60n 10p 10p 0.80n 12n)",
        "Vdescentp descentp 0 PULSE(0 1.2 4.60n 10p 10p 1.40n 12n)",
        "Vdescentn descentn 0 PULSE(0 1.2 4.60n 10p 10p 1.40n 12n)",
        f"Vactsrc actsrc 0 {float(cfg['act_v']):.12g}",
        "Cactrow actrow 0 1f IC=0",
        "Ractrow actrow 0 1e12",
        "Mactrow_n actrow fwd actsrc 0 NMOS W=16u L=180n",
        "Mactrow_p actrow fwdn actsrc vdd PMOS W=32u L=180n",
        "Mactrow_rst actrow rst 0 0 NMOS W=4u L=180n",
        *signed_store_lines(
            positive_node=target_vwp,
            negative_node=target_vwn,
            positive_ic=target_wp,
            negative_ic=target_wn,
        ),
        *signed_store_lines(
            positive_node=wrong_vwp,
            negative_node=wrong_vwn,
            positive_ic=wrong_wp,
            negative_ic=wrong_wn,
        ),
        f"C{wrong_score} {wrong_score} 0 10f IC=0",
        f"C{target_score} {target_score} 0 10f IC=0",
        f"R{wrong_score} {wrong_score} 0 1G",
        f"R{target_score} {target_score} 0 1G",
        f"M{wrong_score}_rst {wrong_score} rst 0 0 NMOS W=4u L=180n",
        f"M{target_score}_rst {target_score} rst 0 0 NMOS W=4u L=180n",
        f"M{wrong_score}_cond actrow {wrong_vwp} {wrong_score} 0 NMOS W={readout_width_u:.6g}u L=180n",
        f"M{target_score}_cond actrow {target_vwp} {target_score} 0 NMOS W={readout_width_u:.6g}u L=180n",
        *_feature_contribution_decision_lines(
            prefix=f"f{feature_idx}_",
            wrong_score=wrong_score,
            target_score=target_score,
            wrong_gt_target=wrong_gt_target,
            target_ge_wrong=target_ge_wrong,
            reset_node="scorepre",
            amp_clock_node="scoreamp",
            decision_clock_node="scoredec",
        ),
        f"C{writer_gate} {writer_gate} 0 4f IC=0",
        f"R{writer_gate} {writer_gate} 0 1G",
        f"M{writer_gate}_rst {writer_gate} rst 0 0 NMOS W=4u L=180n",
        f"R{writer_gate}_m1 {writer_gate}_m1 0 1G",
        f"R{writer_gate}_m2 {writer_gate}_m2 0 1G",
        f"M{writer_gate}_act vdd actrow {writer_gate}_m1 0 NSENSE W=16u L=180n",
        f"M{writer_gate}_dec1 {writer_gate}_m1 {wrong_gt_target} {writer_gate}_m2 0 NMOS W=16u L=180n",
        f"M{writer_gate}_dec2 {writer_gate}_m2 {wrong_gt_target} {writer_gate} 0 NMOS W=16u L=180n",
        f"M{writer_gate}_target_discharge {writer_gate} {target_ge_wrong} 0 0 NMOS W=64u L=180n",
        *class_local_live_label_descent_update_lines(
            class_idx=target_class,
            feature_idx=feature_idx,
            activation_node=writer_gate,
            positive_descent_node="descentp",
            negative_descent_node="0",
            width_u=writer_width_u,
            high_side_topology="pmos-differential",
        ),
        *class_local_live_label_descent_update_lines(
            class_idx=wrong_class,
            feature_idx=feature_idx,
            activation_node=writer_gate,
            positive_descent_node="0",
            negative_descent_node="descentn",
            width_u=writer_width_u,
            high_side_topology="pmos-differential",
        ),
        f".meas tran wrong_contrib FIND V({wrong_score}) AT=2.35n",
        f".meas tran target_contrib FIND V({target_score}) AT=2.35n",
        ".meas tran contrib_margin PARAM='wrong_contrib-target_contrib'",
        f".meas tran wrong_gt_target_after FIND V({wrong_gt_target}) AT=4.50n",
        f".meas tran target_ge_wrong_after FIND V({target_ge_wrong}) AT=4.50n",
        f".meas tran correction_gate_after FIND V({writer_gate}) AT=4.50n",
        f".meas tran target_wp_final FIND V({target_vwp}) AT=6.50n",
        f".meas tran target_wn_final FIND V({target_vwn}) AT=6.50n",
        f".meas tran wrong_wp_final FIND V({wrong_vwp}) AT=6.50n",
        f".meas tran wrong_wn_final FIND V({wrong_vwn}) AT=6.50n",
        f".meas tran target_signed_final PARAM='target_wp_final-target_wn_final'",
        f".meas tran wrong_signed_final PARAM='wrong_wp_final-wrong_wn_final'",
        f".meas tran target_signed_delta PARAM='target_signed_final-({target_wp:.12g}-{target_wn:.12g})'",
        f".meas tran wrong_signed_delta PARAM='wrong_signed_final-({wrong_wp:.12g}-{wrong_wn:.12g})'",
        ".tran 2p 7n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def classify_case(case: str, measures: dict[str, Any], *, min_delta_v: float = 0.010) -> dict[str, Any]:
    target_delta = float(measures["target_signed_delta"])
    wrong_delta = float(measures["wrong_signed_delta"])
    if case in {"wrong_feature_active", "rotated_wrong_feature_active"}:
        passed = target_delta > min_delta_v and wrong_delta < -min_delta_v
    else:
        passed = abs(target_delta) < min_delta_v and abs(wrong_delta) < min_delta_v
    return {"passed": passed, "target_delta_v": target_delta, "wrong_delta_v": wrong_delta}


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for case in CASES:
        path = generated / f"{tag}_{sanitize_tag(case)}.cir"
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(case=case, writer_width_u=args.writer_width_u),
            timeout=args.timeout,
        )
        row = {"case": case, **measures}
        row.update(classify_case(case, row, min_delta_v=args.min_delta_v))
        rows.append(row)
    csv_path = tables / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "simulator": version,
        "architecture": "feature_margin_correction_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "cases": len(rows),
        "passed": all(bool(row["passed"]) for row in rows),
        "min_delta_v": args.min_delta_v,
        "writer_width_u": args.writer_width_u,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spice-bin", default=None)
    parser.add_argument("--tag", default="feature_margin_correction_primitive")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--writer-width-u", type=float, default=0.75)
    parser.add_argument("--min-delta-v", type=float, default=0.010)
    return parser


def main_for_test(argv: list[str] | None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.writer_width_u <= 0.0:
        raise ValueError("writer-width-u must be positive")
    if args.min_delta_v <= 0.0:
        raise ValueError("min-delta-v must be positive")
    return run_cases(args)


def main() -> None:
    main_for_test(None)


if __name__ == "__main__":
    main()
