from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


SCORE_CASES = (
    "positive",
    "negative",
    "neutral",
    "shifted_positive",
    "shifted_negative",
    "tiny_positive",
    "tiny_negative",
    "tiny_neutral",
)
DECISION_TOPOLOGIES = (
    "score-diff",
    "score-diff-reject-ref",
    "score-diff-window",
    "score-diff-gain-window",
    "score-diff-low-gain",
    "score-diff-low-gain-ref",
)


def prefixed(prefix: str, base: str) -> str:
    return f"{prefix}{base}" if prefix else base


def low_gain_ref_state_lines(
    *,
    prefix: str = "",
    gain_capacitance_f: float = 8.0,
    score_amp_ic: float = 1.2,
    reset_node: str = "rstfn",
) -> list[str]:
    score_amp = prefixed(prefix, "score_amp")
    scoren_amp = prefixed(prefix, "scoren_amp")
    return [
        f"C{score_amp} {score_amp} 0 {gain_capacitance_f:.12g}f IC={score_amp_ic:.12g}",
        f"C{scoren_amp} {scoren_amp} 0 {gain_capacitance_f:.12g}f IC={score_amp_ic:.12g}",
        f"R{score_amp} {score_amp} 0 1G",
        f"R{scoren_amp} {scoren_amp} 0 1G",
        f"Mprecharge_{score_amp} {score_amp} {reset_node} vdd vdd PMOS W=4u L=180n",
        f"Mprecharge_{scoren_amp} {scoren_amp} {reset_node} vdd vdd PMOS W=4u L=180n",
    ]


def low_gain_preamp_lines(
    *,
    prefix: str = "",
    score_node: str = "score",
    scoren_node: str = "scoren",
    amp_clock_node: str = "amp",
    gain_input_width: float = 1.0,
    gain_tail_width: float = 8.0,
) -> list[str]:
    score_amp = prefixed(prefix, "score_amp")
    scoren_amp = prefixed(prefix, "scoren_amp")
    score_amp_i = prefixed(prefix, "scoreamp_score_i")
    scoren_amp_i = prefixed(prefix, "scoreamp_scoren_i")
    return [
        f"M{prefix}scoreamp_score_p {score_amp} {score_node} {score_amp_i} vdd PMOS W={gain_input_width:.6g}u L=180n",
        f"M{prefix}scoreamp_score_tail {score_amp_i} {amp_clock_node} 0 0 NMOS W={gain_tail_width:.6g}u L=180n",
        f"M{prefix}scoreamp_scoren_p {scoren_amp} {scoren_node} {scoren_amp_i} vdd PMOS W={gain_input_width:.6g}u L=180n",
        f"M{prefix}scoreamp_scoren_tail {scoren_amp_i} {amp_clock_node} 0 0 NMOS W={gain_tail_width:.6g}u L=180n",
    ]


def low_gain_ref_decision_lines(
    *,
    prefix: str = "",
    score_node: str = "score",
    scoren_node: str = "scoren",
    outref_node: str = "outref",
    decision_node: str | None = None,
    decisionn_node: str | None = None,
    amp_clock_node: str = "amp",
    decision_clock_node: str = "dec2",
    pullup_width: float = 8.0,
    pulldown_width: float = 12.0,
    gain_input_width: float = 1.0,
    gain_tail_width: float = 8.0,
) -> list[str]:
    score_amp = prefixed(prefix, "score_amp")
    scoren_amp = prefixed(prefix, "scoren_amp")
    decision = prefixed(prefix, "decision") if decision_node is None else decision_node
    decisionn = prefixed(prefix, "decisionn") if decisionn_node is None else decisionn_node
    dec_src = prefixed(prefix, "dec_src")
    return [
        "* Low-common-mode PMOS-input preamp followed by a referenced binary latch.",
        *low_gain_preamp_lines(
            prefix=prefix,
            score_node=score_node,
            scoren_node=scoren_node,
            amp_clock_node=amp_clock_node,
            gain_input_width=gain_input_width,
            gain_tail_width=gain_tail_width,
        ),
        "* Positive class wins only when score_amp beats scoren_amp plus the physical outref current.",
        f"M{prefix}dec_low_gain_ref_p {decision} {decisionn} vdd vdd PMOS W={pullup_width:.6g}u L=180n",
        f"M{prefix}decn_low_gain_ref_p {decisionn} {decision} vdd vdd PMOS W={pullup_width:.6g}u L=180n",
        f"M{prefix}dec_low_gain_ref_scorenamp {decision} {scoren_amp} {dec_src} 0 NSENSE W={pulldown_width:.6g}u L=180n",
        f"M{prefix}dec_low_gain_ref_ref {decision} {outref_node} {dec_src} 0 NSENSE W={pulldown_width:.6g}u L=180n",
        f"M{prefix}decn_low_gain_ref_scoreamp {decisionn} {score_amp} {dec_src} 0 NSENSE W={pulldown_width:.6g}u L=180n",
        f"M{prefix}dec_low_gain_ref_tail {dec_src} {decision_clock_node} 0 0 NMOS W={pulldown_width:.6g}u L=180n",
    ]


def score_values(case: str, *, center: float, delta: float) -> tuple[float, float]:
    if case == "positive":
        return center + delta / 2.0, center - delta / 2.0
    if case == "negative":
        return center - delta / 2.0, center + delta / 2.0
    if case == "neutral":
        return center, center
    if case == "shifted_positive":
        return center - delta / 2.0, center + delta / 2.0
    if case == "shifted_negative":
        return center - delta, center + delta
    if case == "tiny_positive":
        return center + 0.0011, center - 0.0011
    if case == "tiny_negative":
        return center - 0.0011, center + 0.0011
    if case == "tiny_neutral":
        return center + 0.000045, center - 0.000045
    raise ValueError(f"score case must be one of {SCORE_CASES}")


def generate_netlist(
    *,
    score_case: str,
    score_center: float = 0.10,
    score_delta: float = 0.04,
    score: float | None = None,
    scoren: float | None = None,
    pullup_width: float = 8.0,
    pulldown_width: float = 12.0,
    scoren_pulldown_scale: float = 1.0,
    decision_topology: str = "score-diff",
    reject_ref: float = 0.075,
    gain_input_width: float = 1.0,
    gain_tail_width: float = 8.0,
    gain_capacitance_f: float = 8.0,
) -> str:
    if score_case not in SCORE_CASES:
        raise ValueError(f"score_case must be one of {SCORE_CASES}")
    if decision_topology not in DECISION_TOPOLOGIES:
        raise ValueError(f"decision_topology must be one of {DECISION_TOPOLOGIES}")
    for name, value in {
        "pullup_width": pullup_width,
        "pulldown_width": pulldown_width,
        "scoren_pulldown_scale": scoren_pulldown_scale,
        "gain_input_width": gain_input_width,
        "gain_tail_width": gain_tail_width,
        "gain_capacitance_f": gain_capacitance_f,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if reject_ref < 0.0 or reject_ref > 1.2:
        raise ValueError("reject_ref must stay within supply rails")
    case_score, case_scoren = score_values(score_case, center=score_center, delta=score_delta)
    score_v = case_score if score is None else score
    scoren_v = case_scoren if scoren is None else scoren
    if min(score_v, scoren_v) < 0.0 or max(score_v, scoren_v) > 1.2:
        raise ValueError("score rails must stay within supply rails")
    measure_time = (
        "5.80n"
        if decision_topology
        in {
            "score-diff-reject-ref",
            "score-diff-gain-window",
            "score-diff-low-gain",
            "score-diff-low-gain-ref",
        }
        else "4.5n"
    )
    lines = [
        "* Score differential precharged decision primitive smoke.",
        "* Tests direct transistor sensing of score/scoren without output-latch indirection.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vscore score 0 {score_v:.12g}",
        f"Vscoren scoren 0 {scoren_v:.12g}",
        f"Voutref outref 0 {reject_ref:.12g}",
        "Vrstfn rstfn 0 PULSE(0 1.2 0.8n 10p 10p 8n 10n)",
        "Vdec dec 0 PULSE(0 1.2 1.0n 10p 10p 3n 10n)",
        "Vamp amp 0 PULSE(0 1.2 1.0n 10p 10p 2.0n 10n)",
        "Vdec2 dec2 0 PULSE(0 1.2 4.6n 10p 10p 0.7n 10n)",
        "Cdecision decision 0 20f IC=0",
        "Cdecisionn decisionn 0 20f IC=0",
        "Rdecision decision 0 1G",
        "Rdecisionn decisionn 0 1G",
        "Mprecharge_decision decision rstfn vdd vdd PMOS W=4u L=180n",
        "Mprecharge_decisionn decisionn rstfn vdd vdd PMOS W=4u L=180n",
    ]
    if decision_topology == "score-diff":
        lines += [
            f"Mdec_scorepc_p decision decisionn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdecn_scorepc_p decisionn decision vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdec_scorepc_n decision scoren dec_src 0 NSENSE W={pulldown_width * scoren_pulldown_scale:.6g}u L=180n",
            f"Mdecn_scorepc_n decisionn score dec_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_scorepc_tail dec_src dec 0 0 NMOS W={pulldown_width:.6g}u L=180n",
        ]
    elif decision_topology == "score-diff-reject-ref":
        lines += [
            "Cdecision_pre decision_pre 0 20f IC=0",
            "Cdecisionn_pre decisionn_pre 0 20f IC=0",
            "Rdecision_pre decision_pre 0 1G",
            "Rdecisionn_pre decisionn_pre 0 1G",
            "Mprecharge_decision_pre decision_pre rstfn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_decisionn_pre decisionn_pre rstfn vdd vdd PMOS W=4u L=180n",
            f"Mdec_scorepre_p decision_pre decisionn_pre vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdecn_scorepre_p decisionn_pre decision_pre vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdec_scorepre_n decision_pre scoren dec_src 0 NSENSE W={pulldown_width * scoren_pulldown_scale:.6g}u L=180n",
            f"Mdecn_scorepre_n decisionn_pre score dec_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_scorepre_tail dec_src dec 0 0 NMOS W={pulldown_width:.6g}u L=180n",
            f"Mdec_reject_p decision decisionn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdecn_reject_p decisionn decision vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdec_reject_n decision decisionn_pre dec2_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdecn_reject_n decisionn outref dec2_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_reject_tail dec2_src dec2 0 0 NMOS W={pulldown_width:.6g}u L=180n",
            ".meas tran decision_pre_after FIND V(decision_pre) AT=4.5n",
            ".meas tran decisionn_pre_after FIND V(decisionn_pre) AT=4.5n",
            ".meas tran decision_pre_diff PARAM='decision_pre_after-decisionn_pre_after'",
        ]
    elif decision_topology == "score-diff-window":
        lines += [
            "Cdecision_posn decision_posn 0 20f IC=0",
            "Cdecision_negn decision_negn 0 20f IC=0",
            "Rdecision_posn decision_posn 0 1G",
            "Rdecision_negn decision_negn 0 1G",
            "Mprecharge_decision_posn decision_posn rstfn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_decision_negn decision_negn rstfn vdd vdd PMOS W=4u L=180n",
            "* Positive window comparator: score must beat scoren plus outref current.",
            f"Mdec_win_pos_p decision decision_posn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdecn_win_pos_p decision_posn decision vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdec_win_pos_scoren decision scoren pos_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_win_pos_ref decision outref pos_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdecn_win_pos_score decision_posn score pos_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_win_pos_tail pos_src dec 0 0 NMOS W={pulldown_width:.6g}u L=180n",
            "* Negative window comparator: scoren must beat score plus outref current.",
            f"Mdec_win_neg_p decisionn decision_negn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdecn_win_neg_p decision_negn decisionn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdec_win_neg_score decisionn score neg_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_win_neg_ref decisionn outref neg_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdecn_win_neg_scoren decision_negn scoren neg_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_win_neg_tail neg_src dec 0 0 NMOS W={pulldown_width:.6g}u L=180n",
            ".meas tran decision_posn_after FIND V(decision_posn) AT=4.5n",
            ".meas tran decision_negn_after FIND V(decision_negn) AT=4.5n",
            ".meas tran positive_window_diff PARAM='decision_after-decision_posn_after'",
            ".meas tran negative_window_diff PARAM='decisionn_after-decision_negn_after'",
        ]
    elif decision_topology == "score-diff-gain-window":
        lines += [
            f"Cscore_amp score_amp 0 {gain_capacitance_f:.12g}f IC=1.2",
            f"Cscoren_amp scoren_amp 0 {gain_capacitance_f:.12g}f IC=1.2",
            "Rscore_amp score_amp 0 1G",
            "Rscoren_amp scoren_amp 0 1G",
            "Mprecharge_score_amp score_amp rstfn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_scoren_amp scoren_amp rstfn vdd vdd PMOS W=4u L=180n",
            "* Dynamic current-integrating preamp. Higher score pulls score_amp lower;",
            "* the window comparator consumes the swapped rails so score_g=scoren_amp.",
            f"Mscoreamp_score score_amp score scoreamp_score_i 0 NSENSE W={gain_input_width:.6g}u L=180n",
            f"Mscoreamp_score_tail scoreamp_score_i amp 0 0 NMOS W={gain_tail_width:.6g}u L=180n",
            f"Mscoreamp_scoren scoren_amp scoren scoreamp_scoren_i 0 NSENSE W={gain_input_width:.6g}u L=180n",
            f"Mscoreamp_scoren_tail scoreamp_scoren_i amp 0 0 NMOS W={gain_tail_width:.6g}u L=180n",
            "Cdecision_posn decision_posn 0 20f IC=0",
            "Cdecision_negn decision_negn 0 20f IC=0",
            "Rdecision_posn decision_posn 0 1G",
            "Rdecision_negn decision_negn 0 1G",
            "Mprecharge_decision_posn decision_posn rstfn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_decision_negn decision_negn rstfn vdd vdd PMOS W=4u L=180n",
            "* Positive window comparator after gain: scoren_amp must beat score_amp plus outref current.",
            f"Mdec_gain_win_pos_p decision decision_posn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdecn_gain_win_pos_p decision_posn decision vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdec_gain_win_pos_scoreamp decision score_amp pos_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_gain_win_pos_ref decision outref pos_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdecn_gain_win_pos_scorenamp decision_posn scoren_amp pos_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_gain_win_pos_tail pos_src dec2 0 0 NMOS W={pulldown_width:.6g}u L=180n",
            "* Negative window comparator after gain: score_amp must beat scoren_amp plus outref current.",
            f"Mdec_gain_win_neg_p decisionn decision_negn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdecn_gain_win_neg_p decision_negn decisionn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdec_gain_win_neg_scorenamp decisionn scoren_amp neg_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_gain_win_neg_ref decisionn outref neg_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdecn_gain_win_neg_scoreamp decision_negn score_amp neg_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_gain_win_neg_tail neg_src dec2 0 0 NMOS W={pulldown_width:.6g}u L=180n",
            ".meas tran score_amp_after FIND V(score_amp) AT=4.5n",
            ".meas tran scoren_amp_after FIND V(scoren_amp) AT=4.5n",
            ".meas tran score_gain_diff PARAM='scoren_amp_after-score_amp_after'",
            ".meas tran decision_posn_after FIND V(decision_posn) AT=5.80n",
            ".meas tran decision_negn_after FIND V(decision_negn) AT=5.80n",
            ".meas tran positive_window_diff PARAM='decision_after-decision_posn_after'",
            ".meas tran negative_window_diff PARAM='decisionn_after-decision_negn_after'",
        ]
    elif decision_topology == "score-diff-low-gain":
        lines += [
            f"Cscore_amp score_amp 0 {gain_capacitance_f:.12g}f IC=1.2",
            f"Cscoren_amp scoren_amp 0 {gain_capacitance_f:.12g}f IC=1.2",
            "Rscore_amp score_amp 0 1G",
            "Rscoren_amp scoren_amp 0 1G",
            "Mprecharge_score_amp score_amp rstfn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_scoren_amp scoren_amp rstfn vdd vdd PMOS W=4u L=180n",
            "* Low-common-mode PMOS-input preamp followed by a regenerative differential latch.",
            f"Mscoreamp_score_p score_amp score scoreamp_score_i vdd PMOS W={gain_input_width:.6g}u L=180n",
            f"Mscoreamp_score_tail scoreamp_score_i amp 0 0 NMOS W={gain_tail_width:.6g}u L=180n",
            f"Mscoreamp_scoren_p scoren_amp scoren scoreamp_scoren_i vdd PMOS W={gain_input_width:.6g}u L=180n",
            f"Mscoreamp_scoren_tail scoreamp_scoren_i amp 0 0 NMOS W={gain_tail_width:.6g}u L=180n",
            f"Mdec_low_gain_p decision decisionn vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdecn_low_gain_p decisionn decision vdd vdd PMOS W={pullup_width:.6g}u L=180n",
            f"Mdec_low_gain_n decision scoren_amp dec_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdecn_low_gain_n decisionn score_amp dec_src 0 NSENSE W={pulldown_width:.6g}u L=180n",
            f"Mdec_low_gain_tail dec_src dec2 0 0 NMOS W={pulldown_width:.6g}u L=180n",
            ".meas tran score_amp_after FIND V(score_amp) AT=4.5n",
            ".meas tran scoren_amp_after FIND V(scoren_amp) AT=4.5n",
            ".meas tran score_gain_diff PARAM='score_amp_after-scoren_amp_after'",
        ]
    elif decision_topology == "score-diff-low-gain-ref":
        lines += [
            *low_gain_ref_state_lines(gain_capacitance_f=gain_capacitance_f),
            *low_gain_ref_decision_lines(
                pullup_width=pullup_width,
                pulldown_width=pulldown_width,
                gain_input_width=gain_input_width,
                gain_tail_width=gain_tail_width,
            ),
            ".meas tran score_amp_after FIND V(score_amp) AT=4.5n",
            ".meas tran scoren_amp_after FIND V(scoren_amp) AT=4.5n",
            ".meas tran score_gain_diff PARAM='score_amp_after-scoren_amp_after'",
        ]
    lines += [
        f".meas tran decision_after FIND V(decision) AT={measure_time}",
        f".meas tran decisionn_after FIND V(decisionn) AT={measure_time}",
        ".meas tran decision_diff PARAM='decision_after-decisionn_after'",
        ".tran 2p 6.2n uic",
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
    if case in {"neutral", "tiny_neutral"}:
        margin = abs(float(row.get("decision_diff", 0.0)))
        return {"decision_classification": "resolved" if margin >= min_abs_margin else "dead_zone"}
    expected = 1.0 if case in {"positive", "shifted_positive", "tiny_positive"} else -1.0
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
                scoren_pulldown_scale=args.scoren_pulldown_scale,
                decision_topology=args.decision_topology,
                reject_ref=args.reject_ref,
                gain_input_width=args.gain_input_width,
                gain_tail_width=args.gain_tail_width,
                gain_capacitance_f=args.gain_capacitance_f,
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
    ap.add_argument("--scoren-pulldown-scale", type=float, default=1.0)
    ap.add_argument("--decision-topology", choices=DECISION_TOPOLOGIES, default="score-diff")
    ap.add_argument("--reject-ref", type=float, default=0.075)
    ap.add_argument("--gain-input-width", type=float, default=1.0)
    ap.add_argument("--gain-tail-width", type=float, default=8.0)
    ap.add_argument("--gain-capacitance-f", type=float, default=8.0)
    ap.add_argument("--min-abs-margin", type=float, default=50e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    for name in [
        "pullup_width",
        "pulldown_width",
        "scoren_pulldown_scale",
        "gain_input_width",
        "gain_tail_width",
        "gain_capacitance_f",
    ]:
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.reject_ref < 0.0 or args.reject_ref > 1.2:
        raise ValueError("reject-ref must stay within supply rails")
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
