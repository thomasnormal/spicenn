from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


APPROACHES = (
    "current-sum",
    "common-mode",
    "pairwise-margin",
    "fixed-total-feedback",
    "soft-wta",
    "charge-share",
    "time-domain",
    "pulse-density",
    "log-domain",
    "learned-calibration",
)

CASES = (
    "target1_clear",
    "target1_low_wrong0",
    "target1_loses2",
    "flat_target1",
)


def class_scores(case: str) -> tuple[tuple[float, float, float], int]:
    if case == "target1_clear":
        return (0.0020, 0.0080, 0.0035), 1
    if case == "target1_low_wrong0":
        return (0.0075, 0.0015, 0.0045), 1
    if case == "target1_loses2":
        return (0.0025, 0.0045, 0.0075), 1
    if case == "flat_target1":
        return (0.0045, 0.0045, 0.0045), 1
    raise ValueError(f"case must be one of {CASES}")


def _err_nodes() -> str:
    return "e0p e0n e1p e1n e2p e2n"


def _subckt_header(name: str) -> list[str]:
    return [
        f".subckt norm_{name} s0 s1 s2 tp0 tp1 tp2 tn0 tn1 tn2 phi rst {_err_nodes()} vdd vss",
        f"* {name} class-evidence normalization candidate.",
    ]


def _error_caps() -> list[str]:
    lines = []
    for idx in range(3):
        for suffix in ("p", "n"):
            node = f"e{idx}{suffix}"
            lines += [
                f"C{node} {node} vss 0.5f IC=0",
                f"R{node} {node} vss 1G",
                f"Mreset_{node} {node} rst vss vss NMOS W=4u L=180n",
            ]
    return lines


def _dense_error_core(
    *,
    score_nodes: tuple[str, str, str],
    mass_node: str = "mass",
    mass_cap_f: float = 0.5,
    sum_width_u: float = 96.0,
    error_width_u: float = 96.0,
    target_drive_node: str | None = None,
) -> list[str]:
    drive = mass_node if target_drive_node is None else target_drive_node
    lines = [
        f"C{mass_node} {mass_node} vss {mass_cap_f:.12g}f IC=0",
        f"R{mass_node} {mass_node} vss 1G",
        f"Mreset_{mass_node} {mass_node} rst vss vss NMOS W=4u L=180n",
    ]
    for idx, score_node in enumerate(score_nodes):
        lines += [
            f"Rmass{idx}_a mass{idx}_a vss 1G",
            f"Rmass{idx}_s mass{idx}_s vss 1G",
            f"Mmass{idx}_label vdd tn{idx} mass{idx}_a vss NSENSE W={sum_width_u:.6g}u L=180n",
            f"Mmass{idx}_score mass{idx}_a {score_node} mass{idx}_s vss NSENSE W={sum_width_u:.6g}u L=180n",
            f"Mmass{idx}_clk mass{idx}_s phi {mass_node} vss NSENSE W={sum_width_u:.6g}u L=180n",
            f"Rep{idx}_a ep{idx}_a vss 1G",
            f"Rep{idx}_m ep{idx}_m vss 1G",
            f"Mep{idx}_label vdd tp{idx} ep{idx}_a vss NSENSE W={error_width_u:.6g}u L=180n",
            f"Mep{idx}_drive ep{idx}_a {drive} ep{idx}_m vss NSENSE W={error_width_u:.6g}u L=180n",
            f"Mep{idx}_clk ep{idx}_m phi e{idx}p vss NSENSE W={error_width_u:.6g}u L=180n",
            f"Ren{idx}_a en{idx}_a vss 1G",
            f"Ren{idx}_s en{idx}_s vss 1G",
            f"Men{idx}_label vdd tn{idx} en{idx}_a vss NSENSE W={error_width_u:.6g}u L=180n",
            f"Men{idx}_score en{idx}_a {score_node} en{idx}_s vss NSENSE W={error_width_u:.6g}u L=180n",
            f"Men{idx}_clk en{idx}_s phi e{idx}n vss NSENSE W={error_width_u:.6g}u L=180n",
        ]
    return lines


def _score_buffer_lines(prefix: str, source: str, *, initial_v: float = 0.0, width_u: float = 64.0) -> list[str]:
    return [
        f"C{prefix} {prefix} vss 1f IC={initial_v:.12g}",
        f"R{prefix} {prefix} vss 1G",
        f"Mreset_{prefix} {prefix} rst vss vss NMOS W=4u L=180n",
        f"R{prefix}_a {prefix}_a vss 1G",
        f"M{prefix}_v vdd {source} {prefix}_a vss NSENSE W={width_u:.6g}u L=180n",
        f"M{prefix}_clk {prefix}_a phi {prefix} vss NSENSE W={width_u:.6g}u L=180n",
    ]


def _common_average_lines(node: str, source_nodes: tuple[str, str, str], *, resistance_ohm: float = 1.0e7) -> list[str]:
    return [
        f"C{node} {node} vss 4f IC=0",
        f"R{node}_leak {node} vss 1G",
        *[f"R{node}_{idx} {node} {source_nodes[idx]} {resistance_ohm:.12g}" for idx in range(3)],
    ]


def _low_gain_score_lines(
    *,
    output_nodes: tuple[str, str, str] = ("sg0", "sg1", "sg2"),
    input_nodes: tuple[str, str, str] = ("s0", "s1", "s2"),
    cap_f: float = 8.0,
    input_width_u: float = 1.0,
    tail_width_u: float = 8.0,
) -> list[str]:
    lines: list[str] = []
    for idx, output in enumerate(output_nodes):
        raw = input_nodes[idx]
        lines += [
            f"C{output} {output} vss {cap_f:.12g}f IC=1.2",
            f"R{output} {output} vss 1G",
            f"Mprecharge_{output} {output} rst vdd vdd PMOS W=4u L=180n",
            f"R{output}_i {output}_i vss 1G",
            f"M{output}_amp_p {output} {raw} {output}_i vdd PMOS W={input_width_u:.6g}u L=180n",
            f"M{output}_amp_tail {output}_i vdd vss vss NMOS W={tail_width_u:.6g}u L=180n",
        ]
    return lines


def _subckt_current_sum() -> str:
    lines = _subckt_header("current_sum") + _error_caps()
    lines += _low_gain_score_lines()
    lines += _dense_error_core(score_nodes=("sg0", "sg1", "sg2"), sum_width_u=32.0, error_width_u=32.0)
    lines.append(".ends norm_current_sum")
    return "\n".join(lines)


def _subckt_common_mode() -> str:
    lines = _subckt_header("common_mode") + _error_caps()
    lines += _low_gain_score_lines()
    lines += _common_average_lines("avg", ("sg0", "sg1", "sg2"))
    for idx in range(3):
        lines += [
            f"Cevid{idx} evid{idx} vss 1f IC=0",
            f"Revid{idx} evid{idx} vss 1G",
            f"Mreset_evid{idx} evid{idx} rst vss vss NMOS W=4u L=180n",
            f"Revid{idx}_up evid{idx}_up vss 1G",
            f"Revid{idx}_dn evid{idx}_dn vss 1G",
            f"Mevid{idx}_up_v vdd sg{idx} evid{idx}_up vss NREL W=192u L=180n",
            f"Mevid{idx}_up_t evid{idx}_up phi evid{idx} vss NSENSE W=192u L=180n",
            f"Mevid{idx}_dn_v evid{idx} avg evid{idx}_dn vss NREL W=48u L=180n",
            f"Mevid{idx}_dn_t evid{idx}_dn phi vss vss NSENSE W=48u L=180n",
        ]
    lines += _dense_error_core(score_nodes=("evid0", "evid1", "evid2"), sum_width_u=48.0, error_width_u=48.0)
    lines.append(".ends norm_common_mode")
    return "\n".join(lines)


def _subckt_pairwise_margin() -> str:
    lines = _subckt_header("pairwise_margin") + _error_caps()
    lines += _low_gain_score_lines()
    for idx in range(3):
        lines += _score_buffer_lines(f"evid{idx}", f"sg{idx}", width_u=64.0)
    lines += _dense_error_core(score_nodes=("evid0", "evid1", "evid2"), sum_width_u=80.0, error_width_u=128.0)
    lines += [
        "* Weak target-margin branches suppress target-wins evidence near ties.",
        "Rmargin0 margin0 vss 1G",
        "Rmargin1 margin1 vss 1G",
        "Rmargin2 margin2 vss 1G",
        "Mmargin0 evid0 tp0 margin0 vss NSENSE W=0.25u L=180n",
        "Mmargin1 evid1 tp1 margin1 vss NSENSE W=0.25u L=180n",
        "Mmargin2 evid2 tp2 margin2 vss NSENSE W=0.25u L=180n",
        "Mmargin0_clk margin0 phi vss vss NMOS W=0.25u L=180n",
        "Mmargin1_clk margin1 phi vss vss NMOS W=0.25u L=180n",
        "Mmargin2_clk margin2 phi vss vss NMOS W=0.25u L=180n",
    ]
    lines.append(".ends norm_pairwise_margin")
    return "\n".join(lines)


def _subckt_fixed_total_feedback() -> str:
    lines = _subckt_header("fixed_total_feedback") + _error_caps()
    lines += _low_gain_score_lines()
    for idx in range(3):
        lines += _score_buffer_lines(f"evid{idx}", f"sg{idx}", width_u=64.0)
    lines += [
        "Ctotal total vss 0.5f IC=0",
        "Rtotal total vss 1G",
        "Mreset_total total rst vss vss NMOS W=4u L=180n",
    ]
    for idx in range(3):
        lines += [
            f"Rtotal{idx}_a total{idx}_a vss 1G",
            f"Mtotal{idx}_v vdd evid{idx} total{idx}_a vss NSENSE W=48u L=180n",
            f"Mtotal{idx}_clk total{idx}_a phi total vss NSENSE W=48u L=180n",
        ]
    lines += _dense_error_core(
        score_nodes=("evid0", "evid1", "evid2"),
        mass_node="mass",
        target_drive_node="total",
        sum_width_u=64.0,
        error_width_u=96.0,
    )
    lines.append(".ends norm_fixed_total_feedback")
    return "\n".join(lines)


def _subckt_soft_wta() -> str:
    lines = _subckt_header("soft_wta") + _error_caps()
    lines += _low_gain_score_lines()
    lines += _common_average_lines("inhibit", ("sg0", "sg1", "sg2"), resistance_ohm=5.0e6)
    for idx in range(3):
        lines += [
            f"Cevid{idx} evid{idx} vss 1f IC=0",
            f"Revid{idx} evid{idx} vss 1G",
            f"Mreset_evid{idx} evid{idx} rst vss vss NMOS W=4u L=180n",
            f"Mevid{idx}_self vdd sg{idx} evid{idx}_a vss NREL W=192u L=180n",
            f"Revid{idx}_a evid{idx}_a vss 1G",
            f"Mevid{idx}_inh evid{idx}_a inhibit evid{idx}_b vss NREL W=32u L=180n",
            f"Revid{idx}_b evid{idx}_b vss 1G",
            f"Mevid{idx}_clk evid{idx}_b phi evid{idx} vss NSENSE W=192u L=180n",
        ]
    lines += _dense_error_core(score_nodes=("evid0", "evid1", "evid2"), sum_width_u=128.0, error_width_u=128.0)
    lines.append(".ends norm_soft_wta")
    return "\n".join(lines)


def _subckt_charge_share() -> str:
    lines = _subckt_header("charge_share") + _error_caps()
    lines += _low_gain_score_lines()
    lines += [
        "Cpool pool vss 2f IC=0",
        "Rpool pool vss 1G",
        "Mreset_pool pool rst vss vss NMOS W=4u L=180n",
    ]
    for idx in range(3):
        lines += [
            f"Cshare{idx} share{idx} vss 1f IC=0",
            f"Rshare{idx} share{idx} vss 1G",
            f"Mreset_share{idx} share{idx} rst vss vss NMOS W=4u L=180n",
            f"Mshare{idx}_pre vdd sg{idx} share{idx} vss NSENSE W=96u L=180n",
            f"Mshare{idx}_pool share{idx} phi pool vss NSENSE W=32u L=180n",
        ]
    lines += _dense_error_core(
        score_nodes=("share0", "share1", "share2"),
        mass_node="mass",
        target_drive_node="pool",
        sum_width_u=96.0,
        error_width_u=96.0,
    )
    lines.append(".ends norm_charge_share")
    return "\n".join(lines)


def _subckt_time_domain() -> str:
    lines = _subckt_header("time_domain") + _error_caps()
    lines += _low_gain_score_lines()
    for idx in range(3):
        lines += [
            f"Cramp{idx} ramp{idx} vss 2f IC=1.2",
            f"Rramp{idx} ramp{idx} vss 1G",
            f"Mprecharge_ramp{idx} ramp{idx} rst vdd vdd PMOS W=4u L=180n",
            f"Mramp{idx}_score ramp{idx} sg{idx} ramp{idx}_dn vss NSENSE W=96u L=180n",
            f"Rramp{idx}_dn ramp{idx}_dn vss 1G",
            f"Mramp{idx}_clk ramp{idx}_dn phi vss vss NMOS W=96u L=180n",
            f"Cevid{idx} evid{idx} vss 1f IC=0",
            f"Revid{idx} evid{idx} vss 1G",
            f"Mreset_evid{idx} evid{idx} rst vss vss NMOS W=4u L=180n",
            f"Mevid{idx}_p evid{idx} ramp{idx} vdd vdd PMOS W=192u L=180n",
        ]
    lines += _dense_error_core(score_nodes=("evid0", "evid1", "evid2"), sum_width_u=96.0, error_width_u=128.0)
    lines.append(".ends norm_time_domain")
    return "\n".join(lines)


def _subckt_pulse_density() -> str:
    lines = _subckt_header("pulse_density") + _error_caps()
    lines += _low_gain_score_lines()
    for idx in range(3):
        lines += _score_buffer_lines(f"count{idx}", f"sg{idx}", width_u=48.0)
        lines += _score_buffer_lines(f"evid{idx}", f"count{idx}", width_u=96.0)
    lines += _dense_error_core(score_nodes=("evid0", "evid1", "evid2"), sum_width_u=96.0, error_width_u=128.0)
    lines.append(".ends norm_pulse_density")
    return "\n".join(lines)


def _subckt_log_domain() -> str:
    lines = _subckt_header("log_domain") + _error_caps()
    lines += _low_gain_score_lines()
    for idx in range(3):
        lines += [
            f"Clog{idx} log{idx} vss 1f IC=0",
            f"Rlog{idx} log{idx} vss 1G",
            f"Mreset_log{idx} log{idx} rst vss vss NMOS W=4u L=180n",
            f"Mlog{idx}_in vdd sg{idx} log{idx}_a vss NSENSE W=128u L=180n",
            f"Rlog{idx}_a log{idx}_a vss 1G",
            f"Mlog{idx}_diode log{idx}_a log{idx} vss vss NREL W=16u L=180n",
            f"Mlog{idx}_store log{idx}_a phi log{idx} vss NSENSE W=64u L=180n",
        ]
    lines += _dense_error_core(score_nodes=("log0", "log1", "log2"), sum_width_u=128.0, error_width_u=128.0)
    lines.append(".ends norm_log_domain")
    return "\n".join(lines)


def _subckt_learned_calibration() -> str:
    lines = _subckt_header("learned_calibration") + _error_caps()
    lines += _low_gain_score_lines()
    for idx, bias in enumerate((0.62, 0.62, 0.62)):
        lines += [
            f"Vcal{idx} cal{idx} vss {bias:.12g}",
            f"Cevid{idx} evid{idx} vss 1f IC=0",
            f"Revid{idx} evid{idx} vss 1G",
            f"Mreset_evid{idx} evid{idx} rst vss vss NMOS W=4u L=180n",
            f"Mevid{idx}_score vdd sg{idx} evid{idx}_a vss NSENSE W=96u L=180n",
            f"Revid{idx}_a evid{idx}_a vss 1G",
            f"Mevid{idx}_cal evid{idx}_a cal{idx} evid{idx}_b vss NSENSE W=24u L=180n",
            f"Revid{idx}_b evid{idx}_b vss 1G",
            f"Mevid{idx}_clk evid{idx}_b phi evid{idx} vss NSENSE W=96u L=180n",
        ]
    lines += _dense_error_core(score_nodes=("evid0", "evid1", "evid2"), sum_width_u=96.0, error_width_u=128.0)
    lines.append(".ends norm_learned_calibration")
    return "\n".join(lines)


SUBCKT_BUILDERS = {
    "current-sum": _subckt_current_sum,
    "common-mode": _subckt_common_mode,
    "pairwise-margin": _subckt_pairwise_margin,
    "fixed-total-feedback": _subckt_fixed_total_feedback,
    "soft-wta": _subckt_soft_wta,
    "charge-share": _subckt_charge_share,
    "time-domain": _subckt_time_domain,
    "pulse-density": _subckt_pulse_density,
    "log-domain": _subckt_log_domain,
    "learned-calibration": _subckt_learned_calibration,
}


def spice_subckt_name(approach: str) -> str:
    if approach not in APPROACHES:
        raise ValueError(f"approach must be one of {APPROACHES}")
    return "norm_" + approach.replace("-", "_")


def normalization_subcircuits(*, approaches: tuple[str, ...] = APPROACHES) -> str:
    return "\n\n".join(SUBCKT_BUILDERS[approach]() for approach in approaches) + "\n"


def generate_netlist(
    *,
    approach: str,
    case: str,
    score_values: tuple[float, ...] | None = None,
    target_class: int | None = None,
) -> str:
    if approach not in APPROACHES:
        raise ValueError(f"approach must be one of {APPROACHES}")
    default_scores, default_target = class_scores(case)
    scores = default_scores if score_values is None else tuple(float(value) for value in score_values)
    target = default_target if target_class is None else int(target_class)
    if len(scores) != 3:
        raise ValueError("score_values must contain three values")
    if target < 0 or target >= 3:
        raise ValueError("target_class must be in [0, 2]")
    if min(scores) < 0.0 or max(scores) > 1.2:
        raise ValueError("score values must stay within supply rails")

    targetp = [1.2 if idx == target else 0.0 for idx in range(3)]
    targetn = [0.0 if idx == target else 1.2 for idx in range(3)]
    subckt = spice_subckt_name(approach)
    lines = [
        f"* Normalization subcircuit smoke: {approach} / {case}.",
        "* Python supplies fixed score/label/clock sources; no behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        normalization_subcircuits(approaches=(approach,)),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(0 1.2 0.0n 10p 10p 0.45n 10n)",
        "Vphi phi 0 PULSE(0 1.2 1.0n 100p 100p 20.0n 50n)",
    ]
    for idx, score in enumerate(scores):
        lines += [
            f"Vs{idx} s{idx} 0 {score:.12g}",
            f"Vtp{idx} tp{idx} 0 {targetp[idx]:.12g}",
            f"Vtn{idx} tn{idx} 0 {targetn[idx]:.12g}",
        ]
    lines += [
        f"Xnorm s0 s1 s2 tp0 tp1 tp2 tn0 tn1 tn2 phi rst {_err_nodes()} vdd 0 {subckt}",
    ]
    for idx in range(3):
        lines += [
            f".meas tran e{idx}p_after FIND V(e{idx}p) AT=4.7n",
            f".meas tran e{idx}n_after FIND V(e{idx}n) AT=4.7n",
            f".meas tran e{idx}_diff PARAM='e{idx}p_after-e{idx}n_after'",
        ]
    lines += [
        ".tran 2p 5n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def classify_case(row: dict[str, Any], *, min_abs_margin: float) -> str:
    case = str(row["case"])
    target = int(row["target_class"])
    target_diff = float(row[f"e{target}_diff"])
    non_target_diffs = [float(row[f"e{idx}_diff"]) for idx in range(3) if idx != target]
    if case == "flat_target1":
        if target_diff > min_abs_margin and all(diff < -min_abs_margin for diff in non_target_diffs):
            return "dense_bootstrap"
        return "weak"
    if target_diff <= min_abs_margin:
        return "weak_target"
    if not all(diff < -min_abs_margin for diff in non_target_diffs):
        return "weak_nontarget"
    return "aligned"


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    approaches = APPROACHES if args.approach == "all" else (args.approach,)
    for approach in approaches:
        for case in CASES:
            scores, target = class_scores(case)
            path = generated / f"{tag}_{sanitize_tag(approach)}_{sanitize_tag(case)}.cir"
            measures = run_netlist(
                spice_bin,
                path,
                generate_netlist(approach=approach, case=case),
                timeout=args.timeout,
            )
            row = {
                "approach": approach,
                "case": case,
                "target_class": target,
                **measures,
            }
            row["classification"] = classify_case(row, min_abs_margin=args.min_abs_margin)
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
    passed = all(str(row["classification"]) in {"aligned", "dense_bootstrap"} for row in rows)
    summary = {
        "simulator": version,
        "architecture": "normalization_subcircuit_library",
        "approaches": list(approaches),
        "cases": CASES,
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
    ap.add_argument("--tag", default="normalization_subcircuits")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--approach", choices=("all", *APPROACHES), default="all")
    ap.add_argument("--min-abs-margin", type=float, default=0.02)
    return ap


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.min_abs_margin <= 0.0:
        raise ValueError("min-abs-margin must be positive")
    return args


def main() -> None:
    summary = run_cases(main_for_test(None))  # type: ignore[arg-type]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
