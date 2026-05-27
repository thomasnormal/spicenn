from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


UPDATE_MODES = ("none", "positive", "negative")
CREDIT_MODES = ("none", "positive", "negative")
PERIOD_NS = 24.0
EDGE_NS = 0.01


def update_rails(mode: str) -> tuple[float, float]:
    if mode == "none":
        return 0.0, 0.0
    if mode == "positive":
        return 1.2, 0.0
    if mode == "negative":
        return 0.0, 1.2
    raise ValueError(f"update mode must be one of {UPDATE_MODES}")


def credit_rails(mode: str) -> tuple[float, float]:
    if mode == "none":
        return 0.0, 0.0
    if mode == "positive":
        return 1.2, 0.0
    if mode == "negative":
        return 0.0, 1.2
    raise ValueError(f"credit mode must be one of {CREDIT_MODES}")


def pwl(points: list[tuple[float, float]]) -> str:
    compact: list[tuple[float, float]] = []
    for time_ns, value in sorted(points):
        if compact and abs(compact[-1][0] - time_ns) < 1e-15:
            compact[-1] = (time_ns, value)
        else:
            compact.append((time_ns, value))
    return "PWL(" + " ".join(f"{time_ns:.12g}n {value:.12g}" for time_ns, value in compact) + ")"


def cycle_window_pwl(
    values: list[float],
    *,
    start_ns: float,
    end_ns: float,
    period_ns: float = PERIOD_NS,
    edge_ns: float = EDGE_NS,
) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for cycle, value in enumerate(values):
        base = period_ns * cycle
        points += [
            (base + start_ns, 0.0),
            (base + start_ns + edge_ns, value),
            (base + end_ns, value),
            (base + end_ns + edge_ns, 0.0),
        ]
    points.append((period_ns * (len(values) - 1) + 18.0, 0.0))
    return pwl(points)


def generate_netlist(
    *,
    wp: float,
    wn: float,
    row: float = 0.85,
    update_mode: str = "positive",
    credit_mode: str = "positive",
    readout_wp: float | None = None,
    readout_wn: float | None = None,
    syn_width: float = 1.0,
    row_drive_width: float = 12.0,
    update_width: float = 0.25,
    credit_width: float = 8.0,
    cycles: int = 1,
    cycle_rows: list[float] | tuple[float, ...] | None = None,
    cycle_update_modes: list[str] | tuple[str, ...] | None = None,
) -> str:
    if update_mode not in UPDATE_MODES:
        raise ValueError(f"update_mode must be one of {UPDATE_MODES}")
    if credit_mode not in CREDIT_MODES:
        raise ValueError(f"credit_mode must be one of {CREDIT_MODES}")
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if cycle_rows is not None and len(cycle_rows) != cycles:
        raise ValueError("cycle_rows length must match cycles")
    if cycle_update_modes is not None and len(cycle_update_modes) != cycles:
        raise ValueError("cycle_update_modes length must match cycles")
    if cycle_update_modes is not None:
        bad_modes = sorted(set(cycle_update_modes) - set(UPDATE_MODES))
        if bad_modes:
            raise ValueError(f"cycle_update_modes entries must be one of {UPDATE_MODES}: {bad_modes}")
    for name, value in {
        "syn_width": syn_width,
        "row_drive_width": row_drive_width,
        "update_width": update_width,
        "credit_width": credit_width,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    per_cycle_rows = list(cycle_rows) if cycle_rows is not None else [row] * cycles
    per_cycle_update_modes = list(cycle_update_modes) if cycle_update_modes is not None else [update_mode] * cycles
    ep, en = update_rails(update_mode)
    ep_values = [update_rails(mode)[0] for mode in per_cycle_update_modes]
    en_values = [update_rails(mode)[1] for mode in per_cycle_update_modes]
    edp, edn = credit_rails(credit_mode)
    rwp = wp if readout_wp is None else readout_wp
    rwn = wn if readout_wn is None else readout_wn
    uses_cycle_sources = cycle_rows is not None or cycle_update_modes is not None
    row_source = (
        cycle_window_pwl(per_cycle_rows, start_ns=1.0, end_ns=4.05)
        if uses_cycle_sources
        else f"PULSE(0 {row:.12g} 1.0n 10p 10p 3.05n 24n)"
    )
    ep_source = (
        cycle_window_pwl(ep_values, start_ns=5.0, end_ns=8.0)
        if uses_cycle_sources
        else f"PULSE(0 {ep:.12g} 5.0n 10p 10p 3.0n 24n)"
    )
    en_source = (
        cycle_window_pwl(en_values, start_ns=5.0, end_ns=8.0)
        if uses_cycle_sources
        else f"PULSE(0 {en:.12g} 5.0n 10p 10p 3.0n 24n)"
    )
    stop_ns = 18.0 + PERIOD_NS * (cycles - 1)
    lines = [
        "* Row-pulsed differential conductance primitive smoke.",
        "* The compute path is row -> conductance(weight gate) -> capacitive rails.",
        "* No behavioral sources or Python-updated state.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(0 1.2 0.0n 10p 10p 0.55n 24n)",
        "Vfwd fwd 0 PULSE(0 1.2 1.0n 10p 10p 3.0n 24n)",
        "Vfwdn fwdn 0 PULSE(1.2 0 1.0n 10p 10p 3.0n 24n)",
        f"Vrow_src row_src 0 {row_source}",
        f"Vep ep 0 {ep_source}",
        f"Ven en 0 {en_source}",
        "Vapply apply 0 PULSE(0 1.2 5.0n 10p 10p 3.0n 24n)",
        f"Vedp edp 0 PULSE(0 {edp:.12g} 11.0n 10p 10p 4.0n 24n)",
        f"Vedn edn 0 PULSE(0 {edn:.12g} 11.0n 10p 10p 4.0n 24n)",
        f"Cwp wp 0 20f IC={wp:.12g}",
        f"Cwn wn 0 20f IC={wn:.12g}",
        f"Cvwp vwp 0 20f IC={rwp:.12g}",
        f"Cvwn vwn 0 20f IC={rwn:.12g}",
        "Rwp wp 0 1e15",
        "Rwn wn 0 1e15",
        "Rvwp vwp 0 1e15",
        "Rvwn vwn 0 1e15",
        "Cpre_p pre_p 0 20f IC=0",
        "Cpre_n pre_n 0 20f IC=0",
        "Chdp hdp 0 12f IC=0",
        "Chdn hdn 0 12f IC=0",
        "Rpre_p pre_p 0 1G",
        "Rpre_n pre_n 0 1G",
        "Rhdp hdp 0 1G",
        "Rhdn hdn 0 1G",
        "",
        "* Row driver: value source is disconnected outside forward, leaving row high-Z except reset/leak.",
        f"Mrow_n row fwd row_src 0 NMOS W={row_drive_width:.6g}u L=180n",
        f"Mrow_p row fwdn row_src vdd PMOS W={2.0 * row_drive_width:.6g}u L=180n",
        f"Mrow_rst row rst 0 0 NMOS W={max(1.0, row_drive_width / 3.0):.6g}u L=180n",
        "Rrow row 0 1e12",
        f"Mpre_p_rst pre_p rst 0 0 NMOS W={max(2.0, row_drive_width / 2.0):.6g}u L=180n",
        f"Mpre_n_rst pre_n rst 0 0 NMOS W={max(2.0, row_drive_width / 2.0):.6g}u L=180n",
        f"Mhdp_rst hdp rst 0 0 NMOS W={max(2.0, row_drive_width / 2.0):.6g}u L=180n",
        f"Mhdn_rst hdn rst 0 0 NMOS W={max(2.0, row_drive_width / 2.0):.6g}u L=180n",
        "",
        "* Forward signed conductance pair: row current/charge sums onto differential pre rails.",
        f"Mwp_fwd row wp pre_p 0 NMOS W={syn_width:.6g}u L=180n",
        f"Mwn_fwd row wn pre_n 0 NMOS W={syn_width:.6g}u L=180n",
        "",
        "* Local update writer. Eligibility is the held pre rail; positive error raises wp/lowers wn, negative error does the opposite.",
        f"Mwp_up_e vdd ep wp_up_e 0 NSENSE W={update_width:.6g}u L=180n",
        f"Mwp_up_x wp_up_e pre_p wp_up_x 0 NSENSE W={update_width:.6g}u L=180n",
        f"Mwp_up_a wp_up_x apply wp 0 NREL W={update_width:.6g}u L=180n",
        f"Mwn_dn_a wn apply wn_dn_a 0 NREL W={update_width:.6g}u L=180n",
        f"Mwn_dn_x wn_dn_a pre_p wn_dn_x 0 NSENSE W={update_width:.6g}u L=180n",
        f"Mwn_dn_e wn_dn_x ep 0 0 NSENSE W={update_width:.6g}u L=180n",
        f"Mwn_up_e vdd en wn_up_e 0 NSENSE W={update_width:.6g}u L=180n",
        f"Mwn_up_x wn_up_e pre_p wn_up_x 0 NSENSE W={update_width:.6g}u L=180n",
        f"Mwn_up_a wn_up_x apply wn 0 NREL W={update_width:.6g}u L=180n",
        f"Mwp_dn_a wp apply wp_dn_a 0 NREL W={update_width:.6g}u L=180n",
        f"Mwp_dn_x wp_dn_a pre_p wp_dn_x 0 NSENSE W={update_width:.6g}u L=180n",
        f"Mwp_dn_e wp_dn_x en 0 0 NSENSE W={update_width:.6g}u L=180n",
        "",
        "* Latch-free analog backward credit through readout conductance pair.",
        f"Mhdp_p edp vwp hdp 0 NSENSE W={credit_width:.6g}u L=180n",
        f"Mhdn_p edp vwn hdn 0 NSENSE W={credit_width:.6g}u L=180n",
        f"Mhdp_n edn vwn hdp 0 NSENSE W={credit_width:.6g}u L=180n",
        f"Mhdn_n edn vwp hdn 0 NSENSE W={credit_width:.6g}u L=180n",
        ".meas tran pre_p_after FIND V(pre_p) AT=4.5n",
        ".meas tran pre_n_after FIND V(pre_n) AT=4.5n",
        ".meas tran forward_margin PARAM='pre_p_after-pre_n_after'",
        ".meas tran wp_before FIND V(wp) AT=4.5n",
        ".meas tran wn_before FIND V(wn) AT=4.5n",
        ".meas tran wp_after FIND V(wp) AT=9.0n",
        ".meas tran wn_after FIND V(wn) AT=9.0n",
        ".meas tran signed_weight_before PARAM='wp_before-wn_before'",
        ".meas tran signed_weight_after PARAM='wp_after-wn_after'",
        ".meas tran signed_weight_delta PARAM='signed_weight_after-signed_weight_before'",
        ".meas tran hdp_after FIND V(hdp) AT=14.5n",
        ".meas tran hdn_after FIND V(hdn) AT=14.5n",
        ".meas tran hidden_credit_margin PARAM='hdp_after-hdn_after'",
        *(
            [
                ".meas tran pre_p_after_cycle2_reset FIND V(pre_p) AT=24.45n",
                ".meas tran pre_n_after_cycle2_reset FIND V(pre_n) AT=24.45n",
                ".meas tran pre_p_after_cycle2 FIND V(pre_p) AT=28.5n",
                ".meas tran pre_n_after_cycle2 FIND V(pre_n) AT=28.5n",
                ".meas tran forward_margin_cycle2 PARAM='pre_p_after_cycle2-pre_n_after_cycle2'",
                ".meas tran wp_after_cycle2 FIND V(wp) AT=33.0n",
                ".meas tran wn_after_cycle2 FIND V(wn) AT=33.0n",
                ".meas tran signed_weight_drift_cycle2 PARAM='(wp_after_cycle2-wn_after_cycle2)-signed_weight_before'",
            ]
            if cycles >= 2
            else []
        ),
        f".tran 5p {stop_ns:.12g}n uic",
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
    if abs(actual) < min_abs_margin:
        return "weak"
    return "flipped"


def classify_row(row: dict[str, Any], *, min_abs_margin: float) -> dict[str, str]:
    wp = float(row["wp"])
    wn = float(row["wn"])
    readout_wp = float(row["readout_wp"])
    readout_wn = float(row["readout_wn"])
    row_v = float(row["row"])
    update_mode = str(row["update_mode"])
    credit_mode = str(row["credit_mode"])
    update_expected = 0.0
    if update_mode == "positive" and row_v > 0.0:
        update_expected = 1.0
    elif update_mode == "negative" and row_v > 0.0:
        update_expected = -1.0
    credit_expected = 0.0
    if credit_mode == "positive":
        credit_expected = readout_wp - readout_wn
    elif credit_mode == "negative":
        credit_expected = readout_wn - readout_wp
    return {
        "forward_classification": classify_sign(
            float(row["forward_margin"]),
            (wp - wn) if row_v > 0.0 else 0.0,
            min_abs_margin=min_abs_margin,
        ),
        "update_classification": classify_sign(
            float(row["signed_weight_delta"]),
            update_expected,
            min_abs_margin=min_abs_margin,
        ),
        "hidden_credit_classification": classify_sign(
            float(row["hidden_credit_margin"]),
            credit_expected,
            min_abs_margin=min_abs_margin,
        ),
    }


def default_cases() -> list[dict[str, Any]]:
    return [
        {
            "case": "forward_positive_credit_positive",
            "wp": 0.70,
            "wn": 0.25,
            "row": 0.85,
            "update_mode": "none",
        },
        {
            "case": "forward_negative_credit_negative",
            "wp": 0.25,
            "wn": 0.70,
            "row": 0.85,
            "update_mode": "none",
        },
        {"case": "zero_row_dead_zone", "wp": 0.70, "wn": 0.25, "row": 0.0, "update_mode": "none", "credit_mode": "none"},
        {"case": "positive_error_strengthens_signed_weight", "wp": 0.45, "wn": 0.40, "row": 0.85, "update_mode": "positive"},
        {"case": "negative_error_weakens_signed_weight", "wp": 0.45, "wn": 0.40, "row": 0.85, "update_mode": "negative"},
        {
            "case": "positive_error_negative_readout_credit",
            "wp": 0.45,
            "wn": 0.40,
            "row": 0.85,
            "credit_mode": "positive",
            "readout_wp": 0.30,
            "readout_wn": 0.55,
        },
        {
            "case": "negative_error_positive_readout_credit",
            "wp": 0.45,
            "wn": 0.40,
            "row": 0.85,
            "credit_mode": "negative",
            "readout_wp": 0.30,
            "readout_wn": 0.55,
        },
        {
            "case": "near_neutral_readout_dead_zone",
            "wp": 0.45,
            "wn": 0.40,
            "row": 0.85,
            "credit_mode": "positive",
            "readout_wp": 0.40,
            "readout_wn": 0.40,
            "update_mode": "none",
        },
    ]


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for case in default_cases():
        row_data = dict(case)
        row_data.setdefault("update_mode", args.update_mode)
        row_data.setdefault("credit_mode", args.credit_mode)
        row_data.setdefault("readout_wp", row_data["wp"])
        row_data.setdefault("readout_wn", row_data["wn"])
        path = generated / f"{tag}_{sanitize_tag(str(row_data['case']))}.cir"
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(
                wp=float(row_data["wp"]),
                wn=float(row_data["wn"]),
                row=float(row_data["row"]),
                update_mode=str(row_data["update_mode"]),
                credit_mode=str(row_data["credit_mode"]),
                readout_wp=float(row_data["readout_wp"]),
                readout_wn=float(row_data["readout_wn"]),
                syn_width=args.syn_width,
                row_drive_width=args.row_drive_width,
                update_width=args.update_width,
                credit_width=args.credit_width,
                cycles=args.cycles,
            ),
            timeout=args.timeout,
        )
        row = {**row_data, **measures}
        row.update(classify_row(row, min_abs_margin=args.min_abs_margin))
        rows.append(row)
    csv_path = tables / f"{tag}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    classification_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in ["forward_classification", "update_classification", "hidden_credit_classification"]:
            classification_counts.setdefault(key, {})
            cls = str(row[key])
            classification_counts[key][cls] = classification_counts[key].get(cls, 0) + 1
    passed = all(
        str(row[key]) in {"aligned", "dead_zone"}
        for row in rows
        for key in ["forward_classification", "update_classification", "hidden_credit_classification"]
    )
    summary = {
        "simulator": version,
        "architecture": "row_pulsed_differential_conductance_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "cases": len(rows),
        "passed": passed,
        "classification_counts": classification_counts,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="row_conductance_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--update-mode", choices=UPDATE_MODES, default="positive")
    ap.add_argument("--credit-mode", choices=CREDIT_MODES, default="positive")
    ap.add_argument("--syn-width", type=float, default=1.0)
    ap.add_argument("--row-drive-width", type=float, default=12.0)
    ap.add_argument("--update-width", type=float, default=0.25)
    ap.add_argument("--credit-width", type=float, default=8.0)
    ap.add_argument("--min-abs-margin", type=float, default=1e-3)
    ap.add_argument("--cycles", type=int, default=1)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    for name in ["syn_width", "row_drive_width", "update_width", "credit_width"]:
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.min_abs_margin < 0.0:
        raise ValueError("min-abs-margin must be nonnegative")
    if args.cycles < 1:
        raise ValueError("cycles must be at least 1")


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
