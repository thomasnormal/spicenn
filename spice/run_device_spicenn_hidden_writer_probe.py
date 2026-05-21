from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from spicenn import DifferentialCapState, FanInTopology, NetlistBuilder, make_sparse_hidden_update_layer
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spicenn import DifferentialCapState, FanInTopology, NetlistBuilder, make_sparse_hidden_update_layer

from _util import parse_measures
from run_device_multicell_classifier import mos_models
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test


WRITE_MODES = (
    "simple_charge_discharge",
    "analog_trace_charge_discharge",
    "diffpair_charge_discharge",
    "inhibit_charge_discharge",
    "cmos_complementary_charge_discharge",
    "hybrid_trace_spike_charge_discharge",
    "senseamp_charge_discharge",
    "senseamp_cmos_complementary_charge_discharge",
)


def pulse(start_ns: float, end_ns: float, high_v: float = 1.2, stop_ns: float = 4.0) -> str:
    return (
        "PWL("
        f"0n 0 {start_ns:.12g}n 0 {start_ns + 0.02:.12g}n {high_v:.12g} "
        f"{end_ns:.12g}n {high_v:.12g} {end_ns + 0.02:.12g}n 0 {stop_ns:.12g}n 0)"
    )


def measurement_lines(write_mode: str) -> str:
    lines = [
        ".meas tran p_before FIND V(wh0_0p) AT=0.95n",
        ".meas tran n_before FIND V(wh0_0n) AT=0.95n",
        ".meas tran p_after FIND V(wh0_0p) AT=3.75n",
        ".meas tran n_after FIND V(wh0_0n) AT=3.75n",
        ".meas tran hdp_mid FIND V(hd0p) AT=2.20n",
        ".meas tran hdn_mid FIND V(hd0n) AT=2.20n",
    ]
    if write_mode in {
        "diffpair_charge_discharge",
        "inhibit_charge_discharge",
        "cmos_complementary_charge_discharge",
        "hybrid_trace_spike_charge_discharge",
        "senseamp_charge_discharge",
        "senseamp_cmos_complementary_charge_discharge",
    }:
        lines.extend(
            [
                ".meas tran hwpos_mid FIND V(hwpos0) AT=2.20n",
                ".meas tran hwneg_mid FIND V(hwneg0) AT=2.20n",
            ]
        )
    if write_mode == "analog_trace_charge_discharge":
        lines.append(".meas tran pretrace_mid FIND V(fhp0_0) AT=2.20n")
    if write_mode in {
        "cmos_complementary_charge_discharge",
        "hybrid_trace_spike_charge_discharge",
        "senseamp_cmos_complementary_charge_discharge",
    }:
        lines.extend(
            [
                ".meas tran pretrace_mid FIND V(fhp0_0) AT=2.20n",
                ".meas tran pretrace_gate_mid FIND V(fhpg0_0) AT=2.20n",
                ".meas tran pretrace_bar_mid FIND V(fhpbar0_0) AT=2.20n",
            ]
        )
    if write_mode in {
        "hybrid_trace_spike_charge_discharge",
        "senseamp_charge_discharge",
        "senseamp_cmos_complementary_charge_discharge",
    }:
        lines.append(".meas tran hwactive_mid FIND V(hwsel0_active) AT=2.20n")
    if write_mode == "cmos_complementary_charge_discharge":
        lines.extend(
            [
                ".meas tran hwpos_bar_mid FIND V(hwsel0_posbar) AT=2.20n",
                ".meas tran hwneg_bar_mid FIND V(hwsel0_negbar) AT=2.20n",
            ]
        )
    return "\n".join(lines)


def netlist(
    *,
    write_mode: str,
    delta_p_v: float,
    delta_n_v: float,
    pre_v: float,
    weight_p_v: float,
    weight_n_v: float,
    update_width_u: float,
    charge_width_u: float,
    discharge_width_u: float,
    selector_width_u: float,
    stop_ns: float,
) -> str:
    if write_mode not in WRITE_MODES:
        raise ValueError(f"unknown hidden writer mode: {write_mode}")

    topology = FanInTopology.from_fanins((0,), 1, {0: (0,)})
    deck = NetlistBuilder()
    deck.render_component(
        DifferentialCapState.from_base(
            "wh0_0",
            cap_f=4.0,
            pos_ic_v=weight_p_v,
            neg_ic_v=weight_n_v,
            leak_to="0",
            leak_ohm="1e15",
        )
    )
    deck.render_component(
        make_sparse_hidden_update_layer(
            "hidden_update",
            topology=topology,
            source_nodes={0: "pre"},
            weight_prefix="wh",
            update_prefix="uh",
            delta_prefix="hd",
            update_width_u=update_width_u,
            charge_width_u=charge_width_u,
            discharge_width_u=discharge_width_u,
            selector_width_u=selector_width_u,
            write_mode=write_mode,
        )
    )
    return f"""
* spicenn single hidden-writer probe
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vwhigh whigh 0 DC 1.0
Vwlow wlow 0 DC 0.1
Vspikeref spikeref 0 DC 0.30
Vpre pre 0 DC {pre_v:.12g}
Vhdp hd0p 0 DC {delta_p_v:.12g}
Vhdn hd0n 0 DC {delta_n_v:.12g}
Vfwd fwd 0 {pulse(0.10, 0.75, stop_ns=stop_ns)}
Vbwd bwd 0 {pulse(1.05, 3.55, stop_ns=stop_ns)}
Vrstf rstf 0 DC 0
Vrste rste 0 DC 0
{deck.render_body()}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p {stop_ns:.12g}n uic
{measurement_lines(write_mode)}
.control
run
.endc
.end
""".lstrip()


def add_derived(measures: dict[str, float], *, delta_p_v: float, delta_n_v: float, pre_v: float) -> dict[str, float]:
    out = dict(measures)
    signed_before = out["p_before"] - out["n_before"]
    signed_after = out["p_after"] - out["n_after"]
    signed_delta = signed_after - signed_before
    common_delta = (out["p_after"] - out["p_before"]) + (out["n_after"] - out["n_before"])
    delta_diff = delta_p_v - delta_n_v
    delta_common = 0.5 * (delta_p_v + delta_n_v)
    expected = pre_v * delta_diff
    selector_diff = (
        out["hwpos_mid"] - out["hwneg_mid"]
        if "hwpos_mid" in out and "hwneg_mid" in out
        else 0.0
    )
    selector_common = (
        out["hwpos_mid"] + out["hwneg_mid"]
        if "hwpos_mid" in out and "hwneg_mid" in out
        else 0.0
    )
    out["signed_before"] = signed_before
    out["signed_after"] = signed_after
    out["signed_delta"] = signed_delta
    out["common_delta"] = common_delta
    out["delta_diff"] = delta_diff
    out["delta_common"] = delta_common
    out["selector_diff"] = selector_diff
    out["selector_common"] = selector_common
    out["selector_margin_abs"] = abs(selector_diff)
    out["expected_direction"] = expected
    out["sign_correct"] = float((expected > 0 and signed_delta > 0) or (expected < 0 and signed_delta < 0) or (expected == 0 and abs(signed_delta) < 1e-6))
    out["effective_mobility"] = signed_delta / expected if abs(expected) > 1e-30 else 0.0
    out["signed_to_common_abs"] = abs(signed_delta) / (abs(common_delta) + 1e-30)
    out["common_to_signed_abs"] = abs(common_delta) / (abs(signed_delta) + 1e-30)
    out["zero_input_quiet"] = float(abs(expected) > 1e-30 or (abs(signed_delta) < 1e-6 and abs(common_delta) < 1e-6))
    return out


def run_case(
    *,
    spice_bin: str,
    path: Path,
    timeout: float,
    write_mode: str,
    delta_p_v: float,
    delta_n_v: float,
    pre_v: float,
    weight_p_v: float,
    weight_n_v: float,
    update_width_u: float,
    charge_width_u: float,
    discharge_width_u: float,
    selector_width_u: float,
    stop_ns: float,
) -> dict[str, Any]:
    text = netlist(
        write_mode=write_mode,
        delta_p_v=delta_p_v,
        delta_n_v=delta_n_v,
        pre_v=pre_v,
        weight_p_v=weight_p_v,
        weight_n_v=weight_n_v,
        update_width_u=update_width_u,
        charge_width_u=charge_width_u,
        discharge_width_u=discharge_width_u,
        selector_width_u=selector_width_u,
        stop_ns=stop_ns,
    )
    proc = run_text_netlist(spice_bin, path, text, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    row: dict[str, Any] = {
        "write_mode": write_mode,
        "delta_p_v": delta_p_v,
        "delta_n_v": delta_n_v,
        "pre_v": pre_v,
        "weight_p_v": weight_p_v,
        "weight_n_v": weight_n_v,
        "update_width_u": update_width_u,
        "charge_width_u": charge_width_u,
        "discharge_width_u": discharge_width_u,
        "selector_width_u": selector_width_u,
    }
    row.update(add_derived(measures, delta_p_v=delta_p_v, delta_n_v=delta_n_v, pre_v=pre_v))
    return row


def default_sweep_modes(selected: str) -> list[str]:
    if selected == "all":
        return list(WRITE_MODES)
    return [selected]


def default_delta_cases() -> list[tuple[str, float, float]]:
    return [
        ("positive", 0.18, 0.12),
        ("negative", 0.12, 0.18),
        ("overlap", 0.16, 0.16),
        ("large_positive", 0.24, 0.10),
        ("large_negative", 0.10, 0.24),
    ]


def parse_float_list(text: str | None) -> list[float]:
    if text is None or text.strip() == "":
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def safe_number(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def delta_cases_from_diff_list(common_v: float, diff_text: str | None) -> list[tuple[str, float, float]]:
    diffs = parse_float_list(diff_text)
    cases: list[tuple[str, float, float]] = []
    for diff in diffs:
        pos = common_v + 0.5 * diff
        neg = common_v - 0.5 * diff
        sign = "p" if diff > 0 else "n" if diff < 0 else "z"
        cases.append((f"diff_{sign}{safe_number(abs(diff))}", pos, neg))
    return cases


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--tag", default="spicenn_hidden_writer_probe")
    ap.add_argument("--mode", choices=(*WRITE_MODES, "all"), default="inhibit_charge_discharge")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--selector-widths-u", default=None)
    ap.add_argument("--update-widths-u", default=None)
    ap.add_argument("--delta-common-v", type=float, default=None)
    ap.add_argument("--delta-diffs-v", default=None)
    ap.add_argument("--delta-p", type=float, default=0.18)
    ap.add_argument("--delta-n", type=float, default=0.12)
    ap.add_argument("--pre-v", type=float, default=0.50)
    ap.add_argument("--weight-p", type=float, default=0.40)
    ap.add_argument("--weight-n", type=float, default=0.40)
    ap.add_argument("--update-width-u", type=float, default=0.004)
    ap.add_argument("--charge-width-u", type=float, default=None)
    ap.add_argument("--discharge-width-u", type=float, default=None)
    ap.add_argument("--selector-width-u", type=float, default=8.0)
    ap.add_argument("--stop-ns", type=float, default=4.0)
    args = ap.parse_args()

    selector_widths = parse_float_list(args.selector_widths_u) or [args.selector_width_u]
    update_widths = parse_float_list(args.update_widths_u) or [args.update_width_u]

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    t0 = time.perf_counter()

    rows: list[dict[str, Any]] = []
    modes = default_sweep_modes(args.mode)
    delta_cases = (
        delta_cases_from_diff_list(args.delta_common_v, args.delta_diffs_v)
        if args.delta_common_v is not None and args.delta_diffs_v is not None
        else default_delta_cases() if args.sweep else [("single", args.delta_p, args.delta_n)]
    )
    for mode in modes:
        for case_name, delta_p_v, delta_n_v in delta_cases:
            for selector_width_u in selector_widths:
                for update_width_u in update_widths:
                    charge_width_u = update_width_u if args.charge_width_u is None else args.charge_width_u
                    discharge_width_u = update_width_u if args.discharge_width_u is None else args.discharge_width_u
                    if charge_width_u < 0 or discharge_width_u < 0:
                        raise ValueError("charge/discharge widths must be nonnegative")
                    path = generated / (
                        f"{safe_tag}_{mode}_{case_name}_sel{safe_number(selector_width_u)}_upd{safe_number(update_width_u)}.cir"
                    )
                    row = run_case(
                        spice_bin=spice_bin,
                        path=path,
                        timeout=args.timeout,
                        write_mode=mode,
                        delta_p_v=delta_p_v,
                        delta_n_v=delta_n_v,
                        pre_v=args.pre_v,
                        weight_p_v=args.weight_p,
                        weight_n_v=args.weight_n,
                        update_width_u=update_width_u,
                        charge_width_u=charge_width_u,
                        discharge_width_u=discharge_width_u,
                        selector_width_u=selector_width_u,
                        stop_ns=args.stop_ns,
                    )
                    row["case"] = case_name
                    rows.append(row)

    if args.csv is not None:
        write_csv(Path(args.csv), rows)

    summary: dict[str, Any] = {
        "simulator": version,
        "architecture": "spicenn_single_hidden_writer_probe",
        "rows": rows,
        "all_sign_correct": all(bool(row["sign_correct"]) for row in rows),
        "wall_time_s": time.perf_counter() - t0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
