from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from run_spice_sweep import ROOT, detect_spice, run_tiny_test


MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def mos_models() -> str:
    return """
.model NMOS NMOS LEVEL=1 VTO=0.35 KP=220u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
""".strip()


def pwl(points: list[tuple[float, float]]) -> str:
    return "PWL(" + " ".join(f"{t:.12g}n {v:.12g}" for t, v in points) + ")"


def phase_waveforms(vin: float, delta_p: float, delta_n: float) -> str:
    vdd = 1.2
    rst = [
        (0.0, vdd),
        (0.5, vdd),
        (0.55, 0.0),
        (2.1, 0.0),
        (2.15, vdd),
        (2.45, vdd),
        (2.50, 0.0),
        (7.7, 0.0),
        (7.75, vdd),
        (8.05, vdd),
        (8.10, 0.0),
        (11.0, 0.0),
    ]
    pix = [(0.0, 0.0), (0.5, 0.0), (0.55, vin), (11.0, vin)]
    acc = [(0.0, 0.0), (2.70, 0.0), (2.75, vdd), (5.00, vdd), (5.05, 0.0), (11.0, 0.0)]
    apply = [(0.0, 0.0), (5.40, 0.0), (5.45, vdd), (7.40, vdd), (7.45, 0.0), (11.0, 0.0)]
    dp = [(0.0, 0.0), (2.65, 0.0), (2.70, delta_p), (5.10, delta_p), (5.15, 0.0), (11.0, 0.0)]
    dn = [(0.0, 0.0), (2.65, 0.0), (2.70, delta_n), (5.10, delta_n), (5.15, 0.0), (11.0, 0.0)]
    return "\n".join(
        [
            f"Vrst rst 0 {pwl(rst)}",
            f"Vpix pix 0 {pwl(pix)}",
            f"Vacc acc 0 {pwl(acc)}",
            f"Vapply apply 0 {pwl(apply)}",
            f"Vdp dp 0 {pwl(dp)}",
            f"Vdn dn 0 {pwl(dn)}",
        ]
    )


def signed_learning_cell_netlist(
    vin: float,
    delta_p: float,
    delta_n: float,
    wp_init: float,
    wn_init: float,
    exclusive_delta_gate: bool = False,
) -> str:
    if exclusive_delta_gate:
        positive_source = "gp_src"
        negative_source = "gn_src"
        exclusive_gate = """
* Mutually inhibit conflicting error rails: dp writes only when dn is low, and conversely.
Mgp_inhibit vdd dn gp_src vdd PMOS W=8u L=180n
Mgn_inhibit vdd dp gn_src vdd PMOS W=8u L=180n
Mgp_kill gp dn 0 0 NMOS W=8u L=180n
Mgn_kill gn dp 0 0 NMOS W=8u L=180n
""".strip()
        exclusive_comment = " with PMOS mutual-inhibit gates"
    else:
        positive_source = "vdd"
        negative_source = "vdd"
        exclusive_gate = ""
        exclusive_comment = ""
    return f"""
* Device-level signed ReLU learning cell.
* No behavioral tanh or arithmetic multiplier appears in the signal path.
* Signed weight is represented by two nonnegative capacitor states: Cwp-Cwn.
* Data-derived gradient caps Cgp/Cgn are charged by input/delta transistor stacks.
* Error-rail selectivity: {exclusive_delta_gate}{exclusive_comment}.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
{phase_waveforms(vin, delta_p, delta_n)}

Cwp wp 0 20f IC={wp_init:.12g}
Cwn wn 0 20f IC={wn_init:.12g}
Cgp gp 0 20f IC=0
Cgn gn 0 20f IC=0
Rwp wp 0 1e15
Rwn wn 0 1e15
Rgp gp 0 1e15
Rgn gn 0 1e15

Mreset_pre pre rst 0 0 NMOS W=2u L=180n
Mreset_act act rst 0 0 NMOS W=2u L=180n

* Positive signed branch: input charges Cpre through a weight-controlled NMOS.
Msyn_pos pix wp pre 0 NMOS W=32u L=180n

* Negative signed branch: input and negative weight open a discharge stack.
Mneg_w pre wn negmid 0 NMOS W=24u L=180n
Mneg_x negmid pix 0 0 NMOS W=24u L=180n

Cpre pre 0 10f IC=0
Rpre pre 0 1G
Mrelu vdd pre act 0 NMOS W=16u L=180n
Cact act 0 20f IC=0
Ract act 0 1G

{exclusive_gate}

* Positive data-derived gradient accumulator: pix AND positive-delta AND acc.
Mgp_x {positive_source} pix gp_x 0 NMOS W=8u L=180n
Mgp_d gp_x dp gp_d 0 NMOS W=8u L=180n
Mgp_a gp_d acc gp 0 NMOS W=8u L=180n

* Negative data-derived gradient accumulator: pix AND negative-delta AND acc.
Mgn_x {negative_source} pix gn_x 0 NMOS W=8u L=180n
Mgn_d gn_x dn gn_d 0 NMOS W=8u L=180n
Mgn_a gn_d acc gn 0 NMOS W=8u L=180n

* Apply positive gradient: charge Cwp and discharge Cwn.
Mwp_up_g vdd gp wp_up 0 NMOS W=4u L=180n
Mwp_up_a wp_up apply wp 0 NMOS W=4u L=180n
Mwn_dn_a wn apply wn_dn 0 NMOS W=4u L=180n
Mwn_dn_g wn_dn gp 0 0 NMOS W=4u L=180n

* Apply negative gradient: charge Cwn and discharge Cwp.
Mwn_up_g vdd gn wn_up 0 NMOS W=4u L=180n
Mwn_up_a wn_up apply wn 0 NMOS W=4u L=180n
Mwp_dn_a wp apply wp_dn 0 NMOS W=4u L=180n
Mwp_dn_g wp_dn gn 0 0 NMOS W=4u L=180n

.options method=gear maxord=2
.tran 10p 11n uic
.meas tran pre_before FIND V(pre) AT=2.0n
.meas tran act_before FIND V(act) AT=2.0n
.meas tran wp_before FIND V(wp) AT=2.0n
.meas tran wn_before FIND V(wn) AT=2.0n
.meas tran gp_after_acc FIND V(gp) AT=5.2n
.meas tran gn_after_acc FIND V(gn) AT=5.2n
.meas tran wp_after_apply FIND V(wp) AT=7.6n
.meas tran wn_after_apply FIND V(wn) AT=7.6n
.meas tran pre_after FIND V(pre) AT=10.8n
.meas tran act_after FIND V(act) AT=10.8n
.meas tran wp_final FIND V(wp) AT=10.8n
.meas tran wn_final FIND V(wn) AT=10.8n
.meas tran signed_before PARAM='wp_before-wn_before'
.meas tran signed_after PARAM='wp_final-wn_final'
.meas tran d_signed PARAM='signed_after-signed_before'
.meas tran d_act PARAM='act_after-act_before'
.control
run
print pre_before act_before wp_before wn_before gp_after_acc gn_after_acc
print wp_after_apply wn_after_apply pre_after act_after wp_final wn_final
print signed_before signed_after d_signed d_act
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    path.write_text(netlist)
    cmd = [spice_bin, "-b", str(path)] if "ngspice" in Path(spice_bin).name.lower() else [spice_bin, str(path)]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("ngspice produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return measures


def is_monotone(xs: list[float], tolerance: float = 1e-6) -> bool:
    return all(b + tolerance >= a for a, b in zip(xs, xs[1:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--tag", default="device_signed_learning_cell")
    ap.add_argument(
        "--exclusive-delta-gate",
        action="store_true",
        help="Add PMOS mutual-inhibit gates so overlapping positive/negative error rails become a no-write event.",
    )
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    cases = []
    for vin in [0.0, 0.2, 0.5, 0.8, 1.0]:
        cases.append(("positive_input_sweep", vin, 1.1, 0.2))
        cases.append(("negative_input_sweep", vin, 0.2, 1.1))
    for delta_p in [0.2, 0.45, 0.7, 0.9, 1.1]:
        cases.append(("positive_delta_sweep", 0.8, delta_p, 0.2))
    for delta_n in [0.2, 0.45, 0.7, 0.9, 1.1]:
        cases.append(("negative_delta_sweep", 0.8, 0.2, delta_n))
    cases.extend(
        [
            ("balanced_low", 0.8, 0.2, 0.2),
            ("balanced_high", 0.8, 1.1, 1.1),
        ]
    )

    for idx, (experiment, vin, delta_p, delta_n) in enumerate(cases):
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_{idx:03d}.cir",
            signed_learning_cell_netlist(
                vin,
                delta_p,
                delta_n,
                0.95,
                0.25,
                exclusive_delta_gate=args.exclusive_delta_gate,
            ),
            args.timeout,
        )
        rows.append(
            {
                "experiment": experiment,
                "vin": vin,
                "delta_p": delta_p,
                "delta_n": delta_n,
                **measures,
            }
        )

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    pos_input = df[df["experiment"] == "positive_input_sweep"].sort_values("vin")
    neg_input = df[df["experiment"] == "negative_input_sweep"].sort_values("vin")
    pos_delta = df[df["experiment"] == "positive_delta_sweep"].sort_values("delta_p")
    neg_delta = df[df["experiment"] == "negative_delta_sweep"].sort_values("delta_n")
    pos_effect = pos_input[pos_input["gp_after_acc"] > 0.35]
    neg_effect = neg_input[neg_input["gn_after_acc"] > 0.35]
    balanced_high = df[df["experiment"] == "balanced_high"].iloc[0]

    summary = {
        "simulator": version,
        "architecture": "device_level_signed_relu_learning_cell",
        "status": "primitive_device_sweep",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "Differential nonnegative weight caps Cwp/Cwn implement signed conductance. "
            "Cpre and Cact store preactivation and ReLU-like activation. Cgp/Cgn are charged by "
            "input/delta/acc transistor stacks and then update Cwp/Cwn during apply."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "exclusive_delta_gate": args.exclusive_delta_gate,
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "rows": len(df),
        "positive_gradient_signed_update_monotone_by_input": is_monotone([float(x) for x in pos_input["d_signed"]]),
        "negative_gradient_signed_update_monotone_by_input": is_monotone([float(-x) for x in neg_input["d_signed"]]),
        "positive_gradient_accumulator_monotone_by_input": is_monotone([float(x) for x in pos_input["gp_after_acc"]]),
        "negative_gradient_accumulator_monotone_by_input": is_monotone([float(x) for x in neg_input["gn_after_acc"]]),
        "positive_gradient_update_monotone_by_delta": is_monotone([float(x) for x in pos_delta["d_signed"]]),
        "negative_gradient_update_monotone_by_delta": is_monotone([float(-x) for x in neg_delta["d_signed"]]),
        "positive_gradient_increases_activation_when_update_strong": bool((pos_effect["d_act"] > 0).all()),
        "negative_gradient_decreases_activation_when_update_strong": bool((neg_effect["d_act"] < 0).all()),
        "strong_update_threshold_v": 0.35,
        "positive_strong_update_cases": int(len(pos_effect)),
        "negative_strong_update_cases": int(len(neg_effect)),
        "max_positive_signed_delta_v": float(pos_input["d_signed"].max()),
        "max_negative_signed_delta_v": float(neg_input["d_signed"].min()),
        "max_positive_activation_delta_v": float(pos_input["d_act"].max()),
        "max_negative_activation_delta_v": float(neg_input["d_act"].min()),
        "max_gp_after_acc_v": float(df["gp_after_acc"].max()),
        "max_gn_after_acc_v": float(df["gn_after_acc"].max()),
        "balanced_high_signed_delta_v": float(balanced_high["d_signed"]),
        "balanced_high_gp_after_acc_v": float(balanced_high["gp_after_acc"]),
        "balanced_high_gn_after_acc_v": float(balanced_high["gn_after_acc"]),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This extends the device-level primitive from preset gradient caps to data-derived gradient accumulation. "
            "It still does not compute output error or backprop deltas from a full network, and it has not been used "
            "for MNIST training."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
