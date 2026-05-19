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


def common_header() -> str:
    return f"""
* Device-level local ReLU/synapse primitive.
* Signal path uses MOSFETs, resistors, capacitors, and voltage sources only.
* Weight state is a floating capacitor voltage that gates an NMOS conductance.
* Activation state is a capacitor charged by an NMOS source follower.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vrst rst 0 PULSE({{VDD}} 0 0 20p 20p 0.5n 20n)
""".strip()


def relu_transfer_netlist(vpre: float) -> str:
    return f"""
{common_header()}
Vpre pre 0 PULSE(0 {vpre:.12g} 0.8n 20p 20p 8n 20n)
Mreset_act act rst 0 0 NMOS W=2u L=180n
Mrelu vdd pre act 0 NMOS W=16u L=180n
Cact act 0 20f IC=0
Ract act 0 1G
.options method=gear maxord=2
.tran 10p 8n uic
.meas tran vact FIND V(act) AT=7.5n
.meas tran vpre_m FIND V(pre) AT=7.5n
.meas tran eact PARAM='0.5*20f*vact*vact'
.control
run
print vpre_m vact eact
.endc
.end
""".lstrip()


def synapse_relu_netlist(vin: float, wgate: float, second_input: float, second_wgate: float) -> str:
    return f"""
{common_header()}
Vpix0 pix0 0 PULSE(0 {vin:.12g} 0.8n 20p 20p 8n 20n)
Vpix1 pix1 0 PULSE(0 {second_input:.12g} 0.8n 20p 20p 8n 20n)
Cw0 w0 0 20f IC={wgate:.12g}
Cw1 w1 0 20f IC={second_wgate:.12g}
Rw0 w0 0 1e15
Rw1 w1 0 1e15
Mreset_pre pre rst 0 0 NMOS W=2u L=180n
Mreset_act act rst 0 0 NMOS W=2u L=180n
Msyn0 pix0 w0 pre 0 NMOS W=32u L=180n
Msyn1 pix1 w1 pre 0 NMOS W=32u L=180n
Cpre pre 0 10f IC=0
Rpre pre 0 1G
Mrelu vdd pre act 0 NMOS W=16u L=180n
Cact act 0 20f IC=0
Ract act 0 1G
.options method=gear maxord=2
.tran 10p 8n uic
.meas tran vpre FIND V(pre) AT=7.5n
.meas tran vact FIND V(act) AT=7.5n
.meas tran vw0 FIND V(w0) AT=7.5n
.meas tran vw1 FIND V(w1) AT=7.5n
.meas tran epre PARAM='0.5*10f*vpre*vpre'
.meas tran eact PARAM='0.5*20f*vact*vact'
.control
run
print vpre vact vw0 vw1 epre eact
.endc
.end
""".lstrip()


def update_cell_netlist(w_init: float, gplus: float, gminus: float) -> str:
    return f"""
{common_header()}
Vapply apply 0 PULSE(0 {{VDD}} 1n 20p 20p 4n 20n)
Cw w 0 20f IC={w_init:.12g}
Cgp gp 0 20f IC={gplus:.12g}
Cgn gn 0 20f IC={gminus:.12g}
Rw w 0 1e15
Rgp gp 0 1e15
Rgn gn 0 1e15
Mup_g vdd gp upmid 0 NMOS W=4u L=180n
Mup_a upmid apply w 0 NMOS W=4u L=180n
Mdn_a w apply dnmid 0 NMOS W=4u L=180n
Mdn_g dnmid gn 0 0 NMOS W=4u L=180n
.options method=gear maxord=2
.tran 10p 8n uic
.meas tran w_before FIND V(w) AT=0.8n
.meas tran w_final FIND V(w) AT=7.5n
.meas tran gp_final FIND V(gp) AT=7.5n
.meas tran gn_final FIND V(gn) AT=7.5n
.meas tran ew_final PARAM='0.5*20f*w_final*w_final'
.control
run
print w_before w_final gp_final gn_final ew_final
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
    ap.add_argument("--tag", default="device_relu_synapse_sweep")
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

    relu_inputs = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    for vpre in relu_inputs:
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_relu_vpre_{vpre:.3f}.cir",
            relu_transfer_netlist(vpre),
            args.timeout,
        )
        rows.append(
            {
                "experiment": "relu_transfer",
                "vin": None,
                "wgate": None,
                "second_input": None,
                "second_wgate": None,
                "forced_vpre": vpre,
                **measures,
            }
        )

    synapse_inputs = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
    weight_gates = [0.2, 0.35, 0.45, 0.55, 0.7, 0.9, 1.1, 1.2]
    for wgate in weight_gates:
        for vin in synapse_inputs:
            measures = run_netlist(
                spice_bin,
                generated / f"{safe_tag}_syn_vin_{vin:.3f}_wg_{wgate:.3f}.cir",
                synapse_relu_netlist(vin, wgate, 0.0, 0.2),
                args.timeout,
            )
            rows.append(
                {
                    "experiment": "single_synapse_relu",
                    "vin": vin,
                    "wgate": wgate,
                    "second_input": 0.0,
                    "second_wgate": 0.2,
                    "forced_vpre": None,
                    **measures,
                }
            )

    two_synapse_cases = [
        (0.1, 0.7, 0.1, 0.7),
        (0.2, 0.7, 0.1, 0.7),
        (0.2, 0.7, 0.2, 0.7),
        (0.4, 0.7, 0.2, 0.7),
        (0.4, 0.9, 0.2, 0.7),
    ]
    for idx, (vin, wgate, second_input, second_wgate) in enumerate(two_synapse_cases):
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_twosyn_{idx:02d}.cir",
            synapse_relu_netlist(vin, wgate, second_input, second_wgate),
            args.timeout,
        )
        rows.append(
            {
                "experiment": "two_synapse_relu",
                "vin": vin,
                "wgate": wgate,
                "second_input": second_input,
                "second_wgate": second_wgate,
                "forced_vpre": None,
                **measures,
            }
        )

    update_cases = [
        (0.5, 0.2, 0.2),
        (0.5, 0.45, 0.2),
        (0.5, 0.7, 0.2),
        (0.5, 1.0, 0.2),
        (0.5, 1.2, 0.2),
        (0.5, 0.2, 0.45),
        (0.5, 0.2, 0.7),
        (0.5, 0.2, 1.0),
        (0.5, 0.2, 1.2),
        (0.5, 1.2, 1.2),
    ]
    for idx, (w_init, gplus, gminus) in enumerate(update_cases):
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_upd_{idx:02d}.cir",
            update_cell_netlist(w_init, gplus, gminus),
            args.timeout,
        )
        rows.append(
            {
                "experiment": "weight_update_cell",
                "vin": None,
                "wgate": w_init,
                "second_input": gplus,
                "second_wgate": gminus,
                "forced_vpre": None,
                **measures,
            }
        )

    df = pd.DataFrame(rows)
    relu = df[df["experiment"] == "relu_transfer"].sort_values("forced_vpre")
    dead_zone_max = float(relu[relu["forced_vpre"] <= 0.35]["vact"].max())
    high_gain = float(
        (relu[relu["forced_vpre"] == 1.0]["vact"].iloc[0] - relu[relu["forced_vpre"] == 0.5]["vact"].iloc[0]) / 0.5
    )
    single = df[df["experiment"] == "single_synapse_relu"].copy()
    updates = df[df["experiment"] == "weight_update_cell"].copy()
    if not updates.empty:
        updates["delta_w"] = updates["w_final"] - updates["w_before"]
        df.loc[updates.index, "delta_w"] = updates["delta_w"]
        update_positive = updates[updates["second_input"] > updates["second_wgate"]].sort_values("second_input")
        update_negative = updates[updates["second_wgate"] > updates["second_input"]].sort_values("second_wgate")
    else:
        update_positive = pd.DataFrame()
        update_negative = pd.DataFrame()
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)
    monotone_by_input = {}
    for wgate, group in single.groupby("wgate"):
        ordered = group.sort_values("vin")
        monotone_by_input[str(wgate)] = is_monotone([float(x) for x in ordered["vpre"]])
    monotone_by_weight = {}
    for vin, group in single.groupby("vin"):
        ordered = group.sort_values("wgate")
        monotone_by_weight[str(vin)] = is_monotone([float(x) for x in ordered["vpre"]])

    summary = {
        "simulator": version,
        "architecture": "device_level_conductance_synapse_relu_capacitor_primitive",
        "status": "primitive_device_sweep",
        "signal_path": "MOS pass-conductance synapses charge Cpre; NMOS source follower charges Cact as a ReLU-like activation.",
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "weight_state": "Cw0/Cw1 capacitor IC voltages drive NMOS synapse gates.",
        "activation_state": "Cact capacitor voltage after source-follower rectification.",
        "gradient_state": "Cgp/Cgn capacitor IC voltages gate differential charge/discharge update paths in the update-cell sweep.",
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "rows": len(df),
        "relu_dead_zone_max_v": dead_zone_max,
        "relu_high_region_gain_v_per_v": high_gain,
        "relu_transfer_monotone": is_monotone([float(x) for x in relu["vact"]]),
        "synapse_pre_monotone_by_input": monotone_by_input,
        "synapse_pre_monotone_by_weight": monotone_by_weight,
        "max_single_synapse_vpre": float(single["vpre"].max()),
        "max_single_synapse_vact": float(single["vact"].max()),
        "single_synapse_can_drive_relu": bool(single["vact"].max() > 0.02),
        "weight_update_positive_monotone": (
            is_monotone([float(x) for x in update_positive["delta_w"]]) if not update_positive.empty else False
        ),
        "weight_update_negative_monotone": (
            is_monotone([float(-x) for x in update_negative["delta_w"]]) if not update_negative.empty else False
        ),
        "max_positive_weight_delta_v": float(update_positive["delta_w"].max()) if not update_positive.empty else None,
        "max_negative_weight_delta_v": float(update_negative["delta_w"].min()) if not update_negative.empty else None,
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This validates the first device-level replacement target for the behavioral local feature cell: "
            "conductance-weighted input integration followed by a capacitor-held ReLU-like activation. "
            "It also validates a primitive capacitor-held differential update path. It does not yet implement "
            "signed forward weights, backprop delta computation, or data-derived gradient accumulation."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
