from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test
from _util import MEAS_RE, parse_measures


def mos_models() -> str:
    return """
.model NMOS NMOS LEVEL=1 VTO=0.35 KP=220u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
""".strip()


def pwl(points: list[tuple[float, float]]) -> str:
    return "PWL(" + " ".join(f"{t:.12g}n {v:.12g}" for t, v in points) + ")"


def common_header() -> str:
    return f"""
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
""".strip()


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return measures


def is_monotone(xs: list[float], tolerance: float = 1e-6) -> bool:
    return all(b + tolerance >= a for a, b in zip(xs, xs[1:]))


def error_cell_netlist(target: float, out: float) -> str:
    err = [(0.0, 0.0), (0.90, 0.0), (0.95, 1.2), (4.80, 1.2), (4.85, 0.0), (6.0, 0.0)]
    rst = [(0.0, 1.2), (0.45, 1.2), (0.50, 0.0), (6.0, 0.0)]
    return f"""
* Device-level output-error cell.
* dplus/dminus are capacitor voltages produced by target/output conductance competition.
{common_header()}
Vrst rst 0 {pwl(rst)}
Verr err 0 {pwl(err)}
Vtarget target 0 {target:.12g}
Vout out 0 {out:.12g}

Mreset_dp dp rst 0 0 NMOS W=4u L=180n
Mreset_dn dn rst 0 0 NMOS W=4u L=180n
Cdp dp 0 20f IC=0
Cdn dn 0 20f IC=0
Rdp dp 0 1G
Rdn dn 0 1G

* dplus is encouraged by target and suppressed by output.
Mdp_t0 vdd target dp_t 0 NMOS W=24u L=180n
Mdp_t1 dp_t err dp 0 NMOS W=24u L=180n
Mdp_y0 dp err dp_y 0 NMOS W=18u L=180n
Mdp_y1 dp_y out 0 0 NMOS W=18u L=180n

* dminus is encouraged by output and suppressed by target.
Mdn_y0 vdd out dn_y 0 NMOS W=24u L=180n
Mdn_y1 dn_y err dn 0 NMOS W=24u L=180n
Mdn_t0 dn err dn_t 0 NMOS W=18u L=180n
Mdn_t1 dn_t target 0 0 NMOS W=18u L=180n

.options method=gear maxord=2
.tran 10p 6n uic
.meas tran dplus FIND V(dp) AT=5.8n
.meas tran dminus FIND V(dn) AT=5.8n
.meas tran dnet PARAM='dplus-dminus'
.control
run
print dplus dminus dnet
.endc
.end
""".lstrip()


def hidden_delta_cell_netlist(dp: float, dn: float, wp: float, wn: float, act: float) -> str:
    bwd = [(0.0, 0.0), (0.90, 0.0), (0.95, 1.2), (5.0, 1.2), (5.05, 0.0), (6.0, 0.0)]
    rst = [(0.0, 1.2), (0.45, 1.2), (0.50, 0.0), (6.0, 0.0)]
    return f"""
* Device-level hidden-delta cell.
* hdp/hdn are charged by conductance stacks implementing sign combinations:
* hdp <= dp*wp + dn*wn, hdn <= dn*wp + dp*wn, gated by activation.
{common_header()}
Vrst rst 0 {pwl(rst)}
Vbwd bwd 0 {pwl(bwd)}
Vdp dp 0 {dp:.12g}
Vdn dn 0 {dn:.12g}
Vwp wp 0 {wp:.12g}
Vwn wn 0 {wn:.12g}
Vact act 0 {act:.12g}

Mreset_hdp hdp rst 0 0 NMOS W=4u L=180n
Mreset_hdn hdn rst 0 0 NMOS W=4u L=180n
Chdp hdp 0 12f IC=0
Chdn hdn 0 12f IC=0
Rhdp hdp 0 1G
Rhdn hdn 0 1G

* Positive hidden delta contributions: d+ with w+ and d- with w-.
Mhdp_a0 vdd dp hdp_a0 0 NMOS W=32u L=180n
Mhdp_a1 hdp_a0 wp hdp_a1 0 NMOS W=32u L=180n
Mhdp_a2 hdp_a1 act hdp_a2 0 NMOS W=32u L=180n
Mhdp_a3 hdp_a2 bwd hdp 0 NMOS W=32u L=180n
Mhdp_b0 vdd dn hdp_b0 0 NMOS W=32u L=180n
Mhdp_b1 hdp_b0 wn hdp_b1 0 NMOS W=32u L=180n
Mhdp_b2 hdp_b1 act hdp_b2 0 NMOS W=32u L=180n
Mhdp_b3 hdp_b2 bwd hdp 0 NMOS W=32u L=180n

* Negative hidden delta contributions: d- with w+ and d+ with w-.
Mhdn_a0 vdd dn hdn_a0 0 NMOS W=32u L=180n
Mhdn_a1 hdn_a0 wp hdn_a1 0 NMOS W=32u L=180n
Mhdn_a2 hdn_a1 act hdn_a2 0 NMOS W=32u L=180n
Mhdn_a3 hdn_a2 bwd hdn 0 NMOS W=32u L=180n
Mhdn_b0 vdd dp hdn_b0 0 NMOS W=32u L=180n
Mhdn_b1 hdn_b0 wn hdn_b1 0 NMOS W=32u L=180n
Mhdn_b2 hdn_b1 act hdn_b2 0 NMOS W=32u L=180n
Mhdn_b3 hdn_b2 bwd hdn 0 NMOS W=32u L=180n

.options method=gear maxord=2
.tran 10p 6n uic
.meas tran hdp_v FIND V(hdp) AT=5.8n
.meas tran hdn_v FIND V(hdn) AT=5.8n
.meas tran hnet PARAM='hdp_v-hdn_v'
.control
run
print hdp_v hdn_v hnet
.endc
.end
""".lstrip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--tag", default="device_delta_cells")
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

    error_cases = [
        ("target_sweep_low_output", 0.2, 0.2),
        ("target_sweep_low_output", 0.5, 0.2),
        ("target_sweep_low_output", 0.8, 0.2),
        ("target_sweep_low_output", 1.1, 0.2),
        ("target_sweep_high_output", 0.2, 0.8),
        ("target_sweep_high_output", 0.5, 0.8),
        ("target_sweep_high_output", 0.8, 0.8),
        ("target_sweep_high_output", 1.1, 0.8),
        ("output_sweep_high_target", 1.0, 0.2),
        ("output_sweep_high_target", 1.0, 0.5),
        ("output_sweep_high_target", 1.0, 0.8),
        ("output_sweep_high_target", 1.0, 1.1),
        ("output_sweep_low_target", 0.2, 0.2),
        ("output_sweep_low_target", 0.2, 0.5),
        ("output_sweep_low_target", 0.2, 0.8),
        ("output_sweep_low_target", 0.2, 1.1),
    ]
    for idx, (experiment, target, out) in enumerate(error_cases):
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_err_{idx:03d}.cir",
            error_cell_netlist(target, out),
            args.timeout,
        )
        rows.append({"cell": "error", "experiment": experiment, "target": target, "out": out, **measures})

    hidden_cases = [
        ("pos_delta_pos_weight", 1.1, 0.2, 1.1, 0.2, 1.1),
        ("neg_delta_pos_weight", 0.2, 1.1, 1.1, 0.2, 1.1),
        ("pos_delta_neg_weight", 1.1, 0.2, 0.2, 1.1, 1.1),
        ("neg_delta_neg_weight", 0.2, 1.1, 0.2, 1.1, 1.1),
        ("inactive_pos_delta_pos_weight", 1.1, 0.2, 1.1, 0.2, 0.2),
        ("balanced_delta", 1.1, 1.1, 1.1, 0.2, 1.1),
        ("balanced_weight", 1.1, 0.2, 1.1, 1.1, 1.1),
    ]
    for idx, (experiment, dp, dn, wp, wn, act) in enumerate(hidden_cases):
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_hid_{idx:03d}.cir",
            hidden_delta_cell_netlist(dp, dn, wp, wn, act),
            args.timeout,
        )
        rows.append(
            {
                "cell": "hidden_delta",
                "experiment": experiment,
                "dp": dp,
                "dn": dn,
                "wp": wp,
                "wn": wn,
                "act": act,
                **measures,
            }
        )

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    error = df[df["cell"] == "error"].copy()
    target_low_output = error[error["experiment"] == "target_sweep_low_output"].sort_values("target")
    target_high_output = error[error["experiment"] == "target_sweep_high_output"].sort_values("target")
    output_high_target = error[error["experiment"] == "output_sweep_high_target"].sort_values("out")
    output_low_target = error[error["experiment"] == "output_sweep_low_target"].sort_values("out")
    hidden = df[df["cell"] == "hidden_delta"].set_index("experiment")

    summary = {
        "simulator": version,
        "architecture": "device_level_error_and_hidden_delta_cells",
        "status": "primitive_device_sweep",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "Output dplus/dminus caps are produced by target/output conductance competition. "
            "Hidden hdp/hdn caps are produced by output-delta/readout-weight sign-combination transistor stacks gated by activation."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "rows": len(df),
        "error_dplus_monotone_by_target_low_output": is_monotone([float(x) for x in target_low_output["dplus"]]),
        "error_dminus_monotone_by_output_high_target": is_monotone([float(x) for x in output_high_target["dminus"]]),
        "error_dminus_monotone_by_output_low_target": is_monotone([float(x) for x in output_low_target["dminus"]]),
        "error_net_monotone_by_target_low_output": is_monotone([float(x) for x in target_low_output["dnet"]]),
        "error_net_monotone_by_target_high_output": is_monotone([float(x) for x in target_high_output["dnet"]]),
        "error_net_decreases_by_output_high_target": is_monotone([float(-x) for x in output_high_target["dnet"]]),
        "error_net_decreases_by_output_low_target": is_monotone([float(-x) for x in output_low_target["dnet"]]),
        "error_prefers_positive_when_target_exceeds_output": bool(
            (error[error["target"] >= error["out"] + 0.3]["dnet"] > 0).all()
        ),
        "error_prefers_negative_when_output_exceeds_target": bool(
            (error[error["out"] >= error["target"] + 0.3]["dnet"] < 0).all()
        ),
        "hidden_pos_delta_pos_weight_positive": bool(hidden.loc["pos_delta_pos_weight", "hnet"] > 0),
        "hidden_neg_delta_pos_weight_negative": bool(hidden.loc["neg_delta_pos_weight", "hnet"] < 0),
        "hidden_pos_delta_neg_weight_negative": bool(hidden.loc["pos_delta_neg_weight", "hnet"] < 0),
        "hidden_neg_delta_neg_weight_positive": bool(hidden.loc["neg_delta_neg_weight", "hnet"] > 0),
        "hidden_inactive_suppresses_delta": bool(
            max(
                abs(float(hidden.loc["inactive_pos_delta_pos_weight", "hdp_v"])),
                abs(float(hidden.loc["inactive_pos_delta_pos_weight", "hdn_v"])),
            )
            < 1e-3
        ),
        "max_error_dplus_v": float(error["dplus"].max()),
        "max_error_dminus_v": float(error["dminus"].max()),
        "max_hidden_hdp_v": float(hidden["hdp_v"].max()),
        "max_hidden_hdn_v": float(hidden["hdn_v"].max()),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This validates primitive capacitor-held output-error and hidden-delta sign paths. "
            "The error cell should be interpreted differentially: raw dplus/dminus branches compete, while dplus-dminus "
            "is the useful signed error quantity. "
            "It still is not a full network: no class readout, no multi-sample loop, and no MNIST training."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
