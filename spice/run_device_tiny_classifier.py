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
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
.model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
""".strip()


def pwl(points: list[tuple[float, float]]) -> str:
    return "PWL(" + " ".join(f"{t:.12g}n {v:.12g}" for t, v in points) + ")"


def phases() -> str:
    vdd = 1.2
    rstf = [(0.0, vdd), (0.50, vdd), (0.55, 0.0), (12.00, 0.0), (12.05, vdd), (12.55, vdd), (12.60, 0.0), (16.0, 0.0)]
    rstg = [(0.0, vdd), (0.50, vdd), (0.55, 0.0), (16.0, 0.0)]
    fwd = [(0.0, 0.0), (0.70, 0.0), (0.75, vdd), (3.00, vdd), (3.05, 0.0), (12.75, 0.0), (12.80, vdd), (15.60, vdd), (15.65, 0.0), (16.0, 0.0)]
    err = [(0.0, 0.0), (3.20, 0.0), (3.25, vdd), (5.00, vdd), (5.05, 0.0), (16.0, 0.0)]
    bwd = [(0.0, 0.0), (5.20, 0.0), (5.25, vdd), (7.00, vdd), (7.05, 0.0), (16.0, 0.0)]
    acc = [(0.0, 0.0), (7.20, 0.0), (7.25, vdd), (9.00, vdd), (9.05, 0.0), (16.0, 0.0)]
    apply = [(0.0, 0.0), (9.20, 0.0), (9.25, vdd), (11.20, vdd), (11.25, 0.0), (16.0, 0.0)]
    return "\n".join(
        [
            f"Vrstf rstf 0 {pwl(rstf)}",
            f"Vrstg rstg 0 {pwl(rstg)}",
            f"Vfwd fwd 0 {pwl(fwd)}",
            f"Verr err 0 {pwl(err)}",
            f"Vbwd bwd 0 {pwl(bwd)}",
            f"Vacc acc 0 {pwl(acc)}",
            f"Vapply apply 0 {pwl(apply)}",
        ]
    )


def tiny_classifier_netlist(
    vin: float,
    target: float,
    whp: float,
    whn: float,
    vwp: float,
    vwn: float,
) -> str:
    return f"""
* Tiny device-level classifier: one input, one hidden ReLU, one signed readout.
* The signal/training path uses MOSFET stacks, capacitors, resistors, and phase sources.
* No behavioral tanh, subtraction, or multiplication appears in the signal path.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vin x 0 {vin:.12g}
Vtarget target 0 {target:.12g}
{phases()}

* Persistent hidden and readout signed weights.
Cwhp whp 0 20f IC={whp:.12g}
Cwhn whn 0 20f IC={whn:.12g}
Cvwp vwp 0 20f IC={vwp:.12g}
Cvwn vwn 0 20f IC={vwn:.12g}
Rwhp whp 0 1e15
Rwhn whn 0 1e15
Rvwp vwp 0 1e15
Rvwn vwn 0 1e15

* Temporary state and gradient capacitors.
Cpre pre 0 10f IC=0
Cact act 0 20f IC=0
Cscore score 0 10f IC=0
Cout out 0 20f IC=0
Cdp dp 0 20f IC=0
Cdn dn 0 20f IC=0
Chdp hdp 0 12f IC=0
Chdn hdn 0 12f IC=0
Cgvp gvp 0 20f IC=0
Cgvn gvn 0 20f IC=0
Cghp ghp 0 10f IC=0
Cghn ghn 0 10f IC=0
Rpre pre 0 1G
Ract act 0 1G
Rscore score 0 1G
Rout out 0 1G
Rdp dp 0 1G
Rdn dn 0 1G
Rhdp hdp 0 1G
Rhdn hdn 0 1G
Rgvp gvp 0 1G
Rgvn gvn 0 1G
Rghp ghp 0 1G
Rghn ghn 0 1G

* Reset only nonpersistent state.
Mreset_pre pre rstf 0 0 NMOS W=4u L=180n
Mreset_act act rstf 0 0 NMOS W=4u L=180n
Mreset_score score rstf 0 0 NMOS W=4u L=180n
Mreset_out out rstf 0 0 NMOS W=4u L=180n
Mreset_dp dp rstg 0 0 NMOS W=4u L=180n
Mreset_dn dn rstg 0 0 NMOS W=4u L=180n
Mreset_hdp hdp rstg 0 0 NMOS W=4u L=180n
Mreset_hdn hdn rstg 0 0 NMOS W=4u L=180n
Mreset_gvp gvp rstg 0 0 NMOS W=4u L=180n
Mreset_gvn gvn rstg 0 0 NMOS W=4u L=180n
Mreset_ghp ghp rstg 0 0 NMOS W=4u L=180n
Mreset_ghn ghn rstg 0 0 NMOS W=4u L=180n

* Hidden forward: signed conductance into Cpre followed by ReLU/source follower onto Cact.
Mhpos_x vdd x hp0 0 NMOS W=32u L=180n
Mhpos_w hp0 whp hp1 0 NMOS W=32u L=180n
Mhpos_f hp1 fwd pre 0 NMOS W=32u L=180n
Mhneg_f pre fwd hn0 0 NMOS W=24u L=180n
Mhneg_x hn0 x hn1 0 NMOS W=24u L=180n
Mhneg_w hn1 whn 0 0 NMOS W=24u L=180n
Mrelu_h vdd pre act 0 NREL W=24u L=180n

* Output forward: signed readout conductance into Cscore followed by ReLU/source follower onto Cout.
Movpos_a vdd act op0 0 NREL W=64u L=180n
Movpos_w op0 vwp op1 0 NREL W=64u L=180n
Movpos_f op1 fwd score 0 NREL W=64u L=180n
Movneg_f score fwd on0 0 NREL W=48u L=180n
Movneg_a on0 act on1 0 NREL W=48u L=180n
Movneg_w on1 vwn 0 0 NREL W=48u L=180n
Mrelu_o vdd score out 0 NREL W=24u L=180n

* Output error: dplus/dminus from target/raw-score conductance competition.
Mdp_t0 vdd target dp_t 0 NSENSE W=32u L=180n
Mdp_t1 dp_t err dp 0 NSENSE W=32u L=180n
Mdp_y0 dp err dp_y 0 NSENSE W=24u L=180n
Mdp_y1 dp_y score 0 0 NSENSE W=24u L=180n
Mdn_y0 vdd score dn_y 0 NSENSE W=32u L=180n
Mdn_y1 dn_y err dn 0 NSENSE W=32u L=180n
Mdn_t0 dn err dn_t 0 NSENSE W=24u L=180n
Mdn_t1 dn_t target 0 0 NSENSE W=24u L=180n

* Hidden delta: sign combinations of output delta, readout weight, and hidden activation.
Mhdp_a0 vdd dp hdp_a0 0 NSENSE W=32u L=180n
Mhdp_a1 hdp_a0 vwp hdp_a1 0 NMOS W=32u L=180n
Mhdp_a2 hdp_a1 act hdp_a2 0 NREL W=32u L=180n
Mhdp_a3 hdp_a2 bwd hdp 0 NMOS W=32u L=180n
Mhdp_b0 vdd dn hdp_b0 0 NSENSE W=32u L=180n
Mhdp_b1 hdp_b0 vwn hdp_b1 0 NMOS W=32u L=180n
Mhdp_b2 hdp_b1 act hdp_b2 0 NREL W=32u L=180n
Mhdp_b3 hdp_b2 bwd hdp 0 NMOS W=32u L=180n
Mhdn_a0 vdd dn hdn_a0 0 NSENSE W=32u L=180n
Mhdn_a1 hdn_a0 vwp hdn_a1 0 NMOS W=32u L=180n
Mhdn_a2 hdn_a1 act hdn_a2 0 NREL W=32u L=180n
Mhdn_a3 hdn_a2 bwd hdn 0 NMOS W=32u L=180n
Mhdn_b0 vdd dp hdn_b0 0 NSENSE W=32u L=180n
Mhdn_b1 hdn_b0 vwn hdn_b1 0 NMOS W=32u L=180n
Mhdn_b2 hdn_b1 act hdn_b2 0 NREL W=32u L=180n
Mhdn_b3 hdn_b2 bwd hdn 0 NMOS W=32u L=180n

* Readout gradient accumulators: hidden activation times output delta.
Mgvp_a vdd act gvp_a 0 NREL W=24u L=180n
Mgvp_d gvp_a dp gvp_d 0 NSENSE W=24u L=180n
Mgvp_g gvp_d acc gvp 0 NREL W=24u L=180n
Mgvn_a vdd act gvn_a 0 NREL W=24u L=180n
Mgvn_d gvn_a dn gvn_d 0 NSENSE W=24u L=180n
Mgvn_g gvn_d acc gvn 0 NREL W=24u L=180n

* Hidden gradient accumulators: input times hidden delta.
Mghp_x vdd x ghp_x 0 NMOS W=48u L=180n
Mghp_d ghp_x hdp ghp_d 0 NSENSE W=48u L=180n
Mghp_g ghp_d acc ghp 0 NMOS W=48u L=180n
Mghn_x vdd x ghn_x 0 NMOS W=48u L=180n
Mghn_d ghn_x hdn ghn_d 0 NSENSE W=48u L=180n
Mghn_g ghn_d acc ghn 0 NMOS W=48u L=180n

* Apply readout positive/negative gradients.
Mvwp_up_g vdd gvp vwp_up 0 NSENSE W=8u L=180n
Mvwp_up_a vwp_up apply vwp 0 NREL W=8u L=180n
Mvwn_dn_a vwn apply vwn_dn 0 NREL W=8u L=180n
Mvwn_dn_g vwn_dn gvp 0 0 NSENSE W=8u L=180n
Mvwn_up_g vdd gvn vwn_up 0 NSENSE W=8u L=180n
Mvwn_up_a vwn_up apply vwn 0 NREL W=8u L=180n
Mvwp_dn_a vwp apply vwp_dn 0 NREL W=8u L=180n
Mvwp_dn_g vwp_dn gvn 0 0 NSENSE W=8u L=180n

* Apply hidden positive/negative gradients.
Mwhp_up_g vdd ghp whp_up 0 NSENSE W=8u L=180n
Mwhp_up_a whp_up apply whp 0 NREL W=8u L=180n
Mwhn_dn_a whn apply whn_dn 0 NREL W=8u L=180n
Mwhn_dn_g whn_dn ghp 0 0 NSENSE W=8u L=180n
Mwhn_up_g vdd ghn whn_up 0 NSENSE W=8u L=180n
Mwhn_up_a whn_up apply whn 0 NREL W=8u L=180n
Mwhp_dn_a whp apply whp_dn 0 NREL W=8u L=180n
Mwhp_dn_g whp_dn ghn 0 0 NSENSE W=8u L=180n

.options method=gear maxord=2
.tran 10p 16n uic
.meas tran act_before FIND V(act) AT=2.95n
.meas tran score_before FIND V(score) AT=2.95n
.meas tran out_before FIND V(out) AT=2.95n
.meas tran score_error FIND V(score) AT=4.25n
.meas tran out_error FIND V(out) AT=4.25n
.meas tran dp_after FIND V(dp) AT=5.10n
.meas tran dn_after FIND V(dn) AT=5.10n
.meas tran hdp_after FIND V(hdp) AT=7.10n
.meas tran hdn_after FIND V(hdn) AT=7.10n
.meas tran gvp_after FIND V(gvp) AT=9.10n
.meas tran gvn_after FIND V(gvn) AT=9.10n
.meas tran ghp_after FIND V(ghp) AT=9.10n
.meas tran ghn_after FIND V(ghn) AT=9.10n
.meas tran whp_after_apply FIND V(whp) AT=11.50n
.meas tran whn_after_apply FIND V(whn) AT=11.50n
.meas tran vwp_after_apply FIND V(vwp) AT=11.50n
.meas tran vwn_after_apply FIND V(vwn) AT=11.50n
.meas tran act_after FIND V(act) AT=15.50n
.meas tran out_after FIND V(out) AT=15.50n
.meas tran hidden_signed_before PARAM='{whp:.12g}-{whn:.12g}'
.meas tran readout_signed_before PARAM='{vwp:.12g}-{vwn:.12g}'
.meas tran hidden_signed_after PARAM='whp_after_apply-whn_after_apply'
.meas tran readout_signed_after PARAM='vwp_after_apply-vwn_after_apply'
.meas tran d_hidden_signed PARAM='hidden_signed_after-hidden_signed_before'
.meas tran d_readout_signed PARAM='readout_signed_after-readout_signed_before'
.meas tran d_out PARAM='out_after-out_before'
.meas tran error_net PARAM='dp_after-dn_after'
.meas tran hidden_delta_net PARAM='hdp_after-hdn_after'
.control
run
print act_before out_before dp_after dn_after hdp_after hdn_after
print gvp_after gvn_after ghp_after ghn_after
print whp_after_apply whn_after_apply vwp_after_apply vwn_after_apply
print act_after out_after hidden_signed_before readout_signed_before
print hidden_signed_after readout_signed_after d_hidden_signed d_readout_signed d_out error_net hidden_delta_net
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return measures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--tag", default="device_tiny_classifier")
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    t0 = time.perf_counter()
    cases = [
        {
            "case": "high_target_should_increase",
            "vin": 1.0,
            "target": 1.1,
            "whp": 0.85,
            "whn": 0.25,
            "vwp": 0.55,
            "vwn": 0.25,
        },
        {
            "case": "low_target_should_decrease",
            "vin": 1.0,
            "target": 0.0,
            "whp": 0.95,
            "whn": 0.25,
            "vwp": 1.1,
            "vwn": 0.2,
        },
    ]
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_{idx:03d}_{case['case']}.cir",
            tiny_classifier_netlist(
                float(case["vin"]),
                float(case["target"]),
                float(case["whp"]),
                float(case["whn"]),
                float(case["vwp"]),
                float(case["vwn"]),
            ),
            args.timeout,
        )
        rows.append({**case, **measures})

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)
    by_case = df.set_index("case")
    high = by_case.loc["high_target_should_increase"]
    low = by_case.loc["low_target_should_decrease"]
    summary = {
        "simulator": version,
        "architecture": "device_level_tiny_forward_backward_update_classifier",
        "status": "primitive_end_to_end_device_learning_smoke",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "One input drives one signed hidden ReLU cell, one signed readout ReLU output, output-error caps, hidden-delta caps, "
            "readout and hidden gradient caps, and apply-phase weight updates. No behavioral tanh, subtraction, or multiplication "
            "appears in the signal path."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "rows": len(df),
        "high_target_error_positive": bool(high["error_net"] > 0),
        "high_target_readout_update_positive": bool(high["d_readout_signed"] > 0),
        "high_target_hidden_update_positive": bool(high["d_hidden_signed"] > 0),
        "high_target_output_increased": bool(high["d_out"] > 0),
        "low_target_error_negative": bool(low["error_net"] < 0),
        "low_target_readout_update_negative": bool(low["d_readout_signed"] < 0),
        "low_target_hidden_update_negative": bool(low["d_hidden_signed"] < 0),
        "low_target_output_decreased": bool(low["d_out"] < 0),
        "high_target_out_before": float(high["out_before"]),
        "high_target_out_after": float(high["out_after"]),
        "low_target_out_before": float(low["out_before"]),
        "low_target_out_after": float(low["out_after"]),
        "high_target_d_readout_signed": float(high["d_readout_signed"]),
        "low_target_d_readout_signed": float(low["d_readout_signed"]),
        "high_target_d_hidden_signed": float(high["d_hidden_signed"]),
        "low_target_d_hidden_signed": float(low["d_hidden_signed"]),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This is the first tiny end-to-end device-level forward/backward/update smoke test. It proves neither scale nor MNIST accuracy, "
            "but it connects the primitive capacitor states into one trainable loop."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
