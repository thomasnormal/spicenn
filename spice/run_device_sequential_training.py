from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test
from _util import MEAS_RE, parse_measures


try:
    from spicenn.timing import CYCLE_NS
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    from spicenn.timing import CYCLE_NS


def mos_models() -> str:
    return """
.model NMOS NMOS LEVEL=1 VTO=0.35 KP=220u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
.model NSENSE NMOS LEVEL=1 VTO=0.02 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
""".strip()


def pwl(points: list[tuple[float, float]]) -> str:
    return "PWL(" + " ".join(f"{t:.12g}n {v:.12g}" for t, v in points) + ")"


def pulse_wave(pulses: list[tuple[float, float]], stop_ns: float, high: float = 1.2) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for start, end in pulses:
        if start <= 0.0:
            points.append((0.0, high))
        else:
            points.append((max(0.0, start - 0.05), 0.0))
            points.append((start, high))
        points.append((end, high))
        points.append((min(stop_ns, end + 0.05), 0.0))
    points.append((stop_ns, 0.0))

    compact: list[tuple[float, float]] = []
    for t, v in sorted(points):
        if compact and abs(compact[-1][0] - t) < 1e-12:
            compact[-1] = (t, v)
        else:
            compact.append((t, v))
    return pwl(compact)


def active_low_pulse_wave(pulses: list[tuple[float, float]], stop_ns: float, high: float = 1.2) -> str:
    points: list[tuple[float, float]] = [(0.0, high)]
    for start, end in pulses:
        if start > 0.0:
            points.append((max(0.0, start - 0.05), high))
        points.append((start, 0.0))
        points.append((end, 0.0))
        points.append((min(stop_ns, end + 0.05), high))
    points.append((stop_ns, high))
    compact: list[tuple[float, float]] = []
    for t, v in sorted(points):
        if compact and abs(compact[-1][0] - t) < 1e-12:
            compact[-1] = (t, v)
        else:
            compact.append((t, v))
    return pwl(compact)


def sample_wave(samples: list[dict[str, float]], key: str, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = float(sample[key])
        if idx == 0:
            points.append((0.0, value))
        else:
            points.append((start - 0.05, float(samples[idx - 1][key])))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.05), value))
    points.append((stop_ns, float(samples[-1][key])))
    return pwl(points)


def repeated_phases(sample_count: int, *, training_enabled: bool = True) -> str:
    stop = sample_count * CYCLE_NS
    rstf: list[tuple[float, float]] = []
    rstg: list[tuple[float, float]] = []
    fwd: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    bwd: list[tuple[float, float]] = []
    acc: list[tuple[float, float]] = []
    apply: list[tuple[float, float]] = []
    for idx in range(sample_count):
        base = idx * CYCLE_NS
        rstf += [(base + 0.00, base + 0.50), (base + 12.05, base + 12.55)]
        rstg += [(base + 0.00, base + 0.50)]
        fwd += [(base + 0.75, base + 3.00), (base + 12.80, base + 15.60)]
        if training_enabled:
            err.append((base + 3.25, base + 5.00))
            bwd.append((base + 5.25, base + 7.00))
            acc.append((base + 7.25, base + 9.00))
            apply.append((base + 9.25, base + 11.20))
    return "\n".join(
        [
            f"Vrstf rstf 0 {pulse_wave(rstf, stop)}",
            f"Vrstg rstg 0 {pulse_wave(rstg, stop)}",
            f"Vfwd fwd 0 {pulse_wave(fwd, stop)}",
            f"Verr err 0 {pulse_wave(err, stop)}",
            f"Vbwd bwd 0 {pulse_wave(bwd, stop)}",
            f"Vacc acc 0 {pulse_wave(acc, stop)}",
            f"Vapply apply 0 {pulse_wave(apply, stop)}",
            f"Vapplyn applyn 0 {active_low_pulse_wave(apply, stop)}",
        ]
    )


def hidden_delta_block(hidden_credit_mode: str) -> str:
    if hidden_credit_mode == "exact_backprop":
        return """
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
""".strip()
    if hidden_credit_mode == "direct_feedback":
        return """
* Hidden delta: fixed direct feedback from output delta and hidden activation.
* This is the transistor-only DFA-like path: it avoids using stale readout sign
* as the hidden credit sign during rapid online recovery.
Mhdp_d0 vdd dp hdp_d0 0 NSENSE W=32u L=180n
Mhdp_d1 hdp_d0 act hdp_d1 0 NREL W=32u L=180n
Mhdp_d2 hdp_d1 bwd hdp 0 NMOS W=32u L=180n
Mhdn_d0 vdd dn hdn_d0 0 NSENSE W=32u L=180n
Mhdn_d1 hdn_d0 act hdn_d1 0 NREL W=32u L=180n
Mhdn_d2 hdn_d1 bwd hdn 0 NMOS W=32u L=180n
""".strip()
    raise ValueError(f"unknown hidden credit mode: {hidden_credit_mode}")


def output_driver_line(output_driver_model: str) -> str:
    if output_driver_model == "nrel":
        return "Mrelu_o vdd score out 0 NREL W=24u L=180n"
    if output_driver_model == "sense":
        return "Mrelu_o vdd score out 0 NSENSE W=24u L=180n"
    raise ValueError(f"unknown output driver model: {output_driver_model}")


def sequential_netlist(
    samples: list[dict[str, float]],
    whp: float,
    whn: float,
    vwp: float,
    vwn: float,
    hidden_credit_mode: str = "direct_feedback",
    training_enabled: bool = True,
    output_driver_model: str = "sense",
) -> str:
    stop = len(samples) * CYCLE_NS
    measures: list[str] = []
    prints: list[str] = []
    for idx in range(len(samples)):
        base = idx * CYCLE_NS
        measures += [
            f".meas tran whp_before_{idx} FIND V(whp) AT={base + 0.60:.2f}n",
            f".meas tran whn_before_{idx} FIND V(whn) AT={base + 0.60:.2f}n",
            f".meas tran vwp_before_{idx} FIND V(vwp) AT={base + 0.60:.2f}n",
            f".meas tran vwn_before_{idx} FIND V(vwn) AT={base + 0.60:.2f}n",
            f".meas tran act_before_{idx} FIND V(act) AT={base + 2.95:.2f}n",
            f".meas tran score_before_{idx} FIND V(score) AT={base + 2.95:.2f}n",
            f".meas tran out_before_{idx} FIND V(out) AT={base + 2.95:.2f}n",
            f".meas tran score_error_{idx} FIND V(score) AT={base + 4.25:.2f}n",
            f".meas tran dp_after_{idx} FIND V(dp) AT={base + 5.10:.2f}n",
            f".meas tran dn_after_{idx} FIND V(dn) AT={base + 5.10:.2f}n",
            f".meas tran hdp_after_{idx} FIND V(hdp) AT={base + 7.10:.2f}n",
            f".meas tran hdn_after_{idx} FIND V(hdn) AT={base + 7.10:.2f}n",
            f".meas tran gvp_after_{idx} FIND V(gvp) AT={base + 9.10:.2f}n",
            f".meas tran gvn_after_{idx} FIND V(gvn) AT={base + 9.10:.2f}n",
            f".meas tran ghp_after_{idx} FIND V(ghp) AT={base + 9.10:.2f}n",
            f".meas tran ghn_after_{idx} FIND V(ghn) AT={base + 9.10:.2f}n",
            f".meas tran whp_after_apply_{idx} FIND V(whp) AT={base + 11.50:.2f}n",
            f".meas tran whn_after_apply_{idx} FIND V(whn) AT={base + 11.50:.2f}n",
            f".meas tran vwp_after_apply_{idx} FIND V(vwp) AT={base + 11.50:.2f}n",
            f".meas tran vwn_after_apply_{idx} FIND V(vwn) AT={base + 11.50:.2f}n",
            f".meas tran act_after_{idx} FIND V(act) AT={base + 15.50:.2f}n",
            f".meas tran out_after_{idx} FIND V(out) AT={base + 15.50:.2f}n",
            f".meas tran hidden_signed_before_{idx} PARAM='whp_before_{idx}-whn_before_{idx}'",
            f".meas tran readout_signed_before_{idx} PARAM='vwp_before_{idx}-vwn_before_{idx}'",
            f".meas tran hidden_signed_after_{idx} PARAM='whp_after_apply_{idx}-whn_after_apply_{idx}'",
            f".meas tran readout_signed_after_{idx} PARAM='vwp_after_apply_{idx}-vwn_after_apply_{idx}'",
            f".meas tran d_hidden_signed_{idx} PARAM='hidden_signed_after_{idx}-hidden_signed_before_{idx}'",
            f".meas tran d_readout_signed_{idx} PARAM='readout_signed_after_{idx}-readout_signed_before_{idx}'",
            f".meas tran d_out_{idx} PARAM='out_after_{idx}-out_before_{idx}'",
            f".meas tran error_net_{idx} PARAM='dp_after_{idx}-dn_after_{idx}'",
            f".meas tran hidden_delta_net_{idx} PARAM='hdp_after_{idx}-hdn_after_{idx}'",
        ]
        prints += [
            f"print out_before_{idx} out_after_{idx} error_net_{idx} d_readout_signed_{idx} d_hidden_signed_{idx}",
            f"print whp_after_apply_{idx} whn_after_apply_{idx} vwp_after_apply_{idx} vwn_after_apply_{idx}",
        ]

    return f"""
* Sequential device-level training: one hidden ReLU/readout loop, repeated samples.
* Persistent Cwh/Cvw weight capacitors are not reset between training samples.
* The signal/training path uses MOSFET stacks, capacitors, resistors, and phase sources.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vin x 0 {sample_wave(samples, "vin", stop)}
Vtarget target 0 {sample_wave(samples, "target", stop)}
{repeated_phases(len(samples), training_enabled=training_enabled)}

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
Cgvp gvp 0 2f IC=0
Cgvn gvn 0 2f IC=0
Cghp ghp 0 10f IC=0
Cghn ghn 0 10f IC=0
Crgp rgp 0 4f IC=1.2
Crgn rgn 0 4f IC=1.2
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
Rrgp rgp vdd 50k
Rrgn rgn vdd 50k

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
{output_driver_line(output_driver_model)}

* Output error: dplus/dminus from target/raw-score conductance competition.
Mdp_t0 vdd target dp_t 0 NSENSE W=32u L=180n
Mdp_t1 dp_t err dp 0 NSENSE W=32u L=180n
Mdp_y0 dp err dp_y 0 NSENSE W=24u L=180n
Mdp_y1 dp_y score 0 0 NSENSE W=24u L=180n
Mdn_y0 vdd score dn_y 0 NSENSE W=32u L=180n
Mdn_y1 dn_y err dn 0 NSENSE W=32u L=180n
Mdn_t0 dn err dn_t 0 NSENSE W=24u L=180n
Mdn_t1 dn_t target 0 0 NSENSE W=24u L=180n

{hidden_delta_block(hidden_credit_mode)}

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
Mrgp_pd rgp gvp 0 0 NSENSE W=16u L=180n
Mrgn_pd rgn gvn 0 0 NSENSE W=16u L=180n
Mvwp_up_p0 vdd rgp vwp_up vdd PMOS W=8u L=180n
Mvwp_up_p1 vwp_up applyn vwp vdd PMOS W=8u L=180n
Mvwn_dn_a vwn apply vwn_dn 0 NREL W=2u L=180n
Mvwn_dn_g vwn_dn gvp 0 0 NSENSE W=2u L=180n
Mvwn_up_p0 vdd rgn vwn_up vdd PMOS W=8u L=180n
Mvwn_up_p1 vwn_up applyn vwn vdd PMOS W=8u L=180n
Mvwp_dn_a vwp apply vwp_dn 0 NREL W=2u L=180n
Mvwp_dn_g vwp_dn gvn 0 0 NSENSE W=2u L=180n

* Apply hidden positive/negative gradients.
Mwhp_up_g vdd ghp whp_up 0 NSENSE W=0.5u L=180n
Mwhp_up_a whp_up apply whp 0 NREL W=0.5u L=180n
Mwhn_dn_a whn apply whn_dn 0 NREL W=0.5u L=180n
Mwhn_dn_g whn_dn ghp 0 0 NSENSE W=0.5u L=180n
Mwhn_up_g vdd ghn whn_up 0 NSENSE W=0.5u L=180n
Mwhn_up_a whn_up apply whn 0 NREL W=0.5u L=180n
Mwhp_dn_a whp apply whp_dn 0 NREL W=0.5u L=180n
Mwhp_dn_g whp_dn ghn 0 0 NSENSE W=0.5u L=180n

.options method=gear maxord=2
.tran 10p {stop:.2f}n uic
{chr(10).join(measures)}
.control
run
{chr(10).join(prints)}
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        stdout_tail = proc.stdout[-3000:] if proc.stdout else "<empty>"
        stderr_tail = proc.stderr[-3000:] if proc.stderr else "<empty>"
        raise RuntimeError(
            f"SPICE failed for {path} with return code {proc.returncode}.\n"
            f"--- stdout tail ---\n{stdout_tail}\n"
            f"--- stderr tail ---\n{stderr_tail}"
        )
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return measures


def expected_positive(target: float) -> bool:
    return target > 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--tag", default="device_sequential_training")
    ap.add_argument(
        "--hidden-credit-mode",
        choices=["direct_feedback", "exact_backprop"],
        default="direct_feedback",
        help=(
            "Hidden credit circuit. direct_feedback is a transistor-only fixed-feedback/DFA path; "
            "exact_backprop multiplies output error by the current readout weight sign."
        ),
    )
    ap.add_argument(
        "--output-driver-model",
        choices=["sense", "nrel"],
        default="sense",
        help="Transistor source-follower model used for the final output/sense node.",
    )
    ap.add_argument(
        "--assert-pass",
        action="store_true",
        help="Exit nonzero unless the sequential transistor smoke passes all readout, output, and hidden polarity gates.",
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
    sequences = [
        {
            "sequence": "high_then_low",
            "initial": {"whp": 0.85, "whn": 0.25, "vwp": 0.55, "vwn": 0.25},
            "samples": [{"vin": 1.0, "target": 1.1}, {"vin": 1.0, "target": 0.0}],
        },
        {
            "sequence": "low_then_high",
            "initial": {"whp": 0.95, "whn": 0.25, "vwp": 1.10, "vwn": 0.20},
            "samples": [{"vin": 1.0, "target": 0.0}, {"vin": 1.0, "target": 1.1}],
        },
    ]

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for seq_idx, sequence in enumerate(sequences):
        initial = sequence["initial"]
        samples = sequence["samples"]
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_{seq_idx:03d}_{sequence['sequence']}.cir",
            sequential_netlist(
                samples,
                float(initial["whp"]),
                float(initial["whn"]),
                float(initial["vwp"]),
                float(initial["vwn"]),
                hidden_credit_mode=args.hidden_credit_mode,
                output_driver_model=args.output_driver_model,
            ),
            args.timeout,
        )
        for sample_idx, sample in enumerate(samples):
            positive = expected_positive(float(sample["target"]))
            row: dict[str, Any] = {
                "sequence": sequence["sequence"],
                "sample_idx": sample_idx,
                "vin": sample["vin"],
                "target": sample["target"],
                "expected_direction": "positive" if positive else "negative",
                "readout_weight_polarity_pass": (
                    measures[f"error_net_{sample_idx}"] > 0
                    and measures[f"d_readout_signed_{sample_idx}"] > 0
                    if positive
                    else measures[f"error_net_{sample_idx}"] < 0
                    and measures[f"d_readout_signed_{sample_idx}"] < 0
                ),
                "output_polarity_pass": (
                    measures[f"d_out_{sample_idx}"] > 0 if positive else measures[f"d_out_{sample_idx}"] < 0
                ),
                "polarity_pass": (
                    measures[f"error_net_{sample_idx}"] > 0
                    and measures[f"d_readout_signed_{sample_idx}"] > 0
                    and measures[f"d_out_{sample_idx}"] > 0
                    if positive
                    else measures[f"error_net_{sample_idx}"] < 0
                    and measures[f"d_readout_signed_{sample_idx}"] < 0
                    and measures[f"d_out_{sample_idx}"] < 0
                ),
                "hidden_polarity_pass": (
                    measures[f"d_hidden_signed_{sample_idx}"] > 0
                    if positive
                    else measures[f"d_hidden_signed_{sample_idx}"] < 0
                ),
            }
            for key, value in measures.items():
                suffix = f"_{sample_idx}"
                if key.endswith(suffix):
                    row[key[: -len(suffix)]] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    readout_weight_polarity_pass = bool(df["readout_weight_polarity_pass"].all())
    output_polarity_pass = bool(df["output_polarity_pass"].all())
    polarity_pass = bool(df["polarity_pass"].all())
    hidden_polarity_pass = bool(df["hidden_polarity_pass"].all())
    device_training_smoke_pass = readout_weight_polarity_pass and output_polarity_pass and hidden_polarity_pass
    summary = {
        "simulator": version,
        "architecture": "device_level_repeated_sample_training_loop",
        "status": "sequential_device_training_smoke",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "hidden_credit_mode": args.hidden_credit_mode,
        "output_driver_model": args.output_driver_model,
        "learning_device_implementation": "transistor_passive",
        "signal_path": (
            "A single SPICE transient repeats forward/error/backward/accumulate/apply guide phases. "
            "Hidden and readout weight capacitors persist inside the deck between samples; only temporary activation, "
            "error, delta, and gradient caps are reset."
        ),
        "hidden_credit_interpretation": (
            "The direct-feedback mode is a fixed physical feedback/DFA-style hidden credit path. "
            "It is not exact backprop, but it avoids letting a stale wrong-sign readout rail invert the next hidden update."
            if args.hidden_credit_mode == "direct_feedback"
            else "The exact-backprop mode computes hidden credit through the current readout rail signs."
        ),
        "no_behavioral_signal_math": True,
        "no_behavioral_learning_devices": True,
        "uses_behavioral_learning_devices": False,
        "transistor_or_passive_learning_path": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "single_training_transient": True,
        "python_weight_updates_between_samples": False,
        "python_checkpointing_between_samples": False,
        "weight_caps_persist_inside_single_spice_transient": True,
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "sequences": len(sequences),
        "samples_per_sequence": len(sequences[0]["samples"]),
        "rows": len(df),
        "all_device_training_smoke_polarities_pass": device_training_smoke_pass,
        "all_error_readout_weight_polarities_pass": readout_weight_polarity_pass,
        "all_output_polarities_pass": output_polarity_pass,
        "all_error_readout_output_polarities_pass": polarity_pass,
        "all_hidden_update_polarities_pass": hidden_polarity_pass,
        "high_then_low_final_readout_signed": float(
            df[(df["sequence"] == "high_then_low") & (df["sample_idx"] == 1)]["readout_signed_after"].iloc[0]
        ),
        "low_then_high_final_readout_signed": float(
            df[(df["sequence"] == "low_then_high") & (df["sample_idx"] == 1)]["readout_signed_after"].iloc[0]
        ),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This moves beyond isolated one-sample netlists: the same transistor-level loop trains on two sequential samples "
            "with persistent weight capacitors in one ngspice transient. It is still a tiny binary/regression smoke test, "
            "not a multi-class MNIST network."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.assert_pass and not device_training_smoke_pass:
        raise SystemExit("sequential transistor training smoke failed polarity gates")


if __name__ == "__main__":
    main()
