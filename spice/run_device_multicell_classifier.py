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


# Timing primitives have moved to ``spicenn.timing``; re-exported here so the
# existing ``from run_device_multicell_classifier import pulse_wave`` imports
# in other run scripts keep working.  Direct ``python spice/foo.py`` execution
# puts ``spice/`` rather than the repository root on ``sys.path``, so add the
# root before importing the package.
try:
    from spicenn.timing import CYCLE_NS, VDD, compact_pwl_points, pulse_wave, pwl
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spicenn.timing import CYCLE_NS, VDD, compact_pwl_points, pulse_wave, pwl


def mos_models() -> str:
    return """
.model NMOS NMOS LEVEL=1 VTO=0.35 KP=220u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
.model NREL NMOS LEVEL=1 VTO=0.12 KP=260u LAMBDA=0.03 GAMMA=0.10 PHI=0.60
.model NSENSE NMOS LEVEL=1 VTO=0.03 KP=260u LAMBDA=0.03 GAMMA=0.05 PHI=0.60
.model PMOS PMOS LEVEL=1 VTO=-0.35 KP=90u LAMBDA=0.03 GAMMA=0.20 PHI=0.60
""".strip()


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


def repeated_phases(sample_count: int) -> str:
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
        ]
    )


def hidden_forward(h: int) -> str:
    return f"""
* Hidden {h}: one local input synapse into Cpre{h}, then ReLU/source follower.
Mh{h}pos_x vdd x{h} h{h}p0 0 NMOS W=32u L=180n
Mh{h}pos_w h{h}p0 wh{h}p h{h}p1 0 NMOS W=32u L=180n
Mh{h}pos_f h{h}p1 fwd pre{h} 0 NMOS W=32u L=180n
Mh{h}neg_f pre{h} fwd h{h}n0 0 NMOS W=24u L=180n
Mh{h}neg_x h{h}n0 x{h} h{h}n1 0 NMOS W=24u L=180n
Mh{h}neg_w h{h}n1 wh{h}n 0 0 NMOS W=24u L=180n
Mrelu_h{h} vdd pre{h} act{h} 0 NREL W=24u L=180n
""".strip()


def output_forward(j: int) -> str:
    lines = [f"* Output {j}: signed readout from both hidden activations into Cscore{j}."]
    for h in range(2):
        lines += [
            f"Mo{j}{h}pos_a vdd act{h} o{j}{h}p0 0 NREL W=64u L=180n",
            f"Mo{j}{h}pos_w o{j}{h}p0 vw{j}{h}p o{j}{h}p1 0 NREL W=64u L=180n",
            f"Mo{j}{h}pos_f o{j}{h}p1 fwd score{j} 0 NREL W=64u L=180n",
            f"Mo{j}{h}neg_f score{j} fwd o{j}{h}n0 0 NREL W=48u L=180n",
            f"Mo{j}{h}neg_a o{j}{h}n0 act{h} o{j}{h}n1 0 NREL W=48u L=180n",
            f"Mo{j}{h}neg_w o{j}{h}n1 vw{j}{h}n 0 0 NREL W=48u L=180n",
        ]
    lines.append(f"Mrelu_o{j} vdd score{j} out{j} 0 NREL W=24u L=180n")
    return "\n".join(lines)


def error_cell(j: int) -> str:
    return f"""
* Output {j} error: dplus/dminus from target/raw-score conductance competition.
Mdp{j}_t0 vdd t{j} dp{j}_t 0 NSENSE W=32u L=180n
Mdp{j}_t1 dp{j}_t err dp{j} 0 NSENSE W=32u L=180n
Mdp{j}_y0 dp{j} err dp{j}_y 0 NSENSE W=24u L=180n
Mdp{j}_y1 dp{j}_y score{j} 0 0 NSENSE W=24u L=180n
Mdn{j}_y0 vdd score{j} dn{j}_y 0 NSENSE W=32u L=180n
Mdn{j}_y1 dn{j}_y err dn{j} 0 NSENSE W=32u L=180n
Mdn{j}_t0 dn{j} err dn{j}_t 0 NSENSE W=24u L=180n
Mdn{j}_t1 dn{j}_t t{j} 0 0 NSENSE W=24u L=180n
""".strip()


def hidden_delta(h: int) -> str:
    lines = [f"* Hidden {h} delta sums both output deltas through readout-weight sign stacks."]
    for j in range(2):
        lines += [
            f"Mhdp{h}{j}a0 vdd dp{j} hdp{h}{j}a0 0 NSENSE W=32u L=180n",
            f"Mhdp{h}{j}a1 hdp{h}{j}a0 vw{j}{h}p hdp{h}{j}a1 0 NMOS W=32u L=180n",
            f"Mhdp{h}{j}a2 hdp{h}{j}a1 act{h} hdp{h}{j}a2 0 NREL W=32u L=180n",
            f"Mhdp{h}{j}a3 hdp{h}{j}a2 bwd hdp{h} 0 NMOS W=32u L=180n",
            f"Mhdp{h}{j}b0 vdd dn{j} hdp{h}{j}b0 0 NSENSE W=32u L=180n",
            f"Mhdp{h}{j}b1 hdp{h}{j}b0 vw{j}{h}n hdp{h}{j}b1 0 NMOS W=32u L=180n",
            f"Mhdp{h}{j}b2 hdp{h}{j}b1 act{h} hdp{h}{j}b2 0 NREL W=32u L=180n",
            f"Mhdp{h}{j}b3 hdp{h}{j}b2 bwd hdp{h} 0 NMOS W=32u L=180n",
            f"Mhdn{h}{j}a0 vdd dn{j} hdn{h}{j}a0 0 NSENSE W=32u L=180n",
            f"Mhdn{h}{j}a1 hdn{h}{j}a0 vw{j}{h}p hdn{h}{j}a1 0 NMOS W=32u L=180n",
            f"Mhdn{h}{j}a2 hdn{h}{j}a1 act{h} hdn{h}{j}a2 0 NREL W=32u L=180n",
            f"Mhdn{h}{j}a3 hdn{h}{j}a2 bwd hdn{h} 0 NMOS W=32u L=180n",
            f"Mhdn{h}{j}b0 vdd dp{j} hdn{h}{j}b0 0 NSENSE W=32u L=180n",
            f"Mhdn{h}{j}b1 hdn{h}{j}b0 vw{j}{h}n hdn{h}{j}b1 0 NMOS W=32u L=180n",
            f"Mhdn{h}{j}b2 hdn{h}{j}b1 act{h} hdn{h}{j}b2 0 NREL W=32u L=180n",
            f"Mhdn{h}{j}b3 hdn{h}{j}b2 bwd hdn{h} 0 NMOS W=32u L=180n",
        ]
    return "\n".join(lines)


def readout_gradients_and_updates() -> str:
    lines: list[str] = []
    for j in range(2):
        for h in range(2):
            lines += [
                f"* Readout gradient/update for output {j}, hidden {h}.",
                f"Mgvp{j}{h}_a vdd act{h} gvp{j}{h}_a 0 NREL W=24u L=180n",
                f"Mgvp{j}{h}_d gvp{j}{h}_a dp{j} gvp{j}{h}_d 0 NSENSE W=24u L=180n",
                f"Mgvp{j}{h}_g gvp{j}{h}_d acc gvp{j}{h} 0 NREL W=24u L=180n",
                f"Mgvn{j}{h}_a vdd act{h} gvn{j}{h}_a 0 NREL W=24u L=180n",
                f"Mgvn{j}{h}_d gvn{j}{h}_a dn{j} gvn{j}{h}_d 0 NSENSE W=24u L=180n",
                f"Mgvn{j}{h}_g gvn{j}{h}_d acc gvn{j}{h} 0 NREL W=24u L=180n",
                f"Mvw{j}{h}p_up_g vdd gvp{j}{h} vw{j}{h}p_up 0 NSENSE W=10u L=180n",
                f"Mvw{j}{h}p_up_a vw{j}{h}p_up apply vw{j}{h}p 0 NREL W=10u L=180n",
                f"Mvw{j}{h}n_dn_a vw{j}{h}n apply vw{j}{h}n_dn 0 NREL W=10u L=180n",
                f"Mvw{j}{h}n_dn_g vw{j}{h}n_dn gvp{j}{h} 0 0 NSENSE W=10u L=180n",
                f"Mvw{j}{h}n_up_g vdd gvn{j}{h} vw{j}{h}n_up 0 NSENSE W=10u L=180n",
                f"Mvw{j}{h}n_up_a vw{j}{h}n_up apply vw{j}{h}n 0 NREL W=10u L=180n",
                f"Mvw{j}{h}p_dn_a vw{j}{h}p apply vw{j}{h}p_dn 0 NREL W=10u L=180n",
                f"Mvw{j}{h}p_dn_g vw{j}{h}p_dn gvn{j}{h} 0 0 NSENSE W=10u L=180n",
            ]
    return "\n".join(lines)


def hidden_gradients_and_updates() -> str:
    lines: list[str] = []
    for h in range(2):
        lines += [
            f"* Hidden gradient/update for hidden {h}.",
            f"Mghp{h}_x vdd x{h} ghp{h}_x 0 NMOS W=48u L=180n",
            f"Mghp{h}_d ghp{h}_x hdp{h} ghp{h}_d 0 NSENSE W=48u L=180n",
            f"Mghp{h}_g ghp{h}_d acc ghp{h} 0 NMOS W=48u L=180n",
            f"Mghn{h}_x vdd x{h} ghn{h}_x 0 NMOS W=48u L=180n",
            f"Mghn{h}_d ghn{h}_x hdn{h} ghn{h}_d 0 NSENSE W=48u L=180n",
            f"Mghn{h}_g ghn{h}_d acc ghn{h} 0 NMOS W=48u L=180n",
            f"Mwh{h}p_up_g vdd ghp{h} wh{h}p_up 0 NSENSE W=8u L=180n",
            f"Mwh{h}p_up_a wh{h}p_up apply wh{h}p 0 NREL W=8u L=180n",
            f"Mwh{h}n_dn_a wh{h}n apply wh{h}n_dn 0 NREL W=8u L=180n",
            f"Mwh{h}n_dn_g wh{h}n_dn ghp{h} 0 0 NSENSE W=8u L=180n",
            f"Mwh{h}n_up_g vdd ghn{h} wh{h}n_up 0 NSENSE W=8u L=180n",
            f"Mwh{h}n_up_a wh{h}n_up apply wh{h}n 0 NREL W=8u L=180n",
            f"Mwh{h}p_dn_a wh{h}p apply wh{h}p_dn 0 NREL W=8u L=180n",
            f"Mwh{h}p_dn_g wh{h}p_dn ghn{h} 0 0 NSENSE W=8u L=180n",
        ]
    return "\n".join(lines)


def persistent_caps(hidden_init: dict[str, float], readout_init: dict[str, float]) -> str:
    lines: list[str] = []
    for h in range(2):
        lines += [
            f"Cwh{h}p wh{h}p 0 20f IC={hidden_init[f'wh{h}p']:.12g}",
            f"Cwh{h}n wh{h}n 0 20f IC={hidden_init[f'wh{h}n']:.12g}",
            f"Rwh{h}p wh{h}p 0 1e15",
            f"Rwh{h}n wh{h}n 0 1e15",
        ]
    for j in range(2):
        for h in range(2):
            lines += [
                f"Cvw{j}{h}p vw{j}{h}p 0 20f IC={readout_init[f'vw{j}{h}p']:.12g}",
                f"Cvw{j}{h}n vw{j}{h}n 0 20f IC={readout_init[f'vw{j}{h}n']:.12g}",
                f"Rvw{j}{h}p vw{j}{h}p 0 1e15",
                f"Rvw{j}{h}n vw{j}{h}n 0 1e15",
            ]
    return "\n".join(lines)


def temporary_caps() -> str:
    lines: list[str] = []
    for h in range(2):
        lines += [
            f"Cpre{h} pre{h} 0 10f IC=0",
            f"Cact{h} act{h} 0 20f IC=0",
            f"Chdp{h} hdp{h} 0 12f IC=0",
            f"Chdn{h} hdn{h} 0 12f IC=0",
            f"Cghp{h} ghp{h} 0 10f IC=0",
            f"Cghn{h} ghn{h} 0 10f IC=0",
            f"Rpre{h} pre{h} 0 1G",
            f"Ract{h} act{h} 0 1G",
            f"Rhdp{h} hdp{h} 0 1G",
            f"Rhdn{h} hdn{h} 0 1G",
            f"Rghp{h} ghp{h} 0 1G",
            f"Rghn{h} ghn{h} 0 1G",
        ]
    for j in range(2):
        lines += [
            f"Cscore{j} score{j} 0 10f IC=0",
            f"Cout{j} out{j} 0 20f IC=0",
            f"Cdp{j} dp{j} 0 20f IC=0",
            f"Cdn{j} dn{j} 0 20f IC=0",
            f"Rscore{j} score{j} 0 1G",
            f"Rout{j} out{j} 0 1G",
            f"Rdp{j} dp{j} 0 1G",
            f"Rdn{j} dn{j} 0 1G",
        ]
        for h in range(2):
            lines += [
                f"Cgvp{j}{h} gvp{j}{h} 0 20f IC=0",
                f"Cgvn{j}{h} gvn{j}{h} 0 20f IC=0",
                f"Rgvp{j}{h} gvp{j}{h} 0 1G",
                f"Rgvn{j}{h} gvn{j}{h} 0 1G",
            ]
    return "\n".join(lines)


def resets() -> str:
    lines: list[str] = []
    for h in range(2):
        lines += [
            f"Mreset_pre{h} pre{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_act{h} act{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_hdp{h} hdp{h} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_hdn{h} hdn{h} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_ghp{h} ghp{h} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_ghn{h} ghn{h} rstg 0 0 NMOS W=4u L=180n",
        ]
    for j in range(2):
        lines += [
            f"Mreset_score{j} score{j} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_out{j} out{j} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_dp{j} dp{j} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_dn{j} dn{j} rstg 0 0 NMOS W=4u L=180n",
        ]
        for h in range(2):
            lines += [
                f"Mreset_gvp{j}{h} gvp{j}{h} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_gvn{j}{h} gvn{j}{h} rstg 0 0 NMOS W=4u L=180n",
            ]
    return "\n".join(lines)


def measure_lines(samples: list[dict[str, float]]) -> tuple[str, str]:
    measures: list[str] = []
    prints: list[str] = []
    for idx in range(len(samples)):
        base = idx * CYCLE_NS
        for h in range(2):
            measures += [
                f".meas tran wh{h}p_before_{idx} FIND V(wh{h}p) AT={base + 0.60:.2f}n",
                f".meas tran wh{h}n_before_{idx} FIND V(wh{h}n) AT={base + 0.60:.2f}n",
                f".meas tran act{h}_before_{idx} FIND V(act{h}) AT={base + 2.95:.2f}n",
                f".meas tran hdp{h}_after_{idx} FIND V(hdp{h}) AT={base + 7.10:.2f}n",
                f".meas tran hdn{h}_after_{idx} FIND V(hdn{h}) AT={base + 7.10:.2f}n",
                f".meas tran ghp{h}_after_{idx} FIND V(ghp{h}) AT={base + 9.10:.2f}n",
                f".meas tran ghn{h}_after_{idx} FIND V(ghn{h}) AT={base + 9.10:.2f}n",
                f".meas tran wh{h}p_after_apply_{idx} FIND V(wh{h}p) AT={base + 11.50:.2f}n",
                f".meas tran wh{h}n_after_apply_{idx} FIND V(wh{h}n) AT={base + 11.50:.2f}n",
                f".meas tran hidden_signed_before_{h}_{idx} PARAM='wh{h}p_before_{idx}-wh{h}n_before_{idx}'",
                f".meas tran hidden_signed_after_{h}_{idx} PARAM='wh{h}p_after_apply_{idx}-wh{h}n_after_apply_{idx}'",
                f".meas tran d_hidden_signed_{h}_{idx} PARAM='hidden_signed_after_{h}_{idx}-hidden_signed_before_{h}_{idx}'",
                f".meas tran hidden_delta_net_{h}_{idx} PARAM='hdp{h}_after_{idx}-hdn{h}_after_{idx}'",
            ]
        for j in range(2):
            measures += [
                f".meas tran score{j}_before_{idx} FIND V(score{j}) AT={base + 2.95:.2f}n",
                f".meas tran out{j}_before_{idx} FIND V(out{j}) AT={base + 2.95:.2f}n",
                f".meas tran score{j}_error_{idx} FIND V(score{j}) AT={base + 4.25:.2f}n",
                f".meas tran dp{j}_after_{idx} FIND V(dp{j}) AT={base + 5.10:.2f}n",
                f".meas tran dn{j}_after_{idx} FIND V(dn{j}) AT={base + 5.10:.2f}n",
                f".meas tran out{j}_after_{idx} FIND V(out{j}) AT={base + 15.50:.2f}n",
                f".meas tran error_net_{j}_{idx} PARAM='dp{j}_after_{idx}-dn{j}_after_{idx}'",
                f".meas tran d_out_{j}_{idx} PARAM='out{j}_after_{idx}-out{j}_before_{idx}'",
            ]
            for h in range(2):
                measures += [
                    f".meas tran vw{j}{h}p_before_{idx} FIND V(vw{j}{h}p) AT={base + 0.60:.2f}n",
                    f".meas tran vw{j}{h}n_before_{idx} FIND V(vw{j}{h}n) AT={base + 0.60:.2f}n",
                    f".meas tran gvp{j}{h}_after_{idx} FIND V(gvp{j}{h}) AT={base + 9.10:.2f}n",
                    f".meas tran gvn{j}{h}_after_{idx} FIND V(gvn{j}{h}) AT={base + 9.10:.2f}n",
                    f".meas tran vw{j}{h}p_after_apply_{idx} FIND V(vw{j}{h}p) AT={base + 11.50:.2f}n",
                    f".meas tran vw{j}{h}n_after_apply_{idx} FIND V(vw{j}{h}n) AT={base + 11.50:.2f}n",
                    f".meas tran readout_signed_before_{j}_{h}_{idx} PARAM='vw{j}{h}p_before_{idx}-vw{j}{h}n_before_{idx}'",
                    f".meas tran readout_signed_after_{j}_{h}_{idx} PARAM='vw{j}{h}p_after_apply_{idx}-vw{j}{h}n_after_apply_{idx}'",
                    f".meas tran d_readout_signed_{j}_{h}_{idx} PARAM='readout_signed_after_{j}_{h}_{idx}-readout_signed_before_{j}_{h}_{idx}'",
                ]
        measures += [
            f".meas tran margin0_before_{idx} PARAM='out0_before_{idx}-out1_before_{idx}'",
            f".meas tran margin1_before_{idx} PARAM='out1_before_{idx}-out0_before_{idx}'",
            f".meas tran margin0_after_{idx} PARAM='out0_after_{idx}-out1_after_{idx}'",
            f".meas tran margin1_after_{idx} PARAM='out1_after_{idx}-out0_after_{idx}'",
            f".meas tran d_margin0_{idx} PARAM='margin0_after_{idx}-margin0_before_{idx}'",
            f".meas tran d_margin1_{idx} PARAM='margin1_after_{idx}-margin1_before_{idx}'",
        ]
        prints += [
            f"print out0_before_{idx} out1_before_{idx} out0_after_{idx} out1_after_{idx}",
            f"print d_margin0_{idx} d_margin1_{idx} error_net_0_{idx} error_net_1_{idx}",
        ]
    return "\n".join(measures), "\n".join(prints)


def multicell_netlist(
    samples: list[dict[str, float]],
    hidden_init: dict[str, float],
    readout_init: dict[str, float],
) -> str:
    stop = len(samples) * CYCLE_NS
    measures, prints = measure_lines(samples)
    return f"""
* Device-level 2-input/2-hidden/2-output classifier.
* Guide waveforms sequence forward, error, backward, accumulate, and update phases.
* Persistent hidden/readout weight capacitors remain charged across samples.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vx0 x0 0 {sample_wave(samples, "x0", stop)}
Vx1 x1 0 {sample_wave(samples, "x1", stop)}
Vt0 t0 0 {sample_wave(samples, "t0", stop)}
Vt1 t1 0 {sample_wave(samples, "t1", stop)}
{repeated_phases(len(samples))}

* Persistent signed weights.
{persistent_caps(hidden_init, readout_init)}

* Temporary neuron, output, error, delta, and gradient storage.
{temporary_caps()}

* Nonpersistent-state resets.
{resets()}

{hidden_forward(0)}
{hidden_forward(1)}

{output_forward(0)}
{output_forward(1)}

{error_cell(0)}
{error_cell(1)}

{hidden_delta(0)}
{hidden_delta(1)}

{readout_gradients_and_updates()}

{hidden_gradients_and_updates()}

.options method=gear maxord=2
.tran 10p {stop:.2f}n uic
{measures}
.control
run
{prints}
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


def row_for_sample(sequence: str, idx: int, sample: dict[str, float], measures: dict[str, float]) -> dict[str, Any]:
    target = int(sample["label"])
    other = 1 - target
    active = 0 if float(sample["x0"]) > float(sample["x1"]) else 1
    target_out_before = measures[f"out{target}_before_{idx}"]
    target_out_after = measures[f"out{target}_after_{idx}"]
    other_out_before = measures[f"out{other}_before_{idx}"]
    other_out_after = measures[f"out{other}_after_{idx}"]
    d_margin = measures[f"d_margin{target}_{idx}"]
    d_target_readout = measures[f"d_readout_signed_{target}_{active}_{idx}"]
    d_other_readout = measures[f"d_readout_signed_{other}_{active}_{idx}"]
    d_active_hidden = measures[f"d_hidden_signed_{active}_{idx}"]
    active_hidden_delta = measures[f"hidden_delta_net_{active}_{idx}"]
    inactive_hidden_delta = measures[f"hidden_delta_net_{1 - active}_{idx}"]
    return {
        "sequence": sequence,
        "sample_idx": idx,
        "label": target,
        "active_hidden": active,
        "x0": sample["x0"],
        "x1": sample["x1"],
        "t0": sample["t0"],
        "t1": sample["t1"],
        "target_out_before": target_out_before,
        "target_out_after": target_out_after,
        "non_target_out_before": other_out_before,
        "non_target_out_after": other_out_after,
        "d_target_out": target_out_after - target_out_before,
        "d_non_target_out": other_out_after - other_out_before,
        "margin_before": measures[f"margin{target}_before_{idx}"],
        "margin_after": measures[f"margin{target}_after_{idx}"],
        "d_margin": d_margin,
        "target_error_net": measures[f"error_net_{target}_{idx}"],
        "non_target_error_net": measures[f"error_net_{other}_{idx}"],
        "d_target_active_readout": d_target_readout,
        "d_non_target_active_readout": d_other_readout,
        "d_active_hidden_signed": d_active_hidden,
        "active_hidden_delta_net": active_hidden_delta,
        "inactive_hidden_delta_net": inactive_hidden_delta,
        "target_output_increased": target_out_after > target_out_before,
        "non_target_output_decreased": other_out_after < other_out_before,
        "margin_improved": d_margin > 0,
        "target_error_positive": measures[f"error_net_{target}_{idx}"] > 0,
        "non_target_error_negative": measures[f"error_net_{other}_{idx}"] < 0,
        "target_active_readout_increased": d_target_readout > 0,
        "non_target_active_readout_decreased": d_other_readout < 0,
        "active_hidden_delta_nonzero": abs(active_hidden_delta) > 1e-6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--tag", default="device_multicell_classifier")
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    hidden_init = {"wh0p": 0.95, "wh0n": 0.25, "wh1p": 0.95, "wh1n": 0.25}
    readout_init = {
        f"vw{j}{h}{pol}": 1.10 if pol == "p" else 0.20
        for j in range(2)
        for h in range(2)
        for pol in ["p", "n"]
    }
    sequences = [
        {
            "sequence": "class0_then_class1",
            "samples": [
                {"x0": 1.0, "x1": 0.0, "t0": 1.1, "t1": 0.0, "label": 0},
                {"x0": 0.0, "x1": 1.0, "t0": 0.0, "t1": 1.1, "label": 1},
            ],
        },
        {
            "sequence": "class1_then_class0",
            "samples": [
                {"x0": 0.0, "x1": 1.0, "t0": 0.0, "t1": 1.1, "label": 1},
                {"x0": 1.0, "x1": 0.0, "t0": 1.1, "t1": 0.0, "label": 0},
            ],
        },
    ]

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for seq_idx, sequence in enumerate(sequences):
        samples = sequence["samples"]
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_{seq_idx:03d}_{sequence['sequence']}.cir",
            multicell_netlist(samples, hidden_init, readout_init),
            args.timeout,
        )
        for sample_idx, sample in enumerate(samples):
            rows.append(row_for_sample(sequence["sequence"], sample_idx, sample, measures))

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    summary = {
        "simulator": version,
        "architecture": "device_level_2input_2hidden_2output_classifier",
        "status": "multicell_multiclass_device_training_smoke",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "Two input voltage rails drive two signed hidden ReLU cells. Two output cells read both hidden activations, "
            "produce one-hot output-error caps, backpropagate hidden deltas through readout-weight sign stacks, accumulate "
            "readout/hidden gradients on capacitors, and apply signed weight-cap updates. Python only generates netlists "
            "and parses measurements."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "weight_caps_persist_inside_single_spice_transient": True,
        "batching_supported": False,
        "batching_design_note": (
            "Current device harnesses are sequential sample-by-sample. If analog batching is added later, "
            "pre-activation storage should account for batch-size scaling, while persistent weight capacitors, "
            "gradient accumulator state, and optimizer/Adam state should not be scaled just because the batch has more samples."
        ),
        "inputs": 2,
        "hidden_cells": 2,
        "outputs": 2,
        "sequences": len(sequences),
        "samples_per_sequence": len(sequences[0]["samples"]),
        "rows": len(df),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "all_target_outputs_increase": bool(df["target_output_increased"].all()),
        "all_non_target_outputs_decrease": bool(df["non_target_output_decreased"].all()),
        "all_margins_improve": bool(df["margin_improved"].all()),
        "all_target_errors_positive": bool(df["target_error_positive"].all()),
        "all_non_target_errors_negative": bool(df["non_target_error_negative"].all()),
        "all_target_active_readouts_increase": bool(df["target_active_readout_increased"].all()),
        "all_non_target_active_readouts_decrease": bool(df["non_target_active_readout_decreased"].all()),
        "all_active_hidden_deltas_nonzero": bool(df["active_hidden_delta_nonzero"].all()),
        "min_margin_improvement_v": float(df["d_margin"].min()),
        "max_margin_improvement_v": float(df["d_margin"].max()),
        "mean_margin_improvement_v": float(df["d_margin"].mean()),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This is the first device-level multi-cell, multi-output classifier smoke test. It demonstrates one-hot class "
            "competition and per-class readout updates in a single SPICE transient with persistent weight capacitors. "
            "It is still a tiny two-class synthetic task, not MNIST."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
