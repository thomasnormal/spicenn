from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from run_device_multicell_classifier import mos_models, pulse_wave
from run_device_xor2_learned_features import (
    BITS,
    CYCLE_NS,
    HIDDEN,
    OUTPUTS,
    hidden_caps,
    hidden_forward,
    sample_wave,
    target_wave,
    xor_label,
)
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def phases(samples: list[dict[str, Any]]) -> str:
    stop = len(samples) * CYCLE_NS
    rstf: list[tuple[float, float]] = []
    rstg: list[tuple[float, float]] = []
    fwd: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    bwd2: list[tuple[float, float]] = []
    bwd1: list[tuple[float, float]] = []
    acc: list[tuple[float, float]] = []
    apply: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        rstf.append((base + 0.00, base + 0.50))
        rstg.append((base + 0.00, base + 0.50))
        fwd.append((base + 0.75, base + 3.45))
        if sample["phase"] == "train":
            err.append((base + 3.70, base + 5.10))
            bwd2.append((base + 5.30, base + 6.70))
            bwd1.append((base + 6.90, base + 8.30))
            acc.append((base + 8.55, base + 10.05))
            apply.append((base + 10.30, base + 12.05))
            rstf.append((base + 12.35, base + 12.80))
            fwd.append((base + 13.00, base + 15.60))
    return "\n".join(
        [
            f"Vrstf rstf 0 {pulse_wave(rstf, stop)}",
            f"Vrstg rstg 0 {pulse_wave(rstg, stop)}",
            f"Vfwd fwd 0 {pulse_wave(fwd, stop)}",
            f"Verr err 0 {pulse_wave(err, stop)}",
            f"Vbwd2 bwd2 0 {pulse_wave(bwd2, stop)}",
            f"Vbwd1 bwd1 0 {pulse_wave(bwd1, stop)}",
            f"Vacc acc 0 {pulse_wave(acc, stop)}",
            f"Vapply apply 0 {pulse_wave(apply, stop)}",
        ]
    )


def middle_caps(diag_v: float, off_v: float, neg_v: float, cap_f: float) -> str:
    lines: list[str] = []
    for dst in range(HIDDEN):
        for src in range(HIDDEN):
            p = diag_v if dst == src else off_v
            n = neg_v
            lines += [
                f"Cwm{dst}{src}p wm{dst}{src}p 0 {cap_f:.12g}f IC={p:.12g}",
                f"Cwm{dst}{src}n wm{dst}{src}n 0 {cap_f:.12g}f IC={n:.12g}",
                f"Rwm{dst}{src}p wm{dst}{src}p 0 1e15",
                f"Rwm{dst}{src}n wm{dst}{src}n 0 1e15",
            ]
    return "\n".join(lines)


def readout_caps(high_v: float, low_v: float, cap_f: float) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            same = out == xor_label(h)
            p = high_v if same else low_v
            n = low_v if same else high_v
            lines += [
                f"Cvw{out}{h}p vw{out}{h}p 0 {cap_f:.12g}f IC={p:.12g}",
                f"Cvw{out}{h}n vw{out}{h}n 0 {cap_f:.12g}f IC={n:.12g}",
                f"Rvw{out}{h}p vw{out}{h}p 0 1e15",
                f"Rvw{out}{h}n vw{out}{h}n 0 1e15",
            ]
    return "\n".join(lines)


def temporary_caps(grad_cap_f: float) -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines += [
            f"Cpre{h} pre{h} 0 10f IC=0",
            f"Cact{h} act{h} 0 20f IC=0",
            f"Cpre2{h} pre2{h} 0 10f IC=0",
            f"Cact2{h} act2{h} 0 20f IC=0",
            f"Ch2dp{h} h2dp{h} 0 12f IC=0",
            f"Ch2dn{h} h2dn{h} 0 12f IC=0",
            f"Ch1dp{h} h1dp{h} 0 12f IC=0",
            f"Ch1dn{h} h1dn{h} 0 12f IC=0",
            f"Rpre{h} pre{h} 0 1G",
            f"Ract{h} act{h} 0 1G",
            f"Rpre2{h} pre2{h} 0 1G",
            f"Ract2{h} act2{h} 0 1G",
            f"Rh2dp{h} h2dp{h} 0 1G",
            f"Rh2dn{h} h2dn{h} 0 1G",
            f"Rh1dp{h} h1dp{h} 0 1G",
            f"Rh1dn{h} h1dn{h} 0 1G",
        ]
        for src in range(HIDDEN):
            lines += [
                f"Cgmp{h}{src} gmp{h}{src} 0 {grad_cap_f:.12g}f IC=0",
                f"Cgmn{h}{src} gmn{h}{src} 0 {grad_cap_f:.12g}f IC=0",
                f"Rgmp{h}{src} gmp{h}{src} 0 1G",
                f"Rgmn{h}{src} gmn{h}{src} 0 1G",
            ]
    for out in range(OUTPUTS):
        lines += [
            f"Cscore{out} score{out} 0 10f IC=0",
            f"Cout{out} out{out} 0 20f IC=0",
            f"Cdp{out} dp{out} 0 20f IC=0",
            f"Cdn{out} dn{out} 0 20f IC=0",
            f"Rscore{out} score{out} 0 1G",
            f"Rout{out} out{out} 0 1G",
            f"Rdp{out} dp{out} 0 1G",
            f"Rdn{out} dn{out} 0 1G",
        ]
        for h in range(HIDDEN):
            lines += [
                f"Cgvp{out}{h} gvp{out}{h} 0 {grad_cap_f:.12g}f IC=0",
                f"Cgvn{out}{h} gvn{out}{h} 0 {grad_cap_f:.12g}f IC=0",
                f"Rgvp{out}{h} gvp{out}{h} 0 1G",
                f"Rgvn{out}{h} gvn{out}{h} 0 1G",
            ]
    return "\n".join(lines)


def resets() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        for node in [
            f"pre{h}",
            f"act{h}",
            f"pre2{h}",
            f"act2{h}",
            f"h2dp{h}",
            f"h2dn{h}",
            f"h1dp{h}",
            f"h1dn{h}",
        ]:
            gate = "rstf" if node.startswith(("pre", "act")) else "rstg"
            lines.append(f"Mreset_{node} {node} {gate} 0 0 NMOS W=4u L=180n")
        for src in range(HIDDEN):
            lines += [
                f"Mreset_gmp{h}{src} gmp{h}{src} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_gmn{h}{src} gmn{h}{src} rstg 0 0 NMOS W=4u L=180n",
            ]
    for out in range(OUTPUTS):
        for node in [f"score{out}", f"out{out}", f"dp{out}", f"dn{out}"]:
            gate = "rstf" if node.startswith(("score", "out")) else "rstg"
            lines.append(f"Mreset_{node} {node} {gate} 0 0 NMOS W=4u L=180n")
        for h in range(HIDDEN):
            lines += [
                f"Mreset_gvp{out}{h} gvp{out}{h} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_gvn{out}{h} gvn{out}{h} rstg 0 0 NMOS W=4u L=180n",
            ]
    return "\n".join(lines)


def hidden2_forward() -> str:
    lines: list[str] = []
    for dst in range(HIDDEN):
        lines.append(f"* Hidden layer 2 cell {dst}: signed conductance from hidden layer 1.")
        for src in range(HIDDEN):
            lines += [
                f"Mm{dst}{src}pos_a vdd act{src} m{dst}{src}p0 0 NREL W=48u L=180n",
                f"Mm{dst}{src}pos_w m{dst}{src}p0 wm{dst}{src}p m{dst}{src}p1 0 NREL W=48u L=180n",
                f"Mm{dst}{src}pos_f m{dst}{src}p1 fwd pre2{dst} 0 NREL W=48u L=180n",
                f"Mm{dst}{src}neg_f pre2{dst} fwd m{dst}{src}n0 0 NREL W=36u L=180n",
                f"Mm{dst}{src}neg_a m{dst}{src}n0 act{src} m{dst}{src}n1 0 NREL W=36u L=180n",
                f"Mm{dst}{src}neg_w m{dst}{src}n1 wm{dst}{src}n 0 0 NREL W=36u L=180n",
            ]
        lines.append(f"Mrelu_h2_{dst} vdd pre2{dst} act2{dst} 0 NREL W=24u L=180n")
    return "\n".join(lines)


def output_forward(output_device: str, output_width_u: float) -> str:
    if output_device not in {"NREL", "NSENSE"}:
        raise ValueError(f"unknown output device: {output_device}")
    lines: list[str] = []
    for out in range(OUTPUTS):
        lines.append(f"* Output {out}: signed readout from hidden layer 2.")
        for h in range(HIDDEN):
            lines += [
                f"Mo{out}{h}pos_a vdd act2{h} o{out}{h}p0 0 NREL W=64u L=180n",
                f"Mo{out}{h}pos_w o{out}{h}p0 vw{out}{h}p o{out}{h}p1 0 NREL W=64u L=180n",
                f"Mo{out}{h}pos_f o{out}{h}p1 fwd score{out} 0 NREL W=64u L=180n",
                f"Mo{out}{h}neg_f score{out} fwd o{out}{h}n0 0 NREL W=48u L=180n",
                f"Mo{out}{h}neg_a o{out}{h}n0 act2{h} o{out}{h}n1 0 NREL W=48u L=180n",
                f"Mo{out}{h}neg_w o{out}{h}n1 vw{out}{h}n 0 0 NREL W=48u L=180n",
            ]
        lines.append(
            f"Mrelu_o{out} vdd score{out} out{out} 0 {output_device} W={output_width_u:.12g}u L=180n"
        )
    return "\n".join(lines)


def error_cells() -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        lines += [
            f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=32u L=180n",
            f"Mdp{out}_t1 dp{out}_t err dp{out} 0 NSENSE W=32u L=180n",
            f"Mdp{out}_y0 dp{out} err dp{out}_y 0 NSENSE W=24u L=180n",
            f"Mdp{out}_y1 dp{out}_y score{out} 0 0 NSENSE W=24u L=180n",
            f"Mdn{out}_y0 vdd score{out} dn{out}_y 0 NSENSE W=32u L=180n",
            f"Mdn{out}_y1 dn{out}_y err dn{out} 0 NSENSE W=32u L=180n",
            f"Mdn{out}_t0 dn{out} err dn{out}_t 0 NSENSE W=24u L=180n",
            f"Mdn{out}_t1 dn{out}_t t{out} 0 0 NSENSE W=24u L=180n",
        ]
    return "\n".join(lines)


def hidden2_delta() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines.append(f"* Hidden layer 2 delta {h}: output error through readout weights.")
        for out in range(OUTPUTS):
            lines += [
                f"Mh2dp{h}{out}a0 vdd dp{out} h2dp{h}{out}a0 0 NSENSE W=32u L=180n",
                f"Mh2dp{h}{out}a1 h2dp{h}{out}a0 vw{out}{h}p h2dp{h}{out}a1 0 NMOS W=32u L=180n",
                f"Mh2dp{h}{out}a2 h2dp{h}{out}a1 act2{h} h2dp{h}{out}a2 0 NREL W=32u L=180n",
                f"Mh2dp{h}{out}a3 h2dp{h}{out}a2 bwd2 h2dp{h} 0 NMOS W=32u L=180n",
                f"Mh2dp{h}{out}b0 vdd dn{out} h2dp{h}{out}b0 0 NSENSE W=32u L=180n",
                f"Mh2dp{h}{out}b1 h2dp{h}{out}b0 vw{out}{h}n h2dp{h}{out}b1 0 NMOS W=32u L=180n",
                f"Mh2dp{h}{out}b2 h2dp{h}{out}b1 act2{h} h2dp{h}{out}b2 0 NREL W=32u L=180n",
                f"Mh2dp{h}{out}b3 h2dp{h}{out}b2 bwd2 h2dp{h} 0 NMOS W=32u L=180n",
                f"Mh2dn{h}{out}a0 vdd dn{out} h2dn{h}{out}a0 0 NSENSE W=32u L=180n",
                f"Mh2dn{h}{out}a1 h2dn{h}{out}a0 vw{out}{h}p h2dn{h}{out}a1 0 NMOS W=32u L=180n",
                f"Mh2dn{h}{out}a2 h2dn{h}{out}a1 act2{h} h2dn{h}{out}a2 0 NREL W=32u L=180n",
                f"Mh2dn{h}{out}a3 h2dn{h}{out}a2 bwd2 h2dn{h} 0 NMOS W=32u L=180n",
                f"Mh2dn{h}{out}b0 vdd dp{out} h2dn{h}{out}b0 0 NSENSE W=32u L=180n",
                f"Mh2dn{h}{out}b1 h2dn{h}{out}b0 vw{out}{h}n h2dn{h}{out}b1 0 NMOS W=32u L=180n",
                f"Mh2dn{h}{out}b2 h2dn{h}{out}b1 act2{h} h2dn{h}{out}b2 0 NREL W=32u L=180n",
                f"Mh2dn{h}{out}b3 h2dn{h}{out}b2 bwd2 h2dn{h} 0 NMOS W=32u L=180n",
            ]
    return "\n".join(lines)


def hidden1_delta() -> str:
    lines: list[str] = []
    for src in range(HIDDEN):
        lines.append(f"* Hidden layer 1 delta {src}: hidden-2 error through middle weights.")
        for dst in range(HIDDEN):
            lines += [
                f"Mh1dp{src}{dst}a0 vdd h2dp{dst} h1dp{src}{dst}a0 0 NSENSE W=28u L=180n",
                f"Mh1dp{src}{dst}a1 h1dp{src}{dst}a0 wm{dst}{src}p h1dp{src}{dst}a1 0 NMOS W=28u L=180n",
                f"Mh1dp{src}{dst}a2 h1dp{src}{dst}a1 act{src} h1dp{src}{dst}a2 0 NREL W=28u L=180n",
                f"Mh1dp{src}{dst}a3 h1dp{src}{dst}a2 bwd1 h1dp{src} 0 NMOS W=28u L=180n",
                f"Mh1dp{src}{dst}b0 vdd h2dn{dst} h1dp{src}{dst}b0 0 NSENSE W=28u L=180n",
                f"Mh1dp{src}{dst}b1 h1dp{src}{dst}b0 wm{dst}{src}n h1dp{src}{dst}b1 0 NMOS W=28u L=180n",
                f"Mh1dp{src}{dst}b2 h1dp{src}{dst}b1 act{src} h1dp{src}{dst}b2 0 NREL W=28u L=180n",
                f"Mh1dp{src}{dst}b3 h1dp{src}{dst}b2 bwd1 h1dp{src} 0 NMOS W=28u L=180n",
                f"Mh1dn{src}{dst}a0 vdd h2dn{dst} h1dn{src}{dst}a0 0 NSENSE W=28u L=180n",
                f"Mh1dn{src}{dst}a1 h1dn{src}{dst}a0 wm{dst}{src}p h1dn{src}{dst}a1 0 NMOS W=28u L=180n",
                f"Mh1dn{src}{dst}a2 h1dn{src}{dst}a1 act{src} h1dn{src}{dst}a2 0 NREL W=28u L=180n",
                f"Mh1dn{src}{dst}a3 h1dn{src}{dst}a2 bwd1 h1dn{src} 0 NMOS W=28u L=180n",
                f"Mh1dn{src}{dst}b0 vdd h2dp{dst} h1dn{src}{dst}b0 0 NSENSE W=28u L=180n",
                f"Mh1dn{src}{dst}b1 h1dn{src}{dst}b0 wm{dst}{src}n h1dn{src}{dst}b1 0 NMOS W=28u L=180n",
                f"Mh1dn{src}{dst}b2 h1dn{src}{dst}b1 act{src} h1dn{src}{dst}b2 0 NREL W=28u L=180n",
                f"Mh1dn{src}{dst}b3 h1dn{src}{dst}b2 bwd1 h1dn{src} 0 NMOS W=28u L=180n",
            ]
    return "\n".join(lines)


def update_cells(readout_width_u: float, middle_width_u: float) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            lines += [
                f"Mgvp{out}{h}_a vdd act2{h} gvp{out}{h}_a 0 NREL W=24u L=180n",
                f"Mgvp{out}{h}_d gvp{out}{h}_a dp{out} gvp{out}{h}_d 0 NSENSE W=24u L=180n",
                f"Mgvp{out}{h}_g gvp{out}{h}_d acc gvp{out}{h} 0 NREL W=24u L=180n",
                f"Mgvn{out}{h}_a vdd act2{h} gvn{out}{h}_a 0 NREL W=24u L=180n",
                f"Mgvn{out}{h}_d gvn{out}{h}_a dn{out} gvn{out}{h}_d 0 NSENSE W=24u L=180n",
                f"Mgvn{out}{h}_g gvn{out}{h}_d acc gvn{out}{h} 0 NREL W=24u L=180n",
                f"Mvw{out}{h}p_up_g vdd gvp{out}{h} vw{out}{h}p_up 0 NSENSE W={readout_width_u:.12g}u L=180n",
                f"Mvw{out}{h}p_up_a vw{out}{h}p_up apply vw{out}{h}p 0 NREL W={readout_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_dn_a vw{out}{h}n apply vw{out}{h}n_dn 0 NREL W={readout_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_dn_g vw{out}{h}n_dn gvp{out}{h} 0 0 NSENSE W={readout_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_up_g vdd gvn{out}{h} vw{out}{h}n_up 0 NSENSE W={readout_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_up_a vw{out}{h}n_up apply vw{out}{h}n 0 NREL W={readout_width_u:.12g}u L=180n",
                f"Mvw{out}{h}p_dn_a vw{out}{h}p apply vw{out}{h}p_dn 0 NREL W={readout_width_u:.12g}u L=180n",
                f"Mvw{out}{h}p_dn_g vw{out}{h}p_dn gvn{out}{h} 0 0 NSENSE W={readout_width_u:.12g}u L=180n",
            ]
    for dst in range(HIDDEN):
        for src in range(HIDDEN):
            lines += [
                f"Mgmp{dst}{src}_a vdd act{src} gmp{dst}{src}_a 0 NREL W=24u L=180n",
                f"Mgmp{dst}{src}_d gmp{dst}{src}_a h2dp{dst} gmp{dst}{src}_d 0 NSENSE W=24u L=180n",
                f"Mgmp{dst}{src}_g gmp{dst}{src}_d acc gmp{dst}{src} 0 NREL W=24u L=180n",
                f"Mgmn{dst}{src}_a vdd act{src} gmn{dst}{src}_a 0 NREL W=24u L=180n",
                f"Mgmn{dst}{src}_d gmn{dst}{src}_a h2dn{dst} gmn{dst}{src}_d 0 NSENSE W=24u L=180n",
                f"Mgmn{dst}{src}_g gmn{dst}{src}_d acc gmn{dst}{src} 0 NREL W=24u L=180n",
                f"Mwm{dst}{src}p_up_g vdd gmp{dst}{src} wm{dst}{src}p_up 0 NSENSE W={middle_width_u:.12g}u L=180n",
                f"Mwm{dst}{src}p_up_a wm{dst}{src}p_up apply wm{dst}{src}p 0 NREL W={middle_width_u:.12g}u L=180n",
                f"Mwm{dst}{src}n_dn_a wm{dst}{src}n apply wm{dst}{src}n_dn 0 NREL W={middle_width_u:.12g}u L=180n",
                f"Mwm{dst}{src}n_dn_g wm{dst}{src}n_dn gmp{dst}{src} 0 0 NSENSE W={middle_width_u:.12g}u L=180n",
                f"Mwm{dst}{src}n_up_g vdd gmn{dst}{src} wm{dst}{src}n_up 0 NSENSE W={middle_width_u:.12g}u L=180n",
                f"Mwm{dst}{src}n_up_a wm{dst}{src}n_up apply wm{dst}{src}n 0 NREL W={middle_width_u:.12g}u L=180n",
                f"Mwm{dst}{src}p_dn_a wm{dst}{src}p apply wm{dst}{src}p_dn 0 NREL W={middle_width_u:.12g}u L=180n",
                f"Mwm{dst}{src}p_dn_g wm{dst}{src}p_dn gmn{dst}{src} 0 0 NSENSE W={middle_width_u:.12g}u L=180n",
            ]
    return "\n".join(lines)


def make_samples(epochs: int, order: list[int]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for pattern in range(4):
        samples.append({"phase": "initial_eval", "pattern": pattern, "label": xor_label(pattern)})
    for _ in range(epochs):
        for pattern in order:
            samples.append({"phase": "train", "pattern": pattern, "label": xor_label(pattern)})
    for pattern in range(4):
        samples.append({"phase": "final_eval", "pattern": pattern, "label": xor_label(pattern)})
    return samples


def measures(samples: list[dict[str, Any]]) -> tuple[str, str]:
    lines: list[str] = []
    prints: list[str] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        pattern = int(sample["pattern"])
        label = int(sample["label"])
        other = 1 - label
        at = base + 3.35
        lines += [
            f".meas tran target_out_{idx} FIND V(out{label}) AT={at:.2f}n",
            f".meas tran other_out_{idx} FIND V(out{other}) AT={at:.2f}n",
            f".meas tran margin_{idx} PARAM='target_out_{idx}-other_out_{idx}'",
            f".meas tran target_score_{idx} FIND V(score{label}) AT={at:.2f}n",
            f".meas tran other_score_{idx} FIND V(score{other}) AT={at:.2f}n",
            f".meas tran score_margin_{idx} PARAM='target_score_{idx}-other_score_{idx}'",
            f".meas tran active_h1_{idx} FIND V(act{pattern}) AT={at:.2f}n",
            f".meas tran active_h2_{idx} FIND V(act2{pattern}) AT={at:.2f}n",
        ]
        if sample["phase"] == "train":
            post_at = base + 15.50
            lines += [
                f".meas tran post_target_out_{idx} FIND V(out{label}) AT={post_at:.2f}n",
                f".meas tran post_other_out_{idx} FIND V(out{other}) AT={post_at:.2f}n",
                f".meas tran post_margin_{idx} PARAM='post_target_out_{idx}-post_other_out_{idx}'",
                f".meas tran post_target_score_{idx} FIND V(score{label}) AT={post_at:.2f}n",
                f".meas tran post_other_score_{idx} FIND V(score{other}) AT={post_at:.2f}n",
                f".meas tran post_score_margin_{idx} PARAM='post_target_score_{idx}-post_other_score_{idx}'",
                f".meas tran d_margin_{idx} PARAM='post_margin_{idx}-margin_{idx}'",
                f".meas tran d_score_margin_{idx} PARAM='post_score_margin_{idx}-score_margin_{idx}'",
                f".meas tran h2dp_active_{idx} FIND V(h2dp{pattern}) AT={base + 6.80:.2f}n",
                f".meas tran h2dn_active_{idx} FIND V(h2dn{pattern}) AT={base + 6.80:.2f}n",
                f".meas tran h1dp_active_{idx} FIND V(h1dp{pattern}) AT={base + 8.40:.2f}n",
                f".meas tran h1dn_active_{idx} FIND V(h1dn{pattern}) AT={base + 8.40:.2f}n",
                f".meas tran h2_delta_net_{idx} PARAM='h2dp_active_{idx}-h2dn_active_{idx}'",
                f".meas tran h1_delta_net_{idx} PARAM='h1dp_active_{idx}-h1dn_active_{idx}'",
            ]
        prints.append(f"print target_out_{idx} other_out_{idx} margin_{idx}")
    final_base = (len(samples) - 1) * CYCLE_NS
    for dst in range(HIDDEN):
        for src in range(HIDDEN):
            lines += [
                f".meas tran wm{dst}{src}p_initial FIND V(wm{dst}{src}p) AT=0.60n",
                f".meas tran wm{dst}{src}n_initial FIND V(wm{dst}{src}n) AT=0.60n",
                f".meas tran wm{dst}{src}p_final FIND V(wm{dst}{src}p) AT={final_base + 0.60:.2f}n",
                f".meas tran wm{dst}{src}n_final FIND V(wm{dst}{src}n) AT={final_base + 0.60:.2f}n",
                f".meas tran d_wm{dst}{src}_signed_total PARAM='(wm{dst}{src}p_final-wm{dst}{src}n_final)-(wm{dst}{src}p_initial-wm{dst}{src}n_initial)'",
            ]
    return "\n".join(lines), "\n".join(prints)


def xor_netlist(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    samples = make_samples(args.epochs, [0, 3, 1, 2])
    stop = len(samples) * CYCLE_NS
    meas, prints = measures(samples)
    return (
        f"""
* Device-level two-hidden-layer XOR architecture probe.
* Layer 1: fixed literal detectors. Layer 2 and readout use capacitor-held signed weights.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vx0 x0 0 {sample_wave(samples, "x0", stop)}
Vx1 x1 0 {sample_wave(samples, "x1", stop)}
Vnx0 nx0 0 {sample_wave(samples, "nx0", stop)}
Vnx1 nx1 0 {sample_wave(samples, "nx1", stop)}
Vt0 t0 0 {target_wave(samples, 0, stop)}
Vt1 t1 0 {target_wave(samples, 1, stop)}
{phases(samples)}

{hidden_caps()}
{middle_caps(args.middle_diag_v, args.middle_off_v, args.middle_neg_v, args.weight_cap_f)}
{readout_caps(args.readout_high_v, args.readout_low_v, args.weight_cap_f)}

{temporary_caps(args.gradient_cap_f)}
{resets()}

{hidden_forward()}
{hidden2_forward()}
{output_forward(args.output_device, args.output_width_u)}
{error_cells()}
{hidden2_delta()}
{hidden1_delta()}
{update_cells(args.readout_update_width_u, args.middle_update_width_u)}

.options method=gear maxord=2
.tran 10p {stop:.2f}n uic
{meas}
.control
run
{prints}
.endc
.end
""".lstrip(),
        samples,
    )


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    path.write_text(netlist)
    cmd = [spice_bin, "-b", str(path)] if "ngspice" in Path(spice_bin).name.lower() else [spice_bin, str(path)]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("ngspice produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--tag", default="device_xor2_two_hidden")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--middle-diag-v", type=float, default=0.82)
    ap.add_argument("--middle-off-v", type=float, default=0.08)
    ap.add_argument("--middle-neg-v", type=float, default=0.04)
    ap.add_argument("--readout-high-v", type=float, default=0.72)
    ap.add_argument("--readout-low-v", type=float, default=0.14)
    ap.add_argument("--weight-cap-f", type=float, default=20.0)
    ap.add_argument("--gradient-cap-f", type=float, default=8.0)
    ap.add_argument("--readout-update-width-u", type=float, default=7.0)
    ap.add_argument("--middle-update-width-u", type=float, default=4.0)
    ap.add_argument("--output-device", choices=["NREL", "NSENSE"], default="NREL")
    ap.add_argument("--output-width-u", type=float, default=24.0)
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    netlist, samples = xor_netlist(args)
    t0 = time.perf_counter()
    parsed = run_netlist(spice_bin, generated / f"{safe_tag}.cir", netlist, args.timeout)

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        phase = str(sample["phase"])
        row: dict[str, Any] = {
            "cycle": idx,
            "phase": phase,
            "pattern": int(sample["pattern"]),
            "label": int(sample["label"]),
            "target_out": parsed[f"target_out_{idx}"],
            "other_out": parsed[f"other_out_{idx}"],
            "margin": parsed[f"margin_{idx}"],
            "target_score": parsed[f"target_score_{idx}"],
            "other_score": parsed[f"other_score_{idx}"],
            "score_margin": parsed[f"score_margin_{idx}"],
            "active_h1": parsed[f"active_h1_{idx}"],
            "active_h2": parsed[f"active_h2_{idx}"],
            "out_correct": parsed[f"margin_{idx}"] > 0,
            "score_correct": parsed[f"score_margin_{idx}"] > 0,
            "correct": parsed[f"score_margin_{idx}"] > 0,
        }
        if phase == "train":
            row |= {
                "post_margin": parsed[f"post_margin_{idx}"],
                "d_margin": parsed[f"d_margin_{idx}"],
                "post_score_margin": parsed[f"post_score_margin_{idx}"],
                "d_score_margin": parsed[f"d_score_margin_{idx}"],
                "h2_delta_net": parsed[f"h2_delta_net_{idx}"],
                "h1_delta_net": parsed[f"h1_delta_net_{idx}"],
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    initial_eval = df[df["phase"] == "initial_eval"]
    final_eval = df[df["phase"] == "final_eval"]
    train = df[df["phase"] == "train"]
    middle_deltas = [
        parsed[f"d_wm{dst}{src}_signed_total"]
        for dst in range(HIDDEN)
        for src in range(HIDDEN)
    ]
    summary = {
        "simulator": version,
        "architecture": "device_level_2bit_xor_two_hidden_layers",
        "status": "two_hidden_architecture_probe",
        "benchmark": "2-bit XOR",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "Fixed literal detector layer -> trainable capacitor-held hidden layer 2 -> trainable readout. "
            "Output errors are transported to hidden layer 2 and then through middle weights to hidden layer 1."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_multipliers": False,
        "hidden_layers": 2,
        "trainable_hidden_layers": 1,
        "epochs": args.epochs,
        "initial_eval_accuracy": float(initial_eval["correct"].mean()),
        "final_eval_accuracy": float(final_eval["correct"].mean()),
        "initial_out_accuracy": float(initial_eval["out_correct"].mean()),
        "final_out_accuracy": float(final_eval["out_correct"].mean()),
        "initial_score_accuracy": float(initial_eval["score_correct"].mean()),
        "final_score_accuracy": float(final_eval["score_correct"].mean()),
        "initial_min_margin_v": float(initial_eval["margin"].min()),
        "final_min_margin_v": float(final_eval["margin"].min()),
        "initial_min_score_margin_v": float(initial_eval["score_margin"].min()),
        "final_min_score_margin_v": float(final_eval["score_margin"].min()),
        "mean_active_h2_initial_v": float(initial_eval["active_h2"].mean()),
        "mean_active_h2_final_v": float(final_eval["active_h2"].mean()),
        "all_train_h2_delta_nonzero": bool((train["h2_delta_net"].abs() > 1e-7).all()) if not train.empty else False,
        "all_train_h1_delta_nonzero": bool((train["h1_delta_net"].abs() > 1e-7).all()) if not train.empty else False,
        "mean_train_margin_change_v": float(train["d_margin"].mean()) if not train.empty else 0.0,
        "mean_train_score_margin_change_v": float(train["d_score_margin"].mean()) if not train.empty else 0.0,
        "max_abs_total_middle_signed_delta_v": float(max(abs(x) for x in middle_deltas)),
        "output_device": args.output_device,
        "output_width_u": args.output_width_u,
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This is not yet a free feature-discovery MLP: layer 1 is a fixed device feature bank. "
            "It tests whether a second trainable hidden layer can carry forward activations, backward deltas, "
            "and local signed updates without Python computing those quantities."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
