from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from run_device_multicell_classifier import mos_models, pulse_wave, pwl
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")
CYCLE_NS = 16.0
BITS = 2
HIDDEN = 4
OUTPUTS = 2


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def bit_value(pattern: int, bit: int) -> int:
    return (pattern >> bit) & 1


def xor_label(pattern: int) -> int:
    return bit_value(pattern, 0) ^ bit_value(pattern, 1)


def input_value(pattern: int, node: str) -> float:
    if node.startswith("nx"):
        return float(1 - bit_value(pattern, int(node[2:])))
    return float(bit_value(pattern, int(node[1:])))


def sample_wave(samples: list[dict[str, Any]], node: str, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = input_value(int(sample["pattern"]), node)
        if idx == 0:
            points.append((0.0, value))
        else:
            points.append((start - 0.05, input_value(int(samples[idx - 1]["pattern"]), node)))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.05), value))
    points.append((stop_ns, input_value(int(samples[-1]["pattern"]), node)))
    return pwl(points)


def target_wave(samples: list[dict[str, Any]], output: int, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = 1.1 if int(sample["label"]) == output else 0.0
        if idx == 0:
            points.append((0.0, value))
        else:
            prev = 1.1 if int(samples[idx - 1]["label"]) == output else 0.0
            points.append((start - 0.05, prev))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.05), value))
    points.append((stop_ns, 1.1 if int(samples[-1]["label"]) == output else 0.0))
    return pwl(points)


def phases(train_count: int, total_count: int) -> str:
    stop = total_count * CYCLE_NS
    rstf: list[tuple[float, float]] = []
    rstg: list[tuple[float, float]] = []
    fwd: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    bwd: list[tuple[float, float]] = []
    acc: list[tuple[float, float]] = []
    apply: list[tuple[float, float]] = []
    for idx in range(total_count):
        base = idx * CYCLE_NS
        rstf.append((base + 0.00, base + 0.50))
        rstg.append((base + 0.00, base + 0.50))
        fwd.append((base + 0.75, base + 3.00))
        if idx < train_count:
            rstf.append((base + 12.05, base + 12.55))
            fwd.append((base + 12.80, base + 15.60))
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


def hidden_caps() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        for bit in range(BITS):
            lines += [
                f"Ch{h}{bit}p wh{h}{bit}p 0 20f IC=1.1",
                f"Ch{h}{bit}n wh{h}{bit}n 0 20f IC=1.1",
                f"Rh{h}{bit}p wh{h}{bit}p 0 1e15",
                f"Rh{h}{bit}n wh{h}{bit}n 0 1e15",
            ]
    return "\n".join(lines)


def readout_caps() -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            same = out == xor_label(h)
            p = 0.45 if same else 0.20
            n = 0.20 if same else 0.45
            lines += [
                f"Cvw{out}{h}p vw{out}{h}p 0 20f IC={p:.12g}",
                f"Cvw{out}{h}n vw{out}{h}n 0 20f IC={n:.12g}",
                f"Rvw{out}{h}p vw{out}{h}p 0 1e15",
                f"Rvw{out}{h}n vw{out}{h}n 0 1e15",
            ]
    return "\n".join(lines)


def temporary_caps() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines += [
            f"Cpre{h} pre{h} 0 10f IC=0",
            f"Cact{h} act{h} 0 20f IC=0",
            f"Chdp{h} hdp{h} 0 12f IC=0",
            f"Chdn{h} hdn{h} 0 12f IC=0",
            f"Rpre{h} pre{h} 0 1G",
            f"Ract{h} act{h} 0 1G",
            f"Rhdp{h} hdp{h} 0 1G",
            f"Rhdn{h} hdn{h} 0 1G",
        ]
        for bit in range(BITS):
            lines += [
                f"Cghp{h}{bit} ghp{h}{bit} 0 10f IC=0",
                f"Cghn{h}{bit} ghn{h}{bit} 0 10f IC=0",
                f"Rghp{h}{bit} ghp{h}{bit} 0 1G",
                f"Rghn{h}{bit} ghn{h}{bit} 0 1G",
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
                f"Cgvp{out}{h} gvp{out}{h} 0 20f IC=0",
                f"Cgvn{out}{h} gvn{out}{h} 0 20f IC=0",
                f"Rgvp{out}{h} gvp{out}{h} 0 1G",
                f"Rgvn{out}{h} gvn{out}{h} 0 1G",
            ]
    return "\n".join(lines)


def resets() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines += [
            f"Mreset_pre{h} pre{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_act{h} act{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_hdp{h} hdp{h} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_hdn{h} hdn{h} rstg 0 0 NMOS W=4u L=180n",
        ]
        for bit in range(BITS):
            lines += [
                f"Mreset_ghp{h}{bit} ghp{h}{bit} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_ghn{h}{bit} ghn{h}{bit} rstg 0 0 NMOS W=4u L=180n",
            ]
    for out in range(OUTPUTS):
        lines += [
            f"Mreset_score{out} score{out} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_out{out} out{out} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_dp{out} dp{out} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_dn{out} dn{out} rstg 0 0 NMOS W=4u L=180n",
        ]
        for h in range(HIDDEN):
            lines += [
                f"Mreset_gvp{out}{h} gvp{out}{h} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_gvn{out}{h} gvn{out}{h} rstg 0 0 NMOS W=4u L=180n",
            ]
    return "\n".join(lines)


def hidden_forward() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines.append(f"* Hidden XOR literal feature {h}.")
        for bit in range(BITS):
            match_lit = f"x{bit}" if bit_value(h, bit) else f"nx{bit}"
            mismatch_lit = f"nx{bit}" if bit_value(h, bit) else f"x{bit}"
            lines += [
                f"Mh{h}{bit}pos_x vdd {match_lit} h{h}{bit}p0 0 NMOS W=32u L=180n",
                f"Mh{h}{bit}pos_w h{h}{bit}p0 wh{h}{bit}p h{h}{bit}p1 0 NMOS W=32u L=180n",
                f"Mh{h}{bit}pos_f h{h}{bit}p1 fwd pre{h} 0 NMOS W=32u L=180n",
                f"Mh{h}{bit}neg_f pre{h} fwd h{h}{bit}n0 0 NMOS W=40u L=180n",
                f"Mh{h}{bit}neg_x h{h}{bit}n0 {mismatch_lit} h{h}{bit}n1 0 NMOS W=40u L=180n",
                f"Mh{h}{bit}neg_w h{h}{bit}n1 wh{h}{bit}n 0 0 NMOS W=40u L=180n",
            ]
        lines.append(f"Mrelu_h{h} vdd pre{h} act{h} 0 NREL W=24u L=180n")
    return "\n".join(lines)


def output_forward() -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        lines.append(f"* Output {out}: signed readout from all hidden features.")
        for h in range(HIDDEN):
            lines += [
                f"Mo{out}{h}pos_a vdd act{h} o{out}{h}p0 0 NREL W=64u L=180n",
                f"Mo{out}{h}pos_w o{out}{h}p0 vw{out}{h}p o{out}{h}p1 0 NREL W=64u L=180n",
                f"Mo{out}{h}pos_f o{out}{h}p1 fwd score{out} 0 NREL W=64u L=180n",
                f"Mo{out}{h}neg_f score{out} fwd o{out}{h}n0 0 NREL W=48u L=180n",
                f"Mo{out}{h}neg_a o{out}{h}n0 act{h} o{out}{h}n1 0 NREL W=48u L=180n",
                f"Mo{out}{h}neg_w o{out}{h}n1 vw{out}{h}n 0 0 NREL W=48u L=180n",
            ]
        lines.append(f"Mrelu_o{out} vdd score{out} out{out} 0 NREL W=24u L=180n")
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


def hidden_delta() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines.append(f"* Hidden delta for feature {h} through learned readout weights.")
        for out in range(OUTPUTS):
            lines += [
                f"Mhdp{h}{out}a0 vdd dp{out} hdp{h}{out}a0 0 NSENSE W=32u L=180n",
                f"Mhdp{h}{out}a1 hdp{h}{out}a0 vw{out}{h}p hdp{h}{out}a1 0 NMOS W=32u L=180n",
                f"Mhdp{h}{out}a2 hdp{h}{out}a1 act{h} hdp{h}{out}a2 0 NREL W=32u L=180n",
                f"Mhdp{h}{out}a3 hdp{h}{out}a2 bwd hdp{h} 0 NMOS W=32u L=180n",
                f"Mhdp{h}{out}b0 vdd dn{out} hdp{h}{out}b0 0 NSENSE W=32u L=180n",
                f"Mhdp{h}{out}b1 hdp{h}{out}b0 vw{out}{h}n hdp{h}{out}b1 0 NMOS W=32u L=180n",
                f"Mhdp{h}{out}b2 hdp{h}{out}b1 act{h} hdp{h}{out}b2 0 NREL W=32u L=180n",
                f"Mhdp{h}{out}b3 hdp{h}{out}b2 bwd hdp{h} 0 NMOS W=32u L=180n",
                f"Mhdn{h}{out}a0 vdd dn{out} hdn{h}{out}a0 0 NSENSE W=32u L=180n",
                f"Mhdn{h}{out}a1 hdn{h}{out}a0 vw{out}{h}p hdn{h}{out}a1 0 NMOS W=32u L=180n",
                f"Mhdn{h}{out}a2 hdn{h}{out}a1 act{h} hdn{h}{out}a2 0 NREL W=32u L=180n",
                f"Mhdn{h}{out}a3 hdn{h}{out}a2 bwd hdn{h} 0 NMOS W=32u L=180n",
                f"Mhdn{h}{out}b0 vdd dp{out} hdn{h}{out}b0 0 NSENSE W=32u L=180n",
                f"Mhdn{h}{out}b1 hdn{h}{out}b0 vw{out}{h}n hdn{h}{out}b1 0 NMOS W=32u L=180n",
                f"Mhdn{h}{out}b2 hdn{h}{out}b1 act{h} hdn{h}{out}b2 0 NREL W=32u L=180n",
                f"Mhdn{h}{out}b3 hdn{h}{out}b2 bwd hdn{h} 0 NMOS W=32u L=180n",
            ]
    return "\n".join(lines)


def readout_gradients_and_updates() -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            lines += [
                f"Mgvp{out}{h}_a vdd act{h} gvp{out}{h}_a 0 NREL W=24u L=180n",
                f"Mgvp{out}{h}_d gvp{out}{h}_a dp{out} gvp{out}{h}_d 0 NSENSE W=24u L=180n",
                f"Mgvp{out}{h}_g gvp{out}{h}_d acc gvp{out}{h} 0 NREL W=24u L=180n",
                f"Mgvn{out}{h}_a vdd act{h} gvn{out}{h}_a 0 NREL W=24u L=180n",
                f"Mgvn{out}{h}_d gvn{out}{h}_a dn{out} gvn{out}{h}_d 0 NSENSE W=24u L=180n",
                f"Mgvn{out}{h}_g gvn{out}{h}_d acc gvn{out}{h} 0 NREL W=24u L=180n",
                f"Mvw{out}{h}p_up_g vdd gvp{out}{h} vw{out}{h}p_up 0 NSENSE W=10u L=180n",
                f"Mvw{out}{h}p_up_a vw{out}{h}p_up apply vw{out}{h}p 0 NREL W=10u L=180n",
                f"Mvw{out}{h}n_dn_a vw{out}{h}n apply vw{out}{h}n_dn 0 NREL W=10u L=180n",
                f"Mvw{out}{h}n_dn_g vw{out}{h}n_dn gvp{out}{h} 0 0 NSENSE W=10u L=180n",
                f"Mvw{out}{h}n_up_g vdd gvn{out}{h} vw{out}{h}n_up 0 NSENSE W=10u L=180n",
                f"Mvw{out}{h}n_up_a vw{out}{h}n_up apply vw{out}{h}n 0 NREL W=10u L=180n",
                f"Mvw{out}{h}p_dn_a vw{out}{h}p apply vw{out}{h}p_dn 0 NREL W=10u L=180n",
                f"Mvw{out}{h}p_dn_g vw{out}{h}p_dn gvn{out}{h} 0 0 NSENSE W=10u L=180n",
            ]
    return "\n".join(lines)


def hidden_gradients_and_updates() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        for bit in range(BITS):
            match_lit = f"x{bit}" if bit_value(h, bit) else f"nx{bit}"
            lines += [
                f"Mghp{h}{bit}_x vdd {match_lit} ghp{h}{bit}_x 0 NMOS W=48u L=180n",
                f"Mghp{h}{bit}_d ghp{h}{bit}_x hdp{h} ghp{h}{bit}_d 0 NSENSE W=48u L=180n",
                f"Mghp{h}{bit}_g ghp{h}{bit}_d acc ghp{h}{bit} 0 NMOS W=48u L=180n",
                f"Mghn{h}{bit}_x vdd {match_lit} ghn{h}{bit}_x 0 NMOS W=48u L=180n",
                f"Mghn{h}{bit}_d ghn{h}{bit}_x hdn{h} ghn{h}{bit}_d 0 NSENSE W=48u L=180n",
                f"Mghn{h}{bit}_g ghn{h}{bit}_d acc ghn{h}{bit} 0 NMOS W=48u L=180n",
                f"Mwh{h}{bit}p_up_g vdd ghp{h}{bit} wh{h}{bit}p_up 0 NSENSE W=8u L=180n",
                f"Mwh{h}{bit}p_up_a wh{h}{bit}p_up apply wh{h}{bit}p 0 NREL W=8u L=180n",
                f"Mwh{h}{bit}n_dn_a wh{h}{bit}n apply wh{h}{bit}n_dn 0 NREL W=8u L=180n",
                f"Mwh{h}{bit}n_dn_g wh{h}{bit}n_dn ghp{h}{bit} 0 0 NSENSE W=8u L=180n",
                f"Mwh{h}{bit}n_up_g vdd ghn{h}{bit} wh{h}{bit}n_up 0 NSENSE W=8u L=180n",
                f"Mwh{h}{bit}n_up_a wh{h}{bit}n_up apply wh{h}{bit}n 0 NREL W=8u L=180n",
                f"Mwh{h}{bit}p_dn_a wh{h}{bit}p apply wh{h}{bit}p_dn 0 NREL W=8u L=180n",
                f"Mwh{h}{bit}p_dn_g wh{h}{bit}p_dn ghn{h}{bit} 0 0 NSENSE W=8u L=180n",
            ]
    return "\n".join(lines)


def measures(samples: list[dict[str, Any]], train_count: int) -> tuple[str, str]:
    lines: list[str] = []
    prints: list[str] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        phase = "train" if idx < train_count else "eval"
        local = idx if idx < train_count else idx - train_count
        pattern = int(sample["pattern"])
        label = int(sample["label"])
        other = 1 - label
        at = base + (15.50 if idx < train_count else 2.95)
        lines += [
            f".meas tran {phase}_target_out_{local} FIND V(out{label}) AT={at:.2f}n",
            f".meas tran {phase}_other_out_{local} FIND V(out{other}) AT={at:.2f}n",
            f".meas tran {phase}_margin_{local} PARAM='{phase}_target_out_{local}-{phase}_other_out_{local}'",
        ]
        if idx < train_count:
            active = pattern
            lines += [
                f".meas tran train_target_before_{local} FIND V(out{label}) AT={base + 2.95:.2f}n",
                f".meas tran train_other_before_{local} FIND V(out{other}) AT={base + 2.95:.2f}n",
                f".meas tran train_margin_before_{local} PARAM='train_target_before_{local}-train_other_before_{local}'",
                f".meas tran train_d_margin_{local} PARAM='train_margin_{local}-train_margin_before_{local}'",
                f".meas tran hdp_active_{local} FIND V(hdp{active}) AT={base + 7.10:.2f}n",
                f".meas tran hdn_active_{local} FIND V(hdn{active}) AT={base + 7.10:.2f}n",
            ]
            for bit in range(BITS):
                lines += [
                    f".meas tran wh{active}{bit}p_before_{local} FIND V(wh{active}{bit}p) AT={base + 0.60:.2f}n",
                    f".meas tran wh{active}{bit}p_after_{local} FIND V(wh{active}{bit}p) AT={base + 11.50:.2f}n",
                    f".meas tran d_wh{active}{bit}p_{local} PARAM='wh{active}{bit}p_after_{local}-wh{active}{bit}p_before_{local}'",
                ]
        prints.append(f"print {phase}_target_out_{local} {phase}_other_out_{local} {phase}_margin_{local}")
    return "\n".join(lines), "\n".join(prints)


def xor_netlist(pattern_order: list[int]) -> str:
    train_samples = [{"pattern": p, "label": xor_label(p)} for p in pattern_order]
    eval_samples = [{"pattern": p, "label": xor_label(p)} for p in range(4)]
    samples = train_samples + eval_samples
    stop = len(samples) * CYCLE_NS
    meas, prints = measures(samples, len(train_samples))
    return f"""
* Device-level 2-bit XOR with hidden feature weight updates.
* Four capacitor-held literal feature cells feed a two-output readout.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vx0 x0 0 {sample_wave(samples, "x0", stop)}
Vx1 x1 0 {sample_wave(samples, "x1", stop)}
Vnx0 nx0 0 {sample_wave(samples, "nx0", stop)}
Vnx1 nx1 0 {sample_wave(samples, "nx1", stop)}
Vt0 t0 0 {target_wave(samples, 0, stop)}
Vt1 t1 0 {target_wave(samples, 1, stop)}
{phases(len(train_samples), len(samples))}

{hidden_caps()}
{readout_caps()}

{temporary_caps()}
{resets()}

{hidden_forward()}
{output_forward()}
{error_cells()}
{hidden_delta()}
{readout_gradients_and_updates()}
{hidden_gradients_and_updates()}

.options method=gear maxord=2
.tran 10p {stop:.2f}n uic
{meas}
.control
run
{prints}
.endc
.end
""".lstrip()


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
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--tag", default="device_xor2_learned_features")
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    orders = [("binary_order", list(range(4))), ("interleaved_order", [0, 3, 1, 2])]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for order_name, order in orders:
        parsed = run_netlist(spice_bin, generated / f"{safe_tag}_{order_name}.cir", xor_netlist(order), args.timeout)
        for pattern in range(4):
            label = xor_label(pattern)
            slot = order.index(pattern)
            rows.append(
                {
                    "order": order_name,
                    "phase": "train",
                    "pattern": pattern,
                    "label": label,
                    "target_out": parsed[f"train_target_out_{slot}"],
                    "other_out": parsed[f"train_other_out_{slot}"],
                    "margin": parsed[f"train_margin_{slot}"],
                    "d_margin": parsed[f"train_d_margin_{slot}"],
                    "hidden_delta_net": parsed[f"hdp_active_{slot}"] - parsed[f"hdn_active_{slot}"],
                    "max_active_hidden_weight_delta": max(abs(parsed[f"d_wh{pattern}{bit}p_{slot}"]) for bit in range(BITS)),
                    "correct": parsed[f"train_margin_{slot}"] > 0,
                }
            )
            rows.append(
                {
                    "order": order_name,
                    "phase": "eval",
                    "pattern": pattern,
                    "label": label,
                    "target_out": parsed[f"eval_target_out_{pattern}"],
                    "other_out": parsed[f"eval_other_out_{pattern}"],
                    "margin": parsed[f"eval_margin_{pattern}"],
                    "d_margin": None,
                    "hidden_delta_net": None,
                    "max_active_hidden_weight_delta": None,
                    "correct": parsed[f"eval_margin_{pattern}"] > 0,
                }
            )

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    train = df[df["phase"] == "train"]
    eval_df = df[df["phase"] == "eval"]
    eval_by_order = eval_df.groupby("order")["correct"].mean().to_dict()
    summary = {
        "simulator": version,
        "architecture": "device_level_2bit_xor_learned_hidden_feature_updates",
        "status": "tiny_learned_hidden_update_device_smoke",
        "benchmark": "2-bit XOR",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "Four literal hidden feature cells and a two-output readout run forward/error/backward/accumulate/apply phases. "
            "Both readout weights and active hidden-feature match weights are capacitor-held and updated in SPICE."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "hidden_feature_weights_trained": True,
        "hidden_features_programmed_initially": True,
        "readout_weights_trained": True,
        "full_dataset_patterns": 4,
        "train_orders": len(orders),
        "eval_accuracy_by_order": {k: float(v) for k, v in eval_by_order.items()},
        "min_eval_accuracy": float(min(eval_by_order.values())),
        "all_eval_patterns_correct": bool(eval_df["correct"].all()),
        "all_train_updates_improve_margin": bool((train["d_margin"] > 0).all()),
        "all_active_hidden_updates_nonzero": bool((train["max_active_hidden_weight_delta"] > 1e-7).all()),
        "min_train_margin_improvement_v": float(train["d_margin"].min()),
        "min_eval_margin_v": float(eval_df["margin"].min()),
        "mean_eval_margin_v": float(eval_df["margin"].mean()),
        "max_active_hidden_weight_delta_v": float(train["max_active_hidden_weight_delta"].max()),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This is the first complete nonlinear tiny benchmark where hidden feature weight capacitors are included in "
            "the update path and measured. The hidden features start from programmed XOR literal detectors, so this is "
            "not random hidden-feature discovery yet."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
