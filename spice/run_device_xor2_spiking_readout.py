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
SPIKE_OFFSETS_NS = [2.20, 2.70, 3.20, 3.70, 4.20, 4.70]


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def phases(samples: list[dict[str, Any]]) -> str:
    stop = len(samples) * CYCLE_NS
    rstf: list[tuple[float, float]] = []
    rstg: list[tuple[float, float]] = []
    fwd: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    acc: list[tuple[float, float]] = []
    apply: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        rstf.append((base + 0.00, base + 0.50))
        rstg.append((base + 0.00, base + 0.50))
        fwd.append((base + 0.75, base + 4.20))
        if sample["phase"] == "train":
            err.append((base + 4.35, base + 5.85))
            acc.append((base + 6.10, base + 7.80))
            apply.append((base + 8.05, base + 10.05))
            rstf.append((base + 12.35, base + 12.80))
            fwd.append((base + 13.00, base + 15.60))
    return "\n".join(
        [
            f"Vrstf rstf 0 {pulse_wave(rstf, stop)}",
            f"Vrstg rstg 0 {pulse_wave(rstg, stop)}",
            f"Vfwd fwd 0 {pulse_wave(fwd, stop)}",
            f"Verr err 0 {pulse_wave(err, stop)}",
            f"Vacc acc 0 {pulse_wave(acc, stop)}",
            f"Vapply apply 0 {pulse_wave(apply, stop)}",
        ]
    )


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


def temporary_caps(grad_cap_f: float, spike_cap_f: float, spike_leak_ohm: float) -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines += [
            f"Cpre{h} pre{h} 0 10f IC=0",
            f"Cact{h} act{h} 0 20f IC=0",
            f"Rpre{h} pre{h} 0 1G",
            f"Ract{h} act{h} 0 1G",
        ]
    for out in range(OUTPUTS):
        lines += [
            f"Cscore{out} score{out} 0 10f IC=0",
            f"Cspk{out} spk{out} 0 {spike_cap_f:.12g}f IC=0",
            f"Cdp{out} dp{out} 0 20f IC=0",
            f"Cdn{out} dn{out} 0 20f IC=0",
            f"Rscore{out} score{out} 0 1G",
            f"Rspk{out} spk{out} 0 {spike_leak_ohm:.12g}",
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
        lines += [
            f"Mreset_pre{h} pre{h} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_act{h} act{h} rstf 0 0 NMOS W=4u L=180n",
        ]
    for out in range(OUTPUTS):
        lines += [
            f"Mreset_score{out} score{out} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_spk{out} spk{out} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_dp{out} dp{out} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_dn{out} dn{out} rstg 0 0 NMOS W=4u L=180n",
        ]
        for h in range(HIDDEN):
            lines += [
                f"Mreset_gvp{out}{h} gvp{out}{h} rstg 0 0 NMOS W=4u L=180n",
                f"Mreset_gvn{out}{h} gvn{out}{h} rstg 0 0 NMOS W=4u L=180n",
            ]
    return "\n".join(lines)


def spiking_output_forward(spike_width_u: float) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        lines.append(f"* Output {out}: signed readout into a leaky spike capacitor.")
        for h in range(HIDDEN):
            lines += [
                f"Mo{out}{h}pos_a vdd act{h} o{out}{h}p0 0 NREL W=64u L=180n",
                f"Mo{out}{h}pos_w o{out}{h}p0 vw{out}{h}p o{out}{h}p1 0 NREL W=64u L=180n",
                f"Mo{out}{h}pos_f o{out}{h}p1 fwd score{out} 0 NREL W=64u L=180n",
                f"Mo{out}{h}neg_f score{out} fwd o{out}{h}n0 0 NREL W=48u L=180n",
                f"Mo{out}{h}neg_a o{out}{h}n0 act{h} o{out}{h}n1 0 NREL W=48u L=180n",
                f"Mo{out}{h}neg_w o{out}{h}n1 vw{out}{h}n 0 0 NREL W=48u L=180n",
            ]
        lines += [
            f"Mspk{out}_sense vdd score{out} spk{out}_sense 0 NREL W={spike_width_u:.12g}u L=180n",
            f"Mspk{out}_gate spk{out}_sense fwd spk{out} 0 NREL W={spike_width_u:.12g}u L=180n",
        ]
    return "\n".join(lines)


def spike_error_cells(error_source: str) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        y = f"{error_source}{out}"
        lines += [
            f"Mdp{out}_t0 vdd t{out} dp{out}_t 0 NSENSE W=32u L=180n",
            f"Mdp{out}_t1 dp{out}_t err dp{out} 0 NSENSE W=32u L=180n",
            f"Mdp{out}_y0 dp{out} err dp{out}_y 0 NSENSE W=24u L=180n",
            f"Mdp{out}_y1 dp{out}_y {y} 0 0 NSENSE W=24u L=180n",
            f"Mdn{out}_y0 vdd {y} dn{out}_y 0 NSENSE W=32u L=180n",
            f"Mdn{out}_y1 dn{out}_y err dn{out} 0 NSENSE W=32u L=180n",
            f"Mdn{out}_t0 dn{out} err dn{out}_t 0 NSENSE W=24u L=180n",
            f"Mdn{out}_t1 dn{out}_t t{out} 0 0 NSENSE W=24u L=180n",
        ]
    return "\n".join(lines)


def readout_updates(update_width_u: float, write_polarity: str) -> str:
    if write_polarity not in {"normal", "reversed"}:
        raise ValueError(f"unknown write polarity: {write_polarity}")
    pos_delta = "dp" if write_polarity == "normal" else "dn"
    neg_delta = "dn" if write_polarity == "normal" else "dp"
    lines: list[str] = []
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            lines += [
                f"Mgvp{out}{h}_a vdd act{h} gvp{out}{h}_a 0 NREL W=24u L=180n",
                f"Mgvp{out}{h}_d gvp{out}{h}_a {pos_delta}{out} gvp{out}{h}_d 0 NSENSE W=24u L=180n",
                f"Mgvp{out}{h}_g gvp{out}{h}_d acc gvp{out}{h} 0 NREL W=24u L=180n",
                f"Mgvn{out}{h}_a vdd act{h} gvn{out}{h}_a 0 NREL W=24u L=180n",
                f"Mgvn{out}{h}_d gvn{out}{h}_a {neg_delta}{out} gvn{out}{h}_d 0 NSENSE W=24u L=180n",
                f"Mgvn{out}{h}_g gvn{out}{h}_d acc gvn{out}{h} 0 NREL W=24u L=180n",
                f"Mvw{out}{h}p_up_g vdd gvp{out}{h} vw{out}{h}p_up 0 NSENSE W={update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}p_up_a vw{out}{h}p_up apply vw{out}{h}p 0 NREL W={update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_dn_a vw{out}{h}n apply vw{out}{h}n_dn 0 NREL W={update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_dn_g vw{out}{h}n_dn gvp{out}{h} 0 0 NSENSE W={update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_up_g vdd gvn{out}{h} vw{out}{h}n_up 0 NSENSE W={update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}n_up_a vw{out}{h}n_up apply vw{out}{h}n 0 NREL W={update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}p_dn_a vw{out}{h}p apply vw{out}{h}p_dn 0 NREL W={update_width_u:.12g}u L=180n",
                f"Mvw{out}{h}p_dn_g vw{out}{h}p_dn gvn{out}{h} 0 0 NSENSE W={update_width_u:.12g}u L=180n",
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
        label = int(sample["label"])
        other = 1 - label
        for offset in SPIKE_OFFSETS_NS:
            suffix = f"{idx}_{int(round(offset * 100)):03d}"
            lines += [
                f".meas tran target_spk_{suffix} FIND V(spk{label}) AT={base + offset:.2f}n",
                f".meas tran other_spk_{suffix} FIND V(spk{other}) AT={base + offset:.2f}n",
                f".meas tran spk_margin_{suffix} PARAM='target_spk_{suffix}-other_spk_{suffix}'",
            ]
        lines += [
            f".meas tran target_score_{idx} FIND V(score{label}) AT={base + 4.15:.2f}n",
            f".meas tran other_score_{idx} FIND V(score{other}) AT={base + 4.15:.2f}n",
            f".meas tran score_margin_{idx} PARAM='target_score_{idx}-other_score_{idx}'",
        ]
        if sample["phase"] == "train":
            lines += [
                f".meas tran dp_target_{idx} FIND V(dp{label}) AT={base + 5.95:.2f}n",
                f".meas tran dn_target_{idx} FIND V(dn{label}) AT={base + 5.95:.2f}n",
                f".meas tran post_target_spk_{idx} FIND V(spk{label}) AT={base + 15.20:.2f}n",
                f".meas tran post_other_spk_{idx} FIND V(spk{other}) AT={base + 15.20:.2f}n",
                f".meas tran post_spk_margin_{idx} PARAM='post_target_spk_{idx}-post_other_spk_{idx}'",
            ]
        prints.append(f"print score_margin_{idx}")
    return "\n".join(lines), "\n".join(prints)


def xor_netlist(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    samples = make_samples(args.epochs, [0, 3, 1, 2])
    stop = len(samples) * CYCLE_NS
    meas, prints = measures(samples)
    return (
        f"""
* Device-level spiking/timed-output XOR readout probe.
* Class evidence charges a leaky spike capacitor; the correct output only needs to win at a timed sample.
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
{readout_caps(args.readout_high_v, args.readout_low_v, args.weight_cap_f)}

{temporary_caps(args.gradient_cap_f, args.spike_cap_f, args.spike_leak_ohm)}
{resets()}

{hidden_forward()}
{spiking_output_forward(args.spike_width_u)}
{spike_error_cells(args.error_source)}
{readout_updates(args.readout_update_width_u, args.write_polarity)}

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
    ap.add_argument("--tag", default="device_xor2_spiking_readout")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--readout-high-v", type=float, default=0.58)
    ap.add_argument("--readout-low-v", type=float, default=0.18)
    ap.add_argument("--weight-cap-f", type=float, default=20.0)
    ap.add_argument("--gradient-cap-f", type=float, default=8.0)
    ap.add_argument("--spike-cap-f", type=float, default=12.0)
    ap.add_argument("--spike-leak-ohm", type=float, default=1.6e8)
    ap.add_argument("--spike-width-u", type=float, default=36.0)
    ap.add_argument("--readout-update-width-u", type=float, default=5.0)
    ap.add_argument("--error-source", choices=["score", "spk"], default="spk")
    ap.add_argument("--write-polarity", choices=["normal", "reversed"], default="normal")
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
        label = int(sample["label"])
        row_base = {
            "cycle": idx,
            "phase": phase,
            "pattern": int(sample["pattern"]),
            "label": label,
            "score_margin": parsed[f"score_margin_{idx}"],
        }
        for offset in SPIKE_OFFSETS_NS:
            suffix = f"{idx}_{int(round(offset * 100)):03d}"
            rows.append(
                row_base
                | {
                    "offset_ns": offset,
                    "target_spike": parsed[f"target_spk_{suffix}"],
                    "other_spike": parsed[f"other_spk_{suffix}"],
                    "spike_margin": parsed[f"spk_margin_{suffix}"],
                    "correct": parsed[f"spk_margin_{suffix}"] > 0,
                }
            )

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    initial = df[df["phase"] == "initial_eval"]
    final = df[df["phase"] == "final_eval"]
    initial_by_offset = initial.groupby("offset_ns")["correct"].mean().to_dict()
    final_by_offset = final.groupby("offset_ns")["correct"].mean().to_dict()
    best_initial_offset, best_initial_acc = max(initial_by_offset.items(), key=lambda kv: kv[1])
    best_final_offset, best_final_acc = max(final_by_offset.items(), key=lambda kv: kv[1])
    summary = {
        "simulator": version,
        "architecture": "device_level_2bit_xor_spiking_timed_readout",
        "status": "spiking_timed_output_probe",
        "benchmark": "2-bit XOR",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "Literal hidden features drive signed capacitor-held readout weights. "
            "The output score charges a leaky spike capacitor, and accuracy is evaluated at fixed time offsets."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_multipliers": False,
        "output_is_timed_spike_voltage": True,
        "error_source": args.error_source,
        "write_polarity": args.write_polarity,
        "epochs": args.epochs,
        "initial_accuracy_by_offset": {str(k): float(v) for k, v in initial_by_offset.items()},
        "final_accuracy_by_offset": {str(k): float(v) for k, v in final_by_offset.items()},
        "best_initial_offset_ns": float(best_initial_offset),
        "best_initial_accuracy": float(best_initial_acc),
        "best_final_offset_ns": float(best_final_offset),
        "best_final_accuracy": float(best_final_acc),
        "mean_final_score_margin_v": float(final.groupby("cycle")["score_margin"].first().mean()),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This tests a timed output code rather than steady-state output accuracy. "
            "The target class only has to produce the larger spike voltage at the chosen read instant."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
