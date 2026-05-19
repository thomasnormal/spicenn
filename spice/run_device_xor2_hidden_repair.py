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
    error_cells,
    hidden_delta,
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
    bwd: list[tuple[float, float]] = []
    acc: list[tuple[float, float]] = []
    apply: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        rstf.append((base + 0.00, base + 0.50))
        rstg.append((base + 0.00, base + 0.50))
        fwd.append((base + 0.75, base + 3.00))
        if sample["phase"] == "train":
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


def hidden_caps(hidden_init_v: float, negative_match_init_v: float, cap_f: float) -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        for bit in range(BITS):
            lines += [
                f"Ch{h}{bit}p wh{h}{bit}p 0 {cap_f:.12g}f IC={hidden_init_v:.12g}",
                f"Ch{h}{bit}n wh{h}{bit}n 0 {cap_f:.12g}f IC={negative_match_init_v:.12g}",
                f"Ch{h}{bit}m wmis{h}{bit} 0 20f IC=1.1",
                f"Rh{h}{bit}p wh{h}{bit}p 0 1e15",
                f"Rh{h}{bit}n wh{h}{bit}n 0 1e15",
                f"Rh{h}{bit}m wmis{h}{bit} 0 1e15",
            ]
    return "\n".join(lines)


def readout_caps(readout_high_v: float, readout_low_v: float) -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        for h in range(HIDDEN):
            same = out == xor_label(h)
            p = readout_high_v if same else readout_low_v
            n = readout_low_v if same else readout_high_v
            lines += [
                f"Cvw{out}{h}p vw{out}{h}p 0 20f IC={p:.12g}",
                f"Cvw{out}{h}n vw{out}{h}n 0 20f IC={n:.12g}",
                f"Rvw{out}{h}p vw{out}{h}p 0 1e15",
                f"Rvw{out}{h}n vw{out}{h}n 0 1e15",
            ]
    return "\n".join(lines)


def hidden_forward() -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        lines.append(f"* Hidden XOR literal feature {h} with signed match weight and fixed mismatch suppression.")
        for bit in range(BITS):
            match_lit = f"x{bit}" if (h >> bit) & 1 else f"nx{bit}"
            mismatch_lit = f"nx{bit}" if (h >> bit) & 1 else f"x{bit}"
            lines += [
                f"Mh{h}{bit}pos_x vdd {match_lit} h{h}{bit}p0 0 NMOS W=32u L=180n",
                f"Mh{h}{bit}pos_w h{h}{bit}p0 wh{h}{bit}p h{h}{bit}p1 0 NMOS W=32u L=180n",
                f"Mh{h}{bit}pos_f h{h}{bit}p1 fwd pre{h} 0 NMOS W=32u L=180n",
                f"Mh{h}{bit}negm_f pre{h} fwd h{h}{bit}nm0 0 NMOS W=24u L=180n",
                f"Mh{h}{bit}negm_x h{h}{bit}nm0 {match_lit} h{h}{bit}nm1 0 NMOS W=24u L=180n",
                f"Mh{h}{bit}negm_w h{h}{bit}nm1 wh{h}{bit}n 0 0 NMOS W=24u L=180n",
                f"Mh{h}{bit}mis_f pre{h} fwd h{h}{bit}mis0 0 NMOS W=40u L=180n",
                f"Mh{h}{bit}mis_x h{h}{bit}mis0 {mismatch_lit} h{h}{bit}mis1 0 NMOS W=40u L=180n",
                f"Mh{h}{bit}mis_w h{h}{bit}mis1 wmis{h}{bit} 0 0 NMOS W=40u L=180n",
            ]
        lines.append(f"Mrelu_h{h} vdd pre{h} act{h} 0 NREL W=24u L=180n")
    return "\n".join(lines)


def output_forward() -> str:
    lines: list[str] = []
    for out in range(OUTPUTS):
        lines.append(f"* Low-voltage output {out}: signed readout from all hidden features.")
        for h in range(HIDDEN):
            lines += [
                f"Mo{out}{h}pos_a vdd act{h} o{out}{h}p0 0 NSENSE W=64u L=180n",
                f"Mo{out}{h}pos_w o{out}{h}p0 vw{out}{h}p o{out}{h}p1 0 NREL W=64u L=180n",
                f"Mo{out}{h}pos_f o{out}{h}p1 fwd score{out} 0 NREL W=64u L=180n",
                f"Mo{out}{h}neg_f score{out} fwd o{out}{h}n0 0 NREL W=48u L=180n",
                f"Mo{out}{h}neg_a o{out}{h}n0 act{h} o{out}{h}n1 0 NSENSE W=48u L=180n",
                f"Mo{out}{h}neg_w o{out}{h}n1 vw{out}{h}n 0 0 NREL W=48u L=180n",
            ]
        lines.append(f"Mrelu_o{out} vdd score{out} out{out} 0 NSENSE W=24u L=180n")
    return "\n".join(lines)


def temporary_caps(gradient_cap_f: float) -> str:
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
                f"Cghp{h}{bit} ghp{h}{bit} 0 {gradient_cap_f:.12g}f IC=0",
                f"Cghn{h}{bit} ghn{h}{bit} 0 {gradient_cap_f:.12g}f IC=0",
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
    return "\n".join(lines)


def hidden_gradients_and_updates(update_width_u: float) -> str:
    lines: list[str] = []
    for h in range(HIDDEN):
        match_lits = [f"x{bit}" if (h >> bit) & 1 else f"nx{bit}" for bit in range(BITS)]
        for bit in range(BITS):
            lines += [
                f"Mghp{h}{bit}_x0 vdd {match_lits[0]} ghp{h}{bit}_x0 0 NMOS W=64u L=180n",
                f"Mghp{h}{bit}_x1 ghp{h}{bit}_x0 {match_lits[1]} ghp{h}{bit}_x1 0 NMOS W=64u L=180n",
                f"Mghp{h}{bit}_d ghp{h}{bit}_x1 hdp{h} ghp{h}{bit}_d 0 NSENSE W=64u L=180n",
                f"Mghp{h}{bit}_a ghp{h}{bit}_d act{h} ghp{h}{bit}_a 0 NREL W=64u L=180n",
                f"Mghp{h}{bit}_g ghp{h}{bit}_a acc ghp{h}{bit} 0 NMOS W=64u L=180n",
                f"Mghn{h}{bit}_x0 vdd {match_lits[0]} ghn{h}{bit}_x0 0 NMOS W=64u L=180n",
                f"Mghn{h}{bit}_x1 ghn{h}{bit}_x0 {match_lits[1]} ghn{h}{bit}_x1 0 NMOS W=64u L=180n",
                f"Mghn{h}{bit}_d ghn{h}{bit}_x1 hdn{h} ghn{h}{bit}_d 0 NSENSE W=64u L=180n",
                f"Mghn{h}{bit}_a ghn{h}{bit}_d act{h} ghn{h}{bit}_a 0 NREL W=64u L=180n",
                f"Mghn{h}{bit}_g ghn{h}{bit}_a acc ghn{h}{bit} 0 NMOS W=64u L=180n",
                f"Mwh{h}{bit}n_dn_a wh{h}{bit}n apply wh{h}{bit}n_dn 0 NREL W={update_width_u:.12g}u L=180n",
                f"Mwh{h}{bit}n_dn_g wh{h}{bit}n_dn ghp{h}{bit} 0 0 NSENSE W={update_width_u:.12g}u L=180n",
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


def measure_lines(samples: list[dict[str, Any]]) -> tuple[str, str]:
    lines: list[str] = []
    prints: list[str] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        pattern = int(sample["pattern"])
        label = int(sample["label"])
        other = 1 - label
        lines += [
            f".meas tran target_out_{idx} FIND V(out{label}) AT={base + 2.95:.2f}n",
            f".meas tran other_out_{idx} FIND V(out{other}) AT={base + 2.95:.2f}n",
            f".meas tran margin_{idx} PARAM='target_out_{idx}-other_out_{idx}'",
            f".meas tran active_act_{idx} FIND V(act{pattern}) AT={base + 2.95:.2f}n",
            f".meas tran target_score_{idx} FIND V(score{label}) AT={base + 2.95:.2f}n",
            f".meas tran other_score_{idx} FIND V(score{other}) AT={base + 2.95:.2f}n",
        ]
        if sample["phase"] == "train":
            lines += [
                f".meas tran train_target_after_{idx} FIND V(out{label}) AT={base + 15.50:.2f}n",
                f".meas tran train_other_after_{idx} FIND V(out{other}) AT={base + 15.50:.2f}n",
                f".meas tran train_margin_after_{idx} PARAM='train_target_after_{idx}-train_other_after_{idx}'",
                f".meas tran train_d_margin_{idx} PARAM='train_margin_after_{idx}-margin_{idx}'",
                f".meas tran hdp_active_{idx} FIND V(hdp{pattern}) AT={base + 7.10:.2f}n",
                f".meas tran hdn_active_{idx} FIND V(hdn{pattern}) AT={base + 7.10:.2f}n",
                f".meas tran hidden_delta_net_{idx} PARAM='hdp_active_{idx}-hdn_active_{idx}'",
            ]
            for bit in range(BITS):
                lines += [
                    f".meas tran wh{pattern}{bit}p_before_{idx} FIND V(wh{pattern}{bit}p) AT={base + 0.60:.2f}n",
                    f".meas tran wh{pattern}{bit}n_before_{idx} FIND V(wh{pattern}{bit}n) AT={base + 0.60:.2f}n",
                    f".meas tran wh{pattern}{bit}p_after_{idx} FIND V(wh{pattern}{bit}p) AT={base + 11.50:.2f}n",
                    f".meas tran wh{pattern}{bit}n_after_{idx} FIND V(wh{pattern}{bit}n) AT={base + 11.50:.2f}n",
                    f".meas tran wh{pattern}{bit}_signed_before_{idx} PARAM='wh{pattern}{bit}p_before_{idx}-wh{pattern}{bit}n_before_{idx}'",
                    f".meas tran wh{pattern}{bit}_signed_after_{idx} PARAM='wh{pattern}{bit}p_after_{idx}-wh{pattern}{bit}n_after_{idx}'",
                    f".meas tran d_wh{pattern}{bit}p_{idx} PARAM='wh{pattern}{bit}p_after_{idx}-wh{pattern}{bit}p_before_{idx}'",
                    f".meas tran d_wh{pattern}{bit}_signed_{idx} PARAM='wh{pattern}{bit}_signed_after_{idx}-wh{pattern}{bit}_signed_before_{idx}'",
                ]
        prints.append(f"print target_out_{idx} other_out_{idx} margin_{idx} active_act_{idx}")
    final_base = (len(samples) - 1) * CYCLE_NS
    for h in range(HIDDEN):
        for bit in range(BITS):
            lines += [
                f".meas tran wh{h}{bit}p_initial FIND V(wh{h}{bit}p) AT=0.60n",
                f".meas tran wh{h}{bit}n_initial FIND V(wh{h}{bit}n) AT=0.60n",
                f".meas tran wh{h}{bit}p_final FIND V(wh{h}{bit}p) AT={final_base + 0.60:.2f}n",
                f".meas tran wh{h}{bit}n_final FIND V(wh{h}{bit}n) AT={final_base + 0.60:.2f}n",
                f".meas tran d_wh{h}{bit}p_total PARAM='wh{h}{bit}p_final-wh{h}{bit}p_initial'",
                f".meas tran wh{h}{bit}_signed_initial PARAM='wh{h}{bit}p_initial-wh{h}{bit}n_initial'",
                f".meas tran wh{h}{bit}_signed_final PARAM='wh{h}{bit}p_final-wh{h}{bit}n_final'",
                f".meas tran d_wh{h}{bit}_signed_total PARAM='wh{h}{bit}_signed_final-wh{h}{bit}_signed_initial'",
            ]
    return "\n".join(lines), "\n".join(prints)


def xor_repair_netlist(
    epochs: int,
    order: list[int],
    hidden_init_v: float,
    mismatch_init_v: float,
    hidden_cap_f: float,
    gradient_cap_f: float,
    update_width_u: float,
    readout_high_v: float,
    readout_low_v: float,
) -> tuple[str, list[dict[str, Any]]]:
    samples = make_samples(epochs, order)
    stop = len(samples) * CYCLE_NS
    meas, prints = measure_lines(samples)
    return (
        f"""
* Device-level 2-bit XOR hidden-feature repair.
* The readout is fixed; only hidden feature weight capacitors are updated.
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

{hidden_caps(hidden_init_v, mismatch_init_v, hidden_cap_f)}
{readout_caps(readout_high_v, readout_low_v)}

{temporary_caps(gradient_cap_f)}
{resets()}

{hidden_forward()}
{output_forward()}
{error_cells()}
{hidden_delta()}
{hidden_gradients_and_updates(update_width_u)}

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
    ap.add_argument("--tag", default="device_xor2_hidden_repair")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--hidden-init-v", type=float, default=0.95)
    ap.add_argument("--mismatch-init-v", type=float, default=0.85)
    ap.add_argument("--hidden-cap-f", type=float, default=2.0)
    ap.add_argument("--gradient-cap-f", type=float, default=4.0)
    ap.add_argument("--update-width-u", type=float, default=160.0)
    ap.add_argument("--readout-high-v", type=float, default=1.10)
    ap.add_argument("--readout-low-v", type=float, default=0.05)
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    netlist, samples = xor_repair_netlist(
        args.epochs,
        [0, 1, 2, 3],
        args.hidden_init_v,
        args.mismatch_init_v,
        args.hidden_cap_f,
        args.gradient_cap_f,
        args.update_width_u,
        args.readout_high_v,
        args.readout_low_v,
    )
    t0 = time.perf_counter()
    parsed = run_netlist(spice_bin, generated / f"{safe_tag}.cir", netlist, args.timeout)

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        phase = str(sample["phase"])
        pattern = int(sample["pattern"])
        row: dict[str, Any] = {
            "cycle": idx,
            "phase": phase,
            "pattern": pattern,
            "label": int(sample["label"]),
            "target_out": parsed[f"target_out_{idx}"],
            "other_out": parsed[f"other_out_{idx}"],
            "margin": parsed[f"margin_{idx}"],
            "active_hidden_act": parsed[f"active_act_{idx}"],
            "correct": parsed[f"margin_{idx}"] > 0,
        }
        if phase == "train":
            row.update(
                {
                    "post_update_margin": parsed[f"train_margin_after_{idx}"],
                    "d_margin_after_update": parsed[f"train_d_margin_{idx}"],
                    "hidden_delta_net": parsed[f"hidden_delta_net_{idx}"],
                    "max_active_hidden_weight_delta": max(
                        abs(parsed[f"d_wh{pattern}{bit}p_{idx}"]) for bit in range(BITS)
                    ),
                    "max_active_hidden_signed_delta": max(
                        abs(parsed[f"d_wh{pattern}{bit}_signed_{idx}"]) for bit in range(BITS)
                    ),
                }
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    initial_eval = df[df["phase"] == "initial_eval"]
    final_eval = df[df["phase"] == "final_eval"]
    train = df[df["phase"] == "train"]
    hidden_total_signed_deltas = [
        parsed[f"d_wh{h}{bit}_signed_total"]
        for h in range(HIDDEN)
        for bit in range(BITS)
    ]
    summary = {
        "simulator": version,
        "architecture": "device_level_2bit_xor_hidden_feature_repair",
        "status": "tiny_hidden_weight_repair_device_smoke",
        "benchmark": "2-bit XOR",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "A fixed class-coded readout sends output errors backward through signed readout capacitor nodes. "
            "Only the hidden feature match-weight capacitors are updated during training cycles."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "readout_weights_trained": False,
        "hidden_feature_weights_trained": True,
        "hidden_feature_topology_programmed": True,
        "hidden_feature_weights_initially_weak": True,
        "epochs": args.epochs,
        "hidden_init_v": args.hidden_init_v,
        "negative_match_init_v": args.mismatch_init_v,
        "fixed_mismatch_suppression_v": 1.1,
        "hidden_cap_f": args.hidden_cap_f,
        "gradient_cap_f": args.gradient_cap_f,
        "update_width_u": args.update_width_u,
        "readout_high_v": args.readout_high_v,
        "readout_low_v": args.readout_low_v,
        "initial_eval_accuracy": float(initial_eval["correct"].mean()),
        "final_eval_accuracy": float(final_eval["correct"].mean()),
        "initial_min_margin_v": float(initial_eval["margin"].min()),
        "final_min_margin_v": float(final_eval["margin"].min()),
        "min_margin_gain_v": float((final_eval["margin"].to_numpy() - initial_eval["margin"].to_numpy()).min()),
        "mean_active_hidden_activation_initial_v": float(initial_eval["active_hidden_act"].mean()),
        "mean_active_hidden_activation_final_v": float(final_eval["active_hidden_act"].mean()),
        "active_hidden_activation_gain_v": float(
            final_eval["active_hidden_act"].mean() - initial_eval["active_hidden_act"].mean()
        ),
        "all_train_hidden_updates_nonzero": bool((train["max_active_hidden_signed_delta"] > 1e-7).all()),
        "max_train_hidden_weight_delta_v": float(train["max_active_hidden_weight_delta"].max()),
        "max_train_hidden_signed_delta_v": float(train["max_active_hidden_signed_delta"].max()),
        "min_total_hidden_signed_delta_v": float(min(hidden_total_signed_deltas)),
        "max_total_hidden_signed_delta_v": float(max(hidden_total_signed_deltas)),
        "all_total_hidden_match_weights_increased": bool(min(hidden_total_signed_deltas) > 0),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This tests hidden-weight repair, not full feature discovery: the literal feature topology and mismatch "
            "suppression are fixed, but the signed hidden match-weight capacitors start weak and are the only trainable "
            "weights in the SPICE transient."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
