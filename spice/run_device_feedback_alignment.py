from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pandas as pd

from run_device_multicell_classifier import (
    CYCLE_NS,
    ROOT,
    error_cell,
    hidden_forward,
    hidden_gradients_and_updates,
    measure_lines,
    mos_models,
    output_forward,
    persistent_caps,
    readout_gradients_and_updates,
    repeated_phases,
    resets,
    row_for_sample,
    run_netlist,
    sample_wave,
    temporary_caps,
)
from run_spice_sweep import detect_spice, run_tiny_test


def feedback_caps(feedback_init: dict[str, float]) -> str:
    lines: list[str] = []
    for j in range(2):
        for h in range(2):
            lines += [
                f"Cfb{j}{h}p fb{j}{h}p 0 20f IC={feedback_init[f'fb{j}{h}p']:.12g}",
                f"Cfb{j}{h}n fb{j}{h}n 0 20f IC={feedback_init[f'fb{j}{h}n']:.12g}",
                f"Rfb{j}{h}p fb{j}{h}p 0 1e15",
                f"Rfb{j}{h}n fb{j}{h}n 0 1e15",
            ]
    return "\n".join(lines)


def feedback_alignment_hidden_delta(h: int) -> str:
    lines = [f"* Hidden {h} direct feedback-alignment delta through fixed feedback caps."]
    for j in range(2):
        lines += [
            f"Mhdp{h}{j}a0 vdd dp{j} hdp{h}{j}a0 0 NSENSE W=32u L=180n",
            f"Mhdp{h}{j}a1 hdp{h}{j}a0 fb{j}{h}p hdp{h}{j}a1 0 NMOS W=32u L=180n",
            f"Mhdp{h}{j}a2 hdp{h}{j}a1 act{h} hdp{h}{j}a2 0 NREL W=32u L=180n",
            f"Mhdp{h}{j}a3 hdp{h}{j}a2 bwd hdp{h} 0 NMOS W=32u L=180n",
            f"Mhdp{h}{j}b0 vdd dn{j} hdp{h}{j}b0 0 NSENSE W=32u L=180n",
            f"Mhdp{h}{j}b1 hdp{h}{j}b0 fb{j}{h}n hdp{h}{j}b1 0 NMOS W=32u L=180n",
            f"Mhdp{h}{j}b2 hdp{h}{j}b1 act{h} hdp{h}{j}b2 0 NREL W=32u L=180n",
            f"Mhdp{h}{j}b3 hdp{h}{j}b2 bwd hdp{h} 0 NMOS W=32u L=180n",
            f"Mhdn{h}{j}a0 vdd dn{j} hdn{h}{j}a0 0 NSENSE W=32u L=180n",
            f"Mhdn{h}{j}a1 hdn{h}{j}a0 fb{j}{h}p hdn{h}{j}a1 0 NMOS W=32u L=180n",
            f"Mhdn{h}{j}a2 hdn{h}{j}a1 act{h} hdn{h}{j}a2 0 NREL W=32u L=180n",
            f"Mhdn{h}{j}a3 hdn{h}{j}a2 bwd hdn{h} 0 NMOS W=32u L=180n",
            f"Mhdn{h}{j}b0 vdd dp{j} hdn{h}{j}b0 0 NSENSE W=32u L=180n",
            f"Mhdn{h}{j}b1 hdn{h}{j}b0 fb{j}{h}n hdn{h}{j}b1 0 NMOS W=32u L=180n",
            f"Mhdn{h}{j}b2 hdn{h}{j}b1 act{h} hdn{h}{j}b2 0 NREL W=32u L=180n",
            f"Mhdn{h}{j}b3 hdn{h}{j}b2 bwd hdn{h} 0 NMOS W=32u L=180n",
        ]
    return "\n".join(lines)


def feedback_alignment_netlist(
    samples: list[dict[str, float]],
    hidden_init: dict[str, float],
    readout_init: dict[str, float],
    feedback_init: dict[str, float],
) -> str:
    stop = len(samples) * CYCLE_NS
    measures, prints = measure_lines(samples)
    return f"""
* Device-level 2-input/2-hidden/2-output classifier with direct feedback alignment.
* Hidden deltas use fixed feedback capacitor nodes, not transported readout weights.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vx0 x0 0 {sample_wave(samples, "x0", stop)}
Vx1 x1 0 {sample_wave(samples, "x1", stop)}
Vt0 t0 0 {sample_wave(samples, "t0", stop)}
Vt1 t1 0 {sample_wave(samples, "t1", stop)}
{repeated_phases(len(samples))}

* Persistent signed forward weights.
{persistent_caps(hidden_init, readout_init)}

* Fixed signed feedback-alignment weights.
{feedback_caps(feedback_init)}

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

{feedback_alignment_hidden_delta(0)}
{feedback_alignment_hidden_delta(1)}

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--tag", default="device_feedback_alignment")
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
    feedback_init = {
        "fb00p": 1.10,
        "fb00n": 0.20,
        "fb01p": 0.20,
        "fb01n": 1.10,
        "fb10p": 0.20,
        "fb10n": 1.10,
        "fb11p": 1.10,
        "fb11n": 0.20,
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
            feedback_alignment_netlist(samples, hidden_init, readout_init, feedback_init),
            args.timeout,
        )
        for sample_idx, sample in enumerate(samples):
            rows.append(row_for_sample(sequence["sequence"], sample_idx, sample, measures))

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    bp_summary_path = results / "device_multicell_classifier_v1_summary.json"
    bp_summary = json.loads(bp_summary_path.read_text()) if bp_summary_path.exists() else {}
    summary = {
        "simulator": version,
        "architecture": "device_level_direct_feedback_alignment_2input_2hidden_2output",
        "status": "alternative_training_rule_device_smoke",
        "training_rule": "direct_feedback_alignment",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "signal_path": (
            "Forward readout learning is unchanged, but hidden deltas are produced by fixed signed feedback "
            "capacitor nodes instead of learned readout weight capacitors. This avoids exact backward weight transport."
        ),
        "no_behavioral_signal_math": True,
        "uses_behavioral_tanh": False,
        "uses_behavioral_multipliers": False,
        "uses_readout_weight_transport_for_hidden_delta": False,
        "fixed_feedback_caps": True,
        "weight_caps_persist_inside_single_spice_transient": True,
        "batching_supported": False,
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
        "backprop_style_mean_margin_improvement_v": bp_summary.get("mean_margin_improvement_v"),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This demonstrates that a fixed-feedback capacitor path can train the tiny multicell one-hot task "
            "without transporting learned readout weights into the hidden-delta circuit. It is still a synthetic "
            "two-class smoke test, not MNIST."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
