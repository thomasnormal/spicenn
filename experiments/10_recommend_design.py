from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    candidate = load_json(ROOT / "results/tables/hardware_mnist_candidate_summary.json")
    value = pd.read_csv(ROOT / "results/tables/value_activation_sweep.csv")
    hw_value = value[value["activation_design"] != "clean_silu_upper"].sort_values(
        ["accuracy", "inference_j"], ascending=[False, True]
    )
    best_value = hw_value.iloc[0].to_dict()
    charge_adc_path = ROOT / "spice/results/charge_adc_sweep.csv"
    charge_summary = {}
    if charge_adc_path.exists():
        charge = pd.read_csv(charge_adc_path)
        usable = charge.groupby(["bits", "cint_f"], as_index=False).agg(
            code_min=("mean_code_0_1", "min"),
            code_max=("mean_code_0_1", "max"),
            ecap_j_max=("ecap_j_mean", "max"),
        )
        usable["code_span"] = usable["code_max"] - usable["code_min"]
        charge_summary = usable.sort_values(["code_span", "ecap_j_max"], ascending=[False, True]).iloc[0].to_dict()

    rec = {
        "recommended_architecture": "hardware_local_evidence",
        "recommended_activation": "4-bit quantized ReLU6 / rectified charge code",
        "full_mnist_candidate": candidate["best_inference_point"],
        "bounded_value_sweep_best_hardware": best_value,
        "spice_charge_adc_best_span": charge_summary,
        "prototype_tile": {
            "input_sheet": "28x28 with local x/y coordinate ramps",
            "channels": "32,64,96",
            "local_kernels": "5x5 then 3x3 then 3x3 then 1x1 class evidence",
            "readout": "spatial mean of 10 class evidence maps; no dense final layer",
            "activation": "conductance-weighted integration, clipped positive charge/voltage, 4-bit local ADC",
            "suggested_cint_f": charge_summary.get("cint_f", 10e-15),
            "max_local_wire_um": candidate["best_inference_point"]["max_wire_length_um"],
        },
        "limitations": [
            "The 4-bit quantized ReLU6 MNIST result uses a high-level quantized activation model, not yet a SPICE-fitted LUT.",
            "The charge ADC SPICE sweep is behavioral and does not include a transistor-level comparator or PDK parasitics.",
            "Energy is an absolute-unit proxy with configurable constants, not a post-layout extraction.",
        ],
    }
    conductance_path = ROOT / "results/tables/conductance_weight_export.json"
    if conductance_path.exists():
        rec["conductance_weight_export"] = json.loads(conductance_path.read_text())
    weak_experiments = []
    for label, path in [
        ("bounded time-to-threshold code", ROOT / "results/tables/hardware_mnist_candidate_timecode_bounded_summary.json"),
        ("bounded smooth time-to-threshold code", ROOT / "results/tables/hardware_mnist_candidate_timecode_smooth_bounded_summary.json"),
    ]:
        if path.exists():
            data = json.loads(path.read_text())
            weak_experiments.append(
                {
                    "experiment": label,
                    "accuracy": data["best_inference_point"]["accuracy"],
                    "note": "not competitive with 4-bit rectified charge on the same bounded MNIST setup",
                }
            )
    layerless_path = ROOT / "results/tables/layerless_recurrent_sheet.csv"
    if layerless_path.exists():
        layerless = pd.read_csv(layerless_path)
        weak_experiments.append(
            {
                "experiment": "layerless recurrent sheet",
                "best_accuracy": float(layerless["accuracy"].max()),
                "note": "naive parallel recurrent sheet did not learn enough to replace staged local layers",
            }
        )
    if weak_experiments:
        rec["weak_or_failed_experiments"] = weak_experiments
    robustness_path = ROOT / "results/tables/best_candidate_robustness.csv"
    if robustness_path.exists():
        rob = pd.read_csv(robustness_path)
        rec["robustness"] = {
            "baseline_accuracy": float(rob[rob["perturbation"] == "baseline"]["accuracy"].iloc[0]),
            "weight_quantization": rob[rob["perturbation"] == "weight_quantization"][
                ["severity", "accuracy"]
            ].to_dict(orient="records"),
            "conductance_drift": rob[rob["perturbation"] == "conductance_drift"][
                ["severity", "accuracy"]
            ].to_dict(orient="records"),
            "main_failure_modes": [
                "2-3 bit weight quantization without quantization-aware retraining",
                "large stuck-zero weight fractions near 10%",
                "large activation offset mismatch near 0.2 normalized units",
            ],
        }
    training_cycle_path = ROOT / "results/tables/training_cycle_energy.json"
    if training_cycle_path.exists():
        rec["training_cycle_energy"] = json.loads(training_cycle_path.read_text())
    spice_training_path = ROOT / "results/tables/spice_mnist_training_best.json"
    if spice_training_path.exists():
        rec["spice_only_mnist_training"] = json.loads(spice_training_path.read_text())["best"]
    frontier_path = ROOT / "results/tables/mnist_time_accuracy_frontier_summary.json"
    pareto_path = ROOT / "results/tables/mnist_time_accuracy_pareto.csv"
    if frontier_path.exists() and pareto_path.exists():
        rec["mnist_time_accuracy_frontier"] = {
            **json.loads(frontier_path.read_text()),
            "pareto_rows": pd.read_csv(pareto_path).to_dict(orient="records"),
        }

    out_json = ROOT / "results/tables/recommended_hardware_design.json"
    out_json.write_text(json.dumps(rec, indent=2) + "\n")
    out_md = ROOT / "results/recommended_hardware_design.md"
    lines = [
        "# Recommended Hardware MNIST Design",
        "",
        f"Architecture: `{rec['recommended_architecture']}`",
        f"Activation: {rec['recommended_activation']}",
        "",
        "## Full MNIST Result",
        "",
        f"- Accuracy: {candidate['best_inference_point']['accuracy']:.4f}",
        f"- Inference energy proxy: {candidate['best_inference_point']['inference_j']:.3e} J",
        f"- Stored weights: {candidate['best_inference_point']['hardware_parameter_count']:,}",
        f"- Synapse instances: {candidate['best_inference_point']['synapse_instances']:,}",
        f"- Max local wire: {candidate['best_inference_point']['max_wire_length_um']:.1f} um",
        "",
        "## Bounded Activation Sweep Best Hardware Mode",
        "",
        f"- Design: {best_value['activation_design']}",
        f"- Accuracy: {best_value['accuracy']:.4f}",
        f"- Inference energy proxy: {best_value['inference_j']:.3e} J",
        "",
        "## SPICE Charge ADC Evidence",
        "",
        f"- Best swept code span config: {charge_summary}",
        "",
        "## Prototype Tile",
        "",
    ]
    for key, val in rec["prototype_tile"].items():
        lines.append(f"- {key}: {val}")
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in rec["limitations"]]
    if "conductance_weight_export" in rec:
        c = rec["conductance_weight_export"]
        lines += [
            "",
            "## Conductance Weight Export",
            "",
            f"- Mapping: {c['mapping']}",
            f"- Stored weights: {c['total_stored_weights']:,}",
            f"- Differential conductance cells: {c['total_differential_conductance_cells']:,}",
            f"- Conductance range: {c['g_min_siemens']:.2e} S to {c['g_max_siemens']:.2e} S",
            f"- Array file: `{c['npz']}`",
        ]
    if weak_experiments:
        lines += ["", "## Weak Or Failed Experiments", ""]
        for item in weak_experiments:
            acc = item.get("accuracy", item.get("best_accuracy"))
            lines.append(f"- {item['experiment']}: accuracy {acc:.4f}; {item['note']}")
    if "robustness" in rec:
        lines += [
            "",
            "## Robustness",
            "",
            f"- Baseline checkpoint accuracy: {rec['robustness']['baseline_accuracy']:.4f}",
            "- 6-8 bit weight quantization is essentially lossless in the current test.",
            "- 4 bit weight quantization remains usable but drops to about 94.4%.",
            "- 2-3 bit weights, 10% stuck-zero weights, and large offset mismatch need retraining or compensation.",
        ]
    if "training_cycle_energy" in rec:
        t = rec["training_cycle_energy"]
        lines += [
            "",
            "## Training Data Cycling",
            "",
            f"- Sample load cycles for 60k MNIST over {t['training_data_cycling']['epochs']} epochs: {t['training_data_cycling']['sample_load_cycles']:,}",
            "- Training still needs a coarse sample sequencer or local handshake, even if tile internals are self-timed.",
            f"- Programmable weight update energy proxy: {t['programmable_weight_updates']['total_training_update_energy_j']:.3e} J",
            "- Weights are programmable conductance states, not hard-wired metal.",
        ]
    if "spice_only_mnist_training" in rec:
        s = rec["spice_only_mnist_training"]
        lines += [
            "",
            "## SPICE-Only MNIST Training",
            "",
            f"- Netlist: `{s['netlist']}`",
            f"- Eval netlist: `{s['eval_netlist']}`",
            f"- Dataset: {s['train_samples']} train / {s['test_samples']} held-out MNIST samples, downsampled to {s['image_size']}x{s['image_size']}",
            f"- Classes: {s['classes']}",
            f"- Train accuracy: {s['train_accuracy']:.4f}",
            f"- Held-out accuracy: {s['heldout_test_accuracy']:.4f}",
            "- Forward pass, class-error backward pass, and update currents are executed inside ngspice.",
        ]
    if "mnist_time_accuracy_frontier" in rec:
        f = rec["mnist_time_accuracy_frontier"]
        best = f["best_accuracy"]
        fast = f["best_accuracy_per_second"]
        lines += [
            "",
            "## MNIST Time/Accuracy Frontier",
            "",
            f"- Best bounded accuracy: {best['accuracy']:.4f} with `{best['family']}` channels `{best['channels']}` for {best['epochs']} epochs.",
            f"- Best accuracy per wall-second: {fast['accuracy']:.4f} with `{fast['family']}` channels `{fast['channels']}` for {fast['epochs']} epochs.",
            f"- Pareto rows: {f['pareto_count']}",
            f"- Trials table: `{f['raw_trials']}`",
            f"- Pareto table: `{f['pareto']}`",
        ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
