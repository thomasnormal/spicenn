from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    candidate = json.loads((ROOT / "results/tables/hardware_mnist_candidate_summary.json").read_text())
    conductance = json.loads((ROOT / "results/tables/conductance_weight_export.json").read_text())
    full = candidate["best_inference_point"]

    train_examples = 60_000
    epochs = int(full["epochs"])
    train_cycles = train_examples * epochs
    programmable_cells = conductance["total_differential_conductance_cells"]
    stored_weights = conductance["total_stored_weights"]

    # Conservative proxy constants; these should be replaced by PDK/device data.
    e_program_pulse_j = 100e-15
    pulses_per_weight_update = 2  # differential pair balance / verify-adjust abstraction
    active_update_fraction = 0.05  # sparse updates after thresholding local gradients
    e_input_load_per_pixel_j = 1e-15
    pixels = 28 * 28

    update_pulses_per_example = stored_weights * active_update_fraction * pulses_per_weight_update
    update_energy_per_example_j = update_pulses_per_example * e_program_pulse_j
    input_cycle_energy_per_example_j = pixels * e_input_load_per_pixel_j
    total_training_update_energy_j = train_cycles * update_energy_per_example_j
    total_input_cycle_energy_j = train_cycles * input_cycle_energy_per_example_j

    rec = {
        "training_data_cycling": {
            "train_examples": train_examples,
            "epochs": epochs,
            "sample_load_cycles": train_cycles,
            "pixels_per_sample": pixels,
            "input_load_energy_per_pixel_j": e_input_load_per_pixel_j,
            "input_cycle_energy_per_example_j": input_cycle_energy_per_example_j,
            "total_input_cycle_energy_j": total_input_cycle_energy_j,
            "need_coarse_clock_or_handshake": True,
            "cycle_phases": [
                "load image/pixel rows or sensor patch",
                "forward integrate/sample through local tiles",
                "present target/error signal during training",
                "apply local conductance programming pulses",
                "verify or settle before next sample if device requires it",
            ],
        },
        "programmable_weight_updates": {
            "stored_weights": stored_weights,
            "differential_conductance_cells": programmable_cells,
            "e_program_pulse_j": e_program_pulse_j,
            "pulses_per_weight_update": pulses_per_weight_update,
            "active_update_fraction": active_update_fraction,
            "update_pulses_per_example": update_pulses_per_example,
            "update_energy_per_example_j": update_energy_per_example_j,
            "total_training_update_energy_j": total_training_update_energy_j,
            "not_hardwired": True,
            "physical_meaning": "Weights are programmable conductance states; metal routing supplies local connectivity only.",
        },
    }
    out = ROOT / "results/tables/training_cycle_energy.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()

