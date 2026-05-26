from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def test_device_mnist01_block_script_help_runs_from_repo_root() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_device_mnist01_block_training.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--block-size" in proc.stdout
    assert "--stride" in proc.stdout
    assert "--channels" in proc.stdout
    assert "--target-polarity" in proc.stdout
    assert "--input-rail-mode" in proc.stdout
    assert "--complement-rail-scale" in proc.stdout
    assert "--readout-gradient-width" in proc.stdout
    assert "--hidden-error-width" in proc.stdout
    assert "--hidden-credit-mode" in proc.stdout
    assert "--readout-feedback-restore-width" in proc.stdout
    assert "--error-signal-mode" in proc.stdout
    assert "--error-restore-width" in proc.stdout
    assert "--hidden-update-width" in proc.stdout
    assert "--hidden-weight-write-width" in proc.stdout
    assert "--hidden-weight-leak-resistance" in proc.stdout
    assert "--hidden-weight-positive-ref" in proc.stdout
    assert "--hidden-weight-negative-ref" in proc.stdout
    assert "--hidden-activation-width" in proc.stdout
    assert "--hidden-input-residual-width" in proc.stdout
    assert "--hidden-stack-shunt-resistance" in proc.stdout
    assert "--hidden-stack-parasitic-capacitance" in proc.stdout
    assert "--hidden-activation-model" in proc.stdout
    assert "--hidden-polarity-init" in proc.stdout
    assert "--readout-forward-model" in proc.stdout
    assert "--learning-activation-gate-model" in proc.stdout
    assert "--readout-weight-leak-resistance" in proc.stdout
    assert "--readout-stack-shunt-resistance" in proc.stdout
    assert "--readout-stack-parasitic-capacitance" in proc.stdout
    assert "--activation-competition-width" in proc.stdout
    assert "--score-activity-inhibition-width" in proc.stdout
    assert "--output-bias" in proc.stdout
    assert "--output-bias-apply-scale" in proc.stdout
    assert "--output-bias-leak-resistance" in proc.stdout
    assert "--score-mode" in proc.stdout
    assert "--measurement-detail" in proc.stdout
    assert "--readout-forward-width" in proc.stdout
    assert "--phase-time-scale" in proc.stdout
    assert "--input-voltage-jitter-sigma" in proc.stdout
    assert "--phase-jitter-sigma-ns" in proc.stdout
    assert "--passive-mismatch-sigma" in proc.stdout
    assert "--state-ic-mismatch-sigma" in proc.stdout
    assert "--perturbation-seed" in proc.stdout
    assert "--hidden-bias-positive-init" in proc.stdout
    assert "--assert-nonbehavioral" in proc.stdout
    assert "--output-differential-stage" in proc.stdout
    assert "--output-score-pullup-width" in proc.stdout
    assert "--output-scoren-pulldown-width" in proc.stdout
    assert "--output-latch-capacitance" in proc.stdout
    assert "--output-decision-stage" in proc.stdout
    assert "--output-decision-ref" in proc.stdout
    assert "--output-decision-ref-source" in proc.stdout
    assert "--output-decision-ref-resistance" in proc.stdout
    assert "--output-decision-ref-capacitance" in proc.stdout
    assert "--output-decision-ref-write-width" in proc.stdout
    assert "--output-decision-pullup-width" in proc.stdout
    assert "--output-decision-pulldown-width" in proc.stdout
    assert "--output-decision-threshold" in proc.stdout
    assert "--continuous-final-eval" in proc.stdout
    assert "--skip-initial-eval" in proc.stdout
    assert "--hidden-forward-topology" in proc.stdout


def test_block_topology_matches_target_10x10_b4_stride2_c2_shape() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    blocks, feature_count = block.block_topology(10, 4, 2, 2)

    assert len(blocks) == 16
    assert feature_count == 32
    assert len(blocks[0]) == 16
    assert blocks[0] == [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
    assert blocks[-1] == [66, 67, 68, 69, 76, 77, 78, 79, 86, 87, 88, 89, 96, 97, 98, 99]


def test_block_training_schedule_can_disable_writes_for_eval_tail() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    phases = block.block_repeated_phases(3, training_enabled=[True, False, True], phase_time_scale=1.0)

    assert "3.25n" in phases
    assert "35.25n" in phases
    assert "19.25n" not in phases
    assert "Vfwd fwd 0 PWL" in phases
    assert "15n" in phases
    assert "31n" in phases
    with pytest.raises(ValueError, match="training schedule length"):
        block.block_repeated_phases(2, training_enabled=[True], phase_time_scale=1.0)


def test_block_training_phase_jitter_is_deterministic_and_changes_edges() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    nominal = block.block_repeated_phases(2, training_enabled=True, phase_time_scale=1.0)
    jittered_a = block.block_repeated_phases(
        2,
        training_enabled=True,
        phase_time_scale=1.0,
        phase_jitter_sigma_ns=0.02,
        phase_jitter_seed=123,
    )
    jittered_b = block.block_repeated_phases(
        2,
        training_enabled=True,
        phase_time_scale=1.0,
        phase_jitter_sigma_ns=0.02,
        phase_jitter_seed=123,
    )

    assert jittered_a == jittered_b
    assert jittered_a != nominal
    assert "Vfwd fwd 0 PWL" in jittered_a
    with pytest.raises(ValueError, match="phase_jitter_sigma_ns"):
        block.block_repeated_phases(1, training_enabled=True, phase_time_scale=1.0, phase_jitter_sigma_ns=-0.01)


def test_input_and_state_perturbations_are_seeded_and_bounded() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    samples = [{"x0": 0.1, "nx0": 1.19, "target": 1.1, "positive_label": 1.0}]
    jittered_a = block.perturb_input_records(samples, sigma=0.02, seed=7)
    jittered_b = block.perturb_input_records(samples, sigma=0.02, seed=7)
    jittered_c = block.perturb_input_records(samples, sigma=0.02, seed=8)

    assert jittered_a == jittered_b
    assert jittered_a != jittered_c
    assert jittered_a[0]["target"] == 1.1
    assert 0.0 <= jittered_a[0]["x0"] <= block.VDD_VALUE
    assert 0.0 <= jittered_a[0]["nx0"] <= block.VDD_VALUE

    weights = {"whp": [[0.5]], "whn": [[0.2]], "bhp": [0.5], "bhn": [0.2], "vwp": [0.52], "vwn": [0.25]}
    state_a = block.perturb_initial_state(weights, sigma=0.01, seed=9)
    state_b = block.perturb_initial_state(weights, sigma=0.01, seed=9)
    assert state_a == state_b
    assert state_a != weights
    with pytest.raises(ValueError, match="input_voltage_jitter_sigma"):
        block.perturb_input_records(samples, sigma=-0.01, seed=1)
    with pytest.raises(ValueError, match="state_ic_mismatch_sigma"):
        block.perturb_initial_state(weights, sigma=-0.01, seed=1)


def test_rows_from_measures_can_label_train_and_eval_segments() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    samples = [
        {"target": 1.1, "positive_label": 1.0, "x0": 0.5},
        {"target": 0.0, "positive_label": 0.0, "x0": 0.4},
    ]
    rows = block.rows_from_measures(
        samples,
        {"out_after_0": 0.8, "out_after_1": 0.1},
        sequence=["train", "final_eval"],
        required_rails=["x0"],
    )

    assert rows["sequence"].tolist() == ["train", "final_eval"]
    assert rows["sample_idx"].tolist() == [0, 1]
    assert rows["out_after"].tolist() == [0.8, 0.1]


def test_block_netlist_output_measurement_detail_omits_state_capacitor_measures() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        measurement_detail="outputs",
    )

    assert "\nB" not in netlist
    assert ".meas tran score_before_0" in netlist
    assert ".meas tran out_after_0" in netlist
    assert ".meas tran error_net_0" in netlist
    assert ".meas tran whp0_0_after_apply_0" not in netlist
    assert ".meas tran act0_before_0" not in netlist
    assert ".subckt signed_store p n Cp=20f Icp=0.5 Icn=0.2 Rleak=1e15" in netlist
    assert "Xwh0_0 whp0_0 whn0_0 signed_store Cp=20f" in netlist
    assert "Mwhp0_0_up_g" in netlist


def test_alternating_channel_hidden_polarity_initializes_absence_channels() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    weights = block.initial_block_weights(4, 2, 2, 2, seed=0, hidden_polarity_init="alternating-channel")
    whp = np.asarray(weights["whp"])
    whn = np.asarray(weights["whn"])
    bhp = np.asarray(weights["bhp"])
    bhn = np.asarray(weights["bhn"])

    assert np.all(whp[0::2] > whn[0::2])
    assert np.all(whp[1::2] < whn[1::2])
    assert np.all(bhp[0::2] > bhn[0::2])
    assert np.all(bhp[1::2] > bhn[1::2])


def test_random_pixel_hidden_polarity_initializes_mixed_stroke_features() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    weights = block.initial_block_weights(4, 2, 2, 3, seed=0, hidden_polarity_init="random-pixel")
    whp = np.asarray(weights["whp"])
    whn = np.asarray(weights["whn"])

    positive_pixels = whp > whn
    assert np.any(positive_pixels)
    assert np.any(~positive_pixels)
    assert all(0 < int(np.sum(row)) < row.size for row in positive_pixels)


def test_initial_block_weights_honor_output_bias_initial_conditions() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    weights = block.initial_block_weights(
        4,
        2,
        2,
        1,
        output_bias_positive_init=0.33,
        output_bias_negative_init=0.31,
    )

    assert weights["obp"] == 0.33
    assert weights["obn"] == 0.31


def test_final_block_weights_preserve_measured_output_bias() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    feature_count = 4
    block_len = 4
    row: dict[str, float] = {"obp_after_apply": 0.41, "obn_after_apply": 0.27}
    for feature in range(feature_count):
        row[f"bhp{feature}_after_apply"] = 0.50
        row[f"bhn{feature}_after_apply"] = 0.20
        row[f"vwp{feature}_after_apply"] = 0.52
        row[f"vwn{feature}_after_apply"] = 0.25
        for pix in range(block_len):
            row[f"whp{feature}_{pix}_after_apply"] = 0.60
            row[f"whn{feature}_{pix}_after_apply"] = 0.15

    weights = block.final_weights_from_rows(pd.DataFrame([row]), feature_count=feature_count, block_len=block_len)

    assert weights["obp"] == 0.41
    assert weights["obn"] == 0.27


def test_output_bias_diagnostics_flag_large_eval_drift() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    diagnostics = block.output_bias_diagnostics(
        pd.DataFrame([{"output_bias_signed_after": 0.05}, {"output_bias_signed_after": 0.67}]),
        pd.DataFrame(
            [
                {"output_bias_signed_after": 0.42},
                {"output_bias_signed_after": 0.22},
                {"output_bias_signed_after": 0.02},
            ]
        ),
        enabled=True,
        decision_threshold=0.1,
    )

    assert diagnostics["output_bias_signed_final_train"] == 0.67
    assert np.isclose(diagnostics["output_bias_signed_final_eval_drift"], -0.4)
    assert np.isclose(diagnostics["output_bias_signed_final_train_to_threshold_ratio"], 6.7)
    assert diagnostics["output_bias_state_drift_warning"] is True


def test_output_bias_diagnostics_are_empty_when_bias_disabled() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    diagnostics = block.output_bias_diagnostics(
        pd.DataFrame([{"output_bias_signed_after": 0.67}]),
        pd.DataFrame([{"output_bias_signed_after": 0.42}]),
        enabled=False,
        decision_threshold=0.1,
    )

    assert diagnostics["output_bias_signed_final_train"] is None
    assert diagnostics["output_bias_state_drift_warning"] is False


def test_adaptive_reference_diagnostics_report_state_drift() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    diagnostics = block.adaptive_reference_diagnostics(
        pd.DataFrame([{"outref_after_apply": 1.12}, {"outref_after_apply": 1.16}]),
        pd.DataFrame(
            [
                {"outref_before": 1.15},
                {"outref_before": 1.13},
                {"outref_before": 1.125},
            ]
        ),
        enabled=True,
        nominal_ref=1.13,
    )

    assert np.isclose(diagnostics["outref_final_train"], 1.16)
    assert np.isclose(diagnostics["outref_final_train_error"], 0.03)
    assert np.isclose(diagnostics["outref_final_eval_drift"], -0.025)
    assert np.isclose(diagnostics["outref_final_eval_max_abs_error"], 0.02)


def test_adaptive_reference_diagnostics_are_empty_when_disabled() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    diagnostics = block.adaptive_reference_diagnostics(
        pd.DataFrame([{"outref_after_apply": 1.16}]),
        pd.DataFrame([{"outref_before": 1.15}]),
        enabled=False,
        nominal_ref=1.13,
    )

    assert diagnostics["outref_final_train"] is None
    assert diagnostics["outref_final_eval_max_abs_error"] is None


def test_score_net_diagnostics_report_class_margin() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    diagnostics = block.score_net_diagnostics(
        pd.DataFrame(
            [
                {"positive_label": 1.0, "score_net": 0.001, "out_after": 0.01},
                {"positive_label": 0.0, "score_net": 0.003, "out_after": 0.02},
            ]
        ),
        pd.DataFrame(
            [
                {"positive_label": 1.0, "score_net": 0.014, "out_after": 0.037},
                {"positive_label": 1.0, "score_net": 0.012, "out_after": 0.038},
                {"positive_label": 0.0, "score_net": 0.011, "out_after": 0.041},
                {"positive_label": 0.0, "score_net": 0.005, "out_after": 0.039},
            ]
        ),
    )

    assert np.isclose(diagnostics["initial_eval_score_net_margin"], -0.002)
    assert np.isclose(diagnostics["final_eval_score_net_positive_mean"], 0.013)
    assert np.isclose(diagnostics["final_eval_score_net_negative_mean"], 0.008)
    assert np.isclose(diagnostics["final_eval_score_net_margin"], 0.005)
    assert np.isclose(diagnostics["final_eval_out_after_margin"], -0.0025)


def test_binary_accuracy_for_signal_can_use_circuit_decision_rail() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    rows = pd.DataFrame(
        [
            {"positive_label": 1.0, "decision_after": 1.06},
            {"positive_label": 0.0, "decision_after": 0.03},
            {"positive_label": 1.0, "decision_after": 0.92},
            {"positive_label": 0.0, "decision_after": 0.72},
        ]
    )

    assert block.binary_accuracy_for_signal(rows, signal="decision_after", threshold=0.6) == 0.75
    assert block.binary_accuracy_for_signal(
        rows,
        signal="decision_after",
        threshold=0.6,
        output_positive_when="low",
    ) == 0.25


def test_nontrivial_learning_requires_initial_eval_baseline() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    assert block.nontrivial_learning_flag(None, 1.0) is False
    assert block.nontrivial_learning_flag(0.5, 0.75) is True
    assert block.nontrivial_learning_flag(0.75, 0.75) is False


def test_threshold_window_diagnostics_report_best_output_threshold() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    rows = pd.DataFrame(
        [
            {"positive_label": 1.0, "out_after": 1.133},
            {"positive_label": 1.0, "out_after": 1.132},
            {"positive_label": 0.0, "out_after": 1.128},
            {"positive_label": 0.0, "out_after": 1.123},
        ]
    )

    diagnostics = block.threshold_window_diagnostics(rows, signal="out_after")

    assert diagnostics["out_after_best_threshold_accuracy"] == 1.0
    assert 1.128 < diagnostics["out_after_best_threshold"] < 1.132
    assert diagnostics["out_after_best_threshold_active_fraction"] == 0.5


def test_block_netlist_emits_per_pixel_trainable_caps_and_no_behavioral_sources() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        readout_gradient_width=36.0,
        hidden_error_width=40.0,
        hidden_update_width=18.0,
        hidden_weight_write_width=0.4,
        hidden_activation_width=96.0,
        hidden_activation_model="sense",
        readout_forward_width=128.0,
        phase_time_scale=2.0,
    )

    assert "\nB" not in netlist
    assert netlist.count("\nXwh") == 16
    assert netlist.count("\nXbh") == 4
    assert netlist.count("\nXvw") == 4
    assert "Vx15 x15 0 PWL" in netlist
    assert "Vx16 x16 0 PWL" not in netlist
    assert "Xwh3_3 whp3_3 whn3_3 signed_store Cp=20f" in netlist
    assert "Xbh3 bhp3 bhn3 signed_store Cp=20f" in netlist
    assert "Xvw3 vwp3 vwn3 signed_store Cp=20f" in netlist
    assert "Mhpos3_3_x vdd x15 hp3_3_0 0 NMOS" in netlist
    assert "Mhbpos3_b vdd bhp3 hbp3_0 0 NMOS" in netlist
    assert "Mghp3_3_d ghp3_3_x hdp3 ghp3_3_d 0 NSENSE W=18u" in netlist
    assert "Mgbp3_d vdd hdp3 gbp3_d 0 NSENSE W=18u" in netlist
    assert "Mhdp3_d0 vdd dp hdp3_d0 0 NSENSE W=40u" in netlist
    assert "Mgvp3_a vdd act3 gvp3_a 0 NREL W=36u" in netlist
    assert "Mbhp3_up_a bhp3_up apply bhp3 0 NREL W=0.4u" in netlist
    assert "Mrelu_h3 vdd pre3 act3 0 NSENSE W=96u" in netlist
    assert "Movpos3_a vdd act3 op3_0 0 NREL W=128u" in netlist
    assert "Movneg3_f score fwd on3_0 0 NREL W=96u" in netlist
    assert "Movpos3_f op3_1 fwd score 0 NREL" in netlist
    assert ".meas tran score_before_0 FIND V(score) AT=5.90n" in netlist
    assert ".meas tran gvp3_after_0 FIND V(gvp3) AT=18.20n" in netlist
    assert ".meas tran gvn3_after_0 FIND V(gvn3) AT=18.20n" in netlist
    assert ".tran 10p 32.00n uic" in netlist
    assert "Mrelu_o vdd score out 0 NSENSE" in netlist


def test_block_netlist_default_hidden_forward_uses_per_pixel_phase_stack() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
    )

    assert "\nB" not in netlist
    assert "Mhfpos0_phase" not in netlist
    assert "Mhfneg0_phase" not in netlist
    assert "Mhpos3_3_x vdd x15 hp3_3_0 0 NMOS" in netlist
    assert "Mhpos3_3_w hp3_3_0 whp3_3 hp3_3_1 0 NMOS" in netlist
    assert "Mhpos3_3_f hp3_3_1 fwd pre3 0 NMOS" in netlist
    assert "Mhneg3_3_f pre3 fwd hn3_3_0 0 NMOS" in netlist
    assert "Mhneg3_3_x hn3_3_0 x15 hn3_3_1 0 NMOS" in netlist
    assert "Mhneg3_3_w hn3_3_1 whn3_3 0 0 NMOS" in netlist


def test_block_netlist_can_emit_shared_phase_hidden_forward_topology() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_forward_width=3.0,
        hidden_forward_topology="shared-phase",
    )

    assert "\nB" not in netlist
    assert "Mhfpos3_phase hfp3_rail fwd pre3 0 NMOS W=12u" in netlist
    assert "Mhfneg3_phase pre3 fwd hfn3_rail 0 NMOS W=12u" in netlist
    assert "Chfp3_rail hfp3_rail 0 0.1f IC=0" in netlist
    assert "Rhfn3_rail hfn3_rail 0 1G" in netlist
    assert "Mhpos3_3_x vdd x15 hp3_3_0 0 NMOS W=3u" in netlist
    assert "Mhpos3_3_w hp3_3_0 whp3_3 hfp3_rail 0 NMOS W=3u" in netlist
    assert "Mhneg3_3_x hfn3_rail x15 hn3_3_0 0 NMOS W=2.25u" in netlist
    assert "Mhneg3_3_w hn3_3_0 whn3_3 0 0 NMOS W=2.25u" in netlist
    assert "Mhpos3_3_f" not in netlist
    assert "Mhneg3_3_f" not in netlist
    assert "hp3_3_1" not in netlist
    assert "hn3_3_1" not in netlist


def test_block_netlist_shared_phase_hidden_stack_conditioning_matches_shallow_nodes() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_forward_topology="shared-phase",
        hidden_stack_parasitic_capacitance=1e-16,
        hidden_stack_shunt_resistance=1e12,
    )

    assert "\nB" not in netlist
    assert "Chread_hp3_3_0 hp3_3_0 0 1e-16" in netlist
    assert "Rhread_hp3_3_0 hp3_3_0 0 1e+12" in netlist
    assert "Chread_hn3_3_0 hn3_3_0 0 1e-16" in netlist
    assert "Rhread_hn3_3_0 hn3_3_0 0 1e+12" in netlist
    assert "Chread_hp3_3_1" not in netlist
    assert "Rhread_hp3_3_1" not in netlist


def test_block_netlist_can_emit_always_on_hidden_forward_topology() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_forward_width=3.0,
        hidden_forward_topology="always-on",
    )

    assert "\nB" not in netlist
    assert "Mhfpos3_phase" not in netlist
    assert "Mhfneg3_phase" not in netlist
    assert "Mhpos3_3_x vdd x15 hp3_3_0 0 NMOS W=3u" in netlist
    assert "Mhpos3_3_w hp3_3_0 whp3_3 pre3 0 NMOS W=3u" in netlist
    assert "Mhneg3_3_x pre3 x15 hn3_3_0 0 NMOS W=2.25u" in netlist
    assert "Mhneg3_3_w hn3_3_0 whn3_3 0 0 NMOS W=2.25u" in netlist
    assert "Mhpos3_3_f" not in netlist
    assert "Mhneg3_3_f" not in netlist
    assert "hp3_3_1" not in netlist
    assert "hn3_3_1" not in netlist
    assert "Mhbpos3_b vdd bhp3 pre3 0 NMOS W=3u" in netlist
    assert "Mhbneg3_b pre3 bhn3 0 0 NMOS W=2.25u" in netlist
    assert "Mhbpos3_f" not in netlist
    assert "Mhbneg3_f" not in netlist


def test_block_netlist_always_on_hidden_stack_conditioning_matches_shallow_nodes() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_forward_topology="always-on",
        hidden_stack_parasitic_capacitance=1e-16,
        hidden_stack_shunt_resistance=1e12,
    )

    assert "\nB" not in netlist
    assert "Chread_hp3_3_0 hp3_3_0 0 1e-16" in netlist
    assert "Rhread_hp3_3_0 hp3_3_0 0 1e+12" in netlist
    assert "Chread_hn3_3_0 hn3_3_0 0 1e-16" in netlist
    assert "Rhread_hn3_3_0 hn3_3_0 0 1e+12" in netlist
    assert "Chread_hp3_3_1" not in netlist
    assert "Rhread_hn3_3_1" not in netlist


def test_block_netlist_can_emit_split_rail_hidden_forward_topology() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_forward_width=3.0,
        hidden_forward_topology="split-rail",
    )

    assert "\nB" not in netlist
    assert "Cpren3 pren3 0 10f IC=0" in netlist
    assert "Rpren3 pren3 0 1G" in netlist
    assert "Mreset_pren3 pren3 rstf 0 0 NMOS W=4u" in netlist
    assert ".subckt split_rail_hidden_pixel x whp whn pre pren Wp=3u Wn=2.25u" in netlist
    assert ".subckt split_rail_hidden_bias vdd bhp bhn pre pren Wp=3u Wn=2.25u" in netlist
    assert ".subckt split_rail_relu_nrel vdd pre pren act W=24u" in netlist
    assert "Xhs3_3 x15 whp3_3 whn3_3 pre3 pren3 split_rail_hidden_pixel Wp=3u Wn=2.25u" in netlist
    assert "Xhb3 vdd bhp3 bhn3 pre3 pren3 split_rail_hidden_bias Wp=3u Wn=2.25u" in netlist
    assert "Xrelu_h3 vdd pre3 pren3 act3 split_rail_relu_nrel W=24u" in netlist
    assert "Mhspos3_3" not in netlist
    assert "Mhsneg3_3" not in netlist
    assert "Mhbpos3_b vdd bhp3 pre3" not in netlist
    assert "Mhbneg3_b vdd bhn3 pren3" not in netlist
    assert "Mrelu_h3_p" not in netlist
    assert "Mrelu_h3_n" not in netlist
    assert "Mhpos3_3_x" not in netlist
    assert "Mhneg3_3_x" not in netlist
    assert "Mhpos3_3_f" not in netlist
    assert "hp3_3_0" not in netlist
    assert "hn3_3_0" not in netlist


def test_block_netlist_split_rail_has_no_per_pixel_hidden_stack_conditioning_nodes() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_forward_topology="split-rail",
        hidden_stack_parasitic_capacitance=1e-16,
        hidden_stack_shunt_resistance=1e12,
    )

    assert "\nB" not in netlist
    assert "Chread_hp3_3_0" not in netlist
    assert "Rhread_hp3_3_0" not in netlist
    assert "Chread_hn3_3_0" not in netlist
    assert "Rhread_hn3_3_0" not in netlist
    assert "Cpren3 pren3 0 10f IC=0" in netlist


def test_block_netlist_can_emit_differential_score_path_without_behavioral_sources() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
    )

    assert "\nB" not in netlist
    assert "Cscoren scoren 0 10f IC=0" in netlist
    assert "Mreset_scoren scoren rstf 0 0 NMOS" in netlist
    assert "Movneg3_f on3_1 fwd scoren 0 NREL" in netlist
    assert "Moutp vdd score out 0 NSENSE" in netlist
    assert "Moutn out scoren 0 0 NSENSE" in netlist
    assert "Mdp_sn0 vdd scoren dp_sn 0 NSENSE" in netlist
    assert "Mdn_sn1 dn_sn scoren 0 0 NSENSE" in netlist
    assert ".meas tran score_net_0 PARAM='score_before_0-scoren_before_0'" in netlist


def test_block_netlist_can_emit_readout_weighted_hidden_credit() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        hidden_credit_mode="readout-weighted",
        hidden_error_width=40.0,
    )

    assert "\nB" not in netlist
    assert "Mhdp3_d0 vdd dp hdp3_d0 0 NSENSE" not in netlist
    assert "Mhdn3_d0 vdd dn hdn3_d0 0 NSENSE" not in netlist
    assert "Mhdp3_pv_e vdd dp hdp3_pv_e 0 NSENSE W=40u" in netlist
    assert "Mhdp3_pv_w hdp3_pv_e vwp3 hdp3_pv_w 0 NSENSE W=40u" in netlist
    assert "Mhdp3_pv_a hdp3_pv_w act3 hdp3_pv_a 0 NREL W=40u" in netlist
    assert "Mhdp3_pv_b hdp3_pv_a bwd hdp3 0 NMOS W=40u" in netlist
    assert "Mhdp3_nv_e vdd dn hdp3_nv_e 0 NSENSE W=40u" in netlist
    assert "Mhdp3_nv_w hdp3_nv_e vwn3 hdp3_nv_w 0 NSENSE W=40u" in netlist
    assert "Mhdn3_pv_e vdd dp hdn3_pv_e 0 NSENSE W=40u" in netlist
    assert "Mhdn3_pv_w hdn3_pv_e vwn3 hdn3_pv_w 0 NSENSE W=40u" in netlist
    assert "Mhdn3_nv_e vdd dn hdn3_nv_e 0 NSENSE W=40u" in netlist
    assert "Mhdn3_nv_w hdn3_nv_e vwp3 hdn3_nv_w 0 NSENSE W=40u" in netlist


def test_block_netlist_can_emit_restored_readout_weighted_hidden_credit() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        hidden_credit_mode="readout-restored",
        hidden_error_width=40.0,
        readout_feedback_restore_width=6.0,
    )

    assert "\nB" not in netlist
    assert "Crvwp3 rvwp3 0 4f IC=0" in netlist
    assert "Crvwn3 rvwn3 0 4f IC=0" in netlist
    assert "Mprecharge_rvwp3 vdd rstgn rvwp3 vdd PMOS W=4u" in netlist
    assert "Mprecharge_rvwn3 vdd rstgn rvwn3 vdd PMOS W=4u" in netlist
    assert "Mrvwp3_p vdd rvwn3 rvwp3 vdd PMOS W=6u" in netlist
    assert "Mrvwn3_p vdd rvwp3 rvwn3 vdd PMOS W=6u" in netlist
    assert "Mrvwp3_n rvwp3 vwn3 rvw_src3 0 NSENSE W=6u" in netlist
    assert "Mrvwn3_n rvwn3 vwp3 rvw_src3 0 NSENSE W=6u" in netlist
    assert "Mrvw3_tail rvw_src3 err 0 0 NMOS W=6u" in netlist
    assert "Mhdp3_pv_w hdp3_pv_e rvwp3 hdp3_pv_w 0 NSENSE W=40u" in netlist
    assert "Mhdp3_nv_w hdp3_nv_e rvwn3 hdp3_nv_w 0 NSENSE W=40u" in netlist
    assert "Mhdn3_pv_w hdn3_pv_e rvwn3 hdn3_pv_w 0 NSENSE W=40u" in netlist
    assert "Mhdn3_nv_w hdn3_nv_e rvwp3 hdn3_nv_w 0 NSENSE W=40u" in netlist


def test_block_netlist_can_emit_restored_error_rails_for_learning() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        hidden_credit_mode="readout-restored",
        error_signal_mode="restored",
        error_restore_width=7.0,
        output_decision_stage="ref-precharged-latched",
        output_differential_stage="latched",
    )

    assert "\nB" not in netlist
    assert "Cedp edp 0 8f IC=0" in netlist
    assert "Cedn edn 0 8f IC=0" in netlist
    assert "Mprecharge_edp vdd rstgn edp vdd PMOS W=4u" in netlist
    assert "Mprecharge_edn vdd rstgn edn vdd PMOS W=4u" in netlist
    assert "Merrstore_p vdd edn edp vdd PMOS W=7u" in netlist
    assert "Merrstore_n vdd edp edn vdd PMOS W=7u" in netlist
    assert "Merrstore_ep edp dn errstore_src 0 NSENSE W=7u" in netlist
    assert "Merrstore_en edn dp errstore_src 0 NSENSE W=7u" in netlist
    assert "Merrstore_tail errstore_src err 0 0 NMOS W=7u" in netlist
    assert "Mhdp3_pv_e vdd edp hdp3_pv_e 0 NSENSE" in netlist
    assert "Mhdp3_nv_e vdd edn hdp3_nv_e 0 NSENSE" in netlist
    assert "Mhdn3_pv_e vdd edp hdn3_pv_e 0 NSENSE" in netlist
    assert "Mhdn3_nv_e vdd edn hdn3_nv_e 0 NSENSE" in netlist
    assert "Mgvp3_d gvp3_a edp gvp3_d 0 NSENSE" in netlist
    assert "Mgvn3_d gvn3_a edn gvn3_d 0 NSENSE" in netlist
    assert "Mdec_pc_n decision outref dec_src 0 NSENSE" in netlist
    assert ".meas tran edp_after_0 FIND V(edp)" in netlist
    assert ".meas tran error_restored_diff_0 PARAM='edp_after_0-edn_after_0'" in netlist


def test_block_netlist_can_restore_only_hidden_error_transport() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        hidden_credit_mode="readout-restored",
        error_signal_mode="restored-hidden",
        output_differential_stage="latched",
    )

    assert "\nB" not in netlist
    assert "Merrstore_ep edp dn errstore_src 0 NSENSE" in netlist
    assert "Mhdp3_pv_e vdd edp hdp3_pv_e 0 NSENSE" in netlist
    assert "Mhdn3_nv_e vdd edn hdn3_nv_e 0 NSENSE" in netlist
    assert "Mgvp3_d gvp3_a dp gvp3_d 0 NSENSE" in netlist
    assert "Mgvn3_d gvn3_a dn gvn3_d 0 NSENSE" in netlist


def test_readout_weighted_hidden_credit_requires_differential_score() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1

    with pytest.raises(ValueError, match="requires differential score_mode"):
        block.block_netlist(
            [sample],
            weights,
            image_size=image_size,
            block_size=2,
            stride=2,
            channels=1,
            training_enabled=True,
            hidden_credit_mode="readout-restored",
            score_mode="single-ended",
        )


def test_block_netlist_can_tune_differential_output_sense_widths() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_score_pullup_width=48.0,
        output_scoren_pulldown_width=96.0,
    )

    assert "\nB" not in netlist
    assert "Moutp vdd score out 0 NSENSE W=48u" in netlist
    assert "Moutn out scoren 0 0 NSENSE W=96u" in netlist


def test_block_netlist_can_emit_score_gated_differential_output_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="score-gated",
        output_score_pullup_width=48.0,
        output_scoren_pulldown_width=96.0,
    )

    assert "\nB" not in netlist
    assert "Moutp_gate vdd scoren outp_gate vdd PMOS W=48u" in netlist
    assert "Moutp_score outp_gate score out 0 NSENSE W=48u" in netlist
    assert "Moutn out scoren 0 0 NSENSE W=96u" in netlist
    assert "Moutp vdd score out 0 NSENSE" not in netlist


def test_block_netlist_can_emit_latched_differential_output_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="latched",
        output_score_pullup_width=48.0,
        output_scoren_pulldown_width=96.0,
    )

    assert "\nB" not in netlist
    assert "Coutn outn 0 20f IC=0" in netlist
    assert "Mreset_outn outn rstf 0 0 NMOS W=4u" in netlist
    assert "Moutlat_p_out vdd outn out vdd PMOS W=48u" in netlist
    assert "Moutlat_n_out out scoren outlat_src 0 NSENSE W=96u" in netlist
    assert "Moutlat_n_outn outn score outlat_src 0 NSENSE W=96u" in netlist
    assert "Moutlat_tail outlat_src fwd 0 0 NMOS W=96u" in netlist
    assert ".meas tran out_diff_0 PARAM='out_after_0-outn_after_0'" in netlist


def test_block_netlist_can_size_latched_output_storage_capacitance() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="latched",
        output_latch_capacitance=200e-15,
    )

    assert "Cout out 0 200f IC=0" in netlist
    assert "Coutn outn 0 200f IC=0" in netlist


def test_block_netlist_can_emit_reference_latched_decision_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="latched",
        output_decision_stage="ref-latched",
        output_decision_ref=1.09,
        output_decision_pullup_width=48.0,
        output_decision_pulldown_width=96.0,
    )

    assert "\nB" not in netlist
    assert "Voutref outref 0 1.09" in netlist
    assert "Cdecision decision 0 20f IC=0" in netlist
    assert "Cdecisionn decisionn 0 20f IC=0" in netlist
    assert "Mreset_decision decision rstf 0 0 NMOS W=4u" in netlist
    assert "Mreset_decisionn decisionn rstf 0 0 NMOS W=4u" in netlist
    assert "Mdec_p decision decisionn vdd vdd PMOS W=48u" in netlist
    assert "Mdecn_p decisionn decision vdd vdd PMOS W=48u" in netlist
    assert "Mdec_n decision outref dec_src 0 NSENSE W=96u" in netlist
    assert "Mdecn_n decisionn out dec_src 0 NSENSE W=96u" in netlist
    assert "Mdec_tail dec_src dec 0 0 NMOS W=96u" in netlist
    assert "Vdec dec 0 PWL" in netlist
    assert ".meas tran decision_after_0 FIND V(decision) AT=15.50n" in netlist
    assert ".meas tran decisionn_after_0 FIND V(decisionn) AT=15.50n" in netlist
    assert ".meas tran decision_diff_0 PARAM='decision_after_0-decisionn_after_0'" in netlist


def test_block_netlist_can_emit_precharged_reference_latched_decision_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="latched",
        output_decision_stage="ref-precharged-latched",
        output_decision_ref=1.13,
        output_decision_pullup_width=48.0,
        output_decision_pulldown_width=96.0,
    )

    assert "\nB" not in netlist
    assert "Vrstfn rstfn 0 PWL" in netlist
    assert "Voutref outref 0 1.13" in netlist
    assert "Cdecision decision 0 20f IC=0" in netlist
    assert "Cdecisionn decisionn 0 20f IC=0" in netlist
    assert "Mreset_decision decision rstf 0 0 NMOS" not in netlist
    assert "Mreset_decisionn decisionn rstf 0 0 NMOS" not in netlist
    assert "Mprecharge_decision vdd rstfn decision vdd PMOS W=4u" in netlist
    assert "Mprecharge_decisionn vdd rstfn decisionn vdd PMOS W=4u" in netlist
    assert "Mdec_pc_p vdd decisionn decision vdd PMOS W=48u" in netlist
    assert "Mdecn_pc_p vdd decision decisionn vdd PMOS W=48u" in netlist
    assert "Mdec_pc_n decision outref dec_src 0 NSENSE W=96u" in netlist
    assert "Mdecn_pc_n decisionn out dec_src 0 NSENSE W=96u" in netlist
    assert "Mdec_pc_tail dec_src dec 0 0 NMOS W=96u" in netlist
    assert ".meas tran decision_diff_0 PARAM='decision_after_0-decisionn_after_0'" in netlist


def test_block_netlist_can_emit_passive_decision_reference_divider() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="latched",
        output_decision_stage="ref-precharged-latched",
        output_decision_ref=0.95,
        output_decision_ref_source="divider",
        output_decision_ref_resistance=1.2e6,
    )

    assert "\nB" not in netlist
    assert "Voutref outref 0" not in netlist
    assert "Routref_top vdd outref 250000" in netlist
    assert "Routref_bot outref 0 950000" in netlist
    assert "Coutref outref 0 1f IC=0" in netlist


def test_passive_mismatch_perturbs_reference_divider_deterministically() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1

    def emit(seed: int) -> str:
        return block.block_netlist(
            [sample],
            weights,
            image_size=image_size,
            block_size=2,
            stride=2,
            channels=1,
            training_enabled=True,
            score_mode="differential",
            output_differential_stage="latched",
            output_decision_stage="ref-precharged-latched",
            output_decision_ref=0.95,
            output_decision_ref_source="divider",
            output_decision_ref_resistance=1.2e6,
            passive_mismatch_sigma=0.01,
            passive_mismatch_seed=seed,
        )

    netlist_a = emit(5)
    netlist_b = emit(5)
    netlist_c = emit(6)

    assert "\nB" not in netlist_a
    assert netlist_a == netlist_b
    assert netlist_a != netlist_c
    assert "Routref_top vdd outref 250000" not in netlist_a
    assert "Routref_bot outref 0 950000" not in netlist_a


def test_block_netlist_can_emit_adaptive_decision_reference_state() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="latched",
        output_decision_stage="ref-precharged-latched",
        output_decision_ref=0.95,
        output_decision_ref_source="adaptive",
        output_decision_ref_resistance=1.2e6,
        output_decision_ref_capacitance=40e-15,
        output_decision_ref_write_width=0.5,
    )

    assert "\nB" not in netlist
    assert "Voutref outref 0" not in netlist
    assert "Coutref outref 0 40f IC=0.95" in netlist
    assert "Routref_top vdd outref 250000" in netlist
    assert "Routref_bot outref 0 950000" in netlist
    assert "Coutref_raise_gate outref_raise_gate 0 4f IC=1.2" in netlist
    assert "Routref_raise_gate outref_raise_gate vdd 50k" in netlist
    assert "Moutref_raise_gate outref_raise_gate dn 0 0 NSENSE W=16u" in netlist
    assert "Moutref_raise_p0 vdd outref_raise_gate outref_raise vdd PMOS W=4u" in netlist
    assert "Moutref_raise_p1 outref_raise applyn outref vdd PMOS W=4u" in netlist
    assert "Moutref_lower_a outref apply outref_lower 0 NREL W=0.5u" in netlist
    assert "Moutref_lower_g outref_lower dp 0 0 NSENSE W=0.5u" in netlist
    assert ".meas tran outref_before_0 FIND V(outref)" in netlist
    assert ".meas tran d_outref_0 PARAM='outref_after_apply_0-outref_before_0'" in netlist


def test_decision_reference_divider_rejects_invalid_reference() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    with pytest.raises(ValueError, match="between 0 and 1.2"):
        block.decision_ref_divider_resistances(1.2, 1e6)
    with pytest.raises(ValueError, match="positive"):
        block.decision_ref_divider_resistances(0.8, 0.0)


def test_adaptive_decision_reference_rejects_invalid_state_devices() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1

    with pytest.raises(ValueError, match="output_decision_ref_capacitance"):
        block.block_netlist(
            [sample],
            weights,
            image_size=image_size,
            block_size=2,
            stride=2,
            channels=1,
            training_enabled=True,
            score_mode="differential",
            output_differential_stage="latched",
            output_decision_stage="ref-precharged-latched",
            output_decision_ref_source="adaptive",
            output_decision_ref_capacitance=0.0,
        )
    with pytest.raises(ValueError, match="output_decision_ref_write_width"):
        block.block_netlist(
            [sample],
            weights,
            image_size=image_size,
            block_size=2,
            stride=2,
            channels=1,
            training_enabled=True,
            score_mode="differential",
            output_differential_stage="latched",
            output_decision_stage="ref-precharged-latched",
            output_decision_ref_source="adaptive",
            output_decision_ref_write_width=0.0,
        )
    with pytest.raises(ValueError, match="requires a reference decision stage"):
        block.block_netlist(
            [sample],
            weights,
            image_size=image_size,
            block_size=2,
            stride=2,
            channels=1,
            training_enabled=True,
            score_mode="differential",
            output_differential_stage="latched",
            output_decision_stage="diff-latched",
            output_decision_ref_source="adaptive",
        )


def test_block_netlist_can_emit_reference_preamp_latched_decision_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="latched",
        output_decision_stage="ref-preamp-latched",
        output_decision_ref=1.13,
        output_decision_pullup_width=64.0,
        output_decision_pulldown_width=128.0,
    )

    assert "\nB" not in netlist
    assert "Voutref outref 0 1.13" in netlist
    assert "Cdecision_pre decision_pre 0 10f IC=0" in netlist
    assert "Cdecisionn_pre decisionn_pre 0 10f IC=0" in netlist
    assert "Mreset_decision_pre decision_pre rstf 0 0 NMOS W=4u" in netlist
    assert "Mreset_decisionn_pre decisionn_pre rstf 0 0 NMOS W=4u" in netlist
    assert "Mdecpre_lp vdd decision_pre decision_pre vdd PMOS W=64u" in netlist
    assert "Mdecpre_ln vdd decisionn_pre decisionn_pre vdd PMOS W=64u" in netlist
    assert "Mdecpre_ref decision_pre outref decpre_src 0 NSENSE W=128u" in netlist
    assert "Mdecpre_out decisionn_pre out decpre_src 0 NSENSE W=128u" in netlist
    assert "Mdecpre_tail decpre_src dec 0 0 NMOS W=128u" in netlist
    assert "Mdec_n decision decisionn_pre dec_src 0 NSENSE W=128u" in netlist
    assert "Mdecn_n decisionn decision_pre dec_src 0 NSENSE W=128u" in netlist
    assert "Mdec_n decision outref dec_src 0 NSENSE" not in netlist
    assert ".meas tran decision_after_0 FIND V(decision) AT=15.50n" in netlist


def test_block_netlist_can_emit_differential_latched_decision_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        score_mode="differential",
        output_differential_stage="latched",
        output_decision_stage="diff-latched",
        output_decision_pullup_width=48.0,
        output_decision_pulldown_width=96.0,
    )

    assert "\nB" not in netlist
    assert "Voutref" not in netlist
    assert "Cdecision decision 0 20f IC=0" in netlist
    assert "Mdec_n decision outn dec_src 0 NSENSE W=96u" in netlist
    assert "Mdecn_n decisionn out dec_src 0 NSENSE W=96u" in netlist
    assert "Mdec_tail dec_src dec 0 0 NMOS W=96u" in netlist
    assert "Mdec_n decision outref dec_src 0 NSENSE" not in netlist
    assert ".meas tran decision_after_0 FIND V(decision) AT=15.50n" in netlist


def test_block_netlist_can_emit_ratioed_inverter_decision_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        output_decision_stage="ratio-inverter",
        output_decision_pullup_width=192.0,
        output_decision_pulldown_width=1.0,
    )

    assert "\nB" not in netlist
    assert "Voutref" not in netlist
    assert "Mdec_inv1_p decisionn out vdd vdd PMOS W=192u" in netlist
    assert "Mdec_inv1_n decisionn out 0 0 NMOS W=1u" in netlist
    assert "Mdec_inv2_p decision decisionn vdd vdd PMOS W=192u" in netlist
    assert "Mdec_inv2_n decision decisionn 0 0 NMOS W=1u" in netlist
    assert ".meas tran decision_after_0 FIND V(decision) AT=15.50n" in netlist


def test_block_netlist_can_emit_stacked_inverter_decision_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        output_decision_stage="stacked-inverter",
        output_decision_pullup_width=192.0,
        output_decision_pulldown_width=1.0,
    )

    assert "\nB" not in netlist
    assert "Voutref" not in netlist
    assert "Cdec_stack0 dec_stack0 0 0.1f IC=0" in netlist
    assert "Cdec_stack1 dec_stack1 0 0.1f IC=0" in netlist
    assert "Mdec_stack_p decisionn out vdd vdd PMOS W=192u" in netlist
    assert "Mdec_stack_n0 decisionn out dec_stack0 0 NMOS W=1u" in netlist
    assert "Mdec_stack_n1 dec_stack0 out dec_stack1 0 NMOS W=1u" in netlist
    assert "Mdec_stack_n2 dec_stack1 out 0 0 NMOS W=1u" in netlist
    assert "Mdec_buf_p decision decisionn vdd vdd PMOS W=192u" in netlist
    assert "Mdec_buf_n decision decisionn 0 0 NMOS W=1u" in netlist
    assert ".meas tran decision_after_0 FIND V(decision) AT=15.50n" in netlist


def test_block_netlist_can_emit_shifted_inverter_decision_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        output_decision_stage="shift-inverter",
        output_decision_pullup_width=32.0,
        output_decision_pulldown_width=4.0,
    )

    assert "\nB" not in netlist
    assert "Cdec_mid dec_mid 0 20f IC=0" in netlist
    assert "Mdec_shift vdd out decisionn 0 NMOS W=4u" in netlist
    assert "Mdec_inv1_p dec_mid decisionn vdd vdd PMOS W=32u" in netlist
    assert "Mdec_inv1_n dec_mid decisionn 0 0 NMOS W=4u" in netlist
    assert "Mdec_inv2_p decision dec_mid vdd vdd PMOS W=32u" in netlist
    assert "Mdec_inv2_n decision dec_mid 0 0 NMOS W=4u" in netlist
    assert ".meas tran decision_after_0 FIND V(decision) AT=15.50n" in netlist


def test_latched_decision_stage_requires_latched_output_stage() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    with pytest.raises(ValueError, match="requires latched output_differential_stage"):
        block.block_netlist(
            [sample],
            weights,
            image_size=image_size,
            block_size=2,
            stride=2,
            channels=1,
            training_enabled=True,
            score_mode="differential",
            output_differential_stage="simple",
            output_decision_stage="ref-latched",
        )


def test_non_simple_output_stage_requires_differential_score_mode() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    with pytest.raises(ValueError, match="requires differential score_mode"):
        block.block_netlist(
            [sample],
            weights,
            image_size=image_size,
            block_size=2,
            stride=2,
            channels=1,
            training_enabled=True,
            score_mode="single-ended",
            output_differential_stage="latched",
        )


def test_block_netlist_can_emit_low_threshold_readout_score_stack() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        readout_forward_model="sense",
        score_mode="differential",
    )

    assert "\nB" not in netlist
    assert "Movpos3_a vdd act3 op3_0 0 NSENSE W=64u" in netlist
    assert "Movpos3_w op3_0 vwp3 op3_1 0 NSENSE W=64u" in netlist
    assert "Movneg3_a vdd act3 on3_0 0 NSENSE W=48u" in netlist
    assert "Movneg3_f on3_1 fwd scoren 0 NSENSE W=48u" in netlist


def test_block_netlist_can_emit_low_threshold_learning_activation_gates() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        learning_activation_gate_model="sense",
    )

    assert "\nB" not in netlist
    assert "Mhdp3_d1 hdp3_d0 act3 hdp3_d1 0 NSENSE W=32u" in netlist
    assert "Mhdn3_d1 hdn3_d0 act3 hdn3_d1 0 NSENSE W=32u" in netlist
    assert "Mgvp3_a vdd act3 gvp3_a 0 NSENSE W=24u" in netlist
    assert "Mgvn3_a vdd act3 gvn3_a 0 NSENSE W=24u" in netlist


def test_block_netlist_can_emit_passive_readout_weight_leak() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        readout_weight_leak_resistance=5e6,
        readout_weight_positive_ref=0.52,
        readout_weight_negative_ref=0.25,
    )

    assert "\nB" not in netlist
    assert "Vvwp_ref vwp_ref 0 0.52" in netlist
    assert "Vvwn_ref vwn_ref 0 0.25" in netlist
    assert "Rvwp0_leak vwp0 vwp_ref 5e+06" in netlist
    assert "Rvwn3_leak vwn3 vwn_ref 5e+06" in netlist


def test_block_netlist_can_emit_passive_hidden_weight_and_bias_leak() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_weight_leak_resistance=4e6,
        hidden_weight_positive_ref=0.49,
        hidden_weight_negative_ref=0.21,
    )

    assert "\nB" not in netlist
    assert "Vwhp_ref whp_ref 0 0.49" in netlist
    assert "Vwhn_ref whn_ref 0 0.21" in netlist
    assert "Rwhp0_0_leak whp0_0 whp_ref 4e+06" in netlist
    assert "Rwhn3_3_leak whn3_3 whn_ref 4e+06" in netlist
    assert "Rbhp0_leak bhp0 whp_ref 4e+06" in netlist
    assert "Rbhn3_leak bhn3 whn_ref 4e+06" in netlist


def test_block_netlist_can_emit_passive_readout_stack_shunts() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        readout_stack_shunt_resistance=1e12,
    )

    assert "\nB" not in netlist
    assert "Rread_op3_0 op3_0 0 1e+12" in netlist
    assert "Rread_op3_1 op3_1 0 1e+12" in netlist
    assert "Rread_on3_0 on3_0 0 1e+12" in netlist
    assert "Rread_on3_1 on3_1 0 1e+12" in netlist


def test_block_netlist_can_emit_passive_readout_stack_parasitic_caps() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        readout_stack_parasitic_capacitance=1e-16,
    )

    assert "\nB" not in netlist
    assert "Cread_op3_0 op3_0 0 1e-16" in netlist
    assert "Cread_op3_1 op3_1 0 1e-16" in netlist
    assert "Cread_on3_0 on3_0 0 1e-16" in netlist
    assert "Cread_on3_1 on3_1 0 1e-16" in netlist


def test_block_netlist_can_emit_transistor_activation_competition() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        activation_competition_width=5.0,
    )

    assert "\nB" not in netlist
    assert "Cactinh actinh 0 10f IC=0" in netlist
    assert "Mreset_actinh actinh rstf 0 0 NMOS" in netlist
    assert "Mactinh_src3_a vdd act3 actinh_src3_a 0 NSENSE W=5u" in netlist
    assert "Mactinh_src3_f actinh_src3_a fwd actinh 0 NMOS W=5u" in netlist
    assert "Mactinh_sink3_i act3 actinh actinh_sink3_i 0 NSENSE W=5u" in netlist
    assert "Mactinh_sink3_f actinh_sink3_i fwd 0 0 NMOS W=5u" in netlist
    assert ".meas tran actinh_before_0 FIND V(actinh) AT=2.95n" in netlist


def test_block_netlist_can_emit_input_dependent_activation_residual() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_input_residual_width=1.5,
    )

    assert "\nB" not in netlist
    assert "Mhres0_0_x vdd x0 hres0_0_0 0 NSENSE W=1.5u" in netlist
    assert "Mhres0_0_f hres0_0_0 fwd act0 0 NMOS W=1.5u" in netlist
    assert "Mhres3_3_x vdd x15 hres3_3_0 0 NSENSE W=1.5u" in netlist
    assert "Mhres3_3_f hres3_3_0 fwd act3 0 NMOS W=1.5u" in netlist


def test_block_netlist_can_emit_passive_hidden_stack_parasitic_caps() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_stack_parasitic_capacitance=1e-16,
    )

    assert "\nB" not in netlist
    assert "Chread_hp3_3_0 hp3_3_0 0 1e-16" in netlist
    assert "Chread_hp3_3_1 hp3_3_1 0 1e-16" in netlist
    assert "Chread_hn3_3_0 hn3_3_0 0 1e-16" in netlist
    assert "Chread_hn3_3_1 hn3_3_1 0 1e-16" in netlist


def test_block_netlist_can_emit_passive_hidden_stack_shunts() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        hidden_stack_shunt_resistance=1e12,
    )

    assert "\nB" not in netlist
    assert "Rhread_hp3_3_0 hp3_3_0 0 1e+12" in netlist
    assert "Rhread_hp3_3_1 hp3_3_1 0 1e+12" in netlist
    assert "Rhread_hn3_3_0 hn3_3_0 0 1e+12" in netlist
    assert "Rhread_hn3_3_1 hn3_3_1 0 1e+12" in netlist


def test_block_netlist_can_emit_score_activity_inhibition() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        activation_competition_width=2.0,
        score_activity_inhibition_width=3.0,
        score_mode="differential",
    )

    assert "\nB" not in netlist
    assert "Mscoreinh_a vdd actinh scoreinh_a 0 NSENSE W=3u" in netlist
    assert "Mscoreinh_f scoreinh_a fwd scoren 0 NMOS W=3u" in netlist


def test_score_activity_inhibition_requires_activation_competition() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    with pytest.raises(ValueError, match="requires activation_competition_width"):
        block.block_netlist(
            [sample],
            weights,
            image_size=image_size,
            block_size=2,
            stride=2,
            channels=1,
            training_enabled=True,
            score_activity_inhibition_width=3.0,
        )


def test_block_netlist_can_emit_trainable_output_bias() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        output_bias_enabled=True,
        score_mode="differential",
    )

    assert "\nB" not in netlist
    assert "Xob obp obn signed_store Cp=20f Icp=0.52 Icn=0.25 Rleak=1e15" in netlist
    assert "Mobpos_w vdd obp obp_f0 0 NREL W=64u" in netlist
    assert "Mobpos_f obp_f0 fwd score 0 NREL W=64u" in netlist
    assert "Mobneg_w vdd obn obn_f0 0 NREL W=48u" in netlist
    assert "Mobneg_f obn_f0 fwd scoren 0 NREL W=48u" in netlist
    assert "Mgop_d vdd dp gop_d 0 NSENSE W=24u" in netlist
    assert "Mgon_d vdd dn gon_d 0 NSENSE W=24u" in netlist
    assert "Mobp_up_p1 obp_up applyn obp vdd PMOS" in netlist
    assert "Mobn_up_p1 obn_up applyn obn vdd PMOS" in netlist
    assert ".meas tran output_bias_signed_after_0 PARAM='obp_after_apply_0-obn_after_apply_0'" in netlist


def test_output_bias_does_not_inherit_readout_weight_leak() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        output_bias_enabled=True,
        readout_weight_leak_resistance=5e6,
    )

    assert "Rvwp0_leak vwp0 vwp_ref 5e+06" in netlist
    assert "Robp_leak" not in netlist
    assert "Robn_leak" not in netlist


def test_output_bias_can_emit_separate_neutral_passive_leak() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
        output_bias_enabled=True,
        output_bias_leak_resistance=7e6,
        output_bias_positive_ref=0.25,
        output_bias_negative_ref=0.25,
    )

    assert "Vobp_ref obp_ref 0 0.25" in netlist
    assert "Vobn_ref obn_ref 0 0.25" in netlist
    assert "Robp_leak obp obp_ref 7e+06" in netlist
    assert "Robn_leak obn obn_ref 7e+06" in netlist


def test_block_netlist_can_emit_target_topology_without_behavioral_sources() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 10
    weights = block.initial_block_weights(image_size, 4, 2, 2, seed=2)
    sample = {f"x{i}": 0.5 for i in range(image_size * image_size)}
    sample.update({f"nx{i}": 0.6 for i in range(image_size * image_size)})
    sample["target"] = 0.0
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=4,
        stride=2,
        channels=2,
        training_enabled=True,
    )

    assert "\nB" not in netlist
    assert netlist.count("\nXwh") == 32 * 16
    assert netlist.count("\nXbh") == 32
    assert netlist.count("\nXvw") == 32
    assert "Vx99 x99 0 PWL" in netlist
    assert "Vnx99 nx99 0 PWL" in netlist
    assert "Xwh31_15 whp31_15 whn31_15 signed_store Cp=20f" in netlist
    assert "Xbh31 bhp31 bhn31 signed_store Cp=20f" in netlist
    assert "Mhpos30_15_x vdd x99 hp30_15_0 0 NMOS" in netlist
    assert "Mhpos31_15_x vdd nx99 hp31_15_0 0 NMOS" in netlist
    assert "Mghp31_15_x vdd nx99 ghp31_15_x 0 NMOS" in netlist
    assert "Mhbpos31_b vdd bhp31 hbp31_0 0 NMOS" in netlist
    assert "Mbhp31_up_a bhp31_up apply bhp31 0 NREL" in netlist
    assert "Movpos31_f op31_1 fwd score 0 NREL" in netlist


def test_block_netlist_rejects_mismatched_weight_shape() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    sample = {f"x{i}": 0.5 for i in range(16)}
    sample["target"] = 1.1
    with pytest.raises(ValueError, match="does not match topology"):
        block.block_netlist(
            [sample],
            block.initial_block_weights(4, 2, 2, 1, seed=0),
            image_size=4,
            block_size=2,
            stride=2,
            channels=2,
            training_enabled=True,
        )


def test_block_netlist_rejects_missing_complement_rail_for_c2_default_mode() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    sample = {f"x{i}": 0.5 for i in range(16)}
    sample["target"] = 1.1
    with pytest.raises(ValueError, match="missing pixel rails: nx0"):
        block.block_netlist(
            [sample],
            block.initial_block_weights(4, 2, 2, 2, seed=0),
            image_size=4,
            block_size=2,
            stride=2,
            channels=2,
            training_enabled=True,
        )


def test_target_polarity_changes_only_label_voltage_convention(monkeypatch) -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    def fake_balanced_digit_indices(labels, count, *, seed, digits):
        del labels, seed
        if count == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.array([0, 1], dtype=np.int64) if digits == (0, 1) else np.array([1, 0], dtype=np.int64)

    class FakeDataset:
        targets = np.array([0, 1], dtype=np.int64)

        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, index):
            import torch

            return torch.zeros((1, 28, 28), dtype=torch.float32), int(index)

    monkeypatch.setattr(block, "balanced_digit_indices", fake_balanced_digit_indices)
    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        SimpleNamespace(datasets=SimpleNamespace(MNIST=FakeDataset), transforms=SimpleNamespace(ToTensor=lambda: None)),
    )

    active_high, _ = block.load_mnist01_block_records(
        2,
        0,
        image_size=4,
        seed=0,
        positive_digit=0,
        negative_digit=1,
        complement_rail_scale=0.5,
        target_polarity="active-high",
        download=False,
    )
    active_low, _ = block.load_mnist01_block_records(
        2,
        0,
        image_size=4,
        seed=0,
        positive_digit=0,
        negative_digit=1,
        complement_rail_scale=0.5,
        target_polarity="active-low",
        download=False,
    )

    assert [row["positive_label"] for row in active_high] == [1.0, 0.0]
    assert [row["target"] for row in active_high] == [1.1, 0.0]
    assert [row["positive_label"] for row in active_low] == [1.0, 0.0]
    assert [row["target"] for row in active_low] == [0.0, 1.1]
