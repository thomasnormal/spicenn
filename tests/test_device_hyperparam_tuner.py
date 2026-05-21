from __future__ import annotations

import json
import sys
from pathlib import Path
from subprocess import CompletedProcess

import optuna
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import tune_device_hyperparams as tuner  # noqa: E402


def test_split_tuner_and_driver_args_uses_double_dash_boundary() -> None:
    own, driver = tuner.split_tuner_and_driver_args(
        ["--profile", "xor2_pmos_cd", "--trials", "3", "--", "--seed", "7", "--epochs", "1"]
    )

    assert own == ["--profile", "xor2_pmos_cd", "--trials", "3"]
    assert driver == ["--seed", "7", "--epochs", "1"]


def test_params_to_driver_args_formats_only_present_values() -> None:
    args = tuner.params_to_driver_args(
        {
            "readout_charge_update_width_u": 0.0002,
            "readout_center_pull_width_u": 0.0,
            "readout_write_gate_device": "NREL",
            "skip_train_refire": True,
            "unused": None,
            "disabled_flag": False,
        }
    )

    assert args == [
        "--readout-charge-update-width-u",
        "0.0002",
        "--readout-center-pull-width-u",
        "0",
        "--readout-write-gate-device",
        "NREL",
        "--skip-train-refire",
    ]


def test_build_trial_command_puts_sampled_args_after_profile_and_before_extra_overrides() -> None:
    profile = tuner.PROFILES["xor2_pmos_cd"]
    command = tuner.build_trial_command(
        profile=profile,
        tag="abc",
        sampled_params={"readout_charge_update_width_u": 0.0004},
        driver_timeout_s=12.0,
        simulator="ngspice",
        extra_driver_args=["--seed", "9"],
    )

    assert command[0] == sys.executable
    assert command[1].endswith("run_device_xor2_random_hidden.py")
    assert command[command.index("--tag") + 1] == "abc"
    assert "--dataset" in command
    assert command[command.index("--readout-charge-update-width-u") + 1] == "0.0004"
    assert command[-4:] == ["--simulator", "ngspice", "--seed", "9"]


def test_objective_score_prefers_best_eval_and_can_penalize_common_mode() -> None:
    summary = {
        "best_eval_accuracy": 0.91,
        "final_score_accuracy": 0.75,
        "final_score_column_centered_accuracy": 1.0,
        "readout_common_to_signed_delta_ratio": 0.2,
        "max_abs_total_hidden_signed_delta_v": 0.01,
    }

    assert tuner.metric_value(summary, "auto") == pytest.approx(0.91)
    assert tuner.metric_value(summary, "score_raw_centered_min") == pytest.approx(0.75)
    assert tuner.metric_value(summary, "score_raw_centered_mean") == pytest.approx(0.875)
    assert tuner.metric_value(summary, "score_raw_centered_product") == pytest.approx(0.75)
    assert tuner.objective_score(
        summary,
        metric="auto",
        common_mode_penalty=0.1,
        hidden_write_penalty=2.0,
    ) == pytest.approx(0.91 - 0.02 - 0.02)


def test_sample_pmos_profile_produces_valid_low_high_rails() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "readout_write_low_v": 0.16,
            "readout_write_high_v": 1.0,
            "readout_center_pull_enabled": True,
            "output_bias_update_enabled": False,
            "readout_write_error_exclusion_width_u": 8.0,
            "readout_charge_update_width_u": 0.002,
            "readout_discharge_update_width_u": 0.0005,
            "readout_center_pull_width_u": 0.0002,
            "readout_center_pull_gate": "apply",
            "readout_center_pull_mode": "state_high",
            "output_bias_offset_v": 0.12,
            "output_bias_forward_width_scale": 0.8,
            "score_cap_f": 20.0,
            "score_reset_v": 0.05,
        }
    )

    params = tuner.sample_pmos_charge_discharge(trial)

    assert params["readout_flow_write_mode"] == "bounded_pmos_charge_discharge"
    assert params["readout_write_error_exclusion"] == "diffpair_bleed"
    assert params["readout_write_low_v"] < params["readout_write_high_v"]
    assert params["readout_center_pull_width_u"] == pytest.approx(0.0002)
    assert params["output_bias_update_width_u"] == 0.0
    assert params["output_bias_offset_v"] == pytest.approx(0.12)
    assert params["output_bias_forward_width_scale"] == pytest.approx(0.8)


def test_target_mistake_profile_uses_micro_bounded_discharge_anchor() -> None:
    profile = tuner.PROFILES["mnist01fixed8_tm_competitive"]
    trial = optuna.trial.FixedTrial(
        {
            "readout_center_pull_enabled": False,
            "readout_update_width_u": 0.00035,
            "output_bias_update_width_u": 0.0,
            "readout_write_low_v": 0.16,
            "readout_write_high_v": 0.58,
            "readout_center_pull_gate": "bwd",
            "readout_center_pull_mode": "always",
            "score_cap_f": 20.0,
            "score_reset_v": 0.0,
            "lead_width_u": 96.0,
            "backward_gate_width_u": 64.0,
            "backward_gate_cap_f": 2.0,
            "output_relu_width_scale": 1.0,
            "output_forward_width_scale": 1.0,
        }
    )

    params = tuner.sample_target_mistake_bounded_discharge(trial)

    assert profile.sampler == "target_mistake_bounded_discharge"
    assert "--backward-gate-mode" in profile.base_args
    assert profile.base_args[profile.base_args.index("--backward-gate-mode") + 1] == "target_mistake"
    assert profile.base_args[profile.base_args.index("--error-rule") + 1] == "score"
    assert "--output-bias-update-width-u" not in profile.base_args
    assert dict(profile.anchor_params)["readout_update_width_u"] == pytest.approx(0.00035)
    assert dict(profile.anchor_params)["output_bias_update_width_u"] == pytest.approx(0.0)
    assert params["readout_flow_write_mode"] == "bounded_discharge"
    assert params["readout_update_width_u"] == pytest.approx(0.00035)
    assert params["output_bias_update_width_u"] == pytest.approx(0.0)
    assert params["lead_width_u"] == pytest.approx(96.0)


def test_mnist3_capstate_profile_replays_low_drive_anchor() -> None:
    profile = tuner.PROFILES["mnist3fixed8_capstate_replay"]
    anchor = dict(profile.anchor_params)
    trial = optuna.trial.FixedTrial(anchor)

    params = tuner.sample_multiclass_capstate_bounded_cd(trial)
    command = tuner.build_trial_command(
        profile=profile,
        tag="mnist3_anchor",
        sampled_params=params,
        driver_timeout_s=120.0,
        simulator=None,
        extra_driver_args=[],
    )

    assert profile.sampler == "multiclass_capstate_bounded_cd"
    assert profile.base_args[profile.base_args.index("--dataset") + 1] == "mnist3fixed8_30"
    assert profile.base_args[profile.base_args.index("--readout-init") + 1] == "csv_cap_state"
    assert profile.base_args[profile.base_args.index("--separator-csv") + 1].endswith(
        "device_readout_capfit_sumtransfer_w1024_c100_scale002_caps.csv"
    )
    assert params["readout_flow_write_mode"] == "bounded_charge_discharge"
    assert params["output_forward_width_scale"] == pytest.approx(0.02)
    assert params["residual_target_width_u"] == pytest.approx(0.06)
    assert params["score_cap_f"] == pytest.approx(0.4)
    assert command[command.index("--output-forward-width-scale") + 1] == "0.02"
    assert command[command.index("--readout-init") + 1] == "csv_cap_state"


def test_mnist3_random_pmos_profile_tracks_from_random_anchor() -> None:
    profile = tuner.PROFILES["mnist3fixed8_random_pmos_chargeonly"]
    anchor = dict(profile.anchor_params)
    trial = optuna.trial.FixedTrial(anchor)

    params = tuner.sample_multiclass_random_pmos_charge_only(trial)
    command = tuner.build_trial_command(
        profile=profile,
        tag="mnist3_random_pmos",
        sampled_params=params,
        driver_timeout_s=120.0,
        simulator=None,
        extra_driver_args=["--dataset", "mnist3fixed8_30"],
    )

    assert profile.sampler == "multiclass_random_pmos_charge_only"
    assert "--readout-init" not in profile.base_args
    assert profile.base_args[profile.base_args.index("--flow-pre-store") + 1] == "synapse_spike"
    assert profile.base_args[profile.base_args.index("--output-head") + 1] == "split_score_caps"
    assert profile.base_args[profile.base_args.index("--error-rule") + 1] == "onehot_limited"
    assert profile.base_args[profile.base_args.index("--decision-source") + 1] == "score"
    assert params["readout_flow_write_mode"] == "bounded_pmos_charge_only"
    assert params["readout_write_error_exclusion"] == "diffpair_bleed"
    assert params["readout_update_width_u"] == pytest.approx(5e-4)
    assert params["readout_flow_polarity"] == "normal"
    assert params["residual_target_width_u"] == pytest.approx(96.0)
    assert params["residual_output_width_u"] == pytest.approx(64.0)
    assert command[command.index("--readout-flow-write-mode") + 1] == "bounded_pmos_charge_only"
    assert command[-2:] == ["--dataset", "mnist3fixed8_30"]


def test_mnist3_random_pmos_derived_profile_exposes_only_global_scales() -> None:
    profile = tuner.PROFILES["mnist3fixed8_random_pmos_chargeonly_derived"]
    anchor = dict(profile.anchor_params)
    trial = optuna.trial.FixedTrial(anchor)

    params = tuner.sample_multiclass_random_pmos_charge_only_derived(trial)
    command = tuner.build_trial_command(
        profile=profile,
        tag="mnist3_derived",
        sampled_params=params,
        driver_timeout_s=120.0,
        simulator=None,
        extra_driver_args=[],
    )

    assert profile.sampler == "multiclass_random_pmos_charge_only_derived"
    assert set(anchor) == {"learning_rate_scale", "error_drive_scale", "score_tau_scale"}
    assert "--flow-pre-store" in profile.base_args
    assert profile.base_args[profile.base_args.index("--flow-pre-store") + 1] == "synapse_spike"
    assert params["readout_update_width_u"] == pytest.approx(5e-4)
    assert params["readout_dp_gate_update_width_u"] == pytest.approx(5e-4)
    assert params["readout_dn_gate_update_width_u"] == pytest.approx(5e-4)
    assert params["readout_write_error_exclusion_width_u"] == pytest.approx(8.0)
    assert params["residual_target_width_u"] == pytest.approx(96.0)
    assert params["residual_output_width_u"] == pytest.approx(48.0)
    assert params["score_cap_f"] == pytest.approx(10.0)
    assert params["output_cap_f"] == pytest.approx(20.0)
    assert params["output_bias_update_width_u"] == pytest.approx(5e-4)
    assert "--learning-rate-scale" not in command
    assert command[command.index("--readout-update-width-u") + 1] == "0.0005"
    assert command[command.index("--readout-dp-gate-update-width-u") + 1] == "0.0005"


def test_mnist3_random_pmos_derived_scales_keep_local_ratios() -> None:
    params = tuner.derive_multiclass_random_pmos_charge_only_params(
        learning_rate_scale=2.0,
        error_drive_scale=0.5,
        score_tau_scale=1.5,
    )

    assert params["readout_update_width_u"] == pytest.approx(1e-3)
    assert params["readout_dp_gate_update_width_u"] == pytest.approx(1e-3)
    assert params["readout_dn_gate_update_width_u"] == pytest.approx(1e-3)
    assert params["output_bias_update_width_u"] == pytest.approx(1e-3)
    assert params["readout_write_error_exclusion_width_u"] == pytest.approx(16.0)
    assert params["residual_target_width_u"] == pytest.approx(48.0)
    assert params["residual_output_width_u"] == pytest.approx(24.0)
    assert params["score_cap_f"] == pytest.approx(15.0)
    assert params["output_cap_f"] == pytest.approx(30.0)

    with pytest.raises(ValueError, match="positive"):
        tuner.derive_multiclass_random_pmos_charge_only_params(learning_rate_scale=0.0)
    with pytest.raises(ValueError, match="write mode"):
        tuner.derive_multiclass_random_pmos_charge_only_params(readout_flow_write_mode="bounded_discharge")


def test_mnist3_random_pmos_charge_discharge_derived_profile_sets_fixed_selector_ratio() -> None:
    profile = tuner.PROFILES["mnist3fixed8_random_pmos_cd_derived"]
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate_scale": 1.0,
            "error_drive_scale": 1.0,
            "score_tau_scale": 1.0,
        }
    )

    params = tuner.sample_profile_params(profile, trial, extra_driver_args=[])

    assert profile.sampler == "multiclass_random_pmos_charge_discharge_derived"
    assert params["readout_flow_write_mode"] == "bounded_pmos_charge_discharge"
    assert params["readout_write_error_exclusion"] == "diffpair_bleed"
    assert params["readout_update_width_u"] == pytest.approx(5e-4)
    assert params["readout_dp_gate_update_width_u"] == pytest.approx(2e-3)
    assert params["readout_dn_gate_update_width_u"] == pytest.approx(5e-4)
    assert params["output_bias_update_width_u"] == pytest.approx(5e-4)
    assert params["score_cap_f"] == pytest.approx(10.0)


def test_mnist3_random_pmos_derived_scales_one_vs_rest_by_class_count() -> None:
    params = tuner.derive_multiclass_random_pmos_charge_only_params(class_count=5)

    assert params["residual_target_width_u"] == pytest.approx(96.0)
    assert params["residual_output_width_u"] == pytest.approx(24.0)
    assert params["output_bias_update_width_u"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="class_count"):
        tuner.derive_multiclass_random_pmos_charge_only_params(class_count=1)


def test_derived_profile_uses_effective_dataset_override_for_class_count() -> None:
    profile = tuner.PROFILES["mnist3fixed8_random_pmos_chargeonly_derived"]
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate_scale": 1.0,
            "error_drive_scale": 1.0,
            "score_tau_scale": 1.0,
        }
    )

    params = tuner.sample_profile_params(
        profile,
        trial,
        extra_driver_args=["--dataset", "mnist5fixed8_25"],
    )

    assert tuner.effective_driver_arg(profile, ["--dataset", "mnist5fixed8_25"], "--dataset") == "mnist5fixed8_25"
    assert tuner.class_count_for_dataset("mnist5fixed8_25") == 5
    assert tuner.class_count_for_dataset("mnistfixed8_20") == 10
    assert params["residual_target_width_u"] == pytest.approx(96.0)
    assert params["residual_output_width_u"] == pytest.approx(24.0)


def test_derived_profile_rescales_score_caps_and_write_by_readout_fanin() -> None:
    profile = tuner.PROFILES["mnist3fixed8_random_pmos_chargeonly_derived"]
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate_scale": 1.0,
            "error_drive_scale": 1.0,
            "score_tau_scale": 1.0,
        }
    )

    dense_wide = tuner.sample_profile_params(
        profile,
        trial,
        extra_driver_args=["--hidden-cells", "16"],
    )
    sparse_fanout = tuner.sample_profile_params(
        profile,
        trial,
        extra_driver_args=[
            "--dataset",
            "mnist5fixed8_25",
            "--hidden-cells",
            "16",
            "--readout-topology",
            "random_fanout",
            "--readout-fan-out",
            "3",
        ],
    )
    balanced_sparse_fanout = tuner.sample_profile_params(
        profile,
        trial,
        extra_driver_args=[
            "--dataset",
            "mnist5fixed8_25",
            "--hidden-cells",
            "16",
            "--readout-topology",
            "balanced_random_fanout",
            "--readout-fan-out",
            "3",
        ],
    )

    assert dense_wide["score_cap_f"] == pytest.approx(20.0)
    assert dense_wide["output_cap_f"] == pytest.approx(40.0)
    assert dense_wide["readout_update_width_u"] == pytest.approx(2.5e-4)
    assert dense_wide["readout_dp_gate_update_width_u"] == pytest.approx(2.5e-4)
    assert dense_wide["readout_dn_gate_update_width_u"] == pytest.approx(2.5e-4)
    assert dense_wide["output_bias_update_width_u"] == pytest.approx(2.5e-4)
    assert dense_wide["readout_write_error_exclusion_width_u"] == pytest.approx(4.0)
    assert sparse_fanout["score_cap_f"] == pytest.approx(12.0)
    assert sparse_fanout["output_cap_f"] == pytest.approx(24.0)
    assert sparse_fanout["readout_update_width_u"] == pytest.approx(5e-4 * 8.0 / 9.6)
    assert sparse_fanout["readout_dp_gate_update_width_u"] == pytest.approx(5e-4 * 8.0 / 9.6)
    assert sparse_fanout["readout_dn_gate_update_width_u"] == pytest.approx(5e-4 * 8.0 / 9.6)
    assert sparse_fanout["output_bias_update_width_u"] == pytest.approx(0.0)
    assert sparse_fanout["residual_output_width_u"] == pytest.approx(24.0)
    assert balanced_sparse_fanout == sparse_fanout


def test_run_one_trial_records_command_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuner, "SPICE_RESULTS", tmp_path / "spice_results")
    monkeypatch.setattr(tuner, "TABLES", tmp_path / "tables")
    tuner.SPICE_RESULTS.mkdir(parents=True)

    args = tuner.parse_args(
        [
            "--profile",
            "xor2_pmos_cd",
            "--study-name",
            "unit_study",
            "--tag-prefix",
            "unit",
            "--driver-timeout",
            "1",
            "--",
            "--seed",
            "4",
        ]
    )
    profile = tuner.PROFILES[args.profile]
    trial = optuna.trial.FixedTrial(
        {
            "readout_write_low_v": 0.16,
            "readout_write_high_v": 1.0,
            "readout_center_pull_enabled": False,
            "output_bias_update_enabled": False,
            "readout_write_error_exclusion_width_u": 8.0,
            "readout_charge_update_width_u": 0.002,
            "readout_discharge_update_width_u": 0.0005,
            "readout_center_pull_gate": "bwd",
            "readout_center_pull_mode": "always",
            "output_bias_offset_v": 0.0,
            "output_bias_forward_width_scale": 1.0,
            "score_cap_f": 20.0,
            "score_reset_v": 0.05,
        }
    )

    def fake_runner(command, **_kwargs):
        tag = command[command.index("--tag") + 1]
        (tuner.SPICE_RESULTS / f"{tag}_summary.json").write_text(
            json.dumps({"best_eval_accuracy": 0.875, "final_score_accuracy": 0.75}) + "\n"
        )
        return CompletedProcess(command, 0, stdout="ok", stderr="")

    value = tuner.run_one_trial(trial, args=args, profile=profile, runner=fake_runner)

    assert value == pytest.approx(0.875)
    trials_csv = tuner.TABLES / "unit_study_trials.csv"
    assert trials_csv.exists()
    text = trials_csv.read_text()
    assert "--seed 4" in text
    assert "param_readout_charge_update_width_u" in text
    assert "status" in text
