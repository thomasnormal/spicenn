from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import parameter_theory as theory  # noqa: E402


def test_rc_tau_matches_hidden_delta_damping_units() -> None:
    assert theory.rc_tau_ns(1e9, 0.02) == pytest.approx(20.0)
    assert theory.leaked_fraction(2.0, 1e9, 0.02) == pytest.approx(1.0 - np.exp(-0.1))
    assert theory.remaining_fraction(20.0, 1e9, 0.02) == pytest.approx(np.exp(-1.0))


def test_damping_bounds_capture_active_and_idle_constraints() -> None:
    bounds = theory.damping_resistance_bounds(
        0.02,
        active_window_ns=2.0,
        idle_window_ns=100.0,
        max_active_loss=0.10,
        max_idle_residue=0.05,
    )

    assert bounds.feasible
    assert bounds.min_tau_ns == pytest.approx(18.98244316)
    assert bounds.max_tau_ns == pytest.approx(33.38082007)
    assert bounds.contains(1e9)


def test_damping_bounds_report_when_passive_reset_is_not_feasible() -> None:
    bounds = theory.damping_resistance_bounds(
        0.02,
        active_window_ns=2.0,
        idle_window_ns=11.2,
        max_active_loss=0.10,
        max_idle_residue=0.05,
    )

    assert not bounds.feasible
    assert bounds.min_resistance_ohm > bounds.max_resistance_ohm


def test_capacitor_step_and_signal_window_ranges() -> None:
    assert theory.capacitor_delta_v(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert theory.required_cap_ff_for_step(0.2, 0.05, 0.01) == pytest.approx(1.0)

    lower, upper = theory.cap_range_for_signal_window_ff(
        min_current_ua=0.01,
        max_current_ua=0.2,
        duration_ns=2.0,
        min_signal_v=0.02,
        max_signal_v=0.8,
    )

    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(1.0)


def test_linear_stability_eta_bound_uses_feature_covariance() -> None:
    features = np.eye(3)
    assert theory.linear_stability_eta_bound(features) == pytest.approx(2.0)

    scaled = 2.0 * np.eye(3)
    assert theory.linear_stability_eta_bound(scaled) == pytest.approx(0.5)
    assert theory.single_sample_eta_bound(4.0) == pytest.approx(0.5)


def test_backward_alignment_cosine_detects_matched_and_reversed_transpose() -> None:
    forward = np.array([[1.0, 2.0], [-0.5, 0.25]])
    matched_backward = forward.T
    reversed_backward = -forward.T
    errors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])

    assert theory.backward_alignment_cosine(forward, matched_backward, errors) == pytest.approx(1.0)
    assert theory.backward_alignment_cosine(forward, reversed_backward, errors) == pytest.approx(-1.0)


def test_mobility_condition_ratio_screens_one_sided_windows() -> None:
    assert theory.mobility_condition_ratio(0.05, 0.10) == pytest.approx(0.5)
    assert theory.mobility_window_is_usable(0.05, 0.10, min_condition_ratio=0.25)
    assert not theory.mobility_window_is_usable(0.005, 0.10, min_condition_ratio=0.25)
    assert not theory.mobility_window_is_usable(0.05, 0.10, min_mobility=0.06)


def test_bounded_write_headroom_rejects_charge_rail_below_initial_state() -> None:
    bad = theory.bounded_write_headroom(center_v=0.64, high_v=0.58, low_v=0.16)
    assert bad.increase_headroom_v == pytest.approx(-0.06)
    assert bad.decrease_headroom_v == pytest.approx(0.48)
    assert not bad.directional()

    good = theory.bounded_write_headroom(center_v=0.64, high_v=1.00, low_v=0.16)
    assert good.increase_headroom_v == pytest.approx(0.36)
    assert good.decrease_headroom_v == pytest.approx(0.48)
    assert good.directional(min_headroom_v=0.05)


def test_one_vs_rest_balance_accounts_for_multiple_non_target_classes() -> None:
    assert theory.one_vs_rest_target_balance_ratio(3, target_mobility=192.0, nontarget_mobility=32.0) == pytest.approx(3.0)
    assert theory.one_vs_rest_width_ratio_is_balanced(3, 192.0, 32.0)
    assert not theory.one_vs_rest_width_ratio_is_balanced(3, 96.0, 64.0)


def test_one_vs_rest_measured_signed_update_balance_reports_epoch_drift() -> None:
    assert theory.one_vs_rest_signed_update_balance_ratio(5, 0.064, -0.016) == pytest.approx(1.0)
    assert theory.one_vs_rest_signed_epoch_delta(5, 0.064, -0.016) == pytest.approx(0.0)

    assert theory.one_vs_rest_signed_update_balance_ratio(5, 0.064, -0.044) == pytest.approx(0.36363636)
    assert theory.one_vs_rest_signed_epoch_delta(5, 0.064, -0.044) == pytest.approx(-0.112)

    with pytest.raises(ValueError, match="nontarget_delta"):
        theory.one_vs_rest_signed_update_balance_ratio(5, 0.064, 0.01)


def test_one_vs_rest_common_epoch_delta_catches_hidden_common_mode_drift() -> None:
    assert theory.one_vs_rest_common_epoch_delta(5, -0.0187, -0.00535) == pytest.approx(-0.0401)
    assert theory.common_drift_to_signed_step_ratio(5, 0.0187, -0.00535, -0.0187, -0.00535) == pytest.approx(
        0.0401 / 0.0187
    )

    assert theory.one_vs_rest_signed_epoch_delta(5, 0.0187, -0.004675) == pytest.approx(0.0)
    assert theory.one_vs_rest_common_epoch_delta(5, -0.0187, -0.004675) == pytest.approx(-0.0374)


def test_multiclass_readout_sizing_derives_topology_dependent_locals() -> None:
    sizing = theory.derive_multiclass_readout_sizing(
        class_count=5,
        effective_readout_fan_in=16.0,
        learning_rate_scale=1.5,
        error_drive_scale=0.5,
        score_tau_scale=2.0,
    )

    assert sizing.readout_fan_in_scale == pytest.approx(2.0)
    assert sizing.readout_update_width_u == pytest.approx(3.75e-4)
    assert sizing.readout_dp_gate_update_width_u == pytest.approx(3.75e-4)
    assert sizing.readout_dn_gate_update_width_u == pytest.approx(3.75e-4)
    assert sizing.output_bias_update_width_u == pytest.approx(3.75e-4)
    assert sizing.readout_write_error_exclusion_width_u == pytest.approx(6.0)
    assert sizing.residual_target_width_u == pytest.approx(48.0)
    assert sizing.residual_output_width_u == pytest.approx(12.0)
    assert sizing.score_cap_f == pytest.approx(40.0)
    assert sizing.output_cap_f == pytest.approx(80.0)
    assert theory.one_vs_rest_target_balance_ratio(
        sizing.class_count,
        sizing.residual_target_width_u,
        sizing.residual_output_width_u,
    ) == pytest.approx(1.0)

    selector_sized = theory.derive_multiclass_readout_sizing(
        class_count=5,
        effective_readout_fan_in=16.0,
        target_selector_ratio=4.0,
        nontarget_selector_ratio=0.5,
        output_bias_update_ratio=0.25,
    )

    assert selector_sized.readout_update_width_u == pytest.approx(2.5e-4)
    assert selector_sized.readout_dp_gate_update_width_u == pytest.approx(1.0e-3)
    assert selector_sized.readout_dn_gate_update_width_u == pytest.approx(1.25e-4)
    assert selector_sized.output_bias_update_width_u == pytest.approx(6.25e-5)

    no_bias_sized = theory.derive_multiclass_readout_sizing(
        class_count=5,
        effective_readout_fan_in=16.0,
        output_bias_update_ratio=0.0,
    )

    assert no_bias_sized.output_bias_update_width_u == pytest.approx(0.0)


def test_multiclass_readout_sizing_rejects_invalid_global_scales() -> None:
    with pytest.raises(ValueError, match="learning_rate_scale"):
        theory.derive_multiclass_readout_sizing(
            class_count=3,
            effective_readout_fan_in=8.0,
            learning_rate_scale=0.0,
        )
    with pytest.raises(ValueError, match="class_count"):
        theory.derive_multiclass_readout_sizing(
            class_count=1,
            effective_readout_fan_in=8.0,
        )


def test_readout_margin_sizing_scales_width_or_capacitance_from_observed_margin() -> None:
    sizing = theory.derive_readout_margin_sizing(
        observed_margin_v=0.00029331,
        target_margin_v=0.010,
        current_readout_width_u=64.0,
        current_score_cap_f=10.0,
        max_readout_width_u=512.0,
        min_score_cap_f=0.5,
    )

    assert sizing.required_signal_scale == pytest.approx(34.09362108349528)
    assert sizing.suggested_readout_width_u == pytest.approx(2181.991749343698)
    assert sizing.readout_width_feasible is False
    assert sizing.suggested_score_cap_f == pytest.approx(0.29331)
    assert sizing.score_cap_feasible is False

    modest = theory.derive_readout_margin_sizing(
        observed_margin_v=0.0025,
        target_margin_v=0.010,
        current_readout_width_u=64.0,
        current_score_cap_f=10.0,
    )
    assert modest.required_signal_scale == pytest.approx(4.0)
    assert modest.suggested_readout_width_u == pytest.approx(256.0)
    assert modest.readout_width_feasible is True
    assert modest.suggested_score_cap_f == pytest.approx(2.5)
    assert modest.score_cap_feasible is True

    with pytest.raises(ValueError, match="observed_margin_v"):
        theory.derive_readout_margin_sizing(
            observed_margin_v=0.0,
            target_margin_v=0.010,
            current_readout_width_u=64.0,
            current_score_cap_f=10.0,
        )


def test_class_evidence_normalizer_sizing_derives_writer_domain_defaults() -> None:
    sizing = theory.derive_class_evidence_normalizer_sizing(
        class_count=3,
        normalized_score_delta_v=4.86e-3,
        contrast_window_ns=2.0,
        mass_window_ns=0.5,
        error_window_ns=0.7,
        target_error_v=0.32,
        score_common_cap_f=4.0,
        contrast_cap_f=10.0,
        error_drive_scale=1.0,
    )

    assert sizing.class_count == 3
    assert sizing.score_common_resistance_ohm == pytest.approx(1.0e7)
    assert sizing.score_common_tau_ns == pytest.approx(40.0)
    assert sizing.contrast_observable_ratio == pytest.approx(0.486)
    assert sizing.mass_cap_f == pytest.approx(0.5)
    assert sizing.error_cap_f == pytest.approx(0.5)
    assert sizing.mass_width_u == pytest.approx(128.0)
    assert sizing.target_error_width_u == pytest.approx(128.0)
    assert sizing.nontarget_error_width_u == pytest.approx(64.0)
    assert theory.one_vs_rest_target_balance_ratio(
        sizing.class_count,
        sizing.target_error_width_u,
        sizing.nontarget_error_width_u,
    ) == pytest.approx(1.0)

    slower = theory.derive_class_evidence_normalizer_sizing(
        class_count=3,
        normalized_score_delta_v=4.86e-3,
        contrast_window_ns=2.0,
        mass_window_ns=0.5,
        error_window_ns=0.7,
        error_drive_scale=0.5,
    )

    assert slower.mass_width_u == pytest.approx(64.0)
    assert slower.target_error_width_u == pytest.approx(64.0)
    assert slower.nontarget_error_width_u == pytest.approx(32.0)


def test_class_evidence_normalizer_sizing_rejects_invisible_contrast() -> None:
    with pytest.raises(ValueError, match="normalized_score_delta_v"):
        theory.derive_class_evidence_normalizer_sizing(
            class_count=3,
            normalized_score_delta_v=0.0,
        )
    with pytest.raises(ValueError, match="contrast_observable_ratio"):
        theory.derive_class_evidence_normalizer_sizing(
            class_count=3,
            normalized_score_delta_v=0.1e-3,
            min_writer_gate_v=0.01,
        )


def test_multiclass_margin_correction_sizing_derives_writer_domain_defaults() -> None:
    sizing = theory.derive_multiclass_margin_correction_sizing(
        class_count=3,
        target_margin_v=1.0e-3,
        score_delta_v=3.0e-3,
        error_window_ns=1.2,
        target_error_v=0.08,
        error_drive_scale=0.5,
    )

    assert sizing.class_count == 3
    assert sizing.target_margin_v == pytest.approx(1.0e-3)
    assert sizing.score_delta_v == pytest.approx(3.0e-3)
    assert sizing.observable_margin_ratio == pytest.approx(1.0)
    assert sizing.pairwise_pullup_width_u == pytest.approx(16.0)
    assert sizing.pairwise_pulldown_width_u == pytest.approx(64.0)
    assert sizing.margin_penalty_width_u == pytest.approx(0.25)
    assert sizing.margin_reference_window_v == pytest.approx(0.256)
    assert sizing.error_width_u == pytest.approx(64.0)
    assert sizing.error_cap_f == pytest.approx(0.5)
    assert sizing.error_clock_high_v == pytest.approx(0.10)


def test_multiclass_margin_correction_sizing_rejects_invisible_margin() -> None:
    with pytest.raises(ValueError, match="target_margin_v"):
        theory.derive_multiclass_margin_correction_sizing(
            class_count=3,
            target_margin_v=0.0,
            score_delta_v=3.0e-3,
        )
    with pytest.raises(ValueError, match="observable"):
        theory.derive_multiclass_margin_correction_sizing(
            class_count=3,
            target_margin_v=0.2e-3,
            score_delta_v=3.0e-3,
            min_observable_score_delta_v=1.0e-3,
        )
