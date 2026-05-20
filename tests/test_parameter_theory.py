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
