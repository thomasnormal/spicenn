from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_device_readout_mobility_sweep as mobility  # noqa: E402


def test_write_actions_match_flow_modes() -> None:
    assert mobility.write_actions("discharge") == (True, False)
    assert mobility.write_actions("bounded_discharge") == (True, False)
    assert mobility.write_actions("charge_only") == (False, True)
    assert mobility.write_actions("bounded_charge_only") == (False, True)
    assert mobility.write_actions("charge_discharge") == (True, True)
    assert mobility.write_actions("bounded_charge_discharge") == (True, True)


def test_signed_mobility_uses_opposite_branch_for_bounded_discharge() -> None:
    df = pd.DataFrame(
        [
            {
                "experiment": "read_positive",
                "theta": 0.34,
                "act": 0.5,
                "read_gain": 0.4,
                "read_slope": 2.0,
            },
            {
                "experiment": "read_negative",
                "theta": 0.34,
                "act": 0.5,
                "read_gain": 0.6,
                "read_slope": 3.0,
            },
            {
                "experiment": "write_discharge",
                "theta": 0.34,
                "act": None,
                "read_gain": None,
                "read_slope": None,
                "state_delta_v": 0.05,
            },
        ]
    )

    signed = mobility.signed_mobility_table(df, "bounded_discharge", 0.24, 0.70)

    row = signed.iloc[0]
    assert row["signed_increase_mobility"] == pytest.approx(0.15)
    assert row["signed_decrease_mobility"] == pytest.approx(0.10)
    assert row["signed_update_sign_aligned"]
    assert row["signed_mobility_balance"] == pytest.approx(2.0 / 3.0)


def test_signed_mobility_uses_same_branch_for_charge_only() -> None:
    df = pd.DataFrame(
        [
            {
                "experiment": "read_positive",
                "theta": 0.46,
                "act": 0.5,
                "read_gain": 0.4,
                "read_slope": 2.0,
            },
            {
                "experiment": "read_negative",
                "theta": 0.46,
                "act": 0.5,
                "read_gain": 0.6,
                "read_slope": 3.0,
            },
            {
                "experiment": "write_charge",
                "theta": 0.46,
                "act": None,
                "read_gain": None,
                "read_slope": None,
                "state_delta_v": 0.04,
            },
        ]
    )

    signed = mobility.signed_mobility_table(df, "charge_only", 0.0, 1.2)

    row = signed.iloc[0]
    assert row["signed_increase_mobility"] == pytest.approx(0.08)
    assert row["signed_decrease_mobility"] == pytest.approx(0.12)
    assert row["signed_update_sign_aligned"]
    assert row["signed_mobility_balance"] == pytest.approx(2.0 / 3.0)


def test_branch_pair_mobility_allows_independent_positive_and_negative_windows() -> None:
    positive_df = pd.DataFrame(
        [
            {
                "experiment": "read_positive",
                "theta": 0.46,
                "act": 0.5,
                "read_gain": 0.7,
                "read_slope": 2.0,
            },
            {
                "experiment": "write_discharge",
                "theta": 0.46,
                "state_delta_v": 0.03,
            },
        ]
    )
    negative_df = pd.DataFrame(
        [
            {
                "experiment": "read_negative",
                "theta": 0.13,
                "act": 0.5,
                "read_gain": 0.2,
                "read_slope": 5.0,
            },
            {
                "experiment": "write_discharge",
                "theta": 0.13,
                "state_delta_v": 0.01,
            },
        ]
    )

    pair = mobility.branch_pair_signed_mobility_table(positive_df, negative_df, "bounded_discharge")

    row = pair.iloc[0]
    assert row["theta_p"] == pytest.approx(0.46)
    assert row["theta_n"] == pytest.approx(0.13)
    assert row["signed_read_gain"] == pytest.approx(0.5)
    assert row["signed_increase_mobility"] == pytest.approx(0.05)
    assert row["signed_decrease_mobility"] == pytest.approx(0.06)
    assert row["signed_update_sign_aligned"]
    assert row["signed_mobility_balance"] == pytest.approx(5.0 / 6.0)


def test_branch_pair_summary_reports_best_operating_pair() -> None:
    pair = pd.DataFrame(
        [
            {
                "theta_p": 0.34,
                "theta_n": 0.10,
                "act": 0.5,
                "signed_read_gain": 0.4,
                "signed_increase_mobility": 0.01,
                "signed_decrease_mobility": 0.08,
                "signed_update_sign_aligned": True,
                "signed_mobility_balance": 0.125,
            },
            {
                "theta_p": 0.46,
                "theta_n": 0.16,
                "act": 0.5,
                "signed_read_gain": 0.05,
                "signed_increase_mobility": 0.05,
                "signed_decrease_mobility": 0.04,
                "signed_update_sign_aligned": True,
                "signed_mobility_balance": 0.8,
            },
        ]
    )

    summary = mobility.summarize_branch_pair_mobility(pair, 0.24, 0.70, 0.10, 0.16, summary_act_v=0.5)

    assert summary["branch_pair_update_sign_aligned_fraction"] == pytest.approx(1.0)
    assert summary["branch_pair_best_theta_p_v"] == pytest.approx(0.46)
    assert summary["branch_pair_best_theta_n_v"] == pytest.approx(0.16)
    assert summary["branch_pair_best_signed_mobility_balance"] == pytest.approx(0.8)
