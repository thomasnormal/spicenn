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
    assert row["physical_gradient_increase_mobility"] == pytest.approx(0.45)
    assert row["physical_gradient_decrease_mobility"] == pytest.approx(0.20)
    assert row["physical_gradient_sign_aligned"]
    assert row["physical_gradient_mobility_balance"] == pytest.approx(4.0 / 9.0)


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
    assert row["physical_gradient_increase_mobility"] == pytest.approx(0.16)
    assert row["physical_gradient_decrease_mobility"] == pytest.approx(0.36)
    assert row["physical_gradient_sign_aligned"]
    assert row["physical_gradient_mobility_balance"] == pytest.approx(4.0 / 9.0)


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
    assert row["physical_gradient_increase_mobility"] == pytest.approx(0.25)
    assert row["physical_gradient_decrease_mobility"] == pytest.approx(0.12)
    assert row["physical_gradient_sign_aligned"]
    assert row["physical_gradient_mobility_balance"] == pytest.approx(0.12 / 0.25)


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
                "physical_gradient_increase_mobility": 0.03,
                "physical_gradient_decrease_mobility": 0.09,
                "physical_gradient_sign_aligned": True,
                "physical_gradient_mobility_balance": 1.0 / 3.0,
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
                "physical_gradient_increase_mobility": 0.05,
                "physical_gradient_decrease_mobility": 0.10,
                "physical_gradient_sign_aligned": True,
                "physical_gradient_mobility_balance": 0.5,
            },
        ]
    )

    summary = mobility.summarize_branch_pair_mobility(pair, 0.24, 0.70, 0.10, 0.16, summary_act_v=0.5)

    assert summary["branch_pair_update_sign_aligned_fraction"] == pytest.approx(1.0)
    assert summary["branch_pair_best_theta_p_v"] == pytest.approx(0.46)
    assert summary["branch_pair_best_theta_n_v"] == pytest.approx(0.16)
    assert summary["branch_pair_best_signed_mobility_balance"] == pytest.approx(0.8)
    assert summary["branch_pair_best_physical_gradient_mobility_balance"] == pytest.approx(0.5)
    assert summary["branch_pair_min_physical_gradient_mobility_balance"] == pytest.approx(1.0 / 3.0)
    assert summary["branch_pair_all_act_negative_signed_read_gain_fraction"] == pytest.approx(0.0)


def test_branch_pair_summary_reports_gain_safe_pair_across_activations() -> None:
    pair = pd.DataFrame(
        [
            {
                "theta_p": 0.46,
                "theta_n": 0.16,
                "act": 0.25,
                "signed_read_gain": -0.2,
                "signed_increase_mobility": 0.08,
                "signed_decrease_mobility": 0.04,
                "signed_update_sign_aligned": True,
                "signed_mobility_balance": 0.5,
                "physical_gradient_increase_mobility": 0.08,
                "physical_gradient_decrease_mobility": 0.04,
                "physical_gradient_sign_aligned": True,
                "physical_gradient_mobility_balance": 0.5,
            },
            {
                "theta_p": 0.46,
                "theta_n": 0.16,
                "act": 0.5,
                "signed_read_gain": 0.2,
                "signed_increase_mobility": 0.08,
                "signed_decrease_mobility": 0.04,
                "signed_update_sign_aligned": True,
                "signed_mobility_balance": 0.5,
                "physical_gradient_increase_mobility": 0.08,
                "physical_gradient_decrease_mobility": 0.04,
                "physical_gradient_sign_aligned": True,
                "physical_gradient_mobility_balance": 0.5,
            },
            {
                "theta_p": 0.34,
                "theta_n": 0.10,
                "act": 0.25,
                "signed_read_gain": 0.1,
                "signed_increase_mobility": 0.02,
                "signed_decrease_mobility": 0.01,
                "signed_update_sign_aligned": True,
                "signed_mobility_balance": 0.5,
                "physical_gradient_increase_mobility": 0.02,
                "physical_gradient_decrease_mobility": 0.01,
                "physical_gradient_sign_aligned": True,
                "physical_gradient_mobility_balance": 0.5,
            },
            {
                "theta_p": 0.34,
                "theta_n": 0.10,
                "act": 0.5,
                "signed_read_gain": 0.3,
                "signed_increase_mobility": 0.02,
                "signed_decrease_mobility": 0.01,
                "signed_update_sign_aligned": True,
                "signed_mobility_balance": 0.5,
                "physical_gradient_increase_mobility": 0.02,
                "physical_gradient_decrease_mobility": 0.01,
                "physical_gradient_sign_aligned": True,
                "physical_gradient_mobility_balance": 0.5,
            },
        ]
    )

    summary = mobility.summarize_branch_pair_mobility(pair, 0.24, 0.70, 0.10, 0.16, summary_act_v=0.5)

    # The single-activation best can still be the high-mobility pair, but the
    # gain-safe summary rejects it because its low-activation read gain flips sign.
    assert summary["branch_pair_best_theta_p_v"] == pytest.approx(0.46)
    assert summary["branch_pair_best_theta_n_v"] == pytest.approx(0.16)
    assert summary["branch_pair_all_act_negative_signed_read_gain_fraction"] == pytest.approx(0.25)
    assert summary["branch_pair_gain_safe_pair_count"] == 1
    assert summary["branch_pair_best_gain_safe_theta_p_v"] == pytest.approx(0.34)
    assert summary["branch_pair_best_gain_safe_theta_n_v"] == pytest.approx(0.10)
    assert summary["branch_pair_best_gain_safe_min_signed_read_gain"] == pytest.approx(0.1)
    assert summary["branch_pair_best_gain_safe_min_physical_gradient_mobility_balance"] == pytest.approx(0.5)


def test_pair_action_netlist_uses_state_window_write_stacks() -> None:
    netlist = mobility.pair_action_mobility_netlist(
        theta_p=0.34,
        theta_n=0.10,
        act=0.5,
        pre=0.65,
        delta=1.0,
        width_u=0.0008,
        signed_action="increase",
        write_mode="bounded_charge_discharge",
        pos_write_high_v=0.70,
        pos_write_low_v=0.24,
        neg_write_high_v=0.16,
        neg_write_low_v=0.10,
        write_state_gate_mode="state_window",
    )

    assert "Vwphigh wphigh 0 0.7" in netlist
    assert "Vwnlow wnlow 0 0.1" in netlist
    assert "Mwp_ch_s wphigh wp wp_ch_s vdd PMOS W=0.0008u" in netlist
    assert "Mwp_ch_d wp_ch_a delta wp 0 NSENSE W=0.0008u" in netlist
    assert "Mwn_dis_s wn wn wn_dis_s 0 NREL W=0.0008u" in netlist
    assert "Mwn_dis_d wn_dis_a delta wnlow 0 NSENSE W=0.0008u" in netlist
    assert ".meas tran desired_signed_read_delta PARAM='signed_read_delta'" in netlist


def test_error_rule_action_netlist_uses_real_dp_dn_write_stacks() -> None:
    netlist = mobility.error_rule_action_mobility_netlist(
        theta_p=0.34,
        theta_n=0.13,
        act=0.5,
        error_action="label0_mistake",
        error_rule="out_competitive",
        write_mode="bounded_charge_discharge",
        width_u=0.0008,
        pos_write_high_v=0.70,
        pos_write_low_v=0.24,
        neg_write_high_v=0.16,
        neg_write_low_v=0.10,
    )

    assert "Mdp0_o0 dp0_t out1 dp0_o 0 NSENSE W=96u" in netlist
    assert "Mdn1_s0 dn1_t out1 dn1_s 0 NSENSE W=96u" in netlist
    assert "Mw0p_ch_d w0p_ch_a dp0 wp0 0 NSENSE W=0.0008u" in netlist
    assert "Mw0n_dis_d w0n_dis_a dp0 wnlow 0 NSENSE W=0.0008u" in netlist
    assert "Mw1p_dis_d w1p_dis_a dn1 wplow 0 NSENSE W=0.0008u" in netlist
    assert "Mw1n_ch_d w1n_ch_a dn1 wn1 0 NSENSE W=0.0008u" in netlist
    assert ".meas tran row0_desired_signed_read_delta PARAM='row0_signed_read_delta'" in netlist
    assert ".meas tran row1_desired_signed_read_delta PARAM='-row1_signed_read_delta'" in netlist


def test_error_rule_action_summary_requires_both_rows_aligned() -> None:
    rows = pd.DataFrame(
        [
            {
                "theta_p": 0.34,
                "theta_n": 0.13,
                "error_rule": "out_competitive",
                "row0_signed_read_delta": 0.02,
                "row0_desired_signed_read_delta": 0.02,
                "row1_signed_read_delta": -0.01,
                "row1_desired_signed_read_delta": 0.01,
                "dp0_probe": 0.9,
                "dn0_probe": 0.0,
                "dp1_probe": 0.0,
                "dn1_probe": 0.8,
            },
            {
                "theta_p": 0.46,
                "theta_n": 0.13,
                "error_rule": "out_competitive",
                "row0_signed_read_delta": -0.03,
                "row0_desired_signed_read_delta": -0.03,
                "row1_signed_read_delta": -0.02,
                "row1_desired_signed_read_delta": 0.02,
                "dp0_probe": 0.9,
                "dn0_probe": 0.0,
                "dp1_probe": 0.0,
                "dn1_probe": 0.8,
            },
        ]
    )

    summary = mobility.summarize_error_rule_action_mobility(rows)

    assert summary["error_rule_action_rows"] == 2
    assert summary["error_rule_action_both_rows_sign_aligned_fraction"] == pytest.approx(0.5)
    assert summary["error_rule_action_best_theta_p_v"] == pytest.approx(0.34)
    assert summary["error_rule_action_best_theta_n_v"] == pytest.approx(0.13)


def test_pair_action_summary_prefers_sign_aligned_direct_effect() -> None:
    pair_action = pd.DataFrame(
        [
            {
                "theta_p": 0.34,
                "theta_n": 0.10,
                "signed_read_delta": 0.04,
                "desired_signed_read_delta": 0.04,
            },
            {
                "theta_p": 0.34,
                "theta_n": 0.10,
                "signed_read_delta": -0.01,
                "desired_signed_read_delta": 0.01,
            },
            {
                "theta_p": 0.46,
                "theta_n": 0.16,
                "signed_read_delta": -0.02,
                "desired_signed_read_delta": -0.02,
            },
        ]
    )

    summary = mobility.summarize_pair_action_mobility(pair_action)

    assert summary["pair_action_rows"] == 3
    assert summary["pair_action_sign_aligned_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["pair_action_best_theta_p_v"] == pytest.approx(0.34)
    assert summary["pair_action_best_theta_n_v"] == pytest.approx(0.10)
    assert summary["pair_action_best_aligned_fraction"] == pytest.approx(1.0)
