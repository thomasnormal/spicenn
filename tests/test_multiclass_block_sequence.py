from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_multiclass_block_sequence as seq  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def _target0_records(count: int) -> list[dict[str, object]]:
    return [{"label": 0, "inputs": {"x0": 0.85}} for _ in range(count)]


def _target0_two_feature_records(count: int) -> list[dict[str, object]]:
    return [{"label": 0, "inputs": {"x0": 0.85, "x1": 0.25}} for _ in range(count)]


def _one_hot_records() -> list[dict[str, object]]:
    return [
        {"label": label, "inputs": {f"x{feature}": 0.85 if feature == label else 0.0 for feature in range(3)}}
        for label in range(3)
    ]


def _two_class_one_hot_records() -> list[dict[str, object]]:
    return [
        {"label": label, "inputs": {f"x{feature}": 0.85 if feature == label else 0.0 for feature in range(2)}}
        for label in range(2)
    ]


def test_multiclass_block_sequence_round_robin_order_preserves_per_class_order() -> None:
    records = [
        {"label": 0, "inputs": {"x0": 0.0}, "id": "0a"},
        {"label": 0, "inputs": {"x0": 0.0}, "id": "0b"},
        {"label": 1, "inputs": {"x0": 0.0}, "id": "1a"},
        {"label": 1, "inputs": {"x0": 0.0}, "id": "1b"},
        {"label": 2, "inputs": {"x0": 0.0}, "id": "2a"},
        {"label": 2, "inputs": {"x0": 0.0}, "id": "2b"},
    ]

    ordered = seq.order_records_by_class_round_robin(records, class_count=3)

    assert [record["id"] for record in ordered] == ["0a", "1a", "2a", "0b", "1b", "2b"]


def test_multiclass_block_sequence_emits_single_continuous_deck() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(2),
        eval_records=_target0_records(1),
        score_capacitance_f=5.0,
        score_load_resistance=3e6,
    )

    assert "\nB" not in netlist
    assert "Vrow0 row0 0 PWL(" in netlist
    assert "Vacc acc 0 PWL(" in netlist
    assert "Vapplyn applyn 0 PWL(" in netlist
    assert "Mhidden_pos0 row0 whp0 pre_p0 0 NMOS" in netlist
    assert "Melig0_n elig0 samp pre_p0 0 NMOS" in netlist
    assert "Mc0_f0_pos_cond actrow0 c0_vwp0 c0_score 0 NMOS" in netlist
    assert "Mc1_f0_vwp_dn_g c1_f0_vwp_dn c1_gvn0 vwlo_ref 0 NSENSE" in netlist
    assert "Cc0_score c0_score 0 5f IC=0" in netlist
    assert "Rc0_score c0_score 0 3000000" in netlist
    assert "* cycle 0 initial_eval label=0" in netlist
    assert "* cycle 1 train label=0" in netlist
    assert "* cycle 2 train label=0" in netlist
    assert "* cycle 3 final_eval label=0" in netlist


def test_multiclass_block_sequence_defaults_score_measure_to_timing_window() -> None:
    records = _target0_records(1)

    late_netlist = seq.generate_netlist(train_records=records, eval_records=records)
    early_netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        score_timing_mode="early",
    )
    explicit_netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        score_timing_mode="early",
        score_measure_ns=5.05,
    )

    assert ".meas tran c0_score_0 FIND V(c0_score) AT=8.50n" in late_netlist
    assert ".meas tran c0_score_0 FIND V(c0_score) AT=5.30n" in early_netlist
    assert ".meas tran c0_score_0 FIND V(c0_score) AT=5.05n" in explicit_netlist


def test_multiclass_block_sequence_can_sample_differential_activation_as_eligibility() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        eligibility_source_mode="act-raw",
    )

    assert "\nB" not in netlist
    assert "Melig0_n elig0 samp act_raw0 0 NMOS W=16u L=180n" in netlist
    assert "Melig0_p elig0 sampn act_raw0 vdd PMOS W=32u L=180n" in netlist
    assert "Melig0_n elig0 samp pre_p0 0 NMOS" not in netlist


def test_multiclass_block_sequence_can_sample_stored_activation_as_eligibility() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        eligibility_source_mode="act",
    )

    assert "\nB" not in netlist
    assert "Melig0_n elig0 samp act0 0 NMOS W=16u L=180n" in netlist
    assert "Melig0_p elig0 sampn act0 vdd PMOS W=32u L=180n" in netlist
    assert "Melig0_n elig0 samp pre_p0 0 NMOS" not in netlist


def test_multiclass_block_sequence_can_scale_nontarget_pressure() -> None:
    netlist = seq.generate_netlist(
        train_records=[
            {"label": 0, "inputs": {"x0": 0.85}},
            {"label": 1, "inputs": {"x0": 0.85}},
        ],
        eval_records=_target0_records(1),
        nontarget_scale=0.5,
        nontarget_width_scale=0.25,
    )

    c0_targetn = next(line for line in netlist.splitlines() if line.startswith("Vc0_targetn "))
    assert "Vc0_targetn c0_targetn 0 PWL(" in c0_targetn
    assert "41n 0.55 41.5n 0.55 41.51n 0" in c0_targetn
    assert "41n 1.1" not in c0_targetn
    assert "41n 0.55 43n 0.55" not in c0_targetn


def test_multiclass_block_sequence_can_gate_nontarget_pressure_with_score() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="score-gated-nontarget",
    )

    assert "Mc1_f0_gvn_label c1_f0_gvn_a c1_targetn c1_f0_gvn_label 0 NSENSE" in netlist
    assert "Mc1_f0_gvn_score c1_f0_gvn_label c1_score c1_f0_gvn_d 0 NSENSE" in netlist
    assert "Mc1_f0_gvn_d c1_f0_gvn_a c1_targetn c1_f0_gvn_d 0 NSENSE" not in netlist


def test_multiclass_block_sequence_can_restore_score_before_nontarget_gate() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="restored-score-nontarget",
    )

    assert "Vscoreamp scoreamp 0 PWL(" in netlist
    assert "Vscoredec scoredec 0 PWL(" in netlist
    assert "Coutref" not in netlist
    assert "Vc0_targetp c0_targetp 0 PWL(" in netlist
    assert "26.8n 1.1 28.8n 1.1" in netlist
    assert "Mc1_f0_gvn_score c1_f0_gvn_label c1_decision c1_f0_gvn_d 0 NSENSE" in netlist
    assert "Mc1_scoreamp_score_p c1_score_amp c1_score c1_scoreamp_score_i vdd PMOS" in netlist


def test_multiclass_block_sequence_can_use_restored_score_binary_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="restored-score-binary-descent",
    )

    assert "\nB" not in netlist
    assert "Cc0_decision c0_decision 0 20f IC=0" in netlist
    assert "Mc0_dec_low_gain_ref_tail c0_dec_src scoredec 0 0 NMOS" in netlist
    assert "Mc0_f0_gvp_decisionn c0_f0_gvp_label c0_decisionn c0_f0_gvp_decisionn 0 NSENSE" in netlist
    assert "Mc0_f0_gvn_decision c0_f0_gvn_label c0_decision c0_f0_gvn_decision 0 NSENSE" in netlist
    assert "Mc0_f0_gvn_score" not in netlist


def test_multiclass_block_sequence_can_use_pairwise_binary_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="pairwise-binary-descent",
    )

    assert "\nB" not in netlist
    assert "Cc0_gt_c1_decision c0_gt_c1_decision 0 20f IC=1.2" in netlist
    assert "Mc0_f0_gvp_pair0_gate c0_f0_gvp_pair0_label c1_gt_c0_decision c0_f0_gvp_pair0_gate 0 NSENSE" in netlist
    assert "Mc0_f0_gvn_pair0_gate c0_f0_gvn_pair0_label c0_gt_c1_decision c0_f0_gvn_pair0_gate 0 NSENSE" in netlist
    assert "Mc0_f0_rgp_pair0_acc c0_f0_rgp_pair0_gate acc 0 0 NREL" in netlist
    assert "Mc0_f0_rgn_pair0_acc c0_f0_rgn_pair0_gate acc 0 0 NREL" in netlist


def test_multiclass_block_sequence_can_use_residual_score_nontarget_gate() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="residual-score-nontarget",
    )

    assert "\nB" not in netlist
    assert "Vscoregaterst scoregaterst 0 PWL(" in netlist
    assert "Cc1_score_gate c1_score_gate 0 4f IC=0" in netlist
    assert "Mc1_score_gate_up_v vdd c1_score_amp c1_score_gate_up_i 0 NREL W=12u" in netlist
    assert "Mc1_score_gate_dn_v c1_score_gate c1_scoren_amp c1_score_gate_dn_i 0 NREL W=24u" in netlist
    assert "Mc1_f0_gvn_score c1_f0_gvn_label c1_score_gate c1_f0_gvn_d 0 NSENSE" in netlist
    assert "Mc1_f0_rgn_res_score c1_f0_rgn_res_label c1_score_gate c1_f0_rgn_res_score 0 NSENSE" in netlist
    assert "Mc1_f0_rgn_res_acc c1_f0_rgn_res_score acc 0 0 NREL" in netlist
    assert "Mc1_dec_low_gain_ref_tail" not in netlist


def test_multiclass_block_sequence_can_use_amplified_score_nontarget_gate() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="amplified-score-nontarget",
    )

    assert "\nB" not in netlist
    assert "Vscoregaterst" not in netlist
    assert "Cc1_score_gate" not in netlist
    assert "Mc1_score_gate_up_v" not in netlist
    assert "Mc1_f0_gvn_score c1_f0_gvn_label c1_score_amp c1_f0_gvn_d 0 NSENSE" in netlist
    assert "Mc1_f0_rgn_res_score c1_f0_rgn_res_label c1_score_amp c1_f0_rgn_res_score 0 NSENSE" in netlist
    assert "Mc1_scoreamp_score_p c1_score_amp c1_score c1_scoreamp_score_i vdd PMOS" in netlist
    assert "Mc1_dec_low_gain_ref_tail" not in netlist


def test_multiclass_block_sequence_can_gate_nontarget_with_common_score_reference() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="common-ref-score-nontarget",
    )

    assert "\nB" not in netlist
    assert "Vscoregaterst scoregaterst 0 PWL(" in netlist
    assert "Cscore_common score_common 0 4f IC=1.2" in netlist
    assert "Rscore_common_c0 score_common c0_score_amp 20000" in netlist
    assert "Rscore_common_c2 score_common c2_score_amp 20000" in netlist
    assert "Cc1_score_common_gate c1_score_common_gate 0 4f IC=0" in netlist
    assert "Mc1_score_common_gate_up_v vdd c1_score_amp c1_score_common_gate_up_i 0 NREL W=48u" in netlist
    assert "Mc1_score_common_gate_dn_v c1_score_common_gate score_common c1_score_common_gate_dn_i 0 NREL W=12u" in netlist
    assert "Mc1_f0_gvn_score c1_f0_gvn_label c1_score_common_gate c1_f0_gvn_d 0 NSENSE" in netlist
    assert ".meas tran c1_score_above_common_1" in netlist


def test_multiclass_block_sequence_can_gate_nontarget_with_raw_common_score_reference() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="raw-common-ref-score-nontarget",
    )

    assert "\nB" not in netlist
    assert "Cscore_common score_common 0 4f IC=1.2" in netlist
    assert "Rscore_common_c0 score_common c0_score 20000" in netlist
    assert "Rscore_common_c2 score_common c2_score 20000" in netlist
    assert "Mc1_score_common_gate_up_v vdd c1_score c1_score_common_gate_up_i 0 NSENSE W=48u" in netlist
    assert "Mc1_scoreamp_score_p" not in netlist
    assert ".meas tran c1_score_above_common_1 PARAM='c1_score_1-score_common_c1_1'" in netlist


def test_multiclass_block_sequence_can_gate_nontarget_with_label_selected_target_score() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="target-ref-score-nontarget",
    )

    assert "\nB" not in netlist
    assert "Vscoregaterst scoregaterst 0 PWL(" in netlist
    assert "Vc0_targetp c0_targetp 0 PWL(" in netlist
    assert "25.55n 1.1" in netlist
    assert "Ctarget_score_ref target_score_ref 0 4f IC=0" in netlist
    assert "Mtarget_score_ref_sel_c0 target_score_ref c0_targetp c0_score_amp 0 NSENSE W=24u" in netlist
    assert "Cc1_score_target_gate c1_score_target_gate 0 4f IC=0" in netlist
    assert "Mc1_score_common_gate_up_v vdd c1_score_amp c1_score_common_gate_up_i 0 NREL W=48u" in netlist
    assert "Mc1_score_common_gate_dn_v c1_score_target_gate target_score_ref c1_score_common_gate_dn_i 0 NREL W=12u" in netlist
    assert "Mc1_f0_gvn_score c1_f0_gvn_label c1_score_target_gate c1_f0_gvn_d 0 NSENSE" in netlist
    assert ".meas tran c1_score_above_target_1" in netlist


def test_multiclass_block_sequence_can_use_amplified_score_competitive_target_boost() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="amplified-score-competitive",
    )

    assert "\nB" not in netlist
    assert "Mc0_f0_gvn_score c0_f0_gvn_label c0_score_amp c0_f0_gvn_d 0 NSENSE" in netlist
    assert "Mc0_f0_gvp_comp0_score c0_f0_gvp_comp0_label c1_score_amp c0_f0_gvp_comp0_score 0 NSENSE" in netlist
    assert "Mc0_f0_rgp_comp0_acc c0_f0_rgp_comp0_score acc 0 0 NREL" in netlist
    assert "Mc0_f0_gvp_comp1_score c0_f0_gvp_comp1_label c2_score_amp c0_f0_gvp_comp1_score 0 NSENSE" in netlist
    assert "Mc1_f0_gvp_comp0_score c1_f0_gvp_comp0_label c0_score_amp c1_f0_gvp_comp0_score 0 NSENSE" in netlist
    assert "Mc1_f0_gvp_comp1_score c1_f0_gvp_comp1_label c2_score_amp c1_f0_gvp_comp1_score 0 NSENSE" in netlist
    assert "Mc0_dec_low_gain_ref_tail" not in netlist


def test_multiclass_block_sequence_can_blend_amplified_score_and_pairwise_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="amplified-score-pairwise",
    )

    assert "\nB" not in netlist
    assert "Mc0_scoreamp_score_p c0_score_amp c0_score c0_scoreamp_score_i vdd PMOS" in netlist
    assert "Cc0_gt_c1_decision c0_gt_c1_decision 0 20f IC=1.2" in netlist
    assert "Mc0_f0_gvn_score c0_f0_gvn_label c0_score_amp c0_f0_gvn_d 0 NSENSE" in netlist
    assert "Mc0_f0_rgn_res_score c0_f0_rgn_res_label c0_score_amp c0_f0_rgn_res_score 0 NSENSE" in netlist
    assert "Mc0_f0_gvp_paircorr0_gate c0_f0_gvp_paircorr0_label c1_gt_c0_decision c0_f0_gvp_paircorr0_gate 0 NSENSE" in netlist
    assert "Mc0_f0_gvn_paircorr0_gate c0_f0_gvn_paircorr0_label c0_gt_c1_decision c0_f0_gvn_paircorr0_gate 0 NSENSE" in netlist
    assert "Mc0_f0_rgp_paircorr0_gate c0_f0_rgp_paircorr0_label c1_gt_c0_decision c0_f0_rgp_paircorr0_gate 0 NSENSE" in netlist
    assert "Mc0_f0_rgn_paircorr0_gate c0_f0_rgn_paircorr0_label c0_gt_c1_decision c0_f0_rgn_paircorr0_gate 0 NSENSE" in netlist
    assert "Mc0_f0_rgp_pair_base" not in netlist
    assert "Mc0_f0_rgn_pair_base" not in netlist


def test_multiclass_block_sequence_can_blend_amplified_score_and_binary_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="amplified-score-binary-descent",
    )

    assert "\nB" not in netlist
    assert "Mc0_scoreamp_score_p c0_score_amp c0_score c0_scoreamp_score_i vdd PMOS" in netlist
    assert "Cc0_decision c0_decision 0 20f IC=0" in netlist
    assert "Mc0_dec_low_gain_ref_tail c0_dec_src scoredec 0 0 NMOS" in netlist
    assert "Mc0_f0_gvn_score c0_f0_gvn_label c0_score_amp c0_f0_gvn_d 0 NSENSE" in netlist
    assert "Mc0_f0_rgn_res_score c0_f0_rgn_res_label c0_score_amp c0_f0_rgn_res_score 0 NSENSE" in netlist
    assert "Mc0_f0_gvp_bincorr_gate c0_f0_gvp_bincorr_label c0_decisionn c0_f0_gvp_bincorr_gate 0 NSENSE" in netlist
    assert "Mc0_f0_gvn_bincorr_gate c0_f0_gvn_bincorr_label c0_decision c0_f0_gvn_bincorr_gate 0 NSENSE" in netlist
    assert "Mc0_f0_rgp_bincorr_gate c0_f0_rgp_bincorr_label c0_decisionn c0_f0_rgp_bincorr_gate 0 NSENSE" in netlist
    assert "Mc0_f0_rgn_bincorr_gate c0_f0_rgn_bincorr_label c0_decision c0_f0_rgn_bincorr_gate 0 NSENSE" in netlist
    assert "Mc0_f0_gvp_decisionn" not in netlist
    assert "Mc0_f0_gvn_decision" not in netlist


def test_multiclass_block_sequence_can_use_score_mass_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="score-mass-descent",
    )

    assert "\nB" not in netlist
    assert "Vscoreamp scoreamp 0 PWL(" in netlist
    assert "Vscoredec scoredec 0 PWL(" in netlist
    assert "Vscoreerr scoreerr 0 PWL(" in netlist
    assert "Vscoregaterst scoregaterst 0 PWL(" in netlist
    assert "25.55n 1.1" in netlist
    assert "Cscore_nontarget_mass score_nontarget_mass 0 8f IC=0" in netlist
    assert "Mmass_nt1_label vdd c1_targetn mass_nt1_a 0 NSENSE W=32u" in netlist
    assert "Mmass_nt1_score mass_nt1_a c1_score_amp mass_nt1_s 0 NSENSE W=32u" in netlist
    assert "Cc0_errp c0_errp 0 8f IC=0" in netlist
    assert "Mc0_errp_mass c0_errp_a score_nontarget_mass c0_errp_m 0 NSENSE W=32u" in netlist
    assert "Mc1_errn_score c1_errn_a c1_score_amp c1_errn_s 0 NSENSE W=32u" in netlist
    assert "Mc0_f0_gvp_e c0_f0_gvp_a c0_errp c0_f0_gvp_d 0 NSENSE" in netlist
    assert "Mc1_f0_gvn_e c1_f0_gvn_a c1_errn c1_f0_gvn_d 0 NSENSE" in netlist
    assert "Mc1_f0_gvn_score" not in netlist
    assert ".meas tran score_nontarget_mass_c0_1" in netlist
    assert ".meas tran c0_errdiff_1" in netlist


def test_multiclass_block_sequence_can_use_normalizer_current_sum_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="normalizer-current-sum-descent",
    )

    assert "\nB" not in netlist
    assert ".subckt norm_current_sum s0 s1 s2 tp0 tp1 tp2 tn0 tn1 tn2 phi rst" in netlist
    assert "Msg0_amp_p sg0 s0 sg0_i vdd PMOS W=1u L=180n" in netlist
    assert (
        "Xscore_normalizer c0_score c1_score c2_score "
        "c0_targetp c1_targetp c2_targetp c0_targetn c1_targetn c2_targetn "
        "scoreerr scoregaterst c0_errp c0_errn c1_errp c1_errn c2_errp c2_errn vdd 0 norm_current_sum"
        in netlist
    )
    assert "Mc0_f0_gvp_e c0_f0_gvp_a c0_errp c0_f0_gvp_d 0 NSENSE" in netlist
    assert "Mc1_f0_gvn_e c1_f0_gvn_a c1_errn c1_f0_gvn_d 0 NSENSE" in netlist
    assert ".meas tran c0_errdiff_1" in netlist


def test_multiclass_block_sequence_can_bound_normalizer_error_clock() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="normalizer-current-sum-descent",
        normalizer_error_clock_high=0.45,
    )

    assert "26.45n 0.45" in netlist


def test_multiclass_block_sequence_reduces_train_error_rail_stats() -> None:
    measures = {
        "c0_errdiff_1": -0.2,
        "c1_errdiff_1": 0.4,
        "c2_errdiff_1": -0.1,
        "c0_errdiff_2": 0.5,
        "c1_errdiff_2": -0.3,
        "c2_errdiff_2": -0.2,
    }
    stats = seq.error_rail_stats(
        measures,
        labels=[0, 1, 0],
        sequence=["initial_eval", "train", "train"],
        class_count=3,
    )

    assert stats["train_errdiff_rows_v"] == [[-0.2, 0.4, -0.1], [0.5, -0.3, -0.2]]
    assert stats["train_target_errdiff_min_v"] == pytest.approx(0.4)
    assert stats["train_target_errdiff_mean_v"] == pytest.approx(0.45)
    assert stats["train_nontarget_errdiff_max_v"] == pytest.approx(-0.1)
    assert stats["train_nontarget_errdiff_mean_v"] == pytest.approx(-0.2)


def test_multiclass_block_sequence_summarizes_train_eligibility_overlap() -> None:
    measures = {
        "elig_f0_1": 0.5,
        "elig_f1_1": 0.01,
        "elig_f2_1": 0.5,
        "elig_f0_2": 0.5,
        "elig_f1_2": 0.5,
        "elig_f2_2": 0.01,
    }

    stats = seq.eligibility_stats(
        measures,
        sequence=["initial_eval", "train", "train"],
        total_feature_count=3,
    )

    assert stats["train_eligibility_rows_v"] == [[0.5, 0.01, 0.5], [0.5, 0.5, 0.01]]
    assert stats["train_eligibility_active_features_25mv_mean"] == 2.0
    assert stats["train_eligibility_active_features_250mv_mean"] == 2.0
    assert stats["train_eligibility_active_features_500mv_mean"] == 0.0
    assert stats["train_eligibility_pairwise_cosine_mean"] == pytest.approx(0.519896, abs=1e-6)


def test_multiclass_block_sequence_summarizes_train_eligibility_gate_activity() -> None:
    measures = {
        "egate_f0_1": 1.1,
        "egate_f1_1": 0.02,
        "egate_f2_1": 0.03,
        "egate_f0_2": 0.1,
        "egate_f1_2": 1.0,
        "egate_f2_2": 0.9,
    }

    stats = seq.eligibility_gate_stats(
        measures,
        sequence=["initial_eval", "train", "train"],
        feature_count=3,
    )

    assert stats["train_eligibility_gate_rows_v"] == [[1.1, 0.02, 0.03], [0.1, 1.0, 0.9]]
    assert stats["train_eligibility_gate_active_features_250mv_mean"] == 1.5
    assert stats["train_eligibility_gate_active_features_250mv_max"] == 2
    assert stats["train_eligibility_gate_active_features_600mv_mean"] == 1.5
    assert stats["train_eligibility_gate_active_features_600mv_max"] == 2
    assert stats["train_eligibility_gate_max_v"] == 1.1


def test_multiclass_block_sequence_summarizes_train_hidden_credit() -> None:
    measures = {
        "hcredit_f0_1": 0.20,
        "hcredit_f1_1": -0.10,
        "hcredit_f0_2": 0.05,
        "hcredit_f1_2": -0.25,
    }

    stats = seq.hidden_credit_stats(
        measures,
        sequence=["initial_eval", "train", "train"],
        feature_count=2,
    )

    assert stats["train_hidden_credit_rows_v"] == [[0.20, -0.10], [0.05, -0.25]]
    assert stats["train_hidden_credit_abs_mean_v"] == pytest.approx(0.15)
    assert stats["train_hidden_credit_abs_max_v"] == pytest.approx(0.25)
    assert stats["train_hidden_credit_positive_mean_v"] == pytest.approx(0.125)
    assert stats["train_hidden_credit_negative_mean_v"] == pytest.approx(-0.175)


def test_multiclass_block_sequence_summarizes_final_signed_projection() -> None:
    measures = {
        "act_f0_2": 0.8,
        "act_f1_2": 0.1,
        "act_f0_3": 0.1,
        "act_f1_3": 0.7,
    }

    stats = seq.signed_readout_projection_stats(
        measures,
        labels=[0, 1, 0, 1],
        sequence=["initial_eval", "train", "final_eval", "final_eval"],
        class_count=2,
        total_feature_count=2,
        final_signed=[
            [0.4, -0.1],
            [-0.2, 0.3],
        ],
    )

    assert stats["final_eval_signed_projection_accuracy"] == 1.0
    assert stats["final_eval_signed_projection_min_margin_v2"] == pytest.approx(0.22)
    assert stats["final_eval_signed_projection_rows"][0]["prediction"] == 0
    assert stats["final_eval_signed_projection_rows"][1]["prediction"] == 1
    assert stats["final_eval_signed_projection_mean_abs_score_v2"] == pytest.approx(0.165)


def test_multiclass_block_sequence_summarizes_conductance_projection() -> None:
    measures = {
        "act_f0_2": 1.0,
        "act_f1_2": 0.5,
        "act_f0_3": 0.4,
        "act_f1_3": 1.0,
    }

    stats = seq.conductance_readout_projection_stats(
        measures,
        labels=[0, 1, 0, 1],
        sequence=["initial_eval", "train", "final_eval", "final_eval"],
        class_count=2,
        total_feature_count=2,
        final_positive=[
            [0.50, 0.34],
            [0.34, 0.52],
        ],
        final_negative=[
            [0.34, 0.36],
            [0.37, 0.34],
        ],
        positive_vto=0.35,
        negative_width_scale=0.75,
    )

    assert stats["final_eval_conductance_projection_accuracy"] == 1.0
    assert stats["final_eval_conductance_projection_min_margin_v2"] == pytest.approx(0.07625)
    assert stats["final_eval_conductance_projection_active_weights"] == 4


def test_multiclass_block_sequence_summarizes_projection_alignment() -> None:
    stats = seq.projection_alignment_stats(
        [
            {
                "cycle": 0,
                "sequence": "final_eval",
                "label": 0,
                "prediction": 0,
                "score_margin_v": 0.2,
                "score_c0_v": 0.30,
                "score_c1_v": 0.10,
            },
            {
                "cycle": 1,
                "sequence": "final_eval",
                "label": 1,
                "prediction": 0,
                "score_margin_v": -0.1,
                "score_c0_v": 0.20,
                "score_c1_v": 0.10,
            },
        ],
        [
            {
                "cycle": 0,
                "label": 0,
                "prediction": 0,
                "score_margin_v2": 0.4,
                "score_c0_v2": 0.60,
                "score_c1_v2": 0.20,
            },
            {
                "cycle": 1,
                "label": 1,
                "prediction": 1,
                "score_margin_v2": 0.1,
                "score_c0_v2": 0.10,
                "score_c1_v2": 0.20,
            },
        ],
        class_count=2,
        prefix="signed_projection",
    )

    assert stats["final_eval_signed_projection_prediction_agreement"] == pytest.approx(0.5)
    assert stats["final_eval_signed_projection_score_correlation_mean"] == pytest.approx(0.0)
    assert stats["final_eval_signed_projection_alignment_rows"][1]["physical_prediction"] == 0
    assert stats["final_eval_signed_projection_alignment_rows"][1]["projection_prediction"] == 1


def test_multiclass_block_sequence_summarizes_readout_matrix_class_bias() -> None:
    stats = seq.readout_weight_matrix_stats(
        final_signed=[
            [-0.08, -0.10, 0.12],
            [-0.06, -0.07, 0.11],
            [0.10, 0.12, 0.10],
        ],
        final_positive=[
            [0.28, 0.28, 0.40],
            [0.28, 0.28, 0.40],
            [0.39, 0.40, 0.40],
        ],
        final_negative=[
            [0.36, 0.38, 0.28],
            [0.34, 0.35, 0.29],
            [0.29, 0.28, 0.30],
        ],
        feature_count=2,
    )

    assert stats["final_readout_signed_feature_class_means_v"] == pytest.approx([-0.09, -0.065, 0.11])
    assert stats["final_readout_signed_feature_class_mean_spread_v"] == pytest.approx(0.20)
    assert stats["final_readout_signed_feature_centered_abs_mean_v"] == pytest.approx(0.008333333333333333)
    assert stats["final_readout_signed_feature_centered_abs_max_v"] == pytest.approx(0.01)
    assert stats["final_readout_signed_class_mean_to_feature_centered_abs_ratio"] == pytest.approx(
        0.20 / 0.008333333333333333
    )
    assert stats["final_readout_common_feature_class_means_v"] == pytest.approx([0.325, 0.3125, 0.34])
    assert stats["final_readout_common_feature_class_mean_spread_v"] == pytest.approx(0.0275)


def test_multiclass_block_sequence_summarizes_activation_prototype_projection() -> None:
    measures = {
        "act_f0_0": 0.9,
        "act_f1_0": 0.1,
        "act_f0_1": 0.1,
        "act_f1_1": 0.8,
        "act_f0_2": 0.7,
        "act_f1_2": 0.2,
        "act_f0_3": 0.2,
        "act_f1_3": 0.9,
    }

    stats = seq.activation_prototype_projection_stats(
        measures,
        labels=[0, 1, 0, 1],
        sequence=["train", "train", "final_eval", "final_eval"],
        class_count=2,
        total_feature_count=2,
    )

    assert stats["final_eval_activation_prototype_accuracy"] == 1.0
    assert stats["final_eval_activation_prototype_min_margin_v2"] == pytest.approx(0.42)
    assert stats["final_eval_activation_prototype_rows"][0]["prediction"] == 0
    assert stats["final_eval_activation_prototype_rows"][1]["prediction"] == 1
    assert stats["final_eval_activation_cosine_prototype_accuracy"] == 1.0
    assert stats["final_eval_activation_cosine_prototype_min_margin"] == pytest.approx(0.59412, abs=1e-5)
    assert stats["final_eval_activation_prototype_pairwise_cosine_mean"] == pytest.approx(0.23285, abs=1e-5)


def test_multiclass_block_sequence_summarizes_hidden_weight_progress() -> None:
    measures = {
        "whsigned_f0_after_train1": 0.79,
        "whsigned_f1_after_train1": 0.82,
        "whsigned_f0_after_train2": 0.76,
        "whsigned_f1_after_train2": 0.84,
        "whsigned_f0_final": 0.75,
        "whsigned_f1_final": 0.85,
    }

    stats = seq.hidden_weight_progress_stats(
        measures,
        train_count=2,
        feature_count=2,
        initial_signed_v=0.80,
    )

    assert stats["hidden_signed_after_each_train_v"] == [[0.79, 0.82], [0.76, 0.84]]
    assert stats["final_hidden_signed_delta_v"] == pytest.approx([-0.05, 0.05])
    assert stats["final_hidden_signed_delta_mean_v"] == pytest.approx(0.0)
    assert stats["final_hidden_signed_delta_abs_mean_v"] == pytest.approx(0.05)
    assert stats["final_hidden_signed_delta_min_v"] == pytest.approx(-0.05)
    assert stats["final_hidden_signed_delta_max_v"] == pytest.approx(0.05)


def test_multiclass_block_sequence_summarizes_readout_update_eligibility() -> None:
    measures = {
        "relig_f0_1": 0.40,
        "relig_f1_1": 0.02,
        "relig_f0_2": 0.30,
        "relig_f1_2": 0.28,
        "relig_update_f0_1": 0.38,
        "relig_update_f1_1": 0.01,
        "relig_update_f0_2": 0.25,
        "relig_update_f1_2": 0.20,
        "relig_pgate_f0_1": 0.70,
        "relig_pgate_f1_1": 1.18,
        "relig_pgate_f0_2": 0.80,
        "relig_pgate_f1_2": 0.85,
    }

    stats = seq.readout_update_eligibility_stats(
        measures,
        sequence=["initial_eval", "train", "train"],
        feature_count=2,
    )

    assert stats["train_readout_update_eligibility_rows_v"] == [[0.40, 0.02], [0.30, 0.28]]
    assert stats["train_readout_update_eligibility_active_features_25mv_mean"] == pytest.approx(1.5)
    assert stats["train_readout_update_eligibility_active_features_250mv_mean"] == pytest.approx(1.5)
    assert stats["train_readout_update_eligibility_active_features_500mv_mean"] == pytest.approx(0.0)
    assert stats["train_readout_update_eligibility_max_v"] == pytest.approx(0.40)
    assert stats["train_readout_update_eligibility_at_writer_rows_v"] == [[0.38, 0.01], [0.25, 0.20]]
    assert stats["train_readout_update_eligibility_at_writer_active_features_250mv_mean"] == pytest.approx(0.5)
    assert stats["train_readout_update_eligibility_at_writer_max_v"] == pytest.approx(0.38)
    assert stats["train_readout_update_eligibility_pgate_rows_v"] == [[0.70, 1.18], [0.80, 0.85]]
    assert stats["train_readout_update_eligibility_pgate_min_v"] == pytest.approx(0.70)
    assert stats["train_readout_update_eligibility_pgate_max_v"] == pytest.approx(1.18)


def test_multiclass_block_sequence_summarizes_readout_train_progress_by_label() -> None:
    stats = seq.readout_train_progress_stats(
        train_progress=[
            [
                [0.10, 0.30, 0.95],
                [-0.05, -0.05, 0.80],
                [-0.02, -0.02, 0.70],
            ],
            [
                [0.08, 0.28, 0.96],
                [-0.06, -0.04, 0.81],
                [0.18, 0.22, 0.71],
            ],
        ],
        train_labels=[0, 2],
        feature_count=2,
        class_count=3,
        initial_signed_v=0.0,
    )

    np.testing.assert_allclose(
        stats["train_readout_signed_feature_class_means_after_each_train_v"],
        [[0.20, -0.05, -0.02], [0.18, -0.05, 0.20]],
    )
    assert stats["train_readout_signed_feature_class_mean_spread_after_each_train_v"] == pytest.approx([0.25, 0.25])
    np.testing.assert_allclose(
        stats["train_readout_signed_feature_class_mean_delta_after_each_train_v"],
        [[0.20, -0.05, -0.02], [-0.02, 0.0, 0.22]],
        atol=1e-12,
    )
    assert stats["train_readout_target_class_feature_mean_delta_by_train_v"] == pytest.approx([0.20, 0.22])
    assert stats["train_readout_nontarget_class_feature_mean_delta_by_train_v"] == pytest.approx([-0.035, -0.01])
    assert stats["train_readout_target_minus_nontarget_delta_by_train_v"] == pytest.approx([0.235, 0.23])
    assert stats["train_readout_target_minus_nontarget_delta_mean_v"] == pytest.approx(0.2325)
    assert stats["train_readout_target_minus_nontarget_delta_min_v"] == pytest.approx(0.23)
    assert stats["train_readout_update_rows"][1]["label"] == 2
    assert stats["train_readout_update_rows"][1]["target_class_feature_mean_after_v"] == pytest.approx(0.20)
    assert stats["train_readout_update_rows"][1]["target_minus_nontarget_delta_v"] == pytest.approx(0.23)


def test_multiclass_block_sequence_can_use_common_score_mass_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="common-score-mass-descent",
    )

    assert "\nB" not in netlist
    assert "Cscore_common score_common 0 4f IC=1.2" in netlist
    assert "Rscore_common_c0 score_common c0_score_amp 20000" in netlist
    assert "Cc1_score_common_gate c1_score_common_gate 0 4f IC=0" in netlist
    assert "Mc1_score_common_gate_up_v vdd c1_score_amp c1_score_common_gate_up_i 0 NREL W=192u" in netlist
    assert "Mc1_score_common_gate_dn_v c1_score_common_gate score_common c1_score_common_gate_dn_i 0 NREL W=6u" in netlist
    assert "Mmass_nt1_score mass_nt1_a c1_score_common_gate mass_nt1_s 0 NSENSE W=128u" in netlist
    assert "Cc1_errn c1_errn 0 0.5f IC=0" in netlist
    assert "Mc1_errn_score c1_errn_a c1_score_common_gate c1_errn_s 0 NSENSE W=128u" in netlist
    assert "Mc1_f0_gvn_e c1_f0_gvn_a c1_errn c1_f0_gvn_d 0 NSENSE" in netlist
    assert ".meas tran c1_score_common_gate_1" in netlist
    assert ".meas tran c1_errdiff_1" in netlist


def test_multiclass_block_sequence_can_use_target_contrast_score_mass_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="target-contrast-score-mass-descent",
    )

    assert "\nB" not in netlist
    assert "Ctarget_score_ref target_score_ref 0 4f IC=0" in netlist
    assert "Mtarget_score_ref_sel_c0 target_score_ref c0_targetp c0_score_amp 0 NSENSE W=24u" in netlist
    assert "Cc1_score_target_gate c1_score_target_gate 0 4f IC=0" in netlist
    assert "Mc1_score_common_gate_up_v vdd c1_score_amp c1_score_common_gate_up_i 0 NREL W=192u" in netlist
    assert "Mc1_score_common_gate_dn_v c1_score_target_gate target_score_ref c1_score_common_gate_dn_i 0 NREL W=6u" in netlist
    assert "Mmass_nt1_score mass_nt1_a c1_score_target_gate mass_nt1_s 0 NSENSE W=128u" in netlist
    assert "Cc1_errn c1_errn 0 0.5f IC=0" in netlist
    assert "Mc1_errn_score c1_errn_a c1_score_target_gate c1_errn_s 0 NSENSE W=128u" in netlist
    assert "Mc1_f0_gvn_e c1_f0_gvn_a c1_errn c1_f0_gvn_d 0 NSENSE" in netlist
    assert ".meas tran c1_score_target_gate_1" in netlist
    assert ".meas tran c1_errdiff_1" in netlist


def test_multiclass_block_sequence_can_use_pairwise_score_competition_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="pairwise-score-competition-descent",
    )

    assert "\nB" not in netlist
    assert "Vscoreerr scoreerr 0 PWL(" in netlist
    assert "Cc0_gt_c1_decision c0_gt_c1_decision 0 4f IC=0" in netlist
    assert "Mc0_gt_c1_scoreamp_score_p c0_gt_c1_score_amp c0_score c0_gt_c1_scoreamp_score_i vdd PMOS" in netlist
    assert "Mc0_gt_c1_dec_pair_tail c0_gt_c1_dec_src scoredec 0 0 NMOS W=64u" in netlist
    assert "Cc1_errp c1_errp 0 0.5f IC=0" in netlist
    assert "Mt0_o1_errp_sup t0_o1_errp_sup c0_gt_c1_decision vdd vdd PMOS W=32u" in netlist
    assert "Mt0_o1_errp_win t0_o1_errp_t c1_gt_c0_decision t0_o1_errp_w 0 NSENSE W=32u" in netlist
    assert "Mc0_f0_gvp_e c0_f0_gvp_a c0_errp c0_f0_gvp_d 0 NSENSE" in netlist
    assert "Mc1_f0_gvn_e c1_f0_gvn_a c1_errn c1_f0_gvn_d 0 NSENSE" in netlist
    assert ".meas tran c0_errdiff_1" in netlist


def test_multiclass_block_sequence_can_use_pairwise_margin_correction_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="pairwise-margin-correction-descent",
    )

    assert "\nB" not in netlist
    assert "Vscoreerr scoreerr 0 PWL(" in netlist
    assert "26.45n 0.45" in netlist
    assert "Cc0_gt_c1_decision c0_gt_c1_decision 0 4f IC=0" in netlist
    assert "Mmpen_t0_o1_label c0_gt_c1_decision c0_targetp mpen_t0_o1_i 0 NSENSE W=0.25u" in netlist
    assert "Mmpen_t0_o1_clk mpen_t0_o1_i scoredec 0 0 NMOS W=0.25u" in netlist
    assert "Cc1_errp c1_errp 0 0.5f IC=0" in netlist
    assert "Mt0_o1_errp_sup t0_o1_errp_sup c0_gt_c1_decision vdd vdd PMOS W=128u" in netlist
    assert "Mc0_f0_gvp_e c0_f0_gvp_a c0_errp c0_f0_gvp_d 0 NSENSE" in netlist
    assert ".meas tran c0_errdiff_1" in netlist

    wider_margin = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="pairwise-margin-correction-descent",
        pairwise_margin_target_v=2.0e-3,
        pairwise_margin_error_drive_scale=0.5,
    )
    assert "Mmpen_t0_o1_label c0_gt_c1_decision c0_targetp mpen_t0_o1_i 0 NSENSE W=0.5u" in wider_margin
    assert "Mt0_o1_errp_sup t0_o1_errp_sup c0_gt_c1_decision vdd vdd PMOS W=64u" in wider_margin


def test_multiclass_block_sequence_can_use_pairwise_margin_centered_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="pairwise-margin-centered-descent",
        readout_update_mode="live",
    )

    assert "\nB" not in netlist
    assert "Cc0_errp_raw c0_errp_raw 0 0.5f IC=0" in netlist
    assert "Cc0_errp c0_errp 0 4f IC=0" in netlist
    assert "Mcenter_c0_common_p vdd c0_errp_raw class_errp_common 0 NSENSE" in netlist
    assert "Mcenter_c0_subtract_common_p vdd class_errp_common c0_errn 0 NSENSE" in netlist
    assert "Mc0_f0_live_pos_up_d c0_f0_live_pos_up c0_errp c0_vwp0 0 NSENSE" in netlist
    assert ".meas tran c0_errdiff_1 PARAM='c0_errp_1-c0_errn_1'" in netlist


def test_multiclass_block_sequence_can_use_pairwise_margin_centered_gain_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="pairwise-margin-centered-gain-descent",
        readout_update_mode="live",
    )

    assert "\nB" not in netlist
    assert "Cc0_errp_raw c0_errp_raw 0 0.5f IC=0" in netlist
    assert "Cc0_errp_ctr c0_errp_ctr 0 4f IC=0" in netlist
    assert "Mcenter_c0_local_p vdd c0_errp_raw c0_errp_ctr 0 NSENSE" in netlist
    assert "Cc0_errp c0_errp 0 1f IC=0" in netlist
    assert "Mrestore_c0_errp vdd c0_errp_ctr c0_errp 0 NSENSE W=128u" in netlist
    assert "Mc0_f0_live_pos_up_d c0_f0_live_pos_up c0_errp c0_vwp0 0 NSENSE" in netlist


def test_multiclass_block_sequence_can_use_common_score_mass_pairwise_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="common-score-mass-pairwise-descent",
    )

    assert "\nB" not in netlist
    assert "Cscore_common score_common 0 4f IC=1.2" in netlist
    assert "Mmass_nt1_score mass_nt1_a c1_score_common_gate mass_nt1_s 0 NSENSE W=128u" in netlist
    assert "Cc1_errp c1_errp 0 0.5f IC=0" in netlist
    assert netlist.count("Cc1_errp c1_errp 0 0.5f IC=0") == 1
    assert "Cc0_gt_c1_decision c0_gt_c1_decision 0 4f IC=0" in netlist
    assert "Mt0_o1_errp_sup t0_o1_errp_sup c0_gt_c1_decision vdd vdd PMOS W=2u" in netlist
    assert "Mt0_o1_errp_win t0_o1_errp_t c1_gt_c0_decision t0_o1_errp_w 0 NSENSE W=2u" in netlist
    assert "Mc0_f0_gvp_e c0_f0_gvp_a c0_errp c0_f0_gvp_d 0 NSENSE" in netlist
    assert ".meas tran score_nontarget_mass_c0_1" in netlist
    assert ".meas tran c0_errdiff_1" in netlist

    scaled = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="common-score-mass-pairwise-descent",
        score_mass_pairwise_error_scale=0.125,
    )
    assert "Mt0_o1_errp_sup t0_o1_errp_sup c0_gt_c1_decision vdd vdd PMOS W=4u" in scaled


def test_multiclass_block_sequence_can_use_contrast_score_mass_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="contrast-score-mass-descent",
    )

    assert "\nB" not in netlist
    assert "Vscore_contrast_ref score_contrast_ref 0 0.6" in netlist
    assert "Cscore_common score_common 0 4f IC=1.2" in netlist
    assert "Cc0_score_contrast c0_score_contrast 0 10f IC=0.6" in netlist
    assert "Mreset_c0_score_contrast c0_score_contrast scoregaterst score_contrast_ref 0 NMOS W=4u" in netlist
    assert "Mc0_score_contrast_up_v vdd c0_score_amp c0_score_contrast_up 0 NREL W=192u" in netlist
    assert "Mc0_score_contrast_dn_v c0_score_contrast score_common c0_score_contrast_dn 0 NREL W=24u" in netlist
    assert "Mmass_nt1_score mass_nt1_a c1_score_contrast mass_nt1_s 0 NSENSE W=128u" in netlist
    assert ".meas tran c0_score_contrast_1" in netlist
    assert ".meas tran c0_errdiff_1" in netlist


def test_multiclass_block_sequence_can_use_low_gain_contrast_score_mass_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="low-gain-contrast-score-mass-descent",
    )

    assert "\nB" not in netlist
    assert "Vscore_contrast_ref score_contrast_ref 0 0.6" in netlist
    assert "Mc0_scoreamp_score_p c0_score_amp c0_score c0_scoreamp_score_i vdd PMOS" in netlist
    assert "Cscore_common score_common 0 4f IC=1.2" in netlist
    assert "Rscore_common_c0 score_common c0_score_amp 10000000" in netlist
    assert "Cc0_score_contrast c0_score_contrast 0 10f IC=0.6" in netlist
    assert "Mc0_score_contrast_up_v vdd c0_score_amp c0_score_contrast_up 0 NREL W=192u" in netlist
    assert "Mc0_score_contrast_dn_v c0_score_contrast score_common c0_score_contrast_dn 0 NREL W=24u" in netlist
    assert "Mmass_nt1_score mass_nt1_a c1_score_contrast mass_nt1_s 0 NSENSE W=128u" in netlist
    assert "Cc1_errn c1_errn 0 0.5f IC=0" in netlist
    assert "Mc1_f0_gvn_e c1_f0_gvn_a c1_errn c1_f0_gvn_d 0 NSENSE" in netlist
    assert ".meas tran c0_score_contrast_1" in netlist
    assert ".meas tran c0_errdiff_1" in netlist


def test_multiclass_block_sequence_can_use_contrast_gated_score_mass_descent() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="contrast-gated-score-mass-descent",
    )

    assert "\nB" not in netlist
    assert "Vscore_contrast_ref score_contrast_ref 0 0.6" in netlist
    assert "Vscoregate scoregate 0 PWL" in netlist
    assert "Cc0_score_contrast c0_score_contrast 0 10f IC=0.6" in netlist
    assert "Cc0_score_contrast_gate c0_score_contrast_gate 0 4f IC=0" in netlist
    assert "Mc0_score_common_gate_up_v vdd c0_score_contrast c0_score_common_gate_up_i 0 NREL W=192u" in netlist
    assert "Mc0_score_common_gate_dn_v c0_score_contrast_gate score_contrast_ref c0_score_common_gate_dn_i 0 NREL W=24u" in netlist
    assert "Mc0_score_common_gate_up_t c0_score_common_gate_up_i scoregate c0_score_contrast_gate 0 NSENSE W=192u" in netlist
    assert "Mmass_nt1_score mass_nt1_a c1_score_contrast_gate mass_nt1_s 0 NSENSE W=128u" in netlist
    assert ".meas tran c0_score_contrast_1" in netlist
    assert ".meas tran c0_errdiff_1" in netlist


def test_multiclass_block_sequence_can_add_target_only_class_bias_row() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        feature_count=1,
        class_bias_mode="target-only",
    )

    assert "\nB" not in netlist
    assert "Vrow1 row1 0 PWL(" in netlist
    assert "Mhidden_pos1 row1 whp1 pre_p1 0 NMOS" in netlist
    assert "Mc0_f1_pos_cond actrow1 c0_vwp1 c0_score 0 NMOS" in netlist
    assert "Mc0_f1_gvp_d c0_f1_gvp_a c0_targetp c0_f1_gvp_d 0 NSENSE" in netlist
    assert "Mc0_f1_gvn_d" not in netlist
    assert "Mc0_f1_rgn_res" not in netlist
    assert ".meas tran c0_f1_signed_final" in netlist


def test_multiclass_block_sequence_can_add_label_descent_class_bias_row() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        feature_count=1,
        class_bias_mode="label-descent",
    )

    assert "\nB" not in netlist
    assert "Vrow1 row1 0 PWL(" in netlist
    assert "Mc0_f1_gvp_d c0_f1_gvp_a c0_targetp c0_f1_gvp_d 0 NSENSE" in netlist
    assert "Mc1_f1_gvn_d c1_f1_gvn_a c1_targetn c1_f1_gvn_d 0 NSENSE" in netlist
    assert "Mc1_f1_rgn_pd c1_rgn1 c1_gvn1 0 0 NSENSE" in netlist


def test_multiclass_block_sequence_can_add_passive_readout_center_leak() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        feature_count=1,
        readout_center_resistance=7e6,
        readout_center_voltage=0.39,
    )

    assert "\nB" not in netlist
    assert "Vreadout_center readout_center 0 0.39" in netlist
    assert "Rc0_vwp0_center c0_vwp0 readout_center 7000000" in netlist
    assert "Rc0_vwn0_center c0_vwn0 readout_center 7000000" in netlist
    assert "Rc2_vwp0_center c2_vwp0 readout_center 7000000" in netlist


def test_multiclass_block_sequence_can_use_live_readout_update_without_gradient_storage() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        readout_update_mode="live",
        error_mode="pairwise-margin-correction-descent",
    )

    assert "\nB" not in netlist
    assert "Vacc acc" not in netlist
    assert "Vapply" not in netlist
    assert "Cc0_gvp0" not in netlist
    assert "Cc0_rgp0" not in netlist
    assert "Vscoreerr scoreerr 0 PWL(" in netlist
    assert "Vscoregaterst scoregaterst 0 PWL(0n 0 0.19n 0 0.2n 1.2 1n 1.2" in netlist
    assert "21.45n 1.2" not in netlist
    assert "Cc0_errp c0_errp 0" in netlist
    assert "Mc0_f0_live_pos_up_e vwhi_ref elig0 c0_f0_live_pos_up 0 NSENSE" in netlist
    assert "Mc0_f0_live_pos_up_d c0_f0_live_pos_up c0_errp c0_vwp0 0 NSENSE" in netlist
    assert "Mc0_f0_live_neg_dn_d c0_f0_live_neg_dn c0_errn vwlo_ref 0 NSENSE" in netlist


def test_multiclass_block_sequence_can_gate_live_writer_with_feature_competition() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_two_feature_records(1),
        eval_records=_target0_two_feature_records(1),
        feature_count=2,
        readout_update_mode="live",
        error_mode="pairwise-margin-correction-descent",
        eligibility_gate_mode="competition",
    )

    assert "\nB" not in netlist
    assert "Vacc acc" not in netlist
    assert "Veligpre eligpre 0 PWL(" in netlist
    assert "Veligdec eligdec 0 PWL(" in netlist
    assert "Veliggate eliggate 0 PWL(" in netlist
    assert "Ce0_gt_e1_decision e0_gt_e1_decision 0 12f IC=1.2" in netlist
    assert "Mprecharge_e0_gt_e1_decision e0_gt_e1_decision eligpre vdd vdd PMOS" in netlist
    assert "Me1_loss_to_e0_mid_dec egate1 e0_gt_e1_decision e1_loss_to_e0_mid 0 NHIGH" in netlist
    assert ".meas tran egate_f0_1 FIND V(egate0) AT=20.85n" in netlist
    assert "Mc0_f0_live_pos_up_e vwhi_ref egate0 c0_f0_live_pos_up 0 NSENSE" in netlist
    assert "Mc0_f1_live_pos_up_e vwhi_ref egate1 c0_f1_live_pos_up 0 NSENSE" in netlist


def test_multiclass_block_sequence_can_rank_gate_live_writer_with_feature_competition() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_two_feature_records(1),
        eval_records=_target0_two_feature_records(1),
        feature_count=2,
        readout_update_mode="live",
        error_mode="pairwise-margin-correction-descent",
        eligibility_gate_mode="rank",
    )

    assert "\nB" not in netlist
    assert "Cegate0 egate0 0 50f IC=0" in netlist
    assert "Me1_rank_loss_to_e0_mid_dec egate1 e0_gt_e1_decision e1_rank_loss_to_e0_mid 0 NHIGH W=0.5u" in netlist
    assert "Me1_loss_to_e0_mid_dec" not in netlist
    assert "Mc0_f0_live_pos_up_e vwhi_ref egate0 c0_f0_live_pos_up 0 NSENSE" in netlist


def test_multiclass_block_sequence_can_contrast_gate_live_writer_with_feature_common_mode() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_two_feature_records(1),
        eval_records=_target0_two_feature_records(1),
        feature_count=2,
        readout_update_mode="live",
        error_mode="pairwise-margin-correction-descent",
        eligibility_gate_mode="contrast",
    )

    assert "\nB" not in netlist
    assert "Veliggate eliggate 0 PWL(" in netlist
    assert "Veligpre" not in netlist
    assert "Ce0_gt_e1_decision" not in netlist
    assert "Celig_common elig_common 0 1f IC=0" in netlist
    assert "Relig_common_e0 elig_common elig0 1000000" in netlist
    assert "Me0_contrast_up_v vdd elig0 e0_contrast_up_i 0 NSENSE W=512u" in netlist
    assert "Me0_contrast_dn_v egate0 elig_common e0_contrast_dn_i 0 NSENSE W=24u" in netlist
    assert "Mc0_f0_live_pos_up_e vwhi_ref egate0 c0_f0_live_pos_up 0 NSENSE" in netlist


def test_multiclass_block_sequence_can_use_current_clamp_score_diagnostic() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_two_feature_records(1),
        eval_records=_target0_two_feature_records(1),
        feature_count=2,
        readout_update_mode="live",
        error_mode="label-descent",
        score_sense_mode="current-clamp",
    )

    assert "\nB" not in netlist
    assert "Vc0_score_clamp c0_score 0 0" in netlist
    assert "Vc0_scoren_clamp c0_scoren 0 0" in netlist
    assert "Cc0_score" not in netlist
    assert "Mreset_c0_score" not in netlist
    assert ".meas tran c0_score_0 FIND I(Vc0_score_clamp)" in netlist
    assert ".meas tran c0_score_net_0 PARAM='c0_score_0-c0_scoren_0'" in netlist


def test_multiclass_block_sequence_can_use_diode_mirror_score_sensor() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_two_feature_records(1),
        eval_records=_target0_two_feature_records(1),
        feature_count=2,
        readout_update_mode="live",
        error_mode="label-rail-descent",
        score_sense_mode="diode-mirror",
    )

    assert "\nB" not in netlist
    assert "Vrstn rstn 0 PWL(" in netlist
    assert "Mc0_score_diode c0_score c0_score 0 0 NSENSE W=64u L=180n" in netlist
    assert "Cc0_score_mirror c0_score_mirror 0 20f IC=1.2" in netlist
    assert "Mc0_score_mirror_rst c0_score_mirror rstn vdd vdd PMOS W=16u L=180n" in netlist
    assert "Mc0_score_mirror_sink c0_score_mirror c0_score 0 0 NSENSE W=4u L=180n" in netlist
    assert "Vc0_score_clamp" not in netlist
    assert ".meas tran c0_score_mirror_0 FIND V(c0_score_mirror)" in netlist
    assert ".meas tran c0_score_0 PARAM='1.2-c0_score_mirror_0'" in netlist
    assert ".meas tran c0_score_net_0 PARAM='c0_score_0-c0_scoren_0'" in netlist


def test_multiclass_block_sequence_can_diode_isolate_readout_forward_path() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_two_feature_records(1),
        eval_records=_target0_two_feature_records(1),
        feature_count=2,
        readout_forward_mode="diode",
    )

    assert "\nB" not in netlist
    assert "Cc0_f0_midp c0_f0_midp 0 0.1f IC=0" in netlist
    assert "Rc0_f1_midn c0_f1_midn 0 1G" in netlist
    assert "Mc0_f0_pos_cond actrow0 c0_vwp0 c0_f0_midp 0 NMOS" in netlist
    assert "Mc0_f0_pos_diode c0_f0_midp c0_f0_midp c0_score 0 NSENSE" in netlist
    assert "Mc0_f1_neg_diode c0_f1_midn c0_f1_midn c0_scoren 0 NSENSE" in netlist
    assert "Mc0_f0_pos_cond actrow0 c0_vwp0 c0_score 0 NMOS" not in netlist


def test_multiclass_block_sequence_can_tune_contrast_gate_common_reference() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_two_feature_records(1),
        eval_records=_target0_two_feature_records(1),
        feature_count=2,
        readout_update_mode="live",
        error_mode="pairwise-margin-correction-descent",
        eligibility_gate_mode="contrast",
        eligibility_contrast_common_resistance_ohm=2.5e6,
        eligibility_contrast_common_capacitance_f=0.5,
    )

    assert "Celig_common elig_common 0 0.5f IC=0" in netlist
    assert "Relig_common_e0 elig_common elig0 2500000" in netlist


def test_multiclass_block_sequence_eligibility_gate_mode_validation() -> None:
    with pytest.raises(ValueError, match="eligibility_gate_mode"):
        seq.generate_netlist(
            train_records=_target0_records(1),
            eval_records=_target0_records(1),
            eligibility_gate_mode="missing",
        )


def test_multiclass_block_sequence_can_sample_score_during_early_differential_window() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        score_measure_ns=5.2,
    )

    assert ".meas tran c0_score_0 FIND V(c0_score) AT=5.20n" in netlist
    assert ".meas tran c0_score_1 FIND V(c0_score) AT=21.20n" in netlist
    assert ".meas tran c0_score_2 FIND V(c0_score) AT=37.20n" in netlist


def test_multiclass_block_sequence_can_move_score_error_timing_to_early_window() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        readout_update_mode="live",
        error_mode="pairwise-margin-correction-descent",
        score_timing_mode="early",
        score_measure_ns=5.2,
    )

    assert "Vout out 0 PWL(0n 0 4.99n 0 5n 1.2 5.35n 1.2" in netlist
    assert "Vscorepre scorepre 0 PWL(" in netlist
    assert "21.45n 0 21.7n 0" in netlist
    assert "Vscoreamp scoreamp 0 PWL(" in netlist
    assert "21.95n 1.2 22.75n 1.2" in netlist
    assert "Vscoredec scoredec 0 PWL(" in netlist
    assert "22.95n 1.2 23.65n 1.2" in netlist
    assert "Vscoreerr scoreerr 0 PWL(" in netlist
    assert "23.8n 0.45 24.1n 0.45" in netlist
    assert ".meas tran c0_gt_c1_decision_1 FIND V(c0_gt_c1_decision) AT=23.85n" in netlist
    assert ".meas tran c0_errp_1 FIND V(c0_errp) AT=24.13n" in netlist
    assert "Vapply" not in netlist


def test_multiclass_block_sequence_can_live_update_hidden_weights_from_readout_credit() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        readout_update_mode="live",
        hidden_update_mode="readout-weighted",
        hidden_credit_width_u=7.5,
        hidden_update_width_u=0.2,
        error_mode="pairwise-margin-correction-descent",
        score_timing_mode="early",
    )

    assert "\nB" not in netlist
    assert "Cxelig0 xelig0 0 20f IC=0" in netlist
    assert "Mxelig0_n xelig0 samp row0 0 NMOS" in netlist
    assert "Ch0_hdp h0_hdp 0 12f IC=0" in netlist
    assert "Ch0_hdn h0_hdn 0 12f IC=0" in netlist
    assert "Mh0_c0_hdp_pv_e vdd c0_errp h0_c0_pv_e 0 NSENSE W=7.5u" in netlist
    assert "Mh0_c0_hdp_pv_w h0_c0_pv_e c0_vwp0 h0_c0_pv_w 0 NSENSE W=7.5u" in netlist
    assert "Mh0_c0_hdp_pv_a h0_c0_pv_w act0 h0_hdp 0 NREL W=7.5u" in netlist
    assert "Mh0_c0_hdn_pv_w h0_c0_pn_e c0_vwn0 h0_c0_pn_w 0 NSENSE W=7.5u" in netlist
    assert "Mh0_c0_hdp_nv_w h0_c0_nv_e c0_vwn0 h0_c0_nv_w 0 NSENSE W=7.5u" in netlist
    assert "Mh0_c0_hdn_nv_w h0_c0_nn_e c0_vwp0 h0_c0_nn_w 0 NSENSE W=7.5u" in netlist
    assert "Mh0_live_pos_up_e vwhi_ref xelig0 h0_live_pos_up 0 NSENSE W=0.2u" in netlist
    assert "Mh0_live_pos_up_d h0_live_pos_up h0_hdp whp0 0 NSENSE W=0.2u" in netlist
    assert "Mh0_live_neg_up_d h0_live_neg_up h0_hdn whn0 0 NSENSE W=0.2u" in netlist
    assert ".meas tran hcredit_f0_1 PARAM='hdp_f0_1-hdn_f0_1'" in netlist
    assert ".meas tran whsigned_f0_final PARAM='whp_f0_final-whn_f0_final'" in netlist


def test_multiclass_block_sequence_can_size_hidden_credit_storage() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        readout_update_mode="live",
        hidden_update_mode="readout-weighted",
        hidden_credit_capacitance_f=50.0,
        hidden_credit_shunt_resistance_ohm=250000.0,
        hidden_credit_activation_model="NMOS",
        error_mode="pairwise-margin-correction-descent",
        score_timing_mode="early",
    )

    assert "Ch0_hdp h0_hdp 0 50f IC=0" in netlist
    assert "Ch0_hdn h0_hdn 0 50f IC=0" in netlist
    assert "Rh0_hdp h0_hdp 0 250000" in netlist
    assert "Rh0_hdn h0_hdn 0 250000" in netlist
    assert "Mh0_c0_hdp_pv_a h0_c0_pv_w act0 h0_hdp 0 NMOS W=8u" in netlist


def test_multiclass_block_sequence_can_directly_update_hidden_weights_from_readout_credit() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        readout_update_mode="live",
        hidden_update_mode="direct-readout-weighted",
        hidden_update_width_u=0.35,
        hidden_credit_activation_model="NMOS",
        error_mode="pairwise-margin-correction-descent",
        score_timing_mode="early",
    )

    assert "\nB" not in netlist
    assert "Cxelig0 xelig0 0 20f IC=0" in netlist
    assert "Ch0_hdp" not in netlist
    assert "Mh0_live_pos_up" not in netlist
    assert "Ch0_c0_direct_vdiff_p h0_c0_direct_vdiff_p 0 2f IC=0" in netlist
    assert "Mh0_c0_direct_vdiff_p_up0 vdd c0_vwp0 h0_c0_direct_vdiff_p_mid 0 NSENSE W=1u" in netlist
    assert "Mh0_c0_direct_vdiff_p_dn h0_c0_direct_vdiff_p c0_vwn0 0 0 NSENSE W=1u" in netlist
    assert "Ch0_c0_direct_pv_pup0 h0_c0_direct_pv_pup0 0 0.05f IC=0" in netlist
    assert "Mh0_c0_direct_pv_pup_e vwhi_ref xelig0 h0_c0_direct_pv_pup0 0 NSENSE W=0.35u" in netlist
    assert "Mh0_c0_direct_pv_pup_a h0_c0_direct_pv_pup0 act0 h0_c0_direct_pv_pup1 0 NMOS W=0.35u" in netlist
    assert "Mh0_c0_direct_pv_pup_r h0_c0_direct_pv_pup1 c0_errp h0_c0_direct_pv_pup2 0 NSENSE W=0.35u" in netlist
    assert "Mh0_c0_direct_pv_pup_w h0_c0_direct_pv_pup2 h0_c0_direct_vdiff_p whp0 0 NSENSE W=0.35u" in netlist
    assert "Mh0_c0_direct_pn_nup_w h0_c0_direct_pn_nup2 h0_c0_direct_vdiff_n whn0 0 NSENSE W=0.35u" in netlist
    assert ".meas tran hcredit_f0_1" not in netlist
    assert ".meas tran whsigned_f0_final PARAM='whp_f0_final-whn_f0_final'" in netlist


def test_multiclass_block_sequence_can_gate_direct_hidden_updates_with_feature_gate() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_two_feature_records(1),
        eval_records=_target0_two_feature_records(1),
        feature_count=2,
        readout_update_mode="live",
        hidden_update_mode="direct-readout-weighted",
        hidden_credit_activation_model="NMOS",
        error_mode="pairwise-margin-correction-descent",
        score_timing_mode="early",
        eligibility_gate_mode="competition",
    )

    assert "\nB" not in netlist
    assert "Vhxsamp hxsamp 0 PWL(" in netlist
    assert "Cxelig0 xelig0 0 20f IC=0" in netlist
    assert "Chxelig0 hxelig0 0 20f IC=0" in netlist
    assert "Mhxelig0_pass_clk hxelig0_pass_mid hxsamp xelig0 0 NSENSE W=16u L=180n" in netlist
    assert "Mhxelig0_pass_gate hxelig0 egate0 hxelig0_pass_mid 0 NSENSE W=16u L=180n" in netlist
    assert ".meas tran hxelig_f0_1 FIND V(hxelig0)" in netlist
    assert "Mh0_c0_direct_pv_pup_e vwhi_ref hxelig0 h0_c0_direct_pv_pup0 0 NSENSE" in netlist
    assert "Mh0_c0_direct_pv_pup_e vwhi_ref xelig0 h0_c0_direct_pv_pup0 0 NSENSE" not in netlist


@pytest.mark.ngspice
def test_multiclass_block_sequence_ngspice_gates_direct_hidden_update_eligibility(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = _target0_two_feature_records(1)
    netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        feature_count=2,
        readout_update_mode="live",
        hidden_update_mode="direct-readout-weighted",
        hidden_credit_activation_model="NMOS",
        hidden_direct_readout_gate_mode="raw",
        error_mode="label-rail-descent",
        score_timing_mode="early",
        eligibility_gate_mode="competition",
        eligibility_source_mode="act",
        score_sense_mode="diode-mirror",
        readout_forward_mode="diode",
        nontarget_scale=0.0,
    )

    measures = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_gated_eligibility.cir",
        netlist,
        timeout=60.0,
    )

    assert float(measures["egate_f0_1"]) > 1.0
    assert float(measures["egate_f1_1"]) < 0.05
    assert float(measures["xelig_f0_1"]) > 0.70
    assert float(measures["xelig_f1_1"]) > 0.20
    assert float(measures["relig_f0_1"]) > 0.30
    assert float(measures["relig_f1_1"]) < 0.05
    assert float(measures["hxelig_f0_1"]) > 0.30
    assert float(measures["hxelig_f1_1"]) < 0.05


def test_multiclass_block_sequence_uses_sampled_gated_readout_eligibility() -> None:
    records = _target0_two_feature_records(1)
    netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        feature_count=2,
        readout_update_mode="live",
        error_mode="label-rail-descent",
        eligibility_gate_mode="competition",
        eligibility_source_mode="act",
    )

    assert "Vrelsamp relsamp 0 PWL(" in netlist
    assert "Vrelig_ref relig_ref 0 1.2" in netlist
    assert "Crelig0 relig0 0 20f IC=0" in netlist
    assert "Crelig0_pgate relig0_pgate 0 2f IC=1.2" in netlist
    assert "Mrelig0_pgate_dis_elig relig0_pgate elig0 relig0_pgate_elig 0 NREL W=16u L=180n" in netlist
    assert "Mrelig0_pgate_dis_gate relig0_pgate_elig egate0 relig0_pgate_gate 0 NSENSE W=16u L=180n" in netlist
    assert "Mrelig0_pgate_dis_clk relig0_pgate_gate relsamp 0 0 NSENSE W=16u L=180n" in netlist
    assert "Mrelig0_restore_p relig0 relig0_pgate relig_ref relig_ref PMOS W=16u L=180n" in netlist
    assert ".meas tran relig_update_f0_1 FIND V(relig0)" in netlist
    assert ".meas tran relig_pgate_f0_1 FIND V(relig0_pgate)" in netlist
    assert "Mc0_f0_live_pos_up_e vwhi_ref relig0 c0_f0_live_pos_up 0 NSENSE W=0.5u L=180n" in netlist
    assert "Mc0_f0_live_pos_up_e vwhi_ref egate0" not in netlist


def test_multiclass_block_sequence_can_use_hybrid_readout_eligibility() -> None:
    records = _target0_two_feature_records(1)
    netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        feature_count=2,
        readout_update_mode="live",
        error_mode="label-rail-descent",
        eligibility_gate_mode="competition",
        eligibility_source_mode="act",
        readout_update_eligibility_mode="hybrid",
        readout_update_eligibility_pgate_capacitance_f=20.0,
        readout_update_eligibility_discharge_width_u=0.5,
        readout_update_eligibility_pass_width_u=8.0,
    )

    assert "Vrelpass relpass 0 PWL(" in netlist
    assert "Vrelboost relboost 0 PWL(" in netlist
    assert "Vrelsamp relsamp 0 PWL(" not in netlist
    assert "Crelig0 relig0 0 5f IC=0" in netlist
    assert "Crelig0_pgate relig0_pgate 0 20f IC=1.2" in netlist
    assert "Mrelig0_pgate_dis_elig relig0_pgate elig0 relig0_pgate_elig 0 NREL W=0.5u L=180n" in netlist
    assert "Mrelig0_pgate_dis_clk relig0_pgate_gate relboost 0 0 NSENSE W=0.5u L=180n" in netlist
    assert "Mrelig0_pass_clk relig0_pass_mid relpass elig0 0 NSENSE W=8u L=180n" in netlist
    assert "Mrelig0_pass_gate relig0 egate0 relig0_pass_mid 0 NSENSE W=8u L=180n" in netlist


@pytest.mark.ngspice
def test_multiclass_block_sequence_ngspice_hybrid_readout_eligibility_uses_small_sample_cap(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = _target0_two_feature_records(1)
    netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        feature_count=2,
        readout_update_mode="live",
        hidden_update_mode="direct-readout-weighted",
        hidden_credit_activation_model="NMOS",
        hidden_direct_readout_gate_mode="raw",
        error_mode="label-rail-descent",
        score_timing_mode="early",
        eligibility_gate_mode="competition",
        eligibility_source_mode="act",
        score_sense_mode="diode-mirror",
        readout_forward_mode="diode",
        nontarget_scale=0.0,
        readout_update_eligibility_mode="hybrid",
        readout_update_eligibility_pgate_capacitance_f=5.0,
        readout_update_eligibility_discharge_width_u=0.5,
        readout_update_eligibility_pass_width_u=8.0,
    )

    measures = run_netlist(
        ngspice_path,
        tmp_path / "hybrid_readout_eligibility_small_sample_cap.cir",
        netlist,
        timeout=60.0,
    )

    assert float(measures["elig_f0_1"]) > 0.40
    assert float(measures["relig_pgate_f0_1"]) < 0.85
    assert float(measures["relig_update_f0_1"]) > 0.80
    assert float(measures["relig_update_f1_1"]) < 0.05


@pytest.mark.ngspice
def test_multiclass_block_sequence_ngspice_bounds_gated_readout_eligibility_ref(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = _target0_two_feature_records(1)
    netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        feature_count=2,
        readout_update_mode="live",
        error_mode="label-rail-descent",
        eligibility_gate_mode="competition",
        eligibility_source_mode="act",
        readout_update_eligibility_ref=0.65,
    )

    measures = run_netlist(
        ngspice_path,
        tmp_path / "bounded_readout_eligibility_ref.cir",
        netlist,
        timeout=60.0,
    )

    assert 0.45 < float(measures["relig_f0_1"]) < 0.75
    assert float(measures["relig_f1_1"]) < 0.05


def _hybrid_readout_eligibility_transfer_netlist() -> str:
    lines = [
        "* Low-level hybrid readout eligibility transfer primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vrelig_ref relig_ref 0 1.2",
        "Vegate egate 0 1.2",
        "Vrelpass relpass 0 PULSE(0 1.2 0.2n 5p 5p 0.2n 2n)",
        "Vrelboost relboost 0 PULSE(0 1.2 0.5n 5p 5p 0.2n 2n)",
    ]
    for idx, eligibility_v in enumerate((0.15, 0.30, 0.45, 0.60)):
        relig = f"relig{idx}"
        pgate = f"{relig}_pgate"
        mid_elig = f"{relig}_pgate_elig"
        mid_gate = f"{relig}_pgate_gate"
        pass_mid = f"{relig}_pass_mid"
        lines += [
            f"Velig{idx} elig{idx} 0 {eligibility_v:.12g}",
            f"C{relig} {relig} 0 20f IC=0",
            f"R{relig} {relig} 0 1G",
            f"C{pgate} {pgate} 0 5f IC=1.2",
            f"R{pgate} {pgate} vdd 50000",
            f"R{mid_elig} {mid_elig} 0 1G",
            f"R{mid_gate} {mid_gate} 0 1G",
            f"R{pass_mid} {pass_mid} 0 1G",
            f"M{relig}_pgate_dis_elig {pgate} elig{idx} {mid_elig} 0 NREL W=0.5u L=180n",
            f"M{relig}_pgate_dis_gate {mid_elig} egate {mid_gate} 0 NSENSE W=0.5u L=180n",
            f"M{relig}_pgate_dis_clk {mid_gate} relboost 0 0 NSENSE W=0.5u L=180n",
            f"M{relig}_restore_p {relig} {pgate} relig_ref relig_ref PMOS W=16u L=180n",
            f"M{relig}_pass_clk {pass_mid} relpass elig{idx} 0 NSENSE W=8u L=180n",
            f"M{relig}_pass_gate {relig} egate {pass_mid} 0 NSENSE W=8u L=180n",
            f".meas tran {relig}_after FIND V({relig}) AT=0.9n",
        ]
    lines += [
        ".tran 2p 1.2n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


@pytest.mark.ngspice
def test_multiclass_block_sequence_ngspice_hybrid_readout_eligibility_is_monotone_and_boosted(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "hybrid_readout_eligibility_transfer.cir",
        _hybrid_readout_eligibility_transfer_netlist(),
        timeout=30.0,
    )

    low = float(measures["relig0_after"])
    mid = float(measures["relig1_after"])
    high = float(measures["relig2_after"])
    stronger = float(measures["relig3_after"])
    assert 0.10 < low < 0.20
    assert 0.25 < mid < 0.40
    assert high > 0.80
    assert stronger > high


def test_multiclass_block_sequence_can_use_raw_direct_hidden_readout_gates() -> None:
    records = _one_hot_records()
    netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        class_count=3,
        feature_count=3,
        readout_update_mode="live",
        hidden_update_mode="direct-readout-weighted",
        hidden_direct_readout_gate_mode="raw",
        error_mode="label-rail-descent",
    )

    assert "h0_c0_direct_vdiff_p" not in netlist
    assert "Mh0_c0_direct_pv_pup_w h0_c0_direct_pv_pup2 c0_vwp0 whp0 0 NSENSE" in netlist


def test_multiclass_block_sequence_can_use_label_rail_descent_for_live_gradient_flow() -> None:
    netlist = seq.generate_netlist(
        train_records=_two_class_one_hot_records(),
        eval_records=_two_class_one_hot_records(),
        class_count=2,
        feature_count=2,
        readout_update_mode="live",
        hidden_update_mode="readout-weighted",
        error_mode="label-rail-descent",
        score_timing_mode="early",
    )

    assert "\nB" not in netlist
    assert "Vscoreerr scoreerr 0 PWL(" in netlist
    assert "Vscoregaterst scoregaterst 0 PWL(" in netlist
    assert "Cc0_errp c0_errp 0 2f IC=0" in netlist
    assert "Cc0_errn c0_errn 0 2f IC=0" in netlist
    assert "Mreset_c0_errp c0_errp scoregaterst 0 0 NMOS W=4u L=180n" in netlist
    assert "Mc0_errp_label vdd c0_targetp c0_errp_label 0 NSENSE W=32u L=180n" in netlist
    assert "Mc0_errp_clk c0_errp_label scoreerr c0_errp 0 NSENSE W=32u L=180n" in netlist
    assert "Mc0_errn_label vdd c0_targetn c0_errn_label 0 NSENSE W=32u L=180n" in netlist
    assert "Mc0_f0_live_pos_up_d c0_f0_live_pos_up c0_errp c0_vwp0 0 NSENSE W=0.5u L=180n" in netlist
    assert "Mc0_f0_live_neg_up_d c0_f0_live_neg_up c0_errn c0_vwn0 0 NSENSE W=0.5u L=180n" in netlist
    assert "Mh0_c0_hdp_pv_e vdd c0_errp h0_c0_pv_e 0 NSENSE W=8u L=180n" in netlist
    assert "Mh0_c0_hdn_pv_w h0_c0_pn_e c0_vwn0 h0_c0_pn_w 0 NSENSE W=8u L=180n" in netlist
    assert ".meas tran c0_errdiff_2 PARAM='c0_errp_2-c0_errn_2'" in netlist
    assert "Vacc acc" not in netlist
    assert "Vapply apply" not in netlist


def test_multiclass_block_sequence_parameterizes_live_readout_update_width() -> None:
    netlist = seq.generate_netlist(
        train_records=_two_class_one_hot_records(),
        eval_records=_two_class_one_hot_records(),
        class_count=2,
        feature_count=2,
        readout_update_mode="live",
        error_mode="label-rail-descent",
        readout_update_width_u=0.125,
    )

    assert "Mc0_f0_live_pos_up_e vwhi_ref elig0 c0_f0_live_pos_up 0 NSENSE W=0.125u L=180n" in netlist
    assert "Mc0_f0_live_pos_up_d c0_f0_live_pos_up c0_errp c0_vwp0 0 NSENSE W=0.125u L=180n" in netlist
    assert "Mc0_f0_live_neg_up_d c0_f0_live_neg_up c0_errn c0_vwn0 0 NSENSE W=0.125u L=180n" in netlist
    assert "W=0.5u L=180n" not in "\n".join(
        line for line in netlist.splitlines() if "_live_" in line
    )


def test_multiclass_block_sequence_can_gate_nontarget_with_restored_winner() -> None:
    netlist = seq.generate_netlist(
        train_records=_target0_records(1),
        eval_records=_target0_records(1),
        error_mode="restored-winner-nontarget",
    )

    assert "Cc0_gt_c1_decision c0_gt_c1_decision 0 20f IC=1.2" in netlist
    assert "Mc1_gt_c0_decision_dis_s c1_gt_c0_decision c0_score c1_gt_c0_decision_dn 0 NSENSE" in netlist
    assert "Mc1_gt_c0_decision_keep c1_gt_c0_decision c0_gt_c1_decision vdd vdd PMOS" in netlist
    assert "Mc1_f0_gvn_gate0 c1_f0_gvn_gate0 c1_gt_c0_decision c1_f0_gvn_gate1 0 NSENSE" in netlist
    assert "Mc1_f0_gvn_gate1 c1_f0_gvn_gate1 c1_gt_c2_decision c1_f0_gvn_gate2 0 NSENSE" in netlist
    assert "Mc1_f0_gvn_score" not in netlist


def test_multiclass_block_sequence_validation() -> None:
    records = _target0_records(1)
    with pytest.raises(ValueError, match="class_count"):
        seq.generate_netlist(train_records=records, eval_records=records, class_count=1)
    with pytest.raises(ValueError, match="feature_count"):
        seq.generate_netlist(train_records=records, eval_records=records, feature_count=0)
    with pytest.raises(ValueError, match="nonempty"):
        seq.generate_netlist(train_records=[], eval_records=records)
    with pytest.raises(ValueError, match="valid class"):
        seq.generate_netlist(train_records=[{"label": 3, "inputs": {"x0": 0.85}}], eval_records=records)
    with pytest.raises(ValueError, match="inputs\\['x0'\\]"):
        seq.generate_netlist(train_records=[{"label": 0, "inputs": {}}], eval_records=records)
    with pytest.raises(ValueError, match="supply rails"):
        seq.generate_netlist(train_records=[{"label": 0, "inputs": {"x0": 1.3}}], eval_records=records)
    with pytest.raises(ValueError, match="class-count"):
        seq.main_for_test(["--class-count", "1"])
    with pytest.raises(ValueError, match="target-class"):
        seq.main_for_test(["--class-count", "3", "--target-class", "3"])
    with pytest.raises(ValueError, match="feature-count"):
        seq.main_for_test(["--feature-count", "0"])
    with pytest.raises(ValueError, match="train-samples"):
        seq.main_for_test(["--train-samples", "0"])
    with pytest.raises(ValueError, match="counted multiclass"):
        seq.main_for_test(["--scenario", "mnist", "--dataset", "mnist01fixed8_6"])
    with pytest.raises(ValueError, match="score-capacitance-f"):
        seq.main_for_test(["--score-capacitance-f", "0"])
    with pytest.raises(ValueError, match="score-load-resistance"):
        seq.main_for_test(["--score-load-resistance", "0"])
    with pytest.raises(ValueError, match="score-measure-ns"):
        seq.main_for_test(["--score-measure-ns", "16"])
    with pytest.raises(ValueError, match="nontarget-scale"):
        seq.main_for_test(["--nontarget-scale", "1.5"])
    with pytest.raises(ValueError, match="nontarget-width-scale"):
        seq.main_for_test(["--nontarget-width-scale", "-0.1"])
    with pytest.raises(ValueError, match="score-mass-sum-width"):
        seq.main_for_test(["--score-mass-sum-width", "0"])
    with pytest.raises(ValueError, match="score-mass-error-width"):
        seq.main_for_test(["--score-mass-error-width", "0"])
    with pytest.raises(ValueError, match="score-mass-pairwise-error-scale"):
        seq.main_for_test(["--score-mass-pairwise-error-scale", "0"])
    with pytest.raises(ValueError, match="initial_readout_states class index"):
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            initial_readout_states={(3, 0): (0.4, 0.4)},
        )
    with pytest.raises(ValueError, match="initial_readout_states values"):
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            initial_readout_states={(0, 0): (1.3, 0.4)},
        )
    with pytest.raises(ValueError, match="nontarget_scale"):
        seq.generate_netlist(train_records=records, eval_records=records, nontarget_scale=-0.1)
    with pytest.raises(ValueError, match="nontarget_width_scale"):
        seq.generate_netlist(train_records=records, eval_records=records, nontarget_width_scale=1.1)
    with pytest.raises(ValueError, match="error_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, error_mode="missing")
    with pytest.raises(ValueError, match="readout_update_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, readout_update_mode="missing")
    with pytest.raises(ValueError, match="live readout_update_mode"):
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            readout_update_mode="live",
            error_mode="score-gated-nontarget",
        )
    with pytest.raises(ValueError, match="readout-update-mode"):
        seq.main_for_test(["--readout-update-mode", "live", "--error-mode", "score-gated-nontarget"])
    with pytest.raises(ValueError, match="readout_update_width_u"):
        seq.generate_netlist(train_records=records, eval_records=records, readout_update_mode="live", readout_update_width_u=0)
    with pytest.raises(ValueError, match="readout-update-width"):
        seq.main_for_test(["--readout-update-width", "0"])
    with pytest.raises(ValueError, match="readout_update_eligibility_ref"):
        seq.generate_netlist(train_records=records, eval_records=records, readout_update_eligibility_ref=0)
    with pytest.raises(ValueError, match="readout_update_eligibility_capacitance_f"):
        seq.generate_netlist(train_records=records, eval_records=records, readout_update_eligibility_capacitance_f=0)
    with pytest.raises(ValueError, match="readout_update_eligibility_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, readout_update_eligibility_mode="missing")
    with pytest.raises(SystemExit):
        seq.main_for_test(["--readout-update-eligibility-mode", "missing"])
    with pytest.raises(ValueError, match="readout-update-eligibility-ref"):
        seq.main_for_test(["--readout-update-eligibility-ref", "1.3"])
    with pytest.raises(ValueError, match="readout-update-eligibility-capacitance-f"):
        seq.main_for_test(["--readout-update-eligibility-capacitance-f", "0"])
    with pytest.raises(ValueError, match="hidden_update_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, hidden_update_mode="missing")
    with pytest.raises(ValueError, match="hidden_direct_readout_gate_mode"):
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            hidden_update_mode="direct-readout-weighted",
            error_mode="pairwise-margin-correction-descent",
            hidden_direct_readout_gate_mode="missing",
        )
    with pytest.raises(ValueError, match="hidden_direct_output_stage"):
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            hidden_update_mode="direct-readout-weighted",
            error_mode="pairwise-margin-correction-descent",
            hidden_direct_output_stage="missing",
        )
    with pytest.raises(ValueError, match="hidden_credit_activation_model"):
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            hidden_update_mode="readout-weighted",
            error_mode="pairwise-margin-correction-descent",
            hidden_credit_activation_model="missing",
        )
    with pytest.raises(ValueError, match="hidden-update-mode"):
        seq.main_for_test(["--hidden-update-mode", "readout-weighted", "--error-mode", "label-descent"])
    with pytest.raises(ValueError, match="hidden-credit-width"):
        seq.main_for_test(["--hidden-credit-width", "0"])
    with pytest.raises(ValueError, match="hidden-credit-capacitance-f"):
        seq.main_for_test(["--hidden-credit-capacitance-f", "0"])
    with pytest.raises(ValueError, match="hidden-credit-shunt-resistance"):
        seq.main_for_test(["--hidden-credit-shunt-resistance", "0"])
    with pytest.raises(ValueError, match="hidden-update-width"):
        seq.main_for_test(["--hidden-update-width", "0"])
    with pytest.raises(ValueError, match="score_timing_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, score_timing_mode="missing")
    with pytest.raises(ValueError, match="score_sense_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, score_sense_mode="missing")
    with pytest.raises(ValueError, match="readout_forward_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, readout_forward_mode="missing")
    with pytest.raises(SystemExit):
        seq.main_for_test(["--readout-forward-mode", "missing"])
    with pytest.raises(ValueError, match="current-clamp"):
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            score_sense_mode="current-clamp",
            error_mode="pairwise-margin-correction-descent",
        )
    with pytest.raises(ValueError, match="class_bias_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, class_bias_mode="missing")
    with pytest.raises(ValueError, match="class-bias-input"):
        seq.main_for_test(["--class-bias-input", "1.3"])
    with pytest.raises(ValueError, match="readout-center-resistance"):
        seq.main_for_test(["--readout-center-resistance", "-1"])
    with pytest.raises(ValueError, match="readout-center-voltage"):
        seq.main_for_test(["--readout-center-voltage", "1.3"])


def _residual_score_gate_netlist(score: float, scoren: float = 0.0) -> str:
    lines = [
        "* Low-level residual score nontarget gate primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vscore c0_score 0 {score:.12g}",
        f"Vscoren c0_scoren 0 {scoren:.12g}",
        "Vscorepre scorepre 0 PULSE(0 1.2 0.8n 10p 10p 8n 10n)",
        "Vscoregaterst scoregaterst 0 PULSE(1.2 0 0.8n 10p 10p 8n 10n)",
        "Vscoreamp scoreamp 0 PULSE(0 1.2 1.0n 10p 10p 2.0n 10n)",
        "Vscoredec scoredec 0 PULSE(0 1.2 3.2n 10p 10p 2.0n 10n)",
        *seq.low_gain_ref_state_lines(prefix="c0_", reset_node="scorepre"),
        *seq.low_gain_preamp_lines(
            prefix="c0_",
            score_node="c0_score",
            scoren_node="c0_scoren",
            amp_clock_node="scoreamp",
        ),
        *seq.class_local_residual_score_gate_lines(class_idx=0),
        ".meas tran score_amp_after FIND V(c0_score_amp) AT=3.1n",
        ".meas tran scoren_amp_after FIND V(c0_scoren_amp) AT=3.1n",
        ".meas tran score_gain_diff PARAM='score_amp_after-scoren_amp_after'",
        ".meas tran score_gate_after FIND V(c0_score_gate) AT=5.5n",
        ".tran 2p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _competitive_target_boost_netlist(opponent_gate: float) -> str:
    lines = [
        "* Low-level amplified-score competitive target boost primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        "Velig elig0 0 0.85",
        "Vtargetp c0_targetp 0 PULSE(0 1.1 1n 10p 10p 2n 20n)",
        "Vtargetn c0_targetn 0 0",
        "Vacc acc 0 PULSE(0 1.2 1n 10p 10p 2n 20n)",
        "Vapply apply 0 PULSE(0 1.2 4n 10p 10p 0.1n 20n)",
        "Vapplyn applyn 0 PULSE(1.2 0 4n 10p 10p 0.1n 20n)",
        "Vown_score_amp c0_score_amp 0 0.05",
        f"Vopp_score_amp opp_score_amp 0 {opponent_gate:.12g}",
        "Cc0_gvp0 c0_gvp0 0 2f IC=0",
        "Cc0_gvn0 c0_gvn0 0 2f IC=0",
        "Cc0_rgp0 c0_rgp0 0 4f IC=1.2",
        "Cc0_rgn0 c0_rgn0 0 4f IC=1.2",
        "Rc0_gvp0 c0_gvp0 0 1G",
        "Rc0_gvn0 c0_gvn0 0 1G",
        "Rc0_rgp0 c0_rgp0 vdd 50k",
        "Rc0_rgn0 c0_rgn0 vdd 50k",
        *seq.signed_store_lines(
            positive_node=seq.class_node(0, "vwp0"),
            negative_node=seq.class_node(0, "vwn0"),
            positive_ic=0.40,
            negative_ic=0.40,
        ),
        *seq.class_local_amplified_score_competitive_gradient_lines(
            class_idx=0,
            feature_idx=0,
            activation_node="elig0",
            own_score_gate_node="c0_score_amp",
            opponent_score_gate_nodes=["opp_score_amp"],
        ),
        *seq.class_local_bounded_update_lines(class_idx=0, feature_idx=0),
        ".meas tran gvp_after FIND V(c0_gvp0) AT=3.5n",
        ".meas tran rgp_after FIND V(c0_rgp0) AT=3.5n",
        ".meas tran vwp_after FIND V(c0_vwp0) AT=5.5n",
        ".meas tran vwn_after FIND V(c0_vwn0) AT=5.5n",
        ".meas tran signed_after PARAM='vwp_after-vwn_after'",
        ".tran 2p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _common_ref_score_gate_netlist(scores: tuple[float, float, float], *, raw_score: bool = False) -> str:
    source_suffix = "score" if raw_score else "score_amp"
    lines = [
        "* Low-level shared score-common reference gate primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vscorepre scorepre 0 1.2",
        "Vscoregaterst scoregaterst 0 PULSE(1.2 0 0.4n 10p 10p 8n 20n)",
        "Vscoredec scoredec 0 PULSE(0 1.2 1.0n 10p 10p 2.0n 20n)",
        *[
            f"Vscoreamp{class_idx} c{class_idx}_{source_suffix} 0 {score:.12g}"
            for class_idx, score in enumerate(scores)
        ],
        *seq.shared_score_common_reference_lines(
            class_count=3,
            source_node_template=f"c{{class_idx}}_{source_suffix}",
        ),
        *[
            line
            for class_idx in range(3)
            for line in seq.class_local_score_common_gate_lines(
                class_idx=class_idx,
                score_input_node=f"c{class_idx}_{source_suffix}",
                analog_model="NSENSE" if raw_score else "NREL",
            )
        ],
        ".meas tran score_common_after FIND V(score_common) AT=3.4n",
        ".meas tran c0_gate_after FIND V(c0_score_common_gate) AT=3.4n",
        ".meas tran c1_gate_after FIND V(c1_score_common_gate) AT=3.4n",
        ".meas tran c2_gate_after FIND V(c2_score_common_gate) AT=3.4n",
        ".tran 2p 4n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _target_ref_score_gate_netlist(scores: tuple[float, float, float], *, target_class: int) -> str:
    lines = [
        "* Low-level label-selected target-score reference gate primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vscorepre scorepre 0 1.2",
        "Vscoregaterst scoregaterst 0 PULSE(1.2 0 0.4n 10p 10p 8n 20n)",
        "Vscoredec scoredec 0 PULSE(0 1.2 1.0n 10p 10p 2.0n 20n)",
        *[
            f"Vscoreamp{class_idx} c{class_idx}_score_amp 0 {score:.12g}"
            for class_idx, score in enumerate(scores)
        ],
        *[
            f"Vtargetp{class_idx} c{class_idx}_targetp 0 {1.1 if class_idx == target_class else 0.0:.12g}"
            for class_idx in range(3)
        ],
        *seq.shared_label_score_reference_lines(class_count=3),
        *[
            line
            for class_idx in range(3)
            for line in seq.class_local_score_common_gate_lines(
                class_idx=class_idx,
                common_node="target_score_ref",
                output_node=seq.class_node(class_idx, "score_target_gate"),
            )
        ],
        ".meas tran target_score_ref_after FIND V(target_score_ref) AT=3.4n",
        ".meas tran c0_gate_after FIND V(c0_score_target_gate) AT=3.4n",
        ".meas tran c1_gate_after FIND V(c1_score_target_gate) AT=3.4n",
        ".meas tran c2_gate_after FIND V(c2_score_target_gate) AT=3.4n",
        ".tran 2p 4n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _score_mass_descent_netlist(
    scores: tuple[float, float, float],
    *,
    target_class: int,
    common_centered: bool = False,
    contrast_centered: bool = False,
    contrast_gated_centered: bool = False,
    target_centered: bool = False,
    pairwise_hybrid: bool = False,
) -> str:
    if sum(bool(value) for value in (common_centered, contrast_centered, contrast_gated_centered, target_centered)) > 1:
        raise ValueError("score centering modes are mutually exclusive")
    if pairwise_hybrid and not common_centered:
        raise ValueError("pairwise_hybrid currently uses common_centered score mass")
    score_suffix = (
        "score_target_gate"
        if target_centered
        else "score_contrast_gate"
        if contrast_gated_centered
        else "score_contrast"
        if contrast_centered
        else "score_common_gate"
        if common_centered
        else "score_amp"
    )
    score_nodes = [f"c{class_idx}_{score_suffix}" for class_idx in range(3)]
    lines = [
        "* Low-level multiclass score-mass descent writer primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        "Vscore_contrast_ref score_contrast_ref 0 0.6",
        "Velig elig0 0 0.85",
        "Vscorepre scorepre 0 1.2",
        "Vscoregaterst scoregaterst 0 PULSE(1.2 0 0.4n 10p 10p 8n 20n)",
        "Vscoreamp scoreamp 0 PULSE(0 1.2 0.7n 10p 10p 1.0n 20n)",
        "Vscoredec scoredec 0 PULSE(0 1.2 1.0n 10p 10p 1.2n 20n)",
        "Vscoregate scoregate 0 PULSE(0 1.2 2.25n 10p 10p 0.12n 20n)",
        "Vscoremass scoremass 0 PULSE(0 1.2 2.4n 10p 10p 0.3n 20n)",
        "Vscoreerr scoreerr 0 PULSE(0 1.2 2.9n 10p 10p 0.3n 20n)",
        "Vacc acc 0 PULSE(0 1.2 3.4n 10p 10p 1.4n 20n)",
        "Vapply apply 0 PULSE(0 1.2 5.2n 10p 10p 0.2n 20n)",
        "Vapplyn applyn 0 PULSE(1.2 0 5.2n 10p 10p 0.2n 20n)",
        *[
            f"Vscoreamp{class_idx} c{class_idx}_score_amp 0 {score:.12g}"
            for class_idx, score in enumerate(scores)
        ],
        *[
            f"Vscore{class_idx} c{class_idx}_score 0 {0.01 * score:.12g}"
            for class_idx, score in enumerate(scores)
        ],
        *[
            f"Vtargetp{class_idx} c{class_idx}_targetp 0 {1.1 if class_idx == target_class else 0.0:.12g}"
            for class_idx in range(3)
        ],
        *[
            f"Vtargetn{class_idx} c{class_idx}_targetn 0 {0.0 if class_idx == target_class else 1.1:.12g}"
            for class_idx in range(3)
        ],
    ]
    if common_centered:
        lines += [
            *seq.shared_score_common_reference_lines(class_count=3),
            *[
                line
                for class_idx in range(3)
                for line in seq.class_local_score_common_gate_lines(class_idx=class_idx)
            ],
        ]
    if contrast_centered:
        lines += [
            *seq.shared_score_common_reference_lines(class_count=3),
            *[
                line
                for class_idx in range(3)
                for line in seq.class_local_score_contrast_lines(class_idx=class_idx)
            ],
        ]
    if contrast_gated_centered:
        lines += [
            *seq.shared_score_common_reference_lines(class_count=3),
            *[
                line
                for class_idx in range(3)
                for line in (
                    *seq.class_local_score_contrast_lines(class_idx=class_idx),
                    *seq.class_local_score_common_gate_lines(
                        class_idx=class_idx,
                        common_node="score_contrast_ref",
                        score_input_node=seq.class_node(class_idx, "score_contrast"),
                        output_node=seq.class_node(class_idx, "score_contrast_gate"),
                        compare_clock="scoregate",
                        pullup_width_u=192.0,
                        pulldown_width_u=24.0,
                    ),
                )
            ],
        ]
    if target_centered:
        lines += [
            *seq.shared_label_score_reference_lines(class_count=3),
            *[
                line
                for class_idx in range(3)
                for line in seq.class_local_score_common_gate_lines(
                    class_idx=class_idx,
                    common_node="target_score_ref",
                    output_node=seq.class_node(class_idx, "score_target_gate"),
                    pullup_width_u=192.0,
                    pulldown_width_u=6.0,
                )
            ],
        ]
    lines += seq.shared_score_mass_error_lines(
        class_count=3,
        score_input_template=f"c{{class_idx}}_{score_suffix}",
        mass_clock_node="scoremass" if contrast_gated_centered else "scoredec",
        sum_width_u=128.0 if (common_centered or contrast_centered or contrast_gated_centered or target_centered) else 32.0,
        error_width_u=128.0 if (common_centered or contrast_centered or contrast_gated_centered or target_centered) else 32.0,
        mass_capacitance_f=0.5 if (common_centered or contrast_centered or contrast_gated_centered or target_centered) else 8.0,
        error_capacitance_f=0.5 if (common_centered or contrast_centered or contrast_gated_centered or target_centered) else 8.0,
    )
    if pairwise_hybrid:
        lines += [
            line
            for class_idx in range(3)
            for opponent_idx in range(class_idx + 1, 3)
            for line in seq.pairwise_low_gain_winner_lines(class_a=class_idx, class_b=opponent_idx)
        ]
        lines += seq.pairwise_score_competition_error_lines(
            class_count=3,
            error_width_u=2.0,
            error_capacitance_f=0.5,
            create_error_nodes=False,
        )
    for class_idx in range(3):
        lines += [
            f"Cc{class_idx}_gvp0 c{class_idx}_gvp0 0 2f IC=0",
            f"Cc{class_idx}_gvn0 c{class_idx}_gvn0 0 2f IC=0",
            f"Cc{class_idx}_rgp0 c{class_idx}_rgp0 0 4f IC=1.2",
            f"Cc{class_idx}_rgn0 c{class_idx}_rgn0 0 4f IC=1.2",
            f"Rc{class_idx}_gvp0 c{class_idx}_gvp0 0 1G",
            f"Rc{class_idx}_gvn0 c{class_idx}_gvn0 0 1G",
            f"Rc{class_idx}_rgp0 c{class_idx}_rgp0 vdd 50k",
            f"Rc{class_idx}_rgn0 c{class_idx}_rgn0 vdd 50k",
            *seq.signed_store_lines(
                positive_node=seq.class_node(class_idx, "vwp0"),
                negative_node=seq.class_node(class_idx, "vwn0"),
                positive_ic=0.40,
                negative_ic=0.40,
            ),
            *seq.class_local_error_rail_gradient_lines(
                class_idx=class_idx,
                feature_idx=0,
                activation_node="elig0",
                positive_error_node=seq.class_node(class_idx, "errp"),
                negative_error_node=seq.class_node(class_idx, "errn"),
            ),
            *seq.class_local_bounded_update_lines(class_idx=class_idx, feature_idx=0),
            *(
                [
                    f".meas tran c{class_idx}_score_gate_after FIND V({score_nodes[class_idx]}) AT=2.35n",
                ]
                if common_centered or contrast_centered or contrast_gated_centered or target_centered
                else []
            ),
            *(
                [
                    f".meas tran c{class_idx}_gt_target_after FIND V({seq.pairwise_decision_node(class_idx, target_class)}) AT=2.35n",
                ]
                if pairwise_hybrid and class_idx != target_class
                else []
            ),
            f".meas tran c{class_idx}_errp_after FIND V({seq.class_node(class_idx, 'errp')}) AT=3.25n",
            f".meas tran c{class_idx}_errn_after FIND V({seq.class_node(class_idx, 'errn')}) AT=3.25n",
            f".meas tran c{class_idx}_errdiff PARAM='c{class_idx}_errp_after-c{class_idx}_errn_after'",
            f".meas tran c{class_idx}_gvp_after FIND V({seq.class_node(class_idx, 'gvp0')}) AT=5.0n",
            f".meas tran c{class_idx}_gvn_after FIND V({seq.class_node(class_idx, 'gvn0')}) AT=5.0n",
            f".meas tran c{class_idx}_vwp_after FIND V({seq.class_node(class_idx, 'vwp0')}) AT=6.0n",
            f".meas tran c{class_idx}_vwn_after FIND V({seq.class_node(class_idx, 'vwn0')}) AT=6.0n",
            f".meas tran c{class_idx}_signed_after PARAM='c{class_idx}_vwp_after-c{class_idx}_vwn_after'",
        ]
    mass_measure_time_ns = 2.75 if contrast_gated_centered else 2.35
    lines += [
        f".meas tran score_nontarget_mass_after FIND V(score_nontarget_mass) AT={mass_measure_time_ns:.2f}n",
        ".tran 2p 7n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _centered_error_rail_netlist(
    raw_positive: tuple[float, float, float],
    raw_negative: tuple[float, float, float],
    *,
    gain_restored: bool = False,
) -> str:
    centered_positive_suffix = "errp_ctr" if gain_restored else "errp"
    centered_negative_suffix = "errn_ctr" if gain_restored else "errn"
    lines = [
        "* Low-level class-centered error rail primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vscoregaterst scoregaterst 0 0",
        *[
            f"Vrawp{class_idx} {seq.class_node(class_idx, 'errp_raw')} 0 {value:.12g}"
            for class_idx, value in enumerate(raw_positive)
        ],
        *[
            f"Vrawn{class_idx} {seq.class_node(class_idx, 'errn_raw')} 0 {value:.12g}"
            for class_idx, value in enumerate(raw_negative)
        ],
        *seq.class_centered_error_rail_lines(
            class_count=3,
            copy_width_u=64.0,
            common_width_u=128.0,
            capacitance_f=4.0,
            common_capacitance_f=4.0,
            positive_suffix=centered_positive_suffix,
            negative_suffix=centered_negative_suffix,
        ),
        *(
            seq.class_error_rail_gain_restore_lines(
                class_count=3,
                restore_width_u=128.0,
                capacitance_f=1.0,
            )
            if gain_restored
            else []
        ),
        ".meas tran common_p_after FIND V(class_errp_common) AT=2.5n",
        ".meas tran common_n_after FIND V(class_errn_common) AT=2.5n",
    ]
    for class_idx in range(3):
        lines += [
            *(
                [
                    f".meas tran c{class_idx}_errp_ctr_after FIND V({seq.class_node(class_idx, 'errp_ctr')}) AT=2.5n",
                    f".meas tran c{class_idx}_errn_ctr_after FIND V({seq.class_node(class_idx, 'errn_ctr')}) AT=2.5n",
                    f".meas tran c{class_idx}_errdiff_ctr PARAM='c{class_idx}_errp_ctr_after-c{class_idx}_errn_ctr_after'",
                ]
                if gain_restored
                else []
            ),
            f".meas tran c{class_idx}_errp_after FIND V({seq.class_node(class_idx, 'errp')}) AT=2.5n",
            f".meas tran c{class_idx}_errn_after FIND V({seq.class_node(class_idx, 'errn')}) AT=2.5n",
            f".meas tran c{class_idx}_errdiff PARAM='c{class_idx}_errp_after-c{class_idx}_errn_after'",
        ]
    lines += [
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _pairwise_winner_error_rail_netlist() -> str:
    decision_values = {
        seq.pairwise_decision_node(1, 0): 1.2,
        seq.pairwise_decision_node(0, 1): 0.0,
        seq.pairwise_decision_node(2, 0): 1.2,
        seq.pairwise_decision_node(0, 2): 0.0,
        seq.pairwise_decision_node(1, 2): 1.2,
        seq.pairwise_decision_node(2, 1): 0.0,
    }
    lines = [
        "* Low-level strongest-impostor pairwise error rail primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vscoreerr scoreerr 0 1.2",
        "Vscoregaterst scoregaterst 0 0",
        "Vtargetp0 c0_targetp 0 1.2",
        "Vtargetn0 c0_targetn 0 0",
        "Vtargetp1 c1_targetp 0 0",
        "Vtargetn1 c1_targetn 0 1.2",
        "Vtargetp2 c2_targetp 0 0",
        "Vtargetn2 c2_targetn 0 1.2",
        *[f"V{name} {name} 0 {value:.12g}" for name, value in decision_values.items()],
        *seq.pairwise_score_competition_error_lines(
            class_count=3,
            error_width_u=32.0,
            error_capacitance_f=2.0,
            strongest_opponent_only=True,
        ),
    ]
    for class_idx in range(3):
        lines += [
            f".meas tran c{class_idx}_errp_after FIND V({seq.class_node(class_idx, 'errp')}) AT=1.0n",
            f".meas tran c{class_idx}_errn_after FIND V({seq.class_node(class_idx, 'errn')}) AT=1.0n",
            f".meas tran c{class_idx}_errdiff PARAM='c{class_idx}_errp_after-c{class_idx}_errn_after'",
        ]
    lines += [
        ".tran 2p 1.5n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _restored_score_binary_descent_netlist(
    *,
    targetp: float,
    targetn: float,
    decision: float,
    decisionn: float,
) -> str:
    lines = [
        "* Low-level restored-score binary-descent gradient primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        "Velig elig0 0 0.85",
        f"Vtargetp c0_targetp 0 PULSE(0 {targetp:.12g} 1n 10p 10p 2n 20n)",
        f"Vtargetn c0_targetn 0 PULSE(0 {targetn:.12g} 1n 10p 10p 2n 20n)",
        f"Vdecision c0_decision 0 {decision:.12g}",
        f"Vdecisionn c0_decisionn 0 {decisionn:.12g}",
        "Vacc acc 0 PULSE(0 1.2 1n 10p 10p 2n 20n)",
        "Vapply apply 0 PULSE(0 1.2 4n 10p 10p 0.1n 20n)",
        "Vapplyn applyn 0 PULSE(1.2 0 4n 10p 10p 0.1n 20n)",
        "Cc0_gvp0 c0_gvp0 0 2f IC=0",
        "Cc0_gvn0 c0_gvn0 0 2f IC=0",
        "Cc0_rgp0 c0_rgp0 0 4f IC=1.2",
        "Cc0_rgn0 c0_rgn0 0 4f IC=1.2",
        "Rc0_gvp0 c0_gvp0 0 1G",
        "Rc0_gvn0 c0_gvn0 0 1G",
        "Rc0_rgp0 c0_rgp0 vdd 50k",
        "Rc0_rgn0 c0_rgn0 vdd 50k",
        *seq.signed_store_lines(
            positive_node=seq.class_node(0, "vwp0"),
            negative_node=seq.class_node(0, "vwn0"),
            positive_ic=0.40,
            negative_ic=0.40,
        ),
        *seq.class_local_restored_score_binary_descent_gradient_lines(
            class_idx=0,
            feature_idx=0,
            activation_node="elig0",
            positive_gate_node="c0_decision",
            negative_gate_node="c0_decisionn",
        ),
        *seq.class_local_bounded_update_lines(class_idx=0, feature_idx=0),
        ".meas tran gvp_after FIND V(c0_gvp0) AT=3.5n",
        ".meas tran gvn_after FIND V(c0_gvn0) AT=3.5n",
        ".meas tran vwp_after FIND V(c0_vwp0) AT=5.5n",
        ".meas tran vwn_after FIND V(c0_vwn0) AT=5.5n",
        ".meas tran signed_after PARAM='vwp_after-vwn_after'",
        ".tran 2p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _pairwise_binary_descent_netlist(
    *,
    targetp: float,
    targetn: float,
    losing_gate: float,
    winning_gate: float,
) -> str:
    lines = [
        "* Low-level pairwise binary-descent gradient primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        "Velig elig0 0 0.85",
        f"Vtargetp c0_targetp 0 PULSE(0 {targetp:.12g} 1n 10p 10p 2n 20n)",
        f"Vtargetn c0_targetn 0 PULSE(0 {targetn:.12g} 1n 10p 10p 2n 20n)",
        f"Vlosing losing_gate 0 {losing_gate:.12g}",
        f"Vwinning winning_gate 0 {winning_gate:.12g}",
        "Vacc acc 0 PULSE(0 1.2 1n 10p 10p 2n 20n)",
        "Vapply apply 0 PULSE(0 1.2 4n 10p 10p 0.1n 20n)",
        "Vapplyn applyn 0 PULSE(1.2 0 4n 10p 10p 0.1n 20n)",
        "Cc0_gvp0 c0_gvp0 0 2f IC=0",
        "Cc0_gvn0 c0_gvn0 0 2f IC=0",
        "Cc0_rgp0 c0_rgp0 0 4f IC=1.2",
        "Cc0_rgn0 c0_rgn0 0 4f IC=1.2",
        "Rc0_gvp0 c0_gvp0 0 1G",
        "Rc0_gvn0 c0_gvn0 0 1G",
        "Rc0_rgp0 c0_rgp0 vdd 50k",
        "Rc0_rgn0 c0_rgn0 vdd 50k",
        *seq.signed_store_lines(
            positive_node=seq.class_node(0, "vwp0"),
            negative_node=seq.class_node(0, "vwn0"),
            positive_ic=0.40,
            negative_ic=0.40,
        ),
        *seq.class_local_pairwise_binary_descent_gradient_lines(
            class_idx=0,
            feature_idx=0,
            activation_node="elig0",
            losing_gate_nodes=["losing_gate"],
            winning_gate_nodes=["winning_gate"],
        ),
        *seq.class_local_bounded_update_lines(class_idx=0, feature_idx=0),
        ".meas tran gvp_after FIND V(c0_gvp0) AT=3.5n",
        ".meas tran gvn_after FIND V(c0_gvn0) AT=3.5n",
        ".meas tran rgp_after FIND V(c0_rgp0) AT=3.5n",
        ".meas tran rgn_after FIND V(c0_rgn0) AT=3.5n",
        ".meas tran vwp_after FIND V(c0_vwp0) AT=5.5n",
        ".meas tran vwn_after FIND V(c0_vwn0) AT=5.5n",
        ".meas tran signed_after PARAM='vwp_after-vwn_after'",
        ".tran 2p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _amplified_pairwise_blend_netlist(
    *,
    targetp: float,
    targetn: float,
    own_score_gate: float,
    losing_gate: float,
    winning_gate: float,
) -> str:
    lines = [
        "* Low-level amplified-score plus pairwise correction gradient primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        "Velig elig0 0 0.85",
        f"Vtargetp c0_targetp 0 PULSE(0 {targetp:.12g} 1n 10p 10p 2n 20n)",
        f"Vtargetn c0_targetn 0 PULSE(0 {targetn:.12g} 1n 10p 10p 2n 20n)",
        f"Vscore c0_score_amp 0 {own_score_gate:.12g}",
        f"Vlosing losing_gate 0 {losing_gate:.12g}",
        f"Vwinning winning_gate 0 {winning_gate:.12g}",
        "Vacc acc 0 PULSE(0 1.2 1n 10p 10p 2n 20n)",
        "Vapply apply 0 PULSE(0 1.2 4n 10p 10p 0.1n 20n)",
        "Vapplyn applyn 0 PULSE(1.2 0 4n 10p 10p 0.1n 20n)",
        "Cc0_gvp0 c0_gvp0 0 2f IC=0",
        "Cc0_gvn0 c0_gvn0 0 2f IC=0",
        "Cc0_rgp0 c0_rgp0 0 4f IC=1.2",
        "Cc0_rgn0 c0_rgn0 0 4f IC=1.2",
        "Rc0_gvp0 c0_gvp0 0 1G",
        "Rc0_gvn0 c0_gvn0 0 1G",
        "Rc0_rgp0 c0_rgp0 vdd 50k",
        "Rc0_rgn0 c0_rgn0 vdd 50k",
        *seq.signed_store_lines(
            positive_node=seq.class_node(0, "vwp0"),
            negative_node=seq.class_node(0, "vwn0"),
            positive_ic=0.40,
            negative_ic=0.40,
        ),
        *seq.class_local_residual_score_nontarget_gradient_lines(
            class_idx=0,
            feature_idx=0,
            activation_node="elig0",
            score_gate_node="c0_score_amp",
        ),
        *seq.class_local_pairwise_binary_descent_correction_lines(
            class_idx=0,
            feature_idx=0,
            activation_node="elig0",
            losing_gate_nodes=["losing_gate"],
            winning_gate_nodes=["winning_gate"],
        ),
        *seq.class_local_bounded_update_lines(class_idx=0, feature_idx=0),
        ".meas tran gvp_after FIND V(c0_gvp0) AT=3.5n",
        ".meas tran gvn_after FIND V(c0_gvn0) AT=3.5n",
        ".meas tran rgp_after FIND V(c0_rgp0) AT=3.5n",
        ".meas tran rgn_after FIND V(c0_rgn0) AT=3.5n",
        ".meas tran vwp_after FIND V(c0_vwp0) AT=5.5n",
        ".meas tran vwn_after FIND V(c0_vwn0) AT=5.5n",
        ".meas tran signed_after PARAM='vwp_after-vwn_after'",
        ".tran 2p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _amplified_binary_blend_netlist(
    *,
    targetp: float,
    targetn: float,
    own_score_gate: float,
    decision: float,
    decisionn: float,
) -> str:
    lines = [
        "* Low-level amplified-score plus binary-decision correction gradient primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vvwhi_ref vwhi_ref 0 0.42",
        "Vvwlo_ref vwlo_ref 0 0.28",
        "Velig elig0 0 0.85",
        f"Vtargetp c0_targetp 0 PULSE(0 {targetp:.12g} 1n 10p 10p 2n 20n)",
        f"Vtargetn c0_targetn 0 PULSE(0 {targetn:.12g} 1n 10p 10p 2n 20n)",
        f"Vscore c0_score_amp 0 {own_score_gate:.12g}",
        f"Vdecision c0_decision 0 {decision:.12g}",
        f"Vdecisionn c0_decisionn 0 {decisionn:.12g}",
        "Vacc acc 0 PULSE(0 1.2 1n 10p 10p 2n 20n)",
        "Vapply apply 0 PULSE(0 1.2 4n 10p 10p 0.1n 20n)",
        "Vapplyn applyn 0 PULSE(1.2 0 4n 10p 10p 0.1n 20n)",
        "Cc0_gvp0 c0_gvp0 0 2f IC=0",
        "Cc0_gvn0 c0_gvn0 0 2f IC=0",
        "Cc0_rgp0 c0_rgp0 0 4f IC=1.2",
        "Cc0_rgn0 c0_rgn0 0 4f IC=1.2",
        "Rc0_gvp0 c0_gvp0 0 1G",
        "Rc0_gvn0 c0_gvn0 0 1G",
        "Rc0_rgp0 c0_rgp0 vdd 50k",
        "Rc0_rgn0 c0_rgn0 vdd 50k",
        *seq.signed_store_lines(
            positive_node=seq.class_node(0, "vwp0"),
            negative_node=seq.class_node(0, "vwn0"),
            positive_ic=0.40,
            negative_ic=0.40,
        ),
        *seq.class_local_residual_score_nontarget_gradient_lines(
            class_idx=0,
            feature_idx=0,
            activation_node="elig0",
            score_gate_node="c0_score_amp",
        ),
        *seq.class_local_restored_score_binary_correction_lines(
            class_idx=0,
            feature_idx=0,
            activation_node="elig0",
            positive_gate_node="c0_decision",
            negative_gate_node="c0_decisionn",
        ),
        *seq.class_local_bounded_update_lines(class_idx=0, feature_idx=0),
        ".meas tran gvp_after FIND V(c0_gvp0) AT=3.5n",
        ".meas tran gvn_after FIND V(c0_gvn0) AT=3.5n",
        ".meas tran rgp_after FIND V(c0_rgp0) AT=3.5n",
        ".meas tran rgn_after FIND V(c0_rgn0) AT=3.5n",
        ".meas tran vwp_after FIND V(c0_vwp0) AT=5.5n",
        ".meas tran vwn_after FIND V(c0_vwn0) AT=5.5n",
        ".meas tran signed_after PARAM='vwp_after-vwn_after'",
        ".tran 2p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _readout_center_leak_netlist(resistance: float = 5e6) -> str:
    lines = [
        "* Low-level passive readout center leak primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vreadout_center readout_center 0 0.40",
        *seq.signed_store_lines(
            positive_node=seq.class_node(0, "vwp0"),
            negative_node=seq.class_node(0, "vwn0"),
            positive_ic=0.52,
            negative_ic=0.25,
        ),
        f"R{seq.class_node(0, 'vwp0')}_center {seq.class_node(0, 'vwp0')} readout_center {resistance:.12g}",
        f"R{seq.class_node(0, 'vwn0')}_center {seq.class_node(0, 'vwn0')} readout_center {resistance:.12g}",
        ".meas tran vwp_after FIND V(c0_vwp0) AT=20n",
        ".meas tran vwn_after FIND V(c0_vwn0) AT=20n",
        ".meas tran signed_after PARAM='vwp_after-vwn_after'",
        ".tran 5p 22n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_readout_weighted_writer_netlist(*, vwp: float, vwn: float) -> str:
    lines = [
        "* Low-level readout-weighted hidden-credit to live hidden-writer primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vwhi vwhi_ref 0 1.05",
        "Vwlo vwlo_ref 0 0.15",
        "Vrst rst 0 PULSE(1.2 0 0.1n 10p 10p 9n 20n)",
        "Verrp c0_errp 0 PULSE(0 1.2 1n 10p 10p 4n 20n)",
        "Verrn c0_errn 0 0",
        "Vact act0 0 1.2",
        "Vxelig xelig0 0 1.2",
        f"Cc0_vwp0 c0_vwp0 0 20f IC={vwp:.12g}",
        f"Cc0_vwn0 c0_vwn0 0 20f IC={vwn:.12g}",
        "Cwhp0 whp0 0 20f IC=0.45",
        "Cwhn0 whn0 0 20f IC=0.40",
        "Rc0_vwp0 c0_vwp0 0 1e15",
        "Rc0_vwn0 c0_vwn0 0 1e15",
        "Rwhp0 whp0 0 1e15",
        "Rwhn0 whn0 0 1e15",
        *seq.hidden_readout_weighted_credit_lines(
            class_count=1,
            feature_idx=0,
            error_positive_nodes=["c0_errp"],
            error_negative_nodes=["c0_errn"],
            width_u=8.0,
        ),
        *seq.hidden_live_weight_update_lines(
            feature_idx=0,
            eligibility_node="xelig0",
            positive_credit_node="h0_hdp",
            negative_credit_node="h0_hdn",
            width_u=0.25,
        ),
        ".meas tran hdp FIND V(h0_hdp) AT=5n",
        ".meas tran hdn FIND V(h0_hdn) AT=5n",
        ".meas tran whp_after FIND V(whp0) AT=8n",
        ".meas tran whn_after FIND V(whn0) AT=8n",
        ".meas tran signed_after PARAM='whp_after-whn_after'",
        ".tran 2p 10n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_direct_readout_weighted_writer_netlist(
    *,
    vwp: float,
    vwn: float,
    whp: float = 0.45,
    whn: float = 0.40,
    readout_high_ref: float = 1.05,
    readout_low_ref: float = 0.15,
    hidden_high_ref: float = 1.05,
    hidden_low_ref: float = 0.15,
    errp: float | str = "PULSE(0 1.2 1n 10p 10p 4n 20n)",
    errn: float | str = 0.0,
    width_u: float = 8.0,
    readout_gate_mode: str = "differential-excess",
    output_stage: str = "nmos-pass",
) -> str:
    lines = [
        "* Low-level direct readout-weighted hidden writer primitive.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vwhi vwhi_ref 0 {readout_high_ref:.12g}",
        f"Vwlo vwlo_ref 0 {readout_low_ref:.12g}",
        f"Vhwhi hidden_whi_ref 0 {hidden_high_ref:.12g}",
        f"Vhwlo hidden_wlo_ref 0 {hidden_low_ref:.12g}",
        "Vrst rst 0 PULSE(1.2 0 0.1n 10p 10p 9n 20n)",
        f"Verrp c0_errp 0 {errp if isinstance(errp, str) else f'{errp:.12g}'}",
        f"Verrn c0_errn 0 {errn if isinstance(errn, str) else f'{errn:.12g}'}",
        "Vact act0 0 1.2",
        "Vxelig xelig0 0 1.2",
        f"Cc0_vwp0 c0_vwp0 0 20f IC={vwp:.12g}",
        f"Cc0_vwn0 c0_vwn0 0 20f IC={vwn:.12g}",
        f"Cwhp0 whp0 0 20f IC={whp:.12g}",
        f"Cwhn0 whn0 0 20f IC={whn:.12g}",
        "Rc0_vwp0 c0_vwp0 0 1e15",
        "Rc0_vwn0 c0_vwn0 0 1e15",
        "Rwhp0 whp0 0 1e15",
        "Rwhn0 whn0 0 1e15",
        *seq.hidden_direct_readout_weighted_update_lines(
            class_count=1,
            feature_idx=0,
            error_positive_nodes=["c0_errp"],
            error_negative_nodes=["c0_errn"],
            eligibility_node="xelig0",
            width_u=width_u,
            readout_gate_mode=readout_gate_mode,
            output_stage=output_stage,
            high_ref_node=(
                "hidden_whi_ref"
                if output_stage in ("pmos-bounded", "pmos-complementary")
                else "vwhi_ref"
            ),
            low_ref_node="hidden_wlo_ref",
        ),
        *(
            [
                ".meas tran vdiff_p FIND V(h0_c0_direct_vdiff_p) AT=5n",
                ".meas tran vdiff_n FIND V(h0_c0_direct_vdiff_n) AT=5n",
            ]
            if readout_gate_mode in ("differential-excess", "restored-excess")
            else []
        ),
        ".meas tran whp_after FIND V(whp0) AT=8n",
        ".meas tran whn_after FIND V(whn0) AT=8n",
        ".meas tran signed_after PARAM='whp_after-whn_after'",
        ".tran 2p 10n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_direct_readout_weighted_two_update_netlist(
    *,
    first_errp: float = 1.2,
    first_errn: float = 0.0,
    second_errp: float = 0.0,
    second_errn: float = 1.2,
    vwp: float = 0.40,
    vwn: float = 0.28,
    whp: float = 0.45,
    whn: float = 0.40,
    width_u: float = 0.125,
    complement_width_scale: float = 0.0625,
    output_stage: str = "pmos-complementary",
) -> str:
    def pulse(level: float, start_ns: float) -> str:
        if level <= 0.0:
            return "0"
        return f"PULSE(0 {level:.12g} {start_ns:.12g}n 10p 10p 4n 20n)"

    lines = [
        "* Two live direct hidden updates without Python state intervention.",
        ".param VDD=1.2",
        seq.mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vwhi vwhi_ref 0 0.42",
        "Vwlo vwlo_ref 0 0.28",
        "Vhwhi hidden_whi_ref 0 1.05",
        "Vhwlo hidden_wlo_ref 0 0.15",
        "Vrst rst 0 PWL(0 1.2 0.1n 1.2 0.12n 0 9n 0 10n 1.2 10.1n 1.2 10.12n 0 20n 0)",
        f"Verrp0 c0_errp 0 {pulse(first_errp, 1.0)}",
        f"Verrn0 c0_errn 0 {pulse(first_errn, 1.0)}",
        f"Verrp1 c1_errp 0 {pulse(second_errp, 11.0)}",
        f"Verrn1 c1_errn 0 {pulse(second_errn, 11.0)}",
        "Vact act0 0 1.2",
        "Vxelig xelig0 0 1.2",
        f"Cc0_vwp0 c0_vwp0 0 20f IC={vwp:.12g}",
        f"Cc0_vwn0 c0_vwn0 0 20f IC={vwn:.12g}",
        f"Cc1_vwp0 c1_vwp0 0 20f IC={vwp:.12g}",
        f"Cc1_vwn0 c1_vwn0 0 20f IC={vwn:.12g}",
        f"Cwhp0 whp0 0 20f IC={whp:.12g}",
        f"Cwhn0 whn0 0 20f IC={whn:.12g}",
        "Rc0_vwp0 c0_vwp0 0 1e15",
        "Rc0_vwn0 c0_vwn0 0 1e15",
        "Rc1_vwp0 c1_vwp0 0 1e15",
        "Rc1_vwn0 c1_vwn0 0 1e15",
        "Rwhp0 whp0 0 1e15",
        "Rwhn0 whn0 0 1e15",
        *seq.hidden_direct_readout_weighted_update_lines(
            class_count=2,
            feature_idx=0,
            error_positive_nodes=["c0_errp", "c1_errp"],
            error_negative_nodes=["c0_errn", "c1_errn"],
            eligibility_node="xelig0",
            width_u=width_u,
            readout_gate_mode="restored-excess",
            output_stage=output_stage,
            high_ref_node="hidden_whi_ref",
            low_ref_node="hidden_wlo_ref",
            complement_width_scale=complement_width_scale,
        ),
        ".meas tran whp_after_first FIND V(whp0) AT=8n",
        ".meas tran whn_after_first FIND V(whn0) AT=8n",
        ".meas tran signed_after_first PARAM='whp_after_first-whn_after_first'",
        ".meas tran whp_after FIND V(whp0) AT=18n",
        ".meas tran whn_after FIND V(whn0) AT=18n",
        ".meas tran signed_after PARAM='whp_after-whn_after'",
        ".tran 2p 20n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def test_multiclass_block_sequence_ngspice_residual_score_gate_is_monotonic_low_floor(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    neutral = run_netlist(
        ngspice_path,
        tmp_path / "residual_score_gate_neutral.cir",
        _residual_score_gate_netlist(0.0),
        timeout=20.0,
    )
    medium = run_netlist(
        ngspice_path,
        tmp_path / "residual_score_gate_medium.cir",
        _residual_score_gate_netlist(0.02),
        timeout=20.0,
    )
    high = run_netlist(
        ngspice_path,
        tmp_path / "residual_score_gate_high.cir",
        _residual_score_gate_netlist(0.10),
        timeout=20.0,
    )

    assert abs(float(neutral["score_gain_diff"])) < 1e-6
    assert float(medium["score_gain_diff"]) > 10e-3
    assert float(high["score_gain_diff"]) > float(medium["score_gain_diff"]) + 20e-3
    assert 0.015 < float(neutral["score_gate_after"]) < 0.05
    assert float(medium["score_gate_after"]) > float(neutral["score_gate_after"]) + 1e-3
    assert float(high["score_gate_after"]) > float(medium["score_gate_after"]) + 3e-3
    assert float(high["score_gate_after"]) < 0.08


def test_multiclass_block_sequence_ngspice_competitive_target_boost_strengthens_positive_write(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    low = run_netlist(
        ngspice_path,
        tmp_path / "competitive_target_boost_low.cir",
        _competitive_target_boost_netlist(0.0),
        timeout=20.0,
    )
    high = run_netlist(
        ngspice_path,
        tmp_path / "competitive_target_boost_high.cir",
        _competitive_target_boost_netlist(1.0),
        timeout=20.0,
    )

    assert float(high["gvp_after"]) > float(low["gvp_after"]) + 5e-3
    assert float(high["rgp_after"]) < float(low["rgp_after"]) - 10e-6
    assert float(high["signed_after"]) > float(low["signed_after"]) + 1e-3


def test_multiclass_block_sequence_ngspice_hidden_credit_drives_live_hidden_writer(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_writer_positive.cir",
        _hidden_readout_weighted_writer_netlist(vwp=1.0, vwn=0.05),
        timeout=20.0,
    )
    negative = run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_writer_negative.cir",
        _hidden_readout_weighted_writer_netlist(vwp=0.05, vwn=1.0),
        timeout=20.0,
    )

    assert float(positive["hdp"]) > float(positive["hdn"]) + 0.5
    assert float(positive["signed_after"]) > 0.5
    assert float(positive["whp_after"]) > 0.80
    assert float(positive["whn_after"]) < 0.20
    assert float(negative["hdn"]) > float(negative["hdp"]) + 0.5
    assert float(negative["signed_after"]) < -0.5
    assert float(negative["whn_after"]) > 0.80
    assert float(negative["whp_after"]) < 0.20


def test_multiclass_block_sequence_ngspice_direct_hidden_writer_preserves_credit_sign(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_positive.cir",
        _hidden_direct_readout_weighted_writer_netlist(vwp=1.0, vwn=0.05),
        timeout=20.0,
    )
    negative = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_negative.cir",
        _hidden_direct_readout_weighted_writer_netlist(vwp=0.05, vwn=1.0),
        timeout=20.0,
    )
    neutral = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_neutral.cir",
        _hidden_direct_readout_weighted_writer_netlist(vwp=0.4, vwn=0.4),
        timeout=20.0,
    )

    assert float(positive["signed_after"]) > 0.20
    assert float(positive["vdiff_p"]) > float(positive["vdiff_n"]) + 0.5
    assert float(positive["whp_after"]) > 0.50
    assert float(positive["whn_after"]) == pytest.approx(0.40, abs=5e-3)
    assert float(negative["signed_after"]) < -0.20
    assert float(negative["vdiff_n"]) > float(negative["vdiff_p"]) + 0.5
    assert float(negative["whn_after"]) > 0.50
    assert float(negative["whp_after"]) == pytest.approx(0.45, abs=5e-3)
    assert abs(float(neutral["vdiff_p"]) - float(neutral["vdiff_n"])) < 20e-3
    assert max(float(neutral["vdiff_p"]), float(neutral["vdiff_n"])) < 0.20
    assert float(neutral["signed_after"]) == pytest.approx(0.05, abs=2e-3)


def test_multiclass_block_sequence_ngspice_direct_hidden_writer_preserves_existing_state_for_neutral_readout(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    neutral = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_neutral_existing_state.cir",
        _hidden_direct_readout_weighted_writer_netlist(vwp=0.4, vwn=0.4, whp=1.0, whn=0.2),
        timeout=20.0,
    )

    assert abs(float(neutral["vdiff_p"]) - float(neutral["vdiff_n"])) < 20e-3
    assert max(float(neutral["vdiff_p"]), float(neutral["vdiff_n"])) < 0.20
    assert float(neutral["signed_after"]) > 0.75


def test_multiclass_block_sequence_ngspice_direct_hidden_writer_pmos_stage_moves_moderate_credit(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    nmos_pass = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_moderate_nmos_pass.cir",
        _hidden_direct_readout_weighted_writer_netlist(vwp=0.40, vwn=0.28, width_u=8.0),
        timeout=20.0,
    )
    pmos_pullup = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_moderate_pmos_pullup.cir",
        _hidden_direct_readout_weighted_writer_netlist(
            vwp=0.40,
            vwn=0.28,
            width_u=0.125,
            readout_gate_mode="restored-excess",
            output_stage="pmos-pullup",
        ),
        timeout=20.0,
    )
    neutral_pmos = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_neutral_existing_state_pmos_pullup.cir",
        _hidden_direct_readout_weighted_writer_netlist(
            vwp=0.40,
            vwn=0.40,
            whp=1.0,
            whn=0.2,
            width_u=0.125,
            readout_gate_mode="restored-excess",
            output_stage="pmos-pullup",
        ),
        timeout=20.0,
    )

    assert float(pmos_pullup["signed_after"]) > float(nmos_pass["signed_after"]) + 20e-3
    assert float(pmos_pullup["signed_after"]) > 0.30
    assert float(neutral_pmos["signed_after"]) > 0.75


def test_multiclass_block_sequence_ngspice_direct_hidden_writer_bounded_pmos_uses_hidden_reference(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    moderate = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_moderate_pmos_bounded_integrated_refs.cir",
        _hidden_direct_readout_weighted_writer_netlist(
            vwp=0.40,
            vwn=0.28,
            readout_high_ref=0.42,
            readout_low_ref=0.28,
            hidden_high_ref=1.05,
            width_u=0.125,
            readout_gate_mode="restored-excess",
            output_stage="pmos-bounded",
        ),
        timeout=20.0,
    )
    neutral = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_neutral_pmos_bounded_integrated_refs.cir",
        _hidden_direct_readout_weighted_writer_netlist(
            vwp=0.40,
            vwn=0.40,
            whp=1.0,
            whn=0.2,
            readout_high_ref=0.42,
            readout_low_ref=0.28,
            hidden_high_ref=1.05,
            width_u=0.125,
            readout_gate_mode="restored-excess",
            output_stage="pmos-bounded",
        ),
        timeout=20.0,
    )

    assert float(moderate["signed_after"]) > 0.30
    assert float(moderate["whp_after"]) < 1.06
    assert float(moderate["whn_after"]) == pytest.approx(0.40, abs=5e-3)
    assert float(neutral["signed_after"]) > 0.75


def test_multiclass_block_sequence_ngspice_complementary_pmos_hidden_writer_does_not_raise_both_rails(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "direct_hidden_writer_two_update_complementary.cir",
        _hidden_direct_readout_weighted_two_update_netlist(),
        timeout=30.0,
    )

    assert float(measures["signed_after_first"]) > 0.30
    assert float(measures["whp_after"]) < 0.45
    assert float(measures["whn_after"]) > 0.40
    assert float(measures["signed_after"]) < -0.10


def test_multiclass_block_sequence_ngspice_live_error_rails_reset_before_eval_writer(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = _two_class_one_hot_records()
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_live_error_reset_hidden.cir",
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            class_count=2,
            feature_count=2,
            readout_update_mode="live",
            hidden_update_mode="readout-weighted",
            error_mode="pairwise-margin-correction-descent",
            score_timing_mode="early",
            score_measure_ns=5.05,
            readout_forward_mode="diode",
        ),
        timeout=60.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(2)]))
        for cycle in range(4, 6)
    ]
    assert final_predictions == [0, 1]
    for class_idx in range(2):
        for feature in range(2):
            assert float(measures[f"c{class_idx}_f{feature}_signed_final"]) == pytest.approx(
                float(measures[f"c{class_idx}_f{feature}_signed_after_train2"]),
                abs=2e-6,
            )
    assert float(measures["c0_f0_signed_final"]) > 100e-3
    assert float(measures["c1_f1_signed_final"]) > 100e-3
    assert float(measures["c0_f1_signed_final"]) < -100e-3
    assert float(measures["c1_f0_signed_final"]) < -100e-3


def test_multiclass_block_sequence_ngspice_label_rail_descent_live_writer(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = _two_class_one_hot_records()
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_label_rail_hidden.cir",
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            class_count=2,
            feature_count=2,
            readout_update_mode="live",
            hidden_update_mode="readout-weighted",
            error_mode="label-rail-descent",
            score_timing_mode="early",
            score_measure_ns=5.05,
            readout_forward_mode="diode",
        ),
        timeout=60.0,
    )

    assert float(measures["c0_errdiff_2"]) > 1.0
    assert float(measures["c1_errdiff_2"]) < -1.0
    assert float(measures["c0_errdiff_3"]) < -1.0
    assert float(measures["c1_errdiff_3"]) > 1.0
    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(2)]))
        for cycle in range(4, 6)
    ]
    assert final_predictions == [0, 1]
    for class_idx in range(2):
        for feature in range(2):
            assert float(measures[f"c{class_idx}_f{feature}_signed_final"]) == pytest.approx(
                float(measures[f"c{class_idx}_f{feature}_signed_after_train2"]),
                abs=2e-6,
            )
    assert float(measures["c0_f0_signed_final"]) > 100e-3
    assert float(measures["c1_f1_signed_final"]) > 100e-3
    assert float(measures["c0_f1_signed_final"]) < -100e-3
    assert float(measures["c1_f0_signed_final"]) < -100e-3


def test_multiclass_block_sequence_ngspice_diode_mirror_score_sensor_reads_live_state(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = _two_class_one_hot_records()
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_diode_mirror_score.cir",
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            class_count=2,
            feature_count=2,
            readout_update_mode="live",
            hidden_update_mode="readout-weighted",
            error_mode="label-rail-descent",
            score_sense_mode="diode-mirror",
            score_timing_mode="early",
            score_measure_ns=5.30,
            readout_forward_mode="diode",
        ),
        timeout=60.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(2)]))
        for cycle in range(4, 6)
    ]
    assert final_predictions == [0, 1]
    assert float(measures["c0_score_net_4"]) > float(measures["c1_score_net_4"]) + 1e-3
    assert float(measures["c1_score_net_5"]) > float(measures["c0_score_net_5"]) + 1e-3
    assert float(measures["c0_f0_signed_final"]) > 100e-3
    assert float(measures["c1_f1_signed_final"]) > 100e-3


def test_multiclass_block_sequence_ngspice_common_ref_gate_tracks_score_above_class_common(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "common_ref_score_gate.cir",
        _common_ref_score_gate_netlist((0.75, 0.45, 0.15)),
        timeout=20.0,
    )

    assert 0.43 < float(measures["score_common_after"]) < 0.50
    assert float(measures["c0_gate_after"]) > float(measures["c1_gate_after"]) + 10e-3
    assert float(measures["c1_gate_after"]) > float(measures["c2_gate_after"]) + 10e-3
    assert float(measures["c0_gate_after"]) > 0.10
    assert float(measures["c2_gate_after"]) < 0.05


def test_multiclass_block_sequence_ngspice_raw_common_ref_gate_tracks_raw_score_above_common(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "raw_common_ref_score_gate.cir",
        _common_ref_score_gate_netlist((0.12, 0.06, 0.0), raw_score=True),
        timeout=20.0,
    )

    assert 0.055 < float(measures["score_common_after"]) < 0.065
    assert float(measures["c0_gate_after"]) > float(measures["c1_gate_after"]) + 1e-3
    assert float(measures["c1_gate_after"]) > float(measures["c2_gate_after"]) + 1e-3
    assert float(measures["c0_gate_after"]) > 5e-3


def test_multiclass_block_sequence_ngspice_target_ref_gate_tracks_score_above_label_score(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "target_ref_score_gate.cir",
        _target_ref_score_gate_netlist((0.75, 0.45, 0.15), target_class=1),
        timeout=20.0,
    )

    assert 0.40 < float(measures["target_score_ref_after"]) < 0.48
    assert float(measures["c0_gate_after"]) > float(measures["c1_gate_after"]) + 10e-3
    assert float(measures["c1_gate_after"]) > float(measures["c2_gate_after"]) + 5e-3
    assert float(measures["c0_gate_after"]) > 0.10
    assert float(measures["c2_gate_after"]) < 0.05


def test_multiclass_block_sequence_ngspice_score_mass_descent_writes_target_and_nontargets(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "score_mass_descent_high_wrong0.cir",
        _score_mass_descent_netlist((0.75, 0.45, 0.15), target_class=1),
        timeout=20.0,
    )

    assert float(measures["score_nontarget_mass_after"]) > 0.40
    assert float(measures["c1_errdiff"]) > 0.10
    assert float(measures["c0_errdiff"]) < -0.10
    assert float(measures["c2_errdiff"]) < -0.01
    assert float(measures["c1_signed_after"]) > 1e-3
    assert float(measures["c0_signed_after"]) < -1e-3
    assert abs(float(measures["c2_signed_after"])) < 1e-6
    assert float(measures["c0_gvn_after"]) > float(measures["c2_gvn_after"]) + 20e-3


def test_multiclass_block_sequence_ngspice_centered_error_rails_cancel_uniform_pressure(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "centered_error_uniform_positive.cir",
        _centered_error_rail_netlist((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        timeout=20.0,
    )

    assert float(measures["common_p_after"]) > 0.45
    for class_idx in range(3):
        assert abs(float(measures[f"c{class_idx}_errdiff"])) < 25e-3


def test_multiclass_block_sequence_ngspice_centered_error_rails_preserve_relative_sign(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "centered_error_target_positive.cir",
        _centered_error_rail_netlist((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
        timeout=20.0,
    )

    assert float(measures["common_p_after"]) > 0.20
    assert float(measures["common_n_after"]) > 0.35
    assert float(measures["c0_errdiff"]) > 25e-3
    assert float(measures["c1_errdiff"]) < -25e-3
    assert float(measures["c2_errdiff"]) < -25e-3
    assert float(measures["c1_errdiff"]) == pytest.approx(float(measures["c2_errdiff"]), abs=10e-3)


def test_multiclass_block_sequence_ngspice_gain_restored_centered_error_rails_preserve_cancel_and_boost(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    uniform = run_netlist(
        ngspice_path,
        tmp_path / "centered_gain_uniform_positive.cir",
        _centered_error_rail_netlist((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), gain_restored=True),
        timeout=20.0,
    )
    directional = run_netlist(
        ngspice_path,
        tmp_path / "centered_gain_target_positive.cir",
        _centered_error_rail_netlist((1.0, 0.0, 0.0), (0.0, 1.0, 1.0), gain_restored=True),
        timeout=20.0,
    )

    for class_idx in range(3):
        assert abs(float(uniform[f"c{class_idx}_errdiff"])) < 75e-3
    assert float(directional["c0_errdiff"]) > float(directional["c0_errdiff_ctr"]) + 10e-3
    assert float(directional["c1_errdiff"]) < float(directional["c1_errdiff_ctr"]) - 10e-3
    assert float(directional["c2_errdiff"]) < float(directional["c2_errdiff_ctr"]) - 10e-3
    assert float(directional["c0_errdiff"]) > 55e-3
    assert float(directional["c1_errdiff"]) < -35e-3
    assert float(directional["c1_errdiff"]) == pytest.approx(float(directional["c2_errdiff"]), abs=20e-3)


def test_multiclass_block_sequence_ngspice_pairwise_winner_error_updates_only_strongest_impostor(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "pairwise_winner_error_rail.cir",
        _pairwise_winner_error_rail_netlist(),
        timeout=30.0,
    )

    assert float(measures["c0_errdiff"]) > 0.20
    assert float(measures["c1_errdiff"]) < -0.20
    assert abs(float(measures["c2_errdiff"])) < 25e-3


def test_multiclass_block_sequence_ngspice_score_mass_target_pressure_tracks_nontarget_mass(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    high_wrong = run_netlist(
        ngspice_path,
        tmp_path / "score_mass_descent_target1_high_wrong0.cir",
        _score_mass_descent_netlist((0.75, 0.45, 0.15), target_class=1),
        timeout=20.0,
    )
    clear = run_netlist(
        ngspice_path,
        tmp_path / "score_mass_descent_target0_clear.cir",
        _score_mass_descent_netlist((0.75, 0.25, 0.15), target_class=0),
        timeout=20.0,
    )

    assert float(high_wrong["score_nontarget_mass_after"]) > float(clear["score_nontarget_mass_after"]) + 0.10
    assert float(high_wrong["c1_errp_after"]) > float(clear["c0_errp_after"]) + 50e-3
    assert float(high_wrong["c1_signed_after"]) > float(clear["c0_signed_after"]) + 1e-3


def test_multiclass_block_sequence_ngspice_common_score_mass_uses_centered_class_contrast(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "common_score_mass_descent_high_wrong0.cir",
        _score_mass_descent_netlist((0.75, 0.45, 0.15), target_class=1, common_centered=True),
        timeout=20.0,
    )

    assert float(measures["c0_score_gate_after"]) > float(measures["c1_score_gate_after"]) + 10e-3
    assert float(measures["c1_score_gate_after"]) > float(measures["c2_score_gate_after"]) + 5e-3
    assert float(measures["score_nontarget_mass_after"]) > 0.05
    assert float(measures["c1_errdiff"]) > 0.01
    assert float(measures["c0_errdiff"]) < -0.01
    assert abs(float(measures["c2_errdiff"])) < abs(float(measures["c0_errdiff"]))
    assert float(measures["c1_signed_after"]) > 1e-6
    assert abs(float(measures["c0_signed_after"])) < 1e-6


def test_multiclass_block_sequence_ngspice_contrast_score_mass_uses_analog_contrast(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "contrast_score_mass_descent_high_wrong0.cir",
        _score_mass_descent_netlist((0.75, 0.45, 0.15), target_class=1, contrast_centered=True),
        timeout=20.0,
    )

    assert float(measures["c0_score_gate_after"]) > float(measures["c1_score_gate_after"]) + 10e-3
    assert float(measures["c1_score_gate_after"]) > float(measures["c2_score_gate_after"]) + 10e-3
    assert float(measures["score_nontarget_mass_after"]) > 0.05
    assert float(measures["c1_errdiff"]) > 0.01
    assert float(measures["c0_errdiff"]) < -0.01
    assert float(measures["c1_signed_after"]) > 1e-6


def test_multiclass_block_sequence_ngspice_common_score_mass_pairwise_adds_mistake_pressure(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    common_only = run_netlist(
        ngspice_path,
        tmp_path / "common_score_mass_only_high_wrong0.cir",
        _score_mass_descent_netlist((0.75, 0.45, 0.15), target_class=1, common_centered=True),
        timeout=20.0,
    )
    hybrid = run_netlist(
        ngspice_path,
        tmp_path / "common_score_mass_pairwise_high_wrong0.cir",
        _score_mass_descent_netlist((0.75, 0.45, 0.15), target_class=1, common_centered=True, pairwise_hybrid=True),
        timeout=20.0,
    )

    assert float(hybrid["c0_gt_target_after"]) > 0.5
    assert float(hybrid["c1_errdiff"]) > float(common_only["c1_errdiff"]) + 0.05
    assert float(hybrid["c0_errdiff"]) < float(common_only["c0_errdiff"]) - 0.05
    assert float(hybrid["c1_signed_after"]) > float(common_only["c1_signed_after"]) + 1e-4


def test_multiclass_block_sequence_ngspice_target_contrast_score_mass_uses_label_reference(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "target_contrast_score_mass_descent_high_wrong0.cir",
        _score_mass_descent_netlist((0.75, 0.45, 0.15), target_class=1, target_centered=True),
        timeout=20.0,
    )

    assert float(measures["c0_score_gate_after"]) > float(measures["c1_score_gate_after"]) + 10e-3
    assert float(measures["c1_score_gate_after"]) > float(measures["c2_score_gate_after"]) + 5e-3
    assert float(measures["score_nontarget_mass_after"]) > 0.05
    assert float(measures["c1_errdiff"]) > 0.01
    assert float(measures["c0_errdiff"]) < -0.01
    assert abs(float(measures["c2_errdiff"])) < abs(float(measures["c0_errdiff"]))
    assert float(measures["c1_signed_after"]) > 1e-6


def test_multiclass_block_sequence_ngspice_restored_binary_descent_gates_target_miss_and_false_positive(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    target_miss = run_netlist(
        ngspice_path,
        tmp_path / "restored_binary_target_miss.cir",
        _restored_score_binary_descent_netlist(targetp=1.1, targetn=0.0, decision=0.0, decisionn=1.2),
        timeout=20.0,
    )
    target_already_wins = run_netlist(
        ngspice_path,
        tmp_path / "restored_binary_target_wins.cir",
        _restored_score_binary_descent_netlist(targetp=1.1, targetn=0.0, decision=1.2, decisionn=0.0),
        timeout=20.0,
    )
    false_positive = run_netlist(
        ngspice_path,
        tmp_path / "restored_binary_false_positive.cir",
        _restored_score_binary_descent_netlist(targetp=0.0, targetn=1.1, decision=1.2, decisionn=0.0),
        timeout=20.0,
    )

    assert float(target_miss["gvp_after"]) > float(target_already_wins["gvp_after"]) + 10e-3
    assert float(target_miss["signed_after"]) > float(target_already_wins["signed_after"]) + 1e-3
    assert float(false_positive["gvn_after"]) > float(target_already_wins["gvn_after"]) + 10e-3
    assert float(false_positive["signed_after"]) < float(target_already_wins["signed_after"]) - 1e-3


def test_multiclass_block_sequence_ngspice_pairwise_binary_descent_gates_loss_and_win(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    target_losing = run_netlist(
        ngspice_path,
        tmp_path / "pairwise_binary_target_losing.cir",
        _pairwise_binary_descent_netlist(targetp=1.1, targetn=0.0, losing_gate=1.2, winning_gate=0.0),
        timeout=20.0,
    )
    target_winning = run_netlist(
        ngspice_path,
        tmp_path / "pairwise_binary_target_winning.cir",
        _pairwise_binary_descent_netlist(targetp=1.1, targetn=0.0, losing_gate=0.0, winning_gate=1.2),
        timeout=20.0,
    )
    nontarget_winning = run_netlist(
        ngspice_path,
        tmp_path / "pairwise_binary_nontarget_winning.cir",
        _pairwise_binary_descent_netlist(targetp=0.0, targetn=1.1, losing_gate=0.0, winning_gate=1.2),
        timeout=20.0,
    )

    assert float(target_losing["gvp_after"]) > float(target_winning["gvp_after"]) + 5e-3
    assert float(target_losing["rgp_after"]) < float(target_winning["rgp_after"]) - 10e-6
    assert float(target_losing["signed_after"]) > float(target_winning["signed_after"]) + 1e-3
    assert float(nontarget_winning["gvn_after"]) > float(target_winning["gvn_after"]) + 5e-3
    assert float(nontarget_winning["rgn_after"]) < float(target_winning["rgn_after"]) - 10e-6
    assert float(nontarget_winning["signed_after"]) < float(target_winning["signed_after"]) - 1e-3


def test_multiclass_block_sequence_ngspice_amplified_pairwise_blend_keeps_base_and_adds_correction(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    target_base = run_netlist(
        ngspice_path,
        tmp_path / "amplified_pairwise_target_base.cir",
        _amplified_pairwise_blend_netlist(
            targetp=1.1,
            targetn=0.0,
            own_score_gate=0.05,
            losing_gate=0.0,
            winning_gate=0.0,
        ),
        timeout=20.0,
    )
    target_losing = run_netlist(
        ngspice_path,
        tmp_path / "amplified_pairwise_target_losing.cir",
        _amplified_pairwise_blend_netlist(
            targetp=1.1,
            targetn=0.0,
            own_score_gate=0.05,
            losing_gate=1.2,
            winning_gate=0.0,
        ),
        timeout=20.0,
    )
    nontarget_base = run_netlist(
        ngspice_path,
        tmp_path / "amplified_pairwise_nontarget_base.cir",
        _amplified_pairwise_blend_netlist(
            targetp=0.0,
            targetn=1.1,
            own_score_gate=0.5,
            losing_gate=0.0,
            winning_gate=0.0,
        ),
        timeout=20.0,
    )
    nontarget_winning = run_netlist(
        ngspice_path,
        tmp_path / "amplified_pairwise_nontarget_winning.cir",
        _amplified_pairwise_blend_netlist(
            targetp=0.0,
            targetn=1.1,
            own_score_gate=0.5,
            losing_gate=0.0,
            winning_gate=1.2,
        ),
        timeout=20.0,
    )

    assert float(target_base["signed_after"]) > 1e-3
    assert float(target_losing["signed_after"]) > float(target_base["signed_after"]) + 1e-6
    assert float(nontarget_base["signed_after"]) < -1e-3
    assert float(nontarget_winning["gvn_after"]) > float(nontarget_base["gvn_after"]) + 5e-3
    assert float(nontarget_winning["rgn_after"]) < float(nontarget_base["rgn_after"]) - 10e-6
    assert float(nontarget_winning["signed_after"]) < float(nontarget_base["signed_after"]) - 1e-3


def test_multiclass_block_sequence_ngspice_amplified_binary_blend_keeps_base_and_adds_correction(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    target_base = run_netlist(
        ngspice_path,
        tmp_path / "amplified_binary_target_base.cir",
        _amplified_binary_blend_netlist(
            targetp=1.1,
            targetn=0.0,
            own_score_gate=0.05,
            decision=1.2,
            decisionn=0.0,
        ),
        timeout=20.0,
    )
    target_miss = run_netlist(
        ngspice_path,
        tmp_path / "amplified_binary_target_miss.cir",
        _amplified_binary_blend_netlist(
            targetp=1.1,
            targetn=0.0,
            own_score_gate=0.05,
            decision=0.0,
            decisionn=1.2,
        ),
        timeout=20.0,
    )
    nontarget_base = run_netlist(
        ngspice_path,
        tmp_path / "amplified_binary_nontarget_base.cir",
        _amplified_binary_blend_netlist(
            targetp=0.0,
            targetn=1.1,
            own_score_gate=0.5,
            decision=0.0,
            decisionn=1.2,
        ),
        timeout=20.0,
    )
    false_positive = run_netlist(
        ngspice_path,
        tmp_path / "amplified_binary_false_positive.cir",
        _amplified_binary_blend_netlist(
            targetp=0.0,
            targetn=1.1,
            own_score_gate=0.5,
            decision=1.2,
            decisionn=0.0,
        ),
        timeout=20.0,
    )

    assert float(target_base["signed_after"]) > 1e-3
    assert float(target_miss["gvp_after"]) > float(target_base["gvp_after"]) + 10e-6
    assert float(target_miss["signed_after"]) > float(target_base["signed_after"]) + 1e-6
    assert float(nontarget_base["signed_after"]) < -1e-3
    assert float(false_positive["gvn_after"]) > float(nontarget_base["gvn_after"]) + 5e-3
    assert float(false_positive["rgn_after"]) < float(nontarget_base["rgn_after"]) - 10e-6
    assert float(false_positive["signed_after"]) < float(nontarget_base["signed_after"]) - 1e-3


def test_multiclass_block_sequence_ngspice_passive_readout_center_leak_reduces_signed_state(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    weak = run_netlist(
        ngspice_path,
        tmp_path / "readout_center_leak_weak.cir",
        _readout_center_leak_netlist(1e12),
        timeout=20.0,
    )
    strong = run_netlist(
        ngspice_path,
        tmp_path / "readout_center_leak_strong.cir",
        _readout_center_leak_netlist(5e6),
        timeout=20.0,
    )

    assert float(strong["vwp_after"]) < float(weak["vwp_after"]) - 10e-3
    assert float(strong["vwn_after"]) > float(weak["vwn_after"]) + 10e-3
    assert float(strong["signed_after"]) < float(weak["signed_after"]) - 20e-3


def test_multiclass_block_sequence_ngspice_amplified_score_gate_depresses_one_hot_off_diagonal(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_amplified_score.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="amplified-score-nontarget",
        ),
        timeout=80.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert float(measures["c1_score_amp_3"]) > 0.40
    assert float(measures["c1_score_amp_3"]) < 0.50
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
        for feature in range(3):
            if feature != class_idx:
                assert float(measures[f"c{class_idx}_f{feature}_signed_final"]) < -10e-3


def test_multiclass_block_sequence_ngspice_common_ref_score_gate_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_common_ref_score.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="common-ref-score-nontarget",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3


def test_multiclass_block_sequence_ngspice_raw_common_ref_score_gate_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_raw_common_ref_score.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="raw-common-ref-score-nontarget",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert abs(float(measures["c0_score_above_common_4"])) > 1e-6
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3


def test_multiclass_block_sequence_ngspice_target_ref_score_gate_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_target_ref_score.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="target-ref-score-nontarget",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
    assert abs(float(measures["c0_score_above_target_4"])) > 1e-6


def test_multiclass_block_sequence_ngspice_amplified_pairwise_blend_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_amplified_pairwise.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="amplified-score-pairwise",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
    assert float(measures["c0_f1_signed_final"]) < -10e-3
    assert float(measures["c1_f0_signed_final"]) < -10e-3


def test_multiclass_block_sequence_ngspice_amplified_binary_blend_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_amplified_binary.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="amplified-score-binary-descent",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
    assert float(measures["c0_f1_signed_final"]) < -10e-3
    assert float(measures["c1_f0_signed_final"]) < -10e-3


def test_multiclass_block_sequence_ngspice_score_mass_descent_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_score_mass.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="score-mass-descent",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert float(measures["score_nontarget_mass_c0_4"]) > 1e-3
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
    assert float(measures["c0_f1_signed_final"]) < -10e-3
    assert float(measures["c1_f0_signed_final"]) < -10e-3


def test_multiclass_block_sequence_ngspice_common_score_mass_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_common_score_mass.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="common-score-mass-descent",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert float(measures["score_nontarget_mass_c0_4"]) > 1e-3
    assert abs(float(measures["c0_score_above_common_4"])) > 1e-6
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 1e-3


def test_multiclass_block_sequence_ngspice_common_score_mass_pairwise_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_common_score_mass_pairwise.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="common-score-mass-pairwise-descent",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert float(measures["score_nontarget_mass_c0_4"]) > 1e-3
    assert abs(float(measures["c0_gt_c1_diff_4"])) > 1e-3
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 1e-3


def test_multiclass_block_sequence_ngspice_contrast_score_mass_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_contrast_score_mass.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="contrast-score-mass-descent",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert float(measures["score_nontarget_mass_c0_4"]) > 1e-3
    assert float(measures["c0_score_contrast_4"]) > 1e-3
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 1e-3


def test_multiclass_block_sequence_ngspice_target_contrast_score_mass_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_target_contrast_score_mass.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="target-contrast-score-mass-descent",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert float(measures["score_nontarget_mass_c0_4"]) > 1e-3
    assert abs(float(measures["c0_score_above_target_4"])) > 1e-6
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 1e-3


def test_multiclass_block_sequence_ngspice_target_only_bias_updates_target_class(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_target_only_bias.cir",
        seq.generate_netlist(
            train_records=_target0_records(1),
            eval_records=_target0_records(1),
            feature_count=1,
            class_bias_mode="target-only",
        ),
        timeout=60.0,
    )

    assert float(measures["c0_f1_signed_final"]) > 10e-3
    assert abs(float(measures["c1_f1_signed_final"])) < 1e-3
    assert abs(float(measures["c2_f1_signed_final"])) < 1e-3
    final_margin = float(measures["c0_score_net_2"]) - max(
        float(measures["c1_score_net_2"]),
        float(measures["c2_score_net_2"]),
    )
    assert final_margin > 2e-3


def test_multiclass_block_sequence_ngspice_label_descent_bias_updates_target_and_nontarget_classes(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_label_descent_bias.cir",
        seq.generate_netlist(
            train_records=_target0_records(1),
            eval_records=_target0_records(1),
            feature_count=1,
            class_bias_mode="label-descent",
        ),
        timeout=60.0,
    )

    assert float(measures["c0_f1_signed_final"]) > 10e-3
    assert float(measures["c1_f1_signed_final"]) < -10e-3
    assert float(measures["c2_f1_signed_final"]) < -10e-3


def test_multiclass_block_sequence_ngspice_nontarget_scale_removes_negative_off_diagonal_updates(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_no_nontarget.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            nontarget_scale=0.0,
        ),
        timeout=60.0,
    )

    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
        for feature in range(3):
            if feature != class_idx:
                assert abs(float(measures[f"c{class_idx}_f{feature}_signed_final"])) < 1e-3


def test_multiclass_block_sequence_ngspice_score_gated_nontarget_keeps_one_hot_diagonal(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_score_gated.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="score-gated-nontarget",
        ),
        timeout=60.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
        for feature in range(3):
            if feature != class_idx:
                assert abs(float(measures[f"c{class_idx}_f{feature}_signed_final"])) < 1e-3


def test_multiclass_block_sequence_ngspice_restored_score_nontarget_keeps_one_hot_learning(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_restored_score.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="restored-score-nontarget",
        ),
        timeout=80.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
        for feature in range(3):
            if feature != class_idx:
                assert float(measures[f"c{class_idx}_f{feature}_signed_final"]) < -10e-3


def test_multiclass_block_sequence_ngspice_restored_score_binary_descent_keeps_one_hot_learning(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_restored_binary.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="restored-score-binary-descent",
        ),
        timeout=80.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
        for feature in range(3):
            if feature != class_idx:
                assert float(measures[f"c{class_idx}_f{feature}_signed_final"]) < -10e-3


def test_multiclass_block_sequence_ngspice_pairwise_binary_descent_keeps_one_hot_predictions(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_pairwise_binary.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="pairwise-binary-descent",
        ),
        timeout=100.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert float(measures["c0_f1_signed_final"]) < -10e-3
    assert float(measures["c1_f1_signed_final"]) > 10e-3
    assert float(measures["c2_f2_signed_final"]) > 10e-3


def test_multiclass_block_sequence_ngspice_pairwise_margin_correction_improves_one_sample_margin(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = [{"label": 1, "inputs": {"x0": 0.85}}]
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_one_sample_pairwise_margin.cir",
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            class_count=3,
            feature_count=1,
            score_capacitance_f=5.0,
            error_mode="pairwise-margin-correction-descent",
        ),
        timeout=80.0,
    )

    initial_scores = [float(measures[f"c{class_idx}_score_net_0"]) for class_idx in range(3)]
    final_scores = [float(measures[f"c{class_idx}_score_net_2"]) for class_idx in range(3)]
    initial_margin = initial_scores[1] - max(initial_scores[0], initial_scores[2])
    final_margin = final_scores[1] - max(final_scores[0], final_scores[2])

    assert float(measures["c1_errdiff_1"]) > 0.35
    assert float(measures["c0_errdiff_1"]) < -0.25
    assert float(measures["c2_errdiff_1"]) < -0.25
    assert float(measures["c1_f0_signed_final"]) > 10e-3
    assert final_margin > initial_margin + 10e-3


def test_multiclass_block_sequence_ngspice_live_pairwise_margin_corrects_wrong_winner(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = [{"label": 1, "inputs": {"x0": 0.85}}]
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_one_sample_live_wrong0_margin.cir",
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            class_count=3,
            feature_count=1,
            score_capacitance_f=5.0,
            error_mode="pairwise-margin-correction-descent",
            readout_update_mode="live",
            initial_readout_states={
                (0, 0): (0.48, 0.28),
                (1, 0): (0.40, 0.40),
                (2, 0): (0.34, 0.40),
            },
        ),
        timeout=80.0,
    )

    initial_scores = [float(measures[f"c{class_idx}_score_net_0"]) for class_idx in range(3)]
    final_scores = [float(measures[f"c{class_idx}_score_net_2"]) for class_idx in range(3)]
    initial_margin = initial_scores[1] - max(initial_scores[0], initial_scores[2])
    final_margin = final_scores[1] - max(final_scores[0], final_scores[2])

    assert initial_scores[0] > initial_scores[1] + 50e-3
    assert float(measures["c1_errdiff_1"]) > 0.35
    assert float(measures["c0_errdiff_1"]) < -0.35
    assert float(measures["c1_f0_signed_final"]) > 0.10
    assert float(measures["c0_f0_signed_final"]) < -0.05
    assert final_margin > initial_margin + 100e-3
    assert final_scores[1] > final_scores[0] + 50e-3


def test_multiclass_block_sequence_ngspice_centered_gain_margin_corrects_wrong_winner(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = [{"label": 1, "inputs": {"x0": 0.85}}]
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_one_sample_centered_gain_wrong0_margin.cir",
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            class_count=3,
            feature_count=1,
            score_capacitance_f=5.0,
            error_mode="pairwise-margin-centered-gain-descent",
            readout_update_mode="live",
            initial_readout_states={
                (0, 0): (0.48, 0.28),
                (1, 0): (0.40, 0.40),
                (2, 0): (0.34, 0.40),
            },
        ),
        timeout=80.0,
    )

    initial_scores = [float(measures[f"c{class_idx}_score_net_0"]) for class_idx in range(3)]
    final_scores = [float(measures[f"c{class_idx}_score_net_2"]) for class_idx in range(3)]
    initial_margin = initial_scores[1] - max(initial_scores[0], initial_scores[2])
    final_margin = final_scores[1] - max(final_scores[0], final_scores[2])

    assert initial_scores[0] > initial_scores[1] + 50e-3
    assert float(measures["c1_errdiff_1"]) > 25e-3
    assert float(measures["c0_errdiff_1"]) < -25e-3
    assert float(measures["c1_f0_signed_final"]) > 0.10
    assert float(measures["c0_f0_signed_final"]) < -10e-3
    assert final_margin > initial_margin + 100e-3
    assert final_scores[1] > final_scores[0] + 25e-3


def test_multiclass_block_sequence_ngspice_restored_winner_blocks_nonwinning_nontargets(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_restored_winner.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
            error_mode="restored-winner-nontarget",
        ),
        timeout=80.0,
    )

    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    assert final_predictions == [0, 1, 2]
    assert abs(float(measures["c0_gt_c1_diff_4"])) > 1.0
    assert float(measures["c0_gt_c1_diff_4"]) == pytest.approx(-float(measures["c1_gt_c0_diff_4"]), abs=1e-6)
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
    assert abs(float(measures["c0_f1_signed_final"])) < 1e-3
    assert abs(float(measures["c0_f2_signed_final"])) < 1e-3
    assert abs(float(measures["c1_f0_signed_final"])) < 1e-3
    assert abs(float(measures["c1_f2_signed_final"])) < 1e-3
    assert abs(float(measures["c2_f0_signed_final"])) < 1e-3
    assert abs(float(measures["c2_f1_signed_final"])) < 1e-3


def test_multiclass_block_sequence_ngspice_persistent_weights_improve_final_margin(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence.cir",
        seq.generate_netlist(train_records=_target0_records(2), eval_records=_target0_records(1)),
        timeout=40.0,
    )

    initial_margin = float(measures["c0_score_net_0"]) - max(
        float(measures["c1_score_net_0"]),
        float(measures["c2_score_net_0"]),
    )
    final_margin = float(measures["c0_score_net_3"]) - max(
        float(measures["c1_score_net_3"]),
        float(measures["c2_score_net_3"]),
    )

    assert abs(initial_margin) < 1e-3
    assert final_margin > initial_margin + 2e-3
    assert float(measures["pre_margin_1"]) > 20e-3
    assert float(measures["act_1"]) > 20e-3
    assert float(measures["elig_1"]) > 20e-3

    c0_after_1 = float(measures["c0_signed_after_train1"])
    c0_after_2 = float(measures["c0_signed_after_train2"])
    c1_after_1 = float(measures["c1_signed_after_train1"])
    c1_after_2 = float(measures["c1_signed_after_train2"])
    assert c0_after_1 > 5e-3
    assert c0_after_2 > c0_after_1 + 5e-3
    assert c1_after_1 < -5e-3
    assert c1_after_2 < c1_after_1 - 5e-3


def test_multiclass_block_sequence_ngspice_one_hot_multiclass_learning(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
        ),
        timeout=60.0,
    )

    initial_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(3)
    ]
    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]

    assert initial_predictions == [0, 0, 0]
    assert final_predictions == [0, 1, 2]
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 10e-3
        for feature in range(3):
            if feature != class_idx:
                assert float(measures[f"c{class_idx}_f{feature}_signed_final"]) < -10e-3


def test_multiclass_block_sequence_ngspice_live_readout_update_moves_one_hot_matrix(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    netlist = seq.generate_netlist(
        train_records=_one_hot_records(),
        eval_records=_one_hot_records(),
        class_count=3,
        feature_count=3,
        readout_update_mode="live",
    )
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_live_onehot.cir",
        netlist,
        timeout=60.0,
    )

    assert "Vacc acc" not in netlist
    assert "Vapply" not in netlist
    assert "Cc0_gvp0" not in netlist
    assert "Cc0_rgp0" not in netlist
    for class_idx in range(3):
        assert float(measures[f"c{class_idx}_f{class_idx}_signed_final"]) > 100e-3
        for feature in range(3):
            if feature != class_idx:
                assert float(measures[f"c{class_idx}_f{feature}_signed_final"]) < -100e-3


def test_multiclass_block_sequence_ngspice_live_readout_is_class_readable_when_sampled_early(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    netlist = seq.generate_netlist(
        train_records=_one_hot_records(),
        eval_records=_one_hot_records(),
        class_count=3,
        feature_count=3,
        readout_update_mode="live",
        score_measure_ns=5.2,
    )
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_live_onehot_early_score.cir",
        netlist,
        timeout=60.0,
    )
    final_predictions = [
        int(np.argmax([float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]))
        for cycle in range(6, 9)
    ]
    final_margins = []
    for cycle, label in zip(range(6, 9), range(3)):
        scores = [float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]
        final_margins.append(scores[label] - max(score for idx, score in enumerate(scores) if idx != label))

    assert "Vacc acc" not in netlist
    assert "Vapply" not in netlist
    assert "Cc0_gvp0" not in netlist
    assert final_predictions == [0, 1, 2]
    assert min(final_margins) > 10e-3


def test_multiclass_block_sequence_ngspice_smaller_score_cap_improves_one_hot_margin(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    default = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_default_score_cap.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=10.0,
        ),
        timeout=60.0,
    )
    smaller = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_sequence_onehot_smaller_score_cap.cir",
        seq.generate_netlist(
            train_records=_one_hot_records(),
            eval_records=_one_hot_records(),
            class_count=3,
            feature_count=3,
            score_capacitance_f=5.0,
        ),
        timeout=60.0,
    )

    def final_min_margin(measures: dict[str, float]) -> float:
        margins = []
        for cycle, label in zip(range(6, 9), range(3)):
            scores = [float(measures[f"c{class_idx}_score_net_{cycle}"]) for class_idx in range(3)]
            margins.append(scores[label] - max(score for idx, score in enumerate(scores) if idx != label))
        return min(margins)

    default_margin = final_min_margin(default)
    smaller_margin = final_min_margin(smaller)
    assert default_margin > 0.0
    assert smaller_margin > 2e-3
    assert smaller_margin > 3.0 * default_margin


def test_multiclass_block_sequence_mnist_scenario_uses_counted_records(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_detect_spice(spice_bin):
        return "/usr/bin/ngspice", "fake-ngspice"

    def fake_dataset_records(name, seed, *, root, download=False):
        calls.append({"name": name, "seed": seed, "root": root, "download": download})
        records = []
        for _ in range(2):
            for label in range(3):
                records.append(
                    {
                        "label": label,
                        "inputs": {f"x{feature}": 0.85 if feature == label else 0.08 for feature in range(8)},
                    }
                )
        return records

    def fake_run_netlist(spice_bin, path, deck, *, timeout):
        assert "Vrow7 row7 0 PWL(" in deck
        assert "Cc2_vwp7 c2_vwp7 0 20f IC=0.4" in deck
        measures = {}
        for cycle in range(9):
            for class_idx in range(3):
                measures[f"c{class_idx}_score_net_{cycle}"] = 1.0 if class_idx == cycle % 3 else 0.0
                measures[f"c{class_idx}_score_{cycle}"] = measures[f"c{class_idx}_score_net_{cycle}"]
                measures[f"c{class_idx}_scoren_{cycle}"] = 0.0
        for class_idx in range(3):
            for feature in range(8):
                measures[f"c{class_idx}_f{feature}_signed_final"] = 0.01
                measures[f"c{class_idx}_f{feature}_vwp_final"] = 0.405
                measures[f"c{class_idx}_f{feature}_vwn_final"] = 0.395
        for train_idx in range(1, 4):
            for class_idx in range(3):
                for feature in range(8):
                    measures[f"c{class_idx}_f{feature}_signed_after_train{train_idx}"] = 0.01
        return measures

    monkeypatch.setattr(seq, "ROOT", tmp_path)
    monkeypatch.setattr(seq, "detect_spice", fake_detect_spice)
    monkeypatch.setattr(seq, "dataset_records", fake_dataset_records)
    monkeypatch.setattr(seq, "run_netlist", fake_run_netlist)

    args = seq.main_for_test(
        [
            "--scenario",
            "mnist",
            "--dataset",
            "mnist3fixed8_6",
            "--class-count",
            "3",
            "--feature-count",
            "8",
            "--train-samples",
            "3",
            "--eval-samples",
            "3",
            "--nontarget-scale",
            "0.5",
            "--nontarget-width-scale",
            "0.75",
            "--error-mode",
            "score-gated-nontarget",
            "--download",
        ]
    )
    summary = seq.run_case(args)

    assert calls == [{"name": "mnist3fixed8_6", "seed": 3, "root": tmp_path, "download": True}]
    assert summary["scenario"] == "mnist"
    assert summary["dataset"] == "mnist3fixed8_6"
    assert summary["sample_order"] == "grouped"
    assert summary["train_samples"] == 3
    assert summary["eval_samples"] == 3
    assert summary["nontarget_scale"] == 0.5
    assert summary["nontarget_width_scale"] == 0.75
    assert summary["error_mode"] == "score-gated-nontarget"
