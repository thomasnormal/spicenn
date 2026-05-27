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


def _one_hot_records() -> list[dict[str, object]]:
    return [
        {"label": label, "inputs": {f"x{feature}": 0.85 if feature == label else 0.0 for feature in range(3)}}
        for label in range(3)
    ]


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
    with pytest.raises(ValueError, match="nontarget-scale"):
        seq.main_for_test(["--nontarget-scale", "1.5"])
    with pytest.raises(ValueError, match="nontarget-width-scale"):
        seq.main_for_test(["--nontarget-width-scale", "-0.1"])
    with pytest.raises(ValueError, match="nontarget_scale"):
        seq.generate_netlist(train_records=records, eval_records=records, nontarget_scale=-0.1)
    with pytest.raises(ValueError, match="nontarget_width_scale"):
        seq.generate_netlist(train_records=records, eval_records=records, nontarget_width_scale=1.1)
    with pytest.raises(ValueError, match="error_mode"):
        seq.generate_netlist(train_records=records, eval_records=records, error_mode="missing")
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
    assert summary["train_samples"] == 3
    assert summary["eval_samples"] == 3
    assert summary["nontarget_scale"] == 0.5
    assert summary["nontarget_width_scale"] == 0.75
    assert summary["error_mode"] == "score-gated-nontarget"
