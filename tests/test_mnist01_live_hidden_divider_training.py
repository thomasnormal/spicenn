from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_mnist01_fixed_feature_divider_training as mnist01_fixed  # noqa: E402
import run_mnist01_live_hidden_divider_training as mnist01_hidden  # noqa: E402


def _hidden_feature_margin(
    measures: dict[str, float],
    records: list[dict[str, object]],
    phase: str,
    sample_idx: int,
    *,
    node: str,
) -> float:
    hidden, _feature = mnist01_hidden._probe_hidden_feature(records[sample_idx])
    values = [measures[f"{phase}_{node}_h{idx}_{sample_idx}"] for idx in range(mnist01_hidden.HIDDEN)]
    active = values[hidden]
    strongest_inactive = max(value for idx, value in enumerate(values) if idx != hidden)
    return active - strongest_inactive


def _hidden_feature_pre_evidence(
    measures: dict[str, float],
    records: list[dict[str, object]],
    phase: str,
    sample_idx: int,
    *,
    hidden_count: int = mnist01_hidden.HIDDEN,
    hidden_init_mode: str = "quadrant",
) -> float:
    hidden, _feature = mnist01_hidden._probe_hidden_feature(
        records[sample_idx],
        hidden_count,
        hidden_init_mode,
    )
    return measures[f"{phase}_pre_signed_h{hidden}_{sample_idx}"]


def _require_mnist_raw() -> None:
    raw = ROOT / "data/MNIST/raw"
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    if not all((raw / name).exists() for name in required):
        pytest.skip("raw MNIST IDX files are not available")


def test_mnist01_live_hidden_records_and_quadrant_mapping_are_real_4x4_mnist() -> None:
    _require_mnist_raw()

    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    assert [sample["label"] for sample in train] == [0, 1]
    assert [sample["label"] for sample in evals] == [0, 1]
    assert mnist01_hidden.hidden_block_for_feature(0, 4) == 0
    assert mnist01_hidden.hidden_block_for_feature(2, 4) == 1
    assert mnist01_hidden.hidden_block_for_feature(8, 4) == 2
    assert mnist01_hidden.hidden_block_for_feature(15, 4) == 3
    assert mnist01_hidden.hidden_unit_for_feature(15, 16, 16, "identity") == 15
    assert mnist01_hidden.patch2x2_hidden_count(16) == 9
    assert mnist01_hidden.patch2x2_features_for_hidden(0, 16, 9) == (0, 1, 4, 5)
    assert mnist01_hidden.patch2x2_features_for_hidden(8, 16, 9) == (10, 11, 14, 15)
    assert mnist01_hidden.patch2x2_hidden_count(32) == 18
    assert mnist01_hidden.patch2x2_features_for_hidden(8, 32, 18) == (10, 11, 14, 15)
    assert mnist01_hidden.patch2x2_features_for_hidden(17, 32, 18) == (26, 27, 30, 31)
    assert mnist01_hidden.hidden_unit_for_feature(15, 16, 9, "patch2x2") == 8
    assert mnist01_hidden.hidden_unit_for_feature(31, 32, 18, "patch2x2") == 17
    for sample in train + evals:
        assert len(sample["features"]) == 16
        assert all(0.0 <= value <= 1.1 for value in sample["features"])
        assert max(sample["features"]) > 0.4


def test_mnist01_live_hidden_netlist_is_live_transistor_path() -> None:
    train = [
        {"features": [1.0] + [0.0] * 15, "label": 0},
        {"features": [0.0, 1.0] + [0.0] * 14, "label": 1},
    ]
    netlist = mnist01_hidden.mnist01_live_hidden_netlist(train, train)

    assert "\nB" not in netlist
    assert "Vapply" not in netlist
    assert "gvp" not in netlist
    assert "ghp" not in netlist
    assert "Mh0f0p_phi px0 featphi h0f0pmid 0 NSENSE" in netlist
    assert "Mhrow1_restore hrow1 hrow1_ctrl vdd vdd PMOS" in netlist
    assert "Mnorm0_score rd0 c0_scorep mir0 0 NSENSE" in netlist
    assert "Mc0_h0_score_pa vdd hrow0 c0_h0_score_pa 0 NSENSE" in netlist
    assert "Mc0_h0_score_pw c0_h0_score_pa c0_vwp0 c0_h0_score_pb 0 NSENSE" in netlist
    assert "Vhcgphi hcgphi 0 PWL" in netlist
    assert "Vhiddenwritephi hiddenwritephi 0 PWL" in netlist
    assert "Cc0_herrp c0_herrp 0 2f IC=0" in netlist
    assert "Mherr_c0p_m vdd b1low herr_c0p_a vdd PMOS" in netlist
    assert "Mh1_c0_cred_pv_e vdd c0_herrp h1_c0_cred_pv_e 0 NSENSE" in netlist
    assert "Mc0_f1_live_pos_up_p c0_vwp1 c0_f1_live_pos_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert "Mh1f6_live_pup_pgate_phi h1f6_live_pup_pgphi errphi 0 0 NSENSE" in netlist
    assert ".meas tran train_hrow_probe_0" in netlist
    assert ".meas tran initial_pre_signed_h0_0" in netlist
    assert ".meas tran initial_act_h0_0" in netlist
    assert ".meas tran train_hcredit_gate_write_probe_0" in netlist
    assert ".meas tran final_hrow_h3_1" in netlist
    assert ".meas tran final_margin_improvement_1" in netlist

    readout_error_credit_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_credit_error_source="readout",
    )
    assert "Mh1_c0_cred_pv_e vdd c0_errp h1_c0_cred_pv_e 0 NSENSE" in readout_error_credit_netlist
    assert "Mh1_c0_cred_pv_e vdd c0_herrp h1_c0_cred_pv_e 0 NSENSE" not in readout_error_credit_netlist

    differential_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_writer_topology="pmos-differential",
    )
    assert (
        "Mh1f6_live_pos_up_ctrl_phi h1f6_live_pos_up_ctrl_phi errphi 0 0 NSENSE"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_neg_up_ctrl_phi h1f6_live_neg_up_ctrl_phi errphi 0 0 NSENSE"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_pos_up_ctrl_latch h1f6_live_pos_up_ctrl h1f6_live_neg_up_ctrl vdd vdd PMOS"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_neg_up_ctrl_latch h1f6_live_neg_up_ctrl h1f6_live_pos_up_ctrl vdd vdd PMOS"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_pos_dn_select h1f6_live_pos_dn h1f6_live_neg_up_ctrl h1f6_live_pos_dn_sel 0 NSENSE"
        in differential_netlist
    )
    assert (
        "Mh1f6_live_neg_dn_select h1f6_live_neg_dn h1f6_live_pos_up_ctrl h1f6_live_neg_dn_sel 0 NSENSE"
        in differential_netlist
    )

    preamp_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_credit_gate_mode="dynamic-preamp",
        hidden_writer_topology="pmos-differential",
        hidden_write_start_train_index=1,
    )
    assert "M1_hcg_sense_p h1_hcg_pre_n h1_hdp h1_hcg_tail 0 NSENSE" in preamp_netlist
    assert "M1_hcg_support_p vdd h1_hdp h1_hcg_support 0 NSENSE" in preamp_netlist
    assert "M1_hcg_write_support hiddenwritephi h1_hcg_support h1_hcg_write_mid 0 NSENSE" in preamp_netlist
    assert "M1_hcg_pos_pull h1_hdp_gate h1_hcg_pre_n 0 0 NMOS" in preamp_netlist
    assert (
        "Mh1f6_live_pos_up_ctrl_phi h1f6_live_pos_up_ctrl_phi h1_hcg_write 0 0 NSENSE"
        in preamp_netlist
    )
    assert (
        "Mh1f6_live_pos_dn_phi h1f6_live_pos_dn_sel h1_hcg_write h1f6_live_pos_dn_phi 0 NSENSE"
        in preamp_netlist
    )

    signcharge_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_credit_gate_mode="dynamic-preamp",
        hidden_writer_topology="pmos-signcharge",
        hidden_write_start_train_index=1,
    )
    assert "Mh1f6_live_pup_pgate_sel h1f6_live_pup_pgate px6 h1f6_live_pup_pgmid 0 NSENSE" in signcharge_netlist
    assert "Ch1f6_live_pup_packet h1f6_live_pup_packet 0 0.25f IC=1.05" in signcharge_netlist
    assert "Mh1f6_live_pup_pmos wh1f6p h1f6_live_pup_pgate h1f6_live_pup_packet vdd PMOS" in signcharge_netlist
    assert "Ch1f6_live_pup_pgmid h1f6_live_pup_pgmid 0 0.001f IC=0" in signcharge_netlist
    assert "Ch1f6_live_pup_pgphi h1f6_live_pup_pgphi 0 0.001f IC=0" in signcharge_netlist
    assert "Mh1f6_live_pos_up_ctrl_phi" not in signcharge_netlist

    hidden_write_phase_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_writer_phase_mode="hidden-write",
        hidden_write_start_train_index=999,
    )
    assert "Mh1f6_live_pup_pgate_phi h1f6_live_pup_pgphi hiddenwritephi 0 0 NSENSE" in hidden_write_phase_netlist
    assert "Mh1f6_live_pup_pgate_phi h1f6_live_pup_pgphi errphi 0 0 NSENSE" not in hidden_write_phase_netlist

    activation_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_activation_mode="differential-preamp",
    )
    assert "Mh0_act_sense_p h0_act_sense_n pre0_p h0_act_sense_tail 0 NSENSE" in activation_netlist
    assert "Mh0_act_sense_tail h0_act_sense_tail featphi 0 0 NSENSE W=4u" in activation_netlist
    assert "Mact0_diff_restore act0 h0_act_sense_n vdd vdd PMOS" in activation_netlist

    input_contrast_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_input_mode="contrast-common-gate",
    )
    assert "Cpx_common px_common 0 8f IC=0" in input_contrast_netlist
    assert "Rpx_common_px0 px_common px0 20000" in input_contrast_netlist
    assert "Mpxcontrast_f0_pass_g px0 pxgate0 pxcontrast_f0_pass_mid 0 NSENSE" in input_contrast_netlist
    assert "Mh0f0p_phi px0 featphi h0f0pinput 0 NSENSE" in input_contrast_netlist
    assert "Mh0f0p_gate h0f0pinput pxgate0 h0f0pmid 0 NSENSE" in input_contrast_netlist
    assert "Mh0f0p_phi px0 featphi h0f0pmid 0 NSENSE" not in input_contrast_netlist

    restored_input_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_input_mode="restored-common-gate",
    )
    assert "Cpxgate0_low pxgate0_low 0 1f IC=1.2" in restored_input_netlist
    assert "Mpxgate0_low_dis_g pxgate0_low pxgate0 pxgate0_low_mid 0 NSENSE" in restored_input_netlist
    assert "Cpxdrive0 pxdrive0 0 4f IC=0" in restored_input_netlist
    assert "Mpxdrive0_rest pxdrive0 pxgate0_low vdd vdd PMOS" in restored_input_netlist
    assert "Mh0f0p_gate h0f0pinput pxdrive0 h0f0pmid 0 NSENSE" in restored_input_netlist

    row_select_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_activation_mode="differential-preamp",
        hidden_row_select_mode="act-common-gate",
    )
    assert "Rhidden_act_common_act0 hidden_act_common act0 100000" in row_select_netlist
    assert "Chactgate0 hactgate0 0 8f IC=0" in row_select_netlist
    assert "Mhactcontrast_h0_pass_g act0 hactgate0 hactcontrast_h0_pass_mid 0 NSENSE" in row_select_netlist
    assert "Mhrow0_ctrl_a hrow0_ctrl act_contrast0 hrow0_mid 0 NMOS" in row_select_netlist
    assert "Mhrow1_ctrl_a hrow1_ctrl act1 hrow1_mid 0 NSENSE" not in row_select_netlist

    pre_diff_readout_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        readout_activation_mode="pre-differential",
    )
    assert "Mc0_h0_score_ppa vdd pre0_p c0_h0_score_ppa 0 NSENSE" in pre_diff_readout_netlist
    assert "Mc0_h0_score_ppw c0_h0_score_ppa c0_vwp0 c0_h0_score_ppb 0 NSENSE" in pre_diff_readout_netlist
    assert "Mc0_h0_score_nna vdd pre0_n c0_h0_score_nna 0 NSENSE" in pre_diff_readout_netlist
    assert "Mc0_h0_score_nnw c0_h0_score_nna c0_vwn0 c0_h0_score_nnb 0 NSENSE" in pre_diff_readout_netlist
    assert "Mc0_h0_score_pna vdd pre0_p c0_h0_score_pna 0 NSENSE" in pre_diff_readout_netlist
    assert "Mc0_h0_score_pnw c0_h0_score_pna c0_vwn0 c0_h0_score_pnb 0 NSENSE" in pre_diff_readout_netlist
    assert "Mc0_h0_score_npa vdd pre0_n c0_h0_score_npa 0 NSENSE" in pre_diff_readout_netlist
    assert "Mc0_h0_score_npw c0_h0_score_npa c0_vwp0 c0_h0_score_npb 0 NSENSE" in pre_diff_readout_netlist

    pre_diff_writer_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        readout_writer_activation_mode="pre-differential",
    )
    assert "Mc0_f0_live_prep_pos_dn_e c0_vwn0 pre0_p c0_f0_live_prep_pos_dn 0 NSENSE" in pre_diff_writer_netlist
    assert "Mc0_f0_live_prep_pos_up_ctrl_e c0_f0_live_prep_pos_up_ctrl pre0_p" in pre_diff_writer_netlist
    assert "Mc0_f0_live_pren_pos_dn_e c0_vwn0 pre0_n c0_f0_live_pren_pos_dn 0 NSENSE" in pre_diff_writer_netlist
    assert "Mc0_f0_live_pren_pos_dn_d c0_f0_live_pren_pos_dn_sel c0_errn vwlo_ref 0 NSENSE" in pre_diff_writer_netlist
    assert "Mc0_f0_live_pren_neg_dn_d c0_f0_live_pren_neg_dn_sel c0_errp vwlo_ref 0 NSENSE" in pre_diff_writer_netlist

    identity_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_count=16,
        hidden_init_mode="identity",
    )
    assert "Cwh0f0p wh0f0p 0 20f IC=1.05" in identity_netlist
    assert "Cwh0f1p wh0f1p 0 20f IC=0.05" in identity_netlist
    assert "Cwh15f15p wh15f15p 0 20f IC=1.05" in identity_netlist
    assert "Mc1_h15_score_pa vdd hrow15 c1_h15_score_pa 0 NSENSE" in identity_netlist
    assert ".meas tran final_hrow_h15_1" in identity_netlist
    nonsquare_train = [
        {"features": [1.0, 0.0, 0.5, 0.2, 0.7, 0.1], "label": 0},
        {"features": [0.0, 1.0, 0.2, 0.5, 0.1, 0.7], "label": 1},
    ]
    nonsquare_identity_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        nonsquare_train,
        nonsquare_train,
        hidden_count=6,
        hidden_init_mode="identity",
    )
    assert "Cwh5f5p wh5f5p 0 20f IC=1.05" in nonsquare_identity_netlist
    assert "Mc1_h5_score_pa vdd hrow5 c1_h5_score_pa 0 NSENSE" in nonsquare_identity_netlist
    sparse_identity_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        nonsquare_train,
        nonsquare_train,
        hidden_count=6,
        hidden_init_mode="identity",
        hidden_connectivity_mode="identity-sparse",
    )
    assert "Cwh5f5p wh5f5p 0 20f IC=1.05" in sparse_identity_netlist
    assert "Mh5f5p_w h5f5pmid wh5f5p pre5_p 0 NSENSE" in sparse_identity_netlist
    assert "Mh5f5_live_pup_pmos wh5f5p" in sparse_identity_netlist
    assert "Cwh0f1p wh0f1p 0 20f" not in sparse_identity_netlist
    assert "Mh0f1p_w h0f1pmid wh0f1p pre0_p 0 NSENSE" not in sparse_identity_netlist
    assert "Mh0f1_live_pup_pmos wh0f1p" not in sparse_identity_netlist

    patch_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_count=9,
        hidden_init_mode="patch2x2",
        hidden_connectivity_mode="patch2x2-sparse",
    )
    assert "Cwh0f0p wh0f0p 0 20f IC=1.05" in patch_netlist
    assert "Cwh0f1p wh0f1p 0 20f IC=1.05" in patch_netlist
    assert "Cwh0f4p wh0f4p 0 20f IC=1.05" in patch_netlist
    assert "Cwh0f5p wh0f5p 0 20f IC=1.05" in patch_netlist
    assert "Cwh8f10p wh8f10p 0 20f IC=1.05" in patch_netlist
    assert "Cwh8f15p wh8f15p 0 20f IC=1.05" in patch_netlist
    assert "Mh0f5p_w h0f5pmid wh0f5p pre0_p 0 NSENSE" in patch_netlist
    assert "Mh8f15p_w h8f15pmid wh8f15p pre8_p 0 NSENSE" in patch_netlist
    assert "Mh8f15_live_pup_pmos wh8f15p" in patch_netlist
    assert "Mc1_h8_score_pa vdd hrow8 c1_h8_score_pa 0 NSENSE" in patch_netlist
    assert ".meas tran final_hrow_h8_1" in patch_netlist
    assert "Cwh0f2p wh0f2p 0 20f" not in patch_netlist
    assert "Cwh8f9p wh8f9p 0 20f" not in patch_netlist
    assert "Mh0f2p_w h0f2pmid wh0f2p pre0_p 0 NSENSE" not in patch_netlist
    assert "Mh8f9_live_pup_pmos wh8f9p" not in patch_netlist

    complement_patch_train = mnist01_fixed.add_complement_features(train, scale=0.5)
    complement_patch_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        complement_patch_train,
        complement_patch_train,
        hidden_count=18,
        hidden_init_mode="patch2x2",
        hidden_connectivity_mode="patch2x2-sparse",
    )
    assert "Cwh0f0p wh0f0p 0 20f IC=1.05" in complement_patch_netlist
    assert "Cwh9f16p wh9f16p 0 20f IC=1.05" in complement_patch_netlist
    assert "Cwh17f31p wh17f31p 0 20f IC=1.05" in complement_patch_netlist
    assert "Mh17f31p_w h17f31pmid wh17f31p pre17_p 0 NSENSE" in complement_patch_netlist
    assert "Mh17f31_live_pup_pmos wh17f31p" in complement_patch_netlist
    assert "Cwh0f16p wh0f16p 0 20f" not in complement_patch_netlist
    assert "Cwh8f31p wh8f31p 0 20f" not in complement_patch_netlist
    assert "Cwh0f18p wh0f18p 0 20f" not in complement_patch_netlist
    assert "Mh0f18p_w h0f18pmid wh0f18p pre0_p 0 NSENSE" not in complement_patch_netlist

    normalized_writer_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        readout_writer_normalization_mode="activity-gate",
    )
    assert "Chrow_activity_gate hrow_activity_gate 0 80f IC=1.2" in normalized_writer_netlist
    assert "Mhrow_activity_gate_h0_phi hrow_activity_gate_h0_mid scorephi 0 0 NSENSE" in normalized_writer_netlist
    assert "Mc0_f0_live_pos_dn_g c0_f0_live_pos_dn_allguard hrow_activity_gate c0_f0_live_pos_dn 0 NREL" in normalized_writer_netlist
    assert "Mc0_f0_live_pos_dn_e c0_vwn0 hrow0 c0_f0_live_pos_dn_allguard 0 NSENSE" in normalized_writer_netlist

    light_measure_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        measure_eval_hidden_states=False,
        tran_step_ps=10.0,
    )
    assert ".meas tran initial_prep_h0_0" not in light_measure_netlist
    assert ".meas tran final_hrow_h3_1" not in light_measure_netlist
    assert ".meas tran initial_margin_0" in light_measure_netlist
    assert ".meas tran final_margin_1" in light_measure_netlist
    assert ".meas tran train_margin_0" in light_measure_netlist
    assert ".meas tran train_target_signed_0" in light_measure_netlist
    assert ".meas tran train_hrow_probe_0" in light_measure_netlist
    assert ".tran 10p " in light_measure_netlist

    readout_measure_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        hidden_count=16,
        hidden_init_mode="identity",
        measure_eval_hidden_states=False,
        measure_readout_states=True,
        tran_step_ps=10.0,
    )
    assert ".meas tran final_readout_c0_vwp_h0" in readout_measure_netlist
    assert ".meas tran final_readout_c1_signed_h15" in readout_measure_netlist


def test_mnist01_live_hidden_forward_metric_rows_cover_every_forward_pass() -> None:
    train = [
        {"features": [1.0] + [0.0] * 15, "label": 0},
        {"features": [0.0, 1.0] + [0.0] * 14, "label": 1},
    ]
    evals = [
        {"features": [0.5] + [0.0] * 15, "label": 0},
        {"features": [0.0, 0.5] + [0.0] * 14, "label": 1},
    ]
    measures: dict[str, float] = {}
    for phase in ("initial", "train", "final"):
        for idx, margin in enumerate((-1.0e-3, 2.0e-3) if phase == "train" else (0.0, 3.0e-3)):
            measures[f"{phase}_target_signed_{idx}"] = margin
            measures[f"{phase}_other_signed_{idx}"] = 0.0
            measures[f"{phase}_margin_{idx}"] = margin

    rows = mnist01_hidden.forward_metric_rows(train, evals, measures, loss_margin_scale_v=1.0e-3)

    assert [row["phase"] for row in rows] == ["initial", "initial", "train", "train", "final", "final"]
    assert [row["phase_index"] for row in rows] == [0, 1, 0, 1, 0, 1]
    assert rows[2]["correct"] == 0
    assert rows[3]["correct"] == 1
    assert rows[3]["softplus_loss"] < rows[2]["softplus_loss"]
    assert rows[3]["phase_cumulative_accuracy"] == pytest.approx(0.5)
    assert rows[-1]["phase_cumulative_accuracy"] == pytest.approx(0.5)
    assert rows[-1]["cumulative_accuracy"] == pytest.approx(3.0 / 6.0)


def test_mnist01_live_hidden_hidden_state_metric_rows_rank_measured_eval_features() -> None:
    evals = [
        {"features": [0.1, 0.9, 0.0, 0.0], "label": 0},
        {"features": [0.0, 0.0, 0.8, 0.2], "label": 1},
    ]
    measures: dict[str, float] = {}
    pre_signed_by_sample = [
        [1.0e-3, 5.0e-3, -2.0e-3, 0.5e-3],
        [0.0, 2.0e-3, 7.0e-3, -1.0e-3],
    ]
    for sample_idx, signed_values in enumerate(pre_signed_by_sample):
        for hidden, signed in enumerate(signed_values):
            measures[f"final_prep_h{hidden}_{sample_idx}"] = signed + 10.0e-3
            measures[f"final_pren_h{hidden}_{sample_idx}"] = 10.0e-3
            measures[f"final_pre_signed_h{hidden}_{sample_idx}"] = signed
            measures[f"final_act_h{hidden}_{sample_idx}"] = max(signed, 0.0)
            measures[f"final_hrow_h{hidden}_{sample_idx}"] = 1.2 if signed > 0.0 else 0.0

    rows = mnist01_hidden.hidden_state_metric_rows(
        evals,
        measures,
        4,
        phase="final",
        hidden_init_mode="identity",
    )

    assert len(rows) == 8
    sample0 = [row for row in rows if row["phase_index"] == 0]
    sample1 = [row for row in rows if row["phase_index"] == 1]
    assert [row["hidden"] for row in sample0 if row["is_probe_hidden"]] == [1]
    assert [row["hidden"] for row in sample1 if row["is_probe_hidden"]] == [2]
    assert [row["hidden"] for row in sample0 if row["is_best_pre_signed"]] == [1]
    assert [row["hidden"] for row in sample1 if row["is_best_pre_signed"]] == [2]
    assert sample0[1]["pre_signed_rank"] == 0
    assert sample1[2]["pre_signed_rank"] == 0

    with pytest.raises(ValueError, match="initial/final"):
        mnist01_hidden.hidden_state_metric_rows(evals, measures, 4, phase="train")


def test_mnist01_live_hidden_readout_state_metric_rows_rank_final_weights() -> None:
    measures: dict[str, float] = {}
    signed_by_class = [
        [2.0e-3, -1.0e-3, 5.0e-3],
        [-3.0e-3, 4.0e-3, 1.0e-3],
    ]
    for output, signed_values in enumerate(signed_by_class):
        for hidden, signed in enumerate(signed_values):
            measures[f"final_readout_c{output}_vwp_h{hidden}"] = 0.40 + signed
            measures[f"final_readout_c{output}_vwn_h{hidden}"] = 0.40
            measures[f"final_readout_c{output}_signed_h{hidden}"] = signed

    rows = mnist01_hidden.readout_state_metric_rows(measures, 3)

    assert len(rows) == 6
    class0 = [row for row in rows if row["class"] == 0]
    class1 = [row for row in rows if row["class"] == 1]
    assert [row["hidden"] for row in class0 if row["signed_rank"] == 0] == [2]
    assert [row["hidden"] for row in class1 if row["signed_rank"] == 0] == [1]
    assert class0[2]["signed_v"] == pytest.approx(5.0e-3)
    assert class1[1]["signed_v"] == pytest.approx(4.0e-3)

    with pytest.raises(ValueError, match="after training"):
        mnist01_hidden.readout_state_metric_rows(measures, 3, phase="train")


@pytest.mark.ngspice
def test_mnist01_live_hidden_forward_metric_rows_from_ngspice_measures(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    train = [
        {"features": [1.0] + [0.0] * 15, "label": 0},
        {"features": [0.0, 1.0] + [0.0] * 14, "label": 1},
    ]
    evals = [
        {"features": [0.5] + [0.0] * 15, "label": 0},
        {"features": [0.0, 0.5] + [0.0] * 14, "label": 1},
    ]

    measures = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_controller_forward_metrics.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            evals,
            hidden_count=16,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
            hidden_activation_mode="differential-preamp",
            hidden_activation_sense_width_u=4.0,
            measure_eval_hidden_states=False,
            tran_step_ps=20.0,
        ),
        timeout=90.0,
    )

    rows = mnist01_hidden.forward_metric_rows(train, evals, measures, loss_margin_scale_v=1.0e-3)

    assert [row["phase"] for row in rows] == ["initial", "initial", "train", "train", "final", "final"]
    assert [row["phase_index"] for row in rows] == [0, 1, 0, 1, 0, 1]
    assert all(row["softplus_loss"] >= 0.0 for row in rows)
    assert all(0.0 <= row["cumulative_accuracy"] <= 1.0 for row in rows)
    assert rows[-2]["correct"] == 1
    assert rows[-1]["correct"] == 1
    assert rows[-2]["softplus_loss"] < rows[0]["softplus_loss"]
    assert rows[-1]["softplus_loss"] < rows[1]["softplus_loss"]
    assert rows[-1]["phase_cumulative_accuracy"] == pytest.approx(1.0)


def test_mnist01_live_hidden_netlist_validation() -> None:
    sample = {"features": [1.0] * 16, "label": 0}

    with pytest.raises(ValueError, match="empty"):
        mnist01_hidden.mnist01_live_hidden_netlist([], [sample])
    with pytest.raises(ValueError, match="exactly four"):
        mnist01_hidden.mnist01_live_hidden_netlist([sample], [sample], hidden_count=3)
    with pytest.raises(ValueError, match="identity"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_count=4,
            hidden_init_mode="identity",
        )
    with pytest.raises(ValueError, match="identity-sparse"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_connectivity_mode="BAD",
        )
    with pytest.raises(ValueError, match="identity-sparse"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_connectivity_mode="identity-sparse",
        )
    with pytest.raises(ValueError, match="patch2x2"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_count=8,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        )
    with pytest.raises(ValueError, match="patch2x2-sparse"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_connectivity_mode="patch2x2-sparse",
        )
    with pytest.raises(ValueError, match="square"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [{"features": [1.0, 0.0, 0.5, 0.2, 0.7, 0.1], "label": 0}],
            [{"features": [1.0, 0.0, 0.5, 0.2, 0.7, 0.1], "label": 0}],
            hidden_count=9,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        )
    with pytest.raises(ValueError, match="hidden_init_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_init_mode="BAD",
        )
    with pytest.raises(ValueError, match="square"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [{"features": [1.0, 0.0], "label": 0}],
            [{"features": [1.0, 0.0], "label": 0}],
        )
    with pytest.raises(ValueError, match="labels"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [{"features": [1.0] * 16, "label": 2}],
            [sample],
        )
    with pytest.raises(ValueError, match="positive"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_error_route_width_u=0.0,
        )
    with pytest.raises(ValueError, match="positive"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            tran_step_ps=0.0,
        )
    with pytest.raises(ValueError, match="hidden_writer_topology"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_writer_topology="BAD",
        )
    with pytest.raises(ValueError, match="hidden_writer_phase_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_writer_phase_mode="BAD",
        )
    with pytest.raises(ValueError, match="owns the hidden writer phase"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_credit_gate_mode="dynamic-preamp",
            hidden_writer_topology="pmos-differential",
            hidden_writer_phase_mode="hidden-write",
        )
    with pytest.raises(ValueError, match="active ordinary hiddenwritephi"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_writer_phase_mode="hidden-write",
            hidden_write_start_train_index=0,
        )
    with pytest.raises(ValueError, match="hidden_credit_gate_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_credit_gate_mode="BAD",
        )
    with pytest.raises(ValueError, match="hidden_credit_error_source"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_credit_error_source="BAD",
        )
    with pytest.raises(ValueError, match="hidden_activation_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_activation_mode="BAD",
        )
    with pytest.raises(ValueError, match="hidden_input_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_input_mode="BAD",
        )
    with pytest.raises(ValueError, match="hidden_row_select_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_row_select_mode="BAD",
        )
    with pytest.raises(ValueError, match="readout_activation_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            readout_activation_mode="BAD",
        )
    with pytest.raises(ValueError, match="readout_writer_activation_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            readout_writer_activation_mode="BAD",
        )
    with pytest.raises(ValueError, match="readout_writer_normalization_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            readout_writer_normalization_mode="BAD",
        )
    with pytest.raises(ValueError, match="pre-differential"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            readout_writer_activation_mode="pre-differential",
            readout_writer_normalization_mode="activity-gate",
        )
    with pytest.raises(ValueError, match="dynamic-preamp"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_credit_gate_mode="dynamic-preamp",
        )
    with pytest.raises(ValueError, match="hidden_write_start_train_index"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_write_start_train_index=-1,
        )
    with pytest.raises(ValueError, match="hidden credit sense window"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_credit_sense_start_ns=6.0,
            hidden_credit_sense_end_ns=5.0,
        )
    with pytest.raises(ValueError, match="hidden write window"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_write_start_ns=8.0,
            hidden_write_end_ns=10.5,
        )
    with pytest.raises(ValueError, match="even"):
        mnist01_hidden.hidden_block_for_feature(0, 5)
    with pytest.raises(ValueError, match="outside"):
        mnist01_hidden.hidden_block_for_feature(16, 4)


@pytest.mark.ngspice
def test_mnist01_live_hidden_differential_activation_reads_unsaturated_synthetic_feature(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    sample = {"features": [1.0] + [0.0] * 15, "label": 0}
    common_kwargs = dict(
        hidden_inside_positive=0.65,
        hidden_outside_positive=0.45,
        hidden_inside_negative=0.45,
        hidden_outside_negative=0.65,
        hidden_activation_sense_width_u=64.0,
    )

    single_ended = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_hidden_single_ended_mild.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            **common_kwargs,
        ),
        timeout=60.0,
    )
    differential = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_hidden_diff_activation_mild.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_activation_mode="differential-preamp",
            **common_kwargs,
        ),
        timeout=60.0,
    )
    common_gate = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_hidden_diff_activation_common_gate_mild.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_activation_mode="differential-preamp",
            hidden_row_select_mode="act-common-gate",
            **common_kwargs,
        ),
        timeout=60.0,
    )

    single_hrows = [single_ended[f"initial_hrow_h{hidden}_0"] for hidden in range(mnist01_hidden.HIDDEN)]
    diff_acts = [differential[f"initial_act_h{hidden}_0"] for hidden in range(mnist01_hidden.HIDDEN)]
    diff_hrows = [differential[f"initial_hrow_h{hidden}_0"] for hidden in range(mnist01_hidden.HIDDEN)]
    common_hrows = [common_gate[f"initial_hrow_h{hidden}_0"] for hidden in range(mnist01_hidden.HIDDEN)]

    assert max(single_hrows) < 1e-3
    assert diff_acts[0] > 1.0
    assert diff_hrows[0] > 1.0
    assert max(diff_acts[1:]) < 1e-3
    assert max(diff_hrows[1:]) < 1e-3
    assert common_hrows[0] > 1.0
    assert max(common_hrows[1:]) < 1e-3


@pytest.mark.ngspice
def test_mnist01_live_hidden_identity_rows_learn_first_real_pair_without_python_weights(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_identity_rows.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            evals,
            hidden_count=16,
            hidden_init_mode="identity",
            hidden_activation_mode="differential-preamp",
            hidden_activation_sense_width_u=4.0,
        ),
        timeout=180.0,
    )

    assert abs(parsed["initial_margin_0"]) < 1e-6
    assert abs(parsed["initial_margin_1"]) < 1e-6
    assert parsed["final_margin_0"] > 1e-3
    assert parsed["final_margin_1"] > 50e-6
    assert parsed["final_margin_improvement_0"] > 1e-3
    assert parsed["final_margin_improvement_1"] > 50e-6
    for train_idx in range(2):
        assert parsed[f"train_target_signed_delta_{train_idx}"] > 3e-3
        assert parsed[f"train_other_signed_delta_{train_idx}"] < -3e-3


@pytest.mark.ngspice
def test_mnist01_live_hidden_sparse_complement_identity_rows_learn_twenty_round_robin_margins(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=10,
        eval_count_per_digit=10,
        image_size=4,
    )
    train = mnist01_fixed.add_complement_features(mnist01_fixed.round_robin_by_label(train), scale=0.5)
    evals = mnist01_fixed.add_complement_features(evals, scale=0.5)

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_sparse_complement_identity.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            evals,
            hidden_count=32,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
            hidden_activation_mode="differential-preamp",
            hidden_activation_sense_width_u=4.0,
            readout_activation_mode="pre-differential",
            readout_writer_activation_mode="pre-differential",
            readout_update_width_u=0.21,
            hidden_writer_phase_mode="hidden-write",
            hidden_write_start_train_index=999,
            measure_eval_hidden_states=False,
            tran_step_ps=10.0,
        ),
        timeout=420.0,
    )

    for sample_idx in range(20):
        assert parsed[f"final_margin_{sample_idx}"] > 0.20e-3
        assert parsed[f"final_margin_improvement_{sample_idx}"] > 0.20e-3
    for train_idx in range(20):
        assert abs(parsed[f"train_wh_probe_signed_delta_{train_idx}"]) < 1.0e-3


@pytest.mark.ngspice
def test_mnist01_live_hidden_sparse_complement_dynamic_hidden_writes_stay_bounded(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=2,
        eval_count_per_digit=2,
        image_size=4,
    )
    train = mnist01_fixed.add_complement_features(mnist01_fixed.round_robin_by_label(train), scale=0.5)
    evals = mnist01_fixed.add_complement_features(evals, scale=0.5)

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_sparse_complement_dynamic_hidden_write.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            evals,
            hidden_count=32,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
            hidden_activation_mode="differential-preamp",
            hidden_activation_sense_width_u=4.0,
            readout_activation_mode="pre-differential",
            readout_writer_activation_mode="pre-differential",
            readout_update_width_u=0.20,
            hidden_credit_gate_mode="dynamic-preamp",
            hidden_writer_topology="pmos-differential",
            hidden_write_start_train_index=0,
            hidden_credit_sense_start_ns=5.00,
            hidden_credit_sense_end_ns=5.35,
            hidden_write_start_ns=5.20,
            hidden_write_end_ns=5.30,
            hidden_update_width_u=0.05,
        ),
        timeout=240.0,
    )

    for sample_idx in range(4):
        assert parsed[f"final_margin_{sample_idx}"] > 0.25e-3
        assert parsed[f"final_margin_improvement_{sample_idx}"] > 0.25e-3
    assert max(abs(parsed[f"train_hcredit_gate_probe_{idx}"]) for idx in range(1, 4)) > 0.5
    for train_idx in range(4):
        assert abs(parsed[f"train_wh_probe_signed_delta_{train_idx}"]) < 2.0e-3


@pytest.mark.ngspice
def test_mnist01_live_hidden_sparse_complement_signcharge_packet_writes_are_bidirectional(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=2,
        eval_count_per_digit=2,
        image_size=4,
    )
    train = mnist01_fixed.add_complement_features(mnist01_fixed.round_robin_by_label(train), scale=0.5)
    evals = mnist01_fixed.add_complement_features(evals, scale=0.5)

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_sparse_complement_signcharge_packet.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            evals,
            hidden_count=32,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
            hidden_inside_positive=0.75,
            hidden_inside_negative=0.15,
            hidden_outside_positive=0.15,
            hidden_outside_negative=0.15,
            hidden_activation_mode="differential-preamp",
            hidden_activation_sense_width_u=4.0,
            readout_activation_mode="pre-differential",
            readout_writer_activation_mode="pre-differential",
            readout_update_width_u=0.20,
            hidden_credit_gate_mode="dynamic-preamp",
            hidden_error_route_width_u=8.0,
            hidden_writer_topology="pmos-signcharge",
            hidden_write_start_train_index=1,
            hidden_credit_sense_start_ns=5.00,
            hidden_credit_sense_end_ns=5.35,
            hidden_write_start_ns=5.35,
            hidden_write_end_ns=5.45,
            hidden_update_width_u=0.2,
        ),
        timeout=300.0,
    )

    for sample_idx in range(4):
        assert parsed[f"final_margin_{sample_idx}"] > 0.25e-3
        assert parsed[f"final_margin_improvement_{sample_idx}"] > 0.25e-3
    assert abs(parsed["train_wh_probe_signed_delta_0"]) < 1.0e-3
    hidden_deltas = [parsed[f"train_wh_probe_signed_delta_{idx}"] for idx in range(1, 4)]
    hidden_credit_gates = [parsed[f"train_hcredit_gate_write_probe_{idx}"] for idx in range(1, 4)]
    assert min(hidden_deltas) < -3.0e-3
    assert max(hidden_deltas) > 3.0e-3
    assert max(abs(delta) for delta in hidden_deltas) > 3.0e-3
    assert max(abs(delta) for delta in hidden_deltas) < 20.0e-3
    assert max(abs(gate) for gate in hidden_credit_gates) > 0.30

    strong_credit_pairs = [
        (gate, delta)
        for gate, delta in zip(hidden_credit_gates, hidden_deltas, strict=True)
        if abs(gate) > 0.30
    ]
    assert strong_credit_pairs
    for gate, delta in strong_credit_pairs:
        assert abs(delta) > 3.0e-3
        assert gate * delta > 0.0

    pre_deltas = [
        _hidden_feature_pre_evidence(
            parsed,
            evals,
            "final",
            sample_idx,
            hidden_count=32,
            hidden_init_mode="identity",
        )
        - _hidden_feature_pre_evidence(
            parsed,
            evals,
            "initial",
            sample_idx,
            hidden_count=32,
            hidden_init_mode="identity",
        )
        for sample_idx in range(4)
    ]
    assert max(abs(delta) for delta in pre_deltas) > 5.0e-3
    assert max(pre_deltas) > 5.0e-3
    assert min(pre_deltas) < -5.0e-3


@pytest.mark.ngspice
def test_mnist01_live_hidden_patch2x2_complement_signcharge_raises_weakest_six_margin(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=3,
        eval_count_per_digit=3,
        image_size=4,
    )
    train = mnist01_fixed.add_complement_features(mnist01_fixed.round_robin_by_label(train), scale=0.5)
    evals = mnist01_fixed.add_complement_features(evals, scale=0.5)
    hidden_count = mnist01_hidden.patch2x2_hidden_count(32)

    common_kwargs = dict(
        hidden_count=hidden_count,
        hidden_init_mode="patch2x2",
        hidden_connectivity_mode="patch2x2-sparse",
        hidden_inside_positive=0.75,
        hidden_inside_negative=0.15,
        hidden_outside_positive=0.15,
        hidden_outside_negative=0.15,
        hidden_activation_mode="differential-preamp",
        hidden_activation_sense_width_u=4.0,
        readout_activation_mode="pre-differential",
        readout_writer_activation_mode="pre-differential",
        readout_width_u=64.0,
        readout_update_width_u=0.25,
        measure_eval_hidden_states=False,
        tran_step_ps=10.0,
    )

    output_only = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_patch2x2_complement_output_only.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            evals,
            **common_kwargs,
            hidden_writer_phase_mode="hidden-write",
            hidden_write_start_train_index=999,
        ),
        timeout=360.0,
    )
    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_patch2x2_complement_signcharge.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            evals,
            **common_kwargs,
            hidden_credit_gate_mode="dynamic-preamp",
            hidden_error_route_width_u=8.0,
            hidden_writer_topology="pmos-signcharge",
            hidden_write_start_train_index=1,
            hidden_credit_sense_start_ns=5.00,
            hidden_credit_sense_end_ns=5.35,
            hidden_write_start_ns=5.35,
            hidden_write_end_ns=5.45,
            hidden_update_width_u=0.05,
        ),
        timeout=480.0,
    )

    output_margins = [output_only[f"final_margin_{sample_idx}"] for sample_idx in range(6)]
    hidden_margins = [parsed[f"final_margin_{sample_idx}"] for sample_idx in range(6)]
    assert min(output_margins) > 0.20e-3
    assert min(hidden_margins) > min(output_margins) + 0.20e-3
    assert min(hidden_margins) > 0.50e-3
    for sample_idx, margin in enumerate(hidden_margins):
        assert margin > 0.50e-3
        assert parsed[f"final_margin_improvement_{sample_idx}"] > 0.50e-3
    assert abs(parsed["train_wh_probe_signed_delta_0"]) < 1.0e-3
    hidden_deltas = [parsed[f"train_wh_probe_signed_delta_{idx}"] for idx in range(1, 6)]
    hidden_credit_gates = [parsed[f"train_hcredit_gate_write_probe_{idx}"] for idx in range(1, 6)]
    assert min(hidden_deltas) > 2.0e-3
    assert max(hidden_deltas) < 8.0e-3
    assert min(hidden_credit_gates) > 0.50
    for gate, delta in zip(hidden_credit_gates, hidden_deltas, strict=True):
        assert gate * delta > 0.0


def _hidden_activation_preamp_probe_netlist(
    *,
    pre_p: float,
    pre_n: float,
    sense_width_u: float,
) -> str:
    lines = [
        "* Hidden differential activation preamp sizing probe.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PWL(0n 1.2 0.45n 1.2 0.48n 0 3n 0)",
        "Vrstn rstn 0 PWL(0n 0 0.45n 0 0.48n 1.2 3n 1.2)",
        "Vfeatphi featphi 0 PWL(0n 0 0.75n 0 0.78n 1.2 2.10n 1.2 2.13n 0 3n 0)",
        f"Vprep pre0_p 0 {pre_p:.12g}",
        f"Vpren pre0_n 0 {pre_n:.12g}",
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            0,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=sense_width_u,
        ),
        ".meas tran sense_p FIND V(h0_act_sense_p) AT=2n",
        ".meas tran sense_n FIND V(h0_act_sense_n) AT=2n",
        ".meas tran act FIND V(act0) AT=2n",
        ".meas tran hrow FIND V(hrow0) AT=2n",
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _pre_differential_readout_probe_netlist(
    *,
    pre_p: float,
    pre_n: float,
    vwp: float = 0.75,
    vwn: float = 0.35,
    hrow: float = 1.2,
    activation_mode: str = "pre-differential",
) -> str:
    lines = [
        "* Pre-differential readout product primitive.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 0",
        "Vscorephi scorephi 0 PULSE(0 1.2 0.50n 10p 10p 1.50n 4n)",
        f"Vprep pre0_p 0 {pre_p:.12g}",
        f"Vpren pre0_n 0 {pre_n:.12g}",
        f"Vhrow hrow0 0 {hrow:.12g}",
        *mnist01_hidden._readout_storage_lines(1, vwp, vwn),
        *mnist01_hidden._score_storage_lines(),
        *mnist01_hidden._score_readout_lines(1, 16.0, activation_mode=activation_mode),
        ".meas tran scorep FIND V(c0_scorep) AT=2.40n",
        ".meas tran scoren FIND V(c0_scoren) AT=2.40n",
        ".meas tran score_net PARAM='scorep-scoren'",
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


@pytest.mark.ngspice
def test_mnist01_pre_differential_readout_tracks_signed_pre_and_weight_product(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_pre_diff_readout_positive.cir",
        _pre_differential_readout_probe_netlist(pre_p=0.55, pre_n=0.35),
        timeout=20.0,
    )
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_pre_diff_readout_negative.cir",
        _pre_differential_readout_probe_netlist(pre_p=0.35, pre_n=0.55),
        timeout=20.0,
    )
    common = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_pre_diff_readout_common.cir",
        _pre_differential_readout_probe_netlist(pre_p=0.45, pre_n=0.45),
        timeout=20.0,
    )

    assert positive["score_net"] > 20e-3
    assert negative["score_net"] < -20e-3
    assert abs(common["score_net"]) < 2e-3
    assert positive["score_net"] == pytest.approx(-negative["score_net"], rel=0.25)


@pytest.mark.ngspice
def test_mnist01_gated_pre_differential_readout_preserves_sign_and_suppresses_off_rows(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    on = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_gated_pre_diff_readout_on.cir",
        _pre_differential_readout_probe_netlist(
            pre_p=0.55,
            pre_n=0.35,
            hrow=1.2,
            activation_mode="pre-differential-gated",
        ),
        timeout=20.0,
    )
    off = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_gated_pre_diff_readout_off.cir",
        _pre_differential_readout_probe_netlist(
            pre_p=0.55,
            pre_n=0.35,
            hrow=0.0,
            activation_mode="pre-differential-gated",
        ),
        timeout=20.0,
    )
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_gated_pre_diff_readout_negative.cir",
        _pre_differential_readout_probe_netlist(
            pre_p=0.35,
            pre_n=0.55,
            hrow=1.2,
            activation_mode="pre-differential-gated",
        ),
        timeout=20.0,
    )

    assert on["score_net"] > 5e-3
    assert negative["score_net"] < -5e-3
    assert abs(off["score_net"]) < abs(on["score_net"]) * 0.20


WEAK_ZERO_PRE_P = (
    0.1770948,
    0.372905,
    0.2479296,
    0.2015259,
    0.3771313,
    0.3056752,
    0.2023425,
    0.4308137,
    0.2395549,
    0.4265476,
    0.3482745,
    0.4026968,
    0.4096666,
    0.3498117,
    0.3939206,
    0.4087584,
    0.3295131,
    0.4139395,
)
WEAK_ZERO_PRE_N = (
    0.04029579,
    0.132638,
    0.06619135,
    0.04040808,
    0.1264236,
    0.1262055,
    0.0401928,
    0.1257492,
    0.06867585,
    0.137868,
    0.1465337,
    0.1435734,
    0.1374579,
    0.1323814,
    0.1409009,
    0.1374621,
    0.1361913,
    0.1345355,
)
WEAK_ZERO_C0_VWP = (
    0.3983066,
    0.3934514,
    0.3951209,
    0.3949562,
    0.4251607,
    0.3994433,
    0.3956214,
    0.4070271,
    0.396642,
    0.409916,
    0.3939899,
    0.3952523,
    0.3987061,
    0.3934496,
    0.3953635,
    0.4003251,
    0.3988637,
    0.4067753,
)
WEAK_ZERO_C0_VWN = (
    0.3982986,
    0.3950124,
    0.3950983,
    0.3949373,
    0.3987968,
    0.3932855,
    0.3956026,
    0.3922848,
    0.3964764,
    0.420144,
    0.4150118,
    0.4163129,
    0.4175983,
    0.4103522,
    0.4129714,
    0.4182057,
    0.4131973,
    0.4162376,
)


def _measured_weak_zero_gated_readout_replay_netlist(active_hidden: set[int]) -> str:
    hidden_count = len(WEAK_ZERO_PRE_P)
    lines = [
        "* Replay measured 4x4 MNIST01 weak-zero hidden/readout state through gated readout.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 0",
        "Vrstn rstn 0 1.2",
        "Vscorephi scorephi 0 PULSE(0 1.2 0.50n 10p 10p 1.50n 4n)",
    ]
    for hidden, (pre_p, pre_n) in enumerate(zip(WEAK_ZERO_PRE_P, WEAK_ZERO_PRE_N, strict=True)):
        hrow = 1.2 if hidden in active_hidden else 0.0
        lines += [
            f"Vpre{hidden}p pre{hidden}_p 0 {pre_p:.12g}",
            f"Vpre{hidden}n pre{hidden}_n 0 {pre_n:.12g}",
            f"Vhrow{hidden} hrow{hidden} 0 {hrow:.12g}",
        ]
    for hidden, (vwp, vwn) in enumerate(zip(WEAK_ZERO_C0_VWP, WEAK_ZERO_C0_VWN, strict=True)):
        lines += [
            f"Vc0vwp{hidden} {mnist01_hidden.class_node(0, f'vwp{hidden}')} 0 {vwp:.12g}",
            f"Vc0vwn{hidden} {mnist01_hidden.class_node(0, f'vwn{hidden}')} 0 {vwn:.12g}",
            f"Vc1vwp{hidden} {mnist01_hidden.class_node(1, f'vwp{hidden}')} 0 {vwn:.12g}",
            f"Vc1vwn{hidden} {mnist01_hidden.class_node(1, f'vwn{hidden}')} 0 {vwp:.12g}",
        ]
    lines += [
        *mnist01_hidden._score_storage_lines(),
        *mnist01_hidden._score_readout_lines(
            hidden_count,
            64.0,
            activation_mode="pre-differential-gated",
        ),
        ".meas tran c0_scorep_at FIND V(c0_scorep) AT=2.40n",
        ".meas tran c0_scoren_at FIND V(c0_scoren) AT=2.40n",
        ".meas tran c1_scorep_at FIND V(c1_scorep) AT=2.40n",
        ".meas tran c1_scoren_at FIND V(c1_scoren) AT=2.40n",
        ".meas tran c0_signed_at PARAM='c0_scorep_at-c0_scoren_at'",
        ".meas tran c1_signed_at PARAM='c1_scorep_at-c1_scoren_at'",
        ".meas tran margin_at PARAM='c0_signed_at-c1_signed_at'",
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


@pytest.mark.ngspice
def test_mnist01_measured_weak_zero_replay_needs_row_competition(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    all_rows = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_weak_zero_replay_all_rows.cir",
        _measured_weak_zero_gated_readout_replay_netlist(set(range(len(WEAK_ZERO_PRE_P)))),
        timeout=30.0,
    )
    top1 = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_weak_zero_replay_top1.cir",
        _measured_weak_zero_gated_readout_replay_netlist({7}),
        timeout=30.0,
    )
    top2 = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_weak_zero_replay_top2.cir",
        _measured_weak_zero_gated_readout_replay_netlist({7, 9}),
        timeout=30.0,
    )

    assert all_rows["margin_at"] < -5e-3
    assert top1["margin_at"] > 10e-3
    assert top2["margin_at"] < 0.0


def _pre_differential_readout_writer_probe_netlist(
    *,
    pre_p: float,
    pre_n: float,
    errp: float = 0.75,
    errn: float = 0.0,
    hrow: float = 1.2,
    activation_mode: str = "pre-differential",
) -> str:
    lines = [
        "* Pre-differential readout writer primitive.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Verrphi errphi 0 PULSE(0 1.2 0.50n 10p 10p 1.50n 4n)",
        f"Vprep pre0_p 0 {pre_p:.12g}",
        f"Vpren pre0_n 0 {pre_n:.12g}",
        f"Vhrow hrow0 0 {hrow:.12g}",
        f"Verrp {mnist01_hidden.class_node(0, 'errp')} 0 {errp:.12g}",
        f"Verrn {mnist01_hidden.class_node(0, 'errn')} 0 {errn:.12g}",
        f"Verrp1 {mnist01_hidden.class_node(1, 'errp')} 0 0",
        f"Verrn1 {mnist01_hidden.class_node(1, 'errn')} 0 0",
        *mnist01_hidden._readout_storage_lines(1, 0.40, 0.40),
        *mnist01_hidden._readout_writer_lines(1, 0.25, activation_mode=activation_mode),
        ".meas tran vwp_after FIND V(c0_vwp0) AT=2.80n",
        ".meas tran vwn_after FIND V(c0_vwn0) AT=2.80n",
        ".meas tran signed_after PARAM='vwp_after-vwn_after'",
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _readout_writer_activity_mass_probe_netlist(
    *,
    active_count: int,
    normalization_mode: str,
) -> str:
    hidden_count = 10
    if not 0 <= active_count <= hidden_count:
        raise ValueError("active_count outside probe hidden_count")
    lines = [
        "* Readout writer active-feature mass normalization probe.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 0",
        "Vscorephi scorephi 0 PULSE(0 1.2 0.20n 10p 10p 1.20n 4n)",
        "Verrphi errphi 0 PULSE(0 1.2 0.50n 10p 10p 2.00n 4n)",
        f"Verrp {mnist01_hidden.class_node(0, 'errp')} 0 0.75",
        f"Verrn {mnist01_hidden.class_node(0, 'errn')} 0 0",
        f"Verrp1 {mnist01_hidden.class_node(1, 'errp')} 0 0",
        f"Verrn1 {mnist01_hidden.class_node(1, 'errn')} 0 0",
        *[
            f"Vhrow{hidden} hrow{hidden} 0 {1.2 if hidden < active_count else 0.0:.12g}"
            for hidden in range(hidden_count)
        ],
        *mnist01_hidden._readout_storage_lines(hidden_count, 0.40, 0.40),
        *mnist01_hidden._readout_writer_lines(
            hidden_count,
            0.25,
            activation_mode="hrow",
            normalization_mode=normalization_mode,
        ),
        "Vrstn rstn 0 1.2",
        ".meas tran hrow_activity_gate_at FIND V(hrow_activity_gate) AT=2.20n"
        if normalization_mode != "none"
        else ".meas tran hrow_activity_gate_at PARAM='1.2'",
        *[
            line
            for hidden in range(hidden_count)
            for line in [
                f".meas tran vwp_h{hidden} FIND V({mnist01_hidden.class_node(0, f'vwp{hidden}')}) AT=2.80n",
                f".meas tran vwn_h{hidden} FIND V({mnist01_hidden.class_node(0, f'vwn{hidden}')}) AT=2.80n",
                f".meas tran signed_h{hidden} PARAM='vwp_h{hidden}-vwn_h{hidden}'",
            ]
        ],
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


@pytest.mark.ngspice
def test_mnist01_readout_writer_activity_gate_bounds_total_active_update_mass(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    raw_five = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_readout_writer_raw_5active.cir",
        _readout_writer_activity_mass_probe_netlist(active_count=5, normalization_mode="none"),
        timeout=30.0,
    )
    raw_ten = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_readout_writer_raw_10active.cir",
        _readout_writer_activity_mass_probe_netlist(active_count=10, normalization_mode="none"),
        timeout=30.0,
    )
    norm_five = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_readout_writer_activity_norm_5active.cir",
        _readout_writer_activity_mass_probe_netlist(active_count=5, normalization_mode="activity-gate"),
        timeout=30.0,
    )
    norm_ten = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_readout_writer_activity_norm_10active.cir",
        _readout_writer_activity_mass_probe_netlist(active_count=10, normalization_mode="activity-gate"),
        timeout=30.0,
    )

    raw_total_five = sum(raw_five[f"signed_h{hidden}"] for hidden in range(5))
    raw_total_ten = sum(raw_ten[f"signed_h{hidden}"] for hidden in range(10))
    norm_total_five = sum(norm_five[f"signed_h{hidden}"] for hidden in range(5))
    norm_total_ten = sum(norm_ten[f"signed_h{hidden}"] for hidden in range(10))
    norm_per_five = norm_total_five / 5.0
    norm_per_ten = norm_total_ten / 10.0

    assert raw_total_five > 5e-3
    assert raw_total_ten / raw_total_five > 1.7
    assert 0.20 < norm_five["hrow_activity_gate_at"] < 1.15
    assert norm_ten["hrow_activity_gate_at"] < norm_five["hrow_activity_gate_at"] - 50e-3
    assert norm_per_five > 0.5e-3
    assert norm_per_ten > 0.0
    assert norm_per_ten < norm_per_five * 0.90
    assert norm_total_ten / norm_total_five < 1.75


@pytest.mark.ngspice
def test_mnist01_pre_differential_writer_uses_signed_hidden_evidence(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_pre_diff_writer_positive.cir",
        _pre_differential_readout_writer_probe_netlist(pre_p=0.55, pre_n=0.35),
        timeout=20.0,
    )
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_pre_diff_writer_negative.cir",
        _pre_differential_readout_writer_probe_netlist(pre_p=0.35, pre_n=0.55),
        timeout=20.0,
    )
    common = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_pre_diff_writer_common.cir",
        _pre_differential_readout_writer_probe_netlist(pre_p=0.45, pre_n=0.45),
        timeout=20.0,
    )

    assert positive["signed_after"] > 2e-3
    assert negative["signed_after"] < -2e-3
    assert abs(common["signed_after"]) < 1e-3


@pytest.mark.ngspice
def test_mnist01_gated_pre_differential_writer_preserves_sign_and_suppresses_off_rows(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    on = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_gated_pre_diff_writer_on.cir",
        _pre_differential_readout_writer_probe_netlist(
            pre_p=0.55,
            pre_n=0.35,
            hrow=1.2,
            activation_mode="pre-differential-gated",
        ),
        timeout=20.0,
    )
    off = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_gated_pre_diff_writer_off.cir",
        _pre_differential_readout_writer_probe_netlist(
            pre_p=0.55,
            pre_n=0.35,
            hrow=0.0,
            activation_mode="pre-differential-gated",
        ),
        timeout=20.0,
    )
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_gated_pre_diff_writer_negative.cir",
        _pre_differential_readout_writer_probe_netlist(
            pre_p=0.35,
            pre_n=0.55,
            hrow=1.2,
            activation_mode="pre-differential-gated",
        ),
        timeout=20.0,
    )

    assert on["signed_after"] > 0.5e-3
    assert negative["signed_after"] < -0.5e-3
    assert abs(off["signed_after"]) < abs(on["signed_after"]) * 0.25


@pytest.mark.ngspice
def test_mnist01_hidden_differential_activation_sizing_rejects_high_common_mode_negative_evidence(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_activation_weak_positive.cir",
        _hidden_activation_preamp_probe_netlist(pre_p=0.45, pre_n=0.32, sense_width_u=4.0),
        timeout=20.0,
    )
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_activation_weak_negative.cir",
        _hidden_activation_preamp_probe_netlist(pre_p=0.32, pre_n=0.45, sense_width_u=4.0),
        timeout=20.0,
    )

    assert positive["sense_n"] < 50e-3
    assert positive["sense_p"] > 0.50
    assert positive["act"] > 1.0
    assert positive["hrow"] > 1.0
    assert negative["sense_n"] > 0.50
    assert negative["sense_p"] < 50e-3
    assert negative["act"] < 1e-3
    assert negative["hrow"] < 1e-3


@pytest.mark.ngspice
def test_mnist01_input_feature_common_gate_tracks_above_common_pixels(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    lines = [
        "* Low-level MNIST pixel common-gate contrast primitive.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 0",
        "Vfeatphi featphi 0 PULSE(0 1.2 0.2n 10p 10p 1.4n 4n)",
        "Vpx0 px0 0 0.42",
        "Vpx1 px1 0 0.30",
        "Vpx2 px2 0 0.06",
        *mnist01_hidden._input_feature_common_gate_lines(
            3,
            common_resistance_ohm=20000.0,
            common_capacitance_f=8.0,
            gate_capacitance_f=8.0,
            contrast_capacitance_f=20.0,
            pullup_width_u=128.0,
            pulldown_width_u=24.0,
            pass_width_u=16.0,
        ),
        ".meas tran common_after FIND V(px_common) AT=2n",
        ".meas tran gate0 FIND V(pxgate0) AT=2n",
        ".meas tran gate1 FIND V(pxgate1) AT=2n",
        ".meas tran gate2 FIND V(pxgate2) AT=2n",
        ".meas tran contrast0 FIND V(px_contrast0) AT=2n",
        ".meas tran contrast1 FIND V(px_contrast1) AT=2n",
        ".meas tran contrast2 FIND V(px_contrast2) AT=2n",
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
    ]

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_input_feature_common_gate.cir",
        "\n".join(lines),
        timeout=20.0,
    )

    assert 0.20 < parsed["common_after"] < 0.30
    assert parsed["gate0"] > parsed["gate1"] + 30e-3
    assert parsed["gate1"] > parsed["gate2"] + 30e-3
    assert parsed["contrast0"] > parsed["contrast1"] + 25e-3
    assert parsed["contrast1"] > parsed["contrast2"] + 25e-3


def _hidden_input_restored_gate_forward_netlist(px0: float, px1: float) -> str:
    lines = [
        "* Restored input common-gate feeding one signed hidden template.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PWL(0n 1.2 0.45n 1.2 0.48n 0 3n 0)",
        "Vrstn rstn 0 PWL(0n 0 0.45n 0 0.48n 1.2 3n 1.2)",
        "Vfeatphi featphi 0 PWL(0n 0 0.75n 0 0.78n 1.2 2.10n 1.2 2.13n 0 3n 0)",
        f"Vpx0 px0 0 {px0:.12g}",
        f"Vpx1 px1 0 {px1:.12g}",
        "Cwh0f0p wh0f0p 0 20f IC=0.75",
        "Rwh0f0p wh0f0p 0 1e15",
        "Cwh0f0n wh0f0n 0 20f IC=0.35",
        "Rwh0f0n wh0f0n 0 1e15",
        "Cwh0f1p wh0f1p 0 20f IC=0.35",
        "Rwh0f1p wh0f1p 0 1e15",
        "Cwh0f1n wh0f1n 0 20f IC=0.75",
        "Rwh0f1n wh0f1n 0 1e15",
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            2,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=64.0,
            input_mode="restored-common-gate",
        ),
        ".meas tran gate0 FIND V(pxgate0) AT=2n",
        ".meas tran gate1 FIND V(pxgate1) AT=2n",
        ".meas tran drive0 FIND V(pxdrive0) AT=2n",
        ".meas tran drive1 FIND V(pxdrive1) AT=2n",
        ".meas tran prep FIND V(pre0_p) AT=2n",
        ".meas tran pren FIND V(pre0_n) AT=2n",
        ".meas tran evidence PARAM='prep-pren'",
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


@pytest.mark.ngspice
def test_mnist01_hidden_input_restored_common_gate_preserves_signed_forward_template(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    match = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_hidden_input_restored_match.cir",
        _hidden_input_restored_gate_forward_netlist(0.45, 0.05),
        timeout=30.0,
    )
    mismatch = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_hidden_input_restored_mismatch.cir",
        _hidden_input_restored_gate_forward_netlist(0.05, 0.45),
        timeout=30.0,
    )
    flat = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_hidden_input_restored_flat.cir",
        _hidden_input_restored_gate_forward_netlist(0.35, 0.35),
        timeout=30.0,
    )

    assert match["drive0"] > 1.0
    assert match["drive1"] < 1e-3
    assert match["evidence"] > 100e-3
    assert mismatch["drive1"] > 1.0
    assert mismatch["drive0"] < 1e-3
    assert mismatch["evidence"] < -100e-3
    assert abs(flat["evidence"]) < 1e-3


@pytest.mark.ngspice
def test_mnist01_hidden_activation_common_gate_selects_above_common_activity(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    lines = [
        "* Low-level MNIST live-hidden row-select common-gate primitive.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 0",
        "Vfeatphi featphi 0 PULSE(0 1.2 0.2n 10p 10p 1.4n 4n)",
        "Vact0 act0 0 0.42",
        "Vact1 act1 0 0.30",
        "Vact2 act2 0 0.06",
        *mnist01_hidden._hidden_activation_common_gate_lines(
            3,
            common_resistance_ohm=100000.0,
            gate_capacitance_f=8.0,
            contrast_capacitance_f=20.0,
            pullup_width_u=128.0,
            pulldown_width_u=24.0,
            pass_width_u=16.0,
        ),
        ".meas tran common_after FIND V(hidden_act_common) AT=2n",
        ".meas tran gate0 FIND V(hactgate0) AT=2n",
        ".meas tran gate1 FIND V(hactgate1) AT=2n",
        ".meas tran gate2 FIND V(hactgate2) AT=2n",
        ".meas tran contrast0 FIND V(act_contrast0) AT=2n",
        ".meas tran contrast1 FIND V(act_contrast1) AT=2n",
        ".meas tran contrast2 FIND V(act_contrast2) AT=2n",
        ".tran 2p 3n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
    ]

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_hidden_activation_common_gate.cir",
        "\n".join(lines),
        timeout=20.0,
    )

    assert 0.20 < parsed["common_after"] < 0.32
    assert parsed["gate0"] > parsed["gate1"] + 40e-3
    assert parsed["gate1"] > parsed["gate2"] + 40e-3
    assert parsed["contrast0"] > parsed["contrast1"] + 30e-3
    assert parsed["contrast1"] > parsed["contrast2"] + 30e-3


def _hidden_forward_update_primitive_netlist() -> str:
    lines = [
        "* One hidden conductance cell: bounded write improves next forward evidence.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PWL(0n 1.2 0.45n 1.2 0.48n 0 4.97n 0 5.00n 1.2 5.45n 1.2 5.48n 0 8n 0)",
        "Vrstn rstn 0 PWL(0n 0 0.45n 0 0.48n 1.2 4.97n 1.2 5.00n 0 5.45n 0 5.48n 1.2 8n 1.2)",
        "Vfeatphi featphi 0 PWL(0n 0 0.75n 0 0.78n 1.2 2.10n 1.2 2.13n 0 5.75n 0 5.78n 1.2 7.10n 1.2 7.13n 0 8n 0)",
        "Verrphi errphi 0 PWL(0n 0 3.00n 0 3.03n 1.2 3.10n 1.2 3.13n 0 8n 0)",
        "Vpx0 px0 0 1.2",
        "Vhdp h0_hdp_gate 0 0.8",
        "Vhdn h0_hdn_gate 0 0",
        "Cwh0f0p wh0f0p 0 20f IC=0.50",
        "Rwh0f0p wh0f0p 0 1e15",
        "Cwh0f0n wh0f0n 0 20f IC=0.45",
        "Rwh0f0n wh0f0n 0 1e15",
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            1,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=64.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            1,
            1,
            0.005,
            0.1,
            0.2,
            "pmos-differential",
            "errphi",
            True,
        ),
        ".meas tran prep_before FIND V(pre0_p) AT=1.95n",
        ".meas tran pren_before FIND V(pre0_n) AT=1.95n",
        ".meas tran evidence_before PARAM='prep_before-pren_before'",
        ".meas tran act_before FIND V(act0) AT=1.95n",
        ".meas tran hrow_before FIND V(hrow0) AT=1.95n",
        ".meas tran whp_before FIND V(wh0f0p) AT=2.50n",
        ".meas tran whn_before FIND V(wh0f0n) AT=2.50n",
        ".meas tran whp_after_write FIND V(wh0f0p) AT=4.80n",
        ".meas tran whn_after_write FIND V(wh0f0n) AT=4.80n",
        ".meas tran signed_delta PARAM='(whp_after_write-whn_after_write)-(whp_before-whn_before)'",
        ".meas tran prep_after FIND V(pre0_p) AT=6.95n",
        ".meas tran pren_after FIND V(pre0_n) AT=6.95n",
        ".meas tran evidence_after PARAM='prep_after-pren_after'",
        ".meas tran act_after FIND V(act0) AT=6.95n",
        ".meas tran hrow_after FIND V(hrow0) AT=6.95n",
        ".meas tran evidence_delta PARAM='evidence_after-evidence_before'",
        ".tran 1p 8n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


@pytest.mark.ngspice
def test_mnist01_hidden_bounded_write_improves_next_forward_evidence(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_hidden_bounded_write_forward_evidence.cir",
        _hidden_forward_update_primitive_netlist(),
        timeout=30.0,
    )

    assert 0.02 < parsed["signed_delta"] < 0.06
    assert 0.03 < parsed["evidence_delta"] < 0.06
    assert parsed["whp_after_write"] < 0.60
    assert parsed["whn_after_write"] > 0.40
    assert parsed["act_after"] >= parsed["act_before"]
    assert parsed["hrow_after"] >= parsed["hrow_before"] - 1e-6


@pytest.mark.ngspice
def test_mnist01_live_hidden_bounded_real_credit_improves_selected_pre_evidence(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, _evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_bounded_real_credit.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            train,
            hidden_inside_positive=0.50,
            hidden_outside_positive=0.45,
            hidden_inside_negative=0.45,
            hidden_outside_negative=0.65,
            hidden_activation_mode="differential-preamp",
            hidden_activation_sense_width_u=64.0,
            hidden_credit_gate_mode="dynamic-preamp",
            hidden_writer_topology="pmos-differential",
            hidden_write_start_train_index=1,
            hidden_credit_sense_start_ns=5.00,
            hidden_credit_sense_end_ns=5.35,
            hidden_write_start_ns=5.20,
            hidden_write_end_ns=5.70,
            hidden_update_width_u=0.005,
            hidden_writer_pmos_width_u=0.1,
        ),
        timeout=160.0,
    )

    initial_evidence = _hidden_feature_pre_evidence(parsed, train, "initial", 1)
    final_evidence = _hidden_feature_pre_evidence(parsed, train, "final", 1)

    assert parsed["train_hcredit_gate_probe_1"] < -0.50
    assert 20e-3 < parsed["train_wh_probe_signed_delta_1"] < 60e-3
    assert parsed["train_wh_probe_p_after_1"] < 0.60
    assert parsed["train_wh_probe_n_after_1"] > 0.40
    assert final_evidence > initial_evidence + 3e-3
    assert parsed["final_margin_0"] < 0.0
    assert parsed["final_margin_1"] > 0.0


@pytest.mark.ngspice
def test_mnist01_live_hidden_divider_ngspice_bootstraps_readout_and_visible_subthreshold_hidden_credit(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, _evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_divider.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(train, train),
        timeout=120.0,
    )

    assert abs(parsed["initial_margin_0"]) < 1e-6
    assert abs(parsed["initial_margin_1"]) < 1e-6
    assert parsed["final_margin_0"] > 0.10e-3
    assert parsed["final_margin_1"] > 1e-6
    assert parsed["final_margin_improvement_0"] > 0.10e-3
    assert parsed["final_margin_improvement_1"] > 1e-6

    for train_idx in range(2):
        ir_sum = abs(parsed[f"train_ir0_{train_idx}"]) + abs(parsed[f"train_ir1_{train_idx}"])
        assert ir_sum == pytest.approx(1.0e-6, rel=0.08)
        assert parsed[f"train_act_probe_{train_idx}"] > 40e-3
        assert parsed[f"train_hrow_probe_{train_idx}"] > 1.0
        assert parsed[f"train_hrow_ctrl_probe_{train_idx}"] < 10e-3
        assert parsed[f"train_target_errp_{train_idx}"] > parsed[f"train_target_errn_{train_idx}"] + 30e-3
        assert parsed[f"train_other_errn_{train_idx}"] > parsed[f"train_other_errp_{train_idx}"] + 30e-3
        assert parsed[f"train_target_herrp_{train_idx}"] > parsed[f"train_target_errp_{train_idx}"] + 0.20
        assert parsed[f"train_other_herrn_{train_idx}"] > parsed[f"train_other_errn_{train_idx}"] + 0.20
        assert parsed[f"train_target_signed_delta_{train_idx}"] > 3e-3
        assert parsed[f"train_other_signed_delta_{train_idx}"] < -3e-3
        assert abs(parsed[f"train_wh_probe_signed_delta_{train_idx}"]) < 1e-3

    assert abs(parsed["train_hcredit_gate_probe_0"]) < 1e-6
    assert parsed["train_hcredit_gate_probe_1"] < -50e-6
    assert abs(parsed["train_hcredit_gate_probe_1"]) < 1e-3


@pytest.mark.ngspice
def test_mnist01_live_hidden_dynamic_preamp_restores_second_sample_credit_after_bootstrap(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, _evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_dynamic_preamp.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            train,
            hidden_credit_gate_mode="dynamic-preamp",
            hidden_writer_topology="pmos-differential",
            hidden_write_start_train_index=1,
        ),
        timeout=120.0,
    )

    assert parsed["final_margin_0"] > 0.10e-3
    assert parsed["final_margin_1"] > 1e-6
    assert parsed["train_target_signed_delta_0"] > 3e-3
    assert parsed["train_other_signed_delta_0"] < -3e-3
    assert parsed["train_target_signed_delta_1"] > 3e-3
    assert parsed["train_other_signed_delta_1"] < -3e-3
    assert abs(parsed["train_wh_probe_signed_delta_0"]) < 1e-3
    assert parsed["train_hcredit_gate_probe_1"] < -0.50
    assert parsed["train_wh_probe_signed_delta_1"] < -50e-3


@pytest.mark.ngspice
def test_mnist01_live_hidden_dynamic_preamp_short_write_bounds_multisample_hidden_update(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=2,
        eval_count_per_digit=2,
        image_size=4,
    )

    parsed = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_live_hidden_dynamic_preamp_bounded.cir",
        mnist01_hidden.mnist01_live_hidden_netlist(
            train,
            evals,
            hidden_credit_gate_mode="dynamic-preamp",
            hidden_writer_topology="pmos-differential",
            hidden_write_start_train_index=1,
            hidden_credit_sense_start_ns=5.00,
            hidden_credit_sense_end_ns=5.35,
            hidden_write_start_ns=5.20,
            hidden_write_end_ns=5.30,
            hidden_update_width_u=0.05,
        ),
        timeout=180.0,
    )

    assert parsed["final_margin_0"] > 10e-3
    assert parsed["final_margin_1"] > 10e-3
    assert parsed["final_margin_2"] < 0.0
    assert parsed["final_margin_3"] < 0.0
    for train_idx in range(4):
        assert abs(parsed[f"train_wh_probe_signed_delta_{train_idx}"]) < 10e-3

    for eval_idx in range(4):
        initial_hrow_margin = _hidden_feature_margin(parsed, evals, "initial", eval_idx, node="hrow")
        final_hrow_margin = _hidden_feature_margin(parsed, evals, "final", eval_idx, node="hrow")
        initial_act_margin = _hidden_feature_margin(parsed, evals, "initial", eval_idx, node="act")
        final_act_margin = _hidden_feature_margin(parsed, evals, "final", eval_idx, node="act")
        assert final_hrow_margin >= initial_hrow_margin - 20e-3
        assert final_act_margin >= initial_act_margin - 20e-3


def _hidden_credit_preamp_primitive_netlist(raw_positive: float, raw_negative: float) -> str:
    lines = [
        "* MNIST-scale hidden-credit dynamic preamp primitive.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(1.2 0 0.4n 10p 10p 9n 20n)",
        "Vrstn rstn 0 PULSE(0 1.2 0.4n 10p 10p 9n 20n)",
        "Vhcgphi hcgphi 0 PULSE(0 1.2 1.0n 10p 10p 2.0n 20n)",
        "Vhiddenwritephi hiddenwritephi 0 PULSE(0 1.2 3.3n 10p 10p 2.0n 20n)",
        "Vpx0 px0 0 PULSE(0 1.2 3.3n 10p 10p 2.0n 20n)",
        f"Vhdp h0_hdp 0 {raw_positive:.12g}",
        f"Vhdn h0_hdn 0 {raw_negative:.12g}",
        "Cwh0f0p wh0f0p 0 20f IC=0.45",
        "Cwh0f0n wh0f0n 0 20f IC=0.40",
        "Rwhp wh0f0p 0 1e15",
        "Rwhn wh0f0n 0 1e15",
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            1,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            1,
            1,
            1.0,
            4.0,
            0.2,
            "pmos-differential",
            "h{hidden}_hcg_write",
            True,
        ),
        ".meas tran gatep_pre FIND V(h0_hdp_gate) AT=3.10n",
        ".meas tran gaten_pre FIND V(h0_hdn_gate) AT=3.10n",
        ".meas tran support_pre FIND V(h0_hcg_support) AT=2.50n",
        ".meas tran write_gate FIND V(h0_hcg_write) AT=4.50n",
        ".meas tran gatep FIND V(h0_hdp_gate) AT=5.50n",
        ".meas tran gaten FIND V(h0_hdn_gate) AT=5.50n",
        ".meas tran gate_diff PARAM='gatep-gaten'",
        ".meas tran whp_after FIND V(wh0f0p) AT=7.00n",
        ".meas tran whn_after FIND V(wh0f0n) AT=7.00n",
        ".meas tran signed_after PARAM='whp_after-whn_after'",
        ".meas tran signed_delta PARAM='signed_after-0.05'",
        ".tran 1p 8n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_credit_signcharge_primitive_netlist(raw_positive: float, raw_negative: float) -> str:
    lines = [
        "* MNIST-scale hidden-credit signcharge writer primitive.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(1.2 0 0.4n 10p 10p 9n 20n)",
        "Vrstn rstn 0 PULSE(0 1.2 0.4n 10p 10p 9n 20n)",
        "Vhcgphi hcgphi 0 PULSE(0 1.2 1.0n 10p 10p 2.0n 20n)",
        "Vhiddenwritephi hiddenwritephi 0 PWL(0n 0 3.27n 0 3.30n 1.2 3.80n 1.2 3.83n 0 8n 0)",
        "Vpx0 px0 0 PULSE(0 1.2 3.3n 10p 10p 2.0n 20n)",
        f"Vhdp h0_hdp 0 {raw_positive:.12g}",
        f"Vhdn h0_hdn 0 {raw_negative:.12g}",
        "Cwh0f0p wh0f0p 0 20f IC=0.45",
        "Cwh0f0n wh0f0n 0 20f IC=0.40",
        "Rwhp wh0f0p 0 1e15",
        "Rwhn wh0f0n 0 1e15",
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            1,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            1,
            1,
            1.0,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
        ),
        ".meas tran gatep FIND V(h0_hdp_gate) AT=3.20n",
        ".meas tran gaten FIND V(h0_hdn_gate) AT=3.20n",
        ".meas tran gate_diff PARAM='gatep-gaten'",
        ".meas tran write_gate FIND V(h0_hcg_write) AT=3.60n",
        ".meas tran whp_after FIND V(wh0f0p) AT=7.00n",
        ".meas tran whn_after FIND V(wh0f0n) AT=7.00n",
        ".meas tran signed_after PARAM='whp_after-whn_after'",
        ".meas tran signed_delta PARAM='signed_after-0.05'",
        ".tran 1p 8n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_signcharge_packet_writer_netlist(
    *,
    positive_gate: float,
    negative_gate: float,
    whp_ic: float,
    whn_ic: float,
) -> str:
    lines = [
        "* Integrated-like signcharge packet writer primitive.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(1.2 0 0.4n 10p 10p 9n 20n)",
        "Vrstn rstn 0 PULSE(0 1.2 0.4n 10p 10p 9n 20n)",
        "Vhcgwrite h0_hcg_write 0 PWL(0n 0 1.27n 0 1.30n 1.2 1.40n 1.2 1.43n 0 5n 0)",
        "Vpx0 px0 0 0.78",
        f"Vhdpg h0_hdp_gate 0 {positive_gate:.12g}",
        f"Vhdng h0_hdn_gate 0 {negative_gate:.12g}",
        f"Cwh0f0p wh0f0p 0 20f IC={whp_ic:.12g}",
        f"Cwh0f0n wh0f0n 0 20f IC={whn_ic:.12g}",
        "Rwhp wh0f0p 0 1e15",
        "Rwhn wh0f0n 0 1e15",
        *mnist01_hidden._hidden_writer_lines(
            1,
            1,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
        ),
        ".meas tran whp_after FIND V(wh0f0p) AT=4.00n",
        ".meas tran whn_after FIND V(wh0f0n) AT=4.00n",
        f".meas tran signed_delta PARAM='whp_after-whn_after-({whp_ic:.12g}-{whn_ic:.12g})'",
        ".tran 1p 5n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_signcharge_forward_after_write_netlist(
    *,
    positive_gate: float,
    negative_gate: float,
    whp_ic: float,
    whn_ic: float,
) -> str:
    lines = [
        "* Hidden signcharge packet writer followed by a second forward read.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PWL(0n 1.2 0.45n 1.2 0.48n 0 2.00n 0 2.03n 1.2 2.45n 1.2 2.48n 0 5n 0)",
        "Vrstn rstn 0 PWL(0n 0 0.45n 0 0.48n 1.2 2.00n 1.2 2.03n 0 2.45n 0 2.48n 1.2 5n 1.2)",
        "Vfeatphi featphi 0 PWL(0n 0 0.75n 0 0.78n 1.2 1.25n 1.2 1.28n 0 2.75n 0 2.78n 1.2 3.25n 1.2 3.28n 0 5n 0)",
        "Vhcgwrite h0_hcg_write 0 PWL(0n 0 1.52n 0 1.55n 1.2 1.65n 1.2 1.68n 0 5n 0)",
        "Vpx0 px0 0 0.78",
        f"Vhdpg h0_hdp_gate 0 {positive_gate:.12g}",
        f"Vhdng h0_hdn_gate 0 {negative_gate:.12g}",
        f"Cwh0f0p wh0f0p 0 20f IC={whp_ic:.12g}",
        f"Cwh0f0n wh0f0n 0 20f IC={whn_ic:.12g}",
        "Rwhp wh0f0p 0 1e15",
        "Rwhn wh0f0n 0 1e15",
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            1,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=4.0,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
        ),
        *mnist01_hidden._hidden_writer_lines(
            1,
            1,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
        ),
        ".meas tran pre_before_p FIND V(pre0_p) AT=1.25n",
        ".meas tran pre_before_n FIND V(pre0_n) AT=1.25n",
        ".meas tran pre_before PARAM='pre_before_p-pre_before_n'",
        ".meas tran pre_after_p FIND V(pre0_p) AT=3.25n",
        ".meas tran pre_after_n FIND V(pre0_n) AT=3.25n",
        ".meas tran pre_after PARAM='pre_after_p-pre_after_n'",
        ".meas tran pre_delta PARAM='pre_after-pre_before'",
        ".meas tran whp_after FIND V(wh0f0p) AT=3.50n",
        ".meas tran whn_after FIND V(wh0f0n) AT=3.50n",
        f".meas tran wh_delta PARAM='whp_after-whn_after-({whp_ic:.12g}-{whn_ic:.12g})'",
        ".tran 1p 5n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_credit_signcharge_forward_after_write_netlist(error_mode: str) -> str:
    if error_mode not in {"positive", "negative"}:
        raise ValueError("error_mode must be positive or negative")
    errp = 0.55 if error_mode == "positive" else 0.0
    errn = 0.55 if error_mode == "negative" else 0.0
    lines = [
        "* Readout-weighted hidden credit followed by signcharge packet write and second forward.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PWL(0n 1.2 0.45n 1.2 0.48n 0 3.00n 0 3.03n 1.2 3.45n 1.2 3.48n 0 6n 0)",
        "Vrstn rstn 0 PWL(0n 0 0.45n 0 0.48n 1.2 3.00n 1.2 3.03n 0 3.45n 0 3.48n 1.2 6n 1.2)",
        "Vfeatphi featphi 0 PWL(0n 0 0.75n 0 0.78n 1.2 1.25n 1.2 1.28n 0 3.75n 0 3.78n 1.2 4.25n 1.2 4.28n 0 6n 0)",
        "Vhcgphi hcgphi 0 PWL(0n 0 1.45n 0 1.48n 1.2 2.25n 1.2 2.28n 0 6n 0)",
        "Vhiddenwritephi hiddenwritephi 0 PWL(0n 0 2.15n 0 2.18n 1.2 2.28n 1.2 2.31n 0 6n 0)",
        "Vpx0 px0 0 0.78",
        f"Vc0herrp c0_herrp 0 {errp:.12g}",
        f"Vc0herrn c0_herrn 0 {errn:.12g}",
        "Vc1herrp c1_herrp 0 0",
        "Vc1herrn c1_herrn 0 0",
        "Cwh0f0p wh0f0p 0 20f IC=0.75",
        "Cwh0f0n wh0f0n 0 20f IC=0.15",
        "Rwhp wh0f0p 0 1e15",
        "Rwhn wh0f0n 0 1e15",
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(0, "vwp0"),
            negative_node=mnist01_hidden.class_node(0, "vwn0"),
            positive_ic=0.46,
            negative_ic=0.24,
        ),
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(1, "vwp0"),
            negative_node=mnist01_hidden.class_node(1, "vwn0"),
            positive_ic=0.40,
            negative_ic=0.40,
        ),
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            1,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=4.0,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
        ),
        *mnist01_hidden._hidden_credit_lines(1, 32.0, 12.0, 0.05, 5.0e7),
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            1,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            1,
            1,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
        ),
        ".meas tran pre_before_p FIND V(pre0_p) AT=1.25n",
        ".meas tran pre_before_n FIND V(pre0_n) AT=1.25n",
        ".meas tran pre_before PARAM='pre_before_p-pre_before_n'",
        ".meas tran hdp FIND V(h0_hdp) AT=2.10n",
        ".meas tran hdn FIND V(h0_hdn) AT=2.10n",
        ".meas tran hcredit PARAM='hdp-hdn'",
        ".meas tran gatep FIND V(h0_hdp_gate) AT=2.20n",
        ".meas tran gaten FIND V(h0_hdn_gate) AT=2.20n",
        ".meas tran gate_diff PARAM='gatep-gaten'",
        ".meas tran pre_after_p FIND V(pre0_p) AT=4.25n",
        ".meas tran pre_after_n FIND V(pre0_n) AT=4.25n",
        ".meas tran pre_after PARAM='pre_after_p-pre_after_n'",
        ".meas tran pre_delta PARAM='pre_after-pre_before'",
        ".meas tran whp_after FIND V(wh0f0p) AT=4.50n",
        ".meas tran whn_after FIND V(wh0f0n) AT=4.50n",
        ".meas tran wh_delta PARAM='whp_after-whn_after-(0.75-0.15)'",
        ".tran 1p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_credit_signcharge_margin_after_write_netlist(target_class: int) -> str:
    if target_class not in (0, 1):
        raise ValueError("target_class must be 0 or 1")
    other_class = 1 - target_class
    herr = {
        (0, "p"): 0.55 if target_class == 0 else 0.0,
        (0, "n"): 0.55 if target_class == 1 else 0.0,
        (1, "p"): 0.55 if target_class == 1 else 0.0,
        (1, "n"): 0.55 if target_class == 0 else 0.0,
    }
    lines = [
        "* Hidden-credit signcharge write followed by downstream score-margin replay.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PWL(0n 1.2 0.45n 1.2 0.48n 0 3.00n 0 3.03n 1.2 3.45n 1.2 3.48n 0 6n 0)",
        "Vrstn rstn 0 PWL(0n 0 0.45n 0 0.48n 1.2 3.00n 1.2 3.03n 0 3.45n 0 3.48n 1.2 6n 1.2)",
        "Vfeatphi featphi 0 PWL(0n 0 0.75n 0 0.78n 1.2 1.25n 1.2 1.28n 0 3.75n 0 3.78n 1.2 4.25n 1.2 4.28n 0 6n 0)",
        "Vscorephi scorephi 0 PWL(0n 0 1.30n 0 1.33n 1.2 1.65n 1.2 1.68n 0 4.30n 0 4.33n 1.2 4.65n 1.2 4.68n 0 6n 0)",
        "Vhcgphi hcgphi 0 PWL(0n 0 1.65n 0 1.68n 1.2 2.25n 1.2 2.28n 0 6n 0)",
        "Vhiddenwritephi hiddenwritephi 0 PWL(0n 0 2.15n 0 2.18n 1.2 2.28n 1.2 2.31n 0 6n 0)",
        "Vpx0 px0 0 0.78",
        f"Vc0herrp c0_herrp 0 {herr[(0, 'p')]:.12g}",
        f"Vc0herrn c0_herrn 0 {herr[(0, 'n')]:.12g}",
        f"Vc1herrp c1_herrp 0 {herr[(1, 'p')]:.12g}",
        f"Vc1herrn c1_herrn 0 {herr[(1, 'n')]:.12g}",
        "Cwh0f0p wh0f0p 0 20f IC=0.50",
        "Cwh0f0n wh0f0n 0 20f IC=0.45",
        "Rwhp wh0f0p 0 1e15",
        "Rwhn wh0f0n 0 1e15",
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(0, "vwp0"),
            negative_node=mnist01_hidden.class_node(0, "vwn0"),
            positive_ic=0.46,
            negative_ic=0.24,
        ),
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(1, "vwp0"),
            negative_node=mnist01_hidden.class_node(1, "vwn0"),
            positive_ic=0.24,
            negative_ic=0.46,
        ),
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            1,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=4.0,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
        ),
        *[
            line
            for output in range(2)
            for kind in ("scorep", "scoren")
            for node in [mnist01_hidden.class_node(output, kind)]
            for line in [
                f"C{node} {node} 0 8f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rst 0 0 NMOS W=4u L=180n",
            ]
        ],
        *mnist01_hidden._score_readout_lines(1, 16.0, activation_mode="pre-differential"),
        *mnist01_hidden._hidden_credit_lines(1, 32.0, 12.0, 0.05, 5.0e7),
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            1,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            1,
            1,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
        ),
        ".meas tran pre_before_p FIND V(pre0_p) AT=1.25n",
        ".meas tran pre_before_n FIND V(pre0_n) AT=1.25n",
        ".meas tran pre_before PARAM='pre_before_p-pre_before_n'",
        ".meas tran c0_scorep_before FIND V(c0_scorep) AT=1.65n",
        ".meas tran c0_scoren_before FIND V(c0_scoren) AT=1.65n",
        ".meas tran c1_scorep_before FIND V(c1_scorep) AT=1.65n",
        ".meas tran c1_scoren_before FIND V(c1_scoren) AT=1.65n",
        ".meas tran c0_signed_before PARAM='c0_scorep_before-c0_scoren_before'",
        ".meas tran c1_signed_before PARAM='c1_scorep_before-c1_scoren_before'",
        ".meas tran hdp FIND V(h0_hdp) AT=2.10n",
        ".meas tran hdn FIND V(h0_hdn) AT=2.10n",
        ".meas tran hcredit PARAM='hdp-hdn'",
        ".meas tran gatep FIND V(h0_hdp_gate) AT=2.20n",
        ".meas tran gaten FIND V(h0_hdn_gate) AT=2.20n",
        ".meas tran gate_diff PARAM='gatep-gaten'",
        ".meas tran pre_after_p FIND V(pre0_p) AT=4.25n",
        ".meas tran pre_after_n FIND V(pre0_n) AT=4.25n",
        ".meas tran pre_after PARAM='pre_after_p-pre_after_n'",
        ".meas tran pre_delta PARAM='pre_after-pre_before'",
        ".meas tran c0_scorep_after FIND V(c0_scorep) AT=4.65n",
        ".meas tran c0_scoren_after FIND V(c0_scoren) AT=4.65n",
        ".meas tran c1_scorep_after FIND V(c1_scorep) AT=4.65n",
        ".meas tran c1_scoren_after FIND V(c1_scoren) AT=4.65n",
        ".meas tran c0_signed_after PARAM='c0_scorep_after-c0_scoren_after'",
        ".meas tran c1_signed_after PARAM='c1_scorep_after-c1_scoren_after'",
        f".meas tran target_margin_before PARAM='c{target_class}_signed_before-c{other_class}_signed_before'",
        f".meas tran target_margin_after PARAM='c{target_class}_signed_after-c{other_class}_signed_after'",
        ".meas tran target_margin_delta PARAM='target_margin_after-target_margin_before'",
        ".meas tran whp_after FIND V(wh0f0p) AT=4.80n",
        ".meas tran whn_after FIND V(wh0f0n) AT=4.80n",
        ".meas tran wh_delta PARAM='whp_after-whn_after-(0.50-0.45)'",
        ".tran 1p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _hidden_divider_signcharge_margin_rescue_netlist(target_class: int) -> str:
    if target_class not in (0, 1):
        raise ValueError("target_class must be 0 or 1")
    other_class = 1 - target_class
    target0 = 1.2 if target_class == 0 else 0.0
    target1 = 1.2 if target_class == 1 else 0.0
    lines = [
        "* Conductance-divider hidden-margin rescue: score -> normalized hidden error -> hidden write -> score replay.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PWL(0n 1.2 0.45n 1.2 0.48n 0 3.00n 0 3.03n 1.2 3.45n 1.2 3.48n 0 6n 0)",
        "Vrstn rstn 0 PWL(0n 0 0.45n 0 0.48n 1.2 3.00n 1.2 3.03n 0 3.45n 0 3.48n 1.2 6n 1.2)",
        "Vfeatphi featphi 0 PWL(0n 0 0.75n 0 0.78n 1.2 1.25n 1.2 1.28n 0 3.75n 0 3.78n 1.2 4.25n 1.2 4.28n 0 6n 0)",
        "Vscorephi scorephi 0 PWL(0n 0 1.30n 0 1.33n 1.2 1.65n 1.2 1.68n 0 4.30n 0 4.33n 1.2 4.65n 1.2 4.68n 0 6n 0)",
        "Verrphi errphi 0 PWL(0n 0 1.68n 0 1.71n 1.2 2.25n 1.2 2.28n 0 6n 0)",
        "Vhcgphi hcgphi 0 PWL(0n 0 2.25n 0 2.28n 1.2 2.75n 1.2 2.78n 0 6n 0)",
        "Vhiddenwritephi hiddenwritephi 0 PWL(0n 0 2.65n 0 2.68n 1.2 2.78n 1.2 2.81n 0 6n 0)",
        "Iprobref vdd rnorm PWL(0n 0 1.68n 0 1.71n 10u 2.25n 10u 2.28n 0 6n 0)",
        "Vpx0 px0 0 1.2",
        f"Vt0 t0 0 {target0:.12g}",
        f"Vt1 t1 0 {target1:.12g}",
        "Cwh0f0p wh0f0p 0 20f IC=0.75",
        "Cwh0f0n wh0f0n 0 20f IC=0.15",
        "Rwhp wh0f0p 0 1e15",
        "Rwhn wh0f0n 0 1e15",
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(0, "vwp0"),
            negative_node=mnist01_hidden.class_node(0, "vwn0"),
            positive_ic=0.90,
            negative_ic=0.10,
        ),
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(1, "vwp0"),
            negative_node=mnist01_hidden.class_node(1, "vwn0"),
            positive_ic=0.10,
            negative_ic=0.90,
        ),
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            1,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=4.0,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
        ),
        *mnist01_hidden._score_storage_lines(),
        *mnist01_hidden._score_readout_lines(1, 64.0, activation_mode="pre-differential"),
        *mnist01_hidden._error_storage_lines(),
        *mnist01_hidden._divider_probability_lines(0.5, 0.02),
        *mnist01_hidden._route_to_hidden_error_rails_lines(16.0),
        *mnist01_hidden._hidden_credit_lines(1, 32.0, 12.0, 0.05, 5.0e7),
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            1,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            1,
            1,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
            hidden_init_mode="identity",
            hidden_connectivity_mode="identity-sparse",
        ),
        ".meas tran pre_before_p FIND V(pre0_p) AT=1.25n",
        ".meas tran pre_before_n FIND V(pre0_n) AT=1.25n",
        ".meas tran pre_before PARAM='pre_before_p-pre_before_n'",
        ".meas tran c0_scorep_before FIND V(c0_scorep) AT=1.65n",
        ".meas tran c0_scoren_before FIND V(c0_scoren) AT=1.65n",
        ".meas tran c1_scorep_before FIND V(c1_scorep) AT=1.65n",
        ".meas tran c1_scoren_before FIND V(c1_scoren) AT=1.65n",
        ".meas tran c0_signed_before PARAM='c0_scorep_before-c0_scoren_before'",
        ".meas tran c1_signed_before PARAM='c1_scorep_before-c1_scoren_before'",
        ".meas tran b0low_err FIND V(b0low) AT=2.10n",
        ".meas tran b1low_err FIND V(b1low) AT=2.10n",
        ".meas tran c0_herrp_probe FIND V(c0_herrp) AT=2.20n",
        ".meas tran c0_herrn_probe FIND V(c0_herrn) AT=2.20n",
        ".meas tran c1_herrp_probe FIND V(c1_herrp) AT=2.20n",
        ".meas tran c1_herrn_probe FIND V(c1_herrn) AT=2.20n",
        ".meas tran target_herrdiff PARAM='c{0}_herrp_probe-c{0}_herrn_probe'".format(target_class),
        ".meas tran other_herrdiff PARAM='c{0}_herrp_probe-c{0}_herrn_probe'".format(other_class),
        ".meas tran hdp FIND V(h0_hdp) AT=2.55n",
        ".meas tran hdn FIND V(h0_hdn) AT=2.55n",
        ".meas tran hcredit PARAM='hdp-hdn'",
        ".meas tran gatep FIND V(h0_hdp_gate) AT=2.70n",
        ".meas tran gaten FIND V(h0_hdn_gate) AT=2.70n",
        ".meas tran gate_diff PARAM='gatep-gaten'",
        ".meas tran pre_after_p FIND V(pre0_p) AT=4.25n",
        ".meas tran pre_after_n FIND V(pre0_n) AT=4.25n",
        ".meas tran pre_after PARAM='pre_after_p-pre_after_n'",
        ".meas tran pre_delta PARAM='pre_after-pre_before'",
        ".meas tran c0_scorep_after FIND V(c0_scorep) AT=4.65n",
        ".meas tran c0_scoren_after FIND V(c0_scoren) AT=4.65n",
        ".meas tran c1_scorep_after FIND V(c1_scorep) AT=4.65n",
        ".meas tran c1_scoren_after FIND V(c1_scoren) AT=4.65n",
        ".meas tran c0_signed_after PARAM='c0_scorep_after-c0_scoren_after'",
        ".meas tran c1_signed_after PARAM='c1_scorep_after-c1_scoren_after'",
        f".meas tran target_margin_before PARAM='c{target_class}_signed_before-c{other_class}_signed_before'",
        f".meas tran target_margin_after PARAM='c{target_class}_signed_after-c{other_class}_signed_after'",
        ".meas tran target_margin_delta PARAM='target_margin_after-target_margin_before'",
        ".meas tran whp_after FIND V(wh0f0p) AT=4.80n",
        ".meas tran whn_after FIND V(wh0f0n) AT=4.80n",
        ".meas tran wh_delta PARAM='whp_after-whn_after-(0.75-0.15)'",
        ".tran 1p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _local_patch2x2_hidden_margin_rescue_netlist(patch_features: list[float]) -> str:
    if len(patch_features) != 4:
        raise ValueError("local patch rescue expects exactly four 2x2 patch features")
    lines = [
        "* Real local 2x2 patch hidden-margin rescue with frozen readout.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PWL(0n 1.2 0.45n 1.2 0.48n 0 3.00n 0 3.03n 1.2 3.45n 1.2 3.48n 0 6n 0)",
        "Vrstn rstn 0 PWL(0n 0 0.45n 0 0.48n 1.2 3.00n 1.2 3.03n 0 3.45n 0 3.48n 1.2 6n 1.2)",
        "Vfeatphi featphi 0 PWL(0n 0 0.75n 0 0.78n 1.2 1.25n 1.2 1.28n 0 3.75n 0 3.78n 1.2 4.25n 1.2 4.28n 0 6n 0)",
        "Vscorephi scorephi 0 PWL(0n 0 1.30n 0 1.33n 1.2 1.65n 1.2 1.68n 0 4.30n 0 4.33n 1.2 4.65n 1.2 4.68n 0 6n 0)",
        "Verrphi errphi 0 PWL(0n 0 1.68n 0 1.71n 1.2 2.25n 1.2 2.28n 0 6n 0)",
        "Vhcgphi hcgphi 0 PWL(0n 0 2.25n 0 2.28n 1.2 2.75n 1.2 2.78n 0 6n 0)",
        "Vhiddenwritephi hiddenwritephi 0 PWL(0n 0 2.65n 0 2.68n 1.2 2.78n 1.2 2.81n 0 6n 0)",
        "Iprobref vdd rnorm PWL(0n 0 1.68n 0 1.71n 10u 2.25n 10u 2.28n 0 6n 0)",
        "Vt0 t0 0 1.2",
        "Vt1 t1 0 0",
        *[f"Vpx{feature} px{feature} 0 {value:.12g}" for feature, value in enumerate(patch_features)],
        *mnist01_hidden._hidden_storage_lines(
            4,
            1,
            init_mode="patch2x2",
            connectivity_mode="patch2x2-sparse",
            inside_positive=0.75,
            outside_positive=0.15,
            inside_negative=0.15,
            outside_negative=0.15,
        ),
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(0, "vwp0"),
            negative_node=mnist01_hidden.class_node(0, "vwn0"),
            positive_ic=0.10,
            negative_ic=0.90,
        ),
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(1, "vwp0"),
            negative_node=mnist01_hidden.class_node(1, "vwn0"),
            positive_ic=0.90,
            negative_ic=0.10,
        ),
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            4,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=4.0,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        ),
        *mnist01_hidden._score_storage_lines(),
        *mnist01_hidden._score_readout_lines(1, 64.0, activation_mode="pre-differential"),
        *mnist01_hidden._error_storage_lines(),
        *mnist01_hidden._divider_probability_lines(0.5, 0.02),
        *mnist01_hidden._route_to_hidden_error_rails_lines(16.0),
        *mnist01_hidden._hidden_credit_lines(1, 32.0, 12.0, 0.05, 5.0e7),
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            1,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            4,
            1,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        ),
        ".meas tran pre_before_p FIND V(pre0_p) AT=1.25n",
        ".meas tran pre_before_n FIND V(pre0_n) AT=1.25n",
        ".meas tran pre_before PARAM='pre_before_p-pre_before_n'",
        ".meas tran c0_scorep_before FIND V(c0_scorep) AT=1.65n",
        ".meas tran c0_scoren_before FIND V(c0_scoren) AT=1.65n",
        ".meas tran c1_scorep_before FIND V(c1_scorep) AT=1.65n",
        ".meas tran c1_scoren_before FIND V(c1_scoren) AT=1.65n",
        ".meas tran c0_signed_before PARAM='c0_scorep_before-c0_scoren_before'",
        ".meas tran c1_signed_before PARAM='c1_scorep_before-c1_scoren_before'",
        ".meas tran c0_herrp_probe FIND V(c0_herrp) AT=2.20n",
        ".meas tran c0_herrn_probe FIND V(c0_herrn) AT=2.20n",
        ".meas tran c1_herrp_probe FIND V(c1_herrp) AT=2.20n",
        ".meas tran c1_herrn_probe FIND V(c1_herrn) AT=2.20n",
        ".meas tran target_herrdiff PARAM='c0_herrp_probe-c0_herrn_probe'",
        ".meas tran other_herrdiff PARAM='c1_herrp_probe-c1_herrn_probe'",
        ".meas tran hdp FIND V(h0_hdp) AT=2.55n",
        ".meas tran hdn FIND V(h0_hdn) AT=2.55n",
        ".meas tran hcredit PARAM='hdp-hdn'",
        ".meas tran gatep FIND V(h0_hdp_gate) AT=2.70n",
        ".meas tran gaten FIND V(h0_hdn_gate) AT=2.70n",
        ".meas tran gate_diff PARAM='gatep-gaten'",
        ".meas tran pre_after_p FIND V(pre0_p) AT=4.25n",
        ".meas tran pre_after_n FIND V(pre0_n) AT=4.25n",
        ".meas tran pre_after PARAM='pre_after_p-pre_after_n'",
        ".meas tran pre_delta PARAM='pre_after-pre_before'",
        ".meas tran c0_scorep_after FIND V(c0_scorep) AT=4.65n",
        ".meas tran c0_scoren_after FIND V(c0_scoren) AT=4.65n",
        ".meas tran c1_scorep_after FIND V(c1_scorep) AT=4.65n",
        ".meas tran c1_scoren_after FIND V(c1_scoren) AT=4.65n",
        ".meas tran c0_signed_after PARAM='c0_scorep_after-c0_scoren_after'",
        ".meas tran c1_signed_after PARAM='c1_scorep_after-c1_scoren_after'",
        ".meas tran target_margin_before PARAM='c0_signed_before-c1_signed_before'",
        ".meas tran target_margin_after PARAM='c0_signed_after-c1_signed_after'",
        ".meas tran target_margin_delta PARAM='target_margin_after-target_margin_before'",
        *[
            line
            for feature in range(4)
            for line in [
                f".meas tran wh{feature}p_after FIND V(wh0f{feature}p) AT=4.80n",
                f".meas tran wh{feature}n_after FIND V(wh0f{feature}n) AT=4.80n",
                f".meas tran wh{feature}_delta PARAM='wh{feature}p_after-wh{feature}n_after-(0.75-0.15)'",
            ]
        ],
        ".tran 1p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _local_patch2x2_hidden_margin_replay_netlist(
    train_patch: list[float],
    neighbor_patch: list[float],
    opposite_patch: list[float],
) -> str:
    if not (len(train_patch) == len(neighbor_patch) == len(opposite_patch) == 4):
        raise ValueError("local patch replay expects three 2x2 patch feature vectors")
    samples = [
        {"features": neighbor_patch, "label": 0, "train": False},
        {"features": opposite_patch, "label": 1, "train": False},
        {"features": train_patch, "label": 0, "train": True},
        {"features": train_patch, "label": 0, "train": False},
        {"features": neighbor_patch, "label": 0, "train": False},
        {"features": opposite_patch, "label": 1, "train": False},
    ]
    stop_ns = len(samples) * mnist01_fixed.CYCLE_NS
    reset = [(idx * mnist01_fixed.CYCLE_NS, idx * mnist01_fixed.CYCLE_NS + 0.45) for idx in range(len(samples))]
    feat = [
        (idx * mnist01_fixed.CYCLE_NS + 0.75, idx * mnist01_fixed.CYCLE_NS + 1.25)
        for idx in range(len(samples))
    ]
    score = [
        (idx * mnist01_fixed.CYCLE_NS + 1.30, idx * mnist01_fixed.CYCLE_NS + 1.65)
        for idx in range(len(samples))
    ]
    err = [(2 * mnist01_fixed.CYCLE_NS + 1.68, 2 * mnist01_fixed.CYCLE_NS + 2.25)]
    hcg = [(2 * mnist01_fixed.CYCLE_NS + 2.25, 2 * mnist01_fixed.CYCLE_NS + 2.75)]
    hiddenwrite = [(2 * mnist01_fixed.CYCLE_NS + 2.65, 2 * mnist01_fixed.CYCLE_NS + 2.78)]
    lines = [
        "* Local 2x2 patch rescue with same/opposite replay and frozen readout.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        f"Vrst rst 0 {mnist01_fixed._pulse_wave(reset, stop_ns)}",
        f"Vrstn rstn 0 {mnist01_fixed._active_low_pulse_wave(reset, stop_ns)}",
        f"Vfeatphi featphi 0 {mnist01_fixed._pulse_wave(feat, stop_ns)}",
        f"Vscorephi scorephi 0 {mnist01_fixed._pulse_wave(score, stop_ns)}",
        f"Verrphi errphi 0 {mnist01_fixed._pulse_wave(err, stop_ns)}",
        f"Vhcgphi hcgphi 0 {mnist01_fixed._pulse_wave(hcg, stop_ns)}",
        f"Vhiddenwritephi hiddenwritephi 0 {mnist01_fixed._pulse_wave(hiddenwrite, stop_ns)}",
        f"Iprobref vdd rnorm {mnist01_fixed._current_pulse_wave(err, stop_ns, 10.0e-6)}",
        "Vt0 t0 0 1.2",
        "Vt1 t1 0 0",
        *[f"Vpx{feature} px{feature} 0 {mnist01_fixed._sample_feature_wave(samples, feature, stop_ns)}" for feature in range(4)],
        *mnist01_hidden._hidden_storage_lines(
            4,
            1,
            init_mode="patch2x2",
            connectivity_mode="patch2x2-sparse",
            inside_positive=0.75,
            outside_positive=0.15,
            inside_negative=0.15,
            outside_negative=0.15,
        ),
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(0, "vwp0"),
            negative_node=mnist01_hidden.class_node(0, "vwn0"),
            positive_ic=0.10,
            negative_ic=0.90,
        ),
        *mnist01_hidden.signed_store_lines(
            positive_node=mnist01_hidden.class_node(1, "vwp0"),
            negative_node=mnist01_hidden.class_node(1, "vwn0"),
            positive_ic=0.90,
            negative_ic=0.10,
        ),
        *mnist01_hidden._hidden_state_lines(1),
        *mnist01_hidden._hidden_forward_lines(
            4,
            1,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=4.0,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        ),
        *mnist01_hidden._score_storage_lines(),
        *mnist01_hidden._score_readout_lines(1, 64.0, activation_mode="pre-differential"),
        *mnist01_hidden._error_storage_lines(),
        *mnist01_hidden._divider_probability_lines(0.5, 0.02),
        *mnist01_hidden._route_to_hidden_error_rails_lines(16.0),
        *mnist01_hidden._hidden_credit_lines(1, 32.0, 12.0, 0.05, 5.0e7),
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            1,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            4,
            1,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        ),
    ]

    def score_measures(prefix: str, idx: int, margin_expr: str) -> list[str]:
        base = idx * mnist01_fixed.CYCLE_NS
        return [
            f".meas tran {prefix}_pre_p FIND V(pre0_p) AT={base + 1.25:.2f}n",
            f".meas tran {prefix}_pre_n FIND V(pre0_n) AT={base + 1.25:.2f}n",
            f".meas tran {prefix}_pre PARAM='{prefix}_pre_p-{prefix}_pre_n'",
            f".meas tran {prefix}_c0_scorep FIND V(c0_scorep) AT={base + 1.65:.2f}n",
            f".meas tran {prefix}_c0_scoren FIND V(c0_scoren) AT={base + 1.65:.2f}n",
            f".meas tran {prefix}_c1_scorep FIND V(c1_scorep) AT={base + 1.65:.2f}n",
            f".meas tran {prefix}_c1_scoren FIND V(c1_scoren) AT={base + 1.65:.2f}n",
            f".meas tran {prefix}_c0_signed PARAM='{prefix}_c0_scorep-{prefix}_c0_scoren'",
            f".meas tran {prefix}_c1_signed PARAM='{prefix}_c1_scorep-{prefix}_c1_scoren'",
            f".meas tran {prefix}_margin PARAM='{margin_expr}'",
        ]

    lines += score_measures("neighbor_before", 0, "neighbor_before_c0_signed-neighbor_before_c1_signed")
    lines += score_measures("opposite_before", 1, "opposite_before_c1_signed-opposite_before_c0_signed")
    lines += score_measures("train_before", 2, "train_before_c0_signed-train_before_c1_signed")
    lines += [
        ".meas tran c0_herrp_probe FIND V(c0_herrp) AT=22.20n",
        ".meas tran c0_herrn_probe FIND V(c0_herrn) AT=22.20n",
        ".meas tran c1_herrp_probe FIND V(c1_herrp) AT=22.20n",
        ".meas tran c1_herrn_probe FIND V(c1_herrn) AT=22.20n",
        ".meas tran target_herrdiff PARAM='c0_herrp_probe-c0_herrn_probe'",
        ".meas tran other_herrdiff PARAM='c1_herrp_probe-c1_herrn_probe'",
        ".meas tran hdp FIND V(h0_hdp) AT=22.55n",
        ".meas tran hdn FIND V(h0_hdn) AT=22.55n",
        ".meas tran hcredit PARAM='hdp-hdn'",
        ".meas tran gatep FIND V(h0_hdp_gate) AT=22.70n",
        ".meas tran gaten FIND V(h0_hdn_gate) AT=22.70n",
        ".meas tran gate_diff PARAM='gatep-gaten'",
    ]
    lines += score_measures("train_after", 3, "train_after_c0_signed-train_after_c1_signed")
    lines += score_measures("neighbor_after", 4, "neighbor_after_c0_signed-neighbor_after_c1_signed")
    lines += score_measures("opposite_after", 5, "opposite_after_c1_signed-opposite_after_c0_signed")
    lines += [
        ".meas tran train_margin_delta PARAM='train_after_margin-train_before_margin'",
        ".meas tran neighbor_margin_delta PARAM='neighbor_after_margin-neighbor_before_margin'",
        ".meas tran opposite_margin_delta PARAM='opposite_after_margin-opposite_before_margin'",
        ".meas tran train_pre_delta PARAM='train_after_pre-train_before_pre'",
        ".meas tran neighbor_pre_delta PARAM='neighbor_after_pre-neighbor_before_pre'",
        ".meas tran opposite_pre_delta PARAM='opposite_after_pre-opposite_before_pre'",
        *[
            line
            for feature in range(4)
            for line in [
                f".meas tran wh{feature}p_after FIND V(wh0f{feature}p) AT=35.00n",
                f".meas tran wh{feature}n_after FIND V(wh0f{feature}n) AT=35.00n",
                f".meas tran wh{feature}_delta PARAM='wh{feature}p_after-wh{feature}n_after-(0.75-0.15)'",
            ]
        ],
        f".tran 5p {stop_ns:.2f}n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _multipatch_patch2x2_hidden_selective_rescue_netlist(
    train_features: list[float],
    neighbor_features: list[float],
    opposite_features: list[float],
) -> str:
    if not (len(train_features) == len(neighbor_features) == len(opposite_features) == 16):
        raise ValueError("multi-patch rescue expects three 4x4 feature vectors")
    hidden_count = mnist01_hidden.patch2x2_hidden_count(16)
    center_hidden = 4
    center_features = mnist01_hidden.patch2x2_features_for_hidden(center_hidden, 16, hidden_count)
    off_hidden = 0
    off_feature = 5
    samples = [
        {"features": neighbor_features, "label": 0, "train": False},
        {"features": opposite_features, "label": 1, "train": False},
        {"features": train_features, "label": 0, "train": True},
        {"features": train_features, "label": 0, "train": False},
        {"features": neighbor_features, "label": 0, "train": False},
        {"features": opposite_features, "label": 1, "train": False},
    ]
    stop_ns = len(samples) * mnist01_fixed.CYCLE_NS
    reset = [(idx * mnist01_fixed.CYCLE_NS, idx * mnist01_fixed.CYCLE_NS + 0.45) for idx in range(len(samples))]
    feat = [
        (idx * mnist01_fixed.CYCLE_NS + 0.75, idx * mnist01_fixed.CYCLE_NS + 1.25)
        for idx in range(len(samples))
    ]
    score = [
        (idx * mnist01_fixed.CYCLE_NS + 1.30, idx * mnist01_fixed.CYCLE_NS + 1.65)
        for idx in range(len(samples))
    ]
    err = [(2 * mnist01_fixed.CYCLE_NS + 1.68, 2 * mnist01_fixed.CYCLE_NS + 2.25)]
    hcg = [(2 * mnist01_fixed.CYCLE_NS + 2.25, 2 * mnist01_fixed.CYCLE_NS + 2.75)]
    hiddenwrite = [(2 * mnist01_fixed.CYCLE_NS + 2.65, 2 * mnist01_fixed.CYCLE_NS + 2.78)]
    lines = [
        "* Full 9-row patch2x2 hidden rescue with center-row credit selectivity.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        f"Vrst rst 0 {mnist01_fixed._pulse_wave(reset, stop_ns)}",
        f"Vrstn rstn 0 {mnist01_fixed._active_low_pulse_wave(reset, stop_ns)}",
        f"Vfeatphi featphi 0 {mnist01_fixed._pulse_wave(feat, stop_ns)}",
        f"Vscorephi scorephi 0 {mnist01_fixed._pulse_wave(score, stop_ns)}",
        f"Verrphi errphi 0 {mnist01_fixed._pulse_wave(err, stop_ns)}",
        f"Vhcgphi hcgphi 0 {mnist01_fixed._pulse_wave(hcg, stop_ns)}",
        f"Vhiddenwritephi hiddenwritephi 0 {mnist01_fixed._pulse_wave(hiddenwrite, stop_ns)}",
        f"Iprobref vdd rnorm {mnist01_fixed._current_pulse_wave(err, stop_ns, 10.0e-6)}",
        "Vt0 t0 0 1.2",
        "Vt1 t1 0 0",
        *[f"Vpx{feature} px{feature} 0 {mnist01_fixed._sample_feature_wave(samples, feature, stop_ns)}" for feature in range(16)],
        *mnist01_hidden._hidden_storage_lines(
            16,
            hidden_count,
            init_mode="patch2x2",
            connectivity_mode="patch2x2-sparse",
            inside_positive=0.75,
            outside_positive=0.15,
            inside_negative=0.15,
            outside_negative=0.15,
        ),
    ]
    for output in range(2):
        for hidden in range(hidden_count):
            if hidden == center_hidden and output == 0:
                positive_ic, negative_ic = 0.10, 0.90
            elif hidden == center_hidden and output == 1:
                positive_ic, negative_ic = 0.90, 0.10
            else:
                positive_ic, negative_ic = 0.40, 0.40
            lines += mnist01_hidden.signed_store_lines(
                positive_node=mnist01_hidden.class_node(output, f"vwp{hidden}"),
                negative_node=mnist01_hidden.class_node(output, f"vwn{hidden}"),
                positive_ic=positive_ic,
                negative_ic=negative_ic,
            )
    lines += [
        *mnist01_hidden._hidden_state_lines(hidden_count),
        *mnist01_hidden._hidden_forward_lines(
            16,
            hidden_count,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=4.0,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        ),
        *mnist01_hidden._score_storage_lines(),
        *mnist01_hidden._score_readout_lines(hidden_count, 64.0, activation_mode="pre-differential"),
        *mnist01_hidden._error_storage_lines(),
        *mnist01_hidden._divider_probability_lines(0.5, 0.02),
        *mnist01_hidden._route_to_hidden_error_rails_lines(16.0),
        *mnist01_hidden._hidden_credit_lines(hidden_count, 32.0, 12.0, 0.05, 5.0e7),
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            hidden_count,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            16,
            hidden_count,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        ),
    ]

    def score_measures(prefix: str, idx: int, margin_expr: str) -> list[str]:
        base = idx * mnist01_fixed.CYCLE_NS
        return [
            f".meas tran {prefix}_center_pre_p FIND V(pre{center_hidden}_p) AT={base + 1.25:.2f}n",
            f".meas tran {prefix}_center_pre_n FIND V(pre{center_hidden}_n) AT={base + 1.25:.2f}n",
            f".meas tran {prefix}_center_pre PARAM='{prefix}_center_pre_p-{prefix}_center_pre_n'",
            f".meas tran {prefix}_off_pre_p FIND V(pre{off_hidden}_p) AT={base + 1.25:.2f}n",
            f".meas tran {prefix}_off_pre_n FIND V(pre{off_hidden}_n) AT={base + 1.25:.2f}n",
            f".meas tran {prefix}_off_pre PARAM='{prefix}_off_pre_p-{prefix}_off_pre_n'",
            f".meas tran {prefix}_c0_scorep FIND V(c0_scorep) AT={base + 1.65:.2f}n",
            f".meas tran {prefix}_c0_scoren FIND V(c0_scoren) AT={base + 1.65:.2f}n",
            f".meas tran {prefix}_c1_scorep FIND V(c1_scorep) AT={base + 1.65:.2f}n",
            f".meas tran {prefix}_c1_scoren FIND V(c1_scoren) AT={base + 1.65:.2f}n",
            f".meas tran {prefix}_c0_signed PARAM='{prefix}_c0_scorep-{prefix}_c0_scoren'",
            f".meas tran {prefix}_c1_signed PARAM='{prefix}_c1_scorep-{prefix}_c1_scoren'",
            f".meas tran {prefix}_margin PARAM='{margin_expr}'",
        ]

    lines += score_measures("neighbor_before", 0, "neighbor_before_c0_signed-neighbor_before_c1_signed")
    lines += score_measures("opposite_before", 1, "opposite_before_c1_signed-opposite_before_c0_signed")
    lines += score_measures("train_before", 2, "train_before_c0_signed-train_before_c1_signed")
    lines += [
        ".meas tran center_hdp FIND V(h4_hdp) AT=22.55n",
        ".meas tran center_hdn FIND V(h4_hdn) AT=22.55n",
        ".meas tran center_hcredit PARAM='center_hdp-center_hdn'",
        ".meas tran center_gatep FIND V(h4_hdp_gate) AT=22.70n",
        ".meas tran center_gaten FIND V(h4_hdn_gate) AT=22.70n",
        ".meas tran center_gate_diff PARAM='center_gatep-center_gaten'",
        ".meas tran off_hdp FIND V(h0_hdp) AT=22.55n",
        ".meas tran off_hdn FIND V(h0_hdn) AT=22.55n",
        ".meas tran off_hcredit PARAM='off_hdp-off_hdn'",
        ".meas tran off_gatep FIND V(h0_hdp_gate) AT=22.70n",
        ".meas tran off_gaten FIND V(h0_hdn_gate) AT=22.70n",
        ".meas tran off_gate_diff PARAM='off_gatep-off_gaten'",
    ]
    lines += score_measures("train_after", 3, "train_after_c0_signed-train_after_c1_signed")
    lines += score_measures("neighbor_after", 4, "neighbor_after_c0_signed-neighbor_after_c1_signed")
    lines += score_measures("opposite_after", 5, "opposite_after_c1_signed-opposite_after_c0_signed")
    lines += [
        ".meas tran train_margin_delta PARAM='train_after_margin-train_before_margin'",
        ".meas tran neighbor_margin_delta PARAM='neighbor_after_margin-neighbor_before_margin'",
        ".meas tran opposite_margin_delta PARAM='opposite_after_margin-opposite_before_margin'",
        ".meas tran train_center_pre_delta PARAM='train_after_center_pre-train_before_center_pre'",
        ".meas tran neighbor_center_pre_delta PARAM='neighbor_after_center_pre-neighbor_before_center_pre'",
        ".meas tran opposite_center_pre_delta PARAM='opposite_after_center_pre-opposite_before_center_pre'",
        ".meas tran train_off_pre_delta PARAM='train_after_off_pre-train_before_off_pre'",
        ".meas tran neighbor_off_pre_delta PARAM='neighbor_after_off_pre-neighbor_before_off_pre'",
        *[
            line
            for feature in center_features
            for line in [
                f".meas tran center_wh{feature}p_after FIND V(wh{center_hidden}f{feature}p) AT=35.00n",
                f".meas tran center_wh{feature}n_after FIND V(wh{center_hidden}f{feature}n) AT=35.00n",
                f".meas tran center_wh{feature}_delta PARAM='center_wh{feature}p_after-center_wh{feature}n_after-(0.75-0.15)'",
            ]
        ],
        f".meas tran off_wh{off_feature}p_after FIND V(wh{off_hidden}f{off_feature}p) AT=35.00n",
        f".meas tran off_wh{off_feature}n_after FIND V(wh{off_hidden}f{off_feature}n) AT=35.00n",
        f".meas tran off_wh{off_feature}_delta PARAM='off_wh{off_feature}p_after-off_wh{off_feature}n_after-(0.75-0.15)'",
        f".tran 5p {stop_ns:.2f}n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _multipatch_patch2x2_readout_bootstrap_hidden_rescue_netlist(
    bootstrap_features: list[float],
    train_features: list[float],
    opposite_features: list[float],
) -> str:
    if not (len(bootstrap_features) == len(train_features) == len(opposite_features) == 16):
        raise ValueError("readout-bootstrap rescue expects three 4x4 feature vectors")
    hidden_count = mnist01_hidden.patch2x2_hidden_count(16)
    center_hidden = 4
    center_features = mnist01_hidden.patch2x2_features_for_hidden(center_hidden, 16, hidden_count)
    samples = [
        {"features": bootstrap_features, "label": 1, "train": True},
        {"features": train_features, "label": 0, "train": False},
        {"features": opposite_features, "label": 1, "train": False},
        {"features": train_features, "label": 0, "train": True},
        {"features": train_features, "label": 0, "train": False},
        {"features": opposite_features, "label": 1, "train": False},
    ]
    stop_ns = len(samples) * mnist01_fixed.CYCLE_NS
    reset = [(idx * mnist01_fixed.CYCLE_NS, idx * mnist01_fixed.CYCLE_NS + 0.45) for idx in range(len(samples))]
    feat = [
        (idx * mnist01_fixed.CYCLE_NS + 0.75, idx * mnist01_fixed.CYCLE_NS + 2.10)
        for idx in range(len(samples))
    ]
    score = [
        (idx * mnist01_fixed.CYCLE_NS + 2.00, idx * mnist01_fixed.CYCLE_NS + 3.20)
        for idx in range(len(samples))
    ]
    err = [
        (0 * mnist01_fixed.CYCLE_NS + 3.50, 0 * mnist01_fixed.CYCLE_NS + 5.40),
        (3 * mnist01_fixed.CYCLE_NS + 3.50, 3 * mnist01_fixed.CYCLE_NS + 5.40),
    ]
    readwrite = [(0 * mnist01_fixed.CYCLE_NS + 3.50, 0 * mnist01_fixed.CYCLE_NS + 5.40)]
    hcg = [(3 * mnist01_fixed.CYCLE_NS + 5.00, 3 * mnist01_fixed.CYCLE_NS + 6.15)]
    hiddenwrite = [(3 * mnist01_fixed.CYCLE_NS + 6.30, 3 * mnist01_fixed.CYCLE_NS + 8.40)]
    lines = [
        "* Patch2x2 hidden rescue using readout leverage created by a live readout writer.",
        ".param VDD=1.2",
        mnist01_hidden.mos_models(),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        f"Vrst rst 0 {mnist01_fixed._pulse_wave(reset, stop_ns)}",
        f"Vrstn rstn 0 {mnist01_fixed._active_low_pulse_wave(reset, stop_ns)}",
        f"Vfeatphi featphi 0 {mnist01_fixed._pulse_wave(feat, stop_ns)}",
        f"Vscorephi scorephi 0 {mnist01_fixed._pulse_wave(score, stop_ns)}",
        f"Verrphi errphi 0 {mnist01_fixed._pulse_wave(err, stop_ns)}",
        f"Vreadwritephi readwritephi 0 {mnist01_fixed._pulse_wave(readwrite, stop_ns)}",
        f"Vhcgphi hcgphi 0 {mnist01_fixed._pulse_wave(hcg, stop_ns)}",
        f"Vhiddenwritephi hiddenwritephi 0 {mnist01_fixed._pulse_wave(hiddenwrite, stop_ns)}",
        f"Iprobref vdd rnorm {mnist01_fixed._current_pulse_wave(err, stop_ns, 10.0e-6)}",
        f"Vt0 t0 0 {mnist01_fixed._target_wave(samples, 0, stop_ns)}",
        f"Vt1 t1 0 {mnist01_fixed._target_wave(samples, 1, stop_ns)}",
        *[f"Vpx{feature} px{feature} 0 {mnist01_fixed._sample_feature_wave(samples, feature, stop_ns)}" for feature in range(16)],
        *mnist01_hidden._hidden_storage_lines(
            16,
            hidden_count,
            init_mode="patch2x2",
            connectivity_mode="patch2x2-sparse",
            inside_positive=0.75,
            outside_positive=0.15,
            inside_negative=0.15,
            outside_negative=0.15,
        ),
        *mnist01_hidden._readout_storage_lines(hidden_count, 0.40, 0.40),
        *mnist01_hidden._hidden_state_lines(hidden_count),
        *mnist01_hidden._hidden_forward_lines(
            16,
            hidden_count,
            8.0,
            activation_mode="differential-preamp",
            activation_sense_width_u=4.0,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        ),
        *mnist01_hidden._score_storage_lines(),
        *mnist01_hidden._score_readout_lines(hidden_count, 64.0, activation_mode="pre-differential"),
        *mnist01_hidden._error_storage_lines(),
        *mnist01_hidden._divider_probability_lines(0.5, 0.02),
        *mnist01_hidden._route_to_error_rails_lines(16.0),
        *mnist01_hidden._route_to_hidden_error_rails_lines(16.0),
        "Vvwhi_ref vwhi_ref 0 0.48",
        "Vvwlo_ref vwlo_ref 0 0.22",
    ]
    for output in range(2):
        lines += mnist01_hidden.class_local_live_label_descent_update_lines(
            class_idx=output,
            feature_idx=center_hidden,
            activation_node=f"hrow{center_hidden}",
            positive_descent_node=mnist01_hidden.class_node(output, "errp"),
            negative_descent_node=mnist01_hidden.class_node(output, "errn"),
            update_guard_node="readwritephi",
            update_guard_model="NSENSE",
            width_u=0.8,
            high_side_topology="pmos-differential",
        )
    lines += [
        *mnist01_hidden._hidden_credit_lines(hidden_count, 32.0, 12.0, 0.05, 5.0e7),
        *mnist01_hidden._hidden_credit_dynamic_preamp_gate_lines(
            hidden_count,
            sense_width_u=32.0,
            latch_pmos_width_u=4.0,
            output_width_u=2.0,
            output_pull_width_u=0.5,
            support_width_u=4.0,
            write_gate_width_u=8.0,
            capacitance_f=2.0,
        ),
        *mnist01_hidden._hidden_writer_lines(
            16,
            hidden_count,
            0.2,
            4.0,
            0.2,
            "pmos-signcharge",
            "h{hidden}_hcg_write",
            True,
            hidden_init_mode="patch2x2",
            hidden_connectivity_mode="patch2x2-sparse",
        ),
    ]

    def score_measures(prefix: str, idx: int, margin_expr: str) -> list[str]:
        base = idx * mnist01_fixed.CYCLE_NS
        return [
            f".meas tran {prefix}_center_pre_p FIND V(pre{center_hidden}_p) AT={base + 2.00:.2f}n",
            f".meas tran {prefix}_center_pre_n FIND V(pre{center_hidden}_n) AT={base + 2.00:.2f}n",
            f".meas tran {prefix}_center_pre PARAM='{prefix}_center_pre_p-{prefix}_center_pre_n'",
            f".meas tran {prefix}_c0_scorep FIND V(c0_scorep) AT={base + 3.15:.2f}n",
            f".meas tran {prefix}_c0_scoren FIND V(c0_scoren) AT={base + 3.15:.2f}n",
            f".meas tran {prefix}_c1_scorep FIND V(c1_scorep) AT={base + 3.15:.2f}n",
            f".meas tran {prefix}_c1_scoren FIND V(c1_scoren) AT={base + 3.15:.2f}n",
            f".meas tran {prefix}_c0_signed PARAM='{prefix}_c0_scorep-{prefix}_c0_scoren'",
            f".meas tran {prefix}_c1_signed PARAM='{prefix}_c1_scorep-{prefix}_c1_scoren'",
            f".meas tran {prefix}_margin PARAM='{margin_expr}'",
        ]

    def center_readout_measures(prefix: str, at_ns: float) -> list[str]:
        return [
            f".meas tran {prefix}_c0_vwp FIND V(c0_vwp{center_hidden}) AT={at_ns:.2f}n",
            f".meas tran {prefix}_c0_vwn FIND V(c0_vwn{center_hidden}) AT={at_ns:.2f}n",
            f".meas tran {prefix}_c1_vwp FIND V(c1_vwp{center_hidden}) AT={at_ns:.2f}n",
            f".meas tran {prefix}_c1_vwn FIND V(c1_vwn{center_hidden}) AT={at_ns:.2f}n",
            f".meas tran {prefix}_c0_signed PARAM='{prefix}_c0_vwp-{prefix}_c0_vwn'",
            f".meas tran {prefix}_c1_signed PARAM='{prefix}_c1_vwp-{prefix}_c1_vwn'",
        ]

    lines += center_readout_measures("readout_initial", 0.55)
    lines += center_readout_measures("readout_after_bootstrap", 8.80)
    lines += score_measures("train_before", 1, "train_before_c0_signed-train_before_c1_signed")
    lines += score_measures("opposite_before", 2, "opposite_before_c1_signed-opposite_before_c0_signed")
    lines += center_readout_measures("readout_before_hidden", 30.55)
    lines += [
        ".meas tran center_hdp FIND V(h4_hdp) AT=35.35n",
        ".meas tran center_hdn FIND V(h4_hdn) AT=35.35n",
        ".meas tran center_hcredit PARAM='center_hdp-center_hdn'",
        ".meas tran center_gatep FIND V(h4_hdp_gate) AT=37.35n",
        ".meas tran center_gaten FIND V(h4_hdn_gate) AT=37.35n",
        ".meas tran center_gate_diff PARAM='center_gatep-center_gaten'",
    ]
    lines += score_measures("train_after", 4, "train_after_c0_signed-train_after_c1_signed")
    lines += score_measures("opposite_after", 5, "opposite_after_c1_signed-opposite_after_c0_signed")
    lines += center_readout_measures("readout_after_hidden", 58.80)
    lines += [
        ".meas tran c0_bootstrap_signed_delta PARAM='readout_after_bootstrap_c0_signed-readout_initial_c0_signed'",
        ".meas tran c1_bootstrap_signed_delta PARAM='readout_after_bootstrap_c1_signed-readout_initial_c1_signed'",
        ".meas tran c0_hidden_signed_drift PARAM='readout_after_hidden_c0_signed-readout_before_hidden_c0_signed'",
        ".meas tran c1_hidden_signed_drift PARAM='readout_after_hidden_c1_signed-readout_before_hidden_c1_signed'",
        ".meas tran train_margin_delta PARAM='train_after_margin-train_before_margin'",
        ".meas tran train_center_pre_delta PARAM='train_after_center_pre-train_before_center_pre'",
        ".meas tran opposite_margin_delta PARAM='opposite_after_margin-opposite_before_margin'",
        *[
            line
            for feature in center_features
            for line in [
                f".meas tran center_wh{feature}p_after FIND V(wh{center_hidden}f{feature}p) AT=48.80n",
                f".meas tran center_wh{feature}n_after FIND V(wh{center_hidden}f{feature}n) AT=48.80n",
                f".meas tran center_wh{feature}_delta PARAM='center_wh{feature}p_after-center_wh{feature}n_after-(0.75-0.15)'",
            ]
        ],
        f".tran 5p {stop_ns:.2f}n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


@pytest.mark.ngspice
def test_hidden_credit_dynamic_preamp_restores_mnist_scale_credit_with_dead_zone(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_preamp_positive.cir",
        _hidden_credit_preamp_primitive_netlist(0.2562769, 0.2559558),
        timeout=30.0,
    )
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_preamp_negative.cir",
        _hidden_credit_preamp_primitive_netlist(0.2559558, 0.2562769),
        timeout=30.0,
    )
    neutral = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_preamp_neutral.cir",
        _hidden_credit_preamp_primitive_netlist(0.2560, 0.2560),
        timeout=30.0,
    )
    tiny = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_preamp_tiny.cir",
        _hidden_credit_preamp_primitive_netlist(0.25605, 0.2560),
        timeout=30.0,
    )
    zero = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_preamp_zero.cir",
        _hidden_credit_preamp_primitive_netlist(0.0, 0.0),
        timeout=30.0,
    )

    assert positive["support_pre"] > 0.15
    assert positive["write_gate"] > 0.15
    assert positive["gate_diff"] > 30e-3
    assert positive["signed_delta"] > 0.50
    assert negative["gate_diff"] < -30e-3
    assert negative["signed_delta"] < -0.50
    assert abs(neutral["gate_diff"]) < 1e-6
    assert abs(neutral["signed_delta"]) < 1e-3
    assert tiny["gate_diff"] > 5e-3
    assert abs(tiny["signed_delta"]) < 1e-3
    assert zero["support_pre"] < 1e-6
    assert zero["write_gate"] < 1e-6
    assert abs(zero["signed_delta"]) < 1e-3


@pytest.mark.ngspice
def test_hidden_signcharge_packet_writer_bounds_saturated_state(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_signcharge_packet_negative.cir",
        _hidden_signcharge_packet_writer_netlist(
            positive_gate=0.0,
            negative_gate=0.69,
            whp_ic=1.05,
            whn_ic=0.05,
        ),
        timeout=30.0,
    )
    positive = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_signcharge_packet_positive.cir",
        _hidden_signcharge_packet_writer_netlist(
            positive_gate=0.69,
            negative_gate=0.0,
            whp_ic=0.05,
            whn_ic=1.05,
        ),
        timeout=30.0,
    )
    neutral = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_signcharge_packet_neutral.cir",
        _hidden_signcharge_packet_writer_netlist(
            positive_gate=0.0,
            negative_gate=0.0,
            whp_ic=1.05,
            whn_ic=0.05,
        ),
        timeout=30.0,
    )

    assert -30e-3 < negative["signed_delta"] < -3e-3
    assert 3e-3 < positive["signed_delta"] < 30e-3
    assert abs(neutral["signed_delta"]) < 100e-6


@pytest.mark.ngspice
def test_hidden_signcharge_packet_write_changes_next_forward_evidence(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive_visible = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_signcharge_forward_positive_visible.cir",
        _hidden_signcharge_forward_after_write_netlist(
            positive_gate=0.69,
            negative_gate=0.0,
            whp_ic=0.75,
            whn_ic=0.15,
        ),
        timeout=30.0,
    )
    positive_saturated = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_signcharge_forward_positive_saturated.cir",
        _hidden_signcharge_forward_after_write_netlist(
            positive_gate=0.69,
            negative_gate=0.0,
            whp_ic=0.90,
            whn_ic=0.15,
        ),
        timeout=30.0,
    )
    negative_visible = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_signcharge_forward_negative_visible.cir",
        _hidden_signcharge_forward_after_write_netlist(
            positive_gate=0.0,
            negative_gate=0.69,
            whp_ic=0.75,
            whn_ic=0.15,
        ),
        timeout=30.0,
    )

    assert positive_visible["wh_delta"] > 3.0e-3
    assert positive_visible["pre_delta"] > 3.0e-3
    assert positive_saturated["wh_delta"] > 1.0e-3
    assert abs(positive_saturated["pre_delta"]) < 1.0e-3
    assert negative_visible["wh_delta"] < -3.0e-3
    assert negative_visible["pre_delta"] < -3.0e-3


@pytest.mark.ngspice
def test_hidden_credit_signcharge_backprop_changes_next_forward_evidence(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    positive = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_forward_positive.cir",
        _hidden_credit_signcharge_forward_after_write_netlist("positive"),
        timeout=30.0,
    )
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_forward_negative.cir",
        _hidden_credit_signcharge_forward_after_write_netlist("negative"),
        timeout=30.0,
    )

    assert positive["hcredit"] > 100e-3
    assert positive["gate_diff"] > 100e-3
    assert positive["wh_delta"] > 3.0e-3
    assert positive["pre_delta"] > 3.0e-3
    assert negative["hcredit"] < -100e-3
    assert negative["gate_diff"] < -100e-3
    assert negative["wh_delta"] < -3.0e-3
    assert negative["pre_delta"] < -3.0e-3


@pytest.mark.ngspice
def test_hidden_credit_signcharge_write_improves_downstream_score_margin(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    target0 = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_margin_target0.cir",
        _hidden_credit_signcharge_margin_after_write_netlist(0),
        timeout=30.0,
    )
    target1 = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_margin_target1.cir",
        _hidden_credit_signcharge_margin_after_write_netlist(1),
        timeout=30.0,
    )

    assert target0["hcredit"] > 100e-3
    assert target0["gate_diff"] > 100e-3
    assert target0["wh_delta"] > 3.0e-3
    assert target0["pre_delta"] > 3.0e-3
    assert target0["target_margin_delta"] > 0.25e-3
    assert target0["c0_signed_after"] > target0["c0_signed_before"]
    assert target0["c1_signed_after"] < target0["c1_signed_before"]

    assert target1["hcredit"] < -100e-3
    assert target1["gate_diff"] < -100e-3
    assert target1["wh_delta"] < -3.0e-3
    assert target1["pre_delta"] < -3.0e-3
    assert target1["target_margin_delta"] > 1.0e-3
    assert target1["c1_signed_after"] > target1["c1_signed_before"]
    assert target1["c0_signed_after"] < target1["c0_signed_before"]


@pytest.mark.ngspice
def test_conductance_divider_hidden_signcharge_rescues_misclassified_margin_without_readout_write(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    rescued = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_divider_signcharge_margin_rescue_target1.cir",
        _hidden_divider_signcharge_margin_rescue_netlist(1),
        timeout=60.0,
    )
    already_correct = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_divider_signcharge_margin_rescue_target0.cir",
        _hidden_divider_signcharge_margin_rescue_netlist(0),
        timeout=60.0,
    )

    assert rescued["target_margin_before"] < -0.25
    assert rescued["target_herrdiff"] > 1.0
    assert rescued["other_herrdiff"] < -1.0
    assert rescued["hcredit"] < -0.50
    assert rescued["gate_diff"] < -0.50
    assert rescued["wh_delta"] < -3.0e-3
    assert rescued["pre_delta"] < -3.0e-3
    assert rescued["target_margin_delta"] > 5.0e-3
    assert rescued["c1_signed_after"] > rescued["c1_signed_before"]
    assert rescued["c0_signed_after"] < rescued["c0_signed_before"]

    assert already_correct["target_margin_before"] > 0.25
    assert abs(already_correct["target_herrdiff"]) < 1.0e-3
    assert abs(already_correct["other_herrdiff"]) < 1.0e-3
    assert abs(already_correct["gate_diff"]) < 1.0e-6
    assert abs(already_correct["wh_delta"]) < 1.0e-4
    assert abs(already_correct["target_margin_delta"]) < 1.0e-4


@pytest.mark.ngspice
def test_real_mnist_patch2x2_hidden_signcharge_rescues_local_margin_without_readout_write(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, _evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=1,
        eval_count_per_digit=1,
        image_size=4,
    )
    digit0 = next(record for record in train if int(record["label"]) == 0)
    center_patch = mnist01_hidden.patch2x2_features_for_hidden(4, 16, 9)
    patch_features = [float(digit0["features"][feature]) for feature in center_patch]
    assert min(patch_features) > 0.50

    rescued = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_local_patch2x2_hidden_margin_rescue.cir",
        _local_patch2x2_hidden_margin_rescue_netlist(patch_features),
        timeout=60.0,
    )

    assert rescued["target_margin_before"] < -0.25
    assert rescued["target_herrdiff"] > 1.0
    assert rescued["other_herrdiff"] < -1.0
    assert rescued["hcredit"] < -0.50
    assert rescued["gate_diff"] < -0.50
    assert rescued["pre_delta"] < -5.0e-3
    assert rescued["target_margin_delta"] > 5.0e-3
    assert rescued["c0_signed_after"] > rescued["c0_signed_before"]
    assert rescued["c1_signed_after"] < rescued["c1_signed_before"]
    assert max(rescued[f"wh{feature}_delta"] for feature in range(4)) < -3.0e-3


@pytest.mark.ngspice
def test_real_mnist_patch2x2_hidden_rescue_preserves_neighbor_and_opposite_margin(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, _evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=3,
        eval_count_per_digit=1,
        image_size=4,
    )
    center_patch = mnist01_hidden.patch2x2_features_for_hidden(4, 16, 9)
    digit0_records = [record for record in train if int(record["label"]) == 0]
    digit1_records = [record for record in train if int(record["label"]) == 1]
    train_patch = [float(digit0_records[0]["features"][feature]) for feature in center_patch]
    neighbor_patch = [float(digit0_records[1]["features"][feature]) for feature in center_patch]
    opposite_patch = [float(digit1_records[0]["features"][feature]) for feature in center_patch]
    assert min(train_patch) > 0.50
    assert min(neighbor_patch) > 0.50
    assert sum(opposite_patch) < sum(train_patch)

    replay = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_local_patch2x2_hidden_margin_replay.cir",
        _local_patch2x2_hidden_margin_replay_netlist(train_patch, neighbor_patch, opposite_patch),
        timeout=90.0,
    )

    assert replay["train_before_margin"] < -0.25
    assert replay["train_margin_delta"] > 5.0e-3
    assert replay["neighbor_before_margin"] < -0.25
    assert replay["neighbor_margin_delta"] > 5.0e-3
    assert replay["opposite_before_margin"] > 5.0e-3
    assert replay["opposite_margin_delta"] > 0.0
    assert replay["opposite_after_margin"] > replay["opposite_before_margin"]
    assert replay["hcredit"] < -0.50
    assert replay["gate_diff"] < -0.50
    assert replay["train_pre_delta"] < -5.0e-3
    assert replay["neighbor_pre_delta"] < -5.0e-3
    assert replay["opposite_pre_delta"] < 0.0
    assert max(replay[f"wh{feature}_delta"] for feature in range(4)) < -3.0e-3


@pytest.mark.ngspice
def test_real_mnist_multipatch_hidden_rescue_is_center_selective(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, _evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=3,
        eval_count_per_digit=1,
        image_size=4,
    )
    digit0_records = [record for record in train if int(record["label"]) == 0]
    digit1_records = [record for record in train if int(record["label"]) == 1]
    center_patch = mnist01_hidden.patch2x2_features_for_hidden(4, 16, 9)
    assert min(float(digit0_records[0]["features"][feature]) for feature in center_patch) > 0.50

    replay = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_multipatch_patch2x2_hidden_selective_rescue.cir",
        _multipatch_patch2x2_hidden_selective_rescue_netlist(
            [float(value) for value in digit0_records[0]["features"]],
            [float(value) for value in digit0_records[1]["features"]],
            [float(value) for value in digit1_records[0]["features"]],
        ),
        timeout=120.0,
    )

    assert replay["train_before_margin"] < -0.25
    assert replay["train_margin_delta"] > 1.0e-3
    assert replay["neighbor_margin_delta"] > 1.0e-3
    assert replay["opposite_after_margin"] > 5.0e-3
    assert replay["opposite_margin_delta"] > -0.5e-3
    assert replay["center_hcredit"] < -0.50
    assert replay["center_gate_diff"] < -0.50
    assert abs(replay["off_hcredit"]) < 1.0e-6
    assert abs(replay["off_gate_diff"]) < 1.0e-6
    assert replay["train_center_pre_delta"] < -5.0e-3
    assert replay["neighbor_center_pre_delta"] < -5.0e-3
    assert abs(replay["train_off_pre_delta"]) < 1.0e-3
    for feature in center_patch:
        assert replay[f"center_wh{feature}_delta"] < -3.0e-3
    assert abs(replay["off_wh5_delta"]) < 50e-6


@pytest.mark.ngspice
def test_real_mnist_patch2x2_hidden_rescue_can_use_live_learned_readout_leverage(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, _evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=3,
        eval_count_per_digit=1,
        image_size=4,
    )
    digit0_records = [record for record in train if int(record["label"]) == 0]
    digit1_records = [record for record in train if int(record["label"]) == 1]
    center_patch = mnist01_hidden.patch2x2_features_for_hidden(4, 16, 9)
    assert min(float(digit0_records[0]["features"][feature]) for feature in center_patch) > 0.50
    assert sum(float(digit1_records[0]["features"][feature]) for feature in center_patch) > 1.0

    replay = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "mnist01_multipatch_patch2x2_readout_bootstrap_hidden_rescue.cir",
        _multipatch_patch2x2_readout_bootstrap_hidden_rescue_netlist(
            [float(value) for value in digit1_records[0]["features"]],
            [float(value) for value in digit0_records[0]["features"]],
            [float(value) for value in digit1_records[1]["features"]],
        ),
        timeout=180.0,
    )

    assert replay["c0_bootstrap_signed_delta"] < -10e-3
    assert replay["c1_bootstrap_signed_delta"] > 10e-3
    assert replay["readout_after_bootstrap_c0_signed"] < -10e-3
    assert replay["readout_after_bootstrap_c1_signed"] > 10e-3
    assert abs(replay["c0_hidden_signed_drift"]) < 1.0e-3
    assert abs(replay["c1_hidden_signed_drift"]) < 1.0e-3
    assert replay["train_before_margin"] < -0.25e-3
    assert replay["center_hcredit"] < -50e-3
    assert replay["center_gate_diff"] < -50e-3
    assert replay["train_center_pre_delta"] < -1.0e-3
    assert replay["train_margin_delta"] > 0.25e-3
    assert replay["opposite_after_margin"] > 0.0
    assert replay["opposite_margin_delta"] > -2.0e-3
    for feature in center_patch:
        assert replay[f"center_wh{feature}_delta"] < -1.0e-3


@pytest.mark.ngspice
def test_hidden_credit_signcharge_writer_preserves_bounded_magnitude(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    negative = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_negative.cir",
        _hidden_credit_signcharge_primitive_netlist(0.25584, 0.25616),
        timeout=30.0,
    )
    neutral = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_neutral.cir",
        _hidden_credit_signcharge_primitive_netlist(0.2560, 0.2560),
        timeout=30.0,
    )
    tiny = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_tiny.cir",
        _hidden_credit_signcharge_primitive_netlist(0.25605, 0.25595),
        timeout=30.0,
    )
    positive = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_positive.cir",
        _hidden_credit_signcharge_primitive_netlist(0.25616, 0.25584),
        timeout=30.0,
    )
    large = mnist01_hidden.run_netlist(
        ngspice_path,
        tmp_path / "hidden_credit_signcharge_large.cir",
        _hidden_credit_signcharge_primitive_netlist(0.2565, 0.2555),
        timeout=30.0,
    )

    assert negative["gate_diff"] < -30e-3
    assert -1e-3 < negative["signed_delta"] < -50e-6
    assert abs(neutral["gate_diff"]) < 1e-6
    assert abs(neutral["signed_delta"]) < 100e-6
    assert tiny["gate_diff"] > 10e-3
    assert abs(tiny["signed_delta"]) < 100e-6
    assert positive["gate_diff"] > 30e-3
    assert 50e-6 < positive["signed_delta"] < 1e-3
    assert positive["signed_delta"] < large["signed_delta"] < 20e-3
