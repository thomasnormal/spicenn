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
    assert ".meas tran final_hrow_h3_1" in netlist
    assert ".meas tran final_margin_improvement_1" in netlist

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

    normalized_writer_netlist = mnist01_hidden.mnist01_live_hidden_netlist(
        train,
        train,
        readout_writer_normalization_mode="activity-gate",
    )
    assert "Chrow_activity_gate hrow_activity_gate 0 80f IC=1.2" in normalized_writer_netlist
    assert "Mhrow_activity_gate_h0_phi hrow_activity_gate_h0_mid scorephi 0 0 NSENSE" in normalized_writer_netlist
    assert "Mc0_f0_live_pos_dn_g c0_f0_live_pos_dn_allguard hrow_activity_gate c0_f0_live_pos_dn 0 NREL" in normalized_writer_netlist
    assert "Mc0_f0_live_pos_dn_e c0_vwn0 hrow0 c0_f0_live_pos_dn_allguard 0 NSENSE" in normalized_writer_netlist


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
def test_mnist01_live_hidden_sparse_complement_identity_rows_learn_ten_round_robin_margins(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    _require_mnist_raw()
    train, evals = mnist01_hidden.load_mnist01_records(
        train_count_per_digit=5,
        eval_count_per_digit=5,
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
            readout_update_width_u=0.20,
            hidden_writer_phase_mode="hidden-write",
            hidden_write_start_train_index=999,
        ),
        timeout=420.0,
    )

    for sample_idx in range(10):
        assert parsed[f"final_margin_{sample_idx}"] > 0.25e-3
        assert parsed[f"final_margin_improvement_{sample_idx}"] > 0.25e-3
    for train_idx in range(10):
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
    assert min(hidden_deltas) < -3.0e-3
    assert max(hidden_deltas) > 3.0e-3
    assert max(abs(delta) for delta in hidden_deltas) > 3.0e-3
    assert max(abs(delta) for delta in hidden_deltas) < 20.0e-3
    assert max(abs(parsed[f"train_hcredit_gate_probe_{idx}"]) for idx in range(1, 4)) > 0.5

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
        *mnist01_hidden._readout_storage_lines(1, vwp, vwn),
        *mnist01_hidden._score_storage_lines(),
        *mnist01_hidden._score_readout_lines(1, 16.0, activation_mode="pre-differential"),
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


def _pre_differential_readout_writer_probe_netlist(
    *,
    pre_p: float,
    pre_n: float,
    errp: float = 0.75,
    errn: float = 0.0,
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
        f"Verrp {mnist01_hidden.class_node(0, 'errp')} 0 {errp:.12g}",
        f"Verrn {mnist01_hidden.class_node(0, 'errn')} 0 {errn:.12g}",
        f"Verrp1 {mnist01_hidden.class_node(1, 'errp')} 0 0",
        f"Verrn1 {mnist01_hidden.class_node(1, 'errn')} 0 0",
        *mnist01_hidden._readout_storage_lines(1, 0.40, 0.40),
        *mnist01_hidden._readout_writer_lines(1, 0.25, activation_mode="pre-differential"),
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
