from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
sys.path.insert(0, str(SPICE_DIR))

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
    assert "Vhcgphi hcgphi 0 PWL" in netlist
    assert "Vhiddenwritephi hiddenwritephi 0 PWL" in netlist
    assert "Cc0_herrp c0_herrp 0 2f IC=0" in netlist
    assert "Mherr_c0p_m vdd b1low herr_c0p_a vdd PMOS" in netlist
    assert "Mh1_c0_cred_pv_e vdd c0_herrp h1_c0_cred_pv_e 0 NSENSE" in netlist
    assert "Mc0_f1_live_pos_up_p c0_vwp1 c0_f1_live_pos_up_ctrl vwhi_ref vdd PMOS" in netlist
    assert "Mh1f6_live_pup_pgate_phi h1f6_live_pup_pgphi errphi 0 0 NSENSE" in netlist
    assert ".meas tran train_hrow_probe_0" in netlist
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


def test_mnist01_live_hidden_netlist_validation() -> None:
    sample = {"features": [1.0] * 16, "label": 0}

    with pytest.raises(ValueError, match="empty"):
        mnist01_hidden.mnist01_live_hidden_netlist([], [sample])
    with pytest.raises(ValueError, match="exactly four"):
        mnist01_hidden.mnist01_live_hidden_netlist([sample], [sample], hidden_count=3)
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
    with pytest.raises(ValueError, match="hidden_credit_gate_mode"):
        mnist01_hidden.mnist01_live_hidden_netlist(
            [sample],
            [sample],
            hidden_credit_gate_mode="BAD",
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
