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
