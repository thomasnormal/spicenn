from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_multiclass_output_head_sequence as seq  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def _one_hot_records() -> list[dict[str, object]]:
    records = []
    for label in range(3):
        records.append(
            {
                "label": label,
                "inputs": {f"x{feature}": 0.92 if feature == label else 0.0 for feature in range(3)},
            }
        )
    return records


def test_multiclass_output_head_sequence_emits_continuous_train_and_eval_deck() -> None:
    records = _one_hot_records()
    netlist = seq.generate_netlist(train_records=records, eval_records=records, class_count=3, feature_count=3)

    assert "\nB" not in netlist
    assert "Vacc acc 0 PWL(" in netlist
    assert "Vapplyn applyn 0 PWL(" in netlist
    assert "Vc2_targetp c2_targetp 0 PWL(" in netlist
    assert "Cc2_vwp2 c2_vwp2 0 20f IC=0.36" in netlist
    assert "Mc2_f2_pos_cond actrow2 c2_vwp2 c2_score 0 NMOS" in netlist
    assert "* cycle 0 initial_eval label=0" in netlist
    assert "* cycle 3 train label=0" in netlist
    assert "* cycle 6 final_eval label=0" in netlist


def test_multiclass_output_head_sequence_can_gate_nontarget_update_with_score() -> None:
    records = _one_hot_records()
    netlist = seq.generate_netlist(
        train_records=records,
        eval_records=records,
        class_count=3,
        feature_count=3,
        error_mode="score-gated-nontarget",
    )

    assert "Vrstscore rstscore 0 PWL(" in netlist
    assert "Mc2_f1_gvn_score c2_f1_gvn_label c2_score c2_f1_gvn_d 0 NSENSE" in netlist
    assert "Mc2_f1_gvn_d c2_f1_gvn_a c2_targetn" not in netlist


def test_multiclass_output_head_sequence_validation() -> None:
    records = _one_hot_records()
    with pytest.raises(ValueError, match="class_count"):
        seq.generate_netlist(train_records=records, eval_records=records, class_count=1, feature_count=3)
    with pytest.raises(ValueError, match="feature_count"):
        seq.generate_netlist(train_records=records, eval_records=records, class_count=3, feature_count=0)
    with pytest.raises(ValueError, match="counted multiclass"):
        seq.main_for_test(["--dataset", "mnist01fixed8_16"])
    with pytest.raises(ValueError, match="nontarget-scale"):
        seq.main_for_test(["--nontarget-scale", "1.5"])
    with pytest.raises(ValueError, match="error_mode"):
        seq.generate_netlist(
            train_records=records,
            eval_records=records,
            class_count=3,
            feature_count=3,
            error_mode="bad",
        )


def test_multiclass_output_head_sequence_balanced_split_covers_each_class() -> None:
    records = _one_hot_records() + _one_hot_records()

    train, eval_ = seq.balanced_train_eval_split(records, class_count=3, train_samples=3, eval_samples=3)

    assert [record["label"] for record in train] == [0, 1, 2]
    assert [record["label"] for record in eval_] == [0, 1, 2]
    with pytest.raises(ValueError, match="divisible"):
        seq.balanced_train_eval_split(records, class_count=3, train_samples=4, eval_samples=3)


def test_multiclass_output_head_sequence_extracts_final_signed_weight_matrix() -> None:
    measures = {
        f"c{class_idx}_f{feature}_signed_final": class_idx + 0.1 * feature
        for class_idx in range(2)
        for feature in range(3)
    }

    assert seq.final_signed_weight_matrix(measures, class_count=2, feature_count=3) == [
        [0.0, 0.1, 0.2],
        [1.0, 1.1, 1.2],
    ]


def test_multiclass_output_head_sequence_ngspice_learns_one_hot_sequence(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    records = _one_hot_records()
    all_records = records + records + records
    sequence = ["initial_eval"] * 3 + ["train"] * 3 + ["final_eval"] * 3
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_output_head_sequence.cir",
        seq.generate_netlist(train_records=records, eval_records=records, class_count=3, feature_count=3),
        timeout=20.0,
    )
    rows = seq.rows_from_measures(all_records, measures, sequence=sequence, class_count=3)

    assert seq.accuracy(rows, "initial_eval") < 1.0
    assert seq.accuracy(rows, "final_eval") == 1.0
    assert min(row["score_margin_v"] for row in rows if row["sequence"] == "final_eval") > 1e-3
