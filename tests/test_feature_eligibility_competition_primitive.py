from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_feature_eligibility_competition_primitive as featcomp  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def _gates(measures: dict[str, object], count: int) -> list[float]:
    return [float(measures[f"egate{idx}_after"]) for idx in range(count)]


def test_feature_eligibility_competition_primitive_emits_pairwise_loss_suppression() -> None:
    netlist = featcomp.generate_netlist(case="one_hot0")

    assert "\nB" not in netlist
    assert "Velig0 elig0 0 0.9" in netlist
    assert "Ce0_gt_e1_decision e0_gt_e1_decision 0 12f IC=1.2" in netlist
    assert "Me0_gt_e1_ab_dis_s e0_gt_e1_decision elig1 e0_gt_e1_ab_dn 0 NMOS" in netlist
    assert "Cegate0 egate0 0 8f IC=0" in netlist
    assert "Me1_loss_to_e0_mid_dec egate1 e0_gt_e1_decision e1_loss_to_e0_mid 0 NHIGH" in netlist
    assert ".meas tran egate0_after FIND V(egate0) AT=4.80n" in netlist


def test_feature_eligibility_competition_primitive_validation() -> None:
    with pytest.raises(ValueError, match="case"):
        featcomp.generate_netlist(case="missing")
    with pytest.raises(ValueError, match="gate_mode"):
        featcomp.generate_netlist(case="one_hot0", gate_mode="missing")
    with pytest.raises(ValueError, match="at least two"):
        featcomp.generate_netlist(eligibility_values=(0.1,))
    with pytest.raises(ValueError, match="supply rails"):
        featcomp.generate_netlist(eligibility_values=(0.1, 1.3))
    with pytest.raises(ValueError, match="positive"):
        featcomp.generate_netlist(case="one_hot0", pairwise_width_u=0)
    with pytest.raises(ValueError, match="positive"):
        featcomp.main_for_test(["--pairwise-width", "0"])
    with pytest.raises(ValueError, match="max-active"):
        featcomp.main_for_test(["--max-active", "-1"])


def test_feature_eligibility_competition_primitive_emits_rank_preserving_gate() -> None:
    netlist = featcomp.generate_netlist(
        case="unique_dense0",
        gate_mode="rank",
        gate_loss_width_u=0.50,
        gate_capacitance_f=50.0,
    )

    assert "\nB" not in netlist
    assert "Cegate0 egate0 0 50f IC=0" in netlist
    assert "Me0_rank_loss_to_e1_mid_dec egate0 e1_gt_e0_decision e0_rank_loss_to_e1_mid 0 NHIGH W=0.5u" in netlist


@pytest.mark.parametrize(
    ("case", "expected_idx", "feature_count"),
    [
        ("one_hot0", 0, 3),
        ("one_hot1", 1, 3),
        ("unique_dense0", 0, 5),
        ("unique_dense4", 4, 6),
    ],
)
def test_feature_eligibility_competition_ngspice_selects_unique_maximum(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    expected_idx: int,
    feature_count: int,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"feature_competition_{case}.cir",
        featcomp.generate_netlist(case=case),
        timeout=30.0,
    )

    gates = _gates(measures, feature_count)
    assert gates[expected_idx] > 0.70
    assert max(value for idx, value in enumerate(gates) if idx != expected_idx) < 0.20


def test_feature_eligibility_competition_ngspice_rotated_unique_maximum_is_not_index_special(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_competition_rotated_unique.cir",
        featcomp.generate_netlist(eligibility_values=(0.080, 0.360, 0.541, 0.313, 0.247)),
        timeout=30.0,
    )

    gates = _gates(measures, 5)
    assert gates[2] > 0.70
    assert max(value for idx, value in enumerate(gates) if idx != 2) < 0.20


def test_feature_eligibility_competition_ngspice_flat_dense_does_not_enable_all_writers(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_competition_flat_dense.cir",
        featcomp.generate_netlist(case="flat_dense"),
        timeout=30.0,
    )

    gates = _gates(measures, 5)
    assert sum(value > 0.60 for value in gates) < 5


@pytest.mark.parametrize(
    ("case", "feature_count"),
    [
        ("fixed8_like0", 9),
        ("fixed8_like1", 9),
        ("fixed8_like2", 9),
    ],
)
def test_feature_eligibility_competition_ngspice_fixed8_like_rows_are_sparsified(
    tmp_path: Path,
    ngspice_path: str,
    case: str,
    feature_count: int,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / f"feature_competition_{case}.cir",
        featcomp.generate_netlist(case=case),
        timeout=60.0,
    )

    gates = _gates(measures, feature_count)
    assert sum(value > 0.60 for value in gates) <= 2
    assert max(gates) - min(gates) > 0.20


def test_feature_eligibility_competition_ngspice_rank_gate_preserves_multiple_features(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_competition_rank_fixed8_like2.cir",
        featcomp.generate_netlist(
            case="fixed8_like2",
            gate_mode="rank",
            gate_loss_width_u=0.50,
            gate_capacitance_f=50.0,
        ),
        timeout=60.0,
    )

    gates = _gates(measures, 9)
    active = sum(value > 0.60 for value in gates)
    assert 1 <= active <= 3
    assert max(gates[idx] for idx in (1, 4, 6, 7, 8)) > 1.0
    assert min(gates) < max(gates) - 0.25


def test_feature_eligibility_competition_ngspice_rank_gate_tracks_unique_dense_order(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "feature_competition_rank_unique_dense0.cir",
        featcomp.generate_netlist(
            case="unique_dense0",
            gate_mode="rank",
            gate_loss_width_u=0.50,
            gate_capacitance_f=50.0,
        ),
        timeout=30.0,
    )

    gates = _gates(measures, 5)
    assert gates[0] > 0.90
    assert gates[3] > 0.30
    assert gates[1] > gates[2]
    assert gates[4] < 0.15
    assert gates[2] < 0.15
