from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_xor_output_normalizer_primitives as xor_norm  # noqa: E402


def _run_case(
    tmp_path: Path,
    ngspice_path: str,
    *,
    approach: str,
    evidence: tuple[float, float],
    target: int = 0,
    name: str = "case",
    **kwargs: float,
) -> dict[str, float]:
    netlist = xor_norm.generate_netlist(
        approach=approach,
        evidence=evidence,
        target=target,
        **kwargs,
    )
    assert "\nB" not in netlist
    return xor_norm.run_netlist(
        ngspice_path,
        tmp_path / f"{approach}_{name}.cir",
        netlist,
        timeout=30.0,
    )


def test_xor_output_normalizer_primitives_define_real_subcircuits() -> None:
    text = xor_norm.normalizer_subcircuits()

    assert "\nB" not in text
    assert ".subckt xor_prob_conductance_divider2 " in text
    assert "Icdref vdd norm PULSE" in text
    assert "Mcd0 d0 gcond ps0 vss NSENSE" in text
    assert ".subckt xor_prob_soft_wta2 " in text
    assert "Mswta0_supp_phi swta0_supp phip vss vss NMOS" in text
    assert ".subckt xor_label_routed_descent2 " in text
    assert "Me0p_p e0p_a p1 e0p_b vss NSENSE" in text
    assert "Me1p_p e1p_a p0 e1p_b vss NSENSE" in text


def test_xor_output_normalizer_primitive_validation() -> None:
    with pytest.raises(ValueError, match="approach"):
        xor_norm.generate_netlist(approach="bad", evidence=(1.0, 1.0), target=0)
    with pytest.raises(ValueError, match="two values"):
        xor_norm.generate_netlist(approach="conductance-divider", evidence=(1.0,), target=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        xor_norm.generate_netlist(approach="conductance-divider", evidence=(-1.0, 1.0), target=0)
    with pytest.raises(ValueError, match="target"):
        xor_norm.generate_netlist(approach="conductance-divider", evidence=(1.0, 1.0), target=2)
    with pytest.raises(ValueError, match="conductance_floor"):
        xor_norm.generate_netlist(
            approach="conductance-divider",
            evidence=(1.0, 1.0),
            target=0,
            conductance_floor=-0.1,
        )


@pytest.mark.parametrize(
    ("evidence", "floor"),
    [
        ((1.0, 1.0), 0.0),
        ((2.0, 1.0), 0.0),
        ((4.0, 2.0), 0.0),
        ((0.0, 0.0), 1.00),
        ((0.0, 3.0), 0.20),
    ],
)
def test_conductance_divider_ngspice_probability_ratio_tracks_conductance_ratio(
    tmp_path: Path,
    ngspice_path: str,
    evidence: tuple[float, float],
    floor: float,
) -> None:
    measures = _run_case(
        tmp_path,
        ngspice_path,
        approach="conductance-divider",
        evidence=evidence,
        conductance_floor=floor,
        name=f"ratio_{evidence[0]}_{evidence[1]}_{floor}",
    )

    i0 = abs(measures["i0_raw"])
    i1 = abs(measures["i1_raw"])
    expected_p0 = (evidence[0] + floor) / (evidence[0] + evidence[1] + 2.0 * floor)
    measured_i0_frac = i0 / (i0 + i1)

    assert i0 + i1 == pytest.approx(1.0e-6, rel=0.08)
    assert measures["p_sum"] > 0.05
    assert 0.0 < measures["norm_after"] < 0.9
    assert measured_i0_frac == pytest.approx(expected_p0, abs=0.08)
    if abs(expected_p0 - 0.5) > 0.05:
        assert (measures["p0_frac"] - 0.5) * (expected_p0 - 0.5) > 0.0


def test_conductance_divider_ngspice_common_scale_drift_is_bounded_and_ordered(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    base = _run_case(
        tmp_path,
        ngspice_path,
        approach="conductance-divider",
        evidence=(1.0, 3.0),
        conductance_floor=0.0,
        conductance_scale=1.0,
        name="scale1",
    )
    scaled = _run_case(
        tmp_path,
        ngspice_path,
        approach="conductance-divider",
        evidence=(1.0, 3.0),
        conductance_floor=0.0,
        conductance_scale=4.0,
        name="scale4",
    )

    base_i0 = abs(base["i0_raw"])
    base_i1 = abs(base["i1_raw"])
    scaled_i0 = abs(scaled["i0_raw"])
    scaled_i1 = abs(scaled["i1_raw"])
    base_i0_frac = base_i0 / (base_i0 + base_i1)
    scaled_i0_frac = scaled_i0 / (scaled_i0 + scaled_i1)

    assert base_i0_frac == pytest.approx(0.25, abs=0.08)
    assert scaled_i0_frac > base_i0_frac
    assert scaled_i0_frac < 0.45
    assert scaled["norm_after"] < base["norm_after"]
    assert scaled_i0 + scaled_i1 == pytest.approx(base_i0 + base_i1, rel=0.08)
    assert base["p0_frac"] < base["p1_frac"]
    assert scaled["p0_frac"] < scaled["p1_frac"]


def test_conductance_divider_ngspice_label_routing_is_cross_entropy_shaped(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    target0_loses = _run_case(
        tmp_path,
        ngspice_path,
        approach="conductance-divider",
        evidence=(1.0, 3.0),
        target=0,
        conductance_floor=0.0,
        name="target0_loses",
    )
    target0_clear = _run_case(
        tmp_path,
        ngspice_path,
        approach="conductance-divider",
        evidence=(3.0, 1.0),
        target=0,
        conductance_floor=0.0,
        name="target0_clear",
    )
    target1_loses = _run_case(
        tmp_path,
        ngspice_path,
        approach="conductance-divider",
        evidence=(3.0, 1.0),
        target=1,
        conductance_floor=0.0,
        name="target1_loses",
    )

    assert target0_loses["e0_diff"] > 0.01
    assert target0_loses["e1_diff"] < -0.01
    assert target0_loses["e0_diff"] > target0_clear["e0_diff"] + 0.01
    assert target1_loses["e1_diff"] > 0.01
    assert target1_loses["e0_diff"] < -0.01
    assert target1_loses["e1_diff"] == pytest.approx(target0_loses["e0_diff"], rel=0.30)


def test_conductance_divider_ngspice_tie_is_symmetric_and_bounded(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    target0 = _run_case(
        tmp_path,
        ngspice_path,
        approach="conductance-divider",
        evidence=(1.0, 1.0),
        target=0,
        conductance_floor=0.1,
        name="tie_target0",
    )
    target1 = _run_case(
        tmp_path,
        ngspice_path,
        approach="conductance-divider",
        evidence=(1.0, 1.0),
        target=1,
        conductance_floor=0.1,
        name="tie_target1",
    )

    assert target0["p0_frac"] == pytest.approx(0.5, abs=0.04)
    assert target0["e0_diff"] > 0.004
    assert target0["e1_diff"] < -0.004
    assert target1["e1_diff"] > 0.004
    assert target1["e0_diff"] < -0.004
    for measures in (target0, target1):
        for name in ("e0p_after", "e0n_after", "e1p_after", "e1n_after"):
            assert 0.0 <= measures[name] < 0.95


@pytest.mark.parametrize(
    "approach",
    ["conductance-divider", "soft-wta"],
)
def test_output_normalizers_ngspice_target_pressure_tracks_wrong_probability(
    tmp_path: Path,
    ngspice_path: str,
    approach: str,
) -> None:
    loses = _run_case(
        tmp_path,
        ngspice_path,
        approach=approach,
        evidence=(1.0, 3.0),
        target=0,
        name="pressure_loses",
        conductance_floor=0.0,
    )
    clear = _run_case(
        tmp_path,
        ngspice_path,
        approach=approach,
        evidence=(3.0, 1.0),
        target=0,
        name="pressure_clear",
        conductance_floor=0.0,
    )

    assert loses["p1_frac"] > clear["p1_frac"] + 0.20
    assert loses["e0_diff"] > clear["e0_diff"] + 0.01
    assert loses["e0_diff"] > 0.01
    assert loses["e1_diff"] < -0.01


def test_soft_wta_ngspice_competition_is_monotonic_and_permutation_symmetric(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    high1 = _run_case(
        tmp_path,
        ngspice_path,
        approach="soft-wta",
        evidence=(1.0, 3.0),
        target=0,
        name="high1",
    )
    high0 = _run_case(
        tmp_path,
        ngspice_path,
        approach="soft-wta",
        evidence=(3.0, 1.0),
        target=1,
        name="high0",
    )
    tie = _run_case(
        tmp_path,
        ngspice_path,
        approach="soft-wta",
        evidence=(2.0, 2.0),
        target=0,
        name="tie",
    )

    assert high1["p1_frac"] > 0.60
    assert high0["p0_frac"] > 0.60
    assert tie["p0_frac"] == pytest.approx(0.5, abs=0.06)
    assert high1["p1_frac"] == pytest.approx(high0["p0_frac"], rel=0.20)
    assert high1["e0_diff"] > 0.02
    assert high1["e1_diff"] < -0.02
    assert high0["e1_diff"] > 0.02
    assert high0["e0_diff"] < -0.02


def test_soft_wta_ngspice_preserves_order_across_score_common_mode(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    low_common = _run_case(
        tmp_path,
        ngspice_path,
        approach="soft-wta",
        evidence=(1.0, 2.0),
        target=0,
        name="low_common",
        conductance_scale=1.0,
    )
    high_common = _run_case(
        tmp_path,
        ngspice_path,
        approach="soft-wta",
        evidence=(1.0, 2.0),
        target=0,
        name="high_common",
        conductance_scale=1.8,
    )

    assert low_common["p1_frac"] > low_common["p0_frac"] + 0.08
    assert high_common["p1_frac"] > high_common["p0_frac"] + 0.08
    assert low_common["e0_diff"] > 0.0
    assert high_common["e0_diff"] > 0.0
    assert high_common["p_sum"] < 1.2


def test_soft_wta_ngspice_near_zero_scores_remain_balanced_not_full_rail(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    near_zero = _run_case(
        tmp_path,
        ngspice_path,
        approach="soft-wta",
        evidence=(0.0, 0.0),
        target=0,
        name="near_zero",
        soft_score_base_v=0.06,
        soft_score_step_v=0.04,
    )
    normal_tie = _run_case(
        tmp_path,
        ngspice_path,
        approach="soft-wta",
        evidence=(2.0, 2.0),
        target=0,
        name="normal_tie",
    )

    assert near_zero["p0_frac"] == pytest.approx(0.5, abs=0.08)
    assert near_zero["p_sum"] < normal_tie["p_sum"]
    assert near_zero["p_sum"] < 0.35
    assert near_zero["e0p_after"] < 0.75
    assert near_zero["e1n_after"] < 0.75
