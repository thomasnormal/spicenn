from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_multiclass_block_smoke as smoke  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_multiclass_block_smoke_emits_row_pulsed_split_rail_class_local_path() -> None:
    netlist = smoke.generate_netlist(class_count=3, target_class=0)

    assert "\nB" not in netlist
    assert "Vrow row0 0 PULSE(0 0.85 1.0n" in netlist
    assert "Mhidden_pos row0 whp pre_p 0 NMOS" in netlist
    assert "Mhidden_neg row0 whn pre_n 0 NMOS" in netlist
    assert "Mact_p vdd pre_p act_raw 0 NREL" in netlist
    assert "Mact_n act_raw pre_n 0 0 NREL" in netlist
    assert "Mact_store_n act0 samp act_raw 0 NMOS" in netlist
    assert "Mact_grad_n actg0 samp act_raw 0 NMOS" in netlist
    assert "Melig_n elig0 samp pre_p 0 NMOS" in netlist
    assert "Mc0_f0_pos_cond actrow0 c0_vwp0 c0_score 0 NMOS" in netlist
    assert "Mc1_f0_neg_cond actrow0 c1_vwn0 c1_scoren 0 NMOS" in netlist
    assert "Mc0_f0_vwp_up_p0 c0_f0_vwp_up c0_rgp0 vwhi_ref vdd PMOS" in netlist
    assert " score " not in netlist
    assert " vwp0 " not in netlist


def test_multiclass_block_smoke_validation() -> None:
    with pytest.raises(ValueError, match="class_count"):
        smoke.generate_netlist(class_count=1)
    with pytest.raises(ValueError, match="target_class"):
        smoke.generate_netlist(class_count=3, target_class=3)
    with pytest.raises(ValueError, match="hidden_width"):
        smoke.generate_netlist(hidden_width_u=0.0)
    with pytest.raises(ValueError, match="supply rails"):
        smoke.generate_netlist(input_v=1.3)
    with pytest.raises(ValueError, match="class-count"):
        smoke.main_for_test(["--class-count", "1"])
    with pytest.raises(ValueError, match="target-class"):
        smoke.main_for_test(["--class-count", "3", "--target-class", "3"])
    with pytest.raises(ValueError, match="min-score-margin"):
        smoke.main_for_test(["--min-score-margin", "-1"])


def test_multiclass_block_smoke_ngspice_forward_scores_and_local_updates(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_block_smoke.cir",
        smoke.generate_netlist(class_count=3, target_class=0),
        timeout=30.0,
    )

    assert float(measures["pre_margin"]) > 20e-3
    assert float(measures["act_after"]) > 20e-3
    assert float(measures["eligibility_after"]) > 20e-3
    assert float(measures["actrow_after"]) > 1e-3

    c0_score = float(measures["c0_score_net"])
    c1_score = float(measures["c1_score_net"])
    c2_score = float(measures["c2_score_net"])
    assert c0_score > 2e-3
    assert c1_score < -2e-3
    assert abs(c2_score) < 5e-3
    assert c0_score - max(c1_score, c2_score) > 2e-3

    assert float(measures["c0_signed_delta"]) > 1e-3
    assert float(measures["c1_signed_delta"]) < -1e-3
    assert float(measures["c2_signed_delta"]) < -1e-3
