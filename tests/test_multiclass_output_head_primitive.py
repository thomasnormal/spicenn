from __future__ import annotations

import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_multiclass_output_head_primitive as head  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_multiclass_output_head_primitive_emits_class_local_nodes() -> None:
    netlist = head.generate_netlist(class_count=3, target_class=1)

    assert "\nB" not in netlist
    assert "Cc0_vwp0 c0_vwp0 0 20f IC=0.36" in netlist
    assert "Cc1_vwp0 c1_vwp0 0 20f IC=0.36" in netlist
    assert "Vc1_targetp c1_targetp 0 PULSE(0 1.1" in netlist
    assert "Vc0_targetn c0_targetn 0 PULSE(0 1.1" in netlist
    assert "Mc1_f0_vwp_up_p0 c1_f0_vwp_up c1_rgp0 vwhi_ref vdd PMOS" in netlist
    assert "Mc0_f0_vwp_dn_g c0_f0_vwp_dn c0_gvn0 vwlo_ref 0 NSENSE" in netlist
    assert "Mc2_f0_pos_cond actrow0 c2_vwp0 c2_score 0 NMOS" in netlist
    assert " score " not in netlist
    assert " dp " not in netlist
    assert " targetp " not in netlist


def test_multiclass_output_head_readout_emitter_can_diode_isolate_score_branch() -> None:
    lines = head.class_local_readout_forward_lines(class_idx=2, feature_idx=3, isolation="diode")
    netlist = "\n".join(lines)

    assert "\nB" not in netlist
    assert "Cc2_f3_midp c2_f3_midp 0 0.1f IC=0" in netlist
    assert "Rc2_f3_midn c2_f3_midn 0 1G" in netlist
    assert "Mc2_f3_pos_cond actrow3 c2_vwp3 c2_f3_midp 0 NMOS W=64u L=180n" in netlist
    assert "Mc2_f3_pos_diode c2_f3_midp c2_f3_midp c2_score 0 NSENSE W=64u L=180n" in netlist
    assert "Mc2_f3_neg_diode c2_f3_midn c2_f3_midn c2_scoren 0 NSENSE W=48u L=180n" in netlist
    assert "actrow3 c2_vwp3 c2_score 0 NMOS" not in netlist


def test_multiclass_output_head_primitive_validation() -> None:
    with pytest.raises(ValueError, match="isolation"):
        head.class_local_readout_forward_lines(class_idx=0, feature_idx=0, isolation="missing")
    with pytest.raises(ValueError, match="class_count"):
        head.generate_netlist(class_count=1)
    with pytest.raises(ValueError, match="target_class"):
        head.generate_netlist(class_count=2, target_class=2)
    with pytest.raises(ValueError, match="supply rails"):
        head.generate_netlist(initial_positive=1.3)
    with pytest.raises(ValueError, match="class-count"):
        head.main_for_test(["--class-count", "1"])
    with pytest.raises(ValueError, match="target-class"):
        head.main_for_test(["--class-count", "3", "--target-class", "3"])
    with pytest.raises(ValueError, match="min-score-margin"):
        head.main_for_test(["--min-score-margin", "-1"])


def test_multiclass_output_head_primitive_ngspice_updates_target_and_nontarget_oppositely(
    tmp_path: Path,
    ngspice_path: str,
) -> None:
    measures = run_netlist(
        ngspice_path,
        tmp_path / "multiclass_output_head.cir",
        head.generate_netlist(class_count=3, target_class=1),
        timeout=20.0,
    )

    assert float(measures["c1_signed_delta"]) > 5e-3
    assert float(measures["c0_signed_delta"]) < -5e-3
    assert float(measures["c2_signed_delta"]) < -5e-3
    assert float(measures["c1_score_net"]) > 5e-3
    assert abs(float(measures["c0_score_net"])) < 1e-3
    assert abs(float(measures["c2_score_net"])) < 1e-3
    assert float(measures["c1_score_net"]) - float(measures["c0_score_net"]) > 5e-3
    assert float(measures["c1_score_net"]) - float(measures["c2_score_net"]) > 5e-3
