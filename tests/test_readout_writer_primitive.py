from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_readout_writer_primitive as writer  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_readout_writer_primitive_emits_bounded_reference_writer() -> None:
    netlist = writer.generate_netlist(update_mode="positive", topology="bounded-ref", update_scale=0.1)

    assert "\nB" not in netlist
    assert "Vvwhi_ref vwhi_ref 0 0.51" in netlist
    assert "Vvwlo_ref vwlo_ref 0 0.19" in netlist
    assert "Mgvp0_a vdd act gvp0_a 0 NSENSE W=24u L=180n" in netlist
    assert "Mrgp0_pd rgp0 gvp0 0 0 NSENSE W=16u L=180n" in netlist
    assert "Mvwp0_up_p0 vwp0_up rgp0 vwhi_ref vdd PMOS W=0.8u L=180n" in netlist
    assert "Mvwn0_dn_g vwn0_dn gvp0 vwlo_ref 0 NSENSE W=0.2u L=180n" in netlist
    assert "Mvwp0_up_p0 vwp0_up rgp0 vdd vdd PMOS" not in netlist


def test_readout_writer_primitive_can_restore_weak_gradient_gate() -> None:
    netlist = writer.generate_netlist(
        update_mode="positive",
        topology="bounded-ref",
        gradient_gate_topology="restored",
        gate_amplitude=0.04,
        gate_restore_width=32.0,
    )

    assert "\nB" not in netlist
    assert "Velig elig 0 PULSE(0 0.04 0.5n 10p 10p 3.0n 30n)" in netlist
    assert "Cegon act 0 10f IC=0" in netlist
    assert "Megate_pd egate elig 0 0 NSENSE W=32u L=180n" in netlist
    assert "Megon_p act egate vdd vdd PMOS W=32u L=180n" in netlist
    assert "Mgvp0_a vdd act gvp0_a 0 NSENSE W=24u L=180n" in netlist


def test_readout_writer_primitive_keeps_legacy_rail_writer_available() -> None:
    netlist = writer.generate_netlist(update_mode="negative", topology="rail", update_scale=0.1)

    assert "Vvwhi_ref" not in netlist
    assert "Vvwlo_ref" not in netlist
    assert "Mvwn0_up_p0 vwn0_up rgn0 vdd vdd PMOS W=0.8u L=180n" in netlist
    assert "Mvwp0_dn_g vwp0_dn gvn0 0 0 NSENSE W=0.2u L=180n" in netlist


def test_readout_writer_primitive_classification() -> None:
    assert writer.classify_sign(0.05, 1.0, min_abs_delta=0.01) == "aligned"
    assert writer.classify_sign(-0.05, -1.0, min_abs_delta=0.01) == "aligned"
    assert writer.classify_sign(0.001, 0.0, min_abs_delta=0.01) == "dead_zone"
    assert writer.classify_sign(-0.05, 1.0, min_abs_delta=0.01) == "wrong_sign"
    row = {"update_mode": "positive", "signed_delta": 0.04, "common_delta": 0.02}
    assert writer.classify_row(row, min_abs_delta=0.01, max_common_delta=0.03) == {
        "update_classification": "aligned",
        "common_mode_classification": "bounded",
    }


def test_readout_writer_primitive_validation() -> None:
    with pytest.raises(ValueError, match="update_mode"):
        writer.generate_netlist(update_mode="bad")
    with pytest.raises(ValueError, match="bounded-ref update references"):
        writer.generate_netlist(update_mode="positive", topology="bounded-ref", negative_ref=0.05, update_span=0.10)
    with pytest.raises(ValueError, match="timeout"):
        writer.main_for_test(["--timeout", "0"])
    with pytest.raises(ValueError, match="error-amplitude"):
        writer.main_for_test(["--error-amplitude", "1.3"])
    with pytest.raises(ValueError, match="gradient-restore-width"):
        writer.main_for_test(["--gradient-restore-width", "0"])
    with pytest.raises(ValueError, match="gate-amplitude"):
        writer.main_for_test(["--gate-amplitude", "1.3"])
    with pytest.raises(ValueError, match="gate-restore-width"):
        writer.main_for_test(["--gate-restore-width", "0"])


@pytest.mark.parametrize(
    ("mode", "expected_sign"),
    [
        ("positive", 1.0),
        ("negative", -1.0),
        ("none", 0.0),
    ],
)
def test_readout_writer_primitive_ngspice_sign_and_common_mode(
    tmp_path: Path,
    mode: str,
    expected_sign: float,
) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    measures = run_netlist(
        ngspice,
        tmp_path / f"readout_writer_{mode}.cir",
        writer.generate_netlist(update_mode=mode, topology="bounded-ref", update_scale=0.10),
        timeout=20.0,
    )

    signed_delta = float(measures["signed_delta"])
    common_delta = abs(float(measures["common_delta"]))
    if expected_sign > 0.0:
        assert signed_delta > 0.05
    elif expected_sign < 0.0:
        assert signed_delta < -0.05
    else:
        assert abs(signed_delta) < 1e-3
    assert common_delta < 0.05


def test_readout_writer_primitive_ngspice_weak_error_starves_default_writer(
    tmp_path: Path,
) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    strong = run_netlist(
        ngspice,
        tmp_path / "readout_writer_strong_error.cir",
        writer.generate_netlist(update_mode="positive", topology="bounded-ref", update_scale=0.10, error_amplitude=1.2),
        timeout=20.0,
    )
    weak = run_netlist(
        ngspice,
        tmp_path / "readout_writer_weak_error.cir",
        writer.generate_netlist(update_mode="positive", topology="bounded-ref", update_scale=0.10, error_amplitude=0.08),
        timeout=20.0,
    )

    assert float(strong["gradient_margin"]) > 0.5
    assert float(strong["signed_delta"]) > 0.05
    assert float(weak["gradient_margin"]) > 0.025
    assert abs(float(weak["signed_delta"])) < 1e-3


def test_readout_writer_primitive_ngspice_stronger_restore_uses_weak_error(
    tmp_path: Path,
) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    measures = run_netlist(
        ngspice,
        tmp_path / "readout_writer_weak_error_stronger_restore.cir",
        writer.generate_netlist(
            update_mode="positive",
            topology="bounded-ref",
            update_scale=0.05,
            error_amplitude=0.08,
            gradient_restore_width=32.0,
        ),
        timeout=20.0,
    )

    assert float(measures["gradient_margin"]) > 0.025
    assert float(measures["signed_delta"]) > 0.05
    assert abs(float(measures["common_delta"])) < 0.05


def test_readout_writer_primitive_ngspice_restored_gate_uses_weak_eligibility(
    tmp_path: Path,
) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    direct = run_netlist(
        ngspice,
        tmp_path / "readout_writer_weak_direct_gate.cir",
        writer.generate_netlist(
            update_mode="positive",
            topology="bounded-ref",
            update_scale=0.05,
            error_amplitude=0.08,
            gradient_restore_width=32.0,
            gate_amplitude=0.04,
        ),
        timeout=20.0,
    )
    restored = run_netlist(
        ngspice,
        tmp_path / "readout_writer_weak_restored_gate.cir",
        writer.generate_netlist(
            update_mode="positive",
            topology="bounded-ref",
            update_scale=0.05,
            error_amplitude=0.08,
            gradient_restore_width=32.0,
            gradient_gate_topology="restored",
            gate_amplitude=0.04,
            gate_restore_width=32.0,
        ),
        timeout=20.0,
    )

    assert float(direct["gate_before_apply"]) < 0.05
    assert float(direct["gradient_margin"]) < 0.025
    assert abs(float(direct["signed_delta"])) < 1e-3

    assert float(restored["gate_before_apply"]) > 0.5
    assert float(restored["gradient_margin"]) > 0.025
    assert float(restored["signed_delta"]) > 0.05
    assert abs(float(restored["common_delta"])) < 0.05
