from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


SPICE_DIR = Path(__file__).resolve().parents[1] / "spice"
sys.path.insert(0, str(SPICE_DIR))

import run_conductance_readout_primitive as readout  # noqa: E402
from run_device_sequential_training import run_netlist  # noqa: E402


def test_conductance_readout_primitive_emits_row_conductance_path() -> None:
    netlist = readout.generate_netlist(readout_case="positive")

    assert "\nB" not in netlist
    assert "Mactrow_n actrow fwd act 0 NMOS" in netlist
    assert "Mactrow_p actrow fwdn act vdd PMOS" in netlist
    assert "Movpos_cond actrow vwp score 0 NMOS W=64u L=180n" in netlist
    assert "Movneg_cond actrow vwn scoren 0 NMOS W=48u L=180n" in netlist
    assert ".meas tran score_margin PARAM='score_after-scoren_after'" in netlist


def test_conductance_readout_primitive_validation() -> None:
    with pytest.raises(ValueError, match="readout_case"):
        readout.generate_netlist(readout_case="bad")
    with pytest.raises(ValueError, match="readout_width"):
        readout.generate_netlist(readout_case="positive", readout_width=0.0)
    with pytest.raises(ValueError, match="timeout"):
        readout.main_for_test(["--timeout", "0"])


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("positive", 1.0),
        ("negative", -1.0),
        ("neutral", 0.0),
    ],
)
def test_conductance_readout_primitive_ngspice_score_polarity(
    tmp_path: Path,
    case: str,
    expected: float,
) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    measures = run_netlist(
        ngspice,
        tmp_path / f"conductance_readout_{case}.cir",
        readout.generate_netlist(readout_case=case, positive_weight=0.50, negative_weight=0.34),
        timeout=20.0,
    )

    margin = float(measures["score_margin"])
    if expected > 0.0:
        assert margin > 1e-3
    elif expected < 0.0:
        assert margin < -1e-3
    else:
        assert abs(margin) < 1e-3


def test_conductance_readout_primitive_ngspice_inactive_row_stays_quiet(tmp_path: Path) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    measures = run_netlist(
        ngspice,
        tmp_path / "conductance_readout_inactive.cir",
        readout.generate_netlist(readout_case="inactive", positive_weight=0.50, negative_weight=0.34),
        timeout=20.0,
    )

    assert abs(float(measures["score_margin"])) < 1e-3
    assert float(measures["score_common"]) < 5e-3
