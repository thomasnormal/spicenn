from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"
if str(SPICE_DIR) not in sys.path:
    sys.path.insert(0, str(SPICE_DIR))

import run_device_spicenn_readout_update_smoke as update_smoke  # noqa: E402


def test_spicenn_readout_update_smoke_netlist_contains_error_and_local_writer() -> None:
    text = update_smoke.netlist(update_width_u=0.004)

    assert "Merr_pos_dp_t0 vdd target_pos err_pos_dp_t 0 NSENSE W=40u L=180n" in text
    assert "Merr_neg_dn_sp1 err_neg_dn_sp err dn_neg_0 0 NSENSE W=32u L=180n" in text
    assert "Mvw_pos_0_0p_ch_d vw_pos_0_0p_ch_a dp_pos_0 vw_pos_0_0p 0 NSENSE W=0.004u" in text
    assert "Mvw_neg_0_0p_flow_d vw_neg_0_0p_flow_a dn_neg_0 wlow 0 NSENSE W=0.004u" in text
    assert ".meas tran pos_dp_err FIND V(dp_pos_0) AT=1.55n" in text
    assert ".meas tran neg_dn_err FIND V(dn_neg_0) AT=1.55n" in text


def test_spicenn_readout_update_smoke_derived_sign_metrics() -> None:
    measures = update_smoke.add_derived(
        {
            "pos_p_before": 0.5,
            "pos_n_before": 0.5,
            "pos_p_after": 0.6,
            "pos_n_after": 0.4,
            "neg_p_before": 0.5,
            "neg_n_before": 0.5,
            "neg_p_after": 0.4,
            "neg_n_after": 0.6,
        }
    )

    assert measures["pos_signed_delta"] == pytest.approx(0.2)
    assert measures["neg_signed_delta"] == pytest.approx(-0.2)
    assert measures["pos_common_delta"] == pytest.approx(0.0)
    assert measures["neg_common_delta"] == pytest.approx(0.0)


def test_spicenn_readout_update_smoke_transient_signs_match_error_direction(tmp_path: Path) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    measures = update_smoke.run_netlist(
        ngspice,
        tmp_path / "spicenn_readout_update_smoke.cir",
        update_smoke.netlist(update_width_u=0.004),
        timeout=20,
    )

    assert measures["pos_dp_err"] > 0.8
    assert measures["pos_dn_err"] < 0.05
    assert measures["neg_dp_err"] < 0.05
    assert measures["neg_dn_err"] > 0.8
    assert measures["pos_signed_delta"] > 0.05
    assert measures["neg_signed_delta"] < -0.05
