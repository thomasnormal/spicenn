from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def test_device_sequential_training_script_help_runs_from_repo_root() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_device_sequential_training.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--timeout" in proc.stdout
    assert "--tag" in proc.stdout
    assert "--hidden-credit-mode" in proc.stdout
    assert "--assert-pass" in proc.stdout


def test_device_sequential_training_netlist_uses_no_behavioral_sources() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_sequential_training as seq

    netlist = seq.sequential_netlist(
        [{"vin": 1.0, "target": 1.1}, {"vin": 1.0, "target": 0.0}],
        whp=0.85,
        whn=0.25,
        vwp=0.55,
        vwn=0.25,
    )

    assert "\nB" not in netlist
    assert "learning_device_implementation" not in netlist
    assert "Vapplyn applyn 0 PWL" in netlist
    assert "Mrgp_pd rgp gvp 0 0 NSENSE" in netlist
    assert "Mvwp_up_p0 vdd rgp vwp_up vdd PMOS" in netlist


def test_direct_feedback_hidden_credit_does_not_gate_delta_with_readout_weight() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_sequential_training as seq

    netlist = seq.sequential_netlist(
        [{"vin": 1.0, "target": 1.1}, {"vin": 1.0, "target": 0.0}],
        whp=0.85,
        whn=0.25,
        vwp=0.55,
        vwn=0.25,
        hidden_credit_mode="direct_feedback",
    )

    hidden_delta_block = netlist.split("* Hidden delta:", 1)[1].split("* Readout gradient", 1)[0]
    assert "\nB" not in hidden_delta_block
    assert " vwp " not in hidden_delta_block
    assert " vwn " not in hidden_delta_block
    assert "Mhdp_d0 vdd dp hdp_d0 0 NSENSE" in hidden_delta_block
    assert "Mhdn_d0 vdd dn hdn_d0 0 NSENSE" in hidden_delta_block


def test_exact_backprop_hidden_credit_keeps_transistor_readout_weight_gates() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_sequential_training as seq

    netlist = seq.sequential_netlist(
        [{"vin": 1.0, "target": 1.1}, {"vin": 1.0, "target": 0.0}],
        whp=0.85,
        whn=0.25,
        vwp=0.55,
        vwn=0.25,
        hidden_credit_mode="exact_backprop",
    )

    hidden_delta_block = netlist.split("* Hidden delta:", 1)[1].split("* Readout gradient", 1)[0]
    assert "\nB" not in hidden_delta_block
    assert "Mhdp_a1 hdp_a0 vwp hdp_a1 0 NMOS" in hidden_delta_block
    assert "Mhdp_b1 hdp_b0 vwn hdp_b1 0 NMOS" in hidden_delta_block
