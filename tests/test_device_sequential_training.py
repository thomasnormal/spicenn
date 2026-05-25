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
    assert "--output-driver-model" in proc.stdout
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


def test_output_driver_model_selects_transistor_source_follower() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_sequential_training as seq

    sense_netlist = seq.sequential_netlist(
        [{"vin": 0.8, "target": 1.1}],
        whp=0.85,
        whn=0.25,
        vwp=0.55,
        vwn=0.25,
        output_driver_model="sense",
    )
    nrel_netlist = seq.sequential_netlist(
        [{"vin": 0.8, "target": 1.1}],
        whp=0.85,
        whn=0.25,
        vwp=0.55,
        vwn=0.25,
        output_driver_model="nrel",
    )

    assert "Mrelu_o vdd score out 0 NSENSE W=24u L=180n" in sense_netlist
    assert "Mrelu_o vdd score out 0 NREL W=24u L=180n" in nrel_netlist
    assert "\nB" not in sense_netlist


def test_eval_mode_disables_training_phase_pulses_without_removing_forward_path() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_sequential_training as seq

    netlist = seq.sequential_netlist(
        [{"vin": 0.8, "target": 1.1}, {"vin": 0.4, "target": 0.0}],
        whp=0.85,
        whn=0.25,
        vwp=0.55,
        vwn=0.25,
        training_enabled=False,
    )

    assert "\nB" not in netlist
    assert "Vfwd fwd 0 PWL" in netlist
    assert "Verr err 0 PWL(0n 0 32n 0)" in netlist
    assert "Vacc acc 0 PWL(0n 0 32n 0)" in netlist
    assert "Vapply apply 0 PWL(0n 0 32n 0)" in netlist
    assert "Vapplyn applyn 0 PWL(0n 1.2 32n 1.2)" in netlist
