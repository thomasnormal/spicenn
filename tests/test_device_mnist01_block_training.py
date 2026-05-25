from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPICE_DIR = ROOT / "spice"


def test_device_mnist01_block_script_help_runs_from_repo_root() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPICE_DIR / "run_device_mnist01_block_training.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--block-size" in proc.stdout
    assert "--stride" in proc.stdout
    assert "--channels" in proc.stdout
    assert "--input-rail-mode" in proc.stdout
    assert "--complement-rail-scale" in proc.stdout
    assert "--hidden-bias-positive-init" in proc.stdout
    assert "--assert-nonbehavioral" in proc.stdout


def test_block_topology_matches_target_10x10_b4_stride2_c2_shape() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    blocks, feature_count = block.block_topology(10, 4, 2, 2)

    assert len(blocks) == 16
    assert feature_count == 32
    assert len(blocks[0]) == 16
    assert blocks[0] == [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
    assert blocks[-1] == [66, 67, 68, 69, 76, 77, 78, 79, 86, 87, 88, 89, 96, 97, 98, 99]


def test_block_netlist_emits_per_pixel_trainable_caps_and_no_behavioral_sources() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 4
    weights = block.initial_block_weights(image_size, 2, 2, 1, seed=1)
    sample = {f"x{i}": 0.2 + 0.01 * i for i in range(image_size * image_size)}
    sample["target"] = 1.1
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=2,
        stride=2,
        channels=1,
        training_enabled=True,
    )

    assert "\nB" not in netlist
    assert netlist.count("Cwhp") == 16
    assert netlist.count("Cbhp") == 4
    assert "Vx15 x15 0 PWL" in netlist
    assert "Vx16 x16 0 PWL" not in netlist
    assert "Cwhp3_3 whp3_3 0 20f" in netlist
    assert "Cbhp3 bhp3 0 20f" in netlist
    assert "Mhpos3_3_x vdd x15 hp3_3_0 0 NMOS" in netlist
    assert "Mhbpos3_b vdd bhp3 hbp3_0 0 NMOS" in netlist
    assert "Mghp3_3_d ghp3_3_x hdp3 ghp3_3_d 0 NSENSE" in netlist
    assert "Mgbp3_d vdd hdp3 gbp3_d 0 NSENSE" in netlist
    assert "Mbhp3_up_a bhp3_up apply bhp3 0 NREL" in netlist
    assert "Movpos3_f op3_1 fwd score 0 NREL" in netlist
    assert "Mrelu_o vdd score out 0 NSENSE" in netlist


def test_block_netlist_can_emit_target_topology_without_behavioral_sources() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import run_device_mnist01_block_training as block

    image_size = 10
    weights = block.initial_block_weights(image_size, 4, 2, 2, seed=2)
    sample = {f"x{i}": 0.5 for i in range(image_size * image_size)}
    sample.update({f"nx{i}": 0.6 for i in range(image_size * image_size)})
    sample["target"] = 0.0
    netlist = block.block_netlist(
        [sample],
        weights,
        image_size=image_size,
        block_size=4,
        stride=2,
        channels=2,
        training_enabled=True,
    )

    assert "\nB" not in netlist
    assert netlist.count("Cwhp") == 32 * 16
    assert netlist.count("Cbhp") == 32
    assert "Vx99 x99 0 PWL" in netlist
    assert "Vnx99 nx99 0 PWL" in netlist
    assert "Cwhp31_15 whp31_15 0 20f" in netlist
    assert "Cbhp31 bhp31 0 20f" in netlist
    assert "Mhpos30_15_x vdd x99 hp30_15_0 0 NMOS" in netlist
    assert "Mhpos31_15_x vdd nx99 hp31_15_0 0 NMOS" in netlist
    assert "Mghp31_15_x vdd nx99 ghp31_15_x 0 NMOS" in netlist
    assert "Mhbpos31_b vdd bhp31 hbp31_0 0 NMOS" in netlist
    assert "Mbhp31_up_a bhp31_up apply bhp31 0 NREL" in netlist
    assert "Movpos31_f op31_1 fwd score 0 NREL" in netlist


def test_block_netlist_rejects_mismatched_weight_shape() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    sample = {f"x{i}": 0.5 for i in range(16)}
    sample["target"] = 1.1
    with pytest.raises(ValueError, match="does not match topology"):
        block.block_netlist(
            [sample],
            block.initial_block_weights(4, 2, 2, 1, seed=0),
            image_size=4,
            block_size=2,
            stride=2,
            channels=2,
            training_enabled=True,
        )


def test_block_netlist_rejects_missing_complement_rail_for_c2_default_mode() -> None:
    sys.path.insert(0, str(SPICE_DIR))
    import pytest
    import run_device_mnist01_block_training as block

    sample = {f"x{i}": 0.5 for i in range(16)}
    sample["target"] = 1.1
    with pytest.raises(ValueError, match="missing pixel rails: nx0"):
        block.block_netlist(
            [sample],
            block.initial_block_weights(4, 2, 2, 2, seed=0),
            image_size=4,
            block_size=2,
            stride=2,
            channels=2,
            training_enabled=True,
        )
