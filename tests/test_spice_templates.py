from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_shared_phase_local_feature_cell_template_runs_in_ngspice() -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    template = ROOT / "spice/templates/shared_phase_local_feature_cell_smoke.cir"
    proc = subprocess.run(
        [ngspice, "-b", str(template)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=True,
    )
    output = proc.stdout.lower()

    assert "no. of data rows" in output
    assert "pre_forward" in output
    assert "act_forward" in output
    assert "score_forward" in output
    assert "hdp_after_bwd" in output


def test_shared_phase_local_feature_cell_full_template_runs_in_ngspice() -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice is not installed")

    template = ROOT / "spice/templates/shared_phase_local_feature_cell_full_smoke.cir"
    proc = subprocess.run(
        [ngspice, "-b", str(template)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=True,
    )
    output = proc.stdout.lower()

    assert "no. of data rows" in output
    assert "pre_forward" in output
    assert "act_forward" in output
    assert "score_forward" in output
    assert "hdp_after_bwd" in output
    assert "hdn_after_bwd" in output
