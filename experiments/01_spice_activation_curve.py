from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--points", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=1.5e-3)
    ap.add_argument("--span", type=float, default=6e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "spice/run_spice_sweep.py"),
        "--trials",
        str(args.trials),
        "--points",
        str(args.points),
        "--sigma",
        str(args.sigma),
        "--span",
        str(args.span),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()

