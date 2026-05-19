from __future__ import annotations

import re


MEAS_RE = re.compile(r"(?im)^\s*(?:y|vint)\s*=\s*([-+0-9.eE]+)")
Y_RE = re.compile(r"(?im)^\s*y\s*=\s*([-+0-9.eE]+)")


def parse_measure(stdout: str, name: str = "y") -> float:
    pat = re.compile(rf"(?im)^\s*{re.escape(name)}\s*=\s*([-+0-9.eE]+)")
    m = pat.search(stdout)
    if not m:
        raise ValueError(f"measurement {name!r} not found in ngspice output")
    return float(m.group(1))


def parse_output_high(stdout: str, vdd: float = 0.8) -> int:
    return int(parse_measure(stdout, "y") > 0.5 * vdd)

