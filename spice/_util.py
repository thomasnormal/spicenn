"""Shared helpers used by the run_device_*.py scripts.

Kept intentionally small: only utilities that are duplicated verbatim across
multiple run scripts belong here.  Anything component-shaped (synapses,
neurons, optimisers, error cells) should live in :mod:`spicenn` instead.
"""

from __future__ import annotations

import re

# Generic name=value measurement line emitted by ngspice ``.meas`` blocks.
# Captures the name and the numeric value separately so the result can be
# bucketed into a ``dict``.  Matched case-insensitively and with multiline
# anchors so it works on both stdout and stderr blobs.
MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")


def parse_measures(text: str) -> dict[str, float]:
    """Pull every ``name = value`` measurement out of an ngspice log."""
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


__all__ = ["MEAS_RE", "parse_measures"]
