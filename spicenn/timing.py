"""Shared timing primitives for the SPICENN run scripts.

A SPICENN training cycle is partitioned into fixed-width windows (reset,
forward, output compare, error formation, backward/write, optional
latched-apply).  This module owns the primitives every run script uses to
emit those windows as ngspice PWL voltage sources: the per-sample ``CYCLE_NS``
length, the ``pwl`` formatter, and the ``pulse_wave`` builder that turns a
list of ``(start_ns, end_ns)`` windows into a square-wave PWL source.

Per-experiment ``phases()`` functions stay in their own run scripts because
the choice of *which* windows fire in *which* samples is experiment-specific;
only these three primitives are universally shared.
"""

from __future__ import annotations


#: Default per-sample cycle length in nanoseconds.  Run scripts that want a
#: longer cycle override this locally.
CYCLE_NS: float = 16.0

#: Default rail high level (matches the LEVEL=1 deck supply rail).
VDD: float = 1.2


def compact_pwl_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """De-duplicate PWL breakpoints at the same time, keeping the latest value."""
    compact: list[tuple[float, float]] = []
    for t, v in sorted(points):
        if compact and abs(compact[-1][0] - t) < 1e-12:
            compact[-1] = (t, v)
        else:
            compact.append((t, v))
    return compact


def pwl(points: list[tuple[float, float]]) -> str:
    """Format a list of ``(time_ns, voltage)`` breakpoints as an ngspice PWL string."""
    compact = compact_pwl_points(points)
    return "PWL(" + " ".join(f"{t:.12g}n {v:.12g}" for t, v in compact) + ")"


def pulse_wave(
    pulses: list[tuple[float, float]],
    stop_ns: float,
    high: float = VDD,
) -> str:
    """Build a square-wave PWL source from a list of ``(start_ns, end_ns)`` pulses.

    Each pulse is rendered as four breakpoints (rise edge, plateau start,
    plateau end, fall edge) with a 50 ps guard band so the slopes don't
    collapse to zero.  The output is bracketed by zero-rail anchors at ``t=0``
    and ``t=stop_ns``.
    """
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for start, end in pulses:
        points.append((max(0.0, start - 0.05), 0.0))
        points.append((start, high))
        points.append((end, high))
        points.append((min(stop_ns, end + 0.05), 0.0))
    points.append((stop_ns, 0.0))
    return pwl(points)


__all__ = ["CYCLE_NS", "VDD", "compact_pwl_points", "pwl", "pulse_wave"]
