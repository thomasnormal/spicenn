from __future__ import annotations

import csv
from pathlib import Path

from .topology import SourceId


ReadoutFanins = dict[int, tuple[int, ...]]


def expected_readout_cap_names(
    *,
    hidden_count: int,
    output_count: int,
    readout_fanins: ReadoutFanins | None = None,
) -> set[str]:
    """Return the capacitor names required for a programmed readout array."""
    if hidden_count <= 0:
        raise ValueError("hidden_count must be positive")
    if output_count <= 0:
        raise ValueError("output_count must be positive")
    expected = {f"vbo{out}{rail}" for out in range(output_count) for rail in ("p", "n")}
    if readout_fanins is None:
        fanins = {out: tuple(range(hidden_count)) for out in range(output_count)}
    else:
        fanins = {out: tuple(readout_fanins.get(out, ())) for out in range(output_count)}
    for out, hidden_ids in fanins.items():
        for hidden in hidden_ids:
            expected.update({f"vw{out}{int(hidden)}p", f"vw{out}{int(hidden)}n"})
    return expected


def load_readout_cap_state_csv(
    path: Path,
    *,
    hidden_count: int,
    output_count: int,
    readout_fanins: ReadoutFanins | None = None,
    rail_min_v: float = 0.0,
    rail_max_v: float = 1.2,
) -> dict[str, float]:
    """Load exact readout capacitor initial voltages from a CSV file.

    The CSV may use either `cap,value` or `name,value` columns.  The loader is
    intentionally strict: missing, extra, duplicate, or out-of-rail capacitors
    indicate that the programmed physical state does not match the readout
    topology being simulated.
    """
    expected = expected_readout_cap_names(
        hidden_count=hidden_count,
        output_count=output_count,
        readout_fanins=readout_fanins,
    )
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"readout cap-state CSV is empty: {path}")
        name_column = "cap" if "cap" in reader.fieldnames else "name" if "name" in reader.fieldnames else None
        if name_column is None or "value" not in reader.fieldnames:
            raise ValueError(f"readout cap-state CSV is missing cap,value columns: {path}")
        init: dict[str, float] = {}
        for row in reader:
            cap = str(row[name_column])
            if cap not in expected:
                raise ValueError(f"readout cap-state CSV contains unexpected capacitor {cap!r}: {path}")
            if cap in init:
                raise ValueError(f"readout cap-state CSV contains duplicate capacitor {cap!r}: {path}")
            value = float(row["value"])
            if not rail_min_v <= value <= rail_max_v:
                raise ValueError(
                    f"readout cap-state CSV capacitor {cap!r} is outside "
                    f"{rail_min_v:g}..{rail_max_v:g} V: {value}"
                )
            init[cap] = value
    missing_caps = sorted(expected - set(init))
    if missing_caps:
        raise ValueError(f"readout cap-state CSV is missing capacitor states {missing_caps}: {path}")
    return init


def normalized_fanins(
    fanins: dict[int, tuple[SourceId, ...]],
    *,
    output_count: int,
) -> ReadoutFanins:
    """Convert topology source ids into int hidden indices for readout state helpers."""
    return {out: tuple(int(source) for source in fanins.get(out, ())) for out in range(output_count)}
