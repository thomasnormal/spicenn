from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PySpice.Spice.Netlist import Circuit


SimulatorKind = Literal["ngspice", "xyce", "generic"]


CONTROL_RE = re.compile(r"(?ims)^\.control\b.*?^\.endc\s*\n?")
END_RE = re.compile(r"(?im)^\s*\.end\s*$")
TITLE_RE = re.compile(r"(?im)^\s*\.title\b.*$")


@dataclass(frozen=True)
class PySpiceDeck:
    circuit: Circuit
    control_block: str


def simulator_kind(spice_bin: str) -> SimulatorKind:
    name = Path(spice_bin).name.lower()
    if "ngspice" in name:
        return "ngspice"
    if "xyce" in name:
        return "xyce"
    return "generic"


def is_ngspice(spice_bin: str) -> bool:
    return simulator_kind(spice_bin) == "ngspice"


def is_xyce(spice_bin: str) -> bool:
    return simulator_kind(spice_bin) == "xyce"


def split_control_block(netlist: str) -> tuple[str, str]:
    match = CONTROL_RE.search(netlist)
    if match is None:
        return netlist, ""
    body = netlist[: match.start()] + netlist[match.end() :]
    return body, match.group(0).strip() + "\n"


def strip_end(netlist: str) -> str:
    return END_RE.sub("", netlist).strip() + "\n"


def raw_netlist_to_pyspice(netlist: str, *, title: str = "spicenn generated deck") -> PySpiceDeck:
    """Convert one generated SPICE deck into a PySpice circuit without parsing devices.

    Most experimental decks are generated as raw SPICE because they use a wide
    mix of MOS stacks, measurements, and simulator directives.  PySpice remains
    the single normalization layer here: it owns the title/end rendering while
    the original circuit body is kept as raw SPICE so ngspice and Xyce receive
    the same generated circuit body.
    """
    body, control = split_control_block(netlist)
    body = strip_end(body)
    body = TITLE_RE.sub("", body).strip() + "\n"
    circuit = Circuit(title)
    circuit.raw_spice = body
    return PySpiceDeck(circuit=circuit, control_block=control)


def render_pyspice_deck(deck: PySpiceDeck, *, include_control: bool) -> str:
    rendered = str(deck.circuit).rstrip() + "\n"
    if include_control and deck.control_block:
        rendered += deck.control_block
    rendered += ".end\n"
    return rendered


def render_for_simulator(netlist: str, spice_bin: str) -> str:
    deck = raw_netlist_to_pyspice(netlist)
    return render_pyspice_deck(deck, include_control=simulator_kind(spice_bin) == "ngspice")


def spice_batch_command(spice_bin: str, netlist: Path) -> list[str]:
    return [spice_bin, "-b", str(netlist)] if is_ngspice(spice_bin) else [spice_bin, str(netlist)]


def write_simulator_netlist(path: Path, netlist: str, spice_bin: str) -> None:
    path.write_text(render_for_simulator(netlist, spice_bin))


def run_simulator_netlist(spice_bin: str, netlist: Path, *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(spice_batch_command(spice_bin, netlist), text=True, capture_output=True, timeout=timeout)


def run_text_netlist(spice_bin: str, path: Path, netlist: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    write_simulator_netlist(path, netlist, spice_bin)
    return run_simulator_netlist(spice_bin, path, timeout=timeout)


def canonical_circuit_body(netlist: str) -> str:
    body, _control = split_control_block(netlist)
    body = strip_end(body)
    body = TITLE_RE.sub("", body)
    lines = []
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines) + "\n"
