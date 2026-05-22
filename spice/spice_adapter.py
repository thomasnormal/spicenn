from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.xyce_simulator import XyceSimulator


SimulatorKind = Literal["ngspice", "xyce", "generic"]


CONTROL_RE = re.compile(r"(?ims)^\.control\b.*?^\.endc\s*\n?")
END_RE = re.compile(r"(?im)^\s*\.end\s*$")
TITLE_RE = re.compile(r"(?im)^\s*\.title\b.*$")


@dataclass(frozen=True)
class SpiceDeck:
    title: str
    body: str
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


def raw_netlist_to_spice_deck(netlist: str, *, title: str = "spicenn generated deck") -> SpiceDeck:
    """Normalize one generated SPICE deck without parsing devices.

    Most experimental decks are generated as raw SPICE because they use a wide
    mix of MOS stacks, measurements, and simulator directives.  Keep the circuit
    body textual so ngspice and Xyce receive the same generated devices, while
    simulator-specific directives such as ngspice control blocks are handled at
    this boundary.
    """
    body, control = split_control_block(netlist)
    body = strip_end(body)
    body = TITLE_RE.sub("", body).strip() + "\n"
    return SpiceDeck(title=title, body=body, control_block=control)


def render_spice_deck(deck: SpiceDeck, *, include_control: bool) -> str:
    rendered = f".title {deck.title}\n{deck.body}".rstrip() + "\n"
    if include_control and deck.control_block:
        rendered += deck.control_block
    rendered += ".end\n"
    return rendered


def render_for_simulator(netlist: str, spice_bin: str) -> str:
    deck = raw_netlist_to_spice_deck(netlist)
    return render_spice_deck(deck, include_control=simulator_kind(spice_bin) == "ngspice")


def spicelib_simulator(spice_bin: str):
    kind = simulator_kind(spice_bin)
    if kind == "ngspice":
        return NGspiceSimulator.create_from(spice_bin)
    if kind == "xyce":
        return XyceSimulator.create_from(spice_bin)
    return None


def simulator_exe_tokens(spice_bin: str) -> list[str]:
    try:
        simulator = spicelib_simulator(spice_bin)
    except FileNotFoundError:
        simulator = None
    if simulator is None:
        return [spice_bin]
    return list(simulator.spice_exe)


def spice_batch_command(spice_bin: str, netlist: Path) -> list[str]:
    exe = simulator_exe_tokens(spice_bin)
    # Keep SPICENN's text-output contract: ngspice control measurements arrive
    # on stdout, and Xyce .print decks create .prn files. spicelib's default
    # rawfile switches change both behaviors.
    if is_ngspice(spice_bin):
        return exe + ["-b", netlist.as_posix()]
    if is_xyce(spice_bin):
        return exe + [netlist.as_posix()]
    return exe + [netlist.as_posix()]


def write_simulator_netlist(path: Path, netlist: str, spice_bin: str) -> None:
    path.write_text(render_for_simulator(netlist, spice_bin))


def _run_simulator_process(spice_bin: str, netlist: Path, *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(spice_batch_command(spice_bin, netlist), text=True, capture_output=True, timeout=timeout)


def run_simulator_netlist(spice_bin: str, netlist: Path, *, timeout: float) -> subprocess.CompletedProcess[str]:
    return _run_simulator_process(spice_bin, netlist, timeout=timeout)


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
