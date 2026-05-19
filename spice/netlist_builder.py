from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union


Value = Union[str, int, float]


def fmt(value: Value) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def param_ref(name: str) -> str:
    return "{" + name + "}"


def v(node: str) -> str:
    return f"V({node})"


def pwl(points: Sequence[tuple[float, float]]) -> str:
    body = " ".join(f"{fmt(float(t))} {fmt(float(val))}" for t, val in points)
    return f"PWL({body})"


@dataclass
class Netlist:
    title: str = ""
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.title:
            self.comment(self.title)

    def comment(self, text: str = "") -> None:
        if not text:
            self.lines.append("*")
            return
        for line in text.splitlines():
            self.lines.append(f"* {line}")

    def blank(self) -> None:
        self.lines.append("")

    def raw(self, line: str = "") -> None:
        self.lines.append(line)

    def param(self, name: str, value: Value) -> None:
        self.lines.append(f".param {name}={fmt(value)}")

    def vsource(self, name: str, p: str, n: str, spec: Value) -> None:
        self.lines.append(f"V{name} {p} {n} {fmt(spec)}")

    def isource(self, name: str, p: str, n: str, spec: Value) -> None:
        self.lines.append(f"I{name} {p} {n} {fmt(spec)}")

    def resistor(self, name: str, p: str, n: str, value: Value) -> None:
        self.lines.append(f"R{name} {p} {n} {fmt(value)}")

    def capacitor(self, name: str, p: str, n: str, value: Value, ic: Optional[Value] = None) -> None:
        line = f"C{name} {p} {n} {fmt(value)}"
        if ic is not None:
            line += f" IC={fmt(ic)}"
        self.lines.append(line)

    def bsource(self, name: str, p: str, n: str, kind: str, expr: str) -> None:
        kind = kind.upper()
        if kind not in {"I", "V"}:
            raise ValueError(f"behavioral source kind must be I or V, got {kind!r}")
        self.lines.append(f"B{name} {p} {n} {kind} = {expr}")

    def options(self, *options: str) -> None:
        self.lines.append(".options " + " ".join(options))

    def control(self, *lines: str) -> None:
        self.lines.append(".control")
        self.lines.extend(lines)
        self.lines.append(".endc")

    def end(self) -> None:
        self.lines.append(".end")

    def render(self) -> str:
        text = "\n".join(self.lines)
        return text if text.endswith("\n") else text + "\n"

    def write(self, path: Path) -> None:
        path.write_text(self.render())


def join_terms(terms: Iterable[str]) -> str:
    return " + ".join(terms)
