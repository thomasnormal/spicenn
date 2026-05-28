from __future__ import annotations

from pathlib import Path
from typing import Any

from _util import parse_measures
from run_device_sequential_training import mos_models, pwl
from run_multiclass_output_head_primitive import class_node, signed_store_lines
from run_multiclass_output_head_sequence import class_local_live_label_descent_update_lines
from run_spice_sweep import run_text_netlist


BITS = 2
FEATURES = 4
OUTPUTS = 2
CYCLE_NS = 10.0


def bit_value(pattern: int, bit: int) -> int:
    return (pattern >> bit) & 1


def xor_label(pattern: int) -> int:
    return bit_value(pattern, 0) ^ bit_value(pattern, 1)


def _input_value(pattern: int, node: str) -> float:
    if node.startswith("nx"):
        return 1.2 * float(1 - bit_value(pattern, int(node[2:])))
    return 1.2 * float(bit_value(pattern, int(node[1:])))


def _pulse_wave(pulses: list[tuple[float, float]], stop_ns: float, high: float = 1.2) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for start, end in pulses:
        if start <= 0.0:
            points.append((0.0, high))
        else:
            points.append((max(0.0, start - 0.03), 0.0))
            points.append((start, high))
        points.append((end, high))
        points.append((min(stop_ns, end + 0.03), 0.0))
    points.append((stop_ns, 0.0))
    return pwl(points)


def _active_low_pulse_wave(pulses: list[tuple[float, float]], stop_ns: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 1.2)]
    for start, end in pulses:
        if start > 0.0:
            points.append((max(0.0, start - 0.03), 1.2))
        points.append((start, 0.0))
        points.append((end, 0.0))
        points.append((min(stop_ns, end + 0.03), 1.2))
    points.append((stop_ns, 1.2))
    return pwl(points)


def _sample_wave(samples: list[dict[str, Any]], node: str, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = _input_value(int(sample["pattern"]), node)
        if idx == 0:
            points.append((0.0, value))
        else:
            prev = _input_value(int(samples[idx - 1]["pattern"]), node)
            points.append((start - 0.03, prev))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.03), value))
    points.append((stop_ns, _input_value(int(samples[-1]["pattern"]), node)))
    return pwl(points)


def _target_wave(samples: list[dict[str, Any]], output: int, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = 1.2 if int(sample["label"]) == output else 0.0
        if idx == 0:
            points.append((0.0, value))
        else:
            prev = 1.2 if int(samples[idx - 1]["label"]) == output else 0.0
            points.append((start - 0.03, prev))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.03), value))
    points.append((stop_ns, 1.2 if int(samples[-1]["label"]) == output else 0.0))
    return pwl(points)


def _current_pulse_wave(pulses: list[tuple[float, float]], stop_ns: float, current_a: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for start, end in pulses:
        if start > 0.0:
            points.append((max(0.0, start - 0.03), 0.0))
        points.append((start, current_a))
        points.append((end, current_a))
        points.append((min(stop_ns, end + 0.03), 0.0))
    points.append((stop_ns, 0.0))
    return pwl(points)


def _sample_plan(train_order: list[int]) -> list[dict[str, Any]]:
    initial = [{"phase": "initial", "pattern": p, "label": xor_label(p), "train": False} for p in range(4)]
    train = [{"phase": "train", "pattern": p, "label": xor_label(p), "train": True} for p in train_order]
    final = [{"phase": "final", "pattern": p, "label": xor_label(p), "train": False} for p in range(4)]
    return initial + train + final


def _clock_lines(samples: list[dict[str, Any]], stop_ns: float, iref_a: float) -> list[str]:
    reset: list[tuple[float, float]] = []
    feat: list[tuple[float, float]] = []
    score: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        reset.append((base + 0.00, base + 0.45))
        feat.append((base + 0.75, base + 2.10))
        score.append((base + 2.00, base + 3.20))
        if bool(sample["train"]):
            err.append((base + 3.50, base + 5.40))
    return [
        f"Vrst rst 0 {_pulse_wave(reset, stop_ns)}",
        f"Vrstn rstn 0 {_active_low_pulse_wave(reset, stop_ns)}",
        f"Vfeatphi featphi 0 {_pulse_wave(feat, stop_ns)}",
        f"Vscorephi scorephi 0 {_pulse_wave(score, stop_ns)}",
        f"Verrphi errphi 0 {_pulse_wave(err, stop_ns)}",
        f"Iprobref vdd rnorm {_current_pulse_wave(err, stop_ns, iref_a)}",
    ]


def _feature_storage_lines() -> list[str]:
    lines: list[str] = []
    for feature in range(FEATURES):
        lines += [
            f"Cact{feature} act{feature} 0 8f IC=0",
            f"Ract{feature} act{feature} 0 1G",
            f"Mreset_act{feature} act{feature} rst 0 0 NMOS W=4u L=180n",
        ]
    return lines


def _feature_decoder_lines() -> list[str]:
    lines: list[str] = []
    for feature in range(FEATURES):
        lit0 = "x0" if bit_value(feature, 0) else "nx0"
        lit1 = "x1" if bit_value(feature, 1) else "nx1"
        lines += [
            f"* Fixed literal feature {feature}.",
            f"Rfeat{feature}_a feat{feature}_a 0 1G",
            f"Rfeat{feature}_b feat{feature}_b 0 1G",
            f"Mfeat{feature}_a vdd {lit0} feat{feature}_a 0 NSENSE W=20u L=180n",
            f"Mfeat{feature}_b feat{feature}_a {lit1} feat{feature}_b 0 NSENSE W=20u L=180n",
            f"Mfeat{feature}_phi feat{feature}_b featphi act{feature} 0 NSENSE W=20u L=180n",
        ]
    return lines


def _readout_storage_lines(initial_positive: float, initial_negative: float) -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        for feature in range(FEATURES):
            lines += signed_store_lines(
                positive_node=class_node(output, f"vwp{feature}"),
                negative_node=class_node(output, f"vwn{feature}"),
                positive_ic=initial_positive,
                negative_ic=initial_negative,
            )
    return lines


def _score_storage_lines() -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        for kind in ("scorep", "scoren"):
            node = class_node(output, kind)
            lines += [
                f"C{node} {node} 0 8f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rst 0 0 NMOS W=4u L=180n",
            ]
    return lines


def _score_readout_lines(width_u: float) -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        scorep = class_node(output, "scorep")
        scoren = class_node(output, "scoren")
        for feature in range(FEATURES):
            prefix = f"c{output}_f{feature}_score_"
            lines += [
                f"R{prefix}pa {prefix}pa 0 1G",
                f"R{prefix}pb {prefix}pb 0 1G",
                f"M{prefix}pa vdd act{feature} {prefix}pa 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pw {prefix}pa {class_node(output, f'vwp{feature}')} {prefix}pb 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pphi {prefix}pb scorephi {scorep} 0 NSENSE W={width_u:.6g}u L=180n",
                f"R{prefix}na {prefix}na 0 1G",
                f"R{prefix}nb {prefix}nb 0 1G",
                f"M{prefix}na vdd act{feature} {prefix}na 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}nw {prefix}na {class_node(output, f'vwn{feature}')} {prefix}nb 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}nphi {prefix}nb scorephi {scoren} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
    return lines


def _error_storage_lines() -> list[str]:
    lines: list[str] = [
        "Vnormfloor normfloor 0 0.62",
        "Rrnorm rnorm 0 1G",
        "Rmir0 mir0 0 1G",
        "Rmir1 mir1 0 1G",
        "Vrsen0 rnorm rd0 0",
        "Vrsen1 rnorm rd1 0",
    ]
    for output in range(OUTPUTS):
        for kind in ("errp", "errn"):
            node = class_node(output, kind)
            lines += [
                f"C{node} {node} 0 2f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rst 0 0 NMOS W=4u L=180n",
            ]
    for branch in (0, 1):
        low = f"b{branch}low"
        lines += [
            f"C{low} {low} 0 1f IC=1.2",
            f"R{low} {low} vdd 1G",
            f"Mpre_{low} {low} rstn vdd vdd PMOS W=4u L=180n",
            f"R{low}_a {low}_a 0 1G",
            f"M{low}_sink {low} mir{branch} {low}_a 0 NMOS W=0.85u L=180n",
            f"M{low}_phi {low}_a errphi 0 0 NSENSE W=0.85u L=180n",
        ]
    return lines


def _divider_probability_lines(branch_width_u: float, floor_width_u: float) -> list[str]:
    lines = [
        f"Mnorm0_score rd0 {class_node(0, 'scorep')} mir0 0 NSENSE W={branch_width_u:.6g}u L=180n",
        f"Mnorm0_floor rd0 normfloor mir0 0 NSENSE W={floor_width_u:.6g}u L=180n",
        f"Mnorm1_score rd1 {class_node(1, 'scorep')} mir1 0 NSENSE W={branch_width_u:.6g}u L=180n",
        f"Mnorm1_floor rd1 normfloor mir1 0 NSENSE W={floor_width_u:.6g}u L=180n",
        "Mnorm0_ref mir0 mir0 0 0 NMOS W=2u L=180n",
        "Mnorm1_ref mir1 mir1 0 0 NMOS W=2u L=180n",
    ]
    return lines


def _route_to_error_rails_lines(route_width_u: float) -> list[str]:
    def charge_lines(name: str, dest: str, low: str, target: str) -> list[str]:
        return [
            f"R{name}_a {name}_a 0 1G",
            f"R{name}_b {name}_b 0 1G",
            f"M{name}_m vdd {low} {name}_a vdd PMOS W={route_width_u:.6g}u L=180n",
            f"M{name}_t {name}_a {target} {name}_b 0 NSENSE W=12u L=180n",
            f"M{name}_phi {name}_b errphi {dest} 0 NSENSE W=12u L=180n",
        ]

    lines: list[str] = []
    lines += charge_lines("err_c0p", class_node(0, "errp"), "b1low", "t0")
    lines += charge_lines("err_c1n", class_node(1, "errn"), "b1low", "t0")
    lines += charge_lines("err_c1p", class_node(1, "errp"), "b0low", "t1")
    lines += charge_lines("err_c0n", class_node(0, "errn"), "b0low", "t1")
    return lines


def _writer_lines(update_width_u: float) -> list[str]:
    lines = [
        "Vvwhi_ref vwhi_ref 0 0.48",
        "Vvwlo_ref vwlo_ref 0 0.22",
    ]
    for output in range(OUTPUTS):
        for feature in range(FEATURES):
            lines += class_local_live_label_descent_update_lines(
                class_idx=output,
                feature_idx=feature,
                activation_node=f"act{feature}",
                positive_descent_node=class_node(output, "errp"),
                negative_descent_node=class_node(output, "errn"),
                width_u=update_width_u,
                high_side_topology="pmos-differential",
            )
    return lines


def _measure_lines(samples: list[dict[str, Any]], train_offset: int) -> list[str]:
    lines: list[str] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        pattern = int(sample["pattern"])
        label = int(sample["label"])
        other = 1 - label
        if sample["phase"] in {"initial", "final"}:
            phase = str(sample["phase"])
            local = pattern
            at = base + 3.15
            lines += [
                f".meas tran {phase}_target_scorep_{local} FIND V({class_node(label, 'scorep')}) AT={at:.2f}n",
                f".meas tran {phase}_target_scoren_{local} FIND V({class_node(label, 'scoren')}) AT={at:.2f}n",
                f".meas tran {phase}_other_scorep_{local} FIND V({class_node(other, 'scorep')}) AT={at:.2f}n",
                f".meas tran {phase}_other_scoren_{local} FIND V({class_node(other, 'scoren')}) AT={at:.2f}n",
                f".meas tran {phase}_target_signed_{local} PARAM='{phase}_target_scorep_{local}-{phase}_target_scoren_{local}'",
                f".meas tran {phase}_other_signed_{local} PARAM='{phase}_other_scorep_{local}-{phase}_other_scoren_{local}'",
                f".meas tran {phase}_margin_{local} PARAM='{phase}_target_signed_{local}-{phase}_other_signed_{local}'",
            ]
        if sample["phase"] == "train":
            local = idx - train_offset
            before = base + 0.55
            after = base + 7.80
            err_at = base + 5.35
            lines += [
                f".meas tran train_ir0_{local} FIND I(Vrsen0) AT={base + 5.00:.2f}n",
                f".meas tran train_ir1_{local} FIND I(Vrsen1) AT={base + 5.00:.2f}n",
                f".meas tran train_rnorm_{local} FIND V(rnorm) AT={base + 5.00:.2f}n",
                f".meas tran train_target_errp_{local} FIND V({class_node(label, 'errp')}) AT={err_at:.2f}n",
                f".meas tran train_target_errn_{local} FIND V({class_node(label, 'errn')}) AT={err_at:.2f}n",
                f".meas tran train_other_errp_{local} FIND V({class_node(other, 'errp')}) AT={err_at:.2f}n",
                f".meas tran train_other_errn_{local} FIND V({class_node(other, 'errn')}) AT={err_at:.2f}n",
            ]
            for output in (label, other):
                role = "target" if output == label else "other"
                lines += [
                    f".meas tran train_{role}_vwp_before_{local} FIND V({class_node(output, f'vwp{pattern}')}) AT={before:.2f}n",
                    f".meas tran train_{role}_vwn_before_{local} FIND V({class_node(output, f'vwn{pattern}')}) AT={before:.2f}n",
                    f".meas tran train_{role}_vwp_after_{local} FIND V({class_node(output, f'vwp{pattern}')}) AT={after:.2f}n",
                    f".meas tran train_{role}_vwn_after_{local} FIND V({class_node(output, f'vwn{pattern}')}) AT={after:.2f}n",
                    f".meas tran train_{role}_signed_before_{local} PARAM='train_{role}_vwp_before_{local}-train_{role}_vwn_before_{local}'",
                    f".meas tran train_{role}_signed_after_{local} PARAM='train_{role}_vwp_after_{local}-train_{role}_vwn_after_{local}'",
                    f".meas tran train_{role}_signed_delta_{local} PARAM='train_{role}_signed_after_{local}-train_{role}_signed_before_{local}'",
                ]
    for pattern in range(4):
        lines.append(
            f".meas tran final_margin_improvement_{pattern} PARAM='final_margin_{pattern}-initial_margin_{pattern}'"
        )
    return lines


def xor_fixed_feature_netlist(
    train_order: list[int],
    *,
    initial_positive: float = 0.40,
    initial_negative: float = 0.40,
    iref_a: float = 1.0e-6,
    readout_width_u: float = 16.0,
    branch_width_u: float = 0.05,
    floor_width_u: float = 0.015,
    route_width_u: float = 3.0,
    update_width_u: float = 0.25,
) -> str:
    if not train_order:
        raise ValueError("train_order must not be empty")
    if any(pattern not in range(4) for pattern in train_order):
        raise ValueError("train_order patterns must be in 0..3")
    if min(
        initial_positive,
        initial_negative,
        iref_a,
        readout_width_u,
        branch_width_u,
        floor_width_u,
        route_width_u,
        update_width_u,
    ) <= 0.0:
        raise ValueError("voltages, current, and widths must be positive")

    samples = _sample_plan(train_order)
    stop_ns = len(samples) * CYCLE_NS
    train_offset = 4
    lines = [
        "* Fixed-feature XOR output learner with conductance-divider normalized error.",
        "* Python supplies inputs, labels, clocks, and diagnostics only; readout weights move in SPICE.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vx0 x0 0 {_sample_wave(samples, 'x0', stop_ns)}",
        f"Vx1 x1 0 {_sample_wave(samples, 'x1', stop_ns)}",
        f"Vnx0 nx0 0 {_sample_wave(samples, 'nx0', stop_ns)}",
        f"Vnx1 nx1 0 {_sample_wave(samples, 'nx1', stop_ns)}",
        f"Vt0 t0 0 {_target_wave(samples, 0, stop_ns)}",
        f"Vt1 t1 0 {_target_wave(samples, 1, stop_ns)}",
        *_clock_lines(samples, stop_ns, iref_a),
        *_feature_storage_lines(),
        *_feature_decoder_lines(),
        *_readout_storage_lines(initial_positive, initial_negative),
        *_score_storage_lines(),
        *_score_readout_lines(readout_width_u),
        *_error_storage_lines(),
        *_divider_probability_lines(branch_width_u, floor_width_u),
        *_route_to_error_rails_lines(route_width_u),
        *_writer_lines(update_width_u),
        f".tran 5p {stop_ns:.2f}n uic",
        *_measure_lines(samples, train_offset),
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float = 60.0) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed
