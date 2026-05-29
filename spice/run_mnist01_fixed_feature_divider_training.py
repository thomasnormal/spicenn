from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from _util import parse_measures
from run_device_sequential_training import mos_models, pwl
from run_multiclass_output_head_primitive import class_node, signed_store_lines
from run_multiclass_output_head_sequence import class_local_live_label_descent_update_lines
from run_spice_sweep import ROOT, run_text_netlist


OUTPUTS = 2
CYCLE_NS = 10.0


def _compact_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    compact: list[tuple[float, float]] = []
    for time, value in sorted(points):
        if compact and abs(compact[-1][0] - time) < 1e-12:
            compact[-1] = (time, value)
        else:
            compact.append((time, value))
    return compact


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
    return pwl(_compact_points(points))


def _active_low_pulse_wave(pulses: list[tuple[float, float]], stop_ns: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 1.2)]
    for start, end in pulses:
        if start > 0.0:
            points.append((max(0.0, start - 0.03), 1.2))
        points.append((start, 0.0))
        points.append((end, 0.0))
        points.append((min(stop_ns, end + 0.03), 1.2))
    points.append((stop_ns, 1.2))
    return pwl(_compact_points(points))


def _current_pulse_wave(pulses: list[tuple[float, float]], stop_ns: float, current_a: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for start, end in pulses:
        if start > 0.0:
            points.append((max(0.0, start - 0.03), 0.0))
        points.append((start, current_a))
        points.append((end, current_a))
        points.append((min(stop_ns, end + 0.03), 0.0))
    points.append((stop_ns, 0.0))
    return pwl(_compact_points(points))


def _sample_feature_wave(samples: list[dict[str, Any]], feature: int, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = float(sample["features"][feature])
        if idx == 0:
            points.append((0.0, value))
        else:
            prev = float(samples[idx - 1]["features"][feature])
            points.append((start - 0.03, prev))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.03), value))
    points.append((stop_ns, float(samples[-1]["features"][feature])))
    return pwl(_compact_points(points))


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
    return pwl(_compact_points(points))


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


def _idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape((count, rows, cols)).astype(np.float64) / 255.0


def _idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.shape[0] != count:
        raise ValueError(f"{path} label count does not match header")
    return data.astype(np.int64)


def downsample_image_area(image: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    rows, cols = image.shape
    row_edges = np.linspace(0, rows, image_size + 1, dtype=int)
    col_edges = np.linspace(0, cols, image_size + 1, dtype=int)
    out = np.zeros((image_size, image_size), dtype=np.float64)
    for row in range(image_size):
        for col in range(image_size):
            block = image[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]]
            out[row, col] = float(np.mean(block)) if block.size else 0.0
    return out


def pixel_to_feature_voltage(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    if clipped < 0.03:
        return 0.0
    return float(np.clip(1.1 * clipped / 0.65, 0.0, 1.1))


def add_complement_features(
    records: list[dict[str, Any]],
    *,
    scale: float = 0.5,
    high: float = 1.1,
) -> list[dict[str, Any]]:
    if scale <= 0.0 or high <= 0.0:
        raise ValueError("complement feature scale and high voltage must be positive")
    out: list[dict[str, Any]] = []
    for record in records:
        features = record.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("records must contain a nonempty feature list")
        raw = [float(value) for value in features]
        if any(value < 0.0 or value > high for value in raw):
            raise ValueError("raw feature voltages must stay between 0 and high")
        complement = [scale * max(0.0, high - value) for value in raw]
        if any(value > 1.2 for value in complement):
            raise ValueError("complement feature voltages must stay within supply rails")
        out.append({**record, "features": raw + complement})
    return out


def round_robin_by_label(
    records: list[dict[str, Any]],
    *,
    labels: tuple[int, ...] = (0, 1),
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("records must not be empty")
    if not labels:
        raise ValueError("labels must not be empty")
    buckets: dict[int, list[dict[str, Any]]] = {int(label): [] for label in labels}
    for record in records:
        label = int(record.get("label", -1))
        if label not in buckets:
            raise ValueError("record label is not in the requested round-robin label set")
        buckets[label].append(record)
    if any(not bucket for bucket in buckets.values()):
        raise ValueError("each round-robin label must have at least one record")
    out: list[dict[str, Any]] = []
    for idx in range(max(len(bucket) for bucket in buckets.values())):
        for label in labels:
            bucket = buckets[int(label)]
            if idx < len(bucket):
                out.append(bucket[idx])
    return out


def load_mnist01_records(
    *,
    train_count_per_digit: int = 1,
    eval_count_per_digit: int = 1,
    image_size: int = 4,
    root: Path = ROOT / "data/MNIST/raw",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_count_per_digit <= 0 or eval_count_per_digit <= 0:
        raise ValueError("sample counts must be positive")
    train_images = _idx_images(root / "train-images-idx3-ubyte")
    train_labels = _idx_labels(root / "train-labels-idx1-ubyte")
    eval_images = _idx_images(root / "t10k-images-idx3-ubyte")
    eval_labels = _idx_labels(root / "t10k-labels-idx1-ubyte")

    def records(images: np.ndarray, labels: np.ndarray, count_per_digit: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for digit in (0, 1):
            chosen = np.flatnonzero(labels == digit)[:count_per_digit]
            if chosen.shape[0] < count_per_digit:
                raise ValueError(f"not enough digit {digit} samples")
            for index in chosen:
                small = downsample_image_area(images[int(index)], image_size)
                features = [pixel_to_feature_voltage(value) for value in small.reshape(-1)]
                out.append(
                    {
                        "features": features,
                        "label": digit,
                        "digit": digit,
                        "mnist_index": int(index),
                        "train": False,
                    }
                )
        return out

    return records(train_images, train_labels, train_count_per_digit), records(
        eval_images,
        eval_labels,
        eval_count_per_digit,
    )


def _sample_plan(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    initial = [{**sample, "phase": "initial", "train": False} for sample in eval_records]
    train = [{**sample, "phase": "train", "train": True} for sample in train_records]
    final = [{**sample, "phase": "final", "train": False} for sample in eval_records]
    return initial + train + final, len(initial), len(train)


def _feature_storage_lines(feature_count: int) -> list[str]:
    lines: list[str] = []
    for feature in range(feature_count):
        lines += [
            f"Cact{feature} act{feature} 0 8f IC=0",
            f"Ract{feature} act{feature} 0 1G",
            f"Mreset_act{feature} act{feature} rst 0 0 NMOS W=4u L=180n",
        ]
    return lines


def _feature_sampler_lines(feature_count: int, width_u: float) -> list[str]:
    return [
        f"Mfeat{feature}_sample px{feature} featphi act{feature} 0 NSENSE W={width_u:.6g}u L=180n"
        for feature in range(feature_count)
    ]


def _readout_storage_lines(feature_count: int, initial_positive: float, initial_negative: float) -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        for feature in range(feature_count):
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


def _score_readout_lines(feature_count: int, width_u: float) -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        scorep = class_node(output, "scorep")
        scoren = class_node(output, "scoren")
        for feature in range(feature_count):
            prefix = f"c{output}_f{feature}_score_"
            lines += [
                f"R{prefix}pa {prefix}pa 0 1G",
                f"R{prefix}pb {prefix}pb 0 1G",
                f"C{prefix}pa {prefix}pa 0 0.05f IC=0",
                f"C{prefix}pb {prefix}pb 0 0.05f IC=0",
                f"M{prefix}pa vdd act{feature} {prefix}pa 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pw {prefix}pa {class_node(output, f'vwp{feature}')} {prefix}pb 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pphi {prefix}pb scorephi {scorep} 0 NSENSE W={width_u:.6g}u L=180n",
                f"R{prefix}na {prefix}na 0 1G",
                f"R{prefix}nb {prefix}nb 0 1G",
                f"C{prefix}na {prefix}na 0 0.05f IC=0",
                f"C{prefix}nb {prefix}nb 0 0.05f IC=0",
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
    return [
        f"Mnorm0_score rd0 {class_node(0, 'scorep')} mir0 0 NSENSE W={branch_width_u:.6g}u L=180n",
        f"Mnorm0_floor rd0 normfloor mir0 0 NSENSE W={floor_width_u:.6g}u L=180n",
        f"Mnorm1_score rd1 {class_node(1, 'scorep')} mir1 0 NSENSE W={branch_width_u:.6g}u L=180n",
        f"Mnorm1_floor rd1 normfloor mir1 0 NSENSE W={floor_width_u:.6g}u L=180n",
        "Mnorm0_ref mir0 mir0 0 0 NMOS W=2u L=180n",
        "Mnorm1_ref mir1 mir1 0 0 NMOS W=2u L=180n",
    ]


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


def _writer_lines(feature_count: int, update_width_u: float) -> list[str]:
    lines = [
        "Vvwhi_ref vwhi_ref 0 0.48",
        "Vvwlo_ref vwlo_ref 0 0.22",
    ]
    for output in range(OUTPUTS):
        for feature in range(feature_count):
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


def _measure_lines(samples: list[dict[str, Any]], eval_count: int, train_count: int) -> list[str]:
    lines: list[str] = []
    train_offset = eval_count
    final_offset = eval_count + train_count
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        label = int(sample["label"])
        other = 1 - label
        if sample["phase"] in {"initial", "final"}:
            phase = str(sample["phase"])
            local = idx if phase == "initial" else idx - final_offset
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
            strongest_feature = int(np.argmax(np.asarray(sample["features"], dtype=float)))
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
                    f".meas tran train_{role}_vwp_before_{local} FIND V({class_node(output, f'vwp{strongest_feature}')}) AT={before:.2f}n",
                    f".meas tran train_{role}_vwn_before_{local} FIND V({class_node(output, f'vwn{strongest_feature}')}) AT={before:.2f}n",
                    f".meas tran train_{role}_vwp_after_{local} FIND V({class_node(output, f'vwp{strongest_feature}')}) AT={after:.2f}n",
                    f".meas tran train_{role}_vwn_after_{local} FIND V({class_node(output, f'vwn{strongest_feature}')}) AT={after:.2f}n",
                    f".meas tran train_{role}_signed_before_{local} PARAM='train_{role}_vwp_before_{local}-train_{role}_vwn_before_{local}'",
                    f".meas tran train_{role}_signed_after_{local} PARAM='train_{role}_vwp_after_{local}-train_{role}_vwn_after_{local}'",
                    f".meas tran train_{role}_signed_delta_{local} PARAM='train_{role}_signed_after_{local}-train_{role}_signed_before_{local}'",
                ]
    for idx in range(eval_count):
        lines.append(f".meas tran final_margin_improvement_{idx} PARAM='final_margin_{idx}-initial_margin_{idx}'")
    return lines


def _validate_records(records: list[dict[str, Any]], feature_count: int, name: str) -> None:
    if not records:
        raise ValueError(f"{name} records must not be empty")
    for record in records:
        features = record.get("features")
        if not isinstance(features, list) or len(features) != feature_count:
            raise ValueError(f"{name} records must all have {feature_count} features")
        if int(record.get("label", -1)) not in (0, 1):
            raise ValueError(f"{name} labels must be 0 or 1")
        if any(not 0.0 <= float(value) <= 1.2 for value in features):
            raise ValueError(f"{name} feature voltages must stay within supply rails")


def mnist01_fixed_feature_netlist(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    initial_positive: float = 0.40,
    initial_negative: float = 0.40,
    iref_a: float = 1.0e-6,
    feature_sample_width_u: float = 16.0,
    readout_width_u: float = 16.0,
    branch_width_u: float = 0.05,
    floor_width_u: float = 0.015,
    route_width_u: float = 3.0,
    update_width_u: float = 0.25,
) -> str:
    if not train_records or not eval_records:
        raise ValueError("train and eval records must not be empty")
    feature_count = len(train_records[0]["features"])
    _validate_records(train_records, feature_count, "train")
    _validate_records(eval_records, feature_count, "eval")
    if min(
        initial_positive,
        initial_negative,
        iref_a,
        feature_sample_width_u,
        readout_width_u,
        branch_width_u,
        floor_width_u,
        route_width_u,
        update_width_u,
    ) <= 0.0:
        raise ValueError("voltages, current, and widths must be positive")

    samples, eval_count, train_count = _sample_plan(train_records, eval_records)
    stop_ns = len(samples) * CYCLE_NS
    lines = [
        "* Fixed-feature MNIST01 output learner with conductance-divider normalized error.",
        "* Python supplies pixels, labels, clocks, and diagnostics only; readout weights move in SPICE.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vt0 t0 0 {_target_wave(samples, 0, stop_ns)}",
        f"Vt1 t1 0 {_target_wave(samples, 1, stop_ns)}",
        *_clock_lines(samples, stop_ns, iref_a),
    ]
    for feature in range(feature_count):
        lines.append(f"Vpx{feature} px{feature} 0 {_sample_feature_wave(samples, feature, stop_ns)}")
    lines += [
        *_feature_storage_lines(feature_count),
        *_feature_sampler_lines(feature_count, feature_sample_width_u),
        *_readout_storage_lines(feature_count, initial_positive, initial_negative),
        *_score_storage_lines(),
        *_score_readout_lines(feature_count, readout_width_u),
        *_error_storage_lines(),
        *_divider_probability_lines(branch_width_u, floor_width_u),
        *_route_to_error_rails_lines(route_width_u),
        *_writer_lines(feature_count, update_width_u),
        f".tran 5p {stop_ns:.2f}n uic",
        *_measure_lines(samples, eval_count, train_count),
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float = 90.0) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed
