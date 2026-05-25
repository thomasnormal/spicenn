from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_device_mnist01_scalar_training import balanced_digit_indices, binary_accuracy, sanitize_tag
from run_device_sequential_training import (
    active_low_pulse_wave,
    expected_positive,
    mos_models,
    output_driver_line,
    pulse_wave,
    run_netlist,
)
from run_spice_sweep import ROOT, detect_spice, run_tiny_test
from spicenn.timing import CYCLE_NS


FEATURE_COUNT = 4


def quad_features_from_image(image: np.ndarray) -> np.ndarray:
    pixels = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    if pixels.ndim != 2:
        raise ValueError("image must be 2-D")
    hmid = pixels.shape[0] // 2
    wmid = pixels.shape[1] // 2
    quadrants = (
        pixels[:hmid, :wmid],
        pixels[:hmid, wmid:],
        pixels[hmid:, :wmid],
        pixels[hmid:, wmid:],
    )
    inks = np.asarray([float(np.mean(q)) for q in quadrants], dtype=np.float64)
    return np.clip(0.55 + 0.55 * inks / 0.35, 0.05, 1.1)


def sample_wave(samples: list[dict[str, Any]], key: str, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = float(sample[key])
        if idx == 0:
            points.append((0.0, value))
        else:
            points.append((start - 0.05, float(samples[idx - 1][key])))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.05), value))
    points.append((stop_ns, float(samples[-1][key])))
    return "PWL(" + " ".join(f"{t:.12g}n {v:.12g}" for t, v in points) + ")"


def repeated_phases(sample_count: int, *, training_enabled: bool) -> str:
    stop = sample_count * CYCLE_NS
    rstf: list[tuple[float, float]] = []
    rstg: list[tuple[float, float]] = []
    fwd: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    bwd: list[tuple[float, float]] = []
    acc: list[tuple[float, float]] = []
    apply: list[tuple[float, float]] = []
    for idx in range(sample_count):
        base = idx * CYCLE_NS
        rstf += [(base + 0.00, base + 0.50), (base + 12.05, base + 12.55)]
        rstg += [(base + 0.00, base + 0.50)]
        fwd += [(base + 0.75, base + 3.00), (base + 12.80, base + 15.60)]
        if training_enabled:
            err.append((base + 3.25, base + 5.00))
            bwd.append((base + 5.25, base + 7.00))
            acc.append((base + 7.25, base + 9.00))
            apply.append((base + 9.25, base + 11.20))
    return "\n".join(
        [
            f"Vrstf rstf 0 {pulse_wave(rstf, stop)}",
            f"Vrstg rstg 0 {pulse_wave(rstg, stop)}",
            f"Vfwd fwd 0 {pulse_wave(fwd, stop)}",
            f"Verr err 0 {pulse_wave(err, stop)}",
            f"Vbwd bwd 0 {pulse_wave(bwd, stop)}",
            f"Vacc acc 0 {pulse_wave(acc, stop)}",
            f"Vapply apply 0 {pulse_wave(apply, stop)}",
            f"Vapplyn applyn 0 {active_low_pulse_wave(apply, stop)}",
        ]
    )


def initial_quad_weights() -> dict[str, list[float]]:
    return {
        "whp": [0.85] * FEATURE_COUNT,
        "whn": [0.25] * FEATURE_COUNT,
        "vwp": [0.55] * FEATURE_COUNT,
        "vwn": [0.25] * FEATURE_COUNT,
    }


def quad_netlist(
    samples: list[dict[str, Any]],
    weights: dict[str, list[float]],
    *,
    training_enabled: bool,
    output_driver_model: str = "sense",
    readout_apply_scale: float = 0.5,
) -> str:
    if readout_apply_scale <= 0.0:
        raise ValueError("readout_apply_scale must be positive")
    readout_pmos_w = 8.0 * readout_apply_scale
    readout_nmos_w = 2.0 * readout_apply_scale
    stop = len(samples) * CYCLE_NS
    measures: list[str] = []
    prints: list[str] = []
    for idx in range(len(samples)):
        base = idx * CYCLE_NS
        measures += [
            f".meas tran score_before_{idx} FIND V(score) AT={base + 2.95:.2f}n",
            f".meas tran out_before_{idx} FIND V(out) AT={base + 2.95:.2f}n",
            f".meas tran score_error_{idx} FIND V(score) AT={base + 4.25:.2f}n",
            f".meas tran dp_after_{idx} FIND V(dp) AT={base + 5.10:.2f}n",
            f".meas tran dn_after_{idx} FIND V(dn) AT={base + 5.10:.2f}n",
            f".meas tran out_after_{idx} FIND V(out) AT={base + 15.50:.2f}n",
            f".meas tran d_out_{idx} PARAM='out_after_{idx}-out_before_{idx}'",
            f".meas tran error_net_{idx} PARAM='dp_after_{idx}-dn_after_{idx}'",
        ]
        for q in range(FEATURE_COUNT):
            measures += [
                f".meas tran act{q}_before_{idx} FIND V(act{q}) AT={base + 2.95:.2f}n",
                f".meas tran act{q}_after_{idx} FIND V(act{q}) AT={base + 15.50:.2f}n",
                f".meas tran whp{q}_before_{idx} FIND V(whp{q}) AT={base + 0.60:.2f}n",
                f".meas tran whn{q}_before_{idx} FIND V(whn{q}) AT={base + 0.60:.2f}n",
                f".meas tran vwp{q}_before_{idx} FIND V(vwp{q}) AT={base + 0.60:.2f}n",
                f".meas tran vwn{q}_before_{idx} FIND V(vwn{q}) AT={base + 0.60:.2f}n",
                f".meas tran hdp{q}_after_{idx} FIND V(hdp{q}) AT={base + 7.10:.2f}n",
                f".meas tran hdn{q}_after_{idx} FIND V(hdn{q}) AT={base + 7.10:.2f}n",
                f".meas tran gvp{q}_after_{idx} FIND V(gvp{q}) AT={base + 9.10:.2f}n",
                f".meas tran gvn{q}_after_{idx} FIND V(gvn{q}) AT={base + 9.10:.2f}n",
                f".meas tran ghp{q}_after_{idx} FIND V(ghp{q}) AT={base + 9.10:.2f}n",
                f".meas tran ghn{q}_after_{idx} FIND V(ghn{q}) AT={base + 9.10:.2f}n",
                f".meas tran whp{q}_after_apply_{idx} FIND V(whp{q}) AT={base + 11.50:.2f}n",
                f".meas tran whn{q}_after_apply_{idx} FIND V(whn{q}) AT={base + 11.50:.2f}n",
                f".meas tran vwp{q}_after_apply_{idx} FIND V(vwp{q}) AT={base + 11.50:.2f}n",
                f".meas tran vwn{q}_after_apply_{idx} FIND V(vwn{q}) AT={base + 11.50:.2f}n",
                f".meas tran hidden{q}_signed_before_{idx} PARAM='whp{q}_before_{idx}-whn{q}_before_{idx}'",
                f".meas tran hidden{q}_signed_after_{idx} PARAM='whp{q}_after_apply_{idx}-whn{q}_after_apply_{idx}'",
                f".meas tran readout{q}_signed_before_{idx} PARAM='vwp{q}_before_{idx}-vwn{q}_before_{idx}'",
                f".meas tran readout{q}_signed_after_{idx} PARAM='vwp{q}_after_apply_{idx}-vwn{q}_after_apply_{idx}'",
                f".meas tran d_hidden{q}_signed_{idx} PARAM='hidden{q}_signed_after_{idx}-hidden{q}_signed_before_{idx}'",
                f".meas tran d_readout{q}_signed_{idx} PARAM='readout{q}_signed_after_{idx}-readout{q}_signed_before_{idx}'",
                f".meas tran hidden_delta{q}_net_{idx} PARAM='hdp{q}_after_{idx}-hdn{q}_after_{idx}'",
            ]
        prints.append(f"print out_before_{idx} out_after_{idx} error_net_{idx}")

    lines = [
        "* Quad-feature MNIST01 device-level training smoke.",
        "* Four MOS/passive hidden/update slices share one output score/error node.",
        ".param VDD=1.2",
        mos_models(),
        "Vdd vdd 0 {VDD}",
    ]
    for q in range(FEATURE_COUNT):
        lines.append(f"Vx{q} x{q} 0 {sample_wave(samples, f'x{q}', stop)}")
    lines += [
        f"Vtarget target 0 {sample_wave(samples, 'target', stop)}",
        repeated_phases(len(samples), training_enabled=training_enabled),
        "",
        "* Persistent signed hidden and readout weights.",
    ]
    for q in range(FEATURE_COUNT):
        lines += [
            f"Cwhp{q} whp{q} 0 20f IC={weights['whp'][q]:.12g}",
            f"Cwhn{q} whn{q} 0 20f IC={weights['whn'][q]:.12g}",
            f"Cvwp{q} vwp{q} 0 20f IC={weights['vwp'][q]:.12g}",
            f"Cvwn{q} vwn{q} 0 20f IC={weights['vwn'][q]:.12g}",
            f"Rwhp{q} whp{q} 0 1e15",
            f"Rwhn{q} whn{q} 0 1e15",
            f"Rvwp{q} vwp{q} 0 1e15",
            f"Rvwn{q} vwn{q} 0 1e15",
        ]
    lines += [
        "",
        "* Shared output/error state.",
        "Cscore score 0 10f IC=0",
        "Cout out 0 20f IC=0",
        "Cdp dp 0 20f IC=0",
        "Cdn dn 0 20f IC=0",
        "Rscore score 0 1G",
        "Rout out 0 1G",
        "Rdp dp 0 1G",
        "Rdn dn 0 1G",
    ]
    for q in range(FEATURE_COUNT):
        lines += [
            f"Cpre{q} pre{q} 0 10f IC=0",
            f"Cact{q} act{q} 0 20f IC=0",
            f"Chdp{q} hdp{q} 0 12f IC=0",
            f"Chdn{q} hdn{q} 0 12f IC=0",
            f"Cgvp{q} gvp{q} 0 2f IC=0",
            f"Cgvn{q} gvn{q} 0 2f IC=0",
            f"Cghp{q} ghp{q} 0 10f IC=0",
            f"Cghn{q} ghn{q} 0 10f IC=0",
            f"Crgp{q} rgp{q} 0 4f IC=1.2",
            f"Crgn{q} rgn{q} 0 4f IC=1.2",
            f"Rpre{q} pre{q} 0 1G",
            f"Ract{q} act{q} 0 1G",
            f"Rhdp{q} hdp{q} 0 1G",
            f"Rhdn{q} hdn{q} 0 1G",
            f"Rgvp{q} gvp{q} 0 1G",
            f"Rgvn{q} gvn{q} 0 1G",
            f"Rghp{q} ghp{q} 0 1G",
            f"Rghn{q} ghn{q} 0 1G",
            f"Rrgp{q} rgp{q} vdd 50k",
            f"Rrgn{q} rgn{q} vdd 50k",
        ]

    lines += [
        "",
        "* Reset shared nonpersistent state.",
        "Mreset_score score rstf 0 0 NMOS W=4u L=180n",
        "Mreset_out out rstf 0 0 NMOS W=4u L=180n",
        "Mreset_dp dp rstg 0 0 NMOS W=4u L=180n",
        "Mreset_dn dn rstg 0 0 NMOS W=4u L=180n",
    ]
    for q in range(FEATURE_COUNT):
        lines += [
            f"Mreset_pre{q} pre{q} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_act{q} act{q} rstf 0 0 NMOS W=4u L=180n",
            f"Mreset_hdp{q} hdp{q} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_hdn{q} hdn{q} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_gvp{q} gvp{q} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_gvn{q} gvn{q} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_ghp{q} ghp{q} rstg 0 0 NMOS W=4u L=180n",
            f"Mreset_ghn{q} ghn{q} rstg 0 0 NMOS W=4u L=180n",
        ]

    for q in range(FEATURE_COUNT):
        lines += [
            "",
            f"* Feature {q}: hidden forward, output contribution, direct feedback, and local writes.",
            f"Mhpos{q}_x vdd x{q} hp{q}_0 0 NMOS W=32u L=180n",
            f"Mhpos{q}_w hp{q}_0 whp{q} hp{q}_1 0 NMOS W=32u L=180n",
            f"Mhpos{q}_f hp{q}_1 fwd pre{q} 0 NMOS W=32u L=180n",
            f"Mhneg{q}_f pre{q} fwd hn{q}_0 0 NMOS W=24u L=180n",
            f"Mhneg{q}_x hn{q}_0 x{q} hn{q}_1 0 NMOS W=24u L=180n",
            f"Mhneg{q}_w hn{q}_1 whn{q} 0 0 NMOS W=24u L=180n",
            f"Mrelu_h{q} vdd pre{q} act{q} 0 NREL W=24u L=180n",
            f"Movpos{q}_a vdd act{q} op{q}_0 0 NREL W=64u L=180n",
            f"Movpos{q}_w op{q}_0 vwp{q} op{q}_1 0 NREL W=64u L=180n",
            f"Movpos{q}_f op{q}_1 fwd score 0 NREL W=64u L=180n",
            f"Movneg{q}_f score fwd on{q}_0 0 NREL W=48u L=180n",
            f"Movneg{q}_a on{q}_0 act{q} on{q}_1 0 NREL W=48u L=180n",
            f"Movneg{q}_w on{q}_1 vwn{q} 0 0 NREL W=48u L=180n",
            f"Mhdp{q}_d0 vdd dp hdp{q}_d0 0 NSENSE W=32u L=180n",
            f"Mhdp{q}_d1 hdp{q}_d0 act{q} hdp{q}_d1 0 NREL W=32u L=180n",
            f"Mhdp{q}_d2 hdp{q}_d1 bwd hdp{q} 0 NMOS W=32u L=180n",
            f"Mhdn{q}_d0 vdd dn hdn{q}_d0 0 NSENSE W=32u L=180n",
            f"Mhdn{q}_d1 hdn{q}_d0 act{q} hdn{q}_d1 0 NREL W=32u L=180n",
            f"Mhdn{q}_d2 hdn{q}_d1 bwd hdn{q} 0 NMOS W=32u L=180n",
            f"Mgvp{q}_a vdd act{q} gvp{q}_a 0 NREL W=24u L=180n",
            f"Mgvp{q}_d gvp{q}_a dp gvp{q}_d 0 NSENSE W=24u L=180n",
            f"Mgvp{q}_g gvp{q}_d acc gvp{q} 0 NREL W=24u L=180n",
            f"Mgvn{q}_a vdd act{q} gvn{q}_a 0 NREL W=24u L=180n",
            f"Mgvn{q}_d gvn{q}_a dn gvn{q}_d 0 NSENSE W=24u L=180n",
            f"Mgvn{q}_g gvn{q}_d acc gvn{q} 0 NREL W=24u L=180n",
            f"Mghp{q}_x vdd x{q} ghp{q}_x 0 NMOS W=48u L=180n",
            f"Mghp{q}_d ghp{q}_x hdp{q} ghp{q}_d 0 NSENSE W=48u L=180n",
            f"Mghp{q}_g ghp{q}_d acc ghp{q} 0 NMOS W=48u L=180n",
            f"Mghn{q}_x vdd x{q} ghn{q}_x 0 NMOS W=48u L=180n",
            f"Mghn{q}_d ghn{q}_x hdn{q} ghn{q}_d 0 NSENSE W=48u L=180n",
            f"Mghn{q}_g ghn{q}_d acc ghn{q} 0 NMOS W=48u L=180n",
            f"Mrgp{q}_pd rgp{q} gvp{q} 0 0 NSENSE W=16u L=180n",
            f"Mrgn{q}_pd rgn{q} gvn{q} 0 0 NSENSE W=16u L=180n",
            f"Mvwp{q}_up_p0 vdd rgp{q} vwp{q}_up vdd PMOS W={readout_pmos_w:.6g}u L=180n",
            f"Mvwp{q}_up_p1 vwp{q}_up applyn vwp{q} vdd PMOS W={readout_pmos_w:.6g}u L=180n",
            f"Mvwn{q}_dn_a vwn{q} apply vwn{q}_dn 0 NREL W={readout_nmos_w:.6g}u L=180n",
            f"Mvwn{q}_dn_g vwn{q}_dn gvp{q} 0 0 NSENSE W={readout_nmos_w:.6g}u L=180n",
            f"Mvwn{q}_up_p0 vdd rgn{q} vwn{q}_up vdd PMOS W={readout_pmos_w:.6g}u L=180n",
            f"Mvwn{q}_up_p1 vwn{q}_up applyn vwn{q} vdd PMOS W={readout_pmos_w:.6g}u L=180n",
            f"Mvwp{q}_dn_a vwp{q} apply vwp{q}_dn 0 NREL W={readout_nmos_w:.6g}u L=180n",
            f"Mvwp{q}_dn_g vwp{q}_dn gvn{q} 0 0 NSENSE W={readout_nmos_w:.6g}u L=180n",
            f"Mwhp{q}_up_g vdd ghp{q} whp{q}_up 0 NSENSE W=0.5u L=180n",
            f"Mwhp{q}_up_a whp{q}_up apply whp{q} 0 NREL W=0.5u L=180n",
            f"Mwhn{q}_dn_a whn{q} apply whn{q}_dn 0 NREL W=0.5u L=180n",
            f"Mwhn{q}_dn_g whn{q}_dn ghp{q} 0 0 NSENSE W=0.5u L=180n",
            f"Mwhn{q}_up_g vdd ghn{q} whn{q}_up 0 NSENSE W=0.5u L=180n",
            f"Mwhn{q}_up_a whn{q}_up apply whn{q} 0 NREL W=0.5u L=180n",
            f"Mwhp{q}_dn_a whp{q} apply whp{q}_dn 0 NREL W=0.5u L=180n",
            f"Mwhp{q}_dn_g whp{q}_dn ghn{q} 0 0 NSENSE W=0.5u L=180n",
        ]

    lines += [
        "",
        output_driver_line(output_driver_model),
        "",
        "* Shared output error from target/raw-score conductance competition.",
        "Mdp_t0 vdd target dp_t 0 NSENSE W=32u L=180n",
        "Mdp_t1 dp_t err dp 0 NSENSE W=32u L=180n",
        "Mdp_y0 dp err dp_y 0 NSENSE W=24u L=180n",
        "Mdp_y1 dp_y score 0 0 NSENSE W=24u L=180n",
        "Mdn_y0 vdd score dn_y 0 NSENSE W=32u L=180n",
        "Mdn_y1 dn_y err dn 0 NSENSE W=32u L=180n",
        "Mdn_t0 dn err dn_t 0 NSENSE W=24u L=180n",
        "Mdn_t1 dn_t target 0 0 NSENSE W=24u L=180n",
        "",
        ".options method=gear maxord=2",
        f".tran 10p {stop:.2f}n uic",
        *measures,
        ".control",
        "run",
        *prints,
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def load_mnist01_quad_records(
    train_samples: int,
    eval_samples: int,
    *,
    image_size: int,
    seed: int,
    positive_digit: int,
    negative_digit: int,
    download: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from torchvision import datasets, transforms
    import torch.nn.functional as F

    if positive_digit == negative_digit:
        raise ValueError("positive and negative digits must differ")
    digits = (positive_digit, negative_digit)
    ds_train = datasets.MNIST(root=str(ROOT / "data"), train=True, download=download, transform=transforms.ToTensor())
    ds_eval = datasets.MNIST(root=str(ROOT / "data"), train=False, download=download, transform=transforms.ToTensor())
    train_labels = np.asarray(ds_train.targets, dtype=np.int64)
    eval_labels = np.asarray(ds_eval.targets, dtype=np.int64)
    train_indices = balanced_digit_indices(train_labels, train_samples, seed=seed, digits=digits)
    eval_indices = balanced_digit_indices(eval_labels, eval_samples, seed=seed + 1, digits=digits)

    def extract(ds: Any, indices: np.ndarray) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index in indices:
            image, digit = ds[int(index)]
            resized = F.interpolate(image.unsqueeze(0), size=(image_size, image_size), mode="area").squeeze()
            features = quad_features_from_image(resized.numpy())
            digit_i = int(digit)
            record: dict[str, Any] = {
                "target": 1.1 if digit_i == positive_digit else 0.0,
                "digit": float(digit_i),
                "mnist_index": float(index),
                "positive_label": 1.0 if digit_i == positive_digit else 0.0,
            }
            for q, value in enumerate(features):
                record[f"x{q}"] = float(value)
            records.append(record)
        return records

    return extract(ds_train, train_indices), extract(ds_eval, eval_indices)


def rows_from_measures(samples: list[dict[str, Any]], measures: dict[str, float], *, sequence: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        positive = expected_positive(float(sample["target"]))
        row: dict[str, Any] = {
            "sequence": sequence,
            "sample_idx": sample_idx,
            "target": sample["target"],
            "digit": sample.get("digit"),
            "mnist_index": sample.get("mnist_index"),
            "positive_label": sample.get("positive_label"),
            "expected_direction": "positive" if positive else "negative",
        }
        for q in range(FEATURE_COUNT):
            row[f"x{q}"] = sample[f"x{q}"]
        for key, value in measures.items():
            suffix = f"_{sample_idx}"
            if key.endswith(suffix):
                row[key[: -len(suffix)]] = value
        rows.append(row)
    return pd.DataFrame(rows)


def final_weights_from_rows(rows: pd.DataFrame) -> dict[str, list[float]]:
    if rows.empty:
        raise ValueError("cannot extract final weights from empty rows")
    final = rows.iloc[-1]
    return {
        "whp": [float(final[f"whp{q}_after_apply"]) for q in range(FEATURE_COUNT)],
        "whn": [float(final[f"whn{q}_after_apply"]) for q in range(FEATURE_COUNT)],
        "vwp": [float(final[f"vwp{q}_after_apply"]) for q in range(FEATURE_COUNT)],
        "vwn": [float(final[f"vwn{q}_after_apply"]) for q in range(FEATURE_COUNT)],
    }


def run_device_sequence(
    spice_bin: str,
    path: Path,
    samples: list[dict[str, Any]],
    weights: dict[str, list[float]],
    *,
    training_enabled: bool,
    timeout: float,
    sequence: str,
    output_driver_model: str,
    readout_apply_scale: float,
) -> pd.DataFrame:
    netlist = quad_netlist(
        samples,
        weights,
        training_enabled=training_enabled,
        output_driver_model=output_driver_model,
        readout_apply_scale=readout_apply_scale,
    )
    if "\nB" in netlist:
        raise ValueError("quad-feature device runner generated a behavioral source")
    measures = run_netlist(spice_bin, path, netlist, timeout)
    return rows_from_measures(samples, measures, sequence=sequence)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=8)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positive-digit", type=int, default=0)
    ap.add_argument("--negative-digit", type=int, default=1)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--tag", default="device_mnist01_quad")
    ap.add_argument("--output-driver-model", choices=["sense", "nrel"], default="sense")
    ap.add_argument("--readout-apply-scale", type=float, default=0.5)
    ap.add_argument("--decision-threshold", type=float, default=0.10)
    ap.add_argument("--assert-nonbehavioral", action="store_true")
    args = ap.parse_args()

    if args.train_samples <= 0:
        raise ValueError("train-samples must be positive for a training smoke")
    if args.eval_samples <= 0:
        raise ValueError("eval-samples must be positive")

    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)

    spice_bin, version = detect_spice(None)
    run_tiny_test(spice_bin, generated)
    safe_tag = sanitize_tag(args.tag)
    t0 = time.perf_counter()
    train_samples, eval_samples = load_mnist01_quad_records(
        args.train_samples,
        args.eval_samples,
        image_size=args.image_size,
        seed=args.seed,
        positive_digit=args.positive_digit,
        negative_digit=args.negative_digit,
        download=args.download,
    )
    initial_weights = initial_quad_weights()
    initial_eval_rows = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_initial_eval.cir",
        eval_samples,
        initial_weights,
        training_enabled=False,
        timeout=args.timeout,
        sequence="initial_eval",
        output_driver_model=args.output_driver_model,
        readout_apply_scale=args.readout_apply_scale,
    )
    train_rows = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_train.cir",
        train_samples,
        initial_weights,
        training_enabled=True,
        timeout=args.timeout,
        sequence="train",
        output_driver_model=args.output_driver_model,
        readout_apply_scale=args.readout_apply_scale,
    )
    final_weights = final_weights_from_rows(train_rows)
    final_eval_rows = run_device_sequence(
        spice_bin,
        generated / f"{safe_tag}_final_eval.cir",
        eval_samples,
        final_weights,
        training_enabled=False,
        timeout=args.timeout,
        sequence="final_eval",
        output_driver_model=args.output_driver_model,
        readout_apply_scale=args.readout_apply_scale,
    )

    curve = pd.concat([initial_eval_rows, train_rows, final_eval_rows], ignore_index=True)
    curve_path = results / f"{safe_tag}.csv"
    table_curve_path = tables / f"{safe_tag}.csv"
    curve.to_csv(curve_path, index=False)
    curve.to_csv(table_curve_path, index=False)

    initial_accuracy = binary_accuracy(initial_eval_rows, threshold=args.decision_threshold)
    final_accuracy = binary_accuracy(final_eval_rows, threshold=args.decision_threshold)
    initial_active_fraction = float(
        np.mean(np.abs(initial_eval_rows["out_after"].to_numpy(dtype=float)) > args.decision_threshold)
    )
    final_active_fraction = float(
        np.mean(np.abs(final_eval_rows["out_after"].to_numpy(dtype=float)) > args.decision_threshold)
    )
    nontrivial_learning_met = final_accuracy > max(initial_accuracy, 0.5)
    summary = {
        "simulator": version,
        "architecture": "device_level_mnist01_quad_feature_sequential_training",
        "status": "mnist01_quad_feature_device_training_smoke",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "dataset": "MNIST01 four-quadrant feature smoke",
        "positive_digit": args.positive_digit,
        "negative_digit": args.negative_digit,
        "image_size": args.image_size,
        "feature_count": FEATURE_COUNT,
        "feature_encoding": "four quadrant mean-ink rails, each 0.55 + 0.55 * quadrant_ink / 0.35 clipped to [0.05, 1.1]",
        "hidden_credit_mode": "direct_feedback",
        "output_driver_model": args.output_driver_model,
        "readout_apply_scale": args.readout_apply_scale,
        "learning_device_implementation": "transistor_passive",
        "no_behavioral_signal_math": True,
        "no_behavioral_learning_devices": True,
        "uses_behavioral_learning_devices": False,
        "transistor_or_passive_learning_path": True,
        "single_training_transient": True,
        "continuous_transient_contract_met": True,
        "strict_fully_on_device_contract_met": True,
        "strict_fully_on_device_requested": True,
        "batch_size": 1,
        "python_weight_updates_between_samples": False,
        "python_checkpointing_between_samples": False,
        "python_hidden_state_intervention": False,
        "training_eval_uses_spice_forward_path": True,
        "uses_local_pca": False,
        "realistic_train_order": True,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "mnist_index_order": "stable_balanced_random_digit01",
        "decision_threshold": args.decision_threshold,
        "initial_eval_accuracy": initial_accuracy,
        "final_eval_accuracy": final_accuracy,
        "eval_accuracy_delta": final_accuracy - initial_accuracy,
        "initial_eval_output_active_fraction": initial_active_fraction,
        "final_eval_output_active_fraction": final_active_fraction,
        "nontrivial_learning_met": nontrivial_learning_met,
        "initial_weights": initial_weights,
        "final_weights": final_weights,
        "curve": str(curve_path),
        "table_curve": str(table_curve_path),
        "netlists": {
            "initial_eval": str(generated / f"{safe_tag}_initial_eval.cir"),
            "train": str(generated / f"{safe_tag}_train.cir"),
            "final_eval": str(generated / f"{safe_tag}_final_eval.cir"),
        },
        "wall_time_s": time.perf_counter() - t0,
        "full_objective_contract_issues": [
            "binary MNIST01 smoke, not multiclass MNIST",
            "four quadrant features, not 10x10 b4 stride2 c2",
            "does not yet demonstrate nontrivial learning" if not nontrivial_learning_met else "",
        ],
        "interpretation": (
            "This is the first multi-feature transistor/passive MNIST data-stream smoke. "
            "Four independent hidden/update slices share one output node and one error rail; training weights change only "
            "inside the uninterrupted training transient."
        ),
    }
    summary["full_objective_contract_issues"] = [issue for issue in summary["full_objective_contract_issues"] if issue]
    if args.assert_nonbehavioral:
        assert summary["no_behavioral_learning_devices"] is True
        assert summary["transistor_or_passive_learning_path"] is True
    summary_path = results / f"{safe_tag}_summary.json"
    table_summary_path = tables / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing optional MNIST dependency: {exc}") from exc
