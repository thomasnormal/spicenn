from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_local_feature_batch_op_train import run_eval, run_train_batch
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def phase_pwl(pulses: list[tuple[float, float]], t_stop: float, edge: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for t_on, t_off in pulses:
        points.append((max(0.0, t_on - edge), 0.0))
        points.append((t_on, 1.0))
        points.append((t_off, 1.0))
        points.append((min(t_stop, t_off + edge), 0.0))
    points.append((t_stop, 0.0))
    cleaned: list[tuple[float, float]] = []
    for t, val in sorted(points):
        if cleaned and abs(cleaned[-1][0] - t) < 1e-18:
            cleaned[-1] = (t, val)
        else:
            cleaned.append((t, val))
    return "PWL(" + " ".join(f"{t:.12g} {val:.12g}" for t, val in cleaned) + ")"


def sample_source_pwl(values: np.ndarray, sample_starts: list[float], t_stop: float, edge: float) -> str:
    points: list[tuple[float, float]] = [(0.0, float(values[0]))]
    for s, val in enumerate(values):
        t = sample_starts[s]
        prev = float(values[s - 1] if s > 0 else val)
        points.append((max(0.0, t - edge), prev))
        points.append((t, float(val)))
    points.append((t_stop, float(values[-1])))
    cleaned: list[tuple[float, float]] = []
    for t, val in points:
        if cleaned and abs(cleaned[-1][0] - t) < 1e-18:
            cleaned[-1] = (t, val)
        else:
            cleaned.append((t, val))
    return "PWL(" + " ".join(f"{t:.12g} {val:.12g}" for t, val in cleaned) + ")"


def tanh_expr(expr: str) -> str:
    return f"(2/(1+exp(-2*({expr})))-1)"


def target_matrix(labels: np.ndarray, n_classes: int, softmax_output: bool = False) -> np.ndarray:
    targets = np.zeros((len(labels), n_classes)) if softmax_output else -np.ones((len(labels), n_classes))
    for s, label in enumerate(labels):
        targets[s, int(label)] = 1.0
    return targets


def parse_measured_vector(stdout: str, n_vec: int) -> np.ndarray:
    vals = np.empty(n_vec, dtype=float)
    for i in range(n_vec):
        name = f"m{i:05d}"
        m = re.search(rf"(?im)^\s*{name}\s*=\s*([-+0-9.eE]+)", stdout)
        if not m:
            raise ValueError(f"missing final-state measurement {name}")
        vals[i] = float(m.group(1))
    return vals


def make_phase_schedule(
    update_batch_size: int,
    updates: int,
    phase: float,
    gap: float,
) -> tuple[dict[str, list[tuple[float, float]]], list[float], float]:
    phases = {"act": [], "score": [], "err": [], "bwd": [], "acc": [], "apply": [], "clear": []}
    sample_starts: list[float] = []
    t = phase
    for _update in range(updates):
        for _sample in range(update_batch_size):
            sample_starts.append(t)
            t += gap
            phases["act"].append((t, t + phase))
            t += phase + gap
            phases["score"].append((t, t + phase))
            t += phase + gap
            phases["err"].append((t, t + phase))
            t += phase + gap
            phases["bwd"].append((t, t + phase))
            t += phase + gap
            phases["acc"].append((t, t + phase))
            t += phase + gap
        phases["apply"].append((t, t + phase))
        t += phase + gap
        phases["clear"].append((t, t + phase))
        t += phase + gap
    return phases, sample_starts, t + phase


def make_phase_transient_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    w: np.ndarray,
    hb: np.ndarray,
    readout: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[list[int]],
    lr: float,
    out_path: Path,
    linear_output: bool,
    update_batch_size: int,
    updates: int,
    phase: float,
    gap: float,
    edge: float,
    settle_ratio: float,
    transient_step: float,
    cw: float,
    cstate: float,
    cgrad: float,
    rleak: float,
    softmax_output: bool = False,
    output_mode: str = "wrdata",
) -> tuple[str, int, float]:
    total_samples = x_batch.shape[0]
    if total_samples != update_batch_size * updates:
        raise ValueError("x_batch length must equal update_batch_size * updates")
    n_blocks, channels, block_len = w.shape
    n_classes = readout.shape[0]
    phases, sample_starts, t_stop = make_phase_schedule(update_batch_size, updates, phase, gap)
    tau = phase / settle_ratio
    targets = target_matrix(y_batch, n_classes, softmax_output)
    lines = [
        "* Phase-resolved transient local-feature/readout training deck.",
        "* Persistent local feature and readout parameters are capacitor voltages.",
        "* Feature activations, class deltas, backward deltas, and gradients are capacitor voltages.",
        f".param LR={lr:.12g}",
        f".param BS={update_batch_size}",
        f".param CW={cw:.12g}",
        f".param CSTATE={cstate:.12g}",
        f".param CGRAD={cgrad:.12g}",
        f".param RLEAK={rleak:.12g}",
        f".param TAU={tau:.12g}",
        f".param TPHASE={phase:.12g}",
        "",
        f"Vpact pact 0 {phase_pwl(phases['act'], t_stop, edge)}",
        f"Vpscore pscore 0 {phase_pwl(phases['score'], t_stop, edge)}",
        f"Vperr perr 0 {phase_pwl(phases['err'], t_stop, edge)}",
        f"Vpbwd pbwd 0 {phase_pwl(phases['bwd'], t_stop, edge)}",
        f"Vpacc pacc 0 {phase_pwl(phases['acc'], t_stop, edge)}",
        f"Vpapply papply 0 {phase_pwl(phases['apply'], t_stop, edge)}",
        f"Vpclear pclear 0 {phase_pwl(phases['clear'], t_stop, edge)}",
        "",
    ]
    for i in range(x_batch.shape[1]):
        lines.append(f"Vpix{i} pix{i} 0 {sample_source_pwl(x_batch[:, i], sample_starts, t_stop, edge)}")
    for k in range(n_classes):
        lines.append(f"Vtarget{k} target{k} 0 {sample_source_pwl(targets[:, k], sample_starts, t_stop, edge)}")
    lines.append("")
    for b in range(n_blocks):
        for c in range(channels):
            for p in range(block_len):
                lines.append(f"Cw{b}_{c}_{p} w{b}_{c}_{p} 0 {{CW}} IC={w[b, c, p]:.12g}")
                lines.append(f"Rw{b}_{c}_{p} w{b}_{c}_{p} 0 {{RLEAK}}")
                lines.append(f"Cgw{b}_{c}_{p} gw{b}_{c}_{p} 0 {{CGRAD}} IC=0")
            lines.append(f"Chb{b}_{c} hb{b}_{c} 0 {{CW}} IC={hb[b, c]:.12g}")
            lines.append(f"Rhb{b}_{c} hb{b}_{c} 0 {{RLEAK}}")
            lines.append(f"Cghb{b}_{c} ghb{b}_{c} 0 {{CGRAD}} IC=0")
            lines.append(f"Ch{b}_{c} h{b}_{c} 0 {{CSTATE}} IC=0")
            lines.append(f"Cdh{b}_{c} dh{b}_{c} 0 {{CSTATE}} IC=0")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                lines.append(f"Cv{k}_{b}_{c} v{k}_{b}_{c} 0 {{CW}} IC={readout[k, b, c]:.12g}")
                lines.append(f"Rv{k}_{b}_{c} v{k}_{b}_{c} 0 {{RLEAK}}")
                lines.append(f"Cgv{k}_{b}_{c} gv{k}_{b}_{c} 0 {{CGRAD}} IC=0")
        lines.append(f"Cob{k} ob{k} 0 {{CW}} IC={output_bias[k]:.12g}")
        lines.append(f"Rob{k} ob{k} 0 {{RLEAK}}")
        lines.append(f"Cgob{k} gob{k} 0 {{CGRAD}} IC=0")
        lines.append(f"Cscore{k} score{k} 0 {{CSTATE}} IC=0")
        lines.append(f"Cd{k} d{k} 0 {{CSTATE}} IC=0")
    lines.append("")
    for b, idxs in enumerate(blocks):
        for c in range(channels):
            terms = [f"V(w{b}_{c}_{p})*V(pix{idx})" for p, idx in enumerate(idxs)]
            terms.append(f"V(hb{b}_{c})")
            h_calc = tanh_expr(" + ".join(terms))
            lines.append(f"Bstore_h{b}_{c} h{b}_{c} 0 I = V(pact)*{{CSTATE}}/{{TAU}}*(V(h{b}_{c})-({h_calc}))")
    for k in range(n_classes):
        score_terms = [f"V(v{k}_{b}_{c})*V(h{b}_{c})" for b in range(n_blocks) for c in range(channels)]
        score_terms.append(f"V(ob{k})")
        lines.append(f"Bscore{k} scorecalc{k} 0 V = " + " + ".join(score_terms))
        lines.append(f"Bstore_score{k} score{k} 0 I = V(pscore)*{{CSTATE}}/{{TAU}}*(V(score{k})-V(scorecalc{k}))")
    if softmax_output:
        denom = " + ".join(f"exp(V(score{k}))" for k in range(n_classes))
        for k in range(n_classes):
            lines.append(f"By{k} y{k} 0 V = exp(V(score{k}))/({denom})")
            lines.append(f"Bstore_d{k} d{k} 0 I = V(perr)*{{CSTATE}}/{{TAU}}*(V(d{k})-(V(target{k})-V(y{k})))")
    else:
        for k in range(n_classes):
            if linear_output:
                lines.append(f"By{k} y{k} 0 V = V(score{k})")
                delta_expr = f"(V(target{k})-V(y{k}))"
            else:
                y_expr = tanh_expr(f"V(score{k})")
                lines.append(f"By{k} y{k} 0 V = {y_expr}")
                delta_expr = f"(V(target{k})-({y_expr}))*(1-({y_expr})*({y_expr}))"
            lines.append(f"Bstore_d{k} d{k} 0 I = V(perr)*{{CSTATE}}/{{TAU}}*(V(d{k})-({delta_expr}))")
    lines.append("")
    for b, idxs in enumerate(blocks):
        for c in range(channels):
            feedback = " + ".join(f"V(v{k}_{b}_{c})*V(d{k})" for k in range(n_classes))
            local_delta = f"({feedback})*(1-V(h{b}_{c})*V(h{b}_{c}))"
            lines.append(f"Bstore_dh{b}_{c} dh{b}_{c} 0 I = V(pbwd)*{{CSTATE}}/{{TAU}}*(V(dh{b}_{c})-({local_delta}))")
            for p, idx in enumerate(idxs):
                grad = f"V(dh{b}_{c})*V(pix{idx})"
                lines.append(f"Bacc_w{b}_{c}_{p} gw{b}_{c}_{p} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*({grad})")
                lines.append(f"Bupd_w{b}_{c}_{p} w{b}_{c}_{p} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gw{b}_{c}_{p})")
                lines.append(f"Bclear_gw{b}_{c}_{p} gw{b}_{c}_{p} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gw{b}_{c}_{p})")
            lines.append(f"Bacc_hb{b}_{c} ghb{b}_{c} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*V(dh{b}_{c})")
            lines.append(f"Bupd_hb{b}_{c} hb{b}_{c} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(ghb{b}_{c})")
            lines.append(f"Bclear_ghb{b}_{c} ghb{b}_{c} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(ghb{b}_{c})")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                grad = f"V(d{k})*V(h{b}_{c})"
                lines.append(f"Bacc_v{k}_{b}_{c} gv{k}_{b}_{c} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*({grad})")
                lines.append(f"Bupd_v{k}_{b}_{c} v{k}_{b}_{c} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gv{k}_{b}_{c})")
                lines.append(f"Bclear_gv{k}_{b}_{c} gv{k}_{b}_{c} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gv{k}_{b}_{c})")
        lines.append(f"Bacc_ob{k} gob{k} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*V(d{k})")
        lines.append(f"Bupd_ob{k} ob{k} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gob{k})")
        lines.append(f"Bclear_gob{k} gob{k} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gob{k})")
    vectors = [f"V(w{b}_{c}_{p})" for b in range(n_blocks) for c in range(channels) for p in range(block_len)]
    vectors += [f"V(hb{b}_{c})" for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(v{k}_{b}_{c})" for k in range(n_classes) for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(ob{k})" for k in range(n_classes)]
    vectors += [f"V(y{k})" for k in range(n_classes)]
    if output_mode not in {"wrdata", "measure"}:
        raise ValueError("output_mode must be 'wrdata' or 'measure'")
    measure_time = max(0.0, t_stop - transient_step)
    lines += ["", ".options method=gear maxord=2", ".control", f"tran {transient_step:.12g} {t_stop:.12g} uic"]
    if output_mode == "wrdata":
        lines.append(f"wrdata {out_path} " + " ".join(vectors))
    else:
        for i, vec in enumerate(vectors):
            lines.append(f"meas tran m{i:05d} FIND {vec} AT={measure_time:.12g}")
    lines += [".endc", ".end", ""]
    return "\n".join(lines), len(vectors), t_stop


def unpack_state(
    vals: np.ndarray,
    w: np.ndarray,
    hb: np.ndarray,
    readout: np.ndarray,
    output_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_classes = readout.shape[0]
    offset = 0
    nw = vals[offset : offset + w.size].reshape(w.shape)
    offset += w.size
    nhb = vals[offset : offset + hb.size].reshape(hb.shape)
    offset += hb.size
    nv = vals[offset : offset + readout.size].reshape(readout.shape)
    offset += readout.size
    nob = vals[offset : offset + output_bias.size]
    offset += output_bias.size
    y = vals[offset : offset + n_classes]
    return nw, nhb, nv, nob, y


def state_metrics(
    ref: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    got: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    names = ["local_weights", "local_bias", "readout", "output_bias"]
    metrics: dict[str, float] = {}
    for name, a, b in zip(names, ref, got):
        diff = np.asarray(b) - np.asarray(a)
        metrics[f"{name}_max_abs_diff"] = float(np.max(np.abs(diff)))
        metrics[f"{name}_mean_abs_diff"] = float(np.mean(np.abs(diff)))
        metrics[f"{name}_rms_diff"] = float(np.sqrt(np.mean(diff * diff)))
    all_diff = np.concatenate([(np.asarray(b) - np.asarray(a)).ravel() for a, b in zip(ref, got)])
    metrics["state_max_abs_diff"] = float(np.max(np.abs(all_diff)))
    metrics["state_mean_abs_diff"] = float(np.mean(np.abs(all_diff)))
    metrics["state_rms_diff"] = float(np.sqrt(np.mean(all_diff * all_diff)))
    return metrics


def update_direction_metrics(
    initial: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ref: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    got: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    eps: float = 1e-12,
) -> dict[str, float]:
    ref_delta = np.concatenate([(np.asarray(after) - np.asarray(before)).ravel() for before, after in zip(initial, ref)])
    got_delta = np.concatenate([(np.asarray(after) - np.asarray(before)).ravel() for before, after in zip(initial, got)])
    ref_norm = float(np.linalg.norm(ref_delta))
    got_norm = float(np.linalg.norm(got_delta))
    metrics = {
        "reference_update_l2": ref_norm,
        "phase_update_l2": got_norm,
    }
    if ref_norm > eps and got_norm > eps:
        metrics["state_update_direction_cosine"] = float(np.dot(ref_delta, got_delta) / (ref_norm * got_norm))
    else:
        metrics["state_update_direction_cosine"] = float("nan")
    mask = np.abs(ref_delta) > eps
    if np.any(mask):
        aligned = np.sign(ref_delta[mask]) == np.sign(got_delta[mask])
        metrics["state_update_sign_alignment_fraction"] = float(np.mean(aligned))
        metrics["state_update_wrong_sign_count"] = float(np.size(aligned) - int(np.sum(aligned)))
    else:
        metrics["state_update_sign_alignment_fraction"] = float("nan")
        metrics["state_update_wrong_sign_count"] = 0.0
    return metrics


def load_or_init_weights(
    init_weights: str,
    rng: np.random.Generator,
    n_blocks: int,
    channels: int,
    block_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected = ((n_blocks, channels, block_len), (n_blocks, channels), (10, n_blocks, channels), (10,))
    if init_weights:
        init = np.load(init_weights)
        if not {"local_weights", "local_bias", "readout", "output_bias"}.issubset(init.files):
            raise ValueError(f"expected local-feature checkpoint keys, got {init.files}")
        w = init["local_weights"].copy()
        hb = init["local_bias"].copy()
        readout = init["readout"].copy()
        output_bias = init["output_bias"].copy()
        actual = (w.shape, hb.shape, readout.shape, output_bias.shape)
        if actual != expected:
            raise ValueError(f"initial weight shapes {actual} do not match expected {expected}")
        return w, hb, readout, output_bias
    w = rng.normal(0.0, 0.05, size=expected[0])
    hb = np.zeros(expected[1])
    readout = rng.normal(0.0, 0.05, size=expected[2])
    output_bias = np.zeros(expected[3])
    return w, hb, readout, output_bias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--eval-samples", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--updates", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--phase", type=float, default=2e-9)
    ap.add_argument("--gap", type=float, default=0.2e-9)
    ap.add_argument("--edge", type=float, default=10e-12)
    ap.add_argument("--settle-ratio", type=float, default=40.0)
    ap.add_argument("--transient-step", type=float, default=20e-12)
    ap.add_argument("--cw", type=float, default=1e-12)
    ap.add_argument("--cstate", type=float, default=1e-12)
    ap.add_argument("--cgrad", type=float, default=1e-12)
    ap.add_argument("--rleak", type=float, default=1e18)
    ap.add_argument("--final-measures", action="store_true")
    ap.add_argument("--tag", default="phase_local_feature")
    args = ap.parse_args()

    if args.batch_size <= 0 or args.updates <= 0:
        raise ValueError("--batch-size and --updates must be positive")
    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.channels <= 0:
        raise ValueError("--channels must be positive")
    if args.phase <= 0 or args.settle_ratio <= 0:
        raise ValueError("--phase and --settle-ratio must be positive")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    total_samples = args.batch_size * args.updates
    if args.train_samples < total_samples:
        raise ValueError("--train-samples must cover --batch-size * --updates")
    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, max(1, args.eval_samples), args.image_size, args.seed)
    x_batch = x_train[:total_samples]
    y_batch = y_train[:total_samples]
    rng = np.random.default_rng(args.seed)
    w, hb, readout, output_bias = load_or_init_weights(
        args.init_weights,
        rng,
        len(blocks),
        args.channels,
        args.block_size * args.block_size,
    )

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    generated.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    stem = f"spice_mnist_local_feature_phase_{sanitize_tag(args.tag)}"
    phase_netlist = generated / f"{stem}.cir"
    phase_data = results / f"{stem}.dat"
    op_netlist = generated / f"{stem}_op_reference.cir"
    op_data = results / f"{stem}_op_reference.dat"

    netlist, n_vec, t_stop = make_phase_transient_netlist(
        x_batch,
        y_batch,
        w,
        hb,
        readout,
        output_bias,
        blocks,
        args.lr,
        phase_data,
        args.linear_output,
        args.batch_size,
        args.updates,
        args.phase,
        args.gap,
        args.edge,
        args.settle_ratio,
        args.transient_step,
        args.cw,
        args.cstate,
        args.cgrad,
        args.rleak,
        args.softmax_output,
        "measure" if args.final_measures else "wrdata",
    )
    phase_netlist.write_text(prepare_netlist_for_simulator(netlist, spice_bin))

    t0 = time.perf_counter()
    proc = run_simulator_netlist(spice_bin, phase_netlist, timeout=args.timeout)
    phase_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    vals = parse_measured_vector(proc.stdout + "\n" + proc.stderr, n_vec) if args.final_measures else read_wrdata_row(phase_data, n_vec)
    phase_w, phase_hb, phase_readout, phase_ob, phase_y = unpack_state(vals, w, hb, readout, output_bias)

    t1 = time.perf_counter()
    op_w, op_hb, op_readout, op_ob = w.copy(), hb.copy(), readout.copy(), output_bias.copy()
    for update in range(args.updates):
        start = update * args.batch_size
        stop = start + args.batch_size
        op_w, op_hb, op_readout, op_ob = run_train_batch(
            spice_bin,
            op_netlist,
            op_data,
            x_batch[start:stop],
            y_batch[start:stop],
            op_w,
            op_hb,
            op_readout,
            op_ob,
            blocks,
            args.lr,
            args.timeout,
            linear_output=args.linear_output,
            softmax_output=args.softmax_output,
            local_activation="tanh",
            relu_clip=1.0,
        )
    op_wall = time.perf_counter() - t1
    metrics = state_metrics((op_w, op_hb, op_readout, op_ob), (phase_w, phase_hb, phase_readout, phase_ob))
    metrics.update(update_direction_metrics((w, hb, readout, output_bias), (op_w, op_hb, op_readout, op_ob), (phase_w, phase_hb, phase_readout, phase_ob)))
    phase_eval_accuracy = None
    op_reference_eval_accuracy = None
    eval_wall = 0.0
    if args.eval_samples > 0:
        t2 = time.perf_counter()
        phase_eval_accuracy = run_eval(
            spice_bin,
            generated / f"{stem}_phase_eval.cir",
            results / f"{stem}_phase_eval.dat",
            x_test[: args.eval_samples],
            y_test[: args.eval_samples],
            phase_w,
            phase_hb,
            phase_readout,
            phase_ob,
            blocks,
            max(1, min(args.eval_samples, 50)),
            args.timeout,
            linear_output=args.linear_output,
            softmax_output=args.softmax_output,
            local_activation="tanh",
            relu_clip=1.0,
        )
        op_reference_eval_accuracy = run_eval(
            spice_bin,
            generated / f"{stem}_op_reference_eval.cir",
            results / f"{stem}_op_reference_eval.dat",
            x_test[: args.eval_samples],
            y_test[: args.eval_samples],
            op_w,
            op_hb,
            op_readout,
            op_ob,
            blocks,
            max(1, min(args.eval_samples, 50)),
            args.timeout,
            linear_output=args.linear_output,
            softmax_output=args.softmax_output,
            local_activation="tanh",
            relu_clip=1.0,
        )
        eval_wall = time.perf_counter() - t2
    final_weights_path = results / f"{stem}_final_weights.npz"
    reference_weights_path = results / f"{stem}_op_reference_final_weights.npz"
    metrics_path = results / f"{stem}_equivalence_metrics.csv"
    np.savez_compressed(
        final_weights_path,
        local_weights=phase_w,
        local_bias=phase_hb,
        readout=phase_readout,
        output_bias=phase_ob,
        y=phase_y,
    )
    np.savez_compressed(
        reference_weights_path,
        local_weights=op_w,
        local_bias=op_hb,
        readout=op_readout,
        output_bias=op_ob,
    )
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    summary = {
        "simulator": version,
        "architecture": "phase_resolved_transient_local_feature_readout",
        "status": "one_batch_equivalence_check" if args.updates == 1 else "multi_update_equivalence_check",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels": args.channels,
        "classes": 10,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "total_samples": total_samples,
        "lr": args.lr,
        "linear_output": bool(args.linear_output),
        "softmax_output": bool(args.softmax_output),
        "init_weights": args.init_weights,
        "phase_netlist": str(phase_netlist),
        "phase_data": str(phase_data),
        "op_reference_netlist": str(op_netlist),
        "op_reference_data": str(op_data),
        "final_weights": str(final_weights_path),
        "op_reference_final_weights": str(reference_weights_path),
        "equivalence_metrics": str(metrics_path),
        "phase_wall_time_s": phase_wall,
        "op_reference_wall_time_s": op_wall,
        "eval_wall_time_s": eval_wall,
        "phase_eval_accuracy": phase_eval_accuracy,
        "op_reference_eval_accuracy": op_reference_eval_accuracy,
        "eval_accuracy_abs_diff": (
            abs(phase_eval_accuracy - op_reference_eval_accuracy)
            if phase_eval_accuracy is not None and op_reference_eval_accuracy is not None
            else None
        ),
        "t_stop_s": t_stop,
        "transient_step_s": args.transient_step,
        "phase_s": args.phase,
        "settle_ratio": args.settle_ratio,
        "output_mode": "measure" if args.final_measures else "wrdata",
        "persistent_state": "local feature weights, local biases, class readout weights, and output biases are capacitor voltages with checkpoint ICs",
        "temporary_state": "feature activations, class scores, class deltas, hidden/backward feature deltas, and gradient accumulators are capacitor voltages",
        "python_role": "Python generates guiding waveforms and compares final state; it does not carry training state during the transient run.",
        "python_checkpointing_between_samples": False,
        "note": "Local-feature phase-transient all-SPICE update smoke; small equivalence check against the existing operating-point SPICE update.",
        **metrics,
    }
    summary_path = results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
