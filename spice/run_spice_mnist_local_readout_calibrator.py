from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import add_local_activation, block_indices, plot_curve
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


def load_local_checkpoint(path: Path, image_size: int, block_size: int, stride: int):
    ckpt = np.load(path, allow_pickle=True)
    weights = np.asarray(ckpt["weights"], dtype=float)
    local_bias = np.asarray(ckpt["local_bias"], dtype=float)
    gains = np.asarray(ckpt["gains"], dtype=float)
    output_bias = np.asarray(ckpt["output_bias"], dtype=float)
    blocks = block_indices(image_size, block_size, stride)
    expected = (10, len(blocks), block_size * block_size)
    if weights.shape != expected:
        raise ValueError(f"weights shape {weights.shape}, expected {expected}")
    if local_bias.shape != (10, len(blocks)) or gains.shape != (10, len(blocks)) or output_bias.shape != (10,):
        raise ValueError(
            f"inconsistent checkpoint shapes: local_bias={local_bias.shape}, gains={gains.shape}, output_bias={output_bias.shape}"
        )
    return weights, local_bias, gains, output_bias, blocks


def feature_expr(raw_expr: str, mode: str) -> str:
    if mode == "raw":
        return raw_expr
    if mode == "unit":
        return f"0.5*(({raw_expr})+1)"
    raise ValueError(f"unknown feature mode {mode!r}")


def make_train_netlist(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    weights: np.ndarray,
    local_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[np.ndarray],
    wmix: np.ndarray,
    bmix: np.ndarray,
    lr: float,
    out_path: Path,
    local_activation: str,
    relu_clip: float,
    feature_mode: str,
    readout_source: str,
) -> str:
    batch, n_in = x_batch.shape
    n_classes = 10
    n_blocks = len(blocks)
    n_features = wmix.shape[1]
    lines = [
        "* Frozen local block evidence plus trainable 10x10 class readout calibration.",
        "* ngspice computes local evidence, softmax class error, and mixer weight updates.",
        f".param LR={lr:.12g}",
        f".param BS={batch}",
        "",
    ]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
        target = np.zeros(n_classes)
        target[int(y_batch[s])] = 1.0
        for k in range(n_classes):
            lines.append(f"Vt{s}_{k} t{s}_{k} 0 DC {target[k]:.12g}")
    lines.append("")
    for k in range(n_classes):
        for b, idxs in enumerate(blocks):
            for p, _idx in enumerate(idxs):
                lines.append(f"Vw{k}_{b}_{p} w{k}_{b}_{p} 0 DC {weights[k, b, p]:.12g}")
            lines.append(f"Vlb{k}_{b} lb{k}_{b} 0 DC {local_bias[k, b]:.12g}")
            lines.append(f"Vg{k}_{b} g{k}_{b} 0 DC {gains[k, b]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {output_bias[k]:.12g}")
        for f in range(n_features):
            lines.append(f"Vwm{k}_{f} wm{k}_{f} 0 DC {wmix[k, f]:.12g}")
        lines.append(f"Vbm{k} bm{k} 0 DC {bmix[k]:.12g}")
    lines.append("")

    for s in range(batch):
        for c in range(n_classes):
            h_terms = []
            for b, idxs in enumerate(blocks):
                terms = [f"V(w{c}_{b}_{p})*V(x{s}_{idx})" for p, idx in enumerate(idxs)]
                terms.append(f"V(lb{c}_{b})")
                h_expr, _deriv = add_local_activation(lines, s, c, b, " + ".join(terms), local_activation, relu_clip)
                local_expr = f"V(g{c}_{b})*{h_expr}"
                h_terms.append(local_expr)
                if readout_source == "local":
                    f_idx = c * n_blocks + b
                    lines.append(f"Bfeat{s}_{f_idx} feat{s}_{f_idx} 0 V = {feature_expr(local_expr, feature_mode)}")
            raw = " + ".join(h_terms + [f"V(ob{c})"])
            lines.append(f"Bbase{s}_{c} base{s}_{c} 0 V = 2/(1+exp(-2*({raw})))-1")
            if readout_source == "class":
                lines.append(f"Bfeat{s}_{c} feat{s}_{c} 0 V = {feature_expr(f'V(base{s}_{c})', feature_mode)}")
        for k in range(n_classes):
            terms = [f"V(wm{k}_{f})*V(feat{s}_{f})" for f in range(n_features)] + [f"V(bm{k})"]
            lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {' + '.join(terms)}")
        denom = " + ".join(f"exp(V(z{s}_{kk}))" for kk in range(n_classes))
        for k in range(n_classes):
            lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
            lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
    lines.append("")
    for k in range(n_classes):
        for f in range(n_features):
            grad = " + ".join(f"V(d{s}_{k})*V(feat{s}_{f})" for s in range(batch))
            lines.append(f"Bnwm{k}_{f} nwm{k}_{f} 0 V = V(wm{k}_{f}) + {{LR}}*(({grad})/{{BS}})")
        grad_b = " + ".join(f"V(d{s}_{k})" for s in range(batch))
        lines.append(f"Bnbm{k} nbm{k} 0 V = V(bm{k}) + {{LR}}*(({grad_b})/{{BS}})")
    vectors = [f"V(nwm{k}_{f})" for k in range(n_classes) for f in range(n_features)]
    vectors += [f"V(nbm{k})" for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def make_eval_netlist(
    x_batch: np.ndarray,
    weights: np.ndarray,
    local_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[np.ndarray],
    wmix: np.ndarray,
    bmix: np.ndarray,
    out_path: Path,
    local_activation: str,
    relu_clip: float,
    feature_mode: str,
    calibrated: bool,
    readout_source: str,
) -> str:
    batch, n_in = x_batch.shape
    n_classes = 10
    n_blocks = len(blocks)
    n_features = wmix.shape[1]
    lines = ["* Frozen local block evidence with optional trainable class mixer eval.", ""]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
    lines.append("")
    for k in range(n_classes):
        for b, idxs in enumerate(blocks):
            for p, _idx in enumerate(idxs):
                lines.append(f"Vw{k}_{b}_{p} w{k}_{b}_{p} 0 DC {weights[k, b, p]:.12g}")
            lines.append(f"Vlb{k}_{b} lb{k}_{b} 0 DC {local_bias[k, b]:.12g}")
            lines.append(f"Vg{k}_{b} g{k}_{b} 0 DC {gains[k, b]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {output_bias[k]:.12g}")
        if calibrated:
            for f in range(n_features):
                lines.append(f"Vwm{k}_{f} wm{k}_{f} 0 DC {wmix[k, f]:.12g}")
            lines.append(f"Vbm{k} bm{k} 0 DC {bmix[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for c in range(n_classes):
            h_terms = []
            for b, idxs in enumerate(blocks):
                terms = [f"V(w{c}_{b}_{p})*V(x{s}_{idx})" for p, idx in enumerate(idxs)]
                terms.append(f"V(lb{c}_{b})")
                h_expr, _deriv = add_local_activation(lines, s, c, b, " + ".join(terms), local_activation, relu_clip)
                local_expr = f"V(g{c}_{b})*{h_expr}"
                h_terms.append(local_expr)
                if readout_source == "local":
                    f_idx = c * n_blocks + b
                    lines.append(f"Bfeat{s}_{f_idx} feat{s}_{f_idx} 0 V = {feature_expr(local_expr, feature_mode)}")
            raw = " + ".join(h_terms + [f"V(ob{c})"])
            lines.append(f"Bbase{s}_{c} base{s}_{c} 0 V = 2/(1+exp(-2*({raw})))-1")
            if readout_source == "class":
                lines.append(f"Bfeat{s}_{c} feat{s}_{c} 0 V = {feature_expr(f'V(base{s}_{c})', feature_mode)}")
        for k in range(n_classes):
            if calibrated:
                terms = [f"V(wm{k}_{f})*V(feat{s}_{f})" for f in range(n_features)] + [f"V(bm{k})"]
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = {' + '.join(terms)}")
            else:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = V(base{s}_{k})")
    vectors = [f"V(y{s}_{k})" for s in range(batch) for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def run_eval(
    spice_bin,
    netlist_path: Path,
    data_path: Path,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    weights: np.ndarray,
    local_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[np.ndarray],
    wmix: np.ndarray,
    bmix: np.ndarray,
    batch_size: int,
    timeout: float,
    local_activation: str,
    relu_clip: float,
    feature_mode: str,
    calibrated: bool,
    readout_source: str,
) -> float:
    correct = 0
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        netlist_path.write_text(
            make_eval_netlist(
                x,
                weights,
                local_bias,
                gains,
                output_bias,
                blocks,
                wmix,
                bmix,
                data_path,
                local_activation,
                relu_clip,
                feature_mode,
                calibrated,
                readout_source,
            )
        )
        proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        vals = read_wrdata_row(data_path, len(y) * 10).reshape(len(y), 10)
        correct += int((np.argmax(vals, axis=1) == y).sum())
    return correct / max(len(y_eval), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-samples", type=int, default=2000)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--identity-scale", type=float, default=1.0)
    ap.add_argument("--feature-mode", choices=["raw", "unit"], default="unit")
    ap.add_argument("--readout-source", choices=["class", "local"], default="class")
    ap.add_argument("--local-activation", choices=["tanh", "relu", "clipped-relu", "diff-clipped-relu"], default="tanh")
    ap.add_argument("--relu-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--tag", default="local_readout_cal")
    args = ap.parse_args()

    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    weights, local_bias, gains, output_bias, blocks = load_local_checkpoint(
        Path(args.checkpoint), args.image_size, args.block_size, args.stride
    )
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_local_readout_cal_{safe_tag}"
    netlist_path = generated / f"{stem}_step.cir"
    eval_netlist = generated / f"{stem}_eval.cir"
    data_path = ROOT / f"spice/results/{stem}_step.dat"

    n_features = 10 if args.readout_source == "class" else 10 * len(blocks)
    wmix = np.zeros((10, n_features))
    if args.readout_source == "class":
        wmix[:, :10] = np.eye(10) * args.identity_scale
        bmix = np.zeros(10)
    else:
        for k in range(10):
            for b in range(len(blocks)):
                wmix[k, k * len(blocks) + b] = args.identity_scale
        bmix = output_bias.copy()
    base_acc = run_eval(
        spice_bin,
        eval_netlist,
        data_path,
        x_test,
        y_test,
        weights,
        local_bias,
        gains,
        output_bias,
        blocks,
        wmix,
        bmix,
        args.batch_size,
        args.timeout,
        args.local_activation,
        args.relu_clip,
        args.feature_mode,
        calibrated=False,
        readout_source=args.readout_source,
    )

    rows = [{"epoch": 0, "heldout_accuracy": base_acc, "epoch_wall_time_s": 0.0}]
    best_acc = base_acc
    best_state = (wmix.copy(), bmix.copy())
    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = np.arange(len(y_train))
        rng.shuffle(order)
        epoch_start = time.perf_counter()
        for n, start in enumerate(range(0, len(order), args.batch_size)):
            idx = order[start : start + args.batch_size]
            netlist_path.write_text(
                make_train_netlist(
                    x_train[idx],
                    y_train[idx],
                    weights,
                    local_bias,
                    gains,
                    output_bias,
                    blocks,
                    wmix,
                    bmix,
                    args.lr,
                    data_path,
                    args.local_activation,
                    args.relu_clip,
                    args.feature_mode,
                    args.readout_source,
                )
            )
            proc = subprocess.run([spice_bin, "-b", str(netlist_path)], text=True, capture_output=True, timeout=args.timeout)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
            n_vals = 10 * n_features + 10
            vals = read_wrdata_row(data_path, n_vals)
            wmix = vals[: 10 * n_features].reshape(10, n_features)
            bmix = vals[10 * n_features : 10 * n_features + 10]
            if (n + 1) % 5 == 0:
                print(f"epoch {epoch + 1} batch {n + 1}", flush=True)
        heldout = run_eval(
            spice_bin,
            eval_netlist,
            data_path,
            x_test,
            y_test,
            weights,
            local_bias,
            gains,
            output_bias,
            blocks,
            wmix,
            bmix,
            args.batch_size,
            args.timeout,
            args.local_activation,
            args.relu_clip,
            args.feature_mode,
            calibrated=True,
            readout_source=args.readout_source,
        )
        row = {"epoch": epoch + 1, "heldout_accuracy": heldout, "epoch_wall_time_s": time.perf_counter() - epoch_start}
        rows.append(row)
        if heldout > best_acc:
            best_acc = heldout
            best_state = (wmix.copy(), bmix.copy())
        print(json.dumps(row), flush=True)

    curve = pd.DataFrame(rows)
    curve_path = ROOT / f"spice/results/{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    fig = ROOT / f"spice/results/{stem}_learning_curve.png"
    plot_curve(curve, fig)
    final_weights = ROOT / f"spice/results/{stem}_final_weights.npz"
    best_weights = ROOT / f"spice/results/{stem}_best_weights.npz"
    np.savez_compressed(final_weights, wmix=wmix, bmix=bmix)
    np.savez_compressed(best_weights, wmix=best_state[0], bmix=best_state[1])
    summary = {
        "simulator": version,
        "dataset": "MNIST train/test split, downsampled",
        "architecture": "frozen_local_block_trainable_10x10_readout",
        "checkpoint": args.checkpoint,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": args.stride,
        "blocks": len(blocks),
        "feature_mode": args.feature_mode,
        "readout_source": args.readout_source,
        "readout_features": int(n_features),
        "local_activation": args.local_activation,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "identity_scale": args.identity_scale,
        "baseline_heldout_accuracy": float(base_acc),
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]),
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()),
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(final_weights),
        "best_weights": str(best_weights),
        "note": (
            "Frozen all-SPICE local block evidence with a trainable 10x10 class mixer. "
            "ngspice computes local evidence, softmax readout error, and programmable mixer updates."
        ),
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
