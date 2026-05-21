from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import (
    add_local_activation,
    add_local_activation_deriv,
    block_indices,
    class_ranges,
    plot_curve,
)
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def make_train_netlist(
    x_batch,
    y_batch,
    weights,
    local_bias,
    gains,
    output_bias,
    blocks,
    lr,
    out_path,
    linear_output,
    softmax_output,
    train_gains,
    local_activation,
    relu_clip,
    class_labels=None,
):
    batch = x_batch.shape[0]
    n_classes, n_blocks, channels, block_len = weights.shape
    if class_labels is None:
        class_labels = np.arange(n_classes)
    class_labels = np.asarray(class_labels)
    if len(class_labels) != n_classes:
        raise ValueError("class_labels length must match weights.shape[0]")
    lines = [
        "* Multichannel local block-evidence batch operating-point SPICE training.",
        "* Each class/block owns multiple analog local evidence cells; ngspice computes forward, backward, and updates.",
        f".param LR={lr:.12g}",
        f".param BS={batch}",
        "",
    ]
    for s in range(batch):
        for i, val in enumerate(x_batch[s]):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(val):.12g}")
        if softmax_output:
            target = np.zeros(n_classes)
        else:
            target = -np.ones(n_classes)
        match = np.flatnonzero(class_labels == int(y_batch[s]))
        if len(match):
            target[int(match[0])] = 1.0
        for k in range(n_classes):
            lines.append(f"Vt{s}_{k} t{s}_{k} 0 DC {target[k]:.12g}")
    lines.append("")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                for p in range(block_len):
                    lines.append(f"Vw{k}_{b}_{c}_{p} w{k}_{b}_{c}_{p} 0 DC {weights[k, b, c, p]:.12g}")
                lines.append(f"Vlb{k}_{b}_{c} lb{k}_{b}_{c} 0 DC {local_bias[k, b, c]:.12g}")
                lines.append(f"Vg{k}_{b}_{c} g{k}_{b}_{c} 0 DC {gains[k, b, c]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {output_bias[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for k in range(n_classes):
            h_names = []
            for b, idxs in enumerate(blocks):
                for c in range(channels):
                    terms = [f"V(w{k}_{b}_{c}_{p})*V(x{s}_{idx})" for p, idx in enumerate(idxs)]
                    terms.append(f"V(lb{k}_{b}_{c})")
                    h_expr, _deriv = add_local_activation(lines, s, k, b * channels + c, " + ".join(terms), local_activation, relu_clip)
                    h_names.append(f"V(g{k}_{b}_{c})*{h_expr}")
            out_sum = " + ".join(h_names + [f"V(ob{k})"])
            if softmax_output:
                lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {out_sum}")
            elif linear_output:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = {out_sum}")
            else:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({out_sum})))-1")
        if softmax_output:
            denom = " + ".join(f"exp(V(z{s}_{j}))" for j in range(n_classes))
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
                lines.append(f"Be{s}_{k} e{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
                lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})")
        else:
            for k in range(n_classes):
                lines.append(f"Be{s}_{k} e{s}_{k} 0 V = V(t{s}_{k})-V(y{s}_{k})")
                if linear_output:
                    lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})")
                else:
                    lines.append(f"Bd{s}_{k} d{s}_{k} 0 V = V(e{s}_{k})*(1-V(y{s}_{k})*V(y{s}_{k}))")
        for k in range(n_classes):
            for b in range(n_blocks):
                for c in range(channels):
                    lines.append(
                        f"Bdh{s}_{k}_{b}_{c} dh{s}_{k}_{b}_{c} 0 V = "
                        f"V(d{s}_{k})*V(g{k}_{b}_{c})*"
                        f"{add_local_activation_deriv(local_activation, relu_clip, s, k, b * channels + c)}"
                    )
    lines.append("")
    for k in range(n_classes):
        for b, idxs in enumerate(blocks):
            for c in range(channels):
                for p, idx in enumerate(idxs):
                    grad = " + ".join(f"V(dh{s}_{k}_{b}_{c})*V(x{s}_{idx})" for s in range(batch))
                    lines.append(
                        f"Bnw{k}_{b}_{c}_{p} nw{k}_{b}_{c}_{p} 0 V = "
                        f"V(w{k}_{b}_{c}_{p}) + {{LR}}*(({grad})/{{BS}})"
                    )
                grad_b = " + ".join(f"V(dh{s}_{k}_{b}_{c})" for s in range(batch))
                lines.append(f"Bnlb{k}_{b}_{c} nlb{k}_{b}_{c} 0 V = V(lb{k}_{b}_{c}) + {{LR}}*(({grad_b})/{{BS}})")
                if train_gains:
                    grad_g = " + ".join(f"V(d{s}_{k})*V(h{s}_{k}_{b * channels + c})" for s in range(batch))
                    lines.append(f"Bng{k}_{b}_{c} ng{k}_{b}_{c} 0 V = V(g{k}_{b}_{c}) + {{LR}}*(({grad_g})/{{BS}})")
        grad_o = " + ".join(f"V(d{s}_{k})" for s in range(batch))
        lines.append(f"Bnob{k} nob{k} 0 V = V(ob{k}) + {{LR}}*(({grad_o})/{{BS}})")
    vectors = [
        f"V(nw{k}_{b}_{c}_{p})"
        for k in range(n_classes)
        for b in range(n_blocks)
        for c in range(channels)
        for p in range(block_len)
    ]
    vectors += [f"V(nlb{k}_{b}_{c})" for k in range(n_classes) for b in range(n_blocks) for c in range(channels)]
    if train_gains:
        vectors += [f"V(ng{k}_{b}_{c})" for k in range(n_classes) for b in range(n_blocks) for c in range(channels)]
    vectors += [f"V(nob{k})" for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def make_eval_netlist(
    x_batch,
    weights,
    local_bias,
    gains,
    output_bias,
    blocks,
    out_path,
    linear_output,
    softmax_output,
    local_activation,
    relu_clip,
):
    batch = x_batch.shape[0]
    n_classes, n_blocks, channels, block_len = weights.shape
    lines = ["* Multichannel local block-evidence batch operating-point SPICE inference.", ""]
    for s in range(batch):
        for i, val in enumerate(x_batch[s]):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(val):.12g}")
    lines.append("")
    for k in range(n_classes):
        for b in range(n_blocks):
            for c in range(channels):
                for p in range(block_len):
                    lines.append(f"Vw{k}_{b}_{c}_{p} w{k}_{b}_{c}_{p} 0 DC {weights[k, b, c, p]:.12g}")
                lines.append(f"Vlb{k}_{b}_{c} lb{k}_{b}_{c} 0 DC {local_bias[k, b, c]:.12g}")
                lines.append(f"Vg{k}_{b}_{c} g{k}_{b}_{c} 0 DC {gains[k, b, c]:.12g}")
        lines.append(f"Vob{k} ob{k} 0 DC {output_bias[k]:.12g}")
    lines.append("")
    for s in range(batch):
        for k in range(n_classes):
            h_names = []
            for b, idxs in enumerate(blocks):
                for c in range(channels):
                    terms = [f"V(w{k}_{b}_{c}_{p})*V(x{s}_{idx})" for p, idx in enumerate(idxs)]
                    terms.append(f"V(lb{k}_{b}_{c})")
                    h_expr, _deriv = add_local_activation(lines, s, k, b * channels + c, " + ".join(terms), local_activation, relu_clip)
                    h_names.append(f"V(g{k}_{b}_{c})*{h_expr}")
            out_sum = " + ".join(h_names + [f"V(ob{k})"])
            if softmax_output:
                lines.append(f"Bz{s}_{k} z{s}_{k} 0 V = {out_sum}")
            elif linear_output:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = {out_sum}")
            else:
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = 2/(1+exp(-2*({out_sum})))-1")
        if softmax_output:
            denom = " + ".join(f"exp(V(z{s}_{j}))" for j in range(n_classes))
            for k in range(n_classes):
                lines.append(f"By{s}_{k} y{s}_{k} 0 V = exp(V(z{s}_{k}))/({denom})")
    vectors = [f"V(y{s}_{k})" for s in range(batch) for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def run_train_batch(
    spice_bin,
    netlist_path,
    data_path,
    x,
    y,
    weights,
    local_bias,
    gains,
    output_bias,
    blocks,
    lr,
    timeout,
    linear_output,
    softmax_output,
    train_gains,
    local_activation,
    relu_clip,
    class_labels=None,
):
    netlist_path.write_text(
        prepare_netlist_for_simulator(
            make_train_netlist(
                x,
                y,
                weights,
                local_bias,
                gains,
                output_bias,
                blocks,
                lr,
                data_path,
                linear_output,
                softmax_output,
                train_gains,
                local_activation,
                relu_clip,
                class_labels=class_labels,
            ),
            spice_bin,
        )
    )
    proc = run_simulator_netlist(spice_bin, netlist_path, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    n_classes, n_blocks, channels, block_len = weights.shape
    n = (
        n_classes * n_blocks * channels * block_len
        + n_classes * n_blocks * channels
        + (n_classes * n_blocks * channels if train_gains else 0)
        + n_classes
    )
    vals = read_wrdata_row(data_path, n)
    offset = 0
    nw = vals[offset : offset + n_classes * n_blocks * channels * block_len].reshape(weights.shape)
    offset += n_classes * n_blocks * channels * block_len
    nlb = vals[offset : offset + n_classes * n_blocks * channels].reshape(local_bias.shape)
    offset += n_classes * n_blocks * channels
    if train_gains:
        ng = vals[offset : offset + n_classes * n_blocks * channels].reshape(gains.shape)
        offset += n_classes * n_blocks * channels
    else:
        ng = gains
    nob = vals[offset : offset + n_classes]
    return nw, nlb, ng, nob


def run_eval(
    spice_bin,
    netlist_path,
    data_path,
    x_eval,
    y_eval,
    weights,
    local_bias,
    gains,
    output_bias,
    blocks,
    batch_size,
    timeout,
    linear_output,
    softmax_output,
    local_activation,
    relu_clip,
    class_chunk_size=0,
):
    correct = 0
    ranges = class_ranges(weights.shape[0], class_chunk_size)
    if len(ranges) > 1 and softmax_output:
        raise ValueError("class chunking is only valid for independent tanh/linear class outputs, not softmax")
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        vals_by_chunk = []
        for cs, ce in ranges:
            netlist_path.write_text(
                prepare_netlist_for_simulator(
                    make_eval_netlist(
                        x,
                        weights[cs:ce],
                        local_bias[cs:ce],
                        gains[cs:ce],
                        output_bias[cs:ce],
                        blocks,
                        data_path,
                        linear_output,
                        softmax_output,
                        local_activation,
                        relu_clip,
                    ),
                    spice_bin,
                )
            )
            proc = run_simulator_netlist(spice_bin, netlist_path, timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
            vals_by_chunk.append(read_wrdata_row(data_path, len(y) * (ce - cs)).reshape(len(y), ce - cs))
        vals = vals_by_chunk[0] if len(vals_by_chunk) == 1 else np.concatenate(vals_by_chunk, axis=1)
        correct += int((np.argmax(vals, axis=1) == y).sum())
    return correct / max(len(y_eval), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=200)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--gain-scale", type=float, default=None)
    ap.add_argument("--train-gains", action="store_true")
    ap.add_argument("--linear-output", action="store_true")
    ap.add_argument("--softmax-output", action="store_true")
    ap.add_argument(
        "--local-activation",
        choices=["tanh", "relu", "clipped-relu", "diff-clipped-relu", "differential-clipped-relu"],
        default="tanh",
    )
    ap.add_argument("--relu-clip", type=float, default=1.0)
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--class-chunk-size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--tag", default="local_block_multichannel")
    args = ap.parse_args()
    if args.linear_output and args.softmax_output:
        raise ValueError("--linear-output and --softmax-output are mutually exclusive")
    if args.class_chunk_size > 0 and args.softmax_output:
        raise ValueError("--class-chunk-size is only valid for independent tanh/linear outputs, not softmax")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    x_train, y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_local_block_mc_{safe_tag}"
    netlist_path = generated / f"{stem}_step.cir"
    eval_netlist = generated / f"{stem}_eval.cir"
    data_path = ROOT / f"spice/results/{stem}_step.dat"
    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0.0, 0.02, size=(10, len(blocks), args.channels, args.block_size * args.block_size))
    local_bias = np.zeros((10, len(blocks), args.channels))
    gain_scale = (1.0 / max(args.channels, 1)) if args.gain_scale is None else args.gain_scale
    gains = np.full((10, len(blocks), args.channels), gain_scale)
    output_bias = np.zeros(10)
    if args.init_weights:
        init = np.load(args.init_weights)
        init_weights = init["weights"]
        init_local_bias = init["local_bias"]
        init_gains = init["gains"]
        init_output_bias = init["output_bias"]
        expected_shapes = (
            (10, len(blocks), args.channels, args.block_size * args.block_size),
            (10, len(blocks), args.channels),
            (10, len(blocks), args.channels),
            (10,),
        )
        actual_shapes = (init_weights.shape, init_local_bias.shape, init_gains.shape, init_output_bias.shape)
        if actual_shapes == expected_shapes:
            weights = init_weights
            local_bias = init_local_bias
            gains = init_gains
            output_bias = init_output_bias
        elif (
            init_weights.ndim == 4
            and init_weights.shape[0] == 10
            and init_weights.shape[1] == len(blocks)
            and init_weights.shape[2] < args.channels
            and init_weights.shape[3] == args.block_size * args.block_size
            and init_local_bias.shape == (10, len(blocks), init_weights.shape[2])
            and init_gains.shape == (10, len(blocks), init_weights.shape[2])
            and init_output_bias.shape == (10,)
        ):
            old_channels = init_weights.shape[2]
            weights[:, :, :old_channels, :] = init_weights
            weights[:, :, old_channels:, :] = 0.0
            local_bias[:, :, :old_channels] = init_local_bias
            local_bias[:, :, old_channels:] = 0.0
            gains[:, :, :old_channels] = init_gains
            output_bias = init_output_bias
        elif (
            init_weights.shape == (10, len(blocks), args.block_size * args.block_size)
            and init_local_bias.shape == (10, len(blocks))
            and init_gains.shape == (10, len(blocks))
            and init_output_bias.shape == (10,)
        ):
            weights[:, :, 0, :] = init_weights
            weights[:, :, 1:, :] = 0.0
            local_bias[:, :, 0] = init_local_bias
            local_bias[:, :, 1:] = 0.0
            gains[:, :, 0] = init_gains
            output_bias = init_output_bias
        else:
            raise ValueError(f"initial weight shapes {actual_shapes} do not match expected {expected_shapes}")
    rows = []
    weights_path = ROOT / f"spice/results/{stem}_final_weights.npz"
    best_weights_path = ROOT / f"spice/results/{stem}_best_weights.npz"
    ranges = class_ranges(weights.shape[0], args.class_chunk_size)

    def save_weights(path: Path) -> None:
        np.savez_compressed(path, weights=weights, local_bias=local_bias, gains=gains, output_bias=output_bias)

    best_acc = -1.0
    t0 = time.perf_counter()
    if args.eval_only or args.epochs == 0:
        epoch_start = time.perf_counter()
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
            args.batch_size,
            args.timeout,
            args.linear_output,
            args.softmax_output,
            args.local_activation,
            args.relu_clip,
            args.class_chunk_size,
        )
        row = {"epoch": 0, "heldout_accuracy": heldout, "epoch_wall_time_s": time.perf_counter() - epoch_start}
        rows.append(row)
        best_acc = heldout
        save_weights(best_weights_path)
        print(json.dumps(row), flush=True)
    for epoch in range(0 if args.eval_only else args.epochs):
        order = np.arange(len(y_train))
        rng.shuffle(order)
        epoch_start = time.perf_counter()
        for n, start in enumerate(range(0, len(order), args.batch_size)):
            idx = order[start : start + args.batch_size]
            for cs, ce in ranges:
                nw, nlb, ng, nob = run_train_batch(
                    spice_bin,
                    netlist_path,
                    data_path,
                    x_train[idx],
                    y_train[idx],
                    weights[cs:ce],
                    local_bias[cs:ce],
                    gains[cs:ce],
                    output_bias[cs:ce],
                    blocks,
                    args.lr,
                    args.timeout,
                    args.linear_output,
                    args.softmax_output,
                    args.train_gains,
                    args.local_activation,
                    args.relu_clip,
                    class_labels=np.arange(cs, ce),
                )
                weights[cs:ce] = nw
                local_bias[cs:ce] = nlb
                gains[cs:ce] = ng
                output_bias[cs:ce] = nob
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
            args.batch_size,
            args.timeout,
            args.linear_output,
            args.softmax_output,
            args.local_activation,
            args.relu_clip,
            args.class_chunk_size,
        )
        row = {"epoch": epoch + 1, "heldout_accuracy": heldout, "epoch_wall_time_s": time.perf_counter() - epoch_start}
        rows.append(row)
        if heldout > best_acc:
            best_acc = heldout
            save_weights(best_weights_path)
        print(json.dumps(row), flush=True)
    curve = pd.DataFrame(rows)
    curve_path = ROOT / f"spice/results/{stem}_learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    fig = ROOT / f"spice/results/{stem}_learning_curve.png"
    plot_curve(curve, fig)
    save_weights(weights_path)
    summary = {
        "simulator": version,
        "dataset": "MNIST train/test split, downsampled",
        "architecture": "local_block_multichannel_class_evidence",
        "activation": "analog_multichannel_voltage_state",
        "local_activation": args.local_activation,
        "relu_clip": args.relu_clip,
        "output_mode": "softmax_class_evidence" if args.softmax_output else ("linear_class_evidence" if args.linear_output else "tanh_class_evidence"),
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "channels_per_class_block": args.channels,
        "gain_scale": gain_scale,
        "train_gains": bool(args.train_gains),
        "class_chunk_size": args.class_chunk_size,
        "local": True,
        "inputs": int(x_train.shape[1]),
        "classes": 10,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "eval_only": bool(args.eval_only),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "init_weights": args.init_weights,
        "linear_output": bool(args.linear_output),
        "softmax_output": bool(args.softmax_output),
        "wall_time_s": time.perf_counter() - t0,
        "learning_curve": str(curve_path),
        "figure": str(fig),
        "final_weights": str(weights_path),
        "best_weights": str(best_weights_path),
        "heldout_test_accuracy": float(curve.iloc[-1]["heldout_accuracy"]),
        "best_heldout_accuracy": float(curve["heldout_accuracy"].max()),
        "note": "Multichannel local block all-SPICE trainer; ngspice computes local evidence and programmable-state updates.",
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
