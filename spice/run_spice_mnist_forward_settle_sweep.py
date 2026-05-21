from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from netlist_builder import Netlist, join_terms, param_ref, pwl, v
from run_spice_mnist_local_block_batch_op_train import block_indices
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)


def tanh_expr(expr: str) -> str:
    return f"(2/(1+exp(-2*({expr})))-1)"


def coef(value: float) -> str:
    return f"({float(value):.12g})"


def step_pwl(value: float, load_time: float, t_stop: float, edge: float) -> str:
    return pwl(
        [
            (0.0, 0.0),
            (max(0.0, load_time - edge), 0.0),
            (load_time, float(value)),
            (t_stop, float(value)),
        ]
    )


def read_wrdata_timeseries(path: Path, n_vec: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2 * n_vec:
        raise ValueError(f"{path} has {arr.shape[1]} columns, expected at least {2 * n_vec}")
    times = np.asarray(arr[:, 0], dtype=float)
    vals = np.asarray(arr[:, 1 : 2 * n_vec : 2], dtype=float)
    return times, vals


def make_forward_settle_netlist(
    x_batch: np.ndarray,
    weights: np.ndarray,
    local_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[list[int]],
    out_path: Path,
    load_time: float,
    t_stop: float,
    transient_step: float,
    edge: float,
    tau_act: float,
    tau_score: float,
    cstate: float,
    pwl_pixels: bool,
    weight_sources: bool,
) -> tuple[str, int]:
    batch = x_batch.shape[0]
    n_classes, n_blocks, block_len = weights.shape
    cstate_ref = param_ref("CSTATE")
    tau_act_ref = param_ref("TAU_ACT")
    tau_score_ref = param_ref("TAU_SCORE")
    deck = Netlist("Forward-settling transient local-block inference deck.")
    deck.comment("Pixels step into input registers; activation and score capacitors settle continuously.")
    deck.param("CSTATE", cstate)
    deck.param("TAU_ACT", tau_act)
    deck.param("TAU_SCORE", tau_score)
    deck.blank()
    deck.vsource("load", "load", "0", step_pwl(1.0, load_time, t_stop, edge))
    deck.blank()
    for s in range(batch):
        for i, val in enumerate(x_batch[s]):
            spec = step_pwl(float(val), load_time, t_stop, edge) if pwl_pixels else f"DC {float(val):.12g}"
            deck.vsource(f"pix{s}_{i}", f"pix{s}_{i}", "0", spec)
    deck.blank()
    if weight_sources:
        for k in range(n_classes):
            for b in range(n_blocks):
                for p in range(block_len):
                    deck.vsource(f"w{k}_{b}_{p}", f"w{k}_{b}_{p}", "0", f"DC {weights[k, b, p]:.12g}")
                deck.vsource(f"lb{k}_{b}", f"lb{k}_{b}", "0", f"DC {local_bias[k, b]:.12g}")
                deck.vsource(f"g{k}_{b}", f"g{k}_{b}", "0", f"DC {gains[k, b]:.12g}")
            deck.vsource(f"ob{k}", f"ob{k}", "0", f"DC {output_bias[k]:.12g}")
        deck.blank()
    for s in range(batch):
        for k in range(n_classes):
            for b, idxs in enumerate(blocks):
                if weight_sources:
                    terms = [f"{v(f'w{k}_{b}_{p}')}*{v(f'pix{s}_{idx}')}" for p, idx in enumerate(idxs)]
                    terms.append(v(f"lb{k}_{b}"))
                else:
                    terms = [f"{coef(weights[k, b, p])}*{v(f'pix{s}_{idx}')}" for p, idx in enumerate(idxs)]
                    terms.append(coef(local_bias[k, b]))
                act_calc = tanh_expr(join_terms(terms))
                deck.capacitor(f"act{s}_{k}_{b}", f"act{s}_{k}_{b}", "0", cstate_ref, ic=0)
                deck.bsource(
                    f"act{s}_{k}_{b}",
                    f"act{s}_{k}_{b}",
                    "0",
                    "I",
                    f"{v('load')}*{cstate_ref}/{tau_act_ref}*({v(f'act{s}_{k}_{b}')}-({act_calc}))",
                )
            if weight_sources:
                score_terms = [f"{v(f'g{k}_{b}')}*{v(f'act{s}_{k}_{b}')}" for b in range(n_blocks)]
                score_terms.append(v(f"ob{k}"))
            else:
                score_terms = [f"{coef(gains[k, b])}*{v(f'act{s}_{k}_{b}')}" for b in range(n_blocks)]
                score_terms.append(coef(output_bias[k]))
            score_calc = join_terms(score_terms)
            deck.capacitor(f"score{s}_{k}", f"score{s}_{k}", "0", cstate_ref, ic=0)
            deck.bsource(
                f"score{s}_{k}",
                f"score{s}_{k}",
                "0",
                "I",
                f"{v('load')}*{cstate_ref}/{tau_score_ref}*({v(f'score{s}_{k}')}-({score_calc}))",
            )
            deck.bsource(f"y{s}_{k}", f"y{s}_{k}", "0", "V", tanh_expr(v(f"score{s}_{k}")))
    vectors = [v(f"y{s}_{k}") for s in range(batch) for k in range(n_classes)]
    deck.blank()
    deck.options("method=gear", "maxord=2")
    deck.control(
        f"tran {transient_step:.12g} {t_stop:.12g} uic",
        f"wrdata {out_path} " + " ".join(vectors),
    )
    deck.end()
    return deck.render(), len(vectors)


def plot_accuracy(df: pd.DataFrame, out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(df["time_after_load_ns"], df["accuracy"], linewidth=2)
    plt.xlabel("time after input load (ns)")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=2000)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--load-time", type=float, default=0.2e-9)
    ap.add_argument("--t-stop", type=float, default=8.0e-9)
    ap.add_argument("--transient-step", type=float, default=20e-12)
    ap.add_argument("--edge", type=float, default=10e-12)
    ap.add_argument("--tau-act", type=float, default=0.5e-9)
    ap.add_argument("--tau-score", type=float, default=0.5e-9)
    ap.add_argument("--cstate", type=float, default=1e-12)
    ap.add_argument(
        "--pwl-pixels",
        action="store_true",
        help="Drive every pixel register with a PWL step. Default uses DC-held pixel registers and gates computation with Vload.",
    )
    ap.add_argument(
        "--weight-sources",
        action="store_true",
        help="Emit fixed checkpoint weights as voltage sources. Default inlines them as constants for faster forward-only diagnostics.",
    )
    ap.add_argument("--timepoints", type=int, default=201)
    ap.add_argument("--tag", default="forward_settle")
    args = ap.parse_args()

    if args.t_stop <= args.load_time:
        raise ValueError("--t-stop must be greater than --load-time")
    if args.batch_size <= 0 or args.timepoints < 2:
        raise ValueError("--batch-size must be positive and --timepoints must be at least 2")

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    _x_train, _y_train, x_test, y_test = load_mnist_sequence(
        args.train_samples, args.test_samples, args.image_size, args.seed
    )

    init = np.load(args.init_weights)
    weights = init["weights"]
    local_bias = init["local_bias"]
    gains = init["gains"]
    output_bias = init["output_bias"]
    expected = (10, len(blocks), args.block_size * args.block_size)
    if weights.shape != expected:
        raise ValueError(f"checkpoint weights have shape {weights.shape}, expected {expected}")

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    figures = ROOT / "results/figures"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    stem = f"spice_mnist_forward_settle_{sanitize_tag(args.tag)}"
    netlist_path = generated / f"{stem}.cir"
    data_path = results / f"{stem}.dat"
    grid = np.linspace(args.load_time, args.t_stop, args.timepoints)
    correct = np.zeros(args.timepoints, dtype=float)
    total = 0
    t0 = time.perf_counter()
    batch_rows = []
    for start in range(0, len(y_test), args.batch_size):
        x_batch = x_test[start : start + args.batch_size]
        y_batch = y_test[start : start + args.batch_size]
        netlist, n_vec = make_forward_settle_netlist(
            x_batch,
            weights,
            local_bias,
            gains,
            output_bias,
            blocks,
            data_path,
            args.load_time,
            args.t_stop,
            args.transient_step,
            args.edge,
            args.tau_act,
            args.tau_score,
            args.cstate,
            args.pwl_pixels,
            args.weight_sources,
        )
        netlist_path.write_text(prepare_netlist_for_simulator(netlist, spice_bin))
        b0 = time.perf_counter()
        proc = run_simulator_netlist(spice_bin, netlist_path, timeout=args.timeout)
        batch_wall = time.perf_counter() - b0
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        times, vals = read_wrdata_timeseries(data_path, n_vec)
        interp = np.empty((len(grid), n_vec), dtype=float)
        for col in range(n_vec):
            interp[:, col] = np.interp(grid, times, vals[:, col])
        scores = interp.reshape(len(grid), len(y_batch), 10)
        pred = np.argmax(scores, axis=2)
        correct += (pred == y_batch[None, :]).sum(axis=1)
        total += len(y_batch)
        batch_rows.append({"start": start, "count": len(y_batch), "wall_time_s": batch_wall})
        print(json.dumps(batch_rows[-1]), flush=True)

    acc = correct / max(total, 1)
    df = pd.DataFrame(
        {
            "time_s": grid,
            "time_after_load_ns": (grid - args.load_time) * 1e9,
            "accuracy": acc,
            "correct": correct.astype(int),
            "total": total,
        }
    )
    csv_path = results / f"{stem}.csv"
    table_path = tables / f"{stem}.csv"
    fig_path = figures / f"{stem}.png"
    df.to_csv(csv_path, index=False)
    df.to_csv(table_path, index=False)
    plot_accuracy(
        df,
        fig_path,
        f"Forward settling accuracy ({args.image_size}x{args.image_size}, n={args.test_samples})",
    )
    final_acc = float(acc[-1])
    within_1pct = df[df["accuracy"] >= final_acc - 0.01]
    max_acc = float(df["accuracy"].max())
    first_max = df[df["accuracy"] == max_acc].iloc[0]
    summary = {
        "simulator": version,
        "architecture": "local_block_forward_settling_transient",
        "init_weights": args.init_weights,
        "train_samples": args.train_samples,
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "classes": 10,
        "test_samples": args.test_samples,
        "batch_size": args.batch_size,
        "load_time_s": args.load_time,
        "t_stop_s": args.t_stop,
        "transient_step_s": args.transient_step,
        "tau_act_s": args.tau_act,
        "tau_score_s": args.tau_score,
        "pwl_pixels": args.pwl_pixels,
        "weight_sources": args.weight_sources,
        "initial_accuracy_at_load": float(acc[0]),
        "steady_accuracy_at_t_stop": final_acc,
        "max_accuracy": max_acc,
        "first_time_within_1pct_of_final_ns": (
            float(within_1pct["time_after_load_ns"].iloc[0]) if len(within_1pct) else None
        ),
        "first_time_of_max_accuracy_ns": float(first_max["time_after_load_ns"]),
        "accuracy_csv": str(csv_path),
        "table_csv": str(table_path),
        "figure": str(fig_path),
        "batch_wall_times": batch_rows,
        "wall_time_s": time.perf_counter() - t0,
        "note": "Transient forward-only settling diagnostic; weights are fixed from checkpoint, while activation and score capacitors settle after the load waveform enables computation from the image registers.",
    }
    summary_path = results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
