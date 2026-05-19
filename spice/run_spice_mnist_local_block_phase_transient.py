from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_spice_mnist_batch_op_train import read_wrdata_row
from run_spice_mnist_local_block_batch_op_train import block_indices, run_train_batch
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


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
    for t, v in sorted(points):
        if cleaned and abs(cleaned[-1][0] - t) < 1e-18:
            cleaned[-1] = (t, v)
        else:
            cleaned.append((t, v))
    return "PWL(" + " ".join(f"{t:.12g} {v:.12g}" for t, v in cleaned) + ")"


def sample_source_pwl(values: np.ndarray, sample_starts: list[float], t_stop: float, edge: float) -> str:
    points: list[tuple[float, float]] = [(0.0, float(values[0]))]
    for s, val in enumerate(values):
        t = sample_starts[s]
        points.append((max(0.0, t - edge), float(values[s - 1] if s > 0 else val)))
        points.append((t, float(val)))
    points.append((t_stop, float(values[-1])))
    cleaned: list[tuple[float, float]] = []
    for t, v in points:
        if cleaned and abs(cleaned[-1][0] - t) < 1e-18:
            cleaned[-1] = (t, v)
        else:
            cleaned.append((t, v))
    return "PWL(" + " ".join(f"{t:.12g} {v:.12g}" for t, v in cleaned) + ")"


def tanh_expr(expr: str) -> str:
    return f"(2/(1+exp(-2*({expr})))-1)"


def target_matrix(labels: np.ndarray, n_classes: int) -> np.ndarray:
    targets = -np.ones((len(labels), n_classes))
    for s, label in enumerate(labels):
        targets[s, int(label)] = 1.0
    return targets


def make_phase_schedule(
    update_batch_size: int,
    updates: int,
    phase: float,
    gap: float,
) -> tuple[dict[str, list[tuple[float, float]]], list[float], float]:
    phases = {"act": [], "score": [], "err": [], "bwd": [], "acc": [], "apply": [], "clear": []}
    sample_starts: list[float] = []
    t = phase
    for _u in range(updates):
        for _s in range(update_batch_size):
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
    weights: np.ndarray,
    local_bias: np.ndarray,
    gains: np.ndarray,
    output_bias: np.ndarray,
    blocks: list[list[int]],
    lr: float,
    out_path: Path,
    train_gains: bool,
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
) -> tuple[str, int, float]:
    total_samples = x_batch.shape[0]
    if total_samples != update_batch_size * updates:
        raise ValueError("x_batch length must equal update_batch_size * updates")
    n_classes, n_blocks, block_len = weights.shape
    phases, sample_starts, t_stop = make_phase_schedule(update_batch_size, updates, phase, gap)
    tau = phase / settle_ratio
    targets = target_matrix(y_batch, n_classes)
    lines = [
        "* Phase-resolved transient local-block training deck.",
        "* Persistent weights/biases/gains are capacitor voltages; gradients are accumulated in capacitor nodes.",
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
    for k in range(n_classes):
        for b in range(n_blocks):
            for p in range(block_len):
                lines.append(f"Cw{k}_{b}_{p} w{k}_{b}_{p} 0 {{CW}} IC={weights[k, b, p]:.12g}")
                lines.append(f"Rw{k}_{b}_{p} w{k}_{b}_{p} 0 {{RLEAK}}")
                lines.append(f"Cgw{k}_{b}_{p} gw{k}_{b}_{p} 0 {{CGRAD}} IC=0")
            lines.append(f"Clb{k}_{b} lb{k}_{b} 0 {{CW}} IC={local_bias[k, b]:.12g}")
            lines.append(f"Rlb{k}_{b} lb{k}_{b} 0 {{RLEAK}}")
            lines.append(f"Cglb{k}_{b} glb{k}_{b} 0 {{CGRAD}} IC=0")
            lines.append(f"Cg{k}_{b} g{k}_{b} 0 {{CW}} IC={gains[k, b]:.12g}")
            lines.append(f"Rg{k}_{b} g{k}_{b} 0 {{RLEAK}}")
            lines.append(f"Cgg{k}_{b} gg{k}_{b} 0 {{CGRAD}} IC=0")
            lines.append(f"Cact{k}_{b} act{k}_{b} 0 {{CSTATE}} IC=0")
            lines.append(f"Cdh{k}_{b} dh{k}_{b} 0 {{CSTATE}} IC=0")
        lines.append(f"Cob{k} ob{k} 0 {{CW}} IC={output_bias[k]:.12g}")
        lines.append(f"Rob{k} ob{k} 0 {{RLEAK}}")
        lines.append(f"Cgob{k} gob{k} 0 {{CGRAD}} IC=0")
        lines.append(f"Cscore{k} score{k} 0 {{CSTATE}} IC=0")
        lines.append(f"Cd{k} d{k} 0 {{CSTATE}} IC=0")
    lines.append("")
    for k in range(n_classes):
        for b, idxs in enumerate(blocks):
            terms = [f"V(w{k}_{b}_{p})*V(pix{idx})" for p, idx in enumerate(idxs)]
            terms.append(f"V(lb{k}_{b})")
            summed = " + ".join(terms)
            act_calc = tanh_expr(summed)
            lines.append(f"Bstore_act{k}_{b} act{k}_{b} 0 I = V(pact)*{{CSTATE}}/{{TAU}}*(V(act{k}_{b})-({act_calc}))")
        score_terms = [f"V(g{k}_{b})*V(act{k}_{b})" for b in range(n_blocks)]
        score_terms.append(f"V(ob{k})")
        score_calc = " + ".join(score_terms)
        lines.append(f"Bscore{k} scorecalc{k} 0 V = {score_calc}")
        lines.append(f"Bstore_score{k} score{k} 0 I = V(pscore)*{{CSTATE}}/{{TAU}}*(V(score{k})-V(scorecalc{k}))")
        y_expr = tanh_expr(f"V(score{k})")
        lines.append(f"By{k} y{k} 0 V = {y_expr}")
        delta_expr = f"(V(target{k})-({y_expr}))*(1-({y_expr})*({y_expr}))"
        lines.append(f"Bstore_d{k} d{k} 0 I = V(perr)*{{CSTATE}}/{{TAU}}*(V(d{k})-({delta_expr}))")
    lines.append("")
    for k in range(n_classes):
        for b, idxs in enumerate(blocks):
            dact = f"(1-V(act{k}_{b})*V(act{k}_{b}))"
            local_delta = f"V(d{k})*V(g{k}_{b})*{dact}"
            lines.append(f"Bstore_dh{k}_{b} dh{k}_{b} 0 I = V(pbwd)*{{CSTATE}}/{{TAU}}*(V(dh{k}_{b})-({local_delta}))")
            for p, idx in enumerate(idxs):
                grad = f"V(dh{k}_{b})*V(pix{idx})"
                lines.append(f"Bacc_w{k}_{b}_{p} gw{k}_{b}_{p} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*({grad})")
                lines.append(f"Bupd_w{k}_{b}_{p} w{k}_{b}_{p} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gw{k}_{b}_{p})")
                lines.append(f"Bclear_gw{k}_{b}_{p} gw{k}_{b}_{p} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gw{k}_{b}_{p})")
            lines.append(f"Bacc_lb{k}_{b} glb{k}_{b} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*V(dh{k}_{b})")
            lines.append(f"Bupd_lb{k}_{b} lb{k}_{b} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(glb{k}_{b})")
            lines.append(f"Bclear_glb{k}_{b} glb{k}_{b} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(glb{k}_{b})")
            gain_grad = f"V(d{k})*V(act{k}_{b})"
            lines.append(f"Bacc_g{k}_{b} gg{k}_{b} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*({gain_grad})")
            if train_gains:
                lines.append(f"Bupd_g{k}_{b} g{k}_{b} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gg{k}_{b})")
            lines.append(f"Bclear_gg{k}_{b} gg{k}_{b} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gg{k}_{b})")
        lines.append(f"Bacc_ob{k} gob{k} 0 I = -V(pacc)*{{CGRAD}}/{{TPHASE}}*V(d{k})")
        lines.append(f"Bupd_ob{k} ob{k} 0 I = -V(papply)*{{CW}}*{{LR}}/({{BS}}*{{TPHASE}})*V(gob{k})")
        lines.append(f"Bclear_gob{k} gob{k} 0 I = V(pclear)*{{CGRAD}}/{{TAU}}*V(gob{k})")
    vectors = [f"V(w{k}_{b}_{p})" for k in range(n_classes) for b in range(n_blocks) for p in range(block_len)]
    vectors += [f"V(lb{k}_{b})" for k in range(n_classes) for b in range(n_blocks)]
    vectors += [f"V(g{k}_{b})" for k in range(n_classes) for b in range(n_blocks)]
    vectors += [f"V(ob{k})" for k in range(n_classes)]
    vectors += [f"V(y{k})" for k in range(n_classes)]
    lines += [
        "",
        ".options method=gear maxord=2",
        ".control",
        f"tran {transient_step:.12g} {t_stop:.12g} uic",
        f"wrdata {out_path} " + " ".join(vectors),
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), len(vectors), t_stop


def unpack_state(vals: np.ndarray, weights: np.ndarray, local_bias: np.ndarray, gains: np.ndarray, output_bias: np.ndarray):
    n_classes, n_blocks, block_len = weights.shape
    offset = 0
    nw = vals[offset : offset + weights.size].reshape(weights.shape)
    offset += weights.size
    nlb = vals[offset : offset + local_bias.size].reshape(local_bias.shape)
    offset += local_bias.size
    ng = vals[offset : offset + gains.size].reshape(gains.shape)
    offset += gains.size
    nob = vals[offset : offset + output_bias.size]
    offset += output_bias.size
    y = vals[offset : offset + n_classes]
    return nw, nlb, ng, nob, y


def state_metrics(ref: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], got: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, float]:
    names = ["weights", "local_bias", "gains", "output_bias"]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=2000)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--updates",
        type=int,
        default=1,
        help="Number of batch update cycles to execute inside one transient deck.",
    )
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--init-weights", required=True)
    ap.add_argument("--no-train-gains", dest="train_gains", action="store_false")
    ap.set_defaults(train_gains=True)
    ap.add_argument("--phase", type=float, default=2e-9)
    ap.add_argument("--gap", type=float, default=0.2e-9)
    ap.add_argument("--edge", type=float, default=10e-12)
    ap.add_argument("--settle-ratio", type=float, default=20.0)
    ap.add_argument("--transient-step", type=float, default=20e-12)
    ap.add_argument("--cw", type=float, default=1e-12)
    ap.add_argument("--cstate", type=float, default=1e-12)
    ap.add_argument("--cgrad", type=float, default=1e-12)
    ap.add_argument("--rleak", type=float, default=1e18)
    ap.add_argument("--tag", default="phase_local_block")
    args = ap.parse_args()

    stride = args.block_size if args.stride is None else args.stride
    blocks = block_indices(args.image_size, args.block_size, stride)
    x_train, y_train, _x_test, _y_test = load_mnist_sequence(args.train_samples, 1, args.image_size, args.seed)
    if args.batch_size <= 0 or args.updates <= 0:
        raise ValueError("--batch-size and --updates must be positive")
    total_samples = args.batch_size * args.updates
    x_batch = x_train[:total_samples]
    y_batch = y_train[:total_samples]

    init = np.load(args.init_weights)
    weights = init["weights"].copy()
    local_bias = init["local_bias"].copy()
    gains = init["gains"].copy()
    output_bias = init["output_bias"].copy()
    expected = (10, len(blocks), args.block_size * args.block_size)
    if weights.shape != expected:
        raise ValueError(f"checkpoint weights have shape {weights.shape}, expected {expected}")

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    generated.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    stem = f"spice_mnist_local_block_phase_{sanitize_tag(args.tag)}"
    phase_netlist = generated / f"{stem}.cir"
    phase_data = results / f"{stem}.dat"
    op_netlist = generated / f"{stem}_op_reference.cir"
    op_data = results / f"{stem}_op_reference.dat"

    netlist, n_vec, t_stop = make_phase_transient_netlist(
        x_batch,
        y_batch,
        weights,
        local_bias,
        gains,
        output_bias,
        blocks,
        args.lr,
        phase_data,
        args.train_gains,
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
    )
    phase_netlist.write_text(netlist)

    t0 = time.perf_counter()
    proc = subprocess.run([spice_bin, "-b", str(phase_netlist)], text=True, capture_output=True, timeout=args.timeout)
    phase_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
    vals = read_wrdata_row(phase_data, n_vec)
    phase_w, phase_lb, phase_g, phase_ob, phase_y = unpack_state(vals, weights, local_bias, gains, output_bias)

    t1 = time.perf_counter()
    op_w, op_lb, op_g, op_ob = weights.copy(), local_bias.copy(), gains.copy(), output_bias.copy()
    for update in range(args.updates):
        start = update * args.batch_size
        stop = start + args.batch_size
        op_w, op_lb, op_g, op_ob = run_train_batch(
            spice_bin,
            op_netlist,
            op_data,
            x_batch[start:stop],
            y_batch[start:stop],
            op_w,
            op_lb,
            op_g,
            op_ob,
            blocks,
            args.lr,
            args.timeout,
            args.train_gains,
            linear_output=False,
            softmax_output=False,
            local_activation="tanh",
            relu_clip=1.0,
        )
    op_wall = time.perf_counter() - t1
    metrics = state_metrics((op_w, op_lb, op_g, op_ob), (phase_w, phase_lb, phase_g, phase_ob))
    np.savez_compressed(
        results / f"{stem}_final_weights.npz",
        weights=phase_w,
        local_bias=phase_lb,
        gains=phase_g,
        output_bias=phase_ob,
        y=phase_y,
    )
    pd.DataFrame([metrics]).to_csv(results / f"{stem}_equivalence_metrics.csv", index=False)
    summary = {
        "simulator": version,
        "architecture": "phase_resolved_transient_local_block_class_evidence",
        "status": "one_batch_equivalence_check" if args.updates == 1 else "multi_update_equivalence_check",
        "image_size": args.image_size,
        "block_size": args.block_size,
        "stride": stride,
        "blocks": len(blocks),
        "classes": 10,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "total_samples": total_samples,
        "lr": args.lr,
        "train_gains": args.train_gains,
        "init_weights": args.init_weights,
        "phase_netlist": str(phase_netlist),
        "phase_data": str(phase_data),
        "op_reference_netlist": str(op_netlist),
        "op_reference_data": str(op_data),
        "final_weights": str(results / f"{stem}_final_weights.npz"),
        "equivalence_metrics": str(results / f"{stem}_equivalence_metrics.csv"),
        "phase_wall_time_s": phase_wall,
        "op_reference_wall_time_s": op_wall,
        "t_stop_s": t_stop,
        "transient_step_s": args.transient_step,
        "phase_s": args.phase,
        "settle_ratio": args.settle_ratio,
        "persistent_state": "weights, local biases, gains, and output biases are capacitor voltages with checkpoint ICs",
        "temporary_state": "local activations, class scores, class deltas, hidden/backward deltas, and gradient accumulators are capacitor voltages",
        "python_role": "Python generates one transient deck and compares final state; it does not carry training state during the transient run.",
        **metrics,
    }
    summary_path = results / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
