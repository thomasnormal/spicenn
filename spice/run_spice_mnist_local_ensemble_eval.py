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
    append_center_block,
    block_indices,
)
from run_spice_mnist_train import load_mnist_sequence
from run_spice_sweep import ROOT, detect_spice, prepare_netlist_for_simulator, run_tiny_test, run_simulator_netlist


def parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_model_spec(text: str) -> dict[str, object]:
    spec: dict[str, object] = {
        "activation": "tanh",
        "output": "tanh",
        "block_size": 7,
        "stride": None,
        "center": False,
        "weight": 1.0,
        "relu_clip": 1.0,
    }
    for part in text.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        if key in {"path", "activation", "output"}:
            spec[key] = value
        elif key in {"block_size", "stride"}:
            spec[key] = int(value)
        elif key == "center":
            spec[key] = parse_bool(value)
        elif key in {"weight", "relu_clip"}:
            spec[key] = float(value)
        else:
            raise ValueError(f"unknown model spec key {key!r}")
    if "path" not in spec:
        raise ValueError(f"model spec needs path=...: {text}")
    if spec["stride"] is None:
        spec["stride"] = int(spec["block_size"])
    return spec


def load_model(spec: dict[str, object], image_size: int) -> dict[str, object]:
    path = Path(str(spec["path"]))
    ckpt = np.load(path, allow_pickle=True)
    weights = np.asarray(ckpt["weights"], dtype=float)
    local_bias = np.asarray(ckpt["local_bias"], dtype=float)
    gains = np.asarray(ckpt["gains"], dtype=float)
    output_bias = np.asarray(ckpt["output_bias"], dtype=float)
    block_size = int(spec["block_size"])
    stride = int(spec["stride"])
    blocks = block_indices(image_size, block_size, stride)
    if bool(spec["center"]):
        blocks = append_center_block(blocks, image_size, block_size)
    expected = (10, len(blocks), block_size * block_size)
    if weights.shape != expected or local_bias.shape != (10, len(blocks)) or gains.shape != (10, len(blocks)):
        raise ValueError(f"{path} has shapes weights={weights.shape}, local_bias={local_bias.shape}, gains={gains.shape}; expected {expected}")
    return {
        **spec,
        "path": path,
        "weights": weights,
        "local_bias": local_bias,
        "gains": gains,
        "output_bias": output_bias,
        "blocks": blocks,
    }


def make_ensemble_eval_netlist(
    x_batch: np.ndarray,
    models: list[dict[str, object]],
    out_path: Path,
    write_branch_scores: bool = False,
) -> str:
    batch, n_in = x_batch.shape
    n_classes = 10
    lines = [
        "* Local block ensemble SPICE inference.",
        "* Each branch computes local nonlinear evidence; ensemble scores are summed in SPICE.",
        "",
    ]
    for s in range(batch):
        for i in range(n_in):
            lines.append(f"Vx{s}_{i} x{s}_{i} 0 DC {float(x_batch[s, i]):.12g}")
    lines.append("")

    for mi, model in enumerate(models):
        weights = model["weights"]
        local_bias = model["local_bias"]
        gains = model["gains"]
        output_bias = model["output_bias"]
        blocks = model["blocks"]
        for k in range(n_classes):
            for b, idxs in enumerate(blocks):
                for p, _idx in enumerate(idxs):
                    lines.append(f"Vw{mi}_{k}_{b}_{p} w{mi}_{k}_{b}_{p} 0 DC {weights[k, b, p]:.12g}")
                lines.append(f"Vlb{mi}_{k}_{b} lb{mi}_{k}_{b} 0 DC {local_bias[k, b]:.12g}")
                lines.append(f"Vg{mi}_{k}_{b} g{mi}_{k}_{b} 0 DC {gains[k, b]:.12g}")
            lines.append(f"Vob{mi}_{k} ob{mi}_{k} 0 DC {output_bias[k]:.12g}")
    lines.append("")

    for s in range(batch):
        calibrated_terms_by_class: list[list[str]] = [[] for _ in range(n_classes)]
        for mi, model in enumerate(models):
            blocks = model["blocks"]
            activation = str(model["activation"])
            output = str(model["output"])
            relu_clip = float(model["relu_clip"])
            weight = float(model["weight"])
            for k in range(n_classes):
                h_terms = []
                for b, idxs in enumerate(blocks):
                    terms = [f"V(w{mi}_{k}_{b}_{p})*V(x{s}_{idx})" for p, idx in enumerate(idxs)]
                    terms.append(f"V(lb{mi}_{k}_{b})")
                    h_expr, _deriv = add_local_activation(lines, s, 1000 * mi + k, b, " + ".join(terms), activation, relu_clip)
                    h_terms.append(f"V(g{mi}_{k}_{b})*{h_expr}")
                out_sum = " + ".join(h_terms + [f"V(ob{mi}_{k})"])
                lines.append(f"Bz{mi}_{s}_{k} z{mi}_{s}_{k} 0 V = {out_sum}")
            if output == "softmax":
                denom = " + ".join(f"exp(V(z{mi}_{s}_{kk}))" for kk in range(n_classes))
                for k in range(n_classes):
                    lines.append(f"By{mi}_{s}_{k} y{mi}_{s}_{k} 0 V = exp(V(z{mi}_{s}_{k}))/({denom})")
                    lines.append(f"Bcal{mi}_{s}_{k} cal{mi}_{s}_{k} 0 V = V(y{mi}_{s}_{k})")
                    calibrated_terms_by_class[k].append(f"{weight:.12g}*V(cal{mi}_{s}_{k})")
            elif output == "linear":
                for k in range(n_classes):
                    lines.append(f"By{mi}_{s}_{k} y{mi}_{s}_{k} 0 V = V(z{mi}_{s}_{k})")
                    lines.append(f"Bcal{mi}_{s}_{k} cal{mi}_{s}_{k} 0 V = V(y{mi}_{s}_{k})")
                    calibrated_terms_by_class[k].append(f"{weight:.12g}*V(cal{mi}_{s}_{k})")
            else:
                for k in range(n_classes):
                    lines.append(f"By{mi}_{s}_{k} y{mi}_{s}_{k} 0 V = 2/(1+exp(-2*V(z{mi}_{s}_{k})))-1")
                    lines.append(f"Bcal{mi}_{s}_{k} cal{mi}_{s}_{k} 0 V = 0.5*(V(y{mi}_{s}_{k})+1)")
                    calibrated_terms_by_class[k].append(f"{weight:.12g}*V(cal{mi}_{s}_{k})")
        for k in range(n_classes):
            lines.append(f"Bens{s}_{k} ens{s}_{k} 0 V = {' + '.join(calibrated_terms_by_class[k])}")

    vectors = [f"V(ens{s}_{k})" for s in range(batch) for k in range(n_classes)]
    if write_branch_scores:
        vectors += [f"V(cal{mi}_{s}_{k})" for mi in range(len(models)) for s in range(batch) for k in range(n_classes)]
    lines += ["", ".control", "op", f"wrdata {out_path} " + " ".join(vectors), ".endc", ".end", ""]
    return "\n".join(lines)


def run_eval(spice_bin, netlist_path, data_path, x_eval, y_eval, models, batch_size, timeout, collect_branch_scores=False):
    correct = 0
    n_classes = 10
    branch_batches = []
    for start in range(0, len(y_eval), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        netlist_path.write_text(
            prepare_netlist_for_simulator(
                make_ensemble_eval_netlist(x, models, data_path, write_branch_scores=collect_branch_scores),
                spice_bin,
            )
        )
        proc = run_simulator_netlist(spice_bin, netlist_path, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:] or proc.stdout[-3000:])
        n_ens = len(y) * n_classes
        n_branch = len(models) * len(y) * n_classes if collect_branch_scores else 0
        vals = read_wrdata_row(data_path, n_ens + n_branch)
        ens = vals[:n_ens].reshape(len(y), n_classes)
        correct += int((np.argmax(ens, axis=1) == y).sum())
        if collect_branch_scores:
            branch_batches.append(vals[n_ens:].reshape(len(models), len(y), n_classes))
    accuracy = correct / max(len(y_eval), 1)
    if collect_branch_scores:
        return accuracy, np.concatenate(branch_batches, axis=1)
    return accuracy, None


def parse_sweep(text: str) -> np.ndarray:
    start, stop, steps = text.split(":")
    return np.linspace(float(start), float(stop), int(steps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="Comma spec: path=...,activation=tanh,output=tanh,block_size=7,stride=7,center=false,weight=1")
    ap.add_argument("--train-samples", type=int, default=2000, help="Only used to reproduce the same RNG train/test split as training runs.")
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--sweep-second-weight", default="", help="Optional offline sweep start:stop:steps using SPICE-computed branch scores for exactly two models.")
    ap.add_argument("--tag", default="local_ensemble")
    args = ap.parse_args()

    _x_train, _y_train, x_test, y_test = load_mnist_sequence(args.train_samples, args.test_samples, args.image_size, args.seed)
    models = [load_model(parse_model_spec(text), args.image_size) for text in args.model]
    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    stem = f"spice_mnist_local_ensemble_{safe_tag}"
    netlist_path = generated / f"{stem}_eval.cir"
    data_path = ROOT / f"spice/results/{stem}_eval.dat"
    t0 = time.perf_counter()
    collect_branch_scores = bool(args.sweep_second_weight)
    accuracy, branch_scores = run_eval(
        spice_bin,
        netlist_path,
        data_path,
        x_test,
        y_test,
        models,
        args.batch_size,
        args.timeout,
        collect_branch_scores=collect_branch_scores,
    )
    rows = [{"heldout_accuracy": accuracy}]
    table_path = ROOT / f"spice/results/{stem}_results.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    sweep_path = None
    sweep_best_weight = None
    sweep_best_accuracy = None
    if args.sweep_second_weight:
        if len(models) != 2:
            raise ValueError("--sweep-second-weight currently expects exactly two models")
        alphas = parse_sweep(args.sweep_second_weight)
        sweep_rows = []
        assert branch_scores is not None
        for alpha in alphas:
            scores = branch_scores[0] + alpha * branch_scores[1]
            acc = float((np.argmax(scores, axis=1) == y_test).mean())
            sweep_rows.append({"second_weight": float(alpha), "heldout_accuracy": acc})
        sweep = pd.DataFrame(sweep_rows)
        best_row = sweep.sort_values(["heldout_accuracy", "second_weight"], ascending=[False, True]).iloc[0]
        sweep_best_weight = float(best_row["second_weight"])
        sweep_best_accuracy = float(best_row["heldout_accuracy"])
        sweep_path = ROOT / f"spice/results/{stem}_weight_sweep.csv"
        sweep.to_csv(sweep_path, index=False)
    summary = {
        "simulator": version,
        "dataset": "MNIST test split, downsampled",
        "architecture": "local_block_spice_ensemble_eval",
        "image_size": args.image_size,
        "train_samples_for_split": args.train_samples,
        "test_samples": args.test_samples,
        "batch_size": args.batch_size,
        "models": [
            {
                "path": str(model["path"]),
                "activation": model["activation"],
                "output": model["output"],
                "block_size": model["block_size"],
                "stride": model["stride"],
                "center": model["center"],
                "weight": model["weight"],
                "blocks": len(model["blocks"]),
            }
            for model in models
        ],
        "heldout_test_accuracy": accuracy,
        "sweep_second_weight": args.sweep_second_weight,
        "sweep_best_second_weight": sweep_best_weight,
        "sweep_best_heldout_accuracy": sweep_best_accuracy,
        "sweep_table": str(sweep_path) if sweep_path else None,
        "wall_time_s": time.perf_counter() - t0,
        "table": str(table_path),
        "note": "Eval-only diagnostic: all branch nonlinear evidence and ensemble scores are computed inside ngspice; weights were trained by prior SPICE runs.",
    }
    out = ROOT / f"spice/results/{stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
