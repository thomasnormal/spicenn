from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from run_device_multicell_classifier import mos_models
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test
from _util import MEAS_RE, parse_measures


def netlist(
    scores: tuple[float, float, float],
    branch_model: str,
    branch_width_u: float,
    tail_width_u: float,
    tail_gate_v: float,
) -> str:
    if branch_model not in {"NMOS", "NREL", "NSENSE"}:
        raise ValueError(f"unknown branch model: {branch_model}")
    score_sources = "\n".join(f"Vscore{k} score{k} 0 {score:.12g}" for k, score in enumerate(scores))
    branches = "\n".join(
        [
            f"Vsense{k} vdd drain{k} 0",
            f"Mbranch{k} drain{k} score{k} src 0 {branch_model} W={branch_width_u:.12g}u L=180n",
        ][line]
        for k in range(3)
        for line in range(2)
    )
    measures = "\n".join(f".meas tran i{k} FIND I(Vsense{k}) AT=0.05n" for k in range(3))
    prints = "\n".join(f"print i{k}" for k in range(3))
    return f"""
* Three-class source-coupled MOS current competition.
* Branch currents are the hardware-native softmax-like class probabilities.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
{score_sources}
Vtail tail 0 {tail_gate_v:.12g}

{branches}
Mtail src tail 0 0 NMOS W={tail_width_u:.12g}u L=180n
Rsrc src 0 1e12

.tran 1p 0.1n
{measures}
.control
run
{prints}
.endc
.end
""".lstrip()


def split_score_netlist(
    score_pairs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    branch_model: str,
    branch_width_u: float,
    inhibit_width_u: float,
    tail_width_u: float,
    tail_gate_v: float,
) -> str:
    if branch_model not in {"NMOS", "NREL", "NSENSE"}:
        raise ValueError(f"unknown branch model: {branch_model}")
    score_sources = "\n".join(
        f"Vscorep{k} scorep{k} 0 {scorep:.12g}\nVscoren{k} scoren{k} 0 {scoren:.12g}"
        for k, (scorep, scoren) in enumerate(score_pairs)
    )
    branches = "\n".join(
        [
            f"Vsense{k} vdd psrc{k} 0",
            f"Minhibit{k} mid{k} scoren{k} psrc{k} vdd PMOS W={inhibit_width_u:.12g}u L=180n",
            f"Mbranch{k} mid{k} scorep{k} src 0 {branch_model} W={branch_width_u:.12g}u L=180n",
        ][line]
        for k in range(3)
        for line in range(3)
    )
    measures = "\n".join(f".meas tran i{k} FIND I(Vsense{k}) AT=0.05n" for k in range(3))
    prints = "\n".join(f"print i{k}" for k in range(3))
    return f"""
* Three-class split-score MOS current competition.
* scorep opens the class branch; scoren inhibits it through a PMOS source device.
* This is a hardware-native positive-current competition for split score capacitors.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
{score_sources}
Vtail tail 0 {tail_gate_v:.12g}

{branches}
Mtail src tail 0 0 NMOS W={tail_width_u:.12g}u L=180n
Rsrc src 0 1e12

.tran 1p 0.1n
{measures}
.control
run
{prints}
.endc
.end
""".lstrip()


def split_differential_pair_netlist(
    score_pairs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    branch_model: str,
    branch_width_u: float,
    tail_width_u: float,
    tail_gate_v: float,
) -> str:
    if branch_model not in {"NMOS", "NREL", "NSENSE"}:
        raise ValueError(f"unknown branch model: {branch_model}")
    score_sources = "\n".join(
        f"Vscorep{k} scorep{k} 0 {scorep:.12g}\nVscoren{k} scoren{k} 0 {scoren:.12g}"
        for k, (scorep, scoren) in enumerate(score_pairs)
    )
    branches = "\n".join(
        [
            f"Vsense{k} vdd posdrain{k} 0",
            f"Vdump{k} vdd negdrain{k} 0",
            f"Mpos{k} posdrain{k} scorep{k} pairsrc{k} 0 {branch_model} W={branch_width_u:.12g}u L=180n",
            f"Mneg{k} negdrain{k} scoren{k} pairsrc{k} 0 {branch_model} W={branch_width_u:.12g}u L=180n",
            f"Mtail{k} pairsrc{k} tail 0 0 NMOS W={tail_width_u:.12g}u L=180n",
            f"Rsrc{k} pairsrc{k} 0 1e12",
        ][line]
        for k in range(3)
        for line in range(6)
    )
    measures = "\n".join(f".meas tran i{k} FIND I(Vsense{k}) AT=0.08n" for k in range(3))
    prints = "\n".join(f"print i{k}" for k in range(3))
    return f"""
* Three-class split-score differential-pair current competition.
* Each class first compares scorep against scoren with a local source-coupled
* pair.  The measured class current is the positive-branch current, so common
* score motion is rejected before classes are normalized by current share.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
{score_sources}
Vtail tail 0 {tail_gate_v:.12g}

{branches}

.tran 1p 0.12n
{measures}
.control
run
{prints}
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, text: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, text, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed


def default_cases() -> list[tuple[float, float, float]]:
    return [
        (0.35, 0.35, 0.35),
        (0.38, 0.35, 0.35),
        (0.42, 0.38, 0.35),
        (0.46, 0.42, 0.35),
        (0.35, 0.38, 0.35),
        (0.35, 0.35, 0.38),
        (0.45, 0.40, 0.38),
    ]


def default_split_score_cases() -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    return [
        ((0.40, 0.35), (0.40, 0.35), (0.40, 0.35)),
        ((0.46, 0.35), (0.40, 0.35), (0.38, 0.35)),
        ((0.40, 0.35), (0.46, 0.35), (0.38, 0.35)),
        ((0.40, 0.35), (0.38, 0.35), (0.46, 0.35)),
        ((0.45, 0.38), (0.43, 0.35), (0.44, 0.42)),
        ((0.55, 0.50), (0.45, 0.40), (0.38, 0.33)),
    ]


def default_differential_pair_cases() -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    return [
        ((0.40, 0.35), (0.40, 0.35), (0.40, 0.35)),
        ((0.46, 0.35), (0.55, 0.50), (0.38, 0.35)),
        ((0.55, 0.50), (0.43, 0.32), (0.38, 0.35)),
        ((0.55, 0.50), (0.38, 0.35), (0.43, 0.32)),
        ((0.35, 0.25), (0.55, 0.48), (0.75, 0.70)),
        ((0.62, 0.56), (0.42, 0.30), (0.68, 0.60)),
        ((0.30, 0.18), (0.65, 0.57), (0.50, 0.43)),
    ]


def current_shares(parsed: dict[str, float]) -> tuple[list[float], list[float], float]:
    currents = [abs(parsed[f"i{k}"]) for k in range(3)]
    total = sum(currents)
    shares = [current / total if total > 0 else 0.0 for current in currents]
    return currents, shares, total


def is_tie(values: list[float], *, eps: float = 1e-12) -> bool:
    return max(values) - min(values) <= eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_softmax_current_competition")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--simulator", default=None)
    ap.add_argument(
        "--mode",
        choices=["single-ended", "split-score", "split-differential-pair"],
        default="split-differential-pair",
    )
    ap.add_argument("--branch-model", choices=["NMOS", "NREL", "NSENSE"], default="NREL")
    ap.add_argument("--branch-width-u", type=float, default=12.0)
    ap.add_argument("--inhibit-width-u", type=float, default=9.0)
    ap.add_argument("--tail-width-u", type=float, default=32.0)
    ap.add_argument("--tail-gate-v", type=float, default=0.42)
    args = ap.parse_args()

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    rows: list[dict[str, float | int | bool | str]] = []
    t0 = time.perf_counter()
    if args.mode == "single-ended":
        case_payloads = default_cases()
    elif args.mode == "split-differential-pair":
        case_payloads = default_differential_pair_cases()
    else:
        case_payloads = default_split_score_cases()
    for idx, payload in enumerate(case_payloads):
        if args.mode == "single-ended":
            scores = list(payload)  # type: ignore[arg-type]
            deck = netlist(
                payload,  # type: ignore[arg-type]
                args.branch_model,
                args.branch_width_u,
                args.tail_width_u,
                args.tail_gate_v,
            )
            base_row = {f"score{k}_v": scores[k] for k in range(3)}
        else:
            pairs = payload  # type: ignore[assignment]
            scores = [scorep - scoren for scorep, scoren in pairs]
            if args.mode == "split-differential-pair":
                deck = split_differential_pair_netlist(
                    pairs,
                    args.branch_model,
                    args.branch_width_u,
                    args.tail_width_u,
                    args.tail_gate_v,
                )
            else:
                deck = split_score_netlist(
                    pairs,
                    args.branch_model,
                    args.branch_width_u,
                    args.inhibit_width_u,
                    args.tail_width_u,
                    args.tail_gate_v,
                )
            base_row = {
                **{f"scorep{k}_v": pairs[k][0] for k in range(3)},
                **{f"scoren{k}_v": pairs[k][1] for k in range(3)},
                **{f"signed_score{k}_v": scores[k] for k in range(3)},
            }
        parsed = run_netlist(spice_bin, generated / f"{safe_tag}_{idx}.cir", deck, args.timeout)
        currents, shares, total = current_shares(parsed)
        expected_argmax = int(max(range(3), key=lambda k: scores[k]))
        share_argmax = int(max(range(3), key=lambda k: shares[k]))
        rows.append(
            {
                **base_row,
                "case": idx,
                "i0_a": currents[0],
                "i1_a": currents[1],
                "i2_a": currents[2],
                "total_i_a": total,
                "share0": shares[0],
                "share1": shares[1],
                "share2": shares[2],
                "score_argmax": expected_argmax,
                "share_argmax": share_argmax,
                "is_tie": is_tie(scores),
                "argmax_matches": expected_argmax == share_argmax,
            }
        )

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    equal = df.iloc[0]
    total_values = df["total_i_a"].to_numpy()
    summary = {
        "simulator": version,
        "architecture": f"three_class_{args.mode.replace('-', '_')}_mos_current_competition",
        "status": "softmax_like_current_normalizer_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a subthreshold/PDK model.",
        "mode": args.mode,
        "branch_model": args.branch_model,
        "branch_width_u": args.branch_width_u,
        "inhibit_width_u": args.inhibit_width_u,
        "tail_width_u": args.tail_width_u,
        "tail_gate_v": args.tail_gate_v,
        "no_behavioral_exp_or_divide_in_signal_path": True,
        "probabilities_measured_offline_as_current_shares": True,
        "all_argmax_matches": bool(df["argmax_matches"].all()),
        "all_nontie_argmax_matches": bool(df.loc[~df["is_tie"].astype(bool), "argmax_matches"].all()),
        "equal_input_max_share_error": float(
            max(abs(float(equal[f"share{k}"]) - 1.0 / 3.0) for k in range(3))
        ),
        "total_current_min_a": float(total_values.min()),
        "total_current_max_a": float(total_values.max()),
        "total_current_relative_span": float(
            (total_values.max() - total_values.min()) / total_values.mean()
            if total_values.mean() > 0
            else 0.0
        ),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "This is not exact softmax in the current LEVEL=1 model. It is a hardware-native normalized "
            "positive-current competition. In subthreshold CMOS, the same topology is the standard path "
            "toward exponential softmax-like branch currents."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
