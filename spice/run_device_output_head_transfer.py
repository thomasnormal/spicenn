from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from run_device_multicell_classifier import mos_models
import run_device_xor2_random_hidden as direct_flow
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test
from _util import MEAS_RE, parse_measures


def default_split_score_cases() -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    return [
        ((0.40, 0.35), (0.40, 0.35), (0.40, 0.35)),
        ((0.46, 0.35), (0.40, 0.35), (0.38, 0.35)),
        ((0.40, 0.35), (0.46, 0.35), (0.38, 0.35)),
        ((0.40, 0.35), (0.38, 0.35), (0.46, 0.35)),
        ((0.45, 0.38), (0.43, 0.35), (0.44, 0.42)),
        ((0.55, 0.50), (0.45, 0.40), (0.38, 0.33)),
        ((0.55, 0.50), (0.43, 0.32), (0.38, 0.35)),
        ((0.35, 0.25), (0.55, 0.48), (0.75, 0.70)),
        ((0.62, 0.56), (0.42, 0.30), (0.68, 0.60)),
    ]


def output_initial_voltage(output_head: str, score_reset_v: float, vdd: float) -> float:
    if output_head in direct_flow.COMMON_MODE_OUT_RESET_HEADS:
        return score_reset_v
    if output_head in direct_flow.LOW_TRUE_OUTPUT_HEADS:
        return vdd
    return 0.0


def output_decision_value(output_head: str, out_v: float) -> float:
    return -out_v if output_head in direct_flow.LOW_TRUE_OUTPUT_HEADS else out_v


def output_head_netlist(
    score_pairs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    output_head: str,
    design_name: str,
    output_width_scale: float,
    output_cap_f: float,
    score_reset_v: float,
    tstop_ns: float,
    score_diode_width_u: float = 1024.0,
    score_mirror_cap_f: float = 20.0,
) -> str:
    original_outputs = direct_flow.OUTPUTS
    try:
        direct_flow.set_output_count(3)
        design = direct_flow.scaled_synapse_design(
            design_name,
            hidden_delta_width_scale=1.0,
            hidden_gradient_width_scale=1.0,
            readout_gradient_width_scale=1.0,
            output_forward_width_scale=1.0,
            output_relu_width_scale=output_width_scale,
        )
        head_cells = "\n".join(
            direct_flow.output_head_from_scores(design, output_head, out, score_diode_width_u, score_mirror_cap_f)
            for out in range(3)
        )
        shared_head_cells = direct_flow.output_head_shared_cells(design, output_head)
    finally:
        direct_flow.set_output_count(original_outputs)

    out_ic = output_initial_voltage(output_head, score_reset_v, 1.2)
    score_sources = "\n".join(
        f"Vscorep{k} scorep{k} 0 {scorep:.12g}\nVscoren{k} scoren{k} 0 {scoren:.12g}\n"
        f"Vscore{k} score{k} 0 {scorep - scoren:.12g}"
        for k, (scorep, scoren) in enumerate(score_pairs)
    )
    output_caps = "\n".join(
        f"Cout{k} out{k} 0 {output_cap_f:.12g}f IC={out_ic:.12g}\nRout{k}_leak out{k} 0 1e12" for k in range(3)
    )
    measure_ns = 0.95 * tstop_ns
    measures = "\n".join(f".meas tran out{k} FIND V(out{k}) AT={measure_ns:.12g}n" for k in range(3))
    return f"""
* Production output-head transfer harness.
* Fixed scorep/scoren voltages drive the same score-to-out MOS fragments used
* by the full device-level network.
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vfwd fwd 0 1.0
Voutg outg 0 1.0
Vrstf rstf 0 0
{score_sources}
{output_caps}

{head_cells}
{shared_head_cells}

.tran 1p {tstop_ns:.12g}n uic
{measures}
.control
run
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_output_head_transfer")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--simulator", default=None)
    ap.add_argument(
        "--output-head",
        choices=[
            "split_score_caps",
            "split_score_diffgate",
            "split_score_chargegate",
            "split_score_diffpair",
            "split_score_diode_diffpair",
            "split_score_compete_tail",
            "split_score_diode_mirror_gate_caps",
            "split_score_diode_mirror_caps",
        ],
        default="split_score_diffpair",
    )
    ap.add_argument("--design", default="split_signed_v1")
    ap.add_argument("--output-width-scale", type=float, default=1.0)
    ap.add_argument("--output-cap-f", type=float, default=10.0)
    ap.add_argument("--score-reset-v", type=float, default=0.30)
    ap.add_argument("--score-diode-width-u", type=float, default=1024.0)
    ap.add_argument("--score-mirror-cap-f", type=float, default=20.0)
    ap.add_argument("--tstop-ns", type=float, default=0.30)
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
    for idx, pairs in enumerate(default_split_score_cases()):
        deck = output_head_netlist(
            pairs,
            args.output_head,
            args.design,
            args.output_width_scale,
            args.output_cap_f,
            args.score_reset_v,
            args.tstop_ns,
            args.score_diode_width_u,
            args.score_mirror_cap_f,
        )
        parsed = run_netlist(spice_bin, generated / f"{safe_tag}_{idx}.cir", deck, args.timeout)
        signed_scores = [scorep - scoren for scorep, scoren in pairs]
        outputs = [parsed[f"out{k}"] for k in range(3)]
        decisions = [output_decision_value(args.output_head, out_v) for out_v in outputs]
        expected_argmax = int(max(range(3), key=lambda k: signed_scores[k]))
        output_argmax = int(max(range(3), key=lambda k: decisions[k]))
        is_tie = max(signed_scores) - min(signed_scores) <= 1e-12
        rows.append(
            {
                "case": idx,
                **{f"scorep{k}_v": pairs[k][0] for k in range(3)},
                **{f"scoren{k}_v": pairs[k][1] for k in range(3)},
                **{f"signed_score{k}_v": signed_scores[k] for k in range(3)},
                **{f"out{k}_v": outputs[k] for k in range(3)},
                **{f"decision{k}": decisions[k] for k in range(3)},
                "score_argmax": expected_argmax,
                "output_argmax": output_argmax,
                "is_tie": is_tie,
                "argmax_matches": expected_argmax == output_argmax,
            }
        )

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)
    nontie = df.loc[~df["is_tie"].astype(bool)]
    summary = {
        "simulator": version,
        "architecture": f"production_{args.output_head}_score_to_output_transfer",
        "output_head": args.output_head,
        "design": args.design,
        "output_width_scale": args.output_width_scale,
        "output_cap_f": args.output_cap_f,
        "score_diode_width_u": args.score_diode_width_u
        if args.output_head in {"split_score_diode_diffpair", *direct_flow.DIODE_MIRROR_OUTPUT_HEADS}
        else None,
        "score_mirror_cap_f": args.score_mirror_cap_f if args.output_head in direct_flow.DIODE_MIRROR_OUTPUT_HEADS else None,
        "tstop_ns": args.tstop_ns,
        "all_argmax_matches": bool(df["argmax_matches"].all()),
        "all_nontie_argmax_matches": bool(nontie["argmax_matches"].all()),
        "n_cases": int(len(df)),
        "n_nontie_cases": int(len(nontie)),
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "wall_time_s": time.perf_counter() - t0,
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
