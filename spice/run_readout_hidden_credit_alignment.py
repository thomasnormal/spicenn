from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


HIDDEN_CREDIT_MODES = ("readout-weighted", "readout-restored-hardgate")


def hidden_credit_lines(mode: str, *, width: float) -> list[str]:
    if mode == "readout-weighted":
        positive_weight = "vwp"
        negative_weight = "vwn"
        weight_model = "NSENSE"
        restore: list[str] = []
    elif mode == "readout-restored-hardgate":
        positive_weight = "rvwp"
        negative_weight = "rvwn"
        weight_model = "NMOS"
        restore = [
            "* Restored readout-weight sign latch.",
            "Mprecharge_rvwp rvwp rstgn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_rvwn rvwn rstgn vdd vdd PMOS W=4u L=180n",
            "Mrvwp_p rvwp rvwn vdd vdd PMOS W=6u L=180n",
            "Mrvwn_p rvwn rvwp vdd vdd PMOS W=6u L=180n",
            "Mrvwp_n rvwp vwn rvw_src 0 NSENSE W=6u L=180n",
            "Mrvwn_n rvwn vwp rvw_src 0 NSENSE W=6u L=180n",
            "Mrvw_tail rvw_src err 0 0 NMOS W=6u L=180n",
            "Crvwp rvwp 0 4f IC=1.2",
            "Crvwn rvwn 0 4f IC=1.2",
            "Rrvwp rvwp 0 1G",
            "Rrvwn rvwn 0 1G",
            "Rrvw_src rvw_src 0 1G",
        ]
    else:
        raise ValueError(f"mode must be one of {HIDDEN_CREDIT_MODES}")
    return [
        *restore,
        "* Four-quadrant hidden-credit transport.",
        f"Mhdp_pv_e vdd edp hdp_pv_e 0 NSENSE W={width:.6g}u L=180n",
        f"Mhdp_pv_w hdp_pv_e {positive_weight} hdp_pv_w 0 {weight_model} W={width:.6g}u L=180n",
        f"Mhdp_pv_a hdp_pv_w act hdp_pv_a 0 NREL W={width:.6g}u L=180n",
        f"Mhdp_pv_b hdp_pv_a bwd hdp 0 NMOS W={width:.6g}u L=180n",
        f"Mhdp_nv_e vdd edn hdp_nv_e 0 NSENSE W={width:.6g}u L=180n",
        f"Mhdp_nv_w hdp_nv_e {negative_weight} hdp_nv_w 0 {weight_model} W={width:.6g}u L=180n",
        f"Mhdp_nv_a hdp_nv_w act hdp_nv_a 0 NREL W={width:.6g}u L=180n",
        f"Mhdp_nv_b hdp_nv_a bwd hdp 0 NMOS W={width:.6g}u L=180n",
        f"Mhdn_pv_e vdd edp hdn_pv_e 0 NSENSE W={width:.6g}u L=180n",
        f"Mhdn_pv_w hdn_pv_e {negative_weight} hdn_pv_w 0 {weight_model} W={width:.6g}u L=180n",
        f"Mhdn_pv_a hdn_pv_w act hdn_pv_a 0 NREL W={width:.6g}u L=180n",
        f"Mhdn_pv_b hdn_pv_a bwd hdn 0 NMOS W={width:.6g}u L=180n",
        f"Mhdn_nv_e vdd edn hdn_nv_e 0 NSENSE W={width:.6g}u L=180n",
        f"Mhdn_nv_w hdn_nv_e {positive_weight} hdn_nv_w 0 {weight_model} W={width:.6g}u L=180n",
        f"Mhdn_nv_a hdn_nv_w act hdn_nv_a 0 NREL W={width:.6g}u L=180n",
        f"Mhdn_nv_b hdn_nv_a bwd hdn 0 NMOS W={width:.6g}u L=180n",
    ]


def generated_internal_node_names() -> list[str]:
    out: list[str] = []
    for rail in ["hdp", "hdn"]:
        for quadrant in ["pv", "nv"]:
            for suffix in ["e", "w", "a"]:
                out.append(f"{rail}_{quadrant}_{suffix}")
    return out


def generate_netlist(
    *,
    mode: str,
    vwp: float,
    vwn: float,
    edp: float = 1.2,
    edn: float = 0.0,
    act: float = 0.75,
    width: float = 32.0,
) -> str:
    if mode not in HIDDEN_CREDIT_MODES:
        raise ValueError(f"mode must be one of {HIDDEN_CREDIT_MODES}")
    if width <= 0.0:
        raise ValueError("width must be positive")
    lines = [
        "* Readout hidden-credit alignment smoke.",
        "* Uses transistor/passive hidden-credit paths; no behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        "Vrstgn rstgn 0 PULSE(0 1.2 1n 10p 10p 0.8n 24n)",
        "Verr err 0 PULSE(0 1.2 2n 10p 10p 6n 24n)",
        "Vbwd bwd 0 PULSE(0 1.2 9n 10p 10p 5n 24n)",
        f"Vedp edp 0 {edp:.12g}",
        f"Vedn edn 0 {edn:.12g}",
        f"Cact act 0 20f IC={act:.12g}",
        "Ract act 0 1G",
        f"Cvwp vwp 0 20f IC={vwp:.12g}",
        f"Cvwn vwn 0 20f IC={vwn:.12g}",
        "Rvwp vwp 0 1G",
        "Rvwn vwn 0 1G",
        "Chdp hdp 0 12f IC=0",
        "Chdn hdn 0 12f IC=0",
        "Rhdp hdp 0 1G",
        "Rhdn hdn 0 1G",
        *[f"R{node} {node} 0 1G" for node in generated_internal_node_names()],
        *hidden_credit_lines(mode, width=width),
        ".meas tran hdp_after FIND V(hdp) AT=15n",
        ".meas tran hdn_after FIND V(hdn) AT=15n",
        ".meas tran hidden_credit_margin PARAM='hdp_after-hdn_after'",
    ]
    if mode == "readout-restored-hardgate":
        lines += [
            ".meas tran rvwp_after FIND V(rvwp) AT=8n",
            ".meas tran rvwn_after FIND V(rvwn) AT=8n",
            ".meas tran rvw_margin PARAM='rvwp_after-rvwn_after'",
        ]
    lines += [
        ".tran 5p 18n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def sweep_points(center: float, deltas: list[float]) -> list[tuple[float, float, float]]:
    return [(delta, center + delta / 2.0, center - delta / 2.0) for delta in deltas]


def classify_row(row: dict[str, Any], *, min_abs_margin: float) -> str:
    delta = float(row["delta"])
    margin = float(row["hidden_credit_margin"])
    if abs(delta) < 1e-15:
        return "dead_zone" if abs(margin) < min_abs_margin else "biased"
    if delta > 0.0 and margin >= min_abs_margin:
        return "aligned"
    if delta < 0.0 and margin <= -min_abs_margin:
        return "aligned"
    if abs(margin) < min_abs_margin:
        return "weak"
    return "flipped"


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    deltas = [float(value) for value in args.deltas.split(",") if value]
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for mode in args.modes:
        for delta, vwp, vwn in sweep_points(args.center, deltas):
            path = generated / f"{tag}_{mode}_{delta:+.3f}.cir".replace("+", "p").replace("-", "m")
            measures = run_netlist(
                spice_bin,
                path,
                generate_netlist(mode=mode, vwp=vwp, vwn=vwn, edp=args.edp, edn=args.edn, act=args.act, width=args.width),
                timeout=args.timeout,
            )
            row = {
                "mode": mode,
                "delta": delta,
                "vwp": vwp,
                "vwn": vwn,
                **measures,
            }
            row["classification"] = classify_row(row, min_abs_margin=args.min_abs_margin)
            rows.append(row)
    csv_path = tables / f"{tag}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        mode = str(row["mode"])
        cls = str(row["classification"])
        counts.setdefault(mode, {})
        counts[mode][cls] = counts[mode].get(cls, 0) + 1
    summary = {
        "simulator": version,
        "architecture": "readout_hidden_credit_alignment",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "modes": args.modes,
        "deltas": deltas,
        "center": args.center,
        "min_abs_margin": args.min_abs_margin,
        "classification_counts": counts,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="readout_hidden_credit_alignment")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--modes", nargs="+", choices=HIDDEN_CREDIT_MODES, default=list(HIDDEN_CREDIT_MODES))
    ap.add_argument("--center", type=float, default=0.35)
    ap.add_argument("--deltas", default="-0.08,-0.04,-0.02,-0.01,0,0.01,0.02,0.04,0.08")
    ap.add_argument("--edp", type=float, default=1.2)
    ap.add_argument("--edn", type=float, default=0.0)
    ap.add_argument("--act", type=float, default=0.75)
    ap.add_argument("--width", type=float, default=32.0)
    ap.add_argument("--min-abs-margin", type=float, default=1e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.width <= 0.0:
        raise ValueError("width must be positive")
    if args.min_abs_margin < 0.0:
        raise ValueError("min-abs-margin must be nonnegative")
    if not [value for value in args.deltas.split(",") if value]:
        raise ValueError("deltas must not be empty")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_sweep(args), indent=2))


if __name__ == "__main__":
    main()
