from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_block_training import readout_weight_update_lines
from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


UPDATE_MODES = ("none", "positive", "negative")
READOUT_WRITER_TOPOLOGIES = ("rail", "bounded-ref")
GRADIENT_GATE_TOPOLOGIES = ("direct", "restored")


def update_rails(mode: str, *, amplitude: float) -> tuple[float, float]:
    if mode == "none":
        return 0.0, 0.0
    if mode == "positive":
        return amplitude, 0.0
    if mode == "negative":
        return 0.0, amplitude
    raise ValueError(f"update mode must be one of {UPDATE_MODES}")


def generate_netlist(
    *,
    update_mode: str,
    topology: str = "bounded-ref",
    vwp: float = 0.36,
    vwn: float = 0.34,
    positive_ref: float = 0.36,
    negative_ref: float = 0.34,
    update_span: float = 0.15,
    gradient_width: float = 24.0,
    gradient_restore_width: float = 16.0,
    gradient_gate_topology: str = "direct",
    gate_amplitude: float = 1.2,
    gate_restore_width: float = 7.0,
    update_scale: float = 0.10,
    error_amplitude: float = 1.2,
) -> str:
    if update_mode not in UPDATE_MODES:
        raise ValueError(f"update_mode must be one of {UPDATE_MODES}")
    if topology not in READOUT_WRITER_TOPOLOGIES:
        raise ValueError(f"topology must be one of {READOUT_WRITER_TOPOLOGIES}")
    if gradient_gate_topology not in GRADIENT_GATE_TOPOLOGIES:
        raise ValueError(f"gradient_gate_topology must be one of {GRADIENT_GATE_TOPOLOGIES}")
    for name, value in {
        "gradient_width": gradient_width,
        "gradient_restore_width": gradient_restore_width,
        "gate_amplitude": gate_amplitude,
        "gate_restore_width": gate_restore_width,
        "update_scale": update_scale,
        "error_amplitude": error_amplitude,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if gate_amplitude > 1.2:
        raise ValueError("gate_amplitude must not exceed VDD")
    if error_amplitude > 1.2:
        raise ValueError("error_amplitude must not exceed VDD")
    if update_span < 0.0:
        raise ValueError("update_span must be nonnegative")
    high_ref = positive_ref + update_span
    low_ref = negative_ref - update_span
    if topology == "bounded-ref" and (high_ref > 1.2 or low_ref < 0.0):
        raise ValueError("bounded-ref update references must stay within supply rails")
    dp, dn = update_rails(update_mode, amplitude=error_amplitude)
    lines = [
        "* Readout writer primitive smoke.",
        "* Tests local gradient storage plus supply-bounded or reference-bounded readout writes.",
        "* No behavioral sources or Python-updated state.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vdp dp 0 PULSE(0 {dp:.12g} 1.0n 10p 10p 2.0n 30n)",
        f"Vdn dn 0 PULSE(0 {dn:.12g} 1.0n 10p 10p 2.0n 30n)",
        "Vacc acc 0 PULSE(0 1.2 1.0n 10p 10p 2.0n 30n)",
        "Vapply apply 0 PULSE(0 1.2 4.0n 10p 10p 4.0n 30n)",
        "Vapplyn applyn 0 PULSE(1.2 0 4.0n 10p 10p 4.0n 30n)",
        f"Cvwp vwp0 0 20f IC={vwp:.12g}",
        f"Cvwn vwn0 0 20f IC={vwn:.12g}",
        "Cgvp gvp0 0 2f IC=0",
        "Cgvn gvn0 0 2f IC=0",
        "Crgp rgp0 0 4f IC=1.2",
        "Crgn rgn0 0 4f IC=1.2",
        "Rvwp vwp0 0 1e15",
        "Rvwn vwn0 0 1e15",
        "Rgvp gvp0 0 1G",
        "Rgvn gvn0 0 1G",
        "Rrgp rgp0 vdd 50k",
        "Rrgn rgn0 vdd 50k",
    ]
    gradient_gate_node = "act"
    if gradient_gate_topology == "direct":
        lines += [
            f"Vact act 0 PULSE(0 {gate_amplitude:.12g} 1.0n 10p 10p 2.0n 30n)",
        ]
    else:
        lines += [
            f"Velig elig 0 PULSE(0 {gate_amplitude:.12g} 0.5n 10p 10p 3.0n 30n)",
            "Cegon act 0 10f IC=0",
            "Cegate egate 0 4f IC=1.2",
            "Regon act 0 1G",
            "Regate egate vdd 50k",
            f"Megate_pd egate elig 0 0 NSENSE W={gate_restore_width:.6g}u L=180n",
            f"Megon_p act egate vdd vdd PMOS W={gate_restore_width:.6g}u L=180n",
        ]
    if topology == "bounded-ref":
        lines += [
            f"Vvwhi_ref vwhi_ref 0 {high_ref:.12g}",
            f"Vvwlo_ref vwlo_ref 0 {low_ref:.12g}",
        ]
    readout_pmos_w = 8.0 * update_scale
    readout_nmos_w = 2.0 * update_scale
    lines += [
        f"Mgvp0_a vdd {gradient_gate_node} gvp0_a 0 NSENSE W={gradient_width:.6g}u L=180n",
        f"Mgvp0_d gvp0_a dp gvp0_d 0 NSENSE W={gradient_width:.6g}u L=180n",
        f"Mgvp0_g gvp0_d acc gvp0 0 NREL W={gradient_width:.6g}u L=180n",
        f"Mgvn0_a vdd {gradient_gate_node} gvn0_a 0 NSENSE W={gradient_width:.6g}u L=180n",
        f"Mgvn0_d gvn0_a dn gvn0_d 0 NSENSE W={gradient_width:.6g}u L=180n",
        f"Mgvn0_g gvn0_d acc gvn0 0 NREL W={gradient_width:.6g}u L=180n",
        f"Mrgp0_pd rgp0 gvp0 0 0 NSENSE W={gradient_restore_width:.6g}u L=180n",
        f"Mrgn0_pd rgn0 gvn0 0 0 NSENSE W={gradient_restore_width:.6g}u L=180n",
        *readout_weight_update_lines(
            0,
            topology=topology,
            readout_pmos_w=readout_pmos_w,
            readout_nmos_w=readout_nmos_w,
        ),
        ".meas tran vwp_before FIND V(vwp0) AT=3.5n",
        ".meas tran vwn_before FIND V(vwn0) AT=3.5n",
        ".meas tran gate_before_apply FIND V(act) AT=3.5n",
        ".meas tran gvp_before_apply FIND V(gvp0) AT=3.5n",
        ".meas tran gvn_before_apply FIND V(gvn0) AT=3.5n",
        ".meas tran gradient_margin PARAM='gvp_before_apply-gvn_before_apply'",
        ".meas tran vwp_after FIND V(vwp0) AT=9.0n",
        ".meas tran vwn_after FIND V(vwn0) AT=9.0n",
        ".meas tran signed_before PARAM='vwp_before-vwn_before'",
        ".meas tran signed_after PARAM='vwp_after-vwn_after'",
        ".meas tran signed_delta PARAM='signed_after-signed_before'",
        ".meas tran common_before PARAM='0.5*(vwp_before+vwn_before)'",
        ".meas tran common_after PARAM='0.5*(vwp_after+vwn_after)'",
        ".meas tran common_delta PARAM='common_after-common_before'",
        ".tran 5p 12n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def classify_sign(actual: float, expected: float, *, min_abs_delta: float) -> str:
    if abs(expected) < 1e-15:
        return "dead_zone" if abs(actual) < min_abs_delta else "biased"
    if expected > 0.0 and actual >= min_abs_delta:
        return "aligned"
    if expected < 0.0 and actual <= -min_abs_delta:
        return "aligned"
    return "wrong_sign"


def classify_row(row: dict[str, Any], *, min_abs_delta: float, max_common_delta: float) -> dict[str, Any]:
    mode = str(row["update_mode"])
    expected = 1.0 if mode == "positive" else -1.0 if mode == "negative" else 0.0
    common_delta = abs(float(row.get("common_delta", 0.0)))
    return {
        "update_classification": classify_sign(float(row.get("signed_delta", 0.0)), expected, min_abs_delta=min_abs_delta),
        "common_mode_classification": "bounded" if common_delta <= max_common_delta else "shifted",
    }


def default_cases() -> list[dict[str, Any]]:
    return [
        {"case": "positive_update", "update_mode": "positive"},
        {"case": "negative_update", "update_mode": "negative"},
        {"case": "no_error_dead_zone", "update_mode": "none"},
    ]


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for case in default_cases():
        path = generated / f"{tag}_{sanitize_tag(str(case['case']))}.cir"
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(
                update_mode=str(case["update_mode"]),
                topology=args.topology,
                vwp=args.vwp,
                vwn=args.vwn,
                positive_ref=args.positive_ref,
                negative_ref=args.negative_ref,
                update_span=args.update_span,
                gradient_width=args.gradient_width,
                gradient_restore_width=args.gradient_restore_width,
                gradient_gate_topology=args.gradient_gate_topology,
                gate_amplitude=args.gate_amplitude,
                gate_restore_width=args.gate_restore_width,
                update_scale=args.update_scale,
                error_amplitude=args.error_amplitude,
            ),
            timeout=args.timeout,
        )
        row = {**case, **measures}
        row.update(classify_row(row, min_abs_delta=args.min_abs_delta, max_common_delta=args.max_common_delta))
        rows.append(row)
    csv_path = tables / f"{tag}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    classification_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in ["update_classification", "common_mode_classification"]:
            classification_counts.setdefault(key, {})
            cls = str(row[key])
            classification_counts[key][cls] = classification_counts[key].get(cls, 0) + 1
    passed = all(
        str(row["update_classification"]) in {"aligned", "dead_zone"}
        and str(row["common_mode_classification"]) == "bounded"
        for row in rows
    )
    summary = {
        "simulator": version,
        "architecture": f"{args.topology}_readout_writer_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "cases": len(rows),
        "passed": passed,
        "classification_counts": classification_counts,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="readout_writer_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--topology", choices=READOUT_WRITER_TOPOLOGIES, default="bounded-ref")
    ap.add_argument("--vwp", type=float, default=0.36)
    ap.add_argument("--vwn", type=float, default=0.34)
    ap.add_argument("--positive-ref", type=float, default=0.36)
    ap.add_argument("--negative-ref", type=float, default=0.34)
    ap.add_argument("--update-span", type=float, default=0.15)
    ap.add_argument("--gradient-width", type=float, default=24.0)
    ap.add_argument("--gradient-restore-width", type=float, default=16.0)
    ap.add_argument("--gradient-gate-topology", choices=GRADIENT_GATE_TOPOLOGIES, default="direct")
    ap.add_argument("--gate-amplitude", type=float, default=1.2)
    ap.add_argument("--gate-restore-width", type=float, default=7.0)
    ap.add_argument("--update-scale", type=float, default=0.10)
    ap.add_argument("--error-amplitude", type=float, default=1.2)
    ap.add_argument("--min-abs-delta", type=float, default=1e-3)
    ap.add_argument("--max-common-delta", type=float, default=50e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    for name in [
        "gradient_width",
        "gradient_restore_width",
        "gate_amplitude",
        "gate_restore_width",
        "update_scale",
        "error_amplitude",
    ]:
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.gate_amplitude > 1.2:
        raise ValueError("gate-amplitude must not exceed VDD")
    if args.error_amplitude > 1.2:
        raise ValueError("error-amplitude must not exceed VDD")
    if args.update_span < 0.0:
        raise ValueError("update-span must be nonnegative")
    if args.min_abs_delta < 0.0:
        raise ValueError("min-abs-delta must be nonnegative")
    if args.max_common_delta < 0.0:
        raise ValueError("max-common-delta must be nonnegative")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_cases(args), indent=2))


if __name__ == "__main__":
    main()
