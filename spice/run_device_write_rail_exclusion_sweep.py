from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from run_device_multicell_classifier import mos_models, pwl
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test
from _util import MEAS_RE, parse_measures


SELECTOR_MODES = ["pmos_inhibit", "diffpair_bleed"]


def write_rail_exclusion_netlist(
    dp_v: float,
    dn_v: float,
    width_u: float,
    rail_cap_f: float = 0.1,
    overlap_decay: bool = False,
    selector_mode: str = "pmos_inhibit",
) -> str:
    if selector_mode not in SELECTOR_MODES:
        raise ValueError(f"unknown write-rail selector mode: {selector_mode}")
    overlap_block = ""
    overlap_measures = ""
    overlap_prints = ""
    if selector_mode == "diffpair_bleed":
        tail_width_u = max(width_u * 0.5, 1e-9)
        bleed_width_u = max(width_u * 0.025, 1e-9)
        selector_block = f"""
Cposbar posbar 0 0.05f IC=1.2
Cnegbar negbar 0 0.05f IC=1.2
Rposbar posbar vdd 1G
Rnegbar negbar vdd 1G
Mposbar_load posbar posbar vdd vdd PMOS W={width_u:.12g}u L=180n
Mnegbar_load negbar negbar vdd vdd PMOS W={width_u:.12g}u L=180n
Mposbar_sel posbar dp sel_src 0 NSENSE W={width_u:.12g}u L=180n
Mnegbar_sel negbar dn sel_src 0 NSENSE W={width_u:.12g}u L=180n
Msel_tail sel_src bwd 0 0 NMOS W={tail_width_u:.12g}u L=180n
Mrwpos_bleed rwpos bwd 0 0 NMOS W={bleed_width_u:.12g}u L=180n
Mrwneg_bleed rwneg bwd 0 0 NMOS W={bleed_width_u:.12g}u L=180n
Mrwpos_p vdd posbar posmid vdd PMOS W={width_u:.12g}u L=180n
Mrwpos_n posmid negbar rwpos 0 NMOS W={width_u:.12g}u L=180n
Mrwneg_p vdd negbar negmid vdd PMOS W={width_u:.12g}u L=180n
Mrwneg_n negmid posbar rwneg 0 NMOS W={width_u:.12g}u L=180n
Cposmid posmid 0 0.02f IC=0
Cnegmid negmid 0 0.02f IC=0
Rposmid posmid 0 1G
Rnegmid negmid 0 1G
Csrc sel_src 0 0.02f IC=0
Rsrc sel_src 0 1G
"""
    else:
        selector_block = f"""
Mrwpos_inh vdd dn rwpos_src vdd PMOS W={width_u:.12g}u L=180n
Mrwpos_gate rwpos_src dp rwpos 0 NSENSE W={width_u:.12g}u L=180n
Mrwneg_inh vdd dp rwneg_src vdd PMOS W={width_u:.12g}u L=180n
Mrwneg_gate rwneg_src dn rwneg 0 NSENSE W={width_u:.12g}u L=180n
Mrwpos_kill rwpos dn 0 0 NMOS W={width_u:.12g}u L=180n
Mrwneg_kill rwneg dp 0 0 NMOS W={width_u:.12g}u L=180n
Csrcp rwpos_src 0 0.02f IC=0
Csrcn rwneg_src 0 0.02f IC=0
Rsrcp rwpos_src 0 1G
Rsrcn rwneg_src 0 1G
"""
    if overlap_decay and selector_mode == "pmos_inhibit":
        overlap_block = f"""
Crwov rwov 0 {rail_cap_f:.12g}f IC=0
Rrwov rwov 0 1G
Mreset_rwov rwov rste 0 0 NMOS W=4u L=180n
Mrwov_p vdd dp rwov_mid 0 NMOS W={width_u:.12g}u L=180n
Mrwov_n rwov_mid dn rwov 0 NMOS W={width_u:.12g}u L=180n
Covmid rwov_mid 0 0.02f IC=0
Rovmid rwov_mid 0 1G
"""
        overlap_measures = """
.meas tran ov_mid FIND V(rwov) AT=2.0n
.meas tran ov_late FIND V(rwov) AT=4.8n
.meas tran ov_after_off FIND V(rwov) AT=7.5n
"""
        overlap_prints = " ov_mid ov_late ov_after_off"
    return f"""
* Isolated mutually inhibited signed write-rail generator.
* Positive rail approximates dp AND (not dn); negative rail approximates dn AND (not dp).
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vrste rste 0 {pwl([(0.0, 1.1), (0.45, 1.1), (0.50, 0.0), (8.0, 0.0)])}
Vbwd bwd 0 {pwl([(0.0, 0.0), (0.50, 0.0), (0.55, 1.1), (5.00, 1.1), (5.05, 0.0), (8.0, 0.0)])}
Vdp dp 0 {pwl([(0.0, 0.0), (0.50, 0.0), (0.55, dp_v), (5.00, dp_v), (5.05, 0.0), (8.0, 0.0)])}
Vdn dn 0 {pwl([(0.0, 0.0), (0.50, 0.0), (0.55, dn_v), (5.00, dn_v), (5.05, 0.0), (8.0, 0.0)])}

Crwpos rwpos 0 {rail_cap_f:.12g}f IC=0
Crwneg rwneg 0 {rail_cap_f:.12g}f IC=0
Rrwpos rwpos 0 1G
Rrwneg rwneg 0 1G
Mreset_rwpos rwpos rste 0 0 NMOS W=4u L=180n
Mreset_rwneg rwneg rste 0 0 NMOS W=4u L=180n
{selector_block}
{overlap_block}

.options method=gear maxord=2
.tran 5p 8n uic
.meas tran pos_mid FIND V(rwpos) AT=2.0n
.meas tran neg_mid FIND V(rwneg) AT=2.0n
.meas tran pos_late FIND V(rwpos) AT=4.8n
.meas tran neg_late FIND V(rwneg) AT=4.8n
.meas tran pos_after_off FIND V(rwpos) AT=7.5n
.meas tran neg_after_off FIND V(rwneg) AT=7.5n
{overlap_measures}
.control
run
print pos_mid neg_mid pos_late neg_late pos_after_off neg_after_off{overlap_prints}
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return measures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--tag", default="device_write_rail_exclusion")
    ap.add_argument("--rail-cap-f", type=float, default=0.1)
    ap.add_argument("--overlap-decay", action="store_true")
    ap.add_argument("--mode", choices=SELECTOR_MODES, default="pmos_inhibit")
    args = ap.parse_args()
    if args.rail_cap_f <= 0:
        raise SystemExit("--rail-cap-f must be positive.")

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    cases = [
        ("positive_only", 1.1, 0.2),
        ("negative_only", 0.2, 1.1),
        ("balanced_high", 1.1, 1.1),
        ("balanced_low", 0.2, 0.2),
    ]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for width_u in [2.0, 4.0, 8.0, 16.0, 32.0]:
        for name, dp_v, dn_v in cases:
            measures = run_netlist(
                spice_bin,
                generated / f"{safe_tag}_{name}_{width_u:g}.cir",
                write_rail_exclusion_netlist(
                    dp_v,
                    dn_v,
                    width_u,
                    args.rail_cap_f,
                    args.overlap_decay,
                    args.mode,
                ),
                args.timeout,
            )
            rows.append(
                {
                    "case": name,
                    "dp_v": dp_v,
                    "dn_v": dn_v,
                    "width_u": width_u,
                    "rail_cap_f": args.rail_cap_f,
                    "selector_mode": args.mode,
                    **measures,
                }
            )

    df = pd.DataFrame(rows)
    curve_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(curve_path, index=False)
    df.to_csv(table_path, index=False)

    positive = df[df["case"] == "positive_only"]
    negative = df[df["case"] == "negative_only"]
    conflict = df[df["case"] == "balanced_high"]
    summary = {
        "simulator": version,
        "architecture": "mutually_inhibited_write_rail_generator",
        "selector_mode": args.mode,
        "overlap_decay": args.overlap_decay,
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "rail_cap_f": args.rail_cap_f,
        "curve": str(curve_path),
        "table_curve": str(table_path),
        "rows": len(df),
        "min_positive_only_pos_late_v": float(positive["pos_late"].min()),
        "max_positive_only_neg_late_v": float(positive["neg_late"].max()),
        "min_negative_only_neg_late_v": float(negative["neg_late"].min()),
        "max_negative_only_pos_late_v": float(negative["pos_late"].max()),
        "max_conflict_pos_late_v": float(conflict["pos_late"].max()),
        "max_conflict_neg_late_v": float(conflict["neg_late"].max()),
        "wall_time_s": time.perf_counter() - t0,
    }
    if args.overlap_decay and args.mode == "pmos_inhibit":
        summary.update(
            {
                "min_conflict_overlap_late_v": float(conflict["ov_late"].min()),
                "max_positive_only_overlap_late_v": float(positive["ov_late"].max()),
                "max_negative_only_overlap_late_v": float(negative["ov_late"].max()),
            }
        )
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
