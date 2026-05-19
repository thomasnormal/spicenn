from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_device_multicell_classifier import mos_models
from run_spice_sweep import ROOT, detect_spice, run_tiny_test


MEAS_RE = re.compile(r"(?im)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")
VDD = 1.2


def parse_measures(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in MEAS_RE.findall(text)}


def common_header() -> str:
    return f"""
* Readout MOS mobility characterization.
* Uses the same LEVEL=1 MOS models and read/write stack topology as the
* direct-flow random-hidden runner.
.param VDD={VDD:.12g}
{mos_models()}
Vdd vdd 0 {{VDD}}
.options method=gear maxord=2
""".strip()


def read_branch_netlist(theta: float, act: float, branch: str, score_ic: float = 0.30) -> str:
    if branch not in {"positive", "negative"}:
        raise ValueError(f"unknown read branch: {branch}")
    if branch == "positive":
        devices = [
            "Mpos_a vdd act pos0 0 NSENSE W=56u L=180n",
            "Mpos_w pos0 w pos1 0 NREL W=56u L=180n",
            "Mpos_f pos1 fwd score 0 NREL W=56u L=180n",
        ]
        score_init = 0.0
        response_expr = "score_final"
    else:
        devices = [
            "Mneg_f score fwd neg0 0 NREL W=48u L=180n",
            "Mneg_a neg0 act neg1 0 NSENSE W=48u L=180n",
            "Mneg_w neg1 w 0 0 NREL W=48u L=180n",
        ]
        score_init = score_ic
        response_expr = "score_initial-score_final"
    return f"""
{common_header()}
Vact act 0 PULSE(0 {act:.12g} 0.75n 20p 20p 3n 8n)
Vfwd fwd 0 PULSE(0 {{VDD}} 0.75n 20p 20p 3n 8n)
Cw w 0 20f IC={theta:.12g}
Rw w 0 1e15
Cscore score 0 10f IC={score_init:.12g}
Rscore score 0 1G
{chr(10).join(devices)}
.tran 5p 4.5n uic
.meas tran score_initial FIND V(score) AT=0.60n
.meas tran score_final FIND V(score) AT=3.75n
.meas tran read_response PARAM='{response_expr}'
.meas tran w_final FIND V(w) AT=3.75n
.control
run
print score_initial score_final read_response w_final
.endc
.end
""".lstrip()


def write_mobility_netlist(theta: float, pre: float, delta: float, width_u: float) -> str:
    return f"""
{common_header()}
Vbwd bwd 0 PULSE(0 {{VDD}} 1.00n 20p 20p 2.0n 8n)
Vpre pre 0 PULSE(0 {pre:.12g} 1.00n 20p 20p 2.0n 8n)
Vdelta delta 0 PULSE(0 {delta:.12g} 1.00n 20p 20p 2.0n 8n)
Cw w 0 20f IC={theta:.12g}
Rw w 0 1e15
Mflow_b w bwd flow_b 0 NREL W={width_u:.12g}u L=180n
Mflow_a flow_b pre flow_a 0 NREL W={width_u:.12g}u L=180n
Mflow_d flow_a delta 0 0 NSENSE W={width_u:.12g}u L=180n
Rflow_b flow_b 0 1G
Rflow_a flow_a 0 1G
Cflow_b flow_b 0 0.02f IC=0
Cflow_a flow_a 0 0.02f IC=0
.tran 5p 4.5n uic
.meas tran w_before FIND V(w) AT=0.80n
.meas tran w_after FIND V(w) AT=3.75n
.meas tran discharge PARAM='w_before-w_after'
.control
run
print w_before w_after discharge
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float) -> dict[str, float]:
    path.write_text(netlist)
    cmd = [spice_bin, "-b", str(path)] if "ngspice" in Path(spice_bin).name.lower() else [spice_bin, str(path)]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("ngspice produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return measures


def add_slopes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["read_slope"] = np.nan
    for (experiment, act), idx in out.groupby(["experiment", "act"], dropna=False).groups.items():
        if not str(experiment).startswith("read_"):
            continue
        ordered_idx = list(idx)
        sub = out.loc[ordered_idx].sort_values("theta")
        slopes = np.gradient(sub["read_response"].to_numpy(), sub["theta"].to_numpy())
        out.loc[sub.index, "read_slope"] = slopes
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_mobility_sweep")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--write-width-u", type=float, default=0.02)
    ap.add_argument("--pre", type=float, default=0.65)
    ap.add_argument("--delta", type=float, default=1.0)
    args = ap.parse_args()

    spice_bin, version = detect_spice(None)
    generated = ROOT / "spice/generated"
    results = ROOT / "spice/results"
    tables = ROOT / "results/tables"
    for directory in [generated, results, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    theta_values = [0.05, 0.10, 0.16, 0.24, 0.34, 0.46, 0.58, 0.70, 0.82, 0.94, 1.06, 1.15]
    act_values = [0.25, 0.50, 0.75]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for act in act_values:
        for branch in ["positive", "negative"]:
            for theta in theta_values:
                measures = run_netlist(
                    spice_bin,
                    generated / f"{safe_tag}_read_{branch}_a{act:.2f}_t{theta:.2f}.cir",
                    read_branch_netlist(theta, act, branch),
                    args.timeout,
                )
                rows.append(
                    {
                        "experiment": f"read_{branch}",
                        "theta": theta,
                        "act": act,
                        "pre": None,
                        "delta": None,
                        "write_width_u": None,
                        **measures,
                    }
                )

    for theta in theta_values:
        measures = run_netlist(
            spice_bin,
            generated / f"{safe_tag}_write_t{theta:.2f}.cir",
            write_mobility_netlist(theta, args.pre, args.delta, args.write_width_u),
            args.timeout,
        )
        rows.append(
            {
                "experiment": "write_discharge",
                "theta": theta,
                "act": None,
                "pre": args.pre,
                "delta": args.delta,
                "write_width_u": args.write_width_u,
                **measures,
            }
        )

    df = add_slopes(pd.DataFrame(rows))
    write = df[df["experiment"] == "write_discharge"][["theta", "discharge"]].rename(
        columns={"discharge": "write_discharge_v"}
    )
    df = df.merge(write, on="theta", how="left")
    df["effective_mobility"] = df["read_slope"] * df["write_discharge_v"]

    csv_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(table_path, index=False)

    read_rows = df[df["experiment"].str.startswith("read_")]
    write_rows = df[df["experiment"] == "write_discharge"]
    summary = {
        "tag": safe_tag,
        "simulator": version,
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "csv": str(csv_path),
        "table_csv": str(table_path),
        "theta_values": theta_values,
        "act_values": act_values,
        "write_width_u": args.write_width_u,
        "write_pre_v": args.pre,
        "write_delta_v": args.delta,
        "min_read_slope": float(read_rows["read_slope"].min()),
        "max_read_slope": float(read_rows["read_slope"].max()),
        "min_write_discharge_v": float(write_rows["discharge"].min()),
        "max_write_discharge_v": float(write_rows["discharge"].max()),
        "min_effective_mobility": float(read_rows["effective_mobility"].min()),
        "max_effective_mobility": float(read_rows["effective_mobility"].max()),
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "Read-response slopes approximate G_eff'(theta); write discharge approximates the natural "
            "direct-flow state mobility s_MOS(theta). Their product is the effective learning mobility "
            "seen by the signed readout branch."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
