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
WRITE_MODES = [
    "discharge",
    "bounded_discharge",
    "charge_only",
    "bounded_charge_only",
    "charge_discharge",
    "bounded_charge_discharge",
]


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


def read_branch_netlist(
    theta: float,
    act: float,
    branch: str,
    score_ic: float = 0.30,
    pos_width_u: float = 56.0,
    neg_width_u: float = 48.0,
    score_cap_f: float = 10.0,
) -> str:
    if branch not in {"positive", "negative"}:
        raise ValueError(f"unknown read branch: {branch}")
    if branch == "positive":
        devices = [
            f"Mpos_a vdd act pos0 0 NSENSE W={pos_width_u:.12g}u L=180n",
            f"Mpos_w pos0 w pos1 0 NREL W={pos_width_u:.12g}u L=180n",
            f"Mpos_f pos1 fwd score 0 NREL W={pos_width_u:.12g}u L=180n",
        ]
        score_init = 0.0
        response_expr = "score_final"
    else:
        devices = [
            f"Mneg_f score fwd neg0 0 NREL W={neg_width_u:.12g}u L=180n",
            f"Mneg_a neg0 act neg1 0 NSENSE W={neg_width_u:.12g}u L=180n",
            f"Mneg_w neg1 w 0 0 NREL W={neg_width_u:.12g}u L=180n",
        ]
        score_init = score_ic
        response_expr = "score_initial-score_final"
    return f"""
{common_header()}
Vact act 0 PULSE(0 {act:.12g} 0.75n 20p 20p 3n 8n)
Vfwd fwd 0 PULSE(0 {{VDD}} 0.75n 20p 20p 3n 8n)
Cw w 0 20f IC={theta:.12g}
Rw w 0 1e15
Cscore score 0 {score_cap_f:.12g}f IC={score_init:.12g}
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


def write_actions(write_mode: str) -> tuple[bool, bool]:
    if write_mode not in WRITE_MODES:
        raise ValueError(f"unknown write mode: {write_mode}")
    discharge_enabled = write_mode in {
        "discharge",
        "bounded_discharge",
        "charge_discharge",
        "bounded_charge_discharge",
    }
    charge_enabled = write_mode in {
        "charge_only",
        "bounded_charge_only",
        "charge_discharge",
        "bounded_charge_discharge",
    }
    return discharge_enabled, charge_enabled


def write_mobility_netlist(
    theta: float,
    pre: float,
    delta: float,
    width_u: float,
    action: str,
    write_mode: str,
    write_high_v: float,
    write_low_v: float,
) -> str:
    if action not in {"discharge", "charge"}:
        raise ValueError(f"unknown write action: {action}")
    bounded_write = write_mode.startswith("bounded_")
    high_node = "whigh" if bounded_write else "vdd"
    low_node = "wlow" if bounded_write else "0"
    write_rails = (
        f"Vwhigh whigh 0 {write_high_v:.12g}\nVwlow wlow 0 {write_low_v:.12g}"
        if bounded_write
        else ""
    )
    if action == "discharge":
        devices = [
            f"Mflow_b w bwd flow_b 0 NREL W={width_u:.12g}u L=180n",
            f"Mflow_a flow_b pre flow_a 0 NREL W={width_u:.12g}u L=180n",
            f"Mflow_d flow_a delta {low_node} 0 NSENSE W={width_u:.12g}u L=180n",
        ]
        state_delta_expr = "w_before-w_after"
    else:
        devices = [
            f"Mflow_b {high_node} bwd flow_b 0 NREL W={width_u:.12g}u L=180n",
            f"Mflow_a flow_b pre flow_a 0 NREL W={width_u:.12g}u L=180n",
            f"Mflow_d flow_a delta w 0 NSENSE W={width_u:.12g}u L=180n",
        ]
        state_delta_expr = "w_after-w_before"
    return f"""
{common_header()}
{write_rails}
Vbwd bwd 0 PULSE(0 {{VDD}} 1.00n 20p 20p 2.0n 8n)
Vpre pre 0 PULSE(0 {pre:.12g} 1.00n 20p 20p 2.0n 8n)
Vdelta delta 0 PULSE(0 {delta:.12g} 1.00n 20p 20p 2.0n 8n)
Cw w 0 20f IC={theta:.12g}
Rw w 0 1e15
{chr(10).join(devices)}
Rflow_b flow_b 0 1G
Rflow_a flow_a 0 1G
Cflow_b flow_b 0 0.02f IC=0
Cflow_a flow_a 0 0.02f IC=0
.tran 5p 4.5n uic
.meas tran w_before FIND V(w) AT=0.80n
.meas tran w_after FIND V(w) AT=3.75n
.meas tran discharge PARAM='w_before-w_after'
.meas tran charge PARAM='w_after-w_before'
.meas tran state_delta_v PARAM='{state_delta_expr}'
.meas tran signed_state_delta_v PARAM='w_after-w_before'
.control
run
print w_before w_after state_delta_v signed_state_delta_v
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
    out["read_gain"] = np.nan
    read_mask = out["experiment"].astype(str).str.startswith("read_") & out["act"].notna()
    out.loc[read_mask, "read_gain"] = out.loc[read_mask, "read_response"] / out.loc[read_mask, "act"]
    for (experiment, act), idx in out.groupby(["experiment", "act"], dropna=False).groups.items():
        if not str(experiment).startswith("read_"):
            continue
        ordered_idx = list(idx)
        sub = out.loc[ordered_idx].sort_values("theta")
        slopes = np.gradient(sub["read_response"].to_numpy(), sub["theta"].to_numpy())
        out.loc[sub.index, "read_slope"] = slopes
    return out


def signed_mobility_table(df: pd.DataFrame, write_mode: str, write_low_v: float, write_high_v: float) -> pd.DataFrame:
    read_pos = (
        df[df["experiment"] == "read_positive"][["theta", "act", "read_gain", "read_slope"]]
        .rename(columns={"read_gain": "read_positive_gain", "read_slope": "read_positive_slope"})
        .copy()
    )
    read_neg = (
        df[df["experiment"] == "read_negative"][["theta", "act", "read_gain", "read_slope"]]
        .rename(columns={"read_gain": "read_negative_gain", "read_slope": "read_negative_slope"})
        .copy()
    )
    signed = read_pos.merge(read_neg, on=["theta", "act"], how="inner")
    signed["write_mode"] = write_mode
    signed["write_low_v"] = write_low_v if write_mode.startswith("bounded_") else np.nan
    signed["write_high_v"] = write_high_v if write_mode.startswith("bounded_") else np.nan

    signed["write_discharge_v"] = 0.0
    signed["write_charge_v"] = 0.0
    discharge = df[df["experiment"] == "write_discharge"][["theta", "state_delta_v"]].rename(
        columns={"state_delta_v": "write_discharge_v"}
    )
    charge = df[df["experiment"] == "write_charge"][["theta", "state_delta_v"]].rename(
        columns={"state_delta_v": "write_charge_v"}
    )
    if not discharge.empty:
        signed = signed.drop(columns=["write_discharge_v"]).merge(discharge, on="theta", how="left")
    if not charge.empty:
        signed = signed.drop(columns=["write_charge_v"]).merge(charge, on="theta", how="left")
    signed["write_discharge_v"] = signed["write_discharge_v"].fillna(0.0)
    signed["write_charge_v"] = signed["write_charge_v"].fillna(0.0)

    # For normal-polarity signed readout:
    #   positive desired update discharges the negative branch and/or charges the positive branch;
    #   negative desired update discharges the positive branch and/or charges the negative branch.
    signed["signed_increase_mobility"] = (
        signed["read_negative_slope"] * signed["write_discharge_v"]
        + signed["read_positive_slope"] * signed["write_charge_v"]
    )
    signed["signed_decrease_mobility"] = (
        signed["read_positive_slope"] * signed["write_discharge_v"]
        + signed["read_negative_slope"] * signed["write_charge_v"]
    )
    # If a write primitive also multiplies its state update by G_eff'(theta),
    # the induced effective-weight mobility gains another slope factor.  These
    # columns are a diagnostic for physical-state-gradient-style writes; the raw
    # columns above remain the actual measured hardware update.
    signed["physical_gradient_increase_mobility"] = (
        signed["read_negative_slope"] ** 2 * signed["write_discharge_v"]
        + signed["read_positive_slope"] ** 2 * signed["write_charge_v"]
    )
    signed["physical_gradient_decrease_mobility"] = (
        signed["read_positive_slope"] ** 2 * signed["write_discharge_v"]
        + signed["read_negative_slope"] ** 2 * signed["write_charge_v"]
    )
    signed["signed_update_sign_aligned"] = (
        (signed["signed_increase_mobility"] > 0) & (signed["signed_decrease_mobility"] > 0)
    )
    signed["physical_gradient_sign_aligned"] = (
        (signed["physical_gradient_increase_mobility"] > 0)
        & (signed["physical_gradient_decrease_mobility"] > 0)
    )
    max_mobility = signed[["signed_increase_mobility", "signed_decrease_mobility"]].max(axis=1)
    min_mobility = signed[["signed_increase_mobility", "signed_decrease_mobility"]].min(axis=1)
    signed["signed_mobility_balance"] = np.where(max_mobility > 0, min_mobility / max_mobility, np.nan)
    physical_max = signed[
        ["physical_gradient_increase_mobility", "physical_gradient_decrease_mobility"]
    ].max(axis=1)
    physical_min = signed[
        ["physical_gradient_increase_mobility", "physical_gradient_decrease_mobility"]
    ].min(axis=1)
    signed["physical_gradient_mobility_balance"] = np.where(
        physical_max > 0,
        physical_min / physical_max,
        np.nan,
    )
    return signed


def _branch_read_table(df: pd.DataFrame, experiment: str, suffix: str) -> pd.DataFrame:
    return (
        df[df["experiment"] == experiment][["theta", "act", "read_gain", "read_slope"]]
        .rename(
            columns={
                "theta": f"theta_{suffix}",
                "read_gain": f"read_{suffix}_gain",
                "read_slope": f"read_{suffix}_slope",
            }
        )
        .copy()
    )


def _branch_write_table(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = pd.DataFrame({f"theta_{suffix}": sorted(df["theta"].dropna().unique())})
    out[f"write_{suffix}_discharge_v"] = 0.0
    out[f"write_{suffix}_charge_v"] = 0.0
    discharge = df[df["experiment"] == "write_discharge"][["theta", "state_delta_v"]].rename(
        columns={"theta": f"theta_{suffix}", "state_delta_v": f"write_{suffix}_discharge_v"}
    )
    charge = df[df["experiment"] == "write_charge"][["theta", "state_delta_v"]].rename(
        columns={"theta": f"theta_{suffix}", "state_delta_v": f"write_{suffix}_charge_v"}
    )
    if not discharge.empty:
        out = out.drop(columns=[f"write_{suffix}_discharge_v"]).merge(discharge, on=f"theta_{suffix}", how="left")
    if not charge.empty:
        out = out.drop(columns=[f"write_{suffix}_charge_v"]).merge(charge, on=f"theta_{suffix}", how="left")
    out[f"write_{suffix}_discharge_v"] = out[f"write_{suffix}_discharge_v"].fillna(0.0)
    out[f"write_{suffix}_charge_v"] = out[f"write_{suffix}_charge_v"].fillna(0.0)
    return out


def branch_pair_signed_mobility_table(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    write_mode: str,
) -> pd.DataFrame:
    """Compute signed readout mobility for independently biased p/n branches.

    signed_mobility_table is the same-theta diagnostic.  This table is the
    branch-specific diagnostic: the positive branch can sit at theta_p while
    the negative branch sits at theta_n.  That is the relevant case for the
    branch-range experiments where the two MOS branches use different storage
    windows.
    """
    discharge_enabled, charge_enabled = write_actions(write_mode)
    pos = _branch_read_table(positive_df, "read_positive", "p").merge(
        _branch_write_table(positive_df, "p"),
        on="theta_p",
        how="left",
    )
    neg = _branch_read_table(negative_df, "read_negative", "n").merge(
        _branch_write_table(negative_df, "n"),
        on="theta_n",
        how="left",
    )
    if not discharge_enabled:
        pos["write_p_discharge_v"] = 0.0
        neg["write_n_discharge_v"] = 0.0
    if not charge_enabled:
        pos["write_p_charge_v"] = 0.0
        neg["write_n_charge_v"] = 0.0
    pair = pos.assign(_pair_key=1).merge(neg.assign(_pair_key=1), on=["act", "_pair_key"]).drop(columns=["_pair_key"])
    pair["write_mode"] = write_mode
    pair["signed_read_gain"] = pair["read_p_gain"] - pair["read_n_gain"]
    pair["signed_increase_mobility"] = (
        pair["read_n_slope"] * pair["write_n_discharge_v"]
        + pair["read_p_slope"] * pair["write_p_charge_v"]
    )
    pair["signed_decrease_mobility"] = (
        pair["read_p_slope"] * pair["write_p_discharge_v"]
        + pair["read_n_slope"] * pair["write_n_charge_v"]
    )
    pair["physical_gradient_increase_mobility"] = (
        pair["read_n_slope"] ** 2 * pair["write_n_discharge_v"]
        + pair["read_p_slope"] ** 2 * pair["write_p_charge_v"]
    )
    pair["physical_gradient_decrease_mobility"] = (
        pair["read_p_slope"] ** 2 * pair["write_p_discharge_v"]
        + pair["read_n_slope"] ** 2 * pair["write_n_charge_v"]
    )
    pair["signed_update_sign_aligned"] = (
        (pair["signed_increase_mobility"] > 0) & (pair["signed_decrease_mobility"] > 0)
    )
    pair["physical_gradient_sign_aligned"] = (
        (pair["physical_gradient_increase_mobility"] > 0)
        & (pair["physical_gradient_decrease_mobility"] > 0)
    )
    max_mobility = pair[["signed_increase_mobility", "signed_decrease_mobility"]].max(axis=1)
    min_mobility = pair[["signed_increase_mobility", "signed_decrease_mobility"]].min(axis=1)
    pair["signed_mobility_balance"] = np.where(max_mobility > 0, min_mobility / max_mobility, np.nan)
    physical_max = pair[
        ["physical_gradient_increase_mobility", "physical_gradient_decrease_mobility"]
    ].max(axis=1)
    physical_min = pair[
        ["physical_gradient_increase_mobility", "physical_gradient_decrease_mobility"]
    ].min(axis=1)
    pair["physical_gradient_mobility_balance"] = np.where(
        physical_max > 0,
        physical_min / physical_max,
        np.nan,
    )
    return pair


def summarize_signed_mobility(signed: pd.DataFrame, write_mode: str, write_low_v: float, write_high_v: float) -> dict[str, Any]:
    operating = signed
    if write_mode.startswith("bounded_"):
        operating = signed[(signed["theta"] >= write_low_v) & (signed["theta"] <= write_high_v)]
    if operating.empty:
        operating = signed
    near_zero = 1e-9
    return {
        "signed_mobility_csv": None,
        "operating_theta_min_v": write_low_v if write_mode.startswith("bounded_") else float(signed["theta"].min()),
        "operating_theta_max_v": write_high_v if write_mode.startswith("bounded_") else float(signed["theta"].max()),
        "signed_mobility_rows": int(len(signed)),
        "operating_signed_mobility_rows": int(len(operating)),
        "signed_update_sign_aligned_fraction": float(operating["signed_update_sign_aligned"].mean()),
        "min_signed_increase_mobility": float(operating["signed_increase_mobility"].min()),
        "max_signed_increase_mobility": float(operating["signed_increase_mobility"].max()),
        "min_signed_decrease_mobility": float(operating["signed_decrease_mobility"].min()),
        "max_signed_decrease_mobility": float(operating["signed_decrease_mobility"].max()),
        "min_signed_mobility_balance": float(operating["signed_mobility_balance"].min(skipna=True)),
        "physical_gradient_sign_aligned_fraction": float(operating["physical_gradient_sign_aligned"].mean()),
        "min_physical_gradient_increase_mobility": float(
            operating["physical_gradient_increase_mobility"].min()
        ),
        "max_physical_gradient_increase_mobility": float(
            operating["physical_gradient_increase_mobility"].max()
        ),
        "min_physical_gradient_decrease_mobility": float(
            operating["physical_gradient_decrease_mobility"].min()
        ),
        "max_physical_gradient_decrease_mobility": float(
            operating["physical_gradient_decrease_mobility"].max()
        ),
        "min_physical_gradient_mobility_balance": float(
            operating["physical_gradient_mobility_balance"].min(skipna=True)
        ),
        "near_zero_signed_increase_count": int((operating["signed_increase_mobility"].abs() < near_zero).sum()),
        "near_zero_signed_decrease_count": int((operating["signed_decrease_mobility"].abs() < near_zero).sum()),
    }


def summarize_branch_pair_mobility(
    pair: pd.DataFrame,
    pos_low_v: float,
    pos_high_v: float,
    neg_low_v: float,
    neg_high_v: float,
    summary_act_v: float | None = None,
) -> dict[str, Any]:
    operating_all_act = pair[
        (pair["theta_p"] >= pos_low_v)
        & (pair["theta_p"] <= pos_high_v)
        & (pair["theta_n"] >= neg_low_v)
        & (pair["theta_n"] <= neg_high_v)
    ]
    if operating_all_act.empty:
        operating_all_act = pair
    operating = operating_all_act
    if summary_act_v is not None:
        at_summary_act = operating_all_act[np.isclose(operating_all_act["act"], summary_act_v)]
        if not at_summary_act.empty:
            operating = at_summary_act
    if operating.empty:
        operating = pair
    aligned = operating[operating["signed_update_sign_aligned"]]
    best_source = aligned if not aligned.empty else operating
    best = best_source.sort_values(
        ["signed_mobility_balance", "signed_increase_mobility", "signed_decrease_mobility"],
        ascending=[False, False, False],
    ).iloc[0]
    gain_safe = (
        operating_all_act.groupby(["theta_p", "theta_n"], as_index=False)
        .agg(
            min_signed_read_gain=("signed_read_gain", "min"),
            mean_signed_read_gain=("signed_read_gain", "mean"),
            min_signed_mobility_balance=("signed_mobility_balance", "min"),
            mean_signed_mobility_balance=("signed_mobility_balance", "mean"),
            aligned_fraction=("signed_update_sign_aligned", "mean"),
            min_signed_increase_mobility=("signed_increase_mobility", "min"),
            min_signed_decrease_mobility=("signed_decrease_mobility", "min"),
            min_physical_gradient_mobility_balance=("physical_gradient_mobility_balance", "min"),
            physical_gradient_aligned_fraction=("physical_gradient_sign_aligned", "mean"),
        )
        .copy()
    )
    gain_safe["gain_sign_safe"] = gain_safe["min_signed_read_gain"] > 0
    gain_safe["mobility_sign_safe"] = gain_safe["aligned_fraction"] >= 1.0
    fully_safe = gain_safe[gain_safe["gain_sign_safe"] & gain_safe["mobility_sign_safe"]]
    gain_safe_source = fully_safe if not fully_safe.empty else gain_safe
    best_gain_safe = gain_safe_source.sort_values(
        [
            "gain_sign_safe",
            "mobility_sign_safe",
            "min_signed_read_gain",
            "min_signed_mobility_balance",
            "mean_signed_read_gain",
        ],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return {
        "branch_pair_mobility_csv": None,
        "branch_pair_mobility_table_csv": None,
        "branch_pair_summary_act_v": summary_act_v,
        "branch_pair_pos_operating_theta_min_v": pos_low_v,
        "branch_pair_pos_operating_theta_max_v": pos_high_v,
        "branch_pair_neg_operating_theta_min_v": neg_low_v,
        "branch_pair_neg_operating_theta_max_v": neg_high_v,
        "branch_pair_rows": int(len(pair)),
        "branch_pair_operating_rows": int(len(operating)),
        "branch_pair_operating_all_act_rows": int(len(operating_all_act)),
        "branch_pair_update_sign_aligned_fraction": float(operating["signed_update_sign_aligned"].mean()),
        "branch_pair_all_act_update_sign_aligned_fraction": float(
            operating_all_act["signed_update_sign_aligned"].mean()
        ),
        "branch_pair_min_signed_read_gain": float(operating["signed_read_gain"].min()),
        "branch_pair_max_signed_read_gain": float(operating["signed_read_gain"].max()),
        "branch_pair_mean_signed_read_gain": float(operating["signed_read_gain"].mean()),
        "branch_pair_all_act_min_signed_read_gain": float(operating_all_act["signed_read_gain"].min()),
        "branch_pair_all_act_negative_signed_read_gain_fraction": float(
            (operating_all_act["signed_read_gain"] <= 0).mean()
        ),
        "branch_pair_min_signed_increase_mobility": float(operating["signed_increase_mobility"].min()),
        "branch_pair_max_signed_increase_mobility": float(operating["signed_increase_mobility"].max()),
        "branch_pair_min_signed_decrease_mobility": float(operating["signed_decrease_mobility"].min()),
        "branch_pair_max_signed_decrease_mobility": float(operating["signed_decrease_mobility"].max()),
        "branch_pair_physical_gradient_sign_aligned_fraction": float(
            operating["physical_gradient_sign_aligned"].mean()
        ),
        "branch_pair_min_physical_gradient_increase_mobility": float(
            operating["physical_gradient_increase_mobility"].min()
        ),
        "branch_pair_max_physical_gradient_increase_mobility": float(
            operating["physical_gradient_increase_mobility"].max()
        ),
        "branch_pair_min_physical_gradient_decrease_mobility": float(
            operating["physical_gradient_decrease_mobility"].min()
        ),
        "branch_pair_max_physical_gradient_decrease_mobility": float(
            operating["physical_gradient_decrease_mobility"].max()
        ),
        "branch_pair_min_physical_gradient_mobility_balance": float(
            operating["physical_gradient_mobility_balance"].min(skipna=True)
        ),
        "branch_pair_best_theta_p_v": float(best["theta_p"]),
        "branch_pair_best_theta_n_v": float(best["theta_n"]),
        "branch_pair_best_signed_read_gain": float(best["signed_read_gain"]),
        "branch_pair_best_signed_increase_mobility": float(best["signed_increase_mobility"]),
        "branch_pair_best_signed_decrease_mobility": float(best["signed_decrease_mobility"]),
        "branch_pair_best_signed_mobility_balance": float(best["signed_mobility_balance"]),
        "branch_pair_best_physical_gradient_increase_mobility": float(
            best["physical_gradient_increase_mobility"]
        ),
        "branch_pair_best_physical_gradient_decrease_mobility": float(
            best["physical_gradient_decrease_mobility"]
        ),
        "branch_pair_best_physical_gradient_mobility_balance": float(
            best["physical_gradient_mobility_balance"]
        ),
        "branch_pair_gain_safe_pair_count": int(len(fully_safe)),
        "branch_pair_best_gain_safe_theta_p_v": float(best_gain_safe["theta_p"]),
        "branch_pair_best_gain_safe_theta_n_v": float(best_gain_safe["theta_n"]),
        "branch_pair_best_gain_safe_min_signed_read_gain": float(best_gain_safe["min_signed_read_gain"]),
        "branch_pair_best_gain_safe_mean_signed_read_gain": float(best_gain_safe["mean_signed_read_gain"]),
        "branch_pair_best_gain_safe_min_signed_mobility_balance": float(
            best_gain_safe["min_signed_mobility_balance"]
        ),
        "branch_pair_best_gain_safe_min_physical_gradient_mobility_balance": float(
            best_gain_safe["min_physical_gradient_mobility_balance"]
        ),
        "branch_pair_best_gain_safe_mean_signed_mobility_balance": float(
            best_gain_safe["mean_signed_mobility_balance"]
        ),
        "branch_pair_best_gain_safe_aligned_fraction": float(best_gain_safe["aligned_fraction"]),
        "branch_pair_best_gain_safe_physical_gradient_aligned_fraction": float(
            best_gain_safe["physical_gradient_aligned_fraction"]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device_readout_mobility_sweep")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--write-width-u", type=float, default=0.02)
    ap.add_argument("--write-mode", choices=WRITE_MODES, default="discharge")
    ap.add_argument("--write-high-v", type=float, default=VDD)
    ap.add_argument("--write-low-v", type=float, default=0.0)
    ap.add_argument("--pre", type=float, default=0.65)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--pos-width-u", type=float, default=56.0)
    ap.add_argument("--neg-width-u", type=float, default=48.0)
    ap.add_argument("--negative-score-ic-v", type=float, default=0.30)
    ap.add_argument("--score-cap-f", type=float, default=10.0)
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
                    read_branch_netlist(
                        theta,
                        act,
                        branch,
                        score_ic=args.negative_score_ic_v,
                        pos_width_u=args.pos_width_u,
                        neg_width_u=args.neg_width_u,
                        score_cap_f=args.score_cap_f,
                    ),
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

    discharge_enabled, charge_enabled = write_actions(args.write_mode)
    for action, enabled in [("discharge", discharge_enabled), ("charge", charge_enabled)]:
        if not enabled:
            continue
        for theta in theta_values:
            measures = run_netlist(
                spice_bin,
                generated / f"{safe_tag}_write_{action}_t{theta:.2f}.cir",
                write_mobility_netlist(
                    theta,
                    args.pre,
                    args.delta,
                    args.write_width_u,
                    action,
                    args.write_mode,
                    args.write_high_v,
                    args.write_low_v,
                ),
                args.timeout,
            )
            rows.append(
                {
                    "experiment": f"write_{action}",
                    "theta": theta,
                    "act": None,
                    "pre": args.pre,
                    "delta": args.delta,
                    "write_width_u": args.write_width_u,
                    **measures,
                }
            )

    df = add_slopes(pd.DataFrame(rows))
    write = df[df["experiment"] == "write_discharge"][["theta", "state_delta_v"]].rename(
        columns={"state_delta_v": "write_discharge_v"}
    )
    if not write.empty:
        df = df.merge(write, on="theta", how="left")
    else:
        df["write_discharge_v"] = np.nan
    df["effective_mobility"] = df["read_slope"] * df["write_discharge_v"]
    signed = signed_mobility_table(df, args.write_mode, args.write_low_v, args.write_high_v)

    csv_path = results / f"{safe_tag}.csv"
    table_path = tables / f"{safe_tag}.csv"
    signed_csv_path = results / f"{safe_tag}_signed_mobility.csv"
    signed_table_path = tables / f"{safe_tag}_signed_mobility.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(table_path, index=False)
    signed.to_csv(signed_csv_path, index=False)
    signed.to_csv(signed_table_path, index=False)

    read_rows = df[df["experiment"].str.startswith("read_")]
    write_rows = df[df["experiment"].str.startswith("write_")]
    signed_summary = summarize_signed_mobility(signed, args.write_mode, args.write_low_v, args.write_high_v)
    signed_summary["signed_mobility_csv"] = str(signed_csv_path)
    signed_summary["signed_mobility_table_csv"] = str(signed_table_path)
    summary = {
        "tag": safe_tag,
        "simulator": version,
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "csv": str(csv_path),
        "table_csv": str(table_path),
        "theta_values": theta_values,
        "act_values": act_values,
        "write_mode": args.write_mode,
        "write_width_u": args.write_width_u,
        "write_high_v": args.write_high_v if args.write_mode.startswith("bounded_") else None,
        "write_low_v": args.write_low_v if args.write_mode.startswith("bounded_") else None,
        "write_pre_v": args.pre,
        "write_delta_v": args.delta,
        "read_positive_width_u": args.pos_width_u,
        "read_negative_width_u": args.neg_width_u,
        "negative_read_score_initial_v": args.negative_score_ic_v,
        "score_cap_f": args.score_cap_f,
        "min_read_slope": float(read_rows["read_slope"].min()),
        "max_read_slope": float(read_rows["read_slope"].max()),
        "min_write_state_delta_v": float(write_rows["state_delta_v"].min()),
        "max_write_state_delta_v": float(write_rows["state_delta_v"].max()),
        "min_write_discharge_v": float(df["write_discharge_v"].min(skipna=True))
        if "write_discharge_v" in df
        else None,
        "max_write_discharge_v": float(df["write_discharge_v"].max(skipna=True))
        if "write_discharge_v" in df
        else None,
        "min_effective_mobility": float(read_rows["effective_mobility"].min()),
        "max_effective_mobility": float(read_rows["effective_mobility"].max()),
        **signed_summary,
        "wall_time_s": time.perf_counter() - t0,
        "interpretation": (
            "Read-response slopes approximate G_eff'(theta); write state deltas approximate the natural "
            "direct-flow state mobility s_MOS(theta). The signed mobility table combines positive- and "
            "negative-branch read slopes with the enabled write actions to estimate the effective-weight "
            "mobility for desired positive and negative readout updates."
        ),
    }
    summary_path = results / f"{safe_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
