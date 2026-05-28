from __future__ import annotations

from pathlib import Path

from _util import parse_measures
from run_device_sequential_training import mos_models
from run_multiclass_block_sequence import (
    hidden_live_weight_update_lines,
    hidden_readout_weighted_credit_lines,
)
from run_multiclass_output_head_primitive import class_node, signed_store_lines
from run_spice_sweep import run_text_netlist
from run_xor_output_normalizer_primitives import normalizer_subcircuits


READOUT_POLARITIES = ("positive", "negative", "neutral")


def _validate_pair(name: str, values: tuple[float, float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    left, right = (float(values[0]), float(values[1]))
    if left < 0.0 or right < 0.0:
        raise ValueError(f"{name} values must be nonnegative")
    return left, right


def _class_readout_initials(polarity: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if polarity == "positive":
        return (0.40, 0.28), (0.28, 0.40)
    if polarity == "negative":
        return (0.28, 0.40), (0.40, 0.28)
    if polarity == "neutral":
        return (0.40, 0.40), (0.40, 0.40)
    raise ValueError(f"readout_polarity must be one of {READOUT_POLARITIES}")


def generate_netlist(
    *,
    evidence: tuple[float, float],
    target: int,
    readout_polarity: str = "positive",
    conductance_floor: float = 0.15,
    conductance_scale: float = 1.0,
    hidden_positive: float = 0.45,
    hidden_negative: float = 0.40,
    activation_level: float = 1.2,
    eligibility_level: float = 1.2,
    hidden_credit_width_u: float = 32.0,
    hidden_update_width_u: float = 0.25,
) -> str:
    g0_raw, g1_raw = _validate_pair("evidence", evidence)
    if target not in (0, 1):
        raise ValueError("target must be 0 or 1")
    if conductance_floor < 0.0:
        raise ValueError("conductance_floor must be nonnegative")
    if conductance_scale <= 0.0:
        raise ValueError("conductance_scale must be positive")
    if not (0.0 <= activation_level <= 1.2 and 0.0 <= eligibility_level <= 1.2):
        raise ValueError("activation and eligibility must stay within supply rails")
    if min(hidden_positive, hidden_negative, hidden_credit_width_u, hidden_update_width_u) < 0.0:
        raise ValueError("hidden initial voltages and widths must be nonnegative")

    t0 = 1.2 if target == 0 else 0.0
    t1 = 1.2 if target == 1 else 0.0
    cd_w0 = g0_raw * conductance_scale
    cd_w1 = g1_raw * conductance_scale
    c0_init, c1_init = _class_readout_initials(readout_polarity)

    lines = [
        "* XOR-scale normalized hidden-update primitive.",
        "* Conductance-divider probability current is routed into error rails;",
        "* readout-weighted analog credit then updates a hidden weight in SPICE.",
        ".param VDD=1.2",
        mos_models(),
        normalizer_subcircuits(
            conductance_w0=cd_w0,
            conductance_w1=cd_w1,
            conductance_floor=conductance_floor,
        ),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vwhi vwhi_ref 0 1.05",
        "Vwlo vwlo_ref 0 0.15",
        "Vrst rst 0 PULSE(1.2 0 0.0n 10p 10p 0.40n 20n)",
        "Vphip phip 0 PULSE(0 1.2 1.0n 20p 20p 2.0n 20n)",
        f"Vt0 t0 0 {t0:.12g}",
        f"Vt1 t1 0 {t1:.12g}",
        f"Vact act0 0 PULSE(0 {activation_level:.12g} 3.15n 20p 20p 2.2n 20n)",
        f"Vxelig xelig0 0 PULSE(0 {eligibility_level:.12g} 3.15n 20p 20p 2.2n 20n)",
        "Vrsen0 rnorm rd0 0",
        "Vrsen1 rnorm rd1 0",
        "Xwroute rnorm rd0 rd1 t0 t1 phip c0_errp c0_errn c1_errp c1_errn vdd 0 xor_current_to_writer_descent2",
    ]
    for class_idx, (vwp, vwn) in enumerate((c0_init, c1_init)):
        lines += signed_store_lines(
            positive_node=class_node(class_idx, "vwp0"),
            negative_node=class_node(class_idx, "vwn0"),
            positive_ic=vwp,
            negative_ic=vwn,
        )
    lines += [
        f"Cwhp0 whp0 0 20f IC={hidden_positive:.12g}",
        f"Cwhn0 whn0 0 20f IC={hidden_negative:.12g}",
        "Rwhp0 whp0 0 1e15",
        "Rwhn0 whn0 0 1e15",
        *hidden_readout_weighted_credit_lines(
            class_count=2,
            feature_idx=0,
            error_positive_nodes=[class_node(0, "errp"), class_node(1, "errp")],
            error_negative_nodes=[class_node(0, "errn"), class_node(1, "errn")],
            width_u=hidden_credit_width_u,
        ),
        *hidden_live_weight_update_lines(
            feature_idx=0,
            eligibility_node="xelig0",
            positive_credit_node="h0_hdp",
            negative_credit_node="h0_hdn",
            width_u=hidden_update_width_u,
        ),
        ".meas tran ir0_raw FIND I(Vrsen0) AT=2.95n",
        ".meas tran ir1_raw FIND I(Vrsen1) AT=2.95n",
        ".meas tran rnorm_after FIND V(rnorm) AT=2.95n",
        ".meas tran e0p_after FIND V(c0_errp) AT=3.10n",
        ".meas tran e0n_after FIND V(c0_errn) AT=3.10n",
        ".meas tran e1p_after FIND V(c1_errp) AT=3.10n",
        ".meas tran e1n_after FIND V(c1_errn) AT=3.10n",
        ".meas tran hdp_after FIND V(h0_hdp) AT=5.35n",
        ".meas tran hdn_after FIND V(h0_hdn) AT=5.35n",
        ".meas tran hcredit_diff PARAM='hdp_after-hdn_after'",
        ".meas tran whp_before FIND V(whp0) AT=3.05n",
        ".meas tran whn_before FIND V(whn0) AT=3.05n",
        ".meas tran signed_before PARAM='whp_before-whn_before'",
        ".meas tran whp_after FIND V(whp0) AT=6.40n",
        ".meas tran whn_after FIND V(whn0) AT=6.40n",
        ".meas tran signed_after PARAM='whp_after-whn_after'",
        ".meas tran signed_delta PARAM='signed_after-signed_before'",
        ".tran 2p 7n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float = 30.0) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed
