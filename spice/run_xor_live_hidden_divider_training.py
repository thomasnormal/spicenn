from __future__ import annotations

from pathlib import Path
from typing import Any

from _util import parse_measures
from run_device_sequential_training import mos_models, pwl
from run_multiclass_output_head_primitive import class_node, signed_store_lines
from run_multiclass_output_head_sequence import class_local_live_label_descent_update_lines
from run_spice_sweep import run_text_netlist


BITS = 2
HIDDEN = 4
OUTPUTS = 2
CYCLE_NS = 10.0
HIDDEN_WRITER_MODES = ("nmos-pass", "pmos-highside")


def bit_value(pattern: int, bit: int) -> int:
    return (pattern >> bit) & 1


def xor_label(pattern: int) -> int:
    return bit_value(pattern, 0) ^ bit_value(pattern, 1)


def _input_value(pattern: int, node: str) -> float:
    if node.startswith("nx"):
        return 1.2 * float(1 - bit_value(pattern, int(node[2:])))
    return 1.2 * float(bit_value(pattern, int(node[1:])))


def _pulses_to_pwl(pulses: list[tuple[float, float]], stop_ns: float, high: float = 1.2) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for start, end in pulses:
        if start > 0.0:
            points.append((max(0.0, start - 0.03), 0.0))
        points.append((start, high))
        points.append((end, high))
        points.append((min(stop_ns, end + 0.03), 0.0))
    points.append((stop_ns, 0.0))
    return pwl(points)


def _active_low_pulses_to_pwl(pulses: list[tuple[float, float]], stop_ns: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 1.2)]
    for start, end in pulses:
        if start > 0.0:
            points.append((max(0.0, start - 0.03), 1.2))
        points.append((start, 0.0))
        points.append((end, 0.0))
        points.append((min(stop_ns, end + 0.03), 1.2))
    points.append((stop_ns, 1.2))
    return pwl(points)


def _sample_wave(samples: list[dict[str, Any]], node: str, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = _input_value(int(sample["pattern"]), node)
        if idx == 0:
            points.append((0.0, value))
        else:
            prev = _input_value(int(samples[idx - 1]["pattern"]), node)
            points.append((start - 0.03, prev))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.03), value))
    points.append((stop_ns, _input_value(int(samples[-1]["pattern"]), node)))
    return pwl(points)


def _target_wave(samples: list[dict[str, Any]], output: int, stop_ns: float) -> str:
    points: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        start = idx * CYCLE_NS
        end = start + CYCLE_NS
        value = 1.2 if int(sample["label"]) == output else 0.0
        if idx == 0:
            points.append((0.0, value))
        else:
            prev = 1.2 if int(samples[idx - 1]["label"]) == output else 0.0
            points.append((start - 0.03, prev))
            points.append((start, value))
        points.append((min(stop_ns, end - 0.03), value))
    points.append((stop_ns, 1.2 if int(samples[-1]["label"]) == output else 0.0))
    return pwl(points)


def _current_pulse_wave(pulses: list[tuple[float, float]], stop_ns: float, current_a: float) -> str:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for start, end in pulses:
        if start > 0.0:
            points.append((max(0.0, start - 0.03), 0.0))
        points.append((start, current_a))
        points.append((end, current_a))
        points.append((min(stop_ns, end + 0.03), 0.0))
    points.append((stop_ns, 0.0))
    return pwl(points)


def _sample_plan(train_order: list[int]) -> list[dict[str, Any]]:
    initial = [{"phase": "initial", "pattern": p, "label": xor_label(p), "train": False} for p in range(4)]
    train = [{"phase": "train", "pattern": p, "label": xor_label(p), "train": True} for p in train_order]
    final = [{"phase": "final", "pattern": p, "label": xor_label(p), "train": False} for p in range(4)]
    return initial + train + final


def _clock_lines(samples: list[dict[str, Any]], stop_ns: float, iref_a: float) -> list[str]:
    reset: list[tuple[float, float]] = []
    feat: list[tuple[float, float]] = []
    score: list[tuple[float, float]] = []
    err: list[tuple[float, float]] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        reset.append((base + 0.00, base + 0.45))
        feat.append((base + 0.75, base + 2.15))
        score.append((base + 2.05, base + 3.25))
        if bool(sample["train"]):
            err.append((base + 3.50, base + 5.60))
    return [
        f"Vrst rst 0 {_pulses_to_pwl(reset, stop_ns)}",
        f"Vrstn rstn 0 {_active_low_pulses_to_pwl(reset, stop_ns)}",
        f"Vfeatphi featphi 0 {_pulses_to_pwl(feat, stop_ns)}",
        f"Vscorephi scorephi 0 {_pulses_to_pwl(score, stop_ns)}",
        f"Verrphi errphi 0 {_pulses_to_pwl(err, stop_ns)}",
        f"Iprobref vdd rnorm {_current_pulse_wave(err, stop_ns, iref_a)}",
    ]


def _hidden_storage_lines(initial_positive: float, initial_negative: float) -> list[str]:
    lines: list[str] = []
    for hidden in range(HIDDEN):
        for bit in range(BITS):
            for kind, value in (("p", initial_positive), ("n", initial_negative)):
                node = f"wh{hidden}{bit}{kind}"
                lines += [
                    f"C{node} {node} 0 20f IC={value:.12g}",
                    f"R{node} {node} 0 1e15",
                ]
    return lines


def _readout_storage_lines(initial_positive: float, initial_negative: float) -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        for hidden in range(HIDDEN):
            lines += signed_store_lines(
                positive_node=class_node(output, f"vwp{hidden}"),
                negative_node=class_node(output, f"vwn{hidden}"),
                positive_ic=initial_positive,
                negative_ic=initial_negative,
            )
    return lines


def _state_storage_lines() -> list[str]:
    lines: list[str] = []
    for hidden in range(HIDDEN):
        for node in (f"pre{hidden}_p", f"pre{hidden}_n", f"act{hidden}"):
            lines += [
                f"C{node} {node} 0 12f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rst 0 0 NMOS W=4u L=180n",
            ]
    for output in range(OUTPUTS):
        for kind in ("scorep", "scoren", "errp", "errn"):
            node = class_node(output, kind)
            cap_f = 2.0 if kind.startswith("err") else 8.0
            lines += [
                f"C{node} {node} 0 {cap_f:.12g}f IC=0",
                f"R{node} {node} 0 1G",
                f"Mreset_{node} {node} rst 0 0 NMOS W=4u L=180n",
            ]
    return lines


def _hidden_forward_lines(width_u: float) -> list[str]:
    lines: list[str] = []
    for hidden in range(HIDDEN):
        pre_p = f"pre{hidden}_p"
        pre_n = f"pre{hidden}_n"
        act = f"act{hidden}"
        lines.append(f"* Hidden feature {hidden}: trainable split-rail literal detector.")
        for bit in range(BITS):
            match_lit = f"x{bit}" if bit_value(hidden, bit) else f"nx{bit}"
            mismatch_lit = f"nx{bit}" if bit_value(hidden, bit) else f"x{bit}"
            lines += [
                f"Rh{hidden}{bit}pmid h{hidden}{bit}pmid 0 1G",
                f"Rh{hidden}{bit}nmid h{hidden}{bit}nmid 0 1G",
                f"Mh{hidden}{bit}p_phi {match_lit} featphi h{hidden}{bit}pmid 0 NSENSE W={width_u:.6g}u L=180n",
                f"Mh{hidden}{bit}p_w h{hidden}{bit}pmid wh{hidden}{bit}p {pre_p} 0 NSENSE W={width_u:.6g}u L=180n",
                f"Mh{hidden}{bit}n_phi {mismatch_lit} featphi h{hidden}{bit}nmid 0 NSENSE W={width_u:.6g}u L=180n",
                f"Mh{hidden}{bit}n_w h{hidden}{bit}nmid wh{hidden}{bit}n {pre_n} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
        lines += [
            f"Mact{hidden}_p vdd {pre_p} {act} 0 NREL W=24u L=180n",
            f"Mact{hidden}_n {act} {pre_n} 0 0 NSENSE W=24u L=180n",
        ]
    return lines


def _score_readout_lines(width_u: float) -> list[str]:
    lines: list[str] = []
    for output in range(OUTPUTS):
        scorep = class_node(output, "scorep")
        scoren = class_node(output, "scoren")
        for hidden in range(HIDDEN):
            prefix = f"c{output}_h{hidden}_score_"
            lines += [
                f"R{prefix}pa {prefix}pa 0 1G",
                f"R{prefix}pb {prefix}pb 0 1G",
                f"M{prefix}pa vdd act{hidden} {prefix}pa 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pw {prefix}pa {class_node(output, f'vwp{hidden}')} {prefix}pb 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pphi {prefix}pb scorephi {scorep} 0 NSENSE W={width_u:.6g}u L=180n",
                f"R{prefix}na {prefix}na 0 1G",
                f"R{prefix}nb {prefix}nb 0 1G",
                f"M{prefix}na vdd act{hidden} {prefix}na 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}nw {prefix}na {class_node(output, f'vwn{hidden}')} {prefix}nb 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}nphi {prefix}nb scorephi {scoren} 0 NSENSE W={width_u:.6g}u L=180n",
            ]
    return lines


def _divider_probability_lines(branch_width_u: float, floor_width_u: float) -> list[str]:
    return [
        "Vnormfloor normfloor 0 0.62",
        "Rrnorm rnorm 0 1G",
        "Rmir0 mir0 0 1G",
        "Rmir1 mir1 0 1G",
        "Vrsen0 rnorm rd0 0",
        "Vrsen1 rnorm rd1 0",
        f"Mnorm0_score rd0 {class_node(0, 'scorep')} mir0 0 NSENSE W={branch_width_u:.6g}u L=180n",
        f"Mnorm0_floor rd0 normfloor mir0 0 NSENSE W={floor_width_u:.6g}u L=180n",
        f"Mnorm1_score rd1 {class_node(1, 'scorep')} mir1 0 NSENSE W={branch_width_u:.6g}u L=180n",
        f"Mnorm1_floor rd1 normfloor mir1 0 NSENSE W={floor_width_u:.6g}u L=180n",
        "Mnorm0_ref mir0 mir0 0 0 NMOS W=2u L=180n",
        "Mnorm1_ref mir1 mir1 0 0 NMOS W=2u L=180n",
    ]


def _error_route_lines(route_width_u: float) -> list[str]:
    def charge_lines(name: str, dest: str, low: str, target: str) -> list[str]:
        return [
            f"R{name}_a {name}_a 0 1G",
            f"R{name}_b {name}_b 0 1G",
            f"M{name}_m vdd {low} {name}_a vdd PMOS W={route_width_u:.6g}u L=180n",
            f"M{name}_t {name}_a {target} {name}_b 0 NSENSE W=12u L=180n",
            f"M{name}_phi {name}_b errphi {dest} 0 NSENSE W=12u L=180n",
        ]

    lines: list[str] = []
    for branch in (0, 1):
        low = f"b{branch}low"
        lines += [
            f"C{low} {low} 0 1f IC=1.2",
            f"R{low} {low} vdd 1G",
            f"Mpre_{low} {low} rstn vdd vdd PMOS W=4u L=180n",
            f"R{low}_a {low}_a 0 1G",
            f"M{low}_sink {low} mir{branch} {low}_a 0 NMOS W=0.85u L=180n",
            f"M{low}_phi {low}_a errphi 0 0 NSENSE W=0.85u L=180n",
        ]
    lines += charge_lines("err_c0p", class_node(0, "errp"), "b1low", "t0")
    lines += charge_lines("err_c1n", class_node(1, "errn"), "b1low", "t0")
    lines += charge_lines("err_c1p", class_node(1, "errp"), "b0low", "t1")
    lines += charge_lines("err_c0n", class_node(0, "errn"), "b0low", "t1")
    return lines


def _readout_writer_lines(width_u: float) -> list[str]:
    lines = [
        "Vvwhi_ref vwhi_ref 0 0.48",
        "Vvwlo_ref vwlo_ref 0 0.22",
    ]
    for output in range(OUTPUTS):
        for hidden in range(HIDDEN):
            lines += class_local_live_label_descent_update_lines(
                class_idx=output,
                feature_idx=hidden,
                activation_node=f"act{hidden}",
                positive_descent_node=class_node(output, "errp"),
                negative_descent_node=class_node(output, "errn"),
                width_u=width_u,
                high_side_topology="pmos-differential",
            )
    return lines


def _hidden_writer_lines(mode: str, width_u: float, pmos_width_u: float, gate_cap_f: float) -> list[str]:
    if mode not in HIDDEN_WRITER_MODES:
        raise ValueError(f"hidden_writer_mode must be one of {HIDDEN_WRITER_MODES}")
    lines = [
        "Vhidden_whi_ref hidden_whi_ref 0 1.05",
        "Vhidden_wlo_ref hidden_wlo_ref 0 0.15",
    ]

    def pmos_charge_lines(prefix: str, dest: str, selector: str, credit: str) -> list[str]:
        gate = f"{prefix}_pgate"
        mid = f"{prefix}_pgmid"
        return [
            f"C{gate} {gate} 0 {gate_cap_f:.12g}f IC=1.05",
            f"R{gate} {gate} hidden_whi_ref 1G",
            f"R{mid} {mid} 0 1G",
            f"M{prefix}_pgate_rst hidden_whi_ref rst {gate} 0 NSENSE W=4u L=180n",
            f"M{prefix}_pgate_sel {gate} {selector} {mid} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_pgate_cred {mid} {credit} 0 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_pmos {dest} {gate} hidden_whi_ref vdd PMOS W={pmos_width_u:.6g}u L=180n",
        ]

    for hidden in range(HIDDEN):
        hdp = f"h{hidden}_hdp_gate"
        hdn = f"h{hidden}_hdn_gate"
        for bit in range(BITS):
            match_lit = f"x{bit}" if bit_value(hidden, bit) else f"nx{bit}"
            mismatch_lit = f"nx{bit}" if bit_value(hidden, bit) else f"x{bit}"
            whp = f"wh{hidden}{bit}p"
            whn = f"wh{hidden}{bit}n"
            prefix = f"h{hidden}b{bit}_live_"
            lines += [
                f"R{prefix}pdn {prefix}pdn 0 1G",
                f"R{prefix}ndn {prefix}ndn 0 1G",
                f"M{prefix}pdn_e {whp} {match_lit} {prefix}pdn 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}pdn_c {prefix}pdn {hdn} hidden_wlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}ndn_e {whn} {mismatch_lit} {prefix}ndn 0 NSENSE W={width_u:.6g}u L=180n",
                f"M{prefix}ndn_c {prefix}ndn {hdp} hidden_wlo_ref 0 NSENSE W={width_u:.6g}u L=180n",
            ]
            if mode == "nmos-pass":
                lines += [
                    f"R{prefix}pup {prefix}pup 0 1G",
                    f"R{prefix}nup {prefix}nup 0 1G",
                    f"M{prefix}pup_e hidden_whi_ref {match_lit} {prefix}pup 0 NSENSE W={width_u:.6g}u L=180n",
                    f"M{prefix}pup_c {prefix}pup {hdp} {whp} 0 NSENSE W={width_u:.6g}u L=180n",
                    f"M{prefix}nup_e hidden_whi_ref {mismatch_lit} {prefix}nup 0 NSENSE W={width_u:.6g}u L=180n",
                    f"M{prefix}nup_c {prefix}nup {hdn} {whn} 0 NSENSE W={width_u:.6g}u L=180n",
                ]
            else:
                lines += pmos_charge_lines(f"{prefix}pup", whp, match_lit, hdp)
                lines += pmos_charge_lines(f"{prefix}nup", whn, mismatch_lit, hdn)
    return lines


def _hidden_credit_gate_lines(width_u: float = 4.0, cap_f: float = 2.0, pull_scale: float = 1.0) -> list[str]:
    lines: list[str] = []
    for hidden in range(HIDDEN):
        raw_p = f"h{hidden}_hdp"
        raw_n = f"h{hidden}_hdn"
        gate_p = f"h{hidden}_hdp_gate"
        gate_n = f"h{hidden}_hdn_gate"
        mid_p = f"{gate_p}_mid"
        mid_n = f"{gate_n}_mid"
        lines += [
            f"C{gate_p} {gate_p} 0 {cap_f:.12g}f IC=0",
            f"C{gate_n} {gate_n} 0 {cap_f:.12g}f IC=0",
            f"R{gate_p} {gate_p} 0 1G",
            f"R{gate_n} {gate_n} 0 1G",
            f"R{mid_p} {mid_p} 0 1G",
            f"R{mid_n} {mid_n} 0 1G",
            f"Mreset_{gate_p} {gate_p} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{gate_n} {gate_n} rst 0 0 NMOS W=4u L=180n",
            f"M{gate_p}_up0 vdd {raw_p} {mid_p} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{gate_p}_up1 {mid_p} {raw_p} {gate_p} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{gate_p}_dn {gate_p} {raw_n} 0 0 NSENSE W={pull_scale * width_u:.6g}u L=180n",
            f"M{gate_n}_up0 vdd {raw_n} {mid_n} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{gate_n}_up1 {mid_n} {raw_n} {gate_n} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{gate_n}_dn {gate_n} {raw_p} 0 0 NSENSE W={pull_scale * width_u:.6g}u L=180n",
        ]
    return lines


def _hidden_credit_lines(
    width_u: float,
    capacitance_f: float,
    activation_model: str,
    internal_cap_f: float,
    internal_shunt_ohm: float,
) -> list[str]:
    lines: list[str] = []

    def term(prefix: str, source: str, err: str, weight: str, dest: str) -> list[str]:
        n0 = f"{prefix}_e"
        n1 = f"{prefix}_w"
        return [
            f"R{n0} {n0} 0 {internal_shunt_ohm:.12g}",
            f"R{n1} {n1} 0 {internal_shunt_ohm:.12g}",
            f"C{n0} {n0} 0 {internal_cap_f:.12g}f IC=0",
            f"C{n1} {n1} 0 {internal_cap_f:.12g}f IC=0",
            f"M{prefix}_e {source} {err} {n0} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_w {n0} {weight} {n1} 0 NSENSE W={width_u:.6g}u L=180n",
            f"M{prefix}_a {n1} act{hidden} {dest} 0 {activation_model} W={width_u:.6g}u L=180n",
        ]

    for hidden in range(HIDDEN):
        hdp = f"h{hidden}_hdp"
        hdn = f"h{hidden}_hdn"
        lines += [
            f"C{hdp} {hdp} 0 {capacitance_f:.12g}f IC=0",
            f"C{hdn} {hdn} 0 {capacitance_f:.12g}f IC=0",
            f"R{hdp} {hdp} 0 1G",
            f"R{hdn} {hdn} 0 1G",
            f"Mreset_{hdp} {hdp} rst 0 0 NMOS W=4u L=180n",
            f"Mreset_{hdn} {hdn} rst 0 0 NMOS W=4u L=180n",
        ]
        for output in range(OUTPUTS):
            errp = class_node(output, "errp")
            errn = class_node(output, "errn")
            vwp = class_node(output, f"vwp{hidden}")
            vwn = class_node(output, f"vwn{hidden}")
            prefix = f"h{hidden}_c{output}_cred"
            lines += term(f"{prefix}_pv", "vdd", errp, vwp, hdp)
            lines += term(f"{prefix}_pn", "vdd", errp, vwn, hdn)
            lines += term(f"{prefix}_nv", "vdd", errn, vwn, hdp)
            lines += term(f"{prefix}_nn", "vdd", errn, vwp, hdn)
    return lines


def _measure_lines(samples: list[dict[str, Any]], train_offset: int) -> list[str]:
    lines: list[str] = []
    for idx, sample in enumerate(samples):
        base = idx * CYCLE_NS
        pattern = int(sample["pattern"])
        label = int(sample["label"])
        other = 1 - label
        if sample["phase"] in {"initial", "final"}:
            phase = str(sample["phase"])
            at = base + 3.20
            lines += [
                f".meas tran {phase}_target_scorep_{pattern} FIND V({class_node(label, 'scorep')}) AT={at:.2f}n",
                f".meas tran {phase}_target_scoren_{pattern} FIND V({class_node(label, 'scoren')}) AT={at:.2f}n",
                f".meas tran {phase}_other_scorep_{pattern} FIND V({class_node(other, 'scorep')}) AT={at:.2f}n",
                f".meas tran {phase}_other_scoren_{pattern} FIND V({class_node(other, 'scoren')}) AT={at:.2f}n",
                f".meas tran {phase}_target_signed_{pattern} PARAM='{phase}_target_scorep_{pattern}-{phase}_target_scoren_{pattern}'",
                f".meas tran {phase}_other_signed_{pattern} PARAM='{phase}_other_scorep_{pattern}-{phase}_other_scoren_{pattern}'",
                f".meas tran {phase}_margin_{pattern} PARAM='{phase}_target_signed_{pattern}-{phase}_other_signed_{pattern}'",
            ]
        if sample["phase"] == "train":
            local = idx - train_offset
            before = base + 0.55
            after = base + 7.80
            err_at = base + 5.55
            credit_at = base + 5.55
            active = pattern
            lines += [
                f".meas tran train_ir0_{local} FIND I(Vrsen0) AT={base + 5.00:.2f}n",
                f".meas tran train_ir1_{local} FIND I(Vrsen1) AT={base + 5.00:.2f}n",
                f".meas tran train_target_errp_{local} FIND V({class_node(label, 'errp')}) AT={err_at:.2f}n",
                f".meas tran train_target_errn_{local} FIND V({class_node(label, 'errn')}) AT={err_at:.2f}n",
                f".meas tran train_other_errp_{local} FIND V({class_node(other, 'errp')}) AT={err_at:.2f}n",
                f".meas tran train_other_errn_{local} FIND V({class_node(other, 'errn')}) AT={err_at:.2f}n",
                f".meas tran train_hdp_active_{local} FIND V(h{active}_hdp) AT={credit_at:.2f}n",
                f".meas tran train_hdn_active_{local} FIND V(h{active}_hdn) AT={credit_at:.2f}n",
                f".meas tran train_hdp_gate_active_{local} FIND V(h{active}_hdp_gate) AT={credit_at:.2f}n",
                f".meas tran train_hdn_gate_active_{local} FIND V(h{active}_hdn_gate) AT={credit_at:.2f}n",
                f".meas tran train_hcredit_active_{local} PARAM='train_hdp_active_{local}-train_hdn_active_{local}'",
                f".meas tran train_hcredit_gate_active_{local} PARAM='train_hdp_gate_active_{local}-train_hdn_gate_active_{local}'",
            ]
            for output in (label, other):
                role = "target" if output == label else "other"
                lines += [
                    f".meas tran train_{role}_vwp_before_{local} FIND V({class_node(output, f'vwp{active}')}) AT={before:.2f}n",
                    f".meas tran train_{role}_vwn_before_{local} FIND V({class_node(output, f'vwn{active}')}) AT={before:.2f}n",
                    f".meas tran train_{role}_vwp_after_{local} FIND V({class_node(output, f'vwp{active}')}) AT={after:.2f}n",
                    f".meas tran train_{role}_vwn_after_{local} FIND V({class_node(output, f'vwn{active}')}) AT={after:.2f}n",
                    f".meas tran train_{role}_signed_before_{local} PARAM='train_{role}_vwp_before_{local}-train_{role}_vwn_before_{local}'",
                    f".meas tran train_{role}_signed_after_{local} PARAM='train_{role}_vwp_after_{local}-train_{role}_vwn_after_{local}'",
                    f".meas tran train_{role}_signed_delta_{local} PARAM='train_{role}_signed_after_{local}-train_{role}_signed_before_{local}'",
                ]
            for bit in range(BITS):
                lines += [
                    f".meas tran train_wh{active}{bit}p_before_{local} FIND V(wh{active}{bit}p) AT={before:.2f}n",
                    f".meas tran train_wh{active}{bit}n_before_{local} FIND V(wh{active}{bit}n) AT={before:.2f}n",
                    f".meas tran train_wh{active}{bit}p_after_{local} FIND V(wh{active}{bit}p) AT={after:.2f}n",
                    f".meas tran train_wh{active}{bit}n_after_{local} FIND V(wh{active}{bit}n) AT={after:.2f}n",
                    f".meas tran train_wh{active}{bit}_signed_before_{local} PARAM='train_wh{active}{bit}p_before_{local}-train_wh{active}{bit}n_before_{local}'",
                    f".meas tran train_wh{active}{bit}_signed_after_{local} PARAM='train_wh{active}{bit}p_after_{local}-train_wh{active}{bit}n_after_{local}'",
                    f".meas tran train_wh{active}{bit}_signed_delta_{local} PARAM='train_wh{active}{bit}_signed_after_{local}-train_wh{active}{bit}_signed_before_{local}'",
                ]
    for pattern in range(4):
        lines.append(f".meas tran final_margin_improvement_{pattern} PARAM='final_margin_{pattern}-initial_margin_{pattern}'")
    return lines


def xor_live_hidden_netlist(
    train_order: list[int],
    *,
    iref_a: float = 1.0e-6,
    hidden_initial_positive: float = 0.85,
    hidden_initial_negative: float = 0.85,
    readout_initial_positive: float = 0.40,
    readout_initial_negative: float = 0.40,
    hidden_forward_width_u: float = 16.0,
    readout_width_u: float = 16.0,
    branch_width_u: float = 0.05,
    floor_width_u: float = 0.015,
    route_width_u: float = 3.0,
    readout_update_width_u: float = 0.25,
    hidden_credit_width_u: float = 32.0,
    hidden_credit_cap_f: float = 12.0,
    hidden_update_width_u: float = 0.05,
    hidden_credit_activation_model: str = "NMOS",
    hidden_credit_internal_cap_f: float = 0.05,
    hidden_credit_internal_shunt_ohm: float = 5.0e7,
    hidden_credit_gate_width_u: float = 4.0,
    hidden_credit_gate_cap_f: float = 2.0,
    hidden_credit_gate_pull_scale: float = 1.0,
    hidden_writer_mode: str = "pmos-highside",
    hidden_writer_pmos_width_u: float = 2.0,
    hidden_writer_gate_cap_f: float = 0.2,
) -> str:
    if not train_order:
        raise ValueError("train_order must not be empty")
    if any(pattern not in range(4) for pattern in train_order):
        raise ValueError("train_order patterns must be in 0..3")
    if min(
        iref_a,
        hidden_initial_positive,
        hidden_initial_negative,
        readout_initial_positive,
        readout_initial_negative,
        hidden_forward_width_u,
        readout_width_u,
        branch_width_u,
        floor_width_u,
        route_width_u,
        readout_update_width_u,
        hidden_credit_width_u,
        hidden_credit_cap_f,
        hidden_update_width_u,
        hidden_credit_internal_cap_f,
        hidden_credit_internal_shunt_ohm,
        hidden_credit_gate_width_u,
        hidden_credit_gate_cap_f,
        hidden_credit_gate_pull_scale,
        hidden_writer_pmos_width_u,
        hidden_writer_gate_cap_f,
    ) <= 0.0:
        raise ValueError("voltages, currents, widths, and capacitances must be positive")
    if hidden_credit_activation_model not in {"NREL", "NMOS", "NSENSE"}:
        raise ValueError("hidden_credit_activation_model must be NREL, NMOS, or NSENSE")
    if hidden_writer_mode not in HIDDEN_WRITER_MODES:
        raise ValueError(f"hidden_writer_mode must be one of {HIDDEN_WRITER_MODES}")

    samples = _sample_plan(train_order)
    stop_ns = len(samples) * CYCLE_NS
    train_offset = 4
    lines = [
        "* Live-hidden XOR learner with conductance-divider normalized error.",
        "* Python supplies inputs, labels, clocks, and diagnostics only.",
        "* Readout and hidden weights move through live transistor/passive writer paths.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vx0 x0 0 {_sample_wave(samples, 'x0', stop_ns)}",
        f"Vx1 x1 0 {_sample_wave(samples, 'x1', stop_ns)}",
        f"Vnx0 nx0 0 {_sample_wave(samples, 'nx0', stop_ns)}",
        f"Vnx1 nx1 0 {_sample_wave(samples, 'nx1', stop_ns)}",
        f"Vt0 t0 0 {_target_wave(samples, 0, stop_ns)}",
        f"Vt1 t1 0 {_target_wave(samples, 1, stop_ns)}",
        *_clock_lines(samples, stop_ns, iref_a),
        *_hidden_storage_lines(hidden_initial_positive, hidden_initial_negative),
        *_readout_storage_lines(readout_initial_positive, readout_initial_negative),
        *_state_storage_lines(),
        *_hidden_forward_lines(hidden_forward_width_u),
        *_score_readout_lines(readout_width_u),
        *_divider_probability_lines(branch_width_u, floor_width_u),
        *_error_route_lines(route_width_u),
        *_readout_writer_lines(readout_update_width_u),
        *_hidden_credit_lines(
            hidden_credit_width_u,
            hidden_credit_cap_f,
            hidden_credit_activation_model,
            hidden_credit_internal_cap_f,
            hidden_credit_internal_shunt_ohm,
        ),
        *_hidden_credit_gate_lines(
            hidden_credit_gate_width_u,
            hidden_credit_gate_cap_f,
            hidden_credit_gate_pull_scale,
        ),
        *_hidden_writer_lines(
            hidden_writer_mode,
            hidden_update_width_u,
            hidden_writer_pmos_width_u,
            hidden_writer_gate_cap_f,
        ),
        f".tran 5p {stop_ns:.2f}n uic",
        *_measure_lines(samples, train_offset),
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def run_netlist(spice_bin: str, path: Path, netlist: str, timeout: float = 90.0) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, netlist, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    parsed = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not parsed:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return parsed
