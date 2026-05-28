from __future__ import annotations

from pathlib import Path

from _util import parse_measures
from run_device_sequential_training import mos_models
from run_multiclass_output_head_primitive import class_node, signed_store_lines
from run_multiclass_output_head_sequence import class_local_live_label_descent_update_lines
from run_spice_sweep import run_text_netlist


APPROACHES = ("conductance-divider", "soft-wta")
DIVIDER_ERROR_MODES = ("sampled-voltage", "current-routed")


def _validate_pair(name: str, values: tuple[float, float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    left, right = (float(values[0]), float(values[1]))
    if left < 0.0 or right < 0.0:
        raise ValueError(f"{name} values must be nonnegative")
    return left, right


def normalizer_subcircuits(
    *,
    conductance_w0: float = 1.0,
    conductance_w1: float = 1.0,
    conductance_floor: float = 0.15,
) -> str:
    cd_w0_u = 0.005 * (conductance_w0 + conductance_floor)
    cd_w1_u = 0.005 * (conductance_w1 + conductance_floor)
    cd_route_w0_u = 0.05 * (conductance_w0 + conductance_floor)
    cd_route_w1_u = 0.05 * (conductance_w1 + conductance_floor)
    return f"""
.subckt xor_prob_conductance_divider2 p0 p1 norm d0 d1 phip rst vdd vss
* A fixed reference current is split by two score/evidence conductances.
* With the branch devices in the small-swing region:
*   I_i ~= IREF * G_i / (G0 + G1)
* Low-impedance ps0/ps1 nodes sense branch current, then p0/p1 sample those
* voltages onto high-impedance hold caps for measurement and error routing.
Vgcd gcond vss 0.62
Icdref vdd norm PULSE(0 1u 1.0n 20p 20p 2.0n 20n)
Cp0 p0 vss 1f IC=0
Cp1 p1 vss 1f IC=0
Cps0 ps0 vss 0.2f IC=0
Cps1 ps1 vss 0.2f IC=0
Rp0 p0 vss 1G
Rp1 p1 vss 1G
Rps0 ps0 vss 60k
Rps1 ps1 vss 60k
Rnorm norm vss 1G
Mreset_p0 p0 rst vss vss NMOS W=4u L=180n
Mreset_p1 p1 rst vss vss NMOS W=4u L=180n
Mreset_ps0 ps0 rst vss vss NMOS W=4u L=180n
Mreset_ps1 ps1 rst vss vss NMOS W=4u L=180n
Mreset_norm norm rst vss vss NMOS W=4u L=180n
Mcd0 d0 gcond ps0 vss NSENSE W={cd_w0_u:.12g}u L=180n
Mcd1 d1 gcond ps1 vss NSENSE W={cd_w1_u:.12g}u L=180n
Msample0 ps0 phip p0 vss NSENSE W=8u L=180n
Msample1 ps1 phip p1 vss NSENSE W=8u L=180n
.ends xor_prob_conductance_divider2

.subckt xor_prob_soft_wta2 s0 s1 p0 p1 inh phip rst vdd vss params: CPROB=2f CINH=2f WEXC=96u WINH=6u WSUP=3u
* Soft-WTA candidate: each score charges its own evidence rail while both
* scores charge a shared inhibitory rail that subtracts activity during the
* probability phase. This is probability-like competition, not exact division.
Cp0 p0 vss {{CPROB}} IC=0
Cp1 p1 vss {{CPROB}} IC=0
Cinh inh vss {{CINH}} IC=0
Rp0 p0 vss 1G
Rp1 p1 vss 1G
Rinh inh vss 1G
Mreset_swta_p0 p0 rst vss vss NMOS W=4u L=180n
Mreset_swta_p1 p1 rst vss vss NMOS W=4u L=180n
Mreset_swta_inh inh rst vss vss NMOS W=4u L=180n
Rswta0_exc swta0_exc vss 1G
Rswta1_exc swta1_exc vss 1G
Mswta0_exc_v vdd s0 swta0_exc vss NSENSE W={{WEXC}} L=180n
Mswta1_exc_v vdd s1 swta1_exc vss NSENSE W={{WEXC}} L=180n
Mswta0_exc_phi swta0_exc phip p0 vss NSENSE W={{WEXC}} L=180n
Mswta1_exc_phi swta1_exc phip p1 vss NSENSE W={{WEXC}} L=180n
Rswta0_inh swta0_inh vss 1G
Rswta1_inh swta1_inh vss 1G
Mswta0_inh_v vdd s0 swta0_inh vss NSENSE W={{WINH}} L=180n
Mswta1_inh_v vdd s1 swta1_inh vss NSENSE W={{WINH}} L=180n
Mswta0_inh_phi swta0_inh phip inh vss NSENSE W={{WINH}} L=180n
Mswta1_inh_phi swta1_inh phip inh vss NSENSE W={{WINH}} L=180n
Rswta0_supp swta0_supp vss 1G
Rswta1_supp swta1_supp vss 1G
Mswta0_supp_v p0 inh swta0_supp vss NMOS W={{WSUP}} L=180n
Mswta1_supp_v p1 inh swta1_supp vss NMOS W={{WSUP}} L=180n
Mswta0_supp_phi swta0_supp phip vss vss NMOS W={{WSUP}} L=180n
Mswta1_supp_phi swta1_supp phip vss vss NMOS W={{WSUP}} L=180n
.ends xor_prob_soft_wta2

.subckt xor_label_routed_descent2 p0 p1 t0 t1 phie rst e0p e0n e1p e1n vdd vss params: CERR=1f WERR=96u
* Cross-entropy-shaped two-class descent routing from probability rails:
* target 0: e0p and e1n are driven from p1.
* target 1: e1p and e0n are driven from p0.
Ce0p e0p vss {{CERR}} IC=0
Ce0n e0n vss {{CERR}} IC=0
Ce1p e1p vss {{CERR}} IC=0
Ce1n e1n vss {{CERR}} IC=0
Re0p e0p vss 1G
Re0n e0n vss 1G
Re1p e1p vss 1G
Re1n e1n vss 1G
Mreset_e0p e0p rst vss vss NMOS W=4u L=180n
Mreset_e0n e0n rst vss vss NMOS W=4u L=180n
Mreset_e1p e1p rst vss vss NMOS W=4u L=180n
Mreset_e1n e1n rst vss vss NMOS W=4u L=180n
Re0p_a e0p_a vss 1G
Re0p_b e0p_b vss 1G
Me0p_t vdd t0 e0p_a vss NSENSE W={{WERR}} L=180n
Me0p_p e0p_a p1 e0p_b vss NSENSE W={{WERR}} L=180n
Me0p_phi e0p_b phie e0p vss NSENSE W={{WERR}} L=180n
Re1n_a e1n_a vss 1G
Re1n_b e1n_b vss 1G
Me1n_t vdd t0 e1n_a vss NSENSE W={{WERR}} L=180n
Me1n_p e1n_a p1 e1n_b vss NSENSE W={{WERR}} L=180n
Me1n_phi e1n_b phie e1n vss NSENSE W={{WERR}} L=180n
Re1p_a e1p_a vss 1G
Re1p_b e1p_b vss 1G
Me1p_t vdd t1 e1p_a vss NSENSE W={{WERR}} L=180n
Me1p_p e1p_a p0 e1p_b vss NSENSE W={{WERR}} L=180n
Me1p_phi e1p_b phie e1p vss NSENSE W={{WERR}} L=180n
Re0n_a e0n_a vss 1G
Re0n_b e0n_b vss 1G
Me0n_t vdd t1 e0n_a vss NSENSE W={{WERR}} L=180n
Me0n_p e0n_a p0 e0n_b vss NSENSE W={{WERR}} L=180n
Me0n_phi e0n_b phie e0n vss NSENSE W={{WERR}} L=180n
.ends xor_label_routed_descent2

.subckt xor_current_routed_descent2 rnorm rd0 rd1 t0 t1 phir e0p e0n e1p e1n vdd vss
* Current-domain cross-entropy-shaped two-class descent routing.
* The same fixed reference-current/conductance split creates branch mirror
* gates mir0/mir1.  Writer-domain outputs are active-low controls:
* target 0 discharges e0p and e1n from branch 1; target 1 discharges e1p and
* e0n from branch 0.
Vgcrd gcrd vss 0.62
Icrdref vdd rnorm PULSE(0 1u 1.0n 20p 20p 2.0n 20n)
Ce0p e0p vss 1f IC=1.2
Ce0n e0n vss 1f IC=1.2
Ce1p e1p vss 1f IC=1.2
Ce1n e1n vss 1f IC=1.2
Re0p e0p vdd 1G
Re0n e0n vdd 1G
Re1p e1p vdd 1G
Re1n e1n vdd 1G
Rrnorm rnorm vss 1G
Rmir0 mir0 vss 1G
Rmir1 mir1 vss 1G
Mcrd0 rd0 gcrd mir0 vss NSENSE W={cd_route_w0_u:.12g}u L=180n
Mcrd1 rd1 gcrd mir1 vss NSENSE W={cd_route_w1_u:.12g}u L=180n
Mcrd0_ref mir0 mir0 vss vss NMOS W=2u L=180n
Mcrd1_ref mir1 mir1 vss vss NMOS W=2u L=180n
Re0p_a e0p_a vss 1G
Re0p_b e0p_b vss 1G
Me0p_sink e0p mir1 e0p_a vss NMOS W=0.2u L=180n
Me0p_t e0p_a t0 e0p_b vss NSENSE W=0.2u L=180n
Me0p_phi e0p_b phir vss vss NSENSE W=0.2u L=180n
Re1n_a e1n_a vss 1G
Re1n_b e1n_b vss 1G
Me1n_sink e1n mir1 e1n_a vss NMOS W=0.2u L=180n
Me1n_t e1n_a t0 e1n_b vss NSENSE W=0.2u L=180n
Me1n_phi e1n_b phir vss vss NSENSE W=0.2u L=180n
Re1p_a e1p_a vss 1G
Re1p_b e1p_b vss 1G
Me1p_sink e1p mir0 e1p_a vss NMOS W=0.2u L=180n
Me1p_t e1p_a t1 e1p_b vss NSENSE W=0.2u L=180n
Me1p_phi e1p_b phir vss vss NSENSE W=0.2u L=180n
Re0n_a e0n_a vss 1G
Re0n_b e0n_b vss 1G
Me0n_sink e0n mir0 e0n_a vss NMOS W=0.2u L=180n
Me0n_t e0n_a t1 e0n_b vss NSENSE W=0.2u L=180n
Me0n_phi e0n_b phir vss vss NSENSE W=0.2u L=180n
.ends xor_current_routed_descent2

.subckt xor_current_to_writer_descent2 rnorm rd0 rd1 t0 t1 phir e0p e0n e1p e1n vdd vss
* Current-domain two-class descent routing into high-going writer rails.
* It uses the divider branch current to build mirror gates, discharges
* branch-local active-low controls, then PMOS-converts those controls into
* the same high-going errp/errn rail convention expected by the live writer:
* target 0: e0p and e1n charge from branch 1.
* target 1: e1p and e0n charge from branch 0.
Vgcrw gcrw vss 0.62
Icrwref vdd rnorm PULSE(0 1u 1.0n 20p 20p 2.0n 20n)
Ce0p e0p vss 2f IC=0
Ce0n e0n vss 2f IC=0
Ce1p e1p vss 2f IC=0
Ce1n e1n vss 2f IC=0
Re0p e0p vss 1G
Re0n e0n vss 1G
Re1p e1p vss 1G
Re1n e1n vss 1G
Rrnorm rnorm vss 1G
Rmir0 mir0 vss 1G
Rmir1 mir1 vss 1G
Cb0low b0low vss 1f IC=1.2
Cb1low b1low vss 1f IC=1.2
Rb0low b0low vdd 1G
Rb1low b1low vdd 1G
Mcrw0 rd0 gcrw mir0 vss NSENSE W={cd_route_w0_u:.12g}u L=180n
Mcrw1 rd1 gcrw mir1 vss NSENSE W={cd_route_w1_u:.12g}u L=180n
Mcrw0_ref mir0 mir0 vss vss NMOS W=2u L=180n
Mcrw1_ref mir1 mir1 vss vss NMOS W=2u L=180n
Rb0low_a b0low_a vss 1G
Mb0low_sink b0low mir0 b0low_a vss NMOS W=0.85u L=180n
Mb0low_phi b0low_a phir vss vss NSENSE W=0.85u L=180n
Rb1low_a b1low_a vss 1G
Mb1low_sink b1low mir1 b1low_a vss NMOS W=0.85u L=180n
Mb1low_phi b1low_a phir vss vss NSENSE W=0.85u L=180n
Re0p_a e0p_a vss 1G
Re0p_b e0p_b vss 1G
Me0p_m vdd b1low e0p_a vdd PMOS W=3u L=180n
Me0p_t e0p_a t0 e0p_b vss NSENSE W=12u L=180n
Me0p_phi e0p_b phir e0p vss NSENSE W=12u L=180n
Re1n_a e1n_a vss 1G
Re1n_b e1n_b vss 1G
Me1n_m vdd b1low e1n_a vdd PMOS W=3u L=180n
Me1n_t e1n_a t0 e1n_b vss NSENSE W=12u L=180n
Me1n_phi e1n_b phir e1n vss NSENSE W=12u L=180n
Re1p_a e1p_a vss 1G
Re1p_b e1p_b vss 1G
Me1p_m vdd b0low e1p_a vdd PMOS W=3u L=180n
Me1p_t e1p_a t1 e1p_b vss NSENSE W=12u L=180n
Me1p_phi e1p_b phir e1p vss NSENSE W=12u L=180n
Re0n_a e0n_a vss 1G
Re0n_b e0n_b vss 1G
Me0n_m vdd b0low e0n_a vdd PMOS W=3u L=180n
Me0n_t e0n_a t1 e0n_b vss NSENSE W=12u L=180n
Me0n_phi e0n_b phir e0n vss NSENSE W=12u L=180n
.ends xor_current_to_writer_descent2
""".strip()


def generate_netlist(
    *,
    approach: str,
    evidence: tuple[float, float],
    target: int,
    conductance_floor: float = 0.15,
    conductance_scale: float = 1.0,
    soft_score_base_v: float = 0.22,
    soft_score_step_v: float = 0.12,
    divider_error_mode: str = "sampled-voltage",
) -> str:
    if approach not in APPROACHES:
        raise ValueError(f"approach must be one of {APPROACHES}")
    g0_raw, g1_raw = _validate_pair("evidence", evidence)
    if target not in (0, 1):
        raise ValueError("target must be 0 or 1")
    if conductance_floor < 0.0:
        raise ValueError("conductance_floor must be nonnegative")
    if conductance_scale <= 0.0:
        raise ValueError("conductance_scale must be positive")
    if divider_error_mode not in DIVIDER_ERROR_MODES:
        raise ValueError(f"divider_error_mode must be one of {DIVIDER_ERROR_MODES}")
    if approach != "conductance-divider" and divider_error_mode != "sampled-voltage":
        raise ValueError("divider_error_mode is only available for conductance-divider")

    t0 = 1.2 if target == 0 else 0.0
    t1 = 1.2 if target == 1 else 0.0
    cd_w0 = g0_raw * conductance_scale
    cd_w1 = g1_raw * conductance_scale
    lines = [
        f"* Isolated XOR-scale output normalizer primitive: {approach}.",
        "* Python supplies only fixed synthetic evidence, label, clocks, and supplies.",
        ".param VDD=1.2",
        mos_models(),
        normalizer_subcircuits(
            conductance_w0=cd_w0,
            conductance_w1=cd_w1,
            conductance_floor=conductance_floor,
        ),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        "Vrst rst 0 PULSE(0 1.2 0.0n 10p 10p 0.40n 20n)",
        "Vphip phip 0 PULSE(0 1.2 1.0n 20p 20p 2.0n 20n)",
        "Vphie phie 0 PULSE(0 1.2 3.4n 20p 20p 1.8n 20n)",
        f"Vt0 t0 0 {t0:.12g}",
        f"Vt1 t1 0 {t1:.12g}",
    ]
    if approach == "conductance-divider":
        lines += [
            "Vsen0 norm d0 0",
            "Vsen1 norm d1 0",
            "Xprob p0 p1 norm d0 d1 phip rst vdd 0 xor_prob_conductance_divider2",
        ]
        if divider_error_mode == "current-routed":
            lines += [
                "Vrsen0 rnorm rd0 0",
                "Vrsen1 rnorm rd1 0",
                "Xroute rnorm rd0 rd1 t0 t1 phip e0p e0n e1p e1n vdd 0 xor_current_routed_descent2",
            ]
    else:
        s0 = soft_score_base_v + soft_score_step_v * g0_raw * conductance_scale
        s1 = soft_score_base_v + soft_score_step_v * g1_raw * conductance_scale
        if max(s0, s1) > 1.2:
            raise ValueError("soft-WTA score controls must stay within supply rails")
        lines += [
            f"Vs0 s0 0 {s0:.12g}",
            f"Vs1 s1 0 {s1:.12g}",
            "Xprob s0 s1 p0 p1 inh phip rst vdd 0 xor_prob_soft_wta2",
            ".meas tran inh_after FIND V(inh) AT=3.25n",
        ]

    if divider_error_mode == "sampled-voltage":
        lines += [
            "Xerr p0 p1 t0 t1 phie rst e0p e0n e1p e1n vdd 0 xor_label_routed_descent2",
        ]
    lines += [
        ".meas tran p0_after FIND V(p0) AT=3.25n",
        ".meas tran p1_after FIND V(p1) AT=3.25n",
        ".meas tran p_sum PARAM='p0_after+p1_after'",
        ".meas tran p0_frac PARAM='p0_after/(p0_after+p1_after)'",
        ".meas tran p1_frac PARAM='p1_after/(p0_after+p1_after)'",
        ".meas tran e0p_after FIND V(e0p) AT=5.45n",
        ".meas tran e0n_after FIND V(e0n) AT=5.45n",
        ".meas tran e1p_after FIND V(e1p) AT=5.45n",
        ".meas tran e1n_after FIND V(e1n) AT=5.45n",
    ]
    if divider_error_mode == "current-routed":
        lines += [
            ".meas tran e0_diff PARAM='e0n_after-e0p_after'",
            ".meas tran e1_diff PARAM='e1n_after-e1p_after'",
        ]
    else:
        lines += [
            ".meas tran e0_diff PARAM='e0p_after-e0n_after'",
            ".meas tran e1_diff PARAM='e1p_after-e1n_after'",
        ]
    if approach == "conductance-divider":
        lines += [
            ".meas tran i0_raw FIND I(Vsen0) AT=2.95n",
            ".meas tran i1_raw FIND I(Vsen1) AT=2.95n",
            ".meas tran norm_after FIND V(norm) AT=2.95n",
        ]
        if divider_error_mode == "current-routed":
            lines += [
                ".meas tran ir0_raw FIND I(Vrsen0) AT=2.95n",
                ".meas tran ir1_raw FIND I(Vrsen1) AT=2.95n",
                ".meas tran rnorm_after FIND V(rnorm) AT=2.95n",
            ]
    lines += [
        ".tran 2p 6n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def generate_writer_correction_netlist(
    *,
    evidence: tuple[float, float],
    target: int,
    conductance_floor: float = 0.15,
    conductance_scale: float = 1.0,
    initial_positive: float = 0.40,
    initial_negative: float = 0.40,
    high_ref: float = 0.48,
    low_ref: float = 0.22,
    update_width_u: float = 0.25,
) -> str:
    g0_raw, g1_raw = _validate_pair("evidence", evidence)
    if target not in (0, 1):
        raise ValueError("target must be 0 or 1")
    if conductance_floor < 0.0:
        raise ValueError("conductance_floor must be nonnegative")
    if conductance_scale <= 0.0:
        raise ValueError("conductance_scale must be positive")
    if min(initial_positive, initial_negative, high_ref, low_ref, update_width_u) < 0.0:
        raise ValueError("voltages and widths must be nonnegative")
    if not low_ref < high_ref <= 1.2:
        raise ValueError("references must satisfy low_ref < high_ref <= 1.2")

    t0 = 1.2 if target == 0 else 0.0
    t1 = 1.2 if target == 1 else 0.0
    cd_w0 = g0_raw * conductance_scale
    cd_w1 = g1_raw * conductance_scale
    lines = [
        "* XOR-scale conductance-divider current-to-writer correction primitive.",
        "* Synthetic evidence is normalized by branch current, then routed into the real PMOS writer.",
        ".param VDD=1.2",
        mos_models(),
        normalizer_subcircuits(
            conductance_w0=cd_w0,
            conductance_w1=cd_w1,
            conductance_floor=conductance_floor,
        ),
        ".options method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7",
        "Vdd vdd 0 {VDD}",
        f"Vvwhi_ref vwhi_ref 0 {high_ref:.12g}",
        f"Vvwlo_ref vwlo_ref 0 {low_ref:.12g}",
        "Vrst rst 0 PULSE(0 1.2 0.0n 10p 10p 0.40n 20n)",
        "Vphip phip 0 PULSE(0 1.2 1.0n 20p 20p 2.0n 20n)",
        "Velig elig0 0 PULSE(0 1.2 3.4n 20p 20p 1.8n 20n)",
        f"Vt0 t0 0 {t0:.12g}",
        f"Vt1 t1 0 {t1:.12g}",
        "Vrsen0 rnorm rd0 0",
        "Vrsen1 rnorm rd1 0",
        "Xwroute rnorm rd0 rd1 t0 t1 phip c0_errp c0_errn c1_errp c1_errn vdd 0 xor_current_to_writer_descent2",
    ]
    for class_idx in (0, 1):
        lines += [
            *signed_store_lines(
                positive_node=class_node(class_idx, "vwp0"),
                negative_node=class_node(class_idx, "vwn0"),
                positive_ic=initial_positive,
                negative_ic=initial_negative,
            ),
            *class_local_live_label_descent_update_lines(
                class_idx=class_idx,
                feature_idx=0,
                activation_node="elig0",
                positive_descent_node=class_node(class_idx, "errp"),
                negative_descent_node=class_node(class_idx, "errn"),
                width_u=update_width_u,
                high_side_topology="pmos-differential",
            ),
            f".meas tran c{class_idx}_vwp_before FIND V({class_node(class_idx, 'vwp0')}) AT=3.3n",
            f".meas tran c{class_idx}_vwn_before FIND V({class_node(class_idx, 'vwn0')}) AT=3.3n",
            f".meas tran c{class_idx}_signed_before PARAM='c{class_idx}_vwp_before-c{class_idx}_vwn_before'",
            f".meas tran c{class_idx}_vwp_after FIND V({class_node(class_idx, 'vwp0')}) AT=6.2n",
            f".meas tran c{class_idx}_vwn_after FIND V({class_node(class_idx, 'vwn0')}) AT=6.2n",
            f".meas tran c{class_idx}_signed_after PARAM='c{class_idx}_vwp_after-c{class_idx}_vwn_after'",
            f".meas tran c{class_idx}_signed_delta PARAM='c{class_idx}_signed_after-c{class_idx}_signed_before'",
            f".meas tran c{class_idx}_pos_ctrl_min MIN V(c{class_idx}_f0_live_pos_up_ctrl) FROM=3.4n TO=5.2n",
            f".meas tran c{class_idx}_neg_ctrl_min MIN V(c{class_idx}_f0_live_neg_up_ctrl) FROM=3.4n TO=5.2n",
        ]
    lines += [
        ".meas tran ir0_raw FIND I(Vrsen0) AT=2.95n",
        ".meas tran ir1_raw FIND I(Vrsen1) AT=2.95n",
        ".meas tran rnorm_after FIND V(rnorm) AT=2.95n",
        ".meas tran e0p_after FIND V(c0_errp) AT=3.25n",
        ".meas tran e0n_after FIND V(c0_errn) AT=3.25n",
        ".meas tran e1p_after FIND V(c1_errp) AT=3.25n",
        ".meas tran e1n_after FIND V(c1_errn) AT=3.25n",
        ".meas tran e0_diff PARAM='e0p_after-e0n_after'",
        ".meas tran e1_diff PARAM='e1p_after-e1n_after'",
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
