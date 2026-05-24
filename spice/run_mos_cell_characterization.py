#!/usr/bin/env python3
"""Run ngspice sanity checks for the MOS local-feature cell sketches.

The decks generated here are intentionally small transistor-level
characterizations of the paper panels.  They use simple Level-1 MOS models
and independent testbench rails so the resulting plots show local sign and
transfer behavior, not a calibrated process result.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "mos_cell_characterization"
FIGURE_DIR = ROOT / "docs" / "figures"

COMMON_MODELS = """
.options abstol=1e-12 reltol=1e-4 method=trap
.model NMOS NMOS (LEVEL=1 VTO=0.55 KP=220u LAMBDA=0.03)
.model PMOS PMOS (LEVEL=1 VTO=-0.55 KP=90u LAMBDA=0.03)
.param LCH=1u WN=24u WP=60u
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_ngspice(deck: str, stem: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    deck_path = RESULT_DIR / f"{stem}.cir"
    log_path = RESULT_DIR / f"{stem}.log"
    deck_path.write_text(deck)
    subprocess.run(
        ["ngspice", "-b", "-o", str(log_path), str(deck_path)],
        check=True,
        cwd=RESULT_DIR,
    )
    data_path = RESULT_DIR / f"{stem}.dat"
    # ngspice wrdata pads rows with a trailing space.  Keep committed
    # characterization data diff-clean without changing the numeric content.
    data_path.write_text("\n".join(line.rstrip() for line in data_path.read_text().splitlines()) + "\n")
    return data_path


def load_wrdata(path: Path, series_count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] == series_count + 1:
        return raw[:, 0], [raw[:, idx + 1] for idx in range(series_count)]
    if raw.shape[1] >= 2 * series_count:
        return raw[:, 0], [raw[:, 2 * idx + 1] for idx in range(series_count)]
    raise ValueError(f"cannot parse {path}: shape={raw.shape}, series={series_count}")


def save_plot(fig: plt.Figure, stem: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"{stem}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def characterize_synapse() -> Path:
    deck = f"""
* Signed MOS synapse transconductance sanity check
{COMMON_MODELS}
VCM cm 0 0.9
VDIFF xp xm -1.4
RXP xp cm 1G
RXM xm cm 1G

* Positive-weight copy: xp steers current to z+, xm to z-.
VWPP wpp 0 1.15
VZPP zpp 0 1.8
VZMP zmp 0 1.8
Mpp zpp xp tailp 0 NMOS L={{LCH}} W={{WN}}
Mpm zmp xm tailp 0 NMOS L={{LCH}} W={{WN}}
Mtp tailp wpp 0 0 NMOS L={{LCH}} W=12u

* Negative-weight copy swaps the input gates, reversing the contribution sign.
VWMN wmn 0 1.15
VZPN zpn 0 1.8
VZMN zmn 0 1.8
Mnp zpn xm tailn 0 NMOS L={{LCH}} W={{WN}}
Mnm zmn xp tailn 0 NMOS L={{LCH}} W={{WN}}
Mtn tailn wmn 0 0 NMOS L={{LCH}} W=12u

.control
set noaskquit
dc VDIFF -1.4 1.4 0.02
wrdata mos_synapse_slice.dat i(VZPP) i(VZMP) i(VZPN) i(VZMN) v(xp) v(xm)
quit
.endc
.end
"""
    store_deck = f"""
* Signed MOS synapse capacitive summing sanity check
{COMMON_MODELS}
.param CSUM=500p WTAIL=2u
VXP xp 0 1.15
VXM xm 0 0.65
VWP wpulse 0 PULSE(0 1.15 0.5u 20n 20n 0.7u 4.0u)

* Positive-weight copy: x+ discharges z+ more than z-, so z- - z+ rises.
CZPP zpp 0 {{CSUM}} IC=1.2
CZMP zmp 0 {{CSUM}} IC=1.2
RZPP zpp 0 100G
RZMP zmp 0 100G
MPP zpp xp tailp 0 NMOS L={{LCH}} W={{WN}}
MPM zmp xm tailp 0 NMOS L={{LCH}} W={{WN}}
MTP tailp wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}

* Negative-weight copy swaps gates, so z- - z+ falls for the same input.
CZPN zpn 0 {{CSUM}} IC=1.2
CZMN zmn 0 {{CSUM}} IC=1.2
RZPN zpn 0 100G
RZMN zmn 0 100G
MNP zpn xm tailn 0 NMOS L={{LCH}} W={{WN}}
MNM zmn xp tailn 0 NMOS L={{LCH}} W={{WN}}
MTN tailn wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}

.control
set noaskquit
tran 5n 2.5u uic
wrdata mos_synapse_store.dat v(zpp) v(zmp) v(zpn) v(zmn) v(wpulse)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_synapse_slice")
    xdiff, cols = load_wrdata(data, 6)
    # ngspice reports current through a voltage source using the source's
    # orientation.  Flip into the contribution convention used in the paper:
    # positive current means a positive signed contribution to z.
    pos_signed = cols[1] - cols[0]
    neg_signed = cols[3] - cols[2]
    require(np.mean(pos_signed[xdiff > 0.2]) > 0, "w+ branch should be positive for x+ > x-")
    require(np.mean(pos_signed[xdiff < -0.2]) < 0, "w+ branch should be negative for x+ < x-")
    require(np.mean(neg_signed[xdiff > 0.2]) < 0, "w- branch should be negative for x+ > x-")
    require(np.mean(neg_signed[xdiff < -0.2]) > 0, "w- branch should be positive for x+ < x-")
    peak = max(float(np.max(np.abs(pos_signed))), 1e-30)
    require(np.max(np.abs(pos_signed + neg_signed)) < 1e-6 * peak, "negative-weight copy should reverse sign")
    require(np.max(np.abs(pos_signed + pos_signed[::-1])) < 2e-3 * peak, "synapse transfer should be odd-symmetric")

    store_data = run_ngspice(store_deck, "mos_synapse_store")
    t, store_cols = load_wrdata(store_data, 5)
    pos_cap_signed = store_cols[1] - store_cols[0]
    neg_cap_signed = store_cols[3] - store_cols[2]

    def at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(t - time_s))])

    require(at(1.25e-6, pos_cap_signed) > 0.06, "w+ synapse should raise signed z storage for x+ > x-")
    require(at(1.25e-6, neg_cap_signed) < -0.06, "w- synapse should lower signed z storage for x+ > x-")
    require(at(2.1e-6, pos_cap_signed) > 0.06, "w+ signed z storage should hold after pulse")
    require(at(2.1e-6, neg_cap_signed) < -0.06, "w- signed z storage should hold after pulse")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2))
    axes[0].plot(xdiff, 1e6 * pos_signed, label="$w^+$ high")
    axes[0].plot(xdiff, 1e6 * neg_signed, label="$w^-$ high")
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_xlabel("$x^+ - x^-$ (V)")
    axes[0].set_ylabel("contribution current (uA)")
    axes[0].set_title("Signed synapse slice reverses contribution sign")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(1e6 * t, pos_cap_signed, label="$w^+$: stored $z^- - z^+$")
    axes[1].plot(1e6 * t, neg_cap_signed, label="$w^-$: stored $z^- - z^+$")
    axes[1].plot(1e6 * t, store_cols[4] / 8.0, color="0.5", alpha=0.45, label="$w_{gate}/8$")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("stored preactivation step (V)")
    axes[1].set_title("Same slice moves and holds summing capacitors")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return save_plot(fig, "mos_synapse_slice_ngspice")


def characterize_forward_pair() -> Path:
    transfer_deck = f"""
* MOS forward differential-pair sanity check
{COMMON_MODELS}
VDD vdd 0 1.8
VCM cm 0 0.9
VDIFF zp zm -0.7
RZP zp cm 1G
RZM zm cm 1G
VTAIL vbias 0 0.95
MP1 hp hp vdd vdd PMOS L={{LCH}} W={{WP}}
MP2 hm hm vdd vdd PMOS L={{LCH}} W={{WP}}
MN1 hp zp tail 0 NMOS L={{LCH}} W={{WN}}
MN2 hm zm tail 0 NMOS L={{LCH}} W={{WN}}
MNT tail vbias 0 0 NMOS L={{LCH}} W={{WN}}
.control
set noaskquit
dc VDIFF -0.7 0.7 0.01
wrdata mos_forward_pair.dat v(hp) v(hm)
quit
.endc
.end
"""
    store_deck = f"""
* MOS forward storage sanity check
{COMMON_MODELS}
.param CSTORE=10p WSW=24u
VDD vdd 0 1.8
VCM cm 0 0.9
VDIFF zp zm PWL(0 0 0.45u 0 0.5u 0.35 1.7u 0.35 1.75u -0.35 3.35u -0.35 3.4u 0 4u 0)
RZP zp cm 1G
RZM zm cm 1G
VTAIL vbias 0 0.95
VPACT pact 0 PULSE(0 1.8 0.5u 20n 20n 0.75u 2.0u)
MP1 hp hp vdd vdd PMOS L={{LCH}} W={{WP}}
MP2 hm hm vdd vdd PMOS L={{LCH}} W={{WP}}
MN1 hp zp tail 0 NMOS L={{LCH}} W={{WN}}
MN2 hm zm tail 0 NMOS L={{LCH}} W={{WN}}
MNT tail vbias 0 0 NMOS L={{LCH}} W={{WN}}
MSP hp pact hcp 0 NMOS L={{LCH}} W={{WSW}}
MSM hm pact hcm 0 NMOS L={{LCH}} W={{WSW}}
CHP hcp 0 {{CSTORE}} IC=1.04
CHM hcm 0 {{CSTORE}} IC=1.04
RLEAKP hcp 0 50G
RLEAKM hcm 0 50G
.control
set noaskquit
tran 5n 4u uic
wrdata mos_forward_store.dat v(hp) v(hm) v(hcp) v(hcm) v(pact)
quit
.endc
.end
"""
    data = run_ngspice(transfer_deck, "mos_forward_pair")
    x, cols = load_wrdata(data, 2)
    xdiff = x
    signed = cols[1] - cols[0]
    require(np.all(np.diff(signed) >= -1e-4), "forward pair transfer should be monotone")
    require(abs(signed[np.argmin(np.abs(xdiff))]) < 1e-3, "forward pair should be centered near zero")
    peak = max(float(np.max(np.abs(signed))), 1e-30)
    require(np.max(np.abs(signed + signed[::-1])) < 0.03 * peak, "forward pair should be approximately odd-symmetric")
    store_data = run_ngspice(store_deck, "mos_forward_store")
    t, store_cols = load_wrdata(store_data, 5)
    load_signed = store_cols[1] - store_cols[0]
    cap_signed = store_cols[3] - store_cols[2]

    def at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(t - time_s))])

    require(at(1.2e-6, cap_signed) > 0.25, "activation cap should store positive phase")
    require(at(1.8e-6, cap_signed) > 0.25, "activation cap should hold after first pact")
    require(at(3.2e-6, cap_signed) < -0.18, "activation cap should store negative phase")
    require(at(3.8e-6, cap_signed) < -0.18, "activation cap should hold after second pact")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))
    # The diode-connected PMOS load is voltage-inverting: the rail that sinks
    # more differential-pair current moves lower.  Plot the usable signed load
    # voltage convention so the transfer rises with z+ - z-.
    axes[0].plot(xdiff, signed, label="$h^- - h^+$")
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_xlabel("$z^+ - z^-$ (V)")
    axes[0].set_ylabel("load differential voltage (V)")
    axes[0].set_title("Forward pair gives monotone bounded transfer")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(1e6 * t, load_signed, label="load $h^- - h^+$", alpha=0.75)
    axes[1].plot(1e6 * t, cap_signed, label="stored $C_{h^-}-C_{h^+}$")
    axes[1].plot(1e6 * t, store_cols[4] / 5.0, color="0.5", alpha=0.45, label="$pact/5$")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("differential voltage (V)")
    axes[1].set_title("Phase switch stores and holds activation on capacitors")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return save_plot(fig, "mos_forward_pair_ngspice")


def characterize_hidden_error() -> Path:
    eps = 0.01
    deck = f"""
* Hidden-error finite-difference replica-pair sanity check.
* The small-signal nudge is applied as a differential perturbation around a
* fixed common mode, matching the MOS forward transfer plotted above.
{COMMON_MODELS}
.param EPS={eps}
VDD vdd 0 1.8
VCM cm 0 0.9
VDIFF zp zm -0.7
RZP zp cm 1G
RZM zm cm 1G
VTAIL vbias 0 0.95

* Testbench nudges around fixed common mode: p-copy uses differential input
* d+eps, m-copy uses d-eps.
VZPP zpp zp {{EPS/2}}
VZMM zm zmm {{EPS/2}}
VZPM zp zpm {{EPS/2}}
VZMP zmp zm {{EPS/2}}

* Positive nudge replica.
MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zmm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

* Negative nudge replica.
MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zmp tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

.control
set noaskquit
dc VDIFF -0.7 0.7 0.01
wrdata mos_hidden_error.dat v(hpp) v(hpm) v(hmp) v(hmm)
quit
.endc
.end
"""
    store_deck = f"""
* Hidden-error sign-selection storage sanity check.
* This is not the finite-difference subtractor; it checks that MOS switches can
* store the positive or negative use of a forward-replica error voltage on
* differential hidden-error capacitors.
{COMMON_MODELS}
.param CERR=10p WSW=24u
VDD vdd 0 1.8
VCM cm 0 0.9
VDIFF zp zm 0.18
RZP zp cm 1G
RZM zm cm 1G
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
VRP rp 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
VRM rm 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)

* Positive-error copy: store h- on delta+ and h+ on delta-.
MPP1 hp_p hp_p vdd vdd PMOS L={{LCH}} W={{WP}}
MPP2 hm_p hm_p vdd vdd PMOS L={{LCH}} W={{WP}}
MNP1 hp_p zp tail_p 0 NMOS L={{LCH}} W={{WN}}
MNP2 hm_p zm tail_p 0 NMOS L={{LCH}} W={{WN}}
MNTP tail_p vbias 0 0 NMOS L={{LCH}} W={{WN}}
CDPP cdp_p 0 {{CERR}} IC=1.04
CDPM cdm_p 0 {{CERR}} IC=1.04
RDP cdp_p 0 50G
RDM cdm_p 0 50G
MSP1 hm_p pbwd nsp1 0 NMOS L={{LCH}} W={{WSW}}
MSP2 nsp1 rp cdp_p 0 NMOS L={{LCH}} W={{WSW}}
MSP3 hp_p pbwd nsp2 0 NMOS L={{LCH}} W={{WSW}}
MSP4 nsp2 rp cdm_p 0 NMOS L={{LCH}} W={{WSW}}

* Negative-error copy swaps the storage rails.
MPN1 hp_n hp_n vdd vdd PMOS L={{LCH}} W={{WP}}
MPN2 hm_n hm_n vdd vdd PMOS L={{LCH}} W={{WP}}
MNN1 hp_n zp tail_n 0 NMOS L={{LCH}} W={{WN}}
MNN2 hm_n zm tail_n 0 NMOS L={{LCH}} W={{WN}}
MNTN tail_n vbias 0 0 NMOS L={{LCH}} W={{WN}}
CDNP cdp_n 0 {{CERR}} IC=1.04
CDNM cdm_n 0 {{CERR}} IC=1.04
RNP cdp_n 0 50G
RNM cdm_n 0 50G
MSN1 hm_n pbwd nsn1 0 NMOS L={{LCH}} W={{WSW}}
MSN2 nsn1 rm cdm_n 0 NMOS L={{LCH}} W={{WSW}}
MSN3 hp_n pbwd nsn2 0 NMOS L={{LCH}} W={{WSW}}
MSN4 nsn2 rm cdp_n 0 NMOS L={{LCH}} W={{WSW}}

.control
set noaskquit
tran 5n 2.5u uic
wrdata mos_hidden_error_store.dat v(hp_p) v(hm_p) v(cdp_p) v(cdm_p) v(cdp_n) v(cdm_n) v(pbwd)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_hidden_error")
    x, cols = load_wrdata(data, 4)
    xdiff = x
    signed_plus_nudge = cols[1] - cols[0]
    signed_minus_nudge = cols[3] - cols[2]
    gain = (signed_plus_nudge - signed_minus_nudge) / (2.0 * eps)
    center_gain = gain[np.argmin(np.abs(xdiff))]
    edge_gain = max(float(np.mean(gain[xdiff < -0.45])), float(np.mean(gain[xdiff > 0.45])))
    require(center_gain > 0.5, "hidden-error derivative gain should be positive near z balance")
    require(edge_gain < 0.65 * center_gain, "hidden-error derivative gain should fall at saturated z")
    require(np.all(gain > -1e-4), "finite-difference gain should stay nonnegative")

    store_data = run_ngspice(store_deck, "mos_hidden_error_store")
    t, store_cols = load_wrdata(store_data, 7)
    pos_stored = store_cols[2] - store_cols[3]
    neg_stored = store_cols[4] - store_cols[5]

    def at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(t - time_s))])

    require(at(1.2e-6, pos_stored) > 0.12, "r+ storage should make positive delta rail differential")
    require(at(1.2e-6, neg_stored) < -0.12, "r- storage should make negative delta rail differential")
    require(at(2.2e-6, pos_stored) > 0.12, "r+ hidden-error storage should hold after phase")
    require(at(2.2e-6, neg_stored) < -0.12, "r- hidden-error storage should hold after phase")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2))
    axes[0].plot(xdiff, gain, label="finite-difference gain")
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_xlabel("$z^+ - z^-$ (V)")
    axes[0].set_ylabel("small-signal gain (V/V)")
    axes[0].set_title("Hidden-error replicas produce derivative window")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(1e6 * t, pos_stored, label="$r^+$: stored $\\delta^+ - \\delta^-$")
    axes[1].plot(1e6 * t, neg_stored, label="$r^-$: stored $\\delta^+ - \\delta^-$")
    axes[1].plot(1e6 * t, store_cols[6] / 10.0, color="0.5", alpha=0.45, label="$pbwd/10$")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("stored hidden-error step (V)")
    axes[1].set_title("MOS pass switches store selected error sign")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return save_plot(fig, "mos_hidden_error_ngspice")


def characterize_reset_precharge() -> Path:
    deck = f"""
* MOS reset/precharge sanity check for reusable state capacitors.
{COMMON_MODELS}
.param CSUM=500p WTAIL=2u WRESETN=24u WRESETP=60u
VDD vdd 0 1.8
VCM vcm 0 0.9
VXP xp 0 1.15
VXM xm 0 0.65
VRST rst 0 PWL(0 0 0.10u 0 0.12u 1.8 0.55u 1.8 0.57u 0 1.80u 0 1.82u 1.8 2.25u 1.8 2.27u 0 3u 0)
VRSTN rstn 0 PWL(0 1.8 0.10u 1.8 0.12u 0 0.55u 0 0.57u 1.8 1.80u 1.8 1.82u 0 2.25u 0 2.27u 1.8 3u 1.8)
VWP wpulse 0 PWL(0 0 0.75u 0 0.77u 1.15 1.45u 1.15 1.47u 0 3u 0)

* Deliberately mismatched preactivation state.
CZP zp 0 {{CSUM}} IC=1.3
CZM zm 0 {{CSUM}} IC=0.5
RZP zp 0 100G
RZM zm 0 100G

* Additional state roles that must reuse the same reset primitive.
CHP hp 0 {{CSUM}} IC=1.25
CHM hm 0 {{CSUM}} IC=0.55
RHP hp 0 100G
RHM hm 0 100G
CDP dp 0 {{CSUM}} IC=0.55
CDM dm 0 {{CSUM}} IC=1.25
RDP dp 0 100G
RDM dm 0 100G

* Transmission-gate reset to common mode.  A single NMOS pass device leaves a
* visible threshold error here, so the reusable primitive uses complementary
* MOS devices.
MRPN zp rst vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRMN zm rst vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRPP zp rstn vcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRMP zm rstn vcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRHN hp rst vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRHM hm rst vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRHP hp rstn vcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRHMP hm rstn vcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRDN dp rst vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRDM dm rst vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRDP dp rstn vcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRDMP dm rstn vcm vdd PMOS L={{LCH}} W={{WRESETP}}

* Reuse the positive signed synapse slice between reset phases.
MPP zp xp tail 0 NMOS L={{LCH}} W={{WN}}
MPM zm xm tail 0 NMOS L={{LCH}} W={{WN}}
MTP tail wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}

.control
set noaskquit
tran 5n 3u uic
wrdata mos_reset_precharge.dat v(zp) v(zm) v(hp) v(hm) v(dp) v(dm) v(rst) v(wpulse)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_reset_precharge")
    t, cols = load_wrdata(data, 8)
    zp, zm, hp, hm, dp, dm, rst, wpulse = cols
    signed = zm - zp
    common = 0.5 * (zp + zm)
    h_signed = hm - hp
    d_signed = dm - dp

    def at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(t - time_s))])

    require(at(0.0, signed) < -0.7, "reset test should start from deliberately mismatched caps")
    require(at(0.0, h_signed) < -0.6, "activation reset test should start mismatched")
    require(at(0.0, d_signed) > 0.6, "hidden-error reset test should start mismatched")
    require(abs(at(0.55e-6, signed)) < 0.06, "first reset should remove most differential state")
    require(abs(at(0.55e-6, common) - 0.9) < 0.05, "first reset should restore common mode")
    require(abs(at(0.55e-6, h_signed)) < 0.06, "first reset should clear activation state")
    require(abs(at(0.55e-6, d_signed)) < 0.06, "first reset should clear hidden-error state")
    require(at(1.65e-6, signed) > 0.05, "synapse write should create a reusable signed state")
    require(abs(at(1.65e-6, h_signed)) < 0.06, "activation state should stay near reset during z write")
    require(abs(at(1.65e-6, d_signed)) < 0.06, "hidden-error state should stay near reset during z write")
    require(abs(at(2.25e-6, signed)) < 0.015, "second reset should clear written differential state")
    require(abs(at(2.25e-6, common) - 0.9) < 0.02, "second reset should restore common mode")
    require(abs(at(2.25e-6, h_signed)) < 0.015, "second reset should keep activation state clear")
    require(abs(at(2.25e-6, d_signed)) < 0.015, "second reset should keep hidden-error state clear")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    axes[0].plot(1e6 * t, zp, label="$z^+$ cap")
    axes[0].plot(1e6 * t, zm, label="$z^-$ cap")
    axes[0].plot(1e6 * t, rst / 2.0, color="0.5", alpha=0.45, label="$reset/2$")
    axes[0].set_ylabel("cap voltage (V)")
    axes[0].set_title("Transmission-gate reset restores preactivation common mode")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(1e6 * t, h_signed, label="activation $h^- - h^+$")
    axes[1].plot(1e6 * t, d_signed, label="hidden error $\\delta^- - \\delta^+$")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_ylabel("differential voltage (V)")
    axes[1].set_title("Same reset phase clears activation and hidden-error state")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[2].plot(1e6 * t, signed, label="stored $z^- - z^+$")
    axes[2].plot(1e6 * t, wpulse / 8.0, color="0.5", alpha=0.45, label="$w_{gate}/8$")
    axes[2].plot(1e6 * t, rst / 10.0, color="0.25", alpha=0.35, label="$reset/10$")
    axes[2].axhline(0, color="0.4", linewidth=0.8)
    axes[2].set_xlabel("time (us)")
    axes[2].set_ylabel("differential voltage (V)")
    axes[2].set_title("Reset, write, and reset again without Python capacitor forcing")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    return save_plot(fig, "mos_reset_precharge_ngspice")


def characterize_writer() -> Path:
    deck = f"""
* Four-quadrant writer coincidence sanity check.
{COMMON_MODELS}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VPACC_N paccn 0 PULSE(1.8 0 0.5u 20n 20n 0.8u 5.0u)
VHI hi 0 1.8
VLO lo 0 0

* Same-sign x+/dh+ copy: selected branch charges W+.
CWP_PP wp_pp 0 {{CWRITE}} IC=0.85
CWM_PP wm_pp 0 {{CWRITE}} IC=0.85
MPPPA vdd paccn n1pp vdd PMOS L={{LCH}} W={{WWRITE}}
MPPPB n1pp lo n2pp vdd PMOS L={{LCH}} W={{WWRITE}}
MPPPC n2pp lo wp_pp vdd PMOS L={{LCH}} W={{WWRITE}}
MPPMIA vdd paccn n3pp vdd PMOS L={{LCH}} W={{WWRITE}}
MPPMIB n3pp lo n4pp vdd PMOS L={{LCH}} W={{WWRITE}}
MPPMIC n4pp hi wm_pp vdd PMOS L={{LCH}} W={{WWRITE}}

* Same-sign x-/dh- copy: the other same-sign branch also charges W+.
CWP_MM wp_mm 0 {{CWRITE}} IC=0.85
CWM_MM wm_mm 0 {{CWRITE}} IC=0.85
MMMPA vdd paccn n1mm vdd PMOS L={{LCH}} W={{WWRITE}}
MMMPB n1mm lo n2mm vdd PMOS L={{LCH}} W={{WWRITE}}
MMMPC n2mm lo wp_mm vdd PMOS L={{LCH}} W={{WWRITE}}
MMMMIA vdd paccn n3mm vdd PMOS L={{LCH}} W={{WWRITE}}
MMMMIB n3mm lo n4mm vdd PMOS L={{LCH}} W={{WWRITE}}
MMMMIC n4mm hi wm_mm vdd PMOS L={{LCH}} W={{WWRITE}}

* Opposite-sign x+/dh- copy: selected branch charges W-.
CWP_PM wp_pm 0 {{CWRITE}} IC=0.85
CWM_PM wm_pm 0 {{CWRITE}} IC=0.85
MPMPIA vdd paccn n1pm vdd PMOS L={{LCH}} W={{WWRITE}}
MPMPIB n1pm lo n2pm vdd PMOS L={{LCH}} W={{WWRITE}}
MPMPIC n2pm hi wp_pm vdd PMOS L={{LCH}} W={{WWRITE}}
MPMMA vdd paccn n3pm vdd PMOS L={{LCH}} W={{WWRITE}}
MPMMB n3pm lo n4pm vdd PMOS L={{LCH}} W={{WWRITE}}
MPMMC n4pm lo wm_pm vdd PMOS L={{LCH}} W={{WWRITE}}

* Opposite-sign x-/dh+ copy: the other opposite-sign branch also charges W-.
CWP_MP wp_mp 0 {{CWRITE}} IC=0.85
CWM_MP wm_mp 0 {{CWRITE}} IC=0.85
MMPPIA vdd paccn n1mp vdd PMOS L={{LCH}} W={{WWRITE}}
MMPPIB n1mp lo n2mp vdd PMOS L={{LCH}} W={{WWRITE}}
MMPPIC n2mp hi wp_mp vdd PMOS L={{LCH}} W={{WWRITE}}
MMPMA vdd paccn n3mp vdd PMOS L={{LCH}} W={{WWRITE}}
MMPMB n3mp lo n4mp vdd PMOS L={{LCH}} W={{WWRITE}}
MMPMC n4mp lo wm_mp vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 10n 4u uic
wrdata mos_writer.dat v(wp_pp) v(wm_pp) v(wp_mm) v(wm_mm) v(wp_pm) v(wm_pm) v(wp_mp) v(wm_mp) v(paccn)
quit
.endc
.end
"""
    widths_us = [0.10, 0.20, 0.40, 0.80, 1.20]
    sweep_devices = []
    sweep_prints = []
    for idx, width in enumerate(widths_us):
        sweep_devices.append(
            f"""
VPACC{idx} paccn{idx} 0 PULSE(1.8 0 0.5u 20n 20n {width}u 5.0u)
CWP{idx} wp{idx} 0 {{CWRITE}} IC=0.85
CWM{idx} wm{idx} 0 {{CWRITE}} IC=0.85
MWP{idx}A vdd paccn{idx} n1_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWP{idx}B n1_{idx} lo n2_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWP{idx}C n2_{idx} lo wp{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM{idx}A vdd paccn{idx} n3_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM{idx}B n3_{idx} lo n4_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM{idx}C n4_{idx} hi wm{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        sweep_prints.extend([f"v(wp{idx})", f"v(wm{idx})"])
    sweep_deck = f"""
* Four-quadrant writer pulse-width sweep.
{COMMON_MODELS}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VHI hi 0 1.8
VLO lo 0 0
{''.join(sweep_devices)}
.control
set noaskquit
tran 5n 2.2u uic
wrdata mos_writer_width.dat {' '.join(sweep_prints)}
quit
.endc
.end
"""
    readback_deck = f"""
* Writer-to-synapse readback sanity check.
{COMMON_MODELS}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VCM cm 0 0.9
VDIFF xp xm 0.5
RXP xp cm 1G
RXM xm cm 1G
VPACC_N paccn 0 PULSE(1.8 0 0.5u 20n 20n 0.8u 5.0u)
VHI hi 0 1.8
VLO lo 0 0

* Same-sign writer charges W+ in the same voltage range used by the positive
* synapse tail-gate read path below.
CWP_P wp_p 0 {{CWRITE}} IC=0.85
CWM_P wm_p 0 {{CWRITE}} IC=0.85
MWPP1 vdd paccn n1p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPP2 n1p lo n2p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPP3 n2p lo wp_p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM1 vdd paccn n3p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM2 n3p lo n4p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM3 n4p hi wm_p vdd PMOS L={{LCH}} W={{WWRITE}}

VZPP zpp 0 1.8
VZMP zmp 0 1.8
MPP zpp xp tailp 0 NMOS L={{LCH}} W={{WN}}
MPM zmp xm tailp 0 NMOS L={{LCH}} W={{WN}}
MTP tailp wp_p 0 0 NMOS L={{LCH}} W=12u

* Opposite-sign writer charges W-.  The read copy swaps differential-pair
* input gates, matching the negative-weight synapse convention.
CWP_N wp_n 0 {{CWRITE}} IC=0.85
CWM_N wm_n 0 {{CWRITE}} IC=0.85
MWNP1 vdd paccn n1n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNP2 n1n lo n2n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNP3 n2n hi wp_n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM1 vdd paccn n3n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM2 n3n lo n4n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM3 n4n lo wm_n vdd PMOS L={{LCH}} W={{WWRITE}}

VZPN zpn 0 1.8
VZMN zmn 0 1.8
MNP zpn xm tailn 0 NMOS L={{LCH}} W={{WN}}
MNM zmn xp tailn 0 NMOS L={{LCH}} W={{WN}}
MTN tailn wm_n 0 0 NMOS L={{LCH}} W=12u

.control
set noaskquit
tran 5n 2.0u uic
wrdata mos_writer_readback.dat v(wp_p) v(wm_p) i(VZPP) i(VZMP) v(wp_n) v(wm_n) i(VZPN) i(VZMN) v(paccn)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_writer")
    t, cols = load_wrdata(data, 9)
    wp_pp_delta = cols[0][-1] - cols[0][0]
    wm_pp_delta = cols[1][-1] - cols[1][0]
    wp_mm_delta = cols[2][-1] - cols[2][0]
    wm_mm_delta = cols[3][-1] - cols[3][0]
    wp_pm_delta = cols[4][-1] - cols[4][0]
    wm_pm_delta = cols[5][-1] - cols[5][0]
    wp_mp_delta = cols[6][-1] - cols[6][0]
    wm_mp_delta = cols[7][-1] - cols[7][0]
    same_selected = np.array([wp_pp_delta, wp_mm_delta])
    same_inactive = np.array([wm_pp_delta, wm_mm_delta])
    opp_selected = np.array([wm_pm_delta, wm_mp_delta])
    opp_inactive = np.array([wp_pm_delta, wp_mp_delta])
    require(np.all((same_selected > 0.05) & (same_selected < 0.25)), "both same-sign branches should make bounded W+ steps")
    require(np.max(np.abs(same_inactive)) < 1e-3, "same-sign branches should leave W- quiet")
    require(np.all((opp_selected > 0.05) & (opp_selected < 0.25)), "both opposite-sign branches should make bounded W- steps")
    require(np.max(np.abs(opp_inactive)) < 1e-3, "opposite-sign branches should leave W+ quiet")

    sweep_data = run_ngspice(sweep_deck, "mos_writer_width")
    _, sweep_cols = load_wrdata(sweep_data, 2 * len(widths_us))
    selected_delta = np.array([sweep_cols[2 * idx][-1] - sweep_cols[2 * idx][0] for idx in range(len(widths_us))])
    inactive_delta = np.array([sweep_cols[2 * idx + 1][-1] - sweep_cols[2 * idx + 1][0] for idx in range(len(widths_us))])
    require(np.all(np.diff(selected_delta) > 0.0), "writer charge should increase with active pulse width")
    require(np.max(np.abs(inactive_delta)) < 1e-3, "inactive writer branch should stay quiet in width sweep")
    # This is an incremental writer, not a rail-to-rail latch.  Over this small
    # voltage excursion the selected charge should be close to proportional to
    # coincidence time.
    fit = np.polyfit(np.array(widths_us), selected_delta, 1)
    predicted = np.polyval(fit, widths_us)
    residual = selected_delta - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((selected_delta - np.mean(selected_delta)) ** 2))
    require(1.0 - ss_res / ss_tot > 0.98, "writer pulse-width response should be near-linear")

    readback_data = run_ngspice(readback_deck, "mos_writer_readback")
    rt, read_cols = load_wrdata(readback_data, 9)
    pos_read_contrib = read_cols[3] - read_cols[2]
    neg_read_contrib = read_cols[7] - read_cols[6]
    baseline_idx = int(np.argmin(np.abs(rt - 0.4e-6)))
    final_idx = int(np.argmin(np.abs(rt - 1.8e-6)))
    baseline_pos = float(pos_read_contrib[baseline_idx])
    final_pos = float(pos_read_contrib[final_idx])
    baseline_neg = float(neg_read_contrib[baseline_idx])
    final_neg = float(neg_read_contrib[final_idx])
    require(read_cols[0][-1] > 0.90, "writer readback should leave W+ in active positive-synapse range")
    require(abs(read_cols[1][-1] - 0.85) < 1e-3, "positive readback should leave inactive W- unchanged")
    require(abs(read_cols[4][-1] - 0.85) < 1e-3, "negative readback should leave inactive W+ unchanged")
    require(read_cols[5][-1] > 0.90, "writer readback should leave W- in active negative-synapse range")
    require(final_pos > baseline_pos + 30e-6, "written W+ should increase positive synapse contribution")
    require(final_neg < baseline_neg - 30e-6, "written W- should increase negative synapse contribution magnitude")

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.4))
    axes[0].plot(1e6 * t, cols[0], label="$x^+\\delta^+ \\to W^+$")
    axes[0].plot(1e6 * t, cols[2], label="$x^-\\delta^- \\to W^+$")
    axes[0].plot(1e6 * t, cols[1], "--", color="0.5", alpha=0.75, label="inactive $W^-$ branches")
    axes[0].plot(1e6 * t, cols[3], ":", color="0.5", alpha=0.75)
    axes[0].set_ylabel("cap voltage (V)")
    axes[0].set_title("Both same-sign coincidence branches make bounded W+ steps")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(1e6 * t, cols[5], label="$x^+\\delta^- \\to W^-$")
    axes[1].plot(1e6 * t, cols[7], label="$x^-\\delta^+ \\to W^-$")
    axes[1].plot(1e6 * t, cols[4], "--", color="0.5", alpha=0.75, label="inactive $W^+$ branches")
    axes[1].plot(1e6 * t, cols[6], ":", color="0.5", alpha=0.75)
    axes[1].plot(1e6 * t, cols[8] / 6.0, color="0.5", alpha=0.45, label="$\\overline{pacc}/6$")
    axes[1].set_ylabel("cap voltage (V)")
    axes[1].set_title("Both opposite-sign coincidence branches make bounded W- steps")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[2].plot(widths_us, selected_delta, "o-", label="selected $W^+$ step")
    axes[2].plot(widths_us, inactive_delta, "o-", label="inactive $W^-$ step")
    axes[2].plot(widths_us, predicted, "--", color="0.35", label="linear fit")
    axes[2].axhline(0, color="0.4", linewidth=0.8)
    axes[2].set_xlabel("active coincidence time (us)")
    axes[2].set_ylabel("$\\Delta V_W$ (V)")
    axes[2].set_title("Incremental write magnitude follows pulse width")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    axes[3].plot(1e6 * rt, 1e6 * pos_read_contrib, label="$W^+$ read contribution")
    axes[3].plot(1e6 * rt, 1e6 * neg_read_contrib, label="$W^-$ read contribution")
    axes[3].plot(1e6 * rt, 100.0 * (read_cols[0] - read_cols[0][0]), label="$100\\Delta W^+$")
    axes[3].plot(1e6 * rt, -100.0 * (read_cols[5] - read_cols[5][0]), label="$-100\\Delta W^-$")
    axes[3].plot(1e6 * rt, read_cols[8] / 6.0, color="0.5", alpha=0.45, label="$\\overline{pacc}/6$")
    axes[3].set_xlabel("time (us)")
    axes[3].set_ylabel("current (uA) / scaled V")
    axes[3].set_title("Written W+ and W- voltages read back with opposite signs")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend()
    fig.tight_layout()
    return save_plot(fig, "mos_writer_ngspice")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["synapse", "forward", "hidden", "reset", "writer"],
        help="Run only one characterization.",
    )
    args = parser.parse_args()
    jobs = {
        "synapse": characterize_synapse,
        "forward": characterize_forward_pair,
        "hidden": characterize_hidden_error,
        "reset": characterize_reset_precharge,
        "writer": characterize_writer,
    }
    selected = [args.only] if args.only else list(jobs)
    for name in selected:
        path = jobs[name]()
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
