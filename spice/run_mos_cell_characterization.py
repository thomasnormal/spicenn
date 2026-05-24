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
    def synapse_cm_devices(name: str, cm: float) -> str:
        return f"""
VCM_{name} cm_{name} 0 {cm:.2f}
VDIFF_{name} xp_{name} xm_{name} PWL(0 -1.4 10u 1.4)
RXP_{name} xp_{name} cm_{name} 1G
RXM_{name} xm_{name} cm_{name} 1G

VWPP_{name} wpp_{name} 0 1.15
VZPP_{name} zpp_{name} 0 1.8
VZMP_{name} zmp_{name} 0 1.8
MPP_{name} zpp_{name} xp_{name} tailp_{name} 0 NMOS L={{LCH}} W={{WN}}
MPM_{name} zmp_{name} xm_{name} tailp_{name} 0 NMOS L={{LCH}} W={{WN}}
MTP_{name} tailp_{name} wpp_{name} 0 0 NMOS L={{LCH}} W=12u

VWMN_{name} wmn_{name} 0 1.15
VZPN_{name} zpn_{name} 0 1.8
VZMN_{name} zmn_{name} 0 1.8
MNP_{name} zpn_{name} xm_{name} tailn_{name} 0 NMOS L={{LCH}} W={{WN}}
MNM_{name} zmn_{name} xp_{name} tailn_{name} 0 NMOS L={{LCH}} W={{WN}}
MTN_{name} tailn_{name} wmn_{name} 0 0 NMOS L={{LCH}} W=12u
"""

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
    cm_cases = [("low", 0.75), ("nominal", 0.90), ("high", 1.05)]
    cm_prints = []
    for name, _cm in cm_cases:
        cm_prints.extend(
            [
                f"v(xp_{name})",
                f"v(xm_{name})",
                f"i(VZPP_{name})",
                f"i(VZMP_{name})",
                f"i(VZPN_{name})",
                f"i(VZMN_{name})",
            ]
        )
    cm_deck = f"""
* Signed MOS synapse input common-mode margin check
{COMMON_MODELS}
{''.join(synapse_cm_devices(name, cm) for name, cm in cm_cases)}
.control
set noaskquit
tran 20n 10u
wrdata mos_synapse_common_mode.dat {' '.join(cm_prints)}
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

* Equal positive and negative copies on the same summing capacitors should
* cancel differentially while still drawing common-mode current.
CZPC zpc 0 {{CSUM}} IC=1.2
CZMC zmc 0 {{CSUM}} IC=1.2
RZPC zpc 0 100G
RZMC zmc 0 100G
MCPP zpc xp tailcp 0 NMOS L={{LCH}} W={{WN}}
MCPM zmc xm tailcp 0 NMOS L={{LCH}} W={{WN}}
MCTP tailcp wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}
MCNP zpc xm tailcn 0 NMOS L={{LCH}} W={{WN}}
MCNM zmc xp tailcn 0 NMOS L={{LCH}} W={{WN}}
MCTN tailcn wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}

.control
set noaskquit
tran 5n 2.5u uic
wrdata mos_synapse_store.dat v(zpp) v(zmp) v(zpn) v(zmn) v(zpc) v(zmc) v(wpulse)
quit
.endc
.end
"""
    weighted_cases = [
        ("wponly", 1.15, 0.00),
        ("wpstrong", 1.15, 1.08),
        ("balanced", 1.10, 1.10),
        ("wmstrong", 1.08, 1.15),
        ("wmonly", 0.00, 1.15),
    ]
    weighted_devices = []
    weighted_prints = []
    for name, wp_amp, wm_amp in weighted_cases:
        weighted_devices.append(
            f"""
VWP_{name} wp_{name} 0 PULSE(0 {wp_amp:.2f} 0.5u 20n 20n 0.7u 4.0u)
VWM_{name} wm_{name} 0 PULSE(0 {wm_amp:.2f} 0.5u 20n 20n 0.7u 4.0u)
CZP_{name} zp_{name} 0 {{CSUM}} IC=1.2
CZM_{name} zm_{name} 0 {{CSUM}} IC=1.2
RZP_{name} zp_{name} 0 100G
RZM_{name} zm_{name} 0 100G
MPP_{name} zp_{name} xp tailp_{name} 0 NMOS L={{LCH}} W={{WN}}
MPM_{name} zm_{name} xm tailp_{name} 0 NMOS L={{LCH}} W={{WN}}
MTP_{name} tailp_{name} wp_{name} 0 0 NMOS L={{LCH}} W={{WTAIL}}
MNP_{name} zp_{name} xm tailn_{name} 0 NMOS L={{LCH}} W={{WN}}
MNM_{name} zm_{name} xp tailn_{name} 0 NMOS L={{LCH}} W={{WN}}
MTN_{name} tailn_{name} wm_{name} 0 0 NMOS L={{LCH}} W={{WTAIL}}
"""
        )
        weighted_prints.extend([f"v(zp_{name})", f"v(zm_{name})"])
    weighted_deck = f"""
* Signed MOS synapse weighted cancellation sanity check.
{COMMON_MODELS}
.param CSUM=500p WTAIL=2u
VXP xp 0 1.15
VXM xm 0 0.65
{''.join(weighted_devices)}
.control
set noaskquit
tran 5n 2.5u uic
wrdata mos_synapse_weighted.dat {' '.join(weighted_prints)} v(wp_wponly)
quit
.endc
.end
"""
    magnitude_deck = f"""
* Signed MOS synapse analog weight-gate magnitude sanity check.
{COMMON_MODELS}
VXP xp 0 1.15
VXM xm 0 0.65
VWG wg 0 0.0

* Positive-weight copy: higher tail-gate voltage increases the positive
* signed contribution for x+ > x-.
VZPP zpp 0 1.8
VZMP zmp 0 1.8
MPP zpp xp tailp 0 NMOS L={{LCH}} W={{WN}}
MPM zmp xm tailp 0 NMOS L={{LCH}} W={{WN}}
MTP tailp wg 0 0 NMOS L={{LCH}} W=12u

* Negative-weight copy swaps the differential gates.  The same analog tail
* voltage should increase magnitude with the opposite signed convention.
VZPN zpn 0 1.8
VZMN zmn 0 1.8
MNP zpn xm tailn 0 NMOS L={{LCH}} W={{WN}}
MNM zmn xp tailn 0 NMOS L={{LCH}} W={{WN}}
MTN tailn wg 0 0 NMOS L={{LCH}} W=12u

.control
set noaskquit
dc VWG 0.0 1.25 0.01
wrdata mos_synapse_weight_gate.dat v(wg) i(VZPP) i(VZMP) i(VZPN) i(VZMN)
quit
.endc
.end
"""
    mismatch_cases = [
        ("left_low", "NMOS_LO", "-20 mV left VTO"),
        ("balanced", "NMOS", "matched"),
        ("left_high", "NMOS_HI", "+20 mV left VTO"),
    ]
    mismatch_devices = []
    mismatch_prints = []
    for name, left_model, _label in mismatch_cases:
        mismatch_devices.append(
            f"""
VCM_MM_{name} cm_mm_{name} 0 0.90
VDIFF_MM_{name} xp_mm_{name} xm_mm_{name} PWL(0 -0.5 10u 0.5)
RXP_MM_{name} xp_mm_{name} cm_mm_{name} 1G
RXM_MM_{name} xm_mm_{name} cm_mm_{name} 1G

VWPP_MM_{name} wpp_mm_{name} 0 1.15
VZPP_MM_{name} zpp_mm_{name} 0 1.8
VZMP_MM_{name} zmp_mm_{name} 0 1.8
MPP_MM_{name} zpp_mm_{name} xp_mm_{name} tailp_mm_{name} 0 {left_model} L={{LCH}} W={{WN}}
MPM_MM_{name} zmp_mm_{name} xm_mm_{name} tailp_mm_{name} 0 NMOS L={{LCH}} W={{WN}}
MTP_MM_{name} tailp_mm_{name} wpp_mm_{name} 0 0 NMOS L={{LCH}} W=12u

VWMN_MM_{name} wmn_mm_{name} 0 1.15
VZPN_MM_{name} zpn_mm_{name} 0 1.8
VZMN_MM_{name} zmn_mm_{name} 0 1.8
MNP_MM_{name} zpn_mm_{name} xm_mm_{name} tailn_mm_{name} 0 {left_model} L={{LCH}} W={{WN}}
MNM_MM_{name} zmn_mm_{name} xp_mm_{name} tailn_mm_{name} 0 NMOS L={{LCH}} W={{WN}}
MTN_MM_{name} tailn_mm_{name} wmn_mm_{name} 0 0 NMOS L={{LCH}} W=12u
"""
        )
        mismatch_prints.extend(
            [
                f"v(xp_mm_{name})",
                f"v(xm_mm_{name})",
                f"i(VZPP_MM_{name})",
                f"i(VZMP_MM_{name})",
                f"i(VZPN_MM_{name})",
                f"i(VZMN_MM_{name})",
            ]
        )
    mismatch_deck = f"""
* Signed MOS synapse threshold-mismatch sanity check.
* The physical z+ branch input transistor is intentionally skewed by +/-20 mV
* VTO in both the positive and gate-swapped negative copies.  The goal is
* bounded offset and preserved sign outside the offset window, not perfect
* cancellation under mismatch.
{COMMON_MODELS}
.model NMOS_LO NMOS (LEVEL=1 VTO=0.53 KP=220u LAMBDA=0.03)
.model NMOS_HI NMOS (LEVEL=1 VTO=0.57 KP=220u LAMBDA=0.03)
{''.join(mismatch_devices)}
.control
set noaskquit
tran 20n 10u
wrdata mos_synapse_mismatch.dat {' '.join(mismatch_prints)}
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

    cm_data = run_ngspice(cm_deck, "mos_synapse_common_mode")
    _cmt, cm_cols = load_wrdata(cm_data, 6 * len(cm_cases))
    cm_curves: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray]] = []
    for idx, (name, cm) in enumerate(cm_cases):
        xp = cm_cols[6 * idx]
        xm = cm_cols[6 * idx + 1]
        cm_pos_signed = cm_cols[6 * idx + 3] - cm_cols[6 * idx + 2]
        cm_neg_signed = cm_cols[6 * idx + 5] - cm_cols[6 * idx + 4]
        cm_xdiff = xp - xm
        cm_peak = max(float(np.max(np.abs(cm_pos_signed))), 1e-30)
        cm_zero = float(cm_pos_signed[np.argmin(np.abs(cm_xdiff))])
        require(np.mean(cm_pos_signed[cm_xdiff > 0.2]) > 0, f"{name} VCM w+ branch should stay positive")
        require(np.mean(cm_pos_signed[cm_xdiff < -0.2]) < 0, f"{name} VCM w+ branch should stay negative")
        require(np.mean(cm_neg_signed[cm_xdiff > 0.2]) < 0, f"{name} VCM w- branch should stay negative")
        require(np.mean(cm_neg_signed[cm_xdiff < -0.2]) > 0, f"{name} VCM w- branch should stay positive")
        require(cm_peak > 0.5e-6, f"{name} VCM synapse branch should retain useful current swing")
        require(abs(cm_zero) < 0.05 * cm_peak, f"{name} VCM synapse transfer should stay centered")
        require(np.max(np.abs(cm_pos_signed + cm_neg_signed)) < 0.02 * cm_peak, f"{name} VCM w- copy should reverse sign")
        cm_curves.append((name, cm, cm_xdiff, cm_pos_signed, cm_neg_signed))

    store_data = run_ngspice(store_deck, "mos_synapse_store")
    t, store_cols = load_wrdata(store_data, 7)
    pos_cap_signed = store_cols[1] - store_cols[0]
    neg_cap_signed = store_cols[3] - store_cols[2]
    cancel_cap_signed = store_cols[5] - store_cols[4]

    def at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(t - time_s))])

    require(at(1.25e-6, pos_cap_signed) > 0.06, "w+ synapse should raise signed z storage for x+ > x-")
    require(at(1.25e-6, neg_cap_signed) < -0.06, "w- synapse should lower signed z storage for x+ > x-")
    require(abs(at(1.25e-6, cancel_cap_signed)) < 0.01, "equal w+ and w- branches should cancel on shared z storage")
    require(at(2.1e-6, pos_cap_signed) > 0.06, "w+ signed z storage should hold after pulse")
    require(at(2.1e-6, neg_cap_signed) < -0.06, "w- signed z storage should hold after pulse")
    require(abs(at(2.1e-6, cancel_cap_signed)) < 0.01, "cancelled shared z storage should hold near zero")

    weighted_data = run_ngspice(weighted_deck, "mos_synapse_weighted")
    wt, weighted_cols = load_wrdata(weighted_data, 2 * len(weighted_cases) + 1)
    weighted_signed = [
        weighted_cols[2 * idx + 1] - weighted_cols[2 * idx]
        for idx in range(len(weighted_cases))
    ]
    weighted_final = np.array([float(series[np.argmin(np.abs(wt - 1.25e-6))]) for series in weighted_signed])
    weighted_hold = np.array([float(series[np.argmin(np.abs(wt - 2.1e-6))]) for series in weighted_signed])
    require(weighted_final[0] > 0.06, "W+ only weighted synapse case should be positive")
    require(weighted_final[4] < -0.06, "W- only weighted synapse case should be negative")
    require(weighted_final[1] > 0.015, "W+ stronger than W- should leave a positive net preactivation")
    require(weighted_final[3] < -0.015, "W- stronger than W+ should leave a negative net preactivation")
    require(abs(weighted_final[2]) < 0.01, "balanced W+ and W- weighted synapse case should cancel")
    require(np.all(np.diff(weighted_final) < -0.01), "weighted synapse net should decrease as W- dominates")
    require(np.max(np.abs(weighted_hold - weighted_final)) < 0.005, "weighted synapse states should hold after pulse")

    magnitude_data = run_ngspice(magnitude_deck, "mos_synapse_weight_gate")
    _mg_sweep, magnitude_cols = load_wrdata(magnitude_data, 5)
    weight_gate = magnitude_cols[0]
    mag_pos_signed = magnitude_cols[2] - magnitude_cols[1]
    mag_neg_signed = magnitude_cols[4] - magnitude_cols[3]
    active = weight_gate >= 0.65
    inactive = weight_gate <= 0.45
    require(np.max(np.abs(mag_pos_signed[inactive])) < 1e-9, "inactive W+ tail gate should be near off")
    require(np.max(np.abs(mag_neg_signed[inactive])) < 1e-9, "inactive W- tail gate should be near off")
    require(np.all(mag_pos_signed[active] > 0.0), "active W+ tail gate should keep positive contribution sign")
    require(np.all(mag_neg_signed[active] < 0.0), "active W- tail gate should keep negative contribution sign")
    require(
        np.all(np.diff(mag_pos_signed[active]) > -2e-9),
        "W+ contribution magnitude should not fall as analog tail gate increases",
    )
    require(
        np.all(np.diff(-mag_neg_signed[active]) > -2e-9),
        "W- contribution magnitude should not fall as analog tail gate increases",
    )
    require(
        mag_pos_signed[-1] > mag_pos_signed[active][0] + 5e-6,
        "W+ analog tail-gate sweep should have useful dynamic range",
    )
    require(
        -mag_neg_signed[-1] > -mag_neg_signed[active][0] + 5e-6,
        "W- analog tail-gate sweep should have useful dynamic range",
    )
    require(
        np.max(np.abs(mag_pos_signed + mag_neg_signed)) < 0.02 * np.max(np.abs(mag_pos_signed)),
        "W- analog tail-gate copy should mirror W+ magnitude with reversed sign",
    )

    mismatch_data = run_ngspice(mismatch_deck, "mos_synapse_mismatch")
    _mt, mismatch_cols = load_wrdata(mismatch_data, 6 * len(mismatch_cases))
    mismatch_curves: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    mismatch_offsets = []
    mismatch_neg_offsets = []
    for idx, (name, _left_model, label) in enumerate(mismatch_cases):
        mxp = mismatch_cols[6 * idx]
        mxm = mismatch_cols[6 * idx + 1]
        mxdiff = mxp - mxm
        mpos = mismatch_cols[6 * idx + 3] - mismatch_cols[6 * idx + 2]
        mneg = mismatch_cols[6 * idx + 5] - mismatch_cols[6 * idx + 4]

        def zero_crossing(series: np.ndarray, series_name: str) -> float:
            changes = np.where(np.diff(np.signbit(series)))[0]
            require(len(changes) == 1, f"{name} {series_name} mismatch transfer should have one zero crossing")
            cross_idx = int(changes[0])
            x0, x1 = mxdiff[cross_idx], mxdiff[cross_idx + 1]
            y0, y1 = series[cross_idx], series[cross_idx + 1]
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))

        pos_zero = zero_crossing(mpos, "W+")
        neg_zero = zero_crossing(mneg, "W-")
        mismatch_offsets.append(pos_zero)
        mismatch_neg_offsets.append(neg_zero)
        require(np.mean(mpos[mxdiff > 0.12]) > 20e-6, f"{name} W+ mismatch branch should stay positive past offset margin")
        require(np.mean(mpos[mxdiff < -0.12]) < -20e-6, f"{name} W+ mismatch branch should stay negative past offset margin")
        require(np.mean(mneg[mxdiff > 0.12]) < -20e-6, f"{name} W- mismatch branch should stay negative past offset margin")
        require(np.mean(mneg[mxdiff < -0.12]) > 20e-6, f"{name} W- mismatch branch should stay positive past offset margin")
        require(np.max(mpos) > 100e-6 and np.min(mpos) < -100e-6, f"{name} W+ mismatch branch should retain useful swing")
        require(np.max(mneg) > 100e-6 and np.min(mneg) < -100e-6, f"{name} W- mismatch branch should retain useful swing")
        mismatch_curves.append((name, label, mxdiff, mpos, mneg))

    low_offset, balanced_offset, high_offset = mismatch_offsets
    require(abs(balanced_offset) < 0.01, "matched synapse zero crossing should stay near zero")
    require(low_offset < balanced_offset - 0.012, "low VTO z+ branch should move W+ zero crossing negative")
    require(high_offset > balanced_offset + 0.012, "high VTO z+ branch should move W+ zero crossing positive")
    require(np.max(np.abs(mismatch_offsets)) < 0.06, "20 mV synapse input mismatch should keep W+ offset bounded")
    require(np.max(np.abs(np.array(mismatch_offsets) + np.array(mismatch_neg_offsets))) < 0.01, "gate-swapped W- mismatch offsets should mirror W+")

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
    axes[1].plot(1e6 * t, cancel_cap_signed, label="$w^+ + w^-$ shared caps")
    axes[1].plot(1e6 * t, store_cols[6] / 8.0, color="0.5", alpha=0.45, label="$w_{gate}/8$")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("stored preactivation step (V)")
    axes[1].set_title("Signed branches add on shared summing capacitors")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    synapse_plot = save_plot(fig, "mos_synapse_slice_ngspice")

    weighted_fig, weighted_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    labels = [
        "$W^+$ only",
        "$W^+ > W^-$",
        "$W^+ = W^-$",
        "$W^- > W^+$",
        "$W^-$ only",
    ]
    for label, series in zip(labels, weighted_signed):
        weighted_axes[0].plot(1e6 * wt, series, label=label)
    weighted_axes[0].plot(1e6 * wt, weighted_cols[-1] / 8.0, color="0.5", alpha=0.35, label="$w^+$ pulse/8")
    weighted_axes[0].axhline(0, color="0.4", linewidth=0.8)
    weighted_axes[0].set_ylabel("stored $z^- - z^+$ (V)")
    weighted_axes[0].set_title("Unequal W+ and W- tails partially cancel on shared summing caps")
    weighted_axes[0].grid(True, alpha=0.25)
    weighted_axes[0].legend(loc="upper right", ncol=2)
    imbalance = np.array([wp - wm for _, wp, wm in weighted_cases])
    order = np.argsort(imbalance)
    weighted_axes[1].plot(imbalance[order], weighted_final[order], "o-", label="after write")
    weighted_axes[1].plot(imbalance[order], weighted_hold[order], "s--", label="after hold")
    weighted_axes[1].axhline(0, color="0.4", linewidth=0.8)
    weighted_axes[1].axvline(0, color="0.4", linewidth=0.8)
    weighted_axes[1].set_xlabel("$V_{W^+} - V_{W^-}$ during pulse (V)")
    weighted_axes[1].set_ylabel("stored preactivation (V)")
    weighted_axes[1].set_title("Net stored sign follows weight-rail imbalance")
    weighted_axes[1].grid(True, alpha=0.25)
    weighted_axes[1].legend()
    weighted_fig.tight_layout()
    save_plot(weighted_fig, "mos_synapse_weighted_ngspice")

    magnitude_fig, magnitude_axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    magnitude_axes[0].plot(weight_gate, 1e6 * mag_pos_signed, label="$W^+$ tail")
    magnitude_axes[0].plot(weight_gate, -1e6 * mag_neg_signed, "s--", markevery=8, label="$W^-$ tail magnitude")
    magnitude_axes[0].axhline(0, color="0.4", linewidth=0.8)
    magnitude_axes[0].axvline(0.55, color="0.5", linewidth=0.8, linestyle=":", label="$V_{TO}$")
    magnitude_axes[0].set_ylabel("signed magnitude (uA)")
    magnitude_axes[0].set_title("Synapse current grows with analog weight-tail voltage")
    magnitude_axes[0].grid(True, alpha=0.25)
    magnitude_axes[0].legend(loc="upper left")
    magnitude_axes[1].plot(weight_gate, 1e6 * mag_pos_signed, label="$W^+$ contribution")
    magnitude_axes[1].plot(weight_gate, 1e6 * mag_neg_signed, label="$W^-$ contribution")
    magnitude_axes[1].axhline(0, color="0.4", linewidth=0.8)
    magnitude_axes[1].axvline(0.55, color="0.5", linewidth=0.8, linestyle=":")
    magnitude_axes[1].set_xlabel("analog weight-tail gate (V)")
    magnitude_axes[1].set_ylabel("contribution current (uA)")
    magnitude_axes[1].set_title("Gate-swapped copy preserves opposite sign over the sweep")
    magnitude_axes[1].grid(True, alpha=0.25)
    magnitude_axes[1].legend(loc="upper left")
    magnitude_fig.tight_layout()
    save_plot(magnitude_fig, "mos_synapse_weight_gate_ngspice")

    mismatch_fig, mismatch_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for (name, label, mxdiff, mpos, mneg), xzero in zip(mismatch_curves, mismatch_offsets):
        mismatch_axes[0].plot(mxdiff, 1e6 * mpos, label=f"$W^+$ {label}")
        mismatch_axes[0].axvline(xzero, linewidth=0.8, linestyle=":", alpha=0.65)
    mismatch_axes[0].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].axvline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].set_xlabel("$x^+ - x^-$ (V)")
    mismatch_axes[0].set_ylabel("contribution current (uA)")
    mismatch_axes[0].set_title("Synapse input-pair mismatch shifts contribution zero")
    mismatch_axes[0].grid(True, alpha=0.25)
    mismatch_axes[0].legend(loc="upper left")
    labels_mm = [case[2] for case in mismatch_cases]
    mismatch_axes[1].plot(labels_mm, mismatch_offsets, "o-", label="$W^+$ zero")
    mismatch_axes[1].plot(labels_mm, mismatch_neg_offsets, "s--", label="$W^-$ zero")
    mismatch_axes[1].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[1].set_xlabel("z+ branch threshold skew")
    mismatch_axes[1].set_ylabel("zero crossing $x^+ - x^-$ (V)")
    mismatch_axes[1].set_title("Gate-swapped copy mirrors the mismatch offset")
    mismatch_axes[1].grid(True, axis="y", alpha=0.25)
    mismatch_axes[1].legend(loc="upper left")
    mismatch_fig.tight_layout()
    save_plot(mismatch_fig, "mos_synapse_mismatch_ngspice")

    cm_fig, cm_axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    for name, cm, cm_xdiff, cm_pos_signed, cm_neg_signed in cm_curves:
        cm_axes[0].plot(cm_xdiff, 1e6 * cm_pos_signed, label=f"$w^+$ VCM={cm:.2f} V")
        cm_axes[1].plot(cm_xdiff, 1e6 * cm_neg_signed, label=f"$w^-$ VCM={cm:.2f} V")
    for ax in cm_axes:
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.axvline(0, color="0.4", linewidth=0.8)
        ax.set_ylabel("contribution current (uA)")
        ax.grid(True, alpha=0.25)
    cm_axes[0].set_title("Positive-weight synapse sign is stable across input common-mode")
    cm_axes[1].set_title("Gate-swapped negative-weight copy reverses sign across input common-mode")
    cm_axes[1].set_xlabel("$x^+ - x^-$ (V)")
    cm_axes[0].legend(loc="upper left")
    cm_axes[1].legend(loc="lower left")
    cm_fig.tight_layout()
    save_plot(cm_fig, "mos_synapse_common_mode_ngspice")
    return synapse_plot


def characterize_forward_pair() -> Path:
    def forward_pair_devices(suffix: str, cm: float) -> str:
        return f"""
VCM_{suffix} cm_{suffix} 0 {cm:.2f}
VDIFF_{suffix} zp_{suffix} zm_{suffix} PWL(0 -0.7 10u 0.7)
RZP_{suffix} zp_{suffix} cm_{suffix} 1G
RZM_{suffix} zm_{suffix} cm_{suffix} 1G
MP1_{suffix} hp_{suffix} hp_{suffix} vdd vdd PMOS L={{LCH}} W={{WP}}
MP2_{suffix} hm_{suffix} hm_{suffix} vdd vdd PMOS L={{LCH}} W={{WP}}
MN1_{suffix} hp_{suffix} zp_{suffix} tail_{suffix} 0 NMOS L={{LCH}} W={{WN}}
MN2_{suffix} hm_{suffix} zm_{suffix} tail_{suffix} 0 NMOS L={{LCH}} W={{WN}}
MNT_{suffix} tail_{suffix} vbias 0 0 NMOS L={{LCH}} W={{WN}}
"""

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
    cm_cases = [("low", 0.75), ("nominal", 0.90), ("high", 1.05)]
    cm_prints = []
    for name, _cm in cm_cases:
        cm_prints.extend([f"v(zp_{name})", f"v(zm_{name})", f"v(hp_{name})", f"v(hm_{name})"])
    cm_deck = f"""
* MOS forward differential-pair common-mode margin check
{COMMON_MODELS}
VDD vdd 0 1.8
VTAIL vbias 0 0.95
{''.join(forward_pair_devices(name, cm) for name, cm in cm_cases)}
.control
set noaskquit
tran 20n 10u
wrdata mos_forward_common_mode.dat {' '.join(cm_prints)}
quit
.endc
.end
"""
    store_sweep_cases = [-0.45, -0.25, 0.00, 0.25, 0.45]
    store_sweep_devices = []
    store_sweep_prints = []
    for idx, diff in enumerate(store_sweep_cases):
        name = f"d{idx}"
        zp = 0.9 + diff / 2.0
        zm = 0.9 - diff / 2.0
        store_sweep_devices.append(
            f"""
* Forward-store magnitude copy for z+ - z- = {diff:.2f} V.
VZP_{name} zp_{name} 0 {zp:.5f}
VZM_{name} zm_{name} 0 {zm:.5f}
MP1_{name} hp_{name} hp_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MP2_{name} hm_{name} hm_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MN1_{name} hp_{name} zp_{name} tail_{name} 0 NMOS L={{LCH}} W={{WN}}
MN2_{name} hm_{name} zm_{name} tail_{name} 0 NMOS L={{LCH}} W={{WN}}
MNT_{name} tail_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}
MSP_{name} hp_{name} pact hcp_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSM_{name} hm_{name} pact hcm_{name} 0 NMOS L={{LCH}} W={{WSW}}
CHP_{name} hcp_{name} 0 {{CSTORE}} IC=1.04
CHM_{name} hcm_{name} 0 {{CSTORE}} IC=1.04
RLEAKP_{name} hcp_{name} 0 50G
RLEAKM_{name} hcm_{name} 0 50G
"""
        )
        store_sweep_prints.extend([f"v(hp_{name})", f"v(hm_{name})", f"v(hcp_{name})", f"v(hcm_{name})"])
    store_sweep_deck = f"""
* MOS forward-store magnitude sweep.
* Static diode-loaded forward-pair copies drive real activation capacitors
* through the same pact pass switches, checking that stored activation is
* graded and retained rather than only sign-selective.
{COMMON_MODELS}
.param CSTORE=10p WSW=24u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPACT pact 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
{''.join(store_sweep_devices)}
.control
set noaskquit
tran 5n 2.5u uic
wrdata mos_forward_store_sweep.dat {' '.join(store_sweep_prints)} v(pact)
quit
.endc
.end
"""
    mismatch_cases = [
        ("left_low", "NMOS_LO", "NMOS", "-20 mV left VTO"),
        ("balanced", "NMOS", "NMOS", "matched"),
        ("left_high", "NMOS_HI", "NMOS", "+20 mV left VTO"),
    ]
    mismatch_devices = []
    mismatch_prints = []
    for name, left_model, right_model, _label in mismatch_cases:
        mismatch_devices.append(
            f"""
VCM_MM_{name} cm_mm_{name} 0 0.90
VDIFF_MM_{name} zp_mm_{name} zm_mm_{name} PWL(0 -0.5 10u 0.5)
RZP_MM_{name} zp_mm_{name} cm_mm_{name} 1G
RZM_MM_{name} zm_mm_{name} cm_mm_{name} 1G
MP1_MM_{name} hp_mm_{name} hp_mm_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MP2_MM_{name} hm_mm_{name} hm_mm_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MN1_MM_{name} hp_mm_{name} zp_mm_{name} tail_mm_{name} 0 {left_model} L={{LCH}} W={{WN}}
MN2_MM_{name} hm_mm_{name} zm_mm_{name} tail_mm_{name} 0 {right_model} L={{LCH}} W={{WN}}
MNT_MM_{name} tail_mm_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}
"""
        )
        mismatch_prints.extend([f"v(zp_mm_{name})", f"v(zm_mm_{name})", f"v(hp_mm_{name})", f"v(hm_mm_{name})"])
    mismatch_deck = f"""
* MOS forward differential-pair threshold-mismatch sanity check.
* The input pair is intentionally skewed by +/-20 mV VTO on the left device.
* The goal is not zero offset, but monotone transfer with bounded crossing
* shift and usable center gain.
{COMMON_MODELS}
.model NMOS_LO NMOS (LEVEL=1 VTO=0.53 KP=220u LAMBDA=0.03)
.model NMOS_HI NMOS (LEVEL=1 VTO=0.57 KP=220u LAMBDA=0.03)
VDD vdd 0 1.8
VTAIL vbias 0 0.95
{''.join(mismatch_devices)}
.control
set noaskquit
tran 20n 10u
wrdata mos_forward_mismatch.dat {' '.join(mismatch_prints)}
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

    cm_data = run_ngspice(cm_deck, "mos_forward_common_mode")
    _cmt, cm_cols = load_wrdata(cm_data, 4 * len(cm_cases))
    cm_curves: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    for idx, (name, cm) in enumerate(cm_cases):
        zp = cm_cols[4 * idx]
        zm = cm_cols[4 * idx + 1]
        hp = cm_cols[4 * idx + 2]
        hm = cm_cols[4 * idx + 3]
        cm_xdiff = zp - zm
        cm_signed = hm - hp
        cm_curve_peak = max(float(np.max(np.abs(cm_signed))), 1e-30)
        cm_zero = float(cm_signed[np.argmin(np.abs(cm_xdiff))])
        mid = int(np.argmin(np.abs(cm_xdiff)))
        local_lo = max(0, mid - 5)
        local_hi = min(len(cm_xdiff), mid + 6)
        local_fit = np.polyfit(cm_xdiff[local_lo:local_hi], cm_signed[local_lo:local_hi], 1)
        require(np.all(np.diff(cm_signed) >= -2e-3), f"{name} common-mode forward transfer should be monotone")
        require(abs(cm_zero) < 0.015, f"{name} common-mode forward transfer should stay centered")
        require(cm_curve_peak > 0.12, f"{name} common-mode forward transfer should retain usable swing")
        require(local_fit[0] > 0.25, f"{name} common-mode forward transfer should retain center gain")
        cm_curves.append((name, cm, cm_xdiff, cm_signed))

    store_sweep_data = run_ngspice(store_sweep_deck, "mos_forward_store_sweep")
    st, store_sweep_cols = load_wrdata(store_sweep_data, 4 * len(store_sweep_cases) + 1)
    sweep_load = []
    sweep_cap = []
    for idx in range(len(store_sweep_cases)):
        sweep_load.append(store_sweep_cols[4 * idx + 1] - store_sweep_cols[4 * idx])
        sweep_cap.append(store_sweep_cols[4 * idx + 3] - store_sweep_cols[4 * idx + 2])

    def sat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(st - time_s))])

    sweep_load_final = np.array([sat(1.2e-6, series) for series in sweep_load])
    sweep_cap_final = np.array([sat(1.2e-6, series) for series in sweep_cap])
    sweep_cap_hold = np.array([sat(2.2e-6, series) for series in sweep_cap])
    require(np.all(np.diff(sweep_cap_final) > 0.05), "stored forward activation should increase with preactivation")
    require(abs(sweep_cap_final[len(store_sweep_cases) // 2]) < 0.01, "stored forward activation should be centered")
    require(sweep_cap_final[0] < -0.20 and sweep_cap_final[-1] > 0.20, "stored forward activation should have useful swing")
    require(
        np.max(np.abs(sweep_cap_final + sweep_cap_final[::-1])) < 0.04 * np.max(np.abs(sweep_cap_final)),
        "stored forward activation should remain approximately odd-symmetric",
    )
    require(
        np.max(np.abs(sweep_cap_final - sweep_cap_hold)) < 0.01,
        "stored forward activation sweep should hold after pact closes",
    )
    require(
        np.max(np.abs(sweep_cap_final - sweep_load_final)) < 0.015,
        "stored forward activation should track the diode-loaded pair output",
    )

    mismatch_data = run_ngspice(mismatch_deck, "mos_forward_mismatch")
    _mt, mismatch_cols = load_wrdata(mismatch_data, 4 * len(mismatch_cases))
    mismatch_curves: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    mismatch_offsets = []
    mismatch_gains = []
    for idx, (name, _left_model, _right_model, label) in enumerate(mismatch_cases):
        mzp = mismatch_cols[4 * idx]
        mzm = mismatch_cols[4 * idx + 1]
        mhp = mismatch_cols[4 * idx + 2]
        mhm = mismatch_cols[4 * idx + 3]
        mxdiff = mzp - mzm
        msigned = mhm - mhp
        require(np.all(np.diff(msigned) >= -2e-3), f"{name} mismatch transfer should remain monotone")
        require(np.max(msigned) > 0.22 and np.min(msigned) < -0.22, f"{name} mismatch transfer should retain swing")
        require(
            float(np.mean(msigned[mxdiff > 0.25])) > 0.18,
            f"{name} mismatch transfer should still be positive beyond offset margin",
        )
        require(
            float(np.mean(msigned[mxdiff < -0.25])) < -0.18,
            f"{name} mismatch transfer should still be negative beyond offset margin",
        )
        sign_changes = np.where(np.diff(np.signbit(msigned)))[0]
        require(len(sign_changes) == 1, f"{name} mismatch transfer should have one zero crossing")
        cross_idx = int(sign_changes[0])
        x0, x1 = mxdiff[cross_idx], mxdiff[cross_idx + 1]
        y0, y1 = msigned[cross_idx], msigned[cross_idx + 1]
        xzero = float(x0 - y0 * (x1 - x0) / (y1 - y0))
        mismatch_offsets.append(xzero)
        mid = int(np.argmin(np.abs(mxdiff - xzero)))
        local_lo = max(0, mid - 5)
        local_hi = min(len(mxdiff), mid + 6)
        mismatch_gains.append(float(np.polyfit(mxdiff[local_lo:local_hi], msigned[local_lo:local_hi], 1)[0]))
        mismatch_curves.append((name, label, mxdiff, msigned))
    left_low_offset, balanced_offset, left_high_offset = mismatch_offsets
    require(abs(balanced_offset) < 0.01, "matched forward pair offset should stay near zero")
    require(left_low_offset < balanced_offset - 0.015, "low left VTO should move zero crossing negative")
    require(left_high_offset > balanced_offset + 0.015, "high left VTO should move zero crossing positive")
    require(np.max(np.abs(mismatch_offsets)) < 0.06, "20 mV VTO mismatch should keep forward offset bounded")
    require(np.min(mismatch_gains) > 0.55, "mismatched forward pair should retain useful center gain")

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
    forward_plot = save_plot(fig, "mos_forward_pair_ngspice")

    cm_fig, cm_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for name, cm, cm_xdiff, cm_signed in cm_curves:
        cm_axes[0].plot(cm_xdiff, cm_signed, label=f"{name} VCM={cm:.2f} V")
        center = np.abs(cm_xdiff) <= 0.18
        cm_axes[1].plot(cm_xdiff[center], cm_signed[center], label=f"{cm:.2f} V")
    cm_axes[0].axhline(0, color="0.4", linewidth=0.8)
    cm_axes[0].axvline(0, color="0.4", linewidth=0.8)
    cm_axes[0].set_ylabel("$h^- - h^+$ (V)")
    cm_axes[0].set_title("Forward pair remains centered across input common-mode")
    cm_axes[0].grid(True, alpha=0.25)
    cm_axes[0].legend()
    cm_axes[1].axhline(0, color="0.4", linewidth=0.8)
    cm_axes[1].axvline(0, color="0.4", linewidth=0.8)
    cm_axes[1].set_xlabel("$z^+ - z^-$ (V)")
    cm_axes[1].set_ylabel("$h^- - h^+$ (V)")
    cm_axes[1].set_title("Center-region zero crossing and gain overlay")
    cm_axes[1].set_xlim(-0.20, 0.20)
    cm_axes[1].set_ylim(-0.20, 0.20)
    cm_axes[1].grid(True, alpha=0.25)
    cm_axes[1].legend(title="$V_{CM}$")
    cm_fig.tight_layout()
    save_plot(cm_fig, "mos_forward_common_mode_ngspice")

    sweep_fig, sweep_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for diff, series in zip(store_sweep_cases, sweep_cap):
        if diff in (-0.45, 0.0, 0.45):
            sweep_axes[0].plot(1e6 * st, series, label=f"$z^+-z^-$={diff:.2f} V")
    sweep_axes[0].plot(1e6 * st, store_sweep_cols[-1] / 5.0, color="0.5", alpha=0.35, label="$pact/5$")
    sweep_axes[0].axhline(0, color="0.4", linewidth=0.8)
    sweep_axes[0].set_ylabel("stored activation (V)")
    sweep_axes[0].set_title("Forward-store capacitor samples graded activations")
    sweep_axes[0].grid(True, alpha=0.25)
    sweep_axes[0].legend(loc="upper right")
    sweep_axes[1].plot(store_sweep_cases, sweep_load_final, "o-", label="load output")
    sweep_axes[1].plot(store_sweep_cases, sweep_cap_final, "s--", label="after pact sample")
    sweep_axes[1].plot(store_sweep_cases, sweep_cap_hold, "d:", color="0.45", label="after hold")
    sweep_axes[1].axhline(0, color="0.4", linewidth=0.8)
    sweep_axes[1].axvline(0, color="0.4", linewidth=0.8)
    sweep_axes[1].set_xlabel("$z^+ - z^-$ (V)")
    sweep_axes[1].set_ylabel("$h^- - h^+$ stored/load (V)")
    sweep_axes[1].set_title("Stored activation is monotone, centered, and retained")
    sweep_axes[1].grid(True, alpha=0.25)
    sweep_axes[1].legend(loc="upper left")
    sweep_fig.tight_layout()
    save_plot(sweep_fig, "mos_forward_store_sweep_ngspice")

    mismatch_fig, mismatch_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for (name, label, mxdiff, msigned), xzero in zip(mismatch_curves, mismatch_offsets):
        mismatch_axes[0].plot(mxdiff, msigned, label=label)
        mismatch_axes[0].axvline(xzero, linewidth=0.8, linestyle=":", alpha=0.65)
    mismatch_axes[0].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].axvline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].set_xlabel("$z^+ - z^-$ (V)")
    mismatch_axes[0].set_ylabel("$h^- - h^+$ (V)")
    mismatch_axes[0].set_title("Forward-pair threshold mismatch shifts the zero crossing")
    mismatch_axes[0].grid(True, alpha=0.25)
    mismatch_axes[0].legend(loc="upper left")
    mismatch_axes[1].plot([case[3] for case in mismatch_cases], mismatch_offsets, "o-", label="zero crossing")
    mismatch_axes[1].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[1].set_xlabel("input-pair threshold skew")
    mismatch_axes[1].set_ylabel("zero crossing $z^+ - z^-$ (V)")
    mismatch_axes[1].set_title("Offset is visible but bounded in the Level-1 mismatch sweep")
    mismatch_axes[1].grid(True, axis="y", alpha=0.25)
    mismatch_axes[1].legend(loc="upper left")
    mismatch_fig.tight_layout()
    save_plot(mismatch_fig, "mos_forward_mismatch_ngspice")
    return forward_plot


def characterize_synapse_forward_chain() -> Path:
    deck = f"""
* Synapse-to-forward-store integration sanity check.
* The signed synapse capacitor convention is voltage-inverting: a positive
* contribution raises z- - z+.  The forward pair therefore reads those summing
* capacitors with crossed gates so its z+ - z- input is the stored
* contribution voltage.
{COMMON_MODELS}
.param CSUM=500p CSTORE=10p WTAIL=2u WSW=24u
VDD vdd 0 1.8
VXP xp 0 1.15
VXM xm 0 0.65
VTAIL vbias 0 0.95
VWP wpulse 0 PULSE(0 1.15 0.5u 20n 20n 0.70u 5u)
VPACT pact 0 PULSE(0 1.8 1.35u 20n 20n 0.65u 5u)

* Positive-weight synapse writes z- > z+ for x+ > x-.
CZPP zpp 0 {{CSUM}} IC=0.9
CZMP zmp 0 {{CSUM}} IC=0.9
RZPP zpp 0 100G
RZMP zmp 0 100G
MPP zpp xp tailp 0 NMOS L={{LCH}} W={{WN}}
MPM zmp xm tailp 0 NMOS L={{LCH}} W={{WN}}
MTP tailp wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}

* Crossed forward gates read the voltage-coded positive contribution.
MPFP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPFM hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MNFP hpp zmp ftailp 0 NMOS L={{LCH}} W={{WN}}
MNFM hmp zpp ftailp 0 NMOS L={{LCH}} W={{WN}}
MNFT ftailp vbias 0 0 NMOS L={{LCH}} W={{WN}}
CHPP hcp_p 0 {{CSTORE}} IC=1.04
CHPM hcm_p 0 {{CSTORE}} IC=1.04
RCHPP hcp_p 0 50G
RCHPM hcm_p 0 50G
MSPP hpp pact hcp_p 0 NMOS L={{LCH}} W={{WSW}}
MSPM hmp pact hcm_p 0 NMOS L={{LCH}} W={{WSW}}

* Negative-weight synapse writes z- < z+ for the same x+ > x- input.
CZPN zpn 0 {{CSUM}} IC=0.9
CZMN zmn 0 {{CSUM}} IC=0.9
RZPN zpn 0 100G
RZMN zmn 0 100G
MNP zpn xm tailn 0 NMOS L={{LCH}} W={{WN}}
MNM zmn xp tailn 0 NMOS L={{LCH}} W={{WN}}
MTN tailn wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}

* The same crossed convention preserves the negative sign.
MPNP hpn hpn vdd vdd PMOS L={{LCH}} W={{WP}}
MPNM hmn hmn vdd vdd PMOS L={{LCH}} W={{WP}}
MNNP hpn zmn ftailn 0 NMOS L={{LCH}} W={{WN}}
MNNM hmn zpn ftailn 0 NMOS L={{LCH}} W={{WN}}
MNNT ftailn vbias 0 0 NMOS L={{LCH}} W={{WN}}
CHNP hcp_n 0 {{CSTORE}} IC=1.04
CHNM hcm_n 0 {{CSTORE}} IC=1.04
RCHNP hcp_n 0 50G
RCHNM hcm_n 0 50G
MSNP hpn pact hcp_n 0 NMOS L={{LCH}} W={{WSW}}
MSNM hmn pact hcm_n 0 NMOS L={{LCH}} W={{WSW}}

.control
set noaskquit
tran 5n 3u uic
wrdata mos_synapse_forward_chain.dat v(zpp) v(zmp) v(hpp) v(hmp) v(hcp_p) v(hcm_p) v(zpn) v(zmn) v(hpn) v(hmn) v(hcp_n) v(hcm_n) v(wpulse) v(pact)
quit
.endc
.end
"""
    cycle_deck = f"""
* Two-cycle synapse-to-forward-store reuse sanity check.
* Reset clears the same preactivation and activation capacitors between two
* chained samples.  The first sample writes a positive synapse contribution;
* the second sample writes a negative contribution through the gate-swapped
* synapse copy.
{COMMON_MODELS}
.param CSUM=500p CSTORE=10p WTAIL=2u WSW=24u WRESETN=24u WRESETP=60u
VDD vdd 0 1.8
VZCM zcm 0 0.9
VHCM hcm_ref 0 1.04
VXP xp 0 1.15
VXM xm 0 0.65
VTAIL vbias 0 0.95
VRST rst 0 PWL(0 0 0.10u 0 0.12u 1.8 0.45u 1.8 0.47u 0 2.20u 0 2.22u 1.8 2.55u 1.8 2.57u 0 5u 0)
VRSTN rstn 0 PWL(0 1.8 0.10u 1.8 0.12u 0 0.45u 0 0.47u 1.8 2.20u 1.8 2.22u 0 2.55u 0 2.57u 1.8 5u 1.8)
VWP wpulse 0 PWL(0 0 0.60u 0 0.62u 1.15 1.20u 1.15 1.22u 0 5u 0)
VWN wnpulse 0 PWL(0 0 2.80u 0 2.82u 1.15 3.40u 1.15 3.42u 0 5u 0)
VPACT pact 0 PWL(0 0 1.35u 0 1.37u 1.8 1.95u 1.8 1.97u 0 3.55u 0 3.57u 1.8 4.15u 1.8 4.17u 0 5u 0)

CZP zp 0 {{CSUM}} IC=0.9
CZM zm 0 {{CSUM}} IC=0.9
RZP zp 0 100G
RZM zm 0 100G
MRZPN zp rst zcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRZMN zm rst zcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRZPP zp rstn zcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRZMP zm rstn zcm vdd PMOS L={{LCH}} W={{WRESETP}}

* Positive and negative signed synapse slices reuse the same summing caps.
MPP zp xp tailp 0 NMOS L={{LCH}} W={{WN}}
MPM zm xm tailp 0 NMOS L={{LCH}} W={{WN}}
MTP tailp wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}
MNP zp xm tailn 0 NMOS L={{LCH}} W={{WN}}
MNM zm xp tailn 0 NMOS L={{LCH}} W={{WN}}
MTN tailn wnpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}

* Crossed forward pair reads voltage-coded z- - z+ contribution.
MPFP hp hp vdd vdd PMOS L={{LCH}} W={{WP}}
MPFM hm hm vdd vdd PMOS L={{LCH}} W={{WP}}
MNFP hp zm ftail 0 NMOS L={{LCH}} W={{WN}}
MNFM hm zp ftail 0 NMOS L={{LCH}} W={{WN}}
MNFT ftail vbias 0 0 NMOS L={{LCH}} W={{WN}}

CHP hcp 0 {{CSTORE}} IC=1.04
CHM hcm 0 {{CSTORE}} IC=1.04
RHP hcp 0 50G
RHM hcm 0 50G
MRHPN hcp rst hcm_ref 0 NMOS L={{LCH}} W={{WRESETN}}
MRHMN hcm rst hcm_ref 0 NMOS L={{LCH}} W={{WRESETN}}
MRHPP hcp rstn hcm_ref vdd PMOS L={{LCH}} W={{WRESETP}}
MRHMP hcm rstn hcm_ref vdd PMOS L={{LCH}} W={{WRESETP}}
MSP hp pact hcp 0 NMOS L={{LCH}} W={{WSW}}
MSM hm pact hcm 0 NMOS L={{LCH}} W={{WSW}}

.control
set noaskquit
tran 5n 4.8u uic
wrdata mos_synapse_forward_cycle.dat v(zp) v(zm) v(hp) v(hm) v(hcp) v(hcm) v(rst) v(wpulse) v(wnpulse) v(pact)
quit
.endc
.end
"""
    timing_cases = [
        ("pre", "pre-write", 0.05, 0.35),
        ("edge", "write edge", 0.48, 0.98),
        ("overlap", "overlap", 0.80, 1.30),
        ("late", "late overlap", 1.10, 1.60),
        ("gap", "settled gap", 1.35, 1.85),
    ]
    timing_devices = []
    timing_prints = ["v(zp)", "v(zm)"]
    for name, _label, start_us, end_us in timing_cases:
        timing_devices.append(
            f"""
VPACT_{name} pact_{name} 0 PWL(0 0 {start_us:.2f}u 0 {start_us + 0.02:.2f}u 1.8 {end_us:.2f}u 1.8 {end_us + 0.02:.2f}u 0 3.0u 0)
MPFP_{name} hp_{name} hp_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPFM_{name} hm_{name} hm_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNFP_{name} hp_{name} zm ftail_{name} 0 NMOS L={{LCH}} W={{WN}}
MNFM_{name} hm_{name} zp ftail_{name} 0 NMOS L={{LCH}} W={{WN}}
MNFT_{name} ftail_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}
CHP_{name} hcp_{name} 0 {{CSTORE}} IC=1.04
CHM_{name} hcm_{name} 0 {{CSTORE}} IC=1.04
RHP_{name} hcp_{name} 0 50G
RHM_{name} hcm_{name} 0 50G
MSP_{name} hp_{name} pact_{name} hcp_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSM_{name} hm_{name} pact_{name} hcm_{name} 0 NMOS L={{LCH}} W={{WSW}}
"""
        )
        timing_prints.extend([f"v(hp_{name})", f"v(hm_{name})", f"v(hcp_{name})", f"v(hcm_{name})", f"v(pact_{name})"])

    timing_deck = f"""
* Synapse-to-forward-store pact timing margin check.
* Several activation storage copies share the same signed synapse and crossed
* forward pair while their pact pulses sample before, during, and after
* preactivation settling.
{COMMON_MODELS}
.param CSUM=500p CSTORE=10p WTAIL=2u WSW=24u
VDD vdd 0 1.8
VXP xp 0 1.15
VXM xm 0 0.65
VTAIL vbias 0 0.95
VWP wpulse 0 PWL(0 0 0.50u 0 0.52u 1.15 1.20u 1.15 1.22u 0 3.0u 0)

CZP zp 0 {{CSUM}} IC=0.9
CZM zm 0 {{CSUM}} IC=0.9
RZP zp 0 100G
RZM zm 0 100G
MPP zp xp tailp 0 NMOS L={{LCH}} W={{WN}}
MPM zm xm tailp 0 NMOS L={{LCH}} W={{WN}}
MTP tailp wpulse 0 0 NMOS L={{LCH}} W={{WTAIL}}

{''.join(timing_devices)}

.control
set noaskquit
tran 5n 3.0u uic
wrdata mos_synapse_forward_phase_timing.dat {' '.join(timing_prints)} v(wpulse)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_synapse_forward_chain")
    t, cols = load_wrdata(data, 14)
    pos_preact = cols[1] - cols[0]
    pos_load = cols[3] - cols[2]
    pos_store = cols[5] - cols[4]
    neg_preact = cols[7] - cols[6]
    neg_load = cols[9] - cols[8]
    neg_store = cols[11] - cols[10]

    def at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(t - time_s))])

    require(at(1.20e-6, pos_preact) > 0.08, "positive synapse should write positive voltage-coded preactivation")
    require(at(1.20e-6, neg_preact) < -0.08, "negative synapse should write negative voltage-coded preactivation")
    require(at(1.25e-6, pos_load) > 0.05, "crossed forward pair should read positive synapse state as positive activation")
    require(at(1.25e-6, neg_load) < -0.05, "crossed forward pair should read negative synapse state as negative activation")
    require(at(1.95e-6, pos_store) > 0.05, "pact should store positive chained activation")
    require(at(1.95e-6, neg_store) < -0.05, "pact should store negative chained activation")
    require(at(2.80e-6, pos_store) > 0.05, "positive chained activation should hold after pact")
    require(at(2.80e-6, neg_store) < -0.05, "negative chained activation should hold after pact")

    cycle_data = run_ngspice(cycle_deck, "mos_synapse_forward_cycle")
    ct, cycle_cols = load_wrdata(cycle_data, 10)
    cycle_preact = cycle_cols[1] - cycle_cols[0]
    cycle_load = cycle_cols[3] - cycle_cols[2]
    cycle_store = cycle_cols[5] - cycle_cols[4]

    def cycle_at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(ct - time_s))])

    require(cycle_at(1.25e-6, cycle_preact) > 0.08, "first cycle should write positive preactivation")
    require(cycle_at(1.90e-6, cycle_store) > 0.05, "first cycle should store positive activation")
    require(abs(cycle_at(2.55e-6, cycle_preact)) < 0.02, "mid-cycle reset should clear preactivation")
    require(abs(cycle_at(2.55e-6, cycle_store)) < 0.02, "mid-cycle reset should clear stored activation")
    require(cycle_at(3.45e-6, cycle_preact) < -0.075, "second cycle should write negative preactivation")
    require(cycle_at(4.10e-6, cycle_store) < -0.05, "second cycle should store negative activation")
    require(cycle_at(4.60e-6, cycle_store) < -0.05, "second-cycle stored activation should hold")

    timing_data = run_ngspice(timing_deck, "mos_synapse_forward_phase_timing")
    tt, timing_cols = load_wrdata(timing_data, len(timing_prints) + 1)

    def timing_at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(tt - time_s))])

    timing_preact = timing_cols[1] - timing_cols[0]
    timing_loads = []
    timing_stores = []
    timing_final = []
    for idx, (_name, _label, _start_us, _end_us) in enumerate(timing_cases):
        load = timing_cols[2 + 5 * idx + 1] - timing_cols[2 + 5 * idx]
        stored = timing_cols[2 + 5 * idx + 3] - timing_cols[2 + 5 * idx + 2]
        timing_loads.append(load)
        timing_stores.append(stored)
        timing_final.append(timing_at(2.75e-6, stored))
    timing_final = np.array(timing_final)
    require(timing_at(1.25e-6, timing_preact) > 0.08, "timing deck should write positive preactivation")
    require(timing_at(1.25e-6, timing_loads[-1]) > 0.05, "timing deck forward pair should read positive activation")
    require(abs(timing_final[0]) < 0.002, "pact pulse before synapse write should not store a signed activation")
    require(timing_final[1] > timing_final[0] + 0.015, "pact pulse at write edge should store a partial positive activation")
    require(timing_final[1] < 0.95 * timing_final[-1], "write-edge pact sample should remain below the settled activation")
    require(np.min(timing_final[2:]) > 0.95 * timing_final[-1], "pact pulses after preactivation settling should store the full activation")
    require(np.max(timing_final[2:]) - np.min(timing_final[2:]) < 0.004, "settled pact timings should agree")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    axes[0].plot(1e6 * t, pos_preact, label="$w^+$ stored $z^- - z^+$")
    axes[0].plot(1e6 * t, neg_preact, "--", label="$w^-$ stored $z^- - z^+$")
    axes[0].plot(1e6 * t, cols[12] / 8.0, color="0.5", alpha=0.35, label="$w_{gate}/8$")
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("preactivation (V)")
    axes[0].set_title("Synapse writes signed preactivation onto real summing caps")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[1].plot(1e6 * t, pos_load, label="positive path load $h^- - h^+$")
    axes[1].plot(1e6 * t, neg_load, "--", label="negative path load $h^- - h^+$")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_ylabel("load output (V)")
    axes[1].set_title("Crossed forward gates preserve the voltage-coded sign")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[2].plot(1e6 * t, pos_store, label="stored positive activation")
    axes[2].plot(1e6 * t, neg_store, "--", label="stored negative activation")
    axes[2].plot(1e6 * t, cols[13] / 8.0, color="0.5", alpha=0.35, label="$pact/8$")
    axes[2].axhline(0, color="0.4", linewidth=0.8)
    axes[2].set_xlabel("time (us)")
    axes[2].set_ylabel("stored activation (V)")
    axes[2].set_title("Activation storage samples and holds the chained result")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    chain_plot = save_plot(fig, "mos_synapse_forward_chain_ngspice")

    cycle_fig, cycle_axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    cycle_axes[0].plot(1e6 * ct, cycle_preact, label="stored $z^- - z^+$")
    cycle_axes[0].plot(1e6 * ct, cycle_cols[6] / 12.0, color="0.4", alpha=0.35, label="$reset/12$")
    cycle_axes[0].plot(1e6 * ct, cycle_cols[7] / 10.0, color="0.5", alpha=0.35, label="$w^+/10$")
    cycle_axes[0].plot(1e6 * ct, -cycle_cols[8] / 10.0, color="0.25", alpha=0.25, label="$-w^-/10$")
    cycle_axes[0].axhline(0, color="0.4", linewidth=0.8)
    cycle_axes[0].set_ylabel("preactivation (V)")
    cycle_axes[0].set_title("Reset lets the same summing caps accept opposite-signed samples")
    cycle_axes[0].grid(True, alpha=0.25)
    cycle_axes[0].legend(loc="upper right", ncol=2)
    cycle_axes[1].plot(1e6 * ct, cycle_load, label="forward load $h^- - h^+$")
    cycle_axes[1].axhline(0, color="0.4", linewidth=0.8)
    cycle_axes[1].set_ylabel("load output (V)")
    cycle_axes[1].set_title("Forward pair follows the reset-and-rewritten preactivation")
    cycle_axes[1].grid(True, alpha=0.25)
    cycle_axes[1].legend(loc="upper right")
    cycle_axes[2].plot(1e6 * ct, cycle_store, label="stored activation")
    cycle_axes[2].plot(1e6 * ct, cycle_cols[9] / 8.0, color="0.5", alpha=0.35, label="$pact/8$")
    cycle_axes[2].plot(1e6 * ct, cycle_cols[6] / 12.0, color="0.4", alpha=0.25, label="$reset/12$")
    cycle_axes[2].axhline(0, color="0.4", linewidth=0.8)
    cycle_axes[2].set_xlabel("time (us)")
    cycle_axes[2].set_ylabel("stored activation (V)")
    cycle_axes[2].set_title("Activation cap is reset before storing the opposite sign")
    cycle_axes[2].grid(True, alpha=0.25)
    cycle_axes[2].legend(loc="upper right", ncol=2)
    cycle_fig.tight_layout()
    save_plot(cycle_fig, "mos_synapse_forward_cycle_ngspice")

    timing_fig, timing_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    timing_axes[0].plot(1e6 * tt, timing_preact, label="stored $z^- - z^+$")
    timing_axes[0].plot(1e6 * tt, timing_loads[-1], label="settled-gap load $h^- - h^+$")
    timing_axes[0].plot(1e6 * tt, timing_cols[-1] / 10.0, color="0.5", alpha=0.35, label="$w^+/10$")
    timing_axes[0].axhline(0, color="0.4", linewidth=0.8)
    timing_axes[0].set_ylabel("differential voltage (V)")
    timing_axes[0].set_title("Synapse and forward load settle before pact sampling")
    timing_axes[0].grid(True, alpha=0.25)
    timing_axes[0].legend(loc="upper right")
    for (_name, label, _start_us, _end_us), stored in zip(timing_cases, timing_stores):
        timing_axes[1].plot(1e6 * tt, stored, label=label)
    timing_axes[1].axhline(0, color="0.4", linewidth=0.8)
    timing_axes[1].set_xlabel("time (us)")
    timing_axes[1].set_ylabel("stored activation (V)")
    timing_axes[1].set_title("pact timing sweep exposes the write/store margin")
    timing_axes[1].grid(True, alpha=0.25)
    timing_axes[1].legend(loc="lower right", ncol=2)
    timing_fig.tight_layout()
    save_plot(timing_fig, "mos_synapse_forward_phase_timing_ngspice")
    return chain_plot


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
    gain_store_cases = [("neg_sat", -0.55), ("center", 0.0), ("pos_sat", 0.55)]
    gain_store_devices = []
    gain_store_prints = []
    for name, diff in gain_store_cases:
        plus_p = 0.9 + (diff + eps) / 2.0
        plus_m = 0.9 - (diff + eps) / 2.0
        minus_p = 0.9 + (diff - eps) / 2.0
        minus_m = 0.9 - (diff - eps) / 2.0
        gain_store_devices.append(
            f"""
* Stored finite-difference replica outputs for {name}, z+ - z- = {diff:.2f} V.
VZPP_{name} zpp_{name} 0 {plus_p:.5f}
VZMM_{name} zmm_{name} 0 {plus_m:.5f}
VZPM_{name} zpm_{name} 0 {minus_p:.5f}
VZMP_{name} zmp_{name} 0 {minus_m:.5f}

MPPPL_{name} hpp_{name} hpp_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPMM_{name} hpm_{name} hpm_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_{name} hpp_{name} zpp_{name} tailp_{name} 0 NMOS L={{LCH}} W={{WN}}
MNPM_{name} hpm_{name} zmm_{name} tailp_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTP_{name} tailp_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMPL_{name} hmp_{name} hmp_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMMM_{name} hmm_{name} hmm_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_{name} hmp_{name} zpm_{name} tailm_{name} 0 NMOS L={{LCH}} W={{WN}}
MNMM_{name} hmm_{name} zmp_{name} tailm_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTM_{name} tailm_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CPP_{name} cpp_{name} 0 {{CERR}} IC=1.04
CPM_{name} cpm_{name} 0 {{CERR}} IC=1.04
CMP_{name} cmp_{name} 0 {{CERR}} IC=1.04
CMM_{name} cmm_{name} 0 {{CERR}} IC=1.04
RPP_{name} cpp_{name} 0 50G
RPM_{name} cpm_{name} 0 50G
RMP_{name} cmp_{name} 0 50G
RMM_{name} cmm_{name} 0 50G
MSPP_{name} hpp_{name} psamp cpp_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSPM_{name} hpm_{name} psamp cpm_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSMP_{name} hmp_{name} psamp cmp_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSMM_{name} hmm_{name} psamp cmm_{name} 0 NMOS L={{LCH}} W={{WSW}}
"""
        )
        gain_store_prints.extend([f"v(cpp_{name})", f"v(cpm_{name})", f"v(cmp_{name})", f"v(cmm_{name})"])
    gain_store_deck = f"""
* Hidden-error derivative-window storage sanity check.
* MOS pass switches sample the +eps and -eps forward-replica outputs onto
* capacitors; the plotted finite difference is computed from those stored
* capacitor voltages by the characterization script.
{COMMON_MODELS}
.param CERR=10p WSW=24u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPSAMP psamp 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
{''.join(gain_store_devices)}
.control
set noaskquit
tran 5n 2.5u uic
wrdata mos_hidden_error_gain_store.dat {' '.join(gain_store_prints)} v(psamp)
quit
.endc
.end
"""
    sign_store_eps = 0.08
    sign_store_cases = [("neg_sat", -0.55), ("center", 0.0), ("pos_sat", 0.55)]
    sign_store_devices = []
    sign_store_prints = []

    def sign_store_path(src: str, gate: str, dst: str, suffix: str) -> str:
        return f"""
MSA_{suffix} {src} pbwd n_{suffix} 0 NMOS L={{LCH}} W={{WSW}}
MSB_{suffix} n_{suffix} {gate} {dst} 0 NMOS L={{LCH}} W={{WSW}}
"""

    for name, diff in sign_store_cases:
        plus_p = 0.9 + (diff + sign_store_eps) / 2.0
        plus_m = 0.9 - (diff + sign_store_eps) / 2.0
        minus_p = 0.9 + (diff - sign_store_eps) / 2.0
        minus_m = 0.9 - (diff - sign_store_eps) / 2.0
        sign_store_devices.append(
            f"""
* Cross-connected finite-difference hidden-error storage for {name}.
VZPPX_{name} zppx_{name} 0 {plus_p:.5f}
VZMMX_{name} zmmx_{name} 0 {plus_m:.5f}
VZPMX_{name} zpmx_{name} 0 {minus_p:.5f}
VZMPX_{name} zmpx_{name} 0 {minus_m:.5f}

MPPXP_{name} hppx_{name} hppx_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMXP_{name} hpmx_{name} hpmx_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPPX_{name} hppx_{name} zppx_{name} tailpx_{name} 0 NMOS L={{LCH}} W={{WN}}
MNPMX_{name} hpmx_{name} zmmx_{name} tailpx_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTPX_{name} tailpx_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMXP2_{name} hmpx_{name} hmpx_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMXM2_{name} hmmx_{name} hmmx_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMPX_{name} hmpx_{name} zpmx_{name} tailmx_{name} 0 NMOS L={{LCH}} W={{WN}}
MNMMX_{name} hmmx_{name} zmpx_{name} tailmx_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTMX_{name} tailmx_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDPPX_{name} cdp_px_{name} 0 {{CERR}} IC=1.04
CDPMX_{name} cdm_px_{name} 0 {{CERR}} IC=1.04
CDNPX_{name} cdp_nx_{name} 0 {{CERR}} IC=1.04
CDNMX_{name} cdm_nx_{name} 0 {{CERR}} IC=1.04
RPPX_{name} cdp_px_{name} 0 50G
RPMX_{name} cdm_px_{name} 0 50G
RNPX_{name} cdp_nx_{name} 0 50G
RNMX_{name} cdm_nx_{name} 0 50G
"""
            + sign_store_path(f"hpmx_{name}", "rp", f"cdp_px_{name}", f"p1_{name}")
            + sign_store_path(f"hmpx_{name}", "rp", f"cdp_px_{name}", f"p2_{name}")
            + sign_store_path(f"hppx_{name}", "rp", f"cdm_px_{name}", f"p3_{name}")
            + sign_store_path(f"hmmx_{name}", "rp", f"cdm_px_{name}", f"p4_{name}")
            + sign_store_path(f"hppx_{name}", "rm", f"cdp_nx_{name}", f"n1_{name}")
            + sign_store_path(f"hmmx_{name}", "rm", f"cdp_nx_{name}", f"n2_{name}")
            + sign_store_path(f"hpmx_{name}", "rm", f"cdm_nx_{name}", f"n3_{name}")
            + sign_store_path(f"hmpx_{name}", "rm", f"cdm_nx_{name}", f"n4_{name}")
        )
        sign_store_prints.extend(
            [
                f"v(cdp_px_{name})",
                f"v(cdm_px_{name})",
                f"v(cdp_nx_{name})",
                f"v(cdm_nx_{name})",
            ]
        )
    sign_store_deck = f"""
* Hidden-error finite-difference sign store sanity check.
* Cross-connected MOS pass networks average +eps and -eps forward-replica
* outputs directly onto hidden-error capacitors.  The positive-error copy
* stores (s_plus - s_minus)/2; the negative-error copy stores the opposite.
{COMMON_MODELS}
.param CERR=10p WSW=24u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
VRP rp 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
VRM rm 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
{''.join(sign_store_devices)}
.control
set noaskquit
tran 5n 2.5u uic
wrdata mos_hidden_error_sign_store.dat {' '.join(sign_store_prints)} v(pbwd)
quit
.endc
.end
"""
    nudge_sweep_values = [0.00, 0.02, 0.04, 0.08, 0.12]
    nudge_sweep_devices = []
    nudge_sweep_prints = []
    for idx, nudge in enumerate(nudge_sweep_values):
        name = f"n{idx}"
        plus_p = 0.9 + nudge / 2.0
        plus_m = 0.9 - nudge / 2.0
        minus_p = 0.9 - nudge / 2.0
        minus_m = 0.9 + nudge / 2.0
        nudge_sweep_devices.append(
            f"""
* Cross-connected finite-difference hidden-error storage, eps={nudge:.2f} V.
VZPPN_{name} zppn_{name} 0 {plus_p:.5f}
VZMMN_{name} zmmn_{name} 0 {plus_m:.5f}
VZPMN_{name} zpmn_{name} 0 {minus_p:.5f}
VZMPN_{name} zmpn_{name} 0 {minus_m:.5f}

MPPNP_{name} hppn_{name} hppn_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPNM_{name} hpmn_{name} hpmn_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPNP_{name} hppn_{name} zppn_{name} tailpn_{name} 0 NMOS L={{LCH}} W={{WN}}
MNPNM_{name} hpmn_{name} zmmn_{name} tailpn_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTPN_{name} tailpn_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMNP_{name} hmpn_{name} hmpn_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMNM_{name} hmmn_{name} hmmn_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMNP_{name} hmpn_{name} zpmn_{name} tailmn_{name} 0 NMOS L={{LCH}} W={{WN}}
MNMNM_{name} hmmn_{name} zmpn_{name} tailmn_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTMN_{name} tailmn_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDPPN_{name} cdp_pn_{name} 0 {{CERR}} IC=1.04
CDPMN_{name} cdm_pn_{name} 0 {{CERR}} IC=1.04
CDNPN_{name} cdp_nn_{name} 0 {{CERR}} IC=1.04
CDNMN_{name} cdm_nn_{name} 0 {{CERR}} IC=1.04
RPPN_{name} cdp_pn_{name} 0 50G
RPMN_{name} cdm_pn_{name} 0 50G
RNPN_{name} cdp_nn_{name} 0 50G
RNMN_{name} cdm_nn_{name} 0 50G
"""
            + sign_store_path(f"hpmn_{name}", "rp", f"cdp_pn_{name}", f"np1_{name}")
            + sign_store_path(f"hmpn_{name}", "rp", f"cdp_pn_{name}", f"np2_{name}")
            + sign_store_path(f"hppn_{name}", "rp", f"cdm_pn_{name}", f"np3_{name}")
            + sign_store_path(f"hmmn_{name}", "rp", f"cdm_pn_{name}", f"np4_{name}")
            + sign_store_path(f"hppn_{name}", "rm", f"cdp_nn_{name}", f"nn1_{name}")
            + sign_store_path(f"hmmn_{name}", "rm", f"cdp_nn_{name}", f"nn2_{name}")
            + sign_store_path(f"hpmn_{name}", "rm", f"cdm_nn_{name}", f"nn3_{name}")
            + sign_store_path(f"hmpn_{name}", "rm", f"cdm_nn_{name}", f"nn4_{name}")
        )
        nudge_sweep_prints.extend(
            [
                f"v(cdp_pn_{name})",
                f"v(cdm_pn_{name})",
                f"v(cdp_nn_{name})",
                f"v(cdm_nn_{name})",
            ]
        )
    nudge_sweep_deck = f"""
* Hidden-error finite-difference nudge-magnitude storage sanity check.
* Matched MOS replica pairs and cross-connected pass switches store a graded
* derivative-weighted hidden-error rail as the replica nudge grows.
{COMMON_MODELS}
.param CERR=10p WSW=24u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
VRP rp 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
VRM rm 0 PULSE(0 1.8 0.5u 20n 20n 0.8u 4.0u)
{''.join(nudge_sweep_devices)}
.control
set noaskquit
tran 5n 2.5u uic
wrdata mos_hidden_error_nudge_sweep.dat {' '.join(nudge_sweep_prints)} v(pbwd)
quit
.endc
.end
"""
    mismatch_cases = [
        ("left_low", "NMOSHE53", "left $V_{TO}=0.53$ V"),
        ("nominal", "NMOSHE55", "left $V_{TO}=0.55$ V"),
        ("left_high", "NMOSHE57", "left $V_{TO}=0.57$ V"),
    ]
    mismatch_models = """
.model NMOSHE53 NMOS (LEVEL=1 VTO=0.53 KP=220u LAMBDA=0.03)
.model NMOSHE55 NMOS (LEVEL=1 VTO=0.55 KP=220u LAMBDA=0.03)
.model NMOSHE57 NMOS (LEVEL=1 VTO=0.57 KP=220u LAMBDA=0.03)
"""
    mismatch_devices = []
    mismatch_prints = []
    for name, left_model, _label in mismatch_cases:
        mismatch_devices.append(
            f"""
* Hidden-error finite-difference replica pair with {left_model} on z+ input devices.
VZPPH_{name} zpph_{name} zp {{EPS/2}}
VZMMH_{name} zm zmmh_{name} {{EPS/2}}
VZPMH_{name} zp zpmh_{name} {{EPS/2}}
VZMPH_{name} zmph_{name} zm {{EPS/2}}

MPPPH_{name} hpph_{name} hpph_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPMH_{name} hpmh_{name} hpmh_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPPH_{name} hpph_{name} zpph_{name} tailph_{name} 0 {left_model} L={{LCH}} W={{WN}}
MNPMH_{name} hpmh_{name} zmmh_{name} tailph_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTPH_{name} tailph_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMPH_{name} hmph_{name} hmph_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMMH_{name} hmmh_{name} hmmh_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMPH_{name} hmph_{name} zpmh_{name} tailmh_{name} 0 {left_model} L={{LCH}} W={{WN}}
MNMMH_{name} hmmh_{name} zmph_{name} tailmh_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTMH_{name} tailmh_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}
"""
        )
        mismatch_prints.extend(
            [f"v(hpph_{name})", f"v(hpmh_{name})", f"v(hmph_{name})", f"v(hmmh_{name})"]
        )
    mismatch_deck = f"""
* Hidden-error derivative-window threshold-mismatch sanity check.
* The z+ input devices in both +eps and -eps replicas are swept across simple
* threshold offsets.  Mismatch should shift/scale the finite-difference window,
* not create negative derivative gain.
{COMMON_MODELS}
{mismatch_models}
.param EPS={eps}
VDD vdd 0 1.8
VCM cm 0 0.9
VDIFF zp zm -0.45
RZP zp cm 1G
RZM zm cm 1G
VTAIL vbias 0 0.95
{''.join(mismatch_devices)}
.control
set noaskquit
dc VDIFF -0.45 0.45 0.01
wrdata mos_hidden_error_mismatch.dat {' '.join(mismatch_prints)}
quit
.endc
.end
"""
    cm_cases = [("cm075", 0.75), ("cm090", 0.90), ("cm105", 1.05)]
    cm_devices = []
    cm_prints = []
    for name, cm_value in cm_cases:
        cm_devices.append(
            f"""
* Hidden-error finite-difference replica pair at input common-mode {cm_value:.2f} V.
VCMH_{name} cmh_{name} 0 {cm_value:.2f}
VDIFFH_{name} zph_{name} zmh_{name} PWL(0 -0.45 10u 0.45)
RZPH_{name} zph_{name} cmh_{name} 1G
RZMH_{name} zmh_{name} cmh_{name} 1G
VZPPHC_{name} zpphc_{name} zph_{name} {{EPS/2}}
VZMMHC_{name} zmh_{name} zmmhc_{name} {{EPS/2}}
VZPMHC_{name} zph_{name} zpmhc_{name} {{EPS/2}}
VZMPHC_{name} zmphc_{name} zmh_{name} {{EPS/2}}

MPPPHC_{name} hpphc_{name} hpphc_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPMHC_{name} hpmhc_{name} hpmhc_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPPHC_{name} hpphc_{name} zpphc_{name} tailphc_{name} 0 NMOS L={{LCH}} W={{WN}}
MNPMHC_{name} hpmhc_{name} zmmhc_{name} tailphc_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTPHC_{name} tailphc_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMPHC_{name} hmphc_{name} hmphc_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMMHC_{name} hmmhc_{name} hmmhc_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMPHC_{name} hmphc_{name} zpmhc_{name} tailmhc_{name} 0 NMOS L={{LCH}} W={{WN}}
MNMMHC_{name} hmmhc_{name} zmphc_{name} tailmhc_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTMHC_{name} tailmhc_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}
"""
        )
        cm_prints.extend(
            [f"v(hpphc_{name})", f"v(hpmhc_{name})", f"v(hmphc_{name})", f"v(hmmhc_{name})"]
        )
    cm_deck = f"""
* Hidden-error derivative-window input-common-mode margin sanity check.
* Matched +eps and -eps forward-replica pairs are instantiated at several
* input common modes.  The derivative proxy should stay nonnegative and keep a
* broad active window without relying on one exact bias point.
{COMMON_MODELS}
.param EPS={eps}
VDD vdd 0 1.8
VTAIL vbias 0 0.95
{''.join(cm_devices)}
.control
set noaskquit
tran 10n 10u
wrdata mos_hidden_error_common_mode.dat {' '.join(cm_prints)}
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

    gain_store_data = run_ngspice(gain_store_deck, "mos_hidden_error_gain_store")
    gt, gain_store_cols = load_wrdata(gain_store_data, 4 * len(gain_store_cases) + 1)
    stored_gain = []
    for idx in range(len(gain_store_cases)):
        plus_signed = gain_store_cols[4 * idx + 1] - gain_store_cols[4 * idx]
        minus_signed = gain_store_cols[4 * idx + 3] - gain_store_cols[4 * idx + 2]
        stored_gain.append((plus_signed - minus_signed) / (2.0 * eps))
    stored_gain_final = np.array([at(1.2e-6, series) for series in stored_gain])
    stored_gain_hold = np.array([at(2.2e-6, series) for series in stored_gain])
    require(stored_gain_final[1] > 0.65, "stored hidden-error gain should be high near z balance")
    require(stored_gain_final[0] < 0.35 * stored_gain_final[1], "negative saturated stored gain should be much lower")
    require(stored_gain_final[2] < 0.35 * stored_gain_final[1], "positive saturated stored gain should be much lower")
    require(np.max(np.abs(stored_gain_hold - stored_gain_final)) < 0.04, "stored hidden-error gain samples should hold")

    sign_store_data = run_ngspice(sign_store_deck, "mos_hidden_error_sign_store")
    st, sign_store_cols = load_wrdata(sign_store_data, 4 * len(sign_store_cases) + 1)
    pos_delta = []
    neg_delta = []
    for idx in range(len(sign_store_cases)):
        pos_delta.append(sign_store_cols[4 * idx] - sign_store_cols[4 * idx + 1])
        neg_delta.append(sign_store_cols[4 * idx + 2] - sign_store_cols[4 * idx + 3])
    pos_final = np.array([at(1.2e-6, series) for series in pos_delta])
    neg_final = np.array([at(1.2e-6, series) for series in neg_delta])
    pos_hold = np.array([at(2.2e-6, series) for series in pos_delta])
    neg_hold = np.array([at(2.2e-6, series) for series in neg_delta])
    require(pos_final[1] > 0.04, "cross-connected r+ derivative store should be positive near z balance")
    require(neg_final[1] < -0.04, "cross-connected r- derivative store should be negative near z balance")
    require(abs(pos_final[0]) < 0.4 * pos_final[1], "negative saturated r+ derivative store should be attenuated")
    require(abs(pos_final[2]) < 0.4 * pos_final[1], "positive saturated r+ derivative store should be attenuated")
    require(np.max(np.abs(pos_final + neg_final)) < 0.01, "r+ and r- cross-connected stores should be opposite")
    require(np.max(np.abs(pos_hold - pos_final)) < 0.01, "r+ derivative-sign store should hold")
    require(np.max(np.abs(neg_hold - neg_final)) < 0.01, "r- derivative-sign store should hold")

    nudge_data = run_ngspice(nudge_sweep_deck, "mos_hidden_error_nudge_sweep")
    nt, nudge_cols = load_wrdata(nudge_data, 4 * len(nudge_sweep_values) + 1)
    nudge_pos_delta = []
    nudge_neg_delta = []
    for idx in range(len(nudge_sweep_values)):
        nudge_pos_delta.append(nudge_cols[4 * idx] - nudge_cols[4 * idx + 1])
        nudge_neg_delta.append(nudge_cols[4 * idx + 2] - nudge_cols[4 * idx + 3])

    def nat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(nt - time_s))])

    nudge_pos_final = np.array([nat(1.2e-6, series) for series in nudge_pos_delta])
    nudge_neg_final = np.array([nat(1.2e-6, series) for series in nudge_neg_delta])
    nudge_pos_hold = np.array([nat(2.2e-6, series) for series in nudge_pos_delta])
    nudge_neg_hold = np.array([nat(2.2e-6, series) for series in nudge_neg_delta])
    require(abs(nudge_pos_final[0]) < 0.005, "zero nudge should not store a positive hidden-error differential")
    require(abs(nudge_neg_final[0]) < 0.005, "zero nudge should not store a negative hidden-error differential")
    require(np.all(np.diff(nudge_pos_final) > 0.006), "r+ hidden-error store should grow with nudge magnitude")
    require(np.all(np.diff(nudge_neg_final) < -0.006), "r- hidden-error store should grow negative with nudge magnitude")
    require(
        np.max(np.abs(nudge_pos_final + nudge_neg_final)) < 0.01,
        "r+ and r- nudge-magnitude stores should mirror each other",
    )
    require(np.max(np.abs(nudge_pos_hold - nudge_pos_final)) < 0.01, "r+ nudge-magnitude store should hold")
    require(np.max(np.abs(nudge_neg_hold - nudge_neg_final)) < 0.01, "r- nudge-magnitude store should hold")

    mismatch_data = run_ngspice(mismatch_deck, "mos_hidden_error_mismatch")
    mt, mismatch_cols = load_wrdata(mismatch_data, 4 * len(mismatch_cases))
    mismatch_gains = []
    mismatch_left_edges = []
    mismatch_right_edges = []
    for idx, (_name, _model, _label) in enumerate(mismatch_cases):
        plus_signed = mismatch_cols[4 * idx + 1] - mismatch_cols[4 * idx]
        minus_signed = mismatch_cols[4 * idx + 3] - mismatch_cols[4 * idx + 2]
        gain_series = (plus_signed - minus_signed) / (2.0 * eps)
        mismatch_gains.append(gain_series)
        require(np.all(gain_series > -1e-4), "hidden-error mismatch gain should stay nonnegative")
        require(np.max(gain_series) > 0.65, "hidden-error mismatch gain should preserve an active derivative window")
        require(
            np.mean(gain_series[np.abs(mt) > 0.35]) < 0.55 * np.max(gain_series),
            "hidden-error mismatch gain should still attenuate away from the active window",
        )
        active = gain_series > 0.5
        left_idx = int(np.argmax(active))
        right_idx = len(active) - 1 - int(np.argmax(active[::-1]))
        mismatch_left_edges.append(float(mt[left_idx]))
        mismatch_right_edges.append(float(mt[right_idx]))
        require(active.any(), "hidden-error mismatch gain should cross the active-window threshold")
        require(mt[right_idx] - mt[left_idx] > 0.55, "hidden-error active window should remain wide under mismatch")
    mismatch_left_edges = np.array(mismatch_left_edges)
    mismatch_right_edges = np.array(mismatch_right_edges)
    require(
        np.all(np.diff(mismatch_left_edges) > 0.005),
        "hidden-error derivative-window left edge should shift monotonically with z+ device threshold",
    )
    require(
        np.all(np.diff(mismatch_right_edges) > 0.005),
        "hidden-error derivative-window right edge should shift monotonically with z+ device threshold",
    )
    nominal_center_gain = mismatch_gains[1][int(np.argmin(np.abs(mt)))]
    require(nominal_center_gain > 0.7, "nominal hidden-error mismatch deck should retain high center gain")

    cm_data = run_ngspice(cm_deck, "mos_hidden_error_common_mode")
    cmt, cm_cols = load_wrdata(cm_data, 4 * len(cm_cases))
    cm_xdiff = -0.45 + 0.90 * cmt / 10e-6
    cm_gains = []
    cm_left_edges = []
    cm_right_edges = []
    cm_center_gains = []
    for idx, (_name, cm_value) in enumerate(cm_cases):
        plus_signed = cm_cols[4 * idx + 1] - cm_cols[4 * idx]
        minus_signed = cm_cols[4 * idx + 3] - cm_cols[4 * idx + 2]
        gain_series = (plus_signed - minus_signed) / (2.0 * eps)
        cm_gains.append(gain_series)
        require(np.all(gain_series > -1e-4), f"{cm_value:.2f} V common-mode hidden-error gain should stay nonnegative")
        require(np.max(gain_series) > 0.65, f"{cm_value:.2f} V common-mode hidden-error gain should preserve an active window")
        center_value = float(gain_series[int(np.argmin(np.abs(cm_xdiff)))])
        cm_center_gains.append(center_value)
        require(center_value > 0.65, f"{cm_value:.2f} V common-mode hidden-error center gain should stay useful")
        require(
            np.mean(gain_series[np.abs(cm_xdiff) > 0.35]) < 0.55 * np.max(gain_series),
            f"{cm_value:.2f} V common-mode hidden-error gain should attenuate at the edges",
        )
        active = gain_series > 0.5
        left_idx = int(np.argmax(active))
        right_idx = len(active) - 1 - int(np.argmax(active[::-1]))
        cm_left_edges.append(float(cm_xdiff[left_idx]))
        cm_right_edges.append(float(cm_xdiff[right_idx]))
        require(active.any(), f"{cm_value:.2f} V common-mode hidden-error gain should cross active threshold")
        require(cm_xdiff[right_idx] - cm_xdiff[left_idx] > 0.45, f"{cm_value:.2f} V common-mode active window should stay usable")
    cm_left_edges = np.array(cm_left_edges)
    cm_right_edges = np.array(cm_right_edges)
    cm_center_gains = np.array(cm_center_gains)

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
    hidden_plot = save_plot(fig, "mos_hidden_error_ngspice")

    gain_fig, gain_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    labels = ["negative saturation", "center", "positive saturation"]
    for label, series in zip(labels, stored_gain):
        gain_axes[0].plot(1e6 * gt, series, label=label)
    gain_axes[0].plot(1e6 * gt, gain_store_cols[-1] / 4.0, color="0.5", alpha=0.35, label="$psamp/4$")
    gain_axes[0].axhline(0, color="0.4", linewidth=0.8)
    gain_axes[0].set_ylabel("stored finite-diff gain (V/V)")
    gain_axes[0].set_title("Stored replica derivative is high only in the active window")
    gain_axes[0].grid(True, alpha=0.25)
    gain_axes[0].legend()
    zdiffs = np.array([diff for _, diff in gain_store_cases])
    order = np.argsort(zdiffs)
    gain_axes[1].plot(zdiffs[order], stored_gain_final[order], "o-", label="after sample")
    gain_axes[1].plot(zdiffs[order], stored_gain_hold[order], "s--", label="after hold")
    gain_axes[1].axhline(0, color="0.4", linewidth=0.8)
    gain_axes[1].axvline(0, color="0.4", linewidth=0.8)
    gain_axes[1].set_xlabel("$z^+ - z^-$ operating point (V)")
    gain_axes[1].set_ylabel("stored finite-diff gain (V/V)")
    gain_axes[1].set_title("Sampled derivative proxy is retained on capacitors")
    gain_axes[1].grid(True, alpha=0.25)
    gain_axes[1].legend()
    gain_fig.tight_layout()
    save_plot(gain_fig, "mos_hidden_error_gain_store_ngspice")

    sign_fig, sign_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    labels = ["negative saturation", "center", "positive saturation"]
    for label, pseries, nseries in zip(labels, pos_delta, neg_delta):
        if label == "center":
            p_label = "$r^+$ center"
            n_label = "$r^-$ center"
            alpha = 1.0
        elif label == "negative saturation":
            p_label = "$r^+$ saturation"
            n_label = "$r^-$ saturation"
            alpha = 0.75
        else:
            p_label = None
            n_label = None
            alpha = 0.75
        sign_axes[0].plot(1e6 * st, pseries, label=p_label, alpha=alpha)
        sign_axes[0].plot(1e6 * st, nseries, "--", label=n_label, alpha=alpha)
    sign_axes[0].plot(1e6 * st, sign_store_cols[-1] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    sign_axes[0].axhline(0, color="0.4", linewidth=0.8)
    sign_axes[0].set_ylabel("stored $\\delta^+ - \\delta^-$ (V)")
    sign_axes[0].set_title("Cross-connected replica outputs store derivative-weighted error sign")
    sign_axes[0].grid(True, alpha=0.25)
    sign_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    zdiffs = np.array([diff for _, diff in sign_store_cases])
    order = np.argsort(zdiffs)
    sign_axes[1].plot(zdiffs[order], pos_final[order], "o-", label="$r^+$ after sample")
    sign_axes[1].plot(zdiffs[order], neg_final[order], "s-", label="$r^-$ after sample")
    sign_axes[1].plot(zdiffs[order], pos_hold[order], "o--", color="0.35", label="$r^+$ after hold")
    sign_axes[1].plot(zdiffs[order], neg_hold[order], "s--", color="0.55", label="$r^-$ after hold")
    sign_axes[1].axhline(0, color="0.4", linewidth=0.8)
    sign_axes[1].axvline(0, color="0.4", linewidth=0.8)
    sign_axes[1].set_xlabel("$z^+ - z^-$ operating point (V)")
    sign_axes[1].set_ylabel("stored $\\delta^+ - \\delta^-$ (V)")
    sign_axes[1].set_title("Stored hidden error is signed and active-window limited")
    sign_axes[1].grid(True, alpha=0.25)
    sign_axes[1].legend(loc="upper right", ncol=2)
    sign_fig.tight_layout()
    save_plot(sign_fig, "mos_hidden_error_sign_store_ngspice")

    nudge_fig, nudge_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for nudge, pseries, nseries in zip(nudge_sweep_values, nudge_pos_delta, nudge_neg_delta):
        if nudge in (0.0, 0.04, 0.12):
            nudge_axes[0].plot(1e6 * nt, pseries, label=f"$r^+$ eps={nudge:.2f} V")
            nudge_axes[0].plot(1e6 * nt, nseries, "--", label=f"$r^-$ eps={nudge:.2f} V")
    nudge_axes[0].plot(1e6 * nt, nudge_cols[-1] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    nudge_axes[0].axhline(0, color="0.4", linewidth=0.8)
    nudge_axes[0].set_ylabel("stored $\\delta^+ - \\delta^-$ (V)")
    nudge_axes[0].set_title("Hidden-error storage scales with replica nudge")
    nudge_axes[0].grid(True, alpha=0.25)
    nudge_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    nudge_axes[1].plot(nudge_sweep_values, nudge_pos_final, "o-", label="$r^+$ after sample")
    nudge_axes[1].plot(nudge_sweep_values, nudge_neg_final, "s-", label="$r^-$ after sample")
    nudge_axes[1].plot(nudge_sweep_values, nudge_pos_hold, "o--", color="0.35", label="$r^+$ after hold")
    nudge_axes[1].plot(nudge_sweep_values, nudge_neg_hold, "s--", color="0.55", label="$r^-$ after hold")
    nudge_axes[1].axhline(0, color="0.4", linewidth=0.8)
    nudge_axes[1].set_xlabel("finite-difference nudge eps (V)")
    nudge_axes[1].set_ylabel("stored hidden-error step (V)")
    nudge_axes[1].set_title("Stored magnitude is graded, mirrored, and retained")
    nudge_axes[1].grid(True, alpha=0.25)
    nudge_axes[1].legend(loc="upper right", ncol=2)
    nudge_fig.tight_layout()
    save_plot(nudge_fig, "mos_hidden_error_nudge_sweep_ngspice")

    mismatch_fig, mismatch_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for (_name, _model, label), gain_series in zip(mismatch_cases, mismatch_gains):
        mismatch_axes[0].plot(mt, gain_series, label=label)
    mismatch_axes[0].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].axvline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].set_xlabel("$z^+ - z^-$ (V)")
    mismatch_axes[0].set_ylabel("finite-diff gain (V/V)")
    mismatch_axes[0].set_title("Hidden-error derivative window shifts under input-pair mismatch")
    mismatch_axes[0].grid(True, alpha=0.25)
    mismatch_axes[0].legend(loc="upper right")
    vt_offsets = np.array([0.53, 0.55, 0.57])
    window_widths = mismatch_right_edges - mismatch_left_edges
    mismatch_axes[1].plot(vt_offsets, mismatch_left_edges, "o-", label="left 0.5-gain edge")
    mismatch_axes[1].plot(vt_offsets, mismatch_right_edges, "s--", label="right 0.5-gain edge")
    mismatch_axes[1].plot(vt_offsets, window_widths, "^-.", label="active-window width")
    mismatch_axes[1].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[1].set_xlabel("z+ input device $V_{TO}$ (V)")
    mismatch_axes[1].set_ylabel("window edge / width (V)")
    mismatch_axes[1].set_title("Mismatch shifts both active-window edges while preserving width")
    mismatch_axes[1].grid(True, alpha=0.25)
    mismatch_axes[1].legend(loc="lower right")
    mismatch_fig.tight_layout()
    save_plot(mismatch_fig, "mos_hidden_error_mismatch_ngspice")

    cm_fig, cm_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for (_name, cm_value), gain_series in zip(cm_cases, cm_gains):
        cm_axes[0].plot(cm_xdiff, gain_series, label=f"$V_{{CM}}={cm_value:.2f}$ V")
    cm_axes[0].axhline(0, color="0.4", linewidth=0.8)
    cm_axes[0].axvline(0, color="0.4", linewidth=0.8)
    cm_axes[0].set_xlabel("$z^+ - z^-$ (V)")
    cm_axes[0].set_ylabel("finite-diff gain (V/V)")
    cm_axes[0].set_title("Hidden-error derivative window survives input common-mode shifts")
    cm_axes[0].grid(True, alpha=0.25)
    cm_axes[0].legend(loc="upper right")
    cm_values = np.array([cm_value for _name, cm_value in cm_cases])
    cm_window_widths = cm_right_edges - cm_left_edges
    cm_axes[1].plot(cm_values, cm_left_edges, "o-", label="left 0.5-gain edge")
    cm_axes[1].plot(cm_values, cm_right_edges, "s--", label="right 0.5-gain edge")
    cm_axes[1].plot(cm_values, cm_window_widths, "^-.", label="active-window width")
    cm_axes[1].plot(cm_values, cm_center_gains, "d:", label="center gain")
    cm_axes[1].axhline(0, color="0.4", linewidth=0.8)
    cm_axes[1].set_xlabel("input common-mode voltage (V)")
    cm_axes[1].set_ylabel("edge / width / gain")
    cm_axes[1].set_title("Bias shifts change gain slightly but keep a broad derivative window")
    cm_axes[1].grid(True, alpha=0.25)
    cm_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    cm_fig.tight_layout()
    save_plot(cm_fig, "mos_hidden_error_common_mode_ngspice")
    return hidden_plot


def characterize_hidden_writer_chain() -> Path:
    eps = 0.10

    def sign_store_path(src: str, gate: str, dst: str, suffix: str) -> str:
        return f"""
MSA_{suffix} {src} pbwd n_{suffix} 0 NMOS L={{LCH}} W={{WSW}}
MSB_{suffix} n_{suffix} {gate} {dst} 0 NMOS L={{LCH}} W={{WSW}}
"""

    deck = f"""
* Hidden-error store to writer/readback integration sanity check.
* The stored differential hidden-error rails are used directly as analog PMOS
* writer gates; there are no ideal comparators or behavioral update sources.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)

* Centered forward-replica pair at z+ - z- +/- epsilon.
VZPP zpp 0 {0.9 + eps / 2.0:.5f}
VZMM zmm 0 {0.9 - eps / 2.0:.5f}
VZPM zpm 0 {0.9 - eps / 2.0:.5f}
VZMP zmp 0 {0.9 + eps / 2.0:.5f}

MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zmm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zmp tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

* Cross-connected positive-error and negative-error stores.
CDP_RP cdp_rp 0 {{CERR}} IC=1.04
CDM_RP cdm_rp 0 {{CERR}} IC=1.04
CDP_RM cdp_rm 0 {{CERR}} IC=1.04
CDM_RM cdm_rm 0 {{CERR}} IC=1.04
RDP_RP cdp_rp 0 50G
RDM_RP cdm_rp 0 50G
RDP_RM cdp_rm 0 50G
RDM_RM cdm_rm 0 50G
{sign_store_path("hpm", "rp", "cdp_rp", "rp1")}
{sign_store_path("hmp", "rp", "cdp_rp", "rp2")}
{sign_store_path("hpp", "rp", "cdm_rp", "rp3")}
{sign_store_path("hmm", "rp", "cdm_rp", "rp4")}
{sign_store_path("hpp", "rm", "cdp_rm", "rm1")}
{sign_store_path("hmm", "rm", "cdp_rm", "rm2")}
{sign_store_path("hpm", "rm", "cdm_rm", "rm3")}
{sign_store_path("hmp", "rm", "cdm_rm", "rm4")}

* Positive activation rails.  The writer is a PMOS coincidence path, so the
* lower complementary rail h- is the active analog gate for x+.
VHP hp 0 1.12
VHM hm 0 0.92

* r+ hidden error: x+ * delta+ should charge W+ more than W-.
CWP_RP wp_rp 0 {{CWRITE}} IC=0.85
CWM_RP wm_rp 0 {{CWRITE}} IC=0.85
MWP_RP_A vdd paccn n_wp_rp_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_B n_wp_rp_a hm n_wp_rp_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_C n_wp_rp_b cdm_rp wp_rp vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_A vdd paccn n_wm_rp_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_B n_wm_rp_a hm n_wm_rp_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_C n_wm_rp_b cdp_rp wm_rp vdd PMOS L={{LCH}} W={{WWRITE}}

* r- hidden error: x+ * delta- should charge W- more than W+.
CWP_RM wp_rm 0 {{CWRITE}} IC=0.85
CWM_RM wm_rm 0 {{CWRITE}} IC=0.85
MWP_RM_A vdd paccn n_wp_rm_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_B n_wp_rm_a hm n_wp_rm_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_C n_wp_rm_b cdm_rm wp_rm vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_A vdd paccn n_wm_rm_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_B n_wm_rm_a hm n_wm_rm_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_C n_wm_rm_b cdp_rm wm_rm vdd PMOS L={{LCH}} W={{WWRITE}}

* Continuous signed synapse readback from the two writer outputs.
VXP xp 0 1.15
VXM xm 0 0.65

VZPP_RP zpp_rp 0 1.8
VZMP_RP zmp_rp 0 1.8
VZPN_RP zpn_rp 0 1.8
VZMN_RP zmn_rp 0 1.8
MPP_RP zpp_rp xp tailpp_rp 0 NMOS L={{LCH}} W={{WN}}
MPM_RP zmp_rp xm tailpp_rp 0 NMOS L={{LCH}} W={{WN}}
MTP_RP tailpp_rp wp_rp 0 0 NMOS L={{LCH}} W=12u
MNP_RP zpn_rp xm tailnn_rp 0 NMOS L={{LCH}} W={{WN}}
MNM_RP zmn_rp xp tailnn_rp 0 NMOS L={{LCH}} W={{WN}}
MTN_RP tailnn_rp wm_rp 0 0 NMOS L={{LCH}} W=12u

VZPP_RM zpp_rm 0 1.8
VZMP_RM zmp_rm 0 1.8
VZPN_RM zpn_rm 0 1.8
VZMN_RM zmn_rm 0 1.8
MPP_RM zpp_rm xp tailpp_rm 0 NMOS L={{LCH}} W={{WN}}
MPM_RM zmp_rm xm tailpp_rm 0 NMOS L={{LCH}} W={{WN}}
MTP_RM tailpp_rm wp_rm 0 0 NMOS L={{LCH}} W=12u
MNP_RM zpn_rm xm tailnn_rm 0 NMOS L={{LCH}} W={{WN}}
MNM_RM zmn_rm xp tailnn_rm 0 NMOS L={{LCH}} W={{WN}}
MTN_RM tailnn_rm wm_rm 0 0 NMOS L={{LCH}} W=12u

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_chain.dat v(cdp_rp) v(cdm_rp) v(cdp_rm) v(cdm_rm) v(wp_rp) v(wm_rp) v(wp_rm) v(wm_rm) i(VZPP_RP) i(VZMP_RP) i(VZPN_RP) i(VZMN_RP) i(VZPP_RM) i(VZMP_RM) i(VZPN_RM) i(VZMN_RM) v(pbwd) v(paccn)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_hidden_writer_chain")
    t, cols = load_wrdata(data, 18)

    def at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(t - time_s))])

    pos_hidden = cols[0] - cols[1]
    neg_hidden = cols[2] - cols[3]
    pos_weight = cols[4] - cols[5]
    neg_weight = cols[6] - cols[7]
    pos_read = (cols[9] - cols[8]) + (cols[11] - cols[10])
    neg_read = (cols[13] - cols[12]) + (cols[15] - cols[14])

    require(at(1.35e-6, pos_hidden) > 0.07, "r+ hidden-error store should be positive before writer phase")
    require(at(1.35e-6, neg_hidden) < -0.07, "r- hidden-error store should be negative before writer phase")
    require(abs(at(1.45e-6, pos_weight)) < 1e-4, "positive-error weight pair should start balanced")
    require(abs(at(1.45e-6, neg_weight)) < 1e-4, "negative-error weight pair should start balanced")
    require(at(2.75e-6, pos_weight) > 0.006, "stored r+ error should steer writer toward W+")
    require(at(2.75e-6, neg_weight) < -0.006, "stored r- error should steer writer toward W-")
    require(at(2.75e-6, pos_read) > at(1.45e-6, pos_read) + 2.0e-6, "r+ writer output should read back as a positive synapse contribution")
    require(at(2.75e-6, neg_read) < at(1.45e-6, neg_read) - 2.0e-6, "r- writer output should read back as a negative synapse contribution")
    require(
        abs(at(3.25e-6, pos_weight) - at(2.75e-6, pos_weight)) < 5e-4,
        "r+ written differential weight should hold after writer phase",
    )
    require(
        abs(at(3.25e-6, neg_weight) - at(2.75e-6, neg_weight)) < 5e-4,
        "r- written differential weight should hold after writer phase",
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    axes[0].plot(1e6 * t, pos_hidden, label="$r^+$ stored $\\delta^+ - \\delta^-$")
    axes[0].plot(1e6 * t, neg_hidden, label="$r^-$ stored $\\delta^+ - \\delta^-$")
    axes[0].plot(1e6 * t, cols[16] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("hidden error (V)")
    axes[0].set_title("MOS hidden-error stores provide signed analog writer gates")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(1e6 * t, pos_weight, label="$r^+$ result: $W^+ - W^-$")
    axes[1].plot(1e6 * t, neg_weight, label="$r^-$ result: $W^+ - W^-$")
    axes[1].plot(1e6 * t, cols[17] / 25.0, color="0.5", alpha=0.35, label="$\\overline{pacc}/25$")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_ylabel("weight differential (V)")
    axes[1].set_title("Stored error sign steers the four-quadrant writer")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(1e6 * t, 1e6 * pos_read, label="$r^+$ signed readback")
    axes[2].plot(1e6 * t, 1e6 * neg_read, label="$r^-$ signed readback")
    axes[2].axhline(0, color="0.4", linewidth=0.8)
    axes[2].set_xlabel("time (us)")
    axes[2].set_ylabel("read current (uA)")
    axes[2].set_title("Written rails read back with the corresponding synapse sign")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    hidden_writer_plot = save_plot(fig, "mos_hidden_writer_chain_ngspice")

    def writer_stack(prefix: str, cap: str, xgate: str, dgate: str) -> str:
        return f"""
M{prefix}A vdd paccn n_{prefix}_a vdd PMOS L={{LCH}} W={{WWRITE}}
M{prefix}B n_{prefix}_a {xgate} n_{prefix}_b vdd PMOS L={{LCH}} W={{WWRITE}}
M{prefix}C n_{prefix}_b {dgate} {cap} vdd PMOS L={{LCH}} W={{WWRITE}}
"""

    quadrant_cases = [
        ("xp_rp", "$x^+ r^+$", "hm_pos", "cdm_rp", "hm_pos", "cdp_rp", 1.0),
        ("xp_rm", "$x^+ r^-$", "hm_pos", "cdm_rm", "hm_pos", "cdp_rm", -1.0),
        ("xm_rp", "$x^- r^+$", "hp_neg", "cdp_rp", "hp_neg", "cdm_rp", -1.0),
        ("xm_rm", "$x^- r^-$", "hp_neg", "cdp_rm", "hp_neg", "cdm_rm", 1.0),
    ]
    quadrant_devices = []
    quadrant_prints = ["v(cdp_rp)", "v(cdm_rp)", "v(cdp_rm)", "v(cdm_rm)"]
    for name, _label, wp_xgate, wp_dgate, wm_xgate, wm_dgate, _expected in quadrant_cases:
        quadrant_devices.append(
            f"""
CWP_{name} wp_{name} 0 {{CWRITE}} IC=0.85
CWM_{name} wm_{name} 0 {{CWRITE}} IC=0.85
"""
            + writer_stack(f"WP_{name}", f"wp_{name}", wp_xgate, wp_dgate)
            + writer_stack(f"WM_{name}", f"wm_{name}", wm_xgate, wm_dgate)
        )
        quadrant_prints.extend([f"v(wp_{name})", f"v(wm_{name})"])

    quadrant_deck = f"""
* Hidden-error-driven four-quadrant writer sanity check.
* The same stored r+ and r- hidden-error capacitor rails steer PMOS writers
* under both positive and negative activation rail polarities.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)

VZPP zpp 0 {0.9 + eps / 2.0:.5f}
VZMM zmm 0 {0.9 - eps / 2.0:.5f}
VZPM zpm 0 {0.9 - eps / 2.0:.5f}
VZMP zmp 0 {0.9 + eps / 2.0:.5f}

MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zmm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zmp tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP cdp_rp 0 {{CERR}} IC=1.04
CDM_RP cdm_rp 0 {{CERR}} IC=1.04
CDP_RM cdp_rm 0 {{CERR}} IC=1.04
CDM_RM cdm_rm 0 {{CERR}} IC=1.04
RDP_RP cdp_rp 0 50G
RDM_RP cdm_rp 0 50G
RDP_RM cdp_rm 0 50G
RDM_RM cdm_rm 0 50G
{sign_store_path("hpm", "rp", "cdp_rp", "qrp1")}
{sign_store_path("hmp", "rp", "cdp_rp", "qrp2")}
{sign_store_path("hpp", "rp", "cdm_rp", "qrp3")}
{sign_store_path("hmm", "rp", "cdm_rp", "qrp4")}
{sign_store_path("hpp", "rm", "cdp_rm", "qrm1")}
{sign_store_path("hmm", "rm", "cdp_rm", "qrm2")}
{sign_store_path("hpm", "rm", "cdm_rm", "qrm3")}
{sign_store_path("hmp", "rm", "cdm_rm", "qrm4")}

* Positive activation: h- is low.  Negative activation: h+ is low.
VHP_POS hp_pos 0 1.12
VHM_POS hm_pos 0 0.92
VHP_NEG hp_neg 0 0.92
VHM_NEG hm_neg 0 1.12

{''.join(quadrant_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_quadrants.dat {' '.join(quadrant_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""
    quadrant_data = run_ngspice(quadrant_deck, "mos_hidden_writer_quadrants")
    qt, qcols = load_wrdata(quadrant_data, len(quadrant_prints) + 2)

    def qat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(qt - time_s))])

    q_pos_hidden = qcols[0] - qcols[1]
    q_neg_hidden = qcols[2] - qcols[3]
    require(qat(1.35e-6, q_pos_hidden) > 0.07, "quadrant r+ hidden-error store should be positive")
    require(qat(1.35e-6, q_neg_hidden) < -0.07, "quadrant r- hidden-error store should be negative")
    qdiffs = []
    for idx, (_name, _label, _wp_xgate, _wp_dgate, _wm_xgate, _wm_dgate, expected) in enumerate(quadrant_cases):
        diff = qcols[4 + 2 * idx] - qcols[5 + 2 * idx]
        qdiffs.append(diff)
        final = qat(2.75e-6, diff)
        hold = qat(3.25e-6, diff)
        require(expected * final > 0.006, f"{_name} should write the expected signed weight differential")
        require(abs(hold - final) < 5e-4, f"{_name} weight differential should hold after writer phase")
    qfinal = np.array([qat(2.75e-6, diff) for diff in qdiffs])
    require(qfinal[0] > 0 and qfinal[1] < 0 and qfinal[2] < 0 and qfinal[3] > 0, "all four update quadrants should have the expected signs")

    quadrant_fig, quadrant_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    quadrant_axes[0].plot(1e6 * qt, q_pos_hidden, label="$r^+$ stored $\\delta^+ - \\delta^-$")
    quadrant_axes[0].plot(1e6 * qt, q_neg_hidden, label="$r^-$ stored $\\delta^+ - \\delta^-$")
    quadrant_axes[0].plot(1e6 * qt, qcols[-2] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    quadrant_axes[0].axhline(0, color="0.4", linewidth=0.8)
    quadrant_axes[0].set_ylabel("hidden error (V)")
    quadrant_axes[0].set_title("One hidden-error store feeds all writer quadrants")
    quadrant_axes[0].grid(True, alpha=0.25)
    quadrant_axes[0].legend(loc="upper right")
    for (_name, label, _wp_xgate, _wp_dgate, _wm_xgate, _wm_dgate, _expected), diff in zip(quadrant_cases, qdiffs):
        quadrant_axes[1].plot(1e6 * qt, diff, label=label)
    quadrant_axes[1].plot(1e6 * qt, qcols[-1] / 25.0, color="0.5", alpha=0.35, label="$\\overline{pacc}/25$")
    quadrant_axes[1].axhline(0, color="0.4", linewidth=0.8)
    quadrant_axes[1].set_xlabel("time (us)")
    quadrant_axes[1].set_ylabel("$W^+ - W^-$ (V)")
    quadrant_axes[1].set_title("Activation/error sign products select W+ or W-")
    quadrant_axes[1].grid(True, alpha=0.25)
    quadrant_axes[1].legend(loc="upper right", ncol=2)
    quadrant_fig.tight_layout()
    save_plot(quadrant_fig, "mos_hidden_writer_quadrants_ngspice")

    magnitude_eps = [0.00, 0.03, 0.06, 0.10, 0.14]
    magnitude_devices = []
    magnitude_prints = []
    for idx, eps_mag in enumerate(magnitude_eps):
        name = f"e{idx}"
        magnitude_devices.append(
            f"""
* Stored-error magnitude to writer copy, epsilon={eps_mag:.2f} V.
VZPP_{name} zpp_{name} 0 {0.9 + eps_mag / 2.0:.5f}
VZMM_{name} zmm_{name} 0 {0.9 - eps_mag / 2.0:.5f}
VZPM_{name} zpm_{name} 0 {0.9 - eps_mag / 2.0:.5f}
VZMP_{name} zmp_{name} 0 {0.9 + eps_mag / 2.0:.5f}

MPPP_{name} hpp_{name} hpp_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_{name} hpm_{name} hpm_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_{name} hpp_{name} zpp_{name} tailp_{name} 0 NMOS L={{LCH}} W={{WN}}
MNPM_{name} hpm_{name} zmm_{name} tailp_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTP_{name} tailp_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_{name} hmp_{name} hmp_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_{name} hmm_{name} hmm_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_{name} hmp_{name} zpm_{name} tailm_{name} 0 NMOS L={{LCH}} W={{WN}}
MNMM_{name} hmm_{name} zmp_{name} tailm_{name} 0 NMOS L={{LCH}} W={{WN}}
MNTM_{name} tailm_{name} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_{name} cdp_{name} 0 {{CERR}} IC=1.04
CDM_{name} cdm_{name} 0 {{CERR}} IC=1.04
RDP_{name} cdp_{name} 0 50G
RDM_{name} cdm_{name} 0 50G
{sign_store_path(f"hpm_{name}", "rp", f"cdp_{name}", f"mag1_{name}")}
{sign_store_path(f"hmp_{name}", "rp", f"cdp_{name}", f"mag2_{name}")}
{sign_store_path(f"hpp_{name}", "rp", f"cdm_{name}", f"mag3_{name}")}
{sign_store_path(f"hmm_{name}", "rp", f"cdm_{name}", f"mag4_{name}")}

CWP_{name} wp_{name} 0 {{CWRITE}} IC=0.85
CWM_{name} wm_{name} 0 {{CWRITE}} IC=0.85
MWP_{name}A vdd paccn n_wp_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_{name}B n_wp_{name}_a hm_pos n_wp_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_{name}C n_wp_{name}_b cdm_{name} wp_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_{name}A vdd paccn n_wm_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_{name}B n_wm_{name}_a hm_pos n_wm_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_{name}C n_wm_{name}_b cdp_{name} wm_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        magnitude_prints.extend([f"v(cdp_{name})", f"v(cdm_{name})", f"v(wp_{name})", f"v(wm_{name})"])

    magnitude_deck = f"""
* Stored hidden-error magnitude to MOS writer magnitude sanity check.
* Each copy stores a different finite-difference hidden-error magnitude, then
* uses those capacitor rails directly as PMOS writer gates under the same
* positive activation and pacc phases.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)
VHP_POS hp_pos 0 1.12
VHM_POS hm_pos 0 0.92
{''.join(magnitude_devices)}
.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_magnitude.dat {' '.join(magnitude_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""

    magnitude_data = run_ngspice(magnitude_deck, "mos_hidden_writer_magnitude")
    mt, mag_cols = load_wrdata(magnitude_data, 4 * len(magnitude_eps) + 2)

    def mat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(mt - time_s))])

    mag_hidden = []
    mag_weight = []
    for idx in range(len(magnitude_eps)):
        mag_hidden.append(mag_cols[4 * idx] - mag_cols[4 * idx + 1])
        mag_weight.append(mag_cols[4 * idx + 2] - mag_cols[4 * idx + 3])
    mag_hidden_final = np.array([mat(1.35e-6, series) for series in mag_hidden])
    mag_weight_final = np.array([mat(2.75e-6, series) for series in mag_weight])
    mag_weight_hold = np.array([mat(3.25e-6, series) for series in mag_weight])
    require(abs(mag_hidden_final[0]) < 0.005, "zero hidden-error nudge should store near-zero hidden error")
    require(abs(mag_weight_final[0]) < 0.001, "zero stored hidden error should not create a differential write")
    require(np.all(np.diff(mag_hidden_final) > 0.015), "stored hidden-error magnitude should grow with replica nudge")
    require(np.all(np.diff(mag_weight_final) > 0.0015), "writer weight step should grow with stored hidden-error magnitude")
    require(mag_weight_final[-1] < 0.03, "largest hidden-error writer step should remain incremental")
    require(
        np.max(np.abs(mag_weight_hold - mag_weight_final)) < 5e-4,
        "hidden-error magnitude writer steps should hold after pacc closes",
    )

    magnitude_fig, magnitude_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for eps_mag, hseries, wseries in zip(magnitude_eps, mag_hidden, mag_weight):
        if eps_mag in (0.00, 0.06, 0.14):
            magnitude_axes[0].plot(1e6 * mt, hseries, label=f"stored eps={eps_mag:.2f} V")
            magnitude_axes[1].plot(1e6 * mt, wseries, label=f"write eps={eps_mag:.2f} V")
    magnitude_axes[0].plot(1e6 * mt, mag_cols[-2] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    magnitude_axes[0].axhline(0, color="0.4", linewidth=0.8)
    magnitude_axes[0].set_ylabel("hidden error (V)")
    magnitude_axes[0].set_title("Stored hidden-error magnitude is available before pacc")
    magnitude_axes[0].grid(True, alpha=0.25)
    magnitude_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    magnitude_axes[1].plot(1e6 * mt, mag_cols[-1] / 30.0, color="0.5", alpha=0.35, label="$\\overline{pacc}/30$")
    magnitude_axes[1].axhline(0, color="0.4", linewidth=0.8)
    magnitude_axes[1].set_xlabel("time (us)")
    magnitude_axes[1].set_ylabel("$W^+ - W^-$ (V)")
    magnitude_axes[1].set_title("MOS writer step grows with stored hidden-error magnitude")
    magnitude_axes[1].grid(True, alpha=0.25)
    magnitude_axes[1].legend(loc="upper right", ncol=2, fontsize="small")
    magnitude_fig.tight_layout()
    save_plot(magnitude_fig, "mos_hidden_writer_magnitude_ngspice")

    timing_cases = [
        ("pre", "pre-store", 0.10),
        ("edge", "store edge", 0.46),
        ("overlap", "overlap", 0.80),
        ("late", "late overlap", 1.10),
        ("gap", "settled gap", 1.55),
    ]
    timing_devices = []
    timing_prints = ["v(cdp_rp)", "v(cdm_rp)"]
    for name, _label, start_us in timing_cases:
        end_us = start_us + 0.40
        timing_devices.append(
            f"""
VPACC_{name} paccn_{name} 0 PWL(0 1.8 {start_us:.2f}u 1.8 {start_us + 0.02:.2f}u 0 {end_us:.2f}u 0 {end_us + 0.02:.2f}u 1.8 3.2u 1.8)
CWP_{name} wp_{name} 0 {{CWRITE}} IC=0.85
CWM_{name} wm_{name} 0 {{CWRITE}} IC=0.85
MWP_{name}A vdd paccn_{name} n_wp_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_{name}B n_wp_{name}_a hm_pos n_wp_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_{name}C n_wp_{name}_b cdm_rp wp_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_{name}A vdd paccn_{name} n_wm_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_{name}B n_wm_{name}_a hm_pos n_wm_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_{name}C n_wm_{name}_b cdp_rp wm_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        timing_prints.extend([f"v(wp_{name})", f"v(wm_{name})", f"v(paccn_{name})"])

    timing_deck = f"""
* Hidden-error to writer phase-timing margin check.
* Multiple writer copies see the same stored r+ hidden-error rails.  Their
* pacc pulses start before, during, and after the pbwd storage phase to expose
* the sequencing margin without behavioral update helpers.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)

VZPP zpp 0 {0.9 + eps / 2.0:.5f}
VZMM zmm 0 {0.9 - eps / 2.0:.5f}
VZPM zpm 0 {0.9 - eps / 2.0:.5f}
VZMP zmp 0 {0.9 + eps / 2.0:.5f}

MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zmm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zmp tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP cdp_rp 0 {{CERR}} IC=1.04
CDM_RP cdm_rp 0 {{CERR}} IC=1.04
RDP_RP cdp_rp 0 50G
RDM_RP cdm_rp 0 50G
{sign_store_path("hpm", "rp", "cdp_rp", "trp1")}
{sign_store_path("hmp", "rp", "cdp_rp", "trp2")}
{sign_store_path("hpp", "rp", "cdm_rp", "trp3")}
{sign_store_path("hmm", "rp", "cdm_rp", "trp4")}

VHP_POS hp_pos 0 1.12
VHM_POS hm_pos 0 0.92

{''.join(timing_devices)}

.control
set noaskquit
tran 5n 3.2u uic
wrdata mos_hidden_writer_phase_timing.dat {' '.join(timing_prints)} v(pbwd)
quit
.endc
.end
"""
    timing_data = run_ngspice(timing_deck, "mos_hidden_writer_phase_timing")
    tt, timing_cols = load_wrdata(timing_data, len(timing_prints) + 1)

    def tat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(tt - time_s))])

    timing_hidden = timing_cols[0] - timing_cols[1]
    timing_diffs = []
    timing_final = []
    for idx, (_name, _label, _start_us) in enumerate(timing_cases):
        diff = timing_cols[2 + 3 * idx] - timing_cols[3 + 3 * idx]
        timing_diffs.append(diff)
        timing_final.append(tat(2.85e-6, diff))
    timing_final = np.array(timing_final)
    require(tat(1.35e-6, timing_hidden) > 0.07, "timing deck should store a positive hidden error")
    require(abs(timing_final[0]) < 0.001, "writer pulse before hidden-error storage should not create signed update")
    require(timing_final[1] > timing_final[0] + 0.002, "writer pulse at storage edge should create a partial signed update")
    require(timing_final[1] < 0.99 * timing_final[-1], "storage-edge writer should be measurably below the settled update")
    require(np.min(timing_final[2:]) > 0.95 * timing_final[-1], "writer pulses after hidden-error settling should reach the full update")
    require(np.max(timing_final[2:]) - np.min(timing_final[2:]) < 2e-4, "settled writer timings should agree")

    timing_fig, timing_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    timing_axes[0].plot(1e6 * tt, timing_hidden, label="stored $r^+$ hidden error")
    timing_axes[0].plot(1e6 * tt, timing_cols[-1] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    timing_axes[0].axhline(0, color="0.4", linewidth=0.8)
    timing_axes[0].set_ylabel("hidden error (V)")
    timing_axes[0].set_title("Hidden-error storage settles quickly after pbwd starts")
    timing_axes[0].grid(True, alpha=0.25)
    timing_axes[0].legend(loc="upper right")
    for (_name, label, _start_us), diff in zip(timing_cases, timing_diffs):
        timing_axes[1].plot(1e6 * tt, diff, label=label)
    timing_axes[1].axhline(0, color="0.4", linewidth=0.8)
    timing_axes[1].set_xlabel("time (us)")
    timing_axes[1].set_ylabel("$W^+ - W^-$ (V)")
    timing_axes[1].set_title("pacc timing sweep exposes the storage/write margin")
    timing_axes[1].grid(True, alpha=0.25)
    timing_axes[1].legend(loc="upper left", ncol=2)
    timing_fig.tight_layout()
    save_plot(timing_fig, "mos_hidden_writer_phase_timing_ngspice")

    mismatch_levels = [
        ("strong", "PMOSHWR50", -0.50),
        ("nominal", "PMOSHWR55", -0.55),
        ("weak", "PMOSHWR60", -0.60),
    ]
    mismatch_model_defs = "\n".join(
        f".model {model} PMOS (LEVEL=1 VTO={vto:.2f} KP=90u LAMBDA=0.03)"
        for _label, model, vto in mismatch_levels
    )
    mismatch_devices = []
    mismatch_prints = ["v(cdp_rp)", "v(cdm_rp)", "v(cdp_rm)", "v(cdm_rm)"]
    for idx, (label, model, _vto) in enumerate(mismatch_levels):
        mismatch_devices.append(
            f"""
* Hidden-writer threshold-mismatch copy: {label} selected and inactive stacks.
CWP_RP_MIS{idx} wp_rp_mis{idx} 0 {{CWRITE}} IC=0.85
CWM_RP_MIS{idx} wm_rp_mis{idx} 0 {{CWRITE}} IC=0.85
MWP_RP_MIS{idx}A vdd paccn n_wp_rp_mis{idx}_a vdd {model} L={{LCH}} W={{WWRITE}}
MWP_RP_MIS{idx}B n_wp_rp_mis{idx}_a hm_pos n_wp_rp_mis{idx}_b vdd {model} L={{LCH}} W={{WWRITE}}
MWP_RP_MIS{idx}C n_wp_rp_mis{idx}_b cdm_rp wp_rp_mis{idx} vdd {model} L={{LCH}} W={{WWRITE}}
MWM_RP_MIS{idx}A vdd paccn n_wm_rp_mis{idx}_a vdd {model} L={{LCH}} W={{WWRITE}}
MWM_RP_MIS{idx}B n_wm_rp_mis{idx}_a hm_pos n_wm_rp_mis{idx}_b vdd {model} L={{LCH}} W={{WWRITE}}
MWM_RP_MIS{idx}C n_wm_rp_mis{idx}_b cdp_rp wm_rp_mis{idx} vdd {model} L={{LCH}} W={{WWRITE}}

CWP_RM_MIS{idx} wp_rm_mis{idx} 0 {{CWRITE}} IC=0.85
CWM_RM_MIS{idx} wm_rm_mis{idx} 0 {{CWRITE}} IC=0.85
MWP_RM_MIS{idx}A vdd paccn n_wp_rm_mis{idx}_a vdd {model} L={{LCH}} W={{WWRITE}}
MWP_RM_MIS{idx}B n_wp_rm_mis{idx}_a hm_pos n_wp_rm_mis{idx}_b vdd {model} L={{LCH}} W={{WWRITE}}
MWP_RM_MIS{idx}C n_wp_rm_mis{idx}_b cdm_rm wp_rm_mis{idx} vdd {model} L={{LCH}} W={{WWRITE}}
MWM_RM_MIS{idx}A vdd paccn n_wm_rm_mis{idx}_a vdd {model} L={{LCH}} W={{WWRITE}}
MWM_RM_MIS{idx}B n_wm_rm_mis{idx}_a hm_pos n_wm_rm_mis{idx}_b vdd {model} L={{LCH}} W={{WWRITE}}
MWM_RM_MIS{idx}C n_wm_rm_mis{idx}_b cdp_rm wm_rm_mis{idx} vdd {model} L={{LCH}} W={{WWRITE}}
"""
        )
        mismatch_prints.extend(
            [
                f"v(wp_rp_mis{idx})",
                f"v(wm_rp_mis{idx})",
                f"v(wp_rm_mis{idx})",
                f"v(wm_rm_mis{idx})",
            ]
        )

    mismatch_deck = f"""
* Integrated hidden-error-store to PMOS-writer threshold-mismatch sweep.
* Nominal MOS finite-difference stores create r+ and r- hidden-error rails.
* Those capacitor rails then drive writer stacks whose PMOS thresholds are
* swept together.  This checks that writer mismatch changes update gain, not
* the sign/selectivity of the integrated backward/update path.
{COMMON_MODELS}
{mismatch_model_defs}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)

VZPP zpp 0 {0.9 + eps / 2.0:.5f}
VZMM zmm 0 {0.9 - eps / 2.0:.5f}
VZPM zpm 0 {0.9 - eps / 2.0:.5f}
VZMP zmp 0 {0.9 + eps / 2.0:.5f}

MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zmm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zmp tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP cdp_rp 0 {{CERR}} IC=1.04
CDM_RP cdm_rp 0 {{CERR}} IC=1.04
CDP_RM cdp_rm 0 {{CERR}} IC=1.04
CDM_RM cdm_rm 0 {{CERR}} IC=1.04
RDP_RP cdp_rp 0 50G
RDM_RP cdm_rp 0 50G
RDP_RM cdp_rm 0 50G
RDM_RM cdm_rm 0 50G
{sign_store_path("hpm", "rp", "cdp_rp", "misrp1")}
{sign_store_path("hmp", "rp", "cdp_rp", "misrp2")}
{sign_store_path("hpp", "rp", "cdm_rp", "misrp3")}
{sign_store_path("hmm", "rp", "cdm_rp", "misrp4")}
{sign_store_path("hpp", "rm", "cdp_rm", "misrm1")}
{sign_store_path("hmm", "rm", "cdp_rm", "misrm2")}
{sign_store_path("hpm", "rm", "cdm_rm", "misrm3")}
{sign_store_path("hmp", "rm", "cdm_rm", "misrm4")}

VHP_POS hp_pos 0 1.12
VHM_POS hm_pos 0 0.92

{''.join(mismatch_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_mismatch.dat {' '.join(mismatch_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""
    mismatch_data = run_ngspice(mismatch_deck, "mos_hidden_writer_mismatch")
    hmt, hmis_cols = load_wrdata(mismatch_data, len(mismatch_prints) + 2)

    def hmat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hmt - time_s))])

    hmis_pos_hidden = hmis_cols[0] - hmis_cols[1]
    hmis_neg_hidden = hmis_cols[2] - hmis_cols[3]
    require(hmat(1.35e-6, hmis_pos_hidden) > 0.07, "mismatch deck should store positive r+ hidden error")
    require(hmat(1.35e-6, hmis_neg_hidden) < -0.07, "mismatch deck should store negative r- hidden error")

    hmis_pos_steps = []
    hmis_neg_steps = []
    hmis_rp_selected = []
    hmis_rm_selected = []
    hmis_rp_complement = []
    hmis_rm_complement = []
    for idx, (_label, _model, _vto) in enumerate(mismatch_levels):
        base = 4 + 4 * idx
        wp_rp = hmis_cols[base]
        wm_rp = hmis_cols[base + 1]
        wp_rm = hmis_cols[base + 2]
        wm_rm = hmis_cols[base + 3]
        pos_diff = wp_rp - wm_rp
        neg_diff = wp_rm - wm_rm
        hmis_pos_steps.append(hmat(2.75e-6, pos_diff))
        hmis_neg_steps.append(hmat(2.75e-6, neg_diff))
        hmis_rp_selected.append(hmat(2.75e-6, wp_rp) - hmat(1.45e-6, wp_rp))
        hmis_rm_selected.append(hmat(2.75e-6, wm_rm) - hmat(1.45e-6, wm_rm))
        hmis_rp_complement.append(hmat(2.75e-6, wm_rp) - hmat(1.45e-6, wm_rp))
        hmis_rm_complement.append(hmat(2.75e-6, wp_rm) - hmat(1.45e-6, wp_rm))
        require(
            abs(hmat(3.25e-6, pos_diff) - hmat(2.75e-6, pos_diff)) < 5e-4,
            f"{_label} r+ integrated writer step should hold",
        )
        require(
            abs(hmat(3.25e-6, neg_diff) - hmat(2.75e-6, neg_diff)) < 5e-4,
            f"{_label} r- integrated writer step should hold",
        )

    hmis_pos_steps = np.array(hmis_pos_steps)
    hmis_neg_steps = np.array(hmis_neg_steps)
    hmis_rp_selected = np.array(hmis_rp_selected)
    hmis_rm_selected = np.array(hmis_rm_selected)
    hmis_rp_complement = np.array(hmis_rp_complement)
    hmis_rm_complement = np.array(hmis_rm_complement)
    require(np.all(hmis_pos_steps > 0.0045), "integrated r+ writer should keep a usable positive step under mismatch")
    require(np.all(hmis_neg_steps < -0.0045), "integrated r- writer should keep a usable negative step under mismatch")
    require(np.all(hmis_pos_steps < 0.04), "integrated r+ writer steps should remain incremental")
    require(np.all(np.abs(hmis_neg_steps) < 0.04), "integrated r- writer steps should remain incremental")
    require(
        np.all(np.diff(hmis_pos_steps) < -5e-4),
        "integrated r+ writer step should decrease as PMOS threshold magnitude increases",
    )
    require(
        np.all(np.diff(np.abs(hmis_neg_steps)) < -5e-4),
        "integrated r- writer magnitude should decrease as PMOS threshold magnitude increases",
    )
    require(
        np.max(np.abs(hmis_pos_steps + hmis_neg_steps)) < 0.003,
        "matched writer threshold offsets should preserve r+/r- update symmetry",
    )
    require(
        np.all(hmis_rp_selected > hmis_rp_complement + 0.006),
        "integrated r+ selected rail should exceed the complementary leakage rail",
    )
    require(
        np.all(hmis_rm_selected > hmis_rm_complement + 0.006),
        "integrated r- selected rail should exceed the complementary leakage rail",
    )
    require(
        np.all(hmis_rp_complement > 0.0) and np.all(hmis_rm_complement > 0.0),
        "complementary rails should show the expected analog PMOS leakage direction",
    )

    mismatch_fig, mismatch_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hmis_x = np.arange(len(mismatch_levels))
    hmis_labels = [f"$V_{{TO}}={vto:.2f}$ V" for _label, _model, vto in mismatch_levels]
    mismatch_axes[0].plot(hmis_x, hmis_pos_steps, "o-", label="$r^+ \\to W^+-W^-$")
    mismatch_axes[0].plot(hmis_x, hmis_neg_steps, "s--", label="$r^- \\to W^+-W^-$")
    mismatch_axes[0].plot(hmis_x, hmis_rp_complement, "o-", color="0.55", alpha=0.75, label="$r^+$ complement $W^-$")
    mismatch_axes[0].plot(hmis_x, hmis_rm_complement, "s:", color="0.55", alpha=0.75, label="$r^-$ complement $W^+$")
    mismatch_axes[0].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].set_xticks(hmis_x)
    mismatch_axes[0].set_xticklabels(hmis_labels)
    mismatch_axes[0].set_ylabel("final step (V)")
    mismatch_axes[0].set_title("Integrated writer mismatch changes gain, not update sign")
    mismatch_axes[0].grid(True, alpha=0.25)
    mismatch_axes[0].legend(loc="upper right", ncol=2)

    for idx, (label, _model, vto) in enumerate(mismatch_levels):
        base = 4 + 4 * idx
        pos_diff = hmis_cols[base] - hmis_cols[base + 1]
        neg_diff = hmis_cols[base + 2] - hmis_cols[base + 3]
        linestyle = "-" if idx != 1 else "--"
        mismatch_axes[1].plot(
            1e6 * hmt,
            pos_diff,
            linestyle,
            label=f"$r^+$ {label} ({vto:.2f} V)",
        )
        mismatch_axes[1].plot(
            1e6 * hmt,
            neg_diff,
            linestyle,
            alpha=0.65,
            label=f"$r^-$ {label} ({vto:.2f} V)",
        )
    mismatch_axes[1].plot(1e6 * hmt, hmis_cols[-1] / 60.0, color="0.5", alpha=0.35, label="$\\overline{pacc}/60$")
    mismatch_axes[1].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[1].set_xlabel("time (us)")
    mismatch_axes[1].set_ylabel("$W^+ - W^-$ (V)")
    mismatch_axes[1].set_title("Stored hidden-error rails drive bounded threshold-corner writes")
    mismatch_axes[1].grid(True, alpha=0.25)
    mismatch_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    mismatch_fig.tight_layout()
    save_plot(mismatch_fig, "mos_hidden_writer_mismatch_ngspice")

    bias_levels = [0.86, 0.92, 0.98, 1.04, 1.10]
    bias_devices = []
    bias_prints = ["v(cdp_rp)", "v(cdm_rp)", "v(cdp_rm)", "v(cdm_rm)"]
    for idx, hgate in enumerate(bias_levels):
        bias_devices.append(
            f"""
* Integrated writer input-gate bias copy: h- gate={hgate:.2f} V.
VHM_BIAS{idx} hm_bias{idx} 0 {hgate:.2f}
CWP_RP_BIAS{idx} wp_rp_bias{idx} 0 {{CWRITE}} IC=0.85
CWM_RP_BIAS{idx} wm_rp_bias{idx} 0 {{CWRITE}} IC=0.85
MWP_RP_BIAS{idx}A vdd paccn n_wp_rp_bias{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_BIAS{idx}B n_wp_rp_bias{idx}_a hm_bias{idx} n_wp_rp_bias{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_BIAS{idx}C n_wp_rp_bias{idx}_b cdm_rp wp_rp_bias{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_BIAS{idx}A vdd paccn n_wm_rp_bias{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_BIAS{idx}B n_wm_rp_bias{idx}_a hm_bias{idx} n_wm_rp_bias{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_BIAS{idx}C n_wm_rp_bias{idx}_b cdp_rp wm_rp_bias{idx} vdd PMOS L={{LCH}} W={{WWRITE}}

CWP_RM_BIAS{idx} wp_rm_bias{idx} 0 {{CWRITE}} IC=0.85
CWM_RM_BIAS{idx} wm_rm_bias{idx} 0 {{CWRITE}} IC=0.85
MWP_RM_BIAS{idx}A vdd paccn n_wp_rm_bias{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_BIAS{idx}B n_wp_rm_bias{idx}_a hm_bias{idx} n_wp_rm_bias{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_BIAS{idx}C n_wp_rm_bias{idx}_b cdm_rm wp_rm_bias{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_BIAS{idx}A vdd paccn n_wm_rm_bias{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_BIAS{idx}B n_wm_rm_bias{idx}_a hm_bias{idx} n_wm_rm_bias{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_BIAS{idx}C n_wm_rm_bias{idx}_b cdp_rm wm_rm_bias{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        bias_prints.extend(
            [
                f"v(wp_rp_bias{idx})",
                f"v(wm_rp_bias{idx})",
                f"v(wp_rm_bias{idx})",
                f"v(wm_rm_bias{idx})",
            ]
        )

    bias_deck = f"""
* Integrated hidden-error-store to writer input-gate bias sweep.
* The hidden-error rails are unchanged from the nominal integrated deck.  Each
* writer copy sees a different positive-activation complementary gate h-.  A
* higher h- gate weakens both PMOS branches, so this sweep exposes the
* selected/complementary margin rather than assuming the previous bias is best.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)

VZPP zpp 0 {0.9 + eps / 2.0:.5f}
VZMM zmm 0 {0.9 - eps / 2.0:.5f}
VZPM zpm 0 {0.9 - eps / 2.0:.5f}
VZMP zmp 0 {0.9 + eps / 2.0:.5f}

MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zmm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zmp tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP cdp_rp 0 {{CERR}} IC=1.04
CDM_RP cdm_rp 0 {{CERR}} IC=1.04
CDP_RM cdp_rm 0 {{CERR}} IC=1.04
CDM_RM cdm_rm 0 {{CERR}} IC=1.04
RDP_RP cdp_rp 0 50G
RDM_RP cdm_rp 0 50G
RDP_RM cdp_rm 0 50G
RDM_RM cdm_rm 0 50G
{sign_store_path("hpm", "rp", "cdp_rp", "biasrp1")}
{sign_store_path("hmp", "rp", "cdp_rp", "biasrp2")}
{sign_store_path("hpp", "rp", "cdm_rp", "biasrp3")}
{sign_store_path("hmm", "rp", "cdm_rp", "biasrp4")}
{sign_store_path("hpp", "rm", "cdp_rm", "biasrm1")}
{sign_store_path("hmm", "rm", "cdp_rm", "biasrm2")}
{sign_store_path("hpm", "rm", "cdm_rm", "biasrm3")}
{sign_store_path("hmp", "rm", "cdm_rm", "biasrm4")}

{''.join(bias_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_gate_bias.dat {' '.join(bias_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""
    bias_data = run_ngspice(bias_deck, "mos_hidden_writer_gate_bias")
    hbt, hbias_cols = load_wrdata(bias_data, len(bias_prints) + 2)

    def hbat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hbt - time_s))])

    hbias_pos_hidden = hbias_cols[0] - hbias_cols[1]
    hbias_neg_hidden = hbias_cols[2] - hbias_cols[3]
    require(hbat(1.35e-6, hbias_pos_hidden) > 0.07, "bias deck should store positive r+ hidden error")
    require(hbat(1.35e-6, hbias_neg_hidden) < -0.07, "bias deck should store negative r- hidden error")

    bias_pos_net = []
    bias_neg_net = []
    bias_selected = []
    bias_complement = []
    for idx, _hgate in enumerate(bias_levels):
        base = 4 + 4 * idx
        wp_rp = hbias_cols[base]
        wm_rp = hbias_cols[base + 1]
        wp_rm = hbias_cols[base + 2]
        wm_rm = hbias_cols[base + 3]
        rp_selected = hbat(2.75e-6, wp_rp) - hbat(1.45e-6, wp_rp)
        rp_complement = hbat(2.75e-6, wm_rp) - hbat(1.45e-6, wm_rp)
        rm_selected = hbat(2.75e-6, wm_rm) - hbat(1.45e-6, wm_rm)
        rm_complement = hbat(2.75e-6, wp_rm) - hbat(1.45e-6, wp_rm)
        bias_selected.append(0.5 * (rp_selected + rm_selected))
        bias_complement.append(0.5 * (rp_complement + rm_complement))
        bias_pos_net.append(hbat(2.75e-6, wp_rp - wm_rp))
        bias_neg_net.append(hbat(2.75e-6, wp_rm - wm_rm))
        require(
            abs(hbat(3.25e-6, wp_rp - wm_rp) - hbat(2.75e-6, wp_rp - wm_rp)) < 5e-4,
            f"h-={_hgate:.2f} r+ bias-sweep writer step should hold",
        )
        require(
            abs(hbat(3.25e-6, wp_rm - wm_rm) - hbat(2.75e-6, wp_rm - wm_rm)) < 5e-4,
            f"h-={_hgate:.2f} r- bias-sweep writer step should hold",
        )

    bias_pos_net = np.array(bias_pos_net)
    bias_neg_net = np.array(bias_neg_net)
    bias_selected = np.array(bias_selected)
    bias_complement = np.array(bias_complement)
    bias_selectivity = bias_selected / np.maximum(bias_complement, 1e-12)
    require(np.all(bias_pos_net > 0.002), "all tested activation-gate biases should keep positive net writes")
    require(np.all(bias_neg_net < -0.002), "all tested activation-gate biases should keep negative net writes")
    require(np.all(np.diff(bias_selected) < -0.001), "selected writer current should weaken as h- gate rises")
    require(np.all(np.diff(bias_complement) < -4e-4), "complement writer leakage should weaken as h- gate rises")
    require(
        np.all(np.diff(bias_selectivity) < -0.03),
        "selected/complement ratio should worsen as h- gate rises",
    )
    require(bias_pos_net[0] > bias_pos_net[-1] + 0.008, "higher h- gate should reduce net update magnitude")
    require(
        bias_complement[0] > bias_complement[-1] + 0.003,
        "higher h- gate should reduce absolute complementary leakage",
    )
    require(bias_selectivity[0] > 2.0, "low h- gate should give the best selected/complement contrast in this sweep")

    bias_fig, bias_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    bias_axes[0].plot(bias_levels, bias_selected, "o-", label="selected rail step")
    bias_axes[0].plot(bias_levels, bias_complement, "s--", label="complement rail step")
    bias_axes[0].plot(bias_levels, bias_pos_net, "^-", label="$r^+$ net $W^+-W^-$")
    bias_axes[0].plot(bias_levels, -bias_neg_net, "v:", label="$r^-$ net magnitude")
    bias_axes[0].axhline(0, color="0.4", linewidth=0.8)
    bias_axes[0].set_ylabel("writer step (V)")
    bias_axes[0].set_title("Activation-gate bias weakens both selected and complement writes")
    bias_axes[0].grid(True, alpha=0.25)
    bias_axes[0].legend(loc="upper right", ncol=2)

    bias_axes[1].plot(bias_levels, bias_selectivity, "o-", label="selected/complement ratio")
    bias_axes[1].plot(bias_levels, bias_pos_net / bias_pos_net[0], "s--", label="normalized net update")
    bias_axes[1].axhline(1.0, color="0.4", linewidth=0.8)
    bias_axes[1].set_xlabel("$h^-$ PMOS writer gate for $x^+$ (V)")
    bias_axes[1].set_ylabel("ratio / normalized gain")
    bias_axes[1].set_title("Bias alone reduces leakage but does not improve selectivity")
    bias_axes[1].grid(True, alpha=0.25)
    bias_axes[1].legend(loc="upper left")
    bias_fig.tight_layout()
    save_plot(bias_fig, "mos_hidden_writer_gate_bias_ngspice")

    restored_deck = f"""
* Restored hidden-error gate to writer characterization.
* A skewed CMOS inverter restorer converts the small stored hidden-error
* differential into wider active-low writer gates.  This is still not an ideal
* comparator or behavioral source: the restored gates are MOS inverter output
* voltages, and those voltages directly drive the PMOS writer stacks.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u WRESTN=12u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)

VZPP zpp 0 {0.9 + eps / 2.0:.5f}
VZMM zmm 0 {0.9 - eps / 2.0:.5f}
VZPM zpm 0 {0.9 - eps / 2.0:.5f}
VZMP zmp 0 {0.9 + eps / 2.0:.5f}

MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zmm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zmp tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP cdp_rp 0 {{CERR}} IC=1.04
CDM_RP cdm_rp 0 {{CERR}} IC=1.04
CDP_RM cdp_rm 0 {{CERR}} IC=1.04
CDM_RM cdm_rm 0 {{CERR}} IC=1.04
RDP_RP cdp_rp 0 50G
RDM_RP cdm_rp 0 50G
RDP_RM cdp_rm 0 50G
RDM_RM cdm_rm 0 50G
{sign_store_path("hpm", "rp", "cdp_rp", "restrp1")}
{sign_store_path("hmp", "rp", "cdp_rp", "restrp2")}
{sign_store_path("hpp", "rp", "cdm_rp", "restrp3")}
{sign_store_path("hmm", "rp", "cdm_rp", "restrp4")}
{sign_store_path("hpp", "rm", "cdp_rm", "restrm1")}
{sign_store_path("hmm", "rm", "cdp_rm", "restrm2")}
{sign_store_path("hpm", "rm", "cdm_rm", "restrm3")}
{sign_store_path("hmp", "rm", "cdm_rm", "restrm4")}

* Skewed CMOS restorers: the higher hidden-error rail pulls its restored gate
* lower, while the lower rail is restored high enough to suppress the
* complementary PMOS writer stack.
MPRP_CDP rgp_rp cdp_rp vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP rgp_rp cdp_rp 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM rgm_rp cdm_rp vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM rgm_rp cdm_rp 0 0 NMOS L={{LCH}} W={{WRESTN}}

MPRM_CDP rgp_rm cdp_rm vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDP rgp_rm cdp_rm 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDM rgm_rm cdm_rm vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDM rgm_rm cdm_rm 0 0 NMOS L={{LCH}} W={{WRESTN}}

VHM_POS hm_pos 0 0.92

* Direct analog-gate baseline.
CWP_RP_DIR wp_rp_dir 0 {{CWRITE}} IC=0.85
CWM_RP_DIR wm_rp_dir 0 {{CWRITE}} IC=0.85
MWP_RP_DIR_A vdd paccn n_wp_rp_dir_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_DIR_B n_wp_rp_dir_a hm_pos n_wp_rp_dir_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_DIR_C n_wp_rp_dir_b cdm_rp wp_rp_dir vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_DIR_A vdd paccn n_wm_rp_dir_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_DIR_B n_wm_rp_dir_a hm_pos n_wm_rp_dir_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_DIR_C n_wm_rp_dir_b cdp_rp wm_rp_dir vdd PMOS L={{LCH}} W={{WWRITE}}

CWP_RM_DIR wp_rm_dir 0 {{CWRITE}} IC=0.85
CWM_RM_DIR wm_rm_dir 0 {{CWRITE}} IC=0.85
MWP_RM_DIR_A vdd paccn n_wp_rm_dir_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_DIR_B n_wp_rm_dir_a hm_pos n_wp_rm_dir_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_DIR_C n_wp_rm_dir_b cdm_rm wp_rm_dir vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_DIR_A vdd paccn n_wm_rm_dir_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_DIR_B n_wm_rm_dir_a hm_pos n_wm_rm_dir_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_DIR_C n_wm_rm_dir_b cdp_rm wm_rm_dir vdd PMOS L={{LCH}} W={{WWRITE}}

* Restored-gate writer.
CWP_RP_RST wp_rp_rst 0 {{CWRITE}} IC=0.85
CWM_RP_RST wm_rp_rst 0 {{CWRITE}} IC=0.85
MWP_RP_RST_A vdd paccn n_wp_rp_rst_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_RST_B n_wp_rp_rst_a hm_pos n_wp_rp_rst_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_RST_C n_wp_rp_rst_b rgp_rp wp_rp_rst vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_RST_A vdd paccn n_wm_rp_rst_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_RST_B n_wm_rp_rst_a hm_pos n_wm_rp_rst_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_RST_C n_wm_rp_rst_b rgm_rp wm_rp_rst vdd PMOS L={{LCH}} W={{WWRITE}}

CWP_RM_RST wp_rm_rst 0 {{CWRITE}} IC=0.85
CWM_RM_RST wm_rm_rst 0 {{CWRITE}} IC=0.85
MWP_RM_RST_A vdd paccn n_wp_rm_rst_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_RST_B n_wp_rm_rst_a hm_pos n_wp_rm_rst_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_RST_C n_wp_rm_rst_b rgp_rm wp_rm_rst vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_RST_A vdd paccn n_wm_rm_rst_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_RST_B n_wm_rm_rst_a hm_pos n_wm_rm_rst_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_RST_C n_wm_rm_rst_b rgm_rm wm_rm_rst vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_restored_gate.dat v(cdp_rp) v(cdm_rp) v(cdp_rm) v(cdm_rm) v(rgp_rp) v(rgm_rp) v(rgp_rm) v(rgm_rm) v(wp_rp_dir) v(wm_rp_dir) v(wp_rm_dir) v(wm_rm_dir) v(wp_rp_rst) v(wm_rp_rst) v(wp_rm_rst) v(wm_rm_rst) v(pbwd) v(paccn)
quit
.endc
.end
"""
    restored_data = run_ngspice(restored_deck, "mos_hidden_writer_restored_gate")
    hrt, hrest_cols = load_wrdata(restored_data, 18)

    def hrat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hrt - time_s))])

    hrest_pos_hidden = hrest_cols[0] - hrest_cols[1]
    hrest_neg_hidden = hrest_cols[2] - hrest_cols[3]
    rp_gate_contrast = hrest_cols[5] - hrest_cols[4]
    rm_gate_contrast = hrest_cols[6] - hrest_cols[7]
    require(hrat(1.35e-6, hrest_pos_hidden) > 0.07, "restored deck should store positive r+ hidden error")
    require(hrat(1.35e-6, hrest_neg_hidden) < -0.07, "restored deck should store negative r- hidden error")
    require(hrat(1.45e-6, rp_gate_contrast) > 0.25, "r+ restored gates should separate before pacc")
    require(hrat(1.45e-6, rm_gate_contrast) > 0.25, "r- restored gates should separate before pacc")

    direct_rp_selected = hrat(2.75e-6, hrest_cols[8]) - hrat(1.45e-6, hrest_cols[8])
    direct_rp_comp = hrat(2.75e-6, hrest_cols[9]) - hrat(1.45e-6, hrest_cols[9])
    direct_rm_comp = hrat(2.75e-6, hrest_cols[10]) - hrat(1.45e-6, hrest_cols[10])
    direct_rm_selected = hrat(2.75e-6, hrest_cols[11]) - hrat(1.45e-6, hrest_cols[11])
    rest_rp_selected = hrat(2.75e-6, hrest_cols[12]) - hrat(1.45e-6, hrest_cols[12])
    rest_rp_comp = hrat(2.75e-6, hrest_cols[13]) - hrat(1.45e-6, hrest_cols[13])
    rest_rm_comp = hrat(2.75e-6, hrest_cols[14]) - hrat(1.45e-6, hrest_cols[14])
    rest_rm_selected = hrat(2.75e-6, hrest_cols[15]) - hrat(1.45e-6, hrest_cols[15])

    direct_pos_net = hrat(2.75e-6, hrest_cols[8] - hrest_cols[9])
    direct_neg_net = hrat(2.75e-6, hrest_cols[10] - hrest_cols[11])
    rest_pos_net = hrat(2.75e-6, hrest_cols[12] - hrest_cols[13])
    rest_neg_net = hrat(2.75e-6, hrest_cols[14] - hrest_cols[15])
    direct_selectivity = 0.5 * (direct_rp_selected + direct_rm_selected) / (
        0.5 * (direct_rp_comp + direct_rm_comp)
    )
    rest_selectivity = 0.5 * (rest_rp_selected + rest_rm_selected) / (
        0.5 * (rest_rp_comp + rest_rm_comp)
    )
    require(rest_pos_net > 0.006, "restored r+ writer should keep a usable positive net update")
    require(rest_neg_net < -0.006, "restored r- writer should keep a usable negative net update")
    require(rest_pos_net < 0.09, "restored r+ writer should remain an incremental update")
    require(abs(rest_neg_net) < 0.09, "restored r- writer should remain an incremental update")
    require(abs(rest_pos_net + rest_neg_net) < 0.003, "restored r+/r- net updates should stay symmetric")
    require(rest_selectivity > direct_selectivity + 1.0, "restored gates should improve selected/complement contrast")
    require(rest_rp_comp < direct_rp_comp * 0.35, "restored r+ complementary rail should be strongly suppressed")
    require(rest_rm_comp < direct_rm_comp * 0.35, "restored r- complementary rail should be strongly suppressed")
    require(rest_rp_selected > direct_rp_selected * 0.50, "restored r+ selected update should remain useful")
    require(rest_rm_selected > direct_rm_selected * 0.50, "restored r- selected update should remain useful")
    require(
        abs(hrat(3.25e-6, hrest_cols[12] - hrest_cols[13]) - rest_pos_net) < 5e-4,
        "restored r+ written differential should hold",
    )
    require(
        abs(hrat(3.25e-6, hrest_cols[14] - hrest_cols[15]) - rest_neg_net) < 5e-4,
        "restored r- written differential should hold",
    )

    restored_fig, restored_axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    restored_axes[0].plot(1e6 * hrt, hrest_pos_hidden, label="stored $r^+$")
    restored_axes[0].plot(1e6 * hrt, hrest_neg_hidden, label="stored $r^-$")
    restored_axes[0].plot(1e6 * hrt, hrest_cols[-2] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    restored_axes[0].axhline(0, color="0.4", linewidth=0.8)
    restored_axes[0].set_ylabel("hidden error (V)")
    restored_axes[0].set_title("Stored hidden-error rails feed MOS gate restorers")
    restored_axes[0].grid(True, alpha=0.25)
    restored_axes[0].legend(loc="upper right")

    restored_axes[1].plot(1e6 * hrt, hrest_cols[4], label="$r^+$ selected gate")
    restored_axes[1].plot(1e6 * hrt, hrest_cols[5], label="$r^+$ complement gate")
    restored_axes[1].plot(1e6 * hrt, hrest_cols[7], "--", label="$r^-$ selected gate")
    restored_axes[1].plot(1e6 * hrt, hrest_cols[6], "--", label="$r^-$ complement gate")
    restored_axes[1].set_ylabel("gate voltage (V)")
    restored_axes[1].set_title("Skewed CMOS restorer widens active-low writer gates")
    restored_axes[1].grid(True, alpha=0.25)
    restored_axes[1].legend(loc="upper right", ncol=2, fontsize="small")

    restored_axes[2].plot(1e6 * hrt, hrest_cols[8] - hrest_cols[9], label="direct $r^+$ net")
    restored_axes[2].plot(1e6 * hrt, hrest_cols[10] - hrest_cols[11], label="direct $r^-$ net")
    restored_axes[2].plot(1e6 * hrt, hrest_cols[12] - hrest_cols[13], "--", label="restored $r^+$ net")
    restored_axes[2].plot(1e6 * hrt, hrest_cols[14] - hrest_cols[15], "--", label="restored $r^-$ net")
    restored_axes[2].plot(1e6 * hrt, hrest_cols[-1] / 60.0, color="0.5", alpha=0.35, label="$\\overline{pacc}/60$")
    restored_axes[2].axhline(0, color="0.4", linewidth=0.8)
    restored_axes[2].set_xlabel("time (us)")
    restored_axes[2].set_ylabel("$W^+ - W^-$ (V)")
    restored_axes[2].set_title("Restored gates suppress complement writes while preserving sign")
    restored_axes[2].grid(True, alpha=0.25)
    restored_axes[2].legend(loc="upper left", ncol=2, fontsize="small")
    restored_fig.tight_layout()
    save_plot(restored_fig, "mos_hidden_writer_restored_gate_ngspice")

    restored_mismatch_cases = [
        ("nominal", 0.55, -0.55, 0.55, -0.55),
        ("selected weak", 0.62, -0.55, 0.55, -0.55),
        ("complement weak", 0.55, -0.55, 0.55, -0.65),
        ("opposed weak", 0.62, -0.55, 0.55, -0.65),
    ]
    restored_mismatch_models = []
    restored_mismatch_devices = []
    restored_mismatch_prints = []
    for idx, (label, nsel_vto, psel_vto, ncomp_vto, pcomp_vto) in enumerate(restored_mismatch_cases):
        restored_mismatch_models.append(
            f"""
.model NRSEL{idx} NMOS (LEVEL=1 VTO={nsel_vto:.2f} KP=220u LAMBDA=0.03)
.model PRSEL{idx} PMOS (LEVEL=1 VTO={psel_vto:.2f} KP=90u LAMBDA=0.03)
.model NRCOMP{idx} NMOS (LEVEL=1 VTO={ncomp_vto:.2f} KP=220u LAMBDA=0.03)
.model PRCOMP{idx} PMOS (LEVEL=1 VTO={pcomp_vto:.2f} KP=90u LAMBDA=0.03)
"""
        )
        restored_mismatch_devices.append(
            f"""
* Restored-gate mismatch copy: {label}.
MPRP_CDP_MIS{idx} rgp_rp_mis{idx} cdp_rp vdd vdd PRSEL{idx} L={{LCH}} W={{WRESTP}}
MNRP_CDP_MIS{idx} rgp_rp_mis{idx} cdp_rp 0 0 NRSEL{idx} L={{LCH}} W={{WRESTN}}
MPRP_CDM_MIS{idx} rgm_rp_mis{idx} cdm_rp vdd vdd PRCOMP{idx} L={{LCH}} W={{WRESTP}}
MNRP_CDM_MIS{idx} rgm_rp_mis{idx} cdm_rp 0 0 NRCOMP{idx} L={{LCH}} W={{WRESTN}}

MPRM_CDP_MIS{idx} rgp_rm_mis{idx} cdp_rm vdd vdd PRCOMP{idx} L={{LCH}} W={{WRESTP}}
MNRM_CDP_MIS{idx} rgp_rm_mis{idx} cdp_rm 0 0 NRCOMP{idx} L={{LCH}} W={{WRESTN}}
MPRM_CDM_MIS{idx} rgm_rm_mis{idx} cdm_rm vdd vdd PRSEL{idx} L={{LCH}} W={{WRESTP}}
MNRM_CDM_MIS{idx} rgm_rm_mis{idx} cdm_rm 0 0 NRSEL{idx} L={{LCH}} W={{WRESTN}}

CWP_RP_RG_MIS{idx} wp_rp_rg_mis{idx} 0 {{CWRITE}} IC=0.85
CWM_RP_RG_MIS{idx} wm_rp_rg_mis{idx} 0 {{CWRITE}} IC=0.85
MWP_RP_RG_MIS{idx}A vdd paccn n_wp_rp_rg_mis{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_RG_MIS{idx}B n_wp_rp_rg_mis{idx}_a hm_pos n_wp_rp_rg_mis{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_RG_MIS{idx}C n_wp_rp_rg_mis{idx}_b rgp_rp_mis{idx} wp_rp_rg_mis{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_RG_MIS{idx}A vdd paccn n_wm_rp_rg_mis{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_RG_MIS{idx}B n_wm_rp_rg_mis{idx}_a hm_pos n_wm_rp_rg_mis{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_RG_MIS{idx}C n_wm_rp_rg_mis{idx}_b rgm_rp_mis{idx} wm_rp_rg_mis{idx} vdd PMOS L={{LCH}} W={{WWRITE}}

CWP_RM_RG_MIS{idx} wp_rm_rg_mis{idx} 0 {{CWRITE}} IC=0.85
CWM_RM_RG_MIS{idx} wm_rm_rg_mis{idx} 0 {{CWRITE}} IC=0.85
MWP_RM_RG_MIS{idx}A vdd paccn n_wp_rm_rg_mis{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_RG_MIS{idx}B n_wp_rm_rg_mis{idx}_a hm_pos n_wp_rm_rg_mis{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_RG_MIS{idx}C n_wp_rm_rg_mis{idx}_b rgp_rm_mis{idx} wp_rm_rg_mis{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_RG_MIS{idx}A vdd paccn n_wm_rm_rg_mis{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_RG_MIS{idx}B n_wm_rm_rg_mis{idx}_a hm_pos n_wm_rm_rg_mis{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_RG_MIS{idx}C n_wm_rm_rg_mis{idx}_b rgm_rm_mis{idx} wm_rm_rg_mis{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        restored_mismatch_prints.extend(
            [
                f"v(rgp_rp_mis{idx})",
                f"v(rgm_rp_mis{idx})",
                f"v(rgp_rm_mis{idx})",
                f"v(rgm_rm_mis{idx})",
                f"v(wp_rp_rg_mis{idx})",
                f"v(wm_rp_rg_mis{idx})",
                f"v(wp_rm_rg_mis{idx})",
                f"v(wm_rm_rg_mis{idx})",
            ]
        )

    restored_mismatch_deck = f"""
* Restored hidden-error gate threshold-offset characterization.
* The hidden-error store is nominal.  The CMOS restorer threshold corners are
* skewed to weaken the selected inverter, weaken the complementary pull-up, or
* combine both effects.  The goal is to prove that the restored writer remains
* signed and complement-suppressed even when the restoring stage is offset.
{COMMON_MODELS}
{''.join(restored_mismatch_models)}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u WRESTN=12u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)

VZPP zpp 0 {0.9 + eps / 2.0:.5f}
VZMM zmm 0 {0.9 - eps / 2.0:.5f}
VZPM zpm 0 {0.9 - eps / 2.0:.5f}
VZMP zmp 0 {0.9 + eps / 2.0:.5f}

MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zmm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zmp tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP cdp_rp 0 {{CERR}} IC=1.04
CDM_RP cdm_rp 0 {{CERR}} IC=1.04
CDP_RM cdp_rm 0 {{CERR}} IC=1.04
CDM_RM cdm_rm 0 {{CERR}} IC=1.04
RDP_RP cdp_rp 0 50G
RDM_RP cdm_rp 0 50G
RDP_RM cdp_rm 0 50G
RDM_RM cdm_rm 0 50G
{sign_store_path("hpm", "rp", "cdp_rp", "rgmisrp1")}
{sign_store_path("hmp", "rp", "cdp_rp", "rgmisrp2")}
{sign_store_path("hpp", "rp", "cdm_rp", "rgmisrp3")}
{sign_store_path("hmm", "rp", "cdm_rp", "rgmisrp4")}
{sign_store_path("hpp", "rm", "cdp_rm", "rgmisrm1")}
{sign_store_path("hmm", "rm", "cdp_rm", "rgmisrm2")}
{sign_store_path("hpm", "rm", "cdm_rm", "rgmisrm3")}
{sign_store_path("hmp", "rm", "cdm_rm", "rgmisrm4")}

VHM_POS hm_pos 0 0.92
{''.join(restored_mismatch_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_restored_gate_mismatch.dat v(cdp_rp) v(cdm_rp) v(cdp_rm) v(cdm_rm) {' '.join(restored_mismatch_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""
    restored_mismatch_data = run_ngspice(
        restored_mismatch_deck,
        "mos_hidden_writer_restored_gate_mismatch",
    )
    rmt, rmis_cols = load_wrdata(restored_mismatch_data, 4 + len(restored_mismatch_prints) + 2)

    def rmat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(rmt - time_s))])

    rmis_pos_hidden = rmis_cols[0] - rmis_cols[1]
    rmis_neg_hidden = rmis_cols[2] - rmis_cols[3]
    require(rmat(1.35e-6, rmis_pos_hidden) > 0.07, "restored mismatch deck should store positive r+ error")
    require(rmat(1.35e-6, rmis_neg_hidden) < -0.07, "restored mismatch deck should store negative r- error")

    gate_selected = []
    gate_complement = []
    rest_selected = []
    rest_complement = []
    rest_pos = []
    rest_neg = []
    for idx, (label, _nsel, _psel, _ncomp, _pcomp) in enumerate(restored_mismatch_cases):
        base = 4 + 8 * idx
        rgp_rp = rmis_cols[base]
        rgm_rp = rmis_cols[base + 1]
        rgp_rm = rmis_cols[base + 2]
        rgm_rm = rmis_cols[base + 3]
        wp_rp = rmis_cols[base + 4]
        wm_rp = rmis_cols[base + 5]
        wp_rm = rmis_cols[base + 6]
        wm_rm = rmis_cols[base + 7]
        selected_gate = 0.5 * (rmat(1.45e-6, rgp_rp) + rmat(1.45e-6, rgm_rm))
        complement_gate = 0.5 * (rmat(1.45e-6, rgm_rp) + rmat(1.45e-6, rgp_rm))
        selected_step = 0.5 * (
            rmat(2.75e-6, wp_rp) - rmat(1.45e-6, wp_rp)
            + rmat(2.75e-6, wm_rm) - rmat(1.45e-6, wm_rm)
        )
        complement_step = 0.5 * (
            rmat(2.75e-6, wm_rp) - rmat(1.45e-6, wm_rp)
            + rmat(2.75e-6, wp_rm) - rmat(1.45e-6, wp_rm)
        )
        pos_net = rmat(2.75e-6, wp_rp - wm_rp)
        neg_net = rmat(2.75e-6, wp_rm - wm_rm)
        gate_selected.append(selected_gate)
        gate_complement.append(complement_gate)
        rest_selected.append(selected_step)
        rest_complement.append(complement_step)
        rest_pos.append(pos_net)
        rest_neg.append(neg_net)
        require(complement_step < 5e-4, f"{label} restored complement writer step should remain suppressed")
        require(abs(pos_net + neg_net) < 0.003, f"{label} restored r+/r- writes should stay symmetric")
        if "selected weak" in label or "opposed" in label:
            require(selected_gate > 1.50, f"{label} selected gate should expose selected-NMOS threshold sensitivity")
            require(abs(pos_net) < 0.001, f"{label} restored r+ net write should collapse rather than flip")
            require(abs(neg_net) < 0.001, f"{label} restored r- net write should collapse rather than flip")
        else:
            require(complement_gate - selected_gate > 0.75, f"{label} restored gates should keep large separation")
            require(selected_step > 0.020, f"{label} restored selected writer step should remain useful")
            require(selected_step < 0.090, f"{label} restored selected writer step should stay incremental")
            require(pos_net > 0.020, f"{label} restored r+ net write should stay positive")
            require(neg_net < -0.020, f"{label} restored r- net write should stay negative")
        require(
            abs(rmat(3.25e-6, wp_rp - wm_rp) - pos_net) < 5e-4,
            f"{label} restored r+ mismatch write should hold",
        )
        require(
            abs(rmat(3.25e-6, wp_rm - wm_rm) - neg_net) < 5e-4,
            f"{label} restored r- mismatch write should hold",
        )

    gate_selected = np.array(gate_selected)
    gate_complement = np.array(gate_complement)
    rest_selected = np.array(rest_selected)
    rest_complement = np.array(rest_complement)
    rest_pos = np.array(rest_pos)
    rest_neg = np.array(rest_neg)
    require(np.max(rest_complement) < 0.001, "all restored mismatch complements should stay near-off")
    require(rest_pos[0] > 0.020 and rest_pos[2] > 0.020, "nominal and complement-weak cases should keep useful r+ writes")
    require(abs(rest_pos[1]) < 0.001 and abs(rest_pos[3]) < 0.001, "selected-weak cases should collapse rather than flip")

    rmis_fig, rmis_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    rmis_x = np.arange(len(restored_mismatch_cases))
    rmis_labels = [label for label, *_ in restored_mismatch_cases]
    rmis_axes[0].plot(rmis_x, gate_selected, "o-", label="selected restored gate")
    rmis_axes[0].plot(rmis_x, gate_complement, "s--", label="complement restored gate")
    rmis_axes[0].set_xticks(rmis_x)
    rmis_axes[0].set_xticklabels(rmis_labels, rotation=15, ha="right")
    rmis_axes[0].set_ylabel("gate voltage (V)")
    rmis_axes[0].set_title("Restored gates retain separation under threshold offsets")
    rmis_axes[0].grid(True, alpha=0.25)
    rmis_axes[0].legend(loc="center right")
    rmis_axes[1].plot(rmis_x, rest_selected, "o-", label="selected rail step")
    rmis_axes[1].plot(rmis_x, rest_complement, "s--", label="complement rail step")
    rmis_axes[1].plot(rmis_x, rest_pos, "^-", label="$r^+$ net")
    rmis_axes[1].plot(rmis_x, -rest_neg, "v:", label="$r^-$ net magnitude")
    rmis_axes[1].axhline(0, color="0.4", linewidth=0.8)
    rmis_axes[1].set_xticks(rmis_x)
    rmis_axes[1].set_xticklabels(rmis_labels, rotation=15, ha="right")
    rmis_axes[1].set_ylabel("writer step (V)")
    rmis_axes[1].set_title("Complement stays off; selected-weak corners collapse the write")
    rmis_axes[1].grid(True, alpha=0.25)
    rmis_axes[1].legend(loc="upper right", ncol=2)
    rmis_fig.tight_layout()
    save_plot(rmis_fig, "mos_hidden_writer_restored_gate_mismatch_ngspice")

    return hidden_writer_plot


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
    widths_us = [0.02, 0.05, 0.10, 0.20, 0.40]
    sweep_devices = []
    sweep_prints = []
    for idx, width in enumerate(widths_us):
        sweep_devices.append(
            f"""
VRST{idx} rst{idx} 0 PULSE(0 1.8 0.10u 5n 5n {width:.2f}u 2u)
VRSTN{idx} rstn{idx} 0 PULSE(1.8 0 0.10u 5n 5n {width:.2f}u 2u)

* NMOS-only reset copy.
CZPN{idx} zpn{idx} 0 {{CSUM}} IC=1.3
CZMN{idx} zmn{idx} 0 {{CSUM}} IC=0.5
RZPN{idx} zpn{idx} 0 100G
RZMN{idx} zmn{idx} 0 100G
MRPN{idx} zpn{idx} rst{idx} vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRMN{idx} zmn{idx} rst{idx} vcm 0 NMOS L={{LCH}} W={{WRESETN}}

* Complementary transmission-gate reset copy.
CZPT{idx} zpt{idx} 0 {{CSUM}} IC=1.3
CZMT{idx} zmt{idx} 0 {{CSUM}} IC=0.5
RZPT{idx} zpt{idx} 0 100G
RZMT{idx} zmt{idx} 0 100G
MRPTN{idx} zpt{idx} rst{idx} vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRMTN{idx} zmt{idx} rst{idx} vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRPTP{idx} zpt{idx} rstn{idx} vcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRMTP{idx} zmt{idx} rstn{idx} vcm vdd PMOS L={{LCH}} W={{WRESETP}}
"""
        )
        sweep_prints.extend([f"v(zpn{idx})", f"v(zmn{idx})", f"v(zpt{idx})", f"v(zmt{idx})"])
    sweep_deck = f"""
* MOS reset pulse-width margin and topology comparison.
{COMMON_MODELS}
.param CSUM=500p WRESETN=24u WRESETP=60u
VDD vdd 0 1.8
VCM vcm 0 0.9
{''.join(sweep_devices)}
.control
set noaskquit
tran 2n 0.8u uic
wrdata mos_reset_width.dat {' '.join(sweep_prints)}
quit
.endc
.end
"""
    write_timing_cases = [
        ("during", "during reset", 0.16, 0.34),
        ("edge", "reset edge", 0.30, 0.60),
        ("post", "post-reset", 0.55, 0.95),
        ("late", "late", 0.80, 1.20),
    ]
    write_timing_devices = []
    write_timing_prints = []
    for name, _label, start_us, end_us in write_timing_cases:
        write_timing_devices.append(
            f"""
VWP_{name} wp_{name} 0 PWL(0 0 {start_us:.2f}u 0 {start_us + 0.02:.2f}u 1.15 {end_us:.2f}u 1.15 {end_us + 0.02:.2f}u 0 2.0u 0)
CZP_{name} zp_{name} 0 {{CSUM}} IC=0.9
CZM_{name} zm_{name} 0 {{CSUM}} IC=0.9
RZP_{name} zp_{name} 0 100G
RZM_{name} zm_{name} 0 100G
MRPN_{name} zp_{name} rst vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRMN_{name} zm_{name} rst vcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRPP_{name} zp_{name} rstn vcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRMP_{name} zm_{name} rstn vcm vdd PMOS L={{LCH}} W={{WRESETP}}
MPP_{name} zp_{name} xp tail_{name} 0 NMOS L={{LCH}} W={{WN}}
MPM_{name} zm_{name} xm tail_{name} 0 NMOS L={{LCH}} W={{WN}}
MTP_{name} tail_{name} wp_{name} 0 0 NMOS L={{LCH}} W={{WTAIL}}
"""
        )
        write_timing_prints.extend([f"v(zp_{name})", f"v(zm_{name})", f"v(wp_{name})"])
    write_timing_deck = f"""
* Reset-release to synapse-write timing margin check.
* Each copy uses the same complementary reset and a positive signed synapse
* pulse at a different time relative to reset release.
{COMMON_MODELS}
.param CSUM=500p WTAIL=2u WRESETN=24u WRESETP=60u
VDD vdd 0 1.8
VCM vcm 0 0.9
VXP xp 0 1.15
VXM xm 0 0.65
VRST rst 0 PWL(0 0 0.10u 0 0.12u 1.8 0.45u 1.8 0.47u 0 2.0u 0)
VRSTN rstn 0 PWL(0 1.8 0.10u 1.8 0.12u 0 0.45u 0 0.47u 1.8 2.0u 1.8)
{''.join(write_timing_devices)}
.control
set noaskquit
tran 5n 2.0u uic
wrdata mos_reset_write_timing.dat {' '.join(write_timing_prints)} v(rst)
quit
.endc
.end
"""
    mismatch_cases = [
        ("strong", "NMOSR53", "PMOSR50", "both strong"),
        ("nominal", "NMOSR55", "PMOSR55", "nominal"),
        ("weak", "NMOSR57", "PMOSR60", "both weak"),
        ("nweak_pstrong", "NMOSR57", "PMOSR50", "N weak / P strong"),
        ("nstrong_pweak", "NMOSR53", "PMOSR60", "N strong / P weak"),
    ]
    mismatch_models = """
.model NMOSR53 NMOS (LEVEL=1 VTO=0.53 KP=220u LAMBDA=0.03)
.model NMOSR55 NMOS (LEVEL=1 VTO=0.55 KP=220u LAMBDA=0.03)
.model NMOSR57 NMOS (LEVEL=1 VTO=0.57 KP=220u LAMBDA=0.03)
.model PMOSR50 PMOS (LEVEL=1 VTO=-0.50 KP=90u LAMBDA=0.03)
.model PMOSR55 PMOS (LEVEL=1 VTO=-0.55 KP=90u LAMBDA=0.03)
.model PMOSR60 PMOS (LEVEL=1 VTO=-0.60 KP=90u LAMBDA=0.03)
"""
    mismatch_devices = []
    mismatch_prints = []
    for idx, (name, n_model, p_model, _label) in enumerate(mismatch_cases):
        mismatch_devices.append(
            f"""
* Reset mismatch copy: {name}.
CZPM{idx} zpm{idx} 0 {{CSUM}} IC=1.3
CZMM{idx} zmm{idx} 0 {{CSUM}} IC=0.5
RZPM{idx} zpm{idx} 0 100G
RZMM{idx} zmm{idx} 0 100G
MRPNM{idx} zpm{idx} rst vcm 0 {n_model} L={{LCH}} W={{WRESETN}}
MRMNM{idx} zmm{idx} rst vcm 0 {n_model} L={{LCH}} W={{WRESETN}}
MRPPM{idx} zpm{idx} rstn vcm vdd {p_model} L={{LCH}} W={{WRESETP}}
MRMPM{idx} zmm{idx} rstn vcm vdd {p_model} L={{LCH}} W={{WRESETP}}
"""
        )
        mismatch_prints.extend([f"v(zpm{idx})", f"v(zmm{idx})"])
    mismatch_deck = f"""
* Complementary reset threshold-mismatch corner check.
* Several reset transmission-gate copies clear the same deliberately mismatched
* capacitor pair under matched and skewed NMOS/PMOS threshold corners.
{COMMON_MODELS}
{mismatch_models}
.param CSUM=500p WRESETN=24u WRESETP=60u
VDD vdd 0 1.8
VCM vcm 0 0.9
VRST rst 0 PULSE(0 1.8 0.10u 5n 5n 0.40u 2u)
VRSTN rstn 0 PULSE(1.8 0 0.10u 5n 5n 0.40u 2u)
{''.join(mismatch_devices)}
.control
set noaskquit
tran 2n 0.8u uic
wrdata mos_reset_mismatch.dat {' '.join(mismatch_prints)} v(rst)
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

    sweep_data = run_ngspice(sweep_deck, "mos_reset_width")
    st, sweep_cols = load_wrdata(sweep_data, 4 * len(widths_us))
    nmos_residual = []
    tg_residual = []
    nmos_common = []
    tg_common = []
    for idx, width in enumerate(widths_us):
        sample_t = 0.10e-6 + width * 1e-6 + 0.10e-6
        sample_idx = int(np.argmin(np.abs(st - sample_t)))
        zpn = sweep_cols[4 * idx][sample_idx]
        zmn = sweep_cols[4 * idx + 1][sample_idx]
        zpt = sweep_cols[4 * idx + 2][sample_idx]
        zmt = sweep_cols[4 * idx + 3][sample_idx]
        nmos_residual.append(abs(zmn - zpn))
        tg_residual.append(abs(zmt - zpt))
        nmos_common.append(0.5 * (zpn + zmn))
        tg_common.append(0.5 * (zpt + zmt))
    nmos_residual = np.array(nmos_residual)
    tg_residual = np.array(tg_residual)
    nmos_common = np.array(nmos_common)
    tg_common = np.array(tg_common)
    require(np.all(np.diff(tg_residual) < 0.0), "transmission-gate reset residual should improve with pulse width")
    require(np.all(tg_residual < nmos_residual), "transmission-gate reset should clear differential state better than NMOS-only reset")
    require(tg_residual[-1] < 0.05, "400 ns transmission-gate reset should leave small differential residue")
    require(nmos_residual[-1] > 0.15, "400 ns NMOS-only reset should still leave visible differential residue")
    require(np.max(np.abs(tg_common - 0.9)) < 0.005, "transmission-gate reset should preserve reset common mode")
    require(np.max(np.abs(nmos_common - 0.9)) > 0.03, "NMOS-only reset should show common-mode error")

    write_timing_data = run_ngspice(write_timing_deck, "mos_reset_write_timing")
    wt, write_timing_cols = load_wrdata(write_timing_data, 3 * len(write_timing_cases) + 1)

    def wat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(wt - time_s))])

    write_signed = []
    write_common = []
    write_final = []
    for idx, (_name, _label, _start_us, _end_us) in enumerate(write_timing_cases):
        signed_case = write_timing_cols[3 * idx + 1] - write_timing_cols[3 * idx]
        common_case = 0.5 * (write_timing_cols[3 * idx + 1] + write_timing_cols[3 * idx])
        write_signed.append(signed_case)
        write_common.append(common_case)
        write_final.append(wat(1.80e-6, signed_case))
    write_final = np.array(write_final)
    require(abs(write_final[0]) < 0.010, "write pulse fully inside reset should be strongly suppressed")
    require(write_final[1] > write_final[0] + 0.02, "write pulse crossing reset release should leave a partial signed state")
    require(write_final[1] < 0.95 * write_final[-1], "reset-edge write should remain below a clean post-reset write")
    require(np.min(write_final[2:]) > 0.95 * write_final[-1], "post-reset writes should reach the full signed state")
    require(np.max(write_final[2:]) - np.min(write_final[2:]) < 0.004, "post-reset write timings should agree")
    require(np.max(np.abs([wat(0.50e-6, common_case) - 0.9 for common_case in write_common])) < 0.02, "reset/write timing copies should leave reset near common mode")

    mismatch_data = run_ngspice(mismatch_deck, "mos_reset_mismatch")
    mt, mismatch_cols = load_wrdata(mismatch_data, 2 * len(mismatch_cases) + 1)
    mismatch_residual = []
    mismatch_common = []
    mismatch_signed_series = []
    sample_idx = int(np.argmin(np.abs(mt - 0.62e-6)))
    for idx, (_name, _n_model, _p_model, _label) in enumerate(mismatch_cases):
        signed_case = mismatch_cols[2 * idx + 1] - mismatch_cols[2 * idx]
        common_case = 0.5 * (mismatch_cols[2 * idx + 1] + mismatch_cols[2 * idx])
        mismatch_signed_series.append(signed_case)
        mismatch_residual.append(abs(float(signed_case[sample_idx])))
        mismatch_common.append(float(common_case[sample_idx]))
    mismatch_residual = np.array(mismatch_residual)
    mismatch_common = np.array(mismatch_common)
    require(np.all(mismatch_residual < 0.08), "mismatched transmission-gate reset should clear most differential residue")
    require(mismatch_residual[2] > mismatch_residual[0], "weak reset threshold corner should clear less than strong corner")
    require(np.max(np.abs(mismatch_common - 0.9)) < 0.015, "mismatched transmission-gate reset should keep common mode near target")

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
    reset_plot = save_plot(fig, "mos_reset_precharge_ngspice")

    sweep_fig, sweep_axes = plt.subplots(2, 1, figsize=(7.2, 5.6))
    sweep_axes[0].semilogy(widths_us, nmos_residual, "o-", label="NMOS-only reset")
    sweep_axes[0].semilogy(widths_us, tg_residual, "s--", label="transmission-gate reset")
    sweep_axes[0].set_ylabel("$|z^- - z^+|$ after reset (V)")
    sweep_axes[0].set_title("Complementary reset gives usable pulse-width margin")
    sweep_axes[0].grid(True, which="both", alpha=0.25)
    sweep_axes[0].legend()
    sweep_axes[1].plot(widths_us, nmos_common, "o-", label="NMOS-only common mode")
    sweep_axes[1].plot(widths_us, tg_common, "s--", label="transmission-gate common mode")
    sweep_axes[1].axhline(0.9, color="0.4", linewidth=0.8, label="$V_{CM}$")
    sweep_axes[1].set_xlabel("reset high time (us)")
    sweep_axes[1].set_ylabel("post-reset common mode (V)")
    sweep_axes[1].set_title("Transmission gate preserves common mode while clearing both rails")
    sweep_axes[1].grid(True, alpha=0.25)
    sweep_axes[1].legend()
    sweep_fig.tight_layout()
    save_plot(sweep_fig, "mos_reset_width_ngspice")

    write_timing_fig, write_timing_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for (_name, label, _start_us, _end_us), signed_case in zip(write_timing_cases, write_signed):
        write_timing_axes[0].plot(1e6 * wt, signed_case, label=label)
    write_timing_axes[0].plot(1e6 * wt, write_timing_cols[-1] / 12.0, color="0.5", alpha=0.35, label="$reset/12$")
    write_timing_axes[0].axhline(0, color="0.4", linewidth=0.8)
    write_timing_axes[0].set_ylabel("$z^- - z^+$ (V)")
    write_timing_axes[0].set_title("Reset suppresses writes until release")
    write_timing_axes[0].grid(True, alpha=0.25)
    write_timing_axes[0].legend(loc="upper right", ncol=2)
    labels = [label for _name, label, _start_us, _end_us in write_timing_cases]
    write_timing_axes[1].bar(labels, write_final)
    write_timing_axes[1].axhline(0, color="0.4", linewidth=0.8)
    write_timing_axes[1].set_ylabel("final $z^- - z^+$ (V)")
    write_timing_axes[1].set_title("Clean post-reset writes agree; reset-edge writes are partial")
    write_timing_axes[1].grid(True, axis="y", alpha=0.25)
    write_timing_fig.tight_layout()
    save_plot(write_timing_fig, "mos_reset_write_timing_ngspice")

    mismatch_fig, mismatch_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    for (_name, _n_model, _p_model, label), signed_case in zip(mismatch_cases, mismatch_signed_series):
        if label in ("both strong", "nominal", "both weak"):
            mismatch_axes[0].plot(1e6 * mt, signed_case, label=label)
        else:
            mismatch_axes[0].plot(1e6 * mt, signed_case, "--", label=label)
    mismatch_axes[0].plot(1e6 * mt, mismatch_cols[-1] / 20.0, color="0.5", alpha=0.35, label="$reset/20$")
    mismatch_axes[0].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].set_ylabel("$z^- - z^+$ (V)")
    mismatch_axes[0].set_title("Transmission-gate reset stays effective across threshold corners")
    mismatch_axes[0].grid(True, alpha=0.25)
    mismatch_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    labels = [label for _name, _n_model, _p_model, label in mismatch_cases]
    xpos = np.arange(len(labels))
    mismatch_axes[1].bar(xpos - 0.18, mismatch_residual, width=0.36, label="residual $|z^- - z^+|$")
    mismatch_axes[1].bar(xpos + 0.18, np.abs(mismatch_common - 0.9), width=0.36, label="common-mode error")
    mismatch_axes[1].set_xticks(xpos)
    mismatch_axes[1].set_xticklabels(labels, rotation=15, ha="right")
    mismatch_axes[1].set_ylabel("post-reset error (V)")
    mismatch_axes[1].set_title("Weak and skewed corners still reset within margin")
    mismatch_axes[1].grid(True, axis="y", alpha=0.25)
    mismatch_axes[1].legend(loc="upper left")
    mismatch_fig.tight_layout()
    save_plot(mismatch_fig, "mos_reset_mismatch_ngspice")
    return reset_plot


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
CWP_P{idx} wp_p{idx} 0 {{CWRITE}} IC=0.85
CWM_P{idx} wm_p{idx} 0 {{CWRITE}} IC=0.85
MWPP{idx}A vdd paccn{idx} n1p_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWPP{idx}B n1p_{idx} lo n2p_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWPP{idx}C n2p_{idx} lo wp_p{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM{idx}A vdd paccn{idx} n3p_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM{idx}B n3p_{idx} lo n4p_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM{idx}C n4p_{idx} hi wm_p{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
CWP_N{idx} wp_n{idx} 0 {{CWRITE}} IC=0.85
CWM_N{idx} wm_n{idx} 0 {{CWRITE}} IC=0.85
MWNP{idx}A vdd paccn{idx} n1n_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWNP{idx}B n1n_{idx} lo n2n_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWNP{idx}C n2n_{idx} hi wp_n{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM{idx}A vdd paccn{idx} n3n_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM{idx}B n3n_{idx} lo n4n_{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM{idx}C n4n_{idx} lo wm_n{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        sweep_prints.extend([f"v(wp_p{idx})", f"v(wm_p{idx})", f"v(wp_n{idx})", f"v(wm_n{idx})"])
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
    analog_gate_levels = [1.20, 1.10, 1.00, 0.90]
    analog_devices = []
    analog_prints = []
    for idx, xgate in enumerate(analog_gate_levels):
        analog_devices.append(
            f"""
VXG_X{idx} xg_x{idx} 0 {xgate:.2f}
VDG_X{idx} dg_x{idx} 0 0.95
CWP_X{idx} wp_x{idx} 0 {{CWRITE}} IC=0.85
CWM_X{idx} wm_x{idx} 0 {{CWRITE}} IC=0.85
MWP_X{idx}A vdd paccn n_wp_x{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_X{idx}B n_wp_x{idx}_a xg_x{idx} n_wp_x{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_X{idx}C n_wp_x{idx}_b dg_x{idx} wp_x{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_X{idx}A vdd paccn n_wm_x{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_X{idx}B n_wm_x{idx}_a xg_x{idx} n_wm_x{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_X{idx}C n_wm_x{idx}_b hi wm_x{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        analog_prints.extend([f"v(wp_x{idx})", f"v(wm_x{idx})"])
    for idx, dgate in enumerate(analog_gate_levels):
        analog_devices.append(
            f"""
VXG_D{idx} xg_d{idx} 0 0.95
VDG_D{idx} dg_d{idx} 0 {dgate:.2f}
CWP_D{idx} wp_d{idx} 0 {{CWRITE}} IC=0.85
CWM_D{idx} wm_d{idx} 0 {{CWRITE}} IC=0.85
MWP_D{idx}A vdd paccn n_wp_d{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_D{idx}B n_wp_d{idx}_a xg_d{idx} n_wp_d{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_D{idx}C n_wp_d{idx}_b dg_d{idx} wp_d{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_D{idx}A vdd paccn n_wm_d{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_D{idx}B n_wm_d{idx}_a xg_d{idx} n_wm_d{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_D{idx}C n_wm_d{idx}_b hi wm_d{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        analog_prints.extend([f"v(wp_d{idx})", f"v(wm_d{idx})"])
    analog_gate_deck = f"""
* Four-quadrant writer analog gate-strength sweep.
* The selected same-sign branch uses active-low PMOS gates.  Lower activation
* or hidden-error gate voltage should increase the W+ update magnitude while
* the inactive W- branch stays quiet.
{COMMON_MODELS}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VPACC_N paccn 0 PULSE(1.8 0 0.5u 20n 20n 0.8u 5.0u)
VHI hi 0 1.8
{''.join(analog_devices)}
.control
set noaskquit
tran 5n 2.0u uic
wrdata mos_writer_analog_gate.dat {' '.join(analog_prints)} v(paccn)
quit
.endc
.end
"""
    mismatch_levels = [
        ("strong", "PMOSW50", -0.50),
        ("nominal", "PMOSW55", -0.55),
        ("weak", "PMOSW60", -0.60),
    ]
    mismatch_devices = []
    mismatch_prints = []
    for idx, (label, model, _vto) in enumerate(mismatch_levels):
        mismatch_devices.append(
            f"""
* Writer threshold-mismatch copy: {label} selected stacks.
CWP_MISP{idx} wp_misp{idx} 0 {{CWRITE}} IC=0.85
CWM_MISP{idx} wm_misp{idx} 0 {{CWRITE}} IC=0.85
MWP_MISP{idx}A vdd paccn n_wp_misp{idx}_a vdd {model} L={{LCH}} W={{WWRITE}}
MWP_MISP{idx}B n_wp_misp{idx}_a xg n_wp_misp{idx}_b vdd {model} L={{LCH}} W={{WWRITE}}
MWP_MISP{idx}C n_wp_misp{idx}_b dg wp_misp{idx} vdd {model} L={{LCH}} W={{WWRITE}}
MWM_MISP{idx}A vdd paccn n_wm_misp{idx}_a vdd {model} L={{LCH}} W={{WWRITE}}
MWM_MISP{idx}B n_wm_misp{idx}_a xg n_wm_misp{idx}_b vdd {model} L={{LCH}} W={{WWRITE}}
MWM_MISP{idx}C n_wm_misp{idx}_b hi wm_misp{idx} vdd {model} L={{LCH}} W={{WWRITE}}

CWP_MISN{idx} wp_misn{idx} 0 {{CWRITE}} IC=0.85
CWM_MISN{idx} wm_misn{idx} 0 {{CWRITE}} IC=0.85
MWP_MISN{idx}A vdd paccn n_wp_misn{idx}_a vdd {model} L={{LCH}} W={{WWRITE}}
MWP_MISN{idx}B n_wp_misn{idx}_a xg n_wp_misn{idx}_b vdd {model} L={{LCH}} W={{WWRITE}}
MWP_MISN{idx}C n_wp_misn{idx}_b hi wp_misn{idx} vdd {model} L={{LCH}} W={{WWRITE}}
MWM_MISN{idx}A vdd paccn n_wm_misn{idx}_a vdd {model} L={{LCH}} W={{WWRITE}}
MWM_MISN{idx}B n_wm_misn{idx}_a xg n_wm_misn{idx}_b vdd {model} L={{LCH}} W={{WWRITE}}
MWM_MISN{idx}C n_wm_misn{idx}_b dg wm_misn{idx} vdd {model} L={{LCH}} W={{WWRITE}}
"""
        )
        mismatch_prints.extend(
            [
                f"v(wp_misp{idx})",
                f"v(wm_misp{idx})",
                f"v(wp_misn{idx})",
                f"v(wm_misn{idx})",
            ]
        )
    mismatch_model_defs = "\n".join(
        f".model {model} PMOS (LEVEL=1 VTO={vto:.2f} KP=90u LAMBDA=0.03)"
        for _label, model, vto in mismatch_levels
    )
    mismatch_deck = f"""
* Four-quadrant writer PMOS-threshold mismatch sweep.
* The same analog activation/error gates drive both selected signs.  Threshold
* offsets should change update gain, not branch sign or inactive-rail quietness.
{COMMON_MODELS}
{mismatch_model_defs}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VPACC_N paccn 0 PULSE(1.8 0 0.5u 20n 20n 0.8u 5.0u)
VXG xg 0 0.90
VDG dg 0 0.90
VHI hi 0 1.8
{''.join(mismatch_devices)}
.control
set noaskquit
tran 5n 2.0u uic
wrdata mos_writer_mismatch.dat {' '.join(mismatch_prints)} v(paccn)
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
    repeated_deck = f"""
* Repeated single-sample writer pulse sanity check.
{COMMON_MODELS}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VPACC_N paccn 0 PULSE(1.8 0 0.3u 10n 10n 0.10u 0.40u)
VHI hi 0 1.8
VLO lo 0 0

* Same-sign path: repeated coincidences should accumulate on W+ only.
CWP_P wp_p 0 {{CWRITE}} IC=0.85
CWM_P wm_p 0 {{CWRITE}} IC=0.85
MWPP1 vdd paccn n1p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPP2 n1p lo n2p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPP3 n2p lo wp_p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM1 vdd paccn n3p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM2 n3p lo n4p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPM3 n4p hi wm_p vdd PMOS L={{LCH}} W={{WWRITE}}

* Opposite-sign path: repeated coincidences should accumulate on W- only.
CWP_N wp_n 0 {{CWRITE}} IC=0.85
CWM_N wm_n 0 {{CWRITE}} IC=0.85
MWNP1 vdd paccn n1n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNP2 n1n lo n2n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNP3 n2n hi wp_n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM1 vdd paccn n3n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM2 n3n lo n4n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNM3 n4n lo wm_n vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 5n 4.4u uic
wrdata mos_writer_repeated.dat v(wp_p) v(wm_p) v(wp_n) v(wm_n) v(paccn)
quit
.endc
.end
"""
    alternating_deck = f"""
* Alternating same-weight writer/readback sanity check.
{COMMON_MODELS}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VCM cm 0 0.9
VDIFF xp xm 0.5
RXP xp cm 1G
RXM xm cm 1G
VHI hi 0 1.8
VLO lo 0 0
VPWP_N pwpn 0 PWL(0 1.8 0.50u 1.8 0.51u 0 0.61u 0 0.62u 1.8 0.90u 1.8 0.91u 0 1.01u 0 1.02u 1.8 1.30u 1.8 1.31u 0 1.41u 0 1.42u 1.8 4.0u 1.8)
VPWM_N pwmn 0 PWL(0 1.8 1.90u 1.8 1.91u 0 2.01u 0 2.02u 1.8 2.30u 1.8 2.31u 0 2.41u 0 2.42u 1.8 2.70u 1.8 2.71u 0 2.81u 0 2.82u 1.8 3.10u 1.8 3.11u 0 3.21u 0 3.22u 1.8 4.0u 1.8)

CWP_ALT wp 0 {{CWRITE}} IC=0.85
CWM_ALT wm 0 {{CWRITE}} IC=0.85

* Same-sign pulses charge W+ on this weight pair.
MWPPA vdd pwpn n1p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPPB n1p lo n2p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPPC n2p lo wp vdd PMOS L={{LCH}} W={{WWRITE}}
MWPMIA vdd pwpn n3p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPMIB n3p lo n4p vdd PMOS L={{LCH}} W={{WWRITE}}
MWPMIC n4p hi wm vdd PMOS L={{LCH}} W={{WWRITE}}

* Opposite-sign pulses later charge W- on the same weight pair.
MWNPIA vdd pwmn n1n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNPIB n1n lo n2n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNPIC n2n hi wp vdd PMOS L={{LCH}} W={{WWRITE}}
MWNMA vdd pwmn n3n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNMB n3n lo n4n vdd PMOS L={{LCH}} W={{WWRITE}}
MWNMC n4n lo wm vdd PMOS L={{LCH}} W={{WWRITE}}

* Continuous read copies use the same stored W+ and W- rails.
VZPP zpp 0 1.8
VZMP zmp 0 1.8
MPP zpp xp tailp 0 NMOS L={{LCH}} W={{WN}}
MPM zmp xm tailp 0 NMOS L={{LCH}} W={{WN}}
MTP tailp wp 0 0 NMOS L={{LCH}} W=12u

VZPN zpn 0 1.8
VZMN zmn 0 1.8
MNP zpn xm tailn 0 NMOS L={{LCH}} W={{WN}}
MNM zmn xp tailn 0 NMOS L={{LCH}} W={{WN}}
MTN tailn wm 0 0 NMOS L={{LCH}} W=12u

.control
set noaskquit
tran 5n 3.8u uic
wrdata mos_writer_alternating.dat v(wp) v(wm) i(VZPP) i(VZMP) i(VZPN) i(VZMN) v(pwpn) v(pwmn)
quit
.endc
.end
"""
    retention_deck = f"""
* Writer retention and synapse read-disturb sanity check.
{COMMON_MODELS}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VCM cm 0 0.9
VDIFF xp xm 0.5
RXP xp cm 1G
RXM xm cm 1G
VPACC_N paccn 0 PULSE(1.8 0 0.5u 20n 20n 0.8u 20u)
VHI hi 0 1.8
VLO lo 0 0

* Same-sign writer charges W+, then the cap is continuously used as a
* positive synapse tail gate for the rest of the transient.
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

* Opposite-sign writer charges W-, then the cap is continuously used as a
* negative synapse tail gate for the rest of the transient.
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
tran 10n 10u uic
wrdata mos_writer_retention.dat v(wp_p) v(wm_p) i(VZPP) i(VZMP) v(wp_n) v(wm_n) i(VZPN) i(VZMN) v(paccn)
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
    _, sweep_cols = load_wrdata(sweep_data, 4 * len(widths_us))
    selected_wp_delta = np.array([sweep_cols[4 * idx][-1] - sweep_cols[4 * idx][0] for idx in range(len(widths_us))])
    inactive_wm_delta = np.array([sweep_cols[4 * idx + 1][-1] - sweep_cols[4 * idx + 1][0] for idx in range(len(widths_us))])
    inactive_wp_delta = np.array([sweep_cols[4 * idx + 2][-1] - sweep_cols[4 * idx + 2][0] for idx in range(len(widths_us))])
    selected_wm_delta = np.array([sweep_cols[4 * idx + 3][-1] - sweep_cols[4 * idx + 3][0] for idx in range(len(widths_us))])
    require(np.all(np.diff(selected_wp_delta) > 0.0), "W+ writer charge should increase with active pulse width")
    require(np.all(np.diff(selected_wm_delta) > 0.0), "W- writer charge should increase with active pulse width")
    require(np.max(np.abs(inactive_wm_delta)) < 1e-3, "inactive W- branch should stay quiet in W+ width sweep")
    require(np.max(np.abs(inactive_wp_delta)) < 1e-3, "inactive W+ branch should stay quiet in W- width sweep")
    # This is an incremental writer, not a rail-to-rail latch.  Over this small
    # voltage excursion the selected charge should be close to proportional to
    # coincidence time.
    fit_wp = np.polyfit(np.array(widths_us), selected_wp_delta, 1)
    predicted_wp = np.polyval(fit_wp, widths_us)
    fit_wm = np.polyfit(np.array(widths_us), selected_wm_delta, 1)
    predicted_wm = np.polyval(fit_wm, widths_us)

    def r_squared(measured: np.ndarray, predicted: np.ndarray) -> float:
        residual = measured - predicted
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((measured - np.mean(measured)) ** 2))
        return 1.0 - ss_res / ss_tot

    require(r_squared(selected_wp_delta, predicted_wp) > 0.98, "W+ writer pulse-width response should be near-linear")
    require(r_squared(selected_wm_delta, predicted_wm) > 0.98, "W- writer pulse-width response should be near-linear")

    analog_data = run_ngspice(analog_gate_deck, "mos_writer_analog_gate")
    atime_gate, analog_cols = load_wrdata(analog_data, 2 * len(analog_gate_levels) * 2 + 1)
    xgate_steps = np.array(
        [analog_cols[2 * idx][-1] - analog_cols[2 * idx][0] for idx in range(len(analog_gate_levels))]
    )
    xgate_inactive = np.array(
        [analog_cols[2 * idx + 1][-1] - analog_cols[2 * idx + 1][0] for idx in range(len(analog_gate_levels))]
    )
    d_offset = 2 * len(analog_gate_levels)
    dgate_steps = np.array(
        [analog_cols[d_offset + 2 * idx][-1] - analog_cols[d_offset + 2 * idx][0] for idx in range(len(analog_gate_levels))]
    )
    dgate_inactive = np.array(
        [analog_cols[d_offset + 2 * idx + 1][-1] - analog_cols[d_offset + 2 * idx + 1][0] for idx in range(len(analog_gate_levels))]
    )
    require(np.all(np.diff(xgate_steps) > 5e-4), "W+ step should increase as activation gate gets more active-low")
    require(np.all(np.diff(dgate_steps) > 5e-4), "W+ step should increase as hidden-error gate gets more active-low")
    require(xgate_steps[-1] - xgate_steps[0] > 0.005, "activation gate sweep should have visible dynamic range")
    require(dgate_steps[-1] - dgate_steps[0] > 0.005, "hidden-error gate sweep should have visible dynamic range")
    require(np.max(np.abs(xgate_inactive)) < 1e-3, "analog activation sweep should leave inactive W- branch quiet")
    require(np.max(np.abs(dgate_inactive)) < 1e-3, "analog hidden-error sweep should leave inactive W- branch quiet")
    require(xgate_steps[0] > 0.0 and dgate_steps[0] > 0.0, "weak analog writer gates should still produce small positive steps")
    require(xgate_steps[-1] < 0.25 and dgate_steps[-1] < 0.25, "strong analog writer gates should remain in incremental range")

    mismatch_data = run_ngspice(mismatch_deck, "mos_writer_mismatch")
    mt, mismatch_cols = load_wrdata(mismatch_data, 4 * len(mismatch_levels) + 1)
    mismatch_wp_selected = np.array(
        [mismatch_cols[4 * idx][-1] - mismatch_cols[4 * idx][0] for idx in range(len(mismatch_levels))]
    )
    mismatch_wm_inactive = np.array(
        [mismatch_cols[4 * idx + 1][-1] - mismatch_cols[4 * idx + 1][0] for idx in range(len(mismatch_levels))]
    )
    mismatch_wp_inactive = np.array(
        [mismatch_cols[4 * idx + 2][-1] - mismatch_cols[4 * idx + 2][0] for idx in range(len(mismatch_levels))]
    )
    mismatch_wm_selected = np.array(
        [mismatch_cols[4 * idx + 3][-1] - mismatch_cols[4 * idx + 3][0] for idx in range(len(mismatch_levels))]
    )
    require(np.all(mismatch_wp_selected > 0.0055), "W+ writer should keep a usable selected step under threshold mismatch")
    require(np.all(mismatch_wm_selected > 0.0055), "W- writer should keep a usable selected step under threshold mismatch")
    require(np.all(mismatch_wp_selected < 0.08), "W+ mismatch steps should remain incremental")
    require(np.all(mismatch_wm_selected < 0.08), "W- mismatch steps should remain incremental")
    require(
        np.all(np.diff(mismatch_wp_selected) < -5e-4),
        "W+ writer step should decrease as PMOS threshold magnitude increases",
    )
    require(
        np.all(np.diff(mismatch_wm_selected) < -5e-4),
        "W- writer step should decrease as PMOS threshold magnitude increases",
    )
    require(
        np.max(np.abs(mismatch_wp_selected - mismatch_wm_selected)) < 0.003,
        "matched threshold offsets should preserve W+/W- write symmetry",
    )
    require(np.max(np.abs(mismatch_wm_inactive)) < 1e-3, "mismatched inactive W- branches should stay quiet")
    require(np.max(np.abs(mismatch_wp_inactive)) < 1e-3, "mismatched inactive W+ branches should stay quiet")

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

    repeated_data = run_ngspice(repeated_deck, "mos_writer_repeated")
    rpt, repeated_cols = load_wrdata(repeated_data, 5)
    sample_times = 0.43e-6 + 0.40e-6 * np.arange(10)
    sample_indices = [int(np.argmin(np.abs(rpt - ts))) for ts in sample_times]
    wp_p_samples = repeated_cols[0][sample_indices]
    wm_p_samples = repeated_cols[1][sample_indices]
    wp_n_samples = repeated_cols[2][sample_indices]
    wm_n_samples = repeated_cols[3][sample_indices]
    wp_p_steps = wp_p_samples - repeated_cols[0][0]
    wm_n_steps = wm_n_samples - repeated_cols[3][0]
    wm_p_steps = wm_p_samples - repeated_cols[1][0]
    wp_n_steps = wp_n_samples - repeated_cols[2][0]
    pulse_numbers = np.arange(1, len(sample_times) + 1)
    wp_p_fit = np.polyfit(pulse_numbers, wp_p_steps, 1)
    wm_n_fit = np.polyfit(pulse_numbers, wm_n_steps, 1)
    wp_p_pred = np.polyval(wp_p_fit, pulse_numbers)
    wm_n_pred = np.polyval(wm_n_fit, pulse_numbers)
    require(np.all(np.diff(wp_p_steps) > 0.004), "repeated W+ writer pulses should accumulate monotonically")
    require(np.all(np.diff(wm_n_steps) > 0.004), "repeated W- writer pulses should accumulate monotonically")
    require(np.max(np.abs(wm_p_steps)) < 1e-3, "inactive W- rail should stay quiet over repeated W+ pulses")
    require(np.max(np.abs(wp_n_steps)) < 1e-3, "inactive W+ rail should stay quiet over repeated W- pulses")
    require(r_squared(wp_p_steps, wp_p_pred) > 0.995, "repeated W+ steps should be near-linear with pulse count")
    require(r_squared(wm_n_steps, wm_n_pred) > 0.995, "repeated W- steps should be near-linear with pulse count")
    require(wp_p_samples[-1] < 1.05, "repeated W+ pulses should remain in the incremental write range")
    require(wm_n_samples[-1] < 1.05, "repeated W- pulses should remain in the incremental write range")

    alternating_data = run_ngspice(alternating_deck, "mos_writer_alternating")
    atime, alternating_cols = load_wrdata(alternating_data, 8)
    alt_pos_contrib = alternating_cols[3] - alternating_cols[2]
    alt_neg_contrib = alternating_cols[5] - alternating_cols[4]
    alt_net_contrib = alt_pos_contrib + alt_neg_contrib

    def alt_at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(atime - time_s))])

    baseline_net = alt_at(0.40e-6, alt_net_contrib)
    positive_net = alt_at(1.65e-6, alt_net_contrib)
    partial_net = alt_at(2.55e-6, alt_net_contrib)
    negative_net = alt_at(3.55e-6, alt_net_contrib)
    require(abs(baseline_net) < 1e-6, "equal initial W+ and W- rails should read near zero net contribution")
    require(positive_net > baseline_net + 15e-6, "W+ pulses should move same-weight readback positive")
    require(partial_net < positive_net - 8e-6, "later W- pulses should reduce the positive readback")
    require(partial_net > 0.0, "partial W- compensation should not overcorrect too early")
    require(negative_net < -5e-6, "enough W- pulses should push same-weight readback negative")
    require(alt_at(3.55e-6, alternating_cols[0]) > alt_at(0.40e-6, alternating_cols[0]) + 0.02, "alternating deck should have written W+")
    require(alt_at(3.55e-6, alternating_cols[1]) > alt_at(0.40e-6, alternating_cols[1]) + 0.03, "alternating deck should have written W-")

    retention_data = run_ngspice(retention_deck, "mos_writer_retention")
    ht, retention_cols = load_wrdata(retention_data, 9)
    hold_pos_contrib = retention_cols[3] - retention_cols[2]
    hold_neg_contrib = retention_cols[7] - retention_cols[6]

    def hold_at(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(ht - time_s))])

    hold_start = 2.0e-6
    hold_end = 9.5e-6
    require(hold_at(hold_start, retention_cols[0]) > 0.90, "retention W+ writer should reach active read range")
    require(hold_at(hold_start, retention_cols[5]) > 0.90, "retention W- writer should reach active read range")
    require(abs(hold_at(hold_start, retention_cols[1]) - 0.85) < 1e-3, "retention inactive W- rail should stay quiet")
    require(abs(hold_at(hold_start, retention_cols[4]) - 0.85) < 1e-3, "retention inactive W+ rail should stay quiet")
    wp_hold_drift = abs(hold_at(hold_end, retention_cols[0]) - hold_at(hold_start, retention_cols[0]))
    wm_hold_drift = abs(hold_at(hold_end, retention_cols[5]) - hold_at(hold_start, retention_cols[5]))
    require(wp_hold_drift < 1e-5, "continuous synapse read should not disturb stored W+ gate voltage")
    require(wm_hold_drift < 1e-5, "continuous synapse read should not disturb stored W- gate voltage")
    pos_hold_drift = abs(hold_at(hold_end, hold_pos_contrib) - hold_at(hold_start, hold_pos_contrib))
    neg_hold_drift = abs(hold_at(hold_end, hold_neg_contrib) - hold_at(hold_start, hold_neg_contrib))
    require(pos_hold_drift < 0.5e-6, "positive read contribution should remain stable during hold")
    require(neg_hold_drift < 0.5e-6, "negative read contribution should remain stable during hold")

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
    axes[2].plot(widths_us, selected_wp_delta, "o-", label="selected $W^+$ step")
    axes[2].plot(widths_us, selected_wm_delta, "o-", label="selected $W^-$ step")
    axes[2].plot(widths_us, inactive_wm_delta, "o-", label="inactive $W^-$ step")
    axes[2].plot(widths_us, inactive_wp_delta, "o-", label="inactive $W^+$ step")
    axes[2].plot(widths_us, predicted_wp, "--", color="0.35", label="$W^+$ linear fit")
    axes[2].plot(widths_us, predicted_wm, ":", color="0.35", label="$W^-$ linear fit")
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
    writer_plot = save_plot(fig, "mos_writer_ngspice")

    repeated_fig, repeated_axes = plt.subplots(2, 1, figsize=(7.2, 5.6))
    repeated_axes[0].plot(1e6 * rpt, repeated_cols[0] - repeated_cols[0][0], label="selected repeated $W^+$")
    repeated_axes[0].plot(
        1e6 * rpt,
        repeated_cols[3] - repeated_cols[3][0],
        "--",
        label="selected repeated $W^-$",
    )
    repeated_axes[0].plot(1e6 * rpt, repeated_cols[1] - repeated_cols[1][0], color="0.5", alpha=0.7, label="inactive $W^-$")
    repeated_axes[0].plot(1e6 * rpt, repeated_cols[2] - repeated_cols[2][0], ":", color="0.5", alpha=0.85, label="inactive $W^+$")
    repeated_axes[0].plot(1e6 * rpt, repeated_cols[4] / 18.0, color="0.35", alpha=0.35, label="$\\overline{pacc}/18$")
    repeated_axes[0].set_ylabel("$\\Delta V_W$ (V)")
    repeated_axes[0].set_title("Repeated single-sample writer pulses accumulate")
    repeated_axes[0].grid(True, alpha=0.25)
    repeated_axes[0].legend(loc="upper left", ncol=2)
    repeated_axes[1].plot(pulse_numbers, wp_p_steps, "o-", label="$W^+$ after each pulse")
    repeated_axes[1].plot(pulse_numbers, wm_n_steps, "s--", label="$W^-$ after each pulse")
    repeated_axes[1].plot(pulse_numbers, wp_p_pred, color="0.35", alpha=0.8, label="$W^+$ linear fit")
    repeated_axes[1].plot(pulse_numbers, wm_n_pred, ":", color="0.35", alpha=0.8, label="$W^-$ linear fit")
    repeated_axes[1].axhline(0, color="0.4", linewidth=0.8)
    repeated_axes[1].set_xlabel("pulse count")
    repeated_axes[1].set_ylabel("$\\Delta V_W$ (V)")
    repeated_axes[1].set_title("Stored update is near-linear over ten online writes")
    repeated_axes[1].grid(True, alpha=0.25)
    repeated_axes[1].legend(loc="upper left", ncol=2)
    repeated_fig.tight_layout()
    save_plot(repeated_fig, "mos_writer_repeated_ngspice")

    analog_fig, analog_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    analog_strength = 1.8 - np.array(analog_gate_levels)
    analog_axes[0].plot(analog_strength, xgate_steps, "o-", label="sweep activation gate")
    analog_axes[0].plot(analog_strength, dgate_steps, "s--", label="sweep hidden-error gate")
    analog_axes[0].plot(analog_strength, xgate_inactive, "o-", color="0.55", alpha=0.75, label="inactive W- during activation sweep")
    analog_axes[0].plot(analog_strength, dgate_inactive, "s:", color="0.55", alpha=0.75, label="inactive W- during error sweep")
    analog_axes[0].axhline(0, color="0.4", linewidth=0.8)
    analog_axes[0].set_ylabel("$\\Delta V_W$ (V)")
    analog_axes[0].set_title("Writer magnitude grows with analog gate strength")
    analog_axes[0].grid(True, alpha=0.25)
    analog_axes[0].legend(loc="upper left", ncol=2)
    for idx, level in enumerate(analog_gate_levels):
        analog_axes[1].plot(1e6 * atime_gate, analog_cols[2 * idx] - analog_cols[2 * idx][0], label=f"$x_g$={level:.2f} V")
    analog_axes[1].plot(1e6 * atime_gate, (1.8 - analog_cols[-1]) / 250.0, color="0.5", alpha=0.35, label="$pacc_{active}/250$")
    analog_axes[1].axhline(0, color="0.4", linewidth=0.8)
    analog_axes[1].set_xlabel("time (us)")
    analog_axes[1].set_ylabel("$\\Delta W^+$ (V)")
    analog_axes[1].set_title("Activation-gate sweep produces graded weight steps")
    analog_axes[1].grid(True, alpha=0.25)
    analog_axes[1].legend(loc="upper left", ncol=2)
    analog_fig.tight_layout()
    save_plot(analog_fig, "mos_writer_analog_gate_ngspice")

    mismatch_fig, mismatch_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    mismatch_labels = [f"$V_{{TO}}={vto:.2f}$ V" for _label, _model, vto in mismatch_levels]
    mismatch_x = np.arange(len(mismatch_levels))
    mismatch_axes[0].plot(mismatch_x, mismatch_wp_selected, "o-", label="selected $W^+$")
    mismatch_axes[0].plot(mismatch_x, mismatch_wm_selected, "s--", label="selected $W^-$")
    mismatch_axes[0].plot(mismatch_x, mismatch_wm_inactive, "o-", color="0.55", alpha=0.75, label="inactive $W^-$")
    mismatch_axes[0].plot(mismatch_x, mismatch_wp_inactive, "s:", color="0.55", alpha=0.75, label="inactive $W^+$")
    mismatch_axes[0].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[0].set_xticks(mismatch_x)
    mismatch_axes[0].set_xticklabels(mismatch_labels)
    mismatch_axes[0].set_ylabel("$\\Delta V_W$ (V)")
    mismatch_axes[0].set_title("PMOS threshold mismatch changes writer gain, not sign")
    mismatch_axes[0].grid(True, alpha=0.25)
    mismatch_axes[0].legend(loc="upper right", ncol=2)
    for idx, (label, _model, vto) in enumerate(mismatch_levels):
        linestyle = "-" if idx != 1 else "--"
        mismatch_axes[1].plot(
            1e6 * mt,
            mismatch_cols[4 * idx] - mismatch_cols[4 * idx][0],
            linestyle,
            label=f"$W^+$ {label} ({vto:.2f} V)",
        )
    mismatch_axes[1].plot(1e6 * mt, (1.8 - mismatch_cols[-1]) / 250.0, color="0.5", alpha=0.35, label="$pacc_{active}/250$")
    mismatch_axes[1].axhline(0, color="0.4", linewidth=0.8)
    mismatch_axes[1].set_xlabel("time (us)")
    mismatch_axes[1].set_ylabel("$\\Delta W^+$ (V)")
    mismatch_axes[1].set_title("Selected writer traces remain monotone and bounded")
    mismatch_axes[1].grid(True, alpha=0.25)
    mismatch_axes[1].legend(loc="upper left")
    mismatch_fig.tight_layout()
    save_plot(mismatch_fig, "mos_writer_mismatch_ngspice")

    alternating_fig, alternating_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    alternating_axes[0].plot(1e6 * atime, 1e6 * alt_net_contrib, label="net signed read contribution")
    alternating_axes[0].plot(1e6 * atime, 1e6 * alt_pos_contrib, "--", label="$W^+$ read component")
    alternating_axes[0].plot(1e6 * atime, 1e6 * alt_neg_contrib, ":", label="$W^-$ read component")
    alternating_axes[0].plot(1e6 * atime, alternating_cols[6] / 100.0, color="0.5", alpha=0.35, label="$\\overline{pacc}_{W+}/100$")
    alternating_axes[0].plot(1e6 * atime, -alternating_cols[7] / 100.0, color="0.25", alpha=0.25, label="$-\\overline{pacc}_{W-}/100$")
    alternating_axes[0].axhline(0, color="0.4", linewidth=0.8)
    alternating_axes[0].set_ylabel("read current (uA)")
    alternating_axes[0].set_title("Alternating W+ then W- writes reverse same-weight readback")
    alternating_axes[0].grid(True, alpha=0.25)
    alternating_axes[0].legend(loc="upper right", ncol=2)
    alternating_axes[1].plot(1e6 * atime, alternating_cols[0] - alternating_cols[0][0], label="$\\Delta W^+$")
    alternating_axes[1].plot(1e6 * atime, alternating_cols[1] - alternating_cols[1][0], label="$\\Delta W^-$")
    alternating_axes[1].axhline(0, color="0.4", linewidth=0.8)
    alternating_axes[1].set_xlabel("time (us)")
    alternating_axes[1].set_ylabel("stored weight step (V)")
    alternating_axes[1].set_title("Both rails retain their writes; sign comes from differential readout")
    alternating_axes[1].grid(True, alpha=0.25)
    alternating_axes[1].legend()
    alternating_fig.tight_layout()
    save_plot(alternating_fig, "mos_writer_alternating_ngspice")

    retention_fig, retention_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    retention_axes[0].plot(1e6 * ht, retention_cols[0], label="$W^+$ written/read")
    retention_axes[0].plot(1e6 * ht, retention_cols[5], label="$W^-$ written/read")
    retention_axes[0].plot(1e6 * ht, retention_cols[1], "--", color="0.5", alpha=0.75, label="inactive rails")
    retention_axes[0].plot(1e6 * ht, retention_cols[4], ":", color="0.5", alpha=0.75)
    retention_axes[0].plot(1e6 * ht, retention_cols[8] / 6.0, color="0.35", alpha=0.35, label="$\\overline{pacc}/6$")
    retention_axes[0].axvspan(2.0, 9.5, color="0.8", alpha=0.18, label="read-disturb hold window")
    retention_axes[0].set_ylabel("weight cap voltage (V)")
    retention_axes[0].set_title("Written weight caps hold while used as synapse tail gates")
    retention_axes[0].grid(True, alpha=0.25)
    retention_axes[0].legend(loc="lower right", ncol=2)
    retention_axes[1].plot(1e6 * ht, 1e6 * hold_pos_contrib, label="$W^+$ read contribution")
    retention_axes[1].plot(1e6 * ht, 1e6 * hold_neg_contrib, label="$W^-$ read contribution")
    retention_axes[1].axvspan(2.0, 9.5, color="0.8", alpha=0.18, label="hold window")
    retention_axes[1].axhline(0, color="0.4", linewidth=0.8)
    retention_axes[1].set_xlabel("time (us)")
    retention_axes[1].set_ylabel("read current (uA)")
    retention_axes[1].set_title("Continuous read current stays stable after write phase")
    retention_axes[1].grid(True, alpha=0.25)
    retention_axes[1].legend(loc="upper right")
    retention_fig.tight_layout()
    save_plot(retention_fig, "mos_writer_retention_ngspice")
    return writer_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["synapse", "forward", "chain", "hidden", "update", "reset", "writer"],
        help="Run only one characterization.",
    )
    args = parser.parse_args()
    jobs = {
        "synapse": characterize_synapse,
        "forward": characterize_forward_pair,
        "chain": characterize_synapse_forward_chain,
        "hidden": characterize_hidden_error,
        "update": characterize_hidden_writer_chain,
        "reset": characterize_reset_precharge,
        "writer": characterize_writer,
    }
    selected = [args.only] if args.only else list(jobs)
    for name in selected:
        path = jobs[name]()
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
