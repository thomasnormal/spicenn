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

    strength_wn_values = [12, 18, 24, 36, 48, 72]
    strength_devices = []
    strength_prints = []
    for idx, wn_rest in enumerate(strength_wn_values):
        strength_devices.append(
            f"""
* Restorer NMOS-strength copy under selected-side weak threshold, WN={wn_rest}u.
MPRP_CDP_STR{idx} rgp_rp_str{idx} cdp_rp vdd vdd PRSEL_STR L={{LCH}} W={{WRESTP}}
MNRP_CDP_STR{idx} rgp_rp_str{idx} cdp_rp 0 0 NRSEL_STR L={{LCH}} W={wn_rest}u
MPRP_CDM_STR{idx} rgm_rp_str{idx} cdm_rp vdd vdd PRCOMP_STR L={{LCH}} W={{WRESTP}}
MNRP_CDM_STR{idx} rgm_rp_str{idx} cdm_rp 0 0 NRCOMP_STR L={{LCH}} W={wn_rest}u

MPRM_CDP_STR{idx} rgp_rm_str{idx} cdp_rm vdd vdd PRCOMP_STR L={{LCH}} W={{WRESTP}}
MNRM_CDP_STR{idx} rgp_rm_str{idx} cdp_rm 0 0 NRCOMP_STR L={{LCH}} W={wn_rest}u
MPRM_CDM_STR{idx} rgm_rm_str{idx} cdm_rm vdd vdd PRSEL_STR L={{LCH}} W={{WRESTP}}
MNRM_CDM_STR{idx} rgm_rm_str{idx} cdm_rm 0 0 NRSEL_STR L={{LCH}} W={wn_rest}u

CWP_RP_STR{idx} wp_rp_str{idx} 0 {{CWRITE}} IC=0.85
CWM_RP_STR{idx} wm_rp_str{idx} 0 {{CWRITE}} IC=0.85
MWP_RP_STR{idx}A vdd paccn n_wp_rp_str{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_STR{idx}B n_wp_rp_str{idx}_a hm_pos n_wp_rp_str{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_STR{idx}C n_wp_rp_str{idx}_b rgp_rp_str{idx} wp_rp_str{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_STR{idx}A vdd paccn n_wm_rp_str{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_STR{idx}B n_wm_rp_str{idx}_a hm_pos n_wm_rp_str{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_STR{idx}C n_wm_rp_str{idx}_b rgm_rp_str{idx} wm_rp_str{idx} vdd PMOS L={{LCH}} W={{WWRITE}}

CWP_RM_STR{idx} wp_rm_str{idx} 0 {{CWRITE}} IC=0.85
CWM_RM_STR{idx} wm_rm_str{idx} 0 {{CWRITE}} IC=0.85
MWP_RM_STR{idx}A vdd paccn n_wp_rm_str{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_STR{idx}B n_wp_rm_str{idx}_a hm_pos n_wp_rm_str{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_STR{idx}C n_wp_rm_str{idx}_b rgp_rm_str{idx} wp_rm_str{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_STR{idx}A vdd paccn n_wm_rm_str{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_STR{idx}B n_wm_rm_str{idx}_a hm_pos n_wm_rm_str{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_STR{idx}C n_wm_rm_str{idx}_b rgm_rm_str{idx} wm_rm_str{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        strength_prints.extend(
            [
                f"v(rgp_rp_str{idx})",
                f"v(rgm_rp_str{idx})",
                f"v(rgp_rm_str{idx})",
                f"v(rgm_rm_str{idx})",
                f"v(wp_rp_str{idx})",
                f"v(wm_rp_str{idx})",
                f"v(wp_rm_str{idx})",
                f"v(wm_rm_str{idx})",
            ]
        )

    strength_deck = f"""
* Restored hidden-error gate NMOS-strength sweep under selected weak threshold.
* The previous restored-gate mismatch deck showed that a selected-side +70 mV
* NMOS threshold offset collapses the write.  This sweep asks whether sizing
* the restorer NMOS wider recovers that corner without turning on the
* complementary writer branch.
{COMMON_MODELS}
.model NRSEL_STR NMOS (LEVEL=1 VTO=0.62 KP=220u LAMBDA=0.03)
.model PRSEL_STR PMOS (LEVEL=1 VTO=-0.55 KP=90u LAMBDA=0.03)
.model NRCOMP_STR NMOS (LEVEL=1 VTO=0.55 KP=220u LAMBDA=0.03)
.model PRCOMP_STR PMOS (LEVEL=1 VTO=-0.55 KP=90u LAMBDA=0.03)
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u WRESTP=300u
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
{sign_store_path("hpm", "rp", "cdp_rp", "rgstrrp1")}
{sign_store_path("hmp", "rp", "cdp_rp", "rgstrrp2")}
{sign_store_path("hpp", "rp", "cdm_rp", "rgstrrp3")}
{sign_store_path("hmm", "rp", "cdm_rp", "rgstrrp4")}
{sign_store_path("hpp", "rm", "cdp_rm", "rgstrrm1")}
{sign_store_path("hmm", "rm", "cdp_rm", "rgstrrm2")}
{sign_store_path("hpm", "rm", "cdm_rm", "rgstrrm3")}
{sign_store_path("hmp", "rm", "cdm_rm", "rgstrrm4")}

VHM_POS hm_pos 0 0.92
{''.join(strength_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_restored_gate_strength.dat v(cdp_rp) v(cdm_rp) v(cdp_rm) v(cdm_rm) {' '.join(strength_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""
    strength_data = run_ngspice(strength_deck, "mos_hidden_writer_restored_gate_strength")
    stt, str_cols = load_wrdata(strength_data, 4 + len(strength_prints) + 2)

    def stat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(stt - time_s))])

    strength_pos_hidden = str_cols[0] - str_cols[1]
    strength_neg_hidden = str_cols[2] - str_cols[3]
    require(stat(1.35e-6, strength_pos_hidden) > 0.07, "strength deck should store positive r+ error")
    require(stat(1.35e-6, strength_neg_hidden) < -0.07, "strength deck should store negative r- error")

    strength_selected_gate = []
    strength_complement_gate = []
    strength_selected_step = []
    strength_complement_step = []
    strength_pos_net = []
    strength_neg_net = []
    for idx, wn_rest in enumerate(strength_wn_values):
        base = 4 + 8 * idx
        rgp_rp = str_cols[base]
        rgm_rp = str_cols[base + 1]
        rgp_rm = str_cols[base + 2]
        rgm_rm = str_cols[base + 3]
        wp_rp = str_cols[base + 4]
        wm_rp = str_cols[base + 5]
        wp_rm = str_cols[base + 6]
        wm_rm = str_cols[base + 7]
        selected_gate = 0.5 * (stat(1.45e-6, rgp_rp) + stat(1.45e-6, rgm_rm))
        complement_gate = 0.5 * (stat(1.45e-6, rgm_rp) + stat(1.45e-6, rgp_rm))
        selected_step = 0.5 * (
            stat(2.75e-6, wp_rp) - stat(1.45e-6, wp_rp)
            + stat(2.75e-6, wm_rm) - stat(1.45e-6, wm_rm)
        )
        complement_step = 0.5 * (
            stat(2.75e-6, wm_rp) - stat(1.45e-6, wm_rp)
            + stat(2.75e-6, wp_rm) - stat(1.45e-6, wp_rm)
        )
        pos_net = stat(2.75e-6, wp_rp - wm_rp)
        neg_net = stat(2.75e-6, wp_rm - wm_rm)
        strength_selected_gate.append(selected_gate)
        strength_complement_gate.append(complement_gate)
        strength_selected_step.append(selected_step)
        strength_complement_step.append(complement_step)
        strength_pos_net.append(pos_net)
        strength_neg_net.append(neg_net)
        require(abs(pos_net + neg_net) < 0.003, f"WN={wn_rest}u recovered writes should stay symmetric")
        require(
            abs(stat(3.25e-6, wp_rp - wm_rp) - pos_net) < 5e-4,
            f"WN={wn_rest}u recovered r+ write should hold",
        )
        require(
            abs(stat(3.25e-6, wp_rm - wm_rm) - neg_net) < 5e-4,
            f"WN={wn_rest}u recovered r- write should hold",
        )

    strength_selected_gate = np.array(strength_selected_gate)
    strength_complement_gate = np.array(strength_complement_gate)
    strength_selected_step = np.array(strength_selected_step)
    strength_complement_step = np.array(strength_complement_step)
    strength_pos_net = np.array(strength_pos_net)
    strength_neg_net = np.array(strength_neg_net)
    recovered = (strength_pos_net > 0.020) & (strength_complement_step < 5e-4)
    overdriven = (strength_complement_step > 0.020) & (np.abs(strength_pos_net) < 0.001)
    require(abs(strength_pos_net[0]) < 0.001, "weakest restorer NMOS should reproduce the selected-weak collapse")
    require(recovered.any(), "some stronger restorer NMOS sizing should recover the selected-weak update")
    require(
        np.all(strength_pos_net[recovered] < 0.12),
        "recovered selected-weak updates should remain incremental",
    )
    require(
        np.max(strength_complement_step[recovered]) < 5e-4,
        "recovered selected-weak sizing should keep complement branch suppressed",
    )
    require(
        overdriven.any(),
        "too-strong restorer NMOS should expose complement turn-on instead of being counted as robust",
    )

    strength_fig, strength_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    recovered_widths = np.array(strength_wn_values)[recovered]
    if recovered_widths.size:
        for axis in strength_axes:
            axis.axvspan(
                float(np.min(recovered_widths)) - 1.0,
                float(np.max(recovered_widths)) + 1.0,
                color="tab:green",
                alpha=0.08,
                label="usable sizing window" if axis is strength_axes[0] else None,
            )
    strength_axes[0].plot(strength_wn_values, strength_selected_gate, "o-", label="selected restored gate")
    strength_axes[0].plot(strength_wn_values, strength_complement_gate, "s--", label="complement restored gate")
    strength_axes[0].axhline(0.9, color="0.4", linewidth=0.8, alpha=0.5)
    strength_axes[0].set_ylabel("gate voltage (V)")
    strength_axes[0].set_title("Restorer NMOS sizing has a finite selected-corner window")
    strength_axes[0].grid(True, alpha=0.25)
    strength_axes[0].legend(loc="center right")
    strength_axes[1].plot(strength_wn_values, strength_selected_step, "o-", label="selected rail step")
    strength_axes[1].plot(strength_wn_values, strength_complement_step, "s--", label="complement rail step")
    strength_axes[1].plot(strength_wn_values, strength_pos_net, "^-", label="$r^+$ net")
    strength_axes[1].plot(strength_wn_values, -strength_neg_net, "v:", label="$r^-$ net magnitude")
    strength_axes[1].axhline(0, color="0.4", linewidth=0.8)
    strength_axes[1].set_xlabel("restorer NMOS width (um)")
    strength_axes[1].set_ylabel("writer step (V)")
    strength_axes[1].set_title("Weak sizing collapses; over-strong sizing cancels through complement turn-on")
    strength_axes[1].grid(True, alpha=0.25)
    strength_axes[1].legend(loc="upper left", ncol=2)
    strength_fig.tight_layout()
    save_plot(strength_fig, "mos_hidden_writer_restored_gate_strength_ngspice")

    swing_eps_values = [0.03, 0.05, 0.07, 0.10, 0.14]
    swing_devices = []
    swing_prints = []
    for idx, eps_swing in enumerate(swing_eps_values):
        swing_devices.append(
            f"""
* Restored-gate hidden-error swing copy, finite-difference epsilon={eps_swing:.2f} V.
VZPP_SW{idx} zpp_sw{idx} 0 {0.9 + eps_swing / 2.0:.5f}
VZMM_SW{idx} zmm_sw{idx} 0 {0.9 - eps_swing / 2.0:.5f}
VZPM_SW{idx} zpm_sw{idx} 0 {0.9 - eps_swing / 2.0:.5f}
VZMP_SW{idx} zmp_sw{idx} 0 {0.9 + eps_swing / 2.0:.5f}

MPPP_SW{idx} hpp_sw{idx} hpp_sw{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_SW{idx} hpm_sw{idx} hpm_sw{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_SW{idx} hpp_sw{idx} zpp_sw{idx} tailp_sw{idx} 0 NMOS L={{LCH}} W={{WN}}
MNPM_SW{idx} hpm_sw{idx} zmm_sw{idx} tailp_sw{idx} 0 NMOS L={{LCH}} W={{WN}}
MNTP_SW{idx} tailp_sw{idx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_SW{idx} hmp_sw{idx} hmp_sw{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_SW{idx} hmm_sw{idx} hmm_sw{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_SW{idx} hmp_sw{idx} zpm_sw{idx} tailm_sw{idx} 0 NMOS L={{LCH}} W={{WN}}
MNMM_SW{idx} hmm_sw{idx} zmp_sw{idx} tailm_sw{idx} 0 NMOS L={{LCH}} W={{WN}}
MNTM_SW{idx} tailm_sw{idx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_SW{idx} cdp_rp_sw{idx} 0 {{CERR}} IC=1.04
CDM_RP_SW{idx} cdm_rp_sw{idx} 0 {{CERR}} IC=1.04
CDP_RM_SW{idx} cdp_rm_sw{idx} 0 {{CERR}} IC=1.04
CDM_RM_SW{idx} cdm_rm_sw{idx} 0 {{CERR}} IC=1.04
RDP_RP_SW{idx} cdp_rp_sw{idx} 0 50G
RDM_RP_SW{idx} cdm_rp_sw{idx} 0 50G
RDP_RM_SW{idx} cdp_rm_sw{idx} 0 50G
RDM_RM_SW{idx} cdm_rm_sw{idx} 0 50G
{sign_store_path(f"hpm_sw{idx}", "rp", f"cdp_rp_sw{idx}", f"swrp1_{idx}")}
{sign_store_path(f"hmp_sw{idx}", "rp", f"cdp_rp_sw{idx}", f"swrp2_{idx}")}
{sign_store_path(f"hpp_sw{idx}", "rp", f"cdm_rp_sw{idx}", f"swrp3_{idx}")}
{sign_store_path(f"hmm_sw{idx}", "rp", f"cdm_rp_sw{idx}", f"swrp4_{idx}")}
{sign_store_path(f"hpp_sw{idx}", "rm", f"cdp_rm_sw{idx}", f"swrm1_{idx}")}
{sign_store_path(f"hmm_sw{idx}", "rm", f"cdp_rm_sw{idx}", f"swrm2_{idx}")}
{sign_store_path(f"hpm_sw{idx}", "rm", f"cdm_rm_sw{idx}", f"swrm3_{idx}")}
{sign_store_path(f"hmp_sw{idx}", "rm", f"cdm_rm_sw{idx}", f"swrm4_{idx}")}

MPRP_CDP_SW{idx} rgp_rp_sw{idx} cdp_rp_sw{idx} vdd vdd PRSEL_SW L={{LCH}} W={{WRESTP}}
MNRP_CDP_SW{idx} rgp_rp_sw{idx} cdp_rp_sw{idx} 0 0 NRSEL_SW L={{LCH}} W={{WRESTN}}
MPRP_CDM_SW{idx} rgm_rp_sw{idx} cdm_rp_sw{idx} vdd vdd PRCOMP_SW L={{LCH}} W={{WRESTP}}
MNRP_CDM_SW{idx} rgm_rp_sw{idx} cdm_rp_sw{idx} 0 0 NRCOMP_SW L={{LCH}} W={{WRESTN}}

MPRM_CDP_SW{idx} rgp_rm_sw{idx} cdp_rm_sw{idx} vdd vdd PRCOMP_SW L={{LCH}} W={{WRESTP}}
MNRM_CDP_SW{idx} rgp_rm_sw{idx} cdp_rm_sw{idx} 0 0 NRCOMP_SW L={{LCH}} W={{WRESTN}}
MPRM_CDM_SW{idx} rgm_rm_sw{idx} cdm_rm_sw{idx} vdd vdd PRSEL_SW L={{LCH}} W={{WRESTP}}
MNRM_CDM_SW{idx} rgm_rm_sw{idx} cdm_rm_sw{idx} 0 0 NRSEL_SW L={{LCH}} W={{WRESTN}}

CWP_RP_SW{idx} wp_rp_sw{idx} 0 {{CWRITE}} IC=0.85
CWM_RP_SW{idx} wm_rp_sw{idx} 0 {{CWRITE}} IC=0.85
MWP_RP_SW{idx}A vdd paccn n_wp_rp_sw{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_SW{idx}B n_wp_rp_sw{idx}_a hm_pos n_wp_rp_sw{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_SW{idx}C n_wp_rp_sw{idx}_b rgp_rp_sw{idx} wp_rp_sw{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_SW{idx}A vdd paccn n_wm_rp_sw{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_SW{idx}B n_wm_rp_sw{idx}_a hm_pos n_wm_rp_sw{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_SW{idx}C n_wm_rp_sw{idx}_b rgm_rp_sw{idx} wm_rp_sw{idx} vdd PMOS L={{LCH}} W={{WWRITE}}

CWP_RM_SW{idx} wp_rm_sw{idx} 0 {{CWRITE}} IC=0.85
CWM_RM_SW{idx} wm_rm_sw{idx} 0 {{CWRITE}} IC=0.85
MWP_RM_SW{idx}A vdd paccn n_wp_rm_sw{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_SW{idx}B n_wp_rm_sw{idx}_a hm_pos n_wp_rm_sw{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_SW{idx}C n_wp_rm_sw{idx}_b rgp_rm_sw{idx} wp_rm_sw{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_SW{idx}A vdd paccn n_wm_rm_sw{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_SW{idx}B n_wm_rm_sw{idx}_a hm_pos n_wm_rm_sw{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_SW{idx}C n_wm_rm_sw{idx}_b rgm_rm_sw{idx} wm_rm_sw{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        swing_prints.extend(
            [
                f"v(cdp_rp_sw{idx})",
                f"v(cdm_rp_sw{idx})",
                f"v(cdp_rm_sw{idx})",
                f"v(cdm_rm_sw{idx})",
                f"v(rgp_rp_sw{idx})",
                f"v(rgm_rp_sw{idx})",
                f"v(rgp_rm_sw{idx})",
                f"v(rgm_rm_sw{idx})",
                f"v(wp_rp_sw{idx})",
                f"v(wm_rp_sw{idx})",
                f"v(wp_rm_sw{idx})",
                f"v(wm_rm_sw{idx})",
            ]
        )

    swing_deck = f"""
* Restored hidden-error gate swing-margin sweep under selected weak threshold.
* The restorer uses a moderate NMOS width from the sizing window.  This deck
* asks how much finite-difference hidden-error swing is needed to overcome the
* selected-side +70 mV NMOS threshold corner while keeping the complement gate
* high.
{COMMON_MODELS}
.model NRSEL_SW NMOS (LEVEL=1 VTO=0.62 KP=220u LAMBDA=0.03)
.model PRSEL_SW PMOS (LEVEL=1 VTO=-0.55 KP=90u LAMBDA=0.03)
.model NRCOMP_SW NMOS (LEVEL=1 VTO=0.55 KP=220u LAMBDA=0.03)
.model PRCOMP_SW PMOS (LEVEL=1 VTO=-0.55 KP=90u LAMBDA=0.03)
.param CERR=10p CWRITE=500p WSW=24u WWRITE=10u WRESTN=24u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)
VHM_POS hm_pos 0 0.92
{''.join(swing_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_restored_gate_swing.dat {' '.join(swing_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""
    swing_data = run_ngspice(swing_deck, "mos_hidden_writer_restored_gate_swing")
    swt, sw_cols = load_wrdata(swing_data, len(swing_prints) + 2)

    def swat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(swt - time_s))])

    swing_hidden = []
    swing_selected_gate = []
    swing_complement_gate = []
    swing_selected_step = []
    swing_complement_step = []
    swing_pos_net = []
    swing_neg_net = []
    swing_neg_hidden_magnitude = []
    for idx, eps_swing in enumerate(swing_eps_values):
        base = 12 * idx
        cdp_rp = sw_cols[base]
        cdm_rp = sw_cols[base + 1]
        cdp_rm = sw_cols[base + 2]
        cdm_rm = sw_cols[base + 3]
        rgp_rp = sw_cols[base + 4]
        rgm_rp = sw_cols[base + 5]
        rgp_rm = sw_cols[base + 6]
        rgm_rm = sw_cols[base + 7]
        wp_rp = sw_cols[base + 8]
        wm_rp = sw_cols[base + 9]
        wp_rm = sw_cols[base + 10]
        wm_rm = sw_cols[base + 11]
        hidden = swat(1.35e-6, cdp_rp - cdm_rp)
        hidden_neg = swat(1.35e-6, cdp_rm - cdm_rm)
        selected_gate = 0.5 * (swat(1.45e-6, rgp_rp) + swat(1.45e-6, rgm_rm))
        complement_gate = 0.5 * (swat(1.45e-6, rgm_rp) + swat(1.45e-6, rgp_rm))
        selected_step = 0.5 * (
            swat(2.75e-6, wp_rp) - swat(1.45e-6, wp_rp)
            + swat(2.75e-6, wm_rm) - swat(1.45e-6, wm_rm)
        )
        complement_step = 0.5 * (
            swat(2.75e-6, wm_rp) - swat(1.45e-6, wm_rp)
            + swat(2.75e-6, wp_rm) - swat(1.45e-6, wp_rm)
        )
        pos_net = swat(2.75e-6, wp_rp - wm_rp)
        neg_net = swat(2.75e-6, wp_rm - wm_rm)
        swing_hidden.append(hidden)
        swing_selected_gate.append(selected_gate)
        swing_complement_gate.append(complement_gate)
        swing_selected_step.append(selected_step)
        swing_complement_step.append(complement_step)
        swing_pos_net.append(pos_net)
        swing_neg_net.append(neg_net)
        swing_neg_hidden_magnitude.append(-hidden_neg)
        require(hidden > 0.0, f"eps={eps_swing:.2f} should store positive r+ hidden error")
        require(hidden_neg < 0.0, f"eps={eps_swing:.2f} should store negative r- hidden error")
        require(abs(hidden + hidden_neg) < 0.003, f"eps={eps_swing:.2f} hidden-error stores should stay symmetric")
        require(abs(pos_net + neg_net) < 0.003, f"eps={eps_swing:.2f} restored writes should stay symmetric")
        require(
            abs(swat(3.25e-6, wp_rp - wm_rp) - pos_net) < 5e-4,
            f"eps={eps_swing:.2f} restored r+ write should hold",
        )
        require(
            abs(swat(3.25e-6, wp_rm - wm_rm) - neg_net) < 5e-4,
            f"eps={eps_swing:.2f} restored r- write should hold",
        )

    swing_hidden = np.array(swing_hidden)
    swing_selected_gate = np.array(swing_selected_gate)
    swing_complement_gate = np.array(swing_complement_gate)
    swing_selected_step = np.array(swing_selected_step)
    swing_complement_step = np.array(swing_complement_step)
    swing_pos_net = np.array(swing_pos_net)
    swing_neg_net = np.array(swing_neg_net)
    swing_neg_hidden_magnitude = np.array(swing_neg_hidden_magnitude)
    swing_usable = (swing_pos_net > 0.020) & (swing_complement_step < 5e-4)
    require(np.all(np.diff(swing_hidden) > 0.015), "stored hidden-error swing should grow with finite-difference epsilon")
    require(np.all(np.diff(swing_selected_gate) < -0.01), "selected restored gate should fall as hidden-error swing grows")
    require(np.all(swing_complement_gate > 1.40), "complement restored gate should remain high across tested swings")
    require(np.all(swing_complement_step < 5e-4), "complement writer should remain suppressed across tested swings")
    require(not swing_usable[0], "smallest hidden-error swing should remain below the selected-weak trip point")
    require(swing_usable.any(), "some hidden-error swing should recover the selected-weak restored writer")
    require(swing_usable[-1], "largest tested hidden-error swing should recover the selected-weak restored writer")
    require(swing_pos_net[-1] > swing_pos_net[0] + 0.020, "larger hidden-error swing should improve the recovered write")
    require(np.all(swing_pos_net[swing_usable] < 0.12), "swing-recovered writes should remain incremental")

    swing_fig, swing_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    swing_eps_arr = np.array(swing_eps_values)
    if swing_usable.any():
        for axis in swing_axes:
            axis.axvspan(
                float(np.min(swing_eps_arr[swing_usable])) - 0.005,
                float(np.max(swing_eps_arr[swing_usable])) + 0.005,
                color="tab:green",
                alpha=0.08,
                label="usable swing window" if axis is swing_axes[0] else None,
            )
    swing_axes[0].plot(swing_eps_values, swing_hidden, "o-", label="stored $r^+$ hidden error")
    swing_axes[0].plot(swing_eps_values, swing_neg_hidden_magnitude, ":", color="tab:blue", label="stored $r^-$ magnitude")
    swing_axes[0].plot(swing_eps_values, swing_selected_gate, "s-", label="selected restored gate")
    swing_axes[0].plot(swing_eps_values, swing_complement_gate, "d--", label="complement restored gate")
    swing_axes[0].axhline(0.9, color="0.4", linewidth=0.8, alpha=0.5)
    swing_axes[0].set_ylabel("voltage (V)")
    swing_axes[0].set_title("Stored error swing moves the selected weak restorer through its trip point")
    swing_axes[0].grid(True, alpha=0.25)
    swing_axes[0].legend(loc="center right", fontsize="small")
    swing_axes[1].plot(swing_eps_values, swing_selected_step, "o-", label="selected rail step")
    swing_axes[1].plot(swing_eps_values, swing_complement_step, "s--", label="complement rail step")
    swing_axes[1].plot(swing_eps_values, swing_pos_net, "^-", label="$r^+$ net")
    swing_axes[1].plot(swing_eps_values, -swing_neg_net, "v:", label="$r^-$ net magnitude")
    swing_axes[1].axhline(0, color="0.4", linewidth=0.8)
    swing_axes[1].set_xlabel("finite-difference epsilon (V)")
    swing_axes[1].set_ylabel("writer step (V)")
    swing_axes[1].set_title("Recovered write appears only after sufficient stored-error swing")
    swing_axes[1].grid(True, alpha=0.25)
    swing_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    swing_fig.tight_layout()
    save_plot(swing_fig, "mos_hidden_writer_restored_gate_swing_ngspice")

    hybrid_eps_values = [0.00, 0.03, 0.06, 0.10, 0.14]
    hybrid_devices = []
    hybrid_prints = []
    for idx, eps_hybrid in enumerate(hybrid_eps_values):
        hybrid_devices.append(
            f"""
* Hybrid restored-enable/analog-error writer copy, finite-difference epsilon={eps_hybrid:.2f} V.
VZPP_HY{idx} zpp_hy{idx} 0 {0.9 + eps_hybrid / 2.0:.5f}
VZMM_HY{idx} zmm_hy{idx} 0 {0.9 - eps_hybrid / 2.0:.5f}
VZPM_HY{idx} zpm_hy{idx} 0 {0.9 - eps_hybrid / 2.0:.5f}
VZMP_HY{idx} zmp_hy{idx} 0 {0.9 + eps_hybrid / 2.0:.5f}

MPPP_HY{idx} hpp_hy{idx} hpp_hy{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HY{idx} hpm_hy{idx} hpm_hy{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HY{idx} hpp_hy{idx} zpp_hy{idx} tailp_hy{idx} 0 NMOS L={{LCH}} W={{WN}}
MNPM_HY{idx} hpm_hy{idx} zmm_hy{idx} tailp_hy{idx} 0 NMOS L={{LCH}} W={{WN}}
MNTP_HY{idx} tailp_hy{idx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HY{idx} hmp_hy{idx} hmp_hy{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HY{idx} hmm_hy{idx} hmm_hy{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HY{idx} hmp_hy{idx} zpm_hy{idx} tailm_hy{idx} 0 NMOS L={{LCH}} W={{WN}}
MNMM_HY{idx} hmm_hy{idx} zmp_hy{idx} tailm_hy{idx} 0 NMOS L={{LCH}} W={{WN}}
MNTM_HY{idx} tailm_hy{idx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HY{idx} cdp_rp_hy{idx} 0 {{CERR}} IC=1.04
CDM_RP_HY{idx} cdm_rp_hy{idx} 0 {{CERR}} IC=1.04
CDP_RM_HY{idx} cdp_rm_hy{idx} 0 {{CERR}} IC=1.04
CDM_RM_HY{idx} cdm_rm_hy{idx} 0 {{CERR}} IC=1.04
RDP_RP_HY{idx} cdp_rp_hy{idx} 0 50G
RDM_RP_HY{idx} cdm_rp_hy{idx} 0 50G
RDP_RM_HY{idx} cdp_rm_hy{idx} 0 50G
RDM_RM_HY{idx} cdm_rm_hy{idx} 0 50G
{sign_store_path(f"hpm_hy{idx}", "rp", f"cdp_rp_hy{idx}", f"hyrp1_{idx}")}
{sign_store_path(f"hmp_hy{idx}", "rp", f"cdp_rp_hy{idx}", f"hyrp2_{idx}")}
{sign_store_path(f"hpp_hy{idx}", "rp", f"cdm_rp_hy{idx}", f"hyrp3_{idx}")}
{sign_store_path(f"hmm_hy{idx}", "rp", f"cdm_rp_hy{idx}", f"hyrp4_{idx}")}
{sign_store_path(f"hpp_hy{idx}", "rm", f"cdp_rm_hy{idx}", f"hyrm1_{idx}")}
{sign_store_path(f"hmm_hy{idx}", "rm", f"cdp_rm_hy{idx}", f"hyrm2_{idx}")}
{sign_store_path(f"hpm_hy{idx}", "rm", f"cdm_rm_hy{idx}", f"hyrm3_{idx}")}
{sign_store_path(f"hmp_hy{idx}", "rm", f"cdm_rm_hy{idx}", f"hyrm4_{idx}")}

MPRP_CDP_HY{idx} rgp_rp_hy{idx} cdp_rp_hy{idx} vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HY{idx} rgp_rp_hy{idx} cdp_rp_hy{idx} 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HY{idx} rgm_rp_hy{idx} cdm_rp_hy{idx} vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HY{idx} rgm_rp_hy{idx} cdm_rp_hy{idx} 0 0 NMOS L={{LCH}} W={{WRESTN}}

MPRM_CDP_HY{idx} rgp_rm_hy{idx} cdp_rm_hy{idx} vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDP_HY{idx} rgp_rm_hy{idx} cdp_rm_hy{idx} 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDM_HY{idx} rgm_rm_hy{idx} cdm_rm_hy{idx} vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDM_HY{idx} rgm_rm_hy{idx} cdm_rm_hy{idx} 0 0 NMOS L={{LCH}} W={{WRESTN}}

CWP_RP_HY{idx} wp_rp_hy{idx} 0 {{CWRITE}} IC=0.85
CWM_RP_HY{idx} wm_rp_hy{idx} 0 {{CWRITE}} IC=0.85
MWP_RP_HY{idx}A vdd paccn n_wp_rp_hy{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_HY{idx}B n_wp_rp_hy{idx}_a hm_pos n_wp_rp_hy{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_HY{idx}C n_wp_rp_hy{idx}_b rgp_rp_hy{idx} n_wp_rp_hy{idx}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RP_HY{idx}D n_wp_rp_hy{idx}_c cdm_rp_hy{idx} wp_rp_hy{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_HY{idx}A vdd paccn n_wm_rp_hy{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_HY{idx}B n_wm_rp_hy{idx}_a hm_pos n_wm_rp_hy{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_HY{idx}C n_wm_rp_hy{idx}_b rgm_rp_hy{idx} n_wm_rp_hy{idx}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RP_HY{idx}D n_wm_rp_hy{idx}_c cdp_rp_hy{idx} wm_rp_hy{idx} vdd PMOS L={{LCH}} W={{WWRITE}}

CWP_RM_HY{idx} wp_rm_hy{idx} 0 {{CWRITE}} IC=0.85
CWM_RM_HY{idx} wm_rm_hy{idx} 0 {{CWRITE}} IC=0.85
MWP_RM_HY{idx}A vdd paccn n_wp_rm_hy{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_HY{idx}B n_wp_rm_hy{idx}_a hm_pos n_wp_rm_hy{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_HY{idx}C n_wp_rm_hy{idx}_b rgp_rm_hy{idx} n_wp_rm_hy{idx}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_RM_HY{idx}D n_wp_rm_hy{idx}_c cdm_rm_hy{idx} wp_rm_hy{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_HY{idx}A vdd paccn n_wm_rm_hy{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_HY{idx}B n_wm_rm_hy{idx}_a hm_pos n_wm_rm_hy{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_HY{idx}C n_wm_rm_hy{idx}_b rgm_rm_hy{idx} n_wm_rm_hy{idx}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_RM_HY{idx}D n_wm_rm_hy{idx}_c cdp_rm_hy{idx} wm_rm_hy{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        hybrid_prints.extend(
            [
                f"v(cdp_rp_hy{idx})",
                f"v(cdm_rp_hy{idx})",
                f"v(cdp_rm_hy{idx})",
                f"v(cdm_rm_hy{idx})",
                f"v(rgp_rp_hy{idx})",
                f"v(rgm_rp_hy{idx})",
                f"v(rgp_rm_hy{idx})",
                f"v(rgm_rm_hy{idx})",
                f"v(wp_rp_hy{idx})",
                f"v(wm_rp_hy{idx})",
                f"v(wp_rm_hy{idx})",
                f"v(wm_rm_hy{idx})",
            ]
        )

    hybrid_deck = f"""
* Hybrid restored-enable plus analog-error local writer.
* The restored hidden-error gate is used only as a branch select/inhibit
* device.  The original stored analog error capacitor remains in series as a
* second PMOS gate so the selected branch can retain update magnitude while
* the complementary branch is held off by a rail-high restored gate.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)
VHM_POS hm_pos 0 0.92
{''.join(hybrid_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_restored_gate_hybrid.dat {' '.join(hybrid_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""
    hybrid_data = run_ngspice(hybrid_deck, "mos_hidden_writer_restored_gate_hybrid")
    hyt, hy_cols = load_wrdata(hybrid_data, len(hybrid_prints) + 2)

    def hyat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyt - time_s))])

    hybrid_hidden = []
    hybrid_selected_gate = []
    hybrid_complement_gate = []
    hybrid_selected_step = []
    hybrid_complement_step = []
    hybrid_pos_net = []
    hybrid_neg_net = []
    for idx, eps_hybrid in enumerate(hybrid_eps_values):
        base = 12 * idx
        cdp_rp = hy_cols[base]
        cdm_rp = hy_cols[base + 1]
        cdp_rm = hy_cols[base + 2]
        cdm_rm = hy_cols[base + 3]
        rgp_rp = hy_cols[base + 4]
        rgm_rp = hy_cols[base + 5]
        rgp_rm = hy_cols[base + 6]
        rgm_rm = hy_cols[base + 7]
        wp_rp = hy_cols[base + 8]
        wm_rp = hy_cols[base + 9]
        wp_rm = hy_cols[base + 10]
        wm_rm = hy_cols[base + 11]
        hidden = hyat(1.35e-6, cdp_rp - cdm_rp)
        hidden_neg = hyat(1.35e-6, cdp_rm - cdm_rm)
        selected_gate = 0.5 * (hyat(1.45e-6, rgp_rp) + hyat(1.45e-6, rgm_rm))
        complement_gate = 0.5 * (hyat(1.45e-6, rgm_rp) + hyat(1.45e-6, rgp_rm))
        selected_step = 0.5 * (
            hyat(2.75e-6, wp_rp) - hyat(1.45e-6, wp_rp)
            + hyat(2.75e-6, wm_rm) - hyat(1.45e-6, wm_rm)
        )
        complement_step = 0.5 * (
            hyat(2.75e-6, wm_rp) - hyat(1.45e-6, wm_rp)
            + hyat(2.75e-6, wp_rm) - hyat(1.45e-6, wp_rm)
        )
        pos_net = hyat(2.75e-6, wp_rp - wm_rp)
        neg_net = hyat(2.75e-6, wp_rm - wm_rm)
        hybrid_hidden.append(hidden)
        hybrid_selected_gate.append(selected_gate)
        hybrid_complement_gate.append(complement_gate)
        hybrid_selected_step.append(selected_step)
        hybrid_complement_step.append(complement_step)
        hybrid_pos_net.append(pos_net)
        hybrid_neg_net.append(neg_net)
        require(abs(hidden + hidden_neg) < 0.003, f"hybrid eps={eps_hybrid:.2f} hidden-error stores should stay symmetric")
        require(abs(pos_net + neg_net) < 0.003, f"hybrid eps={eps_hybrid:.2f} writes should stay symmetric")
        require(
            abs(hyat(3.25e-6, wp_rp - wm_rp) - pos_net) < 5e-4,
            f"hybrid eps={eps_hybrid:.2f} r+ write should hold",
        )
        require(
            abs(hyat(3.25e-6, wp_rm - wm_rm) - neg_net) < 5e-4,
            f"hybrid eps={eps_hybrid:.2f} r- write should hold",
        )

    hybrid_hidden = np.array(hybrid_hidden)
    hybrid_selected_gate = np.array(hybrid_selected_gate)
    hybrid_complement_gate = np.array(hybrid_complement_gate)
    hybrid_selected_step = np.array(hybrid_selected_step)
    hybrid_complement_step = np.array(hybrid_complement_step)
    hybrid_pos_net = np.array(hybrid_pos_net)
    hybrid_neg_net = np.array(hybrid_neg_net)
    require(abs(hybrid_hidden[0]) < 0.005, "hybrid zero-nudge hidden error should stay near zero")
    require(abs(hybrid_pos_net[0]) < 0.001, "hybrid zero hidden error should not create a signed write")
    require(np.all(np.diff(hybrid_hidden) > 0.015), "hybrid stored hidden-error magnitude should grow with epsilon")
    require(np.all(hybrid_complement_gate[1:] > 1.45), "hybrid complement restored gates should stay high")
    require(np.all(hybrid_complement_step < 5e-4), "hybrid complement rails should stay suppressed")
    hybrid_active = hybrid_pos_net > 0.002
    require(hybrid_active[1:].any(), "hybrid writer should become active for some nonzero hidden-error swings")
    require(np.all(np.diff(hybrid_pos_net[hybrid_active]) >= -5e-4), "hybrid active signed write should not lose magnitude ordering")
    require(hybrid_pos_net[-1] > 0.004, "hybrid largest signed write should be useful")
    require(hybrid_pos_net[-1] < 0.080, "hybrid largest signed write should remain incremental")

    hybrid_fig, hybrid_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hybrid_axes[0].plot(hybrid_eps_values, hybrid_hidden, "o-", label="stored $r^+$ hidden error")
    hybrid_axes[0].plot(hybrid_eps_values, hybrid_selected_gate, "s-", label="selected restored gate")
    hybrid_axes[0].plot(hybrid_eps_values, hybrid_complement_gate, "d--", label="complement restored gate")
    hybrid_axes[0].axhline(0.9, color="0.4", linewidth=0.8, alpha=0.5)
    hybrid_axes[0].set_ylabel("voltage (V)")
    hybrid_axes[0].set_title("Hybrid writer uses restoration for selectivity and analog rail for magnitude")
    hybrid_axes[0].grid(True, alpha=0.25)
    hybrid_axes[0].legend(loc="center right", fontsize="small")
    hybrid_axes[1].plot(hybrid_eps_values, hybrid_selected_step, "o-", label="selected rail step")
    hybrid_axes[1].plot(hybrid_eps_values, hybrid_complement_step, "s--", label="complement rail step")
    hybrid_axes[1].plot(hybrid_eps_values, hybrid_pos_net, "^-", label="$r^+$ net")
    hybrid_axes[1].plot(hybrid_eps_values, -hybrid_neg_net, "v:", label="$r^-$ net magnitude")
    hybrid_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hybrid_axes[1].set_xlabel("finite-difference epsilon (V)")
    hybrid_axes[1].set_ylabel("writer step (V)")
    hybrid_axes[1].set_title("Series analog error gate restores graded writes while complement remains off")
    hybrid_axes[1].grid(True, alpha=0.25)
    hybrid_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    hybrid_fig.tight_layout()
    save_plot(hybrid_fig, "mos_hidden_writer_restored_gate_hybrid_ngspice")

    hybrid_mismatch_eps = 0.10
    hybrid_mismatch_cases = [
        ("nominal", 0.55, -0.55, 0.55, -0.55, -0.55),
        ("selected weak", 0.60, -0.55, 0.55, -0.55, -0.55),
        ("comp strong", 0.55, -0.55, 0.50, -0.55, -0.55),
        ("writer weak", 0.55, -0.55, 0.55, -0.55, -0.60),
        ("writer strong", 0.55, -0.55, 0.55, -0.55, -0.50),
        ("combined", 0.60, -0.55, 0.50, -0.55, -0.60),
    ]
    hybrid_mismatch_models = []
    hybrid_mismatch_devices = []
    hybrid_mismatch_prints = []
    for idx, (_label, nsel, psel, ncomp, pcomp, pwrite) in enumerate(hybrid_mismatch_cases):
        hybrid_mismatch_models.append(
            f"""
.model NRSEL_HYM{idx} NMOS (LEVEL=1 VTO={nsel:.2f} KP=220u LAMBDA=0.03)
.model PRSEL_HYM{idx} PMOS (LEVEL=1 VTO={psel:.2f} KP=90u LAMBDA=0.03)
.model NRCOMP_HYM{idx} NMOS (LEVEL=1 VTO={ncomp:.2f} KP=220u LAMBDA=0.03)
.model PRCOMP_HYM{idx} PMOS (LEVEL=1 VTO={pcomp:.2f} KP=90u LAMBDA=0.03)
.model PWRITE_HYM{idx} PMOS (LEVEL=1 VTO={pwrite:.2f} KP=90u LAMBDA=0.03)
"""
        )
        hybrid_mismatch_devices.append(
            f"""
* Hybrid mismatch copy: {_label}.
VZPP_HYM{idx} zpp_hym{idx} 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYM{idx} zmm_hym{idx} 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYM{idx} zpm_hym{idx} 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYM{idx} zmp_hym{idx} 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYM{idx} hpp_hym{idx} hpp_hym{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYM{idx} hpm_hym{idx} hpm_hym{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYM{idx} hpp_hym{idx} zpp_hym{idx} tailp_hym{idx} 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYM{idx} hpm_hym{idx} zmm_hym{idx} tailp_hym{idx} 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYM{idx} tailp_hym{idx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYM{idx} hmp_hym{idx} hmp_hym{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYM{idx} hmm_hym{idx} hmm_hym{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYM{idx} hmp_hym{idx} zpm_hym{idx} tailm_hym{idx} 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYM{idx} hmm_hym{idx} zmp_hym{idx} tailm_hym{idx} 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYM{idx} tailm_hym{idx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYM{idx} cdp_rp_hym{idx} 0 {{CERR}} IC=1.04
CDM_RP_HYM{idx} cdm_rp_hym{idx} 0 {{CERR}} IC=1.04
CDP_RM_HYM{idx} cdp_rm_hym{idx} 0 {{CERR}} IC=1.04
CDM_RM_HYM{idx} cdm_rm_hym{idx} 0 {{CERR}} IC=1.04
RDP_RP_HYM{idx} cdp_rp_hym{idx} 0 50G
RDM_RP_HYM{idx} cdm_rp_hym{idx} 0 50G
RDP_RM_HYM{idx} cdp_rm_hym{idx} 0 50G
RDM_RM_HYM{idx} cdm_rm_hym{idx} 0 50G
{sign_store_path(f"hpm_hym{idx}", "rp", f"cdp_rp_hym{idx}", f"hymrp1_{idx}")}
{sign_store_path(f"hmp_hym{idx}", "rp", f"cdp_rp_hym{idx}", f"hymrp2_{idx}")}
{sign_store_path(f"hpp_hym{idx}", "rp", f"cdm_rp_hym{idx}", f"hymrp3_{idx}")}
{sign_store_path(f"hmm_hym{idx}", "rp", f"cdm_rp_hym{idx}", f"hymrp4_{idx}")}
{sign_store_path(f"hpp_hym{idx}", "rm", f"cdp_rm_hym{idx}", f"hymrm1_{idx}")}
{sign_store_path(f"hmm_hym{idx}", "rm", f"cdp_rm_hym{idx}", f"hymrm2_{idx}")}
{sign_store_path(f"hpm_hym{idx}", "rm", f"cdm_rm_hym{idx}", f"hymrm3_{idx}")}
{sign_store_path(f"hmp_hym{idx}", "rm", f"cdm_rm_hym{idx}", f"hymrm4_{idx}")}

MPRP_CDP_HYM{idx} rgp_rp_hym{idx} cdp_rp_hym{idx} vdd vdd PRSEL_HYM{idx} L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYM{idx} rgp_rp_hym{idx} cdp_rp_hym{idx} 0 0 NRSEL_HYM{idx} L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYM{idx} rgm_rp_hym{idx} cdm_rp_hym{idx} vdd vdd PRCOMP_HYM{idx} L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYM{idx} rgm_rp_hym{idx} cdm_rp_hym{idx} 0 0 NRCOMP_HYM{idx} L={{LCH}} W={{WRESTN}}

MPRM_CDP_HYM{idx} rgp_rm_hym{idx} cdp_rm_hym{idx} vdd vdd PRCOMP_HYM{idx} L={{LCH}} W={{WRESTP}}
MNRM_CDP_HYM{idx} rgp_rm_hym{idx} cdp_rm_hym{idx} 0 0 NRCOMP_HYM{idx} L={{LCH}} W={{WRESTN}}
MPRM_CDM_HYM{idx} rgm_rm_hym{idx} cdm_rm_hym{idx} vdd vdd PRSEL_HYM{idx} L={{LCH}} W={{WRESTP}}
MNRM_CDM_HYM{idx} rgm_rm_hym{idx} cdm_rm_hym{idx} 0 0 NRSEL_HYM{idx} L={{LCH}} W={{WRESTN}}

CWP_RP_HYM{idx} wp_rp_hym{idx} 0 {{CWRITE}} IC=0.85
CWM_RP_HYM{idx} wm_rp_hym{idx} 0 {{CWRITE}} IC=0.85
MWP_RP_HYM{idx}A vdd paccn n_wp_rp_hym{idx}_a vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWP_RP_HYM{idx}B n_wp_rp_hym{idx}_a hm_pos n_wp_rp_hym{idx}_b vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWP_RP_HYM{idx}C n_wp_rp_hym{idx}_b rgp_rp_hym{idx} n_wp_rp_hym{idx}_c vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWP_RP_HYM{idx}D n_wp_rp_hym{idx}_c cdm_rp_hym{idx} wp_rp_hym{idx} vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWM_RP_HYM{idx}A vdd paccn n_wm_rp_hym{idx}_a vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWM_RP_HYM{idx}B n_wm_rp_hym{idx}_a hm_pos n_wm_rp_hym{idx}_b vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWM_RP_HYM{idx}C n_wm_rp_hym{idx}_b rgm_rp_hym{idx} n_wm_rp_hym{idx}_c vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWM_RP_HYM{idx}D n_wm_rp_hym{idx}_c cdp_rp_hym{idx} wm_rp_hym{idx} vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}

CWP_RM_HYM{idx} wp_rm_hym{idx} 0 {{CWRITE}} IC=0.85
CWM_RM_HYM{idx} wm_rm_hym{idx} 0 {{CWRITE}} IC=0.85
MWP_RM_HYM{idx}A vdd paccn n_wp_rm_hym{idx}_a vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWP_RM_HYM{idx}B n_wp_rm_hym{idx}_a hm_pos n_wp_rm_hym{idx}_b vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWP_RM_HYM{idx}C n_wp_rm_hym{idx}_b rgp_rm_hym{idx} n_wp_rm_hym{idx}_c vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWP_RM_HYM{idx}D n_wp_rm_hym{idx}_c cdm_rm_hym{idx} wp_rm_hym{idx} vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWM_RM_HYM{idx}A vdd paccn n_wm_rm_hym{idx}_a vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWM_RM_HYM{idx}B n_wm_rm_hym{idx}_a hm_pos n_wm_rm_hym{idx}_b vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWM_RM_HYM{idx}C n_wm_rm_hym{idx}_b rgm_rm_hym{idx} n_wm_rm_hym{idx}_c vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
MWM_RM_HYM{idx}D n_wm_rm_hym{idx}_c cdp_rm_hym{idx} wm_rm_hym{idx} vdd PWRITE_HYM{idx} L={{LCH}} W={{WWRITE}}
"""
        )
        hybrid_mismatch_prints.extend(
            [
                f"v(cdp_rp_hym{idx})",
                f"v(cdm_rp_hym{idx})",
                f"v(cdp_rm_hym{idx})",
                f"v(cdm_rm_hym{idx})",
                f"v(rgp_rp_hym{idx})",
                f"v(rgm_rp_hym{idx})",
                f"v(rgp_rm_hym{idx})",
                f"v(rgm_rm_hym{idx})",
                f"v(wp_rp_hym{idx})",
                f"v(wm_rp_hym{idx})",
                f"v(wp_rm_hym{idx})",
                f"v(wm_rm_hym{idx})",
            ]
        )

    hybrid_mismatch_deck = f"""
* Hybrid restored-enable/analog-error threshold-corner characterization.
* The hidden-error swing is fixed at a useful value.  The deck perturbs
* restorer trip points and writer PMOS thresholds to test whether the hybrid
* writer keeps sign, symmetry, and complement suppression.
{COMMON_MODELS}
{''.join(hybrid_mismatch_models)}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_N paccn 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)
VHM_POS hm_pos 0 0.92
{''.join(hybrid_mismatch_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_restored_gate_hybrid_mismatch.dat {' '.join(hybrid_mismatch_prints)} v(pbwd) v(paccn)
quit
.endc
.end
"""
    hybrid_mismatch_data = run_ngspice(
        hybrid_mismatch_deck,
        "mos_hidden_writer_restored_gate_hybrid_mismatch",
    )
    hymt, hym_cols = load_wrdata(hybrid_mismatch_data, len(hybrid_mismatch_prints) + 2)

    def hymat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hymt - time_s))])

    hym_hidden = []
    hym_selected_gate = []
    hym_complement_gate = []
    hym_selected_step = []
    hym_complement_step = []
    hym_pos_net = []
    hym_neg_net = []
    for idx, (label, *_rest) in enumerate(hybrid_mismatch_cases):
        base = 12 * idx
        cdp_rp = hym_cols[base]
        cdm_rp = hym_cols[base + 1]
        cdp_rm = hym_cols[base + 2]
        cdm_rm = hym_cols[base + 3]
        rgp_rp = hym_cols[base + 4]
        rgm_rp = hym_cols[base + 5]
        rgp_rm = hym_cols[base + 6]
        rgm_rm = hym_cols[base + 7]
        wp_rp = hym_cols[base + 8]
        wm_rp = hym_cols[base + 9]
        wp_rm = hym_cols[base + 10]
        wm_rm = hym_cols[base + 11]
        hidden = hymat(1.35e-6, cdp_rp - cdm_rp)
        hidden_neg = hymat(1.35e-6, cdp_rm - cdm_rm)
        selected_gate = 0.5 * (hymat(1.45e-6, rgp_rp) + hymat(1.45e-6, rgm_rm))
        complement_gate = 0.5 * (hymat(1.45e-6, rgm_rp) + hymat(1.45e-6, rgp_rm))
        selected_step = 0.5 * (
            hymat(2.75e-6, wp_rp) - hymat(1.45e-6, wp_rp)
            + hymat(2.75e-6, wm_rm) - hymat(1.45e-6, wm_rm)
        )
        complement_step = 0.5 * (
            hymat(2.75e-6, wm_rp) - hymat(1.45e-6, wm_rp)
            + hymat(2.75e-6, wp_rm) - hymat(1.45e-6, wp_rm)
        )
        pos_net = hymat(2.75e-6, wp_rp - wm_rp)
        neg_net = hymat(2.75e-6, wp_rm - wm_rm)
        hym_hidden.append(hidden)
        hym_selected_gate.append(selected_gate)
        hym_complement_gate.append(complement_gate)
        hym_selected_step.append(selected_step)
        hym_complement_step.append(complement_step)
        hym_pos_net.append(pos_net)
        hym_neg_net.append(neg_net)
        require(hidden > 0.07, f"{label} hybrid mismatch hidden-error store should be positive")
        require(hidden_neg < -0.07, f"{label} hybrid mismatch hidden-error store should be negative")
        require(abs(hidden + hidden_neg) < 0.003, f"{label} hybrid mismatch hidden-error stores should stay symmetric")
        require(abs(pos_net + neg_net) < 0.003, f"{label} hybrid mismatch writes should stay symmetric")
        require(
            abs(hymat(3.25e-6, wp_rp - wm_rp) - pos_net) < 5e-4,
            f"{label} hybrid mismatch r+ write should hold",
        )
        require(
            abs(hymat(3.25e-6, wp_rm - wm_rm) - neg_net) < 5e-4,
            f"{label} hybrid mismatch r- write should hold",
        )

    hym_hidden = np.array(hym_hidden)
    hym_selected_gate = np.array(hym_selected_gate)
    hym_complement_gate = np.array(hym_complement_gate)
    hym_selected_step = np.array(hym_selected_step)
    hym_complement_step = np.array(hym_complement_step)
    hym_pos_net = np.array(hym_pos_net)
    hym_neg_net = np.array(hym_neg_net)
    require(np.all(hym_complement_gate > 1.45), "hybrid mismatch complement restored gates should stay high")
    require(np.all(hym_complement_step < 5e-4), "hybrid mismatch complement rails should stay suppressed")
    require(np.all(hym_pos_net > 0.020), "hybrid mismatch r+ net writes should remain useful")
    require(np.all(hym_neg_net < -0.020), "hybrid mismatch r- net writes should remain useful")
    require(np.all(hym_pos_net < 0.090), "hybrid mismatch writes should stay incremental")
    require(hym_pos_net[3] < hym_pos_net[0], "weak writer PMOS should reduce update gain")
    require(hym_pos_net[4] > hym_pos_net[0], "strong writer PMOS should increase update gain")

    hym_fig, hym_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hym_x = np.arange(len(hybrid_mismatch_cases))
    hym_labels = [label for label, *_rest in hybrid_mismatch_cases]
    hym_axes[0].plot(hym_x, hym_selected_gate, "o-", label="selected restored gate")
    hym_axes[0].plot(hym_x, hym_complement_gate, "s--", label="complement restored gate")
    hym_axes[0].axhline(0.9, color="0.4", linewidth=0.8, alpha=0.5)
    hym_axes[0].set_xticks(hym_x)
    hym_axes[0].set_xticklabels(hym_labels, rotation=15, ha="right")
    hym_axes[0].set_ylabel("gate voltage (V)")
    hym_axes[0].set_title("Hybrid restored gates keep selectivity across threshold corners")
    hym_axes[0].grid(True, alpha=0.25)
    hym_axes[0].legend(loc="center right")
    hym_axes[1].plot(hym_x, hym_selected_step, "o-", label="selected rail step")
    hym_axes[1].plot(hym_x, hym_complement_step, "s--", label="complement rail step")
    hym_axes[1].plot(hym_x, hym_pos_net, "^-", label="$r^+$ net")
    hym_axes[1].plot(hym_x, -hym_neg_net, "v:", label="$r^-$ net magnitude")
    hym_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hym_axes[1].set_xticks(hym_x)
    hym_axes[1].set_xticklabels(hym_labels, rotation=15, ha="right")
    hym_axes[1].set_ylabel("writer step (V)")
    hym_axes[1].set_title("Hybrid mismatch changes gain but preserves sign and complement suppression")
    hym_axes[1].grid(True, alpha=0.25)
    hym_axes[1].legend(loc="upper right", ncol=2)
    hym_fig.tight_layout()
    save_plot(hym_fig, "mos_hidden_writer_restored_gate_hybrid_mismatch_ngspice")

    hybrid_timing_cases = [
        ("pre", "pre-quiet", 0.10, 0.40),
        ("early", "early overlap", 0.10, 0.50),
        ("edge", "store edge", 0.46, 0.86),
        ("overlap", "overlap", 0.80, 1.20),
        ("late", "late overlap", 1.10, 1.50),
        ("gap", "settled gap", 1.55, 1.95),
    ]
    hybrid_timing_devices = []
    hybrid_timing_prints = ["v(cdp_rp_hyt)", "v(cdm_rp_hyt)", "v(rgp_rp_hyt)", "v(rgm_rp_hyt)"]
    for name, _label, start_us, end_us in hybrid_timing_cases:
        hybrid_timing_devices.append(
            f"""
VPACC_HYT_{name} paccn_hyt_{name} 0 PWL(0 1.8 {start_us:.2f}u 1.8 {start_us + 0.02:.2f}u 0 {end_us:.2f}u 0 {end_us + 0.02:.2f}u 1.8 3.2u 1.8)
CWP_HYT_{name} wp_hyt_{name} 0 {{CWRITE}} IC=0.85
CWM_HYT_{name} wm_hyt_{name} 0 {{CWRITE}} IC=0.85
MWP_HYT_{name}A vdd paccn_hyt_{name} n_wp_hyt_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYT_{name}B n_wp_hyt_{name}_a hm_pos n_wp_hyt_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYT_{name}C n_wp_hyt_{name}_b rgp_rp_hyt n_wp_hyt_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYT_{name}D n_wp_hyt_{name}_c cdm_rp_hyt wp_hyt_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYT_{name}A vdd paccn_hyt_{name} n_wm_hyt_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYT_{name}B n_wm_hyt_{name}_a hm_pos n_wm_hyt_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYT_{name}C n_wm_hyt_{name}_b rgm_rp_hyt n_wm_hyt_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYT_{name}D n_wm_hyt_{name}_c cdp_rp_hyt wm_hyt_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        hybrid_timing_prints.extend([f"v(wp_hyt_{name})", f"v(wm_hyt_{name})", f"v(paccn_hyt_{name})"])

    hybrid_timing_deck = f"""
* Hybrid restored-enable/analog-error writer phase-overlap characterization.
* A single r+ hidden-error store drives restored select gates and analog
* magnitude gates.  Independent writer copies sweep pacc timing from before
* storage through a separated settled-write phase.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)

VZPP_HYT zpp_hyt 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYT zmm_hyt 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYT zpm_hyt 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYT zmp_hyt 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYT hpp_hyt hpp_hyt vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYT hpm_hyt hpm_hyt vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYT hpp_hyt zpp_hyt tailp_hyt 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYT hpm_hyt zmm_hyt tailp_hyt 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYT tailp_hyt vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYT hmp_hyt hmp_hyt vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYT hmm_hyt hmm_hyt vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYT hmp_hyt zpm_hyt tailm_hyt 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYT hmm_hyt zmp_hyt tailm_hyt 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYT tailm_hyt vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYT cdp_rp_hyt 0 {{CERR}} IC=1.04
CDM_RP_HYT cdm_rp_hyt 0 {{CERR}} IC=1.04
RDP_RP_HYT cdp_rp_hyt 0 50G
RDM_RP_HYT cdm_rp_hyt 0 50G
{sign_store_path("hpm_hyt", "rp", "cdp_rp_hyt", "hytrp1")}
{sign_store_path("hmp_hyt", "rp", "cdp_rp_hyt", "hytrp2")}
{sign_store_path("hpp_hyt", "rp", "cdm_rp_hyt", "hytrp3")}
{sign_store_path("hmm_hyt", "rp", "cdm_rp_hyt", "hytrp4")}

MPRP_CDP_HYT rgp_rp_hyt cdp_rp_hyt vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYT rgp_rp_hyt cdp_rp_hyt 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYT rgm_rp_hyt cdm_rp_hyt vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYT rgm_rp_hyt cdm_rp_hyt 0 0 NMOS L={{LCH}} W={{WRESTN}}

VHM_POS hm_pos 0 0.92
{''.join(hybrid_timing_devices)}

.control
set noaskquit
tran 5n 3.2u uic
wrdata mos_hidden_writer_restored_gate_hybrid_timing.dat {' '.join(hybrid_timing_prints)} v(pbwd)
quit
.endc
.end
"""
    hybrid_timing_data = run_ngspice(
        hybrid_timing_deck,
        "mos_hidden_writer_restored_gate_hybrid_timing",
    )
    hytt, hyt_cols = load_wrdata(hybrid_timing_data, len(hybrid_timing_prints) + 1)

    def hytat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hytt - time_s))])

    hyt_hidden = hyt_cols[0] - hyt_cols[1]
    hyt_selected_gate = hyt_cols[2]
    hyt_complement_gate = hyt_cols[3]
    hyt_diffs = []
    hyt_final = []
    hyt_complement_steps = []
    for idx, (_name, _label, _start_us, _end_us) in enumerate(hybrid_timing_cases):
        wp = hyt_cols[4 + 3 * idx]
        wm = hyt_cols[5 + 3 * idx]
        diff = wp - wm
        hyt_diffs.append(diff)
        hyt_final.append(hytat(2.85e-6, diff))
        hyt_complement_steps.append(hytat(2.85e-6, wm) - hytat(0.05e-6, wm))
    hyt_final = np.array(hyt_final)
    hyt_complement_steps = np.array(hyt_complement_steps)
    require(hytat(1.35e-6, hyt_hidden) > 0.07, "hybrid timing deck should store a positive hidden error")
    require(hytat(1.45e-6, hyt_selected_gate) < 0.30, "hybrid timing selected restored gate should be low after store")
    require(hytat(1.45e-6, hyt_complement_gate) > 1.60, "hybrid timing complement restored gate should be high after store")
    require(abs(hyt_final[0]) < 0.001, "hybrid writer pulse before hidden-error storage should not create signed update")
    require(hyt_final[1] > hyt_final[0] + 0.001, "hybrid writer pulse overlapping pbwd edge should expose a small stray write")
    require(hyt_final[1] < 0.20 * hyt_final[-1], "hybrid early-overlap stray write should stay well below a full update")
    require(np.min(hyt_final[2:]) > 0.95 * hyt_final[-1], "hybrid writer pulses after storage starts should reach full update")
    require(np.max(hyt_final[2:]) - np.min(hyt_final[2:]) < 0.003, "hybrid full writer timings should agree")
    require(np.max(hyt_complement_steps[2:]) < 5e-4, "hybrid full timings should keep complement rail suppressed")

    hyt_fig, hyt_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hyt_axes[0].plot(1e6 * hytt, hyt_hidden, label="stored $r^+$ hidden error")
    hyt_axes[0].plot(1e6 * hytt, hyt_selected_gate, label="selected restored gate")
    hyt_axes[0].plot(1e6 * hytt, hyt_complement_gate, label="complement restored gate")
    hyt_axes[0].plot(1e6 * hytt, hyt_cols[-1] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    hyt_axes[0].axhline(0.9, color="0.4", linewidth=0.8, alpha=0.5)
    hyt_axes[0].set_ylabel("voltage (V)")
    hyt_axes[0].set_title("Hybrid writer select gates settle during the backward-store phase")
    hyt_axes[0].grid(True, alpha=0.25)
    hyt_axes[0].legend(loc="center right", fontsize="small")
    for (_name, label, _start_us, _end_us), diff in zip(hybrid_timing_cases, hyt_diffs):
        hyt_axes[1].plot(1e6 * hytt, diff, label=label)
    hyt_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyt_axes[1].set_xlabel("time (us)")
    hyt_axes[1].set_ylabel("$W^+ - W^-$ (V)")
    hyt_axes[1].set_title("Hybrid pacc timing needs storage-edge separation for full write")
    hyt_axes[1].grid(True, alpha=0.25)
    hyt_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    hyt_fig.tight_layout()
    save_plot(hyt_fig, "mos_hidden_writer_restored_gate_hybrid_timing_ngspice")

    hybrid_timing_corner_cases = [
        ("nom", "nominal", 0.55, -0.55, 0.55, -0.55, -0.55),
        ("weak", "combined weak", 0.60, -0.55, 0.50, -0.55, -0.60),
        ("strong", "writer strong", 0.55, -0.55, 0.55, -0.55, -0.50),
    ]
    hybrid_timing_corner_models = []
    hybrid_timing_corner_devices = []
    hybrid_timing_corner_prints = []
    for cidx, (cname, clabel, nsel, psel, ncomp, pcomp, pwrite) in enumerate(hybrid_timing_corner_cases):
        hybrid_timing_corner_models.append(
            f"""
.model NRSEL_HYTC{cidx} NMOS (LEVEL=1 VTO={nsel:.2f} KP=220u LAMBDA=0.03)
.model PRSEL_HYTC{cidx} PMOS (LEVEL=1 VTO={psel:.2f} KP=90u LAMBDA=0.03)
.model NRCOMP_HYTC{cidx} NMOS (LEVEL=1 VTO={ncomp:.2f} KP=220u LAMBDA=0.03)
.model PRCOMP_HYTC{cidx} PMOS (LEVEL=1 VTO={pcomp:.2f} KP=90u LAMBDA=0.03)
.model PWRITE_HYTC{cidx} PMOS (LEVEL=1 VTO={pwrite:.2f} KP=90u LAMBDA=0.03)
"""
        )
        hybrid_timing_corner_devices.append(
            f"""
* Hybrid timing corner copy: {clabel}.
VZPP_HYTC{cidx} zpp_hytc{cidx} 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYTC{cidx} zmm_hytc{cidx} 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYTC{cidx} zpm_hytc{cidx} 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYTC{cidx} zmp_hytc{cidx} 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYTC{cidx} hpp_hytc{cidx} hpp_hytc{cidx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYTC{cidx} hpm_hytc{cidx} hpm_hytc{cidx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYTC{cidx} hpp_hytc{cidx} zpp_hytc{cidx} tailp_hytc{cidx} 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYTC{cidx} hpm_hytc{cidx} zmm_hytc{cidx} tailp_hytc{cidx} 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYTC{cidx} tailp_hytc{cidx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYTC{cidx} hmp_hytc{cidx} hmp_hytc{cidx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYTC{cidx} hmm_hytc{cidx} hmm_hytc{cidx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYTC{cidx} hmp_hytc{cidx} zpm_hytc{cidx} tailm_hytc{cidx} 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYTC{cidx} hmm_hytc{cidx} zmp_hytc{cidx} tailm_hytc{cidx} 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYTC{cidx} tailm_hytc{cidx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYTC{cidx} cdp_rp_hytc{cidx} 0 {{CERR}} IC=1.04
CDM_RP_HYTC{cidx} cdm_rp_hytc{cidx} 0 {{CERR}} IC=1.04
RDP_RP_HYTC{cidx} cdp_rp_hytc{cidx} 0 50G
RDM_RP_HYTC{cidx} cdm_rp_hytc{cidx} 0 50G
{sign_store_path(f"hpm_hytc{cidx}", "rp", f"cdp_rp_hytc{cidx}", f"hytc{cidx}rp1")}
{sign_store_path(f"hmp_hytc{cidx}", "rp", f"cdp_rp_hytc{cidx}", f"hytc{cidx}rp2")}
{sign_store_path(f"hpp_hytc{cidx}", "rp", f"cdm_rp_hytc{cidx}", f"hytc{cidx}rp3")}
{sign_store_path(f"hmm_hytc{cidx}", "rp", f"cdm_rp_hytc{cidx}", f"hytc{cidx}rp4")}

MPRP_CDP_HYTC{cidx} rgp_rp_hytc{cidx} cdp_rp_hytc{cidx} vdd vdd PRSEL_HYTC{cidx} L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYTC{cidx} rgp_rp_hytc{cidx} cdp_rp_hytc{cidx} 0 0 NRSEL_HYTC{cidx} L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYTC{cidx} rgm_rp_hytc{cidx} cdm_rp_hytc{cidx} vdd vdd PRCOMP_HYTC{cidx} L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYTC{cidx} rgm_rp_hytc{cidx} cdm_rp_hytc{cidx} 0 0 NRCOMP_HYTC{cidx} L={{LCH}} W={{WRESTN}}
"""
        )
        hybrid_timing_corner_prints.extend(
            [
                f"v(cdp_rp_hytc{cidx})",
                f"v(cdm_rp_hytc{cidx})",
                f"v(rgp_rp_hytc{cidx})",
                f"v(rgm_rp_hytc{cidx})",
            ]
        )
        for tname, _tlabel, start_us, end_us in hybrid_timing_cases:
            hybrid_timing_corner_devices.append(
                f"""
VPACC_HYTC{cidx}_{tname} paccn_hytc{cidx}_{tname} 0 PWL(0 1.8 {start_us:.2f}u 1.8 {start_us + 0.02:.2f}u 0 {end_us:.2f}u 0 {end_us + 0.02:.2f}u 1.8 3.2u 1.8)
CWP_HYTC{cidx}_{tname} wp_hytc{cidx}_{tname} 0 {{CWRITE}} IC=0.85
CWM_HYTC{cidx}_{tname} wm_hytc{cidx}_{tname} 0 {{CWRITE}} IC=0.85
MWP_HYTC{cidx}_{tname}A vdd paccn_hytc{cidx}_{tname} n_wp_hytc{cidx}_{tname}_a vdd PWRITE_HYTC{cidx} L={{LCH}} W={{WWRITE}}
MWP_HYTC{cidx}_{tname}B n_wp_hytc{cidx}_{tname}_a hm_pos n_wp_hytc{cidx}_{tname}_b vdd PWRITE_HYTC{cidx} L={{LCH}} W={{WWRITE}}
MWP_HYTC{cidx}_{tname}C n_wp_hytc{cidx}_{tname}_b rgp_rp_hytc{cidx} n_wp_hytc{cidx}_{tname}_c vdd PWRITE_HYTC{cidx} L={{LCH}} W={{WWRITE}}
MWP_HYTC{cidx}_{tname}D n_wp_hytc{cidx}_{tname}_c cdm_rp_hytc{cidx} wp_hytc{cidx}_{tname} vdd PWRITE_HYTC{cidx} L={{LCH}} W={{WWRITE}}
MWM_HYTC{cidx}_{tname}A vdd paccn_hytc{cidx}_{tname} n_wm_hytc{cidx}_{tname}_a vdd PWRITE_HYTC{cidx} L={{LCH}} W={{WWRITE}}
MWM_HYTC{cidx}_{tname}B n_wm_hytc{cidx}_{tname}_a hm_pos n_wm_hytc{cidx}_{tname}_b vdd PWRITE_HYTC{cidx} L={{LCH}} W={{WWRITE}}
MWM_HYTC{cidx}_{tname}C n_wm_hytc{cidx}_{tname}_b rgm_rp_hytc{cidx} n_wm_hytc{cidx}_{tname}_c vdd PWRITE_HYTC{cidx} L={{LCH}} W={{WWRITE}}
MWM_HYTC{cidx}_{tname}D n_wm_hytc{cidx}_{tname}_c cdp_rp_hytc{cidx} wm_hytc{cidx}_{tname} vdd PWRITE_HYTC{cidx} L={{LCH}} W={{WWRITE}}
"""
            )
            hybrid_timing_corner_prints.extend(
                [
                    f"v(wp_hytc{cidx}_{tname})",
                    f"v(wm_hytc{cidx}_{tname})",
                    f"v(paccn_hytc{cidx}_{tname})",
                ]
            )

    hybrid_timing_corner_deck = f"""
* Hybrid restored-enable/analog-error writer phase-overlap corners.
* This is the intersection of the timing and threshold-corner tests: each
* corner has its own MOS hidden-error store, restored gates, and writer PMOS
* threshold, while matched writer copies sweep active-low pacc timing.
{COMMON_MODELS}
{''.join(hybrid_timing_corner_models)}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VHM_POS hm_pos 0 0.92
{''.join(hybrid_timing_corner_devices)}

.control
set noaskquit
tran 5n 3.2u uic
wrdata mos_hidden_writer_restored_gate_hybrid_timing_corner.dat {' '.join(hybrid_timing_corner_prints)} v(pbwd)
quit
.endc
.end
"""
    hybrid_timing_corner_data = run_ngspice(
        hybrid_timing_corner_deck,
        "mos_hidden_writer_restored_gate_hybrid_timing_corner",
    )
    hytct, hytc_cols = load_wrdata(hybrid_timing_corner_data, len(hybrid_timing_corner_prints) + 1)

    def hytcat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hytct - time_s))])

    timing_count = len(hybrid_timing_cases)
    corner_stride = 4 + 3 * timing_count
    hytc_final = np.zeros((len(hybrid_timing_corner_cases), timing_count))
    hytc_comp_step = np.zeros_like(hytc_final)
    hytc_diffs: dict[tuple[str, str], np.ndarray] = {}
    hytc_hidden = []
    hytc_selected_gate = []
    hytc_complement_gate = []
    for cidx, (cname, clabel, *_rest) in enumerate(hybrid_timing_corner_cases):
        base = cidx * corner_stride
        hidden = hytc_cols[base] - hytc_cols[base + 1]
        selected_gate = hytc_cols[base + 2]
        complement_gate = hytc_cols[base + 3]
        hytc_hidden.append(hytcat(1.35e-6, hidden))
        hytc_selected_gate.append(hytcat(1.45e-6, selected_gate))
        hytc_complement_gate.append(hytcat(1.45e-6, complement_gate))
        require(hytc_hidden[-1] > 0.07, f"{clabel} timing-corner deck should store positive hidden error")
        require(hytc_selected_gate[-1] < 0.35, f"{clabel} timing-corner selected restored gate should be low")
        require(hytc_complement_gate[-1] > 1.60, f"{clabel} timing-corner complement restored gate should be high")
        for tidx, (tname, tlabel, _start_us, _end_us) in enumerate(hybrid_timing_cases):
            offset = base + 4 + 3 * tidx
            wp = hytc_cols[offset]
            wm = hytc_cols[offset + 1]
            diff = wp - wm
            hytc_diffs[(cname, tname)] = diff
            hytc_final[cidx, tidx] = hytcat(2.85e-6, diff)
            hytc_comp_step[cidx, tidx] = hytcat(2.85e-6, wm) - hytcat(0.05e-6, wm)
        full = hytc_final[cidx, -1]
        require(abs(hytc_final[cidx, 0]) < 0.001, f"{clabel} pre-store pacc should remain quiet")
        require(full > 0.010, f"{clabel} settled pacc should still produce a useful write")
        require(hytc_final[cidx, 1] < 0.25 * full, f"{clabel} early-overlap write should be a limited fraction of full")
        require(np.min(hytc_final[cidx, 2:]) > 0.90 * full, f"{clabel} post-store pacc timings should reach full write")
        require(
            np.max(hytc_final[cidx, 2:]) - np.min(hytc_final[cidx, 2:]) < 0.004,
            f"{clabel} post-store pacc timings should agree",
        )
        require(np.max(hytc_comp_step[cidx, 2:]) < 5e-4, f"{clabel} complement rail should stay suppressed")

    hytc_hidden = np.array(hytc_hidden)
    hytc_selected_gate = np.array(hytc_selected_gate)
    hytc_complement_gate = np.array(hytc_complement_gate)

    hytc_fig, hytc_axes = plt.subplots(2, 1, figsize=(7.4, 6.0))
    hytc_x = np.arange(timing_count)
    hytc_tlabels = [label for _name, label, _start_us, _end_us in hybrid_timing_cases]
    for cidx, (_cname, clabel, *_rest) in enumerate(hybrid_timing_corner_cases):
        hytc_axes[0].plot(hytc_x, 1e3 * hytc_final[cidx], "o-", label=clabel)
    hytc_axes[0].axhline(0, color="0.4", linewidth=0.8)
    hytc_axes[0].set_xticks(hytc_x)
    hytc_axes[0].set_xticklabels(hytc_tlabels, rotation=15, ha="right")
    hytc_axes[0].set_ylabel("final $W^+ - W^-$ (mV)")
    hytc_axes[0].set_title("Hybrid pacc timing rule survives selected threshold corners")
    hytc_axes[0].grid(True, alpha=0.25)
    hytc_axes[0].legend(loc="upper left", fontsize="small")
    hytc_axes[1].plot(1e6 * hytct, hytc_diffs[("nom", "pre")], color="0.5", linestyle=":", label="nominal pre")
    hytc_axes[1].plot(1e6 * hytct, hytc_diffs[("nom", "gap")], label="nominal settled")
    hytc_axes[1].plot(1e6 * hytct, hytc_diffs[("weak", "early")], label="combined weak early")
    hytc_axes[1].plot(1e6 * hytct, hytc_diffs[("weak", "gap")], label="combined weak settled")
    hytc_axes[1].plot(1e6 * hytct, hytc_diffs[("strong", "gap")], label="writer strong settled")
    hytc_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hytc_axes[1].set_xlabel("time (us)")
    hytc_axes[1].set_ylabel("$W^+ - W^-$ (V)")
    hytc_axes[1].set_title("Cornered traces keep sign; weak corner mainly reduces gain")
    hytc_axes[1].grid(True, alpha=0.25)
    hytc_axes[1].legend(loc="upper left", fontsize="small")
    hytc_fig.tight_layout()
    save_plot(hytc_fig, "mos_hidden_writer_restored_gate_hybrid_timing_corner_ngspice")

    hybrid_noise_cases = [
        ("none", "no kick", 0.000, 0.000),
        ("cmup", "common +25mV", 0.025, 0.025),
        ("cmdn", "common -25mV", -0.025, -0.025),
        ("boost", "diff boost", 0.025, -0.025),
        ("weak25", "diff weaken 25mV", -0.025, 0.025),
        ("weak50", "diff weaken 50mV", -0.050, 0.050),
    ]
    hybrid_noise_prints = []
    hybrid_noise_devices = []

    def kick_current(delta_v: float) -> float:
        return -10e-12 * delta_v / 50e-9

    for idx, (name, _label, d_cdp, d_cdm) in enumerate(hybrid_noise_cases):
        i_cdp = kick_current(d_cdp)
        i_cdm = kick_current(d_cdm)
        hybrid_noise_devices.append(
            f"""
* Hybrid disturbance copy: {name}, target kicks d(cdp)={d_cdp:+.3f} V, d(cdm)={d_cdm:+.3f} V.
VZPP_HYN{idx} zpp_hyn{idx} 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYN{idx} zmm_hyn{idx} 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYN{idx} zpm_hyn{idx} 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYN{idx} zmp_hyn{idx} 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYN{idx} hpp_hyn{idx} hpp_hyn{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYN{idx} hpm_hyn{idx} hpm_hyn{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYN{idx} hpp_hyn{idx} zpp_hyn{idx} tailp_hyn{idx} 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYN{idx} hpm_hyn{idx} zmm_hyn{idx} tailp_hyn{idx} 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYN{idx} tailp_hyn{idx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYN{idx} hmp_hyn{idx} hmp_hyn{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYN{idx} hmm_hyn{idx} hmm_hyn{idx} vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYN{idx} hmp_hyn{idx} zpm_hyn{idx} tailm_hyn{idx} 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYN{idx} hmm_hyn{idx} zmp_hyn{idx} tailm_hyn{idx} 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYN{idx} tailm_hyn{idx} vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYN{idx} cdp_rp_hyn{idx} 0 {{CERR}} IC=1.04
CDM_RP_HYN{idx} cdm_rp_hyn{idx} 0 {{CERR}} IC=1.04
RDP_RP_HYN{idx} cdp_rp_hyn{idx} 0 50G
RDM_RP_HYN{idx} cdm_rp_hyn{idx} 0 50G
{sign_store_path(f"hpm_hyn{idx}", "rp", f"cdp_rp_hyn{idx}", f"hynrp1_{idx}")}
{sign_store_path(f"hmp_hyn{idx}", "rp", f"cdp_rp_hyn{idx}", f"hynrp2_{idx}")}
{sign_store_path(f"hpp_hyn{idx}", "rp", f"cdm_rp_hyn{idx}", f"hynrp3_{idx}")}
{sign_store_path(f"hmm_hyn{idx}", "rp", f"cdm_rp_hyn{idx}", f"hynrp4_{idx}")}

IKDP_HYN{idx} cdp_rp_hyn{idx} 0 PWL(0 0 1.42u 0 1.43u {i_cdp:.6e} 1.48u {i_cdp:.6e} 1.49u 0 3.4u 0)
IKDM_HYN{idx} cdm_rp_hyn{idx} 0 PWL(0 0 1.42u 0 1.43u {i_cdm:.6e} 1.48u {i_cdm:.6e} 1.49u 0 3.4u 0)

MPRP_CDP_HYN{idx} rgp_rp_hyn{idx} cdp_rp_hyn{idx} vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYN{idx} rgp_rp_hyn{idx} cdp_rp_hyn{idx} 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYN{idx} rgm_rp_hyn{idx} cdm_rp_hyn{idx} vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYN{idx} rgm_rp_hyn{idx} cdm_rp_hyn{idx} 0 0 NMOS L={{LCH}} W={{WRESTN}}

CWP_HYN{idx} wp_hyn{idx} 0 {{CWRITE}} IC=0.85
CWM_HYN{idx} wm_hyn{idx} 0 {{CWRITE}} IC=0.85
MWP_HYN{idx}A vdd paccn_hyn n_wp_hyn{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYN{idx}B n_wp_hyn{idx}_a hm_pos n_wp_hyn{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYN{idx}C n_wp_hyn{idx}_b rgp_rp_hyn{idx} n_wp_hyn{idx}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYN{idx}D n_wp_hyn{idx}_c cdm_rp_hyn{idx} wp_hyn{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYN{idx}A vdd paccn_hyn n_wm_hyn{idx}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYN{idx}B n_wm_hyn{idx}_a hm_pos n_wm_hyn{idx}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYN{idx}C n_wm_hyn{idx}_b rgm_rp_hyn{idx} n_wm_hyn{idx}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYN{idx}D n_wm_hyn{idx}_c cdp_rp_hyn{idx} wm_hyn{idx} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        hybrid_noise_prints.extend(
            [
                f"v(cdp_rp_hyn{idx})",
                f"v(cdm_rp_hyn{idx})",
                f"v(rgp_rp_hyn{idx})",
                f"v(rgm_rp_hyn{idx})",
                f"v(wp_hyn{idx})",
                f"v(wm_hyn{idx})",
            ]
        )

    hybrid_noise_deck = f"""
* Hybrid restored-enable/analog-error writer charge-kick disturbance characterization.
* Each copy stores the same r+ hidden error, receives a deterministic small
* current kick on one or both hidden-error capacitors after pbwd closes, and
* then writes through the same hybrid restored-select/analog-magnitude stack.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_HYN paccn_hyn 0 PULSE(1.8 0 1.75u 20n 20n 0.80u 5.0u)
VHM_POS hm_pos 0 0.92
{''.join(hybrid_noise_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_restored_gate_hybrid_noise.dat {' '.join(hybrid_noise_prints)} v(pbwd) v(paccn_hyn)
quit
.endc
.end
"""
    hybrid_noise_data = run_ngspice(
        hybrid_noise_deck,
        "mos_hidden_writer_restored_gate_hybrid_noise",
    )
    hynt, hyn_cols = load_wrdata(hybrid_noise_data, len(hybrid_noise_prints) + 2)

    def hynat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hynt - time_s))])

    hyn_pre_hidden = []
    hyn_post_hidden = []
    hyn_selected_gate = []
    hyn_complement_gate = []
    hyn_selected_step = []
    hyn_complement_step = []
    hyn_net = []
    for idx, (name, _label, _d_cdp, _d_cdm) in enumerate(hybrid_noise_cases):
        base = 6 * idx
        cdp = hyn_cols[base]
        cdm = hyn_cols[base + 1]
        rgp = hyn_cols[base + 2]
        rgm = hyn_cols[base + 3]
        wp = hyn_cols[base + 4]
        wm = hyn_cols[base + 5]
        hidden = cdp - cdm
        pre_hidden = hynat(1.35e-6, hidden)
        post_hidden = hynat(1.62e-6, hidden)
        selected_gate = hynat(1.70e-6, rgp)
        complement_gate = hynat(1.70e-6, rgm)
        selected_step = hynat(2.85e-6, wp) - hynat(1.70e-6, wp)
        complement_step = hynat(2.85e-6, wm) - hynat(1.70e-6, wm)
        net = hynat(2.85e-6, wp - wm)
        hyn_pre_hidden.append(pre_hidden)
        hyn_post_hidden.append(post_hidden)
        hyn_selected_gate.append(selected_gate)
        hyn_complement_gate.append(complement_gate)
        hyn_selected_step.append(selected_step)
        hyn_complement_step.append(complement_step)
        hyn_net.append(net)
        require(pre_hidden > 0.07, f"{name} disturbance copy should store a positive hidden error before kick")
        if name == "weak50":
            require(post_hidden < 0.0, "over-margin destructive differential kick should reverse stored hidden-error sign")
            require(selected_gate > 1.0, "over-margin destructive kick should turn off the original selected gate")
            require(net < 0.0, "over-margin destructive kick should expose wrong-sign write risk")
            require(abs(net) < 0.003, "over-margin destructive kick should only create a small wrong-sign residue")
        else:
            require(post_hidden > 0.025, f"{name} disturbance copy should keep positive hidden-error sign after kick")
            require(selected_gate < 0.45, f"{name} disturbance selected restored gate should stay active-low")
            require(complement_gate > 1.55, f"{name} disturbance complement restored gate should stay inactive")
            require(complement_step < 5e-4, f"{name} disturbance complement rail should stay suppressed")
            require(net > 0.010, f"{name} disturbance should preserve a useful positive signed write")
        require(
            abs(hynat(3.25e-6, wp - wm) - net) < 5e-4,
            f"{name} disturbance write should hold after pacc closes",
        )

    hyn_pre_hidden = np.array(hyn_pre_hidden)
    hyn_post_hidden = np.array(hyn_post_hidden)
    hyn_selected_gate = np.array(hyn_selected_gate)
    hyn_complement_gate = np.array(hyn_complement_gate)
    hyn_selected_step = np.array(hyn_selected_step)
    hyn_complement_step = np.array(hyn_complement_step)
    hyn_net = np.array(hyn_net)
    require(abs(hyn_post_hidden[0] - hyn_pre_hidden[0]) < 0.005, "no-kick copy should hold hidden-error state")
    require(hyn_net[3] > hyn_net[0], "differential boost disturbance should strengthen the write")
    require(hyn_net[4] < hyn_net[0], "small destructive differential disturbance should weaken the write")
    require(hyn_net[5] < hyn_net[4], "larger destructive differential disturbance should weaken the write further")
    require(hyn_net[5] < 0.0, "over-margin destructive differential disturbance should mark the sign boundary")
    require(np.max(np.abs(hyn_net[1:3] - hyn_net[0])) < 0.012, "common-mode disturbance should not dominate signed write gain")

    hyn_fig, hyn_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hyn_labels = [label for _name, label, _d_cdp, _d_cdm in hybrid_noise_cases]
    hyn_x = np.arange(len(hybrid_noise_cases))
    hyn_axes[0].plot(hyn_x, hyn_pre_hidden, "o-", label="before kick")
    hyn_axes[0].plot(hyn_x, hyn_post_hidden, "s-", label="after kick")
    hyn_axes[0].plot(hyn_x, hyn_selected_gate, "^-", label="selected restored gate")
    hyn_axes[0].plot(hyn_x, hyn_complement_gate, "v--", label="complement restored gate")
    hyn_axes[0].axhline(0.9, color="0.4", linewidth=0.8, alpha=0.5)
    hyn_axes[0].set_xticks(hyn_x)
    hyn_axes[0].set_xticklabels(hyn_labels, rotation=15, ha="right")
    hyn_axes[0].set_ylabel("voltage (V)")
    hyn_axes[0].set_title("Hybrid writer keeps restored selectivity after hidden-error kicks")
    hyn_axes[0].grid(True, alpha=0.25)
    hyn_axes[0].legend(loc="center right", fontsize="small")
    hyn_axes[1].plot(hyn_x, hyn_selected_step, "o-", label="selected rail step")
    hyn_axes[1].plot(hyn_x, hyn_complement_step, "s--", label="complement rail step")
    hyn_axes[1].plot(hyn_x, hyn_net, "^-", label="$W^+ - W^-$")
    hyn_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyn_axes[1].set_xticks(hyn_x)
    hyn_axes[1].set_xticklabels(hyn_labels, rotation=15, ha="right")
    hyn_axes[1].set_ylabel("writer step (V)")
    hyn_axes[1].set_title("Small kicks change gain; over-margin differential kick flips sign")
    hyn_axes[1].grid(True, alpha=0.25)
    hyn_axes[1].legend(loc="upper right", ncol=2, fontsize="small")
    hyn_fig.tight_layout()
    save_plot(hyn_fig, "mos_hidden_writer_restored_gate_hybrid_noise_ngspice")

    hybrid_activation_cases = [
        ("h084", "$h^-=0.84$ V", 0.84),
        ("h088", "$h^-=0.88$ V", 0.88),
        ("h092", "$h^-=0.92$ V", 0.92),
        ("h098", "$h^-=0.98$ V", 0.98),
        ("h104", "$h^-=1.04$ V", 1.04),
        ("h110", "$h^-=1.10$ V", 1.10),
    ]
    hybrid_activation_devices = []
    hybrid_activation_prints = ["v(cdp_rp_hya)", "v(cdm_rp_hya)", "v(rgp_rp_hya)", "v(rgm_rp_hya)"]
    for name, _label, hgate in hybrid_activation_cases:
        hybrid_activation_devices.append(
            f"""
VHM_HYA_{name} hm_pos_hya_{name} 0 {hgate:.3f}
CWP_HYA_{name} wp_hya_{name} 0 {{CWRITE}} IC=0.85
CWM_HYA_{name} wm_hya_{name} 0 {{CWRITE}} IC=0.85
MWP_HYA_{name}A vdd paccn_hya n_wp_hya_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYA_{name}B n_wp_hya_{name}_a hm_pos_hya_{name} n_wp_hya_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYA_{name}C n_wp_hya_{name}_b rgp_rp_hya n_wp_hya_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYA_{name}D n_wp_hya_{name}_c cdm_rp_hya wp_hya_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYA_{name}A vdd paccn_hya n_wm_hya_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYA_{name}B n_wm_hya_{name}_a hm_pos_hya_{name} n_wm_hya_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYA_{name}C n_wm_hya_{name}_b rgm_rp_hya n_wm_hya_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYA_{name}D n_wm_hya_{name}_c cdp_rp_hya wm_hya_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        hybrid_activation_prints.extend([f"v(wp_hya_{name})", f"v(wm_hya_{name})"])

    hybrid_activation_deck = f"""
* Hybrid restored-enable/analog-error writer activation-gate characterization.
* The same stored r+ hidden error drives all copies.  Each writer copy sees a
* different active-low positive-activation complementary gate, testing whether
* the hybrid product path still scales with the activation-side operand.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_HYA paccn_hya 0 PULSE(1.8 0 1.55u 20n 20n 0.80u 5.0u)

VZPP_HYA zpp_hya 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYA zmm_hya 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYA zpm_hya 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYA zmp_hya 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYA hpp_hya hpp_hya vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYA hpm_hya hpm_hya vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYA hpp_hya zpp_hya tailp_hya 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYA hpm_hya zmm_hya tailp_hya 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYA tailp_hya vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYA hmp_hya hmp_hya vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYA hmm_hya hmm_hya vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYA hmp_hya zpm_hya tailm_hya 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYA hmm_hya zmp_hya tailm_hya 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYA tailm_hya vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYA cdp_rp_hya 0 {{CERR}} IC=1.04
CDM_RP_HYA cdm_rp_hya 0 {{CERR}} IC=1.04
RDP_RP_HYA cdp_rp_hya 0 50G
RDM_RP_HYA cdm_rp_hya 0 50G
{sign_store_path("hpm_hya", "rp", "cdp_rp_hya", "hyarp1")}
{sign_store_path("hmp_hya", "rp", "cdp_rp_hya", "hyarp2")}
{sign_store_path("hpp_hya", "rp", "cdm_rp_hya", "hyarp3")}
{sign_store_path("hmm_hya", "rp", "cdm_rp_hya", "hyarp4")}

MPRP_CDP_HYA rgp_rp_hya cdp_rp_hya vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYA rgp_rp_hya cdp_rp_hya 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYA rgm_rp_hya cdm_rp_hya vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYA rgm_rp_hya cdm_rp_hya 0 0 NMOS L={{LCH}} W={{WRESTN}}

{''.join(hybrid_activation_devices)}

.control
set noaskquit
tran 5n 3.4u uic
wrdata mos_hidden_writer_restored_gate_hybrid_activation.dat {' '.join(hybrid_activation_prints)} v(pbwd) v(paccn_hya)
quit
.endc
.end
"""
    hybrid_activation_data = run_ngspice(
        hybrid_activation_deck,
        "mos_hidden_writer_restored_gate_hybrid_activation",
    )
    hyatime, hya_cols = load_wrdata(hybrid_activation_data, len(hybrid_activation_prints) + 2)

    def hyaat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyatime - time_s))])

    hya_hidden = hya_cols[0] - hya_cols[1]
    hya_selected_gate = hya_cols[2]
    hya_complement_gate = hya_cols[3]
    hya_selected_step = []
    hya_complement_step = []
    hya_net = []
    for idx, (_name, _label, _hgate) in enumerate(hybrid_activation_cases):
        wp = hya_cols[4 + 2 * idx]
        wm = hya_cols[5 + 2 * idx]
        hya_selected_step.append(hyaat(2.85e-6, wp) - hyaat(1.45e-6, wp))
        hya_complement_step.append(hyaat(2.85e-6, wm) - hyaat(1.45e-6, wm))
        hya_net.append(hyaat(2.85e-6, wp - wm))
    hya_selected_step = np.array(hya_selected_step)
    hya_complement_step = np.array(hya_complement_step)
    hya_net = np.array(hya_net)
    require(hyaat(1.35e-6, hya_hidden) > 0.07, "hybrid activation deck should store a positive hidden error")
    require(hyaat(1.45e-6, hya_selected_gate) < 0.30, "hybrid activation selected restored gate should be low")
    require(hyaat(1.45e-6, hya_complement_gate) > 1.60, "hybrid activation complement restored gate should be high")
    require(np.all(hya_net > 0.004), "hybrid activation sweep should preserve positive signed writes")
    require(np.all(hya_net < 0.090), "hybrid activation sweep should keep writes incremental")
    require(np.all(hya_complement_step < 5e-4), "hybrid activation sweep should keep complement rail suppressed")
    require(np.all(np.diff(hya_net) < -0.001), "hybrid activation write should weaken as active-low activation gate rises")
    require(hya_net[0] > 1.6 * hya_net[-1], "hybrid activation gate should provide useful gain range")

    hya_fig, hya_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hya_gates = np.array([hgate for _name, _label, hgate in hybrid_activation_cases])
    hya_axes[0].plot(1e6 * hyatime, hya_hidden, label="stored $r^+$ hidden error")
    hya_axes[0].plot(1e6 * hyatime, hya_selected_gate, label="selected restored gate")
    hya_axes[0].plot(1e6 * hyatime, hya_complement_gate, label="complement restored gate")
    hya_axes[0].plot(1e6 * hyatime, hya_cols[-1] / 20.0, color="0.5", alpha=0.35, label="$pbwd/20$")
    hya_axes[0].axhline(0.9, color="0.4", linewidth=0.8, alpha=0.5)
    hya_axes[0].set_ylabel("voltage (V)")
    hya_axes[0].set_title("Hybrid activation sweep reuses one stored hidden-error rail")
    hya_axes[0].grid(True, alpha=0.25)
    hya_axes[0].legend(loc="center right", fontsize="small")
    hya_axes[1].plot(hya_gates, hya_selected_step, "o-", label="selected rail step")
    hya_axes[1].plot(hya_gates, hya_complement_step, "s--", label="complement rail step")
    hya_axes[1].plot(hya_gates, hya_net, "^-", label="$W^+ - W^-$")
    hya_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hya_axes[1].set_xlabel("active-low activation PMOS gate $h^-$ (V)")
    hya_axes[1].set_ylabel("writer step (V)")
    hya_axes[1].set_title("Hybrid write gain follows the activation-side analog gate")
    hya_axes[1].grid(True, alpha=0.25)
    hya_axes[1].legend(loc="upper right")
    hya_fig.tight_layout()
    save_plot(hya_fig, "mos_hidden_writer_restored_gate_hybrid_activation_ngspice")

    hybrid_activation_store_deck = f"""
* Hybrid restored-enable/analog-error writer with stored activation gate.
* A real activation capacitor is sampled through a complementary MOS pass gate:
* first to an active-low h- writer gate, then to an inactive-high value.  The
* same stored r+ hidden-error rail and same weight pair are used for both pacc
* pulses, checking that stale activation state is not required for selectivity.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CSTORE=10p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VACT_SRC_HYF hsrc_hyf 0 PWL(0 1.45 0.25u 1.45 0.27u 0.92 0.82u 0.92 0.84u 1.45 1.95u 1.45 2.10u 1.45 2.50u 1.45 4.1u 1.45)
VPSAMP_HYF psamp_hyf 0 PWL(0 0 0.30u 0 0.32u 1.8 0.72u 1.8 0.74u 0 2.10u 0 2.12u 1.8 2.52u 1.8 2.54u 0 4.1u 0)
VPSAMPN_HYF psampn_hyf 0 PWL(0 1.8 0.30u 1.8 0.32u 0 0.72u 0 0.74u 1.8 2.10u 1.8 2.12u 0 2.52u 0 2.54u 1.8 4.1u 1.8)
VPACC1_HYF paccn1_hyf 0 PWL(0 1.8 1.55u 1.8 1.57u 0 1.93u 0 1.95u 1.8 4.1u 1.8)
VPACC2_HYF paccn2_hyf 0 PWL(0 1.8 2.85u 1.8 2.87u 0 3.23u 0 3.25u 1.8 4.1u 1.8)

VZPP_HYF zpp_hyf 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYF zmm_hyf 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYF zpm_hyf 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYF zmp_hyf 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYF hpp_hyf hpp_hyf vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYF hpm_hyf hpm_hyf vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYF hpp_hyf zpp_hyf tailp_hyf 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYF hpm_hyf zmm_hyf tailp_hyf 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYF tailp_hyf vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYF hmp_hyf hmp_hyf vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYF hmm_hyf hmm_hyf vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYF hmp_hyf zpm_hyf tailm_hyf 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYF hmm_hyf zmp_hyf tailm_hyf 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYF tailm_hyf vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYF cdp_rp_hyf 0 {{CERR}} IC=1.04
CDM_RP_HYF cdm_rp_hyf 0 {{CERR}} IC=1.04
RDP_RP_HYF cdp_rp_hyf 0 50G
RDM_RP_HYF cdm_rp_hyf 0 50G
{sign_store_path("hpm_hyf", "rp", "cdp_rp_hyf", "hyfrp1")}
{sign_store_path("hmp_hyf", "rp", "cdp_rp_hyf", "hyfrp2")}
{sign_store_path("hpp_hyf", "rp", "cdm_rp_hyf", "hyfrp3")}
{sign_store_path("hmm_hyf", "rp", "cdm_rp_hyf", "hyfrp4")}

MPRP_CDP_HYF rgp_rp_hyf cdp_rp_hyf vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYF rgp_rp_hyf cdp_rp_hyf 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYF rgm_rp_hyf cdm_rp_hyf vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYF rgm_rp_hyf cdm_rp_hyf 0 0 NMOS L={{LCH}} W={{WRESTN}}

CHM_HYF hm_store_hyf 0 {{CSTORE}} IC=1.45
RHM_HYF hm_store_hyf 0 50G
MSACTN_HYF hsrc_hyf psamp_hyf hm_store_hyf 0 NMOS L={{LCH}} W={{WSW}}
MSACTP_HYF hsrc_hyf psampn_hyf hm_store_hyf vdd PMOS L={{LCH}} W={{WSW}}

CWP_HYF wp_hyf 0 {{CWRITE}} IC=0.85
CWM_HYF wm_hyf 0 {{CWRITE}} IC=0.85
MWP1_HYF_A vdd paccn1_hyf n_wp1_hyf_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP1_HYF_B n_wp1_hyf_a hm_store_hyf n_wp1_hyf_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP1_HYF_C n_wp1_hyf_b rgp_rp_hyf n_wp1_hyf_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP1_HYF_D n_wp1_hyf_c cdm_rp_hyf wp_hyf vdd PMOS L={{LCH}} W={{WWRITE}}
MWM1_HYF_A vdd paccn1_hyf n_wm1_hyf_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM1_HYF_B n_wm1_hyf_a hm_store_hyf n_wm1_hyf_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM1_HYF_C n_wm1_hyf_b rgm_rp_hyf n_wm1_hyf_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM1_HYF_D n_wm1_hyf_c cdp_rp_hyf wm_hyf vdd PMOS L={{LCH}} W={{WWRITE}}
MWP2_HYF_A vdd paccn2_hyf n_wp2_hyf_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP2_HYF_B n_wp2_hyf_a hm_store_hyf n_wp2_hyf_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP2_HYF_C n_wp2_hyf_b rgp_rp_hyf n_wp2_hyf_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP2_HYF_D n_wp2_hyf_c cdm_rp_hyf wp_hyf vdd PMOS L={{LCH}} W={{WWRITE}}
MWM2_HYF_A vdd paccn2_hyf n_wm2_hyf_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM2_HYF_B n_wm2_hyf_a hm_store_hyf n_wm2_hyf_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM2_HYF_C n_wm2_hyf_b rgm_rp_hyf n_wm2_hyf_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM2_HYF_D n_wm2_hyf_c cdp_rp_hyf wm_hyf vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 5n 4.0u uic
wrdata mos_hidden_writer_restored_gate_hybrid_activation_store.dat v(cdp_rp_hyf) v(cdm_rp_hyf) v(rgp_rp_hyf) v(rgm_rp_hyf) v(hsrc_hyf) v(hm_store_hyf) v(wp_hyf) v(wm_hyf) v(psamp_hyf) v(paccn1_hyf) v(paccn2_hyf)
quit
.endc
.end
"""
    hybrid_activation_store_data = run_ngspice(
        hybrid_activation_store_deck,
        "mos_hidden_writer_restored_gate_hybrid_activation_store",
    )
    hyft, hyf_cols = load_wrdata(hybrid_activation_store_data, 11)

    def hyfat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyft - time_s))])

    hyf_hidden = hyf_cols[0] - hyf_cols[1]
    hyf_selected_gate = hyf_cols[2]
    hyf_complement_gate = hyf_cols[3]
    hyf_hsrc = hyf_cols[4]
    hyf_hcap = hyf_cols[5]
    hyf_wp = hyf_cols[6]
    hyf_wm = hyf_cols[7]
    hyf_weight = hyf_wp - hyf_wm
    hyf_first_weight = hyfat(2.05e-6, hyf_weight)
    hyf_second_weight = hyfat(3.35e-6, hyf_weight)
    hyf_second_increment = hyf_second_weight - hyf_first_weight
    require(hyfat(1.35e-6, hyf_hidden) > 0.07, "hybrid activation-store deck should store positive hidden error")
    require(hyfat(1.45e-6, hyf_selected_gate) < 0.30, "hybrid activation-store selected restored gate should be low")
    require(hyfat(1.45e-6, hyf_complement_gate) > 1.60, "hybrid activation-store complement restored gate should be high")
    require(hyfat(0.90e-6, hyf_hcap) < 0.95, "hybrid activation-store should sample active-low activation")
    require(abs(hyfat(1.40e-6, hyf_hcap) - hyfat(0.90e-6, hyf_hcap)) < 0.003, "active activation cap should hold before first pacc")
    require(hyf_first_weight > 0.020, "active stored activation should produce a useful W+ write")
    require(hyfat(2.65e-6, hyf_hcap) > 1.35, "hybrid activation-store should overwrite with inactive-high activation")
    require(abs(hyf_second_increment) < 0.002, "inactive stored activation should suppress the second pacc write")
    require(hyfat(3.85e-6, hyf_weight) - hyf_first_weight < 0.002, "hybrid activation-store final weight should not drift after inactive pulse")
    require(hyfat(3.35e-6, hyf_wm) - hyfat(1.45e-6, hyf_wm) < 5e-4, "stored activation gate should keep complement rail suppressed")

    hyf_fig, hyf_axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    hyf_axes[0].plot(1e6 * hyft, hyf_hidden, label="stored $r^+$")
    hyf_axes[0].plot(1e6 * hyft, hyf_selected_gate, label="selected restored gate")
    hyf_axes[0].plot(1e6 * hyft, hyf_complement_gate, label="complement restored gate")
    hyf_axes[0].set_ylabel("voltage (V)")
    hyf_axes[0].set_title("Hybrid stored-activation deck reuses one hidden-error rail")
    hyf_axes[0].grid(True, alpha=0.25)
    hyf_axes[0].legend(loc="center right", fontsize="small")
    hyf_axes[1].plot(1e6 * hyft, hyf_hsrc, label="activation source")
    hyf_axes[1].plot(1e6 * hyft, hyf_hcap, label="stored activation gate $h^-$")
    hyf_axes[1].plot(1e6 * hyft, hyf_cols[8] / 2.0, color="0.5", alpha=0.35, label="$psamp/2$")
    hyf_axes[1].axhline(0.92, color="0.4", linewidth=0.8, alpha=0.5)
    hyf_axes[1].axhline(1.45, color="0.4", linewidth=0.8, alpha=0.35)
    hyf_axes[1].set_ylabel("activation gate (V)")
    hyf_axes[1].set_title("MOS pass gate samples active, then inactive activation")
    hyf_axes[1].grid(True, alpha=0.25)
    hyf_axes[1].legend(loc="center right", fontsize="small")
    hyf_axes[2].plot(1e6 * hyft, hyf_wp - hyfat(1.45e-6, hyf_wp), label="$W^+$ step")
    hyf_axes[2].plot(1e6 * hyft, hyf_wm - hyfat(1.45e-6, hyf_wm), label="$W^-$ step")
    hyf_axes[2].plot(1e6 * hyft, hyf_weight, label="$W^+ - W^-$")
    hyf_axes[2].plot(1e6 * hyft, hyf_cols[9] / 20.0, color="0.45", alpha=0.3, label="$pacc^1_n/20$")
    hyf_axes[2].plot(1e6 * hyft, hyf_cols[10] / 20.0, color="0.15", alpha=0.25, label="$pacc^2_n/20$")
    hyf_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyf_axes[2].set_xlabel("time (us)")
    hyf_axes[2].set_ylabel("weight step (V)")
    hyf_axes[2].set_title("Stored active gate writes; stored inactive gate suppresses reuse pulse")
    hyf_axes[2].grid(True, alpha=0.25)
    hyf_axes[2].legend(loc="upper left", ncol=2, fontsize="small")
    hyf_fig.tight_layout()
    save_plot(hyf_fig, "mos_hidden_writer_restored_gate_hybrid_activation_store_ngspice")

    hybrid_activation_timing_cases = [
        ("pre", "pre-sample", 1.10, 1.46),
        ("early", "sample edge", 1.45, 1.81),
        ("overlap", "sample overlap", 1.62, 1.98),
        ("late", "post-sample", 2.10, 2.46),
        ("gap", "settled gap", 2.65, 3.01),
    ]
    hybrid_activation_timing_devices = []
    hybrid_activation_timing_prints = ["v(cdp_rp_hyg)", "v(cdm_rp_hyg)", "v(rgp_rp_hyg)", "v(rgm_rp_hyg)"]
    for name, _label, start_us, end_us in hybrid_activation_timing_cases:
        hybrid_activation_timing_devices.append(
            f"""
VPACC_HYG_{name} paccn_hyg_{name} 0 PWL(0 1.8 {start_us:.2f}u 1.8 {start_us + 0.02:.2f}u 0 {end_us:.2f}u 0 {end_us + 0.02:.2f}u 1.8 3.4u 1.8)
CHM_HYG_{name} hm_store_hyg_{name} 0 {{CSTORE}} IC=1.45
RHM_HYG_{name} hm_store_hyg_{name} 0 50G
MSACTN_HYG_{name} hsrc_hyg psamp_hyg hm_store_hyg_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSACTP_HYG_{name} hsrc_hyg psampn_hyg hm_store_hyg_{name} vdd PMOS L={{LCH}} W={{WSW}}
CWP_HYG_{name} wp_hyg_{name} 0 {{CWRITE}} IC=0.85
CWM_HYG_{name} wm_hyg_{name} 0 {{CWRITE}} IC=0.85
MWP_HYG_{name}A vdd paccn_hyg_{name} n_wp_hyg_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYG_{name}B n_wp_hyg_{name}_a hm_store_hyg_{name} n_wp_hyg_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYG_{name}C n_wp_hyg_{name}_b rgp_rp_hyg n_wp_hyg_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYG_{name}D n_wp_hyg_{name}_c cdm_rp_hyg wp_hyg_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYG_{name}A vdd paccn_hyg_{name} n_wm_hyg_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYG_{name}B n_wm_hyg_{name}_a hm_store_hyg_{name} n_wm_hyg_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYG_{name}C n_wm_hyg_{name}_b rgm_rp_hyg n_wm_hyg_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYG_{name}D n_wm_hyg_{name}_c cdp_rp_hyg wm_hyg_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        hybrid_activation_timing_prints.extend(
            [f"v(hm_store_hyg_{name})", f"v(wp_hyg_{name})", f"v(wm_hyg_{name})", f"v(paccn_hyg_{name})"]
        )

    hybrid_activation_timing_deck = f"""
* Hybrid writer activation-store to pacc timing margin check.
* Matched writer copies share one stored r+ hidden-error rail and one activation
* source/sampling phase.  Only pacc timing changes across copies.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CSTORE=10p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VACT_SRC_HYG hsrc_hyg 0 PWL(0 1.45 1.45u 1.45 1.47u 0.92 2.20u 0.92 2.22u 1.45 3.4u 1.45)
VPSAMP_HYG psamp_hyg 0 PWL(0 0 1.55u 0 1.57u 1.8 1.95u 1.8 1.97u 0 3.4u 0)
VPSAMPN_HYG psampn_hyg 0 PWL(0 1.8 1.55u 1.8 1.57u 0 1.95u 0 1.97u 1.8 3.4u 1.8)

VZPP_HYG zpp_hyg 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYG zmm_hyg 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYG zpm_hyg 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYG zmp_hyg 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYG hpp_hyg hpp_hyg vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYG hpm_hyg hpm_hyg vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYG hpp_hyg zpp_hyg tailp_hyg 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYG hpm_hyg zmm_hyg tailp_hyg 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYG tailp_hyg vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYG hmp_hyg hmp_hyg vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYG hmm_hyg hmm_hyg vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYG hmp_hyg zpm_hyg tailm_hyg 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYG hmm_hyg zmp_hyg tailm_hyg 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYG tailm_hyg vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYG cdp_rp_hyg 0 {{CERR}} IC=1.04
CDM_RP_HYG cdm_rp_hyg 0 {{CERR}} IC=1.04
RDP_RP_HYG cdp_rp_hyg 0 50G
RDM_RP_HYG cdm_rp_hyg 0 50G
{sign_store_path("hpm_hyg", "rp", "cdp_rp_hyg", "hygrp1")}
{sign_store_path("hmp_hyg", "rp", "cdp_rp_hyg", "hygrp2")}
{sign_store_path("hpp_hyg", "rp", "cdm_rp_hyg", "hygrp3")}
{sign_store_path("hmm_hyg", "rp", "cdm_rp_hyg", "hygrp4")}

MPRP_CDP_HYG rgp_rp_hyg cdp_rp_hyg vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYG rgp_rp_hyg cdp_rp_hyg 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYG rgm_rp_hyg cdm_rp_hyg vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYG rgm_rp_hyg cdm_rp_hyg 0 0 NMOS L={{LCH}} W={{WRESTN}}

{''.join(hybrid_activation_timing_devices)}

.control
set noaskquit
tran 5n 3.35u uic
wrdata mos_hidden_writer_restored_gate_hybrid_activation_timing.dat {' '.join(hybrid_activation_timing_prints)} v(hsrc_hyg) v(psamp_hyg)
quit
.endc
.end
"""
    hybrid_activation_timing_data = run_ngspice(
        hybrid_activation_timing_deck,
        "mos_hidden_writer_restored_gate_hybrid_activation_timing",
    )
    hyggt, hygg_cols = load_wrdata(hybrid_activation_timing_data, len(hybrid_activation_timing_prints) + 2)

    def hyggat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyggt - time_s))])

    hygg_hidden = hygg_cols[0] - hygg_cols[1]
    hygg_selected_gate = hygg_cols[2]
    hygg_complement_gate = hygg_cols[3]
    hygg_final = []
    hygg_hcap_at_start = []
    hygg_wm_step = []
    for idx, (_name, _label, start_us, end_us) in enumerate(hybrid_activation_timing_cases):
        base = 4 + 4 * idx
        hcap = hygg_cols[base]
        wp = hygg_cols[base + 1]
        wm = hygg_cols[base + 2]
        start_s = start_us * 1e-6
        end_s = end_us * 1e-6
        hygg_hcap_at_start.append(hyggat(start_s + 0.04e-6, hcap))
        hygg_final.append(hyggat(end_s + 0.16e-6, wp - wm))
        hygg_wm_step.append(hyggat(end_s + 0.16e-6, wm) - hyggat(start_s, wm))
    hygg_final = np.array(hygg_final)
    hygg_hcap_at_start = np.array(hygg_hcap_at_start)
    hygg_wm_step = np.array(hygg_wm_step)
    require(hyggat(1.35e-6, hygg_hidden) > 0.07, "hybrid activation-timing deck should store positive hidden error")
    require(hyggat(1.45e-6, hygg_selected_gate) < 0.30, "hybrid activation-timing selected restored gate should be low")
    require(hyggat(1.45e-6, hygg_complement_gate) > 1.60, "hybrid activation-timing complement restored gate should be high")
    require(hygg_final[0] < 0.002, "pacc before activation sampling should be quiet")
    require(hygg_final[1] > hygg_final[0] + 0.002, "pacc at activation-sample edge should expose a partial write")
    require(hygg_final[1] < 0.90 * hygg_final[-1], "activation-edge write should stay below settled write")
    require(np.min(hygg_final[2:]) > 0.90 * hygg_final[-1], "pacc after activation sampling starts should reach full write")
    require(np.max(hygg_final[2:]) - np.min(hygg_final[2:]) < 0.004, "settled activation timing writes should agree")
    require(np.max(hygg_wm_step) < 5e-4, "activation timing sweep should keep complement rail suppressed")

    hygg_fig, hygg_axes = plt.subplots(3, 1, figsize=(7.2, 7.2), gridspec_kw={"height_ratios": [1.15, 1.0, 1.0]})
    hygg_axes[0].plot(1e6 * hyggt, hygg_hidden, label="stored $r^+$")
    hygg_axes[0].plot(1e6 * hyggt, hygg_selected_gate, label="selected restored gate")
    hygg_axes[0].plot(1e6 * hyggt, hygg_complement_gate, label="complement restored gate")
    hygg_axes[0].plot(1e6 * hyggt, hygg_cols[-2], color="0.5", alpha=0.45, label="activation source")
    hygg_axes[0].plot(1e6 * hyggt, hygg_cols[-1] / 2.0, color="0.35", alpha=0.35, label="$psamp/2$")
    hygg_axes[0].set_ylabel("voltage (V)")
    hygg_axes[0].set_title("Hybrid activation-timing sweep shares one hidden-error rail")
    hygg_axes[0].set_xlim(0.35, 2.35)
    hygg_axes[0].grid(True, alpha=0.25)
    hygg_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hygg_x = np.arange(len(hybrid_activation_timing_cases))
    hygg_labels = [label for _name, label, _start, _end in hybrid_activation_timing_cases]
    hygg_axes[1].plot(hygg_x, hygg_hcap_at_start, "^-", color="tab:green", label="$h^-$ near pacc start")
    hygg_axes[1].axhline(0.92, color="tab:green", linewidth=0.9, linestyle=":", label="active sample")
    hygg_axes[1].axhline(1.45, color="0.45", linewidth=0.9, linestyle=":", label="inactive sample")
    hygg_axes[1].set_xticks(hygg_x)
    hygg_axes[1].set_xticklabels(hygg_labels, rotation=15, ha="right")
    hygg_axes[1].set_ylabel("activation gate (V)")
    hygg_axes[1].set_title("sampled activation seen when pacc starts")
    hygg_axes[1].grid(True, alpha=0.25)
    hygg_axes[1].legend(loc="center right", ncol=3, fontsize="small")
    hygg_axes[2].plot(hygg_x, 1e3 * hygg_final, "o-", label="$W^+ - W^-$")
    hygg_axes[2].plot(hygg_x, 1e3 * hygg_wm_step, "s--", label="complement $W^-$ step")
    hygg_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hygg_axes[2].set_xticks(hygg_x)
    hygg_axes[2].set_xticklabels(hygg_labels, rotation=15, ha="right")
    hygg_axes[2].set_ylabel("writer step (mV)")
    hygg_axes[2].set_title("pacc must wait for activation-store sampling margin")
    hygg_axes[2].grid(True, alpha=0.25)
    hygg_axes[2].legend(loc="upper left", ncol=2, fontsize="small")
    hygg_fig.tight_layout()
    save_plot(hygg_fig, "mos_hidden_writer_restored_gate_hybrid_activation_timing_ngspice")

    hybrid_width_cases = [
        ("w040", "40 ns", 2.10, 2.14),
        ("w080", "80 ns", 2.10, 2.18),
        ("w160", "160 ns", 2.10, 2.26),
        ("w320", "320 ns", 2.10, 2.42),
        ("w640", "640 ns", 2.10, 2.74),
    ]
    hybrid_width_devices = []
    hybrid_width_prints = [
        "v(cdp_rp_hyw)",
        "v(cdm_rp_hyw)",
        "v(rgp_rp_hyw)",
        "v(rgm_rp_hyw)",
        "v(hm_store_hyw)",
    ]
    for name, _label, start_us, end_us in hybrid_width_cases:
        hybrid_width_devices.append(
            f"""
VPACC_HYW_{name} paccn_hyw_{name} 0 PWL(0 1.8 {start_us:.2f}u 1.8 {start_us + 0.02:.2f}u 0 {end_us:.2f}u 0 {end_us + 0.02:.2f}u 1.8 3.2u 1.8)
CWP_HYW_{name} wp_hyw_{name} 0 {{CWRITE}} IC=0.85
CWM_HYW_{name} wm_hyw_{name} 0 {{CWRITE}} IC=0.85
MWP_HYW_{name}A vdd paccn_hyw_{name} n_wp_hyw_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYW_{name}B n_wp_hyw_{name}_a hm_store_hyw n_wp_hyw_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYW_{name}C n_wp_hyw_{name}_b rgp_rp_hyw n_wp_hyw_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYW_{name}D n_wp_hyw_{name}_c cdm_rp_hyw wp_hyw_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYW_{name}A vdd paccn_hyw_{name} n_wm_hyw_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYW_{name}B n_wm_hyw_{name}_a hm_store_hyw n_wm_hyw_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYW_{name}C n_wm_hyw_{name}_b rgm_rp_hyw n_wm_hyw_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYW_{name}D n_wm_hyw_{name}_c cdp_rp_hyw wm_hyw_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        hybrid_width_prints.extend([f"v(wp_hyw_{name})", f"v(wm_hyw_{name})", f"v(paccn_hyw_{name})"])

    hybrid_width_deck = f"""
* Hybrid restored-enable/analog-error writer pacc pulse-width characterization.
* Matched writer copies share one stored r+ hidden-error rail and one sampled
* activation capacitor.  Only the active-low pacc pulse width changes, giving
* a transistor-level learning-rate control check for the current hybrid writer.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CSTORE=10p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VACT_SRC_HYW hsrc_hyw 0 PWL(0 1.45 1.45u 1.45 1.47u 0.92 1.90u 0.92 1.92u 1.45 3.2u 1.45)
VPSAMP_HYW psamp_hyw 0 PWL(0 0 1.55u 0 1.57u 1.8 1.85u 1.8 1.87u 0 3.2u 0)
VPSAMPN_HYW psampn_hyw 0 PWL(0 1.8 1.55u 1.8 1.57u 0 1.85u 0 1.87u 1.8 3.2u 1.8)

VZPP_HYW zpp_hyw 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYW zmm_hyw 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYW zpm_hyw 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYW zmp_hyw 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYW hpp_hyw hpp_hyw vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYW hpm_hyw hpm_hyw vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYW hpp_hyw zpp_hyw tailp_hyw 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYW hpm_hyw zmm_hyw tailp_hyw 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYW tailp_hyw vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYW hmp_hyw hmp_hyw vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYW hmm_hyw hmm_hyw vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYW hmp_hyw zpm_hyw tailm_hyw 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYW hmm_hyw zmp_hyw tailm_hyw 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYW tailm_hyw vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYW cdp_rp_hyw 0 {{CERR}} IC=1.04
CDM_RP_HYW cdm_rp_hyw 0 {{CERR}} IC=1.04
RDP_RP_HYW cdp_rp_hyw 0 50G
RDM_RP_HYW cdm_rp_hyw 0 50G
{sign_store_path("hpm_hyw", "rp", "cdp_rp_hyw", "hywrp1")}
{sign_store_path("hmp_hyw", "rp", "cdp_rp_hyw", "hywrp2")}
{sign_store_path("hpp_hyw", "rp", "cdm_rp_hyw", "hywrp3")}
{sign_store_path("hmm_hyw", "rp", "cdm_rp_hyw", "hywrp4")}

MPRP_CDP_HYW rgp_rp_hyw cdp_rp_hyw vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYW rgp_rp_hyw cdp_rp_hyw 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYW rgm_rp_hyw cdm_rp_hyw vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYW rgm_rp_hyw cdm_rp_hyw 0 0 NMOS L={{LCH}} W={{WRESTN}}

CHM_HYW hm_store_hyw 0 {{CSTORE}} IC=1.45
RHM_HYW hm_store_hyw 0 50G
MSACTN_HYW hsrc_hyw psamp_hyw hm_store_hyw 0 NMOS L={{LCH}} W={{WSW}}
MSACTP_HYW hsrc_hyw psampn_hyw hm_store_hyw vdd PMOS L={{LCH}} W={{WSW}}

{''.join(hybrid_width_devices)}

.control
set noaskquit
tran 5n 3.15u uic
wrdata mos_hidden_writer_restored_gate_hybrid_width.dat {' '.join(hybrid_width_prints)} v(hsrc_hyw) v(psamp_hyw)
quit
.endc
.end
"""
    hybrid_width_data = run_ngspice(
        hybrid_width_deck,
        "mos_hidden_writer_restored_gate_hybrid_width",
    )
    hywt, hyw_cols = load_wrdata(hybrid_width_data, len(hybrid_width_prints) + 2)

    def hywat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hywt - time_s))])

    hyw_hidden = hyw_cols[0] - hyw_cols[1]
    hyw_selected_gate = hyw_cols[2]
    hyw_complement_gate = hyw_cols[3]
    hyw_hcap = hyw_cols[4]
    hyw_width_ns = np.array([(end_us - start_us) * 1e3 for _name, _label, start_us, end_us in hybrid_width_cases])
    hyw_final = []
    hyw_wp_step = []
    hyw_wm_step = []
    for idx, (_name, _label, _start_us, end_us) in enumerate(hybrid_width_cases):
        base = 5 + 3 * idx
        wp = hyw_cols[base]
        wm = hyw_cols[base + 1]
        end_s = end_us * 1e-6
        hyw_final.append(hywat(end_s + 0.18e-6, wp - wm))
        hyw_wp_step.append(hywat(end_s + 0.18e-6, wp) - hywat(2.00e-6, wp))
        hyw_wm_step.append(hywat(end_s + 0.18e-6, wm) - hywat(2.00e-6, wm))
    hyw_final = np.array(hyw_final)
    hyw_wp_step = np.array(hyw_wp_step)
    hyw_wm_step = np.array(hyw_wm_step)
    hyw_fit = np.polyval(np.polyfit(hyw_width_ns, hyw_final, 1), hyw_width_ns)
    hyw_r2 = 1.0 - float(np.sum((hyw_final - hyw_fit) ** 2) / np.sum((hyw_final - np.mean(hyw_final)) ** 2))
    require(hywat(1.35e-6, hyw_hidden) > 0.07, "hybrid width deck should store positive hidden error")
    require(hywat(1.45e-6, hyw_selected_gate) < 0.30, "hybrid width selected restored gate should be low")
    require(hywat(1.45e-6, hyw_complement_gate) > 1.60, "hybrid width complement restored gate should be high")
    require(abs(hywat(2.00e-6, hyw_hcap) - 0.92) < 0.010, "hybrid width sampled activation should settle before pacc")
    require(abs(hywat(3.00e-6, hyw_hcap) - hywat(2.00e-6, hyw_hcap)) < 0.002, "hybrid width activation cap should hold through pacc fanout")
    require(np.all(np.diff(hyw_final) > 0.002), "hybrid pacc width sweep should monotonically increase signed write")
    require(hyw_final[0] > 0.001, "short hybrid pacc pulse should create a measurable update")
    require(hyw_final[-1] < 0.12, "long hybrid pacc pulse should remain in incremental range")
    require(hyw_r2 > 0.995, "hybrid pacc width response should be near-linear over the tested width range")
    require(np.max(hyw_wm_step) < 5e-4, "hybrid pacc width sweep should keep complement rail suppressed")
    require(np.max(np.abs(hyw_final - hyw_wp_step)) < 6e-4, "hybrid pacc width signed write should be selected-rail dominated")

    hyw_fig, hyw_axes = plt.subplots(3, 1, figsize=(7.2, 7.2), gridspec_kw={"height_ratios": [1.15, 1.0, 1.0]})
    hyw_axes[0].plot(1e6 * hywt, hyw_hidden, label="stored $r^+$")
    hyw_axes[0].plot(1e6 * hywt, hyw_selected_gate, label="selected restored gate")
    hyw_axes[0].plot(1e6 * hywt, hyw_complement_gate, label="complement restored gate")
    hyw_axes[0].plot(1e6 * hywt, hyw_hcap, color="tab:purple", label="sampled $h^-$")
    hyw_axes[0].set_xlim(0.35, 3.05)
    hyw_axes[0].set_ylabel("voltage (V)")
    hyw_axes[0].set_title("Hybrid width sweep shares hidden-error and sampled activation state")
    hyw_axes[0].grid(True, alpha=0.25)
    hyw_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    for idx, (_name, label, _start_us, _end_us) in enumerate(hybrid_width_cases):
        base = 5 + 3 * idx
        hyw_axes[1].plot(1e6 * hywt, 1e3 * (hyw_cols[base] - hyw_cols[base + 1]), label=label)
    hyw_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyw_axes[1].set_ylabel("$W^+ - W^-$ (mV)")
    hyw_axes[1].set_title("active-low pacc pulse width controls signed update magnitude")
    hyw_axes[1].grid(True, alpha=0.25)
    hyw_axes[1].legend(loc="upper left", ncol=3, fontsize="small")
    hyw_axes[2].plot(hyw_width_ns, 1e3 * hyw_final, "o-", label="$W^+ - W^-$")
    hyw_axes[2].plot(hyw_width_ns, 1e3 * hyw_wm_step, "s--", label="complement $W^-$ step")
    hyw_axes[2].plot(hyw_width_ns, 1e3 * hyw_fit, ":", color="0.3", label=f"linear fit, $R^2={hyw_r2:.3f}$")
    hyw_axes[2].set_xlabel("pacc active-low width (ns)")
    hyw_axes[2].set_ylabel("final step (mV)")
    hyw_axes[2].set_title("learning-rate pulse width is monotone and linear in this window")
    hyw_axes[2].grid(True, alpha=0.25)
    hyw_axes[2].legend(loc="upper left", fontsize="small")
    hyw_fig.tight_layout()
    save_plot(hyw_fig, "mos_hidden_writer_restored_gate_hybrid_width_ngspice")

    hybrid_product_cases = [
        ("pp", "$a^+ e^+$", 0.92, 1.45, "rp", 1.00),
        ("pn", "$a^+ e^-$", 0.92, 1.45, "rm", -1.00),
        ("mp", "$a^- e^+$", 1.45, 0.92, "rp", -1.00),
        ("mn", "$a^- e^-$", 1.45, 0.92, "rm", 1.00),
    ]
    hybrid_product_devices = []
    hybrid_product_prints = [
        "v(cdp_rp_hyq)",
        "v(cdm_rp_hyq)",
        "v(cdp_rm_hyq)",
        "v(cdm_rm_hyq)",
        "v(rgp_rp_hyq)",
        "v(rgm_rp_hyq)",
        "v(rgp_rm_hyq)",
        "v(rgm_rm_hyq)",
    ]
    for name, _label, hplus_src, hminus_src, err_sign, _expected_sign in hybrid_product_cases:
        if err_sign == "rp":
            wp_restored = "rgp_rp_hyq"
            wp_error = "cdm_rp_hyq"
            wm_restored = "rgp_rp_hyq"
            wm_error = "cdm_rp_hyq"
        else:
            wp_restored = "rgm_rm_hyq"
            wp_error = "cdp_rm_hyq"
            wm_restored = "rgm_rm_hyq"
            wm_error = "cdp_rm_hyq"
        hybrid_product_devices.append(
            f"""
VHP_SRC_HYQ_{name} hp_src_hyq_{name} 0 {hplus_src:.3f}
VHM_SRC_HYQ_{name} hm_src_hyq_{name} 0 {hminus_src:.3f}
CHP_HYQ_{name} hp_store_hyq_{name} 0 {{CSTORE}} IC=1.45
CHM_HYQ_{name} hm_store_hyq_{name} 0 {{CSTORE}} IC=1.45
RHP_HYQ_{name} hp_store_hyq_{name} 0 50G
RHM_HYQ_{name} hm_store_hyq_{name} 0 50G
MSHPN_HYQ_{name} hp_src_hyq_{name} psamp_hyq hp_store_hyq_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSHPP_HYQ_{name} hp_src_hyq_{name} psampn_hyq hp_store_hyq_{name} vdd PMOS L={{LCH}} W={{WSW}}
MSHMN_HYQ_{name} hm_src_hyq_{name} psamp_hyq hm_store_hyq_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSHMP_HYQ_{name} hm_src_hyq_{name} psampn_hyq hm_store_hyq_{name} vdd PMOS L={{LCH}} W={{WSW}}
CWP_HYQ_{name} wp_hyq_{name} 0 {{CWRITE}} IC=0.85
CWM_HYQ_{name} wm_hyq_{name} 0 {{CWRITE}} IC=0.85

* Same-sign product branch for this case.
MWP_HYQ_{name}A vdd paccn_hyq n_wp_hyq_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYQ_{name}B n_wp_hyq_{name}_a {'hp_store_hyq_' + name if err_sign == 'rp' else 'hm_store_hyq_' + name} n_wp_hyq_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYQ_{name}C n_wp_hyq_{name}_b {wp_restored} n_wp_hyq_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYQ_{name}D n_wp_hyq_{name}_c {wp_error} wp_hyq_{name} vdd PMOS L={{LCH}} W={{WWRITE}}

* Opposite-sign product branch for this case.
MWM_HYQ_{name}A vdd paccn_hyq n_wm_hyq_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYQ_{name}B n_wm_hyq_{name}_a {'hm_store_hyq_' + name if err_sign == 'rp' else 'hp_store_hyq_' + name} n_wm_hyq_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYQ_{name}C n_wm_hyq_{name}_b {wm_restored} n_wm_hyq_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYQ_{name}D n_wm_hyq_{name}_c {wm_error} wm_hyq_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
"""
        )
        hybrid_product_prints.extend(
            [f"v(hp_store_hyq_{name})", f"v(hm_store_hyq_{name})", f"v(wp_hyq_{name})", f"v(wm_hyq_{name})"]
        )

    hybrid_product_deck = f"""
* Hybrid restored writer sampled-activation four-quadrant product routing.
* Each matched writer copy samples a+ and a- activation gates through MOS pass
* gates, then combines one activation sign with either the stored r+ or r-
* hidden-error sign.  The selected branch must satisfy
*   W+ <- a+e+ + a-e-
*   W- <- a+e- + a-e+
* while the inactive product branch stays quiet.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CSTORE=10p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPSAMP_HYQ psamp_hyq 0 PWL(0 0 1.55u 0 1.57u 1.8 1.85u 1.8 1.87u 0 3.2u 0)
VPSAMPN_HYQ psampn_hyq 0 PWL(0 1.8 1.55u 1.8 1.57u 0 1.85u 0 1.87u 1.8 3.2u 1.8)
VPACC_HYQ paccn_hyq 0 PULSE(1.8 0 2.10u 20n 20n 0.32u 5.0u)

VZPP_HYQ zpp_hyq 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYQ zmm_hyq 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYQ zpm_hyq 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYQ zmp_hyq 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYQ hpp_hyq hpp_hyq vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYQ hpm_hyq hpm_hyq vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYQ hpp_hyq zpp_hyq tailp_hyq 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYQ hpm_hyq zmm_hyq tailp_hyq 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYQ tailp_hyq vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYQ hmp_hyq hmp_hyq vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYQ hmm_hyq hmm_hyq vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYQ hmp_hyq zpm_hyq tailm_hyq 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYQ hmm_hyq zmp_hyq tailm_hyq 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYQ tailm_hyq vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYQ cdp_rp_hyq 0 {{CERR}} IC=1.04
CDM_RP_HYQ cdm_rp_hyq 0 {{CERR}} IC=1.04
CDP_RM_HYQ cdp_rm_hyq 0 {{CERR}} IC=1.04
CDM_RM_HYQ cdm_rm_hyq 0 {{CERR}} IC=1.04
RDP_RP_HYQ cdp_rp_hyq 0 50G
RDM_RP_HYQ cdm_rp_hyq 0 50G
RDP_RM_HYQ cdp_rm_hyq 0 50G
RDM_RM_HYQ cdm_rm_hyq 0 50G
{sign_store_path("hpm_hyq", "rp", "cdp_rp_hyq", "hyqrp1")}
{sign_store_path("hmp_hyq", "rp", "cdp_rp_hyq", "hyqrp2")}
{sign_store_path("hpp_hyq", "rp", "cdm_rp_hyq", "hyqrp3")}
{sign_store_path("hmm_hyq", "rp", "cdm_rp_hyq", "hyqrp4")}
{sign_store_path("hpp_hyq", "rm", "cdp_rm_hyq", "hyqrm1")}
{sign_store_path("hmm_hyq", "rm", "cdp_rm_hyq", "hyqrm2")}
{sign_store_path("hpm_hyq", "rm", "cdm_rm_hyq", "hyqrm3")}
{sign_store_path("hmp_hyq", "rm", "cdm_rm_hyq", "hyqrm4")}

MPRP_CDP_HYQ rgp_rp_hyq cdp_rp_hyq vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYQ rgp_rp_hyq cdp_rp_hyq 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYQ rgm_rp_hyq cdm_rp_hyq vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYQ rgm_rp_hyq cdm_rp_hyq 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDP_HYQ rgp_rm_hyq cdp_rm_hyq vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDP_HYQ rgp_rm_hyq cdp_rm_hyq 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDM_HYQ rgm_rm_hyq cdm_rm_hyq vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDM_HYQ rgm_rm_hyq cdm_rm_hyq 0 0 NMOS L={{LCH}} W={{WRESTN}}

{''.join(hybrid_product_devices)}

.control
set noaskquit
tran 5n 3.15u uic
wrdata mos_hidden_writer_restored_gate_hybrid_product.dat {' '.join(hybrid_product_prints)} v(psamp_hyq) v(paccn_hyq)
quit
.endc
.end
"""
    hybrid_product_data = run_ngspice(
        hybrid_product_deck,
        "mos_hidden_writer_restored_gate_hybrid_product",
    )
    hyqt, hyq_cols = load_wrdata(hybrid_product_data, len(hybrid_product_prints) + 2)

    def hyqat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyqt - time_s))])

    hyq_hidden_pos = hyq_cols[0] - hyq_cols[1]
    hyq_hidden_neg = hyq_cols[2] - hyq_cols[3]
    hyq_rp_selected_gate = hyq_cols[4]
    hyq_rp_complement_gate = hyq_cols[5]
    hyq_rm_complement_gate = hyq_cols[6]
    hyq_rm_selected_gate = hyq_cols[7]
    hyq_signed = []
    hyq_wp_step = []
    hyq_wm_step = []
    hyq_hp_final = []
    hyq_hm_final = []
    for idx, (_name, _label, _hplus_src, _hminus_src, _err_sign, _expected_sign) in enumerate(hybrid_product_cases):
        base = 8 + 4 * idx
        hp = hyq_cols[base]
        hm = hyq_cols[base + 1]
        wp = hyq_cols[base + 2]
        wm = hyq_cols[base + 3]
        hyq_hp_final.append(hyqat(2.00e-6, hp))
        hyq_hm_final.append(hyqat(2.00e-6, hm))
        hyq_signed.append(hyqat(2.70e-6, wp - wm))
        hyq_wp_step.append(hyqat(2.70e-6, wp) - hyqat(2.00e-6, wp))
        hyq_wm_step.append(hyqat(2.70e-6, wm) - hyqat(2.00e-6, wm))
    hyq_signed = np.array(hyq_signed)
    hyq_wp_step = np.array(hyq_wp_step)
    hyq_wm_step = np.array(hyq_wm_step)
    hyq_hp_final = np.array(hyq_hp_final)
    hyq_hm_final = np.array(hyq_hm_final)
    hyq_expected_sign = np.array([expected_sign for *_rest, expected_sign in hybrid_product_cases])
    hyq_abs = np.abs(hyq_signed)
    require(hyqat(1.35e-6, hyq_hidden_pos) > 0.07, "hybrid product r+ store should be positive")
    require(hyqat(1.35e-6, hyq_hidden_neg) < -0.07, "hybrid product r- store should be negative")
    require(abs(hyqat(1.35e-6, hyq_hidden_pos + hyq_hidden_neg)) < 0.003, "hybrid product hidden stores should be symmetric")
    require(hyqat(1.45e-6, hyq_rp_selected_gate) < 0.30, "hybrid product r+ selected gate should be low")
    require(hyqat(1.45e-6, hyq_rp_complement_gate) > 1.60, "hybrid product r+ complement gate should be high")
    require(hyqat(1.45e-6, hyq_rm_selected_gate) < 0.30, "hybrid product r- selected gate should be low")
    require(hyqat(1.45e-6, hyq_rm_complement_gate) > 1.60, "hybrid product r- complement gate should be high")
    require(np.all(np.sign(hyq_signed) == hyq_expected_sign), "hybrid product signed updates should match activation/error product signs")
    require(np.all(hyq_abs > 0.018), "hybrid product quadrants should produce useful signed updates")
    require(np.max(hyq_abs) - np.min(hyq_abs) < 0.004, "hybrid product quadrant magnitudes should agree")
    require(np.max(np.abs(hyq_signed - (hyq_wp_step - hyq_wm_step))) < 5e-4, "hybrid product signed step should match selected rail motion")
    require(np.max(np.minimum(hyq_wp_step, hyq_wm_step)) < 5e-4, "hybrid product inactive rail should stay suppressed")
    require(np.all(np.abs(hyq_hp_final[:2] - 0.92) < 0.010), "hybrid product a+ cases should sample active a+")
    require(np.all(hyq_hm_final[:2] > 1.35), "hybrid product a+ cases should leave a- inactive")
    require(np.all(hyq_hp_final[2:] > 1.35), "hybrid product a- cases should leave a+ inactive")
    require(np.all(np.abs(hyq_hm_final[2:] - 0.92) < 0.010), "hybrid product a- cases should sample active a-")

    hyq_fig, hyq_axes = plt.subplots(3, 1, figsize=(7.2, 7.2), gridspec_kw={"height_ratios": [1.05, 1.0, 1.0]})
    hyq_axes[0].plot(1e6 * hyqt, hyq_hidden_pos, label="stored $e^+$")
    hyq_axes[0].plot(1e6 * hyqt, hyq_hidden_neg, label="stored $e^-$")
    hyq_axes[0].plot(1e6 * hyqt, hyq_rp_selected_gate, label="$e^+$ selected gate")
    hyq_axes[0].plot(1e6 * hyqt, hyq_rm_selected_gate, label="$e^-$ selected gate")
    hyq_axes[0].set_xlim(0.35, 2.75)
    hyq_axes[0].set_ylabel("voltage (V)")
    hyq_axes[0].set_title("Hybrid product router stores both hidden-error signs")
    hyq_axes[0].grid(True, alpha=0.25)
    hyq_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyq_x = np.arange(len(hybrid_product_cases))
    hyq_labels = [label for _name, label, *_rest in hybrid_product_cases]
    hyq_axes[1].plot(hyq_x, hyq_hp_final, "o-", label="sampled $a^+$ gate")
    hyq_axes[1].plot(hyq_x, hyq_hm_final, "s-", label="sampled $a^-$ gate")
    hyq_axes[1].axhline(0.92, color="0.35", linewidth=0.8, linestyle=":", label="active-low")
    hyq_axes[1].axhline(1.45, color="0.65", linewidth=0.8, linestyle=":", label="inactive-high")
    hyq_axes[1].set_xticks(hyq_x)
    hyq_axes[1].set_xticklabels(hyq_labels)
    hyq_axes[1].set_ylabel("activation gate (V)")
    hyq_axes[1].set_title("MOS pass gates sample exactly one activation sign per quadrant")
    hyq_axes[1].grid(True, alpha=0.25)
    hyq_axes[1].legend(loc="center right", ncol=2, fontsize="small")
    hyq_axes[2].bar(hyq_x - 0.18, 1e3 * hyq_wp_step, width=0.36, label="$W^+$ step")
    hyq_axes[2].bar(hyq_x + 0.18, -1e3 * hyq_wm_step, width=0.36, label="$-W^-$ step")
    hyq_axes[2].plot(hyq_x, 1e3 * hyq_signed, "ko-", label="$W^+ - W^-$")
    hyq_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyq_axes[2].set_xticks(hyq_x)
    hyq_axes[2].set_xticklabels(hyq_labels)
    hyq_axes[2].set_ylabel("writer step (mV)")
    hyq_axes[2].set_title("signed update follows the four-quadrant product rule")
    hyq_axes[2].grid(True, axis="y", alpha=0.25)
    hyq_axes[2].legend(loc="upper left", ncol=3, fontsize="small")
    hyq_fig.tight_layout()
    save_plot(hyq_fig, "mos_hidden_writer_restored_gate_hybrid_product_ngspice")

    hybrid_bias_deck = f"""
* Hybrid restored writer local-bias update characterization.
* The local-feature cell also needs db <- e, without an activation operand.
* This deck stores matched e+ and e- hidden-error rails, restores the selected
* branch gates, and writes persistent B+/B- bias capacitors through MOS-only
* phase/restored-select/analog-error stacks.
{COMMON_MODELS}
.param CERR=10p CBIAS=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPACC_HYB paccn_hyb 0 PULSE(1.8 0 1.70u 20n 20n 0.32u 5.0u)

VZPP_HYB zpp_hyb 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYB zmm_hyb 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYB zpm_hyb 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYB zmp_hyb 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYB hpp_hyb hpp_hyb vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYB hpm_hyb hpm_hyb vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYB hpp_hyb zpp_hyb tailp_hyb 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYB hpm_hyb zmm_hyb tailp_hyb 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYB tailp_hyb vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYB hmp_hyb hmp_hyb vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYB hmm_hyb hmm_hyb vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYB hmp_hyb zpm_hyb tailm_hyb 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYB hmm_hyb zmp_hyb tailm_hyb 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYB tailm_hyb vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYB cdp_rp_hyb 0 {{CERR}} IC=1.04
CDM_RP_HYB cdm_rp_hyb 0 {{CERR}} IC=1.04
CDP_RM_HYB cdp_rm_hyb 0 {{CERR}} IC=1.04
CDM_RM_HYB cdm_rm_hyb 0 {{CERR}} IC=1.04
RDP_RP_HYB cdp_rp_hyb 0 50G
RDM_RP_HYB cdm_rp_hyb 0 50G
RDP_RM_HYB cdp_rm_hyb 0 50G
RDM_RM_HYB cdm_rm_hyb 0 50G
{sign_store_path("hpm_hyb", "rp", "cdp_rp_hyb", "hybrp1")}
{sign_store_path("hmp_hyb", "rp", "cdp_rp_hyb", "hybrp2")}
{sign_store_path("hpp_hyb", "rp", "cdm_rp_hyb", "hybrp3")}
{sign_store_path("hmm_hyb", "rp", "cdm_rp_hyb", "hybrp4")}
{sign_store_path("hpp_hyb", "rm", "cdp_rm_hyb", "hybrm1")}
{sign_store_path("hmm_hyb", "rm", "cdp_rm_hyb", "hybrm2")}
{sign_store_path("hpm_hyb", "rm", "cdm_rm_hyb", "hybrm3")}
{sign_store_path("hmp_hyb", "rm", "cdm_rm_hyb", "hybrm4")}

MPRP_CDP_HYB rgp_rp_hyb cdp_rp_hyb vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYB rgp_rp_hyb cdp_rp_hyb 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYB rgm_rp_hyb cdm_rp_hyb vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYB rgm_rp_hyb cdm_rp_hyb 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDP_HYB rgp_rm_hyb cdp_rm_hyb vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDP_HYB rgp_rm_hyb cdp_rm_hyb 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDM_HYB rgm_rm_hyb cdm_rm_hyb vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDM_HYB rgm_rm_hyb cdm_rm_hyb 0 0 NMOS L={{LCH}} W={{WRESTN}}

CBP_RP_HYB bp_rp_hyb 0 {{CBIAS}} IC=0.85
CBM_RP_HYB bm_rp_hyb 0 {{CBIAS}} IC=0.85
MBP_RP_HYBA vdd paccn_hyb n_bp_rp_hyb_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_RP_HYBB n_bp_rp_hyb_a rgp_rp_hyb n_bp_rp_hyb_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_RP_HYBC n_bp_rp_hyb_b cdm_rp_hyb bp_rp_hyb vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_RP_HYBA vdd paccn_hyb n_bm_rp_hyb_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_RP_HYBB n_bm_rp_hyb_a rgm_rp_hyb n_bm_rp_hyb_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_RP_HYBC n_bm_rp_hyb_b cdp_rp_hyb bm_rp_hyb vdd PMOS L={{LCH}} W={{WWRITE}}

CBP_RM_HYB bp_rm_hyb 0 {{CBIAS}} IC=0.85
CBM_RM_HYB bm_rm_hyb 0 {{CBIAS}} IC=0.85
MBP_RM_HYBA vdd paccn_hyb n_bp_rm_hyb_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_RM_HYBB n_bp_rm_hyb_a rgp_rm_hyb n_bp_rm_hyb_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_RM_HYBC n_bp_rm_hyb_b cdm_rm_hyb bp_rm_hyb vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_RM_HYBA vdd paccn_hyb n_bm_rm_hyb_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_RM_HYBB n_bm_rm_hyb_a rgm_rm_hyb n_bm_rm_hyb_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_RM_HYBC n_bm_rm_hyb_b cdp_rm_hyb bm_rm_hyb vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 5n 3.0u uic
wrdata mos_hidden_writer_restored_gate_hybrid_bias.dat v(cdp_rp_hyb) v(cdm_rp_hyb) v(cdp_rm_hyb) v(cdm_rm_hyb) v(rgp_rp_hyb) v(rgm_rp_hyb) v(rgp_rm_hyb) v(rgm_rm_hyb) v(bp_rp_hyb) v(bm_rp_hyb) v(bp_rm_hyb) v(bm_rm_hyb) v(paccn_hyb)
quit
.endc
.end
"""
    hybrid_bias_data = run_ngspice(
        hybrid_bias_deck,
        "mos_hidden_writer_restored_gate_hybrid_bias",
    )
    hybt, hyb_cols = load_wrdata(hybrid_bias_data, 13)

    def hybat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hybt - time_s))])

    hyb_hidden_pos = hyb_cols[0] - hyb_cols[1]
    hyb_hidden_neg = hyb_cols[2] - hyb_cols[3]
    hyb_rp_selected_gate = hyb_cols[4]
    hyb_rp_complement_gate = hyb_cols[5]
    hyb_rm_complement_gate = hyb_cols[6]
    hyb_rm_selected_gate = hyb_cols[7]
    hyb_bp_rp = hyb_cols[8]
    hyb_bm_rp = hyb_cols[9]
    hyb_bp_rm = hyb_cols[10]
    hyb_bm_rm = hyb_cols[11]
    hyb_pos_signed = hyb_bp_rp - hyb_bm_rp
    hyb_neg_signed = hyb_bp_rm - hyb_bm_rm
    hyb_pos_bp_step = hybat(2.55e-6, hyb_bp_rp) - hybat(1.60e-6, hyb_bp_rp)
    hyb_pos_bm_step = hybat(2.55e-6, hyb_bm_rp) - hybat(1.60e-6, hyb_bm_rp)
    hyb_neg_bp_step = hybat(2.55e-6, hyb_bp_rm) - hybat(1.60e-6, hyb_bp_rm)
    hyb_neg_bm_step = hybat(2.55e-6, hyb_bm_rm) - hybat(1.60e-6, hyb_bm_rm)
    hyb_pos_final = hybat(2.55e-6, hyb_pos_signed)
    hyb_neg_final = hybat(2.55e-6, hyb_neg_signed)
    require(hybat(1.35e-6, hyb_hidden_pos) > 0.07, "hybrid bias e+ store should be positive")
    require(hybat(1.35e-6, hyb_hidden_neg) < -0.07, "hybrid bias e- store should be negative")
    require(abs(hybat(1.35e-6, hyb_hidden_pos + hyb_hidden_neg)) < 0.003, "hybrid bias hidden stores should be symmetric")
    require(hybat(1.45e-6, hyb_rp_selected_gate) < 0.30, "hybrid bias e+ selected gate should be low")
    require(hybat(1.45e-6, hyb_rp_complement_gate) > 1.60, "hybrid bias e+ complement gate should be high")
    require(hybat(1.45e-6, hyb_rm_selected_gate) < 0.30, "hybrid bias e- selected gate should be low")
    require(hybat(1.45e-6, hyb_rm_complement_gate) > 1.60, "hybrid bias e- complement gate should be high")
    require(hyb_pos_final > 0.020, "hybrid bias e+ should write positive bias differential")
    require(hyb_neg_final < -0.020, "hybrid bias e- should write negative bias differential")
    require(abs(hyb_pos_final + hyb_neg_final) < 0.003, "hybrid bias signs should write symmetric magnitudes")
    require(hyb_pos_bp_step > 0.020 and hyb_neg_bm_step > 0.020, "hybrid bias selected rails should move")
    require(max(hyb_pos_bm_step, hyb_neg_bp_step) < 5e-4, "hybrid bias inactive rails should stay suppressed")
    require(abs(hybat(2.90e-6, hyb_pos_signed) - hyb_pos_final) < 5e-4, "hybrid bias positive write should hold")
    require(abs(hybat(2.90e-6, hyb_neg_signed) - hyb_neg_final) < 5e-4, "hybrid bias negative write should hold")

    hyb_fig, hyb_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hyb_axes[0].plot(1e6 * hybt, hyb_hidden_pos, label="stored $e^+$")
    hyb_axes[0].plot(1e6 * hybt, hyb_hidden_neg, label="stored $e^-$")
    hyb_axes[0].plot(1e6 * hybt, hyb_rp_selected_gate, label="$e^+$ selected gate")
    hyb_axes[0].plot(1e6 * hybt, hyb_rm_selected_gate, label="$e^-$ selected gate")
    hyb_axes[0].plot(1e6 * hybt, hyb_cols[12] / 20.0, color="0.4", alpha=0.35, label="$pacc_n/20$")
    hyb_axes[0].set_ylabel("voltage (V)")
    hyb_axes[0].set_title("Hybrid bias writer stores both error signs")
    hyb_axes[0].grid(True, alpha=0.25)
    hyb_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyb_axes[1].plot(1e6 * hybt, 1e3 * hyb_pos_signed, label="$B^+ - B^-$ from $e^+$")
    hyb_axes[1].plot(1e6 * hybt, 1e3 * hyb_neg_signed, label="$B^+ - B^-$ from $e^-$")
    hyb_axes[1].plot(1e6 * hybt, 1e3 * (hyb_bm_rp - hybat(1.60e-6, hyb_bm_rp)), "--", label="inactive $B^-$ during $e^+$")
    hyb_axes[1].plot(1e6 * hybt, 1e3 * (hyb_bp_rm - hybat(1.60e-6, hyb_bp_rm)), "--", label="inactive $B^+$ during $e^-$")
    hyb_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyb_axes[1].set_xlabel("time (us)")
    hyb_axes[1].set_ylabel("bias step (mV)")
    hyb_axes[1].set_title("local bias update follows $\\Delta b \\propto e$")
    hyb_axes[1].grid(True, alpha=0.25)
    hyb_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    hyb_fig.tight_layout()
    save_plot(hyb_fig, "mos_hidden_writer_restored_gate_hybrid_bias_ngspice")

    hybrid_cell_update_deck = f"""
* Hybrid restored writer one-sample local-feature update/readback check.
* This combines the sampled activation operand, stored/restored e+ error rail,
* activation-gated W+/W- product write, non-activation-gated B+/B- bias write,
* and transistor synapse readback of both persistent states.  There are no
* behavioral update sources or post-sample capacitor writes in this deck.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CBIAS=500p CSTORE=10p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD_HYU pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP_HYU rp_hyu 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VACT_SRC_HYU hsrc_hyu 0 PWL(0 1.45 1.35u 1.45 1.37u 0.92 1.82u 0.92 1.84u 1.45 3.8u 1.45)
VPSAMP_HYU psamp_hyu 0 PWL(0 0 1.42u 0 1.44u 1.8 1.76u 1.8 1.78u 0 3.8u 0)
VPSAMPN_HYU psampn_hyu 0 PWL(0 1.8 1.42u 1.8 1.44u 0 1.76u 0 1.78u 1.8 3.8u 1.8)
VPACC_HYU paccn_hyu 0 PWL(0 1.8 2.05u 1.8 2.07u 0 2.39u 0 2.41u 1.8 3.8u 1.8)

VZPP_HYU zpp_hyu 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYU zmm_hyu 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYU zpm_hyu 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYU zmp_hyu 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYU hpp_hyu hpp_hyu vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYU hpm_hyu hpm_hyu vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYU hpp_hyu zpp_hyu tailp_hyu 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYU hpm_hyu zmm_hyu tailp_hyu 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYU tailp_hyu vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYU hmp_hyu hmp_hyu vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYU hmm_hyu hmm_hyu vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYU hmp_hyu zpm_hyu tailm_hyu 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYU hmm_hyu zmp_hyu tailm_hyu 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYU tailm_hyu vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYU cdp_rp_hyu 0 {{CERR}} IC=1.04
CDM_RP_HYU cdm_rp_hyu 0 {{CERR}} IC=1.04
RDP_RP_HYU cdp_rp_hyu 0 50G
RDM_RP_HYU cdm_rp_hyu 0 50G
{sign_store_path("hpm_hyu", "rp_hyu", "cdp_rp_hyu", "hyurp1")}
{sign_store_path("hmp_hyu", "rp_hyu", "cdp_rp_hyu", "hyurp2")}
{sign_store_path("hpp_hyu", "rp_hyu", "cdm_rp_hyu", "hyurp3")}
{sign_store_path("hmm_hyu", "rp_hyu", "cdm_rp_hyu", "hyurp4")}

MPRP_CDP_HYU rgp_rp_hyu cdp_rp_hyu vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYU rgp_rp_hyu cdp_rp_hyu 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYU rgm_rp_hyu cdm_rp_hyu vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYU rgm_rp_hyu cdm_rp_hyu 0 0 NMOS L={{LCH}} W={{WRESTN}}

CHM_HYU hm_store_hyu 0 {{CSTORE}} IC=1.45
RHM_HYU hm_store_hyu 0 50G
MSACTN_HYU hsrc_hyu psamp_hyu hm_store_hyu 0 NMOS L={{LCH}} W={{WSW}}
MSACTP_HYU hsrc_hyu psampn_hyu hm_store_hyu vdd PMOS L={{LCH}} W={{WSW}}

CWP_HYU wp_hyu 0 {{CWRITE}} IC=0.85
CWM_HYU wm_hyu 0 {{CWRITE}} IC=0.85
CBP_HYU bp_hyu 0 {{CBIAS}} IC=0.85
CBM_HYU bm_hyu 0 {{CBIAS}} IC=0.85

* Weight product: W+ <- a+e+; W- complement should remain quiet.
MWP_HYU_A vdd paccn_hyu n_wp_hyu_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYU_B n_wp_hyu_a hm_store_hyu n_wp_hyu_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYU_C n_wp_hyu_b rgp_rp_hyu n_wp_hyu_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYU_D n_wp_hyu_c cdm_rp_hyu wp_hyu vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYU_A vdd paccn_hyu n_wm_hyu_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYU_B n_wm_hyu_a hm_store_hyu n_wm_hyu_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYU_C n_wm_hyu_b rgm_rp_hyu n_wm_hyu_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYU_D n_wm_hyu_c cdp_rp_hyu wm_hyu vdd PMOS L={{LCH}} W={{WWRITE}}

* Bias update: B+ <- e+; B- complement should remain quiet.
MBP_HYU_A vdd paccn_hyu n_bp_hyu_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYU_B n_bp_hyu_a rgp_rp_hyu n_bp_hyu_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYU_C n_bp_hyu_b cdm_rp_hyu bp_hyu vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYU_A vdd paccn_hyu n_bm_hyu_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYU_B n_bm_hyu_a rgm_rp_hyu n_bm_hyu_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYU_C n_bm_hyu_b cdp_rp_hyu bm_hyu vdd PMOS L={{LCH}} W={{WWRITE}}

* Continuous transistor readback from weight and bias capacitor pairs.
VXP_HYU xp_hyu 0 1.15
VXM_HYU xm_hyu 0 0.65

VZPP_W_HYU zpp_w_hyu 0 1.8
VZMP_W_HYU zmp_w_hyu 0 1.8
VZPN_W_HYU zpn_w_hyu 0 1.8
VZMN_W_HYU zmn_w_hyu 0 1.8
MPP_W_HYU zpp_w_hyu xp_hyu tail_wp_hyu 0 NMOS L={{LCH}} W={{WN}}
MPM_W_HYU zmp_w_hyu xm_hyu tail_wp_hyu 0 NMOS L={{LCH}} W={{WN}}
MTP_W_HYU tail_wp_hyu wp_hyu 0 0 NMOS L={{LCH}} W=12u
MNP_W_HYU zpn_w_hyu xm_hyu tail_wm_hyu 0 NMOS L={{LCH}} W={{WN}}
MNM_W_HYU zmn_w_hyu xp_hyu tail_wm_hyu 0 NMOS L={{LCH}} W={{WN}}
MTN_W_HYU tail_wm_hyu wm_hyu 0 0 NMOS L={{LCH}} W=12u

VZPP_B_HYU zpp_b_hyu 0 1.8
VZMP_B_HYU zmp_b_hyu 0 1.8
VZPN_B_HYU zpn_b_hyu 0 1.8
VZMN_B_HYU zmn_b_hyu 0 1.8
MPP_B_HYU zpp_b_hyu xp_hyu tail_bp_hyu 0 NMOS L={{LCH}} W={{WN}}
MPM_B_HYU zmp_b_hyu xm_hyu tail_bp_hyu 0 NMOS L={{LCH}} W={{WN}}
MTP_B_HYU tail_bp_hyu bp_hyu 0 0 NMOS L={{LCH}} W=12u
MNP_B_HYU zpn_b_hyu xm_hyu tail_bm_hyu 0 NMOS L={{LCH}} W={{WN}}
MNM_B_HYU zmn_b_hyu xp_hyu tail_bm_hyu 0 NMOS L={{LCH}} W={{WN}}
MTN_B_HYU tail_bm_hyu bm_hyu 0 0 NMOS L={{LCH}} W=12u

.control
set noaskquit
tran 5n 3.75u uic
wrdata mos_hidden_writer_restored_gate_hybrid_cell_update.dat v(cdp_rp_hyu) v(cdm_rp_hyu) v(rgp_rp_hyu) v(rgm_rp_hyu) v(hsrc_hyu) v(hm_store_hyu) v(wp_hyu) v(wm_hyu) v(bp_hyu) v(bm_hyu) i(VZPP_W_HYU) i(VZMP_W_HYU) i(VZPN_W_HYU) i(VZMN_W_HYU) i(VZPP_B_HYU) i(VZMP_B_HYU) i(VZPN_B_HYU) i(VZMN_B_HYU) v(psamp_hyu) v(paccn_hyu)
quit
.endc
.end
"""
    hybrid_cell_update_data = run_ngspice(
        hybrid_cell_update_deck,
        "mos_hidden_writer_restored_gate_hybrid_cell_update",
    )
    hyut, hyu_cols = load_wrdata(hybrid_cell_update_data, 20)

    def hyuat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyut - time_s))])

    hyu_hidden = hyu_cols[0] - hyu_cols[1]
    hyu_selected_gate = hyu_cols[2]
    hyu_complement_gate = hyu_cols[3]
    hyu_hsrc = hyu_cols[4]
    hyu_hcap = hyu_cols[5]
    hyu_wp = hyu_cols[6]
    hyu_wm = hyu_cols[7]
    hyu_bp = hyu_cols[8]
    hyu_bm = hyu_cols[9]
    hyu_weight = hyu_wp - hyu_wm
    hyu_bias = hyu_bp - hyu_bm
    hyu_weight_read = (hyu_cols[11] - hyu_cols[10]) + (hyu_cols[13] - hyu_cols[12])
    hyu_bias_read = (hyu_cols[15] - hyu_cols[14]) + (hyu_cols[17] - hyu_cols[16])
    hyu_weight_final = hyuat(2.75e-6, hyu_weight)
    hyu_bias_final = hyuat(2.75e-6, hyu_bias)
    hyu_weight_read_step = hyuat(2.75e-6, hyu_weight_read) - hyuat(1.95e-6, hyu_weight_read)
    hyu_bias_read_step = hyuat(2.75e-6, hyu_bias_read) - hyuat(1.95e-6, hyu_bias_read)
    require(hyuat(1.35e-6, hyu_hidden) > 0.07, "hybrid cell update should store positive error")
    require(hyuat(1.45e-6, hyu_selected_gate) < 0.30, "hybrid cell update selected restored gate should be low")
    require(hyuat(1.45e-6, hyu_complement_gate) > 1.60, "hybrid cell update complement restored gate should be high")
    require(abs(hyuat(1.85e-6, hyu_hcap) - 0.92) < 0.012, "hybrid cell update should sample active activation gate")
    require(hyu_weight_final > 0.020, "integrated cell update should write positive weight differential")
    require(hyu_bias_final > 0.030, "integrated cell update should write positive bias differential")
    require(hyuat(2.75e-6, hyu_wm) - hyuat(1.95e-6, hyu_wm) < 5e-4, "integrated cell update should keep W- quiet")
    require(hyuat(2.75e-6, hyu_bm) - hyuat(1.95e-6, hyu_bm) < 5e-4, "integrated cell update should keep B- quiet")
    require(hyu_weight_read_step > 3e-6, "weight readback current should increase after weight write")
    require(hyu_bias_read_step > 5e-6, "bias readback current should increase after bias write")
    require(abs(hyuat(3.55e-6, hyu_weight) - hyu_weight_final) < 5e-4, "integrated weight state should hold after pacc")
    require(abs(hyuat(3.55e-6, hyu_bias) - hyu_bias_final) < 5e-4, "integrated bias state should hold after pacc")

    hyu_fig, hyu_axes = plt.subplots(3, 1, figsize=(7.2, 7.4), gridspec_kw={"height_ratios": [1.05, 1.0, 1.0]})
    hyu_axes[0].plot(1e6 * hyut, hyu_hidden, label="stored $e^+$")
    hyu_axes[0].plot(1e6 * hyut, hyu_selected_gate, label="selected error gate")
    hyu_axes[0].plot(1e6 * hyut, hyu_complement_gate, label="complement error gate")
    hyu_axes[0].plot(1e6 * hyut, hyu_hsrc, color="0.55", alpha=0.65, label="activation source")
    hyu_axes[0].plot(1e6 * hyut, hyu_hcap, color="0.15", linestyle="--", label="sampled activation gate")
    hyu_axes[0].set_ylabel("voltage (V)")
    hyu_axes[0].set_title("One MOS sample cycle stores error and activation operands")
    hyu_axes[0].grid(True, alpha=0.25)
    hyu_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyu_axes[1].plot(1e6 * hyut, 1e3 * (hyu_wp - hyuat(1.95e-6, hyu_wp)), label="$W^+$")
    hyu_axes[1].plot(1e6 * hyut, 1e3 * (hyu_wm - hyuat(1.95e-6, hyu_wm)), label="$W^-$")
    hyu_axes[1].plot(1e6 * hyut, 1e3 * (hyu_bp - hyuat(1.95e-6, hyu_bp)), label="$B^+$")
    hyu_axes[1].plot(1e6 * hyut, 1e3 * (hyu_bm - hyuat(1.95e-6, hyu_bm)), label="$B^-$")
    hyu_axes[1].plot(1e6 * hyut, hyu_cols[19] / 20.0, color="0.4", alpha=0.25, label="$pacc_n/20$")
    hyu_axes[1].set_ylabel("state step (mV)")
    hyu_axes[1].set_title("Same pacc pulse writes activation-gated weight and ungated bias")
    hyu_axes[1].grid(True, alpha=0.25)
    hyu_axes[1].legend(loc="upper left", ncol=3, fontsize="small")
    hyu_axes[2].plot(1e6 * hyut, 1e6 * (hyu_weight_read - hyuat(1.95e-6, hyu_weight_read)), label="weight read current step")
    hyu_axes[2].plot(1e6 * hyut, 1e6 * (hyu_bias_read - hyuat(1.95e-6, hyu_bias_read)), label="bias read current step")
    hyu_axes[2].plot(1e6 * hyut, 1e3 * hyu_weight, "--", label="$W^+ - W^-$")
    hyu_axes[2].plot(1e6 * hyut, 1e3 * hyu_bias, "--", label="$B^+ - B^-$")
    hyu_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyu_axes[2].set_xlabel("time (us)")
    hyu_axes[2].set_ylabel("uA / mV")
    hyu_axes[2].set_title("Persistent states immediately read back through NMOS slices")
    hyu_axes[2].grid(True, alpha=0.25)
    hyu_axes[2].legend(loc="upper left", ncol=2, fontsize="small")
    hyu_fig.tight_layout()
    save_plot(hyu_fig, "mos_hidden_writer_restored_gate_hybrid_cell_update_ngspice")

    hybrid_cell_quadrant_devices = []
    hybrid_cell_quadrant_prints = [
        "v(cdp_rp_hyz)",
        "v(cdm_rp_hyz)",
        "v(cdp_rm_hyz)",
        "v(cdm_rm_hyz)",
        "v(rgp_rp_hyz)",
        "v(rgm_rp_hyz)",
        "v(rgp_rm_hyz)",
        "v(rgm_rm_hyz)",
    ]
    for name, _label, hplus_src, hminus_src, err_sign, _expected_sign in hybrid_product_cases:
        if err_sign == "rp":
            wp_restored = "rgp_rp_hyz"
            wp_error = "cdm_rp_hyz"
            wm_restored = "rgp_rp_hyz"
            wm_error = "cdm_rp_hyz"
            bp_restored = "rgp_rp_hyz"
            bp_error = "cdm_rp_hyz"
            bm_restored = "rgm_rp_hyz"
            bm_error = "cdp_rp_hyz"
        else:
            wp_restored = "rgm_rm_hyz"
            wp_error = "cdp_rm_hyz"
            wm_restored = "rgm_rm_hyz"
            wm_error = "cdp_rm_hyz"
            bp_restored = "rgp_rm_hyz"
            bp_error = "cdm_rm_hyz"
            bm_restored = "rgm_rm_hyz"
            bm_error = "cdp_rm_hyz"
        hybrid_cell_quadrant_devices.append(
            f"""
VHP_SRC_HYZ_{name} hp_src_hyz_{name} 0 {hplus_src:.3f}
VHM_SRC_HYZ_{name} hm_src_hyz_{name} 0 {hminus_src:.3f}
CHP_HYZ_{name} hp_store_hyz_{name} 0 {{CSTORE}} IC=1.45
CHM_HYZ_{name} hm_store_hyz_{name} 0 {{CSTORE}} IC=1.45
RHP_HYZ_{name} hp_store_hyz_{name} 0 50G
RHM_HYZ_{name} hm_store_hyz_{name} 0 50G
MSHPN_HYZ_{name} hp_src_hyz_{name} psamp_hyz hp_store_hyz_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSHPP_HYZ_{name} hp_src_hyz_{name} psampn_hyz hp_store_hyz_{name} vdd PMOS L={{LCH}} W={{WSW}}
MSHMN_HYZ_{name} hm_src_hyz_{name} psamp_hyz hm_store_hyz_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSHMP_HYZ_{name} hm_src_hyz_{name} psampn_hyz hm_store_hyz_{name} vdd PMOS L={{LCH}} W={{WSW}}

CWP_HYZ_{name} wp_hyz_{name} 0 {{CWRITE}} IC=0.85
CWM_HYZ_{name} wm_hyz_{name} 0 {{CWRITE}} IC=0.85
CBP_HYZ_{name} bp_hyz_{name} 0 {{CBIAS}} IC=0.85
CBM_HYZ_{name} bm_hyz_{name} 0 {{CBIAS}} IC=0.85

MWP_HYZ_{name}A vdd paccn_hyz n_wp_hyz_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYZ_{name}B n_wp_hyz_{name}_a {'hp_store_hyz_' + name if err_sign == 'rp' else 'hm_store_hyz_' + name} n_wp_hyz_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYZ_{name}C n_wp_hyz_{name}_b {wp_restored} n_wp_hyz_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYZ_{name}D n_wp_hyz_{name}_c {wp_error} wp_hyz_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYZ_{name}A vdd paccn_hyz n_wm_hyz_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYZ_{name}B n_wm_hyz_{name}_a {'hm_store_hyz_' + name if err_sign == 'rp' else 'hp_store_hyz_' + name} n_wm_hyz_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYZ_{name}C n_wm_hyz_{name}_b {wm_restored} n_wm_hyz_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYZ_{name}D n_wm_hyz_{name}_c {wm_error} wm_hyz_{name} vdd PMOS L={{LCH}} W={{WWRITE}}

MBP_HYZ_{name}A vdd paccn_hyz n_bp_hyz_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYZ_{name}B n_bp_hyz_{name}_a {bp_restored} n_bp_hyz_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYZ_{name}C n_bp_hyz_{name}_b {bp_error} bp_hyz_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYZ_{name}A vdd paccn_hyz n_bm_hyz_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYZ_{name}B n_bm_hyz_{name}_a {bm_restored} n_bm_hyz_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYZ_{name}C n_bm_hyz_{name}_b {bm_error} bm_hyz_{name} vdd PMOS L={{LCH}} W={{WWRITE}}

VXP_W_HYZ_{name} xp_w_hyz_{name} 0 1.15
VXM_W_HYZ_{name} xm_w_hyz_{name} 0 0.65
VZPP_W_HYZ_{name} zpp_w_hyz_{name} 0 1.8
VZMP_W_HYZ_{name} zmp_w_hyz_{name} 0 1.8
VZPN_W_HYZ_{name} zpn_w_hyz_{name} 0 1.8
VZMN_W_HYZ_{name} zmn_w_hyz_{name} 0 1.8
MPP_W_HYZ_{name} zpp_w_hyz_{name} xp_w_hyz_{name} tail_wp_hyz_{name} 0 NMOS L={{LCH}} W={{WN}}
MPM_W_HYZ_{name} zmp_w_hyz_{name} xm_w_hyz_{name} tail_wp_hyz_{name} 0 NMOS L={{LCH}} W={{WN}}
MTP_W_HYZ_{name} tail_wp_hyz_{name} wp_hyz_{name} 0 0 NMOS L={{LCH}} W=12u
MNP_W_HYZ_{name} zpn_w_hyz_{name} xm_w_hyz_{name} tail_wm_hyz_{name} 0 NMOS L={{LCH}} W={{WN}}
MNM_W_HYZ_{name} zmn_w_hyz_{name} xp_w_hyz_{name} tail_wm_hyz_{name} 0 NMOS L={{LCH}} W={{WN}}
MTN_W_HYZ_{name} tail_wm_hyz_{name} wm_hyz_{name} 0 0 NMOS L={{LCH}} W=12u

VXP_B_HYZ_{name} xp_b_hyz_{name} 0 1.15
VXM_B_HYZ_{name} xm_b_hyz_{name} 0 0.65
VZPP_B_HYZ_{name} zpp_b_hyz_{name} 0 1.8
VZMP_B_HYZ_{name} zmp_b_hyz_{name} 0 1.8
VZPN_B_HYZ_{name} zpn_b_hyz_{name} 0 1.8
VZMN_B_HYZ_{name} zmn_b_hyz_{name} 0 1.8
MPP_B_HYZ_{name} zpp_b_hyz_{name} xp_b_hyz_{name} tail_bp_hyz_{name} 0 NMOS L={{LCH}} W={{WN}}
MPM_B_HYZ_{name} zmp_b_hyz_{name} xm_b_hyz_{name} tail_bp_hyz_{name} 0 NMOS L={{LCH}} W={{WN}}
MTP_B_HYZ_{name} tail_bp_hyz_{name} bp_hyz_{name} 0 0 NMOS L={{LCH}} W=12u
MNP_B_HYZ_{name} zpn_b_hyz_{name} xm_b_hyz_{name} tail_bm_hyz_{name} 0 NMOS L={{LCH}} W={{WN}}
MNM_B_HYZ_{name} zmn_b_hyz_{name} xp_b_hyz_{name} tail_bm_hyz_{name} 0 NMOS L={{LCH}} W={{WN}}
MTN_B_HYZ_{name} tail_bm_hyz_{name} bm_hyz_{name} 0 0 NMOS L={{LCH}} W=12u
"""
        )
        hybrid_cell_quadrant_prints.extend(
            [
                f"v(hp_store_hyz_{name})",
                f"v(hm_store_hyz_{name})",
                f"v(wp_hyz_{name})",
                f"v(wm_hyz_{name})",
                f"v(bp_hyz_{name})",
                f"v(bm_hyz_{name})",
                f"i(VZPP_W_HYZ_{name})",
                f"i(VZMP_W_HYZ_{name})",
                f"i(VZPN_W_HYZ_{name})",
                f"i(VZMN_W_HYZ_{name})",
                f"i(VZPP_B_HYZ_{name})",
                f"i(VZMP_B_HYZ_{name})",
                f"i(VZPN_B_HYZ_{name})",
                f"i(VZMN_B_HYZ_{name})",
            ]
        )

    hybrid_cell_quadrant_deck = f"""
* Hybrid restored writer four-quadrant local-feature update/readback check.
* Each case samples a+/a- activation gates, uses stored/restored e+/e- rails,
* writes both W+/W- and B+/B- persistent states, and reads both states through
* NMOS synapse slices.  This is the integrated family-completion version of
* the one-quadrant cell-update plot.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CBIAS=500p CSTORE=10p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD_HYZ pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP_HYZ rp_hyz 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM_HYZ rm_hyz 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPSAMP_HYZ psamp_hyz 0 PWL(0 0 1.55u 0 1.57u 1.8 1.85u 1.8 1.87u 0 3.3u 0)
VPSAMPN_HYZ psampn_hyz 0 PWL(0 1.8 1.55u 1.8 1.57u 0 1.85u 0 1.87u 1.8 3.3u 1.8)
VPACC_HYZ paccn_hyz 0 PULSE(1.8 0 2.10u 20n 20n 0.32u 5.0u)

VZPP_HYZ zpp_hyz 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYZ zmm_hyz 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYZ zpm_hyz 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYZ zmp_hyz 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYZ hpp_hyz hpp_hyz vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYZ hpm_hyz hpm_hyz vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYZ hpp_hyz zpp_hyz tailp_hyz 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYZ hpm_hyz zmm_hyz tailp_hyz 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYZ tailp_hyz vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYZ hmp_hyz hmp_hyz vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYZ hmm_hyz hmm_hyz vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYZ hmp_hyz zpm_hyz tailm_hyz 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYZ hmm_hyz zmp_hyz tailm_hyz 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYZ tailm_hyz vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYZ cdp_rp_hyz 0 {{CERR}} IC=1.04
CDM_RP_HYZ cdm_rp_hyz 0 {{CERR}} IC=1.04
CDP_RM_HYZ cdp_rm_hyz 0 {{CERR}} IC=1.04
CDM_RM_HYZ cdm_rm_hyz 0 {{CERR}} IC=1.04
RDP_RP_HYZ cdp_rp_hyz 0 50G
RDM_RP_HYZ cdm_rp_hyz 0 50G
RDP_RM_HYZ cdp_rm_hyz 0 50G
RDM_RM_HYZ cdm_rm_hyz 0 50G
{sign_store_path("hpm_hyz", "rp_hyz", "cdp_rp_hyz", "hyzrp1")}
{sign_store_path("hmp_hyz", "rp_hyz", "cdp_rp_hyz", "hyzrp2")}
{sign_store_path("hpp_hyz", "rp_hyz", "cdm_rp_hyz", "hyzrp3")}
{sign_store_path("hmm_hyz", "rp_hyz", "cdm_rp_hyz", "hyzrp4")}
{sign_store_path("hpp_hyz", "rm_hyz", "cdp_rm_hyz", "hyzrm1")}
{sign_store_path("hmm_hyz", "rm_hyz", "cdp_rm_hyz", "hyzrm2")}
{sign_store_path("hpm_hyz", "rm_hyz", "cdm_rm_hyz", "hyzrm3")}
{sign_store_path("hmp_hyz", "rm_hyz", "cdm_rm_hyz", "hyzrm4")}

MPRP_CDP_HYZ rgp_rp_hyz cdp_rp_hyz vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYZ rgp_rp_hyz cdp_rp_hyz 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYZ rgm_rp_hyz cdm_rp_hyz vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYZ rgm_rp_hyz cdm_rp_hyz 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDP_HYZ rgp_rm_hyz cdp_rm_hyz vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDP_HYZ rgp_rm_hyz cdp_rm_hyz 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDM_HYZ rgm_rm_hyz cdm_rm_hyz vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDM_HYZ rgm_rm_hyz cdm_rm_hyz 0 0 NMOS L={{LCH}} W={{WRESTN}}

{''.join(hybrid_cell_quadrant_devices)}

.control
set noaskquit
tran 5n 3.25u uic
wrdata mos_hidden_writer_restored_gate_hybrid_cell_quadrants.dat {' '.join(hybrid_cell_quadrant_prints)} v(psamp_hyz) v(paccn_hyz)
quit
.endc
.end
"""
    hybrid_cell_quadrant_data = run_ngspice(
        hybrid_cell_quadrant_deck,
        "mos_hidden_writer_restored_gate_hybrid_cell_quadrants",
    )
    hyzt, hyz_cols = load_wrdata(hybrid_cell_quadrant_data, len(hybrid_cell_quadrant_prints) + 2)

    def hyzat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyzt - time_s))])

    hyz_hidden_pos = hyz_cols[0] - hyz_cols[1]
    hyz_hidden_neg = hyz_cols[2] - hyz_cols[3]
    hyz_rp_selected_gate = hyz_cols[4]
    hyz_rp_complement_gate = hyz_cols[5]
    hyz_rm_complement_gate = hyz_cols[6]
    hyz_rm_selected_gate = hyz_cols[7]
    hyz_weight_signed = []
    hyz_bias_signed = []
    hyz_weight_read_step = []
    hyz_bias_read_step = []
    hyz_wp_step = []
    hyz_wm_step = []
    hyz_bp_step = []
    hyz_bm_step = []
    hyz_hp_final = []
    hyz_hm_final = []
    for idx, (_name, _label, _hplus_src, _hminus_src, err_sign, _expected_sign) in enumerate(hybrid_product_cases):
        base = 8 + 14 * idx
        hp = hyz_cols[base]
        hm = hyz_cols[base + 1]
        wp = hyz_cols[base + 2]
        wm = hyz_cols[base + 3]
        bp = hyz_cols[base + 4]
        bm = hyz_cols[base + 5]
        wread = (hyz_cols[base + 7] - hyz_cols[base + 6]) + (hyz_cols[base + 9] - hyz_cols[base + 8])
        bread = (hyz_cols[base + 11] - hyz_cols[base + 10]) + (hyz_cols[base + 13] - hyz_cols[base + 12])
        hyz_hp_final.append(hyzat(2.00e-6, hp))
        hyz_hm_final.append(hyzat(2.00e-6, hm))
        hyz_weight_signed.append(hyzat(2.70e-6, wp - wm))
        hyz_bias_signed.append(hyzat(2.70e-6, bp - bm))
        hyz_weight_read_step.append(hyzat(2.70e-6, wread) - hyzat(2.00e-6, wread))
        hyz_bias_read_step.append(hyzat(2.70e-6, bread) - hyzat(2.00e-6, bread))
        hyz_wp_step.append(hyzat(2.70e-6, wp) - hyzat(2.00e-6, wp))
        hyz_wm_step.append(hyzat(2.70e-6, wm) - hyzat(2.00e-6, wm))
        hyz_bp_step.append(hyzat(2.70e-6, bp) - hyzat(2.00e-6, bp))
        hyz_bm_step.append(hyzat(2.70e-6, bm) - hyzat(2.00e-6, bm))
    hyz_weight_signed = np.array(hyz_weight_signed)
    hyz_bias_signed = np.array(hyz_bias_signed)
    hyz_weight_read_step = np.array(hyz_weight_read_step)
    hyz_bias_read_step = np.array(hyz_bias_read_step)
    hyz_wp_step = np.array(hyz_wp_step)
    hyz_wm_step = np.array(hyz_wm_step)
    hyz_bp_step = np.array(hyz_bp_step)
    hyz_bm_step = np.array(hyz_bm_step)
    hyz_hp_final = np.array(hyz_hp_final)
    hyz_hm_final = np.array(hyz_hm_final)
    hyz_weight_expected = np.array([expected_sign for *_rest, expected_sign in hybrid_product_cases])
    hyz_bias_expected = np.array([1.0 if err_sign == "rp" else -1.0 for *_prefix, err_sign, _expected_sign in hybrid_product_cases])
    require(hyzat(1.35e-6, hyz_hidden_pos) > 0.07, "hybrid cell quadrants e+ store should be positive")
    require(hyzat(1.35e-6, hyz_hidden_neg) < -0.07, "hybrid cell quadrants e- store should be negative")
    require(abs(hyzat(1.35e-6, hyz_hidden_pos + hyz_hidden_neg)) < 0.003, "hybrid cell quadrant hidden stores should be symmetric")
    require(hyzat(1.45e-6, hyz_rp_selected_gate) < 0.30, "hybrid cell quadrants e+ selected gate should be low")
    require(hyzat(1.45e-6, hyz_rp_complement_gate) > 1.60, "hybrid cell quadrants e+ complement gate should be high")
    require(hyzat(1.45e-6, hyz_rm_selected_gate) < 0.30, "hybrid cell quadrants e- selected gate should be low")
    require(hyzat(1.45e-6, hyz_rm_complement_gate) > 1.60, "hybrid cell quadrants e- complement gate should be high")
    require(np.all(np.sign(hyz_weight_signed) == hyz_weight_expected), "integrated quadrant weight signs should follow activation-error products")
    require(np.all(np.sign(hyz_bias_signed) == hyz_bias_expected), "integrated quadrant bias signs should follow error sign only")
    require(np.all(np.abs(hyz_weight_signed) > 0.018), "integrated quadrant weight writes should be useful")
    require(np.all(np.abs(hyz_bias_signed) > 0.030), "integrated quadrant bias writes should be useful")
    require(np.max(np.minimum(hyz_wp_step, hyz_wm_step)) < 5e-4, "integrated quadrant inactive weight rails should stay suppressed")
    require(np.max(np.minimum(hyz_bp_step, hyz_bm_step)) < 5e-4, "integrated quadrant inactive bias rails should stay suppressed")
    require(np.all(np.sign(hyz_weight_read_step) == hyz_weight_expected), "integrated quadrant weight readback signs should match stored weights")
    require(np.all(np.sign(hyz_bias_read_step) == hyz_bias_expected), "integrated quadrant bias readback signs should match stored biases")
    require(np.all(np.abs(hyz_weight_read_step) > 15e-6), "integrated quadrant weight readback should be measurable")
    require(np.all(np.abs(hyz_bias_read_step) > 25e-6), "integrated quadrant bias readback should be measurable")
    require(np.all(np.abs(hyz_hp_final[:2] - 0.92) < 0.010), "integrated quadrant a+ cases should sample active a+")
    require(np.all(hyz_hm_final[:2] > 1.35), "integrated quadrant a+ cases should leave a- inactive")
    require(np.all(hyz_hp_final[2:] > 1.35), "integrated quadrant a- cases should leave a+ inactive")
    require(np.all(np.abs(hyz_hm_final[2:] - 0.92) < 0.010), "integrated quadrant a- cases should sample active a-")

    hyz_x = np.arange(len(hybrid_product_cases))
    hyz_labels = [label for _name, label, *_rest in hybrid_product_cases]
    hyz_fig, hyz_axes = plt.subplots(3, 1, figsize=(7.4, 7.2), gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})
    hyz_axes[0].plot(1e6 * hyzt, hyz_hidden_pos, label="stored $e^+$")
    hyz_axes[0].plot(1e6 * hyzt, hyz_hidden_neg, label="stored $e^-$")
    hyz_axes[0].plot(1e6 * hyzt, hyz_rp_selected_gate, label="$e^+$ selected gate")
    hyz_axes[0].plot(1e6 * hyzt, hyz_rm_selected_gate, label="$e^-$ selected gate")
    hyz_axes[0].plot(1e6 * hyzt, hyz_cols[-2] / 2.0, color="0.45", alpha=0.35, label="$psamp/2$")
    hyz_axes[0].plot(1e6 * hyzt, hyz_cols[-1] / 20.0, color="0.15", alpha=0.25, label="$pacc_n/20$")
    hyz_axes[0].set_xlim(0.35, 2.75)
    hyz_axes[0].set_ylabel("voltage (V)")
    hyz_axes[0].set_title("Integrated quadrant deck stores both error signs once")
    hyz_axes[0].grid(True, alpha=0.25)
    hyz_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyz_axes[1].bar(hyz_x - 0.18, 1e3 * hyz_weight_signed, width=0.36, label="$W^+ - W^-$")
    hyz_axes[1].bar(hyz_x + 0.18, 1e3 * hyz_bias_signed, width=0.36, label="$B^+ - B^-$")
    hyz_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyz_axes[1].set_xticks(hyz_x)
    hyz_axes[1].set_xticklabels(hyz_labels)
    hyz_axes[1].set_ylabel("state step (mV)")
    hyz_axes[1].set_title("weight follows $ae$ while bias follows $e$")
    hyz_axes[1].grid(True, axis="y", alpha=0.25)
    hyz_axes[1].legend(loc="upper right", fontsize="small")
    hyz_axes[2].bar(hyz_x - 0.18, 1e6 * hyz_weight_read_step, width=0.36, label="weight read current")
    hyz_axes[2].bar(hyz_x + 0.18, 1e6 * hyz_bias_read_step, width=0.36, label="bias read current")
    hyz_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyz_axes[2].set_xticks(hyz_x)
    hyz_axes[2].set_xticklabels(hyz_labels)
    hyz_axes[2].set_xlabel("sampled activation/error quadrant")
    hyz_axes[2].set_ylabel("read current step (uA)")
    hyz_axes[2].set_title("readback signs match the stored differential states")
    hyz_axes[2].grid(True, axis="y", alpha=0.25)
    hyz_axes[2].legend(loc="upper right", fontsize="small")
    hyz_fig.tight_layout()
    save_plot(hyz_fig, "mos_hidden_writer_restored_gate_hybrid_cell_quadrants_ngspice")

    hybrid_cell_forward_deck = f"""
* Hybrid restored writer update-to-forward-store integration check.
* A single positive sample writes persistent W+ and B+ capacitors, then those
* same capacitors drive a shared z+/z- summing pair through NMOS synapse
* slices.  A crossed MOS forward pair reads that summed preactivation and
* stores the activation on h+/h- capacitors.  No behavioral update or readback
* source is used.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CBIAS=500p CSTORE=10p CSUM=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u WREAD=24u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD_HYV pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP_HYV rp_hyv 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VACT_SRC_HYV hsrc_hyv 0 PWL(0 1.45 1.35u 1.45 1.37u 0.92 1.82u 0.92 1.84u 1.45 4.6u 1.45)
VPSAMP_HYV psamp_hyv 0 PWL(0 0 1.42u 0 1.44u 1.8 1.76u 1.8 1.78u 0 4.6u 0)
VPSAMPN_HYV psampn_hyv 0 PWL(0 1.8 1.42u 1.8 1.44u 0 1.76u 0 1.78u 1.8 4.6u 1.8)
VPACC_HYV paccn_hyv 0 PWL(0 1.8 2.05u 1.8 2.07u 0 2.39u 0 2.41u 1.8 4.6u 1.8)
VREAD_HYV read_hyv 0 PWL(0 0 2.70u 0 2.72u 1.15 3.36u 1.15 3.38u 0 4.6u 0)
VPACT_HYV pact_hyv 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 4.6u 0)

VXP_HYV xp_hyv 0 1.15
VXM_HYV xm_hyv 0 0.65
VZPP_HYV_SRC zpp_hyv_src 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYV_SRC zmm_hyv_src 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYV_SRC zpm_hyv_src 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYV_SRC zmp_hyv_src 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYV hpp_hyv hpp_hyv vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYV hpm_hyv hpm_hyv vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYV hpp_hyv zpp_hyv_src tailp_hyv 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYV hpm_hyv zmm_hyv_src tailp_hyv 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYV tailp_hyv vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYV hmp_hyv hmp_hyv vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYV hmm_hyv hmm_hyv vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYV hmp_hyv zpm_hyv_src tailm_hyv 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYV hmm_hyv zmp_hyv_src tailm_hyv 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYV tailm_hyv vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYV cdp_rp_hyv 0 {{CERR}} IC=1.04
CDM_RP_HYV cdm_rp_hyv 0 {{CERR}} IC=1.04
RDP_RP_HYV cdp_rp_hyv 0 50G
RDM_RP_HYV cdm_rp_hyv 0 50G
{sign_store_path("hpm_hyv", "rp_hyv", "cdp_rp_hyv", "hyvrp1")}
{sign_store_path("hmp_hyv", "rp_hyv", "cdp_rp_hyv", "hyvrp2")}
{sign_store_path("hpp_hyv", "rp_hyv", "cdm_rp_hyv", "hyvrp3")}
{sign_store_path("hmm_hyv", "rp_hyv", "cdm_rp_hyv", "hyvrp4")}

MPRP_CDP_HYV rgp_rp_hyv cdp_rp_hyv vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYV rgp_rp_hyv cdp_rp_hyv 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYV rgm_rp_hyv cdm_rp_hyv vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYV rgm_rp_hyv cdm_rp_hyv 0 0 NMOS L={{LCH}} W={{WRESTN}}

CHM_HYV hm_store_hyv 0 {{CSTORE}} IC=1.45
RHM_HYV hm_store_hyv 0 50G
MSACTN_HYV hsrc_hyv psamp_hyv hm_store_hyv 0 NMOS L={{LCH}} W={{WSW}}
MSACTP_HYV hsrc_hyv psampn_hyv hm_store_hyv vdd PMOS L={{LCH}} W={{WSW}}

CWP_HYV wp_hyv 0 {{CWRITE}} IC=0.85
CWM_HYV wm_hyv 0 {{CWRITE}} IC=0.85
CBP_HYV bp_hyv 0 {{CBIAS}} IC=0.85
CBM_HYV bm_hyv 0 {{CBIAS}} IC=0.85
MWP_HYV_A vdd paccn_hyv n_wp_hyv_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYV_B n_wp_hyv_a hm_store_hyv n_wp_hyv_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYV_C n_wp_hyv_b rgp_rp_hyv n_wp_hyv_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYV_D n_wp_hyv_c cdm_rp_hyv wp_hyv vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYV_A vdd paccn_hyv n_wm_hyv_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYV_B n_wm_hyv_a hm_store_hyv n_wm_hyv_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYV_C n_wm_hyv_b rgm_rp_hyv n_wm_hyv_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYV_D n_wm_hyv_c cdp_rp_hyv wm_hyv vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYV_A vdd paccn_hyv n_bp_hyv_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYV_B n_bp_hyv_a rgp_rp_hyv n_bp_hyv_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYV_C n_bp_hyv_b cdm_rp_hyv bp_hyv vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYV_A vdd paccn_hyv n_bm_hyv_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYV_B n_bm_hyv_a rgm_rp_hyv n_bm_hyv_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYV_C n_bm_hyv_b cdp_rp_hyv bm_hyv vdd PMOS L={{LCH}} W={{WWRITE}}

CZP_HYV zp_hyv 0 {{CSUM}} IC=0.9
CZM_HYV zm_hyv 0 {{CSUM}} IC=0.9
RZP_HYV zp_hyv 0 100G
RZM_HYV zm_hyv 0 100G
MPP_W_HYV zp_hyv xp_hyv tail_wp_hyv 0 NMOS L={{LCH}} W={{WN}}
MPM_W_HYV zm_hyv xm_hyv tail_wp_hyv 0 NMOS L={{LCH}} W={{WN}}
MTW_GATE_HYV tail_wp_hyv read_hyv tail_wp_store_hyv 0 NMOS L={{LCH}} W={{WREAD}}
MTW_STORE_HYV tail_wp_store_hyv wp_hyv 0 0 NMOS L={{LCH}} W=12u
MNP_W_HYV zp_hyv xm_hyv tail_wm_hyv 0 NMOS L={{LCH}} W={{WN}}
MNM_W_HYV zm_hyv xp_hyv tail_wm_hyv 0 NMOS L={{LCH}} W={{WN}}
MTN_GATE_HYV tail_wm_hyv read_hyv tail_wm_store_hyv 0 NMOS L={{LCH}} W={{WREAD}}
MTN_STORE_HYV tail_wm_store_hyv wm_hyv 0 0 NMOS L={{LCH}} W=12u
MPP_B_HYV zp_hyv xp_hyv tail_bp_hyv 0 NMOS L={{LCH}} W={{WN}}
MPM_B_HYV zm_hyv xm_hyv tail_bp_hyv 0 NMOS L={{LCH}} W={{WN}}
MTB_GATE_HYV tail_bp_hyv read_hyv tail_bp_store_hyv 0 NMOS L={{LCH}} W={{WREAD}}
MTB_STORE_HYV tail_bp_store_hyv bp_hyv 0 0 NMOS L={{LCH}} W=12u
MNP_B_HYV zp_hyv xm_hyv tail_bm_hyv 0 NMOS L={{LCH}} W={{WN}}
MNM_B_HYV zm_hyv xp_hyv tail_bm_hyv 0 NMOS L={{LCH}} W={{WN}}
MTBN_GATE_HYV tail_bm_hyv read_hyv tail_bm_store_hyv 0 NMOS L={{LCH}} W={{WREAD}}
MTBN_STORE_HYV tail_bm_store_hyv bm_hyv 0 0 NMOS L={{LCH}} W=12u

MPFP_HYV hp_hyv hp_hyv vdd vdd PMOS L={{LCH}} W={{WP}}
MPFM_HYV hm_hyv hm_hyv vdd vdd PMOS L={{LCH}} W={{WP}}
MNFP_HYV hp_hyv zm_hyv ftail_hyv 0 NMOS L={{LCH}} W=48u
MNFM_HYV hm_hyv zp_hyv ftail_hyv 0 NMOS L={{LCH}} W=48u
MNFT_HYV ftail_hyv vbias 0 0 NMOS L={{LCH}} W=48u
CHP_HYV hp_store_hyv 0 {{CSTORE}} IC=1.04
CHM_FWD_HYV hm_fwd_store_hyv 0 {{CSTORE}} IC=1.04
RHP_HYV hp_store_hyv 0 50G
RHM_FWD_HYV hm_fwd_store_hyv 0 50G
MSFP_HYV hp_hyv pact_hyv hp_store_hyv 0 NMOS L={{LCH}} W=48u
MSFM_HYV hm_hyv pact_hyv hm_fwd_store_hyv 0 NMOS L={{LCH}} W=48u

.control
set noaskquit
tran 5n 4.55u uic
wrdata mos_hidden_writer_restored_gate_hybrid_update_forward.dat v(cdp_rp_hyv) v(cdm_rp_hyv) v(rgp_rp_hyv) v(rgm_rp_hyv) v(hm_store_hyv) v(wp_hyv) v(wm_hyv) v(bp_hyv) v(bm_hyv) v(zp_hyv) v(zm_hyv) v(hp_hyv) v(hm_hyv) v(hp_store_hyv) v(hm_fwd_store_hyv) v(paccn_hyv) v(read_hyv) v(pact_hyv)
quit
.endc
.end
"""
    hybrid_cell_forward_data = run_ngspice(
        hybrid_cell_forward_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward",
    )
    hyvt, hyv_cols = load_wrdata(hybrid_cell_forward_data, 18)

    def hyvat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyvt - time_s))])

    hyv_hidden = hyv_cols[0] - hyv_cols[1]
    hyv_selected_gate = hyv_cols[2]
    hyv_complement_gate = hyv_cols[3]
    hyv_hcap = hyv_cols[4]
    hyv_weight = hyv_cols[5] - hyv_cols[6]
    hyv_bias = hyv_cols[7] - hyv_cols[8]
    hyv_preact = hyv_cols[10] - hyv_cols[9]
    hyv_forward_load = hyv_cols[12] - hyv_cols[11]
    hyv_forward_store = hyv_cols[14] - hyv_cols[13]
    require(hyvat(1.35e-6, hyv_hidden) > 0.07, "update-forward deck should store positive error")
    require(hyvat(1.45e-6, hyv_selected_gate) < 0.30, "update-forward selected restored gate should be low")
    require(hyvat(1.45e-6, hyv_complement_gate) > 1.60, "update-forward complement restored gate should be high")
    require(abs(hyvat(1.90e-6, hyv_hcap) - 0.92) < 0.012, "update-forward deck should sample active activation")
    require(hyvat(2.55e-6, hyv_weight) > 0.020, "update-forward deck should write positive weight state")
    require(hyvat(2.55e-6, hyv_bias) > 0.030, "update-forward deck should write positive bias state")
    require(abs(hyvat(2.60e-6, hyv_preact)) < 0.003, "summed preactivation should stay quiet before read phase")
    require(hyvat(3.35e-6, hyv_preact) > 0.050, "written W/B states should drive positive summed preactivation")
    require(hyvat(3.35e-6, hyv_forward_load) > 0.040, "forward pair should read the summed W/B preactivation before common-mode collapse")
    require(hyvat(3.50e-6, hyv_forward_store) > 0.040, "pact should store the forward activation during the valid read window")
    require(hyvat(4.45e-6, hyv_forward_store) > 0.040, "stored forward activation should hold after pact")
    require(abs(hyvat(4.45e-6, hyv_weight) - hyvat(2.55e-6, hyv_weight)) < 5e-4, "read/forward phases should not disturb weight state")
    require(abs(hyvat(4.45e-6, hyv_bias) - hyvat(2.55e-6, hyv_bias)) < 5e-4, "read/forward phases should not disturb bias state")

    hyv_fig, hyv_axes = plt.subplots(3, 1, figsize=(7.2, 7.4), gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})
    hyv_axes[0].plot(1e6 * hyvt, hyv_hidden, label="stored $e^+$")
    hyv_axes[0].plot(1e6 * hyvt, hyv_selected_gate, label="selected error gate")
    hyv_axes[0].plot(1e6 * hyvt, hyv_complement_gate, label="complement error gate")
    hyv_axes[0].plot(1e6 * hyvt, hyv_hcap, label="sampled activation gate")
    hyv_axes[0].set_ylabel("voltage (V)")
    hyv_axes[0].set_title("Update-forward deck stores error and activation")
    hyv_axes[0].grid(True, alpha=0.25)
    hyv_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyv_axes[1].plot(1e6 * hyvt, 1e3 * hyv_weight, label="$W^+ - W^-$")
    hyv_axes[1].plot(1e6 * hyvt, 1e3 * hyv_bias, label="$B^+ - B^-$")
    hyv_axes[1].plot(1e6 * hyvt, 1e3 * hyv_preact, label="$z^- - z^+$")
    hyv_axes[1].plot(1e6 * hyvt, hyv_cols[15] / 20.0, color="0.55", alpha=0.25, label="$pacc_n/20$")
    hyv_axes[1].plot(1e6 * hyvt, hyv_cols[16] / 20.0, color="0.25", alpha=0.25, label="$read/20$")
    hyv_axes[1].set_ylabel("mV")
    hyv_axes[1].set_title("Written W/B capacitors drive one shared summing node")
    hyv_axes[1].grid(True, alpha=0.25)
    hyv_axes[1].legend(loc="upper left", ncol=3, fontsize="small")
    hyv_axes[2].plot(1e6 * hyvt, 1e3 * hyv_forward_load, label="forward-pair load")
    hyv_axes[2].plot(1e6 * hyvt, 1e3 * hyv_forward_store, label="stored activation")
    hyv_axes[2].plot(1e6 * hyvt, hyv_cols[17] / 20.0, color="0.35", alpha=0.25, label="$pact/20$")
    hyv_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyv_axes[2].set_xlabel("time (us)")
    hyv_axes[2].set_ylabel("activation differential (mV)")
    hyv_axes[2].set_title("Crossed MOS forward pair stores the summed activation")
    hyv_axes[2].grid(True, alpha=0.25)
    hyv_axes[2].legend(loc="upper left", fontsize="small")
    hyv_fig.tight_layout()
    save_plot(hyv_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_ngspice")

    hybrid_forward_quadrant_devices = []
    hybrid_forward_quadrant_prints = [
        "v(cdp_rp_hyf)",
        "v(cdm_rp_hyf)",
        "v(cdp_rm_hyf)",
        "v(cdm_rm_hyf)",
        "v(rgp_rp_hyf)",
        "v(rgm_rp_hyf)",
        "v(rgp_rm_hyf)",
        "v(rgm_rm_hyf)",
    ]
    for name, _label, hplus_src, hminus_src, err_sign, _expected_weight_sign in hybrid_product_cases:
        if err_sign == "rp":
            wp_activation = f"hp_store_hyf_{name}"
            wm_activation = f"hm_store_hyf_{name}"
            wp_restored = "rgp_rp_hyf"
            wp_error = "cdm_rp_hyf"
            wm_restored = "rgp_rp_hyf"
            wm_error = "cdm_rp_hyf"
            bp_restored = "rgp_rp_hyf"
            bp_error = "cdm_rp_hyf"
            bm_restored = "rgm_rp_hyf"
            bm_error = "cdp_rp_hyf"
        else:
            wp_activation = f"hm_store_hyf_{name}"
            wm_activation = f"hp_store_hyf_{name}"
            wp_restored = "rgm_rm_hyf"
            wp_error = "cdp_rm_hyf"
            wm_restored = "rgm_rm_hyf"
            wm_error = "cdp_rm_hyf"
            bp_restored = "rgp_rm_hyf"
            bp_error = "cdm_rm_hyf"
            bm_restored = "rgm_rm_hyf"
            bm_error = "cdp_rm_hyf"
        xplus = 1.15 if hplus_src < hminus_src else 0.65
        xminus = 0.65 if hplus_src < hminus_src else 1.15
        hybrid_forward_quadrant_devices.append(
            f"""
VHP_SRC_HYF_{name} hp_src_hyf_{name} 0 {hplus_src:.3f}
VHM_SRC_HYF_{name} hm_src_hyf_{name} 0 {hminus_src:.3f}
CHP_ACT_HYF_{name} hp_store_hyf_{name} 0 {{CSTORE}} IC=1.45
CHM_ACT_HYF_{name} hm_store_hyf_{name} 0 {{CSTORE}} IC=1.45
RHP_ACT_HYF_{name} hp_store_hyf_{name} 0 50G
RHM_ACT_HYF_{name} hm_store_hyf_{name} 0 50G
MSHPN_HYF_{name} hp_src_hyf_{name} psamp_hyf hp_store_hyf_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSHPP_HYF_{name} hp_src_hyf_{name} psampn_hyf hp_store_hyf_{name} vdd PMOS L={{LCH}} W={{WSW}}
MSHMN_HYF_{name} hm_src_hyf_{name} psamp_hyf hm_store_hyf_{name} 0 NMOS L={{LCH}} W={{WSW}}
MSHMP_HYF_{name} hm_src_hyf_{name} psampn_hyf hm_store_hyf_{name} vdd PMOS L={{LCH}} W={{WSW}}

CWP_HYF_{name} wp_hyf_{name} 0 {{CWRITE}} IC=0.85
CWM_HYF_{name} wm_hyf_{name} 0 {{CWRITE}} IC=0.85
CBP_HYF_{name} bp_hyf_{name} 0 {{CBIAS}} IC=0.85
CBM_HYF_{name} bm_hyf_{name} 0 {{CBIAS}} IC=0.85

MWP_HYF_{name}A vdd paccn_hyf n_wp_hyf_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYF_{name}B n_wp_hyf_{name}_a {wp_activation} n_wp_hyf_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYF_{name}C n_wp_hyf_{name}_b {wp_restored} n_wp_hyf_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYF_{name}D n_wp_hyf_{name}_c {wp_error} wp_hyf_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYF_{name}A vdd paccn_hyf n_wm_hyf_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYF_{name}B n_wm_hyf_{name}_a {wm_activation} n_wm_hyf_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYF_{name}C n_wm_hyf_{name}_b {wm_restored} n_wm_hyf_{name}_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYF_{name}D n_wm_hyf_{name}_c {wm_error} wm_hyf_{name} vdd PMOS L={{LCH}} W={{WWRITE}}

MBP_HYF_{name}A vdd paccn_hyf n_bp_hyf_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYF_{name}B n_bp_hyf_{name}_a {bp_restored} n_bp_hyf_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYF_{name}C n_bp_hyf_{name}_b {bp_error} bp_hyf_{name} vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYF_{name}A vdd paccn_hyf n_bm_hyf_{name}_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYF_{name}B n_bm_hyf_{name}_a {bm_restored} n_bm_hyf_{name}_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYF_{name}C n_bm_hyf_{name}_b {bm_error} bm_hyf_{name} vdd PMOS L={{LCH}} W={{WWRITE}}

CZP_HYF_{name} zp_hyf_{name} 0 {{CSUM}} IC=0.9
CZM_HYF_{name} zm_hyf_{name} 0 {{CSUM}} IC=0.9
RZP_HYF_{name} zp_hyf_{name} 0 100G
RZM_HYF_{name} zm_hyf_{name} 0 100G
VXP_W_HYF_{name} xp_w_hyf_{name} 0 {xplus:.3f}
VXM_W_HYF_{name} xm_w_hyf_{name} 0 {xminus:.3f}
VXP_B_HYF_{name} xp_b_hyf_{name} 0 1.15
VXM_B_HYF_{name} xm_b_hyf_{name} 0 0.65
MPP_W_HYF_{name} zp_hyf_{name} xp_w_hyf_{name} tail_wp_hyf_{name} 0 NMOS L={{LCH}} W={{WN}}
MPM_W_HYF_{name} zm_hyf_{name} xm_w_hyf_{name} tail_wp_hyf_{name} 0 NMOS L={{LCH}} W={{WN}}
MTW_GATE_HYF_{name} tail_wp_hyf_{name} read_hyf tail_wp_store_hyf_{name} 0 NMOS L={{LCH}} W={{WREAD}}
MTW_STORE_HYF_{name} tail_wp_store_hyf_{name} wp_hyf_{name} 0 0 NMOS L={{LCH}} W=12u
MNP_W_HYF_{name} zp_hyf_{name} xm_w_hyf_{name} tail_wm_hyf_{name} 0 NMOS L={{LCH}} W={{WN}}
MNM_W_HYF_{name} zm_hyf_{name} xp_w_hyf_{name} tail_wm_hyf_{name} 0 NMOS L={{LCH}} W={{WN}}
MTN_GATE_HYF_{name} tail_wm_hyf_{name} read_hyf tail_wm_store_hyf_{name} 0 NMOS L={{LCH}} W={{WREAD}}
MTN_STORE_HYF_{name} tail_wm_store_hyf_{name} wm_hyf_{name} 0 0 NMOS L={{LCH}} W=12u
MPP_B_HYF_{name} zp_hyf_{name} xp_b_hyf_{name} tail_bp_hyf_{name} 0 NMOS L={{LCH}} W={{WN}}
MPM_B_HYF_{name} zm_hyf_{name} xm_b_hyf_{name} tail_bp_hyf_{name} 0 NMOS L={{LCH}} W={{WN}}
MTB_GATE_HYF_{name} tail_bp_hyf_{name} read_hyf tail_bp_store_hyf_{name} 0 NMOS L={{LCH}} W={{WREAD}}
MTB_STORE_HYF_{name} tail_bp_store_hyf_{name} bp_hyf_{name} 0 0 NMOS L={{LCH}} W=12u
MNP_B_HYF_{name} zp_hyf_{name} xm_b_hyf_{name} tail_bm_hyf_{name} 0 NMOS L={{LCH}} W={{WN}}
MNM_B_HYF_{name} zm_hyf_{name} xp_b_hyf_{name} tail_bm_hyf_{name} 0 NMOS L={{LCH}} W={{WN}}
MTBN_GATE_HYF_{name} tail_bm_hyf_{name} read_hyf tail_bm_store_hyf_{name} 0 NMOS L={{LCH}} W={{WREAD}}
MTBN_STORE_HYF_{name} tail_bm_store_hyf_{name} bm_hyf_{name} 0 0 NMOS L={{LCH}} W=12u

MPFP_HYF_{name} hp_hyf_{name} hp_hyf_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MPFM_HYF_{name} hm_hyf_{name} hm_hyf_{name} vdd vdd PMOS L={{LCH}} W={{WP}}
MNFP_HYF_{name} hp_hyf_{name} zm_hyf_{name} ftail_hyf_{name} 0 NMOS L={{LCH}} W=48u
MNFM_HYF_{name} hm_hyf_{name} zp_hyf_{name} ftail_hyf_{name} 0 NMOS L={{LCH}} W=48u
MNFT_HYF_{name} ftail_hyf_{name} vbias 0 0 NMOS L={{LCH}} W=48u
CHP_FWD_HYF_{name} hp_fwd_store_hyf_{name} 0 {{CSTORE}} IC=1.04
CHM_FWD_HYF_{name} hm_fwd_store_hyf_{name} 0 {{CSTORE}} IC=1.04
RHP_FWD_HYF_{name} hp_fwd_store_hyf_{name} 0 50G
RHM_FWD_HYF_{name} hm_fwd_store_hyf_{name} 0 50G
MSFP_HYF_{name} hp_hyf_{name} pact_hyf hp_fwd_store_hyf_{name} 0 NMOS L={{LCH}} W=48u
MSFM_HYF_{name} hm_hyf_{name} pact_hyf hm_fwd_store_hyf_{name} 0 NMOS L={{LCH}} W=48u
"""
        )
        hybrid_forward_quadrant_prints.extend(
            [
                f"v(wp_hyf_{name})",
                f"v(wm_hyf_{name})",
                f"v(bp_hyf_{name})",
                f"v(bm_hyf_{name})",
                f"v(zp_hyf_{name})",
                f"v(zm_hyf_{name})",
                f"v(hp_hyf_{name})",
                f"v(hm_hyf_{name})",
                f"v(hp_fwd_store_hyf_{name})",
                f"v(hm_fwd_store_hyf_{name})",
            ]
        )

    hybrid_forward_quadrant_deck = f"""
* Hybrid restored writer four-quadrant update-to-forward-store check.
* Each matched copy writes W/B from one activation/error quadrant, reads those
* same persistent states into a summing pair, and stores the crossed MOS
* forward-pair activation.  There are no behavioral update or readback sources.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CBIAS=500p CSTORE=10p CSUM=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u WREAD=24u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD_HYF pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRP_HYF rp_hyf 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VRM_HYF rm_hyf 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 5.0u)
VPSAMP_HYF psamp_hyf 0 PWL(0 0 1.42u 0 1.44u 1.8 1.76u 1.8 1.78u 0 4.6u 0)
VPSAMPN_HYF psampn_hyf 0 PWL(0 1.8 1.42u 1.8 1.44u 0 1.76u 0 1.78u 1.8 4.6u 1.8)
VPACC_HYF paccn_hyf 0 PWL(0 1.8 2.05u 1.8 2.07u 0 2.39u 0 2.41u 1.8 4.6u 1.8)
VREAD_HYF read_hyf 0 PWL(0 0 2.70u 0 2.72u 1.15 3.36u 1.15 3.38u 0 4.6u 0)
VPACT_HYF pact_hyf 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 4.6u 0)

VZPP_HYF_SRC zpp_hyf_src 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYF_SRC zmm_hyf_src 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYF_SRC zpm_hyf_src 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYF_SRC zmp_hyf_src 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYF hpp_hyf hpp_hyf vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYF hpm_hyf hpm_hyf vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYF hpp_hyf zpp_hyf_src tailp_hyf 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYF hpm_hyf zmm_hyf_src tailp_hyf 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYF tailp_hyf vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYF hmp_hyf hmp_hyf vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYF hmm_hyf hmm_hyf vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYF hmp_hyf zpm_hyf_src tailm_hyf 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYF hmm_hyf zmp_hyf_src tailm_hyf 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYF tailm_hyf vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYF cdp_rp_hyf 0 {{CERR}} IC=1.04
CDM_RP_HYF cdm_rp_hyf 0 {{CERR}} IC=1.04
CDP_RM_HYF cdp_rm_hyf 0 {{CERR}} IC=1.04
CDM_RM_HYF cdm_rm_hyf 0 {{CERR}} IC=1.04
RDP_RP_HYF cdp_rp_hyf 0 50G
RDM_RP_HYF cdm_rp_hyf 0 50G
RDP_RM_HYF cdp_rm_hyf 0 50G
RDM_RM_HYF cdm_rm_hyf 0 50G
{sign_store_path("hpm_hyf", "rp_hyf", "cdp_rp_hyf", "hyfrp1")}
{sign_store_path("hmp_hyf", "rp_hyf", "cdp_rp_hyf", "hyfrp2")}
{sign_store_path("hpp_hyf", "rp_hyf", "cdm_rp_hyf", "hyfrp3")}
{sign_store_path("hmm_hyf", "rp_hyf", "cdm_rp_hyf", "hyfrp4")}
{sign_store_path("hpp_hyf", "rm_hyf", "cdp_rm_hyf", "hyfrm1")}
{sign_store_path("hmm_hyf", "rm_hyf", "cdp_rm_hyf", "hyfrm2")}
{sign_store_path("hpm_hyf", "rm_hyf", "cdm_rm_hyf", "hyfrm3")}
{sign_store_path("hmp_hyf", "rm_hyf", "cdm_rm_hyf", "hyfrm4")}

MPRP_CDP_HYF rgp_rp_hyf cdp_rp_hyf vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYF rgp_rp_hyf cdp_rp_hyf 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYF rgm_rp_hyf cdm_rp_hyf vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYF rgm_rp_hyf cdm_rp_hyf 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDP_HYF rgp_rm_hyf cdp_rm_hyf vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDP_HYF rgp_rm_hyf cdp_rm_hyf 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDM_HYF rgm_rm_hyf cdm_rm_hyf vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDM_HYF rgm_rm_hyf cdm_rm_hyf 0 0 NMOS L={{LCH}} W={{WRESTN}}

{''.join(hybrid_forward_quadrant_devices)}

.control
set noaskquit
tran 5n 4.55u uic
wrdata mos_hidden_writer_restored_gate_hybrid_update_forward_quadrants.dat {' '.join(hybrid_forward_quadrant_prints)} v(psamp_hyf) v(paccn_hyf) v(read_hyf) v(pact_hyf)
quit
.endc
.end
"""
    hybrid_forward_quadrant_data = run_ngspice(
        hybrid_forward_quadrant_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_quadrants",
    )
    hyft, hyf_cols = load_wrdata(hybrid_forward_quadrant_data, len(hybrid_forward_quadrant_prints) + 4)

    def hyfat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyft - time_s))])

    hyf_hidden_pos = hyf_cols[0] - hyf_cols[1]
    hyf_hidden_neg = hyf_cols[2] - hyf_cols[3]
    hyf_rp_selected_gate = hyf_cols[4]
    hyf_rp_complement_gate = hyf_cols[5]
    hyf_rm_complement_gate = hyf_cols[6]
    hyf_rm_selected_gate = hyf_cols[7]
    hyf_weight = []
    hyf_bias = []
    hyf_preact = []
    hyf_forward_load = []
    hyf_forward_store = []
    for idx, (_name, _label, _hplus_src, _hminus_src, err_sign, _expected_weight_sign) in enumerate(hybrid_product_cases):
        base = 8 + 10 * idx
        wp = hyf_cols[base]
        wm = hyf_cols[base + 1]
        bp = hyf_cols[base + 2]
        bm = hyf_cols[base + 3]
        zp = hyf_cols[base + 4]
        zm = hyf_cols[base + 5]
        hp = hyf_cols[base + 6]
        hm = hyf_cols[base + 7]
        hp_store = hyf_cols[base + 8]
        hm_store = hyf_cols[base + 9]
        hyf_weight.append(hyfat(2.55e-6, wp - wm))
        hyf_bias.append(hyfat(2.55e-6, bp - bm))
        hyf_preact.append(hyfat(3.35e-6, zm - zp))
        hyf_forward_load.append(hyfat(3.35e-6, hm - hp))
        hyf_forward_store.append(hyfat(4.45e-6, hm_store - hp_store))
    hyf_weight = np.array(hyf_weight)
    hyf_bias = np.array(hyf_bias)
    hyf_preact = np.array(hyf_preact)
    hyf_forward_load = np.array(hyf_forward_load)
    hyf_forward_store = np.array(hyf_forward_store)
    hyf_weight_expected = np.array([expected_sign for *_rest, expected_sign in hybrid_product_cases])
    hyf_error_expected = np.array([1.0 if err_sign == "rp" else -1.0 for *_prefix, err_sign, _expected_sign in hybrid_product_cases])
    require(hyfat(1.35e-6, hyf_hidden_pos) > 0.07, "update-forward quadrants e+ store should be positive")
    require(hyfat(1.35e-6, hyf_hidden_neg) < -0.07, "update-forward quadrants e- store should be negative")
    require(abs(hyfat(1.35e-6, hyf_hidden_pos + hyf_hidden_neg)) < 0.003, "update-forward quadrant hidden stores should be symmetric")
    require(hyfat(1.45e-6, hyf_rp_selected_gate) < 0.30, "update-forward quadrants e+ selected gate should be low")
    require(hyfat(1.45e-6, hyf_rp_complement_gate) > 1.60, "update-forward quadrants e+ complement gate should be high")
    require(hyfat(1.45e-6, hyf_rm_selected_gate) < 0.30, "update-forward quadrants e- selected gate should be low")
    require(hyfat(1.45e-6, hyf_rm_complement_gate) > 1.60, "update-forward quadrants e- complement gate should be high")
    require(np.all(np.sign(hyf_weight) == hyf_weight_expected), "update-forward quadrant weights should follow activation-error signs")
    require(np.all(np.sign(hyf_bias) == hyf_error_expected), "update-forward quadrant biases should follow error signs")
    require(np.all(np.sign(hyf_preact) == hyf_error_expected), "update-forward quadrant summing nodes should follow error signs")
    require(np.all(np.sign(hyf_forward_load) == hyf_error_expected), "update-forward quadrant forward loads should follow error signs")
    require(np.all(np.sign(hyf_forward_store) == hyf_error_expected), "update-forward quadrant stored activations should follow error signs")
    require(np.all(np.abs(hyf_weight) > 0.018), "update-forward quadrant weights should have useful magnitude")
    require(np.all(np.abs(hyf_bias) > 0.030), "update-forward quadrant biases should have useful magnitude")
    require(np.all(np.abs(hyf_preact) > 0.045), "update-forward quadrant preactivations should have useful magnitude")
    require(np.all(np.abs(hyf_forward_store) > 0.030), "update-forward quadrant stored activations should have useful magnitude")

    hyf_x = np.arange(len(hybrid_product_cases))
    hyf_labels = [label for _name, label, *_rest in hybrid_product_cases]
    hyf_fig, hyf_axes = plt.subplots(3, 1, figsize=(7.4, 7.2), gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})
    hyf_axes[0].plot(1e6 * hyft, hyf_hidden_pos, label="stored $e^+$")
    hyf_axes[0].plot(1e6 * hyft, hyf_hidden_neg, label="stored $e^-$")
    hyf_axes[0].plot(1e6 * hyft, hyf_rp_selected_gate, label="$e^+$ selected gate")
    hyf_axes[0].plot(1e6 * hyft, hyf_rm_selected_gate, label="$e^-$ selected gate")
    hyf_axes[0].plot(1e6 * hyft, hyf_cols[-4] / 2.0, color="0.45", alpha=0.35, label="$psamp/2$")
    hyf_axes[0].plot(1e6 * hyft, hyf_cols[-3] / 20.0, color="0.15", alpha=0.25, label="$pacc_n/20$")
    hyf_axes[0].set_xlim(0.35, 2.75)
    hyf_axes[0].set_ylabel("voltage (V)")
    hyf_axes[0].set_title("Four matched copies store both error signs once")
    hyf_axes[0].grid(True, alpha=0.25)
    hyf_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyf_axes[1].bar(hyf_x - 0.18, 1e3 * hyf_weight, width=0.36, label="$W^+ - W^-$")
    hyf_axes[1].bar(hyf_x + 0.18, 1e3 * hyf_bias, width=0.36, label="$B^+ - B^-$")
    hyf_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyf_axes[1].set_xticks(hyf_x)
    hyf_axes[1].set_xticklabels(hyf_labels)
    hyf_axes[1].set_ylabel("state step (mV)")
    hyf_axes[1].set_title("local update writes signed W/B capacitor states")
    hyf_axes[1].grid(True, axis="y", alpha=0.25)
    hyf_axes[1].legend(loc="upper right", fontsize="small")
    hyf_axes[2].bar(hyf_x - 0.24, 1e3 * hyf_preact, width=0.24, label="$z^- - z^+$")
    hyf_axes[2].bar(hyf_x, 1e3 * hyf_forward_load, width=0.24, label="forward load")
    hyf_axes[2].bar(hyf_x + 0.24, 1e3 * hyf_forward_store, width=0.24, label="stored activation")
    hyf_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyf_axes[2].set_xticks(hyf_x)
    hyf_axes[2].set_xticklabels(hyf_labels)
    hyf_axes[2].set_xlabel("sampled activation/error quadrant")
    hyf_axes[2].set_ylabel("differential (mV)")
    hyf_axes[2].set_title("read, forward pair, and activation store follow the error sign")
    hyf_axes[2].grid(True, axis="y", alpha=0.25)
    hyf_axes[2].legend(loc="upper right", fontsize="small")
    hyf_fig.tight_layout()
    save_plot(hyf_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_quadrants_ngspice")

    hybrid_forward_reuse_deck = f"""
* Hybrid restored writer same-cell update-to-forward reuse check.
* One physical hidden-error store, activation sample cap, W/B state pair,
* summing pair, and forward-store pair sees two samples.  The first sample is
* a+e+ and writes a positive W/B state.  MOS transmission-gate reset clears
* only the transient hidden/activation/z/h state.  The second sample is a+e-
* and must cancel the persistent W/B differential without Python capacitor
* writes or behavioral update/readback sources.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p CBIAS=500p CSTORE=10p CSUM=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u WREAD=24u WRESETN=60u WRESETP=180u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VECM ecm 0 1.04
VACM acm 0 1.45
VZCM zcm 0 0.90
VHCM hcm 0 1.04
VPBWD_HYR pbwd 0 PWL(0 0 0.45u 0 0.47u 1.8 1.25u 1.8 1.27u 0 4.25u 0 4.27u 1.8 5.05u 1.8 5.07u 0 7.8u 0)
VRP_HYR rp_hyr 0 PWL(0 0 0.45u 0 0.47u 1.8 1.25u 1.8 1.27u 0 7.8u 0)
VRM_HYR rm_hyr 0 PWL(0 0 4.25u 0 4.27u 1.8 5.05u 1.8 5.07u 0 7.8u 0)
VRESET_HYR rst_hyr 0 PWL(0 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 7.8u 0)
VRESETN_HYR rstn_hyr 0 PWL(0 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 7.8u 1.8)
VACT_SRC_HYR hsrc_hyr 0 PWL(0 1.45 1.35u 1.45 1.37u 0.92 1.82u 0.92 1.84u 1.45 5.13u 1.45 5.15u 0.92 5.60u 0.92 5.62u 1.45 7.8u 1.45)
VPSAMP_HYR psamp_hyr 0 PWL(0 0 1.42u 0 1.44u 1.8 1.76u 1.8 1.78u 0 5.20u 0 5.22u 1.8 5.54u 1.8 5.56u 0 7.8u 0)
VPSAMPN_HYR psampn_hyr 0 PWL(0 1.8 1.42u 1.8 1.44u 0 1.76u 0 1.78u 1.8 5.20u 1.8 5.22u 0 5.54u 0 5.56u 1.8 7.8u 1.8)
VPACC_HYR paccn_hyr 0 PWL(0 1.8 2.05u 1.8 2.07u 0 2.39u 0 2.41u 1.8 5.85u 1.8 5.87u 0 6.19u 0 6.21u 1.8 7.8u 1.8)
VREAD_HYR read_hyr 0 PWL(0 0 2.70u 0 2.72u 1.15 3.36u 1.15 3.38u 0 6.50u 0 6.52u 1.15 7.16u 1.15 7.18u 0 7.8u 0)
VPACT_HYR pact_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 6.96u 0 6.98u 1.8 7.11u 1.8 7.13u 0 7.8u 0)

VXP_HYR xp_hyr 0 1.15
VXM_HYR xm_hyr 0 0.65
VZPP_HYR_SRC zpp_hyr_src 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYR_SRC zmm_hyr_src 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYR_SRC zpm_hyr_src 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYR_SRC zmp_hyr_src 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYR hpp_hyr hpp_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYR hpm_hyr hpm_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYR hpp_hyr zpp_hyr_src tailp_hyr 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYR hpm_hyr zmm_hyr_src tailp_hyr 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYR tailp_hyr vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYR hmp_hyr hmp_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYR hmm_hyr hmm_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYR hmp_hyr zpm_hyr_src tailm_hyr 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYR hmm_hyr zmp_hyr_src tailm_hyr 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYR tailm_hyr vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_HYR cdp_hyr 0 {{CERR}} IC=1.04
CDM_HYR cdm_hyr 0 {{CERR}} IC=1.04
RDP_HYR cdp_hyr 0 50G
RDM_HYR cdm_hyr 0 50G
MRDPN_HYR cdp_hyr rst_hyr ecm 0 NMOS L={{LCH}} W={{WRESETN}}
MRDMN_HYR cdm_hyr rst_hyr ecm 0 NMOS L={{LCH}} W={{WRESETN}}
MRDPP_HYR cdp_hyr rstn_hyr ecm vdd PMOS L={{LCH}} W={{WRESETP}}
MRDMP_HYR cdm_hyr rstn_hyr ecm vdd PMOS L={{LCH}} W={{WRESETP}}
{sign_store_path("hpm_hyr", "rp_hyr", "cdp_hyr", "hyrrp1")}
{sign_store_path("hmp_hyr", "rp_hyr", "cdp_hyr", "hyrrp2")}
{sign_store_path("hpp_hyr", "rp_hyr", "cdm_hyr", "hyrrp3")}
{sign_store_path("hmm_hyr", "rp_hyr", "cdm_hyr", "hyrrp4")}
{sign_store_path("hpp_hyr", "rm_hyr", "cdp_hyr", "hyrrm1")}
{sign_store_path("hmm_hyr", "rm_hyr", "cdp_hyr", "hyrrm2")}
{sign_store_path("hpm_hyr", "rm_hyr", "cdm_hyr", "hyrrm3")}
{sign_store_path("hmp_hyr", "rm_hyr", "cdm_hyr", "hyrrm4")}

MPRP_CDP_HYR rgp_hyr cdp_hyr vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYR rgp_hyr cdp_hyr 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYR rgm_hyr cdm_hyr vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYR rgm_hyr cdm_hyr 0 0 NMOS L={{LCH}} W={{WRESTN}}

CHM_ACT_HYR hm_act_hyr 0 {{CSTORE}} IC=1.45
RHM_ACT_HYR hm_act_hyr 0 50G
MRHAN_HYR hm_act_hyr rst_hyr acm 0 NMOS L={{LCH}} W={{WRESETN}}
MRHAP_HYR hm_act_hyr rstn_hyr acm vdd PMOS L={{LCH}} W={{WRESETP}}
MSACTN_HYR hsrc_hyr psamp_hyr hm_act_hyr 0 NMOS L={{LCH}} W={{WSW}}
MSACTP_HYR hsrc_hyr psampn_hyr hm_act_hyr vdd PMOS L={{LCH}} W={{WSW}}

CWP_HYR wp_hyr 0 {{CWRITE}} IC=0.85
CWM_HYR wm_hyr 0 {{CWRITE}} IC=0.85
CBP_HYR bp_hyr 0 {{CBIAS}} IC=0.85
CBM_HYR bm_hyr 0 {{CBIAS}} IC=0.85
MWP_HYR_A vdd paccn_hyr n_wp_hyr_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYR_B n_wp_hyr_a hm_act_hyr n_wp_hyr_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYR_C n_wp_hyr_b rgp_hyr n_wp_hyr_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYR_D n_wp_hyr_c cdm_hyr wp_hyr vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYR_A vdd paccn_hyr n_wm_hyr_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYR_B n_wm_hyr_a hm_act_hyr n_wm_hyr_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYR_C n_wm_hyr_b rgm_hyr n_wm_hyr_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYR_D n_wm_hyr_c cdp_hyr wm_hyr vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYR_A vdd paccn_hyr n_bp_hyr_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYR_B n_bp_hyr_a rgp_hyr n_bp_hyr_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBP_HYR_C n_bp_hyr_b cdm_hyr bp_hyr vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYR_A vdd paccn_hyr n_bm_hyr_a vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYR_B n_bm_hyr_a rgm_hyr n_bm_hyr_b vdd PMOS L={{LCH}} W={{WWRITE}}
MBM_HYR_C n_bm_hyr_b cdp_hyr bm_hyr vdd PMOS L={{LCH}} W={{WWRITE}}

CZP_HYR zp_hyr 0 {{CSUM}} IC=0.9
CZM_HYR zm_hyr 0 {{CSUM}} IC=0.9
RZP_HYR zp_hyr 0 100G
RZM_HYR zm_hyr 0 100G
MRZPN_HYR zp_hyr rst_hyr zcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRZMN_HYR zm_hyr rst_hyr zcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRZPP_HYR zp_hyr rstn_hyr zcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRZMP_HYR zm_hyr rstn_hyr zcm vdd PMOS L={{LCH}} W={{WRESETP}}
MPP_W_HYR zp_hyr xp_hyr tail_wp_hyr 0 NMOS L={{LCH}} W={{WN}}
MPM_W_HYR zm_hyr xm_hyr tail_wp_hyr 0 NMOS L={{LCH}} W={{WN}}
MTW_GATE_HYR tail_wp_hyr read_hyr tail_wp_store_hyr 0 NMOS L={{LCH}} W={{WREAD}}
MTW_STORE_HYR tail_wp_store_hyr wp_hyr 0 0 NMOS L={{LCH}} W=12u
MNP_W_HYR zp_hyr xm_hyr tail_wm_hyr 0 NMOS L={{LCH}} W={{WN}}
MNM_W_HYR zm_hyr xp_hyr tail_wm_hyr 0 NMOS L={{LCH}} W={{WN}}
MTN_GATE_HYR tail_wm_hyr read_hyr tail_wm_store_hyr 0 NMOS L={{LCH}} W={{WREAD}}
MTN_STORE_HYR tail_wm_store_hyr wm_hyr 0 0 NMOS L={{LCH}} W=12u
MPP_B_HYR zp_hyr xp_hyr tail_bp_hyr 0 NMOS L={{LCH}} W={{WN}}
MPM_B_HYR zm_hyr xm_hyr tail_bp_hyr 0 NMOS L={{LCH}} W={{WN}}
MTB_GATE_HYR tail_bp_hyr read_hyr tail_bp_store_hyr 0 NMOS L={{LCH}} W={{WREAD}}
MTB_STORE_HYR tail_bp_store_hyr bp_hyr 0 0 NMOS L={{LCH}} W=12u
MNP_B_HYR zp_hyr xm_hyr tail_bm_hyr 0 NMOS L={{LCH}} W={{WN}}
MNM_B_HYR zm_hyr xp_hyr tail_bm_hyr 0 NMOS L={{LCH}} W={{WN}}
MTBN_GATE_HYR tail_bm_hyr read_hyr tail_bm_store_hyr 0 NMOS L={{LCH}} W={{WREAD}}
MTBN_STORE_HYR tail_bm_store_hyr bm_hyr 0 0 NMOS L={{LCH}} W=12u

MPFP_HYR hp_hyr hp_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MPFM_HYR hm_hyr hm_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MNFP_HYR hp_hyr zm_hyr ftail_hyr 0 NMOS L={{LCH}} W=48u
MNFM_HYR hm_hyr zp_hyr ftail_hyr 0 NMOS L={{LCH}} W=48u
MNFT_HYR ftail_hyr vbias 0 0 NMOS L={{LCH}} W=48u
CHP_FWD_HYR hp_fwd_hyr 0 {{CSTORE}} IC=1.04
CHM_FWD_HYR hm_fwd_hyr 0 {{CSTORE}} IC=1.04
RHP_FWD_HYR hp_fwd_hyr 0 50G
RHM_FWD_HYR hm_fwd_hyr 0 50G
MRFPN_HYR hp_fwd_hyr rst_hyr hcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRFMN_HYR hm_fwd_hyr rst_hyr hcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRFPP_HYR hp_fwd_hyr rstn_hyr hcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRFMP_HYR hm_fwd_hyr rstn_hyr hcm vdd PMOS L={{LCH}} W={{WRESETP}}
MSFP_HYR hp_hyr pact_hyr hp_fwd_hyr 0 NMOS L={{LCH}} W=48u
MSFM_HYR hm_hyr pact_hyr hm_fwd_hyr 0 NMOS L={{LCH}} W=48u

.control
set noaskquit
tran 5n 7.75u uic
wrdata mos_hidden_writer_restored_gate_hybrid_update_forward_reuse.dat v(cdp_hyr) v(cdm_hyr) v(rgp_hyr) v(rgm_hyr) v(hm_act_hyr) v(wp_hyr) v(wm_hyr) v(bp_hyr) v(bm_hyr) v(zp_hyr) v(zm_hyr) v(hp_hyr) v(hm_hyr) v(hp_fwd_hyr) v(hm_fwd_hyr) v(pbwd) v(rp_hyr) v(rm_hyr) v(rst_hyr) v(psamp_hyr) v(paccn_hyr) v(read_hyr) v(pact_hyr)
quit
.endc
.end
"""
    hybrid_forward_reuse_data = run_ngspice(
        hybrid_forward_reuse_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_reuse",
    )
    hyrt, hyr_cols = load_wrdata(hybrid_forward_reuse_data, 23)

    def hyrat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyrt - time_s))])

    hyr_hidden = hyr_cols[0] - hyr_cols[1]
    hyr_rgp = hyr_cols[2]
    hyr_rgm = hyr_cols[3]
    hyr_hcap = hyr_cols[4]
    hyr_weight = hyr_cols[5] - hyr_cols[6]
    hyr_bias = hyr_cols[7] - hyr_cols[8]
    hyr_preact = hyr_cols[10] - hyr_cols[9]
    hyr_forward_load = hyr_cols[12] - hyr_cols[11]
    hyr_forward_store = hyr_cols[14] - hyr_cols[13]
    require(hyrat(1.35e-6, hyr_hidden) > 0.07, "reuse first sample should store positive hidden error")
    require(hyrat(1.45e-6, hyr_rgp) < 0.30, "reuse first sample should select e+ restored gate")
    require(abs(hyrat(1.90e-6, hyr_hcap) - 0.92) < 0.012, "reuse first sample should store active activation")
    require(hyrat(2.55e-6, hyr_weight) > 0.020, "reuse first sample should write positive weight state")
    require(hyrat(2.55e-6, hyr_bias) > 0.030, "reuse first sample should write positive bias state")
    require(hyrat(3.35e-6, hyr_preact) > 0.045, "reuse first sample should read positive preactivation")
    require(hyrat(4.10e-6, np.abs(hyr_hidden)) < 0.004, "reuse reset should clear hidden-error differential")
    require(abs(hyrat(4.10e-6, hyr_hcap) - 1.45) < 0.015, "reuse reset should return activation gate inactive")
    require(abs(hyrat(4.10e-6, hyr_preact)) < 0.001, "reuse reset should clear summing differential")
    require(abs(hyrat(4.10e-6, hyr_forward_store)) < 0.001, "reuse reset should clear stored forward activation")
    require(hyrat(5.15e-6, hyr_hidden) < -0.07, "reuse second sample should store negative hidden error")
    require(hyrat(5.20e-6, hyr_rgm) < 0.30, "reuse second sample should select e- restored gate")
    require(abs(hyrat(5.75e-6, hyr_hcap) - 0.92) < 0.012, "reuse second sample should resample active activation")
    require(abs(hyrat(6.35e-6, hyr_weight)) < 0.004, "reuse second opposite update should cancel weight differential")
    require(abs(hyrat(6.35e-6, hyr_bias)) < 0.004, "reuse second opposite update should cancel bias differential")
    require(abs(hyrat(7.15e-6, hyr_preact)) < 0.001, "reuse cancelled W/B state should read near-zero preactivation")
    require(abs(hyrat(7.45e-6, hyr_forward_store)) < 0.001, "reuse cancelled state should store near-zero activation")

    hyr_fig, hyr_axes = plt.subplots(3, 1, figsize=(7.4, 7.4), gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})
    hyr_axes[0].plot(1e6 * hyrt, hyr_hidden, label="stored error differential")
    hyr_axes[0].plot(1e6 * hyrt, hyr_rgp, label="$e^+$ selected gate")
    hyr_axes[0].plot(1e6 * hyrt, hyr_rgm, label="$e^-$ selected gate")
    hyr_axes[0].plot(1e6 * hyrt, hyr_hcap, label="activation gate")
    hyr_axes[0].plot(1e6 * hyrt, hyr_cols[18] / 2.0, color="0.55", alpha=0.35, label="$reset/2$")
    hyr_axes[0].set_ylabel("voltage (V)")
    hyr_axes[0].set_title("MOS reset reuses the same error and activation storage")
    hyr_axes[0].grid(True, alpha=0.25)
    hyr_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyr_axes[1].plot(1e6 * hyrt, 1e3 * hyr_weight, label="$W^+ - W^-$")
    hyr_axes[1].plot(1e6 * hyrt, 1e3 * hyr_bias, label="$B^+ - B^-$")
    hyr_axes[1].plot(1e6 * hyrt, 1e3 * hyr_preact, label="$z^- - z^+$")
    hyr_axes[1].plot(1e6 * hyrt, hyr_cols[20] / 20.0, color="0.35", alpha=0.25, label="$pacc_n/20$")
    hyr_axes[1].plot(1e6 * hyrt, hyr_cols[21] / 20.0, color="0.15", alpha=0.25, label="$read/20$")
    hyr_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyr_axes[1].set_ylabel("mV")
    hyr_axes[1].set_title("Opposite second sample cancels persistent W/B differentials")
    hyr_axes[1].grid(True, alpha=0.25)
    hyr_axes[1].legend(loc="upper left", ncol=3, fontsize="small")
    hyr_axes[2].plot(1e6 * hyrt, 1e3 * hyr_forward_load, label="forward load")
    hyr_axes[2].plot(1e6 * hyrt, 1e3 * hyr_forward_store, label="stored activation")
    hyr_axes[2].plot(1e6 * hyrt, hyr_cols[22] / 20.0, color="0.35", alpha=0.25, label="$pact/20$")
    hyr_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyr_axes[2].set_xlabel("time (us)")
    hyr_axes[2].set_ylabel("activation differential (mV)")
    hyr_axes[2].set_title("Forward store resets, then stays near zero after cancellation")
    hyr_axes[2].grid(True, alpha=0.25)
    hyr_axes[2].legend(loc="upper left", fontsize="small")
    hyr_fig.tight_layout()
    save_plot(hyr_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_reuse_ngspice")

    def replace_required(deck: str, old: str, new: str) -> str:
        require(old in deck, f"missing deck fragment for replacement: {old[:80]}")
        return deck.replace(old, new)

    reset_strength_cases = [
        (24, 60, "old 24/60"),
        (36, 108, "36/108"),
        (48, 144, "48/144"),
        (60, 180, "chosen 60/180"),
        (72, 216, "72/216"),
    ]
    reset_strength_z = []
    reset_strength_h = []
    reset_strength_first_z = []
    reset_strength_first_h = []
    reset_strength_final_w = []
    reset_strength_final_b = []
    for n_width, p_width, _label in reset_strength_cases:
        stem = f"mos_hidden_writer_restored_gate_hybrid_update_forward_reuse_reset_n{n_width}_p{p_width}"
        strength_deck = replace_required(
            hybrid_forward_reuse_deck,
            "WRESETN=60u WRESETP=180u",
            f"WRESETN={n_width}u WRESETP={p_width}u",
        )
        strength_deck = replace_required(
            strength_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_reuse.dat",
            f"{stem}.dat",
        )
        strength_data = run_ngspice(strength_deck, stem)
        st, strength_cols = load_wrdata(strength_data, 23)

        def sat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(st - time_s))])

        strength_weight = strength_cols[5] - strength_cols[6]
        strength_bias = strength_cols[7] - strength_cols[8]
        strength_preact = strength_cols[10] - strength_cols[9]
        strength_store = strength_cols[14] - strength_cols[13]
        reset_strength_z.append(abs(sat(7.15e-6, strength_preact)))
        reset_strength_h.append(abs(sat(7.45e-6, strength_store)))
        reset_strength_first_z.append(sat(3.35e-6, strength_preact))
        reset_strength_first_h.append(sat(3.45e-6, strength_store))
        reset_strength_final_w.append(abs(sat(6.35e-6, strength_weight)))
        reset_strength_final_b.append(abs(sat(6.35e-6, strength_bias)))

    reset_strength_z = np.array(reset_strength_z)
    reset_strength_h = np.array(reset_strength_h)
    reset_strength_first_z = np.array(reset_strength_first_z)
    reset_strength_first_h = np.array(reset_strength_first_h)
    reset_strength_final_w = np.array(reset_strength_final_w)
    reset_strength_final_b = np.array(reset_strength_final_b)
    require(np.all(np.diff(reset_strength_z) < 0.0), "integrated reset z residue should improve with reset sizing")
    require(np.all(np.diff(reset_strength_h) < 0.0), "integrated reset stored activation residue should improve with reset sizing")
    require(np.max(np.abs(reset_strength_first_z - reset_strength_first_z[0])) < 1e-6, "reset sizing should not change the first valid preactivation")
    require(np.max(np.abs(reset_strength_first_h - reset_strength_first_h[0])) < 1e-6, "reset sizing should not change the first valid stored activation")
    require(reset_strength_z[0] > 1e-3, "old integrated reset sizing should leave visible z residue")
    require(reset_strength_h[0] > 1e-3, "old integrated reset sizing should leave visible stored-activation residue")
    chosen_idx = 3
    require(reset_strength_z[chosen_idx] < 2e-5, "chosen integrated reset sizing should clear z residue below 0.02 mV")
    require(reset_strength_h[chosen_idx] < 2e-5, "chosen integrated reset sizing should clear stored activation below 0.02 mV")
    require(np.max(reset_strength_final_w) < 5e-6, "reset sizing sweep should preserve W cancellation")
    require(np.max(reset_strength_final_b) < 5e-6, "reset sizing sweep should preserve B cancellation")

    strength_x = np.arange(len(reset_strength_cases))
    strength_labels = [label for _n, _p, label in reset_strength_cases]
    strength_fig, strength_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    strength_axes[0].semilogy(strength_x, 1e3 * reset_strength_z, "o-", label="final $|z^- - z^+|$")
    strength_axes[0].semilogy(strength_x, 1e3 * reset_strength_h, "s--", label="final stored $|h^- - h^+|$")
    strength_axes[0].axhline(0.02, color="0.4", linewidth=0.8, alpha=0.7, label="0.02 mV target")
    strength_axes[0].set_xticks(strength_x)
    strength_axes[0].set_xticklabels(strength_labels, rotation=15, ha="right")
    strength_axes[0].set_ylabel("cancelled-state residue (mV)")
    strength_axes[0].set_title("Integrated reset sizing controls reuse residue")
    strength_axes[0].grid(True, which="both", alpha=0.25)
    strength_axes[0].legend(loc="upper right")
    strength_axes[1].plot(strength_x, 1e3 * reset_strength_first_z, "o-", label="first valid $z^- - z^+$")
    strength_axes[1].plot(strength_x, 1e3 * reset_strength_first_h, "s--", label="first stored $h^- - h^+$")
    strength_axes[1].set_xticks(strength_x)
    strength_axes[1].set_xticklabels(strength_labels, rotation=15, ha="right")
    strength_axes[1].set_ylabel("first-sample response (mV)")
    strength_axes[1].set_title("Sizing reset devices does not change the intended write/read/store")
    strength_axes[1].grid(True, alpha=0.25)
    strength_axes[1].legend(loc="center right")
    strength_fig.tight_layout()
    save_plot(strength_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_reset_strength_ngspice")

    read_reuse_replacements = {
        "VPBWD_HYR pbwd 0 PWL(0 0 0.45u 0 0.47u 1.8 1.25u 1.8 1.27u 0 4.25u 0 4.27u 1.8 5.05u 1.8 5.07u 0 7.8u 0)":
            "VPBWD_HYR pbwd 0 PWL(0 0 0.45u 0 0.47u 1.8 1.25u 1.8 1.27u 0 7.8u 0)",
        "VRM_HYR rm_hyr 0 PWL(0 0 4.25u 0 4.27u 1.8 5.05u 1.8 5.07u 0 7.8u 0)":
            "VRM_HYR rm_hyr 0 0",
        "VRESET_HYR rst_hyr 0 PWL(0 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 7.8u 0)":
            "VRESET_HYR rst_hyr 0 PWL(0 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 5.18u 0 5.20u 1.8 5.60u 1.8 5.62u 0 7.8u 0)",
        "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 7.8u 1.8)":
            "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 5.18u 1.8 5.20u 0 5.60u 0 5.62u 1.8 7.8u 1.8)",
        "VACT_SRC_HYR hsrc_hyr 0 PWL(0 1.45 1.35u 1.45 1.37u 0.92 1.82u 0.92 1.84u 1.45 5.13u 1.45 5.15u 0.92 5.60u 0.92 5.62u 1.45 7.8u 1.45)":
            "VACT_SRC_HYR hsrc_hyr 0 PWL(0 1.45 1.35u 1.45 1.37u 0.92 1.82u 0.92 1.84u 1.45 7.8u 1.45)",
        "VPSAMP_HYR psamp_hyr 0 PWL(0 0 1.42u 0 1.44u 1.8 1.76u 1.8 1.78u 0 5.20u 0 5.22u 1.8 5.54u 1.8 5.56u 0 7.8u 0)":
            "VPSAMP_HYR psamp_hyr 0 PWL(0 0 1.42u 0 1.44u 1.8 1.76u 1.8 1.78u 0 7.8u 0)",
        "VPSAMPN_HYR psampn_hyr 0 PWL(0 1.8 1.42u 1.8 1.44u 0 1.76u 0 1.78u 1.8 5.20u 1.8 5.22u 0 5.54u 0 5.56u 1.8 7.8u 1.8)":
            "VPSAMPN_HYR psampn_hyr 0 PWL(0 1.8 1.42u 1.8 1.44u 0 1.76u 0 1.78u 1.8 7.8u 1.8)",
        "VPACC_HYR paccn_hyr 0 PWL(0 1.8 2.05u 1.8 2.07u 0 2.39u 0 2.41u 1.8 5.85u 1.8 5.87u 0 6.19u 0 6.21u 1.8 7.8u 1.8)":
            "VPACC_HYR paccn_hyr 0 PWL(0 1.8 2.05u 1.8 2.07u 0 2.39u 0 2.41u 1.8 7.8u 1.8)",
        "VREAD_HYR read_hyr 0 PWL(0 0 2.70u 0 2.72u 1.15 3.36u 1.15 3.38u 0 6.50u 0 6.52u 1.15 7.16u 1.15 7.18u 0 7.8u 0)":
            "VREAD_HYR read_hyr 0 PWL(0 0 2.70u 0 2.72u 1.15 3.36u 1.15 3.38u 0 4.30u 0 4.32u 1.15 4.96u 1.15 4.98u 0 5.90u 0 5.92u 1.15 6.56u 1.15 6.58u 0 7.8u 0)",
        "VPACT_HYR pact_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 6.96u 0 6.98u 1.8 7.11u 1.8 7.13u 0 7.8u 0)":
            "VPACT_HYR pact_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 4.76u 0 4.78u 1.8 4.91u 1.8 4.93u 0 6.36u 0 6.38u 1.8 6.51u 1.8 6.53u 0 7.8u 0)",
        "mos_hidden_writer_restored_gate_hybrid_update_forward_reuse.dat":
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
    }
    hybrid_forward_read_reuse_deck = hybrid_forward_reuse_deck
    for old, new in read_reuse_replacements.items():
        hybrid_forward_read_reuse_deck = replace_required(hybrid_forward_read_reuse_deck, old, new)
    hybrid_forward_read_reuse_data = run_ngspice(
        hybrid_forward_read_reuse_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse",
    )
    hyrrt, hyrr_cols = load_wrdata(hybrid_forward_read_reuse_data, 23)

    def hyrrat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyrrt - time_s))])

    hyrr_weight = hyrr_cols[5] - hyrr_cols[6]
    hyrr_bias = hyrr_cols[7] - hyrr_cols[8]
    hyrr_preact = hyrr_cols[10] - hyrr_cols[9]
    hyrr_forward_store = hyrr_cols[14] - hyrr_cols[13]
    hyrr_z_times = np.array([3.35, 4.95, 6.55]) * 1e-6
    hyrr_h_times = np.array([3.45, 5.05, 6.70]) * 1e-6
    hyrr_reset_times = np.array([4.10, 5.70]) * 1e-6
    hyrr_z_samples = np.array([hyrrat(ts, hyrr_preact) for ts in hyrr_z_times])
    hyrr_h_samples = np.array([hyrrat(ts, hyrr_forward_store) for ts in hyrr_h_times])
    hyrr_z_reset = np.array([abs(hyrrat(ts, hyrr_preact)) for ts in hyrr_reset_times])
    hyrr_h_reset = np.array([abs(hyrrat(ts, hyrr_forward_store)) for ts in hyrr_reset_times])
    hyrr_weight_after_write = hyrrat(2.55e-6, hyrr_weight)
    hyrr_bias_after_write = hyrrat(2.55e-6, hyrr_bias)
    hyrr_weight_drift = hyrrat(7.45e-6, hyrr_weight) - hyrr_weight_after_write
    hyrr_bias_drift = hyrrat(7.45e-6, hyrr_bias) - hyrr_bias_after_write
    require(hyrr_weight_after_write > 0.020, "read-reuse deck should write positive weight state")
    require(hyrr_bias_after_write > 0.030, "read-reuse deck should write positive bias state")
    require(np.all(hyrr_z_samples > 0.045), "read-reuse cycles should repeatedly read useful positive preactivation")
    require(np.all(hyrr_h_samples > 0.040), "read-reuse cycles should repeatedly store useful positive activation")
    require(np.max(hyrr_z_samples) - np.min(hyrr_z_samples) < 0.001, "read-reuse preactivation cycles should agree")
    require(np.max(hyrr_h_samples) - np.min(hyrr_h_samples) < 0.001, "read-reuse stored activations should agree")
    require(np.max(hyrr_z_reset) < 0.001, "read-reuse reset should clear preactivation between reads")
    require(np.max(hyrr_h_reset) < 0.001, "read-reuse reset should clear stored activation between reads")
    require(abs(hyrr_weight_drift) < 1e-5, "repeated read/reset cycles should not disturb weight state")
    require(abs(hyrr_bias_drift) < 1e-5, "repeated read/reset cycles should not disturb bias state")

    hyrr_fig, hyrr_axes = plt.subplots(3, 1, figsize=(7.4, 7.4), gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})
    hyrr_axes[0].plot(1e6 * hyrrt, 1e3 * hyrr_weight, label="$W^+ - W^-$")
    hyrr_axes[0].plot(1e6 * hyrrt, 1e3 * hyrr_bias, label="$B^+ - B^-$")
    hyrr_axes[0].plot(1e6 * hyrrt, hyrr_cols[20] / 20.0, color="0.45", alpha=0.35, label="$pacc_n/20$")
    hyrr_axes[0].set_ylabel("state differential (mV)")
    hyrr_axes[0].set_title("One local update is reused by repeated forward reads")
    hyrr_axes[0].grid(True, alpha=0.25)
    hyrr_axes[0].legend(loc="upper left", ncol=3, fontsize="small")
    hyrr_axes[1].plot(1e6 * hyrrt, 1e3 * hyrr_preact, label="$z^- - z^+$")
    hyrr_axes[1].plot(1e6 * hyrrt, hyrr_cols[21] / 20.0, color="0.25", alpha=0.25, label="$read/20$")
    hyrr_axes[1].plot(1e6 * hyrrt, hyrr_cols[18] / 20.0, color="0.55", alpha=0.35, label="$reset/20$")
    hyrr_axes[1].plot(1e6 * hyrr_z_times, 1e3 * hyrr_z_samples, "o", color="black", label="read samples")
    hyrr_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyrr_axes[1].set_ylabel("preactivation (mV)")
    hyrr_axes[1].set_title("MOS reset/read cycles reproduce the same summing result")
    hyrr_axes[1].grid(True, alpha=0.25)
    hyrr_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    hyrr_axes[2].plot(1e6 * hyrrt, 1e3 * hyrr_forward_store, label="stored activation")
    hyrr_axes[2].plot(1e6 * hyrrt, hyrr_cols[22] / 20.0, color="0.35", alpha=0.25, label="$pact/20$")
    hyrr_axes[2].plot(1e6 * hyrr_h_times, 1e3 * hyrr_h_samples, "o", color="black", label="activation samples")
    hyrr_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyrr_axes[2].set_xlabel("time (us)")
    hyrr_axes[2].set_ylabel("activation differential (mV)")
    hyrr_axes[2].set_title("Repeated forward-store pulses agree without disturbing W/B")
    hyrr_axes[2].grid(True, alpha=0.25)
    hyrr_axes[2].legend(loc="upper left", fontsize="small")
    hyrr_fig.tight_layout()
    save_plot(hyrr_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse_ngspice")

    negative_read_reuse_replacements = dict(read_reuse_replacements)
    negative_read_reuse_replacements.update(
        {
            "VRP_HYR rp_hyr 0 PWL(0 0 0.45u 0 0.47u 1.8 1.25u 1.8 1.27u 0 7.8u 0)":
                "VRP_HYR rp_hyr 0 0",
            "VRM_HYR rm_hyr 0 PWL(0 0 4.25u 0 4.27u 1.8 5.05u 1.8 5.07u 0 7.8u 0)":
                "VRM_HYR rm_hyr 0 PWL(0 0 0.45u 0 0.47u 1.8 1.25u 1.8 1.27u 0 7.8u 0)",
            "mos_hidden_writer_restored_gate_hybrid_update_forward_reuse.dat":
                "mos_hidden_writer_restored_gate_hybrid_update_forward_negative_read_reuse.dat",
        }
    )
    hybrid_forward_negative_read_reuse_deck = hybrid_forward_reuse_deck
    for old, new in negative_read_reuse_replacements.items():
        hybrid_forward_negative_read_reuse_deck = replace_required(hybrid_forward_negative_read_reuse_deck, old, new)
    hybrid_forward_negative_read_reuse_data = run_ngspice(
        hybrid_forward_negative_read_reuse_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_negative_read_reuse",
    )
    hynrt, hynr_cols = load_wrdata(hybrid_forward_negative_read_reuse_data, 23)

    def hynrat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hynrt - time_s))])

    hynr_weight = hynr_cols[5] - hynr_cols[6]
    hynr_bias = hynr_cols[7] - hynr_cols[8]
    hynr_preact = hynr_cols[10] - hynr_cols[9]
    hynr_forward_store = hynr_cols[14] - hynr_cols[13]
    hynr_z_samples = np.array([hynrat(ts, hynr_preact) for ts in hyrr_z_times])
    hynr_h_samples = np.array([hynrat(ts, hynr_forward_store) for ts in hyrr_h_times])
    hynr_z_reset = np.array([abs(hynrat(ts, hynr_preact)) for ts in hyrr_reset_times])
    hynr_h_reset = np.array([abs(hynrat(ts, hynr_forward_store)) for ts in hyrr_reset_times])
    hynr_weight_after_write = hynrat(2.55e-6, hynr_weight)
    hynr_bias_after_write = hynrat(2.55e-6, hynr_bias)
    hynr_weight_drift = hynrat(7.45e-6, hynr_weight) - hynr_weight_after_write
    hynr_bias_drift = hynrat(7.45e-6, hynr_bias) - hynr_bias_after_write
    require(hynr_weight_after_write < -0.020, "negative read-reuse deck should write negative weight state")
    require(hynr_bias_after_write < -0.030, "negative read-reuse deck should write negative bias state")
    require(np.all(hynr_z_samples < -0.045), "negative read-reuse cycles should repeatedly read useful negative preactivation")
    require(np.all(hynr_h_samples < -0.040), "negative read-reuse cycles should repeatedly store useful negative activation")
    require(np.max(hynr_z_samples) - np.min(hynr_z_samples) < 0.001, "negative read-reuse preactivation cycles should agree")
    require(np.max(hynr_h_samples) - np.min(hynr_h_samples) < 0.001, "negative read-reuse stored activations should agree")
    require(np.max(hynr_z_reset) < 0.001, "negative read-reuse reset should clear preactivation between reads")
    require(np.max(hynr_h_reset) < 0.001, "negative read-reuse reset should clear stored activation between reads")
    require(abs(hynr_weight_drift) < 1e-5, "negative repeated read/reset cycles should not disturb weight state")
    require(abs(hynr_bias_drift) < 1e-5, "negative repeated read/reset cycles should not disturb bias state")
    require(np.max(np.abs(hyrr_z_samples + hynr_z_samples)) < 0.001, "positive and negative read-reuse preactivations should mirror")
    require(np.max(np.abs(hyrr_h_samples + hynr_h_samples)) < 0.001, "positive and negative read-reuse activations should mirror")

    hynr_fig, hynr_axes = plt.subplots(3, 1, figsize=(7.4, 7.4), gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})
    hynr_axes[0].plot(1e6 * hyrrt, 1e3 * hyrr_weight, label="$+W$ state")
    hynr_axes[0].plot(1e6 * hynrt, 1e3 * hynr_weight, "--", label="$-W$ state")
    hynr_axes[0].plot(1e6 * hyrrt, 1e3 * hyrr_bias, label="$+B$ state")
    hynr_axes[0].plot(1e6 * hynrt, 1e3 * hynr_bias, "--", label="$-B$ state")
    hynr_axes[0].plot(1e6 * hyrrt, hyrr_cols[20] / 20.0, color="0.45", alpha=0.25, label="$pacc_n/20$")
    hynr_axes[0].axhline(0, color="0.4", linewidth=0.8)
    hynr_axes[0].set_ylabel("state differential (mV)")
    hynr_axes[0].set_title("Positive and negative local updates both survive repeated reads")
    hynr_axes[0].grid(True, alpha=0.25)
    hynr_axes[0].legend(loc="upper left", ncol=3, fontsize="small")
    hynr_axes[1].plot(1e6 * hyrrt, 1e3 * hyrr_preact, label="+ update reads")
    hynr_axes[1].plot(1e6 * hynrt, 1e3 * hynr_preact, "--", label="- update reads")
    hynr_axes[1].plot(1e6 * hyrr_z_times, 1e3 * hyrr_z_samples, "o", color="black")
    hynr_axes[1].plot(1e6 * hyrr_z_times, 1e3 * hynr_z_samples, "o", color="black")
    hynr_axes[1].plot(1e6 * hyrrt, hyrr_cols[21] / 20.0, color="0.25", alpha=0.25, label="$read/20$")
    hynr_axes[1].plot(1e6 * hyrrt, hyrr_cols[18] / 20.0, color="0.55", alpha=0.35, label="$reset/20$")
    hynr_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hynr_axes[1].set_ylabel("preactivation (mV)")
    hynr_axes[1].set_title("Read/reset cycles preserve the sign and magnitude of the summing result")
    hynr_axes[1].grid(True, alpha=0.25)
    hynr_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    hynr_axes[2].plot(1e6 * hyrrt, 1e3 * hyrr_forward_store, label="+ stored activation")
    hynr_axes[2].plot(1e6 * hynrt, 1e3 * hynr_forward_store, "--", label="- stored activation")
    hynr_axes[2].plot(1e6 * hyrr_h_times, 1e3 * hyrr_h_samples, "o", color="black")
    hynr_axes[2].plot(1e6 * hyrr_h_times, 1e3 * hynr_h_samples, "o", color="black")
    hynr_axes[2].plot(1e6 * hyrrt, hyrr_cols[22] / 20.0, color="0.35", alpha=0.25, label="$pact/20$")
    hynr_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hynr_axes[2].set_xlabel("time (us)")
    hynr_axes[2].set_ylabel("activation differential (mV)")
    hynr_axes[2].set_title("Forward-store reuse mirrors for both trainable-state signs")
    hynr_axes[2].grid(True, alpha=0.25)
    hynr_axes[2].legend(loc="upper left", ncol=2, fontsize="small")
    hynr_fig.tight_layout()
    save_plot(hynr_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_signed_read_reuse_ngspice")

    timing_read_pwl = "VREAD_HYR read_hyr 0 PWL(0 0 2.70u 0 2.72u 1.15 3.36u 1.15 3.38u 0 4.30u 0 4.32u 1.15 4.96u 1.15 4.98u 0 5.90u 0 5.92u 1.15 6.56u 1.15 6.58u 0 7.8u 0)"
    timing_single_read_pwl = "VREAD_HYR read_hyr 0 PWL(0 0 2.70u 0 2.72u 1.15 3.36u 1.15 3.38u 0 7.8u 0)"
    timing_base_pact_pwl = "VPACT_HYR pact_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 4.76u 0 4.78u 1.8 4.91u 1.8 4.93u 0 6.36u 0 6.38u 1.8 6.51u 1.8 6.53u 0 7.8u 0)"
    timing_cases = [
        ("pre_read", "pre-read", "VPACT_HYR pact_hyr 0 PWL(0 0 2.40u 0 2.42u 1.8 2.55u 1.8 2.57u 0 7.8u 0)", 2.65e-6),
        ("read_edge", "read edge", "VPACT_HYR pact_hyr 0 PWL(0 0 2.88u 0 2.90u 1.8 3.03u 1.8 3.05u 0 7.8u 0)", 3.12e-6),
        ("settled", "settled", "VPACT_HYR pact_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 7.8u 0)", 3.45e-6),
        ("late_short", "late short", "VPACT_HYR pact_hyr 0 PWL(0 0 3.40u 0 3.42u 1.8 3.55u 1.8 3.57u 0 7.8u 0)", 3.575e-6),
        ("after_reset", "after reset", "VPACT_HYR pact_hyr 0 PWL(0 0 4.12u 0 4.14u 1.8 4.27u 1.8 4.29u 0 7.8u 0)", 4.38e-6),
    ]
    timing_labels = []
    timing_store_samples = []
    timing_preact_samples = []
    timing_traces = []
    timing_times = None
    for suffix, label, pact_pwl, sample_time in timing_cases:
        stem = f"mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_timing_{suffix}"
        timing_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        timing_deck = replace_required(timing_deck, timing_base_pact_pwl, pact_pwl)
        timing_deck = replace_required(
            timing_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        timing_data = run_ngspice(timing_deck, stem)
        tt, timing_cols = load_wrdata(timing_data, 23)

        def timat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(tt - time_s))])

        timing_preact = timing_cols[10] - timing_cols[9]
        timing_store = timing_cols[14] - timing_cols[13]
        timing_labels.append(label)
        timing_preact_samples.append(timat(sample_time, timing_preact))
        timing_store_samples.append(timat(sample_time, timing_store))
        timing_traces.append((label, tt, timing_preact, timing_store, timing_cols[22]))
        timing_times = tt

    timing_store_samples = np.array(timing_store_samples)
    timing_preact_samples = np.array(timing_preact_samples)
    require(abs(timing_store_samples[0]) < 0.002, "pre-read pact should not store activation before read settles")
    require(
        0.005 < timing_store_samples[1] < 0.90 * timing_store_samples[2],
        "read-edge pact should store a partial activation",
    )
    require(timing_store_samples[2] > 0.040, "settled pact should store a useful activation")
    require(
        0.005 < timing_store_samples[3] < 0.70 * timing_store_samples[2],
        "late short pact should show aperture loss despite valid preactivation",
    )
    require(abs(timing_store_samples[4]) < 0.002, "post-reset pact should store near zero after transient state is cleared")
    require(timing_preact_samples[2] > 0.045, "settled timing case should have useful preactivation")
    require(timing_preact_samples[4] < 0.001, "post-reset timing case should have cleared preactivation")

    timing_x = np.arange(len(timing_cases))
    timing_fig, timing_axes = plt.subplots(2, 1, figsize=(7.4, 6.6), gridspec_kw={"height_ratios": [1.0, 0.9]})
    for label, tt, _preact, store, pact in timing_traces:
        timing_axes[0].plot(1e6 * tt, 1e3 * store, label=label)
    timing_axes[0].plot(1e6 * timing_times, timing_traces[2][4] / 20.0, color="0.35", alpha=0.25, label="$pact/20$")
    timing_axes[0].axhline(0, color="0.4", linewidth=0.8)
    timing_axes[0].set_xlim(2.25, 4.55)
    timing_axes[0].set_ylabel("stored activation (mV)")
    timing_axes[0].set_title("Integrated update-forward store depends on read/reset timing")
    timing_axes[0].grid(True, alpha=0.25)
    timing_axes[0].legend(loc="upper left", ncol=3, fontsize="small")
    timing_axes[1].bar(timing_x - 0.18, 1e3 * timing_preact_samples, width=0.36, label="$z^- - z^+$ at sample")
    timing_axes[1].bar(timing_x + 0.18, 1e3 * timing_store_samples, width=0.36, label="stored $h^- - h^+$")
    timing_axes[1].axhline(0, color="0.4", linewidth=0.8)
    timing_axes[1].set_xticks(timing_x)
    timing_axes[1].set_xticklabels(timing_labels, rotation=15, ha="right")
    timing_axes[1].set_ylabel("sampled differential (mV)")
    timing_axes[1].set_title("Good pact timing needs both valid read state and enough aperture")
    timing_axes[1].grid(True, axis="y", alpha=0.25)
    timing_axes[1].legend(loc="upper left", fontsize="small")
    timing_fig.tight_layout()
    save_plot(timing_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_timing_ngspice")

    aperture_cases_ns = [40, 80, 120, 160, 240, 320]
    aperture_start_us = 3.18
    aperture_labels = []
    aperture_store_samples = []
    aperture_preact_samples = []
    aperture_traces = []
    for width_ns in aperture_cases_ns:
        stem = f"mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_aperture_{width_ns}ns"
        end_us = aperture_start_us + width_ns / 1000.0
        off_us = end_us + 0.02
        sample_time = min((off_us + 0.08) * 1e-6, 3.575e-6)
        pact_pwl = (
            f"VPACT_HYR pact_hyr 0 PWL(0 0 {aperture_start_us - 0.02:.2f}u 0 "
            f"{aperture_start_us:.2f}u 1.8 {end_us:.2f}u 1.8 {off_us:.2f}u 0 7.8u 0)"
        )
        aperture_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        aperture_deck = replace_required(aperture_deck, timing_base_pact_pwl, pact_pwl)
        aperture_deck = replace_required(
            aperture_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        aperture_data = run_ngspice(aperture_deck, stem)
        atime, aperture_cols = load_wrdata(aperture_data, 23)

        def apat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(atime - time_s))])

        aperture_preact = aperture_cols[10] - aperture_cols[9]
        aperture_store = aperture_cols[14] - aperture_cols[13]
        aperture_labels.append(f"{width_ns} ns")
        aperture_preact_samples.append(apat(sample_time, aperture_preact))
        aperture_store_samples.append(apat(sample_time, aperture_store))
        aperture_traces.append((f"{width_ns} ns", atime, aperture_store))

    aperture_preact_samples = np.array(aperture_preact_samples)
    aperture_store_samples = np.array(aperture_store_samples)
    peak_idx = int(np.argmax(aperture_store_samples))
    require(peak_idx == 2, "integrated pact aperture sweep should peak near 120 ns")
    require(aperture_store_samples[2] > 0.045, "120 ns pact aperture should capture a full activation")
    require(aperture_store_samples[0] < 0.85 * aperture_store_samples[2], "40 ns pact aperture should undercharge")
    require(aperture_store_samples[1] > aperture_store_samples[0] + 0.006, "80 ns aperture should improve over 40 ns")
    require(aperture_store_samples[3] < 0.95 * aperture_store_samples[2], "160 ns aperture should begin tracking droop")
    require(aperture_store_samples[4] < 0.75 * aperture_store_samples[2], "240 ns aperture should track too much droop")
    require(aperture_store_samples[5] < 0.75 * aperture_store_samples[2], "320 ns aperture should remain degraded by droop")
    require(np.min(aperture_preact_samples[1:]) > 0.048, "aperture sweep should keep a valid read state for non-short cases")

    aperture_x = np.arange(len(aperture_cases_ns))
    aperture_fig, aperture_axes = plt.subplots(2, 1, figsize=(7.4, 6.6), gridspec_kw={"height_ratios": [1.0, 0.9]})
    for label, atime, store in aperture_traces:
        aperture_axes[0].plot(1e6 * atime, 1e3 * store, label=label)
    aperture_axes[0].axhline(0, color="0.4", linewidth=0.8)
    aperture_axes[0].set_xlim(3.05, 3.65)
    aperture_axes[0].set_ylabel("stored activation (mV)")
    aperture_axes[0].set_title("Activation-store aperture has a finite optimum")
    aperture_axes[0].grid(True, alpha=0.25)
    aperture_axes[0].legend(loc="upper left", ncol=3, fontsize="small")
    aperture_axes[1].bar(aperture_x - 0.18, 1e3 * aperture_preact_samples, width=0.36, label="$z^- - z^+$ at sample")
    aperture_axes[1].bar(aperture_x + 0.18, 1e3 * aperture_store_samples, width=0.36, label="stored $h^- - h^+$")
    aperture_axes[1].axhline(0, color="0.4", linewidth=0.8)
    aperture_axes[1].set_xticks(aperture_x)
    aperture_axes[1].set_xticklabels(aperture_labels)
    aperture_axes[1].set_ylabel("sampled differential (mV)")
    aperture_axes[1].set_xlabel("pact high-time")
    aperture_axes[1].set_title("Too short undercharges; too long tracks the forward-load droop")
    aperture_axes[1].grid(True, axis="y", alpha=0.25)
    aperture_axes[1].legend(loc="upper right", fontsize="small")
    aperture_fig.tight_layout()
    save_plot(aperture_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_aperture_ngspice")

    tg_aperture_store_samples = []
    tg_aperture_preact_samples = []
    tg_aperture_traces = []
    nmos_store_line_p = "MSFP_HYR hp_hyr pact_hyr hp_fwd_hyr 0 NMOS L={LCH} W=48u"
    nmos_store_line_m = "MSFM_HYR hm_hyr pact_hyr hm_fwd_hyr 0 NMOS L={LCH} W=48u"
    tg_store_line_p = nmos_store_line_p + "\nMSFPP_HYR hp_hyr pactn_hyr hp_fwd_hyr vdd PMOS L={LCH} W=120u"
    tg_store_line_m = nmos_store_line_m + "\nMSFMP_HYR hm_hyr pactn_hyr hm_fwd_hyr vdd PMOS L={LCH} W=120u"
    for width_ns in aperture_cases_ns:
        stem = f"mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_tg_aperture_{width_ns}ns"
        end_us = aperture_start_us + width_ns / 1000.0
        off_us = end_us + 0.02
        sample_time = min((off_us + 0.08) * 1e-6, 3.575e-6)
        pact_pwl = (
            f"VPACT_HYR pact_hyr 0 PWL(0 0 {aperture_start_us - 0.02:.2f}u 0 "
            f"{aperture_start_us:.2f}u 1.8 {end_us:.2f}u 1.8 {off_us:.2f}u 0 7.8u 0)"
        )
        pactn_pwl = (
            f"VPACTN_HYR pactn_hyr 0 PWL(0 1.8 {aperture_start_us - 0.02:.2f}u 1.8 "
            f"{aperture_start_us:.2f}u 0 {end_us:.2f}u 0 {off_us:.2f}u 1.8 7.8u 1.8)"
        )
        tg_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        tg_deck = replace_required(tg_deck, timing_base_pact_pwl, pact_pwl + "\n" + pactn_pwl)
        tg_deck = replace_required(tg_deck, nmos_store_line_p, tg_store_line_p)
        tg_deck = replace_required(tg_deck, nmos_store_line_m, tg_store_line_m)
        tg_deck = replace_required(
            tg_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        tg_data = run_ngspice(tg_deck, stem)
        tg_time, tg_cols = load_wrdata(tg_data, 23)

        def tgpat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(tg_time - time_s))])

        tg_preact = tg_cols[10] - tg_cols[9]
        tg_store = tg_cols[14] - tg_cols[13]
        tg_aperture_preact_samples.append(tgpat(sample_time, tg_preact))
        tg_aperture_store_samples.append(tgpat(sample_time, tg_store))
        tg_aperture_traces.append((f"{width_ns} ns", tg_time, tg_store))

    tg_aperture_preact_samples = np.array(tg_aperture_preact_samples)
    tg_aperture_store_samples = np.array(tg_aperture_store_samples)
    require(tg_aperture_store_samples[0] > aperture_store_samples[0] + 0.006, "TG store should improve 40 ns aperture")
    require(tg_aperture_store_samples[1] > aperture_store_samples[1] + 0.004, "TG store should improve 80 ns aperture")
    require(tg_aperture_store_samples[2] > aperture_store_samples[2] + 0.003, "TG store should improve 120 ns aperture")
    require(tg_aperture_store_samples[2] > 0.050, "TG store should capture a full 120 ns activation")
    require(tg_aperture_store_samples[4] < 0.75 * tg_aperture_store_samples[2], "TG 240 ns aperture should still track droop")
    require(tg_aperture_store_samples[5] < 0.75 * tg_aperture_store_samples[2], "TG 320 ns aperture should still track droop")
    require(np.min(tg_aperture_preact_samples[1:]) > 0.048, "TG aperture comparison should keep valid read state")

    store_topology_fig, store_topology_axes = plt.subplots(2, 1, figsize=(7.4, 6.6), gridspec_kw={"height_ratios": [1.0, 0.9]})
    for label, atime, store in tg_aperture_traces:
        if label in {"40 ns", "80 ns", "120 ns", "240 ns"}:
            store_topology_axes[0].plot(1e6 * atime, 1e3 * store, label=f"TG {label}")
    store_topology_axes[0].axhline(0, color="0.4", linewidth=0.8)
    store_topology_axes[0].set_xlim(3.05, 3.65)
    store_topology_axes[0].set_ylabel("stored activation (mV)")
    store_topology_axes[0].set_title("Transmission-gate store charges faster but still has a finite aperture")
    store_topology_axes[0].grid(True, alpha=0.25)
    store_topology_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    store_topology_axes[1].plot(aperture_cases_ns, 1e3 * aperture_store_samples, "o-", label="NMOS-only store")
    store_topology_axes[1].plot(aperture_cases_ns, 1e3 * tg_aperture_store_samples, "s--", label="transmission-gate store")
    store_topology_axes[1].set_xlabel("pact high-time (ns)")
    store_topology_axes[1].set_ylabel("stored $h^- - h^+$ (mV)")
    store_topology_axes[1].set_title("TG improves short apertures; long apertures still follow droop")
    store_topology_axes[1].grid(True, alpha=0.25)
    store_topology_axes[1].legend(loc="upper right")
    store_topology_fig.tight_layout()
    save_plot(store_topology_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_store_topology_ngspice")

    read_gated_store_samples = []
    read_gated_preact_samples = []
    read_gated_traces = []
    read_gated_store_line_p = "\n".join(
        [
            "MSFP_HYR hp_hyr pact_hyr hp_mid_hyr 0 NMOS L={LCH} W=96u",
            "MSFPP_HYR hp_hyr pactn_hyr hp_mid_hyr vdd PMOS L={LCH} W=240u",
            "MSFPV_HYR hp_mid_hyr reads_hyr hp_fwd_hyr 0 NMOS L={LCH} W=96u",
            "MSFPVP_HYR hp_mid_hyr readsn_hyr hp_fwd_hyr vdd PMOS L={LCH} W=240u",
        ]
    )
    read_gated_store_line_m = "\n".join(
        [
            "MSFM_HYR hm_hyr pact_hyr hm_mid_hyr 0 NMOS L={LCH} W=96u",
            "MSFMP_HYR hm_hyr pactn_hyr hm_mid_hyr vdd PMOS L={LCH} W=240u",
            "MSFMV_HYR hm_mid_hyr reads_hyr hm_fwd_hyr 0 NMOS L={LCH} W=96u",
            "MSFMVP_HYR hm_mid_hyr readsn_hyr hm_fwd_hyr vdd PMOS L={LCH} W=240u",
        ]
    )
    for width_ns in aperture_cases_ns:
        stem = f"mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_read_gated_tg_aperture_{width_ns}ns"
        end_us = aperture_start_us + width_ns / 1000.0
        off_us = end_us + 0.02
        sample_time = min((off_us + 0.08) * 1e-6, 3.575e-6)
        pact_pwl = (
            f"VPACT_HYR pact_hyr 0 PWL(0 0 {aperture_start_us - 0.02:.2f}u 0 "
            f"{aperture_start_us:.2f}u 1.8 {end_us:.2f}u 1.8 {off_us:.2f}u 0 7.8u 0)"
        )
        pactn_pwl = (
            f"VPACTN_HYR pactn_hyr 0 PWL(0 1.8 {aperture_start_us - 0.02:.2f}u 1.8 "
            f"{aperture_start_us:.2f}u 0 {end_us:.2f}u 0 {off_us:.2f}u 1.8 7.8u 1.8)"
        )
        reads_pwl = "VREADS_HYR reads_hyr 0 PWL(0 0 2.70u 0 2.72u 1.8 3.36u 1.8 3.38u 0 7.8u 0)"
        readsn_pwl = "VREADSN_HYR readsn_hyr 0 PWL(0 1.8 2.70u 1.8 2.72u 0 3.36u 0 3.38u 1.8 7.8u 1.8)"
        read_gated_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        read_gated_deck = replace_required(
            read_gated_deck,
            timing_base_pact_pwl,
            pact_pwl + "\n" + pactn_pwl + "\n" + reads_pwl + "\n" + readsn_pwl,
        )
        read_gated_deck = replace_required(read_gated_deck, nmos_store_line_p, read_gated_store_line_p)
        read_gated_deck = replace_required(read_gated_deck, nmos_store_line_m, read_gated_store_line_m)
        read_gated_deck = replace_required(
            read_gated_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        read_gated_data = run_ngspice(read_gated_deck, stem)
        rgt, rg_cols = load_wrdata(read_gated_data, 23)

        def rgat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rgt - time_s))])

        rg_preact = rg_cols[10] - rg_cols[9]
        rg_store = rg_cols[14] - rg_cols[13]
        read_gated_preact_samples.append(rgat(sample_time, rg_preact))
        read_gated_store_samples.append(rgat(sample_time, rg_store))
        read_gated_traces.append((f"{width_ns} ns", rgt, rg_store))

    read_gated_preact_samples = np.array(read_gated_preact_samples)
    read_gated_store_samples = np.array(read_gated_store_samples)
    guard_gated_store_samples = []
    guard_gated_preact_samples = []
    guard_gated_traces = []

    def make_guard_store_lines(
        guard_n_width_u: float = 96.0,
        guard_p_width_u: float = 240.0,
        guard_n_model: str = "NMOS",
        guard_p_model: str = "PMOS",
    ) -> tuple[str, str]:
        guard_n_width = f"{guard_n_width_u:g}u"
        guard_p_width = f"{guard_p_width_u:g}u"
        store_p = "\n".join(
            [
                "MSFP_HYR hp_hyr pact_hyr hp_mid_hyr 0 NMOS L={LCH} W=96u",
                "MSFPP_HYR hp_hyr pactn_hyr hp_mid_hyr vdd PMOS L={LCH} W=240u",
                f"MSFPV_HYR hp_mid_hyr guard_hyr hp_fwd_hyr 0 {guard_n_model} L={{LCH}} W={guard_n_width}",
                f"MSFPVP_HYR hp_mid_hyr guardn_hyr hp_fwd_hyr vdd {guard_p_model} L={{LCH}} W={guard_p_width}",
            ]
        )
        store_m = "\n".join(
            [
                "MSFM_HYR hm_hyr pact_hyr hm_mid_hyr 0 NMOS L={LCH} W=96u",
                "MSFMP_HYR hm_hyr pactn_hyr hm_mid_hyr vdd PMOS L={LCH} W=240u",
                f"MSFMV_HYR hm_mid_hyr guard_hyr hm_fwd_hyr 0 {guard_n_model} L={{LCH}} W={guard_n_width}",
                f"MSFMVP_HYR hm_mid_hyr guardn_hyr hm_fwd_hyr vdd {guard_p_model} L={{LCH}} W={guard_p_width}",
            ]
        )
        return store_p, store_m

    guard_store_line_p, guard_store_line_m = make_guard_store_lines()
    for width_ns in aperture_cases_ns:
        stem = f"mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_guarded_tg_aperture_{width_ns}ns"
        end_us = aperture_start_us + width_ns / 1000.0
        off_us = end_us + 0.02
        sample_time = min((off_us + 0.08) * 1e-6, 3.575e-6)
        pact_pwl = (
            f"VPACT_HYR pact_hyr 0 PWL(0 0 {aperture_start_us - 0.02:.2f}u 0 "
            f"{aperture_start_us:.2f}u 1.8 {end_us:.2f}u 1.8 {off_us:.2f}u 0 7.8u 0)"
        )
        pactn_pwl = (
            f"VPACTN_HYR pactn_hyr 0 PWL(0 1.8 {aperture_start_us - 0.02:.2f}u 1.8 "
            f"{aperture_start_us:.2f}u 0 {end_us:.2f}u 0 {off_us:.2f}u 1.8 7.8u 1.8)"
        )
        guard_pwl = "VGUARD_HYR guard_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 7.8u 0)"
        guardn_pwl = "VGUARDN_HYR guardn_hyr 0 PWL(0 1.8 3.16u 1.8 3.18u 0 3.31u 0 3.33u 1.8 7.8u 1.8)"
        guard_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        guard_deck = replace_required(
            guard_deck,
            timing_base_pact_pwl,
            pact_pwl + "\n" + pactn_pwl + "\n" + guard_pwl + "\n" + guardn_pwl,
        )
        guard_deck = replace_required(guard_deck, nmos_store_line_p, guard_store_line_p)
        guard_deck = replace_required(guard_deck, nmos_store_line_m, guard_store_line_m)
        guard_deck = replace_required(
            guard_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        guard_data = run_ngspice(guard_deck, stem)
        gtime, guard_cols = load_wrdata(guard_data, 23)

        def gat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(gtime - time_s))])

        guard_preact = guard_cols[10] - guard_cols[9]
        guard_store = guard_cols[14] - guard_cols[13]
        guard_gated_preact_samples.append(gat(sample_time, guard_preact))
        guard_gated_store_samples.append(gat(sample_time, guard_store))
        guard_gated_traces.append((f"{width_ns} ns", gtime, guard_store))

    guard_gated_preact_samples = np.array(guard_gated_preact_samples)
    guard_gated_store_samples = np.array(guard_gated_store_samples)
    require(read_gated_store_samples[2] > 0.045, "read-gated TG should capture a useful 120 ns activation")
    require(
        read_gated_store_samples[4] > tg_aperture_store_samples[4] + 0.005,
        "read-gated TG should reduce 240 ns post-read droop",
    )
    require(
        read_gated_store_samples[5] > tg_aperture_store_samples[5] + 0.003,
        "read-gated TG should reduce 320 ns post-read droop",
    )
    require(
        read_gated_store_samples[5] > 0.75 * read_gated_store_samples[2],
        "read-gated TG should keep long pact apertures above the ungated droop floor",
    )
    require(np.min(read_gated_preact_samples[1:]) > 0.048, "read-gated TG comparison should keep valid read state")
    require(guard_gated_store_samples[2] > 0.050, "guarded TG should capture a full 120 ns activation")
    require(
        guard_gated_store_samples[4] > read_gated_store_samples[4] + 0.008,
        "guarded TG should improve 240 ns capture beyond raw read-gating",
    )
    require(
        guard_gated_store_samples[5] > read_gated_store_samples[5] + 0.008,
        "guarded TG should improve 320 ns capture beyond raw read-gating",
    )
    require(
        guard_gated_store_samples[5] > 0.90 * guard_gated_store_samples[2],
        "guarded TG should hold long apertures near the clipped valid-window sample",
    )
    require(np.min(guard_gated_preact_samples[1:]) > 0.048, "guarded TG comparison should keep valid read state")

    guard_end_cases_us = [3.24, 3.28, 3.31, 3.33, 3.36, 3.40]
    guard_timing_labels = []
    guard_timing_samples = []
    guard_timing_preact_samples = []
    guard_timing_traces = []
    long_pact_end_us = aperture_start_us + 320 / 1000.0
    long_pact_off_us = long_pact_end_us + 0.02
    long_pact_pwl = (
        f"VPACT_HYR pact_hyr 0 PWL(0 0 {aperture_start_us - 0.02:.2f}u 0 "
        f"{aperture_start_us:.2f}u 1.8 {long_pact_end_us:.2f}u 1.8 {long_pact_off_us:.2f}u 0 7.8u 0)"
    )
    long_pactn_pwl = (
        f"VPACTN_HYR pactn_hyr 0 PWL(0 1.8 {aperture_start_us - 0.02:.2f}u 1.8 "
        f"{aperture_start_us:.2f}u 0 {long_pact_end_us:.2f}u 0 {long_pact_off_us:.2f}u 1.8 7.8u 1.8)"
    )

    def guard_phase_lines(start_us: float, end_us: float, edge_ns: float = 20.0) -> tuple[str, str]:
        edge_us = edge_ns / 1000.0
        rise_start_us = start_us - edge_us
        fall_end_us = end_us + edge_us
        guard_pwl = (
            f"VGUARD_HYR guard_hyr 0 PWL(0 0 {rise_start_us:.3f}u 0 "
            f"{start_us:.3f}u 1.8 {end_us:.3f}u 1.8 {fall_end_us:.3f}u 0 7.8u 0)"
        )
        guardn_pwl = (
            f"VGUARDN_HYR guardn_hyr 0 PWL(0 1.8 {rise_start_us:.3f}u 1.8 "
            f"{start_us:.3f}u 0 {end_us:.3f}u 0 {fall_end_us:.3f}u 1.8 7.8u 1.8)"
        )
        return guard_pwl, guardn_pwl

    def read_phase_line(fall_start_us: float, edge_ns: float = 20.0) -> str:
        edge_us = edge_ns / 1000.0
        fall_end_us = fall_start_us + edge_us
        return (
            f"VREAD_HYR read_hyr 0 PWL(0 0 2.700u 0 2.720u 1.15 "
            f"{fall_start_us:.3f}u 1.15 {fall_end_us:.3f}u 0 7.8u 0)"
        )

    def read_window_line(rise_end_us: float, fall_start_us: float = 3.36, edge_ns: float = 20.0) -> str:
        edge_us = edge_ns / 1000.0
        rise_start_us = rise_end_us - edge_us
        fall_end_us = fall_start_us + edge_us
        return (
            f"VREAD_HYR read_hyr 0 PWL(0 0 {rise_start_us:.3f}u 0 "
            f"{rise_end_us:.3f}u 1.15 {fall_start_us:.3f}u 1.15 {fall_end_us:.3f}u 0 7.8u 0)"
        )

    def pact_slew_lines(edge_ns: float, rise_start_us: float = 3.16, fall_start_us: float = 3.50) -> tuple[str, str]:
        edge_us = edge_ns / 1000.0
        rise_end_us = rise_start_us + edge_us
        fall_end_us = fall_start_us + edge_us
        pact_pwl = (
            f"VPACT_HYR pact_hyr 0 PWL(0 0 {rise_start_us:.3f}u 0 "
            f"{rise_end_us:.3f}u 1.8 {fall_start_us:.3f}u 1.8 {fall_end_us:.3f}u 0 7.8u 0)"
        )
        pactn_pwl = (
            f"VPACTN_HYR pactn_hyr 0 PWL(0 1.8 {rise_start_us:.3f}u 1.8 "
            f"{rise_end_us:.3f}u 0 {fall_start_us:.3f}u 0 {fall_end_us:.3f}u 1.8 7.8u 1.8)"
        )
        return pact_pwl, pactn_pwl

    for guard_end_us in guard_end_cases_us:
        guard_off_us = guard_end_us + 0.02
        label = f"{int(round((guard_end_us - aperture_start_us) * 1000))} ns"
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_timing_{int(round((guard_end_us - aperture_start_us) * 1000))}ns"
        )
        guard_pwl = (
            f"VGUARD_HYR guard_hyr 0 PWL(0 0 {aperture_start_us - 0.02:.2f}u 0 "
            f"{aperture_start_us:.2f}u 1.8 {guard_end_us:.2f}u 1.8 {guard_off_us:.2f}u 0 7.8u 0)"
        )
        guardn_pwl = (
            f"VGUARDN_HYR guardn_hyr 0 PWL(0 1.8 {aperture_start_us - 0.02:.2f}u 1.8 "
            f"{aperture_start_us:.2f}u 0 {guard_end_us:.2f}u 0 {guard_off_us:.2f}u 1.8 7.8u 1.8)"
        )
        timing_guard_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        timing_guard_deck = replace_required(
            timing_guard_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + guard_pwl + "\n" + guardn_pwl,
        )
        timing_guard_deck = replace_required(timing_guard_deck, nmos_store_line_p, guard_store_line_p)
        timing_guard_deck = replace_required(timing_guard_deck, nmos_store_line_m, guard_store_line_m)
        timing_guard_deck = replace_required(
            timing_guard_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        timing_guard_data = run_ngspice(timing_guard_deck, stem)
        gt, guard_timing_cols = load_wrdata(timing_guard_data, 23)

        def gtat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(gt - time_s))])

        guard_timing_preact = guard_timing_cols[10] - guard_timing_cols[9]
        guard_timing_store = guard_timing_cols[14] - guard_timing_cols[13]
        guard_timing_labels.append(label)
        guard_timing_preact_samples.append(gtat(3.575e-6, guard_timing_preact))
        guard_timing_samples.append(gtat(3.575e-6, guard_timing_store))
        guard_timing_traces.append((label, gt, guard_timing_store))

    guard_timing_samples = np.array(guard_timing_samples)
    guard_timing_preact_samples = np.array(guard_timing_preact_samples)
    require(np.min(guard_timing_preact_samples) > 0.048, "guard timing sweep should keep a valid read state")
    require(guard_timing_samples[0] > 0.95 * guard_timing_samples[1], "60 ns guard should already capture most of the valid activation")
    require(
        guard_timing_samples[2] > 0.95 * guard_timing_samples[3],
        "near-nominal guard-off timings should agree",
    )
    require(
        guard_timing_samples[4] < guard_timing_samples[2] - 0.008,
        "guard held past the valid window should begin tracking forward-load droop",
    )
    require(
        guard_timing_samples[-1] < guard_timing_samples[1] - 0.015,
        "late guard-off should start tracking post-valid forward-load droop",
    )

    guard_skew_cases_ns = [-120, -80, -40, 0, 40, 80, 120]
    guard_skew_labels = []
    guard_skew_store_samples = []
    guard_skew_preact_samples = []
    guard_skew_traces = []
    nominal_guard_start_us = aperture_start_us
    guard_width_us = 130 / 1000.0
    for skew_ns in guard_skew_cases_ns:
        guard_start_us = nominal_guard_start_us + skew_ns / 1000.0
        guard_rise_us = guard_start_us - 0.02
        guard_end_us = guard_start_us + guard_width_us
        guard_off_us = guard_end_us + 0.02
        label = f"{skew_ns:+d} ns"
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_skew_{skew_ns:+d}ns".replace("+", "p").replace("-", "m")
        )
        guard_pwl = (
            f"VGUARD_HYR guard_hyr 0 PWL(0 0 {guard_rise_us:.2f}u 0 "
            f"{guard_start_us:.2f}u 1.8 {guard_end_us:.2f}u 1.8 {guard_off_us:.2f}u 0 7.8u 0)"
        )
        guardn_pwl = (
            f"VGUARDN_HYR guardn_hyr 0 PWL(0 1.8 {guard_rise_us:.2f}u 1.8 "
            f"{guard_start_us:.2f}u 0 {guard_end_us:.2f}u 0 {guard_off_us:.2f}u 1.8 7.8u 1.8)"
        )
        skew_guard_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        skew_guard_deck = replace_required(
            skew_guard_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + guard_pwl + "\n" + guardn_pwl,
        )
        skew_guard_deck = replace_required(skew_guard_deck, nmos_store_line_p, guard_store_line_p)
        skew_guard_deck = replace_required(skew_guard_deck, nmos_store_line_m, guard_store_line_m)
        skew_guard_deck = replace_required(
            skew_guard_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        skew_guard_data = run_ngspice(skew_guard_deck, stem)
        gst, guard_skew_cols = load_wrdata(skew_guard_data, 23)

        def gsat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(gst - time_s))])

        guard_skew_preact = guard_skew_cols[10] - guard_skew_cols[9]
        guard_skew_store = guard_skew_cols[14] - guard_skew_cols[13]
        guard_skew_labels.append(label)
        guard_skew_preact_samples.append(gsat(3.575e-6, guard_skew_preact))
        guard_skew_store_samples.append(gsat(3.575e-6, guard_skew_store))
        guard_skew_traces.append((label, gst, guard_skew_store))

    guard_skew_store_samples = np.array(guard_skew_store_samples)
    guard_skew_preact_samples = np.array(guard_skew_preact_samples)
    nominal_skew_idx = guard_skew_cases_ns.index(0)
    require(np.min(guard_skew_preact_samples) > 0.048, "guard skew sweep should keep a valid read state")
    require(
        abs(guard_skew_store_samples[nominal_skew_idx] - guard_timing_samples[2]) < 0.002,
        "nominal guard skew sample should match the nominal timing sample",
    )
    require(guard_skew_store_samples[nominal_skew_idx] > 0.050, "nominal guard skew case should capture a full activation")
    require(
        guard_skew_store_samples[0] < guard_skew_store_samples[nominal_skew_idx] - 0.015,
        "large early guard skew should undercharge the activation store",
    )
    require(
        guard_skew_store_samples[2] > guard_skew_store_samples[nominal_skew_idx] - 0.002,
        "small early guard skew should remain in the useful capture plateau",
    )
    require(
        guard_skew_store_samples[4] < guard_skew_store_samples[nominal_skew_idx] - 0.006,
        "small late guard skew should already begin tracking read-load droop",
    )
    require(
        guard_skew_store_samples[-1] < guard_skew_store_samples[nominal_skew_idx] - 0.010,
        "large late guard skew should track read-path droop",
    )

    guard_edge_cases_ns = [5, 10, 20, 40, 80, 120]
    guard_edge_labels = []
    guard_edge_store_samples = []
    guard_edge_preact_samples = []
    guard_edge_traces = []
    for edge_ns in guard_edge_cases_ns:
        guard_pwl, guardn_pwl = guard_phase_lines(3.18, 3.31, edge_ns)
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_edge_{edge_ns}ns"
        )
        edge_guard_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        edge_guard_deck = replace_required(
            edge_guard_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + guard_pwl + "\n" + guardn_pwl,
        )
        edge_guard_deck = replace_required(edge_guard_deck, nmos_store_line_p, guard_store_line_p)
        edge_guard_deck = replace_required(edge_guard_deck, nmos_store_line_m, guard_store_line_m)
        edge_guard_deck = replace_required(
            edge_guard_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        edge_guard_data = run_ngspice(edge_guard_deck, stem)
        get, guard_edge_cols = load_wrdata(edge_guard_data, 23)

        def geat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(get - time_s))])

        guard_edge_preact = guard_edge_cols[10] - guard_edge_cols[9]
        guard_edge_store = guard_edge_cols[14] - guard_edge_cols[13]
        guard_edge_labels.append(f"{edge_ns} ns")
        guard_edge_preact_samples.append(geat(3.575e-6, guard_edge_preact))
        guard_edge_store_samples.append(geat(3.575e-6, guard_edge_store))
        guard_edge_traces.append((f"{edge_ns} ns", get, guard_edge_store))

    guard_edge_store_samples = np.array(guard_edge_store_samples)
    guard_edge_preact_samples = np.array(guard_edge_preact_samples)
    nominal_edge_idx = guard_edge_cases_ns.index(20)
    require(np.min(guard_edge_preact_samples) > 0.048, "guard edge sweep should keep a valid read state")
    require(guard_edge_store_samples[0] > 0.050, "fast guard edge should capture a full activation")
    require(
        abs(guard_edge_store_samples[nominal_edge_idx] - guard_timing_samples[2]) < 0.002,
        "20 ns guard edge should match the nominal timing sample",
    )
    require(
        guard_edge_store_samples[-1] < guard_edge_store_samples[nominal_edge_idx] - 0.004,
        "slow guard edge should begin tracking the read-path collapse",
    )

    pact_edge_cases_ns = [5, 10, 20, 40, 80, 120]
    pact_edge_labels = []
    pact_edge_store_samples = []
    pact_edge_preact_samples = []
    pact_edge_traces = []
    guard_pwl, guardn_pwl = guard_phase_lines(3.18, 3.31, 20.0)
    for edge_ns in pact_edge_cases_ns:
        pact_pwl, pactn_pwl = pact_slew_lines(edge_ns)
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"pact_edge_{edge_ns}ns"
        )
        pact_edge_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        pact_edge_deck = replace_required(
            pact_edge_deck,
            timing_base_pact_pwl,
            pact_pwl + "\n" + pactn_pwl + "\n" + guard_pwl + "\n" + guardn_pwl,
        )
        pact_edge_deck = replace_required(pact_edge_deck, nmos_store_line_p, guard_store_line_p)
        pact_edge_deck = replace_required(pact_edge_deck, nmos_store_line_m, guard_store_line_m)
        pact_edge_deck = replace_required(
            pact_edge_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        pact_edge_data = run_ngspice(pact_edge_deck, stem)
        pet, pact_edge_cols = load_wrdata(pact_edge_data, 23)

        def peat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(pet - time_s))])

        pact_edge_preact = pact_edge_cols[10] - pact_edge_cols[9]
        pact_edge_store = pact_edge_cols[14] - pact_edge_cols[13]
        pact_edge_labels.append(f"{edge_ns} ns")
        pact_edge_preact_samples.append(peat(3.575e-6, pact_edge_preact))
        pact_edge_store_samples.append(peat(3.575e-6, pact_edge_store))
        pact_edge_traces.append((f"{edge_ns} ns", pet, pact_edge_store, pact_edge_cols[22]))

    pact_edge_store_samples = np.array(pact_edge_store_samples)
    pact_edge_preact_samples = np.array(pact_edge_preact_samples)
    nominal_pact_edge_idx = pact_edge_cases_ns.index(20)
    require(np.min(pact_edge_preact_samples) > 0.048, "pact edge sweep should keep a valid read state")
    require(
        abs(pact_edge_store_samples[nominal_pact_edge_idx] - guard_timing_samples[2]) < 0.002,
        "20 ns pact edge should match the nominal timing sample",
    )
    require(
        np.max(np.abs(pact_edge_store_samples[:3] - pact_edge_store_samples[nominal_pact_edge_idx])) < 0.003,
        "fast pact edges should capture the same activation as the nominal edge",
    )
    require(
        np.all(np.diff(pact_edge_store_samples[2:]) < -0.0002),
        "slower pact rise after nominal should monotonically reduce stored activation",
    )
    require(
        pact_edge_store_samples[-1] < pact_edge_store_samples[nominal_pact_edge_idx] - 0.006,
        "very slow pact edge should visibly undercharge the guarded activation store",
    )

    guard_start_cases_us = [3.10, 3.14, 3.18, 3.22, 3.26, 3.29]
    guard_start_labels = []
    guard_start_store_samples = []
    guard_start_preact_samples = []
    guard_start_traces = []
    for guard_start_us in guard_start_cases_us:
        guard_pwl, guardn_pwl = guard_phase_lines(guard_start_us, 3.31, 20.0)
        label = f"{guard_start_us:.2f} us"
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_start_{int(round(guard_start_us * 1000))}ns"
        )
        start_guard_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        start_guard_deck = replace_required(
            start_guard_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + guard_pwl + "\n" + guardn_pwl,
        )
        start_guard_deck = replace_required(start_guard_deck, nmos_store_line_p, guard_store_line_p)
        start_guard_deck = replace_required(start_guard_deck, nmos_store_line_m, guard_store_line_m)
        start_guard_deck = replace_required(
            start_guard_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        start_guard_data = run_ngspice(start_guard_deck, stem)
        gst, guard_start_cols = load_wrdata(start_guard_data, 23)

        def gsat_start(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(gst - time_s))])

        guard_start_preact = guard_start_cols[10] - guard_start_cols[9]
        guard_start_store = guard_start_cols[14] - guard_start_cols[13]
        guard_start_labels.append(label)
        guard_start_preact_samples.append(gsat_start(3.575e-6, guard_start_preact))
        guard_start_store_samples.append(gsat_start(3.575e-6, guard_start_store))
        guard_start_traces.append((label, gst, guard_start_store))

    guard_start_store_samples = np.array(guard_start_store_samples)
    guard_start_preact_samples = np.array(guard_start_preact_samples)
    nominal_guard_start_idx = guard_start_cases_us.index(3.18)
    require(np.min(guard_start_preact_samples) > 0.048, "guard-start sweep should keep a valid read state")
    require(
        abs(guard_start_store_samples[nominal_guard_start_idx] - guard_timing_samples[2]) < 0.002,
        "nominal guard start should match the nominal guard timing sample",
    )
    require(
        guard_start_store_samples[0] > guard_start_store_samples[nominal_guard_start_idx] - 0.002,
        "early guard start should remain in the valid capture plateau when pact still gates the store",
    )
    require(
        guard_start_store_samples[-1] < guard_start_store_samples[nominal_guard_start_idx] - 0.020,
        "very late guard start should undercharge the activation store",
    )
    require(
        np.all(np.diff(guard_start_store_samples[2:]) < -0.002),
        "guard starts after nominal should monotonically reduce the stored activation",
    )

    read_fall_cases_us = [3.26, 3.30, 3.33, 3.36, 3.40, 3.44]
    read_fall_labels = []
    read_fall_store_samples = []
    read_fall_preact_samples = []
    read_fall_traces = []
    guard_pwl, guardn_pwl = guard_phase_lines(3.18, 3.31, 20.0)
    for read_fall_us in read_fall_cases_us:
        read_pwl = read_phase_line(read_fall_us, 20.0)
        label = f"{read_fall_us:.2f} us"
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"read_fall_{int(round(read_fall_us * 1000))}ns"
        )
        read_fall_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, read_pwl)
        read_fall_deck = replace_required(
            read_fall_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + guard_pwl + "\n" + guardn_pwl,
        )
        read_fall_deck = replace_required(read_fall_deck, nmos_store_line_p, guard_store_line_p)
        read_fall_deck = replace_required(read_fall_deck, nmos_store_line_m, guard_store_line_m)
        read_fall_deck = replace_required(
            read_fall_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        read_fall_data = run_ngspice(read_fall_deck, stem)
        rft, read_fall_cols = load_wrdata(read_fall_data, 23)

        def rfat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rft - time_s))])

        read_fall_preact = read_fall_cols[10] - read_fall_cols[9]
        read_fall_store = read_fall_cols[14] - read_fall_cols[13]
        read_fall_labels.append(label)
        read_fall_preact_samples.append(rfat(3.575e-6, read_fall_preact))
        read_fall_store_samples.append(rfat(3.575e-6, read_fall_store))
        read_fall_traces.append((label, rft, read_fall_store, read_fall_cols[21]))

    read_fall_store_samples = np.array(read_fall_store_samples)
    read_fall_preact_samples = np.array(read_fall_preact_samples)
    nominal_read_fall_idx = read_fall_cases_us.index(3.36)
    require(read_fall_store_samples[nominal_read_fall_idx] > 0.050, "nominal read-fall timing should capture a full activation")
    require(
        read_fall_store_samples[0] > read_fall_store_samples[nominal_read_fall_idx] + 0.004,
        "read-valid falling well before the guard closes should perturb the activation store",
    )
    require(
        np.max(np.abs(read_fall_store_samples[2:] - read_fall_store_samples[nominal_read_fall_idx])) < 0.001,
        "late read fall should not materially change a guard-disconnected store",
    )

    read_rise_cases_us = [2.72, 3.00, 3.10, 3.18, 3.24, 3.28]
    read_rise_labels = []
    read_rise_store_samples = []
    read_rise_preact_samples = []
    read_rise_traces = []
    guard_pwl, guardn_pwl = guard_phase_lines(3.18, 3.31, 20.0)
    for read_rise_us in read_rise_cases_us:
        read_pwl = read_window_line(read_rise_us, 3.36, 20.0)
        label = f"{read_rise_us:.2f} us"
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"read_rise_{int(round(read_rise_us * 1000))}ns"
        )
        read_rise_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, read_pwl)
        read_rise_deck = replace_required(
            read_rise_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + guard_pwl + "\n" + guardn_pwl,
        )
        read_rise_deck = replace_required(read_rise_deck, nmos_store_line_p, guard_store_line_p)
        read_rise_deck = replace_required(read_rise_deck, nmos_store_line_m, guard_store_line_m)
        read_rise_deck = replace_required(
            read_rise_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        read_rise_data = run_ngspice(read_rise_deck, stem)
        rrt, read_rise_cols = load_wrdata(read_rise_data, 23)

        def rrat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rrt - time_s))])

        read_rise_preact = read_rise_cols[10] - read_rise_cols[9]
        read_rise_store = read_rise_cols[14] - read_rise_cols[13]
        read_rise_labels.append(label)
        read_rise_preact_samples.append(rrat(3.575e-6, read_rise_preact))
        read_rise_store_samples.append(rrat(3.575e-6, read_rise_store))
        read_rise_traces.append((label, rrt, read_rise_store, read_rise_cols[21]))

    read_rise_store_samples = np.array(read_rise_store_samples)
    read_rise_preact_samples = np.array(read_rise_preact_samples)
    nominal_read_rise_idx = read_rise_cases_us.index(2.72)
    require(read_rise_store_samples[nominal_read_rise_idx] > 0.050, "nominal read-rise timing should capture a full activation")
    require(
        np.all(np.diff(read_rise_store_samples) < -0.002),
        "later read rise should monotonically reduce the stored activation",
    )
    require(
        read_rise_store_samples[-1] < 0.010,
        "read rising near the end of the guard window should undercharge the activation store",
    )

    hold_reset_line = (
        "VRESET_HYR rst_hyr 0 PWL(0 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 "
        "5.18u 0 5.20u 1.8 5.60u 1.8 5.62u 0 7.8u 0)"
    )
    hold_resetn_line = (
        "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 "
        "5.18u 1.8 5.20u 0 5.60u 0 5.62u 1.8 7.8u 1.8)"
    )
    hold_guard_pwl = "VGUARD_HYR guard_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 7.8u 0)"
    hold_guardn_pwl = "VGUARDN_HYR guardn_hyr 0 PWL(0 1.8 3.16u 1.8 3.18u 0 3.31u 0 3.33u 1.8 7.8u 1.8)"
    hold_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
    hold_deck = replace_required(
        hold_deck,
        timing_base_pact_pwl,
        long_pact_pwl + "\n" + long_pactn_pwl + "\n" + hold_guard_pwl + "\n" + hold_guardn_pwl,
    )
    hold_deck = replace_required(hold_deck, hold_reset_line, "VRESET_HYR rst_hyr 0 0")
    hold_deck = replace_required(hold_deck, hold_resetn_line, "VRESETN_HYR rstn_hyr 0 1.8")
    hold_deck = replace_required(hold_deck, nmos_store_line_p, guard_store_line_p)
    hold_deck = replace_required(hold_deck, nmos_store_line_m, guard_store_line_m)
    hold_deck = replace_required(
        hold_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_hold.dat",
    )
    hold_data = run_ngspice(hold_deck, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_hold")
    ht, hold_cols = load_wrdata(hold_data, 23)

    def ghat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(ht - time_s))])

    hold_preact = hold_cols[10] - hold_cols[9]
    hold_store = hold_cols[14] - hold_cols[13]
    hold_sample_times = np.array([3.575, 4.50, 5.50, 6.50, 7.45]) * 1e-6
    hold_store_samples = np.array([ghat(ts, hold_store) for ts in hold_sample_times])
    hold_preact_samples = np.array([ghat(ts, hold_preact) for ts in hold_sample_times])
    hold_drift = hold_store_samples - hold_store_samples[0]
    require(hold_store_samples[0] > 0.050, "guard hold deck should capture a full activation before hold")
    require(np.min(hold_preact_samples) > 0.048, "guard hold deck should keep the trained preactivation available")
    require(
        np.max(np.abs(hold_drift)) < 0.0002,
        "guarded activation store should hold within 0.2 mV over the no-reset interval",
    )

    stress_read_pwl = (
        "VREAD_HYR read_hyr 0 PWL(0 0 2.70u 0 2.72u 1.15 3.36u 1.15 3.38u 0 "
        "4.30u 0 4.32u 1.15 4.96u 1.15 4.98u 0 "
        "5.90u 0 5.92u 1.15 6.56u 1.15 6.58u 0 7.8u 0)"
    )
    stress_pact_pwl = (
        "VPACT_HYR pact_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.50u 1.8 3.52u 0 "
        "4.76u 0 4.78u 1.8 4.91u 1.8 4.93u 0 "
        "6.36u 0 6.38u 1.8 6.51u 1.8 6.53u 0 7.8u 0)"
    )
    stress_pactn_pwl = (
        "VPACTN_HYR pactn_hyr 0 PWL(0 1.8 3.16u 1.8 3.18u 0 3.50u 0 3.52u 1.8 "
        "4.76u 1.8 4.78u 0 4.91u 0 4.93u 1.8 "
        "6.36u 1.8 6.38u 0 6.51u 0 6.53u 1.8 7.8u 1.8)"
    )
    stress_guard_pwl = (
        "VGUARD_HYR guard_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 "
        "5.20u 0 5.22u 1.8 5.35u 1.8 5.37u 0 "
        "5.98u 0 6.00u 1.8 6.13u 1.8 6.15u 0 7.8u 0)"
    )
    stress_guardn_pwl = (
        "VGUARDN_HYR guardn_hyr 0 PWL(0 1.8 3.16u 1.8 3.18u 0 3.31u 0 3.33u 1.8 "
        "5.20u 1.8 5.22u 0 5.35u 0 5.37u 1.8 "
        "5.98u 1.8 6.00u 0 6.13u 0 6.15u 1.8 7.8u 1.8)"
    )
    stress_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, stress_read_pwl)
    stress_deck = replace_required(
        stress_deck,
        timing_base_pact_pwl,
        stress_pact_pwl + "\n" + stress_pactn_pwl + "\n" + stress_guard_pwl + "\n" + stress_guardn_pwl,
    )
    stress_deck = replace_required(stress_deck, hold_reset_line, "VRESET_HYR rst_hyr 0 0")
    stress_deck = replace_required(stress_deck, hold_resetn_line, "VRESETN_HYR rstn_hyr 0 1.8")
    stress_deck = replace_required(stress_deck, nmos_store_line_p, guard_store_line_p)
    stress_deck = replace_required(stress_deck, nmos_store_line_m, guard_store_line_m)
    stress_deck = replace_required(
        stress_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_off_isolation.dat",
    )
    stress_data = run_ngspice(stress_deck, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_off_isolation")
    got, guard_off_cols = load_wrdata(stress_data, 23)

    def goiat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(got - time_s))])

    guard_off_preact = guard_off_cols[10] - guard_off_cols[9]
    guard_off_store = guard_off_cols[14] - guard_off_cols[13]
    guard_off_sample_times = np.array([3.575, 4.50, 5.05, 5.50, 6.25, 6.75, 7.45]) * 1e-6
    guard_off_store_samples = np.array([goiat(ts, guard_off_store) for ts in guard_off_sample_times])
    guard_off_preact_samples = np.array([goiat(ts, guard_off_preact) for ts in guard_off_sample_times])
    guard_off_drift = guard_off_store_samples - guard_off_store_samples[0]
    require(guard_off_store_samples[0] > 0.050, "guard off-isolation deck should capture a full activation")
    require(
        np.max(np.abs(guard_off_drift)) < 0.0005,
        "off-state control/read toggles should not disturb guarded activation store by more than 0.5 mV",
    )

    negative_hold_deck = replace_required(hybrid_forward_negative_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
    negative_hold_deck = replace_required(
        negative_hold_deck,
        timing_base_pact_pwl,
        long_pact_pwl + "\n" + long_pactn_pwl + "\n" + hold_guard_pwl + "\n" + hold_guardn_pwl,
    )
    negative_hold_deck = replace_required(negative_hold_deck, hold_reset_line, "VRESET_HYR rst_hyr 0 0")
    negative_hold_deck = replace_required(negative_hold_deck, hold_resetn_line, "VRESETN_HYR rstn_hyr 0 1.8")
    negative_hold_deck = replace_required(negative_hold_deck, nmos_store_line_p, guard_store_line_p)
    negative_hold_deck = replace_required(negative_hold_deck, nmos_store_line_m, guard_store_line_m)
    negative_hold_deck = replace_required(
        negative_hold_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_negative_read_reuse.dat",
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_signed_hold_negative.dat",
    )
    negative_hold_data = run_ngspice(
        negative_hold_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_signed_hold_negative",
    )
    nght, negative_hold_cols = load_wrdata(negative_hold_data, 23)

    def nghat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(nght - time_s))])

    negative_hold_preact = negative_hold_cols[10] - negative_hold_cols[9]
    negative_hold_store = negative_hold_cols[14] - negative_hold_cols[13]
    negative_hold_store_samples = np.array([nghat(ts, negative_hold_store) for ts in hold_sample_times])
    negative_hold_preact_samples = np.array([nghat(ts, negative_hold_preact) for ts in hold_sample_times])
    negative_hold_drift = negative_hold_store_samples - negative_hold_store_samples[0]
    signed_hold_mirror_error = hold_store_samples + negative_hold_store_samples
    require(negative_hold_store_samples[0] < -0.050, "negative guarded hold deck should capture a full negative activation")
    require(np.max(np.abs(signed_hold_mirror_error)) < 0.0002, "positive and negative guarded holds should mirror within 0.2 mV")
    require(
        np.max(np.abs(negative_hold_drift)) < 0.0002,
        "negative guarded activation store should hold within 0.2 mV over the no-reset interval",
    )
    require(
        np.max(np.abs(hold_preact_samples + negative_hold_preact_samples)) < 0.002,
        "positive and negative guarded hold preactivations should mirror",
    )

    guard_corner_cases = [
        ("strong", 0.50, -0.50, "strong"),
        ("nominal", 0.55, -0.55, "nominal"),
        ("nweak", 0.64, -0.55, "N weak"),
        ("pweak", 0.55, -0.64, "P weak"),
        ("bothweak", 0.64, -0.64, "both weak"),
    ]
    guard_corner_store_samples = []
    guard_corner_preact_samples = []
    guard_corner_traces = []
    corner_guard_pwl = "VGUARD_HYR guard_hyr 0 PWL(0 0 3.16u 0 3.18u 1.8 3.31u 1.8 3.33u 0 7.8u 0)"
    corner_guardn_pwl = "VGUARDN_HYR guardn_hyr 0 PWL(0 1.8 3.16u 1.8 3.18u 0 3.31u 0 3.33u 1.8 7.8u 1.8)"
    corner_param_line = ".param CERR=10p CWRITE=500p CBIAS=500p CSTORE=10p CSUM=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u WREAD=24u WRESETN=60u WRESETP=180u"
    for name, n_vto, p_vto, label in guard_corner_cases:
        stem = f"mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_guard_corner_{name}"
        n_model = f"NGUARD_{name.upper()}"
        p_model = f"PGUARD_{name.upper()}"
        corner_models = (
            f".model {n_model} NMOS (LEVEL=1 VTO={n_vto:.2f} KP=220u LAMBDA=0.03)\n"
            f".model {p_model} PMOS (LEVEL=1 VTO={p_vto:.2f} KP=90u LAMBDA=0.03)"
        )
        corner_store_line_p, corner_store_line_m = make_guard_store_lines(
            guard_n_model=n_model,
            guard_p_model=p_model,
        )
        corner_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        corner_deck = replace_required(
            corner_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + corner_guard_pwl + "\n" + corner_guardn_pwl,
        )
        corner_deck = replace_required(corner_deck, corner_param_line, corner_models + "\n" + corner_param_line)
        corner_deck = replace_required(corner_deck, nmos_store_line_p, corner_store_line_p)
        corner_deck = replace_required(corner_deck, nmos_store_line_m, corner_store_line_m)
        corner_deck = replace_required(
            corner_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        corner_data = run_ngspice(corner_deck, stem)
        ct, corner_cols = load_wrdata(corner_data, 23)

        def gcat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(ct - time_s))])

        corner_preact = corner_cols[10] - corner_cols[9]
        corner_store = corner_cols[14] - corner_cols[13]
        guard_corner_preact_samples.append(gcat(3.575e-6, corner_preact))
        guard_corner_store_samples.append(gcat(3.575e-6, corner_store))
        guard_corner_traces.append((label, ct, corner_store))

    guard_corner_preact_samples = np.array(guard_corner_preact_samples)
    guard_corner_store_samples = np.array(guard_corner_store_samples)
    require(np.min(guard_corner_preact_samples) > 0.048, "guard corner sweep should keep a valid read state")
    require(np.all(guard_corner_store_samples > 0.035), "guard pass corners should retain positive stored activation")
    require(np.all(guard_corner_store_samples < 0.060), "guard pass corners should remain bounded and incremental")
    require(
        abs(guard_corner_store_samples[1] - guard_timing_samples[2]) < 0.002,
        "nominal guard corner should match the nominal timing sample",
    )
    require(
        np.max(guard_corner_store_samples) - np.min(guard_corner_store_samples) < 0.0008,
        "oversized guard pass gates should keep threshold-corner variation below 0.8 mV",
    )

    guard_size_cases = [
        (0.125, 12.0, 30.0),
        (0.25, 24.0, 60.0),
        (0.50, 48.0, 120.0),
        (1.00, 96.0, 240.0),
        (1.50, 144.0, 360.0),
    ]
    guard_size_labels = []
    guard_size_store_samples = []
    guard_size_preact_samples = []
    guard_size_traces = []
    for scale, n_width_u, p_width_u in guard_size_cases:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_size_{str(scale).replace('.', 'p')}x"
        )
        sized_store_line_p, sized_store_line_m = make_guard_store_lines(
            guard_n_width_u=n_width_u,
            guard_p_width_u=p_width_u,
        )
        size_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        size_deck = replace_required(
            size_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + corner_guard_pwl + "\n" + corner_guardn_pwl,
        )
        size_deck = replace_required(size_deck, nmos_store_line_p, sized_store_line_p)
        size_deck = replace_required(size_deck, nmos_store_line_m, sized_store_line_m)
        size_deck = replace_required(
            size_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        size_data = run_ngspice(size_deck, stem)
        st, size_cols = load_wrdata(size_data, 23)

        def gsizat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(st - time_s))])

        size_preact = size_cols[10] - size_cols[9]
        size_store = size_cols[14] - size_cols[13]
        guard_size_labels.append(f"{scale:g}x")
        guard_size_preact_samples.append(gsizat(3.575e-6, size_preact))
        guard_size_store_samples.append(gsizat(3.575e-6, size_store))
        guard_size_traces.append((f"{scale:g}x", st, size_store))

    guard_size_preact_samples = np.array(guard_size_preact_samples)
    guard_size_store_samples = np.array(guard_size_store_samples)
    require(np.min(guard_size_preact_samples) > 0.048, "guard sizing sweep should keep a valid read state")
    require(guard_size_store_samples[0] > 0.045, "smallest guard TG should still capture a signed activation")
    require(
        guard_size_store_samples[-1] - guard_size_store_samples[0] < 0.010,
        "guard TG sizing should not be the dominant capture limiter above 0.125x",
    )

    guard_cstore_cases_pf = [2, 5, 10, 20, 50, 100]
    guard_cstore_labels = []
    guard_cstore_store_samples = []
    guard_cstore_preact_samples = []
    guard_cstore_traces = []
    for cstore_pf in guard_cstore_cases_pf:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_cstore_{cstore_pf}p"
        )
        cstore_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        cstore_deck = replace_required(
            cstore_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + corner_guard_pwl + "\n" + corner_guardn_pwl,
        )
        cstore_deck = replace_required(cstore_deck, corner_param_line, corner_param_line.replace("CSTORE=10p", f"CSTORE={cstore_pf}p"))
        cstore_deck = replace_required(cstore_deck, nmos_store_line_p, guard_store_line_p)
        cstore_deck = replace_required(cstore_deck, nmos_store_line_m, guard_store_line_m)
        cstore_deck = replace_required(
            cstore_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        cstore_data = run_ngspice(cstore_deck, stem)
        cst, cstore_cols = load_wrdata(cstore_data, 23)

        def gcstat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(cst - time_s))])

        cstore_preact = cstore_cols[10] - cstore_cols[9]
        cstore_store = cstore_cols[14] - cstore_cols[13]
        guard_cstore_labels.append(f"{cstore_pf} pF")
        guard_cstore_preact_samples.append(gcstat(3.575e-6, cstore_preact))
        guard_cstore_store_samples.append(gcstat(3.575e-6, cstore_store))
        guard_cstore_traces.append((f"{cstore_pf} pF", cst, cstore_store))

    guard_cstore_preact_samples = np.array(guard_cstore_preact_samples)
    guard_cstore_store_samples = np.array(guard_cstore_store_samples)
    nominal_cstore_idx = guard_cstore_cases_pf.index(10)
    require(np.min(guard_cstore_preact_samples) > 0.048, "CSTORE sweep should keep a valid read state")
    require(
        abs(guard_cstore_store_samples[nominal_cstore_idx] - guard_timing_samples[2]) < 0.002,
        "10 pF guarded store should match the nominal timing sample",
    )
    require(guard_cstore_store_samples[0] > 0.050, "small activation store cap should capture a full signed activation")
    require(
        guard_cstore_store_samples[0] > guard_cstore_preact_samples[0] + 0.005,
        "very small activation store cap should expose switch-feedthrough overshoot",
    )
    require(
        guard_cstore_store_samples[-1] < guard_cstore_store_samples[nominal_cstore_idx] - 0.010,
        "large activation store cap should visibly undercharge in the fixed guard window",
    )
    require(
        np.all(np.diff(guard_cstore_store_samples[2:]) < -0.001),
        "larger activation store caps should monotonically reduce the sampled activation after nominal sizing",
    )

    guard_control_swing_cases_v = [1.8, 1.6, 1.4, 1.2, 1.0, 0.8]
    guard_control_swing_labels = []
    guard_control_swing_store_samples = []
    guard_control_swing_preact_samples = []
    guard_control_swing_traces = []

    def guard_phase_swing_lines(start_us: float, end_us: float, swing_v: float, edge_ns: float = 20.0) -> tuple[str, str]:
        edge_us = edge_ns / 1000.0
        rise_start_us = start_us - edge_us
        fall_end_us = end_us + edge_us
        comp_low_v = 1.8 - swing_v
        guard_pwl = (
            f"VGUARD_HYR guard_hyr 0 PWL(0 0 {rise_start_us:.3f}u 0 "
            f"{start_us:.3f}u {swing_v:.3f} {end_us:.3f}u {swing_v:.3f} "
            f"{fall_end_us:.3f}u 0 7.8u 0)"
        )
        guardn_pwl = (
            f"VGUARDN_HYR guardn_hyr 0 PWL(0 1.8 {rise_start_us:.3f}u 1.8 "
            f"{start_us:.3f}u {comp_low_v:.3f} {end_us:.3f}u {comp_low_v:.3f} "
            f"{fall_end_us:.3f}u 1.8 7.8u 1.8)"
        )
        return guard_pwl, guardn_pwl

    def pact_swing_lines(swing_v: float, rise_start_us: float = 3.16, fall_start_us: float = 3.50, edge_ns: float = 20.0) -> tuple[str, str]:
        edge_us = edge_ns / 1000.0
        rise_end_us = rise_start_us + edge_us
        fall_end_us = fall_start_us + edge_us
        comp_low_v = 1.8 - swing_v
        pact_pwl = (
            f"VPACT_HYR pact_hyr 0 PWL(0 0 {rise_start_us:.3f}u 0 "
            f"{rise_end_us:.3f}u {swing_v:.3f} {fall_start_us:.3f}u {swing_v:.3f} "
            f"{fall_end_us:.3f}u 0 7.8u 0)"
        )
        pactn_pwl = (
            f"VPACTN_HYR pactn_hyr 0 PWL(0 1.8 {rise_start_us:.3f}u 1.8 "
            f"{rise_end_us:.3f}u {comp_low_v:.3f} {fall_start_us:.3f}u {comp_low_v:.3f} "
            f"{fall_end_us:.3f}u 1.8 7.8u 1.8)"
        )
        return pact_pwl, pactn_pwl

    for swing_v in guard_control_swing_cases_v:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_control_swing_{str(swing_v).replace('.', 'p')}v"
        )
        swing_pact_pwl, swing_pactn_pwl = pact_swing_lines(swing_v)
        swing_guard_pwl, swing_guardn_pwl = guard_phase_swing_lines(3.18, 3.31, swing_v)
        swing_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        swing_deck = replace_required(
            swing_deck,
            timing_base_pact_pwl,
            swing_pact_pwl + "\n" + swing_pactn_pwl + "\n" + swing_guard_pwl + "\n" + swing_guardn_pwl,
        )
        swing_deck = replace_required(swing_deck, nmos_store_line_p, guard_store_line_p)
        swing_deck = replace_required(swing_deck, nmos_store_line_m, guard_store_line_m)
        swing_deck = replace_required(
            swing_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        swing_data = run_ngspice(swing_deck, stem)
        swt, swing_cols = load_wrdata(swing_data, 23)

        def gcsat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(swt - time_s))])

        swing_preact = swing_cols[10] - swing_cols[9]
        swing_store = swing_cols[14] - swing_cols[13]
        guard_control_swing_labels.append(f"{swing_v:.1f} V")
        guard_control_swing_preact_samples.append(gcsat(3.575e-6, swing_preact))
        guard_control_swing_store_samples.append(gcsat(3.575e-6, swing_store))
        guard_control_swing_traces.append((f"{swing_v:.1f} V", swt, swing_store))

    guard_control_swing_preact_samples = np.array(guard_control_swing_preact_samples)
    guard_control_swing_store_samples = np.array(guard_control_swing_store_samples)
    nominal_control_swing_idx = guard_control_swing_cases_v.index(1.8)
    require(np.min(guard_control_swing_preact_samples) > 0.048, "control-swing sweep should keep a valid read state")
    require(
        abs(guard_control_swing_store_samples[nominal_control_swing_idx] - guard_timing_samples[2]) < 0.002,
        "full-swing control case should match the nominal timing sample",
    )
    require(
        np.min(guard_control_swing_store_samples[:3]) > guard_control_swing_store_samples[nominal_control_swing_idx] - 0.004,
        "1.4 V and stronger control swings should retain the nominal capture window",
    )
    require(
        guard_control_swing_store_samples[3] > guard_control_swing_preact_samples[3] + 0.004,
        "marginal 1.2 V control swing should expose pass-stack feedthrough overshoot",
    )
    require(
        guard_control_swing_store_samples[-1] < guard_control_swing_store_samples[nominal_control_swing_idx] - 0.010,
        "weak control swing should visibly undercharge the activation store",
    )

    read_drive_cases_v = [0.75, 0.90, 1.05, 1.15, 1.30, 1.45]
    read_drive_labels = []
    read_drive_store_samples = []
    read_drive_preact_samples = []
    read_drive_traces = []

    def read_drive_line(drive_v: float) -> str:
        return (
            f"VREAD_HYR read_hyr 0 PWL(0 0 2.700u 0 2.720u {drive_v:.3f} "
            f"3.360u {drive_v:.3f} 3.380u 0 7.8u 0)"
        )

    for drive_v in read_drive_cases_v:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_read_drive_{str(drive_v).replace('.', 'p')}v"
        )
        read_drive_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, read_drive_line(drive_v))
        read_drive_deck = replace_required(
            read_drive_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + corner_guard_pwl + "\n" + corner_guardn_pwl,
        )
        read_drive_deck = replace_required(read_drive_deck, nmos_store_line_p, guard_store_line_p)
        read_drive_deck = replace_required(read_drive_deck, nmos_store_line_m, guard_store_line_m)
        read_drive_deck = replace_required(
            read_drive_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        read_drive_data = run_ngspice(read_drive_deck, stem)
        rdt, read_drive_cols = load_wrdata(read_drive_data, 23)

        def rdrat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rdt - time_s))])

        read_drive_preact = read_drive_cols[10] - read_drive_cols[9]
        read_drive_store = read_drive_cols[14] - read_drive_cols[13]
        read_drive_labels.append(f"{drive_v:.2f} V")
        read_drive_preact_samples.append(rdrat(3.575e-6, read_drive_preact))
        read_drive_store_samples.append(rdrat(3.575e-6, read_drive_store))
        read_drive_traces.append((f"{drive_v:.2f} V", rdt, read_drive_preact, read_drive_store, read_drive_cols[21]))

    read_drive_preact_samples = np.array(read_drive_preact_samples)
    read_drive_store_samples = np.array(read_drive_store_samples)
    nominal_read_drive_idx = read_drive_cases_v.index(1.15)
    require(
        abs(read_drive_store_samples[nominal_read_drive_idx] - guard_timing_samples[2]) < 0.002,
        "nominal read-drive case should match the nominal guard timing sample",
    )
    require(
        read_drive_preact_samples[0] < read_drive_preact_samples[nominal_read_drive_idx] - 0.020,
        "low read-drive voltage should visibly reduce the read preactivation",
    )
    require(
        read_drive_store_samples[0] < read_drive_store_samples[nominal_read_drive_idx] - 0.010,
        "low read-drive voltage should visibly undercharge the guarded activation store",
    )
    require(np.all(np.diff(read_drive_preact_samples) > 0.0005), "read preactivation should increase with read-drive voltage")
    require(
        read_drive_store_samples[-1] > read_drive_store_samples[nominal_read_drive_idx] + 0.004,
        "strong read drive should increase the sampled activation in the fixed guard window",
    )

    read_width_cases_u = [6, 12, 24, 48, 96, 192]
    read_width_labels = []
    read_width_store_samples = []
    read_width_preact_samples = []
    read_width_traces = []
    for read_width_u in read_width_cases_u:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_wread_{read_width_u}u"
        )
        read_width_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        read_width_deck = replace_required(
            read_width_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + corner_guard_pwl + "\n" + corner_guardn_pwl,
        )
        read_width_deck = replace_required(read_width_deck, corner_param_line, corner_param_line.replace("WREAD=24u", f"WREAD={read_width_u}u"))
        read_width_deck = replace_required(read_width_deck, nmos_store_line_p, guard_store_line_p)
        read_width_deck = replace_required(read_width_deck, nmos_store_line_m, guard_store_line_m)
        read_width_deck = replace_required(
            read_width_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        read_width_data = run_ngspice(read_width_deck, stem)
        rwt, read_width_cols = load_wrdata(read_width_data, 23)

        def rwdat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rwt - time_s))])

        read_width_preact = read_width_cols[10] - read_width_cols[9]
        read_width_store = read_width_cols[14] - read_width_cols[13]
        read_width_labels.append(f"{read_width_u}u")
        read_width_preact_samples.append(rwdat(3.575e-6, read_width_preact))
        read_width_store_samples.append(rwdat(3.575e-6, read_width_store))
        read_width_traces.append((f"{read_width_u}u", rwt, read_width_store))

    read_width_preact_samples = np.array(read_width_preact_samples)
    read_width_store_samples = np.array(read_width_store_samples)
    nominal_read_width_idx = read_width_cases_u.index(24)
    require(
        abs(read_width_store_samples[nominal_read_width_idx] - guard_timing_samples[2]) < 0.002,
        "nominal WREAD case should match the nominal guard timing sample",
    )
    require(
        read_width_preact_samples[0] < read_width_preact_samples[nominal_read_width_idx] - 0.020,
        "small WREAD should visibly reduce read preactivation",
    )
    require(
        read_width_store_samples[0] < read_width_store_samples[nominal_read_width_idx] - 0.010,
        "small WREAD should visibly undercharge the guarded activation store",
    )
    require(np.all(np.diff(read_width_preact_samples) > 0), "read preactivation should increase with read tail width")
    require(
        read_width_preact_samples[-1] - read_width_preact_samples[-2] < 0.001,
        "largest WREAD points should show the preactivation saturation knee",
    )
    require(
        read_width_store_samples[-1] > read_width_store_samples[nominal_read_width_idx] + 0.004,
        "larger WREAD should improve sampled activation in the fixed guard window",
    )
    require(
        abs(read_width_store_samples[-1] - read_width_store_samples[-2]) < 0.001,
        "largest WREAD points should not materially improve stored activation",
    )

    forward_pair_lines = "\n".join(
        [
            "MNFP_HYR hp_hyr zm_hyr ftail_hyr 0 NMOS L={LCH} W=48u",
            "MNFM_HYR hm_hyr zp_hyr ftail_hyr 0 NMOS L={LCH} W=48u",
            "MNFT_HYR ftail_hyr vbias 0 0 NMOS L={LCH} W=48u",
        ]
    )
    forward_pair_cases_u = [6, 12, 24, 48, 96, 192]
    forward_pair_labels = []
    forward_pair_preact_samples = []
    forward_pair_load_samples = []
    forward_pair_store_samples = []
    forward_pair_traces = []
    for forward_width_u in forward_pair_cases_u:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_forward_pair_{forward_width_u}u"
        )
        sized_forward_pair_lines = forward_pair_lines.replace("W=48u", f"W={forward_width_u}u")
        forward_pair_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        forward_pair_deck = replace_required(
            forward_pair_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + corner_guard_pwl + "\n" + corner_guardn_pwl,
        )
        forward_pair_deck = replace_required(forward_pair_deck, nmos_store_line_p, guard_store_line_p)
        forward_pair_deck = replace_required(forward_pair_deck, nmos_store_line_m, guard_store_line_m)
        forward_pair_deck = replace_required(forward_pair_deck, forward_pair_lines, sized_forward_pair_lines)
        forward_pair_deck = replace_required(
            forward_pair_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        forward_pair_data = run_ngspice(forward_pair_deck, stem)
        fpt, forward_pair_cols = load_wrdata(forward_pair_data, 23)

        def fpat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(fpt - time_s))])

        forward_pair_preact = forward_pair_cols[10] - forward_pair_cols[9]
        forward_pair_load = forward_pair_cols[12] - forward_pair_cols[11]
        forward_pair_store = forward_pair_cols[14] - forward_pair_cols[13]
        forward_pair_labels.append(f"{forward_width_u}u")
        forward_pair_preact_samples.append(fpat(3.575e-6, forward_pair_preact))
        forward_pair_load_samples.append(fpat(3.315e-6, forward_pair_load))
        forward_pair_store_samples.append(fpat(3.575e-6, forward_pair_store))
        forward_pair_traces.append((f"{forward_width_u}u", fpt, forward_pair_load, forward_pair_store))

    forward_pair_preact_samples = np.array(forward_pair_preact_samples)
    forward_pair_load_samples = np.array(forward_pair_load_samples)
    forward_pair_store_samples = np.array(forward_pair_store_samples)
    nominal_forward_pair_idx = forward_pair_cases_u.index(48)
    require(np.min(forward_pair_preact_samples) > 0.048, "forward-pair sweep should keep a valid read state")
    require(
        np.max(forward_pair_preact_samples) - np.min(forward_pair_preact_samples) < 0.001,
        "forward-pair sizing should not perturb the stored preactivation",
    )
    require(
        abs(forward_pair_store_samples[nominal_forward_pair_idx] - guard_timing_samples[2]) < 0.002,
        "nominal forward-pair width should match the nominal guard timing sample",
    )
    require(
        forward_pair_store_samples[0] < forward_pair_store_samples[nominal_forward_pair_idx] - 0.010,
        "small forward pair should visibly underdrive the activation store",
    )
    require(
        np.all(np.diff(forward_pair_store_samples) > 0.005),
        "stored activation should increase with forward-pair strength over the tested range",
    )
    require(
        forward_pair_store_samples[-1] > forward_pair_store_samples[nominal_forward_pair_idx] + 0.050,
        "large forward pair should expose activation gain headroom beyond nominal sizing",
    )

    high_gain_forward_pair_width_u = 96
    high_gain_forward_pair_lines = forward_pair_lines.replace("W=48u", f"W={high_gain_forward_pair_width_u}u")
    high_gain_forward_pair_idx = forward_pair_cases_u.index(high_gain_forward_pair_width_u)

    high_gain_hold_deck = replace_required(hold_deck, forward_pair_lines, high_gain_forward_pair_lines)
    high_gain_hold_deck = replace_required(
        high_gain_hold_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_hold.dat",
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
    )
    high_gain_hold_data = run_ngspice(
        high_gain_hold_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold",
    )
    hght, high_gain_hold_cols = load_wrdata(high_gain_hold_data, 23)

    def hghat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hght - time_s))])

    high_gain_hold_preact = high_gain_hold_cols[10] - high_gain_hold_cols[9]
    high_gain_hold_store = high_gain_hold_cols[14] - high_gain_hold_cols[13]
    high_gain_hold_store_samples = np.array([hghat(ts, high_gain_hold_store) for ts in hold_sample_times])
    high_gain_hold_preact_samples = np.array([hghat(ts, high_gain_hold_preact) for ts in hold_sample_times])
    high_gain_hold_drift = high_gain_hold_store_samples - high_gain_hold_store_samples[0]

    high_gain_negative_hold_deck = replace_required(negative_hold_deck, forward_pair_lines, high_gain_forward_pair_lines)
    high_gain_negative_hold_deck = replace_required(
        high_gain_negative_hold_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_signed_hold_negative.dat",
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_signed_hold_negative.dat",
    )
    high_gain_negative_hold_data = run_ngspice(
        high_gain_negative_hold_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_signed_hold_negative",
    )
    hgnht, high_gain_negative_hold_cols = load_wrdata(high_gain_negative_hold_data, 23)

    def hgnhat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hgnht - time_s))])

    high_gain_negative_hold_preact = high_gain_negative_hold_cols[10] - high_gain_negative_hold_cols[9]
    high_gain_negative_hold_store = high_gain_negative_hold_cols[14] - high_gain_negative_hold_cols[13]
    high_gain_negative_hold_store_samples = np.array([hgnhat(ts, high_gain_negative_hold_store) for ts in hold_sample_times])
    high_gain_negative_hold_preact_samples = np.array([hgnhat(ts, high_gain_negative_hold_preact) for ts in hold_sample_times])
    high_gain_negative_hold_drift = high_gain_negative_hold_store_samples - high_gain_negative_hold_store_samples[0]
    high_gain_signed_mirror_error = high_gain_hold_store_samples + high_gain_negative_hold_store_samples

    high_gain_stress_deck = replace_required(stress_deck, forward_pair_lines, high_gain_forward_pair_lines)
    high_gain_stress_deck = replace_required(
        high_gain_stress_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_off_isolation.dat",
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_off_isolation.dat",
    )
    high_gain_stress_data = run_ngspice(
        high_gain_stress_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_off_isolation",
    )
    hgot, high_gain_guard_off_cols = load_wrdata(high_gain_stress_data, 23)

    def hgoat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hgot - time_s))])

    high_gain_guard_off_preact = high_gain_guard_off_cols[10] - high_gain_guard_off_cols[9]
    high_gain_guard_off_store = high_gain_guard_off_cols[14] - high_gain_guard_off_cols[13]
    high_gain_guard_off_store_samples = np.array([hgoat(ts, high_gain_guard_off_store) for ts in guard_off_sample_times])
    high_gain_guard_off_preact_samples = np.array([hgoat(ts, high_gain_guard_off_preact) for ts in guard_off_sample_times])
    high_gain_guard_off_drift = high_gain_guard_off_store_samples - high_gain_guard_off_store_samples[0]

    require(
        abs(high_gain_hold_store_samples[0] - forward_pair_store_samples[high_gain_forward_pair_idx]) < 0.002,
        "96u high-gain hold fixture should match the 96u forward-pair sweep sample",
    )
    require(
        high_gain_hold_store_samples[0] > guard_timing_samples[2] + 0.020,
        "96u forward pair should retain materially larger stored activation than nominal",
    )
    require(
        np.max(np.abs(high_gain_hold_drift)) < 0.0003,
        "96u forward-pair positive hold should stay within 0.3 mV over the no-reset interval",
    )
    require(
        high_gain_negative_hold_store_samples[0] < -0.070,
        "96u forward-pair negative hold should capture a full negative activation",
    )
    require(
        np.max(np.abs(high_gain_negative_hold_drift)) < 0.0003,
        "96u forward-pair negative hold should stay within 0.3 mV over the no-reset interval",
    )
    require(
        np.max(np.abs(high_gain_signed_mirror_error)) < 0.0003,
        "96u forward-pair positive and negative stores should mirror within 0.3 mV",
    )
    require(
        np.max(np.abs(high_gain_hold_preact_samples + high_gain_negative_hold_preact_samples)) < 0.002,
        "96u forward-pair positive and negative preactivations should mirror",
    )
    require(
        high_gain_guard_off_store_samples[0] > 0.070,
        "96u forward-pair off-isolation deck should capture the larger activation before stress",
    )
    require(
        np.max(np.abs(high_gain_guard_off_drift)) < 0.001,
        "96u forward-pair off-state control/read toggles should not disturb the held activation by more than 1 mV",
    )

    zcm_source_line = "VZCM zcm 0 0.90"
    zcm_cap_p_line = "CZP_HYR zp_hyr 0 {CSUM} IC=0.9"
    zcm_cap_m_line = "CZM_HYR zm_hyr 0 {CSUM} IC=0.9"
    high_gain_zcm_cases_v = [0.75, 0.85, 0.90, 0.95, 1.05]
    high_gain_zcm_labels = []
    high_gain_zcm_preact_samples = []
    high_gain_zcm_load_samples = []
    high_gain_zcm_store_samples = []
    high_gain_zcm_traces = []
    for zcm_v in high_gain_zcm_cases_v:
        zcm_token = str(zcm_v).replace(".", "p")
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_{zcm_token}v"
        )
        zcm_deck = replace_required(high_gain_hold_deck, zcm_source_line, f"VZCM zcm 0 {zcm_v:.2f}")
        zcm_deck = replace_required(zcm_deck, zcm_cap_p_line, f"CZP_HYR zp_hyr 0 {{CSUM}} IC={zcm_v:.2f}")
        zcm_deck = replace_required(zcm_deck, zcm_cap_m_line, f"CZM_HYR zm_hyr 0 {{CSUM}} IC={zcm_v:.2f}")
        zcm_deck = replace_required(
            zcm_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        zcm_data = run_ngspice(zcm_deck, stem)
        zcmt, zcm_cols = load_wrdata(zcm_data, 23)

        def zcmat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(zcmt - time_s))])

        zcm_preact = zcm_cols[10] - zcm_cols[9]
        zcm_load = zcm_cols[12] - zcm_cols[11]
        zcm_store = zcm_cols[14] - zcm_cols[13]
        high_gain_zcm_labels.append(f"{zcm_v:.2f} V")
        high_gain_zcm_preact_samples.append(zcmat(3.575e-6, zcm_preact))
        high_gain_zcm_load_samples.append(zcmat(3.315e-6, zcm_load))
        high_gain_zcm_store_samples.append(zcmat(3.575e-6, zcm_store))
        high_gain_zcm_traces.append((f"{zcm_v:.2f} V", zcmt, zcm_load, zcm_store))

    high_gain_zcm_preact_samples = np.array(high_gain_zcm_preact_samples)
    high_gain_zcm_load_samples = np.array(high_gain_zcm_load_samples)
    high_gain_zcm_store_samples = np.array(high_gain_zcm_store_samples)
    nominal_zcm_idx = high_gain_zcm_cases_v.index(0.90)
    require(
        abs(high_gain_zcm_store_samples[nominal_zcm_idx] - high_gain_hold_store_samples[0]) < 0.002,
        "nominal 96u z common-mode sweep should match the 96u high-gain hold sample",
    )
    require(np.all(high_gain_zcm_preact_samples > 0.045), "96u z common-mode sweep should keep positive read preactivation")
    require(
        high_gain_zcm_store_samples[0] < 0.001,
        "0.75 V z common mode should expose the 96u forward-pair lower-headroom failure",
    )
    require(
        high_gain_zcm_store_samples[1] < high_gain_zcm_store_samples[nominal_zcm_idx] - 0.040,
        "0.85 V z common mode should expose partial 96u forward-pair underdrive",
    )
    require(
        np.min(high_gain_zcm_store_samples[2:]) > 0.070,
        "96u forward pair should keep high activation gain from nominal z common mode upward",
    )
    require(
        np.all(np.diff(high_gain_zcm_store_samples) > 0.003),
        "96u stored activation should increase monotonically with z common mode in this headroom sweep",
    )

    high_gain_tail_bias_line = "MNFT_HYR ftail_hyr vbias 0 0 NMOS L={LCH} W=96u"
    high_gain_tail_rebias_zcm_cases_v = [0.75, 0.85, 0.90]
    high_gain_tail_rebias_cases_v = [0.55, 0.65, 0.75, 0.85, 0.95]
    high_gain_tail_rebias_labels = [f"{bias_v:.2f} V" for bias_v in high_gain_tail_rebias_cases_v]
    high_gain_tail_rebias_store = np.zeros((len(high_gain_tail_rebias_zcm_cases_v), len(high_gain_tail_rebias_cases_v)))
    high_gain_tail_rebias_load = np.zeros_like(high_gain_tail_rebias_store)
    high_gain_tail_rebias_preact = np.zeros_like(high_gain_tail_rebias_store)
    high_gain_tail_rebias_traces = []
    for zi, zcm_v in enumerate(high_gain_tail_rebias_zcm_cases_v):
        for bi, tail_bias_v in enumerate(high_gain_tail_rebias_cases_v):
            stem = (
                "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
                f"forward_pair_96u_zcm_{str(zcm_v).replace('.', 'p')}v_tail_{str(tail_bias_v).replace('.', 'p')}v"
            )
            rebias_deck = replace_required(high_gain_hold_deck, zcm_source_line, f"VZCM zcm 0 {zcm_v:.2f}")
            rebias_deck = replace_required(rebias_deck, zcm_cap_p_line, f"CZP_HYR zp_hyr 0 {{CSUM}} IC={zcm_v:.2f}")
            rebias_deck = replace_required(rebias_deck, zcm_cap_m_line, f"CZM_HYR zm_hyr 0 {{CSUM}} IC={zcm_v:.2f}")
            rebias_deck = replace_required(
                rebias_deck,
                high_gain_tail_bias_line,
                f"VHFBIAS_HYR vhfbias_hyr 0 {tail_bias_v:.2f}\n"
                "MNFT_HYR ftail_hyr vhfbias_hyr 0 0 NMOS L={LCH} W=96u",
            )
            rebias_deck = replace_required(
                rebias_deck,
                "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
                f"{stem}.dat",
            )
            rebias_data = run_ngspice(rebias_deck, stem)
            rbt, rebias_cols = load_wrdata(rebias_data, 23)

            def rbat(time_s: float, values: np.ndarray) -> float:
                return float(values[np.argmin(np.abs(rbt - time_s))])

            rebias_preact = rebias_cols[10] - rebias_cols[9]
            rebias_load = rebias_cols[12] - rebias_cols[11]
            rebias_store = rebias_cols[14] - rebias_cols[13]
            high_gain_tail_rebias_preact[zi, bi] = rbat(3.575e-6, rebias_preact)
            high_gain_tail_rebias_load[zi, bi] = rbat(3.315e-6, rebias_load)
            high_gain_tail_rebias_store[zi, bi] = rbat(3.575e-6, rebias_store)
            if zcm_v in {0.75, 0.85, 0.90} and tail_bias_v in {0.55, 0.75, 0.95}:
                high_gain_tail_rebias_traces.append((f"zcm {zcm_v:.2f}, tail {tail_bias_v:.2f}", rbt, rebias_store))

    require(
        np.min(high_gain_tail_rebias_preact) > 0.040,
        "96u tail-rebias sweep should keep a real read preactivation in every z common-mode case",
    )
    require(
        high_gain_tail_rebias_store[0, -1] < 0.001,
        "96u tail-rebias nominal tail at 0.75 V z common mode should reproduce the headroom failure",
    )
    require(
        np.max(high_gain_tail_rebias_store[0]) < 0.010,
        "forward-tail rebias alone should not rescue the 0.75 V z common-mode failure",
    )
    require(
        np.max(high_gain_tail_rebias_store[1]) < 0.030,
        "forward-tail rebias alone should not materially rescue the partial 0.85 V z common-mode case",
    )
    require(
        high_gain_tail_rebias_store[2, 0] < 0.001,
        "too-low forward-tail bias should collapse even the nominal 0.90 V z common-mode case",
    )
    require(
        np.min(high_gain_tail_rebias_store[2, 1:]) > 0.070,
        "96u tail-rebias sweep should preserve useful nominal-z activation above the underbiased tail corner",
    )

    def shifted_forward_pair_lines(
        couple_pf: float,
        gate_pf: float = 5.0,
        gate_ic_p: float = 0.90,
        gate_ic_m: float = 0.90,
        reset_ref_p: float = 0.90,
        reset_ref_m: float = 0.90,
        forward_p_model: str = "NMOS",
        forward_m_model: str = "NMOS",
        forward_tail_model: str = "NMOS",
        reset_n_model: str = "NMOS",
        reset_p_model: str = "PMOS",
        reset_n_width_u: float | None = None,
        reset_p_width_u: float | None = None,
        reset_ref_series_ohm: float = 0.0,
        reset_ref_shunt_pf: float = 0.0,
    ) -> str:
        def fmt_gate_ic(value: float) -> str:
            if abs(value - round(value, 2)) < 1e-12:
                return f"{value:.2f}"
            return f"{value:.5f}"

        def reset_ref_source(name: str, node: str, value: float) -> list[str]:
            lines: list[str] = []
            if reset_ref_series_ohm <= 0.0:
                lines.append(f"V{name}_HYR {node} 0 {fmt_gate_ic(value)}")
            else:
                source_node = f"{node}_src"
                lines.extend(
                    [
                        f"V{name}_HYR {source_node} 0 {fmt_gate_ic(value)}",
                        f"R{name}_HYR {source_node} {node} {reset_ref_series_ohm:g}",
                    ]
                )
            if reset_ref_shunt_pf > 0.0:
                lines.append(f"C{name}_DECAP_HYR {node} 0 {reset_ref_shunt_pf:g}p IC={fmt_gate_ic(value)}")
            return lines

        if abs(reset_ref_p - reset_ref_m) < 1e-12:
            reset_ref_lines = reset_ref_source("ZGCM", "zgcm_hyr", reset_ref_p)
            reset_ref_p_node = "zgcm_hyr"
            reset_ref_m_node = "zgcm_hyr"
        else:
            reset_ref_lines = [
                *reset_ref_source("ZGRP", "zgrp_hyr", reset_ref_p),
                *reset_ref_source("ZGRM", "zgrm_hyr", reset_ref_m),
            ]
            reset_ref_p_node = "zgrp_hyr"
            reset_ref_m_node = "zgrm_hyr"
        reset_n_width = f"{reset_n_width_u:g}u" if reset_n_width_u is not None else "{WRESETN}"
        reset_p_width = f"{reset_p_width_u:g}u" if reset_p_width_u is not None else "{WRESETP}"

        return "\n".join(
            [
                *reset_ref_lines,
                f"CZPG_HYR zpg_hyr 0 {gate_pf:g}p IC={fmt_gate_ic(gate_ic_p)}",
                f"CZMG_HYR zmg_hyr 0 {gate_pf:g}p IC={fmt_gate_ic(gate_ic_m)}",
                "RZPG_HYR zpg_hyr 0 100G",
                "RZMG_HYR zmg_hyr 0 100G",
                f"MRZGPN_HYR zpg_hyr rst_hyr {reset_ref_p_node} 0 {reset_n_model} L={{LCH}} W={reset_n_width}",
                f"MRZGMN_HYR zmg_hyr rst_hyr {reset_ref_m_node} 0 {reset_n_model} L={{LCH}} W={reset_n_width}",
                f"MRZGPP_HYR zpg_hyr rstn_hyr {reset_ref_p_node} vdd {reset_p_model} L={{LCH}} W={reset_p_width}",
                f"MRZGMP_HYR zmg_hyr rstn_hyr {reset_ref_m_node} vdd {reset_p_model} L={{LCH}} W={reset_p_width}",
                f"CCZP_HYR zp_hyr zpg_hyr {couple_pf:g}p",
                f"CCZM_HYR zm_hyr zmg_hyr {couple_pf:g}p",
                f"MNFP_HYR hp_hyr zmg_hyr ftail_hyr 0 {forward_p_model} L={{LCH}} W=96u",
                f"MNFM_HYR hm_hyr zpg_hyr ftail_hyr 0 {forward_m_model} L={{LCH}} W=96u",
                f"MNFT_HYR ftail_hyr vbias 0 0 {forward_tail_model} L={{LCH}} W=96u",
            ]
        )

    high_gain_shift_couple_cases_pf = [5.0, 10.0, 20.0, 50.0]
    high_gain_shift_couple_labels = [f"{value:g} pF" for value in high_gain_shift_couple_cases_pf]
    high_gain_shift_store_samples = []
    high_gain_shift_preact_samples = []
    high_gain_shift_traces = []
    for couple_pf in high_gain_shift_couple_cases_pf:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_cc{str(couple_pf).replace('.', 'p')}p_cg5p"
        )
        shift_deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
        shift_deck = replace_required(shift_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        shift_deck = replace_required(shift_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        shift_deck = replace_required(shift_deck, high_gain_forward_pair_lines, shifted_forward_pair_lines(couple_pf))
        shift_deck = replace_required(
            shift_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        shift_data = run_ngspice(shift_deck, stem)
        sht, shift_cols = load_wrdata(shift_data, 23)

        def shat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(sht - time_s))])

        shift_preact = shift_cols[10] - shift_cols[9]
        shift_store = shift_cols[14] - shift_cols[13]
        high_gain_shift_preact_samples.append(shat(3.575e-6, shift_preact))
        high_gain_shift_store_samples.append(shat(3.575e-6, shift_store))
        high_gain_shift_traces.append((f"{couple_pf:g} pF", sht, shift_store))

    high_gain_shift_preact_samples = np.array(high_gain_shift_preact_samples)
    high_gain_shift_store_samples = np.array(high_gain_shift_store_samples)
    chosen_shift_idx = high_gain_shift_couple_cases_pf.index(10.0)
    require(
        np.min(high_gain_shift_preact_samples) > 0.040,
        "shifted-gate coupling sweep should keep a real low-common-mode read preactivation",
    )
    require(
        high_gain_shift_store_samples[chosen_shift_idx] > 0.045,
        "10 pF shifted-gate coupling should recover useful 0.75 V z common-mode activation",
    )
    require(
        high_gain_shift_store_samples[chosen_shift_idx] > high_gain_shift_store_samples[0] + 0.006,
        "10 pF shifted-gate coupling should improve over weak 5 pF coupling",
    )
    require(
        high_gain_shift_store_samples[2] < high_gain_shift_store_samples[chosen_shift_idx] - 0.030,
        "overcoupled 20 pF shifted-gate case should expose a sizing window, not monotone improvement",
    )
    require(
        high_gain_shift_store_samples[-1] < 0.005,
        "strongly overcoupled 50 pF shifted-gate case should collapse again",
    )

    chosen_shift_zcm_cases_v = [0.75, 0.85, 0.90]
    chosen_shift_zcm_labels = []
    chosen_shift_zcm_store_samples = []
    chosen_shift_zcm_traces = []
    for zcm_v in chosen_shift_zcm_cases_v:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_{str(zcm_v).replace('.', 'p')}v_shift_cc10p_cg5p"
        )
        shift_zcm_deck = replace_required(high_gain_hold_deck, zcm_source_line, f"VZCM zcm 0 {zcm_v:.2f}")
        shift_zcm_deck = replace_required(shift_zcm_deck, zcm_cap_p_line, f"CZP_HYR zp_hyr 0 {{CSUM}} IC={zcm_v:.2f}")
        shift_zcm_deck = replace_required(shift_zcm_deck, zcm_cap_m_line, f"CZM_HYR zm_hyr 0 {{CSUM}} IC={zcm_v:.2f}")
        shift_zcm_deck = replace_required(shift_zcm_deck, high_gain_forward_pair_lines, shifted_forward_pair_lines(10.0))
        shift_zcm_deck = replace_required(
            shift_zcm_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        shift_zcm_data = run_ngspice(shift_zcm_deck, stem)
        szct, shift_zcm_cols = load_wrdata(shift_zcm_data, 23)

        def szcat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(szct - time_s))])

        shift_zcm_store = shift_zcm_cols[14] - shift_zcm_cols[13]
        chosen_shift_zcm_labels.append(f"{zcm_v:.2f} V")
        chosen_shift_zcm_store_samples.append(szcat(3.575e-6, shift_zcm_store))
        chosen_shift_zcm_traces.append((f"shifted {zcm_v:.2f} V", szct, shift_zcm_store))

    chosen_shift_zcm_store_samples = np.array(chosen_shift_zcm_store_samples)
    require(
        chosen_shift_zcm_store_samples[0] > high_gain_zcm_store_samples[0] + 0.045,
        "chosen shifted-gate latch should rescue the 0.75 V case relative to unshifted 96u",
    )
    require(
        chosen_shift_zcm_store_samples[1] > high_gain_zcm_store_samples[1] + 0.015,
        "chosen shifted-gate latch should improve the partial 0.85 V case",
    )
    require(
        chosen_shift_zcm_store_samples[2] > 0.045,
        "chosen shifted-gate latch should preserve useful nominal-z activation",
    )

    negative_shift_deck = replace_required(high_gain_negative_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
    negative_shift_deck = replace_required(negative_shift_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
    negative_shift_deck = replace_required(negative_shift_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
    negative_shift_deck = replace_required(negative_shift_deck, high_gain_forward_pair_lines, shifted_forward_pair_lines(10.0))
    negative_shift_deck = replace_required(
        negative_shift_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_signed_hold_negative.dat",
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_zcm_0p75v_shift_signed_negative.dat",
    )
    negative_shift_data = run_ngspice(
        negative_shift_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_zcm_0p75v_shift_signed_negative",
    )
    nsht, negative_shift_cols = load_wrdata(negative_shift_data, 23)

    def nshat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(nsht - time_s))])

    negative_shift_store = negative_shift_cols[14] - negative_shift_cols[13]
    negative_shift_store_samples = np.array([nshat(ts, negative_shift_store) for ts in hold_sample_times])
    positive_shift_t = high_gain_shift_traces[chosen_shift_idx][1]
    positive_shift_store = high_gain_shift_traces[chosen_shift_idx][2]
    positive_shift_store_samples = np.array(
        [float(positive_shift_store[np.argmin(np.abs(positive_shift_t - ts))]) for ts in hold_sample_times]
    )
    shifted_signed_mirror_error = positive_shift_store_samples + negative_shift_store_samples
    require(
        negative_shift_store_samples[0] < -0.045,
        "shifted-gate latch should also capture a useful negative 0.75 V activation",
    )
    require(
        np.max(np.abs(positive_shift_store_samples - positive_shift_store_samples[0])) < 0.0003,
        "shifted-gate positive store should hold within 0.3 mV",
    )
    require(
        np.max(np.abs(negative_shift_store_samples - negative_shift_store_samples[0])) < 0.0003,
        "shifted-gate negative store should hold within 0.3 mV",
    )
    require(
        np.max(np.abs(shifted_signed_mirror_error)) < 0.0003,
        "shifted-gate positive and negative 0.75 V stores should mirror within 0.3 mV",
    )

    shift_stress_stem = (
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
        "forward_pair_96u_zcm_0p75v_shift_off_isolation"
    )
    shift_stress_deck = replace_required(stress_deck, zcm_source_line, "VZCM zcm 0 0.75")
    shift_stress_deck = replace_required(shift_stress_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
    shift_stress_deck = replace_required(shift_stress_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
    shift_stress_deck = replace_required(
        shift_stress_deck,
        forward_pair_lines,
        shifted_forward_pair_lines(10.0),
    )
    shift_stress_deck = replace_required(
        shift_stress_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_off_isolation.dat",
        f"{shift_stress_stem}.dat",
    )
    shift_stress_data = run_ngspice(shift_stress_deck, shift_stress_stem)
    ssht, shift_stress_cols = load_wrdata(shift_stress_data, 23)

    def sshat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(ssht - time_s))])

    shift_stress_preact = shift_stress_cols[10] - shift_stress_cols[9]
    shift_stress_store = shift_stress_cols[14] - shift_stress_cols[13]
    shift_stress_store_samples = np.array([sshat(ts, shift_stress_store) for ts in guard_off_sample_times])
    shift_stress_preact_samples = np.array([sshat(ts, shift_stress_preact) for ts in guard_off_sample_times])
    shift_stress_drift = shift_stress_store_samples - shift_stress_store_samples[0]
    require(
        shift_stress_store_samples[0] > 0.045,
        "shifted-gate off-isolation deck should capture a useful low-common-mode activation before stress",
    )
    require(
        shift_stress_preact_samples[0] > 0.040,
        "shifted-gate off-isolation deck should start from a real low-common-mode preactivation",
    )
    require(
        np.max(np.abs(shift_stress_drift)) < 0.001,
        "shifted-gate later off-state control/read toggles should not disturb the held activation by more than 1 mV",
    )

    shift_reset_probe_prefix = "v(zpg_hyr) v(zmg_hyr) v(cdp_hyr)"
    shift_bad_gate_lines = shifted_forward_pair_lines(10.0, gate_ic_p=0.35, gate_ic_m=1.45)

    def add_shifted_gate_probes(deck: str, stem: str) -> str:
        return replace_required(deck, f"{stem}.dat v(cdp_hyr)", f"{stem}.dat {shift_reset_probe_prefix}")

    shift_bad_stem = (
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
        "forward_pair_96u_zcm_0p75v_shift_bad_gate_noreset"
    )
    shift_bad_deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
    shift_bad_deck = replace_required(shift_bad_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
    shift_bad_deck = replace_required(shift_bad_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
    shift_bad_deck = replace_required(shift_bad_deck, high_gain_forward_pair_lines, shift_bad_gate_lines)
    shift_bad_deck = replace_required(
        shift_bad_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
        f"{shift_bad_stem}.dat",
    )
    shift_bad_deck = add_shifted_gate_probes(shift_bad_deck, shift_bad_stem)
    shift_bad_data = run_ngspice(shift_bad_deck, shift_bad_stem)
    sbt, shift_bad_cols = load_wrdata(shift_bad_data, 25)

    def sbat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(sbt - time_s))])

    shift_bad_gate_diff = shift_bad_cols[0] - shift_bad_cols[1]
    shift_bad_gate_common = 0.5 * (shift_bad_cols[0] + shift_bad_cols[1])
    shift_bad_preact = shift_bad_cols[12] - shift_bad_cols[11]
    shift_bad_store = shift_bad_cols[16] - shift_bad_cols[15]

    shift_reset_stem = (
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
        "forward_pair_96u_zcm_0p75v_shift_reset_feedthrough"
    )
    shift_reset_pulse = "VRESET_HYR rst_hyr 0 PWL(0 0 2.48u 0 2.50u 1.8 2.62u 1.8 2.64u 0 7.8u 0)"
    shift_resetn_pulse = (
        "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 2.48u 1.8 2.50u 0 2.62u 0 2.64u 1.8 7.8u 1.8)"
    )
    shift_reset_deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
    shift_reset_deck = replace_required(shift_reset_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
    shift_reset_deck = replace_required(shift_reset_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
    shift_reset_deck = replace_required(shift_reset_deck, high_gain_forward_pair_lines, shift_bad_gate_lines)
    shift_reset_deck = replace_required(shift_reset_deck, "VRESET_HYR rst_hyr 0 0", shift_reset_pulse)
    shift_reset_deck = replace_required(shift_reset_deck, "VRESETN_HYR rstn_hyr 0 1.8", shift_resetn_pulse)
    shift_reset_deck = replace_required(
        shift_reset_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
        f"{shift_reset_stem}.dat",
    )
    shift_reset_deck = add_shifted_gate_probes(shift_reset_deck, shift_reset_stem)
    shift_reset_data = run_ngspice(shift_reset_deck, shift_reset_stem)
    srt, shift_reset_cols = load_wrdata(shift_reset_data, 25)

    def srat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(srt - time_s))])

    shift_reset_gate_diff = shift_reset_cols[0] - shift_reset_cols[1]
    shift_reset_gate_common = 0.5 * (shift_reset_cols[0] + shift_reset_cols[1])
    shift_reset_preact = shift_reset_cols[12] - shift_reset_cols[11]
    shift_reset_store = shift_reset_cols[16] - shift_reset_cols[15]
    shift_reset_gate_residue = np.max(np.abs(shift_reset_gate_diff[(srt > 2.64e-6) & (srt < 2.72e-6)]))
    shift_reset_store_sample = srat(3.575e-6, shift_reset_store)
    shift_bad_store_sample = sbat(3.575e-6, shift_bad_store)
    require(
        abs(sbat(2.70e-6, shift_bad_gate_diff)) > 0.30,
        "bad shifted-gate fixture should start with a large unreset gate differential",
    )
    require(
        shift_bad_store_sample > 0.20,
        "unreset bad shifted-gate fixture should visibly overdrive the activation store",
    )
    require(
        shift_reset_gate_residue < 0.002,
        "physical shifted-gate reset should clear the bad gate differential before read",
    )
    require(
        abs(srat(2.70e-6, shift_reset_gate_common) - 0.90) < 0.005,
        "physical shifted-gate reset should restore the gate common mode",
    )
    require(
        shift_reset_store_sample > 0.045,
        "reset shifted-gate fixture should still capture a useful low-common-mode activation",
    )
    require(
        shift_reset_store_sample < 0.080,
        "reset shifted-gate fixture should avoid the unreset overdrive/rail case",
    )
    require(
        shift_reset_store_sample < 0.25 * shift_bad_store_sample,
        "physical shifted-gate reset should materially reduce bad-initial-state feedthrough",
    )

    shift_reset_size_cases = [
        ("x002", "0.02x", 1.2, 3.6),
        ("x005", "0.05x", 3.0, 9.0),
        ("x010", "0.10x", 6.0, 18.0),
        ("x025", "0.25x", 15.0, 45.0),
        ("x100", "1.00x", 60.0, 180.0),
    ]
    shift_reset_size_labels = []
    shift_reset_size_gate_residue = []
    shift_reset_size_common_error = []
    shift_reset_size_store_samples = []
    shift_reset_size_traces = []
    for name, label, n_width_u, p_width_u in shift_reset_size_cases:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_reset_feedthrough_size_{name}"
        )
        deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
        deck = replace_required(deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(
            deck,
            high_gain_forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                gate_ic_p=0.35,
                gate_ic_m=1.45,
                reset_n_width_u=n_width_u,
                reset_p_width_u=p_width_u,
            ),
        )
        deck = replace_required(deck, "VRESET_HYR rst_hyr 0 0", shift_reset_pulse)
        deck = replace_required(deck, "VRESETN_HYR rstn_hyr 0 1.8", shift_resetn_pulse)
        deck = replace_required(
            deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        deck = add_shifted_gate_probes(deck, stem)
        data = run_ngspice(deck, stem)
        rst, cols = load_wrdata(data, 25)

        def rstat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rst - time_s))])

        gate_diff = cols[0] - cols[1]
        gate_common = 0.5 * (cols[0] + cols[1])
        store = cols[16] - cols[15]
        post_reset = (rst > 2.64e-6) & (rst < 2.72e-6)
        shift_reset_size_labels.append(label)
        shift_reset_size_gate_residue.append(np.max(np.abs(gate_diff[post_reset])))
        shift_reset_size_common_error.append(abs(rstat(2.70e-6, gate_common) - 0.90))
        shift_reset_size_store_samples.append(rstat(3.575e-6, store))
        shift_reset_size_traces.append((label, rst, gate_diff, gate_common, store))

    shift_reset_size_gate_residue = np.array(shift_reset_size_gate_residue)
    shift_reset_size_common_error = np.array(shift_reset_size_common_error)
    shift_reset_size_store_samples = np.array(shift_reset_size_store_samples)
    shift_reset_size_feedthrough = shift_reset_size_store_samples - positive_shift_store_samples[0]
    require(
        shift_reset_size_gate_residue[0] > 0.050,
        "very weak shifted-gate reset sizing should visibly leave bad-gate residue",
    )
    require(
        0.003 < shift_reset_size_gate_residue[1] < 0.010,
        "marginal shifted-gate reset sizing should expose the residue transition",
    )
    require(
        np.all(shift_reset_size_gate_residue[2:] < 0.003),
        "0.10x or stronger shifted-gate reset should clear the bad differential",
    )
    require(
        np.all(shift_reset_size_common_error[2:] < 0.001),
        "0.10x or stronger shifted-gate reset should restore common mode",
    )
    require(
        np.all((shift_reset_size_store_samples[1:] > 0.045) & (shift_reset_size_store_samples[1:] < 0.080)),
        "usable shifted-gate reset sizes should preserve bounded useful activation",
    )
    require(
        np.min(np.abs(shift_reset_size_feedthrough[2:])) < 0.006,
        "clearing shifted-gate reset sizes should keep reset feedthrough to a small activation offset",
    )

    shift_reset_common_cases = [
        ("cm076", "0.76 V", 0.76),
        ("cm078", "0.78 V", 0.78),
        ("cm080", "0.80 V", 0.80),
        ("cm082", "0.82 V", 0.82),
        ("cm084", "0.84 V", 0.84),
        ("cm086", "0.86 V", 0.86),
        ("cm088", "0.88 V", 0.88),
        ("cm090", "0.90 V", 0.90),
        ("cm092", "0.92 V", 0.92),
    ]
    shift_reset_common_labels = []
    shift_reset_common_values = []
    shift_reset_common_gate_pre = []
    shift_reset_common_gate_read = []
    shift_reset_common_gate_residue = []
    shift_reset_common_load_samples = []
    shift_reset_common_store_samples = []
    shift_reset_common_traces = []
    for name, label, reset_common_v in shift_reset_common_cases:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_reset_common_{name}"
        )
        deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
        deck = replace_required(deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(
            deck,
            high_gain_forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                gate_ic_p=0.35,
                gate_ic_m=1.45,
                reset_ref_p=reset_common_v,
                reset_ref_m=reset_common_v,
            ),
        )
        deck = replace_required(deck, "VRESET_HYR rst_hyr 0 0", shift_reset_pulse)
        deck = replace_required(deck, "VRESETN_HYR rstn_hyr 0 1.8", shift_resetn_pulse)
        deck = replace_required(
            deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        deck = add_shifted_gate_probes(deck, stem)
        data = run_ngspice(deck, stem)
        rct, cols = load_wrdata(data, 25)

        def rctat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rct - time_s))])

        gate_diff = cols[0] - cols[1]
        gate_common = 0.5 * (cols[0] + cols[1])
        load = cols[14] - cols[13]
        store = cols[16] - cols[15]
        post_reset = (rct > 2.64e-6) & (rct < 2.72e-6)
        shift_reset_common_labels.append(label)
        shift_reset_common_values.append(reset_common_v)
        shift_reset_common_gate_pre.append(rctat(2.70e-6, gate_common))
        shift_reset_common_gate_read.append(rctat(3.315e-6, gate_common))
        shift_reset_common_gate_residue.append(np.max(np.abs(gate_diff[post_reset])))
        shift_reset_common_load_samples.append(rctat(3.315e-6, load))
        shift_reset_common_store_samples.append(rctat(3.575e-6, store))
        shift_reset_common_traces.append((label, rct, gate_common, load, store))

    shift_reset_common_values = np.array(shift_reset_common_values)
    shift_reset_common_gate_pre = np.array(shift_reset_common_gate_pre)
    shift_reset_common_gate_read = np.array(shift_reset_common_gate_read)
    shift_reset_common_gate_residue = np.array(shift_reset_common_gate_residue)
    shift_reset_common_load_samples = np.array(shift_reset_common_load_samples)
    shift_reset_common_store_samples = np.array(shift_reset_common_store_samples)
    shift_reset_common_store_error = shift_reset_common_store_samples - positive_shift_store_samples[0]
    shift_reset_common_tuned_idx = shift_reset_common_labels.index("0.80 V")
    shift_reset_common_old_idx = shift_reset_common_labels.index("0.90 V")
    require(
        np.max(shift_reset_common_gate_residue) < 0.002,
        "reset-common sweep should clear the bad shifted-gate differential at every common-mode target",
    )
    require(
        abs(shift_reset_common_store_error[shift_reset_common_tuned_idx]) < 0.001,
        "0.80 V shifted-gate reset common should reproduce the initialized-gate activation",
    )
    require(
        shift_reset_common_store_error[shift_reset_common_old_idx] > 0.004,
        "0.90 V shifted-gate reset common should expose the previous activation feedthrough offset",
    )
    require(
        shift_reset_common_store_samples[0] < 0.035,
        "too-low shifted-gate reset common should underdrive the forward store",
    )
    require(
        np.all(np.diff(shift_reset_common_store_samples) > 0.0),
        "shifted-gate reset-common sweep should monotonically increase stored activation in this window",
    )

    shift_reuse_stem = (
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
        "forward_pair_96u_zcm_0p75v_shift_repeated_reset_reuse"
    )
    shift_reuse_deck = replace_required(hybrid_forward_read_reuse_deck, zcm_source_line, "VZCM zcm 0 0.75")
    shift_reuse_deck = replace_required(shift_reuse_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
    shift_reuse_deck = replace_required(shift_reuse_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
    shift_reuse_deck = replace_required(shift_reuse_deck, forward_pair_lines, shifted_forward_pair_lines(10.0))
    shift_reuse_deck = replace_required(
        shift_reuse_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
        f"{shift_reuse_stem}.dat",
    )
    shift_reuse_deck = add_shifted_gate_probes(shift_reuse_deck, shift_reuse_stem)
    shift_reuse_data = run_ngspice(shift_reuse_deck, shift_reuse_stem)
    shrpt, shift_reuse_cols = load_wrdata(shift_reuse_data, 25)

    def shrpat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(shrpt - time_s))])

    shift_reuse_gate_diff = shift_reuse_cols[0] - shift_reuse_cols[1]
    shift_reuse_gate_common = 0.5 * (shift_reuse_cols[0] + shift_reuse_cols[1])
    shift_reuse_weight = shift_reuse_cols[7] - shift_reuse_cols[8]
    shift_reuse_bias = shift_reuse_cols[9] - shift_reuse_cols[10]
    shift_reuse_preact = shift_reuse_cols[12] - shift_reuse_cols[11]
    shift_reuse_load = shift_reuse_cols[14] - shift_reuse_cols[13]
    shift_reuse_store = shift_reuse_cols[16] - shift_reuse_cols[15]
    shift_reuse_z_times = np.array([3.35, 4.95, 6.55]) * 1e-6
    shift_reuse_h_times = np.array([3.45, 5.05, 6.70]) * 1e-6
    shift_reuse_reset_times = np.array([4.10, 5.70]) * 1e-6
    shift_reuse_z_samples = np.array([shrpat(ts, shift_reuse_preact) for ts in shift_reuse_z_times])
    shift_reuse_h_samples = np.array([shrpat(ts, shift_reuse_store) for ts in shift_reuse_h_times])
    shift_reuse_gate_reset_residue = np.array([abs(shrpat(ts, shift_reuse_gate_diff)) for ts in shift_reuse_reset_times])
    shift_reuse_gate_reset_common = np.array([shrpat(ts, shift_reuse_gate_common) for ts in shift_reuse_reset_times])
    shift_reuse_z_reset = np.array([abs(shrpat(ts, shift_reuse_preact)) for ts in shift_reuse_reset_times])
    shift_reuse_h_reset = np.array([abs(shrpat(ts, shift_reuse_store)) for ts in shift_reuse_reset_times])
    shift_reuse_weight_after_write = shrpat(2.55e-6, shift_reuse_weight)
    shift_reuse_bias_after_write = shrpat(2.55e-6, shift_reuse_bias)
    shift_reuse_weight_drift = shrpat(7.45e-6, shift_reuse_weight) - shift_reuse_weight_after_write
    shift_reuse_bias_drift = shrpat(7.45e-6, shift_reuse_bias) - shift_reuse_bias_after_write
    require(
        shift_reuse_weight_after_write > 0.020,
        "shifted-gate repeated-reuse deck should write positive weight state",
    )
    require(
        shift_reuse_bias_after_write > 0.030,
        "shifted-gate repeated-reuse deck should write positive bias state",
    )
    require(
        np.all(shift_reuse_z_samples > 0.040),
        "shifted-gate repeated-reuse cycles should repeatedly read useful low-common-mode preactivation",
    )
    require(
        np.all(shift_reuse_h_samples > 0.045),
        "shifted-gate repeated-reuse cycles should repeatedly store useful low-common-mode activation",
    )
    require(
        np.max(shift_reuse_h_samples) - np.min(shift_reuse_h_samples) < 0.010,
        "shifted-gate repeated-reuse stored activations should stay within a 10 mV cycle window",
    )
    require(
        np.max(shift_reuse_gate_reset_residue) < 0.002,
        "shifted-gate repeated physical resets should clear the gate differential between reads",
    )
    require(
        np.max(np.abs(shift_reuse_gate_reset_common - 0.90)) < 0.005,
        "shifted-gate repeated physical resets should restore gate common mode between reads",
    )
    require(
        np.max(shift_reuse_z_reset) < 0.001,
        "shifted-gate repeated reset should clear preactivation between reads",
    )
    require(
        np.max(shift_reuse_h_reset) < 0.001,
        "shifted-gate repeated reset should clear stored activation between reads",
    )
    require(
        abs(shift_reuse_weight_drift) < 1e-5,
        "shifted-gate repeated read/reset cycles should not disturb weight state",
    )
    require(
        abs(shift_reuse_bias_drift) < 1e-5,
        "shifted-gate repeated read/reset cycles should not disturb bias state",
    )

    shift_reuse_common_stem = (
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
        "forward_pair_96u_zcm_0p75v_shift_repeated_reset_reuse_cm080"
    )
    shift_reuse_common_deck = replace_required(hybrid_forward_read_reuse_deck, zcm_source_line, "VZCM zcm 0 0.75")
    shift_reuse_common_deck = replace_required(shift_reuse_common_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
    shift_reuse_common_deck = replace_required(shift_reuse_common_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
    shift_reuse_common_deck = replace_required(
        shift_reuse_common_deck,
        forward_pair_lines,
        shifted_forward_pair_lines(10.0, reset_ref_p=0.80, reset_ref_m=0.80),
    )
    shift_reuse_common_deck = replace_required(
        shift_reuse_common_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
        f"{shift_reuse_common_stem}.dat",
    )
    shift_reuse_common_deck = add_shifted_gate_probes(shift_reuse_common_deck, shift_reuse_common_stem)
    shift_reuse_common_data = run_ngspice(shift_reuse_common_deck, shift_reuse_common_stem)
    shrct, shift_reuse_common_cols = load_wrdata(shift_reuse_common_data, 25)

    def shrctat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(shrct - time_s))])

    shift_reuse_common_gate_diff = shift_reuse_common_cols[0] - shift_reuse_common_cols[1]
    shift_reuse_common_gate_common = 0.5 * (shift_reuse_common_cols[0] + shift_reuse_common_cols[1])
    shift_reuse_common_weight = shift_reuse_common_cols[7] - shift_reuse_common_cols[8]
    shift_reuse_common_bias = shift_reuse_common_cols[9] - shift_reuse_common_cols[10]
    shift_reuse_common_preact = shift_reuse_common_cols[12] - shift_reuse_common_cols[11]
    shift_reuse_common_load = shift_reuse_common_cols[14] - shift_reuse_common_cols[13]
    shift_reuse_common_store = shift_reuse_common_cols[16] - shift_reuse_common_cols[15]
    shift_reuse_common_z_samples = np.array([shrctat(ts, shift_reuse_common_preact) for ts in shift_reuse_z_times])
    shift_reuse_common_h_samples = np.array([shrctat(ts, shift_reuse_common_store) for ts in shift_reuse_h_times])
    shift_reuse_common_gate_reset_residue = np.array(
        [abs(shrctat(ts, shift_reuse_common_gate_diff)) for ts in shift_reuse_reset_times]
    )
    shift_reuse_common_gate_reset_common = np.array(
        [shrctat(ts, shift_reuse_common_gate_common) for ts in shift_reuse_reset_times]
    )
    shift_reuse_common_z_reset = np.array([abs(shrctat(ts, shift_reuse_common_preact)) for ts in shift_reuse_reset_times])
    shift_reuse_common_h_reset = np.array([abs(shrctat(ts, shift_reuse_common_store)) for ts in shift_reuse_reset_times])
    shift_reuse_common_weight_after_write = shrctat(2.55e-6, shift_reuse_common_weight)
    shift_reuse_common_bias_after_write = shrctat(2.55e-6, shift_reuse_common_bias)
    shift_reuse_common_weight_drift = (
        shrctat(7.45e-6, shift_reuse_common_weight) - shift_reuse_common_weight_after_write
    )
    shift_reuse_common_bias_drift = shrctat(7.45e-6, shift_reuse_common_bias) - shift_reuse_common_bias_after_write
    require(
        np.all(shift_reuse_common_z_samples > 0.040),
        "0.80 V reset-common reuse deck should repeatedly read useful low-common-mode preactivation",
    )
    require(
        np.all(shift_reuse_common_h_samples > 0.045),
        "0.80 V reset-common reuse deck should repeatedly store useful low-common-mode activation",
    )
    require(
        np.max(shift_reuse_common_h_samples) - np.min(shift_reuse_common_h_samples) < 0.001,
        "0.80 V reset-common reuse deck should remove the repeatable reset feedthrough bump",
    )
    require(
        np.max(shift_reuse_h_samples) - np.min(shift_reuse_h_samples) > 0.005,
        "0.90 V reset-common reuse deck should continue exposing the old feedthrough bump",
    )
    require(
        np.max(shift_reuse_common_gate_reset_residue) < 0.002,
        "0.80 V reset-common reuse deck should clear the gate differential between reads",
    )
    require(
        np.max(np.abs(shift_reuse_common_gate_reset_common - 0.80)) < 0.005,
        "0.80 V reset-common reuse deck should restore the tuned gate common mode between reads",
    )
    require(
        np.max(shift_reuse_common_z_reset) < 0.001,
        "0.80 V reset-common reuse deck should clear preactivation state between reads",
    )
    require(
        np.max(shift_reuse_common_h_reset) < 0.001,
        "0.80 V reset-common reuse deck should clear activation state between reads",
    )
    require(
        abs(shift_reuse_common_weight_drift) < 1e-5,
        "0.80 V reset-common reuse deck should not disturb weight state",
    )
    require(
        abs(shift_reuse_common_bias_drift) < 1e-5,
        "0.80 V reset-common reuse deck should not disturb bias state",
    )

    shift_reset_ref_perturb_cases = [
        ("nominal", "nominal", 0.0, 0.0),
        ("common_low_10mv", "common -10 mV", -0.010, 0.0),
        ("common_high_10mv", "common +10 mV", 0.010, 0.0),
        ("helpful_diff_10mv", "helpful diff -10 mV", 0.0, -0.010),
        ("destructive_diff_10mv", "destructive diff +10 mV", 0.0, 0.010),
        ("destructive_diff_20mv", "destructive diff +20 mV", 0.0, 0.020),
    ]
    shift_reset_ref_perturb_labels = []
    shift_reset_ref_perturb_gate_post = []
    shift_reset_ref_perturb_gate_read = []
    shift_reset_ref_perturb_load_samples = []
    shift_reset_ref_perturb_store_samples = []
    shift_reset_ref_perturb_traces = []
    for name, label, common_offset_v, diff_offset_v in shift_reset_ref_perturb_cases:
        reset_ref_p = 0.80 + common_offset_v + 0.5 * diff_offset_v
        reset_ref_m = 0.80 + common_offset_v - 0.5 * diff_offset_v
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_reset_refpert080_{name}"
        )
        deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
        deck = replace_required(deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(
            deck,
            high_gain_forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                gate_ic_p=0.35,
                gate_ic_m=1.45,
                reset_ref_p=reset_ref_p,
                reset_ref_m=reset_ref_m,
            ),
        )
        deck = replace_required(deck, "VRESET_HYR rst_hyr 0 0", shift_reset_pulse)
        deck = replace_required(deck, "VRESETN_HYR rstn_hyr 0 1.8", shift_resetn_pulse)
        deck = replace_required(
            deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        deck = add_shifted_gate_probes(deck, stem)
        data = run_ngspice(deck, stem)
        rpt, cols = load_wrdata(data, 25)

        def rpat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rpt - time_s))])

        gate_diff = cols[0] - cols[1]
        load = cols[14] - cols[13]
        store = cols[16] - cols[15]
        shift_reset_ref_perturb_labels.append(label)
        shift_reset_ref_perturb_gate_post.append(rpat(2.70e-6, gate_diff))
        shift_reset_ref_perturb_gate_read.append(rpat(3.315e-6, gate_diff))
        shift_reset_ref_perturb_load_samples.append(rpat(3.315e-6, load))
        shift_reset_ref_perturb_store_samples.append(rpat(3.575e-6, store))
        shift_reset_ref_perturb_traces.append((label, rpt, gate_diff, load, store))

    shift_reset_ref_perturb_gate_post = np.array(shift_reset_ref_perturb_gate_post)
    shift_reset_ref_perturb_gate_read = np.array(shift_reset_ref_perturb_gate_read)
    shift_reset_ref_perturb_load_samples = np.array(shift_reset_ref_perturb_load_samples)
    shift_reset_ref_perturb_store_samples = np.array(shift_reset_ref_perturb_store_samples)
    reset_ref_perturb_nominal_idx = 0
    reset_ref_perturb_common_idx = [1, 2]
    reset_ref_perturb_helpful_idx = 3
    reset_ref_perturb_destructive_idx = [4, 5]
    require(
        abs(shift_reset_ref_perturb_store_samples[reset_ref_perturb_nominal_idx] - shift_reset_common_store_samples[shift_reset_common_tuned_idx])
        < 0.001,
        "0.80 V reset-reference perturb nominal case should match the reset-common sweep",
    )
    require(
        np.all(shift_reset_ref_perturb_store_samples[reset_ref_perturb_common_idx] > 0.040),
        "10 mV physical reset-reference common perturbations should keep useful activation",
    )
    require(
        np.max(
            np.abs(
                shift_reset_ref_perturb_store_samples[reset_ref_perturb_common_idx]
                - shift_reset_ref_perturb_store_samples[reset_ref_perturb_nominal_idx]
            )
        )
        < 0.010,
        "10 mV physical reset-reference common perturbations should stay a small activation error",
    )
    require(
        shift_reset_ref_perturb_store_samples[reset_ref_perturb_helpful_idx]
        > shift_reset_ref_perturb_store_samples[reset_ref_perturb_nominal_idx],
        "helpful physical reset-reference differential perturbation should increase stored activation",
    )
    require(
        np.all(
            np.diff(
                shift_reset_ref_perturb_store_samples[
                    [reset_ref_perturb_nominal_idx, *reset_ref_perturb_destructive_idx]
                ]
            )
            < -0.004
        ),
        "destructive physical reset-reference differential perturbations should monotonically reduce activation",
    )
    require(
        shift_reset_ref_perturb_store_samples[reset_ref_perturb_destructive_idx[0]] > 0.025,
        "10 mV destructive physical reset-reference differential perturbation should keep useful activation",
    )
    require(
        shift_reset_ref_perturb_store_samples[reset_ref_perturb_destructive_idx[1]] > 0.010,
        "20 mV destructive physical reset-reference differential perturbation should stay positive but expose limited margin",
    )

    shift_noise_cases = [
        ("nominal", "nominal", 0.0, 0.0),
        ("common_low_25mv", "$V_g$ common -25 mV", -0.025, 0.0),
        ("common_high_25mv", "$V_g$ common +25 mV", 0.025, 0.0),
        ("helpful_diff_25mv", "helpful diff -25 mV", 0.0, -0.025),
        ("destructive_diff_25mv", "destructive diff +25 mV", 0.0, 0.025),
        ("destructive_diff_50mv", "destructive diff +50 mV", 0.0, 0.050),
    ]
    shift_noise_labels = []
    shift_noise_gate_samples = []
    shift_noise_load_samples = []
    shift_noise_store_samples = []
    shift_noise_traces = []
    for name, label, gate_common_offset_v, gate_diff_offset_v in shift_noise_cases:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_gate_offset_{name}"
        )
        gate_ic_p = 0.90 + gate_common_offset_v + 0.5 * gate_diff_offset_v
        gate_ic_m = 0.90 + gate_common_offset_v - 0.5 * gate_diff_offset_v
        shift_noise_deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
        shift_noise_deck = replace_required(shift_noise_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        shift_noise_deck = replace_required(shift_noise_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        shift_noise_deck = replace_required(
            shift_noise_deck,
            high_gain_forward_pair_lines,
            shifted_forward_pair_lines(10.0, gate_ic_p=gate_ic_p, gate_ic_m=gate_ic_m),
        )
        shift_noise_deck = replace_required(
            shift_noise_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        shift_noise_deck = add_shifted_gate_probes(shift_noise_deck, stem)
        shift_noise_data = run_ngspice(shift_noise_deck, stem)
        snt, shift_noise_cols = load_wrdata(shift_noise_data, 25)

        def snat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(snt - time_s))])

        shift_noise_gate_diff = shift_noise_cols[0] - shift_noise_cols[1]
        shift_noise_load = shift_noise_cols[14] - shift_noise_cols[13]
        shift_noise_store = shift_noise_cols[16] - shift_noise_cols[15]
        shift_noise_labels.append(label)
        shift_noise_gate_samples.append(snat(3.315e-6, shift_noise_gate_diff))
        shift_noise_load_samples.append(snat(3.315e-6, shift_noise_load))
        shift_noise_store_samples.append(snat(3.575e-6, shift_noise_store))
        shift_noise_traces.append((label, snt, shift_noise_gate_diff, shift_noise_load, shift_noise_store))

    shift_noise_gate_samples = np.array(shift_noise_gate_samples)
    shift_noise_load_samples = np.array(shift_noise_load_samples)
    shift_noise_store_samples = np.array(shift_noise_store_samples)
    nominal_shift_noise_idx = 0
    common_noise_idx = [1, 2]
    helpful_noise_idx = 3
    destructive_noise_idx = [4, 5]
    require(
        shift_noise_store_samples[nominal_shift_noise_idx] > 0.045,
        "shifted-gate offset sweep nominal case should reproduce a useful low-common-mode activation",
    )
    require(
        np.all(shift_noise_store_samples[common_noise_idx] > 0.040),
        "shifted-gate common-mode offsets should keep useful stored activation",
    )
    require(
        np.max(np.abs(shift_noise_store_samples[common_noise_idx] - shift_noise_store_samples[nominal_shift_noise_idx]))
        < 0.012,
        "shifted-gate common-mode offsets should not dominate activation capture",
    )
    require(
        shift_noise_store_samples[helpful_noise_idx] > shift_noise_store_samples[nominal_shift_noise_idx],
        "helpful shifted-gate differential residue should increase the stored activation",
    )
    require(
        np.all(shift_noise_store_samples[destructive_noise_idx] > 0.020),
        "destructive shifted-gate differential residue should reduce but not flip the activation in this window",
    )
    require(
        np.all(np.diff(shift_noise_store_samples[[nominal_shift_noise_idx, *destructive_noise_idx]]) < -0.010),
        "larger destructive shifted-gate differential residue should monotonically reduce stored activation",
    )

    shift_threshold_cases = [
        ("nominal", "nominal", 0.55, 0.55, 0.0),
        ("both_strong_20mv", "both strong -20 mV", 0.53, 0.53, 0.0),
        ("both_weak_20mv", "both weak +20 mV", 0.57, 0.57, 0.0),
        ("hp_weak_20mv", "$h^+$ side weak", 0.57, 0.55, 0.0),
        ("hm_weak_20mv", "$h^-$ side weak", 0.55, 0.57, 0.0),
        ("skew_plusminus_20mv", "skew +20/-20 mV", 0.57, 0.53, 0.0),
        ("skew_minusplus_20mv", "skew -20/+20 mV", 0.53, 0.57, 0.0),
        ("skew_plusminus_trim25mv", "skew +20/-20, trim -25 mV", 0.57, 0.53, -0.025),
        ("skew_plusminus_trim50mv", "skew +20/-20, trim -50 mV", 0.57, 0.53, -0.050),
        ("skew_plusminus_trim75mv", "skew +20/-20, trim -75 mV", 0.57, 0.53, -0.075),
    ]
    shift_threshold_labels = []
    shift_threshold_gate_samples = []
    shift_threshold_load_samples = []
    shift_threshold_store_samples = []
    shift_threshold_traces = []
    for name, label, nfp_vto, nfm_vto, gate_diff_offset_v in shift_threshold_cases:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_threshold_{name}"
        )
        nfp_model = f"NSHFP_{name.upper()}"
        nfm_model = f"NSHFM_{name.upper()}"
        shift_threshold_models = (
            f".model {nfp_model} NMOS (LEVEL=1 VTO={nfp_vto:.2f} KP=220u LAMBDA=0.03)\n"
            f".model {nfm_model} NMOS (LEVEL=1 VTO={nfm_vto:.2f} KP=220u LAMBDA=0.03)"
        )
        shift_threshold_deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
        shift_threshold_deck = replace_required(shift_threshold_deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        shift_threshold_deck = replace_required(shift_threshold_deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        shift_threshold_deck = replace_required(
            shift_threshold_deck,
            corner_param_line,
            shift_threshold_models + "\n" + corner_param_line,
        )
        shift_threshold_deck = replace_required(
            shift_threshold_deck,
            high_gain_forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                gate_ic_p=0.90 + 0.5 * gate_diff_offset_v,
                gate_ic_m=0.90 - 0.5 * gate_diff_offset_v,
                forward_p_model=nfp_model,
                forward_m_model=nfm_model,
            ),
        )
        shift_threshold_deck = replace_required(
            shift_threshold_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        shift_threshold_deck = add_shifted_gate_probes(shift_threshold_deck, stem)
        shift_threshold_data = run_ngspice(shift_threshold_deck, stem)
        stt, shift_threshold_cols = load_wrdata(shift_threshold_data, 25)

        def shtat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(stt - time_s))])

        shift_threshold_gate_diff = shift_threshold_cols[0] - shift_threshold_cols[1]
        shift_threshold_load = shift_threshold_cols[14] - shift_threshold_cols[13]
        shift_threshold_store = shift_threshold_cols[16] - shift_threshold_cols[15]
        shift_threshold_labels.append(label)
        shift_threshold_gate_samples.append(shtat(3.315e-6, shift_threshold_gate_diff))
        shift_threshold_load_samples.append(shtat(3.315e-6, shift_threshold_load))
        shift_threshold_store_samples.append(shtat(3.575e-6, shift_threshold_store))
        shift_threshold_traces.append((label, stt, shift_threshold_load, shift_threshold_store))

    shift_threshold_gate_samples = np.array(shift_threshold_gate_samples)
    shift_threshold_load_samples = np.array(shift_threshold_load_samples)
    shift_threshold_store_samples = np.array(shift_threshold_store_samples)
    nominal_shift_threshold_idx = 0
    require(
        shift_threshold_store_samples[nominal_shift_threshold_idx] > 0.045,
        "shifted-gate threshold nominal case should reproduce a useful low-common-mode activation",
    )
    untrimmed_threshold_indices = np.arange(7)
    threshold_hazard_idx = 5
    threshold_trim25_idx = 7
    threshold_trim50_idx = 8
    threshold_trim75_idx = 9
    require(
        np.all(np.delete(shift_threshold_store_samples[untrimmed_threshold_indices], threshold_hazard_idx) > 0.015),
        "untrimmed shifted-gate threshold corners except the known skew hazard should keep positive activation",
    )
    require(
        shift_threshold_store_samples[threshold_hazard_idx] < -0.010,
        "untrimmed +20/-20 mV input threshold skew should expose the shifted-gate polarity hazard",
    )
    require(
        shift_threshold_store_samples[threshold_trim25_idx] > shift_threshold_store_samples[threshold_hazard_idx] + 0.010,
        "25 mV helpful gate trim should materially improve the +20/-20 mV threshold skew hazard",
    )
    require(
        shift_threshold_store_samples[threshold_trim50_idx] > 0.005,
        "50 mV helpful gate trim should recover positive activation under the +20/-20 mV threshold skew",
    )
    require(
        shift_threshold_store_samples[threshold_trim75_idx] > shift_threshold_store_samples[threshold_trim50_idx] + 0.010,
        "75 mV helpful gate trim should add useful recovery margin under the +20/-20 mV threshold skew",
    )
    require(
        shift_threshold_store_samples[1] > shift_threshold_store_samples[2],
        "matched weak input pair should reduce activation relative to matched strong input pair",
    )
    require(
        np.max(shift_threshold_store_samples) - np.min(shift_threshold_store_samples) > 0.010,
        "threshold skew sweep should expose a visible activation-margin spread",
    )
    require(
        np.max(
            np.abs(
                shift_threshold_gate_samples[untrimmed_threshold_indices]
                - shift_threshold_gate_samples[nominal_shift_threshold_idx]
            )
        )
        < 0.001,
        "untrimmed input-pair threshold sweep should not change the passive shifted-gate sampled differential",
    )

    shift_trimmed_reset_cases = [
        ("untrimmed", "reset trim 0 mV", 0.0),
        ("trim10mv", "reset trim -10 mV", -0.010),
        ("trim15mv", "reset trim -15 mV", -0.015),
        ("trim20mv", "reset trim -20 mV", -0.020),
        ("trim25mv", "reset trim -25 mV", -0.025),
        ("trim50mv", "reset trim -50 mV", -0.050),
        ("trim75mv", "reset trim -75 mV", -0.075),
        ("trim100mv", "reset trim -100 mV", -0.100),
    ]
    shift_trimmed_reset_labels = []
    shift_trimmed_reset_gate_pre_samples = []
    shift_trimmed_reset_gate_common_samples = []
    shift_trimmed_reset_load_samples = []
    shift_trimmed_reset_store_samples = []
    shift_trimmed_reset_traces = []
    for name, label, reset_trim_v in shift_trimmed_reset_cases:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_trimmed_reset_{name}"
        )
        nfp_model = f"NSHTRFP_{name.upper()}"
        nfm_model = f"NSHTRFM_{name.upper()}"
        shift_trimmed_reset_models = (
            f".model {nfp_model} NMOS (LEVEL=1 VTO=0.57 KP=220u LAMBDA=0.03)\n"
            f".model {nfm_model} NMOS (LEVEL=1 VTO=0.53 KP=220u LAMBDA=0.03)"
        )
        shift_trimmed_reset_deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
        shift_trimmed_reset_deck = replace_required(
            shift_trimmed_reset_deck,
            zcm_cap_p_line,
            "CZP_HYR zp_hyr 0 {CSUM} IC=0.75",
        )
        shift_trimmed_reset_deck = replace_required(
            shift_trimmed_reset_deck,
            zcm_cap_m_line,
            "CZM_HYR zm_hyr 0 {CSUM} IC=0.75",
        )
        shift_trimmed_reset_deck = replace_required(
            shift_trimmed_reset_deck,
            corner_param_line,
            shift_trimmed_reset_models + "\n" + corner_param_line,
        )
        shift_trimmed_reset_deck = replace_required(
            shift_trimmed_reset_deck,
            high_gain_forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                gate_ic_p=0.35,
                gate_ic_m=1.45,
                reset_ref_p=0.90 + 0.5 * reset_trim_v,
                reset_ref_m=0.90 - 0.5 * reset_trim_v,
                forward_p_model=nfp_model,
                forward_m_model=nfm_model,
            ),
        )
        shift_trimmed_reset_deck = replace_required(
            shift_trimmed_reset_deck,
            "VRESET_HYR rst_hyr 0 0",
            shift_reset_pulse,
        )
        shift_trimmed_reset_deck = replace_required(
            shift_trimmed_reset_deck,
            "VRESETN_HYR rstn_hyr 0 1.8",
            shift_resetn_pulse,
        )
        shift_trimmed_reset_deck = replace_required(
            shift_trimmed_reset_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        shift_trimmed_reset_deck = add_shifted_gate_probes(shift_trimmed_reset_deck, stem)
        shift_trimmed_reset_data = run_ngspice(shift_trimmed_reset_deck, stem)
        strt, shift_trimmed_reset_cols = load_wrdata(shift_trimmed_reset_data, 25)

        def strtat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(strt - time_s))])

        shift_trimmed_reset_gate_diff = shift_trimmed_reset_cols[0] - shift_trimmed_reset_cols[1]
        shift_trimmed_reset_gate_common = 0.5 * (shift_trimmed_reset_cols[0] + shift_trimmed_reset_cols[1])
        shift_trimmed_reset_load = shift_trimmed_reset_cols[14] - shift_trimmed_reset_cols[13]
        shift_trimmed_reset_store = shift_trimmed_reset_cols[16] - shift_trimmed_reset_cols[15]
        shift_trimmed_reset_labels.append(label)
        shift_trimmed_reset_gate_pre_samples.append(strtat(2.70e-6, shift_trimmed_reset_gate_diff))
        shift_trimmed_reset_gate_common_samples.append(strtat(2.70e-6, shift_trimmed_reset_gate_common))
        shift_trimmed_reset_load_samples.append(strtat(3.315e-6, shift_trimmed_reset_load))
        shift_trimmed_reset_store_samples.append(strtat(3.575e-6, shift_trimmed_reset_store))
        shift_trimmed_reset_traces.append(
            (label, strt, shift_trimmed_reset_gate_diff, shift_trimmed_reset_load, shift_trimmed_reset_store)
        )

    shift_trimmed_reset_gate_pre_samples = np.array(shift_trimmed_reset_gate_pre_samples)
    shift_trimmed_reset_gate_common_samples = np.array(shift_trimmed_reset_gate_common_samples)
    shift_trimmed_reset_load_samples = np.array(shift_trimmed_reset_load_samples)
    shift_trimmed_reset_store_samples = np.array(shift_trimmed_reset_store_samples)
    require(
        shift_trimmed_reset_store_samples[0] < -0.010,
        "physical untrimmed reset should reproduce the +20/-20 mV threshold-skew polarity hazard",
    )
    require(
        np.all(np.diff(shift_trimmed_reset_store_samples) > 0.008),
        "larger helpful split reset trim should monotonically recover the skewed shifted-gate activation",
    )
    require(
        shift_trimmed_reset_store_samples[2] > 0.005,
        "15 mV physical split reset trim should recover positive sign under threshold skew",
    )
    require(
        shift_trimmed_reset_store_samples[4] > 0.020,
        "25 mV physical split reset trim should recover useful margin under threshold skew",
    )
    require(
        shift_trimmed_reset_store_samples[5] > 0.070,
        "50 mV physical split reset trim should add margin beyond the minimally recovered case",
    )
    require(
        np.max(np.abs(shift_trimmed_reset_gate_common_samples - 0.90)) < 0.006,
        "split shifted-gate reset should preserve the shifted-gate common mode before read",
    )
    require(
        np.max(
            np.abs(
                shift_trimmed_reset_gate_pre_samples
                - np.array([case[2] for case in shift_trimmed_reset_cases], dtype=float)
            )
        )
        < 0.003,
        "split shifted-gate reset should establish the requested trim differential before read",
    )

    def run_split_reset_threshold_case(
        stem: str,
        model_tag: str,
        nfp_vto: float,
        nfm_vto: float,
        reset_trim_v: float,
        reset_common_v: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nfp_model = f"NSHBALFP_{model_tag}"
        nfm_model = f"NSHBALFM_{model_tag}"
        models = (
            f".model {nfp_model} NMOS (LEVEL=1 VTO={nfp_vto:.2f} KP=220u LAMBDA=0.03)\n"
            f".model {nfm_model} NMOS (LEVEL=1 VTO={nfm_vto:.2f} KP=220u LAMBDA=0.03)"
        )
        deck = replace_required(high_gain_hold_deck, zcm_source_line, "VZCM zcm 0 0.75")
        deck = replace_required(deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, corner_param_line, models + "\n" + corner_param_line)
        deck = replace_required(
            deck,
            high_gain_forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                gate_ic_p=0.35,
                gate_ic_m=1.45,
                reset_ref_p=reset_common_v + 0.5 * reset_trim_v,
                reset_ref_m=reset_common_v - 0.5 * reset_trim_v,
                forward_p_model=nfp_model,
                forward_m_model=nfm_model,
            ),
        )
        deck = replace_required(deck, "VRESET_HYR rst_hyr 0 0", shift_reset_pulse)
        deck = replace_required(deck, "VRESETN_HYR rstn_hyr 0 1.8", shift_resetn_pulse)
        deck = replace_required(
            deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_hold.dat",
            f"{stem}.dat",
        )
        deck = add_shifted_gate_probes(deck, stem)
        data = run_ngspice(deck, stem)
        time, cols = load_wrdata(data, 25)
        gate_diff = cols[0] - cols[1]
        gate_common = 0.5 * (cols[0] + cols[1])
        load = cols[14] - cols[13]
        store = cols[16] - cols[15]
        return time, gate_diff, gate_common, load, store

    shift_trim_common_cases = [
        ("cm080", "0.80 V tuned common", 0.80, [-0.025, -0.035, -0.045]),
        ("cm090", "0.90 V legacy common", 0.90, [-0.025, -0.035, -0.045]),
    ]
    shift_trim_common_labels = []
    shift_trim_common_values = []
    shift_trim_common_gate_samples = []
    shift_trim_common_common_samples = []
    shift_trim_common_load_samples = []
    shift_trim_common_store_samples = []
    shift_trim_common_traces = []
    for common_name, common_label, reset_common_v, trim_values in shift_trim_common_cases:
        row_gate_samples = []
        row_common_samples = []
        row_load_samples = []
        row_store_samples = []
        for reset_trim_v in trim_values:
            trim_mv = int(round(abs(reset_trim_v) * 1e3))
            stem = (
                "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
                f"forward_pair_96u_zcm_0p75v_shift_trim_common_{common_name}_trim{trim_mv}"
            )
            time, gate_diff, gate_common, load, store = run_split_reset_threshold_case(
                stem,
                f"{common_name.upper()}_TRIM{trim_mv}",
                0.57,
                0.53,
                reset_trim_v,
                reset_common_v=reset_common_v,
            )

            def tcat(time_s: float, values: np.ndarray) -> float:
                return float(values[np.argmin(np.abs(time - time_s))])

            row_gate_samples.append(tcat(2.70e-6, gate_diff))
            row_common_samples.append(tcat(2.70e-6, gate_common))
            row_load_samples.append(tcat(3.315e-6, load))
            row_store_samples.append(tcat(3.575e-6, store))
            if abs(reset_trim_v + 0.035) < 1e-12:
                shift_trim_common_traces.append((common_label, time, load, store))
        shift_trim_common_labels.append(common_label)
        shift_trim_common_values.append(trim_values)
        shift_trim_common_gate_samples.append(row_gate_samples)
        shift_trim_common_common_samples.append(row_common_samples)
        shift_trim_common_load_samples.append(row_load_samples)
        shift_trim_common_store_samples.append(row_store_samples)

    shift_trim_common_values = np.array(shift_trim_common_values)
    shift_trim_common_gate_samples = np.array(shift_trim_common_gate_samples)
    shift_trim_common_common_samples = np.array(shift_trim_common_common_samples)
    shift_trim_common_load_samples = np.array(shift_trim_common_load_samples)
    shift_trim_common_store_samples = np.array(shift_trim_common_store_samples)
    require(
        np.max(np.abs(shift_trim_common_gate_samples - shift_trim_common_values)) < 0.003,
        "split reset should establish requested trim differentials at both reset common modes",
    )
    require(
        np.max(np.abs(shift_trim_common_common_samples[0] - 0.80)) < 0.006
        and np.max(np.abs(shift_trim_common_common_samples[1] - 0.90)) < 0.006,
        "split reset should establish both tested reset common modes before read",
    )
    require(
        np.all(np.diff(shift_trim_common_store_samples, axis=1) > 0.010),
        "helpful split trim should monotonically increase stored activation at both reset common modes",
    )
    require(
        np.all((shift_trim_common_store_samples > 0.020) & (shift_trim_common_store_samples < 0.085)),
        "tested split-trim/common-mode combinations should stay positive and non-overdriven",
    )
    require(
        np.all(shift_trim_common_store_samples[1] - shift_trim_common_store_samples[0] > 0.002),
        "legacy 0.90 V reset common should store slightly higher than tuned 0.80 V for the same trim",
    )
    require(
        0.035 < shift_trim_common_store_samples[0, 1] < 0.055,
        "tuned 0.80 V reset common with the calibrated -35 mV trim should keep useful skew recovery",
    )

    shift_balance_cases = [
        ("nominal", "nominal", 0.55, 0.55, 0.0),
        ("hp_weak_untrimmed", "$h^+$ weak, untrimmed", 0.57, 0.55, 0.0),
        ("hp_weak_trimmed", "$h^+$ weak, trim -20 mV", 0.57, 0.55, -0.020),
        ("hm_weak_untrimmed", "$h^-$ weak, untrimmed", 0.55, 0.57, 0.0),
        ("hm_weak_trimmed", "$h^-$ weak, trim +15 mV", 0.55, 0.57, 0.015),
        ("skew_pm_untrimmed", "skew +20/-20, untrimmed", 0.57, 0.53, 0.0),
        ("skew_pm_trimmed", "skew +20/-20, trim -35 mV", 0.57, 0.53, -0.035),
        ("skew_mp_untrimmed", "skew -20/+20, untrimmed", 0.53, 0.57, 0.0),
        ("skew_mp_trimmed", "skew -20/+20, trim +35 mV", 0.53, 0.57, 0.035),
    ]
    shift_balance_labels = []
    shift_balance_gate_samples = []
    shift_balance_load_samples = []
    shift_balance_store_samples = []
    shift_balance_traces = []
    for name, label, nfp_vto, nfm_vto, reset_trim_v in shift_balance_cases:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_balance_{name}"
        )
        time, gate_diff, _gate_common, load, store = run_split_reset_threshold_case(
            stem,
            name.upper(),
            nfp_vto,
            nfm_vto,
            reset_trim_v,
        )

        def sbalat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(time - time_s))])

        shift_balance_labels.append(label)
        shift_balance_gate_samples.append(sbalat(2.70e-6, gate_diff))
        shift_balance_load_samples.append(sbalat(3.315e-6, load))
        shift_balance_store_samples.append(sbalat(3.575e-6, store))
        shift_balance_traces.append((label, time, load, store))

    shift_balance_gate_samples = np.array(shift_balance_gate_samples)
    shift_balance_load_samples = np.array(shift_balance_load_samples)
    shift_balance_store_samples = np.array(shift_balance_store_samples)
    shift_balance_index = {case[0]: idx for idx, case in enumerate(shift_balance_cases)}
    shift_balance_nominal = shift_balance_store_samples[shift_balance_index["nominal"]]
    for untrimmed_name, trimmed_name in [
        ("hp_weak_untrimmed", "hp_weak_trimmed"),
        ("hm_weak_untrimmed", "hm_weak_trimmed"),
        ("skew_pm_untrimmed", "skew_pm_trimmed"),
        ("skew_mp_untrimmed", "skew_mp_trimmed"),
    ]:
        untrimmed_idx = shift_balance_index[untrimmed_name]
        trimmed_idx = shift_balance_index[trimmed_name]
        require(
            abs(shift_balance_store_samples[trimmed_idx] - shift_balance_nominal)
            < abs(shift_balance_store_samples[untrimmed_idx] - shift_balance_nominal),
            f"{trimmed_name} should move closer to nominal split-reset activation than {untrimmed_name}",
        )
    shift_balance_trimmed_indices = [
        shift_balance_index[name]
        for name in ["hp_weak_trimmed", "hm_weak_trimmed", "skew_pm_trimmed", "skew_mp_trimmed"]
    ]
    require(
        np.all(shift_balance_store_samples[shift_balance_trimmed_indices] > 0.030),
        "calibrated split-reset threshold corners should keep useful positive activation",
    )
    require(
        np.all(shift_balance_store_samples[shift_balance_trimmed_indices] < 0.090),
        "calibrated split-reset threshold corners should avoid the uncalibrated overdrive cases",
    )
    require(
        shift_balance_store_samples[shift_balance_index["skew_pm_untrimmed"]] < -0.010,
        "untrimmed split reset should still expose the +20/-20 threshold-skew sign flip in the balance deck",
    )
    require(
        shift_balance_store_samples[shift_balance_index["skew_mp_untrimmed"]] > 0.10,
        "untrimmed split reset should expose the opposite skew overdrive in the balance deck",
    )

    shift_trimmed_reuse_stem = (
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
        "forward_pair_96u_zcm_0p75v_shift_trimmed_reuse_skew_pm"
    )
    shift_trimmed_reuse_models = (
        ".model NSHTRREUSEFP NMOS (LEVEL=1 VTO=0.57 KP=220u LAMBDA=0.03)\n"
        ".model NSHTRREUSEFM NMOS (LEVEL=1 VTO=0.53 KP=220u LAMBDA=0.03)"
    )
    shift_trimmed_reuse_deck = replace_required(hybrid_forward_read_reuse_deck, zcm_source_line, "VZCM zcm 0 0.75")
    shift_trimmed_reuse_deck = replace_required(
        shift_trimmed_reuse_deck,
        zcm_cap_p_line,
        "CZP_HYR zp_hyr 0 {CSUM} IC=0.75",
    )
    shift_trimmed_reuse_deck = replace_required(
        shift_trimmed_reuse_deck,
        zcm_cap_m_line,
        "CZM_HYR zm_hyr 0 {CSUM} IC=0.75",
    )
    shift_trimmed_reuse_deck = replace_required(
        shift_trimmed_reuse_deck,
        corner_param_line,
        shift_trimmed_reuse_models + "\n" + corner_param_line,
    )
    shift_trimmed_reuse_deck = replace_required(
        shift_trimmed_reuse_deck,
        forward_pair_lines,
        shifted_forward_pair_lines(
            10.0,
            reset_ref_p=0.90 - 0.0175,
            reset_ref_m=0.90 + 0.0175,
            forward_p_model="NSHTRREUSEFP",
            forward_m_model="NSHTRREUSEFM",
        ),
    )
    shift_trimmed_reuse_deck = replace_required(
        shift_trimmed_reuse_deck,
        "VRESET_HYR rst_hyr 0 PWL(0 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 "
        "5.18u 0 5.20u 1.8 5.60u 1.8 5.62u 0 7.8u 0)",
        "VRESET_HYR rst_hyr 0 PWL(0 0 2.48u 0 2.50u 1.8 2.62u 1.8 2.64u 0 "
        "3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 5.18u 0 5.20u 1.8 5.60u 1.8 "
        "5.62u 0 7.8u 0)",
    )
    shift_trimmed_reuse_deck = replace_required(
        shift_trimmed_reuse_deck,
        "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 "
        "5.18u 1.8 5.20u 0 5.60u 0 5.62u 1.8 7.8u 1.8)",
        "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 2.48u 1.8 2.50u 0 2.62u 0 2.64u 1.8 "
        "3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 5.18u 1.8 5.20u 0 5.60u 0 "
        "5.62u 1.8 7.8u 1.8)",
    )
    shift_trimmed_reuse_deck = replace_required(
        shift_trimmed_reuse_deck,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
        f"{shift_trimmed_reuse_stem}.dat",
    )
    shift_trimmed_reuse_deck = add_shifted_gate_probes(shift_trimmed_reuse_deck, shift_trimmed_reuse_stem)
    shift_trimmed_reuse_data = run_ngspice(shift_trimmed_reuse_deck, shift_trimmed_reuse_stem)
    strpt, shift_trimmed_reuse_cols = load_wrdata(shift_trimmed_reuse_data, 25)

    def strpat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(strpt - time_s))])

    shift_trimmed_reuse_gate_diff = shift_trimmed_reuse_cols[0] - shift_trimmed_reuse_cols[1]
    shift_trimmed_reuse_gate_common = 0.5 * (shift_trimmed_reuse_cols[0] + shift_trimmed_reuse_cols[1])
    shift_trimmed_reuse_weight = shift_trimmed_reuse_cols[7] - shift_trimmed_reuse_cols[8]
    shift_trimmed_reuse_bias = shift_trimmed_reuse_cols[9] - shift_trimmed_reuse_cols[10]
    shift_trimmed_reuse_preact = shift_trimmed_reuse_cols[12] - shift_trimmed_reuse_cols[11]
    shift_trimmed_reuse_load = shift_trimmed_reuse_cols[14] - shift_trimmed_reuse_cols[13]
    shift_trimmed_reuse_store = shift_trimmed_reuse_cols[16] - shift_trimmed_reuse_cols[15]
    shift_trimmed_reuse_z_samples = np.array([strpat(ts, shift_trimmed_reuse_preact) for ts in shift_reuse_z_times])
    shift_trimmed_reuse_h_samples = np.array([strpat(ts, shift_trimmed_reuse_store) for ts in shift_reuse_h_times])
    shift_trimmed_reuse_reset_times = np.array([2.70, 4.10, 5.70]) * 1e-6
    shift_trimmed_reuse_gate_reset_diff = np.array(
        [strpat(ts, shift_trimmed_reuse_gate_diff) for ts in shift_trimmed_reuse_reset_times]
    )
    shift_trimmed_reuse_gate_reset_common = np.array(
        [strpat(ts, shift_trimmed_reuse_gate_common) for ts in shift_trimmed_reuse_reset_times]
    )
    shift_trimmed_reuse_z_reset = np.array(
        [abs(strpat(ts, shift_trimmed_reuse_preact)) for ts in shift_trimmed_reuse_reset_times]
    )
    shift_trimmed_reuse_h_reset = np.array(
        [abs(strpat(ts, shift_trimmed_reuse_store)) for ts in shift_trimmed_reuse_reset_times]
    )
    shift_trimmed_reuse_weight_after_write = strpat(2.55e-6, shift_trimmed_reuse_weight)
    shift_trimmed_reuse_bias_after_write = strpat(2.55e-6, shift_trimmed_reuse_bias)
    shift_trimmed_reuse_weight_drift = strpat(7.45e-6, shift_trimmed_reuse_weight) - shift_trimmed_reuse_weight_after_write
    shift_trimmed_reuse_bias_drift = strpat(7.45e-6, shift_trimmed_reuse_bias) - shift_trimmed_reuse_bias_after_write
    require(
        np.all(shift_trimmed_reuse_z_samples > 0.040),
        "trimmed split-reset reuse deck should repeatedly read useful low-common-mode preactivation",
    )
    require(
        np.all(shift_trimmed_reuse_h_samples > 0.035),
        "trimmed split-reset reuse deck should repeatedly store useful calibrated activation",
    )
    require(
        np.max(shift_trimmed_reuse_h_samples) - np.min(shift_trimmed_reuse_h_samples) < 0.020,
        "trimmed split-reset reuse stored activations should stay within a 20 mV cycle window",
    )
    require(
        np.max(np.abs(shift_trimmed_reuse_gate_reset_diff + 0.035)) < 0.003,
        "trimmed split-reset reuse should re-establish the calibrated gate differential between reads",
    )
    require(
        np.max(np.abs(shift_trimmed_reuse_gate_reset_common - 0.90)) < 0.005,
        "trimmed split-reset reuse should preserve shifted-gate common mode between reads",
    )
    require(
        np.max(shift_trimmed_reuse_z_reset) < 0.001,
        "trimmed split-reset reuse reset should clear preactivation between reads",
    )
    require(
        np.max(shift_trimmed_reuse_h_reset) < 0.001,
        "trimmed split-reset reuse reset should clear stored activation between reads",
    )
    require(
        abs(shift_trimmed_reuse_weight_drift) < 1e-5,
        "trimmed split-reset reuse should not disturb weight state",
    )
    require(
        abs(shift_trimmed_reuse_bias_drift) < 1e-5,
        "trimmed split-reset reuse should not disturb bias state",
    )

    def run_trimmed_reuse_margin_case(
        name: str,
        reset_trim_v: float,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        model_tag = name.upper()
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_trimmed_reuse_margin_{name}"
        )
        models = (
            f".model NSHTRMFP_{model_tag} NMOS (LEVEL=1 VTO=0.57 KP=220u LAMBDA=0.03)\n"
            f".model NSHTRMFM_{model_tag} NMOS (LEVEL=1 VTO=0.53 KP=220u LAMBDA=0.03)"
        )
        deck = replace_required(hybrid_forward_read_reuse_deck, zcm_source_line, "VZCM zcm 0 0.75")
        deck = replace_required(deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, corner_param_line, models + "\n" + corner_param_line)
        deck = replace_required(
            deck,
            forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                reset_ref_p=0.90 + 0.5 * reset_trim_v,
                reset_ref_m=0.90 - 0.5 * reset_trim_v,
                forward_p_model=f"NSHTRMFP_{model_tag}",
                forward_m_model=f"NSHTRMFM_{model_tag}",
            ),
        )
        deck = replace_required(
            deck,
            "VRESET_HYR rst_hyr 0 PWL(0 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 "
            "5.18u 0 5.20u 1.8 5.60u 1.8 5.62u 0 7.8u 0)",
            "VRESET_HYR rst_hyr 0 PWL(0 0 2.48u 0 2.50u 1.8 2.62u 1.8 2.64u 0 "
            "3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 5.18u 0 5.20u 1.8 5.60u 1.8 "
            "5.62u 0 7.8u 0)",
        )
        deck = replace_required(
            deck,
            "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 "
            "5.18u 1.8 5.20u 0 5.60u 0 5.62u 1.8 7.8u 1.8)",
            "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 2.48u 1.8 2.50u 0 2.62u 0 2.64u 1.8 "
            "3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 5.18u 1.8 5.20u 0 5.60u 0 "
            "5.62u 1.8 7.8u 1.8)",
        )
        deck = replace_required(
            deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        deck = add_shifted_gate_probes(deck, stem)
        data = run_ngspice(deck, stem)
        return load_wrdata(data, 25)

    shift_trimmed_margin_cases = [
        ("trim25mv", "-25 mV", -0.025),
        ("trim30mv", "-30 mV", -0.030),
        ("trim35mv", "-35 mV", -0.035),
        ("trim40mv", "-40 mV", -0.040),
        ("trim45mv", "-45 mV", -0.045),
    ]
    shift_trimmed_margin_labels = []
    shift_trimmed_margin_trim_values = []
    shift_trimmed_margin_gate_samples = []
    shift_trimmed_margin_common_samples = []
    shift_trimmed_margin_z_samples = []
    shift_trimmed_margin_h_samples = []
    shift_trimmed_margin_traces = []
    for name, label, reset_trim_v in shift_trimmed_margin_cases:
        if abs(reset_trim_v + 0.035) < 1e-12:
            mt, margin_cols = strpt, shift_trimmed_reuse_cols
        else:
            mt, margin_cols = run_trimmed_reuse_margin_case(name, reset_trim_v)

        def smat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(mt - time_s))])

        gate_diff = margin_cols[0] - margin_cols[1]
        gate_common = 0.5 * (margin_cols[0] + margin_cols[1])
        z = margin_cols[12] - margin_cols[11]
        load = margin_cols[14] - margin_cols[13]
        store = margin_cols[16] - margin_cols[15]
        gate_samples = np.array([smat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
        common_samples = np.array([smat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
        z_samples = np.array([smat(ts, z) for ts in shift_reuse_z_times])
        h_samples = np.array([smat(ts, store) for ts in shift_reuse_h_times])
        z_reset = np.array([abs(smat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
        h_reset = np.array([abs(smat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
        require(
            np.max(np.abs(gate_samples - reset_trim_v)) < 0.003,
            f"{label} trimmed reuse margin case should establish requested split-reset differential",
        )
        require(
            np.max(np.abs(common_samples - 0.90)) < 0.006,
            f"{label} trimmed reuse margin case should preserve reset common mode",
        )
        require(
            np.max(z_reset) < 0.001,
            f"{label} trimmed reuse margin case should clear preactivation during reset",
        )
        require(
            np.max(h_reset) < 0.001,
            f"{label} trimmed reuse margin case should clear stored activation during reset",
        )
        require(
            np.all(z_samples > 0.035),
            f"{label} trimmed reuse margin case should keep useful read preactivation",
        )
        require(
            np.all(h_samples > 0.020),
            f"{label} trimmed reuse margin case should keep positive stored activation",
        )
        require(
            np.max(h_samples) < 0.085,
            f"{label} trimmed reuse margin case should avoid overdriving the stored activation",
        )
        require(
            np.max(h_samples) - np.min(h_samples) < 0.025,
            f"{label} trimmed reuse margin case should remain stable across repeated cycles",
        )
        shift_trimmed_margin_labels.append(label)
        shift_trimmed_margin_trim_values.append(reset_trim_v)
        shift_trimmed_margin_gate_samples.append(gate_samples)
        shift_trimmed_margin_common_samples.append(common_samples)
        shift_trimmed_margin_z_samples.append(z_samples)
        shift_trimmed_margin_h_samples.append(h_samples)
        if name in {"trim25mv", "trim35mv", "trim45mv"}:
            shift_trimmed_margin_traces.append((label, mt, load, store))

    shift_trimmed_margin_trim_values = np.array(shift_trimmed_margin_trim_values)
    shift_trimmed_margin_gate_samples = np.array(shift_trimmed_margin_gate_samples)
    shift_trimmed_margin_common_samples = np.array(shift_trimmed_margin_common_samples)
    shift_trimmed_margin_z_samples = np.array(shift_trimmed_margin_z_samples)
    shift_trimmed_margin_h_samples = np.array(shift_trimmed_margin_h_samples)
    shift_trimmed_margin_h_mean = np.mean(shift_trimmed_margin_h_samples, axis=1)
    shift_trimmed_margin_h_min = np.min(shift_trimmed_margin_h_samples, axis=1)
    shift_trimmed_margin_h_max = np.max(shift_trimmed_margin_h_samples, axis=1)
    require(
        np.all(np.diff(shift_trimmed_margin_h_mean) > 0.004),
        "trimmed reuse margin should respond monotonically to stronger helpful reset trim",
    )
    require(
        shift_trimmed_margin_h_min[0] > 0.020 and shift_trimmed_margin_h_max[-1] < 0.085,
        "trimmed reuse margin sweep should bracket a useful non-overdriven calibration window",
    )

    def run_trimmed_reuse_reference_perturb_case(
        name: str,
        reset_trims_v: tuple[float, float, float],
        reset_common_v: tuple[float, float, float],
        nfp_vto: float = 0.570,
        nfm_vto: float = 0.530,
        nominal_reset_trim_v: float = -0.035,
        family: str = "refpert",
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        model_tag = name.upper()
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_trimmed_reuse_{family}_{name}"
        )
        models = (
            f".model NSHTRPFP_{model_tag} NMOS (LEVEL=1 VTO={nfp_vto:.3f} KP=220u LAMBDA=0.03)\n"
            f".model NSHTRPFM_{model_tag} NMOS (LEVEL=1 VTO={nfm_vto:.3f} KP=220u LAMBDA=0.03)"
        )
        ref_p_values = [common + 0.5 * trim for common, trim in zip(reset_common_v, reset_trims_v)]
        ref_m_values = [common - 0.5 * trim for common, trim in zip(reset_common_v, reset_trims_v)]
        nominal_ref_p = 0.90 + 0.5 * nominal_reset_trim_v
        nominal_ref_m = 0.90 - 0.5 * nominal_reset_trim_v

        def fmt_reset_ref(value: float) -> str:
            if abs(value - round(value, 2)) < 1e-12:
                return f"{value:.2f}"
            return f"{value:.5f}"

        def ref_pwl(node: str, values: list[float]) -> str:
            return (
                f"V{node.upper()}_HYR {node}_hyr 0 PWL(0 {values[0]:.5f} "
                f"3.55u {values[1]:.5f} 5.15u {values[2]:.5f} 7.8u {values[2]:.5f})"
            )

        deck = replace_required(hybrid_forward_read_reuse_deck, zcm_source_line, "VZCM zcm 0 0.75")
        deck = replace_required(deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, corner_param_line, models + "\n" + corner_param_line)
        deck = replace_required(
            deck,
            forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                reset_ref_p=nominal_ref_p,
                reset_ref_m=nominal_ref_m,
                forward_p_model=f"NSHTRPFP_{model_tag}",
                forward_m_model=f"NSHTRPFM_{model_tag}",
            ),
        )
        deck = replace_required(
            deck,
            f"VZGRP_HYR zgrp_hyr 0 {fmt_reset_ref(nominal_ref_p)}",
            ref_pwl("zgrp", ref_p_values),
        )
        deck = replace_required(
            deck,
            f"VZGRM_HYR zgrm_hyr 0 {fmt_reset_ref(nominal_ref_m)}",
            ref_pwl("zgrm", ref_m_values),
        )
        deck = replace_required(
            deck,
            "VRESET_HYR rst_hyr 0 PWL(0 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 "
            "5.18u 0 5.20u 1.8 5.60u 1.8 5.62u 0 7.8u 0)",
            "VRESET_HYR rst_hyr 0 PWL(0 0 2.48u 0 2.50u 1.8 2.62u 1.8 2.64u 0 "
            "3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 5.18u 0 5.20u 1.8 5.60u 1.8 "
            "5.62u 0 7.8u 0)",
        )
        deck = replace_required(
            deck,
            "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 "
            "5.18u 1.8 5.20u 0 5.60u 0 5.62u 1.8 7.8u 1.8)",
            "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 2.48u 1.8 2.50u 0 2.62u 0 2.64u 1.8 "
            "3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 5.18u 1.8 5.20u 0 5.60u 0 "
            "5.62u 1.8 7.8u 1.8)",
        )
        deck = replace_required(
            deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        deck = add_shifted_gate_probes(deck, stem)
        data = run_ngspice(deck, stem)
        return load_wrdata(data, 25)

    shift_refpert_cases = [
        ("clean", "clean refs", (-0.035, -0.035, -0.035), (0.90, 0.90, 0.90)),
        ("common_jitter", "common jitter", (-0.035, -0.035, -0.035), (0.88, 0.92, 0.89)),
        ("diff_jitter5", "diff jitter +/-5 mV", (-0.030, -0.035, -0.040), (0.90, 0.90, 0.90)),
        ("diff_jitter10", "diff jitter +/-10 mV", (-0.025, -0.035, -0.045), (0.90, 0.90, 0.90)),
        ("mixed_jitter", "mixed jitter", (-0.030, -0.040, -0.035), (0.885, 0.915, 0.900)),
    ]
    shift_refpert_labels = []
    shift_refpert_gate_samples = []
    shift_refpert_common_samples = []
    shift_refpert_z_samples = []
    shift_refpert_h_samples = []
    shift_refpert_expected_trims = []
    shift_refpert_expected_common = []
    shift_refpert_traces = []
    for name, label, reset_trims_v, reset_common_v in shift_refpert_cases:
        if name == "clean":
            rt, refpert_cols = strpt, shift_trimmed_reuse_cols
        else:
            rt, refpert_cols = run_trimmed_reuse_reference_perturb_case(name, reset_trims_v, reset_common_v)

        def rpat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(rt - time_s))])

        gate_diff = refpert_cols[0] - refpert_cols[1]
        gate_common = 0.5 * (refpert_cols[0] + refpert_cols[1])
        z = refpert_cols[12] - refpert_cols[11]
        load = refpert_cols[14] - refpert_cols[13]
        store = refpert_cols[16] - refpert_cols[15]
        gate_samples = np.array([rpat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
        common_samples = np.array([rpat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
        z_samples = np.array([rpat(ts, z) for ts in shift_reuse_z_times])
        h_samples = np.array([rpat(ts, store) for ts in shift_reuse_h_times])
        z_reset = np.array([abs(rpat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
        h_reset = np.array([abs(rpat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
        expected_trims = np.array(reset_trims_v)
        expected_common = np.array(reset_common_v)
        if np.max(expected_trims) - np.min(expected_trims) < 1e-12:
            require(
                np.max(np.abs(gate_samples - expected_trims)) < 0.006,
                f"{label} should preserve constant reset differential",
            )
        else:
            require(
                np.all((gate_samples > -0.047) & (gate_samples < -0.023)),
                f"{label} should keep sampled reset differential inside the calibrated trim window",
            )
        if np.max(expected_common) - np.min(expected_common) < 1e-12:
            require(
                np.max(np.abs(common_samples - expected_common)) < 0.006,
                f"{label} should preserve constant reset common mode",
            )
        else:
            require(
                np.all((common_samples > 0.86) & (common_samples < 0.93)),
                f"{label} should keep reset common mode bounded under PWL reference perturbation",
            )
        require(
            np.max(z_reset) < 0.001 and np.max(h_reset) < 0.001,
            f"{label} should still clear z/h state during reset",
        )
        require(
            np.all(z_samples > 0.035),
            f"{label} should keep useful read preactivation under reference perturbation",
        )
        require(
            np.all((h_samples > 0.020) & (h_samples < 0.085)),
            f"{label} should keep stored activation positive and non-overdriven under reference perturbation",
        )
        if name == "common_jitter":
            require(
                np.max(h_samples) - np.min(h_samples) < 0.006,
                "common-mode reference jitter should have much smaller activation effect than differential trim jitter",
            )
        if name in {"diff_jitter5", "diff_jitter10"}:
            require(
                np.all(np.diff(h_samples) > 0.004),
                f"{label} should order stored activation by stronger helpful trim each cycle",
            )
        shift_refpert_labels.append(label)
        shift_refpert_gate_samples.append(gate_samples)
        shift_refpert_common_samples.append(common_samples)
        shift_refpert_z_samples.append(z_samples)
        shift_refpert_h_samples.append(h_samples)
        shift_refpert_expected_trims.append(expected_trims)
        shift_refpert_expected_common.append(expected_common)
        if name in {"common_jitter", "diff_jitter10", "mixed_jitter"}:
            shift_refpert_traces.append((label, rt, load, store))

    shift_refpert_gate_samples = np.array(shift_refpert_gate_samples)
    shift_refpert_common_samples = np.array(shift_refpert_common_samples)
    shift_refpert_z_samples = np.array(shift_refpert_z_samples)
    shift_refpert_h_samples = np.array(shift_refpert_h_samples)
    shift_refpert_expected_trims = np.array(shift_refpert_expected_trims)
    shift_refpert_expected_common = np.array(shift_refpert_expected_common)
    shift_refpert_index = {case[0]: idx for idx, case in enumerate(shift_refpert_cases)}
    require(
        np.max(shift_refpert_h_samples[shift_refpert_index["common_jitter"]])
        - np.min(shift_refpert_h_samples[shift_refpert_index["common_jitter"]])
        < 0.25
        * (
            np.max(shift_refpert_h_samples[shift_refpert_index["diff_jitter10"]])
            - np.min(shift_refpert_h_samples[shift_refpert_index["diff_jitter10"]])
        ),
        "common-mode reset reference jitter should be secondary to differential trim jitter",
    )
    require(
        np.max(np.abs(shift_refpert_common_samples[shift_refpert_index["common_jitter"]] - np.array([0.88, 0.92, 0.89])))
        > 0.005,
        "common-mode reference perturbation deck should expose finite reset-settling/feedthrough rather than ideal tracking",
    )

    def run_trimmed_reuse_skew_law_case(
        name: str,
        nfp_vto: float,
        nfm_vto: float,
        reset_trim_v: float,
        family: str = "skewlaw",
        reset_n_vto: float | None = None,
        reset_p_vto: float | None = None,
        reset_ref_series_ohm: float = 0.0,
        reset_ref_shunt_pf: float = 0.0,
        first_reset_active_width_us: float | None = None,
        reset_common_v: float = 0.90,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        model_tag = name.upper()
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_zcm_0p75v_shift_trimmed_reuse_{family}_{name}"
        )
        model_lines = [
            f".model NSHTRSLFP_{model_tag} NMOS (LEVEL=1 VTO={nfp_vto:.3f} KP=220u LAMBDA=0.03)",
            f".model NSHTRSLFM_{model_tag} NMOS (LEVEL=1 VTO={nfm_vto:.3f} KP=220u LAMBDA=0.03)",
        ]
        reset_n_model = "NMOS"
        reset_p_model = "PMOS"
        if reset_n_vto is not None:
            reset_n_model = f"NSHTRSLRN_{model_tag}"
            model_lines.append(
                f".model {reset_n_model} NMOS (LEVEL=1 VTO={reset_n_vto:.3f} KP=220u LAMBDA=0.03)"
            )
        if reset_p_vto is not None:
            reset_p_model = f"NSHTRSLRP_{model_tag}"
            model_lines.append(
                f".model {reset_p_model} PMOS (LEVEL=1 VTO={reset_p_vto:.3f} KP=90u LAMBDA=0.03)"
            )
        models = "\n".join(model_lines)
        deck = replace_required(hybrid_forward_read_reuse_deck, zcm_source_line, "VZCM zcm 0 0.75")
        deck = replace_required(deck, zcm_cap_p_line, "CZP_HYR zp_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, zcm_cap_m_line, "CZM_HYR zm_hyr 0 {CSUM} IC=0.75")
        deck = replace_required(deck, corner_param_line, models + "\n" + corner_param_line)
        deck = replace_required(
            deck,
            forward_pair_lines,
            shifted_forward_pair_lines(
                10.0,
                reset_ref_p=reset_common_v + 0.5 * reset_trim_v,
                reset_ref_m=reset_common_v - 0.5 * reset_trim_v,
                forward_p_model=f"NSHTRSLFP_{model_tag}",
                forward_m_model=f"NSHTRSLFM_{model_tag}",
                reset_n_model=reset_n_model,
                reset_p_model=reset_p_model,
                reset_ref_series_ohm=reset_ref_series_ohm,
                reset_ref_shunt_pf=reset_ref_shunt_pf,
            ),
        )
        if first_reset_active_width_us is None:
            reset_pwl = (
                "VRESET_HYR rst_hyr 0 PWL(0 0 2.48u 0 2.50u 1.8 2.62u 1.8 2.64u 0 "
                "3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 5.18u 0 5.20u 1.8 5.60u 1.8 "
                "5.62u 0 7.8u 0)"
            )
            resetn_pwl = (
                "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 2.48u 1.8 2.50u 0 2.62u 0 2.64u 1.8 "
                "3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 5.18u 1.8 5.20u 0 5.60u 0 "
                "5.62u 1.8 7.8u 1.8)"
            )
        else:
            first_reset_end_us = 2.50 + first_reset_active_width_us
            first_reset_fall_us = first_reset_end_us + 0.02
            reset_pwl = (
                f"VRESET_HYR rst_hyr 0 PWL(0 0 2.48u 0 2.50u 1.8 {first_reset_end_us:.3f}u 1.8 "
                f"{first_reset_fall_us:.3f}u 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 "
                "5.18u 0 5.20u 1.8 5.60u 1.8 5.62u 0 7.8u 0)"
            )
            resetn_pwl = (
                f"VRESETN_HYR rstn_hyr 0 PWL(0 1.8 2.48u 1.8 2.50u 0 {first_reset_end_us:.3f}u 0 "
                f"{first_reset_fall_us:.3f}u 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 "
                "5.18u 1.8 5.20u 0 5.60u 0 5.62u 1.8 7.8u 1.8)"
            )
        deck = replace_required(
            deck,
            "VRESET_HYR rst_hyr 0 PWL(0 0 3.58u 0 3.60u 1.8 4.00u 1.8 4.02u 0 "
            "5.18u 0 5.20u 1.8 5.60u 1.8 5.62u 0 7.8u 0)",
            reset_pwl,
        )
        deck = replace_required(
            deck,
            "VRESETN_HYR rstn_hyr 0 PWL(0 1.8 3.58u 1.8 3.60u 0 4.00u 0 4.02u 1.8 "
            "5.18u 1.8 5.20u 0 5.60u 0 5.62u 1.8 7.8u 1.8)",
            resetn_pwl,
        )
        if first_reset_active_width_us is not None:
            first_read_rise_us = max(2.72, first_reset_fall_us + 0.10)
            if first_read_rise_us >= 3.30:
                raise ValueError("first reset recovery sweep leaves too little read-settle time")
            first_read_pwl = (
                f"VREAD_HYR read_hyr 0 PWL(0 0 {first_read_rise_us - 0.02:.3f}u 0 "
                f"{first_read_rise_us:.3f}u 1.15 3.560u 1.15 3.580u 0 7.8u 0)"
            )
            first_pact_start_us = min(first_read_rise_us + 0.46, 3.30)
            first_pact_end_us = first_pact_start_us + 0.13
            first_pact_pwl = (
                f"VPACT_HYR pact_hyr 0 PWL(0 0 {first_pact_start_us - 0.02:.3f}u 0 "
                f"{first_pact_start_us:.3f}u 1.8 {first_pact_end_us:.3f}u 1.8 "
                f"{first_pact_end_us + 0.02:.3f}u 0 7.8u 0)"
            )
            first_pactn_pwl = (
                f"VPACTN_HYR pactn_hyr 0 PWL(0 1.8 {first_pact_start_us - 0.02:.3f}u 1.8 "
                f"{first_pact_start_us:.3f}u 0 {first_pact_end_us:.3f}u 0 "
                f"{first_pact_end_us + 0.02:.3f}u 1.8 7.8u 1.8)"
            )
            deck = replace_required(deck, timing_read_pwl, first_read_pwl)
            deck = replace_required(deck, timing_base_pact_pwl, first_pact_pwl + "\n" + first_pactn_pwl)
        deck = replace_required(
            deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        deck = add_shifted_gate_probes(deck, stem)
        data = run_ngspice(deck, stem)
        return load_wrdata(data, 25)

    shift_skewlaw_cases = [
        ("d10_trim15", "10 mV skew, trim -15", 0.560, 0.540, -0.015, True),
        ("d15_trim25", "15 mV skew, trim -25", 0.565, 0.535, -0.025, True),
        ("d20_trim35", "20 mV skew, trim -35", 0.570, 0.530, -0.035, True),
        ("d25_trim45", "25 mV skew, trim -45", 0.575, 0.525, -0.045, True),
        ("d30_trim55", "30 mV skew, trim -55", 0.580, 0.520, -0.055, True),
        ("d30_under35", "30 mV skew, trim -35", 0.580, 0.520, -0.035, False),
    ]
    shift_skewlaw_labels = []
    shift_skewlaw_skews_mv = []
    shift_skewlaw_trim_values = []
    shift_skewlaw_gate_samples = []
    shift_skewlaw_common_samples = []
    shift_skewlaw_z_samples = []
    shift_skewlaw_h_samples = []
    shift_skewlaw_calibrated = []
    shift_skewlaw_traces = []
    for name, label, nfp_vto, nfm_vto, reset_trim_v, calibrated in shift_skewlaw_cases:
        if name == "d20_trim35":
            kt, skewlaw_cols = strpt, shift_trimmed_reuse_cols
        else:
            kt, skewlaw_cols = run_trimmed_reuse_skew_law_case(name, nfp_vto, nfm_vto, reset_trim_v)

        def sklat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(kt - time_s))])

        gate_diff = skewlaw_cols[0] - skewlaw_cols[1]
        gate_common = 0.5 * (skewlaw_cols[0] + skewlaw_cols[1])
        z = skewlaw_cols[12] - skewlaw_cols[11]
        load = skewlaw_cols[14] - skewlaw_cols[13]
        store = skewlaw_cols[16] - skewlaw_cols[15]
        gate_samples = np.array([sklat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
        common_samples = np.array([sklat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
        z_samples = np.array([sklat(ts, z) for ts in shift_reuse_z_times])
        h_samples = np.array([sklat(ts, store) for ts in shift_reuse_h_times])
        z_reset = np.array([abs(sklat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
        h_reset = np.array([abs(sklat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
        require(
            np.max(np.abs(gate_samples - reset_trim_v)) < 0.006,
            f"{label} should establish the requested split-reset trim",
        )
        require(
            np.max(np.abs(common_samples - 0.90)) < 0.006,
            f"{label} should preserve reset common mode",
        )
        require(
            np.max(z_reset) < 0.001 and np.max(h_reset) < 0.001,
            f"{label} should still clear z/h state during reset",
        )
        require(
            np.all(z_samples > 0.035),
            f"{label} should keep useful read preactivation",
        )
        require(
            np.all(h_samples < 0.120),
            f"{label} should avoid forward-store overdrive",
        )
        if calibrated:
            require(
                np.all(h_samples > 0.025),
                f"{label} should recover positive activation with the skew-scaled trim",
            )
            require(
                np.max(h_samples) - np.min(h_samples) < 0.030,
                f"{label} should remain stable across repeated cycles",
            )
        shift_skewlaw_labels.append(label)
        shift_skewlaw_skews_mv.append(500.0 * (nfp_vto - nfm_vto))
        shift_skewlaw_trim_values.append(reset_trim_v)
        shift_skewlaw_gate_samples.append(gate_samples)
        shift_skewlaw_common_samples.append(common_samples)
        shift_skewlaw_z_samples.append(z_samples)
        shift_skewlaw_h_samples.append(h_samples)
        shift_skewlaw_calibrated.append(calibrated)
        if name in {"d10_trim15", "d20_trim35", "d30_trim55", "d30_under35"}:
            shift_skewlaw_traces.append((label, kt, load, store))

    shift_skewlaw_skews_mv = np.array(shift_skewlaw_skews_mv)
    shift_skewlaw_trim_values = np.array(shift_skewlaw_trim_values)
    shift_skewlaw_gate_samples = np.array(shift_skewlaw_gate_samples)
    shift_skewlaw_common_samples = np.array(shift_skewlaw_common_samples)
    shift_skewlaw_z_samples = np.array(shift_skewlaw_z_samples)
    shift_skewlaw_h_samples = np.array(shift_skewlaw_h_samples)
    shift_skewlaw_calibrated = np.array(shift_skewlaw_calibrated, dtype=bool)
    shift_skewlaw_h_mean = np.mean(shift_skewlaw_h_samples, axis=1)
    shift_skewlaw_index = {case[0]: idx for idx, case in enumerate(shift_skewlaw_cases)}
    require(
        np.max(shift_skewlaw_h_mean[shift_skewlaw_calibrated])
        - np.min(shift_skewlaw_h_mean[shift_skewlaw_calibrated])
        < 0.070,
        "skew-scaled split trims should keep calibrated activations in the same useful band",
    )
    require(
        shift_skewlaw_h_mean[shift_skewlaw_index["d30_trim55"]]
        > shift_skewlaw_h_mean[shift_skewlaw_index["d30_under35"]] + 0.015,
        "severe skew should visibly need the stronger calibrated trim",
    )

    shift_signlaw_cases = [
        ("pm10_neg15", "+10 mV skew, -15 mV trim", 0.560, 0.540, -0.015, True),
        ("mp10_pos15", "-10 mV skew, +15 mV trim", 0.540, 0.560, 0.015, True),
        ("pm20_neg35", "+20 mV skew, -35 mV trim", 0.570, 0.530, -0.035, True),
        ("mp20_pos35", "-20 mV skew, +35 mV trim", 0.530, 0.570, 0.035, True),
        ("pm30_neg55", "+30 mV skew, -55 mV trim", 0.580, 0.520, -0.055, True),
        ("mp30_pos55", "-30 mV skew, +55 mV trim", 0.520, 0.580, 0.055, True),
        ("pm30_under35", "+30 mV skew, -35 mV trim", 0.580, 0.520, -0.035, False),
        ("mp30_under35", "-30 mV skew, +35 mV trim", 0.520, 0.580, 0.035, False),
    ]
    shift_signlaw_labels = []
    shift_signlaw_skews_mv = []
    shift_signlaw_trim_values = []
    shift_signlaw_gate_samples = []
    shift_signlaw_common_samples = []
    shift_signlaw_z_samples = []
    shift_signlaw_h_samples = []
    shift_signlaw_calibrated = []
    shift_signlaw_traces = []
    for name, label, nfp_vto, nfm_vto, reset_trim_v, calibrated in shift_signlaw_cases:
        st, signlaw_cols = run_trimmed_reuse_skew_law_case(
            name,
            nfp_vto,
            nfm_vto,
            reset_trim_v,
            family="signlaw",
        )

        def sgnat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(st - time_s))])

        gate_diff = signlaw_cols[0] - signlaw_cols[1]
        gate_common = 0.5 * (signlaw_cols[0] + signlaw_cols[1])
        z = signlaw_cols[12] - signlaw_cols[11]
        load = signlaw_cols[14] - signlaw_cols[13]
        store = signlaw_cols[16] - signlaw_cols[15]
        gate_samples = np.array([sgnat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
        common_samples = np.array([sgnat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
        z_samples = np.array([sgnat(ts, z) for ts in shift_reuse_z_times])
        h_samples = np.array([sgnat(ts, store) for ts in shift_reuse_h_times])
        z_reset = np.array([abs(sgnat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
        h_reset = np.array([abs(sgnat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
        require(
            np.max(np.abs(gate_samples - reset_trim_v)) < 0.006,
            f"{label} should establish the requested signed split-reset trim",
        )
        require(
            np.max(np.abs(common_samples - 0.90)) < 0.006,
            f"{label} should preserve reset common mode",
        )
        require(
            np.max(z_reset) < 0.001 and np.max(h_reset) < 0.001,
            f"{label} should clear z/h state during reset",
        )
        require(
            np.all(z_samples > 0.035),
            f"{label} should keep useful read preactivation",
        )
        require(
            np.all(h_samples < 0.130),
            f"{label} should avoid forward-store overdrive",
        )
        if calibrated:
            require(
                np.all(h_samples > 0.025),
                f"{label} should recover positive activation with signed skew-scaled trim",
            )
            require(
                np.max(h_samples) - np.min(h_samples) < 0.035,
                f"{label} should remain stable across repeated cycles",
            )
        shift_signlaw_labels.append(label)
        shift_signlaw_skews_mv.append(500.0 * (nfp_vto - nfm_vto))
        shift_signlaw_trim_values.append(reset_trim_v)
        shift_signlaw_gate_samples.append(gate_samples)
        shift_signlaw_common_samples.append(common_samples)
        shift_signlaw_z_samples.append(z_samples)
        shift_signlaw_h_samples.append(h_samples)
        shift_signlaw_calibrated.append(calibrated)
        if name in {"pm30_neg55", "mp30_pos55", "pm30_under35", "mp30_under35"}:
            shift_signlaw_traces.append((label, st, load, store))

    shift_signlaw_skews_mv = np.array(shift_signlaw_skews_mv)
    shift_signlaw_trim_values = np.array(shift_signlaw_trim_values)
    shift_signlaw_gate_samples = np.array(shift_signlaw_gate_samples)
    shift_signlaw_common_samples = np.array(shift_signlaw_common_samples)
    shift_signlaw_z_samples = np.array(shift_signlaw_z_samples)
    shift_signlaw_h_samples = np.array(shift_signlaw_h_samples)
    shift_signlaw_calibrated = np.array(shift_signlaw_calibrated, dtype=bool)
    shift_signlaw_h_mean = np.mean(shift_signlaw_h_samples, axis=1)
    shift_signlaw_index = {case[0]: idx for idx, case in enumerate(shift_signlaw_cases)}
    require(
        np.max(shift_signlaw_h_mean[shift_signlaw_calibrated])
        - np.min(shift_signlaw_h_mean[shift_signlaw_calibrated])
        < 0.080,
        "signed skew-scaled split trims should keep both polarities in the same useful activation band",
    )
    require(
        shift_signlaw_h_mean[shift_signlaw_index["pm30_neg55"]]
        > shift_signlaw_h_mean[shift_signlaw_index["pm30_under35"]] + 0.010,
        "negative trim for +30 mV skew should recover the under-trimmed low-activation case",
    )
    require(
        shift_signlaw_h_mean[shift_signlaw_index["mp30_pos55"]]
        < shift_signlaw_h_mean[shift_signlaw_index["mp30_under35"]] - 0.010,
        "positive trim for -30 mV skew should reduce the under-trimmed over-activation case",
    )

    shift_polcal_cases = [
        ("pm30_neg45", "+30 mV skew, -45 mV trim", 0.580, 0.520, -0.045, "positive skew"),
        ("pm30_neg55", "+30 mV skew, -55 mV trim", 0.580, 0.520, -0.055, "positive skew"),
        ("pm30_neg65", "+30 mV skew, -65 mV trim", 0.580, 0.520, -0.065, "positive skew"),
        ("mp30_pos55", "-30 mV skew, +55 mV trim", 0.520, 0.580, 0.055, "negative skew"),
        ("mp30_pos65", "-30 mV skew, +65 mV trim", 0.520, 0.580, 0.065, "negative skew"),
        ("mp30_pos75", "-30 mV skew, +75 mV trim", 0.520, 0.580, 0.075, "negative skew"),
    ]
    shift_polcal_labels = []
    shift_polcal_trim_mv = []
    shift_polcal_h_samples = []
    shift_polcal_traces = []
    for name, label, nfp_vto, nfm_vto, reset_trim_v, _group in shift_polcal_cases:
        pt, polcal_cols = run_trimmed_reuse_skew_law_case(
            name,
            nfp_vto,
            nfm_vto,
            reset_trim_v,
            family="polcal",
        )

        def polat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(pt - time_s))])

        gate_diff = polcal_cols[0] - polcal_cols[1]
        gate_common = 0.5 * (polcal_cols[0] + polcal_cols[1])
        z = polcal_cols[12] - polcal_cols[11]
        load = polcal_cols[14] - polcal_cols[13]
        store = polcal_cols[16] - polcal_cols[15]
        gate_samples = np.array([polat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
        common_samples = np.array([polat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
        z_samples = np.array([polat(ts, z) for ts in shift_reuse_z_times])
        h_samples = np.array([polat(ts, store) for ts in shift_reuse_h_times])
        z_reset = np.array([abs(polat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
        h_reset = np.array([abs(polat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
        require(
            np.max(np.abs(gate_samples - reset_trim_v)) < 0.006,
            f"{label} should establish the requested polarity-calibration trim",
        )
        require(
            np.max(np.abs(common_samples - 0.90)) < 0.006,
            f"{label} should preserve reset common mode during polarity calibration",
        )
        require(
            np.max(z_reset) < 0.001 and np.max(h_reset) < 0.001,
            f"{label} should clear z/h state during polarity calibration reset",
        )
        require(
            np.all(z_samples > 0.035),
            f"{label} should keep useful read preactivation during polarity calibration",
        )
        require(
            np.all((h_samples > 0.015) & (h_samples < 0.090)),
            f"{label} should stay inside the non-railed calibration sweep band",
        )
        require(
            np.max(h_samples) - np.min(h_samples) < 0.025,
            f"{label} should remain repeatable across calibration cycles",
        )
        shift_polcal_labels.append(label)
        shift_polcal_trim_mv.append(1e3 * reset_trim_v)
        shift_polcal_h_samples.append(h_samples)
        shift_polcal_traces.append((label, pt, load, store))

    shift_polcal_trim_mv = np.array(shift_polcal_trim_mv)
    shift_polcal_h_samples = np.array(shift_polcal_h_samples)
    shift_polcal_h_mean = np.mean(shift_polcal_h_samples, axis=1)
    shift_polcal_index = {case[0]: idx for idx, case in enumerate(shift_polcal_cases)}
    pm_order = [shift_polcal_index[name] for name in ["pm30_neg45", "pm30_neg55", "pm30_neg65"]]
    mp_order = [shift_polcal_index[name] for name in ["mp30_pos55", "mp30_pos65", "mp30_pos75"]]
    require(
        np.all(np.diff(shift_polcal_h_mean[pm_order]) > 0.010),
        "stronger negative trim should monotonically raise the +30 mV skew activation",
    )
    require(
        np.all(np.diff(shift_polcal_h_mean[mp_order]) < -0.010),
        "stronger positive trim should monotonically lower the -30 mV skew activation",
    )
    require(
        abs(
            shift_polcal_h_mean[shift_polcal_index["pm30_neg55"]]
            - shift_polcal_h_mean[shift_polcal_index["mp30_pos65"]]
        )
        < 0.006,
        "opposite skew polarities should align when the positive-trim branch uses the offset calibration",
    )
    require(
        shift_polcal_h_mean[shift_polcal_index["mp30_pos65"]]
        < shift_polcal_h_mean[shift_polcal_index["mp30_pos55"]] - 0.010,
        "positive-trim offset calibration should improve the old +55 mV case",
    )

    shift_polcal_common_cases = [
        ("pm30_neg55_cm080", "+30 mV skew, -55 mV trim, 0.80 V common", 0.580, 0.520, -0.055, 0.80),
        ("pm30_neg55_cm090", "+30 mV skew, -55 mV trim, 0.90 V common", 0.580, 0.520, -0.055, 0.90),
        ("mp30_pos65_cm080", "-30 mV skew, +65 mV trim, 0.80 V common", 0.520, 0.580, 0.065, 0.80),
        ("mp30_pos65_cm090", "-30 mV skew, +65 mV trim, 0.90 V common", 0.520, 0.580, 0.065, 0.90),
    ]
    shift_polcal_common_labels = []
    shift_polcal_common_modes = []
    shift_polcal_common_groups = []
    shift_polcal_common_gate_samples = []
    shift_polcal_common_common_samples = []
    shift_polcal_common_z_samples = []
    shift_polcal_common_h_samples = []
    shift_polcal_common_traces = []
    for name, label, nfp_vto, nfm_vto, reset_trim_v, reset_common_v in shift_polcal_common_cases:
        pct, polcal_common_cols = run_trimmed_reuse_skew_law_case(
            name,
            nfp_vto,
            nfm_vto,
            reset_trim_v,
            family="polcal_common",
            reset_common_v=reset_common_v,
        )

        def pccat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(pct - time_s))])

        gate_diff = polcal_common_cols[0] - polcal_common_cols[1]
        gate_common = 0.5 * (polcal_common_cols[0] + polcal_common_cols[1])
        z = polcal_common_cols[12] - polcal_common_cols[11]
        load = polcal_common_cols[14] - polcal_common_cols[13]
        store = polcal_common_cols[16] - polcal_common_cols[15]
        gate_samples = np.array([pccat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
        common_samples = np.array([pccat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
        z_samples = np.array([pccat(ts, z) for ts in shift_reuse_z_times])
        h_samples = np.array([pccat(ts, store) for ts in shift_reuse_h_times])
        z_reset = np.array([abs(pccat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
        h_reset = np.array([abs(pccat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
        require(
            np.max(np.abs(gate_samples - reset_trim_v)) < 0.006,
            f"{label} should establish the requested severe-skew trim",
        )
        require(
            np.max(np.abs(common_samples - reset_common_v)) < 0.006,
            f"{label} should establish the requested severe-skew reset common mode",
        )
        require(
            np.max(z_reset) < 0.001 and np.max(h_reset) < 0.001,
            f"{label} should clear z/h state during severe-skew common-mode reset",
        )
        require(
            np.all(z_samples > 0.035),
            f"{label} should keep useful read preactivation",
        )
        require(
            np.all((h_samples > 0.025) & (h_samples < 0.085)),
            f"{label} should keep calibrated activation positive and non-overdriven",
        )
        require(
            np.max(h_samples) - np.min(h_samples) < 0.030,
            f"{label} should remain repeatable across common-mode comparison cycles",
        )
        shift_polcal_common_labels.append(label)
        shift_polcal_common_modes.append(reset_common_v)
        shift_polcal_common_groups.append("positive skew" if reset_trim_v < 0 else "negative skew")
        shift_polcal_common_gate_samples.append(gate_samples)
        shift_polcal_common_common_samples.append(common_samples)
        shift_polcal_common_z_samples.append(z_samples)
        shift_polcal_common_h_samples.append(h_samples)
        shift_polcal_common_traces.append((label, pct, load, store))

    shift_polcal_common_modes = np.array(shift_polcal_common_modes)
    shift_polcal_common_gate_samples = np.array(shift_polcal_common_gate_samples)
    shift_polcal_common_common_samples = np.array(shift_polcal_common_common_samples)
    shift_polcal_common_z_samples = np.array(shift_polcal_common_z_samples)
    shift_polcal_common_h_samples = np.array(shift_polcal_common_h_samples)
    shift_polcal_common_h_mean = np.mean(shift_polcal_common_h_samples, axis=1)
    shift_polcal_common_index = {case[0]: idx for idx, case in enumerate(shift_polcal_common_cases)}
    pm_cm080_idx = shift_polcal_common_index["pm30_neg55_cm080"]
    pm_cm090_idx = shift_polcal_common_index["pm30_neg55_cm090"]
    mp_cm080_idx = shift_polcal_common_index["mp30_pos65_cm080"]
    mp_cm090_idx = shift_polcal_common_index["mp30_pos65_cm090"]
    require(
        abs(shift_polcal_common_h_mean[pm_cm080_idx] - shift_polcal_common_h_mean[mp_cm080_idx]) < 0.018,
        "tuned 0.80 V common mode should preserve severe-skew polarity alignment",
    )
    require(
        abs(shift_polcal_common_h_mean[pm_cm090_idx] - shift_polcal_common_h_mean[mp_cm090_idx]) < 0.018,
        "legacy 0.90 V common mode should preserve severe-skew polarity alignment",
    )
    require(
        np.all(shift_polcal_common_h_mean[[pm_cm090_idx, mp_cm090_idx]] - shift_polcal_common_h_mean[[pm_cm080_idx, mp_cm080_idx]] > 0.002),
        "legacy 0.90 V reset common should store slightly higher than tuned 0.80 V for severe calibrated trims",
    )

    shift_polsens_cases = [
        ("pm30_neg45", "+30 mV skew, -45 mV trim", 0.580, 0.520, -0.045, "positive skew", -10.0),
        ("pm30_neg50", "+30 mV skew, -50 mV trim", 0.580, 0.520, -0.050, "positive skew", -5.0),
        ("pm30_neg55", "+30 mV skew, -55 mV trim", 0.580, 0.520, -0.055, "positive skew", 0.0),
        ("pm30_neg60", "+30 mV skew, -60 mV trim", 0.580, 0.520, -0.060, "positive skew", 5.0),
        ("pm30_neg65", "+30 mV skew, -65 mV trim", 0.580, 0.520, -0.065, "positive skew", 10.0),
        ("mp30_pos55", "-30 mV skew, +55 mV trim", 0.520, 0.580, 0.055, "negative skew", -10.0),
        ("mp30_pos60", "-30 mV skew, +60 mV trim", 0.520, 0.580, 0.060, "negative skew", -5.0),
        ("mp30_pos65", "-30 mV skew, +65 mV trim", 0.520, 0.580, 0.065, "negative skew", 0.0),
        ("mp30_pos70", "-30 mV skew, +70 mV trim", 0.520, 0.580, 0.070, "negative skew", 5.0),
        ("mp30_pos75", "-30 mV skew, +75 mV trim", 0.520, 0.580, 0.075, "negative skew", 10.0),
    ]
    shift_polsens_trim_error_mv = []
    shift_polsens_h_samples = []
    shift_polsens_traces = []
    for name, label, nfp_vto, nfm_vto, reset_trim_v, group, trim_error_mv in shift_polsens_cases:
        pst, polsens_cols = run_trimmed_reuse_skew_law_case(
            name,
            nfp_vto,
            nfm_vto,
            reset_trim_v,
            family="polsens",
        )

        def psat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(pst - time_s))])

        gate_diff = polsens_cols[0] - polsens_cols[1]
        gate_common = 0.5 * (polsens_cols[0] + polsens_cols[1])
        z = polsens_cols[12] - polsens_cols[11]
        load = polsens_cols[14] - polsens_cols[13]
        store = polsens_cols[16] - polsens_cols[15]
        gate_samples = np.array([psat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
        common_samples = np.array([psat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
        z_samples = np.array([psat(ts, z) for ts in shift_reuse_z_times])
        h_samples = np.array([psat(ts, store) for ts in shift_reuse_h_times])
        z_reset = np.array([abs(psat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
        h_reset = np.array([abs(psat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
        require(
            np.max(np.abs(gate_samples - reset_trim_v)) < 0.006,
            f"{label} should establish the requested fine trim code",
        )
        require(
            np.max(np.abs(common_samples - 0.90)) < 0.006,
            f"{label} should preserve reset common mode during fine trim sensitivity",
        )
        require(
            np.max(z_reset) < 0.001 and np.max(h_reset) < 0.001,
            f"{label} should clear z/h state during fine trim sensitivity reset",
        )
        require(
            np.all(z_samples > 0.035),
            f"{label} should keep useful read preactivation during fine trim sensitivity",
        )
        require(
            np.all((h_samples > 0.020) & (h_samples < 0.075)),
            f"{label} should stay inside the local trim-sensitivity range",
        )
        require(
            np.max(h_samples) - np.min(h_samples) < 0.025,
            f"{label} should remain repeatable across fine-trim cycles",
        )
        shift_polsens_trim_error_mv.append(trim_error_mv)
        shift_polsens_h_samples.append(h_samples)
        if trim_error_mv in {-5.0, 0.0, 5.0}:
            shift_polsens_traces.append((group, trim_error_mv, pst, load, store))

    shift_polsens_trim_error_mv = np.array(shift_polsens_trim_error_mv)
    shift_polsens_h_samples = np.array(shift_polsens_h_samples)
    shift_polsens_h_mean = np.mean(shift_polsens_h_samples, axis=1)
    shift_polsens_index = {case[0]: idx for idx, case in enumerate(shift_polsens_cases)}
    pm_sens_order = [
        shift_polsens_index[name]
        for name in ["pm30_neg45", "pm30_neg50", "pm30_neg55", "pm30_neg60", "pm30_neg65"]
    ]
    mp_sens_order = [
        shift_polsens_index[name]
        for name in ["mp30_pos55", "mp30_pos60", "mp30_pos65", "mp30_pos70", "mp30_pos75"]
    ]
    pm_slope = float(
        np.polyfit(
            shift_polsens_trim_error_mv[pm_sens_order],
            1e3 * shift_polsens_h_mean[pm_sens_order],
            1,
        )[0]
    )
    mp_slope = float(
        np.polyfit(
            shift_polsens_trim_error_mv[mp_sens_order],
            1e3 * shift_polsens_h_mean[mp_sens_order],
            1,
        )[0]
    )
    require(
        np.all(np.diff(shift_polsens_h_mean[pm_sens_order]) > 0.006),
        "fine negative-trim increase should monotonically raise +30 mV skew activation",
    )
    require(
        np.all(np.diff(shift_polsens_h_mean[mp_sens_order]) < -0.006),
        "fine positive-trim increase should monotonically lower -30 mV skew activation",
    )
    require(
        1.4 < pm_slope < 2.4,
        "positive-skew fine trim sensitivity should be a visible but bounded gain",
    )
    require(
        -2.4 < mp_slope < -1.4,
        "negative-skew fine trim sensitivity should be a visible but bounded gain",
    )
    require(
        abs(
            shift_polsens_h_mean[shift_polsens_index["pm30_neg55"]]
            - shift_polsens_h_mean[shift_polsens_index["mp30_pos65"]]
        )
        < 0.006,
        "fine trim sensitivity baseline should preserve the calibrated polarity alignment",
    )
    require(
        np.max(
            np.abs(
                1e3
                * (
                    shift_polsens_h_mean[[shift_polsens_index["pm30_neg45"], shift_polsens_index["pm30_neg65"]]]
                    - shift_polsens_h_mean[shift_polsens_index["pm30_neg55"]]
                )
            )
        )
        > 15.0,
        "+/-10 mV trim-code errors should stay visibly resolvable on the +30 mV skew branch",
    )
    require(
        np.max(
            np.abs(
                1e3
                * (
                    shift_polsens_h_mean[[shift_polsens_index["mp30_pos55"], shift_polsens_index["mp30_pos75"]]]
                    - shift_polsens_h_mean[shift_polsens_index["mp30_pos65"]]
                )
            )
        )
        > 15.0,
        "+/-10 mV trim-code errors should stay visibly resolvable on the -30 mV skew branch",
    )

    shift_polref_cases = [
        ("clean_pos65", "clean +65 mV", (0.065, 0.065, 0.065), (0.90, 0.90, 0.90)),
        ("common_jitter_pos65", "common jitter", (0.065, 0.065, 0.065), (0.88, 0.92, 0.89)),
        ("diff_jitter5_pos65", "diff jitter +/-5 mV", (0.060, 0.065, 0.070), (0.90, 0.90, 0.90)),
        ("diff_jitter10_pos65", "diff jitter +/-10 mV", (0.055, 0.065, 0.075), (0.90, 0.90, 0.90)),
        ("mixed_jitter_pos65", "mixed jitter", (0.060, 0.070, 0.065), (0.885, 0.915, 0.900)),
    ]
    shift_polref_labels = []
    shift_polref_gate_samples = []
    shift_polref_common_samples = []
    shift_polref_h_samples = []
    shift_polref_traces = []
    for name, label, reset_trims_v, reset_common_v in shift_polref_cases:
        prt, polref_cols = run_trimmed_reuse_reference_perturb_case(
            name,
            reset_trims_v,
            reset_common_v,
            nfp_vto=0.520,
            nfm_vto=0.580,
            nominal_reset_trim_v=0.065,
            family="polrefpert",
        )

        def prlat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(prt - time_s))])

        gate_diff = polref_cols[0] - polref_cols[1]
        gate_common = 0.5 * (polref_cols[0] + polref_cols[1])
        z = polref_cols[12] - polref_cols[11]
        load = polref_cols[14] - polref_cols[13]
        store = polref_cols[16] - polref_cols[15]
        gate_samples = np.array([prlat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
        common_samples = np.array([prlat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
        z_samples = np.array([prlat(ts, z) for ts in shift_reuse_z_times])
        h_samples = np.array([prlat(ts, store) for ts in shift_reuse_h_times])
        z_reset = np.array([abs(prlat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
        h_reset = np.array([abs(prlat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
        expected_trims = np.array(reset_trims_v)
        expected_common = np.array(reset_common_v)
        if np.max(expected_trims) - np.min(expected_trims) < 1e-12:
            require(
                np.max(np.abs(gate_samples - expected_trims)) < 0.006,
                f"{label} should preserve the calibrated positive reset differential",
            )
        else:
            require(
                np.all((gate_samples > 0.050) & (gate_samples < 0.080)),
                f"{label} should keep sampled positive trim inside the calibrated trim window",
            )
        if np.max(expected_common) - np.min(expected_common) < 1e-12:
            require(
                np.max(np.abs(common_samples - expected_common)) < 0.006,
                f"{label} should preserve positive-branch reset common mode",
            )
        else:
            require(
                np.all((common_samples > 0.86) & (common_samples < 0.93)),
                f"{label} should keep positive-branch reset common mode bounded",
            )
        require(
            np.max(z_reset) < 0.001 and np.max(h_reset) < 0.001,
            f"{label} should still clear z/h state for the positive calibrated branch",
        )
        require(
            np.all(z_samples > 0.035),
            f"{label} should keep useful read preactivation for the positive calibrated branch",
        )
        require(
            np.all((h_samples > 0.020) & (h_samples < 0.085)),
            f"{label} should keep the positive calibrated branch non-railed under reset perturbation",
        )
        if name == "common_jitter_pos65":
            require(
                np.max(h_samples) - np.min(h_samples) < 0.006,
                "positive calibrated branch should be weakly sensitive to common-mode reference jitter",
            )
        if name in {"diff_jitter5_pos65", "diff_jitter10_pos65"}:
            require(
                np.all(np.diff(h_samples) < -0.004),
                f"{label} should order stored activation by stronger positive trim each cycle",
            )
        shift_polref_labels.append(label)
        shift_polref_gate_samples.append(gate_samples)
        shift_polref_common_samples.append(common_samples)
        shift_polref_h_samples.append(h_samples)
        if name in {"common_jitter_pos65", "diff_jitter10_pos65", "mixed_jitter_pos65"}:
            shift_polref_traces.append((label, prt, load, store))

    shift_polref_gate_samples = np.array(shift_polref_gate_samples)
    shift_polref_common_samples = np.array(shift_polref_common_samples)
    shift_polref_h_samples = np.array(shift_polref_h_samples)
    shift_polref_index = {case[0]: idx for idx, case in enumerate(shift_polref_cases)}
    require(
        np.max(shift_polref_h_samples[shift_polref_index["common_jitter_pos65"]])
        - np.min(shift_polref_h_samples[shift_polref_index["common_jitter_pos65"]])
        < 0.25
        * (
            np.max(shift_polref_h_samples[shift_polref_index["diff_jitter10_pos65"]])
            - np.min(shift_polref_h_samples[shift_polref_index["diff_jitter10_pos65"]])
        ),
        "positive calibrated common-mode jitter should remain secondary to differential trim jitter",
    )

    shift_reset_corner_specs = [
        ("strong", "strong TG", 0.500, -0.500),
        ("nominal", "nominal TG", 0.550, -0.550),
        ("weak", "weak TG", 0.600, -0.600),
        ("nweak_pstrong", "N weak / P strong", 0.600, -0.500),
        ("nstrong_pweak", "N strong / P weak", 0.500, -0.600),
    ]
    shift_reset_branch_specs = [
        ("neg_trim", "+30 mV skew, -55 mV trim", 0.580, 0.520, -0.055),
        ("pos_trim", "-30 mV skew, +65 mV trim", 0.520, 0.580, 0.065),
    ]
    shift_reset_corner_trim_error = []
    shift_reset_corner_common_error = []
    shift_reset_corner_h_mean = []
    shift_reset_corner_h_spread = []
    shift_reset_corner_traces = []
    for branch_name, branch_label, nfp_vto, nfm_vto, reset_trim_v in shift_reset_branch_specs:
        branch_trim_error = []
        branch_common_error = []
        branch_h_mean = []
        branch_h_spread = []
        for corner_name, corner_label, reset_n_vto, reset_p_vto in shift_reset_corner_specs:
            rt, rs_cols = run_trimmed_reuse_skew_law_case(
                f"{branch_name}_{corner_name}",
                nfp_vto,
                nfm_vto,
                reset_trim_v,
                family="resetsw",
                reset_n_vto=reset_n_vto,
                reset_p_vto=reset_p_vto,
            )

            def rsat(time_s: float, values: np.ndarray) -> float:
                return float(values[np.argmin(np.abs(rt - time_s))])

            gate_diff = rs_cols[0] - rs_cols[1]
            gate_common = 0.5 * (rs_cols[0] + rs_cols[1])
            z = rs_cols[12] - rs_cols[11]
            load = rs_cols[14] - rs_cols[13]
            store = rs_cols[16] - rs_cols[15]
            gate_samples = np.array([rsat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
            common_samples = np.array([rsat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
            z_samples = np.array([rsat(ts, z) for ts in shift_reuse_z_times])
            h_samples = np.array([rsat(ts, store) for ts in shift_reuse_h_times])
            z_reset = np.array([abs(rsat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
            h_reset = np.array([abs(rsat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
            trim_error = float(np.max(np.abs(gate_samples - reset_trim_v)))
            common_error = float(np.max(np.abs(common_samples - 0.90)))
            h_mean = float(np.mean(h_samples))
            h_spread = float(np.max(h_samples) - np.min(h_samples))
            require(
                trim_error < 0.008,
                f"{branch_label} {corner_label} reset switch corner should preserve the calibrated split trim",
            )
            require(
                common_error < 0.010,
                f"{branch_label} {corner_label} reset switch corner should preserve shifted-gate common mode",
            )
            require(
                np.max(z_reset) < 0.002 and np.max(h_reset) < 0.002,
                f"{branch_label} {corner_label} reset switch corner should clear z/h state",
            )
            require(
                np.all(z_samples > 0.035),
                f"{branch_label} {corner_label} reset switch corner should keep useful read preactivation",
            )
            require(
                np.all((h_samples > 0.020) & (h_samples < 0.085)),
                f"{branch_label} {corner_label} reset switch corner should keep stored activation non-railed",
            )
            require(
                h_spread < 0.025,
                f"{branch_label} {corner_label} reset switch corner should remain repeatable across cycles",
            )
            branch_trim_error.append(trim_error)
            branch_common_error.append(common_error)
            branch_h_mean.append(h_mean)
            branch_h_spread.append(h_spread)
            if corner_name in {"nominal", "weak", "nweak_pstrong"}:
                shift_reset_corner_traces.append((branch_label, corner_label, rt, load, store))
        shift_reset_corner_trim_error.append(branch_trim_error)
        shift_reset_corner_common_error.append(branch_common_error)
        shift_reset_corner_h_mean.append(branch_h_mean)
        shift_reset_corner_h_spread.append(branch_h_spread)

    shift_reset_corner_trim_error = np.array(shift_reset_corner_trim_error)
    shift_reset_corner_common_error = np.array(shift_reset_corner_common_error)
    shift_reset_corner_h_mean = np.array(shift_reset_corner_h_mean)
    shift_reset_corner_h_spread = np.array(shift_reset_corner_h_spread)
    require(
        np.all(np.max(shift_reset_corner_h_mean, axis=1) - np.min(shift_reset_corner_h_mean, axis=1) < 0.012),
        "reset switch threshold corners should not materially move the calibrated activation",
    )
    require(
        np.max(shift_reset_corner_common_error) < 0.5 * np.max(np.abs([case[4] for case in shift_reset_branch_specs])),
        "reset switch threshold corner common-mode error should stay below the intentional split trim scale",
    )

    shift_refz_cases = [
        ("r0", "0", 0.0),
        ("r1k", "1k", 1e3),
        ("r10k", "10k", 1e4),
        ("r100k", "100k", 1e5),
        ("r1m", "1M", 1e6),
        ("r3m", "3M", 3e6),
    ]
    shift_refz_trim_error = []
    shift_refz_common_error = []
    shift_refz_h_mean = []
    shift_refz_h_spread = []
    shift_refz_traces = []
    for branch_name, branch_label, nfp_vto, nfm_vto, reset_trim_v in shift_reset_branch_specs:
        branch_trim_error = []
        branch_common_error = []
        branch_h_mean = []
        branch_h_spread = []
        for case_name, case_label, series_ohm in shift_refz_cases:
            rt, rz_cols = run_trimmed_reuse_skew_law_case(
                f"{branch_name}_{case_name}",
                nfp_vto,
                nfm_vto,
                reset_trim_v,
                family="refz",
                reset_ref_series_ohm=series_ohm,
            )

            def rzat(time_s: float, values: np.ndarray) -> float:
                return float(values[np.argmin(np.abs(rt - time_s))])

            gate_diff = rz_cols[0] - rz_cols[1]
            gate_common = 0.5 * (rz_cols[0] + rz_cols[1])
            z = rz_cols[12] - rz_cols[11]
            load = rz_cols[14] - rz_cols[13]
            store = rz_cols[16] - rz_cols[15]
            gate_samples = np.array([rzat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
            common_samples = np.array([rzat(ts, gate_common) for ts in shift_trimmed_reuse_reset_times])
            z_samples = np.array([rzat(ts, z) for ts in shift_reuse_z_times])
            h_samples = np.array([rzat(ts, store) for ts in shift_reuse_h_times])
            z_reset = np.array([abs(rzat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
            h_reset = np.array([abs(rzat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
            trim_error = float(np.max(np.abs(gate_samples - reset_trim_v)))
            common_error = float(np.max(np.abs(common_samples - 0.90)))
            h_mean = float(np.mean(h_samples))
            h_spread = float(np.max(h_samples) - np.min(h_samples))
            require(
                np.max(z_reset) < 0.010 and np.max(h_reset) < 0.010,
                f"{branch_label} {case_label} reset-reference source should still clear z/h during reset",
            )
            require(
                np.all(z_samples > 0.035),
                f"{branch_label} {case_label} reset-reference source should keep useful read preactivation",
            )
            branch_trim_error.append(trim_error)
            branch_common_error.append(common_error)
            branch_h_mean.append(h_mean)
            branch_h_spread.append(h_spread)
            if case_name in {"r0", "r100k", "r1m", "r3m"}:
                shift_refz_traces.append((branch_label, case_label, rt, load, store))
        shift_refz_trim_error.append(branch_trim_error)
        shift_refz_common_error.append(branch_common_error)
        shift_refz_h_mean.append(branch_h_mean)
        shift_refz_h_spread.append(branch_h_spread)

    shift_refz_trim_error = np.array(shift_refz_trim_error)
    shift_refz_common_error = np.array(shift_refz_common_error)
    shift_refz_h_mean = np.array(shift_refz_h_mean)
    shift_refz_h_spread = np.array(shift_refz_h_spread)
    require(
        np.all(shift_refz_trim_error[:, :2] < 0.001),
        "reset-reference source impedance up through 1k should preserve split trim",
    )
    require(
        np.all((shift_refz_h_mean[:, :2] > 0.040) & (shift_refz_h_mean[:, :2] < 0.055)),
        "reset-reference source impedance up through 1k should preserve calibrated activation",
    )
    require(
        np.all(np.abs(shift_refz_h_mean[:, 2] - shift_refz_h_mean[:, 0]) > 0.015),
        "10k reset-reference sources should already visibly move the stored activation with this reset pulse",
    )
    require(
        np.all(np.diff(shift_refz_trim_error, axis=1) >= -0.002),
        "reset-reference trim error should not improve as source impedance increases",
    )
    require(
        np.all(shift_refz_trim_error[:, -1] > 0.020),
        "megaohm reset-reference sources should visibly fail to deliver the calibrated split trim",
    )
    require(
        np.all(np.abs(shift_refz_h_mean[:, -1] - shift_refz_h_mean[:, 0]) > 0.020),
        "megaohm reset-reference sources should visibly move the stored activation",
    )

    shift_refz_width_cases = [
        ("w120ns", "120 ns", 0.120),
        ("w180ns", "180 ns", 0.180),
        ("w240ns", "240 ns", 0.240),
        ("w320ns", "320 ns", 0.320),
        ("w400ns", "400 ns", 0.400),
    ]
    shift_refz_width_source_cases = [
        ("r10k", "10k", 1e4),
        ("r100k", "100k", 1e5),
    ]
    shift_refz_width_sample_time = 3.555e-6
    shift_refz_width_trim_error = []
    shift_refz_width_h_sample = []
    shift_refz_width_z_sample = []
    shift_refz_width_reset_z = []
    shift_refz_width_traces = []
    for branch_name, branch_label, nfp_vto, nfm_vto, reset_trim_v in shift_reset_branch_specs:
        branch_trim_error = []
        branch_h_sample = []
        branch_z_sample = []
        branch_reset_z = []
        for source_name, source_label, series_ohm in shift_refz_width_source_cases:
            source_trim_error = []
            source_h_sample = []
            source_z_sample = []
            source_reset_z = []
            for width_name, width_label, width_us in shift_refz_width_cases:
                rt, rw_cols = run_trimmed_reuse_skew_law_case(
                    f"{branch_name}_{source_name}_{width_name}",
                    nfp_vto,
                    nfm_vto,
                    reset_trim_v,
                    family="refzwidth",
                    reset_ref_series_ohm=series_ohm,
                    first_reset_active_width_us=width_us,
                )

                def rwfat(time_s: float, values: np.ndarray) -> float:
                    return float(values[np.argmin(np.abs(rt - time_s))])

                gate_diff = rw_cols[0] - rw_cols[1]
                gate_common = 0.5 * (rw_cols[0] + rw_cols[1])
                z = rw_cols[12] - rw_cols[11]
                store = rw_cols[16] - rw_cols[15]
                gate_sample_time = (2.50 + width_us + 0.08) * 1e-6
                trim_error = abs(rwfat(gate_sample_time, gate_diff) - reset_trim_v)
                z_reset = abs(rwfat(gate_sample_time, z))
                h_reset = abs(rwfat(gate_sample_time, store))
                h_sample = rwfat(shift_refz_width_sample_time, store)
                z_sample = rwfat(shift_refz_width_sample_time, z)
                require(
                    z_reset < 0.010 and h_reset < 0.010,
                    f"{branch_label} {source_label} {width_label} first-reset recovery should clear z/h",
                )
                require(
                    z_sample > 0.030,
                    f"{branch_label} {source_label} {width_label} first-reset recovery should keep useful read preactivation",
                )
                source_trim_error.append(trim_error)
                source_h_sample.append(h_sample)
                source_z_sample.append(z_sample)
                source_reset_z.append(z_reset)
                if width_name in {"w120ns", "w400ns"}:
                    shift_refz_width_traces.append(
                        (branch_label, source_label, width_label, rt, store)
                    )
            branch_trim_error.append(source_trim_error)
            branch_h_sample.append(source_h_sample)
            branch_z_sample.append(source_z_sample)
            branch_reset_z.append(source_reset_z)
        shift_refz_width_trim_error.append(branch_trim_error)
        shift_refz_width_h_sample.append(branch_h_sample)
        shift_refz_width_z_sample.append(branch_z_sample)
        shift_refz_width_reset_z.append(branch_reset_z)

    shift_refz_width_trim_error = np.array(shift_refz_width_trim_error)
    shift_refz_width_h_sample = np.array(shift_refz_width_h_sample)
    shift_refz_width_z_sample = np.array(shift_refz_width_z_sample)
    shift_refz_width_reset_z = np.array(shift_refz_width_reset_z)
    r10k_idx = 0
    r100k_idx = 1
    require(
        np.all(np.diff(shift_refz_width_trim_error[:, r10k_idx, :], axis=1) < 0.0),
        "10k reset-reference trim error should improve monotonically with first-reset width",
    )
    require(
        np.all(shift_refz_width_trim_error[:, r10k_idx, -1] < 0.006),
        "400 ns first reset should recover 10k trim delivery",
    )
    require(
        np.all(
            np.abs(shift_refz_width_h_sample[:, r10k_idx, -1] - shift_refz_width_h_sample[:, r10k_idx, 0])
            > 0.020
        ),
        "400 ns first reset should visibly improve the 10k first-cycle activation error",
    )
    require(
        np.all(shift_refz_width_trim_error[:, r100k_idx, -1] > 0.020),
        "400 ns first reset should still not be enough for 100k trim-source impedance",
    )
    require(
        np.all(np.abs(shift_refz_width_h_sample[:, r100k_idx, -1] - shift_refz_width_h_sample[:, r10k_idx, -1]) > 0.020),
        "100k trim-source impedance should still visibly miss the recovered 10k activation",
    )

    shift_refz_decap_cases = [
        ("c0p", "0 pF", 0.0),
        ("c25p", "25 pF", 25.0),
        ("c50p", "50 pF", 50.0),
        ("c100p", "100 pF", 100.0),
        ("c250p", "250 pF", 250.0),
    ]
    shift_refz_decap_sample_time = 3.555e-6
    shift_refz_decap_trim_error = []
    shift_refz_decap_h_sample = []
    shift_refz_decap_z_sample = []
    shift_refz_decap_traces = []
    for branch_name, branch_label, nfp_vto, nfm_vto, reset_trim_v in shift_reset_branch_specs:
        branch_trim_error = []
        branch_h_sample = []
        branch_z_sample = []
        for cap_name, cap_label, cap_pf in shift_refz_decap_cases:
            dt, dc_cols = run_trimmed_reuse_skew_law_case(
                f"{branch_name}_r100k_{cap_name}",
                nfp_vto,
                nfm_vto,
                reset_trim_v,
                family="refzdecap",
                reset_ref_series_ohm=1e5,
                reset_ref_shunt_pf=cap_pf,
                first_reset_active_width_us=0.120,
            )

            def dcat(time_s: float, values: np.ndarray) -> float:
                return float(values[np.argmin(np.abs(dt - time_s))])

            gate_diff = dc_cols[0] - dc_cols[1]
            z = dc_cols[12] - dc_cols[11]
            store = dc_cols[16] - dc_cols[15]
            gate_sample_time = (2.50 + 0.120 + 0.08) * 1e-6
            trim_error = abs(dcat(gate_sample_time, gate_diff) - reset_trim_v)
            z_reset = abs(dcat(gate_sample_time, z))
            h_reset = abs(dcat(gate_sample_time, store))
            h_sample = dcat(shift_refz_decap_sample_time, store)
            z_sample = dcat(shift_refz_decap_sample_time, z)
            require(
                z_reset < 0.010 and h_reset < 0.010,
                f"{branch_label} {cap_label} local trim decoupling should still clear z/h",
            )
            require(
                z_sample > 0.030,
                f"{branch_label} {cap_label} local trim decoupling should keep useful read preactivation",
            )
            branch_trim_error.append(trim_error)
            branch_h_sample.append(h_sample)
            branch_z_sample.append(z_sample)
            if cap_name in {"c0p", "c100p", "c250p"}:
                shift_refz_decap_traces.append((branch_label, cap_label, dt, store))
        shift_refz_decap_trim_error.append(branch_trim_error)
        shift_refz_decap_h_sample.append(branch_h_sample)
        shift_refz_decap_z_sample.append(branch_z_sample)

    shift_refz_decap_trim_error = np.array(shift_refz_decap_trim_error)
    shift_refz_decap_h_sample = np.array(shift_refz_decap_h_sample)
    shift_refz_decap_z_sample = np.array(shift_refz_decap_z_sample)
    require(
        np.all(np.diff(shift_refz_decap_trim_error, axis=1) < 0.0),
        "local reset-reference decoupling should monotonically reduce 100k trim error",
    )
    require(
        np.all(shift_refz_decap_trim_error[:, -1] < 0.006),
        "250 pF local reset-reference decoupling should recover 100k trim delivery",
    )
    require(
        np.all(np.abs(shift_refz_decap_h_sample[:, -1] - shift_refz_decap_h_sample[:, 0]) > 0.060),
        "local reset-reference decoupling should visibly recover 100k first-cycle activation",
    )
    require(
        np.all((shift_refz_decap_h_sample[:, -1] > 0.020) & (shift_refz_decap_h_sample[:, -1] < 0.090)),
        "250 pF local reset-reference decoupling should put first-cycle activation in the useful band",
    )

    shift_refz_recharge_cases = [
        ("c0p", "0 pF", 0.0),
        ("c50p", "50 pF", 50.0),
        ("c100p", "100 pF", 100.0),
        ("c250p", "250 pF", 250.0),
    ]
    shift_refz_recharge_trim_error = []
    shift_refz_recharge_h_samples = []
    shift_refz_recharge_z_samples = []
    shift_refz_recharge_traces = []
    for branch_name, branch_label, nfp_vto, nfm_vto, reset_trim_v in shift_reset_branch_specs:
        branch_trim_error = []
        branch_h_samples = []
        branch_z_samples = []
        for cap_name, cap_label, cap_pf in shift_refz_recharge_cases:
            rt, rc_cols = run_trimmed_reuse_skew_law_case(
                f"{branch_name}_r100k_{cap_name}",
                nfp_vto,
                nfm_vto,
                reset_trim_v,
                family="refzrecharge",
                reset_ref_series_ohm=1e5,
                reset_ref_shunt_pf=cap_pf,
            )

            def rcat(time_s: float, values: np.ndarray) -> float:
                return float(values[np.argmin(np.abs(rt - time_s))])

            gate_diff = rc_cols[0] - rc_cols[1]
            z = rc_cols[12] - rc_cols[11]
            store = rc_cols[16] - rc_cols[15]
            gate_samples = np.array([rcat(ts, gate_diff) for ts in shift_trimmed_reuse_reset_times])
            z_samples = np.array([rcat(ts, z) for ts in shift_reuse_z_times])
            h_samples = np.array([rcat(ts, store) for ts in shift_reuse_h_times])
            z_reset = np.array([abs(rcat(ts, z)) for ts in shift_trimmed_reuse_reset_times])
            h_reset = np.array([abs(rcat(ts, store)) for ts in shift_trimmed_reuse_reset_times])
            trim_error = np.abs(gate_samples - reset_trim_v)
            require(
                np.max(z_reset) < 0.010 and np.max(h_reset) < 0.010,
                f"{branch_label} {cap_label} recharge case should still clear z/h between reads",
            )
            require(
                np.all(z_samples > 0.030),
                f"{branch_label} {cap_label} recharge case should keep useful read preactivation",
            )
            branch_trim_error.append(trim_error)
            branch_h_samples.append(h_samples)
            branch_z_samples.append(z_samples)
            if cap_name in {"c0p", "c100p", "c250p"}:
                shift_refz_recharge_traces.append((branch_label, cap_label, rt, store))
        shift_refz_recharge_trim_error.append(branch_trim_error)
        shift_refz_recharge_h_samples.append(branch_h_samples)
        shift_refz_recharge_z_samples.append(branch_z_samples)

    shift_refz_recharge_trim_error = np.array(shift_refz_recharge_trim_error)
    shift_refz_recharge_h_samples = np.array(shift_refz_recharge_h_samples)
    shift_refz_recharge_z_samples = np.array(shift_refz_recharge_z_samples)
    shift_refz_recharge_max_trim_error = np.max(shift_refz_recharge_trim_error, axis=2)
    require(
        np.all(np.diff(shift_refz_recharge_max_trim_error, axis=1) < 0.0),
        "local decoupling recharge max trim error should improve monotonically with capacitance",
    )
    require(
        np.all(shift_refz_recharge_max_trim_error[:, -1] < 0.006),
        "250 pF local decoupling should preserve 100k trim delivery across repeated resets",
    )
    require(
        np.all((shift_refz_recharge_h_samples[:, -1, :] > 0.020) & (shift_refz_recharge_h_samples[:, -1, :] < 0.090)),
        "250 pF local decoupling should keep all repeated activations in the useful band",
    )
    require(
        np.all(np.max(shift_refz_recharge_h_samples[:, -1, :], axis=1) - np.min(shift_refz_recharge_h_samples[:, -1, :], axis=1) < 0.025),
        "250 pF local decoupling repeated activations should stay within a 25 mV cycle window",
    )
    require(
        np.all(np.abs(shift_refz_recharge_h_samples[:, -1, 0] - shift_refz_recharge_h_samples[:, 0, 0]) > 0.060),
        "250 pF local decoupling should visibly improve the first repeated-schedule activation over no decap",
    )

    shift_refz_startup_cases = [
        ("cold", "cold", 0.0),
        ("tau0p5", "0.5 tau", 0.5),
        ("tau1", "1 tau", 1.0),
        ("tau2", "2 tau", 2.0),
        ("tau3", "3 tau", 3.0),
        ("ideal", "initialized", None),
    ]

    def run_reset_ref_startup_case(
        branch_name: str,
        reset_trim_v: float,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_shifted_gate_reset_ref_startup_{branch_name}"
        )
        reset_ref_p = 0.90 + 0.5 * reset_trim_v
        reset_ref_m = 0.90 - 0.5 * reset_trim_v
        deck_lines = [
            "* Cold-start check for the local split-reset trim-reference reservoir.",
            "* Each copy has a 100k source, 250 pF local reservoir, 5 pF shifted-gate",
            "* load, and real complementary reset transmission gates.",
            COMMON_MODELS,
            ".param CREF=250p CGATE=5p WRESETN=60u WRESETP=180u",
            "VDD vdd 0 1.8",
            "VRST rst 0 PWL(0 0 2.48u 0 2.50u 1.8 2.62u 1.8 2.64u 0 3.0u 0)",
            "VRSTN rstn 0 PWL(0 1.8 2.48u 1.8 2.50u 0 2.62u 0 2.64u 1.8 3.0u 1.8)",
        ]
        prints = []
        for idx, (case_name, _case_label, tau_count) in enumerate(shift_refz_startup_cases):
            if tau_count is None:
                charged_fraction = 1.0
            else:
                charged_fraction = 1.0 - float(np.exp(-tau_count))
            ref_p_ic = reset_ref_p * charged_fraction
            ref_m_ic = reset_ref_m * charged_fraction
            deck_lines.extend(
                [
                    f"* {case_name}: reservoir precharged to {charged_fraction:.6f} of target.",
                    f"VZRP_{idx} zrp_src_{idx} 0 {reset_ref_p:.5f}",
                    f"VZRM_{idx} zrm_src_{idx} 0 {reset_ref_m:.5f}",
                    f"RZRP_{idx} zrp_src_{idx} zrp_{idx} 100k",
                    f"RZRM_{idx} zrm_src_{idx} zrm_{idx} 100k",
                    f"CZRP_{idx} zrp_{idx} 0 {{CREF}} IC={ref_p_ic:.5f}",
                    f"CZRM_{idx} zrm_{idx} 0 {{CREF}} IC={ref_m_ic:.5f}",
                    f"CZPG_{idx} zpg_{idx} 0 {{CGATE}} IC=0.90",
                    f"CZMG_{idx} zmg_{idx} 0 {{CGATE}} IC=0.90",
                    f"MRZGPN_{idx} zpg_{idx} rst zrp_{idx} 0 NMOS L={{LCH}} W={{WRESETN}}",
                    f"MRZGMN_{idx} zmg_{idx} rst zrm_{idx} 0 NMOS L={{LCH}} W={{WRESETN}}",
                    f"MRZGPP_{idx} zpg_{idx} rstn zrp_{idx} vdd PMOS L={{LCH}} W={{WRESETP}}",
                    f"MRZGMP_{idx} zmg_{idx} rstn zrm_{idx} vdd PMOS L={{LCH}} W={{WRESETP}}",
                ]
            )
            prints.extend([f"v(zrp_{idx})", f"v(zrm_{idx})", f"v(zpg_{idx})", f"v(zmg_{idx})"])
        deck_lines.extend(
            [
                ".control",
                "set noaskquit",
                "tran 5n 3.0u uic",
                f"wrdata {stem}.dat " + " ".join(prints),
                "quit",
                ".endc",
                ".end",
            ]
        )
        data = run_ngspice("\n".join(deck_lines), stem)
        return load_wrdata(data, len(prints))

    shift_refz_startup_trim_error = []
    shift_refz_startup_gate_common = []
    shift_refz_startup_gate_diff = []
    shift_refz_startup_traces = []
    for branch_name, branch_label, _nfp_vto, _nfm_vto, reset_trim_v in shift_reset_branch_specs:
        stt, startup_cols = run_reset_ref_startup_case(branch_name, reset_trim_v)

        def stat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(stt - time_s))])

        branch_trim_error = []
        branch_gate_common = []
        branch_gate_diff = []
        for case_idx, (_case_name, case_label, _tau_count) in enumerate(shift_refz_startup_cases):
            ref_p = startup_cols[4 * case_idx]
            ref_m = startup_cols[4 * case_idx + 1]
            gate_p = startup_cols[4 * case_idx + 2]
            gate_m = startup_cols[4 * case_idx + 3]
            gate_diff = gate_p - gate_m
            gate_common = 0.5 * (gate_p + gate_m)
            sampled_diff = stat(2.70e-6, gate_diff)
            sampled_common = stat(2.70e-6, gate_common)
            branch_trim_error.append(abs(sampled_diff - reset_trim_v))
            branch_gate_common.append(sampled_common)
            branch_gate_diff.append(sampled_diff)
            if case_label in {"cold", "3 tau", "initialized"}:
                shift_refz_startup_traces.append(
                    (branch_label, case_label, stt, ref_p - ref_m, gate_diff, gate_common)
                )
        shift_refz_startup_trim_error.append(branch_trim_error)
        shift_refz_startup_gate_common.append(branch_gate_common)
        shift_refz_startup_gate_diff.append(branch_gate_diff)

    shift_refz_startup_trim_error = np.array(shift_refz_startup_trim_error)
    shift_refz_startup_gate_common = np.array(shift_refz_startup_gate_common)
    shift_refz_startup_gate_diff = np.array(shift_refz_startup_gate_diff)
    cold_idx = 0
    tau3_idx = 4
    initialized_idx = 5
    require(
        np.all(shift_refz_startup_trim_error[:, cold_idx] > 0.040),
        "cold 250 pF trim-reference reservoirs should visibly under-deliver split trim",
    )
    require(
        np.all(np.diff(shift_refz_startup_trim_error, axis=1) < 0.0),
        "trim-reference startup error should improve monotonically with reservoir precharge",
    )
    require(
        np.all(shift_refz_startup_trim_error[:, tau3_idx] < 0.005),
        "three RC time constants of reservoir precharge should recover split-trim delivery below 5 mV",
    )
    require(
        np.all(shift_refz_startup_gate_common[:, tau3_idx] > 0.84),
        "three RC time constants should restore enough reset-reference common mode",
    )
    require(
        np.all(shift_refz_startup_trim_error[:, initialized_idx] < 0.0015),
        "initialized trim-reference reservoirs should match the earlier decoupling trim accuracy",
    )

    shift_refz_precharge_cases = [
        ("nopre", "none", 0.0),
        ("pre2ns", "2 ns", 2.0),
        ("pre5ns", "5 ns", 5.0),
        ("pre10ns", "10 ns", 10.0),
        ("pre20ns", "20 ns", 20.0),
        ("pre40ns", "40 ns", 40.0),
    ]

    def run_reset_ref_precharge_case(
        branch_name: str,
        reset_trim_v: float,
        reset_common_v: float = 0.90,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        common_tag = "" if abs(reset_common_v - 0.90) < 1e-12 else f"_cm{int(round(reset_common_v * 100)):03d}"
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_shifted_gate_reset_ref_precharge{common_tag}_{branch_name}"
        )
        reset_ref_p = reset_common_v + 0.5 * reset_trim_v
        reset_ref_m = reset_common_v - 0.5 * reset_trim_v
        deck_lines = [
            "* MOS startup-precharge check for the local split-reset trim reservoir.",
            "* Reservoir capacitors start cold. A temporary low-impedance TG path",
            "* charges them before the ordinary reset pulse samples the rails.",
            COMMON_MODELS,
            ".param CREF=250p CGATE=5p WRESETN=60u WRESETP=180u WPREN=300u WPREP=900u",
            "VDD vdd 0 1.8",
            "VRST rst 0 PWL(0 0 2.48u 0 2.50u 1.8 2.62u 1.8 2.64u 0 3.0u 0)",
            "VRSTN rstn 0 PWL(0 1.8 2.48u 1.8 2.50u 0 2.62u 0 2.64u 1.8 3.0u 1.8)",
        ]
        prints = []
        for idx, (case_name, _case_label, width_ns) in enumerate(shift_refz_precharge_cases):
            if width_ns <= 0.0:
                pre_pwl = f"VPRE_{idx} pre_{idx} 0 0"
                pren_pwl = f"VPREN_{idx} pren_{idx} 0 1.8"
            else:
                pre_end_us = 0.420 + 0.001 * width_ns
                pre_fall_us = pre_end_us + 0.020
                pre_pwl = (
                    f"VPRE_{idx} pre_{idx} 0 PWL(0 0 0.400u 0 0.420u 1.8 "
                    f"{pre_end_us:.3f}u 1.8 {pre_fall_us:.3f}u 0 3.0u 0)"
                )
                pren_pwl = (
                    f"VPREN_{idx} pren_{idx} 0 PWL(0 1.8 0.400u 1.8 0.420u 0 "
                    f"{pre_end_us:.3f}u 0 {pre_fall_us:.3f}u 1.8 3.0u 1.8)"
                )
            deck_lines.extend(
                [
                    f"* {case_name}: cold reservoir, startup precharge width {width_ns:g} ns.",
                    pre_pwl,
                    pren_pwl,
                    f"VZRP_PRE_{idx} zrp_pre_src_{idx} 0 {reset_ref_p:.5f}",
                    f"VZRM_PRE_{idx} zrm_pre_src_{idx} 0 {reset_ref_m:.5f}",
                    f"RZRP_PRE_{idx} zrp_pre_src_{idx} zrp_pre_{idx} 100k",
                    f"RZRM_PRE_{idx} zrm_pre_src_{idx} zrm_pre_{idx} 100k",
                    f"CZRP_PRE_{idx} zrp_pre_{idx} 0 {{CREF}} IC=0",
                    f"CZRM_PRE_{idx} zrm_pre_{idx} 0 {{CREF}} IC=0",
                    f"MPREPN_{idx} zrp_pre_{idx} pre_{idx} zrp_pre_src_{idx} 0 NMOS L={{LCH}} W={{WPREN}}",
                    f"MPREMN_{idx} zrm_pre_{idx} pre_{idx} zrm_pre_src_{idx} 0 NMOS L={{LCH}} W={{WPREN}}",
                    f"MPREPP_{idx} zrp_pre_{idx} pren_{idx} zrp_pre_src_{idx} vdd PMOS L={{LCH}} W={{WPREP}}",
                    f"MPREMP_{idx} zrm_pre_{idx} pren_{idx} zrm_pre_src_{idx} vdd PMOS L={{LCH}} W={{WPREP}}",
                    f"CZPG_PRE_{idx} zpg_pre_{idx} 0 {{CGATE}} IC={reset_common_v:.5f}",
                    f"CZMG_PRE_{idx} zmg_pre_{idx} 0 {{CGATE}} IC={reset_common_v:.5f}",
                    f"MRZGPN_PRE_{idx} zpg_pre_{idx} rst zrp_pre_{idx} 0 NMOS L={{LCH}} W={{WRESETN}}",
                    f"MRZGMN_PRE_{idx} zmg_pre_{idx} rst zrm_pre_{idx} 0 NMOS L={{LCH}} W={{WRESETN}}",
                    f"MRZGPP_PRE_{idx} zpg_pre_{idx} rstn zrp_pre_{idx} vdd PMOS L={{LCH}} W={{WRESETP}}",
                    f"MRZGMP_PRE_{idx} zmg_pre_{idx} rstn zrm_pre_{idx} vdd PMOS L={{LCH}} W={{WRESETP}}",
                ]
            )
            prints.extend(
                [
                    f"v(zrp_pre_{idx})",
                    f"v(zrm_pre_{idx})",
                    f"v(zpg_pre_{idx})",
                    f"v(zmg_pre_{idx})",
                ]
            )
        deck_lines.extend(
            [
                ".control",
                "set noaskquit",
                "tran 1n 3.0u uic",
                f"wrdata {stem}.dat " + " ".join(prints),
                "quit",
                ".endc",
                ".end",
            ]
        )
        data = run_ngspice("\n".join(deck_lines), stem)
        return load_wrdata(data, len(prints))

    shift_refz_precharge_trim_error = []
    shift_refz_precharge_gate_common = []
    shift_refz_precharge_gate_diff = []
    shift_refz_precharge_traces = []
    for branch_name, branch_label, _nfp_vto, _nfm_vto, reset_trim_v in shift_reset_branch_specs:
        pct, precharge_cols = run_reset_ref_precharge_case(branch_name, reset_trim_v)

        def pcat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(pct - time_s))])

        branch_trim_error = []
        branch_gate_common = []
        branch_gate_diff = []
        for case_idx, (_case_name, case_label, _width_ns) in enumerate(shift_refz_precharge_cases):
            ref_p = precharge_cols[4 * case_idx]
            ref_m = precharge_cols[4 * case_idx + 1]
            gate_p = precharge_cols[4 * case_idx + 2]
            gate_m = precharge_cols[4 * case_idx + 3]
            gate_diff = gate_p - gate_m
            gate_common = 0.5 * (gate_p + gate_m)
            sampled_diff = pcat(2.70e-6, gate_diff)
            sampled_common = pcat(2.70e-6, gate_common)
            branch_trim_error.append(abs(sampled_diff - reset_trim_v))
            branch_gate_common.append(sampled_common)
            branch_gate_diff.append(sampled_diff)
            if case_label in {"none", "5 ns", "10 ns", "20 ns"}:
                shift_refz_precharge_traces.append(
                    (branch_label, case_label, pct, ref_p - ref_m, gate_diff, gate_common)
                )
        shift_refz_precharge_trim_error.append(branch_trim_error)
        shift_refz_precharge_gate_common.append(branch_gate_common)
        shift_refz_precharge_gate_diff.append(branch_gate_diff)

    shift_refz_precharge_trim_error = np.array(shift_refz_precharge_trim_error)
    shift_refz_precharge_gate_common = np.array(shift_refz_precharge_gate_common)
    shift_refz_precharge_gate_diff = np.array(shift_refz_precharge_gate_diff)
    pre_none_idx = 0
    pre_5ns_idx = 2
    pre_10ns_idx = 3
    pre_20ns_idx = 4
    require(
        np.all(shift_refz_precharge_trim_error[:, pre_none_idx] > 0.040),
        "cold reservoirs without startup precharge should reproduce the startup failure",
    )
    require(
        np.all(np.diff(shift_refz_precharge_trim_error, axis=1) < 0.0),
        "startup precharge trim error should improve monotonically with pulse width",
    )
    require(
        np.all(shift_refz_precharge_trim_error[:, pre_5ns_idx] > 0.006),
        "5 ns startup precharge should still be visibly marginal",
    )
    require(
        np.all(shift_refz_precharge_trim_error[:, pre_10ns_idx] < 0.004),
        "10 ns startup precharge should recover split-trim delivery below 4 mV",
    )
    require(
        np.all(shift_refz_precharge_trim_error[:, pre_20ns_idx] < 0.0015),
        "20 ns startup precharge should recover initialized-reservoir trim accuracy",
    )
    require(
        np.all(shift_refz_precharge_gate_common[:, pre_20ns_idx] > 0.89),
        "20 ns startup precharge should restore reset-reference common mode",
    )

    shift_refz_precharge_tuned_trim_error = []
    shift_refz_precharge_tuned_gate_common = []
    shift_refz_precharge_tuned_gate_diff = []
    shift_refz_precharge_tuned_traces = []
    shift_refz_precharge_tuned_common_v = 0.80
    for branch_name, branch_label, _nfp_vto, _nfm_vto, reset_trim_v in shift_reset_branch_specs:
        pct, precharge_cols = run_reset_ref_precharge_case(
            branch_name,
            reset_trim_v,
            reset_common_v=shift_refz_precharge_tuned_common_v,
        )

        def pcat080(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(pct - time_s))])

        branch_trim_error = []
        branch_gate_common = []
        branch_gate_diff = []
        for case_idx, (_case_name, case_label, _width_ns) in enumerate(shift_refz_precharge_cases):
            ref_p = precharge_cols[4 * case_idx]
            ref_m = precharge_cols[4 * case_idx + 1]
            gate_p = precharge_cols[4 * case_idx + 2]
            gate_m = precharge_cols[4 * case_idx + 3]
            gate_diff = gate_p - gate_m
            gate_common = 0.5 * (gate_p + gate_m)
            sampled_diff = pcat080(2.70e-6, gate_diff)
            sampled_common = pcat080(2.70e-6, gate_common)
            branch_trim_error.append(abs(sampled_diff - reset_trim_v))
            branch_gate_common.append(sampled_common)
            branch_gate_diff.append(sampled_diff)
            if case_label in {"none", "5 ns", "10 ns", "20 ns"}:
                shift_refz_precharge_tuned_traces.append(
                    (branch_label, case_label, pct, ref_p - ref_m, gate_diff, gate_common)
                )
        shift_refz_precharge_tuned_trim_error.append(branch_trim_error)
        shift_refz_precharge_tuned_gate_common.append(branch_gate_common)
        shift_refz_precharge_tuned_gate_diff.append(branch_gate_diff)

    shift_refz_precharge_tuned_trim_error = np.array(shift_refz_precharge_tuned_trim_error)
    shift_refz_precharge_tuned_gate_common = np.array(shift_refz_precharge_tuned_gate_common)
    shift_refz_precharge_tuned_gate_diff = np.array(shift_refz_precharge_tuned_gate_diff)
    require(
        np.all(shift_refz_precharge_tuned_trim_error[:, pre_none_idx] > 0.040),
        "cold tuned-common reservoirs without startup precharge should reproduce the startup failure",
    )
    require(
        np.all(np.diff(shift_refz_precharge_tuned_trim_error, axis=1) < 0.0),
        "tuned-common startup precharge trim error should improve monotonically with pulse width",
    )
    require(
        np.all(shift_refz_precharge_tuned_trim_error[:, pre_5ns_idx] > 0.006),
        "5 ns tuned-common startup precharge should still be visibly marginal",
    )
    require(
        np.all(shift_refz_precharge_tuned_trim_error[:, pre_10ns_idx] < 0.006),
        "10 ns tuned-common startup precharge should recover split-trim delivery below the 6 mV gate",
    )
    require(
        np.all(shift_refz_precharge_tuned_trim_error[:, pre_20ns_idx] < 0.0017),
        "20 ns tuned-common startup precharge should recover initialized-reservoir trim accuracy",
    )
    require(
        np.all(np.abs(shift_refz_precharge_tuned_gate_common[:, pre_20ns_idx] - shift_refz_precharge_tuned_common_v) < 0.006),
        "20 ns tuned-common startup precharge should restore the 0.80 V reset-reference common mode",
    )

    shift_refz_precharge_strength_widths = [
        ("w0125", "0.125x", 37.5, 112.5),
        ("w025", "0.25x", 75.0, 225.0),
        ("w05", "0.5x", 150.0, 450.0),
        ("w10", "1.0x", 300.0, 900.0),
    ]
    shift_refz_precharge_strength_pulses = [
        ("pre20ns", "20 ns", 20.0),
        ("pre40ns", "40 ns", 40.0),
    ]

    def run_reset_ref_precharge_strength_case(
        branch_name: str,
        reset_trim_v: float,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_"
            f"forward_pair_96u_shifted_gate_reset_ref_precharge_strength_{branch_name}"
        )
        reset_ref_p = 0.90 + 0.5 * reset_trim_v
        reset_ref_m = 0.90 - 0.5 * reset_trim_v
        deck_lines = [
            "* MOS startup-precharge strength margin check for the local trim reservoir.",
            "* Reservoir capacitors start cold. The startup TG width and pulse width",
            "* are swept before the ordinary reset samples the split trim rails.",
            COMMON_MODELS,
            ".param CREF=250p CGATE=5p WRESETN=60u WRESETP=180u",
            "VDD vdd 0 1.8",
            "VRST rst 0 PWL(0 0 2.48u 0 2.50u 1.8 2.62u 1.8 2.64u 0 3.0u 0)",
            "VRSTN rstn 0 PWL(0 1.8 2.48u 1.8 2.50u 0 2.62u 0 2.64u 1.8 3.0u 1.8)",
        ]
        prints = []
        case_idx = 0
        for width_name, width_label, wpre_n_um, wpre_p_um in shift_refz_precharge_strength_widths:
            for pulse_name, pulse_label, pulse_width_ns in shift_refz_precharge_strength_pulses:
                pre_end_us = 0.420 + 0.001 * pulse_width_ns
                pre_fall_us = pre_end_us + 0.020
                deck_lines.extend(
                    [
                        (
                            f"* {width_name}_{pulse_name}: cold reservoir, startup TG {width_label}, "
                            f"startup pulse {pulse_label}."
                        ),
                        (
                            f"VPRE_STR_{case_idx} pre_str_{case_idx} 0 "
                            f"PWL(0 0 0.400u 0 0.420u 1.8 {pre_end_us:.3f}u 1.8 "
                            f"{pre_fall_us:.3f}u 0 3.0u 0)"
                        ),
                        (
                            f"VPREN_STR_{case_idx} pren_str_{case_idx} 0 "
                            f"PWL(0 1.8 0.400u 1.8 0.420u 0 {pre_end_us:.3f}u 0 "
                            f"{pre_fall_us:.3f}u 1.8 3.0u 1.8)"
                        ),
                        f"VZRP_STR_{case_idx} zrp_str_src_{case_idx} 0 {reset_ref_p:.5f}",
                        f"VZRM_STR_{case_idx} zrm_str_src_{case_idx} 0 {reset_ref_m:.5f}",
                        f"RZRP_STR_{case_idx} zrp_str_src_{case_idx} zrp_str_{case_idx} 100k",
                        f"RZRM_STR_{case_idx} zrm_str_src_{case_idx} zrm_str_{case_idx} 100k",
                        f"CZRP_STR_{case_idx} zrp_str_{case_idx} 0 {{CREF}} IC=0",
                        f"CZRM_STR_{case_idx} zrm_str_{case_idx} 0 {{CREF}} IC=0",
                        (
                            f"MPREPN_STR_{case_idx} zrp_str_{case_idx} pre_str_{case_idx} "
                            f"zrp_str_src_{case_idx} 0 NMOS L={{LCH}} W={wpre_n_um:g}u"
                        ),
                        (
                            f"MPREMN_STR_{case_idx} zrm_str_{case_idx} pre_str_{case_idx} "
                            f"zrm_str_src_{case_idx} 0 NMOS L={{LCH}} W={wpre_n_um:g}u"
                        ),
                        (
                            f"MPREPP_STR_{case_idx} zrp_str_{case_idx} pren_str_{case_idx} "
                            f"zrp_str_src_{case_idx} vdd PMOS L={{LCH}} W={wpre_p_um:g}u"
                        ),
                        (
                            f"MPREMP_STR_{case_idx} zrm_str_{case_idx} pren_str_{case_idx} "
                            f"zrm_str_src_{case_idx} vdd PMOS L={{LCH}} W={wpre_p_um:g}u"
                        ),
                        f"CZPG_STR_{case_idx} zpg_str_{case_idx} 0 {{CGATE}} IC=0.90",
                        f"CZMG_STR_{case_idx} zmg_str_{case_idx} 0 {{CGATE}} IC=0.90",
                        (
                            f"MRZGPN_STR_{case_idx} zpg_str_{case_idx} rst zrp_str_{case_idx} "
                            f"0 NMOS L={{LCH}} W={{WRESETN}}"
                        ),
                        (
                            f"MRZGMN_STR_{case_idx} zmg_str_{case_idx} rst zrm_str_{case_idx} "
                            f"0 NMOS L={{LCH}} W={{WRESETN}}"
                        ),
                        (
                            f"MRZGPP_STR_{case_idx} zpg_str_{case_idx} rstn zrp_str_{case_idx} "
                            f"vdd PMOS L={{LCH}} W={{WRESETP}}"
                        ),
                        (
                            f"MRZGMP_STR_{case_idx} zmg_str_{case_idx} rstn zrm_str_{case_idx} "
                            f"vdd PMOS L={{LCH}} W={{WRESETP}}"
                        ),
                    ]
                )
                prints.extend(
                    [
                        f"v(zrp_str_{case_idx})",
                        f"v(zrm_str_{case_idx})",
                        f"v(zpg_str_{case_idx})",
                        f"v(zmg_str_{case_idx})",
                    ]
                )
                case_idx += 1
        deck_lines.extend(
            [
                ".control",
                "set noaskquit",
                "tran 1n 3.0u uic",
                f"wrdata {stem}.dat " + " ".join(prints),
                "quit",
                ".endc",
                ".end",
            ]
        )
        data = run_ngspice("\n".join(deck_lines), stem)
        return load_wrdata(data, len(prints))

    shift_refz_precharge_strength_trim_error = []
    shift_refz_precharge_strength_common = []
    for branch_name, _branch_label, _nfp_vto, _nfm_vto, reset_trim_v in shift_reset_branch_specs:
        stt, strength_cols = run_reset_ref_precharge_strength_case(branch_name, reset_trim_v)

        def scat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(stt - time_s))])

        branch_trim_error = []
        branch_common = []
        case_idx = 0
        for _width_name, _width_label, _wpre_n_um, _wpre_p_um in shift_refz_precharge_strength_widths:
            pulse_trim_error = []
            pulse_common = []
            for _pulse_name, _pulse_label, _pulse_width_ns in shift_refz_precharge_strength_pulses:
                gate_p = strength_cols[4 * case_idx + 2]
                gate_m = strength_cols[4 * case_idx + 3]
                gate_diff = gate_p - gate_m
                gate_common = 0.5 * (gate_p + gate_m)
                pulse_trim_error.append(abs(scat(2.70e-6, gate_diff) - reset_trim_v))
                pulse_common.append(scat(2.70e-6, gate_common))
                case_idx += 1
            branch_trim_error.append(pulse_trim_error)
            branch_common.append(pulse_common)
        shift_refz_precharge_strength_trim_error.append(branch_trim_error)
        shift_refz_precharge_strength_common.append(branch_common)

    shift_refz_precharge_strength_trim_error = np.array(shift_refz_precharge_strength_trim_error)
    shift_refz_precharge_strength_common = np.array(shift_refz_precharge_strength_common)
    strength_20ns_idx = 0
    strength_40ns_idx = 1
    strength_0125_idx = 0
    strength_025_idx = 1
    strength_05_idx = 2
    strength_10_idx = 3
    require(
        np.all(np.diff(shift_refz_precharge_strength_trim_error[:, :, strength_20ns_idx], axis=1) < 0.0),
        "20 ns startup-precharge trim error should improve monotonically with TG strength",
    )
    require(
        np.all(np.diff(shift_refz_precharge_strength_trim_error[:, :, strength_40ns_idx], axis=1) < 0.0),
        "40 ns startup-precharge trim error should improve monotonically with TG strength",
    )
    require(
        np.all(shift_refz_precharge_strength_trim_error[:, strength_0125_idx, strength_40ns_idx] > 0.015),
        "0.125x startup precharge should remain visibly too weak even at 40 ns",
    )
    require(
        np.all(shift_refz_precharge_strength_trim_error[:, strength_025_idx, strength_20ns_idx] > 0.014),
        "0.25x startup precharge should remain marginal at 20 ns",
    )
    require(
        np.all(shift_refz_precharge_strength_trim_error[:, strength_025_idx, strength_40ns_idx] < 0.006),
        "0.25x startup precharge should recover below the 6 mV trim-error gate at 40 ns",
    )
    require(
        np.all(shift_refz_precharge_strength_trim_error[:, strength_05_idx, strength_20ns_idx] < 0.006),
        "0.5x startup precharge should recover below the 6 mV trim-error gate at 20 ns",
    )
    require(
        np.all(shift_refz_precharge_strength_trim_error[:, strength_10_idx, strength_20ns_idx] < 0.0015),
        "nominal startup precharge should retain initialized-reservoir accuracy at 20 ns",
    )
    require(
        np.all(shift_refz_precharge_strength_common[:, strength_05_idx, strength_20ns_idx] > 0.84),
        "0.5x 20 ns startup precharge should restore enough reset-reference common mode",
    )

    tail_bias_forward_tail_line = "MNFT_HYR ftail_hyr vbias 0 0 NMOS L={LCH} W=48u"
    tail_bias_cases_v = [0.70, 0.80, 0.90, 0.95, 1.05, 1.15, 1.25]
    tail_bias_labels = []
    tail_bias_preact_samples = []
    tail_bias_load_samples = []
    tail_bias_store_samples = []
    tail_bias_traces = []
    for tail_bias_v in tail_bias_cases_v:
        stem = (
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_pact_"
            f"guard_forward_tail_bias_{str(tail_bias_v).replace('.', 'p')}v"
        )
        tail_bias_deck = replace_required(hybrid_forward_read_reuse_deck, timing_read_pwl, timing_single_read_pwl)
        tail_bias_deck = replace_required(
            tail_bias_deck,
            timing_base_pact_pwl,
            long_pact_pwl + "\n" + long_pactn_pwl + "\n" + corner_guard_pwl + "\n" + corner_guardn_pwl,
        )
        tail_bias_deck = replace_required(tail_bias_deck, nmos_store_line_p, guard_store_line_p)
        tail_bias_deck = replace_required(tail_bias_deck, nmos_store_line_m, guard_store_line_m)
        tail_bias_deck = replace_required(
            tail_bias_deck,
            tail_bias_forward_tail_line,
            f"VFBIAS_HYR vfbias_hyr 0 {tail_bias_v:.2f}\n"
            "MNFT_HYR ftail_hyr vfbias_hyr 0 0 NMOS L={LCH} W=48u",
        )
        tail_bias_deck = replace_required(
            tail_bias_deck,
            "mos_hidden_writer_restored_gate_hybrid_update_forward_read_reuse.dat",
            f"{stem}.dat",
        )
        tail_bias_data = run_ngspice(tail_bias_deck, stem)
        tbt, tail_bias_cols = load_wrdata(tail_bias_data, 23)

        def tbat(time_s: float, values: np.ndarray) -> float:
            return float(values[np.argmin(np.abs(tbt - time_s))])

        tail_bias_preact = tail_bias_cols[10] - tail_bias_cols[9]
        tail_bias_load = tail_bias_cols[12] - tail_bias_cols[11]
        tail_bias_store = tail_bias_cols[14] - tail_bias_cols[13]
        tail_bias_labels.append(f"{tail_bias_v:.2f} V")
        tail_bias_preact_samples.append(tbat(3.575e-6, tail_bias_preact))
        tail_bias_load_samples.append(tbat(3.315e-6, tail_bias_load))
        tail_bias_store_samples.append(tbat(3.575e-6, tail_bias_store))
        tail_bias_traces.append((f"{tail_bias_v:.2f} V", tbt, tail_bias_load, tail_bias_store))

    tail_bias_preact_samples = np.array(tail_bias_preact_samples)
    tail_bias_load_samples = np.array(tail_bias_load_samples)
    tail_bias_store_samples = np.array(tail_bias_store_samples)
    nominal_tail_bias_idx = tail_bias_cases_v.index(0.95)
    require(np.min(tail_bias_preact_samples) > 0.048, "tail-bias sweep should keep a valid read state")
    require(
        np.max(tail_bias_preact_samples) - np.min(tail_bias_preact_samples) < 0.001,
        "tail-bias sweep should not perturb the stored preactivation",
    )
    require(
        abs(tail_bias_store_samples[nominal_tail_bias_idx] - guard_timing_samples[2]) < 0.002,
        "nominal tail bias should match the nominal guard timing sample",
    )
    require(
        tail_bias_store_samples[0] < tail_bias_store_samples[nominal_tail_bias_idx],
        "weak isolated tail bias should slightly reduce the activation store",
    )
    require(
        np.all(np.diff(tail_bias_store_samples) > 0),
        "stored activation should increase monotonically with isolated forward tail bias",
    )
    require(
        tail_bias_store_samples[-1] - tail_bias_store_samples[0] < 0.003,
        "isolated forward tail bias should act as a fine trim, not a large gain knob in this sizing",
    )

    read_gated_fig, read_gated_axes = plt.subplots(2, 1, figsize=(7.4, 6.6), gridspec_kw={"height_ratios": [1.0, 0.9]})
    for label, rgt, store in guard_gated_traces:
        if label in {"40 ns", "120 ns", "240 ns", "320 ns"}:
            read_gated_axes[0].plot(1e6 * rgt, 1e3 * store, label=f"guarded {label}")
    read_gated_axes[0].axhline(0, color="0.4", linewidth=0.8)
    read_gated_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    read_gated_axes[0].axvline(3.38, color="0.5", linestyle=":", linewidth=0.9, alpha=0.6, label="read off")
    read_gated_axes[0].set_xlim(3.05, 3.65)
    read_gated_axes[0].set_ylabel("stored activation (mV)")
    read_gated_axes[0].set_title("Guarded transmission store disconnects before forward-load collapse")
    read_gated_axes[0].grid(True, alpha=0.25)
    read_gated_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    read_gated_axes[1].plot(aperture_cases_ns, 1e3 * aperture_store_samples, "o-", label="NMOS-only")
    read_gated_axes[1].plot(aperture_cases_ns, 1e3 * tg_aperture_store_samples, "s--", label="TG")
    read_gated_axes[1].plot(aperture_cases_ns, 1e3 * read_gated_store_samples, "^-.", label="read-gated TG")
    read_gated_axes[1].plot(aperture_cases_ns, 1e3 * guard_gated_store_samples, "d:", label="guarded TG")
    read_gated_axes[1].set_xlabel("pact high-time (ns)")
    read_gated_axes[1].set_ylabel("stored $h^- - h^+$ (mV)")
    read_gated_axes[1].set_title("Guarding the store with read-valid widens the safe aperture")
    read_gated_axes[1].grid(True, alpha=0.25)
    read_gated_axes[1].legend(loc="lower right")
    read_gated_fig.tight_layout()
    save_plot(read_gated_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_read_gated_store_ngspice")

    guard_timing_fig, guard_timing_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, gt, store in guard_timing_traces:
        if label in {"60 ns", "130 ns", "180 ns", "220 ns"}:
            guard_timing_axes[0].plot(1e6 * gt, 1e3 * store, label=f"guard {label}")
    guard_timing_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_timing_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="nominal guard")
    guard_timing_axes[0].axvline(3.38, color="0.5", linestyle=":", linewidth=0.9, alpha=0.6, label="read off")
    guard_timing_axes[0].set_xlim(3.05, 3.65)
    guard_timing_axes[0].set_ylabel("stored activation (mV)")
    guard_timing_axes[0].set_title("Guard-off timing has a broad plateau before droop")
    guard_timing_axes[0].grid(True, alpha=0.25)
    guard_timing_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    guard_timing_x = np.arange(len(guard_timing_labels))
    guard_timing_axes[1].bar(guard_timing_x - 0.18, 1e3 * guard_timing_preact_samples, width=0.36, label="$z^- - z^+$")
    guard_timing_axes[1].bar(guard_timing_x + 0.18, 1e3 * guard_timing_samples, width=0.36, label="stored $h^- - h^+$")
    guard_timing_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_timing_axes[1].set_xticks(guard_timing_x)
    guard_timing_axes[1].set_xticklabels(guard_timing_labels)
    guard_timing_axes[1].set_xlabel("guard high-time")
    guard_timing_axes[1].set_ylabel("sampled differential (mV)")
    guard_timing_axes[1].set_title("Guard can close early after capture, but not after forward-load collapse")
    guard_timing_axes[1].grid(True, axis="y", alpha=0.25)
    guard_timing_axes[1].legend(loc="upper right", fontsize="small")
    guard_timing_fig.tight_layout()
    save_plot(guard_timing_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_timing_ngspice")

    guard_skew_fig, guard_skew_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, gst, store in guard_skew_traces:
        if label in {"-120 ns", "-40 ns", "+0 ns", "+80 ns", "+120 ns"}:
            guard_skew_axes[0].plot(1e6 * gst, 1e3 * store, label=f"skew {label}")
    guard_skew_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_skew_axes[0].axvline(3.18, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="nominal start")
    guard_skew_axes[0].axvline(3.33, color="0.5", linestyle=":", linewidth=0.9, alpha=0.6, label="nominal off")
    guard_skew_axes[0].set_xlim(3.05, 3.65)
    guard_skew_axes[0].set_ylabel("stored activation (mV)")
    guard_skew_axes[0].set_title("Guard clock skew trades undercharge against late droop")
    guard_skew_axes[0].grid(True, alpha=0.25)
    guard_skew_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    guard_skew_x = np.arange(len(guard_skew_labels))
    guard_skew_axes[1].bar(guard_skew_x - 0.18, 1e3 * guard_skew_preact_samples, width=0.36, label="$z^- - z^+$")
    guard_skew_axes[1].bar(guard_skew_x + 0.18, 1e3 * guard_skew_store_samples, width=0.36, label="stored $h^- - h^+$")
    guard_skew_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_skew_axes[1].set_xticks(guard_skew_x)
    guard_skew_axes[1].set_xticklabels(guard_skew_labels)
    guard_skew_axes[1].set_xlabel("guard-window shift")
    guard_skew_axes[1].set_ylabel("sampled differential (mV)")
    guard_skew_axes[1].set_title("The usable skew window is asymmetric around the nominal guard")
    guard_skew_axes[1].grid(True, axis="y", alpha=0.25)
    guard_skew_axes[1].legend(loc="upper right", fontsize="small")
    guard_skew_fig.tight_layout()
    save_plot(guard_skew_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_skew_ngspice")

    guard_edge_fig, guard_edge_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, get, store in guard_edge_traces:
        if label in {"5 ns", "20 ns", "40 ns", "80 ns", "120 ns"}:
            guard_edge_axes[0].plot(1e6 * get, 1e3 * store, label=f"edge {label}")
    guard_edge_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_edge_axes[0].axvline(3.31, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="fall starts")
    guard_edge_axes[0].axvline(3.38, color="0.5", linestyle=":", linewidth=0.9, alpha=0.6, label="read off")
    guard_edge_axes[0].set_xlim(3.15, 3.48)
    guard_edge_axes[0].set_ylabel("stored activation (mV)")
    guard_edge_axes[0].set_title("Slow guard fall edges extend the effective capture window")
    guard_edge_axes[0].grid(True, alpha=0.25)
    guard_edge_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    guard_edge_x = np.arange(len(guard_edge_labels))
    guard_edge_axes[1].bar(guard_edge_x - 0.18, 1e3 * guard_edge_preact_samples, width=0.36, label="$z^- - z^+$")
    guard_edge_axes[1].bar(guard_edge_x + 0.18, 1e3 * guard_edge_store_samples, width=0.36, label="stored $h^- - h^+$")
    guard_edge_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_edge_axes[1].set_xticks(guard_edge_x)
    guard_edge_axes[1].set_xticklabels(guard_edge_labels)
    guard_edge_axes[1].set_xlabel("guard rise/fall edge")
    guard_edge_axes[1].set_ylabel("sampled differential (mV)")
    guard_edge_axes[1].set_title("The tested edge slew costs about 5 mV only at 120 ns")
    guard_edge_axes[1].grid(True, axis="y", alpha=0.25)
    guard_edge_axes[1].legend(loc="lower right", ncol=2, fontsize="small")
    guard_edge_fig.tight_layout()
    save_plot(guard_edge_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_edge_ngspice")

    pact_edge_fig, pact_edge_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, pet, store, pact_ctl in pact_edge_traces:
        if label in {"5 ns", "20 ns", "40 ns", "80 ns", "120 ns"}:
            pact_edge_axes[0].plot(1e6 * pet, 1e3 * store, label=f"pact edge {label}")
    pact_edge_axes[0].plot(1e6 * pact_edge_traces[nominal_pact_edge_idx][1], pact_edge_traces[nominal_pact_edge_idx][3] / 20.0, color="0.5", alpha=0.25, label="$pact/20$")
    pact_edge_axes[0].axhline(0, color="0.4", linewidth=0.8)
    pact_edge_axes[0].axvline(3.18, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard rise")
    pact_edge_axes[0].axvline(3.31, color="0.45", linestyle=":", linewidth=0.9, alpha=0.6, label="guard fall")
    pact_edge_axes[0].set_xlim(3.05, 3.45)
    pact_edge_axes[0].set_ylabel("stored activation (mV)")
    pact_edge_axes[0].set_title("Slow pact rise reduces charge time inside the guard window")
    pact_edge_axes[0].grid(True, alpha=0.25)
    pact_edge_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    pact_edge_x = np.arange(len(pact_edge_labels))
    pact_edge_axes[1].bar(pact_edge_x - 0.18, 1e3 * pact_edge_preact_samples, width=0.36, label="$z^- - z^+$")
    pact_edge_axes[1].bar(pact_edge_x + 0.18, 1e3 * pact_edge_store_samples, width=0.36, label="stored $h^- - h^+$")
    pact_edge_axes[1].axhline(0, color="0.4", linewidth=0.8)
    pact_edge_axes[1].set_xticks(pact_edge_x)
    pact_edge_axes[1].set_xticklabels(pact_edge_labels)
    pact_edge_axes[1].set_xlabel("pact rise/fall edge")
    pact_edge_axes[1].set_ylabel("sampled differential (mV)")
    pact_edge_axes[1].set_title("Pact must be mostly valid when the guard window opens")
    pact_edge_axes[1].grid(True, axis="y", alpha=0.25)
    pact_edge_axes[1].legend(loc="lower left", ncol=2, fontsize="small")
    pact_edge_fig.tight_layout()
    save_plot(pact_edge_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_pact_edge_ngspice")

    guard_start_fig, guard_start_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, gst, store in guard_start_traces:
        if label in {"3.10 us", "3.18 us", "3.26 us", "3.29 us"}:
            guard_start_axes[0].plot(1e6 * gst, 1e3 * store, label=f"start {label}")
    guard_start_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_start_axes[0].axvline(3.18, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="nominal start")
    guard_start_axes[0].axvline(3.31, color="0.45", linestyle=":", linewidth=0.9, alpha=0.6, label="fixed fall")
    guard_start_axes[0].set_xlim(3.05, 3.45)
    guard_start_axes[0].set_ylabel("stored activation (mV)")
    guard_start_axes[0].set_title("Guard-start timing isolates charge-time loss with fixed guard close")
    guard_start_axes[0].grid(True, alpha=0.25)
    guard_start_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    guard_start_x = np.arange(len(guard_start_labels))
    guard_start_axes[1].bar(guard_start_x - 0.18, 1e3 * guard_start_preact_samples, width=0.36, label="$z^- - z^+$")
    guard_start_axes[1].bar(guard_start_x + 0.18, 1e3 * guard_start_store_samples, width=0.36, label="stored $h^- - h^+$")
    guard_start_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_start_axes[1].set_xticks(guard_start_x)
    guard_start_axes[1].set_xticklabels(guard_start_labels)
    guard_start_axes[1].set_xlabel("guard rise end")
    guard_start_axes[1].set_ylabel("sampled differential (mV)")
    guard_start_axes[1].set_title("Opening before pact is harmless; opening after nominal cuts charge time")
    guard_start_axes[1].grid(True, axis="y", alpha=0.25)
    guard_start_axes[1].legend(loc="lower left", ncol=2, fontsize="small")
    guard_start_fig.tight_layout()
    save_plot(guard_start_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_start_ngspice")

    read_fall_fig, read_fall_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, rft, store, read_ctl in read_fall_traces:
        if label in {"3.26 us", "3.30 us", "3.36 us", "3.44 us"}:
            read_fall_axes[0].plot(1e6 * rft, 1e3 * store, label=f"read fall {label}")
    read_fall_axes[0].plot(1e6 * read_fall_traces[nominal_read_fall_idx][1], read_fall_traces[nominal_read_fall_idx][3] / 20.0, color="0.5", alpha=0.3, label="$read/20$")
    read_fall_axes[0].axhline(0, color="0.4", linewidth=0.8)
    read_fall_axes[0].axvline(3.31, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard fall")
    read_fall_axes[0].set_xlim(3.15, 3.55)
    read_fall_axes[0].set_ylabel("stored activation (mV)")
    read_fall_axes[0].set_title("Early read fall perturbs the value while the guard is still open")
    read_fall_axes[0].grid(True, alpha=0.25)
    read_fall_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    read_fall_x = np.arange(len(read_fall_labels))
    read_fall_axes[1].bar(read_fall_x - 0.18, 1e3 * read_fall_preact_samples, width=0.36, label="$z^- - z^+$")
    read_fall_axes[1].bar(read_fall_x + 0.18, 1e3 * read_fall_store_samples, width=0.36, label="stored $h^- - h^+$")
    read_fall_axes[1].axhline(0, color="0.4", linewidth=0.8)
    read_fall_axes[1].set_xticks(read_fall_x)
    read_fall_axes[1].set_xticklabels(read_fall_labels)
    read_fall_axes[1].set_xlabel("read fall start")
    read_fall_axes[1].set_ylabel("sampled differential (mV)")
    read_fall_axes[1].set_title("After the guard is off, read fall no longer changes the stored activation")
    read_fall_axes[1].grid(True, axis="y", alpha=0.25)
    read_fall_axes[1].legend(loc="lower right", ncol=2, fontsize="small")
    read_fall_fig.tight_layout()
    save_plot(read_fall_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_read_fall_ngspice")

    read_rise_fig, read_rise_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, rrt, store, read_ctl in read_rise_traces:
        if label in {"2.72 us", "3.10 us", "3.18 us", "3.28 us"}:
            read_rise_axes[0].plot(1e6 * rrt, 1e3 * store, label=f"read rise {label}")
    read_rise_axes[0].plot(1e6 * read_rise_traces[nominal_read_rise_idx][1], read_rise_traces[nominal_read_rise_idx][3] / 20.0, color="0.5", alpha=0.3, label="$read/20$")
    read_rise_axes[0].axhline(0, color="0.4", linewidth=0.8)
    read_rise_axes[0].axvline(3.18, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard rise")
    read_rise_axes[0].axvline(3.31, color="0.45", linestyle=":", linewidth=0.9, alpha=0.6, label="guard fall")
    read_rise_axes[0].set_xlim(3.05, 3.45)
    read_rise_axes[0].set_ylabel("stored activation (mV)")
    read_rise_axes[0].set_title("Late read rise tests settling before the guard closes")
    read_rise_axes[0].grid(True, alpha=0.25)
    read_rise_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    read_rise_x = np.arange(len(read_rise_labels))
    read_rise_axes[1].bar(read_rise_x - 0.18, 1e3 * read_rise_preact_samples, width=0.36, label="$z^- - z^+$")
    read_rise_axes[1].bar(read_rise_x + 0.18, 1e3 * read_rise_store_samples, width=0.36, label="stored $h^- - h^+$")
    read_rise_axes[1].axhline(0, color="0.4", linewidth=0.8)
    read_rise_axes[1].set_xticks(read_rise_x)
    read_rise_axes[1].set_xticklabels(read_rise_labels)
    read_rise_axes[1].set_xlabel("read rise end")
    read_rise_axes[1].set_ylabel("sampled differential (mV)")
    read_rise_axes[1].set_title("The store only sees the read state available inside the guard window")
    read_rise_axes[1].grid(True, axis="y", alpha=0.25)
    read_rise_axes[1].legend(loc="lower right", ncol=2, fontsize="small")
    read_rise_fig.tight_layout()
    save_plot(read_rise_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_read_rise_ngspice")

    hold_fig, hold_axes = plt.subplots(2, 1, figsize=(7.4, 6.2), gridspec_kw={"height_ratios": [1.0, 0.8]})
    hold_axes[0].plot(1e6 * ht, 1e3 * hold_preact, label="$z^- - z^+$")
    hold_axes[0].plot(1e6 * ht, 1e3 * hold_store, label="stored $h^- - h^+$")
    hold_axes[0].plot(1e6 * ht, hold_cols[22] / 20.0, color="0.35", alpha=0.25, label="$pact/20$")
    hold_axes[0].axhline(0, color="0.4", linewidth=0.8)
    hold_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    hold_axes[0].set_xlim(3.05, 7.55)
    hold_axes[0].set_ylabel("differential (mV)")
    hold_axes[0].set_title("Guarded activation store holds after read-valid closes")
    hold_axes[0].grid(True, alpha=0.25)
    hold_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    hold_sample_labels = [f"{ts * 1e6:.2f}" for ts in hold_sample_times]
    hold_x = np.arange(len(hold_sample_times))
    hold_axes[1].plot(hold_x, 1e6 * hold_drift, "o-", label="drift from first sample")
    hold_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hold_axes[1].set_xticks(hold_x)
    hold_axes[1].set_xticklabels(hold_sample_labels)
    hold_axes[1].set_xlabel("sample time (us)")
    hold_axes[1].set_ylabel("drift ($\\mu$V)")
    hold_axes[1].set_title("No-reset hold drift stays below one microvolt")
    hold_axes[1].grid(True, axis="y", alpha=0.25)
    hold_axes[1].legend(loc="upper right", fontsize="small")
    hold_fig.tight_layout()
    save_plot(hold_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_hold_ngspice")

    guard_off_fig, guard_off_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    guard_off_axes[0].plot(1e6 * got, 1e3 * guard_off_preact, label="$z^- - z^+$")
    guard_off_axes[0].plot(1e6 * got, 1e3 * guard_off_store, label="stored $h^- - h^+$")
    guard_off_axes[0].plot(1e6 * got, guard_off_cols[21] / 20.0, color="0.25", alpha=0.25, label="$read/20$")
    guard_off_axes[0].plot(1e6 * got, guard_off_cols[22] / 20.0, color="0.45", alpha=0.25, label="$pact/20$")
    for start_us, end_us, label in [(5.22, 5.35, "guard-only"), (6.00, 6.13, "guard+read only")]:
        guard_off_axes[0].axvspan(start_us, end_us, color="0.75", alpha=0.18, label=label)
    guard_off_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_off_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="first guard off")
    guard_off_axes[0].set_xlim(3.05, 7.55)
    guard_off_axes[0].set_ylabel("differential / control (mV)")
    guard_off_axes[0].set_title("Guarded activation store rejects later off-state control toggles")
    guard_off_axes[0].grid(True, alpha=0.25)
    guard_off_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    guard_off_sample_labels = [f"{ts * 1e6:.2f}" for ts in guard_off_sample_times]
    guard_off_x = np.arange(len(guard_off_sample_times))
    guard_off_axes[1].plot(guard_off_x, 1e6 * guard_off_drift, "o-", label="stored activation drift")
    guard_off_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_off_axes[1].set_xticks(guard_off_x)
    guard_off_axes[1].set_xticklabels(guard_off_sample_labels)
    guard_off_axes[1].set_xlabel("sample time (us)")
    guard_off_axes[1].set_ylabel("drift ($\\mu$V)")
    guard_off_axes[1].set_title("Pact-only, guard-only, and read pulses leave the held value unchanged")
    guard_off_axes[1].grid(True, axis="y", alpha=0.25)
    guard_off_axes[1].legend(loc="lower right", fontsize="small")
    guard_off_fig.tight_layout()
    save_plot(guard_off_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_off_isolation_ngspice")

    signed_hold_fig, signed_hold_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    signed_hold_axes[0].plot(1e6 * ht, 1e3 * hold_store, label="+ stored $h^- - h^+$")
    signed_hold_axes[0].plot(1e6 * nght, 1e3 * negative_hold_store, "--", label="- stored $h^- - h^+$")
    signed_hold_axes[0].plot(1e6 * ht, 1e3 * hold_preact, color="0.35", alpha=0.35, label="+ $z^- - z^+$")
    signed_hold_axes[0].plot(1e6 * nght, 1e3 * negative_hold_preact, "--", color="0.55", alpha=0.45, label="- $z^- - z^+$")
    signed_hold_axes[0].plot(1e6 * ht, hold_cols[22] / 20.0, color="0.20", alpha=0.20, label="$pact/20$")
    signed_hold_axes[0].axhline(0, color="0.4", linewidth=0.8)
    signed_hold_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    signed_hold_axes[0].set_xlim(3.05, 7.55)
    signed_hold_axes[0].set_ylabel("differential (mV)")
    signed_hold_axes[0].set_title("Guarded activation store mirrors positive and negative signed states")
    signed_hold_axes[0].grid(True, alpha=0.25)
    signed_hold_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    signed_hold_x = np.arange(len(hold_sample_times))
    signed_hold_axes[1].plot(signed_hold_x, 1e6 * hold_drift, "o-", label="+ drift")
    signed_hold_axes[1].plot(signed_hold_x, 1e6 * negative_hold_drift, "s--", label="- drift")
    signed_hold_axes[1].plot(signed_hold_x, 1e3 * signed_hold_mirror_error, "d:", label="mirror error")
    signed_hold_axes[1].axhline(0, color="0.4", linewidth=0.8)
    signed_hold_axes[1].set_xticks(signed_hold_x)
    signed_hold_axes[1].set_xticklabels(hold_sample_labels)
    signed_hold_axes[1].set_xlabel("sample time (us)")
    signed_hold_axes[1].set_ylabel("$\\mu$V / mV")
    signed_hold_axes[1].set_title("Both signs hold; mirror error stays below the assertion bound")
    signed_hold_axes[1].grid(True, axis="y", alpha=0.25)
    signed_hold_axes[1].legend(loc="upper right", ncol=3, fontsize="small")
    signed_hold_fig.tight_layout()
    save_plot(signed_hold_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_signed_hold_ngspice")

    guard_corner_fig, guard_corner_axes = plt.subplots(2, 1, figsize=(7.4, 6.2), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, ct, store in guard_corner_traces:
        guard_corner_axes[0].plot(1e6 * ct, 1e3 * store, label=label)
    guard_corner_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_corner_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    guard_corner_axes[0].set_xlim(3.15, 3.38)
    guard_corner_axes[0].set_ylabel("stored activation (mV)")
    guard_corner_axes[0].set_title("Guarded store remains signed correctly across pass-gate corners")
    guard_corner_axes[0].grid(True, alpha=0.25)
    guard_corner_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    guard_corner_x = np.arange(len(guard_corner_cases))
    guard_corner_labels = [label for _name, _n_vto, _p_vto, label in guard_corner_cases]
    guard_corner_axes[1].bar(guard_corner_x - 0.18, 1e3 * guard_corner_preact_samples, width=0.36, label="$z^- - z^+$")
    guard_corner_axes[1].bar(guard_corner_x + 0.18, 1e3 * guard_corner_store_samples, width=0.36, label="stored $h^- - h^+$")
    guard_corner_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_corner_axes[1].set_xticks(guard_corner_x)
    guard_corner_axes[1].set_xticklabels(guard_corner_labels)
    guard_corner_axes[1].set_ylabel("sampled differential (mV)")
    guard_corner_axes[1].set_title("Correct guard-device corners span about 0.62 mV")
    guard_corner_axes[1].grid(True, axis="y", alpha=0.25)
    guard_corner_axes[1].legend(loc="lower right", ncol=2, fontsize="small")
    guard_corner_fig.tight_layout()
    save_plot(guard_corner_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_corner_ngspice")

    guard_size_fig, guard_size_axes = plt.subplots(2, 1, figsize=(7.4, 6.2), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, st, store in guard_size_traces:
        guard_size_axes[0].plot(1e6 * st, 1e3 * store, label=label)
    guard_size_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_size_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    guard_size_axes[0].set_xlim(3.15, 3.38)
    guard_size_axes[0].set_ylabel("stored activation (mV)")
    guard_size_axes[0].set_title("Guard TG width sweep after the timing fix")
    guard_size_axes[0].grid(True, alpha=0.25)
    guard_size_axes[0].legend(loc="upper left", ncol=3, fontsize="small")
    guard_size_x = np.arange(len(guard_size_labels))
    guard_size_axes[1].bar(guard_size_x - 0.18, 1e3 * guard_size_preact_samples, width=0.36, label="$z^- - z^+$")
    guard_size_axes[1].bar(guard_size_x + 0.18, 1e3 * guard_size_store_samples, width=0.36, label="stored $h^- - h^+$")
    guard_size_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_size_axes[1].set_xticks(guard_size_x)
    guard_size_axes[1].set_xticklabels(guard_size_labels)
    guard_size_axes[1].set_xlabel("guard TG width scale")
    guard_size_axes[1].set_ylabel("sampled differential (mV)")
    guard_size_axes[1].set_title("The width knee tells where the guard switch stops being oversized")
    guard_size_axes[1].grid(True, axis="y", alpha=0.25)
    guard_size_axes[1].legend(loc="lower right", ncol=2, fontsize="small")
    guard_size_fig.tight_layout()
    save_plot(guard_size_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_size_ngspice")

    guard_cstore_fig, guard_cstore_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, cst, store in guard_cstore_traces:
        if label in {"2 pF", "10 pF", "50 pF", "100 pF"}:
            guard_cstore_axes[0].plot(1e6 * cst, 1e3 * store, label=label)
    guard_cstore_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_cstore_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    guard_cstore_axes[0].set_xlim(3.05, 3.65)
    guard_cstore_axes[0].set_ylabel("stored activation (mV)")
    guard_cstore_axes[0].set_title("Small caps overshoot; large caps undercharge in the fixed guard window")
    guard_cstore_axes[0].grid(True, alpha=0.25)
    guard_cstore_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    guard_cstore_x = np.arange(len(guard_cstore_labels))
    guard_cstore_axes[1].bar(guard_cstore_x - 0.18, 1e3 * guard_cstore_preact_samples, width=0.36, label="$z^- - z^+$")
    guard_cstore_axes[1].bar(guard_cstore_x + 0.18, 1e3 * guard_cstore_store_samples, width=0.36, label="stored $h^- - h^+$")
    guard_cstore_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_cstore_axes[1].set_xticks(guard_cstore_x)
    guard_cstore_axes[1].set_xticklabels(guard_cstore_labels)
    guard_cstore_axes[1].set_xlabel("activation store capacitance")
    guard_cstore_axes[1].set_ylabel("sampled differential (mV)")
    guard_cstore_axes[1].set_title("Store capacitance sets both feedthrough sensitivity and charge time")
    guard_cstore_axes[1].grid(True, axis="y", alpha=0.25)
    guard_cstore_axes[1].legend(loc="lower left", ncol=2, fontsize="small")
    guard_cstore_fig.tight_layout()
    save_plot(guard_cstore_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_cstore_ngspice")

    guard_control_swing_fig, guard_control_swing_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, swt, store in guard_control_swing_traces:
        if label in {"1.8 V", "1.4 V", "1.2 V", "1.0 V"}:
            guard_control_swing_axes[0].plot(1e6 * swt, 1e3 * store, label=label)
    guard_control_swing_axes[0].axhline(0, color="0.4", linewidth=0.8)
    guard_control_swing_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    guard_control_swing_axes[0].set_xlim(3.05, 3.65)
    guard_control_swing_axes[0].set_ylabel("stored activation (mV)")
    guard_control_swing_axes[0].set_title("Reduced pact/guard swing tests phase-driver headroom")
    guard_control_swing_axes[0].grid(True, alpha=0.25)
    guard_control_swing_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    guard_control_swing_x = np.arange(len(guard_control_swing_labels))
    guard_control_swing_axes[1].bar(guard_control_swing_x - 0.18, 1e3 * guard_control_swing_preact_samples, width=0.36, label="$z^- - z^+$")
    guard_control_swing_axes[1].bar(guard_control_swing_x + 0.18, 1e3 * guard_control_swing_store_samples, width=0.36, label="stored $h^- - h^+$")
    guard_control_swing_axes[1].axhline(0, color="0.4", linewidth=0.8)
    guard_control_swing_axes[1].set_xticks(guard_control_swing_x)
    guard_control_swing_axes[1].set_xticklabels(guard_control_swing_labels)
    guard_control_swing_axes[1].set_xlabel("pact/guard control swing")
    guard_control_swing_axes[1].set_ylabel("sampled differential (mV)")
    guard_control_swing_axes[1].set_title("Moderate loss is tolerated; marginal swing exposes feedthrough/failure")
    guard_control_swing_axes[1].grid(True, axis="y", alpha=0.25)
    guard_control_swing_axes[1].legend(loc="lower left", ncol=2, fontsize="small")
    guard_control_swing_fig.tight_layout()
    save_plot(guard_control_swing_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_control_swing_ngspice")

    read_drive_fig, read_drive_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, rdt, _preact, store, read_ctl in read_drive_traces:
        if label in {"0.75 V", "1.05 V", "1.15 V", "1.45 V"}:
            read_drive_axes[0].plot(1e6 * rdt, 1e3 * store, label=label)
    read_drive_axes[0].plot(1e6 * read_drive_traces[nominal_read_drive_idx][1], read_drive_traces[nominal_read_drive_idx][4] / 20.0, color="0.45", alpha=0.25, label="$read/20$")
    read_drive_axes[0].axhline(0, color="0.4", linewidth=0.8)
    read_drive_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    read_drive_axes[0].set_xlim(3.05, 3.65)
    read_drive_axes[0].set_ylabel("stored activation (mV)")
    read_drive_axes[0].set_title("Read-valid drive sets preactivation strength before the guard closes")
    read_drive_axes[0].grid(True, alpha=0.25)
    read_drive_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    read_drive_x = np.arange(len(read_drive_labels))
    read_drive_axes[1].bar(read_drive_x - 0.18, 1e3 * read_drive_preact_samples, width=0.36, label="$z^- - z^+$")
    read_drive_axes[1].bar(read_drive_x + 0.18, 1e3 * read_drive_store_samples, width=0.36, label="stored $h^- - h^+$")
    read_drive_axes[1].axhline(0, color="0.4", linewidth=0.8)
    read_drive_axes[1].set_xticks(read_drive_x)
    read_drive_axes[1].set_xticklabels(read_drive_labels)
    read_drive_axes[1].set_xlabel("read-valid gate drive")
    read_drive_axes[1].set_ylabel("sampled differential (mV)")
    read_drive_axes[1].set_title("Weak read drive is a real signal-path limiter, not a store-switch issue")
    read_drive_axes[1].grid(True, axis="y", alpha=0.25)
    read_drive_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    read_drive_fig.tight_layout()
    save_plot(read_drive_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_read_drive_ngspice")

    read_width_fig, read_width_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, rwt, store in read_width_traces:
        if label in {"6u", "24u", "96u", "192u"}:
            read_width_axes[0].plot(1e6 * rwt, 1e3 * store, label=label)
    read_width_axes[0].axhline(0, color="0.4", linewidth=0.8)
    read_width_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    read_width_axes[0].set_xlim(3.05, 3.65)
    read_width_axes[0].set_ylabel("stored activation (mV)")
    read_width_axes[0].set_title("Read-tail width sets the same headroom as read-valid drive")
    read_width_axes[0].grid(True, alpha=0.25)
    read_width_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    read_width_x = np.arange(len(read_width_labels))
    read_width_axes[1].bar(read_width_x - 0.18, 1e3 * read_width_preact_samples, width=0.36, label="$z^- - z^+$")
    read_width_axes[1].bar(read_width_x + 0.18, 1e3 * read_width_store_samples, width=0.36, label="stored $h^- - h^+$")
    read_width_axes[1].axhline(0, color="0.4", linewidth=0.8)
    read_width_axes[1].set_xticks(read_width_x)
    read_width_axes[1].set_xticklabels(read_width_labels)
    read_width_axes[1].set_xlabel("read-tail NMOS width")
    read_width_axes[1].set_ylabel("sampled differential (mV)")
    read_width_axes[1].set_title("Sizing helps until the forward/store path saturates")
    read_width_axes[1].grid(True, axis="y", alpha=0.25)
    read_width_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    read_width_fig.tight_layout()
    save_plot(read_width_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_read_width_ngspice")

    forward_pair_fig, forward_pair_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, fpt, load, store in forward_pair_traces:
        if label in {"6u", "48u", "96u", "192u"}:
            forward_pair_axes[0].plot(1e6 * fpt, 1e3 * load, label=f"{label} load")
            forward_pair_axes[0].plot(1e6 * fpt, 1e3 * store, "--", label=f"{label} store")
    forward_pair_axes[0].axhline(0, color="0.4", linewidth=0.8)
    forward_pair_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    forward_pair_axes[0].set_xlim(3.05, 3.65)
    forward_pair_axes[0].set_ylabel("activation differential (mV)")
    forward_pair_axes[0].set_title("Forward-pair width is a real activation-gain knob")
    forward_pair_axes[0].grid(True, alpha=0.25)
    forward_pair_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    forward_pair_x = np.arange(len(forward_pair_labels))
    forward_pair_axes[1].bar(forward_pair_x - 0.24, 1e3 * forward_pair_preact_samples, width=0.24, label="$z^- - z^+$")
    forward_pair_axes[1].bar(forward_pair_x, 1e3 * forward_pair_load_samples, width=0.24, label="forward load")
    forward_pair_axes[1].bar(forward_pair_x + 0.24, 1e3 * forward_pair_store_samples, width=0.24, label="stored $h^- - h^+$")
    forward_pair_axes[1].axhline(0, color="0.4", linewidth=0.8)
    forward_pair_axes[1].set_xticks(forward_pair_x)
    forward_pair_axes[1].set_xticklabels(forward_pair_labels)
    forward_pair_axes[1].set_xlabel("forward-pair/tail NMOS width")
    forward_pair_axes[1].set_ylabel("sampled differential (mV)")
    forward_pair_axes[1].set_title("Read state is fixed while activation gain keeps increasing")
    forward_pair_axes[1].grid(True, axis="y", alpha=0.25)
    forward_pair_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    forward_pair_fig.tight_layout()
    save_plot(forward_pair_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_ngspice")

    high_gain_fig, high_gain_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    high_gain_axes[0].plot(1e6 * hght, 1e3 * high_gain_hold_store, label="+ 96u hold")
    high_gain_axes[0].plot(1e6 * hgnht, 1e3 * high_gain_negative_hold_store, "--", label="- 96u hold")
    high_gain_axes[0].plot(1e6 * hgot, 1e3 * high_gain_guard_off_store, ":", label="+ 96u off-isolation")
    high_gain_axes[0].plot(1e6 * hght, 1e3 * high_gain_hold_preact, color="0.35", alpha=0.30, label="+ $z^- - z^+$")
    high_gain_axes[0].axhline(0, color="0.4", linewidth=0.8)
    high_gain_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    high_gain_axes[0].set_xlim(3.05, 7.55)
    high_gain_axes[0].set_ylabel("differential (mV)")
    high_gain_axes[0].set_title("96u forward pair keeps the larger stored activation in hold")
    high_gain_axes[0].grid(True, alpha=0.25)
    high_gain_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    high_gain_drift_labels = ["+ hold", "- hold", "off stress"]
    high_gain_drift_values = np.array(
        [
            np.max(np.abs(high_gain_hold_drift)),
            np.max(np.abs(high_gain_negative_hold_drift)),
            np.max(np.abs(high_gain_guard_off_drift)),
        ]
    )
    high_gain_x = np.arange(len(high_gain_drift_labels))
    high_gain_axes[1].bar(high_gain_x, 1e6 * high_gain_drift_values)
    high_gain_axes[1].set_xticks(high_gain_x)
    high_gain_axes[1].set_xticklabels(high_gain_drift_labels)
    high_gain_axes[1].set_ylabel("max held-value drift ($\\mu$V)")
    high_gain_axes[1].set_ylim(0.0, max(1.0, 1.25 * 1e6 * float(np.max(high_gain_drift_values))))
    high_gain_axes[1].set_title("Sign symmetry and later control activity stay at sub-microvolt drift")
    high_gain_axes[1].grid(True, axis="y", alpha=0.25)
    high_gain_fig.tight_layout()
    save_plot(high_gain_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_robustness_ngspice")

    high_gain_zcm_fig, high_gain_zcm_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, zcmt, load, store in high_gain_zcm_traces:
        if label in {"0.75 V", "0.90 V", "1.05 V"}:
            high_gain_zcm_axes[0].plot(1e6 * zcmt, 1e3 * load, label=f"{label} load")
            high_gain_zcm_axes[0].plot(1e6 * zcmt, 1e3 * store, "--", label=f"{label} store")
    high_gain_zcm_axes[0].axhline(0, color="0.4", linewidth=0.8)
    high_gain_zcm_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    high_gain_zcm_axes[0].set_xlim(3.05, 3.65)
    high_gain_zcm_axes[0].set_ylabel("activation differential (mV)")
    high_gain_zcm_axes[0].set_title("96u forward pair exposes a lower summing-common-mode headroom limit")
    high_gain_zcm_axes[0].grid(True, alpha=0.25)
    high_gain_zcm_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    high_gain_zcm_x = np.arange(len(high_gain_zcm_labels))
    high_gain_zcm_axes[1].bar(high_gain_zcm_x - 0.24, 1e3 * high_gain_zcm_preact_samples, width=0.24, label="$z^- - z^+$")
    high_gain_zcm_axes[1].bar(high_gain_zcm_x, 1e3 * high_gain_zcm_load_samples, width=0.24, label="forward load")
    high_gain_zcm_axes[1].bar(high_gain_zcm_x + 0.24, 1e3 * high_gain_zcm_store_samples, width=0.24, label="stored $h^- - h^+$")
    high_gain_zcm_axes[1].axhline(0, color="0.4", linewidth=0.8)
    high_gain_zcm_axes[1].set_xticks(high_gain_zcm_x)
    high_gain_zcm_axes[1].set_xticklabels(high_gain_zcm_labels)
    high_gain_zcm_axes[1].set_xlabel("initial/reset summing common mode")
    high_gain_zcm_axes[1].set_ylabel("sampled differential (mV)")
    high_gain_zcm_axes[1].set_title("The larger forward pair needs roughly nominal or higher z common mode")
    high_gain_zcm_axes[1].grid(True, axis="y", alpha=0.25)
    high_gain_zcm_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    high_gain_zcm_fig.tight_layout()
    save_plot(high_gain_zcm_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_zcm_ngspice")

    tail_rebias_fig, tail_rebias_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.9]})
    for label, rbt, store in high_gain_tail_rebias_traces:
        tail_rebias_axes[0].plot(1e6 * rbt, 1e3 * store, label=label)
    tail_rebias_axes[0].axhline(0, color="0.4", linewidth=0.8)
    tail_rebias_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    tail_rebias_axes[0].set_xlim(3.05, 3.65)
    tail_rebias_axes[0].set_ylim(-4, 82)
    tail_rebias_axes[0].set_ylabel("stored activation (mV)")
    tail_rebias_axes[0].set_title("Stored activation shows the same low-common-mode limit")
    tail_rebias_axes[0].grid(True, alpha=0.25)
    tail_rebias_axes[0].legend(loc="upper left", ncol=2, fontsize="x-small")
    tail_rebias_x = np.arange(len(high_gain_tail_rebias_labels))
    for zi, zcm_v in enumerate(high_gain_tail_rebias_zcm_cases_v):
        tail_rebias_axes[1].plot(
            tail_rebias_x,
            1e3 * high_gain_tail_rebias_store[zi],
            marker="o",
            label=f"zcm {zcm_v:.2f} V",
        )
    tail_rebias_axes[1].axhline(0, color="0.4", linewidth=0.8)
    tail_rebias_axes[1].set_xticks(tail_rebias_x)
    tail_rebias_axes[1].set_xticklabels(high_gain_tail_rebias_labels)
    tail_rebias_axes[1].set_xlabel("private 96u forward-tail bias")
    tail_rebias_axes[1].set_ylabel("stored activation (mV)")
    tail_rebias_axes[1].set_title("0.75 V stays failed; 0.85 V stays partial; too-low tail bias kills nominal")
    tail_rebias_axes[1].grid(True, alpha=0.25)
    tail_rebias_axes[1].legend(loc="upper right", fontsize="small")
    tail_rebias_fig.tight_layout()
    save_plot(tail_rebias_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_tail_rebias_ngspice")

    shift_fig, shift_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.85]})
    unshifted_075 = next((trace for trace in high_gain_zcm_traces if trace[0] == "0.75 V"), None)
    if unshifted_075 is not None:
        _label, ust, _load, ustore = unshifted_075
        shift_axes[0].plot(1e6 * ust, 1e3 * ustore, "k:", label="unshifted 0.75 V")
    for label, szct, store in chosen_shift_zcm_traces:
        shift_axes[0].plot(1e6 * szct, 1e3 * store, label=label)
    shift_axes[0].plot(1e6 * positive_shift_t, 1e3 * positive_shift_store, "--", color="tab:blue", label="+ signed hold")
    shift_axes[0].plot(1e6 * nsht, 1e3 * negative_shift_store, "--", color="tab:red", label="- signed hold")
    shift_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    shift_axes[0].set_xlim(3.05, 3.65)
    shift_axes[0].set_ylabel("stored activation (mV)")
    shift_axes[0].set_title("Passive shifted-gate latch restores low-z headroom")
    shift_axes[0].grid(True, alpha=0.25)
    shift_axes[0].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_x = np.arange(len(high_gain_shift_couple_labels) + 1)
    shift_labels = ["none", *high_gain_shift_couple_labels]
    shift_values = np.array([high_gain_zcm_store_samples[0], *high_gain_shift_store_samples])
    shift_axes[1].plot(shift_x, 1e3 * shift_values, marker="o", label="zcm 0.75 V")
    shift_axes[1].axhline(1e3 * high_gain_zcm_store_samples[nominal_zcm_idx], color="0.35", linestyle="--", linewidth=0.9, label="unshifted 0.90 V")
    shift_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_axes[1].set_xticks(shift_x)
    shift_axes[1].set_xticklabels(shift_labels)
    shift_axes[1].set_xlabel("coupling capacitance into 5 pF shifted gate latch")
    shift_axes[1].set_ylabel("stored activation (mV)")
    shift_axes[1].set_title("The useful coupling ratio is a window, not monotone gain")
    shift_axes[1].grid(True, alpha=0.25)
    shift_axes[1].legend(loc="upper right", fontsize="small")
    shift_fig.tight_layout()
    save_plot(shift_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_ngspice")

    shift_stress_fig, shift_stress_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.4),
        gridspec_kw={"height_ratios": [1.0, 0.8]},
    )
    shift_stress_axes[0].plot(1e6 * ssht, 1e3 * shift_stress_preact, label="$z^- - z^+$")
    shift_stress_axes[0].plot(1e6 * ssht, 1e3 * shift_stress_store, label="stored shifted $h^- - h^+$")
    shift_stress_axes[0].plot(1e6 * ssht, shift_stress_cols[21] / 20.0, color="0.25", alpha=0.25, label="$read/20$")
    shift_stress_axes[0].plot(1e6 * ssht, shift_stress_cols[22] / 20.0, color="0.45", alpha=0.25, label="$pact/20$")
    for start_us, end_us, label in [(5.22, 5.35, "guard-only"), (6.00, 6.13, "guard+read only")]:
        shift_stress_axes[0].axvspan(start_us, end_us, color="0.75", alpha=0.18, label=label)
    shift_stress_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_stress_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="first guard off")
    shift_stress_axes[0].set_xlim(3.05, 7.55)
    shift_stress_axes[0].set_ylabel("differential / control (mV)")
    shift_stress_axes[0].set_title("Shifted-gate latch holds through later off-state stress")
    shift_stress_axes[0].grid(True, alpha=0.25)
    shift_stress_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    shift_stress_x = np.arange(len(guard_off_sample_times))
    shift_stress_axes[1].plot(shift_stress_x, 1e6 * shift_stress_drift, "o-", label="stored activation drift")
    shift_stress_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_stress_axes[1].set_xticks(shift_stress_x)
    shift_stress_axes[1].set_xticklabels(guard_off_sample_labels)
    shift_stress_axes[1].set_xlabel("sample time (us)")
    shift_stress_axes[1].set_ylabel("drift ($\\mu$V)")
    shift_stress_axes[1].set_title("Pact-only, guard-only, and read pulses do not disturb the shifted store")
    shift_stress_axes[1].grid(True, axis="y", alpha=0.25)
    shift_stress_axes[1].legend(loc="upper right", fontsize="small")
    shift_stress_fig.tight_layout()
    save_plot(
        shift_stress_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_off_isolation_ngspice",
    )

    shift_reset_fig, shift_reset_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.4),
        gridspec_kw={"height_ratios": [1.0, 0.8]},
    )
    shift_reset_axes[0].plot(1e6 * sbt, 1e3 * shift_bad_gate_diff, ":", label="unreset gate diff")
    shift_reset_axes[0].plot(1e6 * srt, 1e3 * shift_reset_gate_diff, label="reset gate diff")
    shift_reset_axes[0].plot(1e6 * srt, 1e3 * (shift_reset_gate_common - 0.90), "--", label="reset gate common error")
    shift_reset_axes[0].plot(1e6 * srt, shift_reset_cols[20] / 20.0, color="0.45", alpha=0.25, label="$reset/20$")
    shift_reset_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_reset_axes[0].axvspan(2.50, 2.62, color="0.75", alpha=0.18, label="reset high")
    shift_reset_axes[0].axvline(2.72, color="0.45", linestyle="--", linewidth=0.9, alpha=0.6, label="read starts")
    shift_reset_axes[0].set_xlim(2.35, 3.75)
    shift_reset_axes[0].set_ylabel("shifted-gate error (mV)")
    shift_reset_axes[0].set_title("Physical shifted-gate reset removes bad initial gate differential")
    shift_reset_axes[0].grid(True, alpha=0.25)
    shift_reset_axes[0].legend(loc="lower left", ncol=2, fontsize="x-small")
    shift_reset_axes[1].plot(1e6 * sbt, 1e3 * shift_bad_store, ":", label="unreset bad init")
    shift_reset_axes[1].plot(1e6 * srt, 1e3 * shift_reset_store, label="with physical reset")
    shift_reset_axes[1].plot(1e6 * positive_shift_t, 1e3 * positive_shift_store, "--", label="nominal shifted")
    shift_reset_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_reset_axes[1].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    shift_reset_axes[1].set_xlim(3.05, 3.75)
    shift_reset_axes[1].set_ylim(-10, max(390, 1e3 * shift_bad_store_sample * 1.05))
    shift_reset_axes[1].set_xlabel("time (us)")
    shift_reset_axes[1].set_ylabel("stored activation (mV)")
    shift_reset_axes[1].set_title("Reset prevents the bad-initial-state overdrive path")
    shift_reset_axes[1].grid(True, alpha=0.25)
    shift_reset_axes[1].legend(loc="upper right", fontsize="small")
    shift_reset_fig.tight_layout()
    save_plot(
        shift_reset_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_feedthrough_ngspice",
    )

    shift_reset_size_fig, shift_reset_size_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.85, 0.85, 1.0]},
    )
    reset_size_x = np.arange(len(shift_reset_size_labels))
    shift_reset_size_axes[0].plot(
        reset_size_x,
        1e3 * shift_reset_size_gate_residue,
        "o-",
        label="max post-reset gate diff",
    )
    shift_reset_size_axes[0].plot(
        reset_size_x,
        1e3 * shift_reset_size_common_error,
        "s--",
        label="common-mode error",
    )
    shift_reset_size_axes[0].axhline(2.0, color="0.45", linestyle=":", linewidth=0.9, label="2 mV residue")
    shift_reset_size_axes[0].set_xticks(reset_size_x)
    shift_reset_size_axes[0].set_xticklabels(shift_reset_size_labels)
    shift_reset_size_axes[0].set_ylabel("reset error (mV)")
    shift_reset_size_axes[0].set_title("Shifted-gate reset strength clears bad initial gate state")
    shift_reset_size_axes[0].set_yscale("log")
    shift_reset_size_axes[0].grid(True, axis="y", alpha=0.25)
    shift_reset_size_axes[0].legend(loc="upper right", fontsize="small")
    shift_reset_size_axes[1].bar(
        reset_size_x - 0.18,
        1e3 * shift_reset_size_store_samples,
        width=0.36,
        label="with reset",
    )
    shift_reset_size_axes[1].bar(
        reset_size_x + 0.18,
        1e3 * shift_reset_size_feedthrough,
        width=0.36,
        label="offset from initialized",
    )
    shift_reset_size_axes[1].axhline(1e3 * positive_shift_store_samples[0], color="0.35", linestyle="--", linewidth=0.9)
    shift_reset_size_axes[1].axhspan(45, 80, color="0.7", alpha=0.12, label="useful bounded window")
    shift_reset_size_axes[1].set_xticks(reset_size_x)
    shift_reset_size_axes[1].set_xticklabels(shift_reset_size_labels)
    shift_reset_size_axes[1].set_ylabel("stored $h$ / offset (mV)")
    shift_reset_size_axes[1].set_title("Reset feedthrough is a small sizing-dependent activation offset")
    shift_reset_size_axes[1].grid(True, axis="y", alpha=0.25)
    shift_reset_size_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    for label, rst, gate_diff, _gate_common, store in shift_reset_size_traces:
        if label in {"0.02x", "0.05x", "0.10x", "1.00x"}:
            shift_reset_size_axes[2].plot(1e6 * rst, 1e3 * store, label=f"{label} $h$")
            shift_reset_size_axes[2].plot(1e6 * rst, 1e3 * gate_diff, "--", alpha=0.55, label=f"{label} gate")
    shift_reset_size_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_reset_size_axes[2].axvspan(2.50, 2.62, color="0.75", alpha=0.16, label="reset high")
    shift_reset_size_axes[2].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    shift_reset_size_axes[2].set_xlim(2.35, 3.75)
    shift_reset_size_axes[2].set_xlabel("time (us)")
    shift_reset_size_axes[2].set_ylabel("differential (mV)")
    shift_reset_size_axes[2].set_title("Time traces show weak-reset residue versus usable reset sizes")
    shift_reset_size_axes[2].grid(True, alpha=0.25)
    shift_reset_size_axes[2].legend(loc="upper left", ncol=2, fontsize="xx-small")
    shift_reset_size_fig.tight_layout()
    save_plot(
        shift_reset_size_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_feedthrough_size_ngspice",
    )

    shift_reset_common_fig, shift_reset_common_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.2),
        gridspec_kw={"height_ratios": [0.95, 0.8, 0.9]},
    )
    shift_reset_common_axes[0].plot(
        shift_reset_common_values,
        1e3 * shift_reset_common_store_samples,
        "o-",
        label="physical reset",
    )
    shift_reset_common_axes[0].axhline(
        1e3 * positive_shift_store_samples[0],
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        label="initialized-gate result",
    )
    shift_reset_common_axes[0].axhspan(45, 80, color="0.7", alpha=0.12, label="useful bounded window")
    shift_reset_common_axes[0].axvline(0.80, color="tab:green", linestyle=":", linewidth=1.1, label="tuned 0.80 V")
    shift_reset_common_axes[0].axvline(0.90, color="tab:red", linestyle=":", linewidth=1.1, label="old 0.90 V")
    shift_reset_common_axes[0].set_ylabel("stored $h$ (mV)")
    shift_reset_common_axes[0].set_title("Reset common-mode tunes out shifted-gate activation feedthrough")
    shift_reset_common_axes[0].grid(True, alpha=0.25)
    shift_reset_common_axes[0].legend(loc="lower right", ncol=2, fontsize="x-small")
    shift_reset_common_axes[1].plot(
        shift_reset_common_values,
        shift_reset_common_gate_pre,
        "o-",
        label="post-reset gate common",
    )
    shift_reset_common_axes[1].plot(
        shift_reset_common_values,
        shift_reset_common_gate_read,
        "s--",
        label="read-time gate common",
    )
    shift_reset_common_axes[1].set_ylabel("gate common (V)")
    shift_reset_common_axes[1].set_title("Read coupling preserves the common-mode offset set by reset")
    shift_reset_common_axes[1].grid(True, alpha=0.25)
    shift_reset_common_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    reuse_cycle_x = np.arange(1, len(shift_reuse_h_samples) + 1)
    shift_reset_common_axes[2].plot(
        reuse_cycle_x,
        1e3 * shift_reuse_h_samples,
        "o-",
        label="0.90 V reset common",
    )
    shift_reset_common_axes[2].plot(
        reuse_cycle_x,
        1e3 * shift_reuse_common_h_samples,
        "s-",
        label="0.80 V reset common",
    )
    shift_reset_common_axes[2].set_xticks(reuse_cycle_x)
    shift_reset_common_axes[2].set_xlabel("read/store cycle")
    shift_reset_common_axes[2].set_ylabel("stored $h$ (mV)")
    shift_reset_common_axes[2].set_title("Tuned reset common removes the repeated-reset bump")
    shift_reset_common_axes[2].grid(True, alpha=0.25)
    shift_reset_common_axes[2].legend(loc="upper right", fontsize="small")
    shift_reset_common_fig.tight_layout()
    save_plot(
        shift_reset_common_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_common_ngspice",
    )

    shift_reuse_fig, shift_reuse_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.9]},
    )
    shift_reuse_axes[0].plot(1e6 * shrct, 1e3 * shift_reuse_common_gate_diff, label="shifted-gate diff")
    shift_reuse_axes[0].plot(
        1e6 * shrct,
        1e3 * (shift_reuse_common_gate_common - 0.80),
        "--",
        label="gate common - 0.80 V",
    )
    shift_reuse_axes[0].plot(
        1e6 * shrct,
        1e3 * shift_reuse_common_cols[20] / 20.0,
        color="0.45",
        alpha=0.25,
        label="$reset/20$",
    )
    for start_us, end_us, label in [(3.60, 4.00, "reset 1"), (5.20, 5.60, "reset 2")]:
        shift_reuse_axes[0].axvspan(start_us, end_us, color="0.75", alpha=0.16, label=label)
    shift_reuse_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_reuse_axes[0].set_xlim(2.9, 7.1)
    shift_reuse_axes[0].set_ylabel("gate diff / common offset (mV)")
    shift_reuse_axes[0].set_title("Physical resets restore the shifted-gate state between repeated reads")
    shift_reuse_axes[0].grid(True, alpha=0.25)
    shift_reuse_axes[0].legend(loc="lower left", ncol=2, fontsize="small")
    shift_reuse_axes[1].plot(1e6 * shrct, 1e3 * shift_reuse_common_preact, label="$z^- - z^+$")
    shift_reuse_axes[1].plot(1e6 * shrct, 1e3 * shift_reuse_common_load, label="forward load")
    shift_reuse_axes[1].plot(1e6 * shrct, 1e3 * shift_reuse_common_store, label="stored activation")
    shift_reuse_axes[1].plot(
        1e6 * shrct,
        1e3 * shift_reuse_common_cols[23] / 20.0,
        color="0.35",
        alpha=0.25,
        label="$read/20$",
    )
    shift_reuse_axes[1].plot(
        1e6 * shrct,
        1e3 * shift_reuse_common_cols[24] / 20.0,
        color="0.15",
        alpha=0.25,
        label="$pact/20$",
    )
    shift_reuse_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_reuse_axes[1].set_xlim(2.9, 7.1)
    shift_reuse_axes[1].set_ylabel("differential (mV)")
    shift_reuse_axes[1].set_title("Same MOS-written W/B state drives three low-common-mode reads")
    shift_reuse_axes[1].grid(True, alpha=0.25)
    shift_reuse_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    shift_reuse_x = np.arange(len(shift_reuse_common_h_samples))
    shift_reuse_axes[2].bar(shift_reuse_x - 0.18, 1e3 * shift_reuse_common_z_samples, width=0.36, label="$z$ sample")
    shift_reuse_axes[2].bar(shift_reuse_x + 0.18, 1e3 * shift_reuse_common_h_samples, width=0.36, label="$h$ sample")
    shift_reuse_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_reuse_axes[2].set_xticks(shift_reuse_x)
    shift_reuse_axes[2].set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    shift_reuse_axes[2].set_xlabel("read/reset/store cycle")
    shift_reuse_axes[2].set_ylabel("sampled differential (mV)")
    shift_reuse_axes[2].set_title("Repeated shifted-gate captures remain useful after physical reset")
    shift_reuse_axes[2].grid(True, axis="y", alpha=0.25)
    shift_reuse_axes[2].legend(loc="upper right", fontsize="small")
    shift_reuse_fig.tight_layout()
    save_plot(
        shift_reuse_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_repeated_reset_reuse_ngspice",
    )

    shift_reset_ref_perturb_fig, shift_reset_ref_perturb_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.4),
        gridspec_kw={"height_ratios": [1.0, 0.85]},
    )
    for label, rpt, _gate_diff, _load, store in shift_reset_ref_perturb_traces:
        if label in {"nominal", "helpful diff -10 mV", "destructive diff +10 mV", "destructive diff +20 mV"}:
            shift_reset_ref_perturb_axes[0].plot(1e6 * rpt, 1e3 * store, label=label)
    shift_reset_ref_perturb_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_reset_ref_perturb_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    shift_reset_ref_perturb_axes[0].set_xlim(3.05, 3.75)
    shift_reset_ref_perturb_axes[0].set_ylabel("stored activation (mV)")
    shift_reset_ref_perturb_axes[0].set_title("Physical reset-reference differential error changes shifted-gate margin")
    shift_reset_ref_perturb_axes[0].grid(True, alpha=0.25)
    shift_reset_ref_perturb_axes[0].legend(loc="upper right", fontsize="small")
    shift_reset_ref_perturb_x = np.arange(len(shift_reset_ref_perturb_labels))
    shift_reset_ref_perturb_axes[1].bar(
        shift_reset_ref_perturb_x - 0.24,
        1e3 * shift_reset_ref_perturb_gate_post,
        width=0.24,
        label="post-reset gate diff",
    )
    shift_reset_ref_perturb_axes[1].bar(
        shift_reset_ref_perturb_x,
        1e3 * shift_reset_ref_perturb_load_samples,
        width=0.24,
        label="forward load",
    )
    shift_reset_ref_perturb_axes[1].bar(
        shift_reset_ref_perturb_x + 0.24,
        1e3 * shift_reset_ref_perturb_store_samples,
        width=0.24,
        label="stored $h$",
    )
    shift_reset_ref_perturb_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_reset_ref_perturb_axes[1].set_xticks(shift_reset_ref_perturb_x)
    shift_reset_ref_perturb_axes[1].set_xticklabels(shift_reset_ref_perturb_labels, rotation=18, ha="right")
    shift_reset_ref_perturb_axes[1].set_ylabel("sampled differential (mV)")
    shift_reset_ref_perturb_axes[1].set_title("Common reset-reference errors are mild; differential errors consume margin")
    shift_reset_ref_perturb_axes[1].grid(True, axis="y", alpha=0.25)
    shift_reset_ref_perturb_axes[1].legend(loc="upper right", ncol=3, fontsize="small")
    shift_reset_ref_perturb_fig.tight_layout()
    save_plot(
        shift_reset_ref_perturb_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_refpert080_ngspice",
    )

    shift_noise_fig, shift_noise_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.4),
        gridspec_kw={"height_ratios": [1.0, 0.85]},
    )
    for label, snt, gate_diff, _load, store in shift_noise_traces:
        if label in {"nominal", "helpful diff -25 mV", "destructive diff +25 mV", "destructive diff +50 mV"}:
            shift_noise_axes[0].plot(1e6 * snt, 1e3 * store, label=label)
    shift_noise_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_noise_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    shift_noise_axes[0].set_xlim(3.05, 3.75)
    shift_noise_axes[0].set_ylabel("stored activation (mV)")
    shift_noise_axes[0].set_title("Shifted-gate activation under deterministic gate differential residue")
    shift_noise_axes[0].grid(True, alpha=0.25)
    shift_noise_axes[0].legend(loc="upper right", fontsize="small")
    shift_noise_x = np.arange(len(shift_noise_labels))
    shift_noise_axes[1].bar(shift_noise_x - 0.24, 1e3 * shift_noise_gate_samples, width=0.24, label="gate diff")
    shift_noise_axes[1].bar(shift_noise_x, 1e3 * shift_noise_load_samples, width=0.24, label="forward load")
    shift_noise_axes[1].bar(shift_noise_x + 0.24, 1e3 * shift_noise_store_samples, width=0.24, label="stored $h$")
    shift_noise_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_noise_axes[1].set_xticks(shift_noise_x)
    shift_noise_axes[1].set_xticklabels(shift_noise_labels, rotation=18, ha="right")
    shift_noise_axes[1].set_ylabel("sampled differential (mV)")
    shift_noise_axes[1].set_title("Common offsets are mild; destructive differential residue reduces margin")
    shift_noise_axes[1].grid(True, axis="y", alpha=0.25)
    shift_noise_axes[1].legend(loc="upper right", ncol=3, fontsize="small")
    shift_noise_fig.tight_layout()
    save_plot(
        shift_noise_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_offset_noise_ngspice",
    )

    shift_threshold_fig, shift_threshold_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.4),
        gridspec_kw={"height_ratios": [1.0, 0.85]},
    )
    for label, stt, load, store in shift_threshold_traces:
        if label in {
            "nominal",
            "both strong -20 mV",
            "both weak +20 mV",
            "skew +20/-20 mV",
            "skew +20/-20, trim -50 mV",
            "skew +20/-20, trim -75 mV",
        }:
            shift_threshold_axes[0].plot(1e6 * stt, 1e3 * store, label=label)
    shift_threshold_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_threshold_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    shift_threshold_axes[0].set_xlim(3.05, 3.75)
    shift_threshold_axes[0].set_ylabel("activation differential (mV)")
    shift_threshold_axes[0].set_title("Shifted-gate forward pair under input-threshold corners")
    shift_threshold_axes[0].grid(True, alpha=0.25)
    shift_threshold_axes[0].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_threshold_x = np.arange(len(shift_threshold_labels))
    shift_threshold_axes[1].bar(
        shift_threshold_x - 0.24,
        1e3 * shift_threshold_gate_samples,
        width=0.24,
        label="gate diff",
    )
    shift_threshold_axes[1].bar(
        shift_threshold_x,
        1e3 * shift_threshold_load_samples,
        width=0.24,
        label="forward load",
    )
    shift_threshold_axes[1].bar(
        shift_threshold_x + 0.24,
        1e3 * shift_threshold_store_samples,
        width=0.24,
        label="stored $h$",
    )
    shift_threshold_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_threshold_axes[1].set_xticks(shift_threshold_x)
    shift_threshold_axes[1].set_xticklabels(shift_threshold_labels, rotation=18, ha="right")
    shift_threshold_axes[1].set_ylabel("sampled differential (mV)")
    shift_threshold_axes[1].set_title("Threshold skew changes gain but must not flip the shifted activation")
    shift_threshold_axes[1].grid(True, axis="y", alpha=0.25)
    shift_threshold_axes[1].legend(loc="upper right", ncol=3, fontsize="small")
    shift_threshold_fig.tight_layout()
    save_plot(
        shift_threshold_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_threshold_ngspice",
    )

    shift_trimmed_reset_fig, shift_trimmed_reset_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.4),
        gridspec_kw={"height_ratios": [1.0, 0.85]},
    )
    for label, strt, _gate_diff, _load, store in shift_trimmed_reset_traces:
        if label in {
            "reset trim 0 mV",
            "reset trim -15 mV",
            "reset trim -50 mV",
            "reset trim -75 mV",
            "reset trim -100 mV",
        }:
            shift_trimmed_reset_axes[0].plot(1e6 * strt, 1e3 * store, label=label)
    shift_trimmed_reset_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_trimmed_reset_axes[0].axvspan(2.50, 2.62, color="0.75", alpha=0.18, label="split reset")
    shift_trimmed_reset_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    shift_trimmed_reset_axes[0].set_xlim(2.35, 3.75)
    shift_trimmed_reset_axes[0].set_ylabel("stored activation (mV)")
    shift_trimmed_reset_axes[0].set_title("Split physical reset trims the skewed shifted-gate forward pair")
    shift_trimmed_reset_axes[0].grid(True, alpha=0.25)
    shift_trimmed_reset_axes[0].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_trimmed_reset_x = np.arange(len(shift_trimmed_reset_labels))
    shift_trimmed_reset_axes[1].bar(
        shift_trimmed_reset_x - 0.24,
        1e3 * shift_trimmed_reset_gate_pre_samples,
        width=0.24,
        label="pre-read gate diff",
    )
    shift_trimmed_reset_axes[1].bar(
        shift_trimmed_reset_x,
        1e3 * shift_trimmed_reset_load_samples,
        width=0.24,
        label="forward load",
    )
    shift_trimmed_reset_axes[1].bar(
        shift_trimmed_reset_x + 0.24,
        1e3 * shift_trimmed_reset_store_samples,
        width=0.24,
        label="stored $h$",
    )
    shift_trimmed_reset_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_trimmed_reset_axes[1].set_xticks(shift_trimmed_reset_x)
    shift_trimmed_reset_axes[1].set_xticklabels(shift_trimmed_reset_labels, rotation=18, ha="right")
    shift_trimmed_reset_axes[1].set_ylabel("sampled differential (mV)")
    shift_trimmed_reset_axes[1].set_title("Real reset rails establish the trim; activation recovers with enough offset budget")
    shift_trimmed_reset_axes[1].grid(True, axis="y", alpha=0.25)
    shift_trimmed_reset_axes[1].legend(loc="upper right", ncol=3, fontsize="small")
    shift_trimmed_reset_fig.tight_layout()
    save_plot(
        shift_trimmed_reset_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_trimmed_reset_ngspice",
    )

    shift_trim_common_fig, shift_trim_common_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.2),
        gridspec_kw={"height_ratios": [0.9, 1.0]},
    )
    shift_trim_common_trim_mv = 1e3 * shift_trim_common_values[0]
    for idx, label in enumerate(shift_trim_common_labels):
        shift_trim_common_axes[0].plot(
            shift_trim_common_trim_mv,
            1e3 * shift_trim_common_gate_samples[idx],
            marker="o",
            label=f"{label} gate diff",
        )
        shift_trim_common_axes[0].plot(
            shift_trim_common_trim_mv,
            1e3 * (shift_trim_common_common_samples[idx] - (0.80 if idx == 0 else 0.90)),
            "--",
            marker="s",
            label=f"{label} common error",
        )
    shift_trim_common_axes[0].plot(
        shift_trim_common_trim_mv,
        shift_trim_common_trim_mv,
        ":",
        color="0.25",
        label="commanded trim",
    )
    shift_trim_common_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_trim_common_axes[0].set_ylabel("reset sample (mV)")
    shift_trim_common_axes[0].set_title("Split reset tracks trim at both legacy and tuned common modes")
    shift_trim_common_axes[0].grid(True, alpha=0.25)
    shift_trim_common_axes[0].legend(loc="lower left", ncol=2, fontsize="x-small")
    for idx, label in enumerate(shift_trim_common_labels):
        shift_trim_common_axes[1].plot(
            shift_trim_common_trim_mv,
            1e3 * shift_trim_common_store_samples[idx],
            marker="o",
            label=f"{label} stored $h$",
        )
        shift_trim_common_axes[1].plot(
            shift_trim_common_trim_mv,
            1e3 * shift_trim_common_load_samples[idx],
            "--",
            alpha=0.65,
            label=f"{label} load",
        )
    shift_trim_common_axes[1].axhspan(20, 85, color="0.7", alpha=0.12, label="accepted window")
    shift_trim_common_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_trim_common_axes[1].set_xlabel("split-reset trim command (mV)")
    shift_trim_common_axes[1].set_ylabel("sampled differential (mV)")
    shift_trim_common_axes[1].set_title("Tuned 0.80 V common keeps the calibrated skew-trim branch useful")
    shift_trim_common_axes[1].grid(True, alpha=0.25)
    shift_trim_common_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_trim_common_fig.tight_layout()
    save_plot(
        shift_trim_common_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_trimmed_reset_common_ngspice",
    )

    shift_balance_fig, shift_balance_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.4),
        gridspec_kw={"height_ratios": [1.0, 0.9]},
    )
    for label, sbt, _load, store in shift_balance_traces:
        if label in {
            "nominal",
            "skew +20/-20, untrimmed",
            "skew +20/-20, trim -35 mV",
            "skew -20/+20, untrimmed",
            "skew -20/+20, trim +35 mV",
        }:
            shift_balance_axes[0].plot(1e6 * sbt, 1e3 * store, label=label)
    shift_balance_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_balance_axes[0].axvspan(2.50, 2.62, color="0.75", alpha=0.18, label="split reset")
    shift_balance_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    shift_balance_axes[0].set_xlim(2.35, 3.75)
    shift_balance_axes[0].set_ylabel("stored activation (mV)")
    shift_balance_axes[0].set_title("Signed split-reset trim pulls opposite threshold skews back toward nominal")
    shift_balance_axes[0].grid(True, alpha=0.25)
    shift_balance_axes[0].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_balance_x = np.arange(len(shift_balance_labels))
    shift_balance_axes[1].bar(
        shift_balance_x - 0.24,
        1e3 * shift_balance_gate_samples,
        width=0.24,
        label="pre-read gate diff",
    )
    shift_balance_axes[1].bar(
        shift_balance_x,
        1e3 * shift_balance_load_samples,
        width=0.24,
        label="forward load",
    )
    shift_balance_axes[1].bar(
        shift_balance_x + 0.24,
        1e3 * shift_balance_store_samples,
        width=0.24,
        label="stored $h$",
    )
    shift_balance_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_balance_axes[1].set_xticks(shift_balance_x)
    shift_balance_axes[1].set_xticklabels(shift_balance_labels, rotation=18, ha="right")
    shift_balance_axes[1].set_ylabel("sampled differential (mV)")
    shift_balance_axes[1].set_title("The same split-reset primitive corrects both skew signs when trim polarity is chosen")
    shift_balance_axes[1].grid(True, axis="y", alpha=0.25)
    shift_balance_axes[1].legend(loc="upper right", ncol=3, fontsize="small")
    shift_balance_fig.tight_layout()
    save_plot(
        shift_balance_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_balanced_trim_ngspice",
    )

    shift_trimmed_reuse_fig, shift_trimmed_reuse_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.9]},
    )
    shift_trimmed_reuse_axes[0].plot(1e6 * strpt, 1e3 * shift_trimmed_reuse_gate_diff, label="shifted-gate diff")
    shift_trimmed_reuse_reset_mask = shift_trimmed_reuse_cols[20] > 0.9
    shift_trimmed_reuse_axes[0].plot(
        1e6 * strpt,
        1e3 * np.where(shift_trimmed_reuse_reset_mask, shift_trimmed_reuse_gate_common - 0.90, np.nan),
        "--",
        label="common error during reset",
    )
    shift_trimmed_reuse_axes[0].plot(
        1e6 * strpt,
        1e3 * shift_trimmed_reuse_cols[20] / 20.0,
        color="0.45",
        alpha=0.25,
        label="$reset/20$",
    )
    shift_trimmed_reuse_axes[0].axhline(-35.0, color="0.2", linewidth=0.8, linestyle=":", label="trim target")
    shift_trimmed_reuse_axes[0].scatter(
        1e6 * shift_trimmed_reuse_reset_times,
        1e3 * shift_trimmed_reuse_gate_reset_diff,
        color="tab:blue",
        s=20,
        zorder=4,
        label="reset samples",
    )
    for start_us, end_us, label in [(2.50, 2.62, "pre-reset"), (3.60, 4.00, "reset 1"), (5.20, 5.60, "reset 2")]:
        shift_trimmed_reuse_axes[0].axvspan(start_us, end_us, color="0.75", alpha=0.16, label=label)
    shift_trimmed_reuse_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_trimmed_reuse_axes[0].set_xlim(2.35, 7.1)
    shift_trimmed_reuse_axes[0].set_ylim(-85, 105)
    shift_trimmed_reuse_axes[0].set_ylabel("gate error (mV)")
    shift_trimmed_reuse_axes[0].set_title("Trimmed split resets re-center the skewed shifted gate between reads")
    shift_trimmed_reuse_axes[0].grid(True, alpha=0.25)
    shift_trimmed_reuse_axes[0].legend(loc="lower left", ncol=2, fontsize="x-small")
    shift_trimmed_reuse_axes[1].plot(1e6 * strpt, 1e3 * shift_trimmed_reuse_preact, label="$z^- - z^+$")
    shift_trimmed_reuse_axes[1].plot(1e6 * strpt, 1e3 * shift_trimmed_reuse_load, label="forward load")
    shift_trimmed_reuse_axes[1].plot(1e6 * strpt, 1e3 * shift_trimmed_reuse_store, label="stored activation")
    shift_trimmed_reuse_axes[1].plot(
        1e6 * strpt,
        1e3 * shift_trimmed_reuse_cols[23] / 20.0,
        color="0.35",
        alpha=0.25,
        label="$read/20$",
    )
    shift_trimmed_reuse_axes[1].plot(
        1e6 * strpt,
        1e3 * shift_trimmed_reuse_cols[24] / 20.0,
        color="0.15",
        alpha=0.25,
        label="$pact/20$",
    )
    shift_trimmed_reuse_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_trimmed_reuse_axes[1].set_xlim(2.35, 7.1)
    shift_trimmed_reuse_axes[1].set_ylabel("differential (mV)")
    shift_trimmed_reuse_axes[1].set_title("Same skewed pair and MOS-written W/B state produce repeated useful captures")
    shift_trimmed_reuse_axes[1].grid(True, alpha=0.25)
    shift_trimmed_reuse_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    shift_trimmed_reuse_x = np.arange(len(shift_trimmed_reuse_h_samples))
    shift_trimmed_reuse_axes[2].bar(
        shift_trimmed_reuse_x - 0.18,
        1e3 * shift_trimmed_reuse_z_samples,
        width=0.36,
        label="$z$ sample",
    )
    shift_trimmed_reuse_axes[2].bar(
        shift_trimmed_reuse_x + 0.18,
        1e3 * shift_trimmed_reuse_h_samples,
        width=0.36,
        label="$h$ sample",
    )
    shift_trimmed_reuse_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_trimmed_reuse_axes[2].set_xticks(shift_trimmed_reuse_x)
    shift_trimmed_reuse_axes[2].set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    shift_trimmed_reuse_axes[2].set_xlabel("read/reset/store cycle")
    shift_trimmed_reuse_axes[2].set_ylabel("sampled differential (mV)")
    shift_trimmed_reuse_axes[2].set_title("Calibrated reuse stays positive over repeated physical resets")
    shift_trimmed_reuse_axes[2].grid(True, axis="y", alpha=0.25)
    shift_trimmed_reuse_axes[2].legend(loc="upper right", fontsize="small")
    shift_trimmed_reuse_fig.tight_layout()
    save_plot(
        shift_trimmed_reuse_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_trimmed_reuse_ngspice",
    )

    shift_trimmed_margin_fig, shift_trimmed_margin_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.2),
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0]},
    )
    shift_trimmed_margin_x = np.arange(len(shift_trimmed_margin_labels))
    shift_trimmed_margin_axes[0].errorbar(
        1e3 * shift_trimmed_margin_trim_values,
        1e3 * np.mean(shift_trimmed_margin_gate_samples, axis=1),
        yerr=1e3
        * np.vstack(
            [
                np.mean(shift_trimmed_margin_gate_samples, axis=1) - np.min(shift_trimmed_margin_gate_samples, axis=1),
                np.max(shift_trimmed_margin_gate_samples, axis=1) - np.mean(shift_trimmed_margin_gate_samples, axis=1),
            ]
        ),
        marker="o",
        capsize=3,
        label="measured reset diff",
    )
    shift_trimmed_margin_axes[0].plot(
        1e3 * shift_trimmed_margin_trim_values,
        1e3 * shift_trimmed_margin_trim_values,
        ":",
        color="0.25",
        label="commanded trim",
    )
    shift_trimmed_margin_axes[0].plot(
        1e3 * shift_trimmed_margin_trim_values,
        1e3 * (np.mean(shift_trimmed_margin_common_samples, axis=1) - 0.90),
        "--",
        label="common error",
    )
    shift_trimmed_margin_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_trimmed_margin_axes[0].set_ylabel("reset sample (mV)")
    shift_trimmed_margin_axes[0].set_title("Physical split reset tracks trim command across a local calibration window")
    shift_trimmed_margin_axes[0].grid(True, alpha=0.25)
    shift_trimmed_margin_axes[0].legend(loc="upper left", fontsize="small")
    shift_trimmed_margin_axes[1].errorbar(
        shift_trimmed_margin_x - 0.08,
        1e3 * shift_trimmed_margin_h_mean,
        yerr=1e3
        * np.vstack(
            [
                shift_trimmed_margin_h_mean - shift_trimmed_margin_h_min,
                shift_trimmed_margin_h_max - shift_trimmed_margin_h_mean,
            ]
        ),
        fmt="o",
        capsize=4,
        label="$h$ cycles",
    )
    shift_trimmed_margin_axes[1].plot(
        shift_trimmed_margin_x + 0.08,
        1e3 * np.mean(shift_trimmed_margin_z_samples, axis=1),
        "s",
        label="$z$ cycles",
    )
    shift_trimmed_margin_axes[1].axhspan(20, 85, color="0.7", alpha=0.12, label="accepted window")
    shift_trimmed_margin_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_trimmed_margin_axes[1].set_xticks(shift_trimmed_margin_x)
    shift_trimmed_margin_axes[1].set_xticklabels(shift_trimmed_margin_labels)
    shift_trimmed_margin_axes[1].set_ylabel("cycle samples (mV)")
    shift_trimmed_margin_axes[1].set_title("Stored activation stays positive and bounded for +/-10 mV trim error")
    shift_trimmed_margin_axes[1].grid(True, axis="y", alpha=0.25)
    shift_trimmed_margin_axes[1].legend(loc="upper left", fontsize="small")
    for label, tmt, load, store in shift_trimmed_margin_traces:
        shift_trimmed_margin_axes[2].plot(1e6 * tmt, 1e3 * store, label=f"{label} $h$")
        shift_trimmed_margin_axes[2].plot(1e6 * tmt, 1e3 * load, "--", alpha=0.65, label=f"{label} load")
    shift_trimmed_margin_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_trimmed_margin_axes[2].set_xlim(2.35, 7.1)
    shift_trimmed_margin_axes[2].set_xlabel("time (us)")
    shift_trimmed_margin_axes[2].set_ylabel("differential (mV)")
    shift_trimmed_margin_axes[2].set_title("Repeated captures remain ordered across weak/nominal/strong trims")
    shift_trimmed_margin_axes[2].grid(True, alpha=0.25)
    shift_trimmed_margin_axes[2].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_trimmed_margin_fig.tight_layout()
    save_plot(
        shift_trimmed_margin_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_trimmed_margin_ngspice",
    )

    shift_refpert_fig, shift_refpert_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
    )
    shift_refpert_x = np.arange(3)
    for idx, label in enumerate(shift_refpert_labels):
        shift_refpert_axes[0].plot(
            shift_refpert_x,
            1e3 * shift_refpert_gate_samples[idx],
            marker="o",
            label=label,
        )
    shift_refpert_axes[0].set_xticks(shift_refpert_x)
    shift_refpert_axes[0].set_xticklabels(["reset 0", "reset 1", "reset 2"])
    shift_refpert_axes[0].set_ylabel("gate diff (mV)")
    shift_refpert_axes[0].set_title("PWL reset references produce bounded cycle-by-cycle trim errors")
    shift_refpert_axes[0].grid(True, alpha=0.25)
    shift_refpert_axes[0].legend(loc="lower left", ncol=2, fontsize="x-small")
    for idx, label in enumerate(shift_refpert_labels):
        shift_refpert_axes[1].plot(
            shift_refpert_x,
            1e3 * shift_refpert_h_samples[idx],
            marker="o",
            label=label,
        )
    shift_refpert_axes[1].plot(
        shift_refpert_x,
        1e3 * shift_refpert_z_samples[shift_refpert_index["clean"]],
        "s--",
        color="0.35",
        label="clean $z$",
    )
    shift_refpert_axes[1].axhspan(20, 85, color="0.7", alpha=0.12, label="accepted window")
    shift_refpert_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_refpert_axes[1].set_xticks(shift_refpert_x)
    shift_refpert_axes[1].set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    shift_refpert_axes[1].set_ylabel("cycle samples (mV)")
    shift_refpert_axes[1].set_title("Differential trim jitter dominates; common-mode jitter is small")
    shift_refpert_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refpert_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    for label, rpt, load, store in shift_refpert_traces:
        shift_refpert_axes[2].plot(1e6 * rpt, 1e3 * store, label=f"{label} $h$")
        shift_refpert_axes[2].plot(1e6 * rpt, 1e3 * load, "--", alpha=0.55, label=f"{label} load")
    shift_refpert_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_refpert_axes[2].set_xlim(2.35, 7.1)
    shift_refpert_axes[2].set_xlabel("time (us)")
    shift_refpert_axes[2].set_ylabel("differential (mV)")
    shift_refpert_axes[2].set_title("Time-domain traces stay bounded under cycle-varying reset references")
    shift_refpert_axes[2].grid(True, alpha=0.25)
    shift_refpert_axes[2].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_refpert_fig.tight_layout()
    save_plot(
        shift_refpert_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_refpert_ngspice",
    )

    shift_skewlaw_fig, shift_skewlaw_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.5),
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0]},
    )
    calibrated_idx = np.flatnonzero(shift_skewlaw_calibrated)
    under_idx = np.flatnonzero(~shift_skewlaw_calibrated)
    shift_skewlaw_axes[0].plot(
        shift_skewlaw_skews_mv[calibrated_idx],
        1e3 * shift_skewlaw_trim_values[calibrated_idx],
        "o-",
        label="requested trim",
    )
    shift_skewlaw_axes[0].errorbar(
        shift_skewlaw_skews_mv[calibrated_idx],
        1e3 * np.mean(shift_skewlaw_gate_samples[calibrated_idx], axis=1),
        yerr=1e3
        * np.vstack(
            [
                np.mean(shift_skewlaw_gate_samples[calibrated_idx], axis=1)
                - np.min(shift_skewlaw_gate_samples[calibrated_idx], axis=1),
                np.max(shift_skewlaw_gate_samples[calibrated_idx], axis=1)
                - np.mean(shift_skewlaw_gate_samples[calibrated_idx], axis=1),
            ]
        ),
        fmt="s--",
        capsize=3,
        label="sampled reset diff",
    )
    if len(under_idx):
        shift_skewlaw_axes[0].scatter(
            shift_skewlaw_skews_mv[under_idx],
            1e3 * shift_skewlaw_trim_values[under_idx],
            marker="x",
            s=48,
            label="under-trimmed severe skew",
        )
    shift_skewlaw_axes[0].set_ylabel("split trim (mV)")
    shift_skewlaw_axes[0].set_title("Physical split reset can encode a skew-scaled trim law")
    shift_skewlaw_axes[0].grid(True, alpha=0.25)
    shift_skewlaw_axes[0].legend(loc="lower left", ncol=2, fontsize="small")
    shift_skewlaw_x = np.arange(3)
    for idx in calibrated_idx:
        shift_skewlaw_axes[1].plot(
            shift_skewlaw_x,
            1e3 * shift_skewlaw_h_samples[idx],
            marker="o",
            label=shift_skewlaw_labels[idx],
        )
    if len(under_idx):
        for idx in under_idx:
            shift_skewlaw_axes[1].plot(
                shift_skewlaw_x,
                1e3 * shift_skewlaw_h_samples[idx],
                "x--",
                linewidth=1.5,
                label=shift_skewlaw_labels[idx],
            )
    shift_skewlaw_axes[1].axhspan(25, 115, color="0.7", alpha=0.12, label="accepted window")
    shift_skewlaw_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_skewlaw_axes[1].set_xticks(shift_skewlaw_x)
    shift_skewlaw_axes[1].set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    shift_skewlaw_axes[1].set_ylabel("stored $h$ (mV)")
    shift_skewlaw_axes[1].set_title("Calibrated trims keep repeated captures positive across skew magnitudes")
    shift_skewlaw_axes[1].grid(True, axis="y", alpha=0.25)
    shift_skewlaw_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    for label, kpt, load, store in shift_skewlaw_traces:
        shift_skewlaw_axes[2].plot(1e6 * kpt, 1e3 * store, label=f"{label} $h$")
        shift_skewlaw_axes[2].plot(1e6 * kpt, 1e3 * load, "--", alpha=0.55, label=f"{label} load")
    shift_skewlaw_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_skewlaw_axes[2].set_xlim(2.35, 7.1)
    shift_skewlaw_axes[2].set_xlabel("time (us)")
    shift_skewlaw_axes[2].set_ylabel("differential (mV)")
    shift_skewlaw_axes[2].set_title("Under-trimmed severe skew is visibly weaker than the calibrated case")
    shift_skewlaw_axes[2].grid(True, alpha=0.25)
    shift_skewlaw_axes[2].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_skewlaw_fig.tight_layout()
    save_plot(
        shift_skewlaw_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_skew_trim_law_ngspice",
    )

    shift_signlaw_fig, shift_signlaw_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.5),
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0]},
    )
    sign_cal_idx = np.flatnonzero(shift_signlaw_calibrated)
    sign_under_idx = np.flatnonzero(~shift_signlaw_calibrated)
    order = np.argsort(shift_signlaw_skews_mv[sign_cal_idx])
    sign_cal_sorted = sign_cal_idx[order]
    shift_signlaw_axes[0].plot(
        shift_signlaw_skews_mv[sign_cal_sorted],
        1e3 * shift_signlaw_trim_values[sign_cal_sorted],
        "o-",
        label="requested trim",
    )
    shift_signlaw_axes[0].errorbar(
        shift_signlaw_skews_mv[sign_cal_sorted],
        1e3 * np.mean(shift_signlaw_gate_samples[sign_cal_sorted], axis=1),
        yerr=1e3
        * np.vstack(
            [
                np.mean(shift_signlaw_gate_samples[sign_cal_sorted], axis=1)
                - np.min(shift_signlaw_gate_samples[sign_cal_sorted], axis=1),
                np.max(shift_signlaw_gate_samples[sign_cal_sorted], axis=1)
                - np.mean(shift_signlaw_gate_samples[sign_cal_sorted], axis=1),
            ]
        ),
        fmt="s--",
        capsize=3,
        label="sampled reset diff",
    )
    shift_signlaw_axes[0].scatter(
        shift_signlaw_skews_mv[sign_under_idx],
        1e3 * shift_signlaw_trim_values[sign_under_idx],
        marker="x",
        s=48,
        label="under-trimmed 30 mV skew",
    )
    shift_signlaw_axes[0].axhline(0, color="0.4", linewidth=0.8)
    shift_signlaw_axes[0].axvline(0, color="0.4", linewidth=0.8)
    shift_signlaw_axes[0].set_ylabel("split trim (mV)")
    shift_signlaw_axes[0].set_title("Split-reset trim law is signed across both skew polarities")
    shift_signlaw_axes[0].grid(True, alpha=0.25)
    shift_signlaw_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    sign_x = np.arange(3)
    for idx in sign_cal_sorted:
        shift_signlaw_axes[1].plot(
            sign_x,
            1e3 * shift_signlaw_h_samples[idx],
            marker="o",
            label=shift_signlaw_labels[idx],
        )
    for idx in sign_under_idx:
        shift_signlaw_axes[1].plot(
            sign_x,
            1e3 * shift_signlaw_h_samples[idx],
            "x--",
            linewidth=1.5,
            label=shift_signlaw_labels[idx],
        )
    shift_signlaw_axes[1].axhspan(25, 125, color="0.7", alpha=0.12, label="accepted window")
    shift_signlaw_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_signlaw_axes[1].set_xticks(sign_x)
    shift_signlaw_axes[1].set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    shift_signlaw_axes[1].set_ylabel("stored $h$ (mV)")
    shift_signlaw_axes[1].set_title("Both signed trims recover positive repeated captures")
    shift_signlaw_axes[1].grid(True, axis="y", alpha=0.25)
    shift_signlaw_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    for label, spt, load, store in shift_signlaw_traces:
        shift_signlaw_axes[2].plot(1e6 * spt, 1e3 * store, label=f"{label} $h$")
        shift_signlaw_axes[2].plot(1e6 * spt, 1e3 * load, "--", alpha=0.55, label=f"{label} load")
    shift_signlaw_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_signlaw_axes[2].set_xlim(2.35, 7.1)
    shift_signlaw_axes[2].set_xlabel("time (us)")
    shift_signlaw_axes[2].set_ylabel("differential (mV)")
    shift_signlaw_axes[2].set_title("Severe under-trim leaves opposite low/high activation errors")
    shift_signlaw_axes[2].grid(True, alpha=0.25)
    shift_signlaw_axes[2].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_signlaw_fig.tight_layout()
    save_plot(
        shift_signlaw_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_signed_skew_trim_law_ngspice",
    )

    shift_polcal_fig, shift_polcal_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.0),
        gridspec_kw={"height_ratios": [0.95, 1.05]},
    )
    pm_trim_abs = np.abs(shift_polcal_trim_mv[pm_order])
    mp_trim_abs = np.abs(shift_polcal_trim_mv[mp_order])
    shift_polcal_axes[0].plot(
        pm_trim_abs,
        1e3 * shift_polcal_h_mean[pm_order],
        "o-",
        label="+30 mV skew, negative trim",
    )
    shift_polcal_axes[0].plot(
        mp_trim_abs,
        1e3 * shift_polcal_h_mean[mp_order],
        "s--",
        label="-30 mV skew, positive trim",
    )
    shift_polcal_axes[0].axhline(
        1e3 * shift_polcal_h_mean[shift_polcal_index["pm30_neg55"]],
        color="0.25",
        linestyle=":",
        linewidth=1.0,
        label="-55 mV reference",
    )
    shift_polcal_axes[0].axvline(65, color="0.5", linestyle="--", linewidth=0.9, alpha=0.75)
    shift_polcal_axes[0].set_xlabel("trim magnitude (mV)")
    shift_polcal_axes[0].set_ylabel("mean stored $h$ (mV)")
    shift_polcal_axes[0].set_title("Opposite skew polarity needs an offset trim code")
    shift_polcal_axes[0].grid(True, alpha=0.25)
    shift_polcal_axes[0].legend(loc="upper right", fontsize="small")
    for label, pt, load, store in shift_polcal_traces:
        if "+30" in label and "-55" not in label:
            continue
        if "-30" in label and "+65" not in label:
            continue
        shift_polcal_axes[1].plot(1e6 * pt, 1e3 * store, label=f"{label} $h$")
        shift_polcal_axes[1].plot(1e6 * pt, 1e3 * load, "--", alpha=0.55, label=f"{label} load")
    shift_polcal_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_polcal_axes[1].set_xlim(2.35, 7.1)
    shift_polcal_axes[1].set_xlabel("time (us)")
    shift_polcal_axes[1].set_ylabel("differential (mV)")
    shift_polcal_axes[1].set_title("+65 mV positive trim aligns the opposite-polarity branch")
    shift_polcal_axes[1].grid(True, alpha=0.25)
    shift_polcal_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_polcal_fig.tight_layout()
    save_plot(
        shift_polcal_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_polarity_calibration_ngspice",
    )

    shift_polcal_common_fig, shift_polcal_common_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.0),
        gridspec_kw={"height_ratios": [0.95, 1.05]},
    )
    polcal_common_modes = np.array([0.80, 0.90])
    pm_common_order = [pm_cm080_idx, pm_cm090_idx]
    mp_common_order = [mp_cm080_idx, mp_cm090_idx]
    shift_polcal_common_axes[0].plot(
        polcal_common_modes,
        1e3 * shift_polcal_common_h_mean[pm_common_order],
        "o-",
        label="+30 mV skew, -55 mV trim",
    )
    shift_polcal_common_axes[0].plot(
        polcal_common_modes,
        1e3 * shift_polcal_common_h_mean[mp_common_order],
        "s--",
        label="-30 mV skew, +65 mV trim",
    )
    shift_polcal_common_axes[0].axhspan(25, 85, color="0.7", alpha=0.12, label="accepted window")
    shift_polcal_common_axes[0].axvline(0.80, color="tab:green", linestyle=":", linewidth=1.0, label="tuned common")
    shift_polcal_common_axes[0].axvline(0.90, color="tab:red", linestyle=":", linewidth=1.0, label="legacy common")
    shift_polcal_common_axes[0].set_xticks(polcal_common_modes)
    shift_polcal_common_axes[0].set_ylabel("mean stored $h$ (mV)")
    shift_polcal_common_axes[0].set_title("Severe trim calibration survives the tuned reset common mode")
    shift_polcal_common_axes[0].grid(True, alpha=0.25)
    shift_polcal_common_axes[0].legend(loc="upper left", ncol=2, fontsize="x-small")
    common_cycle_x = np.arange(3)
    for idx in [pm_cm080_idx, pm_cm090_idx, mp_cm080_idx, mp_cm090_idx]:
        label = shift_polcal_common_labels[idx]
        short_label = label.replace(" mV skew, ", " ").replace(" mV trim, ", " ").replace(" V common", " Vcm")
        shift_polcal_common_axes[1].plot(
            common_cycle_x,
            1e3 * shift_polcal_common_h_samples[idx],
            marker="o" if "0.80" in label else "s",
            linestyle="-" if "+30" in label else "--",
            label=short_label,
        )
    shift_polcal_common_axes[1].axhspan(25, 85, color="0.7", alpha=0.12)
    shift_polcal_common_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_polcal_common_axes[1].set_xticks(common_cycle_x)
    shift_polcal_common_axes[1].set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    shift_polcal_common_axes[1].set_xlabel("read/store cycle")
    shift_polcal_common_axes[1].set_ylabel("stored $h$ (mV)")
    shift_polcal_common_axes[1].set_title("Both severe-skew polarities remain repeatable at 0.80 V")
    shift_polcal_common_axes[1].grid(True, axis="y", alpha=0.25)
    shift_polcal_common_axes[1].legend(loc="upper left", ncol=2, fontsize="xx-small")
    shift_polcal_common_fig.tight_layout()
    save_plot(
        shift_polcal_common_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_polarity_common_ngspice",
    )

    shift_polsens_fig, shift_polsens_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.0),
        gridspec_kw={"height_ratios": [0.95, 1.05]},
    )
    shift_polsens_axes[0].plot(
        shift_polsens_trim_error_mv[pm_sens_order],
        1e3 * shift_polsens_h_mean[pm_sens_order],
        "o-",
        label=f"+30 mV skew, slope {pm_slope:.2f} mV/mV",
    )
    shift_polsens_axes[0].plot(
        shift_polsens_trim_error_mv[mp_sens_order],
        1e3 * shift_polsens_h_mean[mp_sens_order],
        "s--",
        label=f"-30 mV skew, slope {mp_slope:.2f} mV/mV",
    )
    shift_polsens_axes[0].axvline(0, color="0.4", linewidth=0.8)
    shift_polsens_axes[0].axhline(
        1e3 * shift_polsens_h_mean[shift_polsens_index["pm30_neg55"]],
        color="0.25",
        linestyle=":",
        linewidth=1.0,
        label="calibrated target",
    )
    shift_polsens_axes[0].set_xlabel("trim-code error from calibrated value (mV)")
    shift_polsens_axes[0].set_ylabel("mean stored $h$ (mV)")
    shift_polsens_axes[0].set_title("Trim-code errors remain locally linear over +/-10 mV")
    shift_polsens_axes[0].grid(True, alpha=0.25)
    shift_polsens_axes[0].legend(loc="upper right", fontsize="small")
    for group, trim_error_mv, pst, load, store in shift_polsens_traces:
        if trim_error_mv != 0.0:
            continue
        shift_polsens_axes[1].plot(1e6 * pst, 1e3 * store, label=f"{group}, calibrated $h$")
        shift_polsens_axes[1].plot(1e6 * pst, 1e3 * load, "--", alpha=0.55, label=f"{group}, load")
    shift_polsens_axes[1].axhline(0, color="0.4", linewidth=0.8)
    shift_polsens_axes[1].set_xlim(2.35, 7.1)
    shift_polsens_axes[1].set_xlabel("time (us)")
    shift_polsens_axes[1].set_ylabel("differential (mV)")
    shift_polsens_axes[1].set_title("Calibrated fine-sensitivity baselines remain aligned over repeated cycles")
    shift_polsens_axes[1].grid(True, alpha=0.25)
    shift_polsens_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_polsens_fig.tight_layout()
    save_plot(
        shift_polsens_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_trim_sensitivity_ngspice",
    )

    shift_polref_fig, shift_polref_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.2),
        gridspec_kw={"height_ratios": [0.8, 0.9, 1.0]},
    )
    polref_x = np.arange(3)
    for idx, label in enumerate(shift_polref_labels):
        style = "o-" if idx in {0, 1} else "s--"
        shift_polref_axes[0].plot(polref_x, 1e3 * shift_polref_gate_samples[idx], style, label=label)
    shift_polref_axes[0].axhline(65, color="0.25", linestyle=":", linewidth=1.0, label="+65 mV target")
    shift_polref_axes[0].set_xticks(polref_x)
    shift_polref_axes[0].set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    shift_polref_axes[0].set_ylabel("sampled trim (mV)")
    shift_polref_axes[0].set_title("Positive calibrated split reset tracks cycle-varying trim commands")
    shift_polref_axes[0].grid(True, axis="y", alpha=0.25)
    shift_polref_axes[0].legend(loc="upper left", ncol=2, fontsize="x-small")
    for idx, label in enumerate(shift_polref_labels):
        style = "o-" if idx in {0, 1} else "s--"
        shift_polref_axes[1].plot(polref_x, 1e3 * shift_polref_h_samples[idx], style, label=label)
    shift_polref_axes[1].axhspan(25, 70, color="0.7", alpha=0.12, label="useful window")
    shift_polref_axes[1].set_xticks(polref_x)
    shift_polref_axes[1].set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    shift_polref_axes[1].set_ylabel("stored $h$ (mV)")
    shift_polref_axes[1].set_title("Differential trim jitter dominates over common-mode reference jitter")
    shift_polref_axes[1].grid(True, axis="y", alpha=0.25)
    shift_polref_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    for label, prt, load, store in shift_polref_traces:
        shift_polref_axes[2].plot(1e6 * prt, 1e3 * store, label=f"{label} $h$")
        shift_polref_axes[2].plot(1e6 * prt, 1e3 * load, "--", alpha=0.55, label=f"{label} load")
    shift_polref_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_polref_axes[2].set_xlim(2.35, 7.1)
    shift_polref_axes[2].set_xlabel("time (us)")
    shift_polref_axes[2].set_ylabel("differential (mV)")
    shift_polref_axes[2].set_title("Positive calibrated branch remains bounded under reset-reference perturbation")
    shift_polref_axes[2].grid(True, alpha=0.25)
    shift_polref_axes[2].legend(loc="upper left", ncol=2, fontsize="x-small")
    shift_polref_fig.tight_layout()
    save_plot(
        shift_polref_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_positive_refpert_ngspice",
    )

    shift_reset_corner_fig, shift_reset_corner_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.8, 0.9, 1.0]},
    )
    reset_corner_x = np.arange(len(shift_reset_corner_specs))
    reset_corner_labels = [label for _name, label, _nvto, _pvto in shift_reset_corner_specs]
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_reset_corner_axes[0].plot(
            reset_corner_x,
            1e3 * shift_reset_corner_trim_error[branch_idx],
            marker,
            label=branch_label,
        )
    shift_reset_corner_axes[0].set_xticks(reset_corner_x)
    shift_reset_corner_axes[0].set_xticklabels(reset_corner_labels, rotation=15, ha="right")
    shift_reset_corner_axes[0].set_ylabel("max trim error (mV)")
    shift_reset_corner_axes[0].set_title("Reset TG threshold corners preserve calibrated split trim")
    shift_reset_corner_axes[0].grid(True, axis="y", alpha=0.25)
    shift_reset_corner_axes[0].legend(loc="upper left", fontsize="x-small")
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_reset_corner_axes[1].plot(
            reset_corner_x,
            1e3 * shift_reset_corner_h_mean[branch_idx],
            marker,
            label=branch_label,
        )
    shift_reset_corner_axes[1].axhspan(35, 60, color="0.7", alpha=0.12, label="aligned window")
    shift_reset_corner_axes[1].set_xticks(reset_corner_x)
    shift_reset_corner_axes[1].set_xticklabels(reset_corner_labels, rotation=15, ha="right")
    shift_reset_corner_axes[1].set_ylabel("mean stored $h$ (mV)")
    shift_reset_corner_axes[1].set_title("Stored activation is dominated by calibration code, not reset TG corner")
    shift_reset_corner_axes[1].grid(True, axis="y", alpha=0.25)
    shift_reset_corner_axes[1].legend(loc="upper left", fontsize="x-small")
    for branch_label, corner_label, rst, load, store in shift_reset_corner_traces:
        if ("+30" in branch_label and corner_label == "nweak_pstrong") or (
            "-30" in branch_label and corner_label == "weak TG"
        ):
            linestyle = "--"
        else:
            linestyle = "-"
        shift_reset_corner_axes[2].plot(
            1e6 * rst,
            1e3 * store,
            linestyle,
            label=f"{branch_label.split(', ')[1]}, {corner_label} $h$",
        )
    shift_reset_corner_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_reset_corner_axes[2].set_xlim(2.35, 7.1)
    shift_reset_corner_axes[2].set_xlabel("time (us)")
    shift_reset_corner_axes[2].set_ylabel("stored $h$ (mV)")
    shift_reset_corner_axes[2].set_title("Physical reset-switch corners still clear and reuse the shifted gate")
    shift_reset_corner_axes[2].grid(True, alpha=0.25)
    shift_reset_corner_axes[2].legend(loc="lower right", ncol=1, fontsize="xx-small")
    shift_reset_corner_fig.tight_layout()
    save_plot(
        shift_reset_corner_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_switch_corner_ngspice",
    )

    shift_refz_fig, shift_refz_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.8, 0.9, 1.0]},
    )
    refz_x = np.arange(len(shift_refz_cases))
    refz_labels = [label for _name, label, _series in shift_refz_cases]
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_axes[0].plot(
            refz_x,
            1e3 * shift_refz_trim_error[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_axes[0].axhline(6, color="0.4", linestyle=":", linewidth=0.9, label="6 mV trim-error gate")
    shift_refz_axes[0].set_xticks(refz_x)
    shift_refz_axes[0].set_xticklabels(refz_labels)
    shift_refz_axes[0].set_ylabel("max trim error (mV)")
    shift_refz_axes[0].set_title("Finite reset-reference impedance limits split-trim delivery")
    shift_refz_axes[0].grid(True, axis="y", alpha=0.25)
    shift_refz_axes[0].legend(loc="upper left", ncol=2, fontsize="x-small")
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_axes[1].plot(
            refz_x,
            1e3 * shift_refz_h_mean[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_axes[1].axhspan(40, 55, color="0.7", alpha=0.12, label="calibrated window")
    shift_refz_axes[1].set_xticks(refz_x)
    shift_refz_axes[1].set_xticklabels(refz_labels)
    shift_refz_axes[1].set_ylabel("mean stored $h$ (mV)")
    shift_refz_axes[1].set_title("Weak trim drivers eventually revert toward under-trimmed behavior")
    shift_refz_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refz_axes[1].legend(loc="upper left", ncol=2, fontsize="x-small")
    for branch_label, case_label, rzt, load, store in shift_refz_traces:
        if case_label in {"100k"}:
            continue
        shift_refz_axes[2].plot(
            1e6 * rzt,
            1e3 * store,
            label=f"{branch_label.split(', ')[1]}, {case_label} $h$",
        )
    shift_refz_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_refz_axes[2].set_xlim(2.35, 7.1)
    shift_refz_axes[2].set_xlabel("time (us)")
    shift_refz_axes[2].set_ylabel("stored $h$ (mV)")
    shift_refz_axes[2].set_title("High source impedance leaves the shifted gate under-calibrated")
    shift_refz_axes[2].grid(True, alpha=0.25)
    shift_refz_axes[2].legend(loc="upper left", ncol=2, fontsize="xx-small")
    shift_refz_fig.tight_layout()
    save_plot(
        shift_refz_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_ref_impedance_ngspice",
    )

    shift_refz_width_fig, shift_refz_width_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.85, 0.85, 1.0]},
    )
    refz_width_x = np.array([width_us * 1e3 for _name, _label, width_us in shift_refz_width_cases])
    for source_idx, (_source_name, source_label, _series_ohm) in enumerate(shift_refz_width_source_cases):
        for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
            marker = "o-" if branch_idx == 0 else "s--"
            alpha = 1.0 if source_label == "10k" else 0.55
            shift_refz_width_axes[0].plot(
                refz_width_x,
                1e3 * shift_refz_width_trim_error[branch_idx, source_idx],
                marker,
                alpha=alpha,
                label=f"{branch_label.split(', ')[1]}, {source_label}",
            )
    shift_refz_width_axes[0].axhline(6, color="0.4", linestyle=":", linewidth=0.9, label="6 mV trim-error gate")
    shift_refz_width_axes[0].set_ylabel("max trim error (mV)")
    shift_refz_width_axes[0].set_title("Longer first reset recovers moderate trim-source impedance")
    shift_refz_width_axes[0].grid(True, axis="y", alpha=0.25)
    shift_refz_width_axes[0].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for source_idx, (_source_name, source_label, _series_ohm) in enumerate(shift_refz_width_source_cases):
        for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
            marker = "o-" if branch_idx == 0 else "s--"
            alpha = 1.0 if source_label == "10k" else 0.55
            shift_refz_width_axes[1].plot(
                refz_width_x,
                1e3 * shift_refz_width_h_sample[branch_idx, source_idx],
                marker,
                alpha=alpha,
                label=f"{branch_label.split(', ')[1]}, {source_label}",
            )
    shift_refz_width_axes[1].axhspan(40, 55, color="0.7", alpha=0.12, label="calibrated window")
    shift_refz_width_axes[1].set_ylabel("first stored $h$ (mV)")
    shift_refz_width_axes[1].set_title("Trim recovery improves 10k activation, but schedule still matters")
    shift_refz_width_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refz_width_axes[1].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for branch_label, source_label, width_label, rwt, store in shift_refz_width_traces:
        if source_label == "100k" and width_label == "120 ns":
            continue
        linestyle = "-" if source_label == "10k" else ":"
        shift_refz_width_axes[2].plot(
            1e6 * rwt,
            1e3 * store,
            linestyle,
            label=f"{branch_label.split(', ')[1]}, {source_label}, {width_label}",
        )
    shift_refz_width_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_refz_width_axes[2].set_xlim(2.35, 3.75)
    shift_refz_width_axes[2].set_xlabel("time (us)")
    shift_refz_width_axes[2].set_ylabel("stored $h$ (mV)")
    shift_refz_width_axes[2].set_title("First-cycle traces expose reset-width recovery")
    shift_refz_width_axes[2].grid(True, alpha=0.25)
    shift_refz_width_axes[2].legend(loc="upper left", ncol=2, fontsize="xx-small")
    shift_refz_width_fig.tight_layout()
    save_plot(
        shift_refz_width_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_ref_width_ngspice",
    )

    shift_refz_decap_fig, shift_refz_decap_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.85, 0.85, 1.0]},
    )
    refz_decap_x = np.array([cap_pf for _name, _label, cap_pf in shift_refz_decap_cases])
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_decap_axes[0].plot(
            refz_decap_x,
            1e3 * shift_refz_decap_trim_error[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_decap_axes[0].axhline(6, color="0.4", linestyle=":", linewidth=0.9, label="6 mV trim-error gate")
    shift_refz_decap_axes[0].set_ylabel("trim error (mV)")
    shift_refz_decap_axes[0].set_title("Local trim-reference capacitance buffers a 100k source")
    shift_refz_decap_axes[0].grid(True, axis="y", alpha=0.25)
    shift_refz_decap_axes[0].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_decap_axes[1].plot(
            refz_decap_x,
            1e3 * shift_refz_decap_h_sample[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_decap_axes[1].axhspan(40, 55, color="0.7", alpha=0.12, label="calibrated window")
    shift_refz_decap_axes[1].set_ylabel("first stored $h$ (mV)")
    shift_refz_decap_axes[1].set_title("Buffered trim rails recover first-cycle activation")
    shift_refz_decap_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refz_decap_axes[1].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for branch_label, cap_label, dct, store in shift_refz_decap_traces:
        linestyle = "-" if cap_label == "0 pF" else "--" if cap_label == "100 pF" else ":"
        shift_refz_decap_axes[2].plot(
            1e6 * dct,
            1e3 * store,
            linestyle,
            label=f"{branch_label.split(', ')[1]}, {cap_label}",
        )
    shift_refz_decap_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_refz_decap_axes[2].set_xlim(2.35, 3.75)
    shift_refz_decap_axes[2].set_xlabel("time (us)")
    shift_refz_decap_axes[2].set_ylabel("stored $h$ (mV)")
    shift_refz_decap_axes[2].set_title("Passive local capacitance supplies the reset-edge trim charge")
    shift_refz_decap_axes[2].grid(True, alpha=0.25)
    shift_refz_decap_axes[2].legend(loc="upper left", ncol=2, fontsize="xx-small")
    shift_refz_decap_fig.tight_layout()
    save_plot(
        shift_refz_decap_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_ref_decap_ngspice",
    )

    shift_refz_recharge_fig, shift_refz_recharge_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.85, 0.85, 1.0]},
    )
    refz_recharge_x = np.array([cap_pf for _name, _label, cap_pf in shift_refz_recharge_cases])
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_recharge_axes[0].plot(
            refz_recharge_x,
            1e3 * shift_refz_recharge_max_trim_error[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_recharge_axes[0].axhline(6, color="0.4", linestyle=":", linewidth=0.9, label="6 mV trim-error gate")
    shift_refz_recharge_axes[0].set_ylabel("max trim error (mV)")
    shift_refz_recharge_axes[0].set_title("Local trim-reference caps recharge through a 100k source")
    shift_refz_recharge_axes[0].grid(True, axis="y", alpha=0.25)
    shift_refz_recharge_axes[0].legend(loc="upper right", ncol=2, fontsize="xx-small")
    cycle_x = np.arange(1, len(shift_reuse_h_times) + 1)
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        shift_refz_recharge_axes[1].plot(
            cycle_x,
            1e3 * shift_refz_recharge_h_samples[branch_idx, -1],
            "o-" if branch_idx == 0 else "s--",
            label=f"{branch_label.split(', ')[1]}, 250 pF",
        )
        shift_refz_recharge_axes[1].plot(
            cycle_x,
            1e3 * shift_refz_recharge_h_samples[branch_idx, 0],
            ":" if branch_idx == 0 else "-.",
            alpha=0.65,
            label=f"{branch_label.split(', ')[1]}, 0 pF",
        )
    shift_refz_recharge_axes[1].axhspan(40, 55, color="0.7", alpha=0.12, label="calibrated window")
    shift_refz_recharge_axes[1].set_xticks(cycle_x)
    shift_refz_recharge_axes[1].set_ylabel("stored $h$ (mV)")
    shift_refz_recharge_axes[1].set_title("The passive reservoir stays useful across repeated resets")
    shift_refz_recharge_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refz_recharge_axes[1].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for branch_label, cap_label, rct, store in shift_refz_recharge_traces:
        linestyle = "-" if cap_label == "0 pF" else "--" if cap_label == "100 pF" else ":"
        shift_refz_recharge_axes[2].plot(
            1e6 * rct,
            1e3 * store,
            linestyle,
            label=f"{branch_label.split(', ')[1]}, {cap_label}",
        )
    shift_refz_recharge_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_refz_recharge_axes[2].set_xlim(2.35, 7.25)
    shift_refz_recharge_axes[2].set_xlabel("time (us)")
    shift_refz_recharge_axes[2].set_ylabel("stored $h$ (mV)")
    shift_refz_recharge_axes[2].set_title("Repeated reads expose recharge and retention behavior")
    shift_refz_recharge_axes[2].grid(True, alpha=0.25)
    shift_refz_recharge_axes[2].legend(loc="upper left", ncol=2, fontsize="xx-small")
    shift_refz_recharge_fig.tight_layout()
    save_plot(
        shift_refz_recharge_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_ref_recharge_ngspice",
    )

    shift_refz_startup_fig, shift_refz_startup_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.85, 0.85, 1.0]},
    )
    refz_startup_x = np.arange(len(shift_refz_startup_cases))
    refz_startup_labels = [label for _name, label, _tau_count in shift_refz_startup_cases]
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_startup_axes[0].plot(
            refz_startup_x,
            1e3 * shift_refz_startup_trim_error[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_startup_axes[0].axhline(6, color="0.4", linestyle=":", linewidth=0.9, label="6 mV trim-error gate")
    shift_refz_startup_axes[0].set_xticks(refz_startup_x)
    shift_refz_startup_axes[0].set_xticklabels(refz_startup_labels)
    shift_refz_startup_axes[0].set_ylabel("trim error (mV)")
    shift_refz_startup_axes[0].set_title("Local trim reservoirs need startup precharge")
    shift_refz_startup_axes[0].grid(True, axis="y", alpha=0.25)
    shift_refz_startup_axes[0].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_startup_axes[1].plot(
            refz_startup_x,
            shift_refz_startup_gate_common[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_startup_axes[1].axhline(0.90, color="0.35", linestyle="--", linewidth=0.9, label="target common")
    shift_refz_startup_axes[1].axhline(0.84, color="0.5", linestyle=":", linewidth=0.9, label="startup gate")
    shift_refz_startup_axes[1].set_xticks(refz_startup_x)
    shift_refz_startup_axes[1].set_xticklabels(refz_startup_labels)
    shift_refz_startup_axes[1].set_ylabel("gate common (V)")
    shift_refz_startup_axes[1].set_title("Common-mode recovery follows the same local RC reservoir")
    shift_refz_startup_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refz_startup_axes[1].legend(loc="lower right", ncol=2, fontsize="xx-small")
    for branch_label, case_label, srt, ref_diff, gate_diff, gate_common in shift_refz_startup_traces:
        if case_label == "cold":
            linestyle = ":"
        elif case_label == "3 tau":
            linestyle = "--"
        else:
            linestyle = "-"
        shift_refz_startup_axes[2].plot(
            1e6 * srt,
            1e3 * gate_diff,
            linestyle,
            label=f"{branch_label.split(', ')[1]}, {case_label} gate diff",
        )
    shift_refz_startup_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_refz_startup_axes[2].set_xlim(2.35, 2.85)
    shift_refz_startup_axes[2].set_xlabel("time (us)")
    shift_refz_startup_axes[2].set_ylabel("shifted-gate trim (mV)")
    shift_refz_startup_axes[2].set_title("Real reset switches sample only the charge present in the local reservoir")
    shift_refz_startup_axes[2].grid(True, alpha=0.25)
    shift_refz_startup_axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        fontsize="xx-small",
    )
    shift_refz_startup_fig.tight_layout()
    save_plot(
        shift_refz_startup_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_ref_startup_ngspice",
    )

    shift_refz_precharge_fig, shift_refz_precharge_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.85, 0.85, 1.0]},
    )
    refz_precharge_x = np.array([width_ns for _name, _label, width_ns in shift_refz_precharge_cases])
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_precharge_axes[0].plot(
            refz_precharge_x,
            1e3 * shift_refz_precharge_trim_error[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_precharge_axes[0].axhline(6, color="0.4", linestyle=":", linewidth=0.9, label="6 mV trim-error gate")
    shift_refz_precharge_axes[0].set_ylabel("trim error (mV)")
    shift_refz_precharge_axes[0].set_title("A MOS startup-precharge path recovers cold trim reservoirs")
    shift_refz_precharge_axes[0].grid(True, axis="y", alpha=0.25)
    shift_refz_precharge_axes[0].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_precharge_axes[1].plot(
            refz_precharge_x,
            shift_refz_precharge_gate_common[branch_idx],
            marker,
            label=branch_label,
        )
    shift_refz_precharge_axes[1].axhline(0.90, color="0.35", linestyle="--", linewidth=0.9, label="target common")
    shift_refz_precharge_axes[1].axhline(0.84, color="0.5", linestyle=":", linewidth=0.9, label="startup gate")
    shift_refz_precharge_axes[1].set_ylabel("gate common (V)")
    shift_refz_precharge_axes[1].set_title("The startup switch restores the reservoir common mode too")
    shift_refz_precharge_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refz_precharge_axes[1].legend(loc="lower right", ncol=2, fontsize="xx-small")
    for branch_label, case_label, pct, _ref_diff, gate_diff, _gate_common in shift_refz_precharge_traces:
        if case_label == "none":
            linestyle = ":"
        elif case_label == "5 ns":
            linestyle = "-."
        elif case_label == "10 ns":
            linestyle = "--"
        else:
            linestyle = "-"
        shift_refz_precharge_axes[2].plot(
            1e6 * pct,
            1e3 * gate_diff,
            linestyle,
            label=f"{branch_label.split(', ')[1]}, {case_label}",
        )
    shift_refz_precharge_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_refz_precharge_axes[2].set_xlim(2.35, 2.85)
    shift_refz_precharge_axes[2].set_xlabel("time (us)")
    shift_refz_precharge_axes[2].set_ylabel("shifted-gate trim (mV)")
    shift_refz_precharge_axes[2].set_title("First reset samples the startup-charged local reservoir")
    shift_refz_precharge_axes[2].grid(True, alpha=0.25)
    shift_refz_precharge_axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        fontsize="xx-small",
    )
    shift_refz_precharge_fig.tight_layout()
    save_plot(
        shift_refz_precharge_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_ref_precharge_ngspice",
    )

    shift_refz_precharge_tuned_fig, shift_refz_precharge_tuned_axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.4),
        gridspec_kw={"height_ratios": [0.85, 0.85, 1.0]},
    )
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_precharge_tuned_axes[0].plot(
            refz_precharge_x,
            1e3 * shift_refz_precharge_tuned_trim_error[branch_idx],
            marker,
            label=f"{branch_label}, 0.80 Vcm",
        )
        shift_refz_precharge_tuned_axes[0].plot(
            refz_precharge_x,
            1e3 * shift_refz_precharge_trim_error[branch_idx],
            ":" if branch_idx == 0 else "-.",
            alpha=0.65,
            label=f"{branch_label}, 0.90 Vcm",
        )
    shift_refz_precharge_tuned_axes[0].axhline(
        6, color="0.4", linestyle=":", linewidth=0.9, label="6 mV trim-error gate"
    )
    shift_refz_precharge_tuned_axes[0].set_ylabel("trim error (mV)")
    shift_refz_precharge_tuned_axes[0].set_title("The MOS startup-precharge path also works at tuned 0.80 V common")
    shift_refz_precharge_tuned_axes[0].grid(True, axis="y", alpha=0.25)
    shift_refz_precharge_tuned_axes[0].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        marker = "o-" if branch_idx == 0 else "s--"
        shift_refz_precharge_tuned_axes[1].plot(
            refz_precharge_x,
            shift_refz_precharge_tuned_gate_common[branch_idx],
            marker,
            label=f"{branch_label}, 0.80 Vcm",
        )
    shift_refz_precharge_tuned_axes[1].axhline(0.80, color="0.35", linestyle="--", linewidth=0.9, label="0.80 V target")
    shift_refz_precharge_tuned_axes[1].axhline(0.74, color="0.5", linestyle=":", linewidth=0.9, label="startup gate")
    shift_refz_precharge_tuned_axes[1].set_ylabel("gate common (V)")
    shift_refz_precharge_tuned_axes[1].set_title("Tuned-common reservoirs recover the lower common-mode target")
    shift_refz_precharge_tuned_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refz_precharge_tuned_axes[1].legend(loc="lower right", ncol=2, fontsize="xx-small")
    for branch_label, case_label, pct, _ref_diff, gate_diff, _gate_common in shift_refz_precharge_tuned_traces:
        if case_label == "none":
            linestyle = ":"
        elif case_label == "5 ns":
            linestyle = "-."
        elif case_label == "10 ns":
            linestyle = "--"
        else:
            linestyle = "-"
        shift_refz_precharge_tuned_axes[2].plot(
            1e6 * pct,
            1e3 * gate_diff,
            linestyle,
            label=f"{branch_label.split(', ')[1]}, {case_label}",
        )
    shift_refz_precharge_tuned_axes[2].axhline(0, color="0.4", linewidth=0.8)
    shift_refz_precharge_tuned_axes[2].set_xlim(2.35, 2.85)
    shift_refz_precharge_tuned_axes[2].set_xlabel("time (us)")
    shift_refz_precharge_tuned_axes[2].set_ylabel("shifted-gate trim (mV)")
    shift_refz_precharge_tuned_axes[2].set_title("First reset samples the tuned startup-charged reservoir")
    shift_refz_precharge_tuned_axes[2].grid(True, alpha=0.25)
    shift_refz_precharge_tuned_axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        fontsize="xx-small",
    )
    shift_refz_precharge_tuned_fig.tight_layout()
    save_plot(
        shift_refz_precharge_tuned_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_ref_precharge_080_ngspice",
    )

    shift_refz_precharge_strength_fig, shift_refz_precharge_strength_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.9),
        gridspec_kw={"height_ratios": [1.0, 0.85]},
    )
    precharge_strength_x = np.array(
        [wpre_n_um / 300.0 for _name, _label, wpre_n_um, _wpre_p_um in shift_refz_precharge_strength_widths]
    )
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        for pulse_idx, (_pulse_name, pulse_label, _pulse_width_ns) in enumerate(shift_refz_precharge_strength_pulses):
            marker = "o-" if pulse_idx == 0 else "s--"
            alpha = 1.0 if branch_idx == 0 else 0.72
            shift_refz_precharge_strength_axes[0].plot(
                precharge_strength_x,
                1e3 * shift_refz_precharge_strength_trim_error[branch_idx, :, pulse_idx],
                marker,
                alpha=alpha,
                label=f"{branch_label.split(', ')[1]}, {pulse_label}",
            )
    shift_refz_precharge_strength_axes[0].axhline(
        6,
        color="0.4",
        linestyle=":",
        linewidth=0.9,
        label="6 mV trim-error gate",
    )
    shift_refz_precharge_strength_axes[0].set_xscale("log", base=2)
    shift_refz_precharge_strength_axes[0].set_xticks(precharge_strength_x)
    shift_refz_precharge_strength_axes[0].set_xticklabels(
        [label for _name, label, _wpre_n_um, _wpre_p_um in shift_refz_precharge_strength_widths]
    )
    shift_refz_precharge_strength_axes[0].set_ylabel("trim error (mV)")
    shift_refz_precharge_strength_axes[0].set_title("Startup precharge has a width/time tradeoff")
    shift_refz_precharge_strength_axes[0].grid(True, axis="y", alpha=0.25)
    shift_refz_precharge_strength_axes[0].legend(loc="upper right", ncol=2, fontsize="xx-small")
    for branch_idx, (_branch_name, branch_label, _nfp, _nfm, _trim) in enumerate(shift_reset_branch_specs):
        for pulse_idx, (_pulse_name, pulse_label, _pulse_width_ns) in enumerate(shift_refz_precharge_strength_pulses):
            marker = "o-" if pulse_idx == 0 else "s--"
            alpha = 1.0 if branch_idx == 0 else 0.72
            shift_refz_precharge_strength_axes[1].plot(
                precharge_strength_x,
                shift_refz_precharge_strength_common[branch_idx, :, pulse_idx],
                marker,
                alpha=alpha,
                label=f"{branch_label.split(', ')[1]}, {pulse_label}",
            )
    shift_refz_precharge_strength_axes[1].axhline(0.90, color="0.35", linestyle="--", linewidth=0.9)
    shift_refz_precharge_strength_axes[1].axhline(0.84, color="0.5", linestyle=":", linewidth=0.9)
    shift_refz_precharge_strength_axes[1].set_xscale("log", base=2)
    shift_refz_precharge_strength_axes[1].set_xticks(precharge_strength_x)
    shift_refz_precharge_strength_axes[1].set_xticklabels(
        [label for _name, label, _wpre_n_um, _wpre_p_um in shift_refz_precharge_strength_widths]
    )
    shift_refz_precharge_strength_axes[1].set_xlabel("startup precharge TG width scale")
    shift_refz_precharge_strength_axes[1].set_ylabel("gate common (V)")
    shift_refz_precharge_strength_axes[1].set_title("Common-mode restoration tracks the same strength margin")
    shift_refz_precharge_strength_axes[1].grid(True, axis="y", alpha=0.25)
    shift_refz_precharge_strength_fig.tight_layout()
    save_plot(
        shift_refz_precharge_strength_fig,
        "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_pair_96u_shifted_gate_reset_ref_precharge_strength_ngspice",
    )

    tail_bias_fig, tail_bias_axes = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw={"height_ratios": [1.0, 0.8]})
    for label, tbt, load, store in tail_bias_traces:
        if label in {"0.70 V", "0.90 V", "0.95 V", "1.15 V", "1.25 V"}:
            tail_bias_axes[0].plot(1e6 * tbt, 1e3 * load, label=f"{label} load")
            tail_bias_axes[0].plot(1e6 * tbt, 1e3 * store, "--", label=f"{label} store")
    tail_bias_axes[0].axhline(0, color="0.4", linewidth=0.8)
    tail_bias_axes[0].axvline(3.33, color="0.25", linestyle="--", linewidth=0.9, alpha=0.6, label="guard off")
    tail_bias_axes[0].set_xlim(3.05, 3.65)
    tail_bias_axes[0].set_ylabel("activation differential (mV)")
    tail_bias_axes[0].set_title("Isolated forward tail bias is only a small trim in this sizing")
    tail_bias_axes[0].grid(True, alpha=0.25)
    tail_bias_axes[0].legend(loc="upper left", ncol=2, fontsize="small")
    tail_bias_x = np.arange(len(tail_bias_labels))
    tail_bias_axes[1].bar(tail_bias_x - 0.24, 1e3 * tail_bias_preact_samples, width=0.24, label="$z^- - z^+$")
    tail_bias_axes[1].bar(tail_bias_x, 1e3 * tail_bias_load_samples, width=0.24, label="forward load")
    tail_bias_axes[1].bar(tail_bias_x + 0.24, 1e3 * tail_bias_store_samples, width=0.24, label="stored $h^- - h^+$")
    tail_bias_axes[1].axhline(0, color="0.4", linewidth=0.8)
    tail_bias_axes[1].set_xticks(tail_bias_x)
    tail_bias_axes[1].set_xticklabels(tail_bias_labels)
    tail_bias_axes[1].set_xlabel("forward-pair tail bias")
    tail_bias_axes[1].set_ylabel("sampled differential (mV)")
    tail_bias_axes[1].set_title("Width, not tail-bias overdrive, provides the large gain range")
    tail_bias_axes[1].grid(True, axis="y", alpha=0.25)
    tail_bias_axes[1].legend(loc="lower right", ncol=3, fontsize="small")
    tail_bias_fig.tight_layout()
    save_plot(tail_bias_fig, "mos_hidden_writer_restored_gate_hybrid_update_forward_guard_forward_tail_bias_ngspice")

    hybrid_repeated_deck = f"""
* Hybrid restored-enable/analog-error writer repeated-pulse accumulation check.
* One stored r+ hidden-error rail and one activation gate drive the same
* persistent weight capacitors through a train of pacc pulses.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 8.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 8.0u)
VPACC_HYR paccn_hyr 0 PULSE(1.8 0 1.55u 20n 20n 0.16u 0.45u)
VHM_POS hm_pos 0 0.92

VZPP_HYR zpp_hyr 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYR zmm_hyr 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYR zpm_hyr 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYR zmp_hyr 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYR hpp_hyr hpp_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYR hpm_hyr hpm_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYR hpp_hyr zpp_hyr tailp_hyr 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYR hpm_hyr zmm_hyr tailp_hyr 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYR tailp_hyr vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYR hmp_hyr hmp_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYR hmm_hyr hmm_hyr vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYR hmp_hyr zpm_hyr tailm_hyr 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYR hmm_hyr zmp_hyr tailm_hyr 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYR tailm_hyr vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYR cdp_rp_hyr 0 {{CERR}} IC=1.04
CDM_RP_HYR cdm_rp_hyr 0 {{CERR}} IC=1.04
RDP_RP_HYR cdp_rp_hyr 0 50G
RDM_RP_HYR cdm_rp_hyr 0 50G
{sign_store_path("hpm_hyr", "rp", "cdp_rp_hyr", "hyrrp1")}
{sign_store_path("hmp_hyr", "rp", "cdp_rp_hyr", "hyrrp2")}
{sign_store_path("hpp_hyr", "rp", "cdm_rp_hyr", "hyrrp3")}
{sign_store_path("hmm_hyr", "rp", "cdm_rp_hyr", "hyrrp4")}

MPRP_CDP_HYR rgp_rp_hyr cdp_rp_hyr vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYR rgp_rp_hyr cdp_rp_hyr 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYR rgm_rp_hyr cdm_rp_hyr vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYR rgm_rp_hyr cdm_rp_hyr 0 0 NMOS L={{LCH}} W={{WRESTN}}

CWP_HYR wp_hyr 0 {{CWRITE}} IC=0.85
CWM_HYR wm_hyr 0 {{CWRITE}} IC=0.85
MWP_HYR_A vdd paccn_hyr n_wp_hyr_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYR_B n_wp_hyr_a hm_pos n_wp_hyr_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYR_C n_wp_hyr_b rgp_rp_hyr n_wp_hyr_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_HYR_D n_wp_hyr_c cdm_rp_hyr wp_hyr vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYR_A vdd paccn_hyr n_wm_hyr_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYR_B n_wm_hyr_a hm_pos n_wm_hyr_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYR_C n_wm_hyr_b rgm_rp_hyr n_wm_hyr_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_HYR_D n_wm_hyr_c cdp_rp_hyr wm_hyr vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 5n 4.65u uic
wrdata mos_hidden_writer_restored_gate_hybrid_repeated.dat v(cdp_rp_hyr) v(cdm_rp_hyr) v(rgp_rp_hyr) v(rgm_rp_hyr) v(wp_hyr) v(wm_hyr) v(paccn_hyr) v(pbwd)
quit
.endc
.end
"""
    hybrid_repeated_data = run_ngspice(
        hybrid_repeated_deck,
        "mos_hidden_writer_restored_gate_hybrid_repeated",
    )
    hyrt, hyr_cols = load_wrdata(hybrid_repeated_data, 8)

    def hyrat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyrt - time_s))])

    hyr_hidden = hyr_cols[0] - hyr_cols[1]
    hyr_selected_gate = hyr_cols[2]
    hyr_complement_gate = hyr_cols[3]
    hyr_wp = hyr_cols[4]
    hyr_wm = hyr_cols[5]
    hyr_diff = hyr_wp - hyr_wm
    hyr_sample_times = np.array([1.78, 2.23, 2.68, 3.13, 3.58, 4.03, 4.48]) * 1e-6
    hyr_steps = np.array([hyrat(ts, hyr_diff) for ts in hyr_sample_times])
    hyr_wp_steps = np.array([hyrat(ts, hyr_wp) - hyrat(1.45e-6, hyr_wp) for ts in hyr_sample_times])
    hyr_wm_steps = np.array([hyrat(ts, hyr_wm) - hyrat(1.45e-6, hyr_wm) for ts in hyr_sample_times])
    hyr_increments = np.diff(hyr_steps)
    require(hyrat(1.35e-6, hyr_hidden) > 0.07, "hybrid repeated deck should store a positive hidden error")
    require(hyrat(1.45e-6, hyr_selected_gate) < 0.30, "hybrid repeated selected restored gate should be low")
    require(hyrat(1.45e-6, hyr_complement_gate) > 1.60, "hybrid repeated complement restored gate should be high")
    require(np.all(np.diff(hyr_steps) > 0.010), "hybrid repeated write should accumulate monotonically")
    require(hyr_steps[-1] > 0.080, "hybrid repeated write should build a large readable differential")
    require(hyr_steps[-1] < 0.20, "hybrid repeated write should remain incremental across seven pulses")
    require(np.max(hyr_wm_steps) < 5e-4, "hybrid repeated complement rail should remain suppressed")
    require(np.min(hyr_increments) > 0.45 * np.max(hyr_increments), "hybrid repeated increments should not collapse over seven pulses")
    require(abs(hyrat(4.60e-6, hyr_diff) - hyr_steps[-1]) < 5e-4, "hybrid repeated write should hold after pulse train")

    hyr_fig, hyr_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hyr_axes[0].plot(1e6 * hyrt, hyr_hidden, label="stored $r^+$ hidden error")
    hyr_axes[0].plot(1e6 * hyrt, hyr_selected_gate, label="selected restored gate")
    hyr_axes[0].plot(1e6 * hyrt, hyr_complement_gate, label="complement restored gate")
    hyr_axes[0].plot(1e6 * hyrt, hyr_cols[6] / 20.0, color="0.5", alpha=0.35, label="$pacc_n/20$")
    hyr_axes[0].set_ylabel("voltage (V)")
    hyr_axes[0].set_title("Hybrid repeated writer reuses one stored hidden-error rail")
    hyr_axes[0].grid(True, alpha=0.25)
    hyr_axes[0].legend(loc="center right", fontsize="small")
    hyr_axes[1].plot(1e6 * hyrt, hyr_wp - hyrat(1.45e-6, hyr_wp), label="selected $W^+$ step")
    hyr_axes[1].plot(1e6 * hyrt, hyr_wm - hyrat(1.45e-6, hyr_wm), label="complement $W^-$ step")
    hyr_axes[1].plot(1e6 * hyrt, hyr_diff, label="$W^+ - W^-$")
    hyr_axes[1].plot(1e6 * hyr_sample_times, hyr_steps, "o", color="black", label="post-pulse samples")
    hyr_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyr_axes[1].set_xlabel("time (us)")
    hyr_axes[1].set_ylabel("writer step (V)")
    hyr_axes[1].set_title("Repeated hybrid pacc pulses accumulate while complement stays off")
    hyr_axes[1].grid(True, alpha=0.25)
    hyr_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    hyr_fig.tight_layout()
    save_plot(hyr_fig, "mos_hidden_writer_restored_gate_hybrid_repeated_ngspice")

    hybrid_alternating_deck = f"""
* Hybrid restored-enable/analog-error writer alternating-sign cancellation check.
* One r+ store and one r- store drive two phase-separated writer stacks into
* the same persistent W+/W- capacitor pair.  The first three pacc pulses write
* W+, then four opposite-sign pacc pulses write W- and must cancel/cross the
* same physical signed weight.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 8.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 8.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 8.0u)
VPACC_POS_HYC paccn_pos_hyc 0 PWL(0 1.8 1.55u 1.8 1.57u 0 1.73u 0 1.75u 1.8 2.00u 1.8 2.02u 0 2.18u 0 2.20u 1.8 2.45u 1.8 2.47u 0 2.63u 0 2.65u 1.8 5u 1.8)
VPACC_NEG_HYC paccn_neg_hyc 0 PWL(0 1.8 2.90u 1.8 2.92u 0 3.08u 0 3.10u 1.8 3.35u 1.8 3.37u 0 3.53u 0 3.55u 1.8 3.80u 1.8 3.82u 0 3.98u 0 4.00u 1.8 4.25u 1.8 4.27u 0 4.43u 0 4.45u 1.8 5u 1.8)
VHM_POS hm_pos_hyc 0 0.92

VZPP_HYC zpp_hyc 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYC zmm_hyc 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYC zpm_hyc 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYC zmp_hyc 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYC hpp_hyc hpp_hyc vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYC hpm_hyc hpm_hyc vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYC hpp_hyc zpp_hyc tailp_hyc 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYC hpm_hyc zmm_hyc tailp_hyc 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYC tailp_hyc vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYC hmp_hyc hmp_hyc vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYC hmm_hyc hmm_hyc vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYC hmp_hyc zpm_hyc tailm_hyc 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYC hmm_hyc zmp_hyc tailm_hyc 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYC tailm_hyc vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYC cdp_rp_hyc 0 {{CERR}} IC=1.04
CDM_RP_HYC cdm_rp_hyc 0 {{CERR}} IC=1.04
CDP_RM_HYC cdp_rm_hyc 0 {{CERR}} IC=1.04
CDM_RM_HYC cdm_rm_hyc 0 {{CERR}} IC=1.04
RDP_RP_HYC cdp_rp_hyc 0 50G
RDM_RP_HYC cdm_rp_hyc 0 50G
RDP_RM_HYC cdp_rm_hyc 0 50G
RDM_RM_HYC cdm_rm_hyc 0 50G
{sign_store_path("hpm_hyc", "rp", "cdp_rp_hyc", "hycrp1")}
{sign_store_path("hmp_hyc", "rp", "cdp_rp_hyc", "hycrp2")}
{sign_store_path("hpp_hyc", "rp", "cdm_rp_hyc", "hycrp3")}
{sign_store_path("hmm_hyc", "rp", "cdm_rp_hyc", "hycrp4")}
{sign_store_path("hpp_hyc", "rm", "cdp_rm_hyc", "hycrm1")}
{sign_store_path("hmm_hyc", "rm", "cdp_rm_hyc", "hycrm2")}
{sign_store_path("hpm_hyc", "rm", "cdm_rm_hyc", "hycrm3")}
{sign_store_path("hmp_hyc", "rm", "cdm_rm_hyc", "hycrm4")}

MPRP_CDP_HYC rgp_rp_hyc cdp_rp_hyc vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYC rgp_rp_hyc cdp_rp_hyc 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYC rgm_rp_hyc cdm_rp_hyc vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYC rgm_rp_hyc cdm_rp_hyc 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDP_HYC rgp_rm_hyc cdp_rm_hyc vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDP_HYC rgp_rm_hyc cdp_rm_hyc 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDM_HYC rgm_rm_hyc cdm_rm_hyc vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDM_HYC rgm_rm_hyc cdm_rm_hyc 0 0 NMOS L={{LCH}} W={{WRESTN}}

CWP_HYC wp_hyc 0 {{CWRITE}} IC=0.85
CWM_HYC wm_hyc 0 {{CWRITE}} IC=0.85
MWP_POS_HYC_A vdd paccn_pos_hyc n_wp_pos_hyc_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYC_B n_wp_pos_hyc_a hm_pos_hyc n_wp_pos_hyc_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYC_C n_wp_pos_hyc_b rgp_rp_hyc n_wp_pos_hyc_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYC_D n_wp_pos_hyc_c cdm_rp_hyc wp_hyc vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYC_A vdd paccn_pos_hyc n_wm_pos_hyc_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYC_B n_wm_pos_hyc_a hm_pos_hyc n_wm_pos_hyc_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYC_C n_wm_pos_hyc_b rgm_rp_hyc n_wm_pos_hyc_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYC_D n_wm_pos_hyc_c cdp_rp_hyc wm_hyc vdd PMOS L={{LCH}} W={{WWRITE}}

MWP_NEG_HYC_A vdd paccn_neg_hyc n_wp_neg_hyc_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYC_B n_wp_neg_hyc_a hm_pos_hyc n_wp_neg_hyc_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYC_C n_wp_neg_hyc_b rgp_rm_hyc n_wp_neg_hyc_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYC_D n_wp_neg_hyc_c cdm_rm_hyc wp_hyc vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYC_A vdd paccn_neg_hyc n_wm_neg_hyc_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYC_B n_wm_neg_hyc_a hm_pos_hyc n_wm_neg_hyc_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYC_C n_wm_neg_hyc_b rgm_rm_hyc n_wm_neg_hyc_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYC_D n_wm_neg_hyc_c cdp_rm_hyc wm_hyc vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 5n 4.75u uic
wrdata mos_hidden_writer_restored_gate_hybrid_alternating.dat v(cdp_rp_hyc) v(cdm_rp_hyc) v(cdp_rm_hyc) v(cdm_rm_hyc) v(rgp_rp_hyc) v(rgm_rp_hyc) v(rgp_rm_hyc) v(rgm_rm_hyc) v(wp_hyc) v(wm_hyc) v(paccn_pos_hyc) v(paccn_neg_hyc) v(pbwd)
quit
.endc
.end
"""
    hybrid_alternating_data = run_ngspice(
        hybrid_alternating_deck,
        "mos_hidden_writer_restored_gate_hybrid_alternating",
    )
    hyct, hyc_cols = load_wrdata(hybrid_alternating_data, 13)

    def hycat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyct - time_s))])

    hyc_hidden_pos = hyc_cols[0] - hyc_cols[1]
    hyc_hidden_neg = hyc_cols[2] - hyc_cols[3]
    hyc_pos_selected_gate = hyc_cols[4]
    hyc_pos_complement_gate = hyc_cols[5]
    hyc_neg_complement_gate = hyc_cols[6]
    hyc_neg_selected_gate = hyc_cols[7]
    hyc_wp = hyc_cols[8]
    hyc_wm = hyc_cols[9]
    hyc_diff = hyc_wp - hyc_wm
    hyc_initial_wp = hycat(1.45e-6, hyc_wp)
    hyc_initial_wm = hycat(1.45e-6, hyc_wm)
    hyc_sample_times = np.array([1.78, 2.23, 2.68, 3.13, 3.58, 4.03, 4.48]) * 1e-6
    hyc_steps = np.array([hycat(ts, hyc_diff) for ts in hyc_sample_times])
    hyc_wp_steps = np.array([hycat(ts, hyc_wp) - hyc_initial_wp for ts in hyc_sample_times])
    hyc_wm_steps = np.array([hycat(ts, hyc_wm) - hyc_initial_wm for ts in hyc_sample_times])
    hyc_pos_phase_steps = hyc_steps[:3]
    hyc_neg_phase_steps = hyc_steps[3:]
    require(hycat(1.35e-6, hyc_hidden_pos) > 0.07, "hybrid alternating r+ store should be positive")
    require(hycat(1.35e-6, hyc_hidden_neg) < -0.07, "hybrid alternating r- store should be negative")
    require(abs(hycat(1.35e-6, hyc_hidden_pos + hyc_hidden_neg)) < 0.003, "hybrid alternating hidden-error stores should be symmetric")
    require(hycat(1.45e-6, hyc_pos_selected_gate) < 0.30, "hybrid alternating r+ selected restored gate should be low")
    require(hycat(1.45e-6, hyc_pos_complement_gate) > 1.60, "hybrid alternating r+ complement restored gate should be high")
    require(hycat(1.45e-6, hyc_neg_selected_gate) < 0.30, "hybrid alternating r- selected restored gate should be low")
    require(hycat(1.45e-6, hyc_neg_complement_gate) > 1.60, "hybrid alternating r- complement restored gate should be high")
    require(np.all(np.diff(hyc_pos_phase_steps) > 0.010), "hybrid alternating positive pulses should build positive signed weight")
    require(np.max(hyc_wm_steps[:3]) < 5e-4, "hybrid alternating W- rail should stay quiet during positive pulses")
    require(np.all(np.diff(hyc_neg_phase_steps) < -0.010), "hybrid alternating negative pulses should reduce signed weight")
    require(hyc_neg_phase_steps[0] < hyc_pos_phase_steps[-1] - 0.010, "first negative pulse should partially cancel the positive state")
    require(hyc_neg_phase_steps[-1] < -0.006, "extra negative pulse should cross the same signed weight below zero")
    require(np.max(np.abs(hyc_wp_steps[3:] - hyc_wp_steps[2])) < 5e-4, "hybrid alternating W+ rail should hold during negative pulses")
    require(np.min(np.diff(hyc_wm_steps[2:])) > 0.010, "hybrid alternating W- rail should accumulate during negative pulses")
    require(abs(hycat(4.65e-6, hyc_diff) - hyc_steps[-1]) < 5e-4, "hybrid alternating signed weight should hold after final pulse")

    hyc_fig, hyc_axes = plt.subplots(2, 1, figsize=(7.2, 5.8))
    hyc_axes[0].plot(1e6 * hyct, hyc_hidden_pos, label="stored $r^+$")
    hyc_axes[0].plot(1e6 * hyct, hyc_hidden_neg, label="stored $r^-$")
    hyc_axes[0].plot(1e6 * hyct, hyc_pos_selected_gate, label="$r^+$ selected gate")
    hyc_axes[0].plot(1e6 * hyct, hyc_neg_selected_gate, label="$r^-$ selected gate")
    hyc_axes[0].plot(1e6 * hyct, hyc_cols[10] / 20.0, color="0.45", alpha=0.35, label="$pacc^+_n/20$")
    hyc_axes[0].plot(1e6 * hyct, hyc_cols[11] / 20.0, color="0.15", alpha=0.25, label="$pacc^-_n/20$")
    hyc_axes[0].axhline(0, color="0.4", linewidth=0.8)
    hyc_axes[0].set_ylabel("voltage (V)")
    hyc_axes[0].set_title("Hybrid alternating writer stores both hidden-error signs")
    hyc_axes[0].grid(True, alpha=0.25)
    hyc_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyc_axes[1].plot(1e6 * hyct, hyc_wp - hyc_initial_wp, label="$W^+$ step")
    hyc_axes[1].plot(1e6 * hyct, hyc_wm - hyc_initial_wm, label="$W^-$ step")
    hyc_axes[1].plot(1e6 * hyct, hyc_diff, label="$W^+ - W^-$")
    hyc_axes[1].plot(1e6 * hyc_sample_times, hyc_steps, "o", color="black", label="post-pulse samples")
    hyc_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyc_axes[1].set_xlabel("time (us)")
    hyc_axes[1].set_ylabel("writer step (V)")
    hyc_axes[1].set_title("Same weight pair accumulates, cancels, and crosses sign")
    hyc_axes[1].grid(True, alpha=0.25)
    hyc_axes[1].legend(loc="upper left", ncol=2, fontsize="small")
    hyc_fig.tight_layout()
    save_plot(hyc_fig, "mos_hidden_writer_restored_gate_hybrid_alternating_ngspice")

    hybrid_readback_deck = f"""
* Hybrid restored-enable/analog-error writer readback and retention check.
* The same r+/r- stores and restored-select writer stacks alternately write one
* W+/W- capacitor pair.  The written rails are continuously connected as MOS
* synapse tail gates during and after the write train.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VPBWD pbwd 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 12.0u)
VRP rp 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 12.0u)
VRM rm 0 PULSE(0 1.8 0.45u 20n 20n 0.80u 12.0u)
VPACC_POS_HYD paccn_pos_hyd 0 PWL(0 1.8 1.55u 1.8 1.57u 0 1.73u 0 1.75u 1.8 2.00u 1.8 2.02u 0 2.18u 0 2.20u 1.8 2.45u 1.8 2.47u 0 2.63u 0 2.65u 1.8 10u 1.8)
VPACC_NEG_HYD paccn_neg_hyd 0 PWL(0 1.8 2.90u 1.8 2.92u 0 3.08u 0 3.10u 1.8 3.35u 1.8 3.37u 0 3.53u 0 3.55u 1.8 3.80u 1.8 3.82u 0 3.98u 0 4.00u 1.8 4.25u 1.8 4.27u 0 4.43u 0 4.45u 1.8 10u 1.8)
VHM_POS_HYD hm_pos_hyd 0 0.92

VZPP_HYD zpp_hyd 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYD zmm_hyd 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYD zpm_hyd 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYD zmp_hyd 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYD hpp_hyd hpp_hyd vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYD hpm_hyd hpm_hyd vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYD hpp_hyd zpp_hyd tailp_hyd 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYD hpm_hyd zmm_hyd tailp_hyd 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYD tailp_hyd vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYD hmp_hyd hmp_hyd vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYD hmm_hyd hmm_hyd vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYD hmp_hyd zpm_hyd tailm_hyd 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYD hmm_hyd zmp_hyd tailm_hyd 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYD tailm_hyd vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_RP_HYD cdp_rp_hyd 0 {{CERR}} IC=1.04
CDM_RP_HYD cdm_rp_hyd 0 {{CERR}} IC=1.04
CDP_RM_HYD cdp_rm_hyd 0 {{CERR}} IC=1.04
CDM_RM_HYD cdm_rm_hyd 0 {{CERR}} IC=1.04
RDP_RP_HYD cdp_rp_hyd 0 50G
RDM_RP_HYD cdm_rp_hyd 0 50G
RDP_RM_HYD cdp_rm_hyd 0 50G
RDM_RM_HYD cdm_rm_hyd 0 50G
{sign_store_path("hpm_hyd", "rp", "cdp_rp_hyd", "hydrp1")}
{sign_store_path("hmp_hyd", "rp", "cdp_rp_hyd", "hydrp2")}
{sign_store_path("hpp_hyd", "rp", "cdm_rp_hyd", "hydrp3")}
{sign_store_path("hmm_hyd", "rp", "cdm_rp_hyd", "hydrp4")}
{sign_store_path("hpp_hyd", "rm", "cdp_rm_hyd", "hydrm1")}
{sign_store_path("hmm_hyd", "rm", "cdp_rm_hyd", "hydrm2")}
{sign_store_path("hpm_hyd", "rm", "cdm_rm_hyd", "hydrm3")}
{sign_store_path("hmp_hyd", "rm", "cdm_rm_hyd", "hydrm4")}

MPRP_CDP_HYD rgp_rp_hyd cdp_rp_hyd vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYD rgp_rp_hyd cdp_rp_hyd 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYD rgm_rp_hyd cdm_rp_hyd vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYD rgm_rp_hyd cdm_rp_hyd 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDP_HYD rgp_rm_hyd cdp_rm_hyd vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDP_HYD rgp_rm_hyd cdp_rm_hyd 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRM_CDM_HYD rgm_rm_hyd cdm_rm_hyd vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRM_CDM_HYD rgm_rm_hyd cdm_rm_hyd 0 0 NMOS L={{LCH}} W={{WRESTN}}

CWP_HYD wp_hyd 0 {{CWRITE}} IC=0.85
CWM_HYD wm_hyd 0 {{CWRITE}} IC=0.85
MWP_POS_HYD_A vdd paccn_pos_hyd n_wp_pos_hyd_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYD_B n_wp_pos_hyd_a hm_pos_hyd n_wp_pos_hyd_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYD_C n_wp_pos_hyd_b rgp_rp_hyd n_wp_pos_hyd_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYD_D n_wp_pos_hyd_c cdm_rp_hyd wp_hyd vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYD_A vdd paccn_pos_hyd n_wm_pos_hyd_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYD_B n_wm_pos_hyd_a hm_pos_hyd n_wm_pos_hyd_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYD_C n_wm_pos_hyd_b rgm_rp_hyd n_wm_pos_hyd_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYD_D n_wm_pos_hyd_c cdp_rp_hyd wm_hyd vdd PMOS L={{LCH}} W={{WWRITE}}

MWP_NEG_HYD_A vdd paccn_neg_hyd n_wp_neg_hyd_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYD_B n_wp_neg_hyd_a hm_pos_hyd n_wp_neg_hyd_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYD_C n_wp_neg_hyd_b rgp_rm_hyd n_wp_neg_hyd_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYD_D n_wp_neg_hyd_c cdm_rm_hyd wp_hyd vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYD_A vdd paccn_neg_hyd n_wm_neg_hyd_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYD_B n_wm_neg_hyd_a hm_pos_hyd n_wm_neg_hyd_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYD_C n_wm_neg_hyd_b rgm_rm_hyd n_wm_neg_hyd_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYD_D n_wm_neg_hyd_c cdp_rm_hyd wm_hyd vdd PMOS L={{LCH}} W={{WWRITE}}

VXP_HYD xp_hyd 0 1.15
VXM_HYD xm_hyd 0 0.65
VZPP_READ_HYD zpp_read_hyd 0 1.8
VZMP_READ_HYD zmp_read_hyd 0 1.8
MPP_READ_HYD zpp_read_hyd xp_hyd tailp_read_hyd 0 NMOS L={{LCH}} W={{WN}}
MPM_READ_HYD zmp_read_hyd xm_hyd tailp_read_hyd 0 NMOS L={{LCH}} W={{WN}}
MTP_READ_HYD tailp_read_hyd wp_hyd 0 0 NMOS L={{LCH}} W=12u
VZPN_READ_HYD zpn_read_hyd 0 1.8
VZMN_READ_HYD zmn_read_hyd 0 1.8
MNP_READ_HYD zpn_read_hyd xm_hyd tailn_read_hyd 0 NMOS L={{LCH}} W={{WN}}
MNM_READ_HYD zmn_read_hyd xp_hyd tailn_read_hyd 0 NMOS L={{LCH}} W={{WN}}
MTN_READ_HYD tailn_read_hyd wm_hyd 0 0 NMOS L={{LCH}} W=12u

.control
set noaskquit
tran 10n 9.8u uic
wrdata mos_hidden_writer_restored_gate_hybrid_readback.dat v(cdp_rp_hyd) v(cdm_rp_hyd) v(cdp_rm_hyd) v(cdm_rm_hyd) v(rgp_rp_hyd) v(rgm_rp_hyd) v(rgp_rm_hyd) v(rgm_rm_hyd) v(wp_hyd) v(wm_hyd) i(VZPP_READ_HYD) i(VZMP_READ_HYD) i(VZPN_READ_HYD) i(VZMN_READ_HYD) v(paccn_pos_hyd) v(paccn_neg_hyd) v(pbwd)
quit
.endc
.end
"""
    hybrid_readback_data = run_ngspice(
        hybrid_readback_deck,
        "mos_hidden_writer_restored_gate_hybrid_readback",
    )
    hydt, hyd_cols = load_wrdata(hybrid_readback_data, 17)

    def hydat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hydt - time_s))])

    hyd_hidden_pos = hyd_cols[0] - hyd_cols[1]
    hyd_hidden_neg = hyd_cols[2] - hyd_cols[3]
    hyd_wp = hyd_cols[8]
    hyd_wm = hyd_cols[9]
    hyd_weight = hyd_wp - hyd_wm
    hyd_pos_read = hyd_cols[11] - hyd_cols[10]
    hyd_neg_read = hyd_cols[13] - hyd_cols[12]
    hyd_net_read = hyd_pos_read + hyd_neg_read
    hyd_baseline_read = hydat(1.45e-6, hyd_net_read)
    hyd_positive_read = hydat(2.75e-6, hyd_net_read)
    hyd_near_cancel_read = hydat(4.05e-6, hyd_net_read)
    hyd_negative_read = hydat(4.55e-6, hyd_net_read)
    hyd_hold_start = 4.55e-6
    hyd_hold_end = 9.5e-6
    require(hydat(1.35e-6, hyd_hidden_pos) > 0.07, "hybrid readback r+ store should be positive")
    require(hydat(1.35e-6, hyd_hidden_neg) < -0.07, "hybrid readback r- store should be negative")
    require(abs(hydat(1.35e-6, hyd_hidden_pos + hyd_hidden_neg)) < 0.003, "hybrid readback hidden-error stores should be symmetric")
    require(hydat(1.45e-6, hyd_cols[4]) < 0.30, "hybrid readback r+ selected gate should be low")
    require(hydat(1.45e-6, hyd_cols[5]) > 1.60, "hybrid readback r+ complement gate should be high")
    require(hydat(1.45e-6, hyd_cols[7]) < 0.30, "hybrid readback r- selected gate should be low")
    require(hydat(1.45e-6, hyd_cols[6]) > 1.60, "hybrid readback r- complement gate should be high")
    require(abs(hyd_baseline_read) < 1e-6, "hybrid readback should start balanced")
    require(hyd_positive_read > hyd_baseline_read + 8e-6, "hybrid W+ pulses should read back as a positive contribution")
    require(abs(hyd_near_cancel_read) < 2e-6, "hybrid alternating writes should read back near zero at cancellation")
    require(hyd_negative_read < hyd_baseline_read - 4e-6, "hybrid extra W- pulse should read back as a negative contribution")
    require(hydat(2.75e-6, hyd_weight) > 0.025, "hybrid readback W+ pulses should build positive stored weight")
    require(hydat(hyd_hold_start, hyd_weight) < -0.006, "hybrid readback final stored weight should be negative")
    require(abs(hydat(hyd_hold_end, hyd_wp) - hydat(hyd_hold_start, hyd_wp)) < 1e-5, "hybrid readback should not disturb W+ cap during hold")
    require(abs(hydat(hyd_hold_end, hyd_wm) - hydat(hyd_hold_start, hyd_wm)) < 1e-5, "hybrid readback should not disturb W- cap during hold")
    require(abs(hydat(hyd_hold_end, hyd_net_read) - hydat(hyd_hold_start, hyd_net_read)) < 0.5e-6, "hybrid read current should hold after writes")

    hyd_fig, hyd_axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    hyd_axes[0].plot(1e6 * hydt, hyd_hidden_pos, label="stored $r^+$")
    hyd_axes[0].plot(1e6 * hydt, hyd_hidden_neg, label="stored $r^-$")
    hyd_axes[0].plot(1e6 * hydt, hyd_cols[14] / 20.0, color="0.45", alpha=0.35, label="$pacc^+_n/20$")
    hyd_axes[0].plot(1e6 * hydt, hyd_cols[15] / 20.0, color="0.15", alpha=0.25, label="$pacc^-_n/20$")
    hyd_axes[0].axhline(0, color="0.4", linewidth=0.8)
    hyd_axes[0].set_ylabel("voltage (V)")
    hyd_axes[0].set_title("Hybrid readback deck stores both writer signs")
    hyd_axes[0].grid(True, alpha=0.25)
    hyd_axes[0].legend(loc="center right", ncol=2, fontsize="small")
    hyd_axes[1].plot(1e6 * hydt, hyd_wp - hydat(1.45e-6, hyd_wp), label="$W^+$ step")
    hyd_axes[1].plot(1e6 * hydt, hyd_wm - hydat(1.45e-6, hyd_wm), label="$W^-$ step")
    hyd_axes[1].plot(1e6 * hydt, hyd_weight, label="$W^+ - W^-$")
    hyd_axes[1].axvspan(4.55, 9.5, color="0.8", alpha=0.18, label="read hold")
    hyd_axes[1].axhline(0, color="0.4", linewidth=0.8)
    hyd_axes[1].set_ylabel("weight step (V)")
    hyd_axes[1].set_title("Written weight caps hold while driving MOS synapse gates")
    hyd_axes[1].grid(True, alpha=0.25)
    hyd_axes[1].legend(loc="upper right", ncol=2, fontsize="small")
    hyd_axes[2].plot(1e6 * hydt, 1e6 * hyd_net_read, label="net signed readback")
    hyd_axes[2].plot(1e6 * hydt, 1e6 * hyd_pos_read, "--", label="$W^+$ read component")
    hyd_axes[2].plot(1e6 * hydt, 1e6 * hyd_neg_read, ":", label="$W^-$ read component")
    hyd_axes[2].axvspan(4.55, 9.5, color="0.8", alpha=0.18, label="read hold")
    hyd_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hyd_axes[2].set_xlabel("time (us)")
    hyd_axes[2].set_ylabel("read current (uA)")
    hyd_axes[2].set_title("Hybrid-written rails read back, cancel, reverse, and retain sign")
    hyd_axes[2].grid(True, alpha=0.25)
    hyd_axes[2].legend(loc="upper right", ncol=2, fontsize="small")
    hyd_fig.tight_layout()
    save_plot(hyd_fig, "mos_hidden_writer_restored_gate_hybrid_readback_ngspice")

    hybrid_reuse_deck = f"""
* Hybrid restored-enable/analog-error writer hidden-error reset/reuse check.
* One hidden-error capacitor pair is first written as r+, reset by
* complementary MOS transmission gates back to the neutral error common mode,
* then rewritten as r-.  The same restored gates and analog caps steer one
* persistent W+/W- pair in both sample phases.
{COMMON_MODELS}
.param CERR=10p CWRITE=500p WSW=24u WWRITE=24u WRESTN=18u WRESTP=300u WRESETN=24u WRESETP=60u
VDD vdd 0 1.8
VTAIL vbias 0 0.95
VERRCM verrcm 0 1.04
VPBWD_HYE pbwd 0 PWL(0 0 0.45u 0 0.47u 1.8 1.22u 1.8 1.24u 0 2.70u 0 2.72u 1.8 3.47u 1.8 3.49u 0 6.4u 0)
VRST_HYE rst_hye 0 PWL(0 0 1.95u 0 1.97u 1.8 2.42u 1.8 2.44u 0 6.4u 0)
VRSTN_HYE rstn_hye 0 PWL(0 1.8 1.95u 1.8 1.97u 0 2.42u 0 2.44u 1.8 6.4u 1.8)
VRP_HYE rp_hye 0 PWL(0 0 0.45u 0 0.47u 1.8 1.22u 1.8 1.24u 0 6.4u 0)
VRM_HYE rm_hye 0 PWL(0 0 2.70u 0 2.72u 1.8 3.47u 1.8 3.49u 0 6.4u 0)
VPACC_POS_HYE paccn_pos_hye 0 PWL(0 1.8 1.45u 1.8 1.47u 0 1.83u 0 1.85u 1.8 6.4u 1.8)
VPACC_NEG_HYE paccn_neg_hye 0 PWL(0 1.8 3.70u 1.8 3.72u 0 4.08u 0 4.10u 1.8 6.4u 1.8)
VHM_POS_HYE hm_pos_hye 0 0.92

VZPP_HYE zpp_hye 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}
VZMM_HYE zmm_hye 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZPM_HYE zpm_hye 0 {0.9 - hybrid_mismatch_eps / 2.0:.5f}
VZMP_HYE zmp_hye 0 {0.9 + hybrid_mismatch_eps / 2.0:.5f}

MPPP_HYE hpp_hye hpp_hye vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM_HYE hpm_hye hpm_hye vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP_HYE hpp_hye zpp_hye tailp_hye 0 NMOS L={{LCH}} W={{WN}}
MNPM_HYE hpm_hye zmm_hye tailp_hye 0 NMOS L={{LCH}} W={{WN}}
MNTP_HYE tailp_hye vbias 0 0 NMOS L={{LCH}} W={{WN}}

MPMP_HYE hmp_hye hmp_hye vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM_HYE hmm_hye hmm_hye vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP_HYE hmp_hye zpm_hye tailm_hye 0 NMOS L={{LCH}} W={{WN}}
MNMM_HYE hmm_hye zmp_hye tailm_hye 0 NMOS L={{LCH}} W={{WN}}
MNTM_HYE tailm_hye vbias 0 0 NMOS L={{LCH}} W={{WN}}

CDP_HYE cdp_hye 0 {{CERR}} IC=1.04
CDM_HYE cdm_hye 0 {{CERR}} IC=1.04
RDP_HYE cdp_hye 0 50G
RDM_HYE cdm_hye 0 50G
MRDPN_HYE cdp_hye rst_hye verrcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRDMN_HYE cdm_hye rst_hye verrcm 0 NMOS L={{LCH}} W={{WRESETN}}
MRDPP_HYE cdp_hye rstn_hye verrcm vdd PMOS L={{LCH}} W={{WRESETP}}
MRDMP_HYE cdm_hye rstn_hye verrcm vdd PMOS L={{LCH}} W={{WRESETP}}
{sign_store_path("hpm_hye", "rp_hye", "cdp_hye", "hyerp1")}
{sign_store_path("hmp_hye", "rp_hye", "cdp_hye", "hyerp2")}
{sign_store_path("hpp_hye", "rp_hye", "cdm_hye", "hyerp3")}
{sign_store_path("hmm_hye", "rp_hye", "cdm_hye", "hyerp4")}
{sign_store_path("hpp_hye", "rm_hye", "cdp_hye", "hyerm1")}
{sign_store_path("hmm_hye", "rm_hye", "cdp_hye", "hyerm2")}
{sign_store_path("hpm_hye", "rm_hye", "cdm_hye", "hyerm3")}
{sign_store_path("hmp_hye", "rm_hye", "cdm_hye", "hyerm4")}

MPRP_CDP_HYE rgp_hye cdp_hye vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDP_HYE rgp_hye cdp_hye 0 0 NMOS L={{LCH}} W={{WRESTN}}
MPRP_CDM_HYE rgm_hye cdm_hye vdd vdd PMOS L={{LCH}} W={{WRESTP}}
MNRP_CDM_HYE rgm_hye cdm_hye 0 0 NMOS L={{LCH}} W={{WRESTN}}

CWP_HYE wp_hye 0 {{CWRITE}} IC=0.85
CWM_HYE wm_hye 0 {{CWRITE}} IC=0.85
MWP_POS_HYE_A vdd paccn_pos_hye n_wp_pos_hye_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYE_B n_wp_pos_hye_a hm_pos_hye n_wp_pos_hye_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYE_C n_wp_pos_hye_b rgp_hye n_wp_pos_hye_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_POS_HYE_D n_wp_pos_hye_c cdm_hye wp_hye vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYE_A vdd paccn_pos_hye n_wm_pos_hye_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYE_B n_wm_pos_hye_a hm_pos_hye n_wm_pos_hye_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYE_C n_wm_pos_hye_b rgm_hye n_wm_pos_hye_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_POS_HYE_D n_wm_pos_hye_c cdp_hye wm_hye vdd PMOS L={{LCH}} W={{WWRITE}}

MWP_NEG_HYE_A vdd paccn_neg_hye n_wp_neg_hye_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYE_B n_wp_neg_hye_a hm_pos_hye n_wp_neg_hye_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYE_C n_wp_neg_hye_b rgp_hye n_wp_neg_hye_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWP_NEG_HYE_D n_wp_neg_hye_c cdm_hye wp_hye vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYE_A vdd paccn_neg_hye n_wm_neg_hye_a vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYE_B n_wm_neg_hye_a hm_pos_hye n_wm_neg_hye_b vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYE_C n_wm_neg_hye_b rgm_hye n_wm_neg_hye_c vdd PMOS L={{LCH}} W={{WWRITE}}
MWM_NEG_HYE_D n_wm_neg_hye_c cdp_hye wm_hye vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 5n 6.2u uic
wrdata mos_hidden_writer_restored_gate_hybrid_reuse.dat v(cdp_hye) v(cdm_hye) v(rgp_hye) v(rgm_hye) v(wp_hye) v(wm_hye) v(rp_hye) v(rm_hye) v(rst_hye) v(paccn_pos_hye) v(paccn_neg_hye)
quit
.endc
.end
"""
    hybrid_reuse_data = run_ngspice(
        hybrid_reuse_deck,
        "mos_hidden_writer_restored_gate_hybrid_reuse",
    )
    hyet, hye_cols = load_wrdata(hybrid_reuse_data, 11)

    def hyeat(time_s: float, values: np.ndarray) -> float:
        return float(values[np.argmin(np.abs(hyet - time_s))])

    hye_hidden = hye_cols[0] - hye_cols[1]
    hye_common = 0.5 * (hye_cols[0] + hye_cols[1])
    hye_rgp = hye_cols[2]
    hye_rgm = hye_cols[3]
    hye_wp = hye_cols[4]
    hye_wm = hye_cols[5]
    hye_weight = hye_wp - hye_wm
    hye_pos_step = hyeat(1.90e-6, hye_weight)
    hye_reset_hidden = hyeat(2.55e-6, hye_hidden)
    hye_neg_hidden = hyeat(3.60e-6, hye_hidden)
    hye_final_weight = hyeat(4.25e-6, hye_weight)
    require(hyeat(1.35e-6, hye_hidden) > 0.07, "hybrid reuse first store should create positive hidden error")
    require(hyeat(1.35e-6, hye_rgp) < 0.30, "hybrid reuse first selected restored gate should be low")
    require(hyeat(1.35e-6, hye_rgm) > 1.60, "hybrid reuse first complement restored gate should be high")
    require(hye_pos_step > 0.020, "hybrid reuse first sample should write W+")
    require(abs(hye_reset_hidden) < 0.004, "hybrid reuse reset should clear hidden-error differential")
    require(abs(hyeat(2.55e-6, hye_common) - 1.04) < 0.005, "hybrid reuse reset should restore hidden-error common mode")
    require(abs(hyeat(2.55e-6, hye_weight) - hye_pos_step) < 5e-4, "hybrid reuse reset should not disturb persistent weight caps")
    require(hye_neg_hidden < -0.07, "hybrid reuse second store should create negative hidden error")
    require(hyeat(3.60e-6, hye_rgm) < 0.30, "hybrid reuse second selected restored gate should be low")
    require(hyeat(3.60e-6, hye_rgp) > 1.60, "hybrid reuse second complement restored gate should be high")
    require(hye_final_weight < hye_pos_step - 0.020, "hybrid reuse negative sample should cancel the prior W+ write")
    require(hye_final_weight < 0.010, "hybrid reuse final signed weight should be near-zero or negative after opposite sign")
    require(abs(hyeat(5.90e-6, hye_weight) - hye_final_weight) < 5e-4, "hybrid reuse final weight should hold")

    hye_fig, hye_axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    hye_axes[0].plot(1e6 * hyet, hye_hidden, label="$c_{d+}-c_{d-}$")
    hye_axes[0].plot(1e6 * hyet, hye_common - 1.04, label="common-mode error")
    hye_axes[0].plot(1e6 * hyet, hye_cols[6] / 20.0, color="0.45", alpha=0.3, label="$rp/20$")
    hye_axes[0].plot(1e6 * hyet, hye_cols[7] / 20.0, color="0.15", alpha=0.25, label="$rm/20$")
    hye_axes[0].plot(1e6 * hyet, hye_cols[8] / 20.0, color="0.6", alpha=0.3, label="$reset/20$")
    hye_axes[0].axhline(0, color="0.4", linewidth=0.8)
    hye_axes[0].set_ylabel("hidden state (V)")
    hye_axes[0].set_title("One hidden-error capacitor pair is reset and reused")
    hye_axes[0].grid(True, alpha=0.25)
    hye_axes[0].legend(loc="upper right", ncol=2, fontsize="small")
    hye_axes[1].plot(1e6 * hyet, hye_rgp, label="$c_{d+}$ restored gate")
    hye_axes[1].plot(1e6 * hyet, hye_rgm, label="$c_{d-}$ restored gate")
    hye_axes[1].axhline(0.9, color="0.4", linewidth=0.8, alpha=0.5)
    hye_axes[1].set_ylabel("gate voltage (V)")
    hye_axes[1].set_title("Restored select gate swaps after physical reset")
    hye_axes[1].grid(True, alpha=0.25)
    hye_axes[1].legend(loc="center right", fontsize="small")
    hye_axes[2].plot(1e6 * hyet, hye_wp - hyeat(1.35e-6, hye_wp), label="$W^+$ step")
    hye_axes[2].plot(1e6 * hyet, hye_wm - hyeat(1.35e-6, hye_wm), label="$W^-$ step")
    hye_axes[2].plot(1e6 * hyet, hye_weight, label="$W^+ - W^-$")
    hye_axes[2].plot(1e6 * hyet, hye_cols[9] / 20.0, color="0.45", alpha=0.3, label="$pacc^+_n/20$")
    hye_axes[2].plot(1e6 * hyet, hye_cols[10] / 20.0, color="0.15", alpha=0.25, label="$pacc^-_n/20$")
    hye_axes[2].axhline(0, color="0.4", linewidth=0.8)
    hye_axes[2].set_xlabel("time (us)")
    hye_axes[2].set_ylabel("weight step (V)")
    hye_axes[2].set_title("Persistent weight caps survive reset and accept opposite update")
    hye_axes[2].grid(True, alpha=0.25)
    hye_axes[2].legend(loc="upper right", ncol=2, fontsize="small")
    hye_fig.tight_layout()
    save_plot(hye_fig, "mos_hidden_writer_restored_gate_hybrid_reuse_ngspice")

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
