#!/usr/bin/env python3
"""Run ngspice sanity checks for the MOS local-feature cell sketches.

The decks generated here are intentionally small transistor-level
characterizations of the four paper panels.  They use simple Level-1 MOS models
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
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(xdiff, 1e6 * pos_signed, label="$w^+$ high")
    ax.plot(xdiff, 1e6 * neg_signed, label="$w^-$ high")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.axvline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("$x^+ - x^-$ (V)")
    ax.set_ylabel("contribution current (uA)")
    ax.set_title("Signed synapse slice reverses contribution sign")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_plot(fig, "mos_synapse_slice_ngspice")


def characterize_forward_pair() -> Path:
    deck = f"""
* MOS forward differential-pair sanity check
{COMMON_MODELS}
VDD vdd 0 1.8
VZP zp 0 0.2
VZM zm 0 0.9
VTAIL vbias 0 0.95
MP1 hp hp vdd vdd PMOS L={{LCH}} W={{WP}}
MP2 hm hm vdd vdd PMOS L={{LCH}} W={{WP}}
MN1 hp zp tail 0 NMOS L={{LCH}} W={{WN}}
MN2 hm zm tail 0 NMOS L={{LCH}} W={{WN}}
MNT tail vbias 0 0 NMOS L={{LCH}} W={{WN}}
.control
set noaskquit
dc VZP 0.2 1.6 0.01
wrdata mos_forward_pair.dat v(hp) v(hm)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_forward_pair")
    x, cols = load_wrdata(data, 2)
    xdiff = x - 0.9
    signed = cols[1] - cols[0]
    require(np.all(np.diff(signed) >= -1e-4), "forward pair transfer should be monotone")
    require(abs(signed[np.argmin(np.abs(xdiff))]) < 1e-3, "forward pair should be centered near zero")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # The diode-connected PMOS load is voltage-inverting: the rail that sinks
    # more differential-pair current moves lower.  Plot the usable signed load
    # voltage convention so the transfer rises with z+ - z-.
    ax.plot(xdiff, signed, label="$h^- - h^+$")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.axvline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("$z^+ - z^-$ (V)")
    ax.set_ylabel("load differential voltage (V)")
    ax.set_title("Forward MOS differential pair gives monotone transfer")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_plot(fig, "mos_forward_pair_ngspice")


def characterize_hidden_error() -> Path:
    eps = 0.01
    deck = f"""
* Hidden-error finite-difference replica-pair sanity check.
* The small-signal nudge is applied to the swept preactivation rail so this
* measures the local slope of the same MOS forward transfer plotted above.
{COMMON_MODELS}
.param EPS={eps}
VDD vdd 0 1.8
VZP zp 0 0.2
VZM zm 0 0.9
VTAIL vbias 0 0.95

* Testbench nudges: p-copy uses z+ + eps.
VZP_P zpp zp {{EPS}}
* m-copy uses z+ - eps.
VZP_M zp zpm {{EPS}}

* Positive nudge replica.
MPPP hpp hpp vdd vdd PMOS L={{LCH}} W={{WP}}
MPPM hpm hpm vdd vdd PMOS L={{LCH}} W={{WP}}
MNPP hpp zpp tailp 0 NMOS L={{LCH}} W={{WN}}
MNPM hpm zm tailp 0 NMOS L={{LCH}} W={{WN}}
MNTP tailp vbias 0 0 NMOS L={{LCH}} W={{WN}}

* Negative nudge replica.
MPMP hmp hmp vdd vdd PMOS L={{LCH}} W={{WP}}
MPMM hmm hmm vdd vdd PMOS L={{LCH}} W={{WP}}
MNMP hmp zpm tailm 0 NMOS L={{LCH}} W={{WN}}
MNMM hmm zm tailm 0 NMOS L={{LCH}} W={{WN}}
MNTM tailm vbias 0 0 NMOS L={{LCH}} W={{WN}}

.control
set noaskquit
dc VZP 0.2 1.6 0.01
wrdata mos_hidden_error.dat v(hpp) v(hpm) v(hmp) v(hmm)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_hidden_error")
    x, cols = load_wrdata(data, 4)
    xdiff = x - 0.9
    signed_plus_nudge = cols[1] - cols[0]
    signed_minus_nudge = cols[3] - cols[2]
    gain = (signed_plus_nudge - signed_minus_nudge) / (2.0 * eps)
    pos_fb = gain
    neg_fb = -gain
    center_gain = gain[np.argmin(np.abs(xdiff))]
    edge_gain = max(float(np.mean(gain[xdiff < -0.45])), float(np.mean(gain[xdiff > 0.45])))
    require(center_gain > 0.5, "hidden-error derivative gain should be positive near z balance")
    require(edge_gain < 0.65 * center_gain, "hidden-error derivative gain should fall at saturated z")
    require(np.all(pos_fb > -1e-4), "positive feedback gain should stay nonnegative")
    require(np.all(neg_fb < 1e-4), "negative feedback gain should stay nonpositive")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x - 0.9, pos_fb, label="$r^+$ active")
    ax.plot(x - 0.9, neg_fb, label="$r^-$ active")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.axvline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("$z^+ - z^-$ (V)")
    ax.set_ylabel("small-signal error gain (V/V)")
    ax.set_title("Hidden-error replicas produce derivative-shaped signed gain")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_plot(fig, "mos_hidden_error_ngspice")


def characterize_writer() -> Path:
    deck = f"""
* Four-quadrant writer coincidence sanity check
{COMMON_MODELS}
.param WWRITE=2u CWRITE=100p
VPACC pacc 0 PULSE(0 1.8 0.5u 20n 20n 3.0u 6.0u)

* Same-sign copy: x+ and dh+ discharge W+.
VXP_S xp_s 0 1.8
VXM_S xm_s 0 0.0
VDHP_S dhp_s 0 1.8
VDHM_S dhm_s 0 0.0
CWP_S wp_s 0 {{CWRITE}} IC=1.8
CWM_S wm_s 0 {{CWRITE}} IC=1.8
MSSP1 wp_s pacc n1s 0 NMOS L={{LCH}} W={{WWRITE}}
MSSP2 n1s xp_s n2s 0 NMOS L={{LCH}} W={{WWRITE}}
MSSP3 n2s dhp_s 0 0 NMOS L={{LCH}} W={{WWRITE}}
MSSM1 wm_s pacc n3s 0 NMOS L={{LCH}} W={{WWRITE}}
MSSM2 n3s xp_s n4s 0 NMOS L={{LCH}} W={{WWRITE}}
MSSM3 n4s dhm_s 0 0 NMOS L={{LCH}} W={{WWRITE}}

* Opposite-sign copy: x+ and dh- discharge W-.
VXP_O xp_o 0 1.8
VXM_O xm_o 0 0.0
VDHP_O dhp_o 0 0.0
VDHM_O dhm_o 0 1.8
CWP_O wp_o 0 {{CWRITE}} IC=1.8
CWM_O wm_o 0 {{CWRITE}} IC=1.8
MOSP1 wp_o pacc n1o 0 NMOS L={{LCH}} W={{WWRITE}}
MOSP2 n1o xp_o n2o 0 NMOS L={{LCH}} W={{WWRITE}}
MOSP3 n2o dhp_o 0 0 NMOS L={{LCH}} W={{WWRITE}}
MOSM1 wm_o pacc n3o 0 NMOS L={{LCH}} W={{WWRITE}}
MOSM2 n3o xp_o n4o 0 NMOS L={{LCH}} W={{WWRITE}}
MOSM3 n4o dhm_o 0 0 NMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 10n 4u uic
wrdata mos_writer.dat v(wp_s) v(wm_s) v(wp_o) v(wm_o) v(pacc)
quit
.endc
.end
"""
    data = run_ngspice(deck, "mos_writer")
    t, cols = load_wrdata(data, 5)
    require(cols[0][-1] < 0.2 and cols[1][-1] > 1.7, "same-sign writer should select only W+")
    require(cols[3][-1] < 0.2 and cols[2][-1] > 1.7, "opposite-sign writer should select only W-")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(1e6 * t, cols[0], label="$W^+$, same sign")
    axes[0].plot(1e6 * t, cols[1], label="$W^-$, same sign")
    axes[0].set_ylabel("cap voltage (V)")
    axes[0].set_title("Same-sign coincidence selects W+ branch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(1e6 * t, cols[2], label="$W^+$, opposite sign")
    axes[1].plot(1e6 * t, cols[3], label="$W^-$, opposite sign")
    axes[1].plot(1e6 * t, cols[4], color="0.5", alpha=0.45, label="$pacc$")
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("cap voltage (V)")
    axes[1].set_title("Opposite-sign coincidence selects W- branch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return save_plot(fig, "mos_writer_ngspice")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["synapse", "forward", "hidden", "writer"],
        help="Run only one characterization.",
    )
    args = parser.parse_args()
    jobs = {
        "synapse": characterize_synapse,
        "forward": characterize_forward_pair,
        "hidden": characterize_hidden_error,
        "writer": characterize_writer,
    }
    selected = [args.only] if args.only else list(jobs)
    for name in selected:
        path = jobs[name]()
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
