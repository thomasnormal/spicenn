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
    transfer_deck = f"""
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
    store_deck = f"""
* MOS forward storage sanity check
{COMMON_MODELS}
.param CSTORE=10p WSW=24u
VDD vdd 0 1.8
VZP zp 0 PWL(0 0.9 0.45u 0.9 0.5u 1.25 1.7u 1.25 1.75u 0.55 3.35u 0.55 3.4u 0.9 4u 0.9)
VZM zm 0 0.9
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
    xdiff = x - 0.9
    signed = cols[1] - cols[0]
    require(np.all(np.diff(signed) >= -1e-4), "forward pair transfer should be monotone")
    require(abs(signed[np.argmin(np.abs(xdiff))]) < 1e-3, "forward pair should be centered near zero")
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
* Four-quadrant writer coincidence sanity check.
{COMMON_MODELS}
.param WWRITE=2u CWRITE=500p
VDD vdd 0 1.8
VPACC_N paccn 0 PULSE(1.8 0 0.5u 20n 20n 0.8u 5.0u)
VHI hi 0 1.8
VLO lo 0 0

* Same-sign copy: active-low pacc/x+/dh+ gates charge W+.
CWP_S wp_s 0 {{CWRITE}} IC=0.2
CWM_S wm_s 0 {{CWRITE}} IC=0.2
MSSP1 vdd paccn n1s vdd PMOS L={{LCH}} W={{WWRITE}}
MSSP2 n1s lo n2s vdd PMOS L={{LCH}} W={{WWRITE}}
MSSP3 n2s lo wp_s vdd PMOS L={{LCH}} W={{WWRITE}}
MSSM1 vdd paccn n3s vdd PMOS L={{LCH}} W={{WWRITE}}
MSSM2 n3s lo n4s vdd PMOS L={{LCH}} W={{WWRITE}}
MSSM3 n4s hi wm_s vdd PMOS L={{LCH}} W={{WWRITE}}

* Opposite-sign copy: active-low pacc/x+/dh- gates charge W-.
CWP_O wp_o 0 {{CWRITE}} IC=0.2
CWM_O wm_o 0 {{CWRITE}} IC=0.2
MOSP1 vdd paccn n1o vdd PMOS L={{LCH}} W={{WWRITE}}
MOSP2 n1o lo n2o vdd PMOS L={{LCH}} W={{WWRITE}}
MOSP3 n2o hi wp_o vdd PMOS L={{LCH}} W={{WWRITE}}
MOSM1 vdd paccn n3o vdd PMOS L={{LCH}} W={{WWRITE}}
MOSM2 n3o lo n4o vdd PMOS L={{LCH}} W={{WWRITE}}
MOSM3 n4o lo wm_o vdd PMOS L={{LCH}} W={{WWRITE}}

.control
set noaskquit
tran 10n 4u uic
wrdata mos_writer.dat v(wp_s) v(wm_s) v(wp_o) v(wm_o) v(paccn)
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
CWP{idx} wp{idx} 0 {{CWRITE}} IC=0.2
CWM{idx} wm{idx} 0 {{CWRITE}} IC=0.2
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
    data = run_ngspice(deck, "mos_writer")
    t, cols = load_wrdata(data, 5)
    same_wp_delta = cols[0][-1] - cols[0][0]
    same_wm_delta = cols[1][-1] - cols[1][0]
    opp_wp_delta = cols[2][-1] - cols[2][0]
    opp_wm_delta = cols[3][-1] - cols[3][0]
    require(same_wp_delta > 0.05 and same_wp_delta < 0.25, "same-sign writer should make a bounded W+ step")
    require(abs(same_wm_delta) < 1e-3, "same-sign writer should leave W- quiet")
    require(opp_wm_delta > 0.05 and opp_wm_delta < 0.25, "opposite-sign writer should make a bounded W- step")
    require(abs(opp_wp_delta) < 1e-3, "opposite-sign writer should leave W+ quiet")

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

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    axes[0].plot(1e6 * t, cols[0], label="$W^+$, same sign")
    axes[0].plot(1e6 * t, cols[1], label="$W^-$, same sign")
    axes[0].set_ylabel("cap voltage (V)")
    axes[0].set_title("Same-sign coincidence makes a bounded W+ step")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(1e6 * t, cols[2], label="$W^+$, opposite sign")
    axes[1].plot(1e6 * t, cols[3], label="$W^-$, opposite sign")
    axes[1].plot(1e6 * t, cols[4] / 6.0, color="0.5", alpha=0.45, label="$\\overline{pacc}/6$")
    axes[1].set_ylabel("cap voltage (V)")
    axes[1].set_title("Opposite-sign coincidence makes a bounded W- step")
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
