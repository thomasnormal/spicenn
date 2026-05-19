from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf


def probit(x, theta, sigma):
    sigma = np.maximum(np.abs(sigma), 1e-12)
    return 0.5 * (1.0 + erf((x - theta) / (sigma * np.sqrt(2.0))))


def logistic(x, theta, scale):
    scale = np.maximum(np.abs(scale), 1e-12)
    return 1.0 / (1.0 + np.exp(-(x - theta) / scale))


def fit_activation(csv_path: str | Path, json_path: str | Path, plot_path: str | Path) -> dict:
    df = pd.read_csv(csv_path)
    x = df["signal"].to_numpy(dtype=float)
    y = df["p_high"].to_numpy(dtype=float)
    n = df.get("trials", pd.Series(np.ones(len(df)))).to_numpy(dtype=float)
    sigma0 = max((x.max() - x.min()) / 6.0, 1e-9)
    popt_p, pcov_p = curve_fit(probit, x, y, p0=[0.0, sigma0], bounds=([-np.inf, 1e-12], [np.inf, np.inf]), maxfev=10000)
    popt_l, pcov_l = curve_fit(logistic, x, y, p0=[0.0, sigma0], bounds=([-np.inf, 1e-12], [np.inf, np.inf]), maxfev=10000)
    theta, sigma = [float(v) for v in popt_p]
    ci = 1.96 * np.sqrt(np.diag(pcov_p))
    result = {
        "theta_eff": theta,
        "sigma_eff": sigma,
        "slope_at_threshold": float(1.0 / (sigma * np.sqrt(2.0 * np.pi))),
        "transition_width_10_90": float(2.563103131 * sigma),
        "theta_ci95": [float(theta - ci[0]), float(theta + ci[0])],
        "sigma_ci95": [float(sigma - ci[1]), float(sigma + ci[1])],
        "logistic_theta": float(popt_l[0]),
        "logistic_scale": float(popt_l[1]),
        "trials_total": int(n.sum()),
    }
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    xs = np.linspace(x.min(), x.max(), 300)
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, "o", label="sweep")
    plt.plot(xs, probit(xs, *popt_p), label="probit fit")
    plt.plot(xs, logistic(xs, *popt_l), "--", label="logistic fit")
    plt.xlabel("signal")
    plt.ylabel("P(out high)")
    plt.ylim(-0.03, 1.03)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--json", default="spice/results/activation_curve_fit.json")
    ap.add_argument("--plot", default="spice/results/activation_curve.png")
    args = ap.parse_args()
    print(json.dumps(fit_activation(args.csv_path, args.json, args.plot), indent=2))


if __name__ == "__main__":
    main()

