"""Small analytic helpers for SPICENN device-parameter ranges.

The SPICE decks are the ground truth, but many knobs can be screened with
first-order capacitor and gradient-flow estimates before launching long runs.
All public helpers use the same engineering units as the device runner flags:
capacitance in femtofarads, time in nanoseconds, current in microamps, and
resistance in ohms.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")


def rc_tau_ns(resistance_ohm: float, capacitance_ff: float) -> float:
    """Return the RC time constant in ns for R in ohms and C in fF."""
    _require_positive("resistance_ohm", resistance_ohm)
    _require_positive("capacitance_ff", capacitance_ff)
    return resistance_ohm * capacitance_ff * 1e-6


def rc_resistance_for_tau_ohm(tau_ns: float, capacitance_ff: float) -> float:
    """Return R in ohms needed to get ``tau_ns`` with ``capacitance_ff``."""
    _require_positive("tau_ns", tau_ns)
    _require_positive("capacitance_ff", capacitance_ff)
    return tau_ns / (capacitance_ff * 1e-6)


def remaining_fraction(duration_ns: float, resistance_ohm: float, capacitance_ff: float) -> float:
    """Fraction of an initial capacitor voltage left after passive RC leakage."""
    _require_positive("duration_ns", duration_ns)
    tau = rc_tau_ns(resistance_ohm, capacitance_ff)
    return exp(-duration_ns / tau)


def leaked_fraction(duration_ns: float, resistance_ohm: float, capacitance_ff: float) -> float:
    """Fraction of an initial capacitor voltage lost after passive RC leakage."""
    return 1.0 - remaining_fraction(duration_ns, resistance_ohm, capacitance_ff)


def capacitor_delta_v(current_ua: float, duration_ns: float, capacitance_ff: float) -> float:
    """Voltage step from integrating a constant current on a capacitor.

    With these units, 1 uA for 1 ns on 1 fF gives exactly 1 V.
    """
    _require_positive("duration_ns", duration_ns)
    _require_positive("capacitance_ff", capacitance_ff)
    if not isfinite(current_ua):
        raise ValueError("current_ua must be finite.")
    return current_ua * duration_ns / capacitance_ff


def required_cap_ff_for_step(current_ua: float, duration_ns: float, max_delta_v: float) -> float:
    """Minimum capacitance that bounds an integration step to ``max_delta_v``."""
    _require_positive("current_ua", abs(current_ua))
    _require_positive("duration_ns", duration_ns)
    _require_positive("max_delta_v", max_delta_v)
    return abs(current_ua) * duration_ns / max_delta_v


def cap_range_for_signal_window_ff(
    min_current_ua: float,
    max_current_ua: float,
    duration_ns: float,
    min_signal_v: float,
    max_signal_v: float,
) -> tuple[float, float]:
    """Capacitance interval that keeps useful signals above noise and below rails.

    For an integrating node ``dV = I T / C``.  The lower bound prevents the
    largest expected current from saturating the node; the upper bound keeps the
    smallest expected current observable.
    """
    _require_positive("min_current_ua", min_current_ua)
    _require_positive("max_current_ua", max_current_ua)
    _require_positive("duration_ns", duration_ns)
    _require_positive("min_signal_v", min_signal_v)
    _require_positive("max_signal_v", max_signal_v)
    if min_current_ua > max_current_ua:
        raise ValueError("min_current_ua must not exceed max_current_ua.")
    if min_signal_v >= max_signal_v:
        raise ValueError("min_signal_v must be below max_signal_v.")
    lower = max_current_ua * duration_ns / max_signal_v
    upper = min_current_ua * duration_ns / min_signal_v
    return lower, upper


@dataclass(frozen=True)
class DampingBounds:
    """Leak-resistor bounds for a parasitic damping capacitor."""

    cap_ff: float
    min_resistance_ohm: float
    max_resistance_ohm: float
    min_tau_ns: float
    max_tau_ns: float

    @property
    def feasible(self) -> bool:
        return self.min_resistance_ohm <= self.max_resistance_ohm

    def contains(self, resistance_ohm: float) -> bool:
        return self.min_resistance_ohm <= resistance_ohm <= self.max_resistance_ohm


def damping_resistance_bounds(
    cap_ff: float,
    active_window_ns: float,
    idle_window_ns: float,
    *,
    max_active_loss: float = 0.10,
    max_idle_residue: float = 0.05,
) -> DampingBounds:
    """Return resistor range for a damping cap used as a weak DC path.

    The lower bound keeps leakage from removing more than ``max_active_loss`` of
    the node voltage during the useful active window.  The upper bound makes
    passive leakage remove old charge to ``max_idle_residue`` during the idle
    interval.  If the interval is infeasible, the circuit needs an explicit
    reset/bleed switch or looser requirements.
    """
    _require_positive("cap_ff", cap_ff)
    _require_positive("active_window_ns", active_window_ns)
    _require_positive("idle_window_ns", idle_window_ns)
    if not 0.0 < max_active_loss < 1.0:
        raise ValueError("max_active_loss must be between 0 and 1.")
    if not 0.0 < max_idle_residue < 1.0:
        raise ValueError("max_idle_residue must be between 0 and 1.")

    min_tau_ns = active_window_ns / -log(1.0 - max_active_loss)
    max_tau_ns = idle_window_ns / -log(max_idle_residue)
    return DampingBounds(
        cap_ff=cap_ff,
        min_resistance_ohm=rc_resistance_for_tau_ohm(min_tau_ns, cap_ff),
        max_resistance_ohm=rc_resistance_for_tau_ohm(max_tau_ns, cap_ff),
        min_tau_ns=min_tau_ns,
        max_tau_ns=max_tau_ns,
    )


def linear_stability_eta_bound(features: np.ndarray) -> float:
    """Batch squared-error stability bound for a linear readout.

    For ``y = Xw`` and loss ``0.5 ||Xw-t||^2``, fixed-step gradient descent is
    stable for ``0 < eta < 2 / lambda_max(X^T X)``.  This is only a yardstick for
    the hardware effective mobility, not a claim that the MOS circuit performs
    exact SGD.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    if x.size == 0:
        raise ValueError("features must be nonempty.")
    gram = x.T @ x
    largest = float(np.linalg.eigvalsh(gram).max())
    _require_positive("largest eigenvalue", largest)
    return 2.0 / largest


def single_sample_eta_bound(feature_norm_sq: float) -> float:
    """One-sample squared-error stability yardstick ``eta < 2 / ||a||^2``."""
    _require_positive("feature_norm_sq", feature_norm_sq)
    return 2.0 / feature_norm_sq


def backward_alignment_cosine(forward_gain: np.ndarray, backward_gain: np.ndarray, errors: np.ndarray) -> float:
    """Cosine between hardware backward flow and the ideal forward transpose.

    ``forward_gain`` has shape ``(outputs, inputs)``.  ``backward_gain`` has
    shape ``(inputs, outputs)``.  ``errors`` may be a single output-error vector
    or a batch of row vectors.  The return value is the cosine between
    ``backward_gain @ e`` and ``forward_gain.T @ e`` after flattening the tested
    error set.  Positive values are feedback-alignment-like; one is exact
    transpose alignment up to gain.
    """
    gf = np.asarray(forward_gain, dtype=float)
    gb = np.asarray(backward_gain, dtype=float)
    e = np.asarray(errors, dtype=float)
    if gf.ndim != 2 or gb.ndim != 2:
        raise ValueError("forward_gain and backward_gain must be 2D matrices.")
    if gb.shape != (gf.shape[1], gf.shape[0]):
        raise ValueError("backward_gain must have shape (inputs, outputs).")
    if e.ndim == 1:
        if e.shape[0] != gf.shape[0]:
            raise ValueError("error vector length must match output count.")
        ideal = gf.T @ e
        actual = gb @ e
    elif e.ndim == 2:
        if e.shape[1] != gf.shape[0]:
            raise ValueError("error matrix width must match output count.")
        ideal = e @ gf
        actual = e @ gb.T
    else:
        raise ValueError("errors must be a vector or a 2D batch matrix.")
    ideal_flat = np.ravel(ideal)
    actual_flat = np.ravel(actual)
    denom = float(np.linalg.norm(ideal_flat) * np.linalg.norm(actual_flat))
    _require_positive("alignment denominator", denom)
    return float(np.dot(actual_flat, ideal_flat) / denom)


def mobility_condition_ratio(increase_mobility: float, decrease_mobility: float) -> float:
    """Return ``min(mu+)/max(mu+)`` for two positive signed-update mobilities."""
    _require_positive("increase_mobility", increase_mobility)
    _require_positive("decrease_mobility", decrease_mobility)
    return min(increase_mobility, decrease_mobility) / max(increase_mobility, decrease_mobility)


def mobility_window_is_usable(
    increase_mobility: float,
    decrease_mobility: float,
    *,
    min_condition_ratio: float = 0.1,
    min_mobility: float = 0.0,
) -> bool:
    """Screen whether a signed branch pair has nonzero, not-too-one-sided updates."""
    if not 0.0 < min_condition_ratio <= 1.0:
        raise ValueError("min_condition_ratio must be in (0, 1].")
    if min_mobility < 0 or not isfinite(min_mobility):
        raise ValueError("min_mobility must be nonnegative and finite.")
    return (
        increase_mobility > min_mobility
        and decrease_mobility > min_mobility
        and mobility_condition_ratio(increase_mobility, decrease_mobility) >= min_condition_ratio
    )


@dataclass(frozen=True)
class BoundedWriteHeadroom:
    """Voltage headroom around an initialized state for bounded write rails."""

    center_v: float
    high_v: float
    low_v: float

    @property
    def increase_headroom_v(self) -> float:
        return self.high_v - self.center_v

    @property
    def decrease_headroom_v(self) -> float:
        return self.center_v - self.low_v

    def directional(self, min_headroom_v: float = 0.0) -> bool:
        if min_headroom_v < 0 or not isfinite(min_headroom_v):
            raise ValueError("min_headroom_v must be nonnegative and finite.")
        return self.increase_headroom_v > min_headroom_v and self.decrease_headroom_v > min_headroom_v


def bounded_write_headroom(center_v: float, high_v: float, low_v: float) -> BoundedWriteHeadroom:
    """Return whether bounded charge/discharge rails can move a state both ways.

    A bounded charge path only increases a capacitor state if ``high_v`` is
    above the current state.  Likewise a bounded discharge path needs ``low_v``
    below the current state.  This is a necessary headroom screen, not a full
    MOS mobility model.
    """
    for name, value in [("center_v", center_v), ("high_v", high_v), ("low_v", low_v)]:
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if not low_v < high_v:
        raise ValueError("low_v must be below high_v.")
    return BoundedWriteHeadroom(center_v=center_v, high_v=high_v, low_v=low_v)


def one_vs_rest_target_balance_ratio(class_count: int, target_mobility: float, nontarget_mobility: float) -> float:
    """Return target drive divided by total one-vs-rest non-target drive.

    In a balanced K-class epoch, each output row sees one target exposure and
    K-1 non-target exposures.  A ratio above one means the row's target-side
    mobility exceeds the total opposing non-target mobility before considering
    feature-dependent differences.
    """
    if class_count < 2:
        raise ValueError("class_count must be at least two.")
    _require_positive("target_mobility", target_mobility)
    _require_positive("nontarget_mobility", nontarget_mobility)
    return target_mobility / ((class_count - 1) * nontarget_mobility)


def one_vs_rest_width_ratio_is_balanced(
    class_count: int,
    target_width_u: float,
    nontarget_width_u: float,
    *,
    min_ratio: float = 1.0,
) -> bool:
    """Screen one-vs-rest rail widths against balanced-class exposure counts."""
    _require_positive("min_ratio", min_ratio)
    return one_vs_rest_target_balance_ratio(class_count, target_width_u, nontarget_width_u) >= min_ratio
