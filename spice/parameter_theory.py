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


def _require_nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite.")


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


def one_vs_rest_signed_update_balance_ratio(
    class_count: int,
    target_delta: float,
    nontarget_delta: float,
) -> float:
    """Return target write movement divided by total non-target movement.

    ``target_delta`` should be positive and ``nontarget_delta`` should be
    negative for the usual one-vs-rest correction direction.  A value near one
    means one target exposure balances the ``class_count - 1`` non-target
    exposures a row sees in a balanced epoch.
    """
    if class_count < 2:
        raise ValueError("class_count must be at least two.")
    _require_positive("target_delta", target_delta)
    if not isfinite(nontarget_delta) or nontarget_delta >= 0:
        raise ValueError("nontarget_delta must be negative and finite.")
    return target_delta / ((class_count - 1) * abs(nontarget_delta))


def one_vs_rest_signed_epoch_delta(
    class_count: int,
    target_delta: float,
    nontarget_delta: float,
) -> float:
    """Return net row movement after one target and all non-target exposures."""
    if class_count < 2:
        raise ValueError("class_count must be at least two.")
    if not isfinite(target_delta) or not isfinite(nontarget_delta):
        raise ValueError("deltas must be finite.")
    return target_delta + (class_count - 1) * nontarget_delta


def one_vs_rest_common_epoch_delta(
    class_count: int,
    target_common_delta: float,
    nontarget_common_delta: float,
) -> float:
    """Return net branch-common movement for one balanced one-vs-rest epoch.

    This is separate from the signed learning balance.  A writer can have
    nearly zero signed epoch drift while still walking both physical branches
    toward a rail if target and non-target events have same-sign common motion.
    """
    if class_count < 2:
        raise ValueError("class_count must be at least two.")
    if not isfinite(target_common_delta) or not isfinite(nontarget_common_delta):
        raise ValueError("common deltas must be finite.")
    return target_common_delta + (class_count - 1) * nontarget_common_delta


def common_drift_to_signed_step_ratio(
    class_count: int,
    target_delta: float,
    nontarget_delta: float,
    target_common_delta: float,
    nontarget_common_delta: float,
) -> float:
    """Compare balanced-epoch common-mode drift to the useful signed step size."""
    signed_scale = max(abs(target_delta), abs(nontarget_delta))
    _require_positive("signed step scale", signed_scale)
    return abs(one_vs_rest_common_epoch_delta(class_count, target_common_delta, nontarget_common_delta)) / signed_scale


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


@dataclass(frozen=True)
class MulticlassReadoutSizing:
    """Derived local sizes for a one-vs-rest multiclass readout family.

    These are not independent hyperparameters.  They are the local widths and
    capacitances implied by the chosen topology and three global scales.
    """

    class_count: int
    effective_readout_fan_in: float
    readout_fan_in_scale: float
    readout_update_width_u: float
    readout_dp_gate_update_width_u: float
    readout_dn_gate_update_width_u: float
    output_bias_update_width_u: float
    readout_write_error_exclusion_width_u: float
    residual_target_width_u: float
    residual_output_width_u: float
    score_cap_f: float
    output_cap_f: float


@dataclass(frozen=True)
class ClassEvidenceNormalizerSizing:
    """Derived local sizes for score-centering to writer-domain error rails."""

    class_count: int
    normalized_score_delta_v: float
    score_common_resistance_ohm: float
    score_common_tau_ns: float
    contrast_observable_ratio: float
    score_common_cap_f: float
    contrast_cap_f: float
    mass_cap_f: float
    error_cap_f: float
    mass_width_u: float
    target_error_width_u: float
    nontarget_error_width_u: float


@dataclass(frozen=True)
class MulticlassMarginCorrectionSizing:
    """Derived local sizes for target-vs-impostor margin correction."""

    class_count: int
    target_margin_v: float
    score_delta_v: float
    observable_margin_ratio: float
    pairwise_pullup_width_u: float
    pairwise_pulldown_width_u: float
    margin_penalty_width_u: float
    margin_reference_window_v: float
    error_width_u: float
    error_cap_f: float
    error_window_ns: float
    error_clock_high_v: float
    target_error_v: float


@dataclass(frozen=True)
class ReadoutMarginSizing:
    """First-order W/C scale needed to make a sign-correct score observable."""

    observed_margin_v: float
    target_margin_v: float
    required_signal_scale: float
    current_readout_width_u: float
    suggested_readout_width_u: float
    max_readout_width_u: float
    readout_width_feasible: bool
    current_score_cap_f: float
    suggested_score_cap_f: float
    min_score_cap_f: float
    score_cap_feasible: bool


def derive_readout_margin_sizing(
    *,
    observed_margin_v: float,
    target_margin_v: float,
    current_readout_width_u: float,
    current_score_cap_f: float,
    max_readout_width_u: float = 512.0,
    min_score_cap_f: float = 0.5,
) -> ReadoutMarginSizing:
    """Estimate readout W/C scaling needed for a target score margin.

    For the same topology and measurement offset, first-order score swing is
    approximately proportional to readout conductance and inversely
    proportional to score capacitance:

    ``Vmargin ~ Wreadout / Cscore``.

    This helper is intentionally only valid for rows whose class sign is already
    correct. If ``observed_margin_v`` is not positive, no amount of positive
    W/C scaling fixes the classification sign.
    """
    for name, value in [
        ("observed_margin_v", observed_margin_v),
        ("target_margin_v", target_margin_v),
        ("current_readout_width_u", current_readout_width_u),
        ("current_score_cap_f", current_score_cap_f),
        ("max_readout_width_u", max_readout_width_u),
        ("min_score_cap_f", min_score_cap_f),
    ]:
        _require_positive(name, float(value))
    required_signal_scale = target_margin_v / observed_margin_v
    suggested_readout_width_u = current_readout_width_u * required_signal_scale
    suggested_score_cap_f = current_score_cap_f / required_signal_scale
    return ReadoutMarginSizing(
        observed_margin_v=observed_margin_v,
        target_margin_v=target_margin_v,
        required_signal_scale=required_signal_scale,
        current_readout_width_u=current_readout_width_u,
        suggested_readout_width_u=suggested_readout_width_u,
        max_readout_width_u=max_readout_width_u,
        readout_width_feasible=suggested_readout_width_u <= max_readout_width_u,
        current_score_cap_f=current_score_cap_f,
        suggested_score_cap_f=suggested_score_cap_f,
        min_score_cap_f=min_score_cap_f,
        score_cap_feasible=suggested_score_cap_f >= min_score_cap_f,
    )


def derive_multiclass_readout_sizing(
    *,
    class_count: int,
    effective_readout_fan_in: float,
    learning_rate_scale: float = 1.0,
    error_drive_scale: float = 1.0,
    score_tau_scale: float = 1.0,
    anchor_fan_in: float = 8.0,
    anchor_update_width_u: float = 5e-4,
    anchor_guard_width_u: float = 8.0,
    anchor_target_width_u: float = 96.0,
    anchor_score_cap_f: float = 10.0,
    output_to_score_cap_ratio: float = 2.0,
    target_selector_ratio: float = 1.0,
    nontarget_selector_ratio: float = 1.0,
    output_bias_update_ratio: float = 1.0,
) -> MulticlassReadoutSizing:
    """Derive local readout sizes from topology plus global circuit scales.

    The formulas encode three first-order constraints:

    * row update mobility scales like ``1 / effective_readout_fan_in`` to keep
      the feature-norm stability yardstick near the anchor circuit;
    * score/output capacitance scales with fan-in to keep voltage swing roughly
      invariant under wider rows;
    * non-target one-vs-rest error drive scales as ``1 / (class_count - 1)`` so
      balanced epochs do not create a class-count-dependent row drift.
    * target/non-target selector widths and output-bias write width are fixed
      circuit-family ratios around the derived readout update width, not
      independent experiment knobs.
    """
    if class_count < 2:
        raise ValueError("class_count must be at least two.")
    for name, value in [
        ("effective_readout_fan_in", effective_readout_fan_in),
        ("learning_rate_scale", learning_rate_scale),
        ("error_drive_scale", error_drive_scale),
        ("score_tau_scale", score_tau_scale),
        ("anchor_fan_in", anchor_fan_in),
        ("anchor_update_width_u", anchor_update_width_u),
        ("anchor_guard_width_u", anchor_guard_width_u),
        ("anchor_target_width_u", anchor_target_width_u),
        ("anchor_score_cap_f", anchor_score_cap_f),
        ("output_to_score_cap_ratio", output_to_score_cap_ratio),
        ("target_selector_ratio", target_selector_ratio),
        ("nontarget_selector_ratio", nontarget_selector_ratio),
    ]:
        _require_positive(name, float(value))
    _require_nonnegative("output_bias_update_ratio", float(output_bias_update_ratio))

    readout_fan_in_scale = effective_readout_fan_in / anchor_fan_in
    write_scale = learning_rate_scale / readout_fan_in_scale
    readout_update_width_u = anchor_update_width_u * write_scale
    residual_target_width_u = anchor_target_width_u * error_drive_scale
    residual_output_width_u = residual_target_width_u / (class_count - 1)

    if not one_vs_rest_width_ratio_is_balanced(
        class_count,
        residual_target_width_u,
        residual_output_width_u,
    ):
        raise AssertionError("derived one-vs-rest widths should be balanced")

    score_cap_f = anchor_score_cap_f * score_tau_scale * readout_fan_in_scale

    return MulticlassReadoutSizing(
        class_count=class_count,
        effective_readout_fan_in=effective_readout_fan_in,
        readout_fan_in_scale=readout_fan_in_scale,
        readout_update_width_u=readout_update_width_u,
        readout_dp_gate_update_width_u=target_selector_ratio * readout_update_width_u,
        readout_dn_gate_update_width_u=nontarget_selector_ratio * readout_update_width_u,
        output_bias_update_width_u=output_bias_update_ratio * readout_update_width_u,
        readout_write_error_exclusion_width_u=anchor_guard_width_u * write_scale,
        residual_target_width_u=residual_target_width_u,
        residual_output_width_u=residual_output_width_u,
        score_cap_f=score_cap_f,
        output_cap_f=output_to_score_cap_ratio * score_cap_f,
    )


def derive_class_evidence_normalizer_sizing(
    *,
    class_count: int,
    normalized_score_delta_v: float,
    contrast_window_ns: float = 2.0,
    mass_window_ns: float = 0.5,
    error_window_ns: float = 0.7,
    target_error_v: float = 0.32,
    min_writer_gate_v: float = 0.01,
    score_common_cap_f: float = 4.0,
    contrast_cap_f: float = 10.0,
    anchor_score_common_tau_ns: float = 40.0,
    anchor_mass_cap_f: float = 0.5,
    anchor_error_cap_f: float = 0.5,
    anchor_mass_width_u: float = 128.0,
    anchor_error_width_u: float = 128.0,
    error_drive_scale: float = 1.0,
) -> ClassEvidenceNormalizerSizing:
    """Derive local class-evidence normalizer sizes from circuit constraints.

    The normalizer bridges low-common score evidence into writer-domain rails.
    These defaults encode the successful low-level primitive constraints:

    * common-reference loading must be slow relative to the contrast window so
      millivolt score separation is not shorted away;
    * score-mass and error caps are sized for a writer-useful rail voltage, not
      for final classification directly;
    * non-target error width is scaled by ``1 / (class_count - 1)`` so a
      balanced one-vs-rest epoch is not class-count biased.

    Bayesian search should normally tune ``error_drive_scale`` and high-level
    margins around this profile, not the individual local widths/capacitances.
    """
    if class_count < 2:
        raise ValueError("class_count must be at least two.")
    for name, value in [
        ("normalized_score_delta_v", normalized_score_delta_v),
        ("contrast_window_ns", contrast_window_ns),
        ("mass_window_ns", mass_window_ns),
        ("error_window_ns", error_window_ns),
        ("target_error_v", target_error_v),
        ("min_writer_gate_v", min_writer_gate_v),
        ("score_common_cap_f", score_common_cap_f),
        ("contrast_cap_f", contrast_cap_f),
        ("anchor_score_common_tau_ns", anchor_score_common_tau_ns),
        ("anchor_mass_cap_f", anchor_mass_cap_f),
        ("anchor_error_cap_f", anchor_error_cap_f),
        ("anchor_mass_width_u", anchor_mass_width_u),
        ("anchor_error_width_u", anchor_error_width_u),
        ("error_drive_scale", error_drive_scale),
    ]:
        _require_positive(name, float(value))

    contrast_observable_ratio = normalized_score_delta_v / min_writer_gate_v
    if contrast_observable_ratio < 0.25:
        raise ValueError("contrast_observable_ratio is too small for a writer-domain normalizer.")

    score_common_resistance_ohm = rc_resistance_for_tau_ohm(anchor_score_common_tau_ns, score_common_cap_f)
    mass_cap_f = required_cap_ff_for_step(
        target_error_v * anchor_mass_cap_f / max(mass_window_ns, 1e-30),
        mass_window_ns,
        target_error_v,
    )
    error_cap_f = required_cap_ff_for_step(
        target_error_v * anchor_error_cap_f / max(error_window_ns, 1e-30),
        error_window_ns,
        target_error_v,
    )
    mass_width_u = anchor_mass_width_u * error_drive_scale
    target_error_width_u = anchor_error_width_u * error_drive_scale
    nontarget_error_width_u = target_error_width_u / (class_count - 1)

    if not one_vs_rest_width_ratio_is_balanced(
        class_count,
        target_error_width_u,
        nontarget_error_width_u,
    ):
        raise AssertionError("derived normalizer error widths should be one-vs-rest balanced")

    return ClassEvidenceNormalizerSizing(
        class_count=class_count,
        normalized_score_delta_v=normalized_score_delta_v,
        score_common_resistance_ohm=score_common_resistance_ohm,
        score_common_tau_ns=rc_tau_ns(score_common_resistance_ohm, score_common_cap_f),
        contrast_observable_ratio=contrast_observable_ratio,
        score_common_cap_f=score_common_cap_f,
        contrast_cap_f=contrast_cap_f,
        mass_cap_f=mass_cap_f,
        error_cap_f=error_cap_f,
        mass_width_u=mass_width_u,
        target_error_width_u=target_error_width_u,
        nontarget_error_width_u=nontarget_error_width_u,
    )


def derive_multiclass_margin_correction_sizing(
    *,
    class_count: int,
    target_margin_v: float,
    score_delta_v: float,
    min_observable_score_delta_v: float = 1.0e-3,
    error_window_ns: float = 1.2,
    target_error_v: float = 0.43,
    sense_threshold_v: float = 0.02,
    anchor_error_cap_f: float = 0.5,
    anchor_error_width_u: float = 128.0,
    anchor_pairwise_pullup_width_u: float = 16.0,
    anchor_pairwise_pulldown_width_u: float = 64.0,
    margin_reference_window_v: float = 0.256,
    error_drive_scale: float = 1.0,
) -> MulticlassMarginCorrectionSizing:
    """Derive sizes for a pairwise target-margin correction primitive.

    The circuit contract is mistake/margin driven:

    ``score_opponent + target_margin > score_target``

    should create a writer-domain positive target rail and a negative rail for
    the offending opponent. Local device sizes are anchored to the existing
    low-gain pairwise score comparator and then scaled by the few high-level
    knobs that should remain visible to Bayesian search.
    """
    if class_count < 2:
        raise ValueError("class_count must be at least two.")
    for name, value in [
        ("target_margin_v", target_margin_v),
        ("score_delta_v", score_delta_v),
        ("min_observable_score_delta_v", min_observable_score_delta_v),
        ("error_window_ns", error_window_ns),
        ("target_error_v", target_error_v),
        ("sense_threshold_v", sense_threshold_v),
        ("anchor_error_cap_f", anchor_error_cap_f),
        ("anchor_error_width_u", anchor_error_width_u),
        ("anchor_pairwise_pullup_width_u", anchor_pairwise_pullup_width_u),
        ("anchor_pairwise_pulldown_width_u", anchor_pairwise_pulldown_width_u),
        ("margin_reference_window_v", margin_reference_window_v),
        ("error_drive_scale", error_drive_scale),
    ]:
        _require_positive(name, float(value))

    observable_margin_ratio = min(target_margin_v, score_delta_v) / min_observable_score_delta_v
    if observable_margin_ratio < 1.0:
        raise ValueError("target_margin_v and score_delta_v are below the observable score-delta floor.")

    error_cap_f = required_cap_ff_for_step(
        target_error_v * anchor_error_cap_f / error_window_ns,
        error_window_ns,
        target_error_v,
    )
    error_width_u = anchor_error_width_u * error_drive_scale
    margin_penalty_width_u = anchor_pairwise_pulldown_width_u * target_margin_v / margin_reference_window_v
    error_clock_high_v = min(1.2, target_error_v + sense_threshold_v)

    return MulticlassMarginCorrectionSizing(
        class_count=class_count,
        target_margin_v=target_margin_v,
        score_delta_v=score_delta_v,
        observable_margin_ratio=observable_margin_ratio,
        pairwise_pullup_width_u=anchor_pairwise_pullup_width_u,
        pairwise_pulldown_width_u=anchor_pairwise_pulldown_width_u,
        margin_penalty_width_u=margin_penalty_width_u,
        margin_reference_window_v=margin_reference_window_v,
        error_width_u=error_width_u,
        error_cap_f=error_cap_f,
        error_window_ns=error_window_ns,
        error_clock_high_v=error_clock_high_v,
        target_error_v=target_error_v,
    )
