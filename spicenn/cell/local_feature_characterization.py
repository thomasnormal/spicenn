from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .characterization import CharacterizationResult, PromotionLevel
from .local_feature import (
    LocalFeatureCell,
    activation_derivative_np,
    activation_np,
    synapse_transfer_np,
)


def characterize_local_feature_cell(cell: LocalFeatureCell, *, seed: int = 0) -> CharacterizationResult:
    rng = np.random.default_rng(seed)
    z = np.linspace(-2.5, 2.5, 401)
    h = activation_np(z, cell.local_activation, cell.relu_clip, cell.relu_leak, cell.softplus_beta)
    # Allow tiny numerical noise, but require the transfer to be monotone.
    monotonicity_violations = int(np.sum(np.diff(h) < -1e-9))

    deriv = activation_derivative_np(
        z,
        h,
        cell.local_activation,
        cell.relu_clip,
        cell.relu_leak,
        cell.softplus_beta,
        cell.activation_derivative,
        cell.derivative_floor,
        cell.derivative_gate_threshold,
    )
    derivative_negative = int(np.sum(deriv < -1e-12))

    # Synthetic single-cell update alignment: the reference local rule is
    # sign(delta * x).  Since x is nonnegative, a valid local write should align
    # with the hidden delta sign over the operating range.
    x = rng.uniform(0.0, 1.0, size=256)
    feedback = rng.normal(size=256)
    pre = rng.normal(size=256)
    act = activation_np(pre, cell.local_activation, cell.relu_clip, cell.relu_leak, cell.softplus_beta)
    gate = activation_derivative_np(
        pre,
        act,
        cell.local_activation,
        cell.relu_clip,
        cell.relu_leak,
        cell.softplus_beta,
        cell.activation_derivative,
        cell.derivative_floor,
        cell.derivative_gate_threshold,
    )
    update = feedback * gate * x
    reference = feedback * np.maximum(gate, 0.0) * x
    denom = float(np.linalg.norm(update) * np.linalg.norm(reference))
    update_cosine = 0.0 if denom == 0.0 else float(np.dot(update, reference) / denom)
    active = np.abs(reference) > 1e-12
    update_sign_alignment = 1.0 if not np.any(active) else float(np.mean(np.sign(update[active]) == np.sign(reference[active])))

    # A bounded effective synapse is preferred but not required.  This metric
    # detects NaNs and pathological transfer blow-up in candidate settings.
    weights = np.linspace(-4.0, 4.0, 129)
    eff_w = synapse_transfer_np(weights, cell.hidden_synapse_mode, cell.synapse_clip)
    finite_synapse = bool(np.all(np.isfinite(eff_w)))

    passed = (
        monotonicity_violations == 0
        and derivative_negative == 0
        and update_cosine > 0.0
        and update_sign_alignment == 1.0
        and finite_synapse
    )
    return CharacterizationResult(
        passed=passed,
        protocol_family=cell.protocol_family,
        promotion_level=PromotionLevel.LEARNING_ALIGNED if passed else PromotionLevel.CONTRACT_VALID,
        forward_error=None,
        forward_monotonicity_violations=monotonicity_violations,
        update_cosine=update_cosine,
        update_sign_alignment=update_sign_alignment,
        notes=[] if passed else [f"derivative_negative={derivative_negative}", f"finite_synapse={finite_synapse}"],
    )


def characterization_row(cell: LocalFeatureCell, *, seed: int = 0) -> dict[str, object]:
    result = characterize_local_feature_cell(cell, seed=seed)
    row = asdict(result)
    row["cell"] = cell.name
    row["description"] = cell.description
    row["protocol_family"] = cell.protocol_family.value
    row["promotion_level"] = result.promotion_level.name
    return row
