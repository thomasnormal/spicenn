from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


OBJECTIVE = (
    "Build the simplest possible programmable, trainable, preferably self-timed "
    "local neural hardware architecture that runs forward and training entirely "
    "in SPICE, is robust to realistic circuit noise, and achieves over 90% "
    "accuracy on full MNIST. Stochastic operation is allowed but not required."
)


def load_json(path: str) -> dict[str, Any] | None:
    p = ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pct(x: Any) -> str:
    if x is None:
        return "missing"
    try:
        return f"{100.0 * float(x):.2f}%"
    except (TypeError, ValueError):
        return str(x)


def sci(x: Any) -> str:
    if x is None:
        return "missing"
    try:
        return f"{float(x):.3g}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    spice_forward = load_json("spice/results/spice_mnist_mlp14x14_h32_200_op_inference_summary.json")
    digits_forward = load_json("spice/results/spice_digits_logreg_inference_summary.json")
    spice_lut = load_json("results/tables/spice_lut_mnist_calibration_summary.json")
    candidate = load_json("results/tables/hardware_mnist_candidate_summary.json")
    forward_settle = load_json(
        "spice/results/spice_mnist_forward_settle_14x14_b7_100test_tau05ns_literalw_dt50ps_from879_summary.json"
    )
    settling_pareto = load_json(
        "spice/results/spice_mnist_settling_pareto_14x14_b7_1000test_taugrid_from879_summary.json"
    )
    local_feature_settling_b5 = load_json(
        "spice/results/spice_mnist_local_feature_settling_pareto_smallnet10_b5_s2_c2_2k_lr003_taugrid_summary.json"
    )
    local_feature_settling_b4 = load_json(
        "spice/results/spice_mnist_local_feature_settling_pareto_smallnet10_b4_s2_c2_2k_lr008_taugrid_summary.json"
    )
    bias_cal_subset = load_json(
        "spice/results/spice_mnist_output_bias_cal_14x14_b7_cal2000_test1000_from879_summary.json"
    )
    bias_cal_full_14 = load_json(
        "spice/results/spice_mnist_output_bias_cal_14x14_b7_cal60000_test10000_from879_summary.json"
    )
    bias_cal_full_28 = load_json(
        "spice/results/spice_mnist_output_bias_cal_28x28_b14_cal60000_test10000_from881_summary.json"
    )
    bias_cal_spice_eval = load_json(
        "spice/results/spice_mnist_local_block_local_block_14x14_b7_biascal_eval1000_from879_summary.json"
    )
    local_feature_torch = load_json(
        "spice/results/spice_mnist_local_feature_torch_14x14_b7_c32_full_e20_lr001_summary.json"
    )
    local_feature_spice_eval = load_json(
        "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c32_torchbest_eval1000_spice_summary.json"
    )
    local_feature_spice_update = load_json(
        "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c32_torchbest_spice_update_smoke_summary.json"
    )
    local_feature_phase = load_json(
        "spice/results/spice_mnist_local_feature_phase_torchbest_14x14_b7_c32_linear_bs1_u1_summary.json"
    )
    local_feature_batch_phase = load_json(
        "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs2_u1_phase4ns_summary.json"
    )
    local_feature_multibatch_phase = load_json(
        "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs2_u2_phase4ns_summary.json"
    )
    local_feature_phase_2ns = load_json(
        "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs2_u2_phase2ns_summary.json"
    )
    local_feature_phase_1ns = load_json(
        "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs2_u2_phase1ns_summary.json"
    )
    local_feature_phase_train = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_c8_msetanh_lrspice03_eval100_bs2_u2_phase1ns_fromsmallrule_summary.json"
    )
    local_feature_small_sweep = load_json(
        "spice/results/spice_mnist_local_feature_small_rule_2k1k_msetanh_c8-16_e60_summary.json"
    )
    local_feature_noise_sweep = load_json(
        "spice/results/spice_mnist_local_feature_update_noise_2k1k_c8-16_e60_summary.json"
    )
    local_feature_small_frontier = load_json(
        "spice/results/spice_mnist_local_feature_small_frontier_1k_i8-14_c4-16_e60_summary.json"
    )
    local_feature_overlap_frontier = load_json(
        "spice/results/spice_mnist_local_feature_overlap_frontier_1k_i8-14_c2-4_e60_summary.json"
    )
    local_feature_overlap_refine = load_json(
        "spice/results/spice_mnist_local_feature_overlap_refine_1k2k_i12-14_e80_summary.json"
    )
    local_feature_overlap_noise = load_json(
        "spice/results/spice_mnist_local_feature_overlap_noise_2k_i12_c2-4_e80_summary.json"
    )
    local_feature_overlap_lowlr = load_json(
        "spice/results/spice_mnist_local_feature_overlap_lowlr_2k_i12_c2_e120_summary.json"
    )
    local_feature_overlap_batch2 = load_json(
        "spice/results/spice_mnist_local_feature_overlap_batch2_2k_i12_c2_e40_summary.json"
    )
    local_feature_smallnets_screen = load_json(
        "spice/results/spice_mnist_local_feature_smallnets_batch2_screen_1k_i8-12_e35_summary.json"
    )
    local_feature_smallnets_promote = load_json(
        "spice/results/spice_mnist_local_feature_smallnets_batch2_promote_2k_i10-12_e60_summary.json"
    )
    local_feature_smallnet10_b5_fast_lr003 = load_json(
        "spice/results/fast_mnist_smallnet10_b5_s2_c2_lr003_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b5_fast_lr005 = load_json(
        "spice/results/fast_mnist_smallnet10_b5_s2_c2_lr005_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b5_scale_5k2k_hilr = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b5_s2_c2_scale_5k2k_e120_hilr_summary.json"
    )
    local_feature_smallnet10_b5_scale_10k2k_hilr = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b5_s2_c2_scale_10k2k_e120_hilr_summary.json"
    )
    local_feature_smallnet10_b5_scale_10k2k_fast_lr003 = load_json(
        "spice/results/fast_mnist_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_fastcont_lr003_0to212_summary.json"
    )
    local_feature_smallnet10_b5_scale_10k2k_fast_lr003_resume1024 = load_json(
        "spice/results/fast_mnist_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_fastcont_lr003_212to1024_summary.json"
    )
    local_feature_smallnet10_b5_phase_scale_10k2k_lr003 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_track12_lr003_eval2000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume60 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_resume12to60_lr003_eval2000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b5_scale_10k2k_fast_from_spice60_lr003 = load_json(
        "spice/results/fast_mnist_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_spice60_fast_lr003_60to212_summary.json"
    )
    local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume212 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_resume60to212_lr003_eval2000_phase1ns_summary.json"
    )
    local_feature_tiny_below_b5_screen = load_json(
        "spice/results/spice_mnist_local_feature_tiny_below_b5_screen_2k2k_e60_summary.json"
    )
    local_feature_tiny_below_b5_promote = load_json(
        "spice/results/spice_mnist_local_feature_tiny_below_b5_promote_5k10k_e120_summary.json"
    )
    local_feature_tiny_below_b5_c1_fast1024 = load_json(
        "spice/results/fast_mnist_tiny_below_b5_i10_b4_s2_c1_10k2k_e120_lr003_fastcont_lr002_0to1024_summary.json"
    )
    local_feature_tiny_below_b5_c1_phase12 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_below_b5_i10_b4_s2_c1_10k2k_e120_lr003_track12_lr002_eval2000_phase1ns_summary.json"
    )
    local_feature_tiny_below_b5_c2_fast1024 = load_json(
        "spice/results/fast_mnist_tiny_below_b5_i9_b4_s2_c2_10k2k_e120_lr005_fastcont_lr002_0to1024_summary.json"
    )
    local_feature_tiny_below_b5_c2_phase12 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_below_b5_i9_b4_s2_c2_10k2k_e120_lr005_track12_lr002_eval2000_phase1ns_summary.json"
    )
    local_feature_tiny_below_b5_c2_fast_from_spice12 = load_json(
        "spice/results/fast_mnist_tiny_below_b5_i9_b4_s2_c2_10k2k_e120_lr005_spice12_fast_lr002_12to60_summary.json"
    )
    local_feature_tiny_below_b5_c2_phase60 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_below_b5_i9_b4_s2_c2_10k2k_e120_lr005_resume12to60_lr002_eval2000_phase1ns_summary.json"
    )
    local_feature_sub936_promote_5k = load_json(
        "spice/results/spice_mnist_local_feature_sub936_promote_5k2k_e140_summary.json"
    )
    local_feature_i10_b6_s2_c1_promote_10k = load_json(
        "spice/results/spice_mnist_local_feature_i10_b6_s2_c1_promote_10k2k_e180_summary.json"
    )
    local_feature_i10_b6_s2_c1_fast212 = load_json(
        "spice/results/fast_mnist_tiny_i10_b6_s2_c1_10k2k_e180_lr001_fastcont_lr001_0to212_summary.json"
    )
    local_feature_i10_b6_s2_c1_fast1024 = load_json(
        "spice/results/fast_mnist_tiny_i10_b6_s2_c1_10k2k_e180_lr001_fastcont_lr001_0to1024_summary.json"
    )
    local_feature_i10_b6_s2_c1_phase12 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_i10_b6_s2_c1_10k2k_e180_lr001_track12_eval2000_phase1ns_summary.json"
    )
    local_feature_i10_b6_s2_c1_fast_from_spice12_to60 = load_json(
        "spice/results/fast_mnist_tiny_i10_b6_s2_c1_10k2k_e180_lr001_spice12_fast_lr001_12to60_summary.json"
    )
    local_feature_i10_b6_s2_c1_phase60 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_i10_b6_s2_c1_10k2k_e180_lr001_resume12to60_eval2000_phase1ns_summary.json"
    )
    local_feature_i10_b6_s2_c1_fast_from_spice12 = load_json(
        "spice/results/fast_mnist_tiny_i10_b6_s2_c1_10k2k_e180_lr001_spice12_fast_lr001_12to212_summary.json"
    )
    local_feature_sub904_broad_2k = load_json(
        "spice/results/spice_mnist_local_feature_sub904_broad_2k2k_e80_summary.json"
    )
    local_feature_sub904_promote_5k = load_json(
        "spice/results/spice_mnist_local_feature_sub904_promote_5k2k_e160_summary.json"
    )
    local_feature_i7_b3_s2_c2_promote_10k = load_json(
        "spice/results/spice_mnist_local_feature_i7_b3_s2_c2_promote_10k2k_e180_summary.json"
    )
    local_feature_i7_b3_s2_c2_fast1024 = load_json(
        "spice/results/fast_mnist_tiny_i7_b3_s2_c2_10k2k_e180_lr001_fastcont_lr001_0to1024_summary.json"
    )
    local_feature_i7_b3_s2_c2_phase12 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_i7_b3_s2_c2_10k2k_e180_lr001_track12_eval2000_phase1ns_summary.json"
    )
    local_feature_i7_b3_s2_c2_fast_from_spice12 = load_json(
        "spice/results/fast_mnist_tiny_i7_b3_s2_c2_10k2k_e180_lr001_spice12_fast_lr001_12to212_summary.json"
    )
    local_feature_i7_b3_s2_c2_fast_from_spice12_1024 = load_json(
        "spice/results/fast_mnist_tiny_i7_b3_s2_c2_10k2k_e180_lr001_spice12_fast_lr001_12to1024_summary.json"
    )
    local_feature_i7_b3_s2_c2_fast_from_spice12_to60 = load_json(
        "spice/results/fast_mnist_tiny_i7_b3_s2_c2_10k2k_e180_lr001_spice12_fast_lr001_12to60_summary.json"
    )
    local_feature_i7_b3_s2_c2_phase60 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_i7_b3_s2_c2_10k2k_e180_lr001_resume12to60_eval2000_phase1ns_summary.json"
    )
    local_feature_i7_b3_s2_c2_phase212 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_i7_b3_s2_c2_10k2k_e180_lr001_resume60to212_eval2000_phase1ns_summary.json"
    )
    local_feature_i7_b3_s2_c2_settling = load_json(
        "spice/results/spice_mnist_local_feature_settling_pareto_tiny_i7_b3_s2_c2_10k2k_lr001_taugrid_summary.json"
    )
    local_feature_sub796_promote_10k = load_json(
        "spice/results/spice_mnist_local_feature_sub796_promote_10k2k_e220_summary.json"
    )
    local_feature_sub706_pilot_2k = load_json(
        "spice/results/spice_mnist_local_feature_sub706_pilot_2k2k_e80_summary.json"
    )
    local_feature_shared_kernel_pilot_2k = load_json(
        "spice/results/spice_mnist_local_feature_shared_kernel_pilot_2k2k_e80_summary.json"
    )
    local_feature_shared_kernel_promote_5k = load_json(
        "spice/results/spice_mnist_local_feature_shared_kernel_promote_5k2k_e120_summary.json"
    )
    local_feature_partial_sharing_pilot_2k = load_json(
        "spice/results/spice_mnist_local_feature_partial_sharing_pilot_2k2k_e80_summary.json"
    )
    local_feature_partial_sharing_promote_5k = load_json(
        "spice/results/spice_mnist_local_feature_partial_sharing_promote_5k2k_e120_summary.json"
    )
    local_feature_partial_sharing_i9_b4_fast212 = load_json(
        "spice/results/fast_mnist_partialsem_i9_b4_s2_sh1_pr1_5k2k_lr003_fastcont_lr003_0to212_summary.json"
    )
    local_feature_partial_sharing_i9_b3_fast1024 = load_json(
        "spice/results/fast_mnist_partialsem_i9_b3_s2_sh2_pr0_5k2k_lr005_fastcont_lr001_0to1024_summary.json"
    )
    local_feature_partial_sharing_i7_b3_fast1024 = load_json(
        "spice/results/fast_mnist_partialsem_i7_b3_s2_sh2_pr1_5k2k_lr003_fastcont_lr001_0to1024_summary.json"
    )
    local_feature_partial_phase_u1 = load_json(
        "spice/results/spice_mnist_partial_sharing_phase_partial_i7_b3_s2_sh2_pr1_lr001_bs2_u1_phase1ns_meas_summary.json"
    )
    local_feature_partial_phase_u2 = load_json(
        "spice/results/spice_mnist_partial_sharing_phase_partial_i7_b3_s2_sh2_pr1_lr001_bs2_u2_phase1ns_meas_summary.json"
    )
    local_feature_partial_phase_train12 = load_json(
        "spice/results/spice_mnist_partial_sharing_phase_train_partial_i7_b3_s2_sh2_pr1_5k2k_lr003_track12_lr001_eval200_phase1ns_summary.json"
    )
    local_feature_partial_phase12_eval2000 = load_json(
        "spice/results/spice_mnist_local_feature_partial_i7_b3_s2_sh2_pr1_phase12_eval2000_summary.json"
    )
    local_feature_partial_i9_b3_phase_train12 = load_json(
        "spice/results/spice_mnist_partial_sharing_phase_train_partial_i9_b3_s2_sh2_pr0_5k2k_lr005_track12_lr001_eval200_phase1ns_summary.json"
    )
    local_feature_partial_i9_b3_phase12_eval2000 = load_json(
        "spice/results/spice_mnist_local_feature_partial_i9_b3_s2_sh2_pr0_phase12_eval2000_summary.json"
    )
    local_feature_i7_b5_s1_c1_fast212 = load_json(
        "spice/results/fast_mnist_tiny_i7_b5_s1_c1_10k2k_e220_lr002_fastcont_lr002_0to212_summary.json"
    )
    local_feature_i7_b5_s1_c1_fast1024 = load_json(
        "spice/results/fast_mnist_tiny_i7_b5_s1_c1_10k2k_e220_lr002_fastcont_lr002_0to1024_summary.json"
    )
    local_feature_i7_b5_s1_c1_phase12 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_tiny_i7_b5_s1_c1_10k2k_e220_lr002_track12_eval2000_phase1ns_summary.json"
    )
    local_feature_i7_b5_s1_c1_fast_from_spice12 = load_json(
        "spice/results/fast_mnist_tiny_i7_b5_s1_c1_10k2k_e220_lr002_spice12_fast_lr002_12to212_summary.json"
    )
    local_feature_i7_b5_s1_c1_fast_from_spice12_1024 = load_json(
        "spice/results/fast_mnist_tiny_i7_b5_s1_c1_10k2k_e220_lr002_spice12_fast_lr002_12to1024_summary.json"
    )
    local_feature_smallnet10_b4_fast_lr008 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_lr008_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b4_fast_lr005 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_lr005_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b5_noise = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b5_s2_c2_noise_2k_e60_summary.json"
    )
    local_feature_smallnet10_b4_noise = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_noise_2k_e60_summary.json"
    )
    local_feature_smallnet10_b4_phase_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_smallnet10_b4_s2_c2_lr008_bs2_u2_phase1ns_meas_train2000_summary.json"
    )
    local_feature_smallnet10_b4_phase_tracker_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_track12_eval1000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_phase_resume12to60_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_resume12to60_eval1000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_phase_resume60to120_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_resume60to120_eval1000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_phase_resume120to180_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_resume120to180_eval1000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_phase_resume180to212_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_resume180to212_eval1000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_handoff_e12 = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_handoff_e12_summary.json"
    )
    local_feature_smallnet10_b4_handoff_e15 = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_handoff_e15_summary.json"
    )
    local_feature_smallnet10_b4_handoff_e15_fast_lr015 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_handoff_e15_lr015_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b4_phase_handoff_e15_lr015 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_handoff_e15_lr015_track212_eval1000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_handoff_e20 = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_handoff_e20_summary.json"
    )
    local_feature_smallnet10_b4_handoff_e20_fast_lr01 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_handoff_e20_lr01_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b4_phase_handoff_e20_lr01 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_handoff_e20_lr01_track212_eval1000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_scale_5k2k = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_scale_5k2k_e25_summary.json"
    )
    local_feature_smallnet10_b4_scale_5k2k_fast_lr008 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_5k2k_e25_lr008_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b4_phase_scale_5k2k_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_5k2k_e25_lr008_track212_eval2000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_scale_10k2k = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_scale_10k2k_e25_summary.json"
    )
    local_feature_smallnet10_b4_scale_10k2k_fast_lr008 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_10k2k_e25_lr008_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b4_phase_scale_10k2k_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_10k2k_e25_lr008_track212_eval2000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_scale_10kfull_fast_lr008 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_10kfull_e25_lr008_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b4_phase_scale_10kfull_lr008 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_10kfull_e25_lr008_track212_eval10000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_scale_60kfull = load_json(
        "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_scale_60kfull_e25_summary.json"
    )
    local_feature_smallnet10_b4_scale_60kfull_fast_lr003 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_fast_0to212_summary.json"
    )
    local_feature_smallnet10_b4_phase_scale_60kfull_lr003 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_track212_eval10000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume1024 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_fast_resume212to1024_summary.json"
    )
    local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume2048 = load_json(
        "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_fast_resume1024to2048_summary.json"
    )
    local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume424 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_resume212to424_eval10000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume600 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_resume424to600_eval10000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume1024 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_resume600to1024_eval10000_phase1ns_summary.json"
    )
    local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume2048 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_resume1024to2048_eval10000_phase1ns_summary.json"
    )
    local_feature_overlap_handoff_e6_lr005 = load_json(
        "spice/results/spice_mnist_local_feature_overlap_batch2_handoff_e6_lr005_summary.json"
    )
    local_feature_overlap_handoff_e10_lr005 = load_json(
        "spice/results/spice_mnist_local_feature_overlap_batch2_handoff_e10_lr005_summary.json"
    )
    local_feature_overlap_phase_lr03 = load_json(
        "spice/results/spice_mnist_local_feature_phase_overlap12_b6_s2_c2_lr03_bs2_u2_phase1ns_meas_summary.json"
    )
    local_feature_overlap_phase_lr05 = load_json(
        "spice/results/spice_mnist_local_feature_phase_overlap12_b6_s2_c2_lr05_bs2_u2_phase1ns_meas_summary.json"
    )
    local_feature_overlap_phase_train12_1ns = load_json(
        "spice/results/spice_mnist_local_feature_phase_overlap12_b6_s2_c2_lr03_bs2_u2_phase1ns_meas_train12_summary.json"
    )
    local_feature_overlap_phase_train12_4ns = load_json(
        "spice/results/spice_mnist_local_feature_phase_overlap12_b6_s2_c2_lr03_bs2_u2_phase4ns_meas_train12_summary.json"
    )
    local_feature_overlap_phase_lr015 = load_json(
        "spice/results/spice_mnist_local_feature_phase_overlap12_b6_s2_c2_lr015_bs2_u2_phase1ns_meas_train12_summary.json"
    )
    local_feature_overlap_phase_batch2_lr01 = load_json(
        "spice/results/spice_mnist_local_feature_phase_overlap12_b6_s2_c2_batch2_lr01_bs2_u2_phase1ns_meas_train2000_summary.json"
    )
    local_feature_overlap_phase_tracker = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_lr03_track12_phase1ns_summary.json"
    )
    local_feature_overlap_phase_tracker_lr015 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_lr015_track12_phase1ns_summary.json"
    )
    local_feature_overlap_phase_resume_lr015 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_lr015_resume12to32_eval200_phase1ns_summary.json"
    )
    local_feature_overlap_phase_tracker_batch2_lr01 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_batch2_lr01_track12_eval1000_phase1ns_summary.json"
    )
    local_feature_overlap_fast_resume_batch2_lr01 = load_json(
        "spice/results/fast_mnist_overlap12_b6_s2_c2_batch2_lr01_fast_resume12to212_summary.json"
    )
    local_feature_overlap_phase_resume_batch2_lr01 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_batch2_lr01_resume12to60_eval1000_phase1ns_summary.json"
    )
    local_feature_overlap_fast_handoff_e6_lr005 = load_json(
        "spice/results/fast_mnist_overlap12_b6_s2_c2_batch2_lr005_e6_fast_0to212_summary.json"
    )
    local_feature_overlap_fast_handoff_e10_lr005 = load_json(
        "spice/results/fast_mnist_overlap12_b6_s2_c2_batch2_lr005_e10_fast_0to212_summary.json"
    )
    local_feature_overlap_phase_handoff_e10_lr005 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_batch2_lr005_e10_track60_eval1000_phase1ns_summary.json"
    )
    local_feature_overlap_phase_handoff_e10_lr005_resume = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_batch2_lr005_e10_resume60to120_eval1000_phase1ns_summary.json"
    )
    local_feature_frontier_phase = load_json(
        "spice/results/spice_mnist_local_feature_phase_smallfrontier14_c8_lr08_bs2_u2_phase1ns_meas_summary.json"
    )
    local_feature_phase_train_tracker = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr08_random_track12_phase1ns_summary.json"
    )
    local_feature_phase_train_tracker_lr03 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr03_random_track12_phase1ns_summary.json"
    )
    local_feature_phase_train_tracker_lr08_2ns = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr08_random_track12_phase2ns_summary.json"
    )
    local_feature_phase_train_tracker_lr03_20 = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr03_random_track20_phase1ns_summary.json"
    )
    local_feature_phase_train_resume_eval = load_json(
        "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr03_resume20to40_eval200_phase1ns_summary.json"
    )
    device_relu_synapse = load_json("spice/results/device_relu_synapse_update_v0_summary.json")
    device_signed_learning = load_json("spice/results/device_signed_learning_cell_v2_summary.json")
    device_delta_cells = load_json("spice/results/device_delta_cells_v1_summary.json")
    device_tiny_classifier = load_json("spice/results/device_tiny_classifier_v6_summary.json")
    device_sequential_training = load_json("spice/results/device_sequential_training_v0_summary.json")
    device_multicell_classifier = load_json("spice/results/device_multicell_classifier_v1_summary.json")
    device_feedback_alignment = load_json("spice/results/device_feedback_alignment_v0_summary.json")
    device_parity3_readout = load_json("spice/results/device_parity3_readout_v0_summary.json")
    device_xor2_learned_features = load_json("spice/results/device_xor2_learned_features_v0_summary.json")
    device_xor2_hidden_repair = load_json("spice/results/device_xor2_hidden_repair_v11_summary.json")
    device_xor2_random_hidden = load_json("spice/results/device_xor2_random_hidden_v2_summary.json")
    device_xor2_random_hidden_wide = load_json("spice/results/device_xor2_random_hidden_v31_summary.json")
    device_xor2_readout_rule = load_json("spice/results/device_xor2_random_hidden_v48_summary.json")
    device_xor2_lowtarget_rule = load_json("spice/results/device_xor2_random_hidden_v53_summary.json")
    device_xor2_separator_readout = load_json("spice/results/device_xor2_random_hidden_v69_summary.json")
    device_xor2_separator_score_train = load_json("spice/results/device_xor2_random_hidden_v76_summary.json")
    device_xor2_mistake_gate_random = load_json("spice/results/device_xor2_random_hidden_v88_summary.json")
    device_xor2_mistake_gate_separator = load_json("spice/results/device_xor2_random_hidden_v89_summary.json")
    device_xor2_local_loss_random = load_json("spice/results/device_xor2_random_hidden_v92_summary.json")
    device_xor2_local_loss_separator = load_json("spice/results/device_xor2_random_hidden_v91_summary.json")
    device_xor2_local_loss_strong = load_json("spice/results/device_xor2_random_hidden_v94_summary.json")
    device_xor2_output_senseamp_batch = load_json("spice/results/device_xor2_random_hidden_v102_summary.json")
    device_xor2_output_senseamp_online = load_json("spice/results/device_xor2_random_hidden_v104_summary.json")
    device_xor2_output_senseamp_best = load_json("spice/results/device_xor2_random_hidden_v116_summary.json")
    device_xor2_backprop_score = load_json("spice/results/device_xor2_random_hidden_v119_backprop_score_e8_summary.json")
    device_xor2_backprop_grad_probe = load_json("spice/results/device_xor2_random_hidden_v127_backprop_score_e1_hd64_meas_summary.json")
    device_xor2_backprop_hgsense = load_json(
        "spice/results/device_xor2_random_hidden_v140_backprop_score_e8_hgsense_w96_c20_hu001_summary.json"
    )
    device_xor2_backprop_gate = load_json(
        "spice/results/device_xor2_random_hidden_v141_backprop_score_e8_hgsense_refactor_gate_summary.json"
    )
    device_xor2_flow_smoke = load_json(
        "spice/results/device_xor2_random_hidden_v142_flow_smoke_summary.json"
    )
    device_xor2_flow_prestore_gate = load_json(
        "spice/results/device_xor2_random_hidden_v146_flow_prestore_gate_e1_summary.json"
    )
    device_xor2_flow_prestore_consume = load_json(
        "spice/results/device_xor2_random_hidden_v147_flow_prestore_consume_e1_summary.json"
    )
    device_xor2_flow_readout_only_solved = load_json(
        "spice/results/device_xor2_random_hidden_v159_flow_readout_only_e32_w015_b002_summary.json"
    )
    device_xor2_flow_prestore_readout_only_solved = load_json(
        "spice/results/device_xor2_random_hidden_v161_flow_prestore_gate_readout_only_e32_w015_b002_summary.json"
    )
    device_xor2_flow_alllayer_solved = load_json(
        "spice/results/device_xor2_random_hidden_v165_flow_alllayer_light_e32_w015_h000001_b002_summary.json"
    )
    device_mnist01_8_flow_readout_only_strong = load_json(
        "spice/results/device_mnist01_8_random_hidden_v16_flow_readout_only_e8_w015_b002_summary.json"
    )
    device_mnist01_8_flow_readout_only_best = load_json(
        "spice/results/device_mnist01_8_random_hidden_v19_flow_readout_only_light_e10_w01_b0015_summary.json"
    )
    device_moons8_eval = load_json("spice/results/device_moons8_random_hidden_v0_eval_smoke_summary.json")
    device_moons8_score = load_json("spice/results/device_moons8_random_hidden_v1_score_e2_hgsense_summary.json")
    device_moons8_score_strong = load_json(
        "spice/results/device_moons8_random_hidden_v3_score_e2_strong_readout_summary.json"
    )
    device_mnist01_8_eval = load_json(
        "spice/results/device_mnist01_8_random_hidden_v1_norm_eval_smoke_summary.json"
    )
    device_mnist01_8_score_strong = load_json(
        "spice/results/device_mnist01_8_random_hidden_v2_norm_score_e2_strong_readout_summary.json"
    )
    device_mnist01_8_score_gentle = load_json(
        "spice/results/device_mnist01_8_random_hidden_v3_norm_score_e3_gentle_readout_summary.json"
    )
    device_mnist01_8_outcomp_bias = load_json(
        "spice/results/device_mnist01_8_random_hidden_v6_outcomp_e3_bias50m_summary.json"
    )
    device_mnist01_8_outcomp_hd16 = load_json(
        "spice/results/device_mnist01_8_random_hidden_v9_outcomp_e4_bias50m_hd16_summary.json"
    )

    spice_train_runs = []
    for path in sorted((ROOT / "spice/results").glob("spice_mnist*_summary.json")):
        data = json.loads(path.read_text())
        if "heldout_test_accuracy" not in data:
            continue
        if data.get("eval_only"):
            continue
        spice_train_runs.append((path, data))
    substantive_spice_train_runs = [
        (path, data)
        for path, data in spice_train_runs
        if int(data.get("train_samples") or 0) >= 1000 and int(data.get("test_samples") or 0) >= 1000
    ]
    best_spice_train_acc = None
    best_spice_train_note = "No all-SPICE MNIST training summary found."
    if substantive_spice_train_runs:
        best_path, best_train = max(
            substantive_spice_train_runs,
            key=lambda item: float(item[1].get("heldout_test_accuracy") or -1),
        )
        best_spice_train_acc = best_train.get("heldout_test_accuracy")
        best_spice_train_note = (
            f"Best substantive all-ngspice training run: {best_train.get('train_samples')} train / "
            f"{best_train.get('test_samples')} held-out, image_size={best_train.get('image_size')}, "
            f"held-out accuracy {pct(best_spice_train_acc)}, summary {best_path}."
        )
    elif spice_train_runs:
        best_path, best_train = max(spice_train_runs, key=lambda item: float(item[1].get("heldout_test_accuracy") or -1))
        best_spice_train_acc = best_train.get("heldout_test_accuracy")
        best_spice_train_note = (
            f"Only tiny all-ngspice training/smoke runs found; highest small-run accuracy {pct(best_spice_train_acc)} "
            f"at {best_train.get('train_samples')} train / {best_train.get('test_samples')} held-out, summary {best_path}."
        )

    best_spice_forward_acc = spice_forward.get("spice_accuracy") if spice_forward else None
    best_spice_forward_note = (
        "No SPICE forward-inference MNIST summary found."
        if not spice_forward
        else (
            f"SPICE forward-only MNIST proxy: {spice_forward.get('test_samples')} held-out samples, "
            f"image_size={spice_forward.get('image_size')}, hidden={spice_forward.get('hidden')}, "
            f"accuracy {pct(best_spice_forward_acc)}; training was offline."
        )
    )

    best_digits_forward_note = (
        "No SPICE digits forward summary found."
        if not digits_forward
        else (
            f"SPICE forward-only sklearn digits proxy: {digits_forward.get('test_samples')} held-out samples, "
            f"accuracy {pct(digits_forward.get('spice_accuracy'))}; not MNIST."
        )
    )

    forward_settle_note = (
        "No SPICE forward-settling transient summary found."
        if not forward_settle
        else (
            f"SPICE local-block transient settling probe: {forward_settle.get('test_samples')} held-out samples, "
            f"image_size={forward_settle.get('image_size')}, endpoint accuracy "
            f"{pct(forward_settle.get('steady_accuracy_at_t_stop'))}, max transient accuracy "
            f"{pct(forward_settle.get('max_accuracy'))}; fixed checkpoint weights, not a training run."
        )
    )
    device_relu_note = (
        "No transistor/device-level ReLU synapse primitive sweep found."
        if not device_relu_synapse
        else (
            "A MOS device-level primitive sweep now tests the intended lower-level replacement for the behavioral tanh/multiplier cell: "
            f"conductance synapses charge a pre-activation capacitor, an NMOS source follower stores a ReLU-like activation, "
            f"and differential gradient-cap voltages charge/discharge a weight capacitor during an apply pulse. "
            f"The sweep used ngspice built-in LEVEL=1 MOS models rather than a foundry PDK; ReLU transfer monotone="
            f"{device_relu_synapse.get('relu_transfer_monotone')}, max single-synapse preactivation "
            f"{sci(device_relu_synapse.get('max_single_synapse_vpre'))} V, max activation "
            f"{sci(device_relu_synapse.get('max_single_synapse_vact'))} V, max positive weight delta "
            f"{sci(device_relu_synapse.get('max_positive_weight_delta_v'))} V, and max negative weight delta "
            f"{sci(device_relu_synapse.get('max_negative_weight_delta_v'))} V. "
            "This is a primitive validation, not an MNIST training result."
        )
    )
    device_signed_note = (
        "No transistor/device-level signed learning-cell sweep found."
        if not device_signed_learning
        else (
            "A follow-up MOS device-level signed learning-cell sweep now adds differential positive/negative weight "
            "capacitors and data-derived gradient accumulation: input/delta/acc transistor stacks charge Cgp/Cgn, "
            "then apply-phase transistor stacks update Cwp/Cwn. "
            f"Positive and negative accumulator monotonicity by input were "
            f"{device_signed_learning.get('positive_gradient_accumulator_monotone_by_input')}/"
            f"{device_signed_learning.get('negative_gradient_accumulator_monotone_by_input')}; signed updates were monotone "
            f"by input and delta, strong positive updates increased activation="
            f"{device_signed_learning.get('positive_gradient_increases_activation_when_update_strong')}, and strong negative "
            f"updates decreased activation={device_signed_learning.get('negative_gradient_decreases_activation_when_update_strong')}. "
            f"Max signed-weight changes were +{sci(device_signed_learning.get('max_positive_signed_delta_v'))} V and "
            f"{sci(device_signed_learning.get('max_negative_signed_delta_v'))} V. "
            "This still does not compute network output error or backprop deltas."
        )
    )
    device_delta_note = (
        "No transistor/device-level error/delta cell sweep found."
        if not device_delta_cells
        else (
            "A further MOS device-level sweep adds output-error and hidden-delta capacitor cells: target/output "
            "conductance competition writes dplus/dminus, and output-delta/readout-weight sign-combination stacks "
            "write hidden positive/negative delta caps. "
            f"Differential error was monotone with target at low/high output="
            f"{device_delta_cells.get('error_net_monotone_by_target_low_output')}/"
            f"{device_delta_cells.get('error_net_monotone_by_target_high_output')}; differential error decreased with output="
            f"{device_delta_cells.get('error_net_decreases_by_output_high_target')}/"
            f"{device_delta_cells.get('error_net_decreases_by_output_low_target')}; all four hidden-delta sign cases and "
            f"inactive ReLU suppression passed. Max error caps were "
            f"{sci(device_delta_cells.get('max_error_dplus_v'))}/{sci(device_delta_cells.get('max_error_dminus_v'))} V, "
            f"and max hidden delta caps were {sci(device_delta_cells.get('max_hidden_hdp_v'))}/"
            f"{sci(device_delta_cells.get('max_hidden_hdn_v'))} V. "
            "This is still a primitive sweep, not a multi-cell classifier."
        )
    )
    device_tiny_note = (
        "No transistor/device-level tiny classifier loop found."
        if not device_tiny_classifier
        else (
            "A tiny MOS device-level classifier now connects the primitive cells into one forward/error/backward/update loop: "
            "one input drives one signed hidden ReLU cell, one signed readout ReLU output, output-error caps, hidden-delta caps, "
            "readout/hidden gradient caps, and apply-phase weight updates. "
            f"High-target checks passed for positive error/readout update/hidden update/output increase="
            f"{device_tiny_classifier.get('high_target_error_positive')}/"
            f"{device_tiny_classifier.get('high_target_readout_update_positive')}/"
            f"{device_tiny_classifier.get('high_target_hidden_update_positive')}/"
            f"{device_tiny_classifier.get('high_target_output_increased')}; low-target checks passed for negative "
            f"error/readout update/hidden update/output decrease="
            f"{device_tiny_classifier.get('low_target_error_negative')}/"
            f"{device_tiny_classifier.get('low_target_readout_update_negative')}/"
            f"{device_tiny_classifier.get('low_target_hidden_update_negative')}/"
            f"{device_tiny_classifier.get('low_target_output_decreased')}. "
            f"Readout signed-weight deltas were +{sci(device_tiny_classifier.get('high_target_d_readout_signed'))} V and "
            f"{sci(device_tiny_classifier.get('low_target_d_readout_signed'))} V. "
            "This is a two-case smoke test under LEVEL=1 MOS models, not a scaled MNIST result."
        )
    )
    device_sequential_note = (
        "No transistor/device-level sequential training loop found."
        if not device_sequential_training
        else (
            "A repeated-sample MOS device-level training deck now keeps hidden/readout weight capacitors persistent inside "
            "one ngspice transient while guide waveforms reset only temporary activation, error, delta, and gradient caps "
            "between samples. "
            f"Sequences={device_sequential_training.get('sequences')}, samples per sequence="
            f"{device_sequential_training.get('samples_per_sequence')}; error/readout/output polarity checks passed="
            f"{device_sequential_training.get('all_error_readout_output_polarities_pass')}; hidden update polarity checks passed="
            f"{device_sequential_training.get('all_hidden_update_polarities_pass')}. "
            f"Final signed readout weights were {sci(device_sequential_training.get('high_then_low_final_readout_signed'))} V "
            f"for high-then-low and {sci(device_sequential_training.get('low_then_high_final_readout_signed'))} V "
            "for low-then-high. This is still a tiny two-sample smoke test with coarse updates, not stable MNIST training."
        )
    )
    device_multicell_note = (
        "No transistor/device-level multicell classifier found."
        if not device_multicell_classifier
        else (
            "A 2-input/2-hidden/2-output MOS device-level classifier now tests one-hot class competition with persistent "
            "hidden/readout weight capacitors in single ngspice transients. "
            f"Target output increase={device_multicell_classifier.get('all_target_outputs_increase')}, non-target output "
            f"decrease={device_multicell_classifier.get('all_non_target_outputs_decrease')}, margins improve="
            f"{device_multicell_classifier.get('all_margins_improve')}, target/non-target active readout updates pass="
            f"{device_multicell_classifier.get('all_target_active_readouts_increase')}/"
            f"{device_multicell_classifier.get('all_non_target_active_readouts_decrease')}. "
            f"Margin improvement ranged from {sci(device_multicell_classifier.get('min_margin_improvement_v'))} V to "
            f"{sci(device_multicell_classifier.get('max_margin_improvement_v'))} V. "
            f"Batching supported={device_multicell_classifier.get('batching_supported')}. "
            "This is still a synthetic two-class smoke test, not MNIST-scale training."
        )
    )
    device_feedback_alignment_note = (
        "No transistor/device-level feedback-alignment variant found."
        if not device_feedback_alignment
        else (
            "A direct-feedback-alignment MOS variant now replaces hidden-delta transport through learned readout weights "
            "with fixed signed feedback capacitor nodes. "
            f"Uses readout weight transport for hidden delta="
            f"{device_feedback_alignment.get('uses_readout_weight_transport_for_hidden_delta')}; fixed feedback caps="
            f"{device_feedback_alignment.get('fixed_feedback_caps')}. "
            f"Target output increase={device_feedback_alignment.get('all_target_outputs_increase')}, non-target output "
            f"decrease={device_feedback_alignment.get('all_non_target_outputs_decrease')}, margins improve="
            f"{device_feedback_alignment.get('all_margins_improve')}. "
            f"Mean margin improvement was {sci(device_feedback_alignment.get('mean_margin_improvement_v'))} V versus "
            f"{sci(device_feedback_alignment.get('backprop_style_mean_margin_improvement_v'))} V for the backprop-style "
            "multicell smoke. This is still a hand-sized synthetic two-class test, not MNIST-scale evidence."
        )
    )
    device_parity3_note = (
        "No transistor/device-level 3-bit parity benchmark found."
        if not device_parity3_readout
        else (
            "A complete tiny-dataset parity benchmark now trains capacitor-held readout weights over all eight 3-bit "
            "patterns and evaluates all eight patterns in the same ngspice transient. "
            f"Hidden feature weights trained={device_parity3_readout.get('hidden_feature_weights_trained')}; readout "
            f"weights trained={device_parity3_readout.get('readout_weights_trained')}; min eval accuracy="
            f"{pct(device_parity3_readout.get('min_eval_accuracy'))}; all eval patterns correct="
            f"{device_parity3_readout.get('all_eval_patterns_correct')}; all train updates improved margin="
            f"{device_parity3_readout.get('all_train_updates_improve_margin')}. "
            f"Minimum eval margin was {sci(device_parity3_readout.get('min_eval_margin_v'))} V. "
            "This validates complete tiny-dataset sequencing/readout learning, but the hidden literal features are programmed."
        )
    )
    device_xor2_note = (
        "No transistor/device-level 2-bit XOR learned-hidden-update benchmark found."
        if not device_xor2_learned_features
        else (
            "A 2-bit XOR device benchmark now includes hidden-feature weight capacitors in the update path: four literal "
            "hidden cells and a two-output readout run forward/error/backward/accumulate/apply phases in ngspice. "
            f"Hidden feature weights trained={device_xor2_learned_features.get('hidden_feature_weights_trained')}; "
            f"hidden features programmed initially={device_xor2_learned_features.get('hidden_features_programmed_initially')}; "
            f"readout weights trained={device_xor2_learned_features.get('readout_weights_trained')}; min eval accuracy="
            f"{pct(device_xor2_learned_features.get('min_eval_accuracy'))}; all eval patterns correct="
            f"{device_xor2_learned_features.get('all_eval_patterns_correct')}; all active hidden updates nonzero="
            f"{device_xor2_learned_features.get('all_active_hidden_updates_nonzero')}. "
            f"Minimum train margin improvement was {sci(device_xor2_learned_features.get('min_train_margin_improvement_v'))} V, "
            f"minimum eval margin was {sci(device_xor2_learned_features.get('min_eval_margin_v'))} V, and max active "
            f"hidden-weight movement was {sci(device_xor2_learned_features.get('max_active_hidden_weight_delta_v'))} V. "
            "This is the first nonlinear hidden-update tiny benchmark, but the literal hidden features are still programmed "
            "rather than discovered from random initialization."
        )
    )
    device_xor2_hidden_repair_note = (
        "No transistor/device-level XOR hidden-repair benchmark found."
        if not device_xor2_hidden_repair
        else (
            "A second 2-bit XOR device benchmark freezes the readout and trains only signed hidden match-weight "
            "capacitors. Each literal feature has fixed mismatch suppression plus a differential match weight, so "
            "positive hidden error improves the signed match weight by discharging the negative match capacitor. "
            f"Readout weights trained={device_xor2_hidden_repair.get('readout_weights_trained')}; hidden feature "
            f"weights trained={device_xor2_hidden_repair.get('hidden_feature_weights_trained')}; initial/final eval "
            f"accuracy={pct(device_xor2_hidden_repair.get('initial_eval_accuracy'))}/"
            f"{pct(device_xor2_hidden_repair.get('final_eval_accuracy'))}; min margin gain="
            f"{sci(device_xor2_hidden_repair.get('min_margin_gain_v'))} V; active hidden activation gain="
            f"{sci(device_xor2_hidden_repair.get('active_hidden_activation_gain_v'))} V; all total signed hidden "
            f"match weights increased={device_xor2_hidden_repair.get('all_total_hidden_match_weights_increased')}. "
            "This is stronger than a readout-only benchmark, but it still uses programmed literal-feature topology and "
            "fixed mismatch suppression rather than discovering hidden features from random weights."
        )
    )
    device_xor2_random_hidden_note = (
        "No transistor/device-level XOR general random-hidden benchmark found."
        if not device_xor2_random_hidden
        else (
            "A 2-bit XOR device benchmark now removes the programmed literal-detector hidden topology: four hidden "
            "ReLU cells are fully connected to x0/nx0/x1/nx1 rails with deterministic pseudo-random signed capacitor "
            "weights, and both readout and hidden weights update in ngspice. "
            f"Hidden topology programmed as literals={device_xor2_random_hidden.get('hidden_topology_programmed_as_literals')}; "
            f"readout/hidden weights trained={device_xor2_random_hidden.get('readout_weights_trained')}/"
            f"{device_xor2_random_hidden.get('hidden_feature_weights_trained')}; initial/final eval accuracy="
            f"{pct(device_xor2_random_hidden.get('initial_eval_accuracy'))}/"
            f"{pct(device_xor2_random_hidden.get('final_eval_accuracy'))}; final min margin="
            f"{sci(device_xor2_random_hidden.get('final_min_margin_v'))} V; all train cycles update readout/hidden="
            f"{device_xor2_random_hidden.get('all_train_cycles_update_readout')}/"
            f"{device_xor2_random_hidden.get('all_train_cycles_update_hidden')}. "
            "This is the first nonliteral hidden-layer device run, but it reaches only 75% on XOR and has coarse signed updates."
        )
    )
    device_xor2_random_hidden_wide_note = (
        "No wider transistor/device-level XOR random-hidden benchmark found."
        if not device_xor2_random_hidden_wide
        else (
            "A wider follow-up uses eight nonliteral hidden ReLU cells, a hidden bias rail, capacitor-held output-bias "
            "weights, internal parasitic caps/leaks for convergence, and an optional batch-apply schedule that keeps "
            "gradient caps across a four-pattern mini-epoch before one apply pulse. "
            f"Batch apply={device_xor2_random_hidden_wide.get('batch_apply')}; train/apply cycles="
            f"{device_xor2_random_hidden_wide.get('train_cycles')}/"
            f"{device_xor2_random_hidden_wide.get('train_apply_cycles')}; initial/final eval accuracy="
            f"{pct(device_xor2_random_hidden_wide.get('initial_eval_accuracy'))}/"
            f"{pct(device_xor2_random_hidden_wide.get('final_eval_accuracy'))}; final min margin="
            f"{sci(device_xor2_random_hidden_wide.get('final_min_margin_v'))} V; final hidden-feature separability="
            f"{device_xor2_random_hidden_wide.get('final_hidden_feature_separability', {}).get('linearly_separable')} "
            f"with diagnostic min margin "
            f"{sci(device_xor2_random_hidden_wide.get('final_hidden_feature_separability', {}).get('min_margin'))}. "
            "This shows the SPICE-measured nonliteral hidden representation can be separable and the gradient-cap "
            "batch schedule is wired, but the local analog readout/update rule still does not find the separating weights."
        )
    )
    device_xor2_readout_rule_note = (
        "No transistor/device-level XOR readout-rule follow-up found."
        if not device_xor2_readout_rule
        else (
            "Follow-up local readout-rule sweeps added score, perceptron, margin, and competitive error options. "
            "The strongest direct perceptron control shows the write path can move hidden-to-output readout weights "
            f"by {sci(device_xor2_readout_rule.get('max_abs_total_readout_weight_signed_delta_v'))} V total, "
            f"but still ends at {pct(device_xor2_readout_rule.get('final_eval_accuracy'))} eval accuracy with final min margin "
            f"{sci(device_xor2_readout_rule.get('final_min_margin_v'))} V. "
            "Score-gated margin/competitive variants stayed at leakage-scale movement, so the next blocker is a "
            "stronger mistake/low-margin comparator or local-loss error latch, not the capacitor write path."
        )
    )
    device_xor2_calibrated_readout_note = (
        "No transistor/device-level XOR calibrated-readout check found."
        if not device_xor2_separator_readout
        else (
            "A programmed-separator sanity check initializes the same eight nonliteral hidden-cell readout from a "
            "SPICE-measured hidden-activation separator, with an added differential output-bias offset. "
            f"At separator scale {device_xor2_separator_readout.get('separator_scale')} and offset "
            f"{device_xor2_separator_readout.get('separator_offset_v')} V, zero-epoch ngspice evaluation reaches "
            f"{pct(device_xor2_separator_readout.get('final_eval_accuracy'))} eval accuracy with final min margin "
            f"{sci(device_xor2_separator_readout.get('final_min_margin_v'))} V. "
            "This proves the current nonliteral hidden activations plus the transistor-level readout can represent XOR, "
            "but the readout weights were programmed rather than learned."
        )
    )
    device_xor2_separator_score_note = (
        ""
        if not device_xor2_separator_score_train
        else (
            " Starting from that calibrated separator, a three-epoch batch score-rule run preserved "
            f"{pct(device_xor2_separator_score_train.get('final_eval_accuracy'))} eval accuracy with final min margin "
            f"{sci(device_xor2_separator_score_train.get('final_min_margin_v'))} V, while the direct perceptron and "
            "lowtarget controls degraded the same calibrated start to 75%. This is preservation from a programmed "
            "solution, not discovery from random readout weights."
        )
    )
    device_xor2_lowtarget_rule_note = (
        ""
        if not device_xor2_lowtarget_rule
        else (
            " A low-score target-gated rule also moved readout weights materially "
            f"({sci(device_xor2_lowtarget_rule.get('max_abs_total_readout_weight_signed_delta_v'))} V total in the "
            "strongest run), but the generated low-score gate did not discriminate the two outputs well enough and "
            f"the run stayed at {pct(device_xor2_lowtarget_rule.get('final_eval_accuracy'))}."
        )
    )
    device_xor2_mistake_gate_note = (
        ""
        if not device_xor2_mistake_gate_random or not device_xor2_mistake_gate_separator
        else (
            " A transistor-level mistake-gated follow-up added score-lead latch nodes `lead01`/`lead10` and an "
            "image-generated architecture diagram at results/figures/device_xor2_mistake_gate_architecture.png. "
            f"The random-readout diagnostic stayed at {pct(device_xor2_mistake_gate_random.get('final_eval_accuracy'))}; "
            f"the calibrated-separator diagnostic stayed at {pct(device_xor2_mistake_gate_separator.get('final_eval_accuracy'))}, "
            "but this is not evidence of learned correction because the lead latch did not produce a usable gate voltage. "
            f"The strongest measured mean absolute lead difference was only "
            f"{sci(max(device_xor2_mistake_gate_random.get('mean_abs_train_lead_diff_v') or 0.0, device_xor2_mistake_gate_separator.get('mean_abs_train_lead_diff_v') or 0.0))} V."
        )
    )
    device_xor2_local_loss_note = (
        ""
        if not device_xor2_local_loss_random or not device_xor2_local_loss_separator
        else (
            " A comparator-free local-loss follow-up then avoided the failed lead latch: the target output is boosted "
            "when its low-score node is high, and the non-target output is depressed when its own score node is high. "
            f"From random readout, three batch epochs moved hidden-to-output readout weights by "
            f"{sci(device_xor2_local_loss_random.get('max_abs_total_readout_weight_signed_delta_v'))} V total but still "
            f"ended at {pct(device_xor2_local_loss_random.get('final_eval_accuracy'))} with final min margin "
            f"{sci(device_xor2_local_loss_random.get('final_min_margin_v'))} V. From the calibrated separator, one "
            f"batch epoch preserved {pct(device_xor2_local_loss_separator.get('final_eval_accuracy'))} with final min "
            f"margin {sci(device_xor2_local_loss_separator.get('final_min_margin_v'))} V. This confirms usable write "
            "strength without solving readout discovery from random weights."
            + (
                ""
                if not device_xor2_local_loss_strong
                else (
                    f" Increasing readout write strength to 10x moved readout weights by "
                    f"{sci(device_xor2_local_loss_strong.get('max_abs_total_readout_weight_signed_delta_v'))} V total "
                    f"but still ended at {pct(device_xor2_local_loss_strong.get('final_eval_accuracy'))}, so the blocker "
                    "is update direction/error construction rather than raw write amplitude."
                )
            )
        )
    )
    device_xor2_output_senseamp_note = (
        ""
        if not device_xor2_output_senseamp_batch or not device_xor2_output_senseamp_online
        else (
            " A stronger transistor-level output-senseamp mistake gate now samples the stored output activations, "
            "precharges `lead01/lead10`, and discharges the losing side during a separate compare pulse before the "
            "error/update pulse. "
            f"The batch diagnostic tracked the score winner with mean absolute lead separation "
            f"{sci(device_xor2_output_senseamp_batch.get('mean_abs_train_lead_diff_v'))} V but stayed at "
            f"{pct(device_xor2_output_senseamp_batch.get('final_eval_accuracy'))}. "
            f"Switching to online per-sample applies with stronger readout writes raised the dense nonliteral XOR circuit "
            f"from {pct(device_xor2_output_senseamp_online.get('initial_eval_accuracy'))} to "
            f"{pct(device_xor2_output_senseamp_online.get('final_eval_accuracy'))}, with "
            f"{sci(device_xor2_output_senseamp_online.get('max_abs_total_readout_weight_signed_delta_v'))} V total "
            "readout movement. It still does not solve XOR, but it is the first random-readout improvement from the "
            "new hardware mistake gate."
            + (
                ""
                if not device_xor2_output_senseamp_best
                else (
                    f" A later 16-epoch online run with a stronger readout write solved the same dense nonliteral XOR "
                    f"benchmark from random readout at {pct(device_xor2_output_senseamp_best.get('final_eval_accuracy'))}, "
                    f"with final minimum margin {sci(device_xor2_output_senseamp_best.get('final_min_margin_v'))} V and "
                    f"{sci(device_xor2_output_senseamp_best.get('max_abs_total_readout_weight_signed_delta_v'))} V total "
                    "readout movement; hidden writes were still effectively near-frozen."
                )
            )
        )
    )
    device_xor2_backprop_synapse_note = (
        ""
        if not device_xor2_backprop_score or not device_xor2_backprop_grad_probe
        else (
            " The dense XOR generator now has selectable synapse/backward designs: `--hidden-error-rule backprop` gates "
            "hidden-delta transistor stacks with the same capacitor-held readout weight nodes used by the forward readout, "
            "while `--hidden-error-rule dfa` swaps in fixed feedback capacitor nodes. "
            f"With continuous score-error and real readout-weight-transport backprop, an eight-epoch run reached "
            f"{pct(device_xor2_backprop_score.get('final_eval_accuracy'))} final XOR accuracy and moved readout weights by "
            f"{sci(device_xor2_backprop_score.get('max_abs_total_readout_weight_signed_delta_v'))} V. "
            f"A 64x hidden-delta/hidden-gradient probe measured only "
            f"{sci(device_xor2_backprop_grad_probe.get('max_train_hidden_grad_signal_v'))} V on hidden gradient caps, "
            "so the direct hidden-gradient storage/apply cell was the bottleneck. "
            + (
                ""
                if not device_xor2_backprop_hgsense
                else (
                    "A follow-up selectable hidden-gradient sense/write cell (`--hidden-apply-mode grad_senseamp`) "
                    f"with a smaller hidden update solved the dense XOR device benchmark at "
                    f"{pct(device_xor2_backprop_hgsense.get('final_eval_accuracy'))} final accuracy, final minimum margin "
                    f"{sci(device_xor2_backprop_hgsense.get('final_min_margin_v'))} V, and "
                    f"{sci(device_xor2_backprop_hgsense.get('max_abs_total_hidden_signed_delta_v'))} V total hidden-weight movement. "
                    + (
                        ""
                        if not device_xor2_backprop_gate
                        else (
                            "After refactoring the generator to support multiple datasets, the same current code path "
                            f"re-ran `--dataset xor2` and again reached "
                            f"{pct(device_xor2_backprop_gate.get('final_eval_accuracy'))} final accuracy with final minimum margin "
                            f"{sci(device_xor2_backprop_gate.get('final_min_margin_v'))} V, so XOR now clears the 95% gate "
                            "before further moons or MNIST scaling. "
                        )
                    )
                    + "This is a real all-layer device-level backprop result on XOR, but it is still a tiny benchmark and the "
                    "sense/write cell needs better differential/common-mode control before scaling."
                    + (
                        ""
                        if not device_moons8_eval or not device_moons8_score_strong
                        else (
                            " The same generator now supports a deterministic continuous `moons8` dataset on the "
                            "`x0/nx0/x1/nx1` rails. Zero-training moons eval verified the nonliteral hidden representation "
                            f"is separable, while a two-epoch stronger score-error run stayed at "
                            f"{pct(device_moons8_score_strong.get('final_eval_accuracy'))} final accuracy but improved the "
                            f"worst final margin to {sci(device_moons8_score_strong.get('final_min_margin_v'))} V with "
                            f"{sci(device_moons8_score_strong.get('max_abs_total_readout_weight_signed_delta_v'))} V "
                            "readout movement. This moves the transistor-level backprop harness beyond XOR, but the "
                            "continuous-sample readout error rule is not yet strong enough."
                        )
                    )
                    + (
                        ""
                        if not device_mnist01_8_eval or not device_mnist01_8_score_gentle
                        else (
                            " A tiny binary MNIST bridge (`mnist01_8`) now selects local MNIST zeros/ones, "
                            "downsamples each image to 2x2, and drives the same four input rails. Zero-training eval "
                            "shows the measured hidden activations are linearly separable, but the random readout starts "
                            f"at {pct(device_mnist01_8_eval.get('final_eval_accuracy'))}. A three-epoch gentle score run "
                            "with real hidden backprop through readout synapse caps moved readout weights by "
                            f"{sci(device_mnist01_8_score_gentle.get('max_abs_total_readout_weight_signed_delta_v'))} V "
                            "and hidden weights by "
                            f"{sci(device_mnist01_8_score_gentle.get('max_abs_total_hidden_signed_delta_v'))} V, "
                            f"but stayed at {pct(device_mnist01_8_score_gentle.get('final_eval_accuracy'))} with final "
                            f"minimum margin {sci(device_mnist01_8_score_gentle.get('final_min_margin_v'))} V. "
                            "This makes the continuous readout/error update rule the current blocker before broader "
                            "8x8 MNIST scaling."
                        )
                    )
                    + (
                        ""
                        if not device_mnist01_8_outcomp_bias
                        else (
                            " A new output-competitive error rule uses the stored output activation caps as local "
                            "analog gates instead of raw score residuals or the saturated lead latch. On `mnist01_8`, "
                            "the bias-strengthened three-epoch run kept lead tracking correct and improved the worst "
                            f"final margin to {sci(device_mnist01_8_outcomp_bias.get('final_min_margin_v'))} V, but "
                            f"still ended at {pct(device_mnist01_8_outcomp_bias.get('final_eval_accuracy'))}. "
                        )
                    )
                    + (
                        ""
                        if not device_xor2_flow_smoke
                        else (
                            " The generator now also exposes direct backward/write flow mode with no gradient "
                            "accumulator caps in the weight update path and no separate apply phase. The one-epoch "
                            f"XOR flow smoke completed with {sci(device_xor2_flow_smoke.get('max_abs_total_readout_weight_signed_delta_v'))} V "
                            "readout movement and "
                            f"{sci(device_xor2_flow_smoke.get('max_abs_total_hidden_signed_delta_v'))} V hidden-weight movement, "
                            "but the first tiny MNIST01 flow smoke timed out, so this simpler architecture is wired "
                            "but not yet the stable scaling path."
                        )
                    )
                    + (
                        ""
                        if not device_xor2_flow_prestore_gate
                        else (
                            " A per-synapse pre-activation trace variant now captures local source activity through MOS store paths during "
                            "the forward phase and uses those trace capacitors during the direct backward/write "
                            f"window. The non-destructive trace-gate XOR check moved readout weights by "
                            f"{sci(device_xor2_flow_prestore_gate.get('max_abs_total_readout_weight_signed_delta_v'))} V "
                            f"and ended at {pct(device_xor2_flow_prestore_gate.get('final_eval_accuracy'))}; "
                            "the destructive trace-consume variant reduced readout movement to "
                            f"{sci(device_xor2_flow_prestore_consume.get('max_abs_total_readout_weight_signed_delta_v') if device_xor2_flow_prestore_consume else None)} V, "
                            "so the consume path needs timing/capacitance tuning."
                        )
                    )
                    + (
                        ""
                        if not device_xor2_flow_readout_only_solved
                        else (
                            " Longer readout-only direct-flow runs now solve XOR without gradient accumulator "
                            "caps or a separate apply pulse: the shared-source run reached "
                            f"{pct(device_xor2_flow_readout_only_solved.get('final_eval_accuracy'))} with "
                            f"{sci(device_xor2_flow_readout_only_solved.get('final_min_margin_v'))} V minimum margin, "
                            "and the per-synapse trace-gate run reached "
                            f"{pct(device_xor2_flow_prestore_readout_only_solved.get('final_eval_accuracy') if device_xor2_flow_prestore_readout_only_solved else None)}. "
                            "These are not end-to-end hidden backprop results because hidden writes were disabled."
                        )
                    )
                    + (
                        ""
                        if not device_xor2_flow_alllayer_solved
                        else (
                            " A later light-measurement all-layer direct-flow run enabled hidden writes and "
                            "readout-weight-transport hidden deltas, still without gradient accumulator caps or "
                            "a separate apply pulse. It reached "
                            f"{pct(device_xor2_flow_alllayer_solved.get('final_eval_accuracy'))} XOR accuracy with "
                            f"{sci(device_xor2_flow_alllayer_solved.get('final_min_margin_v'))} V minimum margin and "
                            f"{sci(device_xor2_flow_alllayer_solved.get('max_abs_total_hidden_signed_delta_v'))} V "
                            "total hidden signed-weight movement, making direct-flow hidden backprop electrically "
                            "active on XOR. The next scaling blocker is tiny MNIST01, not XOR."
                        )
                    )
                    + (
                        ""
                        if not device_mnist01_8_flow_readout_only_best
                        else (
                            " The best tiny MNIST01 readout-only direct-flow run so far reached "
                            f"{pct(device_mnist01_8_flow_readout_only_best.get('final_eval_accuracy'))} after "
                            f"{device_mnist01_8_flow_readout_only_best.get('epochs')} epochs, with final minimum "
                            f"margin {sci(device_mnist01_8_flow_readout_only_best.get('final_min_margin_v'))} V. "
                            "That improves the first stronger readout-only transfer but remains below a solved "
                            "binary MNIST bridge."
                        )
                    )
                )
            )
        )
    )
    settling_pareto_note = (
        "No settling time/accuracy Pareto summary found."
        if not settling_pareto
        else (
            f"Analytical settling frontier validated against SPICE: {settling_pareto.get('test_samples')} held-out samples, "
            f"best fixed-checkpoint transient accuracy {pct(settling_pareto.get('best_accuracy'))} at "
            f"{settling_pareto.get('best_readout_time_ns')} ns versus exact steady-state "
            f"{pct(settling_pareto.get('steady_state_accuracy'))}; not a training run."
        )
    )
    local_feature_settling_note = (
        "No local-feature settling time/accuracy Pareto summaries found."
        if not local_feature_settling_b5 or not local_feature_settling_b4
        else (
            f"Local-feature settling frontiers for the small candidates show finite readout is a useful but modest meta-parameter: "
            f"the 1,372-state 10x10 b5 stride2 c2 branch peaks at {pct(local_feature_settling_b5.get('best_accuracy'))} "
            f"at {local_feature_settling_b5.get('best_readout_time_ns')} ns versus "
            f"{pct(local_feature_settling_b5.get('steady_state_accuracy'))} steady-state, and stays within one point at "
            f"{local_feature_settling_b5.get('fastest_within_1pct_of_best_readout_time_ns')} ns; "
            f"the 1,832-state 10x10 b4 stride2 c2 branch peaks at {pct(local_feature_settling_b4.get('best_accuracy'))} "
            f"at {local_feature_settling_b4.get('best_readout_time_ns')} ns versus "
            f"{pct(local_feature_settling_b4.get('steady_state_accuracy'))} steady-state, and stays within one point at "
            f"{local_feature_settling_b4.get('fastest_within_1pct_of_best_readout_time_ns')} ns. "
            "These are fixed-checkpoint analytical timing sweeps, not training runs."
        )
    )
    bias_cal_note = (
        "No output-bias capacitor calibration summary found."
        if not bias_cal_subset
        else (
            f"Output-bias capacitor calibration on the 2,000-train split improved the 1,000-held-out subset "
            f"from {pct(bias_cal_subset.get('base_test_accuracy'))} to "
            f"{pct(bias_cal_subset.get('calibrated_test_accuracy'))}; "
            f"ngspice eval verified {pct(bias_cal_spice_eval.get('heldout_test_accuracy') if bias_cal_spice_eval else None)}. "
            f"Full 60k/10k analytical checks reached "
            f"{pct(bias_cal_full_14.get('calibrated_test_accuracy') if bias_cal_full_14 else None)} at 14x14 and "
            f"{pct(bias_cal_full_28.get('calibrated_test_accuracy') if bias_cal_full_28 else None)} at 28x28."
        )
    )
    local_feature_note = (
        "No 32-channel local-feature topology result found."
        if not local_feature_torch
        else (
            f"Richer local-feature topology: 32 tanh feature cells per block trained in PyTorch reached "
            f"{pct(local_feature_torch.get('best_test_accuracy'))} on full 60k/10k MNIST and exports a SPICE checkpoint; "
            f"ngspice eval of that checkpoint reached "
            f"{pct(local_feature_spice_eval.get('heldout_test_accuracy') if local_feature_spice_eval else None)} "
            f"on a 1,000-image held-out slice. A tiny ngspice update smoke "
            f"{'completed' if local_feature_spice_update else 'is missing'}; "
            f"a one-sample phase-transient capacitor update from the same c32 checkpoint "
            f"{'matched the OP update with max state diff ' + sci(local_feature_phase.get('state_max_abs_diff')) if local_feature_phase else 'is missing'}. "
            f"A c8 small-rule phase check accumulated "
            f"{local_feature_batch_phase.get('batch_size') if local_feature_batch_phase else 'missing'} samples in gradient capacitors "
            f"before one apply pulse and matched the OP batch update with max state diff "
            f"{sci(local_feature_batch_phase.get('state_max_abs_diff')) if local_feature_batch_phase else 'missing'}. "
            f"A follow-up c8 phase check ran "
            f"{local_feature_multibatch_phase.get('updates') if local_feature_multibatch_phase else 'missing'} such batch updates "
            f"in one transient while retaining updated weight capacitors between cycles, with max state diff "
            f"{sci(local_feature_multibatch_phase.get('state_max_abs_diff')) if local_feature_multibatch_phase else 'missing'}. "
            f"Shorter c8 timing checks kept the same two-batch update sequence: 2 ns phases took "
            f"{sci(local_feature_phase_2ns.get('phase_wall_time_s')) if local_feature_phase_2ns else 'missing'} s with RMS state diff "
            f"{sci(local_feature_phase_2ns.get('state_rms_diff')) if local_feature_phase_2ns else 'missing'}, and 1 ns phases took "
            f"{sci(local_feature_phase_1ns.get('phase_wall_time_s')) if local_feature_phase_1ns else 'missing'} s with RMS state diff "
            f"{sci(local_feature_phase_1ns.get('state_rms_diff')) if local_feature_phase_1ns else 'missing'}. "
            f"A repeated phase-training harness ran a c8 1 ns phase chunk over "
            f"{local_feature_phase_train.get('phase_train_samples') if local_feature_phase_train else 'missing'} samples "
            f"and SPICE-evaluated "
            f"{local_feature_phase_train.get('test_samples') if local_feature_phase_train else 'missing'} held-out images, ending at "
            f"{pct(local_feature_phase_train.get('final_heldout_accuracy')) if local_feature_phase_train else 'missing'}; "
            "this is a harness smoke, not a benchmark. "
            f"Full training has not moved into transient SPICE."
        )
    )
    small_rule_note = (
        "No small-network local-feature rule sweep found."
        if not local_feature_small_sweep
        else (
            f"Fast small-network sweep: phase-portable mse_tanh + plain SGD reached "
            f"{pct(local_feature_small_sweep.get('best_phase_spice_portable', {}).get('test_accuracy'))} "
            f"on {local_feature_small_sweep.get('train_samples')} train / "
            f"{local_feature_small_sweep.get('test_samples')} held-out with "
            f"{local_feature_small_sweep.get('best_phase_spice_portable', {}).get('channels')} channels per block; "
            "this is a PyTorch experiment to choose smaller SPICE targets, not an all-SPICE result."
        )
    )
    noise_sweep_note = (
        "No local-feature update-noise surrogate sweep found."
        if not local_feature_noise_sweep
        else (
            f"Update-noise surrogate for the same manual phase-portable rule reached "
            f"{pct(local_feature_noise_sweep.get('best_overall', {}).get('test_accuracy'))} "
            f"on {local_feature_noise_sweep.get('train_samples')} train / "
            f"{local_feature_noise_sweep.get('test_samples')} held-out with "
            f"{local_feature_noise_sweep.get('best_overall', {}).get('channels')} channels per block, "
            f"lr_spice={local_feature_noise_sweep.get('best_overall', {}).get('lr_spice')}, and "
            f"absolute per-update noise std "
            f"{local_feature_noise_sweep.get('best_overall', {}).get('update_noise_std')}; "
            "this is a fast robustness surrogate for prioritizing small SPICE checks, not all-SPICE training."
        )
    )
    frontier_note = (
        "No small-network local-feature frontier sweep found."
        if not local_feature_small_frontier
        else (
            f"Small-network frontier sweep: the same manual phase-portable rule reached "
            f"{pct(local_feature_small_frontier.get('best_overall', {}).get('test_accuracy'))} "
            f"on {local_feature_small_frontier.get('train_samples_list')} train / "
            f"{local_feature_small_frontier.get('test_samples')} held-out with "
            f"image_size={local_feature_small_frontier.get('best_overall', {}).get('image_size')}, "
            f"channels={local_feature_small_frontier.get('best_overall', {}).get('channels')}, "
            f"lr_spice={local_feature_small_frontier.get('best_overall', {}).get('lr_spice')}, and "
            f"{local_feature_small_frontier.get('best_overall', {}).get('phase_state_values')} phase-state values; "
            "this is the non-overlap experiment-first scale-up ladder before the overlapping-block refinement. "
            + (
                "No overlapping-block frontier sweep found."
                if not local_feature_overlap_frontier
                else (
                    f"An overlapping-block sweep found a same-state replacement for 14x14/c8: "
                    f"image_size={local_feature_overlap_frontier.get('best_overall', {}).get('image_size')}, "
                    f"block_size={local_feature_overlap_frontier.get('best_overall', {}).get('block_size')}, "
                    f"stride={local_feature_overlap_frontier.get('best_overall', {}).get('stride')}, "
                    f"channels={local_feature_overlap_frontier.get('best_overall', {}).get('channels')} reached "
                    f"{pct(local_feature_overlap_frontier.get('best_overall', {}).get('test_accuracy'))} "
                    f"with {local_feature_overlap_frontier.get('best_overall', {}).get('phase_state_values')} "
                    "phase-state values."
                )
            )
            + " "
            + (
                "No overlapping-block refinement sweep found."
                if not local_feature_overlap_refine
                else (
                    f"A 1k/2k refinement found the smallest >90% overlapping point at "
                    f"image_size={local_feature_overlap_refine.get('smallest_target_hit', {}).get('image_size')}, "
                    f"stride={local_feature_overlap_refine.get('smallest_target_hit', {}).get('stride')}, "
                    f"channels={local_feature_overlap_refine.get('smallest_target_hit', {}).get('channels')}, "
                    f"train_samples={local_feature_overlap_refine.get('smallest_target_hit', {}).get('train_samples')}, "
                    f"accuracy={pct(local_feature_overlap_refine.get('smallest_target_hit', {}).get('test_accuracy'))}, "
                    f"and {local_feature_overlap_refine.get('smallest_target_hit', {}).get('phase_state_values')} "
                    "phase-state values."
                )
            )
            + " "
            + (
                "No overlapping-block update-noise sweep found."
                if not local_feature_overlap_noise
                else (
                    f"The 12x12 overlap noise surrogate found a smallest >90% point at "
                    f"channels={local_feature_overlap_noise.get('smallest_target_hit', {}).get('channels')}, "
                    f"stride={local_feature_overlap_noise.get('smallest_target_hit', {}).get('stride')}, "
                    f"lr_spice={local_feature_overlap_noise.get('smallest_target_hit', {}).get('lr_spice')}, "
                    f"noise_std={local_feature_overlap_noise.get('smallest_target_hit', {}).get('update_noise_std')}, "
                    f"accuracy={pct(local_feature_overlap_noise.get('smallest_target_hit', {}).get('test_accuracy'))}, "
                    f"and {local_feature_overlap_noise.get('smallest_target_hit', {}).get('phase_state_values')} "
                    "phase-state values, so the smaller 12x12 overlap candidate should be prioritized before longer "
                    "14x14/c8 phase-training."
                )
            )
            + " "
            + (
                "No low-learning-rate overlap sweep found."
                if not local_feature_overlap_lowlr
                else (
                    "A lower-lr overlap sweep found lr_spice=0.15 still clears the fast target at "
                    "90.6% "
                    "on the 2k/1k surrogate, while lr_spice=0.1 falls just below 90%."
                )
            )
            + " "
            + (
                "No batch-size-matched overlap sweep found."
                if not local_feature_overlap_batch2
                else (
                    f"A batch-size-matched overlap sweep then found the same 12x12/c2 topology reaches "
                    f"{pct(local_feature_overlap_batch2.get('best_overall', {}).get('test_accuracy'))} "
                    f"with batch size 2 at lr_spice="
                    f"{local_feature_overlap_batch2.get('best_overall', {}).get('lr_spice')}, matching the "
                    "transient deck's current update granularity."
                )
            )
            + " "
            + (
                "No renewed small-network batch-2 screen found."
                if not local_feature_smallnets_screen
                else (
                    f"A renewed batch-size-matched small-network screen found a smaller 1k/1k target: "
                    f"image_size={local_feature_smallnets_screen.get('smallest_target_hit', {}).get('image_size')}, "
                    f"block_size={local_feature_smallnets_screen.get('smallest_target_hit', {}).get('block_size')}, "
                    f"stride={local_feature_smallnets_screen.get('smallest_target_hit', {}).get('stride')}, "
                    f"channels={local_feature_smallnets_screen.get('smallest_target_hit', {}).get('channels')}, "
                    f"accuracy={pct(local_feature_smallnets_screen.get('smallest_target_hit', {}).get('test_accuracy'))}, "
                    f"and {local_feature_smallnets_screen.get('smallest_target_hit', {}).get('phase_state_values')} "
                    "phase-state values."
                )
            )
            + " "
            + (
                "No promoted small-network batch-2 sweep found."
                if not local_feature_smallnets_promote
                else (
                    f"Promoting the small-network Pareto candidates to 2k/1k found a much smaller fast target: "
                    f"image_size={local_feature_smallnets_promote.get('smallest_target_hit', {}).get('image_size')}, "
                    f"block_size={local_feature_smallnets_promote.get('smallest_target_hit', {}).get('block_size')}, "
                    f"stride={local_feature_smallnets_promote.get('smallest_target_hit', {}).get('stride')}, "
                    f"channels={local_feature_smallnets_promote.get('smallest_target_hit', {}).get('channels')} reached "
                    f"{pct(local_feature_smallnets_promote.get('smallest_target_hit', {}).get('test_accuracy'))} "
                    f"with {local_feature_smallnets_promote.get('smallest_target_hit', {}).get('phase_state_values')} "
                    f"phase-state values; the best promoted small network reached "
                    f"{pct(local_feature_smallnets_promote.get('best_overall', {}).get('test_accuracy'))} "
                    f"with {local_feature_smallnets_promote.get('best_overall', {}).get('phase_state_values')} values. "
                    "These promoted sweep results are fast-surrogate results; the following continuation noise and bounded "
                    "SPICE checks choose which small candidate to validate."
                )
            )
            + " "
            + (
                "No 10x10 small-network continuation stress screen found."
                if not local_feature_smallnet10_b4_fast_lr008
                else (
                    f"Fast continuation over {local_feature_smallnet10_b4_fast_lr008.get('end_sample')} sequential samples "
                    "selected the 10x10 4x4 stride-2/c2 candidate over the smaller 10x10 5x5 branch: "
                    f"the 4x4 lr_spice=0.08 branch ended at "
                    f"{pct(local_feature_smallnet10_b4_fast_lr008.get('final_eval', {}).get('test_accuracy'))}, "
                    + (
                        "while no matching 5x5 continuation was found."
                        if not local_feature_smallnet10_b5_fast_lr003
                        else (
                            f"while the 5x5 lr_spice=0.03 branch ended at "
                            f"{pct(local_feature_smallnet10_b5_fast_lr003.get('final_eval', {}).get('test_accuracy'))}."
                        )
                    )
                )
            )
            + " "
            + (
                "No 10x10 small-network noise stress screen found."
                if not local_feature_smallnet10_b4_noise
                else (
                    "The targeted update-noise screen kept 10x10 4x4 stride-2/c2 above 91% for every tested noise std "
                    "through 0.003, while the 10x10 5x5 stride-2/c2 branch had much less margin and dropped below 90% "
                    "at higher-noise settings."
                )
            )
            + " "
            + (
                "No scaled 10x10 b5 high-lr experiment found."
                if not local_feature_smallnet10_b5_scale_10k2k_hilr
                else (
                    f"Revisiting the smaller 1,372-state 10x10 b5 stride2 c2 branch with longer high-lr fast training reached "
                    f"{pct(local_feature_smallnet10_b5_scale_5k2k_hilr.get('best_overall', {}).get('test_accuracy') if local_feature_smallnet10_b5_scale_5k2k_hilr else None)} "
                    f"on 5,000 train / 2,000 held-out and "
                    f"{pct(local_feature_smallnet10_b5_scale_10k2k_hilr.get('best_overall', {}).get('test_accuracy'))} "
                    f"on 10,000 train / 2,000 held-out. Continuing the 10k checkpoint for 212 fast-reference samples at "
                    f"lr_spice=0.03 ended at "
                    f"{pct(local_feature_smallnet10_b5_scale_10k2k_fast_lr003.get('final_eval', {}).get('test_accuracy') if local_feature_smallnet10_b5_scale_10k2k_fast_lr003 else None)}. "
                    f"A longer fast gate to 1,024 total samples ended at "
                    f"{pct(local_feature_smallnet10_b5_scale_10k2k_fast_lr003_resume1024.get('final_eval', {}).get('test_accuracy') if local_feature_smallnet10_b5_scale_10k2k_fast_lr003_resume1024 else None)}. "
                    f"A 12-sample real 1 ns SPICE phase-training gate ended at "
                    f"{pct(local_feature_smallnet10_b5_phase_scale_10k2k_lr003.get('final_heldout_accuracy') if local_feature_smallnet10_b5_phase_scale_10k2k_lr003 else None)} "
                    f"with RMS drift "
                    f"{sci(local_feature_smallnet10_b5_phase_scale_10k2k_lr003.get('final_fast_reference_state_rms_diff') if local_feature_smallnet10_b5_phase_scale_10k2k_lr003 else None)}. "
                    f"Resuming the same SPICE capacitor trajectory to 60 total samples ended at "
                    f"{pct(local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume60.get('final_heldout_accuracy') if local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume60 else None)} "
                    f"with RMS drift "
                    f"{sci(local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume60.get('final_fast_reference_state_rms_diff') if local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume60 else None)}. "
                    f"Fast continuation from the actual 60-sample SPICE capacitor state predicted "
                    f"{pct(local_feature_smallnet10_b5_scale_10k2k_fast_from_spice60_lr003.get('final_eval', {}).get('test_accuracy') if local_feature_smallnet10_b5_scale_10k2k_fast_from_spice60_lr003 else None)} "
                    f"at 212 total samples. Resuming the real SPICE trajectory to 212 total samples ended at "
                    f"{pct(local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume212.get('final_heldout_accuracy') if local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume212 else None)} "
                    f"with RMS drift "
                    f"{sci(local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume212.get('final_fast_reference_state_rms_diff') if local_feature_smallnet10_b5_phase_scale_10k2k_lr003_resume212 else None)}."
                )
            )
            + " "
            + (
                "No below-b5 tiny fast Pareto revisit found."
                if not local_feature_tiny_below_b5_promote
                else (
                    "A smaller-than-b5 fast revisit found new candidates before spending more SPICE time: "
                    "a 936-state 10x10 b4 stride2 c1 branch reached 91.6% on 10,000 train / 2,000 held-out, "
                    "and a 1,048-state 9x9 b4 stride2 c2 branch reached 92.4%. "
                    f"The 936-state branch stayed at "
                    f"{pct(local_feature_tiny_below_b5_c1_fast1024.get('final_eval', {}).get('test_accuracy') if local_feature_tiny_below_b5_c1_fast1024 else None)} "
                    f"after 1,024 fast continuation samples; the 1,048-state branch stayed at "
                    f"{pct(local_feature_tiny_below_b5_c2_fast1024.get('final_eval', {}).get('test_accuracy') if local_feature_tiny_below_b5_c2_fast1024 else None)}. "
                    f"A 12-sample real 1 ns SPICE gate on the 936-state branch ended at "
                    f"{pct(local_feature_tiny_below_b5_c1_phase12.get('final_heldout_accuracy') if local_feature_tiny_below_b5_c1_phase12 else None)} "
                    f"with fast/reference accuracies matched and RMS drift "
                    f"{sci(local_feature_tiny_below_b5_c1_phase12.get('final_fast_reference_state_rms_diff') if local_feature_tiny_below_b5_c1_phase12 else None)}. "
                    f"A 12-sample real 1 ns SPICE gate on the 1,048-state branch ended at "
                    f"{pct(local_feature_tiny_below_b5_c2_phase12.get('final_heldout_accuracy') if local_feature_tiny_below_b5_c2_phase12 else None)} "
                    f"with RMS drift "
                    f"{sci(local_feature_tiny_below_b5_c2_phase12.get('final_fast_reference_state_rms_diff') if local_feature_tiny_below_b5_c2_phase12 else None)}. "
                    f"Fast continuation from the actual 12-sample SPICE capacitor state predicted "
                    f"{pct(local_feature_tiny_below_b5_c2_fast_from_spice12.get('final_eval', {}).get('test_accuracy') if local_feature_tiny_below_b5_c2_fast_from_spice12 else None)} "
                    "at 60 total samples, and the real SPICE resume matched that at "
                    f"{pct(local_feature_tiny_below_b5_c2_phase60.get('final_heldout_accuracy') if local_feature_tiny_below_b5_c2_phase60 else None)} "
                    f"with RMS drift {sci(local_feature_tiny_below_b5_c2_phase60.get('final_fast_reference_state_rms_diff') if local_feature_tiny_below_b5_c2_phase60 else None)}. "
                    "A follow-up fast sub-936 pass found a new 904-state 10x10 b6 stride2 c1 branch: it reached "
                    f"{pct(local_feature_sub936_promote_5k.get('smallest_target_hit', {}).get('test_accuracy') if local_feature_sub936_promote_5k else None)} "
                    "on 5,000 train / 2,000 held-out, while the lower-state branches stayed below target. "
                    "Promoting just that 904-state branch to 10,000 train / 2,000 held-out reached "
                    f"{pct(local_feature_i10_b6_s2_c1_promote_10k.get('best_overall', {}).get('test_accuracy') if local_feature_i10_b6_s2_c1_promote_10k else None)} "
                    "and a 1,024-sample fast continuation gate ended at "
                    f"{pct(local_feature_i10_b6_s2_c1_fast1024.get('final_eval', {}).get('test_accuracy') if local_feature_i10_b6_s2_c1_fast1024 else None)}. "
                    "A 12-sample real 1 ns SPICE gate on the 904-state branch ended at "
                    f"{pct(local_feature_i10_b6_s2_c1_phase12.get('final_heldout_accuracy') if local_feature_i10_b6_s2_c1_phase12 else None)} "
                    f"with RMS drift {sci(local_feature_i10_b6_s2_c1_phase12.get('final_fast_reference_state_rms_diff') if local_feature_i10_b6_s2_c1_phase12 else None)}, "
                    "and fast continuation from that actual SPICE capacitor state predicted "
                    f"{pct(local_feature_i10_b6_s2_c1_fast_from_spice12.get('final_eval', {}).get('test_accuracy') if local_feature_i10_b6_s2_c1_fast_from_spice12 else None)} "
                    "at 212 total samples. The exact fast prediction for the 12-to-60 window was "
                    f"{pct(local_feature_i10_b6_s2_c1_fast_from_spice12_to60.get('final_eval', {}).get('test_accuracy') if local_feature_i10_b6_s2_c1_fast_from_spice12_to60 else None)}, "
                    "and the real SPICE resume matched it at "
                    f"{pct(local_feature_i10_b6_s2_c1_phase60.get('final_heldout_accuracy') if local_feature_i10_b6_s2_c1_phase60 else None)} "
                    f"with RMS drift {sci(local_feature_i10_b6_s2_c1_phase60.get('final_fast_reference_state_rms_diff') if local_feature_i10_b6_s2_c1_phase60 else None)}. "
                    "Shifting back to smaller fast experiments changed the frontier again: a broad sub-904 2,000/2,000 pilot found no >90% hit, "
                    "but a focused 5,000/2,000 promotion found a 796-state 7x7 b3 stride2 c2 branch above target. "
                    "Promoting that same 796-state branch to 10,000 train / 2,000 held-out reached "
                    f"{pct(local_feature_i7_b3_s2_c2_promote_10k.get('best_overall', {}).get('test_accuracy') if local_feature_i7_b3_s2_c2_promote_10k else None)} "
                    "and its 1,024-sample fast continuation ended at "
                    f"{pct(local_feature_i7_b3_s2_c2_fast1024.get('final_eval', {}).get('test_accuracy') if local_feature_i7_b3_s2_c2_fast1024 else None)}. "
                    "A 12-sample real 1 ns SPICE gate on this 796-state branch ended at "
                    f"{pct(local_feature_i7_b3_s2_c2_phase12.get('final_heldout_accuracy') if local_feature_i7_b3_s2_c2_phase12 else None)} "
                    f"with RMS drift {sci(local_feature_i7_b3_s2_c2_phase12.get('final_fast_reference_state_rms_diff') if local_feature_i7_b3_s2_c2_phase12 else None)}, "
                    "making it the smallest short-SPICE-validated above-90 local-feature path so far. "
                    "Fast continuation from the actual 12-sample 796-state SPICE capacitor state predicted "
                    f"{pct(local_feature_i7_b3_s2_c2_fast_from_spice12.get('final_eval', {}).get('test_accuracy') if local_feature_i7_b3_s2_c2_fast_from_spice12 else None)} "
                    "at 212 total samples and "
                    f"{pct(local_feature_i7_b3_s2_c2_fast_from_spice12_1024.get('final_eval', {}).get('test_accuracy') if local_feature_i7_b3_s2_c2_fast_from_spice12_1024 else None)} "
                    "at 1,024 total samples. The exact fast prediction for the 12-to-60 window was "
                    f"{pct(local_feature_i7_b3_s2_c2_fast_from_spice12_to60.get('final_eval', {}).get('test_accuracy') if local_feature_i7_b3_s2_c2_fast_from_spice12_to60 else None)}, "
                    "and the real SPICE resume ended at "
                    f"{pct(local_feature_i7_b3_s2_c2_phase60.get('final_heldout_accuracy') if local_feature_i7_b3_s2_c2_phase60 else None)} "
                    f"with RMS drift {sci(local_feature_i7_b3_s2_c2_phase60.get('final_fast_reference_state_rms_diff') if local_feature_i7_b3_s2_c2_phase60 else None)}. "
                    "Continuing the same private/local SPICE capacitor trajectory to 212 total samples ended at "
                    f"{pct(local_feature_i7_b3_s2_c2_phase212.get('final_heldout_accuracy') if local_feature_i7_b3_s2_c2_phase212 else None)} "
                    f"with fast reference accuracy {pct(local_feature_i7_b3_s2_c2_phase212.get('final_fast_reference_accuracy') if local_feature_i7_b3_s2_c2_phase212 else None)} "
                    f"and RMS drift {sci(local_feature_i7_b3_s2_c2_phase212.get('final_fast_reference_state_rms_diff') if local_feature_i7_b3_s2_c2_phase212 else None)}. "
                    "The same 796-state checkpoint's timing surrogate reaches "
                    f"{pct(local_feature_i7_b3_s2_c2_settling.get('fastest_within_1pct_of_best_accuracy') if local_feature_i7_b3_s2_c2_settling else None)} "
                    f"by {local_feature_i7_b3_s2_c2_settling.get('fastest_within_1pct_of_best_readout_time_ns') if local_feature_i7_b3_s2_c2_settling else 'unknown'} ns. "
                    "Promoting the remaining sub-796 near misses found an even smaller but low-margin 706-state 7x7 b5 stride1 c1 branch: it reached "
                    f"{pct(local_feature_sub796_promote_10k.get('best_overall', {}).get('test_accuracy') if local_feature_sub796_promote_10k else None)} "
                    "on 10,000 train / 2,000 held-out, while the 712-state branch stayed below target. "
                    "Its 1,024-sample fast continuation ended at "
                    f"{pct(local_feature_i7_b5_s1_c1_fast1024.get('final_eval', {}).get('test_accuracy') if local_feature_i7_b5_s1_c1_fast1024 else None)} "
                    "but its 212-sample fast continuation dipped to "
                    f"{pct(local_feature_i7_b5_s1_c1_fast212.get('final_eval', {}).get('test_accuracy') if local_feature_i7_b5_s1_c1_fast212 else None)}. "
                    "A short 12-sample real 1 ns SPICE gate on the 706-state branch ended at "
                    f"{pct(local_feature_i7_b5_s1_c1_phase12.get('final_heldout_accuracy') if local_feature_i7_b5_s1_c1_phase12 else None)} "
                    f"with RMS drift {sci(local_feature_i7_b5_s1_c1_phase12.get('final_fast_reference_state_rms_diff') if local_feature_i7_b5_s1_c1_phase12 else None)}, "
                    "making it the smallest short-SPICE-validated above-90 path, but not the best-margin candidate."
                    + (
                        " No sub-706 pilot found."
                        if not local_feature_sub706_pilot_2k
                        else (
                            f" A follow-up sub-706 fast pilot on 2,000 train / 2,000 held-out over 56 compact topology/lr trials "
                            f"found no target hit; the best overall point was "
                            f"{pct(local_feature_sub706_pilot_2k.get('best_overall', {}).get('test_accuracy'))} "
                            f"at {local_feature_sub706_pilot_2k.get('best_overall', {}).get('phase_state_values')} states, "
                            "and the best true sub-706 point was 85.2% at 544 states. This argues against spending SPICE on these smaller shapes yet."
                        )
                    )
                    + (
                        " No shared-kernel compact pilot found."
                        if not local_feature_shared_kernel_promote_5k
                        else (
                            " A shared-kernel compact follow-up tied one learned local kernel per channel across block positions; "
                            f"the 2,000/2,000 pilot peaked at "
                            f"{pct(local_feature_shared_kernel_pilot_2k.get('best_overall', {}).get('test_accuracy') if local_feature_shared_kernel_pilot_2k else None)} "
                            f"with {local_feature_shared_kernel_pilot_2k.get('best_overall', {}).get('phase_state_values') if local_feature_shared_kernel_pilot_2k else 'missing'} states, "
                            f"and the focused 5,000/2,000 promotion peaked at "
                            f"{pct(local_feature_shared_kernel_promote_5k.get('best_overall', {}).get('test_accuracy'))} "
                            f"with {local_feature_shared_kernel_promote_5k.get('best_overall', {}).get('phase_state_values')} states. "
                            "That is useful state compression, but it changes the hardware assumption from per-block synaptic storage to shared capacitor broadcast or time-multiplexed reuse, and it is still below the accuracy threshold."
                        )
                    )
                    + (
                        " No partial-sharing compact promotion found."
                        if not local_feature_partial_sharing_promote_5k
                        else (
                            " A partial-sharing follow-up kept shared kernel capacitors for some channels and per-block private capacitors for others. "
                            f"The 2,000/2,000 pilot peaked at "
                            f"{pct(local_feature_partial_sharing_pilot_2k.get('best_overall', {}).get('test_accuracy') if local_feature_partial_sharing_pilot_2k else None)} "
                            f"with {local_feature_partial_sharing_pilot_2k.get('best_overall', {}).get('phase_state_values') if local_feature_partial_sharing_pilot_2k else 'missing'} states. "
                            f"Focused 5,000/2,000 promotion reached "
                            f"{pct(local_feature_partial_sharing_promote_5k.get('best_overall', {}).get('test_accuracy'))} "
                            f"with {local_feature_partial_sharing_promote_5k.get('best_overall', {}).get('phase_state_values')} states; "
                            f"its smallest target hit was "
                            f"{pct(local_feature_partial_sharing_promote_5k.get('smallest_target_hit', {}).get('test_accuracy'))} "
                            f"with {local_feature_partial_sharing_promote_5k.get('smallest_target_hit', {}).get('phase_state_values')} states. "
                            f"Correct shared-capacitor fast continuation keeps the 784-state 9x9 b3 shared branch at "
                            f"{pct(local_feature_partial_sharing_i9_b3_fast1024.get('final_eval', {}).get('test_accuracy') if local_feature_partial_sharing_i9_b3_fast1024 else None)} "
                            f"after 1,024 samples and the 854-state 7x7 b3 partial branch at "
                            f"{pct(local_feature_partial_sharing_i7_b3_fast1024.get('final_eval', {}).get('test_accuracy') if local_feature_partial_sharing_i7_b3_fast1024 else None)}. "
                            "These are compact accelerator candidates rather than the cleanest local-synapse architecture, because the shared channels use one physical weight-capacitor state across block positions."
                        )
                    )
                    + (
                        " No partial-sharing phase deck validation found."
                        if not local_feature_partial_phase_train12
                        else (
                            " The partial-sharing phase deck now implements those shared-capacitor semantics. "
                            f"A two-update 1 ns equivalence check matched the fast partial-sharing rule with RMS drift "
                            f"{sci(local_feature_partial_phase_u2.get('state_rms_diff') if local_feature_partial_phase_u2 else None)} "
                            f"and max drift {sci(local_feature_partial_phase_u2.get('state_max_abs_diff') if local_feature_partial_phase_u2 else None)}. "
                            f"A 12-sample repeated 1 ns SPICE gate on the 854-state branch ended with RMS drift "
                            f"{sci(local_feature_partial_phase_train12.get('final_fast_reference_state_rms_diff'))} "
                            f"against the correct fast reference, and a full 2,000-image ngspice eval of that final checkpoint reached "
                            f"{pct(local_feature_partial_phase12_eval2000.get('heldout_test_accuracy') if local_feature_partial_phase12_eval2000 else None)}. "
                            f"The smaller 784-state 9x9 b3 shared branch also passed a 12-sample repeated 1 ns SPICE gate with RMS drift "
                            f"{sci(local_feature_partial_i9_b3_phase_train12.get('final_fast_reference_state_rms_diff') if local_feature_partial_i9_b3_phase_train12 else None)} "
                            f"and reached {pct(local_feature_partial_i9_b3_phase12_eval2000.get('heldout_test_accuracy') if local_feature_partial_i9_b3_phase12_eval2000 else None)} "
                            "on a 2,000-image ngspice eval. The private/local 796-state and 706-state branches remain the cleaner evidence for the original weights-in-local-synapses architecture."
                        )
                    )
                )
            )
            + " "
            + (
                "No 10x10 small-network phase-transient check found."
                if not local_feature_smallnet10_b4_phase_lr008
                else (
                    f"A bounded SPICE equivalence check for the selected 10x10 4x4 stride-2/c2 lr_spice=0.08 checkpoint "
                    f"matched OP SPICE at 1 ns with RMS state diff "
                    f"{sci(local_feature_smallnet10_b4_phase_lr008.get('state_rms_diff'))} and max diff "
                    f"{sci(local_feature_smallnet10_b4_phase_lr008.get('state_max_abs_diff'))}, with phase wall time "
                    f"{sci(local_feature_smallnet10_b4_phase_lr008.get('phase_wall_time_s'))} s for four samples."
                )
            )
            + " "
            + (
                "No 10x10 small-network repeated phase tracker found."
                if not local_feature_smallnet10_b4_phase_tracker_lr008
                else (
                    f"A short repeated SPICE tracker for the same 10x10 4x4 stride-2/c2 checkpoint kept the "
                    f"1,000-image ngspice eval at "
                    f"{pct(local_feature_smallnet10_b4_phase_tracker_lr008.get('final_heldout_accuracy'))} "
                    f"after {local_feature_smallnet10_b4_phase_tracker_lr008.get('phase_train_samples')} real 1 ns "
                    "phase-trained samples, with phase/reference fast accuracies matched and RMS drift "
                    f"{sci(local_feature_smallnet10_b4_phase_tracker_lr008.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No longer 10x10 small-network phase resume found."
                if not local_feature_smallnet10_b4_phase_resume12to60_lr008
                else (
                    f"Resuming the 10x10 branch from 12 to 60 total samples stayed above target at "
                    f"{pct(local_feature_smallnet10_b4_phase_resume12to60_lr008.get('final_heldout_accuracy'))} "
                    f"with phase/reference fast accuracies matched and RMS drift "
                    f"{sci(local_feature_smallnet10_b4_phase_resume12to60_lr008.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No 10x10 recovery-window phase resume found."
                if not local_feature_smallnet10_b4_phase_resume60to120_lr008
                else (
                    f"Continuing the same 10x10 branch from 60 to 120 total samples recovered to "
                    f"{pct(local_feature_smallnet10_b4_phase_resume60to120_lr008.get('final_heldout_accuracy'))} "
                    f"with phase/reference fast accuracies matched and RMS drift "
                    f"{sci(local_feature_smallnet10_b4_phase_resume60to120_lr008.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No 10x10 180-sample phase resume found."
                if not local_feature_smallnet10_b4_phase_resume120to180_lr008
                else (
                    f"Continuing the same 10x10 branch from 120 to 180 total samples stayed at "
                    f"{pct(local_feature_smallnet10_b4_phase_resume120to180_lr008.get('final_heldout_accuracy'))} "
                    f"with phase/reference fast accuracies matched and RMS drift "
                    f"{sci(local_feature_smallnet10_b4_phase_resume120to180_lr008.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No 10x10 212-sample endpoint phase resume found."
                if not local_feature_smallnet10_b4_phase_resume180to212_lr008
                else (
                    f"Continuing the same 10x10 branch from 180 to 212 total samples reached the fast-predicted endpoint at "
                    f"{pct(local_feature_smallnet10_b4_phase_resume180to212_lr008.get('final_heldout_accuracy'))} "
                    f"with phase/reference fast accuracies matched and RMS drift "
                    f"{sci(local_feature_smallnet10_b4_phase_resume180to212_lr008.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No reduced-handoff 10x10 small-network screen found."
                if not local_feature_smallnet10_b4_handoff_e20
                else (
                    "A reduced-pretraining screen showed that 12 fast epochs were too low-margin after continuation, "
                    f"while 15 fast epochs produced a viable 10x10 b4 stride-2/c2 lr_spice=0.15 checkpoint at "
                    f"{pct(local_feature_smallnet10_b4_handoff_e15.get('best_overall', {}).get('test_accuracy'))}; "
                    "the 20-epoch screen also exposed a cleaner lr_spice=0.1 checkpoint that actually peaked at epoch 16."
                )
            )
            + " "
            + (
                "No reduced-handoff 10x10 SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_handoff_e15_lr015
                else (
                    f"The 15-epoch lr_spice=0.15 handoff then ran {local_feature_smallnet10_b4_phase_handoff_e15_lr015.get('phase_train_samples')} "
                    f"real 1 ns phase-trained samples from chunk 0 and stayed above target at "
                    f"{pct(local_feature_smallnet10_b4_phase_handoff_e15_lr015.get('final_heldout_accuracy'))}, with "
                    "phase/reference fast accuracies matched but higher RMS drift "
                    f"{sci(local_feature_smallnet10_b4_phase_handoff_e15_lr015.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No larger-coverage 10x10 small-network screen found."
                if not local_feature_smallnet10_b4_scale_5k2k
                else (
                    f"Scaling the same 10x10 b4 stride-2/c2 branch to {local_feature_smallnet10_b4_scale_5k2k.get('train_samples_list')} "
                    f"train / {local_feature_smallnet10_b4_scale_5k2k.get('test_samples')} held-out kept strong fast margin: "
                    f"best held-out accuracy was {pct(local_feature_smallnet10_b4_scale_5k2k.get('best_overall', {}).get('test_accuracy'))} "
                    f"with {local_feature_smallnet10_b4_scale_5k2k.get('best_overall', {}).get('phase_state_values')} phase-state values."
                )
            )
            + " "
            + (
                "No larger-coverage 10x10 SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_scale_5k2k_lr008
                else (
                    f"The larger-coverage lr_spice=0.08 checkpoint ran {local_feature_smallnet10_b4_phase_scale_5k2k_lr008.get('phase_train_samples')} "
                    f"real 1 ns phase-trained samples and ended at "
                    f"{pct(local_feature_smallnet10_b4_phase_scale_5k2k_lr008.get('final_heldout_accuracy'))} "
                    f"on {local_feature_smallnet10_b4_phase_scale_5k2k_lr008.get('test_samples')} ngspice-held-out images, "
                    f"with RMS drift {sci(local_feature_smallnet10_b4_phase_scale_5k2k_lr008.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No 10k-coverage 10x10 small-network screen found."
                if not local_feature_smallnet10_b4_scale_10k2k
                else (
                    f"Scaling the same branch again to {local_feature_smallnet10_b4_scale_10k2k.get('train_samples_list')} "
                    f"train / {local_feature_smallnet10_b4_scale_10k2k.get('test_samples')} held-out improved the fast margin: "
                    f"best held-out accuracy was {pct(local_feature_smallnet10_b4_scale_10k2k.get('best_overall', {}).get('test_accuracy'))} "
                    f"with the same {local_feature_smallnet10_b4_scale_10k2k.get('best_overall', {}).get('phase_state_values')} phase-state values. "
                    f"The lr_spice=0.08 fast continuation ended at "
                    f"{pct((local_feature_smallnet10_b4_scale_10k2k_fast_lr008 or {}).get('final_eval', {}).get('test_accuracy'))} "
                    f"after {(local_feature_smallnet10_b4_scale_10k2k_fast_lr008 or {}).get('final_eval', {}).get('samples_seen')} seen samples."
                )
            )
            + " "
            + (
                "No 10k-coverage 10x10 SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_scale_10k2k_lr008
                else (
                    f"The 10k/2k lr_spice=0.08 checkpoint ran {local_feature_smallnet10_b4_phase_scale_10k2k_lr008.get('phase_train_samples')} "
                    f"real 1 ns phase-trained samples and ended at "
                    f"{pct(local_feature_smallnet10_b4_phase_scale_10k2k_lr008.get('final_heldout_accuracy'))} "
                    f"on {local_feature_smallnet10_b4_phase_scale_10k2k_lr008.get('test_samples')} ngspice-held-out images, "
                    f"with RMS drift {sci(local_feature_smallnet10_b4_phase_scale_10k2k_lr008.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No full-test 10x10 fast continuation found."
                if not local_feature_smallnet10_b4_scale_10kfull_fast_lr008
                else (
                    f"Re-evaluating that 10k-trained lr_spice=0.08 checkpoint on all "
                    f"{local_feature_smallnet10_b4_scale_10kfull_fast_lr008.get('test_samples')} MNIST test images gave "
                    f"{pct(local_feature_smallnet10_b4_scale_10kfull_fast_lr008.get('best_eval', {}).get('test_accuracy'))} initially and "
                    f"{pct(local_feature_smallnet10_b4_scale_10kfull_fast_lr008.get('final_eval', {}).get('test_accuracy'))} after the "
                    f"{local_feature_smallnet10_b4_scale_10kfull_fast_lr008.get('final_eval', {}).get('samples_seen')}-sample fast continuation."
                )
            )
            + " "
            + (
                "No full-test 10x10 SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_scale_10kfull_lr008
                else (
                    f"The corresponding full-test SPICE phase run ended at "
                    f"{pct(local_feature_smallnet10_b4_phase_scale_10kfull_lr008.get('final_heldout_accuracy'))} "
                    f"on {local_feature_smallnet10_b4_phase_scale_10kfull_lr008.get('test_samples')} ngspice-held-out images, "
                    f"with fast reference accuracy {pct(local_feature_smallnet10_b4_phase_scale_10kfull_lr008.get('final_fast_reference_accuracy'))} "
                    f"and RMS drift {sci(local_feature_smallnet10_b4_phase_scale_10kfull_lr008.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No full-train/full-test 10x10 fast screen found."
                if not local_feature_smallnet10_b4_scale_60kfull
                else (
                    f"Training the same 1,832-state branch on all "
                    f"{int(local_feature_smallnet10_b4_scale_60kfull.get('best_overall', {}).get('train_samples', 0))} MNIST training images "
                    f"with the fast phase-portable rule raised the best checkpoint to "
                    f"{pct(local_feature_smallnet10_b4_scale_60kfull.get('best_overall', {}).get('test_accuracy'))} "
                    f"on all {local_feature_smallnet10_b4_scale_60kfull.get('test_samples')} test images at lr_spice="
                    f"{local_feature_smallnet10_b4_scale_60kfull.get('best_overall', {}).get('lr_spice')}."
                )
            )
            + " "
            + (
                "No full-train/full-test 10x10 fast continuation found."
                if not local_feature_smallnet10_b4_scale_60kfull_fast_lr003
                else (
                    f"The full-train lr_spice=0.03 checkpoint stayed at "
                    f"{pct(local_feature_smallnet10_b4_scale_60kfull_fast_lr003.get('final_eval', {}).get('test_accuracy'))} "
                    f"after {local_feature_smallnet10_b4_scale_60kfull_fast_lr003.get('final_eval', {}).get('samples_seen')} fast continuation samples."
                )
            )
            + " "
            + (
                "No full-train/full-test 10x10 SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_scale_60kfull_lr003
                else (
                    f"The full-train/full-test SPICE phase run ended at "
                    f"{pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003.get('final_heldout_accuracy'))} "
                    f"on all {local_feature_smallnet10_b4_phase_scale_60kfull_lr003.get('test_samples')} ngspice-held-out images, "
                    f"with fast reference accuracy {pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003.get('final_fast_reference_accuracy'))} "
                    f"and RMS drift {sci(local_feature_smallnet10_b4_phase_scale_60kfull_lr003.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No longer full-train/full-test fast continuation found."
                if not local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume1024
                else (
                    f"A longer fast continuation from the 212-sample state predicted the branch stays at "
                    f"{pct(local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume1024.get('final_eval', {}).get('test_accuracy'))} "
                    f"after {local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume1024.get('final_eval', {}).get('samples_seen')} total samples, "
                    f"with a best point of {pct(local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume1024.get('best_eval', {}).get('test_accuracy'))}."
                )
            )
            + " "
            + (
                "No resumed full-train/full-test 10x10 SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume424
                else (
                    f"Resuming the SPICE capacitor state from sample "
                    f"{local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume424.get('phase_train_start_sample')} to "
                    f"{local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume424.get('phase_train_end_sample')} ended at "
                    f"{pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume424.get('final_heldout_accuracy'))} "
                    f"on all {local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume424.get('test_samples')} ngspice-held-out images, "
                    f"with fast reference accuracy {pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume424.get('final_fast_reference_accuracy'))} "
                    f"and RMS drift {sci(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume424.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No 600-sample resumed full-train/full-test SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume600
                else (
                    f"Continuing the same SPICE capacitor trajectory to "
                    f"{local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume600.get('phase_train_end_sample')} total samples ended at "
                    f"{pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume600.get('final_heldout_accuracy'))}, "
                    f"with fast reference accuracy {pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume600.get('final_fast_reference_accuracy'))} "
                    f"and RMS drift {sci(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume600.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No 1024-sample resumed full-train/full-test SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume1024
                else (
                    f"Continuing to the fast-predicted {local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume1024.get('phase_train_end_sample')}-sample endpoint ended at "
                    f"{pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume1024.get('final_heldout_accuracy'))}, "
                    f"with fast reference accuracy {pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume1024.get('final_fast_reference_accuracy'))} "
                    f"and RMS drift {sci(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume1024.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No 2048-sample full-train/full-test fast continuation found."
                if not local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume2048
                else (
                    f"A smaller-step fast gate from 1,024 to "
                    f"{local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume2048.get('final_eval', {}).get('samples_seen')} "
                    f"total samples predicted {pct(local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume2048.get('final_eval', {}).get('test_accuracy'))} "
                    f"final accuracy, with a best point of "
                    f"{pct(local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume2048.get('best_eval', {}).get('test_accuracy'))} "
                    f"at {local_feature_smallnet10_b4_scale_60kfull_fast_lr003_resume2048.get('best_eval', {}).get('samples_seen')} samples."
                )
            )
            + " "
            + (
                "No 2048-sample resumed full-train/full-test SPICE phase tracker found."
                if not local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume2048
                else (
                    f"Resuming the same SPICE capacitor trajectory from "
                    f"{local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume2048.get('phase_train_start_sample')} to "
                    f"{local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume2048.get('phase_train_end_sample')} "
                    f"total samples ended at {pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume2048.get('final_heldout_accuracy'))}, "
                    f"with fast reference accuracy {pct(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume2048.get('final_fast_reference_accuracy'))} "
                    f"and RMS drift {sci(local_feature_smallnet10_b4_phase_scale_60kfull_lr003_resume2048.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No phase-transient overlap check found."
                if not local_feature_overlap_phase_lr03
                else (
                    f"The 12x12 overlap c2 checkpoint matched OP SPICE on the first 2k-sequence four-sample batch at "
                    f"1 ns and lr_spice=0.3 with RMS state diff "
                    f"{sci(local_feature_overlap_phase_lr03.get('state_rms_diff'))}; "
                    + (
                        "no lr_spice=0.5 comparison was found."
                        if not local_feature_overlap_phase_lr05
                        else (
                            f"the lr_spice=0.5 comparison had RMS state diff "
                            f"{sci(local_feature_overlap_phase_lr05.get('state_rms_diff'))}."
                        )
                    )
                )
            )
            + " "
            + (
                "No difficult-batch overlap timing check found."
                if not local_feature_overlap_phase_train12_1ns
                else (
                    f"On the first tracker-sequence batch, however, the same lr_spice=0.3 overlap deck had "
                    f"1 ns RMS state diff {sci(local_feature_overlap_phase_train12_1ns.get('state_rms_diff'))}; "
                    + (
                        "no 4 ns timing comparison was found."
                        if not local_feature_overlap_phase_train12_4ns
                        else (
                            f"4 ns reduced this to "
                            f"{sci(local_feature_overlap_phase_train12_4ns.get('state_rms_diff'))}."
                        )
                    )
                )
            )
            + " "
            + (
                "No stabilized low-lr overlap phase check found."
                if not local_feature_overlap_phase_lr015
                else (
                    f"The lr_spice=0.15 trained overlap checkpoint had 1 ns difficult-batch RMS state diff "
                    f"{sci(local_feature_overlap_phase_lr015.get('state_rms_diff'))}, preserving a >90% fast branch "
                    "with much lower phase error than lr_spice=0.3."
                )
            )
            + " "
            + (
                "No batch-size-matched lr_spice=0.1 phase check found."
                if not local_feature_overlap_phase_batch2_lr01
                else (
                    f"The batch-size-matched lr_spice=0.1 checkpoint had 1 ns RMS state diff "
                    f"{sci(local_feature_overlap_phase_batch2_lr01.get('state_rms_diff'))} on the first 2k-sequence "
                    "four-sample batch."
                )
            )
            + " "
            + (
                "No repeated overlap phase-training tracker found."
                if not local_feature_overlap_phase_tracker
                else (
                    f"A 12-sample repeated 1 ns overlap tracker ended with RMS drift "
                    f"{sci(local_feature_overlap_phase_tracker.get('final_fast_reference_state_rms_diff'))} and max drift "
                    f"{sci(local_feature_overlap_phase_tracker.get('final_fast_reference_state_max_abs_diff'))}, "
                    "so the smaller topology is SPICE-runnable but not yet stable enough for long phase training."
                )
            )
            + " "
            + (
                "No stabilized low-lr repeated overlap tracker found."
                if not local_feature_overlap_phase_tracker_lr015
                else (
                    f"A matched lr_spice=0.15 12-sample overlap tracker ended with RMS drift "
                    f"{sci(local_feature_overlap_phase_tracker_lr015.get('final_fast_reference_state_rms_diff'))} and max drift "
                    f"{sci(local_feature_overlap_phase_tracker_lr015.get('final_fast_reference_state_max_abs_diff'))}, "
                    "restoring bounded repeated-chunk behavior on the same 1 ns schedule."
                )
            )
            + " "
            + (
                "No batch-size-matched repeated overlap tracker found."
                if not local_feature_overlap_phase_tracker_batch2_lr01
                else (
                    f"The batch-size-matched lr_spice=0.1 tracker SPICE-evaluated the checkpoint at "
                    f"{pct(local_feature_overlap_phase_tracker_batch2_lr01.get('best_eval', {}).get('heldout_accuracy'))} "
                    f"and ended after 12 phase-trained samples at "
                    f"{pct(local_feature_overlap_phase_tracker_batch2_lr01.get('final_heldout_accuracy'))}, with RMS drift "
                    f"{sci(local_feature_overlap_phase_tracker_batch2_lr01.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No fast continuation prediction for the batch-size-matched branch found."
                if not local_feature_overlap_fast_resume_batch2_lr01
                else (
                    f"A fast checkpoint-continuation prediction from the 12-sample batch-matched checkpoint stayed at "
                    f"{pct(local_feature_overlap_fast_resume_batch2_lr01.get('best_eval', {}).get('test_accuracy'))} "
                    f"near the start and ended at "
                    f"{pct(local_feature_overlap_fast_resume_batch2_lr01.get('final_eval', {}).get('test_accuracy'))} "
                    f"after {local_feature_overlap_fast_resume_batch2_lr01.get('end_sample')} seen samples."
                )
            )
            + " "
            + (
                "No longer batch-size-matched SPICE continuation found."
                if not local_feature_overlap_phase_resume_batch2_lr01
                else (
                    f"Resuming the same SPICE phase state from chunk 3 to chunk 15 added "
                    f"{local_feature_overlap_phase_resume_batch2_lr01.get('phase_train_samples')} real 1 ns phase-trained "
                    f"samples and ended at {pct(local_feature_overlap_phase_resume_batch2_lr01.get('final_heldout_accuracy'))} "
                    f"on the 1,000-image ngspice slice with RMS drift "
                    f"{sci(local_feature_overlap_phase_resume_batch2_lr01.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No reduced-pretraining handoff check found."
                if not local_feature_overlap_phase_handoff_e10_lr005
                else (
                    f"Reducing the handoff checkpoint to 10 fast epochs at lr_spice=0.05 still left a "
                    f"{pct(local_feature_overlap_handoff_e10_lr005.get('best_overall', {}).get('test_accuracy'))} "
                    f"fast checkpoint; after {local_feature_overlap_phase_handoff_e10_lr005.get('phase_train_samples')} "
                    f"real 1 ns SPICE phase-trained samples it ended at "
                    f"{pct(local_feature_overlap_phase_handoff_e10_lr005.get('final_heldout_accuracy'))} with RMS drift "
                    f"{sci(local_feature_overlap_phase_handoff_e10_lr005.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No 120-sample reduced-pretraining continuation found."
                if not local_feature_overlap_phase_handoff_e10_lr005_resume
                else (
                    f"Continuing that 10-epoch handoff to 120 total seen samples ended at "
                    f"{pct(local_feature_overlap_phase_handoff_e10_lr005_resume.get('final_heldout_accuracy'))} "
                    f"on the 1,000-image ngspice slice with phase/reference fast accuracies both "
                    f"{pct(local_feature_overlap_phase_handoff_e10_lr005_resume.get('final_phase_fast_accuracy'))} "
                    f"and RMS drift {sci(local_feature_overlap_phase_handoff_e10_lr005_resume.get('final_fast_reference_state_rms_diff'))}."
                )
            )
            + " "
            + (
                "No phase-transient check of the frontier point found."
                if not local_feature_frontier_phase
                else (
                    f"A 1 ns phase-transient SPICE two-batch check of that c8/lr_spice=0.8 frontier point matched "
                    f"the operating-point SPICE update with max state diff "
                    f"{sci(local_feature_frontier_phase.get('state_max_abs_diff'))} and RMS state diff "
                    f"{sci(local_feature_frontier_phase.get('state_rms_diff'))}, using scalar final-state measurements."
                )
            )
            + " "
            + (
                "No random-state phase-training tracker found."
                if not local_feature_phase_train_tracker
                else (
                    f"A random-state repeated phase-training tracker ran "
                    f"{local_feature_phase_train_tracker.get('phase_train_samples')} samples through the same 14x14/c8 "
                    f"1 ns phase deck and compared against the fast reference rule; final RMS state drift was "
                    f"{sci(local_feature_phase_train_tracker.get('final_fast_reference_state_rms_diff'))}, max drift was "
                    f"{sci(local_feature_phase_train_tracker.get('final_fast_reference_state_max_abs_diff'))}, and the "
                    "dominant drift was in output/readout state."
                )
            )
            + " "
            + (
                "No lower-learning-rate phase-training tracker found."
                if not local_feature_phase_train_tracker_lr03
                else (
                    f"A matched lr_spice=0.3 tracker on the same 12-sample 1 ns schedule reduced final RMS drift to "
                    f"{sci(local_feature_phase_train_tracker_lr03.get('final_fast_reference_state_rms_diff'))} and max drift to "
                    f"{sci(local_feature_phase_train_tracker_lr03.get('final_fast_reference_state_max_abs_diff'))}, "
                    "showing update scale is a direct control knob for repeated-chunk phase error."
                )
            )
            + " "
            + (
                "No 2 ns high-learning-rate tracker found."
                if not local_feature_phase_train_tracker_lr08_2ns
                else (
                    f"A matched lr_spice=0.8 tracker with 2 ns phases reduced final RMS drift to "
                    f"{sci(local_feature_phase_train_tracker_lr08_2ns.get('final_fast_reference_state_rms_diff'))} and max drift to "
                    f"{sci(local_feature_phase_train_tracker_lr08_2ns.get('final_fast_reference_state_max_abs_diff'))}, "
                    "about 2x better than 1 ns at the same learning rate but still worse than the lr_spice=0.3 branch."
                )
            )
            + " "
            + (
                "No longer lr_spice=0.3 tracked phase-training curve found."
                if not local_feature_phase_train_tracker_lr03_20
                else (
                    f"A longer resumable lr_spice=0.3 tracker ran "
                    f"{local_feature_phase_train_tracker_lr03_20.get('phase_train_samples')} samples over "
                    f"{local_feature_phase_train_tracker_lr03_20.get('chunks')} chunks with per-chunk checkpoints; "
                    f"final RMS drift was {sci(local_feature_phase_train_tracker_lr03_20.get('final_fast_reference_state_rms_diff'))} "
                    f"and max drift was {sci(local_feature_phase_train_tracker_lr03_20.get('final_fast_reference_state_max_abs_diff'))}."
                )
            )
            + " "
            + (
                "No resumed lr_spice=0.3 phase-training evaluation branch found."
                if not local_feature_phase_train_resume_eval
                else (
                    f"Continuing the same c8 branch from samples "
                    f"{local_feature_phase_train_resume_eval.get('phase_train_start_sample')} to "
                    f"{local_feature_phase_train_resume_eval.get('phase_train_end_sample')} improved the 200-image "
                    f"SPICE held-out slice to {pct(local_feature_phase_train_resume_eval.get('final_heldout_accuracy'))}; "
                    f"final RMS state drift against the fast reference was "
                    f"{sci(local_feature_phase_train_resume_eval.get('final_fast_reference_state_rms_diff'))}."
                )
            )
        )
    )

    best_spice_lut_note = "No SPICE-LUT local-model summary found."
    if spice_lut and spice_lut.get("best_spice_lut"):
        row = spice_lut["best_spice_lut"]
        best_spice_lut_note = (
            f"PyTorch local model with ngspice-derived 4-bit charge-ADC activation LUT: "
            f"{row.get('train_limit')} train / {row.get('test_limit')} test, accuracy {pct(row.get('accuracy'))}; "
            "network execution/training was not in SPICE."
        )

    candidate_note = "No full-MNIST high-level candidate summary found."
    if candidate:
        candidate_note = (
            f"High-level full-MNIST local candidate: accuracy {pct(candidate.get('accuracy'))}; "
            "PyTorch execution, not all-SPICE."
        )

    criteria = [
        {
            "criterion": "Programmable trainable weights, not hard-wired constants",
            "status": "partial",
            "evidence": (
                "Small SPICE training demos store weights as capacitor node voltages and update them with behavioral currents. "
                "The >90% SPICE forward proxies use fixed coefficients or offline-trained weights. "
                + bias_cal_note
                + " "
                + local_feature_note
                + " "
                + small_rule_note
                + " "
                + noise_sweep_note
                + " "
                + frontier_note
                + " "
                + device_relu_note
                + " "
                + device_signed_note
                + " "
                + device_delta_note
                + " "
                + device_tiny_note
                + " "
                + device_sequential_note
                + " "
                + device_multicell_note
                + " "
                + device_feedback_alignment_note
                + " "
                + device_parity3_note
                + " "
                + device_xor2_note
                + " "
                + device_xor2_hidden_repair_note
                + " "
                + device_xor2_random_hidden_note
                + " "
                + device_xor2_random_hidden_wide_note
                + " "
                + device_xor2_readout_rule_note
                + device_xor2_lowtarget_rule_note
                + " "
                + device_xor2_calibrated_readout_note
                + device_xor2_separator_score_note
                + device_xor2_mistake_gate_note
                + device_xor2_local_loss_note
                + device_xor2_output_senseamp_note
                + device_xor2_backprop_synapse_note
            ),
            "blocking_gap": "Need the target local MNIST architecture to store and update its weights inside SPICE.",
        },
        {
            "criterion": "Forward pass runs entirely in SPICE",
            "status": "partial",
            "evidence": (
                best_spice_forward_note
                + " "
                + best_digits_forward_note
                + " "
                + forward_settle_note
                + " "
                + settling_pareto_note
                + " "
                + local_feature_settling_note
                + " "
                + bias_cal_note
                + " "
                + local_feature_note
                + " "
                + small_rule_note
                + " "
                + noise_sweep_note
                + " "
                + frontier_note
                + " "
                + device_relu_note
                + " "
                + device_signed_note
                + " "
                + device_delta_note
                + " "
                + device_tiny_note
                + " "
                + device_sequential_note
                + " "
                + device_multicell_note
                + " "
                + device_feedback_alignment_note
                + " "
                + device_parity3_note
                + " "
                + device_xor2_note
                + " "
                + device_xor2_hidden_repair_note
                + " "
                + device_xor2_random_hidden_note
                + " "
                + device_xor2_random_hidden_wide_note
                + " "
                + device_xor2_readout_rule_note
                + device_xor2_lowtarget_rule_note
                + " "
                + device_xor2_calibrated_readout_note
                + device_xor2_separator_score_note
                + device_xor2_mistake_gate_note
                + device_xor2_local_loss_note
                + device_xor2_output_senseamp_note
                + device_xor2_backprop_synapse_note
            ),
            "blocking_gap": "Need the intended local noise-robust architecture, not a dense/offline proxy, evaluated over full MNIST.",
        },
        {
            "criterion": "Training/backward/update path runs entirely in SPICE",
            "status": "partial",
            "evidence": (
                best_spice_train_note
                + " "
                + bias_cal_note
                + " "
                + local_feature_note
                + " "
                + small_rule_note
                + " "
                + noise_sweep_note
                + " "
                + frontier_note
                + " "
                + device_relu_note
                + " "
                + device_signed_note
                + " "
                + device_delta_note
                + " "
                + device_tiny_note
                + " "
                + device_sequential_note
                + " "
                + device_multicell_note
                + " "
                + device_feedback_alignment_note
                + " "
                + device_parity3_note
                + " "
                + device_xor2_note
                + " "
                + device_xor2_hidden_repair_note
                + " "
                + device_xor2_random_hidden_note
                + " "
                + device_xor2_random_hidden_wide_note
                + " "
                + device_xor2_readout_rule_note
                + device_xor2_lowtarget_rule_note
                + " "
                + device_xor2_calibrated_readout_note
                + device_xor2_separator_score_note
                + device_xor2_mistake_gate_note
                + device_xor2_local_loss_note
                + device_xor2_output_senseamp_note
                + device_xor2_backprop_synapse_note
            ),
            "blocking_gap": "Current all-SPICE training is tiny and far below 90%; full MNIST training has not run in SPICE.",
        },
        {
            "criterion": "Architecture is local and hardware-plausible",
            "status": "partial",
            "evidence": (
                candidate_note
                + " "
                + best_spice_lut_note
                + " "
                + local_feature_note
                + " "
                + small_rule_note
                + " "
                + noise_sweep_note
                + " "
                + frontier_note
                + " "
                + device_relu_note
                + " "
                + device_signed_note
                + " "
                + device_delta_note
                + " "
                + device_tiny_note
                + " "
                + device_sequential_note
                + " "
                + device_multicell_note
                + " "
                + device_feedback_alignment_note
                + " "
                + device_parity3_note
                + " "
                + device_xor2_note
                + " "
                + device_xor2_hidden_repair_note
                + " "
                + device_xor2_random_hidden_note
                + " "
                + device_xor2_random_hidden_wide_note
                + " "
                + device_xor2_readout_rule_note
                + device_xor2_lowtarget_rule_note
                + " "
                + device_xor2_calibrated_readout_note
                + device_xor2_separator_score_note
                + device_xor2_mistake_gate_note
                + device_xor2_local_loss_note
                + device_xor2_output_senseamp_note
                + device_xor2_backprop_synapse_note
            ),
            "blocking_gap": "The richer local architecture still clears 90% using behavioral cells or fast-trained checkpoints; need to migrate the high-accuracy path to transistor/conductance primitives and prove training robustness.",
        },
        {
            "criterion": "Electronic neuron primitive is noise-tolerant; stochasticity is optional",
            "status": "partial",
            "evidence": (
                "Comparator and charge-ADC primitives have ngspice sweeps, and a PyTorch local model can use the SPICE LUT. "
                "The local all-SPICE batch-op trainer now uses configurable analog tanh, ReLU, clipped-ReLU, or differential "
                "clipped-ReLU voltage states and can inject input noise, weight mismatch, local offset, and output offset. "
                "The differential clipped-ReLU state is a bounded signed voltage built from two rectifier branches. "
                "Robustness of a high-accuracy all-SPICE trainable "
                "network has still not been demonstrated. "
                + noise_sweep_note
                + " "
                + device_relu_note
                + " "
                + device_signed_note
                + " "
                + device_delta_note
                + " "
                + device_tiny_note
                + " "
                + device_sequential_note
                + " "
                + device_multicell_note
                + " "
                + device_feedback_alignment_note
                + " "
                + device_parity3_note
                + " "
                + device_xor2_note
                + " "
                + device_xor2_hidden_repair_note
                + " "
                + device_xor2_random_hidden_note
                + " "
                + device_xor2_random_hidden_wide_note
                + " "
                + device_xor2_readout_rule_note
                + device_xor2_lowtarget_rule_note
                + " "
                + device_xor2_calibrated_readout_note
                + device_xor2_separator_score_note
                + device_xor2_mistake_gate_note
                + device_xor2_local_loss_note
                + device_xor2_output_senseamp_note
                + device_xor2_backprop_synapse_note
            ),
            "blocking_gap": (
                "Need the all-SPICE trainable network to include explicit noise/nonidealities or extracted circuit "
                "models, then pass accuracy tests under noise/drift/mismatch. It does not have to sample stochastic bits."
            ),
        },
        {
            "criterion": "Preferably self-timed operation",
            "status": "not_achieved",
            "evidence": "Only architecture notes mention local handshake/self-timing. Existing SPICE demos use explicit sample sequencing.",
            "blocking_gap": "Need a self-timed or locally handshaked tile/control model, or an explicit decision to use a minimal clock.",
        },
        {
            "criterion": "Full MNIST, not sklearn digits or small subsets",
            "status": "not_achieved",
            "evidence": (
                "All-ngspice training uses small downsampled MNIST subsets. SPICE forward MNIST proxy uses 200 held-out samples. "
                "PyTorch local results use bounded 10k/2k or full MNIST but not all-SPICE. "
                + bias_cal_note
                + " "
                + local_feature_note
                + " "
                + small_rule_note
                + " "
                + noise_sweep_note
                + " "
                + frontier_note
            ),
            "blocking_gap": "Need 60k-train/10k-test MNIST data cycling and evaluation in SPICE or a clearly accepted equivalent SPICE execution strategy.",
        },
        {
            "criterion": "Accuracy greater than 90% on full MNIST",
            "status": "not_achieved",
            "evidence": (
                f"Best all-SPICE training held-out accuracy: {pct(best_spice_train_acc)}. "
                f"Best SPICE forward-only MNIST proxy: {pct(best_spice_forward_acc)} on 200 samples, but training was offline. "
                + forward_settle_note
                + " "
                + settling_pareto_note
                + " "
                + local_feature_settling_note
                + " "
                + bias_cal_note
                + " "
                + local_feature_note
                + " "
                + small_rule_note
                + " "
                + noise_sweep_note
                + " "
                + frontier_note
            ),
            "blocking_gap": "Need >90% from the same full-MNIST all-SPICE trainable local, noise-robust architecture.",
        },
    ]

    achieved = all(c["status"] == "achieved" for c in criteria)
    audit = {
        "objective": OBJECTIVE,
        "achieved": achieved,
        "all_spice_training_runs_found": len(spice_train_runs),
        "criteria": criteria,
        "scaling_evidence": [
            {
                "artifact": "spice/run_device_relu_synapse_sweep.py",
                "finding": "Adds a first transistor/device-level primitive harness for the hardware pivot: MOS pass-conductance synapses charge a pre-activation capacitor, an NMOS source follower stores a ReLU-like activation on Cact, and differential gradient capacitor voltages drive charge/discharge paths for a weight capacitor. The sweep intentionally avoids behavioral tanh and behavioral multipliers in the signal path.",
            },
            {
                "artifact": "spice/results/device_relu_synapse_update_v0_summary.json",
                "finding": "The device-level primitive sweep passed the basic operating checks under ngspice LEVEL=1 MOS models: ReLU transfer was monotone, synapse preactivation was monotone in input and weight-gate voltage, the resized single synapse drove a ReLU activation up to about 90.6 mV, and the differential update cell moved the weight capacitor monotonically for positive and negative gradient-cap voltages. This is not yet a signed, trainable MNIST cell or a foundry-PDK result.",
            },
            {
                "artifact": "spice/run_device_signed_learning_cell.py",
                "finding": "Adds a device-level signed learning-cell harness: positive and negative weight capacitors implement differential signed conductance; input/delta/acc transistor stacks charge Cgp/Cgn; and apply-phase transistor stacks update Cwp/Cwn before a second forward readout. The signal path avoids behavioral tanh and behavioral multipliers.",
            },
            {
                "artifact": "spice/results/device_signed_learning_cell_v2_summary.json",
                "finding": "The signed learning-cell sweep passed primitive data-derived update checks under ngspice LEVEL=1 MOS models: positive/negative gradient accumulator caps were monotone in input, signed updates were monotone in input and delta, strong positive updates increased activation, and strong negative updates decreased activation. Max signed-weight changes were about +0.25 V and -0.95 V. This still lacks network error computation, backprop delta cells, full MNIST dataflow, and foundry-PDK models.",
            },
            {
                "artifact": "spice/run_device_delta_cells.py",
                "finding": "Adds device-level output-error and hidden-delta primitive cells. The output-error cell uses target/output conductance competition to write dplus/dminus caps. The hidden-delta cell uses output-delta/readout-weight sign-combination transistor stacks, gated by activation, to write hdp/hdn caps. The signal path avoids behavioral subtraction and multiplication.",
            },
            {
                "artifact": "spice/results/device_delta_cells_v1_summary.json",
                "finding": "The delta-cell sweep passed primitive checks under ngspice LEVEL=1 MOS models: differential error dplus-dminus was monotone with target and decreased with output; target>output selected positive error polarity; output>target selected negative error polarity; all four hidden-delta sign cases passed; and inactive ReLU gating suppressed hidden-delta storage. This is still not a multi-cell classifier or MNIST training run.",
            },
            {
                "artifact": "spice/run_device_tiny_classifier.py",
                "finding": "Adds a tiny end-to-end device-level classifier harness: one input, one signed hidden ReLU cell, one signed readout ReLU output, output-error caps, hidden-delta caps, readout/hidden gradient caps, and apply-phase weight updates. The signal path avoids behavioral tanh, subtraction, and multiplication; Python only instantiates the netlists and parses measurements.",
            },
            {
                "artifact": "spice/results/device_tiny_classifier_v6_summary.json",
                "finding": "The tiny classifier smoke test passed both polarity cases under ngspice LEVEL=1 MOS models: high target produced positive error, positive readout and hidden signed-weight updates, and increased output; low target produced negative error, negative readout and hidden signed-weight updates, and decreased output. This is still only a two-case device-level smoke test, not scaled MNIST training or a foundry-PDK result.",
            },
            {
                "artifact": "spice/run_device_sequential_training.py",
                "finding": "Adds a repeated-sample device-level training harness. A single transient deck repeats the guide phases over multiple samples while persistent hidden/readout weight capacitors remain in SPICE; only temporary activation, error, hidden-delta, and gradient caps are reset between samples.",
            },
            {
                "artifact": "spice/results/device_sequential_training_v0_summary.json",
                "finding": "The sequential training smoke test passed both high-then-low and low-then-high two-sample sequences under ngspice LEVEL=1 MOS models: error polarity, readout signed-weight update polarity, hidden signed-weight update polarity, and output movement all matched the target direction. This demonstrates weight-cap state persistence across samples inside one SPICE transient, but it still uses a tiny single-hidden-cell binary loop with coarse updates.",
            },
            {
                "artifact": "spice/run_device_multicell_classifier.py",
                "finding": "Adds a 2-input/2-hidden/2-output device-level classifier harness. Two hidden ReLU capacitor cells feed two output cells; one-hot output-error caps drive readout-gradient caps and hidden-delta caps; gradient caps then update persistent hidden and readout weight capacitors inside a single transient. The signal path avoids behavioral tanh, subtraction, and multiplication.",
            },
            {
                "artifact": "spice/results/device_multicell_classifier_v1_summary.json",
                "finding": "The multicell classifier smoke test passed class-0/class-1 and class-1/class-0 two-sample sequences under ngspice LEVEL=1 MOS models. For every sample, target output increased, non-target output decreased, target-vs-non-target margin improved, target active readout increased, non-target active readout decreased, and active hidden-delta paths were nonzero. This is a synthetic two-class identity task, not MNIST training.",
            },
            {
                "artifact": "spice/run_device_feedback_alignment.py",
                "finding": "Adds the first alternative training-rule device harness: a direct-feedback-alignment variant of the multicell classifier. Hidden deltas are generated through fixed signed feedback capacitor nodes instead of learned readout weight capacitors, so the hidden update path avoids exact readout-weight transport.",
            },
            {
                "artifact": "spice/results/device_feedback_alignment_v0_summary.json",
                "finding": "The direct-feedback-alignment smoke test passed the same synthetic two-class checks as the backprop-style multicell harness: target outputs rose, non-target outputs fell, margins improved, and fixed feedback caps produced nonzero hidden deltas. Mean margin improvement matched the backprop-style smoke on this symmetric toy. This is not evidence that random feedback scales to MNIST.",
            },
            {
                "artifact": "spice/run_device_parity3_readout.py",
                "finding": "Adds a complete tiny-dataset device benchmark for 3-bit parity. Six literal input rails drive eight capacitor-held pattern feature cells, then a two-output readout is trained with capacitor-held gradient accumulators and persistent readout weight caps over all eight patterns.",
            },
            {
                "artifact": "spice/results/device_parity3_readout_v0_summary.json",
                "finding": "The 3-bit parity readout benchmark trained over all eight parity patterns and evaluated all eight patterns in the same ngspice transient. Both binary and interleaved train orders reached 100% eval accuracy after one readout-training epoch, with all train updates improving the target-vs-other margin. Hidden literal features were programmed capacitor states, not learned hidden weights.",
            },
            {
                "artifact": "spice/run_device_xor2_learned_features.py",
                "finding": "Adds a nonlinear 2-bit XOR device benchmark where four literal hidden feature cells and a two-output readout run forward/error/backward/accumulate/apply phases. Both readout weights and active hidden-feature match weights are capacitor-held and updated in SPICE, with hidden deltas propagated through signed readout weight capacitors.",
            },
            {
                "artifact": "spice/results/device_xor2_learned_features_v0_summary.json",
                "finding": "The 2-bit XOR learned-hidden-update benchmark reached 100% eval accuracy for binary and interleaved train orders in ngspice LEVEL=1 MOS, with all train updates improving margin and all active hidden-feature weight updates nonzero. The hidden features were initialized as programmed literal detectors, so this is a hidden-update milestone, not random hidden-feature discovery.",
            },
            {
                "artifact": "spice/run_device_xor2_hidden_repair.py",
                "finding": "Adds a hidden-only 2-bit XOR repair benchmark. The readout is frozen; each literal hidden feature has fixed mismatch suppression and a trainable differential signed match weight, so positive hidden error updates only hidden match-weight capacitors inside the SPICE transient.",
            },
            {
                "artifact": "spice/results/device_xor2_hidden_repair_v11_summary.json",
                "finding": "The hidden-only repair benchmark preserved 100% eval accuracy while improving the minimum margin from 23.1761 mV to 23.1883 mV. All hidden signed match weights increased by 0.151362 mV over three epochs. This is still literal-feature repair, not random hidden-feature discovery.",
            },
            {
                "artifact": "spice/run_device_xor2_random_hidden.py",
                "finding": "Adds a nonliteral hidden-layer device benchmark for 2-bit XOR. The current harness can run eight hidden ReLU cells fully connected to x0/nx0/x1/nx1 plus a bias rail, train capacitor-held output-bias/readout/hidden weights, add internal parasitic caps/leaks for convergence, and optionally accumulate gradient caps over a four-pattern mini-epoch before one apply pulse.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v2_summary.json",
                "finding": "The dense random-hidden XOR device run improved eval accuracy from 25% to 75% while updating both hidden and readout signed weights in a single ngspice transient. It does not solve XOR: one final pattern remains wrong and the final minimum margin is -1.525 mV, so stable nonliteral hidden training is still open.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v31_summary.json",
                "finding": "The wider random-hidden XOR device run uses eight nonliteral hidden cells and batch gradient accumulation over 12 train cycles with 3 apply cycles. It still ends at 50% eval accuracy, but the SPICE-measured final hidden activations are linearly separable in a post-run diagnostic, so the blocker is the local analog readout/update rule rather than hidden representability.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v48_summary.json",
                "finding": "Readout-rule sweeps on the same eight-hidden XOR circuit added direct perceptron, pairwise margin, and competitive local-loss style error rules. The strongest direct perceptron run moved hidden-to-output readout weights by 318.419 mV total but still ended at 50% eval accuracy; score-gated margin and competitive rules produced only microvolt-scale movement. This narrows the blocker to a stronger mistake/low-margin comparator or error latch.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v53_summary.json",
                "finding": "The low-score target-gated rule moved hidden-to-output readout weights by 47.9312 mV total, but the low-score gate voltages did not separate the two class outputs well enough and the run stayed at 50% eval accuracy. This is a negative result for the simple resistor-load low-score gate.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v69_summary.json",
                "finding": "The programmed-separator readout check initialized the same eight nonliteral hidden cells from a SPICE-measured hidden-activation separator plus a 40 mV differential output-bias offset. With zero training epochs, the ngspice transient reached 100% XOR eval accuracy. This proves the current hidden representation and transistor-level readout can express XOR, but the readout was programmed rather than learned.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v76_summary.json",
                "finding": "Starting from the programmed separator, a three-epoch batch score-rule run preserved 100% XOR eval accuracy with a 24.9948 uV final minimum margin. The same calibrated start degrades to 75% under the direct perceptron and lowtarget controls, so preserving a correct analog separator is easier than discovering one from random readout weights.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v77_summary.json",
                "finding": "The programmed-separator plus direct perceptron update control starts at 100% but falls to 75% after three batch epochs. It is a negative control showing that an always-on target/anti-target update can perturb a barely calibrated separator out of the correct region.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v78_summary.json",
                "finding": "The programmed-separator plus lowtarget update control also starts at 100% but falls to 75% after three batch epochs. This reinforces that the simple low-score gate is not a reliable mistake/low-margin detector.",
            },
            {
                "artifact": "results/figures/device_xor2_mistake_gate_architecture.png",
                "finding": "Image-generated architecture diagram for the attempted mistake-gated dense-hidden XOR circuit: input rails feed dense hidden ReLU cells, readout scores feed score-lead latch nodes, and a mistake gate routes local positive/negative updates through capacitor-held gradient and weight states.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v88_summary.json",
                "finding": "The stronger score-lead mistake-gate diagnostic from random readout stayed at 50% XOR accuracy. The lead-latch tracker reported only an underflow-scale mean absolute lead difference, so the intended comparator gate did not actually drive the update path.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v89_summary.json",
                "finding": "The same mistake-gated rule from the calibrated programmed separator preserved 100% XOR accuracy, but the lead nodes again remained effectively zero. This is a negative circuit result for the current lead-latch topology, not learned correction.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v90_summary.json",
                "finding": "The first comparator-free local-loss run from random readout moved hidden-to-output readout weights by about 15.984 mV in one batch epoch, but still ended at 50% XOR accuracy.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v91_summary.json",
                "finding": "The comparator-free local-loss rule from the calibrated programmed separator preserved 100% XOR accuracy for one batch epoch, with final minimum margin reduced to 34.174 uV. This is preservation from a programmed readout, not discovery.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v92_summary.json",
                "finding": "The three-epoch comparator-free local-loss run from random readout moved hidden-to-output readout weights by 47.9314 mV total, but still ended at 50% XOR accuracy. The rule has usable electrical write strength, but its update direction still does not find the separating readout.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v93_summary.json",
                "finding": "A 5x readout-width local-loss control moved hidden-to-output readout weights by 239.087 mV total over three batch epochs and still ended at 50% XOR accuracy.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v94_summary.json",
                "finding": "A 10x readout-width local-loss control moved hidden-to-output readout weights by 477.778 mV total over three batch epochs and still ended at 50% XOR accuracy. This rules out insufficient write amplitude as the main issue for this local-loss formulation.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v102_summary.json",
                "finding": "The corrected output-senseamp mistake-gate diagnostic uses stored output activation caps rather than collapsed score nodes for winner detection. It tracks the winning output with about 47.238 mV mean lead separation, but batch-apply training still ends at 50% XOR accuracy.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v103_summary.json",
                "finding": "The same output-senseamp mistake gate with online per-sample applies over three epochs keeps lead tracking correct and moves readout weights by 49.968 mV total. It still ends at 50% XOR accuracy, but the worst margin improves from -7.068 mV to -6.05989 mV.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v104_summary.json",
                "finding": "An eight-epoch online output-senseamp mistake-gated run with stronger readout writes improves the nonliteral dense-hidden XOR circuit from 50% to 75% accuracy from random readout. Lead tracking remains correct with about 44.8 mV mean lead separation and readout weights move by 660.895 mV total. XOR is still not solved, but this is the first positive random-readout movement from the hardware mistake gate.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v116_summary.json",
                "finding": "A 16-epoch online output-senseamp mistake-gated run with 10x readout writes solves the dense nonliteral XOR device benchmark from random readout at 100% final accuracy. This primarily validates the hardware mistake-gated readout path; hidden writes remained near-frozen.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v119_backprop_score_e8_summary.json",
                "finding": "The random-hidden generator now exposes a real hidden backprop path using the same capacitor-held readout weight nodes that drive the forward readout synapses. With continuous score-error and real readout-weight transport, the eight-epoch run reached 75% XOR accuracy but did not solve the task.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v127_backprop_score_e1_hd64_meas_summary.json",
                "finding": "A 64x hidden-delta/hidden-gradient width probe with real readout-weight-transport backprop measured only microvolt-scale hidden gradient accumulator signals, identifying the hidden-gradient storage/apply cell as the next transistor-level bottleneck.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v140_backprop_score_e8_hgsense_w96_c20_hu001_summary.json",
                "finding": "A selectable hidden-gradient sense/write cell amplified the hidden-gradient differential before applying hidden-weight updates. With continuous score-error, real hidden backprop through the capacitor-held readout synapses, and a 10x smaller hidden update, the dense nonliteral XOR device benchmark reached 100% final accuracy with +5.15 mV final minimum margin and 56.8 mV total hidden signed-weight movement.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v141_backprop_score_e8_hgsense_refactor_gate_summary.json",
                "finding": "After the generator was refactored to support multiple datasets, the current `--dataset xor2` code path re-ran the same real-backprop hidden-gradient sense/write configuration and again reached 100% final XOR accuracy with +5.15 mV final minimum margin. This is the formal XOR >=95% gate before spending more work on moons or 8x8 MNIST scaling.",
            },
            {
                "artifact": "spice/results/device_moons8_random_hidden_v0_eval_smoke_summary.json",
                "finding": "The dense random-hidden device generator now supports a deterministic continuous two-moons dataset using the same x0/nx0/x1/nx1 rails. A zero-training moons8 ngspice eval completed and showed the measured nonliteral hidden activations are linearly separable, while the random readout starts at 50% accuracy.",
            },
            {
                "artifact": "spice/results/device_moons8_random_hidden_v3_score_e2_strong_readout_summary.json",
                "finding": "A two-epoch moons8 score-error run used real hidden backprop through readout synapse caps and the hidden-gradient sense/write cell. It moved readout weights by 474 mV and hidden weights by 2.92 mV, improved the worst final margin to -4.75 mV, but remained at 50% final accuracy. Mistake-gated and perceptron two-epoch moons variants timed out at 600 s, so they are runtime/stiffness issues rather than completed learning evidence.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v1_norm_eval_smoke_summary.json",
                "finding": "The same dense random-hidden device generator now supports a tiny binary MNIST bridge: four zeros and four ones are area-downsampled to 2x2 and mapped onto x0/nx0/x1/nx1 rails. The zero-training ngspice eval starts at 50% accuracy, but the measured hidden activations are linearly separable with +6.63 mV post-run perceptron margin.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v3_norm_score_e3_gentle_readout_summary.json",
                "finding": "A three-epoch interleaved MNIST01 score-error run used real hidden backprop through capacitor-held readout synapses and the hidden-gradient sense/write cell. It moved readout weights by 714 mV and hidden weights by 3.86 mV, but stayed at 50% accuracy with -29.8 mV final margin. This is a negative scaling result for the current continuous-sample readout/error update rule, not a hidden-representation failure.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v6_outcomp_e3_bias50m_summary.json",
                "finding": "An output-competitive error rule uses stored output activation capacitors as local analog gates for the competing-output update. On the tiny MNIST01 bridge it kept lead tracking correct and improved the worst final margin to -2.14 mV, but remained at 50% accuracy because the decision boundary crossed the zeros while making the ones slightly negative.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v142_flow_smoke_summary.json",
                "finding": "The dense random-hidden generator now supports direct backward/write flow mode: during bwd, error and hidden-delta conductance paths discharge capacitor-held weights directly, without gradient accumulator caps, gcmp, or a separate apply pulse. A one-epoch XOR smoke completed with nonzero readout and hidden-weight movement, but still ended at 50% accuracy.",
            },
            {
                "artifact": "spice/generated/device_mnist01_8_random_hidden_v10_flow_e1_smoke.cir",
                "finding": "The first tiny MNIST01 direct-flow smoke timed out after 650 s. This is evidence that the simpler flow architecture is wired, but it still needs stiffness/runtime tuning before it can replace the sampled accumulate/apply path on the MNIST bridge.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v146_flow_prestore_gate_e1_summary.json",
                "finding": "Adds a per-synapse pre-activation trace mode for direct backward/write flow. Each readout synapse captures its hidden activation and each hidden-input synapse captures its input/bias rail through MOS store paths during fwd; the backward/write stacks use those local trace capacitors instead of shared source nodes. The one-epoch XOR trace-gate check remained at 50% accuracy but produced comparable readout and hidden-weight movement to the shared-node flow run.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v147_flow_prestore_consume_e1_summary.json",
                "finding": "Adds a destructive pre-activation trace-consume mode, where local trace caps are discharged during bwd while they gate the direct write. The one-epoch XOR check remained at 50% accuracy and reduced total readout-weight movement to about 1.09 mV, showing that the current consume device drains the eligibility trace too aggressively.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v159_flow_readout_only_e32_w015_b002_summary.json",
                "finding": "A longer readout-only direct-flow XOR run solved the task without gradient accumulator caps, gcmp, or a separate apply pulse: flow_hidden_write=off, readout_update_width=0.15 u, 32 epochs, 100% final accuracy, and +1.64 mV final minimum margin. This proves the direct-flow write path can train the readout, but it is not end-to-end hidden-layer backprop.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v161_flow_prestore_gate_readout_only_e32_w015_b002_summary.json",
                "finding": "The same solved readout-only direct-flow XOR setting also works with per-synapse pre-activation trace capacitors gating the readout writes, reaching 100% final accuracy with +1.76 mV final minimum margin.",
            },
            {
                "artifact": "spice/generated/device_xor2_random_hidden_v160_flow_alllayer_e40_w01_h00001_b002.cir",
                "finding": "The first corresponding all-layer direct-flow run with hidden_delta/readout-weight-transport enabled and tiny hidden writes timed out after 900 s. This identified stiffness/runtime as the obstacle for direct-flow hidden writes.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v165_flow_alllayer_light_e32_w015_h000001_b002_summary.json",
                "finding": "After adding light measurement output and reducing hidden-write strength, the all-layer direct-flow XOR run completed with hidden writes enabled, no gradient accumulator caps, no separate apply pulse, real readout-weight transport into hidden deltas, 100% final accuracy, +2.00 mV final minimum margin, and 4.64 mV total hidden signed-weight movement. This is the first solved direct-flow hidden-backprop XOR result.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v166_flow_alllayer_gate_e32_w015_h000001_b002_summary.json",
                "finding": "The all-layer direct-flow XOR path also solved with per-synapse pre-activation trace capacitors gating the direct write, reaching 100% final accuracy with +3.69 mV final minimum margin and 35.1 mV total hidden signed-weight movement. This supports the local eligibility-trace architecture as a viable direct-flow variant.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v16_flow_readout_only_e8_w015_b002_summary.json",
                "finding": "Applying the stronger solved-XOR readout-only direct-flow setting to the tiny MNIST01 bridge reached 62.5% final accuracy and -18.6 mV final minimum margin, with class-1 samples strongly positive but several zero samples pushed negative. This is a negative transfer result, not a scaling solution.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v19_flow_readout_only_light_e10_w01_b0015_summary.json",
                "finding": "A gentler readout-only direct-flow tiny-MNIST01 setting improved the bridge to 75% final accuracy after 10 epochs, with -2.75 mV final minimum margin. This is the best direct-flow MNIST01 bridge result so far, but it still lacks hidden direct-flow writes and remains below a solved binary MNIST smoke.",
            },
            {
                "artifact": "spice/generated/device_mnist01_8_random_hidden_v22_flow_alllayer_light_e2_w01_h000001_b0015.cir",
                "finding": "The first short all-layer direct-flow MNIST01 smoke with hidden writes enabled failed electrically with a timestep-too-small error at hidden-delta internal node hdp_a61_0. This localized the continuous-sample stiffness to the hidden-delta transport chain.",
            },
            {
                "artifact": "spice/generated/device_mnist01_8_random_hidden_v23_flow_alllayer_light_e2_hd4_w01_h000001_b0015.cir",
                "finding": "Reducing hidden-delta width scale to 4 did not by itself remove the all-layer MNIST01 stiffness; the run still failed with a timestep-too-small error at hdp_a51_0.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v25_flow_alllayer_light_e2_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "Adding explicit hidden-delta internal damping (0.02 fF and 1 GOhm on transport-chain internal nodes) let the all-layer direct-flow MNIST01 out-competitive run complete with hidden writes enabled. It reached 75% final accuracy after 2 epochs, with -11.2 mV final minimum margin and 4.29 mV total hidden signed-weight movement, so damping fixes the transient but not the tiny MNIST bridge yet.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v26_flow_alllayer_light_e8_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "Extending the damped all-layer direct-flow MNIST01 bridge to 8 epochs solved the 8-sample binary task with hidden writes enabled: 100% final accuracy, +14.2 mV final minimum margin, real readout-weight transport for hidden deltas, no gradient accumulator caps, no separate apply phase, and 17.1 mV total hidden signed-weight movement. This is the first solved continuous-sample all-layer direct-flow bridge, but it is still far smaller than full MNIST.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v0_flow_alllayer_light_e8_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "Promoting the same damped all-layer direct-flow settings to the 12-sample binary MNIST bridge completed 96 train cycles and reached 91.67% final accuracy with hidden writes enabled. One digit-1 sample remained wrong at -12.9 mV margin; total hidden signed-weight movement was 25.9 mV. This is the first >90% continuous-sample all-layer direct-flow bridge, but it is still only a 12-sample binary task.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v167_transient_offsets_smoke_e0_summary.json",
                "finding": "The dense device harness now supports transient readout timing sweeps via --readout-sample-offsets-ns. The smoke run measured offsets 2.0, 2.5, 2.95, 3.5, and 4.05 ns and emitted readout_offset_stats plus best_final_transient_* fields, so accuracy can be evaluated at a spike/transient arrival time rather than only at the old fixed 2.95 ns sample point.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v2_flow_alllayer_light_e8_outcomp_damped_offsets_summary.json",
                "finding": "A transient timing sweep on the 12-sample all-layer direct-flow MNIST01 bridge showed that the one remaining error is not fixed by sampling the existing forward window differently: offsets 2.0, 2.5, 2.95, 3.5, 4.05, and 4.5 ns all stayed at 91.67% final accuracy, with the least-bad final minimum margin at 4.5 ns.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v1_flow_alllayer_light_e12_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "Extending the same damped all-layer direct-flow 12-sample MNIST01 run to 12 epochs did not fix the remaining error; final accuracy stayed at 91.67% and the worst margin worsened to -14.0 mV. This points to ordering/update-balance calibration rather than simply more epochs.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v3_flow_alllayer_interleave_light_e8_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "Interleaving the 12-sample binary MNIST stream instead of presenting all zeros before all ones did not solve the remaining sample at the original update strength; final accuracy stayed at 91.67% and the worst margin was -13.2 mV.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v4_flow_alllayer_interleave_light_e8_outcomp_damped_w008_h000001_b0015_summary.json",
                "finding": "A gentler interleaved readout update (0.08 u instead of 0.10 u) also stayed at 91.67%, though it slightly improved the worst margin to -12.8 mV. This suggests the next lever is the feature/readout balance or a stronger local margin trigger, not just ordering or epoch count.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v168_lead_or_bwd_smoke_e0_summary.json",
                "finding": "The device harness now has a self-timed backward-rail mode, --backward-gate-mode lead_or: Python provides a broad bwd_src window, but the effective bwd node is charged in-circuit only when bwd_src and an output lead-latch rail are high. A zero-training XOR smoke completed, proving the gated rail itself is electrically viable.",
            },
            {
                "artifact": "spice/generated/device_xor2_random_hidden_v169_flow_leador_alllayer_e32_w015_h000001_b002.cir",
                "finding": "The first undamped all-layer lead-gated backward training attempt failed with a timestep-too-small error at hidden-delta internal node hdn_a01_0. This mirrors the earlier MNIST hidden-delta stiffness and indicates the self-timed path should be evaluated with the same hidden-delta internal damping.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v170_flow_leador_alllayer_damped_e32_w015_h000001_b002_summary.json",
                "finding": "With hidden-delta damping enabled, the lead-gated self-timed backward rail solved all-layer XOR: 100% final accuracy, +3.09 mV final minimum margin, no gradient accumulator caps, no separate apply phase, and real readout-weight transport into hidden deltas.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v27_flow_alllayer_leador_e8_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "The same lead-gated self-timed backward rail solved the 8-sample MNIST01 all-layer direct-flow bridge with hidden writes enabled, reaching 100% final accuracy and +15.4 mV final minimum margin. This is the first solved bridge where the effective backward/write rail is opened by an in-circuit output event latch rather than directly by Python.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v5_flow_alllayer_leador_light_e8_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "Promoting the lead-gated self-timed backward rail to the 12-sample MNIST01 bridge did not fix the remaining ambiguous digit-1 sample: final accuracy stayed at 91.67% and the worst margin worsened to -18.0 mV. Self-timing improves the architecture but is not by itself the missing readout/feature-balance lever.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v6_flow_alllayer_outmistake_light_e8_damped_w01_h000001_b0015_summary.json",
                "finding": "The old output-latch mistake rule was a negative control on the 12-sample all-layer direct-flow MNIST01 bridge: it ended at 50% final accuracy with -49.4 mV margin. This exposed that the literal lead01/lead10 naming did not match the out_senseamp winner polarity.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v7_flow_alllayer_local_loss_light_e8_damped_w008_h000001_b0015_summary.json",
                "finding": "A local-loss style output rule was also a negative control for this bridge, ending at 50% final accuracy with -199 mV margin. It is not a useful replacement for the current out-competitive update in this transistor-level flow deck.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v8_flow_alllayer_outcomp_light_e8_damped_w006_h000002_b0015_summary.json",
                "finding": "Changing the out-competitive balance to lower readout writes and stronger hidden writes preserved 91.67% final accuracy but worsened the worst margin to -15.8 mV, so the existing 0.10 u readout / 1e-6 u hidden setting remains the better 12-sample baseline.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v28_flow_alllayer_latchmistake_light_e8_damped_w01_h000001_b0015_summary.json",
                "finding": "After correcting the out_senseamp lead-latch polarity, the new out_latch_mistake rule still solved the 8-sample MNIST01 bridge with 100% final accuracy and +10.9 mV margin. This validates the corrected latch polarity on the solved bridge.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v9_flow_alllayer_latchmistake_light_e8_damped_w01_h000001_b0015_summary.json",
                "finding": "The corrected latch-mistake rule at 8 epochs reduced the stubborn digit-1 error from -12.9 mV to -7.11 mV, but broke one marginal digit-0 sample and ended at 83.33% final accuracy. The latch signal is useful as a margin assist, but too blunt as the only output update rule.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v10_flow_alllayer_light_e8_outcomp_damped_wide_offsets_summary.json",
                "finding": "A wider transient readout sweep over 0.5 ns through 5.5 ns confirms that steady-state correctness is not required by the scoring harness, but it still does not rescue the 12-sample bridge. Very early samples are unstable; every offset from 1.0 ns onward remains at 91.67%, with the least-bad late margin about -12.9 mV.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v11_flow_alllayer_leador_light_e8_outcomp_damped_wide_offsets_summary.json",
                "finding": "Repeating the wide transient sweep with the lead-gated self-timed backward rail again stayed at 91.67% for offsets from 1.0 ns onward, with a worse late margin around -18.0 mV. Self-timing remains architecturally useful but is not the missing correction for this ambiguous sample.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v12_flow_alllayer_latchmistake_light_e12_damped_w01_h000001_b0015_summary.json",
                "finding": "Extending the corrected latch-mistake rule to 12 epochs recovered the 12-sample bridge to 91.67% final accuracy and left only the original digit-1 sample wrong, with an improved but still negative -8.15 mV margin. This points to a softer latch/margin boost or feature/readout balance experiment next.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v29_flow_alllayer_latchboost32_light_e8_damped_w01_h000001_b0015_summary.json",
                "finding": "The new out_competitive_latchboost rule keeps the base out-competitive update and adds a weaker corrected-latch path in parallel. With a 32 u latch boost it preserved the solved 8-sample MNIST01 bridge at 100% final accuracy and +13.5 mV margin.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v30_flow_alllayer_latchboost64_light_e8_damped_w01_h000001_b0015_summary.json",
                "finding": "Increasing the latch boost to 64 u also preserved the solved 8-sample MNIST01 bridge at 100% final accuracy, with +13.8 mV margin. The hybrid latch path is electrically safe on the solved bridge.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v13_flow_alllayer_latchboost32_light_e8_damped_w01_h000001_b0015_summary.json",
                "finding": "The same hybrid latch-boost rule did not move the 12-sample bridge at 32 u: final accuracy stayed at 91.67% and the worst margin was -13.0 mV, slightly worse than the original out-competitive baseline.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v14_flow_alllayer_latchboost64_light_e8_damped_w01_h000001_b0015_summary.json",
                "finding": "The 64 u latch-boost variant also stayed at 91.67% on the 12-sample bridge, with -12.9 mV worst margin. This makes latch boosting a safe architecture variant, but not a scaling fix for the ambiguous digit-1 sample.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v31_flow_alllayer_tracegate_e8_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "A per-synapse pre-activation trace version of all-layer direct-flow training now solves the 8-sample MNIST01 bridge: every synapse captures its own pre activity into a local capacitor during forward and uses that local trace capacitor during backward/write, reaching 100% final accuracy and +14.2 mV margin.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v15_flow_alllayer_tracegate_e8_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "Promoting the per-synapse trace-gate architecture to the 12-sample bridge stayed at 91.67% with -13.0 mV margin. Local saved pre-activations are viable and closer to the intended synapse-local architecture, but they do not by themselves fix the remaining output/feature calibration gap.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v16_csvsep_initial_s002_e0_summary.json",
                "finding": "The random-hidden generator now supports --readout-init csv_separator, fitting a perceptron to measured hidden activation columns from a prior SPICE CSV and mapping it into readout capacitor ICs. On the 12-sample bridge, scale 0.02 kept zero-training MOS readout accuracy at 50%, despite the hidden activations being linearly separable.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v18_csvsep_initial_s004_offm008_e0_summary.json",
                "finding": "Tuning the CSV-derived separator to scale 0.04 with -80 mV output-bias offset improved the zero-training programmed-readout diagnostic only to 66.67%; all ones were correct, but four zeros were barely wrong at sub-millivolt margins. The linear hidden separator does not map cleanly into the current nonlinear MOS readout parameterization.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v21_csvsep_initial_s008_offm016_e0_summary.json",
                "finding": "Doubling the CSV-derived separator scale to 0.08 and canceling the larger learned bias again reached only 66.67% zero-training accuracy. This reinforces that the remaining small-bridge gap is a readout-cell/parameterization issue, not just absence of a linear separator in hidden activation space.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v24_csvsep_initial_s004_offm0079_e0_summary.json",
                "finding": "A finer offset check around the best scale-0.04 programmed separator did not expose a calibrated readout pocket: -79 mV, -78 mV, and -77 mV offset runs all fell back to 50% zero-training accuracy, with the best of the three still at -0.361 mV worst margin.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v28_csvsep_initial_s004_offm008_c042_e0_summary.json",
                "finding": "The separator initializer now exposes --readout-center-v. Lowering the differential separator common-mode to 0.35, 0.42, or 0.50 V did not solve the 12-sample bridge; the best runs still reached only 66.67% zero-training accuracy.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v38_csvrect_initial_s004_offm003_c012_e0_summary.json",
                "finding": "A new csv_rectified_separator mode maps the fitted separator into one-sided positive/negative readout branches. With readout_center_v=0.12 and separator_offset_v from -30 mV to 0 V, the zero-training readout reached 91.67% and fixed the previously hard digit-1 sample, but one digit-0 sample remained wrong.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v42_csvrect_initial_s004_offm003_c012_offsets_e0_summary.json",
                "finding": "Sampling the rectified readout at multiple transient offsets from 1.0 ns through 5.0 ns did not expose a 100% spiking/readout instant; the best offset still stayed at 91.67%. Two- and eight-epoch trained rectified starts then timed out at the current transient size.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v51_csvthresh_initial_s004_thr008_c012_e0_summary.json",
                "finding": "A thresholded CSV-separator initializer was a negative control: the best scale-0.04, threshold-80 mV zero-training readout stayed at 91.67% and had a large -37.6 mV worst margin, so thresholding the old gate-stack readout does not fix the separator-to-device mismatch.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v54_passact_csvrect_s004_off000_c012_e0_summary.json",
                "finding": "Adding split_signed_passact_v1, where the positive readout branch uses the stored hidden activation voltage as the pass-device source while keeping capacitor-held signed weights and readout-weight transport, cleanly expressed the rectified CSV separator in SPICE: the 12-sample bridge reached 100% zero-training accuracy with +14.8 mV final margin.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v58_passact_csvrect_s004_off000_c012_offsets_e0_summary.json",
                "finding": "The pass-activation readout was robust across transient readout timing: offsets from 1.0 ns through 5.0 ns all stayed at 100% final accuracy, with the best measured final margin +17.6 mV at 5.0 ns. This is a structural readout fix, not a one-instant spike artifact.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v59_passact_csvrect_train_s004_off000_c012_e1_w002_b0005_summary.json",
                "finding": "Starting from the programmed pass-activation separator, one short all-layer training epoch preserved 100% accuracy but degraded the margin from +14.8 mV to +4.94 mV while moving readout weights by 13.2 mV, output biases by 3.34 mV, and hidden weights by 3.27 mV.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v60_passact_csvrect_train_s004_off000_c012_e2_w001_b00025_summary.json",
                "finding": "Two gentler programmed-start pass-activation epochs drifted out of the correct pocket, ending at 91.67% and -1.35 mV margin. The readout cell now has enough expressivity, but the direct-flow update rule still needs margin/mistake gating or better local stabilization.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v32_passact_train_e8_outcomp_damped_w01_h000001_b0015_summary.json",
                "finding": "Training split_signed_passact_v1 from random readout on the otherwise solved 8-sample MNIST01 bridge stayed at 50% despite 298 mV of readout movement and 17.1 mV of hidden movement. The structural readout fix does not by itself solve learning from random initialization.",
            },
            {
                "artifact": "spice/run_device_xor2_random_hidden.py",
                "finding": "Adds --backward-gate-mode target_mistake: the scheduled Python waveform now drives bwd_src, while the actual bwd capacitor charges only through target-label and output-sense-latch transistor stacks. A PMOS inhibit from the target's winning lead suppresses ambiguous both-high latch cases.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v66_passact_csvrect_targetmistake_e1_gatecheck_summary.json",
                "finding": "The first loose target-mistake gate preserved 100% programmed-start pass-activation accuracy, but a correctly classified both-high sense-latch sample still opened bwd to about 1.04 V and degraded the margin to +13.2 mV. This identified the need for an explicit winning-lead inhibit.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v67_passact_csvrect_targetmistake_strict_e1_gatecheck_summary.json",
                "finding": "The strict target-mistake gate held the programmed-start train bwd rail to millivolt leakage (6.38 mV max), preserved 100% accuracy, and ended at +14.9 mV margin after one epoch. This validates circuit-level no-update-if-correct gating for the pass-activation readout.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v68_passact_csvrect_targetmistake_strict_e2_gatecheck_summary.json",
                "finding": "The strict target-mistake gate also preserved the programmed 12-sample separator for two epochs: 100% final accuracy, +14.9 mV margin, only 0.119 mV total readout movement, and max train bwd about 6.39 mV.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v36_passact_random_targetmistake_strict_e2_w01_b0015_summary.json",
                "finding": "From random readout on the 8-sample bridge, strict target-mistake direct flow fired on mistakes and improved the worst margin from -0.803 mV to -0.515 mV over two epochs, but accuracy stayed at 50%. The new gate fixes destructive updates on correct programmed starts, not random-readout learning.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v38_passact_random_targetmistake_strict_e2_cap1f_summary.json",
                "finding": "A persistent weight-capacitance sweep showed useful charge-level sensitivity: lowering the hidden/readout/bias storage caps from 4 fF to 1 fF improved the 8-sample random-start worst margin from -0.515 mV to -0.208 mV after two epochs, though accuracy stayed at 50%.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v42_passact_random_targetmistake_strict_e2_cap05f_summary.json",
                "finding": "Pushing persistent storage lower to 0.5 fF improved the same random-start worst margin further to -0.156 mV after two epochs, with larger readout/hidden movement. This made lower charge storage the main charge-level lever to keep sweeping.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v43_passact_random_targetmistake_strict_e4_cap1f_summary.json",
                "finding": "Extending the 1 fF charge-level run from two to four epochs stalled at -0.203 mV worst margin and 50% accuracy, essentially tied with the two-epoch 1 fF result. Repetition alone is not enough; the update/error rule still needs a better bias or noise source.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v39_passact_random_targetmistake_strict_e2_cap8f_summary.json",
                "finding": "Increasing persistent weight capacitance to 8 fF weakened training movement and regressed the same random-start bridge to -0.642 mV worst margin, reinforcing that lower charge storage currently helps more than higher charge storage.",
            },
            {
                "artifact": "spice/run_device_xor2_random_hidden.py",
                "finding": "Adds --cap-dither-v, --cap-dither-seed, and --cap-dither-scope to perturb hidden/readout persistent capacitor initial charges deterministically. This is an initial charge-offset/noise proxy, not true transient thermal noise.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v40_passact_random_targetmistake_strict_e2_dither_readout20mv_summary.json",
                "finding": "A 20 mV readout-only initial charge dither made the 8-sample random-start initial margin worse (-1.30 mV) and still ended at 50% with -1.07 mV margin, so this dither seed/amplitude did not help training.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v41_passact_random_targetmistake_strict_e2_dither_all20mv_summary.json",
                "finding": "A 20 mV all-persistent-cap initial charge dither also stayed at 50% and ended at -0.986 mV worst margin. Initial charge disorder has not helped in this first seed; lower capacitance is the more promising charge-level knob so far.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v47_passact_targetmistake_cap025f_no_noise_summary.json",
                "finding": "Lowering the hidden activation/storage capacitance to 0.25 fF with strict target-mistake direct flow improved the random-start 8-sample bridge to 75% after two epochs, with final margin -41.1 uV.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v48_passact_targetmistake_cap025f_e4_no_noise_summary.json",
                "finding": "Continuing the 0.25 fF point to four epochs reached 87.5% and narrowed the worst margin to -16.4 uV, but still did not fully solve the bridge.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v49_passact_targetmistake_cap0125f_e2_no_noise_summary.json",
                "finding": "The lower 0.125 fF point reached 87.5% after two epochs but with a worse -338 uV margin, showing that the charge-level benefit is not monotonic.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v50_passact_targetmistake_cap02f_e2_no_noise_summary.json",
                "finding": "The 0.20 fF point solved the random-start 8-sample MNIST01 bridge in ngspice with strict target-mistake direct flow: 100% final accuracy, +27.2 uV final margin, no gradient accumulator caps, no separate apply pulse, and real readout-weight transport into hidden deltas.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v52_passact_targetmistake_cap02f_tracegate_e2_no_noise_summary.json",
                "finding": "The stricter local pre-activation trace-cap version of the 0.20 fF setting also solved the 8-sample bridge at 100% with +9.97 uV margin. Each readout and hidden-input synapse captures its pre activity through MOS store paths during fwd and uses that local trace capacitor during backward/write.",
            },
            {
                "artifact": "spice/run_device_xor2_random_hidden.py",
                "finding": "Adds counted dataset variants such as mnist01_16 and moons16, --init-seed for separating dataset selection from capacitor initialization, and --hidden-cells for scaling the dense random-hidden device bridge without editing global constants.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v69_passact_targetmistake_cap02f_tracegate_e2_summary.json",
                "finding": "Promoting the strict trace-cap target-mistake direct-flow bridge from 8 to 12 MNIST01 samples reached 91.67% after two epochs at 0.20 fF, with final margin -152 uV. This keeps the architecture constraints from the 8-sample solve but leaves one label-1 sample wrong.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v70_passact_targetmistake_cap02f_tracegate_e4_summary.json",
                "finding": "Repeating the same 0.20 fF 12-sample trace-cap setting for four epochs did not fix the remaining sample: final accuracy stayed at 91.67% with final margin -153 uV, so repetition alone is not enough.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v72_passact_targetmistake_cap022f_tracegate_e2_summary.json",
                "finding": "A small charge-level sweep on the 12-sample trace-cap bridge found 0.22 fF was the best tested point: 91.67% final accuracy with -120 uV margin, better than 0.20 fF (-152 uV) and 0.18 fF (-213 uV) but still one sample short.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v74_passact_targetmistake_cap022f_tracegate_e2_init5_summary.json",
                "finding": "The first separated initialization-seed sweep at the 0.22 fF 12-sample point was sensitive: init seed 5 reached 83.33% with -115 uV margin, while neighboring init seeds 4 and 6 fell to 50%. The newer summaries explicitly mark mos_store_trace_caps, hidden-delta activation gating, and direct weight-write paths.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v76_passact_targetmistake_cap022f_tracegate_h8_offsets_e2_summary.json",
                "finding": "A multi-offset readout check at the best 8-hidden 0.22 fF 12-sample setting stayed at 91.67% for every sampled output time from 2.0 ns to 4.5 ns; the best margin was still -120 uV, so the remaining error is not just a transient readout-timing issue.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v77_passact_targetmistake_cap022f_tracegate_h12_e2_summary.json",
                "finding": "Increasing the strict trace-cap bridge to 12 hidden cells without retuning collapsed to 50% final accuracy. All digit-1 samples were correct, including the hard pattern 8, but all digit-0 samples were wrong, indicating a class-balance/readout-bias shift rather than a clean capacity win.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v78_passact_targetmistake_cap022f_tracegate_h16_e2_summary.json",
                "finding": "The 16-hidden version showed the same failure mode as 12 hidden cells: 50% final accuracy, all ones correct, all zeros wrong, and final margin -413 uV. Blind hidden-count scaling is not enough for the current direct-flow rule.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v79_passact_targetmistake_cap022f_tracegate_h8_readoutonly_e2_summary.json",
                "finding": "Disabling hidden writes on the 8-hidden 12-sample trace-cap bridge reached 83.33% with -127 uV margin. The all-layer hidden-write path is still beneficial for the current best 91.67% setting.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v80_passact_targetmistake_cap022f_tracegate_h8_hiddenweak_e2_summary.json",
                "finding": "Weakening hidden writes by 5x reached only 75% despite a slightly less negative worst margin of -97.3 uV, so simply damping hidden writes does not solve the 12-sample bridge.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v81_passact_targetmistake_cap022f_tracegate_h8_biasweak_e2_summary.json",
                "finding": "Weakening the output-bias write by 3x collapsed the same 8-hidden 12-sample setup to 50% with -338 uV margin. The output-bias path is not merely too strong; the next retune should focus on asymmetric class/readout balance.",
            },
            {
                "artifact": "spice/run_device_xor2_random_hidden.py",
                "finding": "Adds --output-bias-offset-v, a signed initial-condition perturbation for differential output-bias capacitor nodes. This is a circuit-state initialization knob, not behavioral postprocessing.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v83_passact_targetmistake_cap022f_tracegate_h12_biasp00050_e2_summary.json",
                "finding": "Tiny positive output-bias offsets did not fix the 12-hidden class-balance failure: +0.25 mV and +0.5 mV both stayed at 50% final accuracy, with every digit-1 sample correct and every digit-0 sample wrong.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v85_passact_targetmistake_cap022f_tracegate_h8_biasm00010_e2_summary.json",
                "finding": "A tiny -0.1 mV output-bias offset on the 8-hidden best branch preserved 91.67% final accuracy and moved the worst margin only slightly, from about -120 uV to -116 uV. Static bias offset is too weak to solve the remaining label-1 sample.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v89_passact_targetmistake_cap022f_tracegate_h8_biasm020_e2_summary.json",
                "finding": "Larger static output-bias offsets were actively harmful: +20/+50 mV checks on wider hidden counts stayed at 50%, and a -20 mV 8-hidden check collapsed to the same all-zeros-wrong mode as the wider branches.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v90_passact_targetmistake_cap022f_tracegate_h8_rw02_e2_summary.json",
                "finding": "Doubling readout update width to 0.2 u on the 8-hidden branch did not improve the 12-sample bridge: final accuracy stayed at 91.67% and the worst margin stayed essentially unchanged at -120 uV.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v91_passact_targetmistake_cap022f_tracegate_h12_rw02_e2_summary.json",
                "finding": "The same doubled readout update width on the 12-hidden branch stayed at 50% final accuracy, with all digit-1 samples correct and all digit-0 samples wrong. Raw readout write amplitude is not the h12 fix.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v92_passact_targetmistake_cap022f_tracegate_h12_e4_summary.json",
                "finding": "Running the 12-hidden baseline for four epochs also stayed at 50%, again with all zeros wrong and all ones correct. The h12/h16 blocker is dynamic output-error/update balance, not simply epoch count.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v94_passact_targetmistake_cap024f_tracegate_h8_e2_summary.json",
                "finding": "Extending the 8-hidden charge sweep to 0.24 fF preserved 91.67% final accuracy and improved the worst margin to -106 uV, with only the hard label-1 pattern 8 still wrong.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v98_passact_targetmistake_cap028f_tracegate_h8_e2_summary.json",
                "finding": "Pushing the same charge sweep to 0.26 and 0.28 fF improved the hard-pattern margin further, but made one digit-0 sample wrong and dropped accuracy to 83.33%. The 8-hidden branch has a narrow charge/balance ridge rather than a monotonic fix.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v95_passact_latchmistake_cap022f_tracegate_h12_e2_summary.json",
                "finding": "The corrected latch-mistake rule did not fix the 12-hidden pass-activation/target-mistake branch: final accuracy stayed at 50%, with all digit-1 samples correct and all digit-0 samples wrong.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v96_passact_targetmistake_cap022f_tracegate_h12_fullprobe_e1_summary.json",
                "finding": "A one-epoch full-measure h12 probe showed substantial readout/output-bias movement and millivolt-scale hidden weight-cap movement, but sampled hidden-delta nodes were essentially zero. This suggests some measured hidden movement may be weight-gate feedthrough rather than useful backprop current.",
            },
            {
                "artifact": "spice/run_device_xor2_random_hidden.py",
                "finding": "Adds full-measure diagnostics for output-delta nodes and hidden-delta nodes both at the backward sample and during the direct write window, and fixes generated comments/summaries to report the actual hidden-cell count. The next full probe can distinguish real hidden backprop current from parasitic weight-cap motion.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v172_diag_absnode_smoke_summary.json",
                "finding": "A short full-measure smoke validates the new diagnostics. It reports output-delta node magnitude around 64 mV but hidden-delta node magnitude around 1.35e-10 V while hidden weight caps still move by millivolts, strengthening the feedthrough-vs-backprop concern.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v174_diag_hdelta_nogate_summary.json",
                "finding": "A follow-up raw hidden-delta transport probe showed that the path is mostly common-mode: output-delta nodes were about 64 mV, hdp/hdn reached about 36 mV common-mode, but the useful differential hidden-delta signal was only about 34 uV even with the activation gate removed.",
            },
            {
                "artifact": "spice/run_device_xor2_random_hidden.py",
                "finding": "Adds --hidden-delta-weight-device, --hidden-delta-cap-f, and --hidden-delta-output-mode senseamp. These are circuit-topology knobs: the hidden-delta path still flows through output-error caps, capacitor-held readout weights, and optional activation gating before any local latch amplifies it.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v180_diag_hdelta_senseamp_nogate_summary.json",
                "finding": "The new hidden-delta senseamp converted a raw hidden-delta differential of only about 34 uV into about 1.2 V of local hdpg/hdng write-gate separation in ngspice. This proves a transistor-level local amplification option for the real readout-weight-transport path, though the one-epoch four-hidden XOR run itself stayed at 50%.",
            },
            {
                "artifact": "spice/results/device_xor2_random_hidden_v181_senseamp_h8_e8_summary.json",
                "finding": "Using the local hidden-delta senseamp on the eight-hidden direct-flow XOR setting solved XOR after 8 epochs: 100% final accuracy, +3.76 mV final margin, hidden writes enabled, real readout-weight transport into hidden deltas, no gradient accumulator caps, and no behavioral gradient math.",
            },
            {
                "artifact": "spice/results/device_mnist01_12_random_hidden_v99_passact_targetmistake_h8_cap024_hdnsense_e2_summary.json",
                "finding": "Repeating the 8-hidden 0.24 fF 12-sample MNIST01 bridge with the stronger activation-gated hidden-delta probe setting preserved 91.67% final accuracy and -106 uV worst margin, confirming the prior narrow charge/balance ridge rather than solving the remaining sample.",
            },
            {
                "artifact": "spice/run_device_xor2_random_hidden.py",
                "finding": "Adds --train-charge-noise-* controls for transistor-gated stochastic charge bleed during training. Python only seeds the random pulse rails; actual perturbation current must pass through bwd and a MOS discharge path into a selected signed weight-cap branch.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v44_passact_targetmistake_cap05f_noise_readout_w002_p035_summary.json",
                "finding": "Training-time readout charge noise at 0.5 fF, width 0.002 u, probability 0.35 stayed at 50% and worsened the final margin to -50.1 mV, so this stochastic charge bleed setting is far too destructive.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v45_passact_targetmistake_cap05f_noise_readout_w005_p035_summary.json",
                "finding": "Increasing the same readout noise width to 0.005 u worsened the margin further to -73.6 mV at 50% accuracy.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v46_passact_targetmistake_cap05f_noise_all_w002_p020_summary.json",
                "finding": "All-scope charge noise at 0.5 fF, width 0.002 u, probability 0.20 stayed at 50% and ended at -27.0 mV, with excessive hidden-weight movement.",
            },
            {
                "artifact": "spice/results/device_mnist01_8_random_hidden_v51_passact_targetmistake_cap025f_noise_readout_w0002_p010_summary.json",
                "finding": "A much gentler 0.25 fF readout-noise setting (0.0002 u, probability 0.10, 0.10 ns pulse) still regressed to 75% with -18.4 mV margin. Charge level helps; explicit random charge bleed has not helped so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_train_local_audit_8x8_80_e3_lr80k_summary.json",
                "finding": "Monolithic transient all-SPICE training completed for 8x8, 80 train / 80 held-out, reaching 75% held-out.",
            },
            {
                "artifact": "spice/generated/spice_mnist_train_local_audit_8x8_200_e3_lr80k.cir",
                "finding": "Monolithic transient all-SPICE training for 8x8, 200 train / 200 held-out timed out after 300 s.",
            },
            {
                "artifact": "spice/results/spice_mnist_stream_stream_8x8_100_e2_lr80k_summary.json",
                "finding": "Chunked all-SPICE training completed for 8x8, 100 train / 100 held-out, but took about 329 s and reached 42% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_op_op_8x8_200_e5_lr004_summary.json",
                "finding": "Per-sample operating-point training completed for 8x8, 200 train / 200 held-out, reaching 70% held-out but requiring one ngspice launch per sample/update.",
            },
            {
                "artifact": "spice/results/spice_mnist_batch_op_batch_op_8x8_200_e20_lr03_b50_summary.json",
                "finding": "Batch operating-point training completed for 8x8, 200 train / 200 held-out, reaching 68.5% best held-out in about 428 s; fastest scalable all-SPICE training path so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_8x8_b4_200_e20_lr02_summary.json",
                "finding": "First local block-evidence batch-op all-SPICE training run completed for 8x8, 200 train / 200 held-out, reaching 68% held-out in about 398 s.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e5_lr02_summary.json",
                "finding": "Same local block architecture at 14x14 resolution with four 7x7 blocks reached 66% after 5 epochs, but required about 561 s.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_cont3_lr02_b50_summary.json",
                "finding": "Resuming the 14x14 local block run from saved programmable weights for 3 more epochs reached 71% held-out, the best local all-SPICE batch-op result so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_cont5_lr02_b50_summary.json",
                "finding": "Two further resumed epochs from the 71% checkpoint fell back to 70%, suggesting this small 200-sample setup is near a plateau.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_best_robustness_summary.json",
                "finding": "Eval-only robustness check of the 71% 14x14 checkpoint matched 71% clean accuracy and held about 70.25% mean accuracy under normalized perturbation sigma 0.01.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_eval1000_summary.json",
                "finding": "The 71% 14x14 checkpoint evaluated at 73.3% on a larger 1,000-sample held-out subset.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_1000_e1_from200_lr01_summary.json",
                "finding": "Continuing the 14x14 checkpoint for one epoch on 1,000 train / 1,000 held-out samples reached 76.4% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_1000_e2_from200_lr01_summary.json",
                "finding": "A second 1,000-sample continuation epoch reached 81.0% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_1000_e3_from200_lr01_summary.json",
                "finding": "A third 1,000-sample continuation epoch reached 84.1% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_1000_e4_from200_lr01_summary.json",
                "finding": "A fourth 1,000-sample continuation epoch reached 85.0% held-out, the best local all-SPICE result so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_1000_e4_robust01_summary.json",
                "finding": "Eval-only robustness check of the 85% checkpoint reproduced 85.0% clean accuracy and reached 84.8% under one normalized perturbation sigma 0.01 draw.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e1_from1000e4_lr01_summary.json",
                "finding": "Continuing the 85% checkpoint for one epoch on 2,000 train / 1,000 held-out samples reached 86.0% held-out, the best local all-SPICE result so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e2_from1000e4_lr01_summary.json",
                "finding": "A second 2,000-sample continuation epoch reached 86.9% held-out, the best local all-SPICE result so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e3_frome2_lr005_b100_summary.json",
                "finding": "Continuing the 86.9% checkpoint for one more 2,000-train / 1,000-held-out epoch with batch size 100 and lr 0.05 reached 87.5% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e4_frome3_lr005_b100_summary.json",
                "finding": "A second conservative continuation from the 87.5% checkpoint reached 87.7% held-out, the best saved local all-SPICE subset result so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e5_frome4_traingains_lr001_b100_summary.json",
                "finding": "Continuing the 87.7% checkpoint for one more all-SPICE epoch with trainable per-class/per-block gains reached 87.9% held-out, the new best local all-SPICE subset result.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e6_frome5_traingains_lr0005_b100_summary.json",
                "finding": "A second conservative trainable-gain continuation at learning rate 0.005 stayed at 87.9% held-out, suggesting this four-block gain-tuned path is plateauing below 90%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_center_expand_eval1000_summary.json",
                "finding": "Expanding the 87.9% four-block checkpoint to a five-block model with one centered overlapping block reproduced 87.9% on the 1,000-image held-out set, confirming the added block starts neutral.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_center_2000_e1_from4block_lr005_b100_summary.json",
                "finding": "Training the five-block centered-overlap model for one 2,000-train / 1,000-held-out epoch at learning rate 0.005 stayed at 87.9%, so the smallest overlap-capacity increase did not break the plateau.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_mc_14x14_b7_c2_expand_eval1000_summary.json",
                "finding": "Expanding the 87.9% four-block checkpoint to a two-channel local-template model reproduced 87.9%, confirming the added template channel starts neutral.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_mc_14x14_b7_c2_2000_e1_from4block_lr005_b100_summary.json",
                "finding": "Training the two-channel local-template model for one 2,000-train / 1,000-held-out epoch reached 88.0%, the new best all-SPICE subset result, but required about 2,096 s.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_mc_14x14_b7_c2_2000_e2_frome1_lr005_b100_summary.json",
                "finding": "A second two-channel continuation stayed at 88.0%, suggesting the channel-expanded path also plateaus quickly while doubling runtime.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_mc_14x14_b7_c2_2000_e2_eval1000_classchunk1_summary.json",
                "finding": "Class-chunked eval of the two-channel 88.0% checkpoint reproduced 88.0% and reduced eval time from about 187 s to about 128 s.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_mc_14x14_b7_c2_2000_e3_frome2_lr005_b100_classchunk1_summary.json",
                "finding": "Class-chunked two-channel continuation preserved 88.0% while reducing one epoch from about 2,102 s to about 1,058 s; useful for scaling but not an accuracy breakthrough.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_mc_14x14_b7_c2_4000_e1_from2000e3_lr003_b100_classchunk1_summary.json",
                "finding": "Continuing the class-chunked two-channel checkpoint for one 4,000-train / 1,000-held-out epoch at lr 0.003 dropped to 87.2%, so data scaling alone did not beat the 88.0% subset plateau.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_mc_14x14_b7_c3_expand_from_c2_eval1000_classchunk1_summary.json",
                "finding": "Expanding the 88.0% two-channel checkpoint to three local-template channels reproduced 88.0%, confirming lower-to-higher channel expansion starts neutral.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_mc_14x14_b7_c3_2000_e1_from_c2_lr005_b100_classchunk1_summary.json",
                "finding": "One class-chunked three-channel continuation stayed at 88.0%, so extra within-block template capacity alone did not break the plateau.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e7_softmax_frome6_traingains_lr0001_b100_summary.json",
                "finding": "Continuing the 87.9% checkpoint for one epoch with SPICE-computed softmax class-competition updates at learning rate 0.001 also stayed at 87.9%, so changing the output loss/update signal alone did not break the plateau.",
            },
            {
                "artifact": "spice/run_spice_mnist_local_ensemble_eval.py",
                "finding": "Adds an eval-only local-block ensemble diagnostic where each branch computes local nonlinear evidence inside ngspice and branch scores are summed by SPICE behavioral nodes.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_ensemble_tanh879_control_eval1000_split2000_summary.json",
                "finding": "The local ensemble evaluator's one-branch control reproduced the known 87.9% accuracy of the best tanh local-block checkpoint on the same 2,000/1,000 RNG split.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_ensemble_tanh879_plus_diffrelu809_eval1000_split2000_summary.json",
                "finding": "Combining the 87.9% tanh branch with the 80.9% differential-clipped-ReLU softmax branch by summing scores inside ngspice reached only 86.7%, worse than the best branch alone.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_ensemble_tanh879_plus_diffrelu809_sweep_split2000_summary.json",
                "finding": "Sweeping the rectified branch gain from 0 to 1 using SPICE-computed branch scores found the best gain was 0.0; any positive contribution reduced accuracy below the 87.9% tanh branch.",
            },
            {
                "artifact": "results/tables/spice_local_ensemble_comparison.csv",
                "finding": "Tabulates local ensemble SPICE diagnostics.",
            },
            {
                "artifact": "spice/run_spice_mnist_local_readout_calibrator.py",
                "finding": "Adds a frozen-local-evidence plus trainable 10x10 class mixer experiment; ngspice computes the frozen local evidence, softmax readout error, and programmable mixer updates.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_readout_cal_eval_14x14_b7_2000_1000_identity_summary.json",
                "finding": "Eval-only readout calibrator reproduced the known 87.9% local-block checkpoint on the 2,000/1,000 split.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_readout_cal_cal_14x14_b7_2000_1000_e1_lr002_summary.json",
                "finding": "One conservative trainable 10x10 mixer epoch with identity scale 1.0 reduced accuracy to 87.7%, so naive readout calibration did not help.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_readout_cal_cal_14x14_b7_2000_1000_e1_lr002_id4_summary.json",
                "finding": "A stronger identity-scale mixer preserved 87.9% but still did not improve it, suggesting the current plateau is not only a simple 10-class readout calibration issue.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_readout_cal_localfeat_14x14_b7_2000_1000_e1_lr002_summary.json",
                "finding": "Exposing all 40 class/block local evidence features to a trainable 10x40 mixer also reduced accuracy to 87.7%, so the plateau is not fixed by a small programmable readout over frozen local features.",
            },
            {
                "artifact": "results/tables/spice_local_readout_calibration.csv",
                "finding": "Tabulates frozen-local-block trainable-readout calibration checks.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_2000_e4_robust01_b100_summary.json",
                "finding": "Eval-only robustness check of the 87.7% checkpoint reproduced 87.7% clean accuracy and reached 87.7% under one combined normalized perturbation sigma 0.01 draw.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_4000_e1_from2000e4_lr003_b100_summary.json",
                "finding": "Continuing the 87.7% checkpoint for one 4,000-train / 1,000-held-out epoch with lr 0.03 reached 87.5%, so extra samples alone did not beat the current best four-block local checkpoint.",
            },
            {
                "artifact": "spice/generated/spice_mnist_local_block_local_block_14x14_b7_s3_200_e2_lr02_b100_step.cir",
                "finding": "Overlapping 9-block stride-3 local variant with batch size 100 timed out after 90 s on the first ngspice training solve.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_s3_200_e1_lr02_b50_summary.json",
                "finding": "The same overlapping 9-block stride-3 variant with batch size 50 completed one 200/200 epoch in about 272 s and reached only 21% held-out, so naive overlap is currently too slow and weak.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_28x28_b14_100_e1_lr02_b25_summary.json",
                "finding": "Full-resolution local feasibility run with 28x28 inputs and four non-overlapping 14x14 blocks completed one 100/100 all-SPICE epoch, reaching 32% held-out in about 538 s; full resolution works but is too slow in the direct batch-op formulation.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_28x28_b14_100_e1_lr02_b25_classchunk1_summary.json",
                "finding": "Class-chunked full-resolution run solved each independent class evidence update separately, reproducing 32% held-out while reducing wall time from about 538 s to 305 s.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_28x28_b14_100_e1_lr02_b50_classchunk1_summary.json",
                "finding": "Full-resolution class-chunked batch-size-50 run reached 41% held-out after one 100/100 epoch in about 307 s.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_28x28_b14_100_e2_lr02_b50_classchunk1_summary.json",
                "finding": "Continuing that 41% full-resolution checkpoint for one more epoch fell to 33% held-out, so the simple four-block full-resolution model is not yet stable.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_28x28_b14_eval200_from14x14_upsample_summary.json",
                "finding": "A 28x28 full-resolution model initialized by 2x upsampling the best 14x14 checkpoint reached 88.0% on a 200-image held-out sample, far better than full-resolution training from scratch.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_28x28_b14_eval1000_from14x14_upsample_summary.json",
                "finding": "The same upsampled full-resolution checkpoint reproduced 87.9% on the 1,000-image held-out split, so full resolution preserves but does not improve the best 14x14 local result.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_28x28_b14_200of2000_e1_from14x14_upsample_lr005_summary.json",
                "finding": "A bounded full-resolution fine-tune using 200 shuffled samples from the 2,000-sample train split reached 88.1% on the 1,000-image held-out split, the current best local all-SPICE subset result.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_28x28_b14_200of2000_e2_frome1_lr003_summary.json",
                "finding": "A second bounded full-resolution fine-tune at lower learning rate stayed at 88.1%, suggesting this initialized full-resolution path is also plateauing.",
            },
            {
                "artifact": "spice/run_spice_mnist_local_block_phase_transient.py",
                "finding": "Adds the first phase-resolved transient version of the local-block model: Python generates one deck and PWL inputs, while weights, biases, gains, activations, errors, gradients, and updates live as SPICE capacitor states during the run.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_phase_phase_14x14_b7_bs1_from_localblock879_lr005_summary.json",
                "finding": "One-sample all-class transient deck initialized from the 87.9% 14x14 local-block checkpoint matched the old operating-point SPICE update with max state difference about 2.65e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_phase_phase_14x14_b7_bs2_from_localblock879_lr005_summary.json",
                "finding": "Two-sample all-class transient deck verified repeated phase sequencing and gradient accumulation before a single update pulse, matching the old update with max state difference about 1.34e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_phase_phase_14x14_b7_bs1_u2_from_localblock879_lr005_summary.json",
                "finding": "Two separate one-sample update cycles ran inside one all-class transient deck and matched sequential operating-point SPICE updates with max state difference about 2.67e-5, confirming SPICE-held weights can be updated and reused without Python carrying intermediate state.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_phase_phase_14x14_b7_bs1_u2_bwd_from_localblock879_lr005_summary.json",
                "finding": "The phase-resolved deck now includes an explicit backward phase and hidden/backward delta capacitor nodes; two update cycles matched sequential operating-point SPICE updates with max state difference about 2.67e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_phase_phase_28x28_b14_bs1_from_localblock881_lr005_summary.json",
                "finding": "Full-resolution one-sample all-class transient deck initialized from the 88.1% 28x28 local-block checkpoint matched the old operating-point SPICE update with max state difference about 2.55e-5, but took about 356 s for one sample.",
            },
            {
                "artifact": "results/tables/spice_phase_transient_comparison.csv",
                "finding": "Tabulates phase-resolved transient local-block equivalence checks against the old Python-orchestrated operating-point update.",
            },
            {
                "artifact": "spice/run_spice_mnist_forward_settle_sweep.py",
                "finding": "Adds a forward-settling transient diagnostic that loads image registers, lets activation and score capacitors settle, samples class predictions over time, and writes CSV/PNG accuracy curves.",
            },
            {
                "artifact": "spice/results/spice_mnist_forward_settle_14x14_b7_100test_tau05ns_literalw_dt50ps_from879_summary.json",
                "finding": "Forward-settling probe from the 87.9% 14x14 local-block checkpoint reached 85% at 8 ns on a 100-image held-out slice, with 87% maximum transient accuracy and within-one-percentage-point settling by about 0.195 ns.",
            },
            {
                "artifact": "results/tables/spice_forward_settling_summary.csv",
                "finding": "Tabulates the forward-settling transient accuracy probe and links its CSV/PNG artifacts.",
            },
            {
                "artifact": "spice/run_spice_mnist_settling_pareto.py",
                "finding": "Adds a SPICE-validated analytical settling surrogate that sweeps readout time and activation/score time constants to produce time/accuracy Pareto frontiers for a fixed checkpoint.",
            },
            {
                "artifact": "spice/results/spice_mnist_settling_pareto_14x14_b7_1000test_taugrid_from879_summary.json",
                "finding": "The 1,000-held-out settling frontier found 88.2% best transient accuracy for the 87.9% 14x14 checkpoint, 0.3 percentage points above exact steady state; with tau_act=tau_score=0.5 ns the peak is 88.2% at 1.05 ns and the 8 ns endpoint is 87.9%.",
            },
            {
                "artifact": "results/tables/spice_mnist_settling_pareto_14x14_b7_1000test_taugrid_from879_frontier.csv",
                "finding": "Tabulates the nondominated time/accuracy readout frontier for the fixed-checkpoint settling sweep.",
            },
            {
                "artifact": "results/tables/spice_mnist_settling_pareto_14x14_b7_1000test_taugrid_from879_frontier_targets.csv",
                "finding": "Reports the fastest readout settings within 0, 0.5, 1, 2, and 5 percentage points of the best fixed-checkpoint settling accuracy.",
            },
            {
                "artifact": "spice/run_spice_mnist_local_feature_settling_pareto.py",
                "finding": "Adds the same fixed-checkpoint readout-time and time-constant frontier for the active local-feature checkpoint format with hidden-feature, readout, and output-bias capacitor timing.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_settling_pareto_smallnet10_b5_s2_c2_2k_lr003_taugrid_summary.json",
                "finding": "The 1,372-state 10x10 b5 stride2 c2 local-feature candidate peaked at 91.3% around 1.245 ns versus 91.2% steady-state; it stayed within one percentage point of peak at 0.38 ns.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_settling_pareto_smallnet10_b4_s2_c2_2k_lr008_taugrid_summary.json",
                "finding": "The 1,832-state 10x10 b4 stride2 c2 local-feature candidate peaked at 92.9% around 0.55 ns versus 92.8% steady-state; it stayed within one percentage point of peak at 0.33 ns.",
            },
            {
                "artifact": "spice/run_spice_mnist_output_bias_calibrator.py",
                "finding": "Adds train-side coordinate calibration for the ten output-bias capacitor initial voltages, then saves a recalibrated checkpoint for SPICE eval.",
            },
            {
                "artifact": "spice/results/spice_mnist_output_bias_cal_14x14_b7_cal2000_test1000_from879_summary.json",
                "finding": "Output-bias capacitor calibration on the 2,000-train split improved the 14x14 local-block checkpoint from 87.9% to 89.0% on the 1,000-image held-out split.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_biascal_eval1000_from879_summary.json",
                "finding": "ngspice eval verified the output-bias-calibrated 14x14 checkpoint at 89.0% on the 2,000/1,000 subset.",
            },
            {
                "artifact": "spice/results/spice_mnist_output_bias_cal_14x14_b7_cal60000_test10000_from879_summary.json",
                "finding": "Full train/test analytical calibration for the 14x14 checkpoint improved 86.7% to 87.39%, showing the 89.0% subset gain does not solve full MNIST.",
            },
            {
                "artifact": "spice/results/spice_mnist_output_bias_cal_28x28_b14_cal60000_test10000_from881_summary.json",
                "finding": "Full train/test analytical calibration for the 28x28 fine-tuned checkpoint improved 86.76% to 87.28%, so output-bias calibration alone is insufficient.",
            },
            {
                "artifact": "results/tables/spice_output_bias_calibration.csv",
                "finding": "Tabulates output-bias capacitor calibration results and distinguishes SPICE-verified subset eval from analytical full-MNIST checks.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_gain_8x8_b4_200_e20_lr03_summary.json",
                "finding": "Trainable local-to-class gain variant completed for 8x8, 200 train / 200 held-out, but regressed to 40% held-out; fixed gains remain better.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_8x8_b4_c4_200_e10_lr1_summary.json",
                "finding": "Local feature/readout batch-op all-SPICE training completed for 8x8, 200 train / 200 held-out, reaching 46.5% held-out; not competitive with fixed local class evidence.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_8x8_b4_c4_100_e10_softmax_diffrelu_lr1_summary.json",
                "finding": "After adding softmax output, differential clipped-ReLU local activation, and checkpoint resume support to the local feature/readout trainer, a 100/100 8x8 all-SPICE run reached 67% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_8x8_b4_c4_200_e20_softmax_diffrelu_from100e10_lr05_summary.json",
                "finding": "Continuing the patched local feature/readout checkpoint on a 200/200 8x8 subset reached 73% best and 71.5% final held-out, a better capacity signal but still below the best 14x14 local block path.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c4_200_e10_softmax_diffrelu_lr05_summary.json",
                "finding": "Scaling the patched local feature/readout model to 14x14 with four 7x7 blocks reached 72% held-out after 10 epochs on a 200/200 subset.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c4_200_e20_softmax_diffrelu_frome10_lr025_summary.json",
                "finding": "Continuing that 14x14 shared-feature checkpoint for 10 more epochs reached 75% best and 74.5% final held-out. This is a useful shared-feature capacity branch but still below the best class-specific local block path.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c8_200_e10_softmax_diffrelu_lr05_summary.json",
                "finding": "Doubling the 14x14 shared-feature model from 4 to 8 channels reached 74.5% best and 74.0% final held-out in about 924 s, so width alone did not beat the 4-channel continuation and was much slower.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c10_import_eval200_from_localblock_summary.json",
                "finding": "The local feature/readout trainer can now import the best direct local-block checkpoint as 10 feature channels; eval-only ngspice inference reproduced 88.0% on 200 held-out samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c10_import_eval1000_from_localblock_summary.json",
                "finding": "The same imported two-stage local feature wrapper reproduced 87.9% on the 1,000-image held-out split, so it is a faithful Baseline A initialization rather than a new accuracy result.",
            },
            {
                "artifact": "results/tables/spice_local_feature_comparison.csv",
                "finding": "Tabulates local feature/readout SPICE checks.",
            },
            {
                "artifact": "spice/run_spice_mnist_local_feature_torch_train.py",
                "finding": "Adds a high-level PyTorch trainer/exporter for the same local-feature/readout topology used by the SPICE deck, so architecture capacity can be tested before expensive SPICE training.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_torch_14x14_b7_c32_full_e20_lr001_summary.json",
                "finding": "A 14x14 local-feature model with four 7x7 blocks and 32 tanh feature cells per block reached 97.86% on full 60k/10k MNIST in PyTorch and exported a SPICE-evaluable checkpoint.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c32_torchbest_eval1000_spice_summary.json",
                "finding": "ngspice eval of the exported 32-channel local-feature checkpoint reached 98.2% on a 1,000-image held-out slice, confirming the SPICE deck can compute this high-capacity local topology.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c32_torchbest_spice_update_smoke_summary.json",
                "finding": "A tiny ngspice backprop/update smoke from the exported 32-channel checkpoint completed one update epoch over 10 training samples, confirming SPICE-computed updates are wired for this topology; it is not a benchmark.",
            },
            {
                "artifact": "spice/run_spice_mnist_local_feature_phase_transient.py",
                "finding": "Adds a phase-resolved transient update path for the shared local-feature/readout topology: local feature weights, readout weights, output biases, activations, deltas, backward feature deltas, and gradient accumulators are SPICE capacitor states while Python only generates guiding waveforms.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_smoke_14x14_b7_c8_linear_bs1_u1_summary.json",
                "finding": "A 14x14 four-block shared-feature transient deck with 8 channels and linear readout matched the operating-point SPICE update with max state difference about 5.26e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_torchbest_14x14_b7_c32_linear_bs1_u1_summary.json",
                "finding": "A one-sample phase-transient update from the PyTorch-trained 32-channel full-MNIST checkpoint matched the operating-point SPICE update with max state difference about 1.79e-5; this is a migration smoke, not a training benchmark.",
            },
            {
                "artifact": "experiments/19_local_feature_small_rule_sweep.py",
                "finding": "Adds a fast small-network experiment harness for local-feature training rules, separating architecture and optimizer choices before expensive SPICE runs.",
            },
            {
                "artifact": "tests/test_local_feature_rule_mapping.py",
                "finding": "Verifies that PyTorch mse_tanh + plain SGD maps to the SPICE local-feature tanh-output delta update when lr_spice = 2 * lr_torch / 10.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_small_rule_2k1k_msetanh_c8-16_e60_summary.json",
                "finding": "A phase-portable local-feature rule, tanh-output MSE with plain SGD, reached 93.5% on the 2,000-train / 1,000-held-out 14x14 MNIST subset with 16 channels per block; this is a fast PyTorch experiment, not all-SPICE training.",
            },
            {
                "artifact": "results/tables/spice_mnist_local_feature_small_rule_2k1k_msetanh_c8-16_e60_best_by_trial.csv",
                "finding": "Focused c8/c12/c16 sweep shows even 8 channels per block can reach 92.9% on the 2,000/1,000 subset with the phase-portable mse_tanh + SGD rule.",
            },
            {
                "artifact": "experiments/20_local_feature_update_noise_sweep.py",
                "finding": "Adds a fast manual-update surrogate that injects absolute parameter noise after each phase-portable local-feature update, so finite phase/integration error can be screened before running long transient SPICE jobs.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_update_noise_2k1k_c8-16_e60_summary.json",
                "finding": "The c8/c16 update-noise surrogate stayed above 90% on the 2,000-train / 1,000-held-out subset for tested noise stds up to 0.003; the best c16 lr_spice=0.8 setting reached 93.6%. This is not an all-SPICE result.",
            },
            {
                "artifact": "results/tables/spice_mnist_local_feature_update_noise_2k1k_c8-16_e60_best_by_trial.csv",
                "finding": "Tabulates c8/c16 accuracy versus lr_spice and injected per-update parameter noise; c8 lr_spice=0.3 remains above 91% around the measured phase RMS scale.",
            },
            {
                "artifact": "experiments/21_local_feature_small_frontier.py",
                "finding": "Adds a fast small-network frontier harness for the phase-portable local-feature rule, sweeping image size, block size, channels, learning rate, train-set size, and optional update noise before choosing SPICE targets.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_small_frontier_1k_i8-14_c4-16_e60_summary.json",
                "finding": "A 1,000-train / 1,000-held-out frontier sweep found that 14x14, four 7x7 blocks, 8 channels per block, and lr_spice=0.8 crossed 90% at epoch 43 with 90.2% accuracy and 3,944 phase-state values; c16 did not materially improve the result.",
            },
            {
                "artifact": "results/tables/spice_mnist_local_feature_small_frontier_1k_i8-14_c4-16_e60_pareto_by_state.csv",
                "finding": "The non-overlap fast Pareto-by-state ladder is 8x8/c4 at 82.2%, 10x10/c4 at 85.5%, 14x14/c4 at 86.7%, 10x10/c8 at 88.4%, 12x12/c8 at 89.8%, and 14x14/c8 at 90.2%; it is now the baseline that the overlapping-block sweeps improved on.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_overlap_frontier_1k_i8-14_c2-4_e60_summary.json",
                "finding": "A fast overlapping-block frontier sweep found that 14x14 with 7x7 blocks, stride 2, and only 2 channels reached 90.8% on the 1,000/1,000 surrogate with 3,944 phase-state values, the same state count as the previous non-overlap 14x14/c8 point but higher accuracy.",
            },
            {
                "artifact": "results/tables/spice_mnist_local_feature_overlap_refine_1k2k_i12-14_e80_pareto_by_state.csv",
                "finding": "Refining overlapping 12x12/14x14 candidates found a smaller >90% point: 12x12 with 6x6 blocks, stride 2, 2 channels, and 2,000 train samples reached 92.4% with 3,112 phase-state values; 12x12 stride-3/c4 reached 93.4% with 3,496 states, and 14x14 stride-3/c4 reached 93.8% with 4,432 states.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_overlap_noise_2k_i12_c2-4_e80_summary.json",
                "finding": "A targeted update-noise surrogate for the 12x12 overlap candidates kept all tested c2 stride-2 settings above 91% through noise std 0.003, with the smallest target-hit summary at 92.2% and 3,112 phase-state values; the c4 stride-3 branch reached 93.7% with 3,496 states.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_overlap_lowlr_2k_i12_c2_e120_summary.json",
                "finding": "A lower-learning-rate fast sweep for the 12x12 stride-2/c2 overlap candidate found that lr_spice=0.15 still reaches 90.6% on the 2,000/1,000 surrogate with 3,112 phase-state values, while lr_spice=0.1 reaches 89.5%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_overlap_batch2_2k_i12_c2_e40_summary.json",
                "finding": "Matching the fast experiment to the SPICE transient batch size changes the preferred rate: the same 12x12 stride-2/c2 topology trained from scratch with batch size 2 reached 93.2% on the 2,000/1,000 surrogate at lr_spice=0.1 and 3,112 phase-state values; every tested lr from 0.01 to 0.2 cleared 90% by 40 epochs.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnets_batch2_screen_1k_i8-12_e35_summary.json",
                "finding": "A renewed batch-size-matched small-network screen over high-overlap 8x8/10x10/12x12 candidates found that 10x10 with 4x4 blocks, stride 1, 1 channel, and lr_spice=0.2 reached 90.2% on the 1,000/1,000 surrogate with 2,784 phase-state values; the best screened point was 12x12 with 4x4 blocks, stride 2, 2 channels, and 90.5% with 2,840 values.",
            },
            {
                "artifact": "results/tables/spice_mnist_local_feature_smallnets_batch2_screen_1k_i8-12_e35_pareto_by_state.csv",
                "finding": "The 1,000/1,000 small-network Pareto-by-state ladder is 10x10 b5 stride2 c2 at 89.0% and 1,372 states, 10x10 b4 stride2 c2 at 89.9% and 1,832 states, 10x10 b4 stride1 c1 at 90.2% and 2,784 states, and 12x12 b4 stride2 c2 at 90.5% and 2,840 states.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnets_batch2_promote_2k_i10-12_e60_summary.json",
                "finding": "Promoting the small-network Pareto candidates to 2,000 train / 1,000 held-out found a new much smaller fast target: 10x10 with 5x5 blocks, stride 2, 2 channels, and lr_spice=0.03 reached 91.2% with only 1,372 phase-state values. The best promoted point, 12x12 b4 stride2 c2 at lr_spice=0.05, reached 93.5% with 2,840 states.",
            },
            {
                "artifact": "results/tables/spice_mnist_local_feature_smallnets_batch2_promote_2k_i10-12_e60_pareto_by_state.csv",
                "finding": "The promoted small-network Pareto ladder is 10x10 b5 stride2 c2 at 91.3% and 1,372 states, 10x10 b4 stride2 c2 at 92.8% and 1,832 states, 10x10 b5 stride1 c1 at 92.9% and 2,704 states, 10x10 b4 stride1 c1 at 93.3% and 2,784 states, and 12x12 b4 stride2 c2 at 93.5% and 2,840 states. This dominates the older 12x12 b6 stride2 c2 fast target on capacitor state.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b5_s2_c2_lr003_fast_0to212_summary.json",
                "finding": "Fast continuation from the smallest 10x10 b5 stride2 c2 lr_spice=0.03 checkpoint stayed above target but only barely, ending at 90.1% after 212 sequential seen samples.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_lr008_fast_0to212_summary.json",
                "finding": "Fast continuation from the 10x10 b4 stride2 c2 lr_spice=0.08 checkpoint retained better margin, ending at 92.3% after 212 sequential seen samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b5_s2_c2_noise_2k_e60_summary.json",
                "finding": "The targeted update-noise sweep for the smallest 10x10 b5 stride2 c2 candidate kept some settings above 90%, but the branch had little margin and fell below target at higher-noise settings.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b5_s2_c2_scale_5k2k_e120_hilr_summary.json",
                "finding": "Longer high-lr fast training revived the smaller 1,372-state 10x10 b5 stride2 c2 branch at 5,000 train / 2,000 held-out: best accuracy reached 92.05% at lr_spice=0.3.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b5_s2_c2_scale_10k2k_e120_hilr_summary.json",
                "finding": "The same 1,372-state 10x10 b5 stride2 c2 branch reached 92.4% at 10,000 train / 2,000 held-out with longer high-lr fast training.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_fastcont_lr003_0to212_summary.json",
                "finding": "A lower-lr continuation gate from the 10k-trained b5 checkpoint stayed above target after 212 sequential samples, ending at 91.9% with lr_spice=0.03.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_fastcont_lr003_212to1024_summary.json",
                "finding": "Extending the same lower-lr fast b5 continuation from 212 to 1,024 total samples stayed above target but low-margin, ending at 90.5%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_track12_lr003_eval2000_phase1ns_summary.json",
                "finding": "A short real 1 ns SPICE phase-training gate for the smaller 1,372-state b5 branch ran 12 samples and ended at 92.3% on 2,000 ngspice-held-out images, matching the fast reference with RMS state drift 4.90e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_resume12to60_lr003_eval2000_phase1ns_summary.json",
                "finding": "Resuming that smaller b5 SPICE capacitor state from 12 to 60 total phase-trained samples ended at 91.6% on the 2,000-image ngspice eval, matching the fast reference with RMS state drift 1.98e-4.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_spice60_fast_lr003_60to212_summary.json",
                "finding": "Fast continuation from the actual 60-sample b5 SPICE capacitor state predicted the branch would remain above target through 212 total samples, ending at 91.85%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b5_s2_c2_scale_10k2k_e120_lr02_resume60to212_lr003_eval2000_phase1ns_summary.json",
                "finding": "Resuming the smaller 1,372-state b5 SPICE capacitor state from 60 to 212 total phase-trained samples ended at 91.9% on the 2,000-image ngspice eval, matching the fast reference with RMS state drift 2.99e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_tiny_below_b5_screen_2k2k_e60_summary.json",
                "finding": "A below-b5 fast screen found a new 1,048-state 9x9 b4 stride2 c2 candidate that reached 90.4% on the 2,000/2,000 screen, while several more aggressive 706-1,016-state c1/four-block variants stayed below target.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_tiny_below_b5_promote_5k10k_e120_summary.json",
                "finding": "Promoting the below-b5 candidates to 5,000 and 10,000 training samples found a 936-state 10x10 b4 stride2 c1 target at 91.6%, a 1,048-state 9x9 b4 stride2 c2 target at 92.4%, and a 1,372-state b5 baseline at 93.25%.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_below_b5_i10_b4_s2_c1_10k2k_e120_lr003_fastcont_lr002_0to1024_summary.json",
                "finding": "The 936-state c1 candidate stayed above target through a 1,024-sample fast continuation gate, ending at 91.7% with lr_spice=0.02.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_below_b5_i10_b4_s2_c1_10k2k_e120_lr003_track12_lr002_eval2000_phase1ns_summary.json",
                "finding": "The 936-state 10x10 b4 stride2 c1 candidate passed a 12-sample real 1 ns SPICE phase-training gate, ending at 91.45% on the 2,000-image ngspice eval with matched fast/reference accuracy and RMS state drift 6.89e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_sub936_promote_5k2k_e140_summary.json",
                "finding": "A fast sub-936 revisit over 706-, 796-, 904-, and 936-state candidates found a new 904-state 10x10 b6 stride2 c1 branch that reached 90.15% on the 5,000/2,000 split; all sub-904 branches stayed below 90%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_i10_b6_s2_c1_promote_10k2k_e180_summary.json",
                "finding": "Promoting the 904-state 10x10 b6 stride2 c1 branch to 10,000 train / 2,000 held-out reached 91.05% at lr_spice=0.01, making it the smallest current fast above-90 local-feature candidate.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i10_b6_s2_c1_10k2k_e180_lr001_fastcont_lr001_0to212_summary.json",
                "finding": "A 212-sample fast continuation gate from the 904-state 10k checkpoint stayed above target, ending at 90.5% with lr_spice=0.01.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i10_b6_s2_c1_10k2k_e180_lr001_fastcont_lr001_0to1024_summary.json",
                "finding": "A 1,024-sample fast continuation gate from the same 904-state checkpoint ended at 90.75%, making the branch plausible enough for short SPICE validation.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_i10_b6_s2_c1_10k2k_e180_lr001_track12_eval2000_phase1ns_summary.json",
                "finding": "The 904-state 10x10 b6 stride2 c1 branch passed a 12-sample real 1 ns SPICE phase-training gate, ending at 91.05% on the 2,000-image ngspice eval with matched fast/reference accuracy, RMS state drift 3.56e-5, and max state drift 4.12e-4.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i10_b6_s2_c1_10k2k_e180_lr001_spice12_fast_lr001_12to60_summary.json",
                "finding": "Fast continuation from the actual 12-sample 904-state SPICE capacitor state predicted 90.8% at 60 total samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_i10_b6_s2_c1_10k2k_e180_lr001_resume12to60_eval2000_phase1ns_summary.json",
                "finding": "Resuming the 904-state 10x10 b6 stride2 c1 SPICE capacitor state from 12 to 60 total phase-trained samples ended at 90.8% on the 2,000-image ngspice eval with matched fast/reference accuracy, RMS state drift 9.99e-5, and max state drift 4.69e-4.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i10_b6_s2_c1_10k2k_e180_lr001_spice12_fast_lr001_12to212_summary.json",
                "finding": "Fast continuation from the actual 12-sample 904-state SPICE capacitor state predicted the branch stays above target through 212 total samples, ending at 90.5%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_sub904_broad_2k2k_e80_summary.json",
                "finding": "A broad sub-904 fast pilot over 706-, 712-, 796-, 808-, and 840-state topologies found no 2,000/2,000 hit; the best point was a 796-state 7x7 b3 stride2 c2 branch at 88.4%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_sub904_promote_5k2k_e160_summary.json",
                "finding": "Focused promotion of the best sub-904 families to 5,000 train / 2,000 held-out found a 796-state 7x7 b3 stride2 c2 branch at 91.0%, while the 706- and 712-state near misses stayed below 90%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_i7_b3_s2_c2_promote_10k2k_e180_summary.json",
                "finding": "Promoting the 796-state 7x7 b3 stride2 c2 branch to 10,000 train / 2,000 held-out reached 91.35% at lr_spice=0.01, with every tested lr in the 0.008-0.05 grid still above 90.5%.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i7_b3_s2_c2_10k2k_e180_lr001_fastcont_lr001_0to1024_summary.json",
                "finding": "A 1,024-sample fast continuation gate from the 796-state 10k checkpoint ended at 91.4%, so the new smaller branch has better long-window fast margin than the previous 904-state branch.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_i7_b3_s2_c2_10k2k_e180_lr001_track12_eval2000_phase1ns_summary.json",
                "finding": "The 796-state 7x7 b3 stride2 c2 branch passed a 12-sample real 1 ns SPICE phase-training gate, ending at 91.35% on the 2,000-image ngspice eval with matched fast/reference accuracy, RMS state drift 1.36e-5, and max state drift 8.63e-5.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i7_b3_s2_c2_10k2k_e180_lr001_spice12_fast_lr001_12to1024_summary.json",
                "finding": "Fast continuation from the actual 12-sample 796-state SPICE capacitor state predicted the branch stays above target through 1,024 total samples, ending at 91.4%.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i7_b3_s2_c2_10k2k_e180_lr001_spice12_fast_lr001_12to60_summary.json",
                "finding": "Fast continuation from the actual 12-sample 796-state SPICE capacitor state predicted 91.05% at 60 total samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_i7_b3_s2_c2_10k2k_e180_lr001_resume12to60_eval2000_phase1ns_summary.json",
                "finding": "Resuming the 796-state 7x7 b3 stride2 c2 SPICE capacitor state from 12 to 60 total phase-trained samples ended at 91.0% on the 2,000-image ngspice eval, with fast reference accuracy 91.05%, RMS state drift 1.28e-4, and max state drift 9.31e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_i7_b3_s2_c2_10k2k_e180_lr001_resume60to212_eval2000_phase1ns_summary.json",
                "finding": "Continuing the same 796-state private/local SPICE capacitor trajectory from 60 to 212 total phase-trained samples stayed above target, peaking at 91.25% after 152 samples and ending at 90.9% on the 2,000-image ngspice eval; final fast reference accuracy was 90.95%, RMS state drift was 1.69e-4, and max state drift was 1.55e-3.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_settling_pareto_tiny_i7_b3_s2_c2_10k2k_lr001_taugrid_summary.json",
                "finding": "The 796-state checkpoint's finite-readout timing surrogate peaks at its 91.35% steady-state accuracy and reaches 90.45% by 0.53 ns, giving a fast time/accuracy Pareto point without more SPICE.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_sub796_promote_10k2k_e220_summary.json",
                "finding": "Promoting the sub-796 near misses found a 706-state 7x7 b5 stride1 c1 fast checkpoint at 90.55% on 10,000 train / 2,000 held-out; the 712-state 9x9 b3 stride2 c1 branch stayed below target, peaking at 89.45%.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i7_b5_s1_c1_10k2k_e220_lr002_fastcont_lr002_0to1024_summary.json",
                "finding": "The 706-state 7x7 b5 stride1 c1 fast checkpoint is low-margin: it dipped to 89.95% after 212 continuation samples but ended at 90.5% after 1,024 samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_i7_b5_s1_c1_10k2k_e220_lr002_track12_eval2000_phase1ns_summary.json",
                "finding": "The 706-state 7x7 b5 stride1 c1 branch passed a 12-sample real 1 ns SPICE phase-training gate, ending at 90.5% on the 2,000-image ngspice eval with matched fast/reference accuracy, RMS state drift 4.10e-5, and max state drift 2.15e-4.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_i7_b5_s1_c1_10k2k_e220_lr002_spice12_fast_lr002_12to1024_summary.json",
                "finding": "Fast continuation from the actual 12-sample 706-state SPICE capacitor state predicted 89.95% at 212 total samples and 90.5% at 1,024 total samples, so it is a smallest-state short validation rather than a high-margin long-window path.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_sub706_pilot_2k2k_e80_summary.json",
                "finding": "A follow-up sub-706 fast pilot over 56 compact topology/lr trials found no target hit on 2,000 train / 2,000 held-out. The best overall point was the same-state 706-value 9x9 b5 stride2 c1 branch at 86.8%, and the best true sub-706 point was a 544-state 8x8 b4 stride2 c1 branch at 85.2%, so the current 706-state short-SPICE-validated branch remains the small-state floor for this feature family.",
            },
            {
                "artifact": "experiments/23_local_feature_shared_kernel_frontier.py",
                "finding": "Added a fast shared-kernel local-feature harness where one learned local kernel per channel is reused across block positions while the class readout remains position-specific; it reports train, gradient, temporary, and total phase-state values.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_shared_kernel_pilot_2k2k_e80_summary.json",
                "finding": "The shared-kernel 2,000/2,000 pilot found no target hit. Its state Pareto rose from 76.7% at 272 phase-state values to 87.6% at 540 phase-state values, so sharing buys compactness but loses too much accuracy at the smallest settings.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_shared_kernel_promote_5k2k_e120_summary.json",
                "finding": "Focused shared-kernel promotion of the best 504-, 540-, and 624-state candidates to 5,000 train / 2,000 held-out still found no target hit; the best point was 88.45% at 540 phase-state values, and the best 504-state point was 88.35%.",
            },
            {
                "artifact": "experiments/24_local_feature_partial_sharing_frontier.py",
                "finding": "Added a fast partial-sharing local-feature harness with shared kernel capacitors for selected channels and private per-block capacitors for other channels, while keeping the readout position-specific. This is a compact accelerator parameterization, not the cleanest local-synapse architecture, because shared channels use one physical weight-capacitor state across block positions.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_partial_sharing_pilot_2k2k_e80_summary.json",
                "finding": "The partial-sharing 2,000/2,000 pilot found a near-hit: 7x7 b3 stride2 with two shared and one private channel reached 89.3% at 854 phase-state values; the sub-796 Pareto point reached 88.7% at 784 values.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_partial_sharing_promote_5k2k_e120_summary.json",
                "finding": "Focused partial-sharing promotion to 5,000 train / 2,000 held-out found a new fast small-state frontier: 9x9 b4 stride2 with one shared and one private channel reached 90.15% at 776 phase-state values, 9x9 b3 stride2 with two shared channels reached 90.85% at 784 values, and 7x7 b3 stride2 with two shared plus one private channel reached 92.3% at 854 values.",
            },
            {
                "artifact": "experiments/25_partial_sharing_checkpoint_continuation.py",
                "finding": "Added a continuation gate that preserves partial-sharing capacitor semantics; using the generic local-feature continuation would incorrectly expand shared kernels into independent per-block weights after the first update.",
            },
            {
                "artifact": "spice/results/fast_mnist_partialsem_i9_b4_s2_sh1_pr1_5k2k_lr003_fastcont_lr003_0to212_summary.json",
                "finding": "Correct partial-sharing continuation shows the 776-state 9x9 b4 stride2 one-shared/one-private branch is low-margin: it starts at 90.15% and falls to 89.85% after 212 samples.",
            },
            {
                "artifact": "spice/results/fast_mnist_partialsem_i9_b3_s2_sh2_pr0_5k2k_lr005_fastcont_lr001_0to1024_summary.json",
                "finding": "Correct partial-sharing continuation keeps the 784-state 9x9 b3 stride2 two-shared-channel branch above target through 1,024 samples, ending at 90.25%.",
            },
            {
                "artifact": "spice/results/fast_mnist_partialsem_i7_b3_s2_sh2_pr1_5k2k_lr003_fastcont_lr001_0to1024_summary.json",
                "finding": "Correct partial-sharing continuation keeps the 854-state 7x7 b3 stride2 two-shared/one-private branch above target through 1,024 samples, ending at 91.65%. It is the stronger fast partial-sharing candidate, but still needs a SPICE deck that preserves shared-kernel capacitor semantics.",
            },
            {
                "artifact": "spice/run_spice_mnist_partial_sharing_phase_transient.py",
                "finding": "Added a phase-resolved transient SPICE deck for partial-sharing local features. Shared kernel weights/biases, private weights/biases, readout weights, output biases, activations, deltas, and gradient accumulators are capacitor voltages; shared kernel gradient capacitors sum across block positions before one shared apply pulse. The shared kernel is a real shared capacitor state broadcast into several block computations, not a per-block local synapse.",
            },
            {
                "artifact": "spice/results/spice_mnist_partial_sharing_phase_partial_i7_b3_s2_sh2_pr1_lr001_bs2_u1_phase1ns_meas_summary.json",
                "finding": "A one-update 1 ns partial-sharing phase-transient check on the 854-state 7x7 b3 stride2 two-shared/one-private checkpoint matched the correct fast partial-sharing reference with RMS state drift 2.63e-6 and max drift 1.92e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_partial_sharing_phase_partial_i7_b3_s2_sh2_pr1_lr001_bs2_u2_phase1ns_meas_summary.json",
                "finding": "A two-update 1 ns partial-sharing phase-transient check verified repeated accumulate/apply/clear behavior inside one transient, matching the fast partial-sharing reference with RMS state drift 2.26e-5 and max drift 1.49e-4.",
            },
            {
                "artifact": "spice/run_spice_mnist_partial_sharing_phase_train.py",
                "finding": "Added a repeated partial-sharing phase-training harness that launches transient chunks from capacitor checkpoints and evaluates expanded readout-equivalent checkpoints with ngspice inference.",
            },
            {
                "artifact": "spice/results/spice_mnist_partial_sharing_phase_train_partial_i7_b3_s2_sh2_pr1_5k2k_lr003_track12_lr001_eval200_phase1ns_summary.json",
                "finding": "The 854-state partial-sharing branch passed a 12-sample real 1 ns SPICE phase-training gate. The final SPICE capacitor state matched the correct fast reference with RMS drift 4.64e-5 and max drift 5.88e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_partial_i7_b3_s2_sh2_pr1_phase12_eval2000_summary.json",
                "finding": "A 2,000-image ngspice eval of the 12-sample partial-sharing phase-trained checkpoint reached 92.15%, confirming the SPICE-updated 854-state branch remains above target on the same held-out size as the fast promotion.",
            },
            {
                "artifact": "spice/results/spice_mnist_partial_sharing_phase_train_partial_i9_b3_s2_sh2_pr0_5k2k_lr005_track12_lr001_eval200_phase1ns_summary.json",
                "finding": "The smaller 784-state 9x9 b3 stride2 two-shared-channel branch also passed a 12-sample real 1 ns SPICE phase-training gate, with final RMS drift 5.52e-5 and max drift 2.96e-4 against the correct fast partial-sharing reference.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_partial_i9_b3_s2_sh2_pr0_phase12_eval2000_summary.json",
                "finding": "A 2,000-image ngspice eval of the 784-state branch after the 12-sample SPICE phase gate reached 90.95%, making it the smallest partial-sharing SPICE-validated above-90 branch so far, but with less margin than the 854-state branch and with a shared-capacitor broadcast assumption.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_below_b5_i9_b4_s2_c2_10k2k_e120_lr005_fastcont_lr002_0to1024_summary.json",
                "finding": "The 1,048-state 9x9 c2 candidate stayed above target through a 1,024-sample fast continuation gate, ending at 92.05% with lr_spice=0.02.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_below_b5_i9_b4_s2_c2_10k2k_e120_lr005_track12_lr002_eval2000_phase1ns_summary.json",
                "finding": "The 1,048-state 9x9 b4 stride2 c2 candidate passed a 12-sample real 1 ns SPICE phase-training gate, ending at 92.3% on the 2,000-image ngspice eval with matched fast/reference accuracy and RMS state drift 2.95e-5.",
            },
            {
                "artifact": "spice/results/fast_mnist_tiny_below_b5_i9_b4_s2_c2_10k2k_e120_lr005_spice12_fast_lr002_12to60_summary.json",
                "finding": "Fast continuation from the actual 12-sample 1,048-state SPICE capacitor state predicted the branch would remain at 92.15% through 60 total samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_tiny_below_b5_i9_b4_s2_c2_10k2k_e120_lr005_resume12to60_lr002_eval2000_phase1ns_summary.json",
                "finding": "Resuming the 1,048-state 9x9 b4 stride2 c2 SPICE capacitor state from 12 to 60 total phase-trained samples ended at 92.15% on the 2,000-image ngspice eval with matched fast/reference accuracy and RMS state drift 1.13e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_noise_2k_e60_summary.json",
                "finding": "The targeted update-noise sweep for 10x10 b4 stride2 c2 kept all tested lr_spice=0.05 and lr_spice=0.08 settings at or above 91.7% through noise std 0.003, making it the preferred small SPICE target.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_smallnet10_b4_s2_c2_lr008_bs2_u2_phase1ns_meas_train2000_summary.json",
                "finding": "A bounded phase-transient SPICE check of the selected 10x10 b4 stride2 c2 lr_spice=0.08 checkpoint ran two batch-2 updates with 1 ns phases and scalar final-state measurements. It matched the operating-point SPICE reference with max state diff 5.99e-6 and RMS state diff 1.98e-6, using only 5.51 s phase wall time for four samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_track12_eval1000_phase1ns_summary.json",
                "finding": "A short repeated phase-transient tracker for the selected 10x10 b4 stride2 c2 checkpoint ran 12 real 1 ns phase-trained samples. The 1,000-image ngspice eval stayed at 92.8%, phase/reference fast accuracies matched at 92.8%, and final RMS drift against the fast reference was 3.76e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_resume12to60_eval1000_phase1ns_summary.json",
                "finding": "Resuming the selected 10x10 b4 stride2 c2 branch from 12 to 60 total seen samples hit the fast-predicted low point but stayed above target: the 1,000-image ngspice eval was 91.5%, phase/reference fast accuracies matched at 91.5%, and final RMS drift was 5.10e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_resume60to120_eval1000_phase1ns_summary.json",
                "finding": "Continuing the selected 10x10 b4 stride2 c2 branch from 60 to 120 total seen samples reached the fast-predicted recovery region: the 1,000-image ngspice eval was 92.7%, phase/reference fast accuracies matched at 92.7%, and final RMS drift was 6.57e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_resume120to180_eval1000_phase1ns_summary.json",
                "finding": "Continuing the selected 10x10 b4 stride2 c2 branch from 120 to 180 total seen samples stayed in the predicted >92% window: the 1,000-image ngspice eval was 92.5%, phase/reference fast accuracies matched at 92.5%, and final RMS drift was 7.42e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_lr008_resume180to212_eval1000_phase1ns_summary.json",
                "finding": "Continuing the selected 10x10 b4 stride2 c2 branch from 180 to 212 total seen samples reached the fast-predicted endpoint: the 1,000-image ngspice eval was 92.3%, phase/reference fast accuracies matched at 92.3%, and final RMS drift was 7.47e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_handoff_e12_summary.json",
                "finding": "A reduced-pretraining fast screen for the selected 10x10 b4 stride2 c2 topology found that 12 epochs can cross 90% briefly, but its 212-sample fast continuations fell below target and should not be prioritized for SPICE.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_handoff_e15_summary.json",
                "finding": "A 15-epoch reduced-pretraining fast screen for the selected 10x10 b4 stride2 c2 topology found a viable but higher-gain handoff: lr_spice=0.15 reached 90.9%.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_handoff_e15_lr015_fast_0to212_summary.json",
                "finding": "The 15-epoch lr_spice=0.15 handoff remained at 91.1% after 212 fast-reference continuation samples, while the other tested 15-epoch branches ended below 90%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_handoff_e15_lr015_track212_eval1000_phase1ns_summary.json",
                "finding": "The 15-epoch lr_spice=0.15 handoff ran 212 real 1 ns phase-trained samples from chunk 0 and ended at 91.1% on the 1,000-image ngspice eval; phase/reference fast accuracies matched at 91.1%, but final RMS drift rose to 4.09e-3.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_handoff_e20_summary.json",
                "finding": "A 20-epoch reduced-pretraining fast screen for the selected 10x10 b4 stride2 c2 topology found a cleaner handoff: lr_spice=0.05 reached 91.4%, and the lr_spice=0.1 checkpoint used for phase validation actually peaked at epoch 16 with 91.3%.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_handoff_e20_lr01_fast_0to212_summary.json",
                "finding": "The epoch-16 lr_spice=0.1 checkpoint from the 20-epoch screen remained at 91.0% after 212 fast-reference continuation samples, making it worth one SPICE validation despite reduced offline pretraining.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_handoff_e20_lr01_track212_eval1000_phase1ns_summary.json",
                "finding": "The epoch-16 lr_spice=0.1 checkpoint from the 20-epoch screen ran 212 real 1 ns phase-trained samples from chunk 0 and ended at 91.0% on the 1,000-image ngspice eval; phase/reference fast accuracies matched at 91.0%, and final RMS drift was 1.03e-3.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_scale_5k2k_e25_summary.json",
                "finding": "Scaling the selected 10x10 b4 stride2 c2 topology to 5,000 train / 2,000 held-out samples kept the same 1,832-state circuit above target: the best fast checkpoint reached 93.1% and all tested final epoch-25 settings stayed at or above 91.2%.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_5k2k_e25_lr008_fast_0to212_summary.json",
                "finding": "The 5k/2k lr_spice=0.08 checkpoint stayed at 92.6% after a 212-sample fast-reference continuation, making it the larger-coverage candidate selected for SPICE validation.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_5k2k_e25_lr008_track212_eval2000_phase1ns_summary.json",
                "finding": "The 5k/2k lr_spice=0.08 checkpoint ran 212 real 1 ns phase-trained samples from chunk 0 and ended at 92.6% on a 2,000-image ngspice eval; phase/reference fast accuracies matched at 92.6%, and final RMS drift was 1.17e-3.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_scale_10k2k_e25_summary.json",
                "finding": "Scaling the selected 10x10 b4 stride2 c2 topology to 10,000 train / 2,000 held-out samples kept the same 1,832-state circuit and raised the fast best checkpoint to 94.1%.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_10k2k_e25_lr008_fast_0to212_summary.json",
                "finding": "The 10k/2k lr_spice=0.08 checkpoint stayed at 93.8% after a 212-sample fast-reference continuation, making it the highest-margin experiment-first candidate selected for SPICE validation so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_10k2k_e25_lr008_track212_eval2000_phase1ns_summary.json",
                "finding": "The 10k/2k lr_spice=0.08 checkpoint ran 212 real 1 ns phase-trained samples from chunk 0 and ended at 93.8% on a 2,000-image ngspice eval; phase/reference fast accuracies matched at 93.8%, and final RMS drift was 1.29e-3.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_10kfull_e25_lr008_fast_0to212_summary.json",
                "finding": "The same 10k-trained lr_spice=0.08 checkpoint generalizes to the full 10,000-image MNIST test set: it started at 93.57% and stayed at 93.44% after the 212-sample fast-reference continuation.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_10kfull_e25_lr008_track212_eval10000_phase1ns_summary.json",
                "finding": "The corresponding full-test SPICE phase run evaluated all 10,000 MNIST test images after 212 real 1 ns phase-trained samples and ended at 93.45%; the fast reference was 93.44%, and final RMS drift was 1.29e-3.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_smallnet10_b4_s2_c2_scale_60kfull_e25_summary.json",
                "finding": "Training the selected 10x10 b4 stride2 c2 topology on all 60,000 MNIST training images with the fast phase-portable rule kept the same 1,832-state circuit and reached 94.96% on the full 10,000-image test set.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_fast_0to212_summary.json",
                "finding": "The full-train lr_spice=0.03 checkpoint stayed at 94.81% after a 212-sample fast-reference continuation on the full 10,000-image test set, with a transient peak at 95.02%.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_track212_eval10000_phase1ns_summary.json",
                "finding": "The full-train lr_spice=0.03 checkpoint ran 212 real 1 ns phase-trained samples from chunk 0 and ended at 94.77% on all 10,000 MNIST test images in ngspice; the fast reference was 94.81%, and final RMS drift was 2.83e-4.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_fast_resume212to1024_summary.json",
                "finding": "A longer fast-reference continuation from the 212-sample full-train state predicted the branch remains high-margin through 1,024 total samples: final full-test accuracy was 94.77%, with a 94.90% best point near 600 samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_resume212to424_eval10000_phase1ns_summary.json",
                "finding": "Resuming the saved SPICE capacitor state from 212 to 424 total phase-trained samples ended at 94.82% on all 10,000 MNIST test images in ngspice; the resumed fast reference was 94.83%, and final RMS drift was 3.49e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_resume424to600_eval10000_phase1ns_summary.json",
                "finding": "Continuing the saved SPICE capacitor trajectory from 424 to the fast-predicted 600-sample peak ended at 94.89% on all 10,000 MNIST test images in ngspice; the fast reference was 94.90%, and final RMS drift was 4.16e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_resume600to1024_eval10000_phase1ns_summary.json",
                "finding": "Continuing the same SPICE trajectory from 600 to 1,024 total phase-trained samples ended at 94.75% on all 10,000 MNIST test images in ngspice; the fast reference was 94.77%, and final RMS drift was 5.54e-4.",
            },
            {
                "artifact": "spice/results/fast_mnist_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_fast_resume1024to2048_summary.json",
                "finding": "A longer fast-reference continuation from 1,024 to 2,048 total samples predicted 94.80% final full-test accuracy and a 94.82% best point around 1,792 samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallnet10_b4_s2_c2_scale_60kfull_e25_lr003_resume1024to2048_eval10000_phase1ns_summary.json",
                "finding": "Continuing the same SPICE trajectory from 1,024 to 2,048 total phase-trained samples ended at 94.79% on all 10,000 MNIST test images in ngspice; the fast reference was 94.80%, and final RMS drift was 6.92e-4.",
            },
            {
                "artifact": "results/tables/spice_overlap_phase_validation.csv",
                "finding": "Tabulates phase-transient validation for overlapping local-feature candidates. The older 12x12 overlapping c2 branch matched OP SPICE with RMS state diff 1.58e-6 for the batch-size-matched lr_spice=0.1 checkpoint; the new 10x10 b4 stride2 c2 lr_spice=0.08 checkpoint matches similarly tightly at 1.98e-6 RMS with lower phase wall time.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_overlap12_b6_s2_c2_lr03_bs2_u2_phase1ns_meas_train12_summary.json",
                "finding": "The same 12x12 overlapping c2 lr_spice=0.3 phase deck showed sample-sensitive finite-phase error on the first four samples of the 12-sample tracker sequence: RMS state diff was about 2.40e-3 at 1 ns.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_overlap12_b6_s2_c2_lr03_bs2_u2_phase4ns_meas_train12_summary.json",
                "finding": "Increasing phase length on that difficult overlap batch improved but did not eliminate the error; 4 ns phases reduced RMS state diff to about 6.00e-4 with phase wall time about 55.6 s for four training samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_lr03_track12_phase1ns_summary.json",
                "finding": "A repeated 12-sample phase-training tracker for the 12x12 overlapping c2 checkpoint at lr_spice=0.3 accumulated large drift over three 1 ns chunks, ending at RMS state drift about 2.75e-2 and max drift about 0.158 against the fast reference; this branch needs timing or update-scaling stabilization before longer phase training.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_lr015_track12_phase1ns_summary.json",
                "finding": "Reducing the 12x12 overlapping c2 branch to lr_spice=0.15 restored bounded repeated-chunk behavior while preserving a >90% fast candidate: the matched 12-sample 1 ns tracker ended at RMS drift about 9.19e-4 and max drift about 3.95e-3, with phase/reference fast accuracies aligned on the 200-image slice.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_lr015_resume12to32_eval200_phase1ns_summary.json",
                "finding": "Resuming the stabilized 12x12 overlapping c2 lr_spice=0.15 branch from chunk 3 to chunk 8 added 20 more SPICE-trained samples with 1 ns phases and SPICE held-out evaluation. The 200-image SPICE slice was 85.0% at resume, peaked at 87.0% after chunk 4, and ended at 84.0% after 32 seen samples; final phase/reference fast accuracies were 84.0% versus 84.5%, with RMS state drift about 1.05e-3.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_batch2_lr01_track12_eval1000_phase1ns_summary.json",
                "finding": "The batch-size-matched 12x12 overlapping c2 lr_spice=0.1 checkpoint was verified by ngspice at 93.2% on the 1,000-image slice before phase training and 93.1% after three 1 ns phase chunks over 12 samples. Final phase/reference fast accuracies matched at 93.1%, with RMS state drift about 8.95e-6 and max drift about 8.11e-5.",
            },
            {
                "artifact": "experiments/22_local_feature_checkpoint_continuation.py",
                "finding": "Adds a fast reference continuation harness for local-feature checkpoints, so candidate phase-training windows can be screened before spending ngspice transient time.",
            },
            {
                "artifact": "spice/results/fast_mnist_overlap12_b6_s2_c2_batch2_lr01_fast_resume12to212_summary.json",
                "finding": "Fast continuation from the 12-sample batch-matched checkpoint predicted that the lr_spice=0.1 branch would stay at 93.1% through about 60 seen samples and remain at 92.6% after 212 seen samples, selecting the first extension window for SPICE validation.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_batch2_lr01_resume12to60_eval1000_phase1ns_summary.json",
                "finding": "The selected SPICE continuation resumed the batch-size-matched 12x12 overlapping c2 lr_spice=0.1 phase state from chunk 3 to chunk 15, adding 48 more real 1 ns phase-trained samples. The 1,000-image ngspice eval stayed at 93.1% after 60 total seen samples, phase/reference fast accuracies matched at 93.1%, and RMS state drift was about 6.53e-5.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_overlap_batch2_handoff_e6_lr005_summary.json",
                "finding": "An earlier handoff checkpoint at lr_spice=0.05 after only 6 fast batch-2 epochs reached 90.5% on the 2,000/1,000 surrogate, but its fast sequential continuation dropped below 90% after 40-60 seen samples, making it a brittle SPICE target.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_overlap_batch2_handoff_e10_lr005_summary.json",
                "finding": "A slightly later lr_spice=0.05 handoff after 10 fast batch-2 epochs reached 91.1% on the 2,000/1,000 surrogate with the same 3,112 phase-state values, cutting fast pretraining from 36 epochs to 10 before the SPICE handoff.",
            },
            {
                "artifact": "spice/results/fast_mnist_overlap12_b6_s2_c2_batch2_lr005_e10_fast_0to212_summary.json",
                "finding": "Fast continuation from the 10-epoch lr_spice=0.05 handoff predicted that sequential batch-2 updates would remain at 90.4% after 212 seen samples, with 90.5% at 60 seen samples.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_batch2_lr005_e10_track60_eval1000_phase1ns_summary.json",
                "finding": "The reduced-pretraining SPICE handoff from the 10-epoch lr_spice=0.05 checkpoint ran 60 real 1 ns phase-training samples and ended at 90.4% on the 1,000-image ngspice slice. Phase/reference fast accuracies stayed aligned at 90.4% versus 90.5%, with RMS state drift about 2.89e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_overlap12_b6_s2_c2_batch2_lr005_e10_resume60to120_eval1000_phase1ns_summary.json",
                "finding": "Continuing the same 10-epoch handoff from 60 to 120 total seen samples hit the predicted low-margin point: the 1,000-image ngspice eval ended exactly at 90.0%, phase/reference fast accuracies both read 90.0%, and RMS state drift was about 3.32e-4. This is a threshold pass, not a robust endpoint.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_smallfrontier14_c8_lr08_bs2_u2_phase1ns_meas_summary.json",
                "finding": "The smallest fast-frontier point above 90% on the 1k/1k surrogate, 14x14/c8 with lr_spice=0.8, matched a two-update operating-point SPICE reference in the phase-transient capacitor deck with RMS state difference about 6.44e-5 and max state difference about 7.21e-4 at 1 ns phases.",
            },
            {
                "artifact": "results/tables/spice_small_frontier_validation.csv",
                "finding": "Tabulates fast-frontier candidates that have been spot-validated by phase-transient SPICE equivalence checks.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs1_u2_phase4ns_summary.json",
                "finding": "Two phase-transient updates from the moderate c8 mse_tanh checkpoint, using lr_spice=0.3, matched the operating-point SPICE reference with max state difference about 3.36e-3 and RMS state difference about 3.91e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs2_u1_phase4ns_summary.json",
                "finding": "A c8 mse_tanh phase-transient batch update, using lr_spice=0.3 and batch size 2, accumulated two samples in gradient capacitors before one apply pulse and matched the operating-point batch update with max state difference about 7.46e-4 and RMS state difference about 1.15e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs2_u2_phase4ns_summary.json",
                "finding": "A c8 mse_tanh phase-transient multi-batch run, using lr_spice=0.3, ran two two-sample batch updates in one transient while retaining the updated weight/readout capacitor voltages between cycles; it matched sequential operating-point batch references with max state difference about 7.44e-4 and RMS state difference about 1.15e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs2_u2_phase2ns_summary.json",
                "finding": "The same c8 two-batch update schedule with 2 ns phases and the same 25 ps tau target reduced phase wall time from about 95.7 s to 56.8 s, with RMS state difference increasing from about 1.15e-4 to 2.31e-4.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_smallrule_c8_msetanh_lrspice03_bs2_u2_phase1ns_summary.json",
                "finding": "The same c8 two-batch update schedule with 1 ns phases reduced phase wall time to about 34.6 s, with RMS state difference about 4.63e-4; this is the current aggressive timing point for small c8 phase-training checks.",
            },
            {
                "artifact": "results/tables/spice_local_feature_phase_timing_frontier.csv",
                "finding": "Tabulates the c8 local-feature phase timing/error frontier for the same two-batch transient update at 4 ns, 2 ns, and 1 ns phase lengths.",
            },
            {
                "artifact": "spice/run_spice_mnist_local_feature_phase_train.py",
                "finding": "Adds a repeated phase-transient local-feature training harness: Python generates guiding waveforms and launches chunks, each chunk stores weights, activations, deltas, and gradient accumulators as SPICE capacitor voltages, then the harness evaluates checkpoints with SPICE inference.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_c8_msetanh_lrspice03_eval100_bs2_u2_phase1ns_fromsmallrule_summary.json",
                "finding": "The c8 phase-training harness ran one 1 ns four-sample phase-training chunk from the phase-portable checkpoint and evaluated 100 held-out images in SPICE before and after; accuracy changed from 94% to 87%, so this is a workflow smoke rather than an accuracy benchmark.",
            },
            {
                "artifact": "results/tables/spice_phase_training_comparison.csv",
                "finding": "Tabulates repeated phase-training harness smoke runs separately from equivalence-only phase-transient checks.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smoke_8x8_c2_bs1_u1_phase1ns_meas_summary.json",
                "finding": "A tiny phase-training harness smoke passed with final-state scalar measurements instead of a full transient wrdata dump, confirming the measurement-output path for reducing generated data volume before larger spot checks.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr08_random_track12_phase1ns_summary.json",
                "finding": "A random-state 14x14/c8 phase-training tracker ran three 1 ns phase chunks over 12 MNIST samples with scalar final-state measurements and compared the resulting capacitor state against the exact fast reference rule; final RMS state drift was about 3.53e-3 and max drift about 3.77e-2, dominated by output-bias/readout state.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr03_random_track12_phase1ns_summary.json",
                "finding": "A matched random-state 14x14/c8 tracker at lr_spice=0.3 on the same 12 MNIST samples and 1 ns phase schedule reduced final RMS drift to about 5.92e-4 and max drift to about 7.97e-3, about 6x and 4.7x lower than the lr_spice=0.8 tracker.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr08_random_track12_phase2ns_summary.json",
                "finding": "A matched lr_spice=0.8 tracker with 2 ns phases and the same 25 ps tau target reduced final RMS drift to about 1.77e-3 and max drift to about 1.89e-2, about 2x better than 1 ns at the same learning rate but still roughly 3x worse in RMS drift than the 1 ns lr_spice=0.3 branch while taking about 1.56x longer.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smoke_resume_stage1_8x8_c1_summary.json",
                "finding": "Per-chunk phase-training resume support now saves both SPICE capacitor state and fast-reference state; a staged 8x8/c1 resume smoke exactly matched an uninterrupted two-chunk run with zero phase-state and reference-state difference.",
            },
            {
                "artifact": "results/tables/spice_phase_training_resume_validation.csv",
                "finding": "Tabulates the resume validation comparing staged versus uninterrupted phase-training chunks.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr03_random_track20_phase1ns_summary.json",
                "finding": "A longer 14x14/c8 lr_spice=0.3 phase-training tracker ran five 1 ns phase chunks over 20 MNIST samples with per-chunk phase and fast-reference checkpoints; final RMS drift was about 4.61e-4 and max drift about 4.71e-3.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_phase_train_smallfrontier14_c8_lr03_resume20to40_eval200_phase1ns_summary.json",
                "finding": "The same 14x14/c8 lr_spice=0.3 phase-training branch resumed from chunk 5 to chunk 10, covering samples 20-40; SPICE held-out evaluation on the 200-image slice improved from 10.5% at chunk 5 to 15.0% at chunk 10, with final RMS drift about 6.46e-4 and max drift about 5.23e-3 against the fast reference state.",
            },
            {
                "artifact": "results/tables/spice_phase_training_reference_tracking.csv",
                "finding": "Tabulates repeated phase-training runs that track SPICE capacitor state against the fast reference update rule chunk by chunk.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_feature_local_feature_14x14_b7_c8_msetanh_lr15_eval100_spice_summary.json",
                "finding": "ngspice eval of the moderate c8 mse_tanh checkpoint reached 88% on a 100-image random slice; the fast model gives the same 88% on that exact slice, so this is a consistency check rather than a benchmark.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_analog_noise_smoke_summary.json",
                "finding": "Analog local block-evidence SPICE smoke run completed with explicit noise/mismatch robustness rows; the 20-sample accuracy is only a functional check, not a meaningful benchmark.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_linear_out_8x8_b4_200_e10_lr02_summary.json",
                "finding": "Linear analog class-evidence readout was tested for the local block trainer on 8x8, 200 train / 200 held-out; it reached 67% best held-out, not better than the 68% tanh-output local baseline.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_softmax_8x8_b4_200_e10_lr05_summary.json",
                "finding": "SPICE-computed softmax class competition was tested for the local block trainer on 8x8, 200 train / 200 held-out; it reached 64% best held-out, also not better than the 68% baseline.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e2_tanh_lr02_b100_summary.json",
                "finding": "Controlled 14x14 local block baseline with tanh local evidence, 200 train / 200 held-out, two epochs, and batch size 100 reached 35.5% best held-out and 32.5% final held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e2_relu_lr02_b100_summary.json",
                "finding": "The same controlled run using algebraic ReLU local evidence reached 23.5% best/final held-out, so raw ReLU is currently weaker than tanh in this short all-SPICE local update test.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e2_clipped_relu_lr02_b100_summary.json",
                "finding": "The same controlled run using algebraic clipped-ReLU local evidence also reached 23.5% best/final held-out; clipping alone did not fix early local training.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e2_relu_softmax_lr05_b100_summary.json",
                "finding": "Pairing algebraic ReLU local evidence with SPICE-computed softmax class competition reached 41.0% best/final held-out in the same two-epoch 14x14 subset setup, better than the tanh-output rectified runs.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e2_clipped_relu_softmax_lr05_b100_summary.json",
                "finding": "Pairing algebraic clipped-ReLU local evidence with SPICE-computed softmax class competition also reached 41.0% best/final held-out; bounded rectified evidence is viable enough for further biasing/differential-channel tests, but is not close to the final goal.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e2_diff_clipped_relu_lr02_b100_summary.json",
                "finding": "Differential clipped-ReLU local evidence, implemented as two bounded rectifier branches, reached 34.5% best / 32.5% final held-out with tanh-style signed output on the controlled 14x14, 200/200, two-epoch subset.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e2_diff_clipped_relu_softmax_lr05_b100_summary.json",
                "finding": "The same differential clipped-ReLU local evidence with SPICE-computed softmax class competition reached 55.5% best/final held-out after two epochs, substantially better than one-sided ReLU in this short controlled test.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e5_diff_clipped_relu_softmax_lr05_b100_summary.json",
                "finding": "Continuing the differential clipped-ReLU + softmax checkpoint for three more 200-sample epochs reached 66.5% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_200_e5_diff_clipped_relu_softmax_eval1000_summary.json",
                "finding": "Eval-only check of that 66.5% checkpoint on a 1,000-image held-out subset reached 69.0%, so the 200-sample result was not only a tiny held-out artifact.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_1000_e1_diff_clipped_relu_softmax_from200e5_lr03_b100_summary.json",
                "finding": "Scaling differential clipped-ReLU + softmax to one 1,000-train / 1,000-held-out all-SPICE epoch reached 78.0% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_1000_e2_diff_clipped_relu_softmax_frome1_lr02_b100_summary.json",
                "finding": "A second 1,000-sample continuation epoch for differential clipped-ReLU + softmax reached 79.9% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_local_block_14x14_b7_1000_e3_diff_clipped_relu_softmax_frome2_lr015_b100_summary.json",
                "finding": "A third 1,000-sample continuation epoch for differential clipped-ReLU + softmax reached 80.9% held-out. This is trainable and hardware-plausible, but still behind the best tanh local path and far from full-MNIST completion.",
            },
            {
                "artifact": "results/tables/spice_local_activation_comparison.csv",
                "finding": "Tabulates the local activation comparisons and records which clipped-ReLU rows predate the derivative helper rewrite.",
            },
            {
                "artifact": "spice/run_spice_mnist_sparse_random_train.py",
                "finding": "Adds a random sparse hidden all-SPICE trainer with local sparse input fan-in, optional shortcut inputs, simple bounded activations, SPICE-computed backprop or direct feedback alignment hidden errors, and analog/clipped/legacy-quantized/symmetric-quantized/pulse-count/dithered-pulse/local-residual-pulse gradient coding modes.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_stronger_analog_summary.json",
                "finding": "Random sparse hidden network with analog gradients, 8x8 MNIST, 100 train / 100 held-out, hidden=32, fan_in=16, reached 34% best held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_quant4_summary.json",
                "finding": "The same sparse network with 4-bit quantized gradients reached 72% best held-out, suggesting coarse pulse-like gradient coding can help hidden credit assignment in this small sparse SPICE test.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_quant8_summary.json",
                "finding": "The same sparse network with 8-bit quantized gradients reached 34% best held-out, similar to analog; more nominal gradient precision was not automatically better in this setup.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_analog_lr5_summary.json",
                "finding": "Increasing analog learning rate to 5.0 reached only 19% best held-out, so the 4-bit result was not reproduced by simply scaling analog updates.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_symquant4_summary.json",
                "finding": "A more electronic symmetric signed-magnitude 4-bit gradient encoding reached only 29% best held-out, so removing the legacy quantizer's zero-offset behavior hurt this sparse setup.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_pulsecount4_summary.json",
                "finding": "A true sign/magnitude 4-bit pulse-count update encoding also reached only 29% best held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_pulsedither4_summary.json",
                "finding": "Deterministic dithered 4-bit pulse-count updates improved to 47% best/final held-out, better than plain pulses but still below the legacy 4-bit quantized run.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e10_pulsedither4_frome5_lr05_summary.json",
                "finding": "Resuming the dithered pulse-count checkpoint for five more epochs at learning rate 0.5 reached 50% best/final held-out, showing the hardware-natural pulse branch can keep improving but remains far behind the best local SPICE subset model.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_pulseresid4_summary.json",
                "finding": "A local residual pulse accumulation mode, where each synapse keeps a residual state and emits integer programming pulses, reached 30% best/final held-out in the same sparse 8x8 100/100 setup.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e5_dfa_quant4_summary.json",
                "finding": "Direct feedback alignment mode replaces hidden-layer weight transport with fixed random class-error feedback; on the same sparse 8x8 100/100 setup with 4-bit quantized gradients it reached 61% after five epochs.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e10_dfa_quant4_frome5_lr05_summary.json",
                "finding": "Continuing the DFA sparse checkpoint for five more epochs at learning rate 0.5 peaked at 75% and finished at 70%, roughly matching the earlier 72% exact-backprop sparse baseline with a more plausible hidden error path.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e13_dfa_pulsedither4_fromdfaquant_lr05_summary.json",
                "finding": "Switching the DFA sparse checkpoint to deterministic dithered pulse updates for three epochs reached 79% held-out, the best sparse hardware-plausible update result so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_100_e16_dfa_pulsedither4_frome13_lr025_summary.json",
                "finding": "A lower-rate dithered pulse continuation from the 79% checkpoint slipped to 77%, so the prior checkpoint is better.",
            },
            {
                "artifact": "spice/results/spice_mnist_sparse_random_sparse_8x8_h32_f16_200_e18_dfa_pulsedither4_from100e13best_lr025_summary.json",
                "finding": "Scaling the 79% dithered-pulse DFA sparse checkpoint to 200 train / 200 held-out samples reached 71.5%, so the 100/100 gain did not scale cleanly.",
            },
            {
                "artifact": "results/tables/spice_sparse_random_gradient_precision.csv",
                "finding": "Tabulates sparse random network gradient precision experiments.",
            },
            {
                "artifact": "results/biological_sparse_precision_notes.md",
                "finding": "Records the current more-biological hardware direction: sparse recurrent sheets, simple rectifying/saturating activations, and gradient precision from pulse/dither/residual-charge accumulation rather than 4-bit floating-point datapaths.",
            },
            {
                "artifact": "spice/run_spice_mnist_recurrent_sparse_sheet_train.py",
                "finding": "Adds a random sparse recurrent sheet all-SPICE trainer: local sparse input fan-in, local recurrent fan-in plus optional shortcuts, parallel recurrent ticks, SPICE-computed softmax error, unrolled recurrent backprop through time or direct feedback alignment hidden errors, and programmable updates.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h16_if8_rf4_t3_100_e5_analog_summary.json",
                "finding": "The recurrent sparse sheet with analog gradients reached 29% best/final held-out on a 100/100 8x8 MNIST subset.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h16_if8_rf4_t3_100_e5_quant4_summary.json",
                "finding": "The same recurrent sparse sheet with 4-bit quantized gradients reached 55% best/final held-out after five epochs.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h16_if8_rf4_t3_100_e10_quant4_frome5_lr05_summary.json",
                "finding": "Continuing the 4-bit recurrent sparse sheet checkpoint for five more epochs at learning rate 0.5 reached 68% best/final held-out on the 100/100 subset.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h16_if8_rf4_t3_100_e5_dfa_quant4_summary.json",
                "finding": "The comparable 16-cell recurrent sparse sheet with direct feedback alignment reached 56% held-out after five epochs, matching the first exact-BPTT 55% run while avoiding trainable weight transport into hidden errors.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h16_if8_rf4_t3_100_e10_dfa_quant4_frome5_lr05_summary.json",
                "finding": "Continuing the 16-cell recurrent DFA checkpoint for five more epochs at learning rate 0.5 peaked at 65% and finished at 64%, below but near the 68% exact-BPTT continuation.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h16_if8_rf4_t3_200_e15_quant4_from100e10_lr025_summary.json",
                "finding": "Continuing the recurrent sparse sheet on 200 train / 200 held-out samples reached 69% best/final held-out, so this branch did not collapse when the sample count doubled.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_100_e5_quant4_summary.json",
                "finding": "Doubling recurrent sheet capacity to 32 cells reached 58% best/final held-out after five 100/100 epochs.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_100_e10_quant4_frome5_lr05_summary.json",
                "finding": "Continuing the 32-cell recurrent sparse sheet checkpoint for five more epochs at learning rate 0.5 reached 74% best and 73% final held-out on 100/100.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_200_e13_quant4_from100e10_lr025_summary.json",
                "finding": "Continuing the 32-cell recurrent sparse sheet on 200 train / 200 held-out samples reached 73% best/final held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_100_e5_quant4_mem03_inh01_summary.json",
                "finding": "Adding fixed self-memory 0.3 and local inhibition 0.1 to the 32-cell recurrent sparse sheet reached 61% best/final held-out after five 100/100 epochs.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_100_e10_quant4_mem03_inh01_frome5_lr05_summary.json",
                "finding": "Continuing the self-memory/local-inhibition recurrent sheet reached 76% best and 74% final held-out on 100/100, the best recurrent sheet result so far.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_100_e5_dfa_quant4_mem03_inh01_summary.json",
                "finding": "The 32-cell self-memory/local-inhibition recurrent DFA branch reached 51% best after five epochs, weaker than the comparable exact-BPTT start.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_100_e10_dfa_quant4_mem03_inh01_frome5_lr05_summary.json",
                "finding": "Continuing the 32-cell recurrent DFA branch peaked at 72% and finished at 66%, useful but below the 76% exact-BPTT recurrent best.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_200_e13_dfa_quant4_mem03_inh01_from100e10best_lr025_summary.json",
                "finding": "Scaling the best 32-cell recurrent DFA checkpoint to 200 train / 200 held-out samples peaked at 67.5% and finished at 65%, below the comparable exact-BPTT recurrent scale check.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h32_if8_rf4_t3_200_e13_quant4_mem03_inh01_from100e10_lr025_summary.json",
                "finding": "Continuing the self-memory/local-inhibition recurrent sheet on 200 train / 200 held-out samples reached 75.5% best but fell to 68% final, so this branch is useful but unstable.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_smoke_8x8_h12_if6_rf3_t2_40_e2_pulseresid4_summary.json",
                "finding": "A recurrent sparse sheet residual-pulse smoke test ran end-to-end but reached only 20% best/final held-out on 40/40.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h16_if8_rf4_t3_100_e5_pulseresid4_mem03_inh01_summary.json",
                "finding": "A proper 100/100 residual-pulse recurrent sparse sheet check with self-memory and local inhibition reached 29% after five epochs, far below the quantized-gradient recurrent branch.",
            },
            {
                "artifact": "spice/results/spice_mnist_recurrent_sparse_sheet_recurrent_8x8_h16_if8_rf4_t3_100_e5_pulseresid6_step001_mem03_inh01_summary.json",
                "finding": "Increasing recurrent residual update precision with a smaller fixed pulse quantum and 63 max pulses improved early learning but reached only 30% after five epochs.",
            },
            {
                "artifact": "results/tables/spice_recurrent_sparse_sheet_comparison.csv",
                "finding": "Tabulates recurrent sparse sheet SPICE experiments.",
            },
            {
                "artifact": "spice/results/spice_mnist_shared_local_smoke_summary.json",
                "finding": "Shared local class-evidence SPICE trainer smoke test completed, proving the shared-kernel update netlist runs; early larger checks were slow and weak, so it is not yet a candidate.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_8x8_b4_c2_100_e5_lr02_summary.json",
                "finding": "Naive two-channel local block evidence with gain 0.5 completed for 8x8, 100 train / 100 held-out, but stalled at 30% held-out.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_8x8_b4_c2_g1_100_e3_lr02_summary.json",
                "finding": "Naive two-channel local block evidence with gain 1.0 also stalled at 30% held-out; extra duplicated local cells are not enough.",
            },
            {
                "artifact": "spice/results/spice_mnist_local_block_mc_local_block_mc_8x8_b4_c2_100_e5_traingains_softmax_diffrelu_lr05_summary.json",
                "finding": "After adding trainable channel gains, softmax output, and differential clipped-ReLU support to the multichannel local trainer, a 100/100 8x8 SPICE run reached 37% held-out. This improves over the old 30% multichannel checks but remains far below the single-channel local path.",
            },
            {
                "artifact": "results/tables/spice_multichannel_local_comparison.csv",
                "finding": "Tabulates multichannel local SPICE checks.",
            },
            {
                "artifact": "spice/results/spice_mnist_stream_stream_8x8_200_e3_lr80k_epoch01_eval_trace.csv",
                "finding": "Chunked 8x8, 200-sample run reached epoch 1 with recovered 60.5% held-out accuracy before being stopped for runtime.",
            },
        ],
        "next_smallest_valid_milestone": {
            "name": "Small-frontier to all-SPICE local-feature milestone",
            "description": (
                "Use the fast local-feature frontier as the main design loop, then validate only Pareto candidates in SPICE. "
                "The old non-overlap ladder found 14x14/c8 as the first 1k/1k point above 90%, and the first overlapping "
                "batch-size-matched target was 12x12 with 6x6 blocks, stride 2, 2 channels, 93.2% on the 2k/1k surrogate, "
                "and 3,112 phase-state values. A smaller experiment-first pass now gives better fast targets: 10x10 b5 "
                "stride2 c2 reaches about 91.2-91.3% with only 1,372 phase-state values, and 10x10 b4 stride2 c2 reaches "
                "92.8% with 1,832 values. The best promoted small candidate, 12x12 b4 stride2 c2, reaches 93.5% with "
                "2,840 values. Local-feature settling sweeps add the timing side of the small-network Pareto view: the b5 branch "
                "peaks at 91.3% and remains within one point of peak by 0.38 ns, while the b4 branch peaks at 92.9% and remains "
                "within one point by 0.33 ns. Initial continuation and noise screens selected 10x10 b4 stride2 c2 over the smaller b5 branch: "
                "the b4 lr_spice=0.08 checkpoint remains at 92.3% after 212 sequential samples and all tested b4 noise "
                "settings stay at or above 91.7% through noise std 0.003. A later high-lr revisit of the smaller b5 branch "
                "reached 92.4% on 10,000 train / 2,000 held-out, and a lower-lr continuation gate from that checkpoint stayed "
                "at 91.9% after 212 sequential samples and 90.5% after 1,024 total samples. A short 12-sample real 1 ns SPICE phase-training gate for this b5 checkpoint "
                "ended at 92.3% with RMS drift 4.90e-5, resuming it to 60 total samples ended at 91.6% with RMS drift 1.98e-4, "
                "and resuming the same real SPICE trajectory to 212 total samples ended at 91.9% with RMS drift 2.99e-4. "
                "This made b5 the smallest SPICE-validated above-90 local-feature path at 1,372 phase-state values. "
                "A new below-b5 fast pass then found still smaller candidates: 10x10 b4 stride2 c1 reaches 91.6% "
                "with 936 phase-state values and remains at 91.7% after a 1,024-sample fast continuation, while 9x9 b4 stride2 c2 reaches "
                "92.4% with 1,048 phase-state values and remains at 92.05% after the same 1,024-sample gate. The 936-state c1 branch "
                "also passed a short real 1 ns SPICE phase-training gate at 91.45% with RMS drift 6.89e-5, making it the smallest "
                "SPICE-validated above-90 local-feature path at that point. The 1,048-state branch also passes a short real 1 ns SPICE gate at "
                "92.3% with RMS drift 2.95e-5, then resumes to 60 total phase-trained samples at 92.15% with RMS drift 1.13e-4, "
                "making it the better-margin below-b5 resumed SPICE path. "
                "A newer fast-only sub-936 pass found that 10x10 b6 stride2 c1 reaches 90.15% with 904 phase-state values on 5,000/2,000, "
                "then 91.05% with 904 values on 10,000/2,000 and remains at 90.75% after a 1,024-sample fast continuation. "
                "It now also passes a 12-sample real 1 ns SPICE phase-training gate at 91.05% with RMS drift 3.56e-5, making it the smallest "
                "SPICE-validated above-90 local-feature path so far. Resuming the same real SPICE capacitor trajectory to 60 total samples "
                "matched the fast prediction at 90.8% with RMS drift 9.99e-5; a fast continuation from the 12-sample SPICE state predicts 90.5% at 212 total samples. "
                "A renewed sub-904 experiment-first pass then found a smaller 796-state 7x7 b3 stride2 c2 branch: the broad 2,000/2,000 pilot peaked at 88.4%, "
                "the focused 5,000/2,000 promotion reached 91.0%, and the 10,000/2,000 promotion reached 91.35% with the same 796 values. "
                "Its 1,024-sample fast continuation ended at 91.4%, and a 12-sample real 1 ns SPICE phase-training gate ended at 91.35% with RMS drift 1.36e-5. "
                "Fast continuation from the actual 12-sample 796-state SPICE capacitor state predicts 90.9% at 212 total samples and 91.4% at 1,024 total samples, "
                "and resuming that same real SPICE capacitor trajectory to 60 total samples ended at 91.0% with RMS drift 1.28e-4. "
                "Continuing the same private/local SPICE trajectory to 212 total samples stayed above target, peaking at 91.25% after 152 samples and ending at 90.9% with RMS drift 1.69e-4. "
                "Its timing surrogate reaches 90.45% by 0.53 ns. "
                "Promoting the remaining sub-796 near misses found a still smaller 706-state 7x7 b5 stride1 c1 branch at 90.55% on 10,000/2,000, while the 712-state branch stayed below target. "
                "The 706-state branch passed a 12-sample real 1 ns SPICE phase-training gate at 90.5% with RMS drift 4.10e-5, making it the smallest short-SPICE-validated above-90 path so far; "
                "however, its fast continuation dips to 89.95% at 212 samples and recovers only to 90.5% at 1,024 samples, so the 796-state branch remains the stronger small candidate. "
                "A follow-up sub-706 fast pilot over 56 compact topology/lr trials produced no promotion candidate: the best overall point was 86.8% at the same 706-state budget, and the best true sub-706 point was 85.2% at 544 states. "
                "A shared-kernel local-feature variant compressed the same kind of model by reusing one learned local kernel per channel across all block positions; this is a shared-capacitor broadcast/time-multiplexing assumption rather than pure per-block synaptic storage. The 2,000/2,000 pilot peaked at 87.6% with 540 phase-state values, and focused 5,000/2,000 promotion only reached 88.45% with 540 values. "
                "A partial-sharing variant recovers much more accuracy under the same compact-accelerator assumption: focused 5,000/2,000 promotion reached 90.15% at 776 phase-state values, 90.85% at 784 values, and 92.3% at 854 values. "
                "A corrected shared-capacitor continuation gate keeps the 784-state branch at 90.25% after 1,024 samples and the 854-state branch at 91.65%, while the 776-state branch dips below target after 212 samples. "
                "The partial-sharing phase deck now preserves that shared-capacitor update directly: a two-update 1 ns transient check matched the correct fast reference with RMS drift 2.26e-5, and a 12-sample repeated 1 ns SPICE gate on the 854-state branch ended with RMS drift 4.64e-5 against the correct fast reference. "
                "A full 2,000-image ngspice eval of the 12-sample SPICE-updated checkpoint reached 92.15%, making the 854-state branch the strongest small partial-sharing SPICE validation so far. "
                "The smaller 784-state 9x9 b3 shared branch also passed a 12-sample SPICE gate with RMS drift 5.52e-5 and reached 90.95% on a full 2,000-image ngspice eval, making it the smallest partial-sharing SPICE-validated above-90 branch so far. Because these shared branches rely on shared physical weight-capacitor state, the 796-state and 706-state private/local branches remain the cleaner evidence for the original local-neuron/synapse architecture. "
                "The 10x10 b4 stride2 c2 checkpoint remains the higher-margin long-window path: it is "
                "SPICE-validated for bounded equivalence, matching OP SPICE at 1 ns with RMS state diff 1.98e-6 and max "
                "diff 5.99e-6. A short repeated tracker for the same checkpoint keeps the 1,000-image ngspice eval at "
                "92.8% after 12 real 1 ns phase-trained samples, with RMS drift 3.76e-5. Extending the same tracker to "
                "60 total samples stayed above target at 91.5%, and continuing to 120 total samples recovered to 92.7% "
                "with phase/reference fast accuracies matched. Continuing to 180 total samples stayed at 92.5% with "
                "phase/reference fast accuracies matched and RMS drift 7.42e-4. The 212-sample endpoint stayed at 92.3% "
                "with matched phase/reference fast accuracies and RMS drift 7.47e-4. Reducing the handoff pretraining is viable but costs margin: "
                "12 fast epochs crossed 90% only briefly and fell below target in the 212-sample fast continuation gate, "
                "while a 15-epoch lr_spice=0.15 handoff ran 212 real 1 ns phase-trained samples from chunk 0 and ended at "
                "91.1% on the 1,000-image ngspice eval with matched phase/reference fast accuracies, but higher RMS drift "
                "4.09e-3. A cleaner lr_spice=0.1 checkpoint from the 20-epoch screen actually peaked at epoch 16; it ended "
                "at 91.0% after the same 212-sample SPICE validation with RMS drift 1.03e-3. Scaling the same 1,832-state "
                "10x10 branch to 5,000 train / 2,000 held-out improved the fast margin: the best checkpoint reached 93.1%, "
                "the lr_spice=0.08 checkpoint stayed at 92.6% after a 212-sample fast continuation, and the corresponding "
                "SPICE phase run ended at 92.6% on 2,000 ngspice-held-out images with RMS drift 1.17e-3. "
                "Scaling again to 10,000 train / 2,000 held-out raised the fast best checkpoint to 94.1%; the lr_spice=0.08 "
                "checkpoint stayed at 93.8% after the same 212-sample fast continuation, and the corresponding SPICE phase run "
                "also ended at 93.8% on 2,000 ngspice-held-out images with RMS drift 1.29e-3. "
                "Evaluating that same 10k-trained checkpoint on all 10,000 MNIST test images kept the fast continuation at 93.44%, "
                "and the corresponding full-test SPICE phase run ended at 93.45% with fast reference accuracy 93.44% and RMS drift 1.29e-3. "
                "Using all 60,000 MNIST training images in the fast phase-portable rule raised the same 1,832-state branch to 94.96% "
                "on the full 10,000-image test set; the lr_spice=0.03 checkpoint stayed at 94.81% after a 212-sample fast continuation, "
                "and the corresponding full-train/full-test SPICE phase run ended at 94.77% with fast reference accuracy 94.81% and RMS drift 2.83e-4. "
                "A longer fast prediction stayed high-margin through 1,024 total samples at 94.77%; resuming the SPICE capacitor state from "
                "212 to 424 total phase-trained samples ended at 94.82% on the full test set with fast reference accuracy 94.83% and RMS drift 3.49e-4. "
                "Continuing the same trajectory to the 600-sample fast-predicted peak ended at 94.89% with fast reference accuracy 94.90% and RMS drift 4.16e-4; "
                "continuing to 1,024 total samples ended at 94.75% with fast reference accuracy 94.77% and RMS drift 5.54e-4; "
                "continuing again to 2,048 total samples ended at 94.79% with fast reference accuracy 94.80% and RMS drift 6.92e-4. "
                "The older batch-size-matched "
                "12x12 b6 stride2 c2 lr_spice=0.1 path remains a higher-state repeated-training validation: "
                "it reproduces 93.2% in ngspice eval and remains at 93.1% after 60 total "
                "seen samples with 48 samples added by resumed 1 ns phase-transient SPICE training. A reduced-pretraining "
                "handoff from only 10 fast epochs at lr_spice=0.05 "
                "ended at 90.4% after 60 real phase-trained samples and exactly 90.0% after 120 total seen samples, so it "
                "is viable but low-margin. The next useful work should shift back to small-network sweeps and Pareto-frontier thinking, then use short SPICE "
                "windows as validation gates before pushing beyond the 2,048-sample repeated phase window. Keep scalar "
                "final-state measurement output enabled to control transient data volume."
            ),
            "why_not_complete_goal": "It is a stepping stone; the final goal still requires the training sequence itself to run in SPICE, not only a bounded phase-training window from a fast-trained checkpoint.",
        },
    }

    tables = ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    json_path = tables / "full_spice_goal_audit.json"
    json_path.write_text(json.dumps(audit, indent=2) + "\n")

    md = ROOT / "results/full_spice_goal_audit.md"
    lines = [
        "# Full SPICE Goal Audit",
        "",
        f"Objective: {OBJECTIVE}",
        "",
        f"Achieved: `{achieved}`",
        "",
        "## Checklist",
        "",
    ]
    for c in criteria:
        lines += [
            f"### {c['criterion']}",
            "",
            f"- Status: `{c['status']}`",
            f"- Evidence: {c['evidence']}",
            f"- Blocking gap: {c['blocking_gap']}",
            "",
        ]
    lines += [
        "## Scaling Evidence",
        "",
    ]
    for item in audit["scaling_evidence"]:
        lines += [
            f"- `{item['artifact']}`: {item['finding']}",
        ]
    lines += [
        "",
        "## Next Smallest Valid Milestone",
        "",
        audit["next_smallest_valid_milestone"]["description"],
        "",
        f"Why this is not completion: {audit['next_smallest_valid_milestone']['why_not_complete_goal']}",
        "",
    ]
    md.write_text("\n".join(lines))
    print(json.dumps({"audit": str(json_path), "markdown": str(md), "achieved": achieved}, indent=2))


if __name__ == "__main__":
    main()
