from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from run_device_mnist01_scalar_training import sanitize_tag
from run_device_sequential_training import mos_models, run_netlist
from run_spice_sweep import ROOT, detect_spice


ERROR_CASES = (
    "target_positive_score_negative",
    "target_negative_score_positive",
    "target_positive_score_positive",
    "target_negative_score_negative",
    "neutral",
)
ERROR_TOPOLOGIES = ("competition", "binary-descent", "label-descent")


def case_values(case: str, *, score_center: float, score_delta: float) -> tuple[float, float, float]:
    positive_score = score_center + score_delta / 2.0
    negative_score = score_center - score_delta / 2.0
    if case == "target_positive_score_negative":
        return 1.2, negative_score, positive_score
    if case == "target_negative_score_positive":
        return 0.0, positive_score, negative_score
    if case == "target_positive_score_positive":
        return 1.2, positive_score, negative_score
    if case == "target_negative_score_negative":
        return 0.0, negative_score, positive_score
    if case == "neutral":
        return 0.0, score_center, score_center
    raise ValueError(f"error case must be one of {ERROR_CASES}")


def label_values(case: str) -> tuple[float, float]:
    if case in {"target_positive_score_negative", "target_positive_score_positive"}:
        return 1.2, 0.0
    if case in {"target_negative_score_positive", "target_negative_score_negative"}:
        return 0.0, 1.2
    if case == "neutral":
        return 0.0, 0.0
    raise ValueError(f"error case must be one of {ERROR_CASES}")


def generate_netlist(
    *,
    error_case: str,
    score_center: float = 0.10,
    score_delta: float = 0.08,
    restore_error: bool = True,
    error_restore_width: float = 7.0,
    error_topology: str = "competition",
) -> str:
    if error_case not in ERROR_CASES:
        raise ValueError(f"error_case must be one of {ERROR_CASES}")
    if error_topology not in ERROR_TOPOLOGIES:
        raise ValueError(f"error_topology must be one of {ERROR_TOPOLOGIES}")
    if score_delta < 0.0:
        raise ValueError("score_delta must be nonnegative")
    if error_restore_width <= 0.0:
        raise ValueError("error_restore_width must be positive")
    target, score, scoren = case_values(error_case, score_center=score_center, score_delta=score_delta)
    targetp, targetn = label_values(error_case)
    if min(target, targetp, targetn, score, scoren) < 0.0 or max(target, targetp, targetn, score, scoren) > 1.2:
        raise ValueError("target and score rails must stay within supply rails")
    lines = [
        "* Output score/error primitive smoke.",
        "* Tests target and differential score rails -> dp/dn -> optional edp/edn.",
        "* Matches the transistor/passive error motif used by the MNIST01 block runner.",
        "* No behavioral sources.",
        ".param VDD=1.2",
        mos_models(),
        ".options method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6",
        "Vdd vdd 0 {VDD}",
        f"Vtarget target 0 {target:.12g}",
        f"Vtargetp targetp 0 {targetp:.12g}",
        f"Vtargetn targetn 0 {targetn:.12g}",
        f"Vscore score 0 {score:.12g}",
        f"Vscoren scoren 0 {scoren:.12g}",
        "Verr err 0 PULSE(0 1.2 1.0n 10p 10p 3.0n 10n)",
        "Vrstg rstg 0 PULSE(0 1.2 0.0n 10p 10p 0.55n 10n)",
        "Vrstgn rstgn 0 PULSE(0 1.2 0.0n 10p 10p 0.55n 10n)",
        "Cdp dp 0 20f IC=0",
        "Cdn dn 0 20f IC=0",
        "Rdp dp 0 1G",
        "Rdn dn 0 1G",
        "Mreset_dp dp rstg 0 0 NMOS W=4u L=180n",
        "Mreset_dn dn rstg 0 0 NMOS W=4u L=180n",
    ]
    if restore_error:
        lines += [
            "Cedp edp 0 8f IC=0",
            "Cedn edn 0 8f IC=0",
            "Cerrstore_src errstore_src 0 0.1f IC=0",
            "Redp edp 0 1G",
            "Redn edn 0 1G",
            "Rerrstore_src errstore_src 0 1G",
            "Mprecharge_edp edp rstgn vdd vdd PMOS W=4u L=180n",
            "Mprecharge_edn edn rstgn vdd vdd PMOS W=4u L=180n",
        ]
    if error_topology == "competition":
        lines += [
            "",
            "* Shared output error from target/raw-score conductance competition.",
            "Mdp_t0 vdd target dp_t 0 NSENSE W=32u L=180n",
            "Mdp_t1 dp_t err dp 0 NSENSE W=32u L=180n",
            "Mdp_sn0 vdd scoren dp_sn 0 NSENSE W=24u L=180n",
            "Mdp_sn1 dp_sn err dp 0 NSENSE W=24u L=180n",
            "Mdp_y0 dp err dp_y 0 NSENSE W=24u L=180n",
            "Mdp_y1 dp_y score 0 0 NSENSE W=24u L=180n",
            "Mdn_y0 vdd score dn_y 0 NSENSE W=32u L=180n",
            "Mdn_y1 dn_y err dn 0 NSENSE W=32u L=180n",
            "Mdn_sn0 dn err dn_sn 0 NSENSE W=24u L=180n",
            "Mdn_sn1 dn_sn scoren 0 0 NSENSE W=24u L=180n",
            "Mdn_t0 dn err dn_t 0 NSENSE W=24u L=180n",
            "Mdn_t1 dn_t target 0 0 NSENSE W=24u L=180n",
        ]
    elif error_topology == "binary-descent":
        lines += [
            "",
            "* Binary descent output error: dp ~= targetp*scoren, dn ~= targetn*score.",
            "Mdp_bd_t vdd targetp dp_bd_t 0 NSENSE W=48u L=180n",
            "Mdp_bd_s dp_bd_t scoren dp_bd_s 0 NSENSE W=48u L=180n",
            "Mdp_bd_e dp_bd_s err dp 0 NSENSE W=48u L=180n",
            "Mdn_bd_t vdd targetn dn_bd_t 0 NSENSE W=48u L=180n",
            "Mdn_bd_s dn_bd_t score dn_bd_s 0 NSENSE W=48u L=180n",
            "Mdn_bd_e dn_bd_s err dn 0 NSENSE W=48u L=180n",
        ]
    else:
        lines += [
            "",
            "* Label descent output error: dp ~= targetp, dn ~= targetn during err.",
            "Mdp_ld_t vdd targetp dp_ld_t 0 NSENSE W=48u L=180n",
            "Mdp_ld_e dp_ld_t err dp 0 NSENSE W=48u L=180n",
            "Mdn_ld_t vdd targetn dn_ld_t 0 NSENSE W=48u L=180n",
            "Mdn_ld_e dn_ld_t err dn 0 NSENSE W=48u L=180n",
        ]
    if restore_error:
        lines += [
            "",
            "* Restored output-error latch: raw dp/dn select full-swing edp/edn learning rails.",
            f"Merrstore_p edp edn vdd vdd PMOS W={error_restore_width:.6g}u L=180n",
            f"Merrstore_n edn edp vdd vdd PMOS W={error_restore_width:.6g}u L=180n",
            f"Merrstore_ep edp dn errstore_src 0 NSENSE W={error_restore_width:.6g}u L=180n",
            f"Merrstore_en edn dp errstore_src 0 NSENSE W={error_restore_width:.6g}u L=180n",
            f"Merrstore_tail errstore_src err 0 0 NMOS W={error_restore_width:.6g}u L=180n",
        ]
    lines += [
        ".meas tran dp_after FIND V(dp) AT=3.5n",
        ".meas tran dn_after FIND V(dn) AT=3.5n",
        ".meas tran raw_error_diff PARAM='dp_after-dn_after'",
        *(
            [
                ".meas tran edp_after FIND V(edp) AT=3.5n",
                ".meas tran edn_after FIND V(edn) AT=3.5n",
                ".meas tran restored_error_diff PARAM='edp_after-edn_after'",
            ]
            if restore_error
            else []
        ),
        ".tran 2p 5n uic",
        ".control",
        "run",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def expected_raw_sign(case: str) -> float:
    if case in {"target_positive_score_negative", "target_positive_score_positive"}:
        return 1.0
    if case == "target_negative_score_positive":
        return -1.0
    return 0.0


def expected_binary_descent_sign(case: str) -> float:
    if case in {"target_positive_score_negative", "target_positive_score_positive"}:
        return 1.0
    if case in {"target_negative_score_positive", "target_negative_score_negative"}:
        return -1.0
    return 0.0


def expected_label_descent_sign(case: str) -> float:
    if case in {"target_positive_score_negative", "target_positive_score_positive"}:
        return 1.0
    if case in {"target_negative_score_positive", "target_negative_score_negative"}:
        return -1.0
    return 0.0


def classify_sign(actual: float, expected: float, *, min_abs_margin: float) -> str:
    if abs(expected) < 1e-15:
        return "dead_zone" if abs(actual) < min_abs_margin else "biased"
    if expected > 0.0 and actual >= min_abs_margin:
        return "aligned"
    if expected < 0.0 and actual <= -min_abs_margin:
        return "aligned"
    if abs(actual) < min_abs_margin:
        return "weak"
    return "flipped"


def classify_row(row: dict[str, Any], *, min_abs_margin: float) -> dict[str, str]:
    topology = str(row.get("error_topology", "competition"))
    if topology == "binary-descent":
        expected = expected_binary_descent_sign(str(row["error_case"]))
    elif topology == "label-descent":
        expected = expected_label_descent_sign(str(row["error_case"]))
    else:
        expected = expected_raw_sign(str(row["error_case"]))
    classifications = {
        "raw_error_classification": classify_sign(
            float(row["raw_error_diff"]),
            expected,
            min_abs_margin=min_abs_margin,
        )
    }
    if "restored_error_diff" in row:
        classifications["restored_error_classification"] = classify_sign(
            float(row["restored_error_diff"]),
            expected,
            min_abs_margin=min_abs_margin,
        )
    return classifications


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    generated = ROOT / "spice/generated"
    tables = ROOT / "results/tables"
    generated.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    spice_bin, version = detect_spice(args.spice_bin)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for case in ERROR_CASES:
        path = generated / f"{tag}_{sanitize_tag(case)}.cir"
        measures = run_netlist(
            spice_bin,
            path,
            generate_netlist(
                error_case=case,
                score_center=args.score_center,
                score_delta=args.score_delta,
                restore_error=not args.no_restore_error,
                error_restore_width=args.error_restore_width,
                error_topology=args.error_topology,
            ),
            timeout=args.timeout,
        )
        row = {"error_case": case, "error_topology": args.error_topology, **measures}
        row.update(classify_row(row, min_abs_margin=args.min_abs_margin))
        rows.append(row)
    csv_path = tables / f"{tag}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    classification_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key, value in row.items():
            if key.endswith("_classification"):
                counts = classification_counts.setdefault(key, {})
                cls = str(value)
                counts[cls] = counts.get(cls, 0) + 1
    passed = all(
        str(value) in {"aligned", "dead_zone", "biased"}
        for row in rows
        for key, value in row.items()
        if key.endswith("_classification")
    )
    summary = {
        "simulator": version,
        "architecture": "score_error_primitive",
        "model_level": "ngspice built-in LEVEL=1 MOS models; not a foundry PDK.",
        "cases": len(rows),
        "passed": passed,
        "classification_counts": classification_counts,
        "csv": str(csv_path),
        "wall_time_s": time.perf_counter() - start,
    }
    summary_path = tables / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-bin", default=None)
    ap.add_argument("--tag", default="score_error_primitive")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--score-center", type=float, default=0.10)
    ap.add_argument("--score-delta", type=float, default=0.08)
    ap.add_argument("--no-restore-error", action="store_true")
    ap.add_argument("--error-restore-width", type=float, default=7.0)
    ap.add_argument("--error-topology", choices=ERROR_TOPOLOGIES, default="competition")
    ap.add_argument("--min-abs-margin", type=float, default=25e-3)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")
    if args.score_delta < 0.0:
        raise ValueError("score-delta must be nonnegative")
    if args.error_restore_width <= 0.0:
        raise ValueError("error-restore-width must be positive")
    if args.min_abs_margin < 0.0:
        raise ValueError("min-abs-margin must be nonnegative")


def main_for_test(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    print(json.dumps(run_cases(args), indent=2))


if __name__ == "__main__":
    main()
