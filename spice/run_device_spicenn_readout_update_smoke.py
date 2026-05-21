from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from spicenn import (
        CapState,
        DifferentialCapState,
        FanInTopology,
        NetlistBuilder,
        SignedScoreErrorCell,
        make_sparse_readout_update_layer,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spicenn import (
        CapState,
        DifferentialCapState,
        FanInTopology,
        NetlistBuilder,
        SignedScoreErrorCell,
        make_sparse_readout_update_layer,
    )

from _util import parse_measures
from run_device_multicell_classifier import mos_models
from run_spice_sweep import ROOT, detect_spice, run_text_netlist, run_tiny_test


def pulse(start_ns: float, end_ns: float, high_v: float = 1.2, stop_ns: float = 4.0) -> str:
    return (
        "PWL("
        f"0n 0 {start_ns:.12g}n 0 {start_ns + 0.02:.12g}n {high_v:.12g} "
        f"{end_ns:.12g}n {high_v:.12g} {end_ns + 0.02:.12g}n 0 {stop_ns:.12g}n 0)"
    )


def render_update_case(
    prefix: str,
    *,
    target_v: float,
    score_pos_v: float,
    score_neg_v: float,
    update_width_u: float,
) -> str:
    topology = FanInTopology.from_fanins((0,), 1, {0: (0,)})
    weight_prefix = f"vw_{prefix}_"
    dp_prefix = f"dp_{prefix}_"
    dn_prefix = f"dn_{prefix}_"
    deck = NetlistBuilder()
    weight = DifferentialCapState.from_base(
        f"{weight_prefix}0_0",
        cap_f=4.0,
        pos_ic_v=0.55,
        neg_ic_v=0.55,
        leak_to="0",
        leak_ohm="1e15",
    )
    error = SignedScoreErrorCell(
        f"err_{prefix}",
        target_node=f"target_{prefix}",
        score_pos_node=f"scorep_{prefix}",
        score_neg_node=f"scoren_{prefix}",
        positive_error=CapState(f"{dp_prefix}0", f"{dp_prefix}0", 6.0, leak_to="0", leak_ohm="1G"),
        negative_error=CapState(f"{dn_prefix}0", f"{dn_prefix}0", 6.0, leak_to="0", leak_ohm="1G"),
        target_width_u=40.0,
        score_width_u=32.0,
    )
    update = make_sparse_readout_update_layer(
        f"update_{prefix}",
        topology=topology,
        source_nodes={0: f"act_{prefix}"},
        weight_prefix=weight_prefix,
        update_prefix=f"uw_{prefix}_",
        positive_error_prefix=dp_prefix,
        negative_error_prefix=dn_prefix,
        update_width_u=update_width_u,
    )
    for component in (weight, error, update):
        deck.render_component(component)
    return "\n".join(
        [
            f"Vact_{prefix} act_{prefix} 0 DC 1.0",
            f"Vtarget_{prefix} target_{prefix} 0 DC {target_v:.12g}",
            f"Vscorep_{prefix} scorep_{prefix} 0 DC {score_pos_v:.12g}",
            f"Vscoren_{prefix} scoren_{prefix} 0 DC {score_neg_v:.12g}",
            deck.render_body(),
        ]
    )


def measurement_lines(prefix: str) -> str:
    p = f"vw_{prefix}_0_0p"
    n = f"vw_{prefix}_0_0n"
    dp = f"dp_{prefix}_0"
    dn = f"dn_{prefix}_0"
    return "\n".join(
        [
            f".meas tran {prefix}_p_before FIND V({p}) AT=1.65n",
            f".meas tran {prefix}_n_before FIND V({n}) AT=1.65n",
            f".meas tran {prefix}_p_after FIND V({p}) AT=3.65n",
            f".meas tran {prefix}_n_after FIND V({n}) AT=3.65n",
            f".meas tran {prefix}_dp_err FIND V({dp}) AT=1.55n",
            f".meas tran {prefix}_dn_err FIND V({dn}) AT=1.55n",
        ]
    )


def add_derived(measures: dict[str, float]) -> dict[str, float]:
    enriched = dict(measures)
    for prefix in ("pos", "neg"):
        before = enriched[f"{prefix}_p_before"] - enriched[f"{prefix}_n_before"]
        after = enriched[f"{prefix}_p_after"] - enriched[f"{prefix}_n_after"]
        enriched[f"{prefix}_signed_before"] = before
        enriched[f"{prefix}_signed_after"] = after
        enriched[f"{prefix}_signed_delta"] = after - before
        enriched[f"{prefix}_common_delta"] = (
            enriched[f"{prefix}_p_after"]
            - enriched[f"{prefix}_p_before"]
            + enriched[f"{prefix}_n_after"]
            - enriched[f"{prefix}_n_before"]
        )
    return enriched


def netlist(*, update_width_u: float = 0.004) -> str:
    return f"""
* spicenn signed readout error/update smoke
.param VDD=1.2
{mos_models()}
Vdd vdd 0 {{VDD}}
Vwhigh whigh 0 DC 1.0
Vwlow wlow 0 DC 0.1
Verr err 0 {pulse(0.50, 1.45)}
Vbwd bwd 0 {pulse(1.75, 3.45)}
{render_update_case("pos", target_v=1.1, score_pos_v=0.0, score_neg_v=0.0, update_width_u=update_width_u)}
{render_update_case("neg", target_v=0.0, score_pos_v=1.1, score_neg_v=0.0, update_width_u=update_width_u)}
.options method=gear maxord=2 reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e12
.tran 5p 4n uic
{measurement_lines("pos")}
{measurement_lines("neg")}
.control
run
.endc
.end
""".lstrip()


def run_netlist(spice_bin: str, path: Path, text: str, timeout: float) -> dict[str, float]:
    proc = run_text_netlist(spice_bin, path, text, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    measures = parse_measures(proc.stdout + "\n" + proc.stderr)
    if not measures:
        raise RuntimeError("SPICE produced no parseable measurements:\n" + (proc.stdout + proc.stderr)[-3000:])
    return add_derived(measures)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--tag", default="spicenn_readout_update_smoke")
    ap.add_argument("--update-width-u", type=float, default=0.004)
    args = ap.parse_args()

    spice_bin, version = detect_spice(args.simulator)
    generated = ROOT / "spice/generated"
    generated.mkdir(parents=True, exist_ok=True)
    run_tiny_test(spice_bin, generated)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag)
    text = netlist(update_width_u=args.update_width_u)
    t0 = time.perf_counter()
    measures = run_netlist(spice_bin, generated / f"{safe_tag}.cir", text, args.timeout)
    summary: dict[str, Any] = {
        "simulator": version,
        "architecture": "spicenn_signed_readout_error_update_smoke",
        "status": "local_update_direction_smoke",
        "measures": measures,
        "pos_update_sign_correct": measures["pos_signed_delta"] > 0,
        "neg_update_sign_correct": measures["neg_signed_delta"] < 0,
        "wall_time_s": time.perf_counter() - t0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
