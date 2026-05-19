from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.hardware_metrics import local_evidence_net_layers


def draw_architecture(channels: tuple[int, int, int], out_path: Path) -> None:
    layers = local_evidence_net_layers(channels=channels)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 66)
    ax.axis("off")

    colors = ["#e8eef8", "#d9ead3", "#fff2cc", "#f4cccc"]
    x0 = 8
    widths = [19, 19, 19, 15]
    heights = [48, 34, 24, 24]
    labels = [
        "28x28 input tile sheet\n+ x/y coordinate ramps",
        f"5x5 local sum\n{channels[0]} channels",
        f"3x3 local sum\n{channels[1]} channels\n2x2 pool",
        f"3x3 local sum\n{channels[2]} channels\n1x1 class evidence",
    ]
    xs = [x0, 34, 60, 84]
    ys = [9, 16, 21, 21]
    for i, (x, y, w, h) in enumerate(zip(xs, ys, widths, heights)):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.8",
            linewidth=1.4,
            edgecolor="#2f3a45",
            facecolor=colors[i],
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h + 2.0, labels[i], ha="center", va="bottom", fontsize=9)
        grid_n = 7 if i else 14
        for gx in range(grid_n + 1):
            ax.plot([x + gx * w / grid_n, x + gx * w / grid_n], [y, y + h], color="#6b7280", lw=0.25, alpha=0.45)
        for gy in range(grid_n + 1):
            ax.plot([x, x + w], [y + gy * h / grid_n, y + gy * h / grid_n], color="#6b7280", lw=0.25, alpha=0.45)
        if i > 0:
            ax.text(x + w / 2, y - 3.5, f"max wire {layers[i-1].max_wire_length_um:.0f} um", ha="center", fontsize=8)

    for x_a, x_b in zip(xs[:-1], xs[1:]):
        ax.annotate(
            "",
            xy=(x_b - 1.5, 33),
            xytext=(x_a + widths[xs.index(x_a)] + 1.5, 33),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#374151"),
        )

    ax.text(91.5, 14.5, "spatial mean\n10 logits", ha="center", va="top", fontsize=9)
    ax.annotate("", xy=(100, 33), xytext=(101.5, 33), arrowprops=dict(arrowstyle="->", lw=1.5, color="#374151"))
    ax.add_patch(patches.Rectangle((102, 28), 2.0, 10.0, facecolor="#cfe2f3", edgecolor="#2f3a45"))
    ax.text(103, 25.5, "output", ha="center", fontsize=8)

    summary = (
        f"Current candidate: local evidence MNIST tile, channels {channels[0]}-{channels[1]}-{channels[2]}\n"
        "No dense final layer. Shared local kernels are mapped as repeated local synapse neighborhoods.\n"
        "Coordinate ramps preserve spatial position for global class-evidence averaging."
    )
    ax.text(8, 61.5, summary, ha="left", va="top", fontsize=11, weight="bold")

    totals = {
        "synapse_instances": sum(layer.synapse_instances for layer in layers),
        "parameters": sum(layer.parameter_count for layer in layers),
        "wire_um": sum(layer.total_wire_length_um for layer in layers),
        "max_wire_um": max(layer.max_wire_length_um for layer in layers),
    }
    metrics = (
        f"Synapse instances: {totals['synapse_instances']:,}\n"
        f"Stored weights: {totals['parameters']:,}\n"
        f"Total local wire: {totals['wire_um'] / 1e6:.1f} m-equivalent on tile grid\n"
        f"Max local wire: {totals['max_wire_um']:.0f} um"
    )
    ax.text(8, 4, metrics, ha="left", va="bottom", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="32,64,96")
    ap.add_argument("--out", default=str(ROOT / "results/figures/current_hardware_architecture.png"))
    args = ap.parse_args()
    channels = tuple(int(v) for v in args.channels.split(","))
    if len(channels) != 3:
        raise ValueError("--channels must be three comma-separated integers")
    draw_architecture(channels, Path(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
