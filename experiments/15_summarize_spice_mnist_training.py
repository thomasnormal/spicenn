from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for path in sorted((ROOT / "spice/results").glob("spice_mnist*_summary.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "summary": str(path),
                "netlist": data.get("netlist"),
                "eval_netlist": data.get("eval_netlist"),
                "dataset": data.get("dataset"),
                "image_size": data.get("image_size"),
                "inputs": data.get("inputs"),
                "hidden": data.get("hidden", 0),
                "classes": data.get("classes"),
                "train_samples": data.get("train_samples"),
                "test_samples": data.get("test_samples"),
                "epochs": data.get("epochs"),
                "lr": data.get("lr"),
                "train_accuracy": data.get("final", {}).get("train_accuracy"),
                "heldout_test_accuracy": data.get("heldout_test_accuracy"),
                "learning_curve": data.get("learning_curve"),
                "final_weights": data.get("final_weights"),
                "note": data.get("note"),
            }
        )
    df = pd.DataFrame(rows)
    out_csv = ROOT / "results/tables/spice_mnist_training_runs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    valid = df.dropna(subset=["heldout_test_accuracy"])
    best = valid.sort_values(["heldout_test_accuracy", "train_accuracy"], ascending=[False, False]).iloc[0].to_dict()
    out_json = ROOT / "results/tables/spice_mnist_training_best.json"
    out_json.write_text(json.dumps({"best": best, "runs": rows}, indent=2) + "\n")
    print(json.dumps({"table": str(out_csv), "best": best}, indent=2))


if __name__ == "__main__":
    main()

