import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORRECTED_ROWS = ROOT / "experiments" / "corrected_publication_metrics" / "rows.json"
OUTPUT = ROOT / "Scientific_Reports_submission" / "figs" / "fig_primary_replicate_comparison.pdf"


def load_rows(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped = {"agf": {}, "control": {}}
    for row in rows:
        grouped[row["kind"]].setdefault(row["dataset"], []).append(row)
    return {
        kind: {
            dataset: sorted(dataset_rows, key=lambda item: item["seed"])
            for dataset, dataset_rows in by_dataset.items()
        }
        for kind, by_dataset in grouped.items()
    }


rows = load_rows(CORRECTED_ROWS)
datasets = [
    ("temperature", "Temperature"),
    ("humidity", "Relative humidity"),
    ("component_of_wind", "Surface wind"),
    ("cloud_cover", "Cloud cover"),
]

fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
for ax, (key, label) in zip(axes, datasets):
    full_by_seed = {row["seed"]: row["mae"] for row in rows["agf"][key]}
    control_by_seed = {row["seed"]: row["mae"] for row in rows["control"][key]}
    seeds = sorted(full_by_seed)
    control_values = np.array([control_by_seed[seed] for seed in seeds])
    full_values = np.array([full_by_seed[seed] for seed in seeds])

    jitter = np.linspace(-0.07, 0.07, len(seeds))
    for offset, left, right in zip(jitter, control_values, full_values):
        ax.scatter(0 + offset, left, color="#4c78a8", s=28, zorder=3, alpha=0.9)
        ax.scatter(1 + offset, right, color="#f58518", s=28, zorder=3, alpha=0.9)

    ax.errorbar(
        [0, 1],
        [control_values.mean(), full_values.mean()],
        yerr=[control_values.std(ddof=1), full_values.std(ddof=1)],
        fmt="D",
        color="black",
        capsize=3,
        markersize=5,
        linewidth=1.2,
        zorder=4,
    )
    delta = 100.0 * (full_values.mean() - control_values.mean()) / control_values.mean()
    ax.set_title(f"{label}\nmean change {delta:+.2f}%", fontsize=10)
    ax.set_xticks([0, 1], ["Control", "CLCRN-AGF"])
    ax.set_ylabel("Test MAE")
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Five-replicate comparison of the adaptive graph-fusion extension", fontsize=12)
fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
print(OUTPUT)
