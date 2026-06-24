"""Generate the Scientific Reports component-ablation figure.

The figure uses the current manuscript data sources:

- temperature 100-epoch, 3-seed ablation:
  experiments_nmi/B_clean_ablation/seed_{2023,2024,2025}/aggregate_results.json
- humidity w/o adaptive graph and full CLCRN-AGF:
  experiments/corrected_publication_metrics/rows.json, restricted to seeds
  2023--2025
- humidity w/o gated fusion:
  experiments/humidity_ablation_100ep_amp_b32/aggregate_results.json

Output:
  Scientific_Reports_submission/figs/fig_component_ablation_100ep.pdf
  Scientific_Reports_submission/figs/fig_component_ablation_100ep.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "Scientific_Reports_submission"
    / "figs"
    / "fig_component_ablation_100ep.pdf"
)
OUTPUT_PNG = OUTPUT.with_suffix(".png")

SEEDS = [2023, 2024, 2025]
VARIANTS = [
    ("wo_adaptive_graph", "w/o adaptive\ngraph", "#4c78a8"),
    ("wo_gated_fusion", "w/o gated\nfusion", "#72b7b2"),
    ("full_model", "Full\nCLCRN-AGF", "#f58518"),
]
METRICS = [("mae", "MAE"), ("rmse", "RMSE")]


def load_temperature() -> dict[str, dict[str, list[float]]]:
    out = {variant: {"mae": [], "rmse": []} for variant, _, _ in VARIANTS}
    for seed in SEEDS:
        path = (
            ROOT
            / "experiments_nmi"
            / "B_clean_ablation"
            / f"seed_{seed}"
            / "aggregate_results.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["results"]:
            variant = row["variant"]
            if variant in out:
                out[variant]["mae"].append(float(row["mae"]))
                out[variant]["rmse"].append(float(row["rmse"]))
    return out


def load_humidity() -> dict[str, dict[str, list[float]]]:
    out = {variant: {"mae": [], "rmse": []} for variant, _, _ in VARIANTS}

    rows = json.loads(
        (ROOT / "experiments" / "corrected_publication_metrics" / "rows.json").read_text(
            encoding="utf-8"
        )
    )
    kind_to_variant = {"control": "wo_adaptive_graph", "agf": "full_model"}
    for row in rows:
        if row["dataset"] != "humidity" or row["seed"] not in SEEDS:
            continue
        variant = kind_to_variant.get(row["kind"])
        if variant:
            out[variant]["mae"].append(float(row["mae"]))
            out[variant]["rmse"].append(float(row["rmse"]))

    wo_gated = json.loads(
        (
            ROOT
            / "experiments"
            / "humidity_ablation_100ep_amp_b32"
            / "aggregate_results.json"
        ).read_text(encoding="utf-8")
    )
    for row in wo_gated["per_seed"]:
        out["wo_gated_fusion"]["mae"].append(float(row["mae"]))
        out["wo_gated_fusion"]["rmse"].append(float(row["rmse"]))

    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def draw_panel(ax, data, metric: str, task_title: str) -> None:
    x = np.arange(len(VARIANTS), dtype=float)
    means = []
    stds = []
    colors = []
    for variant, _, color in VARIANTS:
        mu, sd = mean_std(data[variant][metric])
        means.append(mu)
        stds.append(sd)
        colors.append(color)

    ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        alpha=0.84,
        edgecolor="black",
        linewidth=0.7,
        capsize=4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )

    jitter = np.array([-0.09, 0.0, 0.09])
    for idx, (variant, _, _) in enumerate(VARIANTS):
        vals = np.asarray(data[variant][metric], dtype=float)
        ax.scatter(
            np.full_like(vals, x[idx]) + jitter[: len(vals)],
            vals,
            s=20,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in VARIANTS], fontsize=8)
    ax.set_title(task_title, fontsize=10, pad=5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    temperature = load_temperature()
    humidity = load_humidity()
    tasks = [
        ("Temperature", temperature),
        ("Relative humidity", humidity),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=False)

    for col, (task_title, task_data) in enumerate(tasks):
        for row, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row, col]
            draw_panel(ax, task_data, metric, task_title if row == 0 else "")
            ax.set_ylabel(f"Test {metric_label}")

    axes[0, 0].text(
        -0.18,
        1.08,
        "a",
        transform=axes[0, 0].transAxes,
        fontweight="bold",
        fontsize=12,
    )
    axes[0, 1].text(
        -0.18,
        1.08,
        "b",
        transform=axes[0, 1].transAxes,
        fontweight="bold",
        fontsize=12,
    )
    axes[1, 0].text(
        -0.18,
        1.08,
        "c",
        transform=axes[1, 0].transAxes,
        fontweight="bold",
        fontsize=12,
    )
    axes[1, 1].text(
        -0.18,
        1.08,
        "d",
        transform=axes[1, 1].transAxes,
        fontweight="bold",
        fontsize=12,
    )

    fig.suptitle(
        "Full-budget component ablations over three seed replicates",
        fontsize=12,
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Bars show mean ± sample standard deviation; dots show individual seeds. Lower is better.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0.02, 0.04, 1.0, 0.96])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
