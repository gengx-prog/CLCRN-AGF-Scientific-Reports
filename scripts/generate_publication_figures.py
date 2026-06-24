import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SUMMARY = ROOT / "experiments" / "weatherbench_clcrn_paper_run" / "summary_all.json"
IMPROVED_SUMMARY = ROOT / "experiments" / "weatherbench_asttn_full_100ep" / "summary_all.json"
PAPER_COMPARISON = ROOT / "experiments" / "weatherbench_asttn_full_100ep" / "comparison_vs_pdf.json"
OUT_DIR = ROOT / "experiments" / "weatherbench_publication_figures"

DATASET_LABELS = {
    "temperature": "Temperature",
    "humidity": "Humidity",
    "component_of_wind": "Wind",
    "cloud_cover": "Cloud Cover",
}

COLORS = {
    "paper": "#284B63",
    "original": "#D9A441",
    "improved": "#2A9D8F",
    "gain": "#E76F51",
    "loss": "#577590",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_map(rows):
    return {row["dataset"]: row for row in rows}


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )


def ordered_datasets(rows):
    order = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
    available = {row["dataset"] for row in rows}
    return [ds for ds in order if ds in available]


def add_value_labels(ax, bars, fmt="{:.3f}", pad=0.01):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax * pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_figure(fig, stem: str):
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metric(datasets, paper, original, improved, metric: str, title: str, stem: str):
    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.8))

    paper_vals = [paper[ds][metric] for ds in datasets]
    orig_vals = [original[ds][metric] for ds in datasets]
    imp_vals = [improved[ds][metric] for ds in datasets]

    bars1 = ax.bar(x - width, paper_vals, width, label="Paper CLCRN", color=COLORS["paper"])
    bars2 = ax.bar(x, orig_vals, width, label="Reproduced CLCRN", color=COLORS["original"])
    bars3 = ax.bar(x + width, imp_vals, width, label="Improved CLCRN", color=COLORS["improved"])

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets])
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)
    add_value_labels(ax, bars3)
    save_figure(fig, stem)


def plot_improvement_dumbbell(datasets, original, improved, metric: str, title: str, stem: str):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(datasets))[::-1]

    for idx, ds in enumerate(datasets):
        y_pos = y[idx]
        old_val = original[ds][metric]
        new_val = improved[ds][metric]
        ax.plot([old_val, new_val], [y_pos, y_pos], color="#C8D5B9", linewidth=3, zorder=1)
        ax.scatter(old_val, y_pos, s=90, color=COLORS["original"], label="Reproduced CLCRN" if idx == 0 else None, zorder=2)
        ax.scatter(new_val, y_pos, s=90, color=COLORS["improved"], label="Improved CLCRN" if idx == 0 else None, zorder=3)
        delta_pct = (new_val - old_val) / old_val * 100.0
        ax.text(
            max(old_val, new_val) + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01 if idx else max(old_val, new_val),
            y_pos + 0.07,
            f"{delta_pct:+.2f}%",
            fontsize=10,
            color=COLORS["gain"] if delta_pct < 0 else COLORS["loss"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABELS[ds] for ds in datasets])
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, stem)


def plot_horizon_curves(datasets, original, improved, metric: str, title: str, stem: str):
    horizons = [3, 6, 12]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    axes = axes.flatten()

    for ax, ds in zip(axes, datasets):
        orig_vals = [original[ds]["step_metrics"][f"{metric}_{h}"] for h in horizons]
        imp_vals = [improved[ds]["step_metrics"][f"{metric}_{h}"] for h in horizons]
        ax.plot(horizons, orig_vals, marker="o", linewidth=2.4, color=COLORS["original"], label="Reproduced CLCRN")
        ax.plot(horizons, imp_vals, marker="o", linewidth=2.4, color=COLORS["improved"], label="Improved CLCRN")
        ax.fill_between(horizons, orig_vals, imp_vals, color="#C8D5B9", alpha=0.18)
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(horizons)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, y=1.04, fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, stem)


def plot_paper_gap_heatmap(datasets, comparison_rows, stem: str):
    mae_gaps = [next(row for row in comparison_rows if row["dataset"] == ds)["delta_mae_pct"] for ds in datasets]
    rmse_gaps = [next(row for row in comparison_rows if row["dataset"] == ds)["delta_rmse_pct"] for ds in datasets]
    values = np.array([mae_gaps, rmse_gaps])

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    sns.heatmap(
        values,
        annot=True,
        fmt=".2f",
        cmap=sns.diverging_palette(20, 150, as_cmap=True),
        center=0.0,
        linewidths=1,
        cbar_kws={"label": "Gap vs Paper (%)"},
        ax=ax,
    )
    ax.set_yticklabels(["MAE Gap", "RMSE Gap"], rotation=0)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets], rotation=0)
    ax.set_title("Relative Gap Between Improved CLCRN and Paper CLCRN")
    save_figure(fig, stem)


def write_gallery(datasets, paper, original, improved, comparison_rows):
    path = OUT_DIR / "figure_gallery.md"
    lines = [
        "# WeatherBench Publication Figures",
        "",
        "This folder contains paper-ready figures generated from the completed 100-epoch results.",
        "",
        "## Figures",
        "",
        "- `fig01_mae_overall.png/pdf`: Overall MAE comparison among paper, reproduced CLCRN, and improved CLCRN.",
        "- `fig02_rmse_overall.png/pdf`: Overall RMSE comparison among paper, reproduced CLCRN, and improved CLCRN.",
        "- `fig03_mae_dumbbell.png/pdf`: MAE shift from reproduced CLCRN to improved CLCRN.",
        "- `fig04_rmse_dumbbell.png/pdf`: RMSE shift from reproduced CLCRN to improved CLCRN.",
        "- `fig05_mae_horizon_curves.png/pdf`: Multi-horizon MAE curves for the original and improved models.",
        "- `fig06_rmse_horizon_curves.png/pdf`: Multi-horizon RMSE curves for the original and improved models.",
        "- `fig07_gap_vs_paper_heatmap.png/pdf`: Relative percentage gap between improved CLCRN and paper CLCRN.",
        "",
        "## Final 100-Epoch Metrics",
        "",
        "| Dataset | Paper MAE | Reproduced MAE | Improved MAE | Paper RMSE | Reproduced RMSE | Improved RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ds in datasets:
        lines.append(
            f"| {DATASET_LABELS[ds]} | "
            f"{paper[ds]['mae']:.4f} | {original[ds]['mae']:.4f} | {improved[ds]['mae']:.4f} | "
            f"{paper[ds]['rmse']:.4f} | {original[ds]['rmse']:.4f} | {improved[ds]['rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gap vs Paper",
            "",
            "| Dataset | MAE Gap (%) | RMSE Gap (%) |",
            "| --- | ---: | ---: |",
        ]
    )
    for ds in datasets:
        row = next(row for row in comparison_rows if row["dataset"] == ds)
        lines.append(
            f"| {DATASET_LABELS[ds]} | {row['delta_mae_pct']:+.2f} | {row['delta_rmse_pct']:+.2f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_out_dir()
    setup_style()

    original = metric_map(load_json(ORIGINAL_SUMMARY))
    improved = metric_map(load_json(IMPROVED_SUMMARY))
    comparison = load_json(PAPER_COMPARISON)
    paper = metric_map(comparison["paper"])
    comparison_rows = comparison["comparisons_vs_paper"]

    datasets = ordered_datasets(load_json(IMPROVED_SUMMARY))

    plot_grouped_metric(
        datasets,
        paper,
        original,
        improved,
        metric="mae",
        title="WeatherBench MAE: Paper vs Reproduced vs Improved CLCRN",
        stem="fig01_mae_overall",
    )
    plot_grouped_metric(
        datasets,
        paper,
        original,
        improved,
        metric="rmse",
        title="WeatherBench RMSE: Paper vs Reproduced vs Improved CLCRN",
        stem="fig02_rmse_overall",
    )
    plot_improvement_dumbbell(
        datasets,
        original,
        improved,
        metric="mae",
        title="MAE Shift From Reproduced CLCRN to Improved CLCRN",
        stem="fig03_mae_dumbbell",
    )
    plot_improvement_dumbbell(
        datasets,
        original,
        improved,
        metric="rmse",
        title="RMSE Shift From Reproduced CLCRN to Improved CLCRN",
        stem="fig04_rmse_dumbbell",
    )
    plot_horizon_curves(
        datasets,
        original,
        improved,
        metric="mae",
        title="Multi-Horizon MAE Curves",
        stem="fig05_mae_horizon_curves",
    )
    plot_horizon_curves(
        datasets,
        original,
        improved,
        metric="rmse",
        title="Multi-Horizon RMSE Curves",
        stem="fig06_rmse_horizon_curves",
    )
    plot_paper_gap_heatmap(
        datasets,
        comparison_rows,
        stem="fig07_gap_vs_paper_heatmap",
    )
    write_gallery(datasets, paper, original, improved, comparison_rows)
    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
