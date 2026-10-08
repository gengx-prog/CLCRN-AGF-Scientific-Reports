"""Redraw old manuscript diagnostic styles using the October 2026 run data.

No model fitting or fabricated observations. Usage:
python diagnostics.py --data-root /path/to/revision_2026_10_08/data --output-root /path/to/output
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

TASKS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
LABELS = ["Temperature", "Humidity", "Wind", "Cloud Cover"]
LABEL = dict(zip(TASKS, LABELS))
COLORS5 = dict(zip(TASKS, ["#1b69b9", "#e65d45", "#42ab16", "#8135a7"]))
COLORS7 = dict(zip(TASKS, ["#DD8452", "#4C72B0", "#55A868", "#DA8BC3"]))
COLORS8 = dict(zip(TASKS, ["#D18E52", "#51A98F", "#2588C5", "#8271BE"]))
UNITS = dict(zip(TASKS, ["K", "percentage points", r"m s$^{-1}$", "fraction"]))
SEEDS = [2021, 2022, 2023, 2024, 2025]
METRICS = ["exact_mae", "cumulative_mae", "exact_rmse", "cumulative_rmse"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def save(fig, stem, out):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out / "figures" / f"{stem}.{ext}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(font="DejaVu Sans"):
    plt.rcdefaults()
    plt.rcParams.update({"font.family": font, "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": False, "savefig.transparent": False})


def fig5(data, out):
    source = pd.read_csv(data / "horizon_per_seed.csv")
    h = source.loc[source.kind.eq("agf")].copy()
    assert len(h) == 4 * 5 * 12
    assert not h.duplicated(["dataset", "seed", "lead_hours"]).any()
    checks = []
    for (task, seed), g in h.groupby(["dataset", "seed"]):
        g = g.sort_values("lead_hours")
        assert g.lead_hours.tolist() == list(range(1, 13))
        np.testing.assert_allclose(g.cumulative_mae, np.cumsum(g.exact_mae) / np.arange(1, 13), atol=1e-12, rtol=0)
        np.testing.assert_allclose(g.cumulative_rmse, np.sqrt(np.cumsum(g.exact_rmse**2) / np.arange(1, 13)), atol=1e-12, rtol=0)
        raw_path = data / f"inference_agf_{task}_{seed}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert raw["metrics"]["n_examples"] == 657
        for metric in METRICS:
            np.testing.assert_allclose(g[metric], raw["metrics"][metric], atol=1e-12, rtol=0)
        checks.append({"file": raw_path.name, "sha256": sha(raw_path), "test_examples": 657,
                       "checkpoint_sha256": raw["checkpoint"]["sha256"]})
    grouped = h.groupby(["dataset", "lead_hours"])[METRICS].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(x) for x in grouped.columns]
    summary = grouped.reset_index()
    table = pd.read_csv(data / "table1_primary_statistics.csv")
    ends = []
    for task in TASKS:
        assert sorted(h.loc[h.dataset.eq(task), "seed"].unique()) == SEEDS
        end = summary.loc[summary.dataset.eq(task) & summary.lead_hours.eq(12)].iloc[0]
        for metric in ["mae", "rmse"]:
            target = table.loc[table.dataset.eq(task) & table.metric.eq(metric)].iloc[0]
            for stat, col in [("mean", "agf_mean"), ("std", "agf_sd")]:
                err = float(end[f"cumulative_{metric}_{stat}"] - target[col])
                assert abs(err) < 1e-9
                ends.append({"dataset": task, "metric": metric, "statistic": stat, "difference_from_table1": err})
    write_csv(h, out / "source_data/fig5_horizon_per_seed.csv")
    write_csv(summary, out / "source_data/fig5_horizon_summary.csv")
    style("DejaVu Serif")
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 6.7))
    for task, ax in zip(TASKS, axes.flat):
        frame = summary.loc[summary.dataset.eq(task)].sort_values("lead_hours")
        x = frame.lead_hours.to_numpy()
        for metric, label, color, marker, ls, alpha in [
            ("exact_mae", "Exact MAE", COLORS5[task], "o", "-", 1),
            ("cumulative_mae", "Cumulative MAE", COLORS5[task], "s", "--", .74),
            ("exact_rmse", "Exact RMSE", "#666666", "^", ":", .78),
        ]:
            y, sd = frame[f"{metric}_mean"].to_numpy(), frame[f"{metric}_std"].to_numpy()
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=.10, linewidth=0)
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=ls,
                    linewidth=1.6 if metric != "exact_rmse" else 1.2, markersize=3.4, alpha=alpha)
        ax.set_title(LABEL[task], fontweight="bold", pad=9)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(f"Error ({UNITS[task]})")
        ax.set_xticks([1, 3, 5, 7, 9, 11, 12])
        ax.set_xlim(.5, 12.5)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=.55, alpha=.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.4, loc="upper left", framealpha=.92, fancybox=False,
                  edgecolor="#cccccc", borderpad=.35, labelspacing=.32, handlelength=2)
    fig.suptitle("Per-Step Forecast Error vs. Prediction Horizon", fontsize=13, fontweight="bold", y=.992)
    fig.text(.5, .008, "CLCRN-AGF · five training seeds · 657 test windows per seed · shading: ±1 sample SD", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .027, 1, .97), h_pad=1.5, w_pad=2)
    save(fig, "fig5_horizon_curves", out)
    return {"caption": "Per-step forecast error of five CLCRN-AGF models per task on all 657 test windows. Solid circles show exact-lead MAE, dashed squares show MAE accumulated over leads 1 through h, and grey triangles show exact-lead RMSE. Lines denote the training-seed mean and shading denotes one sample standard deviation (ddof=1). These 20 AGF checkpoints are the AGF half of the 40 primary runs in Table 1. At 12 h the cumulative MAE equals the corresponding Table 1 value. Independent vertical scales retain each task's physical units. Valid zero targets are included and spatial weights are uniform.",
            "styling_notes": "Preserves the old 2-by-2 serif layout, task colours and three line/marker encodings; adds sample-SD bands and explicit physical units. Cumulative RMSE is checked against Table 1 in the exported source data but is not plotted.",
            "checks": {"per_seed_horizon_rows": len(h), "inference_sources": checks, "table1_endpoints": ends},
            "source_data": ["fig5_horizon_per_seed.csv", "fig5_horizon_summary.csv"]}


def fig7(data, out):
    full = pd.read_csv(data / "robustness_per_seed_mask.csv")
    raw = full.loc[full.kind.eq("agf")].copy()
    assert len(raw) == 4 * 5 * 13
    assert set(raw.n_examples) == {64}
    assert not raw.duplicated(["dataset", "seed", "ratio", "maskseed"]).any()
    recomputed = 100 * (raw.mae / raw.clean_mae_same_subset - 1)
    np.testing.assert_allclose(raw.relative_increase_percent, recomputed, atol=1e-10, rtol=0)
    for (task, seed), frame in raw.groupby(["dataset", "seed"]):
        assert len(frame.loc[frame.ratio.eq(0)]) == 1
        clean = frame.loc[frame.ratio.eq(0), "mae"].iloc[0]
        np.testing.assert_allclose(frame.clean_mae_same_subset, clean, atol=1e-12, rtol=0)
        for ratio in [.1, .2, .3, .4]:
            assert sorted(frame.loc[frame.ratio.eq(ratio), "maskseed"].tolist()) == [11, 22, 33]
    per_seed = raw.groupby(["dataset", "seed", "ratio"], as_index=False).agg(
        relative_mae_change_percent=("relative_increase_percent", "mean"),
        absolute_mae=("mae", "mean"), clean_mae_same_subset=("clean_mae_same_subset", "first"),
        n_mask_draws=("mae", "size"), n_examples=("n_examples", "first"))
    summary = per_seed.groupby(["dataset", "ratio"], as_index=False).agg(
        mean_relative_mae_change_percent=("relative_mae_change_percent", "mean"),
        sample_sd_relative_mae_change_percent=("relative_mae_change_percent", "std"),
        n_training_seeds=("seed", "nunique"), mean_absolute_mae=("absolute_mae", "mean"))
    assert set(summary.n_training_seeds) == {5}
    write_csv(raw, out / "source_data/fig7_robustness_per_mask.csv")
    write_csv(per_seed, out / "source_data/fig7_robustness_per_seed.csv")
    write_csv(summary, out / "source_data/fig7_robustness_summary.csv")
    style()
    fig, ax = plt.subplots(figsize=(8.1, 5.35))
    ax.set_facecolor("#EAEAF2")
    patterns = ["-", (0, (4, 1.5)), (0, (1, 1)), (0, (3, 1.25, 1.5, 1.25))]
    for task, ls in zip(TASKS, patterns):
        frame = summary.loc[summary.dataset.eq(task)].sort_values("ratio")
        x = frame.ratio.to_numpy()
        y = frame.mean_relative_mae_change_percent.to_numpy()
        sd = frame.sample_sd_relative_mae_change_percent.to_numpy()
        ax.fill_between(x, y - sd, y + sd, color=COLORS7[task], alpha=.16, linewidth=0)
        ax.plot(x, y, color=COLORS7[task], linestyle=ls, linewidth=2, label=LABEL[task])
    ax.set_xlabel("Missing node ratio")
    ax.set_ylabel("MAE change from clean input (%)")
    ax.set_title("Forecast sensitivity to missing input nodes", fontsize=12, pad=13)
    ax.set_xticks([0, .1, .2, .3, .4])
    ax.grid(color="white", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.legend(title="Variable", loc="upper left", frameon=True, framealpha=.94,
              facecolor="#EAEAF2", edgecolor="#CCCCCC", fontsize=9.5, title_fontsize=10)
    fig.text(.5, .017, "64 fixed test windows · five AGF training seeds · three mask draws per nonzero ratio", ha="center", fontsize=8.5)
    fig.text(.5, -.015, "Shading: ±1 sample SD across training seeds after averaging the mask draws", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .044, 1, 1))
    save(fig, "fig7_missing_node_robustness", out)
    indices = []
    for task in TASKS:
        p = data / f"robustness_indices_{task}.json"
        indices.append({"file": p.name, "sha256": sha(p)})
    return {"caption": "Missing-input-node sensitivity of five AGF checkpoints per task, evaluated on the same fixed subset of 64 test windows at every ratio. The same node mask is applied across all 12 input hours and signal channels; standardized values are set to zero, corresponding to training-mean imputation. The plotted quantity is 100 times (masked MAE / clean MAE - 1), with each checkpoint's clean MAE measured on that same subset. At each nonzero missing-node ratio, mask seeds 11, 22 and 33 are averaged within a training seed before computing the mean and sample standard deviation across five training seeds. The clean point is zero by definition. Shaded bands indicate descriptive training-seed variation, not confidence intervals or a significance test. Tasks use distinct colours and line patterns. This diagnostic uses 64 of the 657 test windows.",
            "styling_notes": "Preserves the old grey plot surface, white grid and task-specific line patterns and palette. The common axis now uses percent MAE changes rather than combining errors in different physical units.",
            "checks": {"mask_rows": len(raw), "training_seed_rows": len(per_seed), "summary_rows": len(summary), "test_windows": 64, "training_seeds": SEEDS, "mask_seeds": [11, 22, 33], "subset_indices": indices},
            "source_data": ["fig7_robustness_per_mask.csv", "fig7_robustness_per_seed.csv", "fig7_robustness_summary.csv"]}


def binned_quantile(left, right, counts, q):
    """Quantile estimated by linear interpolation within its histogram bin."""
    target = q * np.sum(counts)
    cumulative = np.cumsum(counts)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    before = cumulative[idx - 1] if idx else 0
    return float(left[idx] + (target - before) / counts[idx] * (right[idx] - left[idx]))


def fig8(data, out):
    raw = pd.read_csv(data / "gate_histogram.csv")
    seed_stats = pd.read_csv(data / "gate_per_seed.csv")
    stats = pd.read_csv(data / "table4_gate_statistics.csv")
    assert len(raw) == 4 * 5 * 100
    assert not raw.duplicated(["dataset", "seed", "bin_left"]).any()
    assert not seed_stats.duplicated(["dataset", "seed"]).any()
    quantiles, pooled_rows, checks = [], [], []
    for task in TASKS:
        frame = raw.loc[raw.dataset.eq(task)]
        per_seed = seed_stats.loc[seed_stats.dataset.eq(task)]
        table = stats.loc[stats.dataset.eq(task)].iloc[0]
        assert sorted(per_seed.seed.tolist()) == SEEDS
        assert set(per_seed.n_scalar_gates) == {657 * 12 * 2048 * 8}
        assert set(per_seed.n_test_examples) == {657}
        assert set(per_seed.gate_channels) == {8}
        for seed in SEEDS:
            hist = frame.loc[frame.seed.eq(seed)].sort_values("bin_left")
            assert len(hist) == 100 and int(hist["count"].sum()) == 657 * 12 * 2048 * 8
            np.testing.assert_allclose(hist.bin_left, np.arange(100) / 100, atol=1e-14, rtol=0)
            np.testing.assert_allclose(hist.bin_right, np.arange(1, 101) / 100, atol=1e-14, rtol=0)
        pooled = frame.groupby(["dataset", "bin_left", "bin_right"], as_index=False)["count"].sum()
        pooled = pooled.sort_values("bin_left")
        n = int(pooled["count"].sum())
        assert n == int(table.n_scalar_gates) == 645857280
        mean = np.average(per_seed["mean"], weights=per_seed.n_scalar_gates)
        var = np.average(per_seed.population_std**2 + per_seed["mean"]**2, weights=per_seed.n_scalar_gates) - mean**2
        assert abs(mean - table["mean"]) < 1e-12
        assert abs(np.sqrt(var) - table.pooled_population_std) < 1e-12
        left, right, counts = (pooled[col].to_numpy() for col in ["bin_left", "bin_right", "count"])
        pooled["density"] = counts / n / (right - left)
        pooled_rows.append(pooled)
        q = {f"approx_q{int(p * 100):02d}": binned_quantile(left, right, counts, p) for p in [.05, .25, .5, .75, .95]}
        quantiles.append({"dataset": task, "exact_mean": float(mean), "exact_pooled_population_sd": float(np.sqrt(var)), "n_scalar_gates": n,
                          "n_seeds": 5, "n_test_examples_per_seed": 657, "input_steps": 12, "nodes": 2048, "gate_channels": 8, **q})
        checks.append({"dataset": task, "histogram_count_equals_table4": True, "pooled_moments_equal_table4": True, "scalar_gates": n})
    summary = pd.DataFrame(quantiles)
    hist = pd.concat(pooled_rows, ignore_index=True)
    write_csv(raw, out / "source_data/fig8_gate_histogram_per_seed.csv")
    write_csv(seed_stats, out / "source_data/fig8_gate_statistics_per_seed.csv")
    write_csv(hist, out / "source_data/fig8_gate_histogram_pooled.csv")
    write_csv(summary, out / "source_data/fig8_gate_summary.csv")
    style()
    fig, ax = plt.subplots(figsize=(10, 5.3))
    positions = [4, 3, 2, 1]
    for task, y in zip(TASKS, positions):
        row = summary.loc[summary.dataset.eq(task)].iloc[0]
        p = hist.loc[hist.dataset.eq(task)].sort_values("bin_left")
        centers = (p.bin_left.to_numpy() + p.bin_right.to_numpy()) / 2
        width = p.density.to_numpy() / p.density.max() * .32
        # Histogram-centre interpolation preserves the observed binned density.
        # This is not a KDE fitted to made-up normal or sampled observations.
        x = np.r_[0, centers, 1]
        w = np.r_[0, width, 0]
        ax.fill_between(x, y - w, y + w, facecolor=COLORS8[task], edgecolor=COLORS8[task], alpha=.57, linewidth=.9)
        ax.plot([row.approx_q05, row.approx_q95], [y, y], color="#304354", linewidth=1.25)
        ax.add_patch(Rectangle((row.approx_q25, y - .065), row.approx_q75 - row.approx_q25, .13,
                               facecolor="white", edgecolor="#304354", linewidth=1.1, zorder=3))
        ax.plot([row.approx_q50, row.approx_q50], [y - .085, y + .085], color="#304354", linewidth=1.15, zorder=4)
        ax.scatter([row.exact_mean], [y], marker="D", s=37, c=[COLORS8[task]], edgecolors="white", linewidth=.8, zorder=5)
        ax.text(1.025, y + .044, rf"$\mu$={row.exact_mean:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                color="#304354", fontsize=10, fontweight="bold", clip_on=False)
        ax.text(1.025, y - .13, rf"$\sigma$={row.exact_pooled_population_sd:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                color="#687b8e", fontsize=9, clip_on=False)
    ax.axvline(.5, color="#688099", linewidth=1.1, linestyle=(0, (5, 4)), zorder=0)
    ax.text(.505, 4.50, r"$g=0.5$", color="#58718b", fontsize=9)
    ax.set_yticks(positions, LABELS, fontweight="bold")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(.43, 4.65)
    ax.grid(axis="x", color="#dbe4ed", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#95A9BE")
    ax.set_xlabel("Fusion-gate activation $g$ (0 = residual projection, 1 = graph-aggregated projection)", labelpad=9, fontsize=9.5)
    ax.set_title("Distribution of learned gate activations", loc="left", fontweight="bold", fontsize=13, color="#304354", pad=16)
    handles = [Line2D([0], [0], color="#304354", linewidth=1.25, label="5–95% (approx.)"),
               Patch(facecolor="white", edgecolor="#304354", label="IQR (approx.)"),
               Line2D([0], [0], marker="D", color="white", markerfacecolor="#2588C5", markersize=6, label="Mean")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3, frameon=False, fontsize=9,
              columnspacing=1.9, handlelength=1.8)
    fig.text(.51, .008, "Five AGF seeds × 657 test windows × 12 input steps × 2,048 nodes × all 8 gate channels", ha="center", fontsize=8.3)
    fig.subplots_adjust(left=.17, right=.86, bottom=.25, top=.87)
    save(fig, "fig8_gate_distribution", out)
    return {"caption": "Distribution of scalar gate activations for all four tasks. Each task pools 645,857,280 activations from five AGF checkpoints, all 657 test windows per checkpoint, 12 input steps, 2,048 nodes and all eight channels, without channel averaging. Mirrored profiles interpolate the centres of the recorded 100-bin histogram on [0,1]; widths are normalized separately within each task. White boxes, centre lines and whiskers are approximate 25th–75th, 50th and 5th–95th percentiles computed by linear interpolation of the binned cumulative counts. Diamonds and right-side mean and population standard deviation are the exact full-stream statistics reported in Table 4. The dashed line is g=0.5. The gate combines a graph-aggregated projection and a residual linear projection of the same pre-CLConv input.",
            "styling_notes": "Preserves the horizontal mirrored distribution, white IQR box, mean diamond, g=0.5 reference and right-side moment annotations. Adds temperature and uses the full [0,1] gate range. No artificial samples or normal-distribution assumption are used; neither moments nor shapes come from the old channel-averaged arrays.",
            "checks": checks, "source_data": ["fig8_gate_histogram_per_seed.csv", "fig8_gate_statistics_per_seed.csv", "fig8_gate_histogram_pooled.csv", "fig8_gate_summary.csv"],
            "limitation": "Raw per-scalar gates are not stored in the supplied current package. Distribution shape is represented by the actual 100-bin histogram; displayed quantiles are binned approximations, whereas means and population SDs are exact recorded stream statistics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for dirname in ["figures", "source_data", "metadata"]:
        (args.output_root / dirname).mkdir(parents=True, exist_ok=True)
    source_names = ["horizon_per_seed.csv", "robustness_per_seed_mask.csv", "gate_histogram.csv", "gate_per_seed.csv", "table1_primary_statistics.csv", "table4_gate_statistics.csv"]
    metadata = {"purpose": "Old manuscript figure styling with new, internally consistent experimental data", "data_root": str(args.data_root.resolve()),
                "sources": [{"file": name, "sha256": sha(args.data_root / name)} for name in source_names],
                "software": {"python": __import__("sys").version, "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "figures": {"fig5": fig5(args.data_root, args.output_root), "fig7": fig7(args.data_root, args.output_root), "fig8": fig8(args.data_root, args.output_root)}}
    artifacts = []
    for directory in ["figures", "source_data"]:
        for p in sorted((args.output_root / directory).glob("fig*")):
            if p.name.startswith(("fig5", "fig7", "fig8")):
                artifacts.append({"file": p.relative_to(args.output_root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)})
    metadata["artifacts"] = artifacts
    metadata["script_sha256"] = sha(__file__)
    (args.output_root / "metadata/diagnostics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "artifacts": len(artifacts), "metadata": str(args.output_root / "metadata/diagnostics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
